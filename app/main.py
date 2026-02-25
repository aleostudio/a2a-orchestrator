from typing import Any
import asyncio
import json
import re
import time
import uuid
import httpx
import ollama as ollama_client
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from a2a.client.client_factory import ClientFactory
from a2a.types import AgentCard, Message, Part, Role, TextPart
from app.logger import logger
from app.config import (
    APP_NAME,
    APP_VERSION,
    APP_HOST,
    APP_PORT,
    REGISTRY_URL,
    REGISTRY_TIMEOUT_S,
    REGISTRY_RETRIES,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_S,
    AGENT_TIMEOUT_S,
    AGENT_RETRIES,
    CIRCUIT_FAILURE_THRESHOLD,
    CIRCUIT_RECOVERY_SECONDS,
)


# Ollama client
ollama = ollama_client.Client(host=OLLAMA_BASE_URL)


# Routing policy
MATH_OPERATORS = ("+", "-", "*", "/", "%", "^", "(", ")")
MATH_EXPR_RE = re.compile(r"[0-9\s\+\-\*\/\%\(\)\.^]{3,}")


# Prompt
INTENT_SYSTEM_PROMPT = """You are an intent classifier. Given a user message, determine what kind of help is needed.

Respond with a JSON object only, no other text:
{"skill": "<skill_tag>", "payload": "<text to send to the agent>"}

Rules:
- If the message requires mathematical calculation, respond: {"skill": "math", "payload": "<the math expression>"}
- For anything else, respond: {"skill": "general", "payload": "<the original message>"}
"""


class CircuitBreaker:
    # Initialize thresholds and in-memory circuit state.
    def __init__(self, threshold: int, recovery_seconds: int):
        self.threshold = threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.opened_until = 0.0

    # Check whether calls are currently allowed.
    def allow(self) -> bool:
        return time.time() >= self.opened_until

    # Reset breaker state after a successful call.
    def on_success(self) -> None:
        self.failures = 0
        self.opened_until = 0.0

    # Record a failure and open the circuit when threshold is reached.
    def on_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_until = time.time() + self.recovery_seconds

    # Return breaker status for observability endpoints.
    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        return {
            "failures": self.failures,
            "open": not self.allow(),
            "opened_until": self.opened_until,
            "retry_in_seconds": max(0.0, self.opened_until - now),
        }


registry_breaker = CircuitBreaker(CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_SECONDS)
agent_breakers: dict[str, CircuitBreaker] = {}


# Get request id from header/state or generate a new one
def _get_request_id(req: Request) -> str:
    return (
        req.headers.get("x-request-id")
        or getattr(req.state, "request_id", None)
        or str(uuid.uuid4())
    )


# Normalize math syntax before forwarding to specialists
def normalize_expression(expression: str) -> str:
    return expression.replace("^", "**").strip()


# Extract the first math-like expression from user input
def _extract_math_expression(user_input: str) -> str | None:
    for candidate in MATH_EXPR_RE.findall(user_input):
        trimmed = candidate.strip()
        if any(op in trimmed for op in MATH_OPERATORS) and any(ch.isdigit() for ch in trimmed):
            return normalize_expression(trimmed)

    return None


# Apply fast deterministic routing rules
def rule_based_route(user_input: str) -> dict[str, str] | None:
    expression = _extract_math_expression(user_input)
    if expression:
        return {"skill": "math", "payload": expression, "route_source": "rule"}

    return None


# Classify intent with the configured Ollama model
def _llm_classify_sync(user_input: str) -> dict[str, str]:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        format="json",
    )

    data = json.loads(response.message.content)

    skill = data.get("skill", "general")
    if skill not in {"general", "math"}:
        skill = "general"

    payload = data.get("payload") or user_input
    if skill == "math":
        payload = normalize_expression(payload)

    return {"skill": skill, "payload": payload, "route_source": "llm"}


# Choose route using rules first, then LLM, then fallback
async def select_route(user_input: str) -> dict[str, str]:
    policy_result = rule_based_route(user_input)
    if policy_result:
        logger.info("route_selected skill=%s source=rule", policy_result["skill"])

        return policy_result

    try:
        llm_result = await asyncio.wait_for(asyncio.to_thread(_llm_classify_sync, user_input), timeout=OLLAMA_TIMEOUT_S)
        logger.info("route_selected skill=%s source=llm", llm_result["skill"])

        return llm_result

    except Exception as exc:
        logger.warning("llm_classification_failed err=%s; fallback=general", exc)
        return {"skill": "general", "payload": user_input, "route_source": "fallback"}


# Execute a single registry discover call
async def _discover_once(skill: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=REGISTRY_TIMEOUT_S) as client:
        resp = await client.get(f"{REGISTRY_URL}/discover", params={"skill": skill})
        resp.raise_for_status()

        return resp.json()


# Discover a compatible agent with retries and circuit breaker
async def discover_agent(skill: str) -> dict | None:
    if not registry_breaker.allow():
        logger.warning("registry_circuit_open skill=%s", skill)
        return None

    last_error: Exception | None = None
    for attempt in range(REGISTRY_RETRIES + 1):
        try:
            agents = await _discover_once(skill)
            registry_breaker.on_success()

            return agents[0] if agents else None

        except Exception as exc:
            last_error = exc
            if attempt < REGISTRY_RETRIES:
                await asyncio.sleep(0.25 * (attempt + 1))
                continue

    registry_breaker.on_failure()
    logger.error("registry_discovery_failed skill=%s err=%s", skill, last_error)

    return None


# Return or create a per-agent circuit breaker
def _agent_breaker(agent_url: str) -> CircuitBreaker:
    if agent_url not in agent_breakers:
        agent_breakers[agent_url] = CircuitBreaker(CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_SECONDS)

    return agent_breakers[agent_url]


# Return first text found in a parts collection
def _first_text_from_parts(parts: Any) -> str | None:
    if not parts:
        return None

    for part in parts:
        text = getattr(getattr(part, "root", None), "text", None)
        if text:
            return text

    return None


# Return first text found in task artifacts
def _first_text_from_artifacts(task: Any) -> str | None:
    artifacts = getattr(task, "artifacts", None)
    if not artifacts:
        return None

    for artifact in artifacts:
        text = _first_text_from_parts(getattr(artifact, "parts", None))
        if text:
            return text

    return None


# Extract first text payload from A2A stream events
def _extract_text_from_event(event: Any) -> str | None:
    if isinstance(event, Message):
        return _first_text_from_parts(event.parts)

    task, _update = event
    if not task:
        return None

    artifact_text = _first_text_from_artifacts(task)
    if artifact_text:
        return artifact_text

    status = getattr(task, "status", None)
    status_message = getattr(status, "message", None)
    return _first_text_from_parts(getattr(status_message, "parts", None))


# Send one request to an agent and read first text response
async def _call_agent_once(agent_info: dict, payload: str) -> str | None:
    try:
        a2a_client = await ClientFactory.connect(agent_info["url"])
    except Exception:
        card = AgentCard.model_validate(agent_info["card"])
        a2a_client = await ClientFactory.connect(card)

    user_message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=payload))],
        message_id=str(uuid.uuid4()),
    )

    async for event in a2a_client.send_message(user_message):
        text = _extract_text_from_event(event)
        if text:
            return text

    return None


# Call an agent with timeout, retry and breaker protections
async def call_agent(agent_info: dict, payload: str) -> tuple[str | None, str | None]:
    agent_url = agent_info.get("url", "unknown")
    breaker = _agent_breaker(agent_url)

    if not breaker.allow():
        logger.warning("agent_circuit_open url=%s", agent_url)
        return None, "agent_circuit_open"

    last_error: Exception | None = None
    for attempt in range(AGENT_RETRIES + 1):
        try:
            result = await asyncio.wait_for(_call_agent_once(agent_info, payload), timeout=AGENT_TIMEOUT_S)
            if result:
                breaker.on_success()
                return result, None
            last_error = RuntimeError("empty_agent_response")
        except Exception as exc:
            last_error = exc

        if attempt < AGENT_RETRIES:
            await asyncio.sleep(0.25 * (attempt + 1))

    breaker.on_failure()
    logger.error("agent_call_failed url=%s err=%s", agent_url, last_error)

    return None, "agent_call_failed"


# Optionally rephrase math outputs into natural language
def _format_response_sync(user_input: str, agent_result: str, skill: str) -> str:
    if skill != "math":
        return agent_result

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. The user asked a math question and we computed the result. Provide a brief, natural response incorporating the result. Be concise."},
            {"role": "user", "content": f"User asked: {user_input}\nComputed result: {agent_result}"},
        ],
    )

    return response.message.content


# Format final response asynchronously with safe fallback
async def format_response(user_input: str, agent_result: str, skill: str) -> str:
    if skill != "math":
        return agent_result
    try:
        return await asyncio.wait_for(asyncio.to_thread(_format_response_sync, user_input, agent_result, skill), timeout=OLLAMA_TIMEOUT_S)

    except Exception as exc:
        logger.warning("response_formatting_failed err=%s", exc)
        return agent_result


# Pydantic models
class InferenceRequest(BaseModel):
    message: str

class InferenceResponse(BaseModel):
    response: str
    agent: str
    request_id: str
    status: str = "ok"
    route_source: str = "unknown"
    error_code: str | None = None


# App init
app = FastAPI(title=APP_NAME, version=APP_VERSION)


# Health API
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "registry_url": REGISTRY_URL,
        "registry_breaker": registry_breaker.snapshot(),
        "agent_breakers": {url: breaker.snapshot() for url, breaker in agent_breakers.items()},
    }


# Inference API - Orchestrate routing, discovery and delegation
@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, req: Request, response: Response):
    request_id = _get_request_id(req)
    response.headers["x-request-id"] = request_id
    route = await select_route(request.message)
    skill = route.get("skill", "general")
    payload = route.get("payload", request.message)
    route_source = route.get("route_source", "unknown")

    agent_info = await discover_agent(skill)
    if not agent_info:
        return InferenceResponse(
            response=f"No agent found for skill '{skill}'.",
            agent="none",
            request_id=request_id,
            status="fallback",
            route_source=route_source,
            error_code="no_agent_found",
        )

    agent_name = agent_info.get("card", {}).get("name", agent_info.get("url", "unknown"))
    logger.info("routing_to_agent name=%s url=%s skill=%s", agent_name, agent_info.get("url"), skill)

    result, error_code = await call_agent(agent_info, payload)
    if not result:
        return InferenceResponse(
            response=f"Agent '{agent_name}' did not return a result.",
            agent=agent_name,
            request_id=request_id,
            status="fallback",
            route_source=route_source,
            error_code=error_code or "agent_no_result",
        )

    response_text = await format_response(request.message, result, skill)

    return InferenceResponse(
        response=response_text,
        agent=agent_name,
        request_id=request_id,
        status="ok",
        route_source=route_source,
    )


# App entrypoint
def main() -> None:
    logger.info("#####################################################")
    logger.info("#            ___                            _       #")
    logger.info("#      /\\   |__ \\    /\\                    | |      #")
    logger.info("#     /  \\     ) |  /  \\      ___  _ __ ___| |__    #")
    logger.info("#    / /\\ \\   / /  / /\\ \\    / _ \\| '__/ __| '_ \\   #")
    logger.info("#   / ____ \\ / /_ / ____ \\  | (_) | | | (__| | | |  #")
    logger.info("#  /_/    \\_\\____/_/    \\_\\  \\___/|_|  \\___|_| |_|  #")
    logger.info("#                                                   #")
    logger.info("#        alessandro.orru <at> aleostudio.com        #")
    logger.info("#                                                   #")
    logger.info("#####################################################")

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)


# App launch if invoked directly
if __name__ == "__main__":
    main()
