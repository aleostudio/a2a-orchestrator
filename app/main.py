from typing import Any
import asyncio
import json
import re
import uuid
import ollama as ollama_client
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from app.a2a import a2a_service
from app.config import APP_HOST, APP_NAME, APP_PORT, APP_VERSION, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_S
from app.logger import logger


# Ollama client
ollama = ollama_client.Client(host=OLLAMA_BASE_URL)


# Routing policy
MATH_OPERATORS = ("+", "-", "*", "/", "%", "^", "(", ")")
MATH_EXPR_RE = re.compile(r"[0-9\s\+\-\*\/\%\(\)\.^]{3,}")


# Get request id from header/state or generate a new one
def _get_request_id(req: Request) -> str:
    return req.headers.get("x-request-id") or getattr(req.state, "request_id", None) or str(uuid.uuid4())


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


# Build unique sorted skill tags from registry skills payload
def _available_skill_tags(registry_skills: list[dict[str, Any]]) -> list[str]:
    tags: set[str] = set()
    for skill in registry_skills:
        for tag in skill.get("tags", []):
            if isinstance(tag, str) and tag.strip():
                tags.add(tag.strip().lower())

    return sorted(tags)


# Build classifier prompt from live registry skills
def _build_intent_system_prompt(registry_skills: list[dict[str, Any]], fallback_skill: str) -> str:
    if not registry_skills:
        return (
            "You are an intent classifier. "
            "Respond only with valid JSON in this format: "
            '{"skill": "general", "payload": "<original user message>"}. '
            "Always use skill='general'."
        )

    skills_lines: list[str] = []
    for skill in registry_skills:
        tags = ", ".join(skill.get("tags", []))
        skills_lines.append(
            f"- agent: {skill.get('agent', 'unknown')} | "
            f"skill: {skill.get('name', 'unknown')} | "
            f"tags: [{tags}] | "
            f"description: {skill.get('description', '')}"
        )

    return (
        "You are an intent classifier for an A2A orchestrator.\n"
        "Choose exactly one tag from the available tags listed below.\n"
        "Return only valid JSON with this schema:\n"
        '{"skill": "<one_tag>", "payload": "<text to send to the chosen agent>"}\n\n'
        "Rules:\n"
        "- Use only one of the provided tags.\n"
        "- Keep payload concise and aligned to the selected skill.\n"
        f"- If uncertain, use fallback tag '{fallback_skill}'.\n\n"
        "Available skills from registry:\n"
        + "\n".join(skills_lines)
    )


# Classify intent with the configured Ollama model
def _llm_classify_sync(user_input: str, system_prompt: str, allowed_tags: set[str], fallback_skill: str) -> dict[str, str]:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        format="json",
    )

    data = json.loads(response.message.content)

    skill = str(data.get("skill", fallback_skill)).strip().lower()
    if skill not in allowed_tags:
        skill = fallback_skill

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

    fallback_skill = "general"
    try:
        registry_skills = await a2a_service.list_skills()
        available_tags = _available_skill_tags(registry_skills)
        if available_tags:
            fallback_skill = "general" if "general" in available_tags else available_tags[0]

        system_prompt = _build_intent_system_prompt(registry_skills, fallback_skill)
        llm_result = await asyncio.wait_for(
            asyncio.to_thread(
                _llm_classify_sync,
                user_input,
                system_prompt,
                set(available_tags) if available_tags else {fallback_skill},
                fallback_skill,
            ),
            timeout=OLLAMA_TIMEOUT_S,
        )
        logger.info("route_selected skill=%s source=llm", llm_result["skill"])
        return llm_result

    except Exception as exc:
        logger.warning("llm_classification_failed err=%s; fallback=%s", exc, fallback_skill)
        return {"skill": fallback_skill, "payload": user_input, "route_source": "fallback"}


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


# Expose service health and breaker snapshots
@app.get("/health")
async def health():
    return {"status": "ok", **a2a_service.health_snapshot()}


# Orchestrate routing, discovery, delegation and final reply
@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, req: Request, response: Response):
    request_id = _get_request_id(req)
    response.headers["x-request-id"] = request_id
    route = await select_route(request.message)
    skill = route.get("skill", "general")
    payload = route.get("payload", request.message)
    route_source = route.get("route_source", "unknown")

    agent_info = await a2a_service.discover_agent(skill)
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

    result, error_code = await a2a_service.call_agent(agent_info, payload)
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


# Start the API server and print startup banner
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
