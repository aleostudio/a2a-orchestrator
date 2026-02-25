from typing import Any
import asyncio
from contextlib import asynccontextmanager
import json
import re
import time
import uuid
import ollama as ollama_client
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from app.a2a import a2a_service
from app.config import (APP_HOST, APP_NAME, APP_PORT, APP_VERSION, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_S, ROUTE_CONFIDENCE_THRESHOLD, SKILLS_CACHE_REFRESH_INTERVAL_S, SKILLS_CACHE_TTL_S)
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


# Build unique sorted canonical skill IDs from registry skills payload.
def _available_skill_ids(registry_skills: list[dict[str, Any]]) -> list[str]:
    skill_ids: set[str] = set()
    for skill in registry_skills:
        skill_id = skill.get("skill_id")
        if isinstance(skill_id, str) and skill_id.strip():
            skill_ids.add(skill_id.strip().lower())

    return sorted(skill_ids)


# Select the default fallback skill ID from registry catalog.
def _fallback_skill_id(registry_skills: list[dict[str, Any]], available_skill_ids: list[str]) -> str:
    for skill in registry_skills:
        if skill.get("route_tag") == "general":
            candidate = skill.get("skill_id")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().lower()

    if available_skill_ids:
        return available_skill_ids[0]

    return "fallback.general"


# Build classifier prompt from live registry skills
def _build_intent_system_prompt(registry_skills: list[dict[str, Any]], fallback_skill_id: str) -> str:
    if not registry_skills:
        return (
            "You are an intent classifier. "
            "Respond only with valid JSON in this format: "
            '{"skill_id": "fallback.general", "payload": "<original user message>"}. '
            "Always use skill_id='fallback.general'."
        )

    skills_lines: list[str] = []
    for skill in registry_skills:
        skills_lines.append(
            f"- skill_id: {skill.get('skill_id', 'unknown')} | "
            f"- agent: {skill.get('agent', 'unknown')} | "
            f"skill: {skill.get('name', 'unknown')} | "
            f"route_tag: {skill.get('route_tag', 'unknown')} | "
            f"description: {skill.get('description', '')}"
        )

    return (
        "You are an intent classifier for an A2A orchestrator.\n"
        "Choose exactly one skill_id from the available skills listed below.\n"
        "Return only valid JSON with this exact schema:\n"
        '{"skill_id": "<one_skill_id>", "payload": "<text to send to the chosen agent>", "confidence": <0_to_1>, "reason": "<short rationale>", "needs_clarification": <true_or_false>}\n\n'
        "Rules:\n"
        "- Use only one of the provided skill_id values.\n"
        "- confidence must be a float between 0 and 1.\n"
        "- Set needs_clarification=true when the request is ambiguous, multi-intent, or missing key details.\n"
        "- Keep payload concise and preserve user constraints and data.\n"
        f"- If uncertain, choose fallback skill_id '{fallback_skill_id}'.\n\n"
        "Available skills from registry:\n"
        + "\n".join(skills_lines)
    )


# Parse a float-like value with bounds fallback.
def _coerce_confidence(raw_confidence: Any) -> float:
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return 0.0

    if confidence < 0.0:
        return 0.0

    if confidence > 1.0:
        return 1.0

    return confidence


# Keep a refreshed in-memory cache for classifier skills and prompt.
class SkillsPromptCache:

    # Initialize cache settings and default classifier context.
    def __init__(self, ttl_seconds: float, refresh_interval_seconds: float):
        self.ttl_seconds = max(1.0, ttl_seconds)
        self.refresh_interval_seconds = max(0.0, refresh_interval_seconds)
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None
        self._poller_task: asyncio.Task | None = None
        self._startup_refresh_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._prompt = _build_intent_system_prompt([], "fallback.general")
        self._allowed_skill_ids: set[str] = {"fallback.general"}
        self._fallback_skill_id = "fallback.general"
        self._skill_id_to_route_tag: dict[str, str] = {"fallback.general": "general"}
        self._updated_at_monotonic = 0.0
        self._last_refresh_error: str | None = None


    # Return whether cache content is still valid by TTL.
    def _is_fresh_locked(self) -> bool:
        if self._updated_at_monotonic <= 0.0:
            return False

        return (time.monotonic() - self._updated_at_monotonic) <= self.ttl_seconds


    # Refresh cache data by fetching skills from registry.
    async def _refresh_once(self) -> None:
        registry_skills = await a2a_service.list_skills()
        available_skill_ids = _available_skill_ids(registry_skills)
        fallback_skill_id = _fallback_skill_id(registry_skills, available_skill_ids)
        allowed_skill_ids = set(available_skill_ids) if available_skill_ids else {fallback_skill_id}
        skill_id_to_route_tag = {
            skill["skill_id"]: skill["route_tag"]
            for skill in registry_skills
            if isinstance(skill.get("skill_id"), str) and isinstance(skill.get("route_tag"), str)
        }
        if fallback_skill_id not in skill_id_to_route_tag:
            skill_id_to_route_tag[fallback_skill_id] = "general"
        prompt = _build_intent_system_prompt(registry_skills, fallback_skill_id)

        async with self._lock:
            self._prompt = prompt
            self._allowed_skill_ids = allowed_skill_ids
            self._fallback_skill_id = fallback_skill_id
            self._skill_id_to_route_tag = skill_id_to_route_tag
            self._updated_at_monotonic = time.monotonic()
            self._last_refresh_error = None

        logger.info(
            "Agents skills cache refreshed: skills=%s allowed_skill_ids=%s fallback=%s",
            len(registry_skills),
            len(allowed_skill_ids),
            fallback_skill_id,
        )


    # Refresh cache once when stale or when forced.
    async def refresh(self, force: bool = False) -> None:
        async with self._lock:
            needs_refresh = force or not self._is_fresh_locked()
            if not needs_refresh:
                return
            if self._refresh_task is None:
                self._refresh_task = asyncio.create_task(self._refresh_once())
            task = self._refresh_task

        try:
            await task
        except Exception as exc:
            async with self._lock:
                self._last_refresh_error = str(exc)
            logger.warning("skills_prompt_cache_refresh_failed err=%s", exc)
        finally:
            async with self._lock:
                if self._refresh_task is task:
                    self._refresh_task = None


    # Return classifier context ensuring TTL-based freshness.
    async def get_classifier_context(self) -> tuple[str, set[str], str, dict[str, str]]:
        await self.refresh(force=False)
        async with self._lock:
            return (
                self._prompt,
                set(self._allowed_skill_ids),
                self._fallback_skill_id,
                dict(self._skill_id_to_route_tag),
            )


    # Return cache snapshot for readiness checks and diagnostics.
    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            age_seconds = None
            if self._updated_at_monotonic > 0.0:
                age_seconds = max(0.0, time.monotonic() - self._updated_at_monotonic)

            return {
                "initialized": self._updated_at_monotonic > 0.0,
                "age_seconds": age_seconds,
                "ttl_seconds": self.ttl_seconds,
                "allowed_skill_ids_count": len(self._allowed_skill_ids),
                "fallback_skill_id": self._fallback_skill_id,
                "last_refresh_error": self._last_refresh_error,
            }


    # Start periodic background refresh task.
    async def start(self) -> None:
        if self.refresh_interval_seconds <= 0:
            return
        async with self._lock:
            if self._poller_task is not None:
                return

            self._stop_event.clear()
            self._poller_task = asyncio.create_task(self._poll_loop())


    # Schedule a one-shot refresh task and keep a strong reference.
    async def schedule_startup_refresh(self) -> None:
        async with self._lock:
            if self._startup_refresh_task is not None and not self._startup_refresh_task.done():
                return

            self._startup_refresh_task = asyncio.create_task(self.refresh(force=True))


    # Stop periodic background refresh task.
    async def stop(self) -> None:
        async with self._lock:
            task = self._poller_task
            startup_task = self._startup_refresh_task
            self._poller_task = None
            self._startup_refresh_task = None
            self._stop_event.set()

        if task is not None:
            task.cancel()
            results = await asyncio.gather(task, return_exceptions=True)
            error = results[0]
            if isinstance(error, Exception) and not isinstance(error, asyncio.CancelledError):
                logger.warning("skills_prompt_cache_poller_stop_failed err=%s", error)

        if startup_task is not None:
            startup_task.cancel()
            results = await asyncio.gather(startup_task, return_exceptions=True)
            error = results[0]
            if isinstance(error, Exception) and not isinstance(error, asyncio.CancelledError):
                logger.warning("skills_prompt_cache_startup_task_stop_failed err=%s", error)


    # Periodically refresh cache while app is running.
    async def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.refresh_interval_seconds)
                break
            except asyncio.TimeoutError:
                await self.refresh(force=True)


skills_prompt_cache = SkillsPromptCache(
    ttl_seconds=SKILLS_CACHE_TTL_S,
    refresh_interval_seconds=SKILLS_CACHE_REFRESH_INTERVAL_S,
)


# Classify intent with the configured Ollama model
def _llm_classify_sync(user_input: str, system_prompt: str, allowed_skill_ids: set[str], fallback_skill_id: str, skill_id_to_route_tag: dict[str, str], min_confidence: float) -> dict[str, str]:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        format="json",
    )

    data = json.loads(response.message.content)

    skill_id = str(data.get("skill_id", fallback_skill_id)).strip().lower()
    payload = str(data.get("payload") or user_input)
    confidence = _coerce_confidence(data.get("confidence"))
    needs_clarification = bool(data.get("needs_clarification", False))

    use_fallback = skill_id not in allowed_skill_ids or confidence < min_confidence or needs_clarification
    fallback_route_tag = skill_id_to_route_tag.get(fallback_skill_id, "general")
    if use_fallback:
        logger.info("LLM guardrail fallback selected '%s', confidence %.2f, needs clarification '%s', fallback '%s'", skill_id, confidence, needs_clarification, fallback_skill_id)

        return {
            "skill_id": fallback_skill_id,
            "skill": fallback_route_tag,
            "payload": user_input,
            "route_source": "llm_guardrail",
        }

    route_tag = skill_id_to_route_tag.get(skill_id, fallback_route_tag)
    if route_tag == "math":
        payload = normalize_expression(payload)

    return {"skill_id": skill_id, "skill": route_tag, "payload": payload, "route_source": "llm"}


# Choose route using rules first, then LLM, then fallback
async def select_route(user_input: str) -> dict[str, str]:
    policy_result = rule_based_route(user_input)
    if policy_result:
        logger.info("Route selected: skill '%s', source 'rule'", policy_result["skill"])
        return policy_result

    fallback_skill_id = "fallback.general"
    try:
        system_prompt, allowed_skill_ids, fallback_skill_id, skill_id_to_route_tag = await skills_prompt_cache.get_classifier_context()
        llm_result = await asyncio.wait_for(
            asyncio.to_thread(
                _llm_classify_sync,
                user_input,
                system_prompt,
                allowed_skill_ids,
                fallback_skill_id,
                skill_id_to_route_tag,
                ROUTE_CONFIDENCE_THRESHOLD,
            ),
            timeout=OLLAMA_TIMEOUT_S,
        )
        logger.info("Route selected: skill_id '%s', route_tag '%s', source 'LLM'", llm_result.get("skill_id"), llm_result.get("skill"))

        return llm_result

    except Exception as exc:
        logger.warning("LLM classification failed: err=%s; fallback=%s", exc, fallback_skill_id)
        return {"skill_id": fallback_skill_id, "skill": "general", "payload": user_input, "route_source": "fallback"}


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


# Manage startup and shutdown resources for the API app.
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await skills_prompt_cache.start()
    await skills_prompt_cache.schedule_startup_refresh()
    try:
        yield
    finally:
        await skills_prompt_cache.stop()


# App init
app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)


# Expose service health and breaker snapshots
@app.get("/health")
async def health():
    return {"status": "ok", **a2a_service.health_snapshot()}


# Expose readiness state based on registry breaker and skills cache.
@app.get("/ready")
async def ready(response: Response):
    await skills_prompt_cache.refresh(force=False)
    cache_snapshot = await skills_prompt_cache.snapshot()
    registry_open = not a2a_service.registry_breaker.allow()
    is_ready = cache_snapshot["initialized"] and not registry_open
    if not is_ready:
        response.status_code = 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "registry_circuit_open": registry_open,
        "skills_cache": cache_snapshot,
    }


# Orchestrate routing, discovery, delegation and final reply
@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, req: Request, response: Response):
    request_id = _get_request_id(req)
    response.headers["x-request-id"] = request_id
    route = await select_route(request.message)
    skill = route.get("skill", "general")
    _skill_id = route.get("skill_id", "unknown")
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
    logger.info("Routing to agent: '%s' (%s): skill '%s'", agent_name, agent_info.get("url"), skill)

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

    return InferenceResponse(
        response=result,
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
