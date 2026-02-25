from typing import Any
import asyncio
import re
import time
import uuid
import httpx
from a2a.client.client_factory import ClientFactory
from a2a.types import AgentCard, Message, Part, Role, TextPart
from app.config import (AGENT_RETRIES, AGENT_TIMEOUT_S, CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_SECONDS, REGISTRY_RETRIES, REGISTRY_TIMEOUT_S, REGISTRY_URL)
from app.logger import logger


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return clean.strip("-") or "unknown"


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


class A2AService:

    # Initialize registry and per-agent breaker state.
    def __init__(self):
        self.registry_breaker = CircuitBreaker(CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_SECONDS)
        self.agent_breakers: dict[str, CircuitBreaker] = {}


    # Return health snapshot including breaker states.
    def health_snapshot(self) -> dict[str, Any]:
        return {
            "registry_url": REGISTRY_URL,
            "registry_breaker": self.registry_breaker.snapshot(),
            "agent_breakers": {url: breaker.snapshot() for url, breaker in self.agent_breakers.items()},
        }


    # Execute a single registry discover call.
    async def _discover_once(self, skill: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=REGISTRY_TIMEOUT_S) as client:
            resp = await client.get(f"{REGISTRY_URL}/discover", params={"skill": skill})
            resp.raise_for_status()

            return resp.json()


    # Execute a single registry agents call
    async def _list_agents_once(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=REGISTRY_TIMEOUT_S) as client:
            resp = await client.get(f"{REGISTRY_URL}/agents")
            resp.raise_for_status()

            return resp.json()


    # Discover a compatible agent with retries and circuit breaker
    async def discover_agent(self, skill: str) -> dict[str, Any] | None:
        if not self.registry_breaker.allow():
            logger.warning("registry_circuit_open skill=%s", skill)
            return None

        last_error: Exception | None = None
        for attempt in range(REGISTRY_RETRIES + 1):
            try:
                agents = await self._discover_once(skill)
                self.registry_breaker.on_success()
                return agents[0] if agents else None
            except Exception as exc:
                last_error = exc
                if attempt < REGISTRY_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))

        self.registry_breaker.on_failure()
        logger.error("registry_discovery_failed skill=%s err=%s", skill, last_error)

        return None


    # List all registered agents with retries and breaker protection
    async def list_agents(self) -> list[dict[str, Any]]:
        if not self.registry_breaker.allow():
            logger.warning("registry_circuit_open action=list_agents")
            return []

        last_error: Exception | None = None
        for attempt in range(REGISTRY_RETRIES + 1):
            try:
                agents = await self._list_agents_once()
                self.registry_breaker.on_success()
                return agents
            except Exception as exc:
                last_error = exc
                if attempt < REGISTRY_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))

        self.registry_breaker.on_failure()
        logger.error("registry_list_agents_failed err=%s", last_error)

        return []


    # Extract registry skills for dynamic intent classification prompts
    async def list_skills(self) -> list[dict[str, Any]]:
        agents = await self.list_agents()
        skills: list[dict[str, Any]] = []

        for agent in agents:
            card = agent.get("card", {})
            agent_name = card.get("name", "unknown")
            agent_slug = _slug(agent_name)
            for skill in card.get("skills", []):
                tags = [str(tag).strip().lower() for tag in skill.get("tags", []) if str(tag).strip()]
                if not tags:
                    continue
                skill_name = str(skill.get("name", "unknown"))
                skill_slug = _slug(skill_name)
                deduped_tags = sorted(set(tags))
                description = skill.get("description", "")
                for tag in deduped_tags:
                    skill_id = f"{agent_slug}.{skill_slug}.{_slug(tag)}"
                    skills.append(
                        {
                            "skill_id": skill_id,
                            "route_tag": tag,
                            "agent": agent_name,
                            "name": skill_name,
                            "description": description,
                            "tags": deduped_tags,
                        }
                    )

        return skills


    # Return or create a per-agent circuit breaker
    def _agent_breaker(self, agent_url: str) -> CircuitBreaker:
        if agent_url not in self.agent_breakers:
            self.agent_breakers[agent_url] = CircuitBreaker(CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_SECONDS)
        return self.agent_breakers[agent_url]


    # Return first text found in a parts collection
    def _first_text_from_parts(self, parts: Any) -> str | None:
        if not parts:
            return None

        for part in parts:
            text = getattr(getattr(part, "root", None), "text", None)
            if text:
                return text

        return None


    # Return first text found in task artifacts
    def _first_text_from_artifacts(self, task: Any) -> str | None:
        artifacts = getattr(task, "artifacts", None)
        if not artifacts:
            return None

        for artifact in artifacts:
            text = self._first_text_from_parts(getattr(artifact, "parts", None))
            if text:
                return text

        return None


    # Extract first text payload from A2A stream events
    def _extract_text_from_event(self, event: Any) -> str | None:
        if isinstance(event, Message):
            return self._first_text_from_parts(event.parts)

        task, _update = event
        if not task:
            return None

        artifact_text = self._first_text_from_artifacts(task)
        if artifact_text:
            return artifact_text

        status = getattr(task, "status", None)
        status_message = getattr(status, "message", None)

        return self._first_text_from_parts(getattr(status_message, "parts", None))


    # Send one request to an agent and read first text response
    async def _call_agent_once(self, agent_info: dict[str, Any], payload: str) -> str | None:
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
            text = self._extract_text_from_event(event)
            if text:
                return text

        return None


    # Call an agent with timeout, retry and breaker protections.
    async def call_agent(self, agent_info: dict[str, Any], payload: str) -> tuple[str | None, str | None]:
        agent_url = agent_info.get("url", "unknown")
        breaker = self._agent_breaker(agent_url)

        if not breaker.allow():
            logger.warning("agent_circuit_open url=%s", agent_url)
            return None, "agent_circuit_open"

        last_error: Exception | None = None
        for attempt in range(AGENT_RETRIES + 1):
            try:
                result = await asyncio.wait_for(self._call_agent_once(agent_info, payload), timeout=AGENT_TIMEOUT_S)
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


# Shared service instance
a2a_service = A2AService()
