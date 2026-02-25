import asyncio
import time
from types import SimpleNamespace

import pytest

from app.a2a import A2AService


@pytest.fixture
def service() -> A2AService:
    return A2AService()


def test_list_skills_extracts_tags(service, monkeypatch):
    async def fake_list_agents():
        return [
            {
                "url": "http://localhost:8001",
                "card": {
                    "name": "General Assistant",
                    "skills": [
                        {
                            "name": "General Knowledge",
                            "description": "General tasks",
                            "tags": ["general", "qa"],
                        }
                    ],
                },
            }
        ]

    monkeypatch.setattr(service, "list_agents", fake_list_agents)
    skills = asyncio.run(service.list_skills())

    assert len(skills) == 1
    assert skills[0]["agent"] == "General Assistant"
    assert skills[0]["name"] == "General Knowledge"
    assert skills[0]["tags"] == ["general", "qa"]


def test_discover_agent_returns_first_result(service, monkeypatch):
    async def fake_discover_once(_skill: str):
        return [
            {"url": "http://agent-1", "card": {"name": "Agent 1"}},
            {"url": "http://agent-2", "card": {"name": "Agent 2"}},
        ]

    monkeypatch.setattr(service, "_discover_once", fake_discover_once)
    agent = asyncio.run(service.discover_agent("general"))
    assert agent == {"url": "http://agent-1", "card": {"name": "Agent 1"}}
    assert service.registry_breaker.failures == 0


def test_discover_agent_opens_circuit_after_failures(service, monkeypatch):
    async def fake_discover_once(_skill: str):
        raise RuntimeError("registry down")

    monkeypatch.setattr(service, "_discover_once", fake_discover_once)
    agent = asyncio.run(service.discover_agent("general"))

    assert agent is None
    assert service.registry_breaker.failures == 1


def test_discover_agent_short_circuits_when_breaker_open(service, monkeypatch):
    service.registry_breaker.opened_until = time.time() + 60

    async def fake_discover_once(_skill: str):
        pytest.fail("discover should not run when breaker is open")

    monkeypatch.setattr(service, "_discover_once", fake_discover_once)
    assert asyncio.run(service.discover_agent("general")) is None


def test_call_agent_returns_error_when_breaker_open(service):
    agent_info = {"url": "http://agent-1", "card": {"name": "Agent 1"}}
    breaker = service._agent_breaker(agent_info["url"])
    breaker.opened_until = time.time() + 60

    result, error_code = asyncio.run(service.call_agent(agent_info, "payload"))
    assert result is None
    assert error_code == "agent_circuit_open"


def test_call_agent_success(service, monkeypatch):
    async def fake_call_once(_agent_info: dict, _payload: str):
        return "ok"

    monkeypatch.setattr(service, "_call_agent_once", fake_call_once)

    result, error_code = asyncio.run(
        service.call_agent({"url": "http://agent-1", "card": {"name": "Agent 1"}}, "payload")
    )
    assert result == "ok"
    assert error_code is None


def test_call_agent_failure_returns_fallback_error(service, monkeypatch):
    async def fake_call_once(_agent_info: dict, _payload: str):
        raise RuntimeError("agent failed")

    monkeypatch.setattr(service, "_call_agent_once", fake_call_once)

    result, error_code = asyncio.run(
        service.call_agent({"url": "http://agent-1", "card": {"name": "Agent 1"}}, "payload")
    )
    assert result is None
    assert error_code == "agent_call_failed"


def test_extract_text_from_event_reads_status_message(service):
    part = SimpleNamespace(root=SimpleNamespace(text="hello from status"))
    status_message = SimpleNamespace(parts=[part])
    status = SimpleNamespace(message=status_message)
    task = SimpleNamespace(artifacts=[], status=status)

    extracted = service._extract_text_from_event((task, None))
    assert extracted == "hello from status"
