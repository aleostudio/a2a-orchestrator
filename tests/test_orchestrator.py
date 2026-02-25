import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main as orchestrator


@pytest.fixture(autouse=True)
def reset_breakers():
    orchestrator.registry_breaker.failures = 0
    orchestrator.registry_breaker.opened_until = 0.0
    orchestrator.agent_breakers.clear()


def test_normalize_expression_converts_caret():
    assert orchestrator.normalize_expression(" 2^8 ") == "2**8"


def test_rule_based_route_extracts_math_expression():
    route = orchestrator.rule_based_route("what is 125 * 37 + 89?")
    assert route is not None
    assert route["skill"] == "math"
    assert route["payload"] == "125 * 37 + 89"
    assert route["route_source"] == "rule"


def test_rule_based_route_returns_none_for_non_math():
    assert orchestrator.rule_based_route("tell me about python") is None


def test_select_route_uses_rule_first(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_llm_classify_sync",
        lambda _msg: pytest.fail("LLM classifier should not be called for rule hits"),
    )

    route = asyncio.run(orchestrator.select_route("1 + 1"))
    assert route["route_source"] == "rule"
    assert route["skill"] == "math"


def test_select_route_uses_llm_when_rule_misses(monkeypatch):
    monkeypatch.setattr(orchestrator, "rule_based_route", lambda _msg: None)
    monkeypatch.setattr(
        orchestrator,
        "_llm_classify_sync",
        lambda _msg: {"skill": "general", "payload": "hello", "route_source": "llm"},
    )

    route = asyncio.run(orchestrator.select_route("hello"))
    assert route == {"skill": "general", "payload": "hello", "route_source": "llm"}


def test_select_route_falls_back_when_llm_fails(monkeypatch):
    monkeypatch.setattr(orchestrator, "rule_based_route", lambda _msg: None)

    def _raise(_msg):
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(orchestrator, "_llm_classify_sync", _raise)

    route = asyncio.run(orchestrator.select_route("hello world"))
    assert route == {"skill": "general", "payload": "hello world", "route_source": "fallback"}


def test_discover_agent_returns_first_result(monkeypatch):
    async def fake_discover_once(_skill: str):
        return [
            {"url": "http://agent-1", "card": {"name": "Agent 1"}},
            {"url": "http://agent-2", "card": {"name": "Agent 2"}},
        ]

    monkeypatch.setattr(orchestrator, "_discover_once", fake_discover_once)
    agent = asyncio.run(orchestrator.discover_agent("general"))
    assert agent == {"url": "http://agent-1", "card": {"name": "Agent 1"}}
    assert orchestrator.registry_breaker.failures == 0


def test_discover_agent_opens_circuit_after_retries(monkeypatch):
    async def fake_discover_once(_skill: str):
        raise RuntimeError("registry down")

    monkeypatch.setattr(orchestrator, "_discover_once", fake_discover_once)
    monkeypatch.setattr(orchestrator, "REGISTRY_RETRIES", 0)

    agent = asyncio.run(orchestrator.discover_agent("general"))
    assert agent is None
    assert orchestrator.registry_breaker.failures == 1


def test_discover_agent_short_circuits_when_breaker_open(monkeypatch):
    orchestrator.registry_breaker.opened_until = time.time() + 60

    async def fake_discover_once(_skill: str):
        pytest.fail("discover should not be called when breaker is open")

    monkeypatch.setattr(orchestrator, "_discover_once", fake_discover_once)
    assert asyncio.run(orchestrator.discover_agent("general")) is None


def test_call_agent_returns_error_when_breaker_open():
    agent_info = {"url": "http://agent-1", "card": {"name": "Agent 1"}}
    breaker = orchestrator._agent_breaker(agent_info["url"])
    breaker.opened_until = time.time() + 60

    result, error_code = asyncio.run(orchestrator.call_agent(agent_info, "payload"))
    assert result is None
    assert error_code == "agent_circuit_open"


def test_call_agent_success(monkeypatch):
    async def fake_call_once(_agent_info: dict, _payload: str):
        return "ok"

    monkeypatch.setattr(orchestrator, "_call_agent_once", fake_call_once)

    result, error_code = asyncio.run(
        orchestrator.call_agent({"url": "http://agent-1", "card": {"name": "Agent 1"}}, "payload")
    )
    assert result == "ok"
    assert error_code is None


def test_call_agent_failure_returns_fallback_error(monkeypatch):
    async def fake_call_once(_agent_info: dict, _payload: str):
        raise RuntimeError("agent failed")

    monkeypatch.setattr(orchestrator, "_call_agent_once", fake_call_once)
    monkeypatch.setattr(orchestrator, "AGENT_RETRIES", 0)

    result, error_code = asyncio.run(
        orchestrator.call_agent({"url": "http://agent-1", "card": {"name": "Agent 1"}}, "payload")
    )
    assert result is None
    assert error_code == "agent_call_failed"


def test_format_response_passthrough_for_general_skill():
    assert asyncio.run(orchestrator.format_response("hello", "agent output", "general")) == "agent output"


def test_format_response_math_fallback_when_llm_fails(monkeypatch):
    def fake_format_sync(_user_input: str, _agent_result: str, _skill: str):
        raise RuntimeError("ollama timeout")

    monkeypatch.setattr(orchestrator, "_format_response_sync", fake_format_sync)
    result = asyncio.run(orchestrator.format_response("2+2", "4", "math"))
    assert result == "4"


def test_extract_text_from_event_tuple_status_message():
    part = SimpleNamespace(root=SimpleNamespace(text="hello from status"))
    status_message = SimpleNamespace(parts=[part])
    status = SimpleNamespace(message=status_message)
    task = SimpleNamespace(artifacts=[], status=status)

    extracted = orchestrator._extract_text_from_event((task, None))
    assert extracted == "hello from status"


def test_health_endpoint_includes_breakers():
    client = TestClient(orchestrator.app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "registry_breaker" in body
    assert "agent_breakers" in body


def test_inference_returns_fallback_when_no_agent(monkeypatch):
    async def fake_select_route(_message: str):
        return {"skill": "general", "payload": "hello", "route_source": "rule"}

    async def fake_discover_agent(_skill: str):
        return None

    monkeypatch.setattr(orchestrator, "select_route", fake_select_route)
    monkeypatch.setattr(orchestrator, "discover_agent", fake_discover_agent)

    client = TestClient(orchestrator.app)
    response = client.post("/inference", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "fallback"
    assert body["error_code"] == "no_agent_found"
    assert body["agent"] == "none"
    assert body["request_id"]
    assert response.headers.get("x-request-id") == body["request_id"]


def test_inference_returns_fallback_when_agent_fails(monkeypatch):
    async def fake_select_route(_message: str):
        return {"skill": "general", "payload": "hello", "route_source": "rule"}

    async def fake_discover_agent(_skill: str):
        return {"url": "http://localhost:9300", "card": {"name": "General Assistant"}}

    async def fake_call_agent(_agent_info: dict, _payload: str):
        return None, "agent_call_failed"

    monkeypatch.setattr(orchestrator, "select_route", fake_select_route)
    monkeypatch.setattr(orchestrator, "discover_agent", fake_discover_agent)
    monkeypatch.setattr(orchestrator, "call_agent", fake_call_agent)

    client = TestClient(orchestrator.app)
    response = client.post("/inference", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "fallback"
    assert body["error_code"] == "agent_call_failed"
    assert body["agent"] == "General Assistant"


def test_inference_success(monkeypatch):
    async def fake_select_route(_message: str):
        return {"skill": "general", "payload": "hello", "route_source": "rule"}

    async def fake_discover_agent(_skill: str):
        return {
            "url": "http://localhost:9300",
            "card": {"name": "General Assistant"},
        }

    async def fake_call_agent(_agent_info: dict, _payload: str):
        return "Hi from agent", None

    async def fake_format_response(_user_input: str, agent_result: str, _skill: str):
        return agent_result

    monkeypatch.setattr(orchestrator, "select_route", fake_select_route)
    monkeypatch.setattr(orchestrator, "discover_agent", fake_discover_agent)
    monkeypatch.setattr(orchestrator, "call_agent", fake_call_agent)
    monkeypatch.setattr(orchestrator, "format_response", fake_format_response)

    client = TestClient(orchestrator.app)
    response = client.post("/inference", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["agent"] == "General Assistant"
    assert body["response"] == "Hi from agent"
    assert body["route_source"] == "rule"
