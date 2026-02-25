import asyncio

import pytest
from fastapi.testclient import TestClient

from app import main as orchestrator


@pytest.fixture(autouse=True)
def reset_skills_prompt_cache():
    orchestrator.skills_prompt_cache._prompt = orchestrator._build_intent_system_prompt([], "general")
    orchestrator.skills_prompt_cache._allowed_tags = {"general"}
    orchestrator.skills_prompt_cache._fallback_skill = "general"
    orchestrator.skills_prompt_cache._updated_at_monotonic = 0.0
    orchestrator.skills_prompt_cache._refresh_task = None


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


def test_available_skill_tags_collects_unique_tags():
    tags = orchestrator._available_skill_tags(
        [
            {"tags": ["general", "assistant"]},
            {"tags": ["math", "general"]},
        ]
    )
    assert tags == ["assistant", "general", "math"]


def test_build_intent_system_prompt_includes_registry_skills():
    prompt = orchestrator._build_intent_system_prompt(
        [
            {
                "agent": "Math Specialist",
                "name": "Math Evaluation",
                "description": "Evaluates arithmetic",
                "tags": ["math", "calculator"],
            }
        ],
        fallback_skill="math",
    )

    assert "Math Specialist" in prompt
    assert "Math Evaluation" in prompt
    assert "math" in prompt


def test_select_route_uses_rule_first(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_llm_classify_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("LLM should not run for rule hit")),
    )

    route = asyncio.run(orchestrator.select_route("1 + 1"))
    assert route["route_source"] == "rule"
    assert route["skill"] == "math"


def test_select_route_uses_dynamic_skills_for_llm(monkeypatch):
    async def fake_list_skills():
        return [
            {
                "agent": "General Assistant",
                "name": "General Knowledge",
                "description": "General tasks",
                "tags": ["general", "qa"],
            }
        ]

    captured = {"prompt": ""}

    def fake_llm(user_input: str, prompt: str, allowed_tags: set[str], fallback_skill: str):
        captured["prompt"] = prompt
        assert user_input == "hello"
        assert allowed_tags == {"general", "qa"}
        assert fallback_skill == "general"
        return {"skill": "qa", "payload": "hello", "route_source": "llm"}

    monkeypatch.setattr(orchestrator, "rule_based_route", lambda _msg: None)
    monkeypatch.setattr(orchestrator.a2a_service, "list_skills", fake_list_skills)
    monkeypatch.setattr(orchestrator, "_llm_classify_sync", fake_llm)

    route = asyncio.run(orchestrator.select_route("hello"))
    assert route == {"skill": "qa", "payload": "hello", "route_source": "llm"}
    assert "General Knowledge" in captured["prompt"]


def test_select_route_falls_back_when_llm_fails(monkeypatch):
    async def fake_list_skills():
        return [
            {
                "agent": "General Assistant",
                "name": "General Knowledge",
                "description": "General tasks",
                "tags": ["general"],
            }
        ]

    def fake_llm(*_args, **_kwargs):
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(orchestrator, "rule_based_route", lambda _msg: None)
    monkeypatch.setattr(orchestrator.a2a_service, "list_skills", fake_list_skills)
    monkeypatch.setattr(orchestrator, "_llm_classify_sync", fake_llm)

    route = asyncio.run(orchestrator.select_route("hello world"))
    assert route == {"skill": "general", "payload": "hello world", "route_source": "fallback"}


def test_skills_prompt_cache_uses_ttl(monkeypatch):
    calls = {"count": 0}

    async def fake_list_skills():
        calls["count"] += 1
        return [{"agent": "General Assistant", "name": "General Knowledge", "description": "", "tags": ["general"]}]

    monkeypatch.setattr(orchestrator.a2a_service, "list_skills", fake_list_skills)
    cache = orchestrator.SkillsPromptCache(ttl_seconds=60.0, refresh_interval_seconds=30.0)

    asyncio.run(cache.get_classifier_context())
    asyncio.run(cache.get_classifier_context())

    assert calls["count"] == 1


def test_skills_prompt_cache_refreshes_periodically(monkeypatch):
    calls = {"count": 0}

    async def fake_list_skills():
        calls["count"] += 1
        return [{"agent": "General Assistant", "name": "General Knowledge", "description": "", "tags": ["general"]}]

    monkeypatch.setattr(orchestrator.a2a_service, "list_skills", fake_list_skills)
    cache = orchestrator.SkillsPromptCache(ttl_seconds=60.0, refresh_interval_seconds=0.05)

    async def run_cycle():
        await cache.start()
        await asyncio.sleep(0.12)
        await cache.stop()

    asyncio.run(run_cycle())
    assert calls["count"] >= 1


def test_format_response_passthrough_for_general_skill():
    assert asyncio.run(orchestrator.format_response("hello", "agent output", "general")) == "agent output"


def test_format_response_math_fallback_when_llm_fails(monkeypatch):
    def fake_format_sync(_user_input: str, _agent_result: str, _skill: str):
        raise RuntimeError("ollama timeout")

    monkeypatch.setattr(orchestrator, "_format_response_sync", fake_format_sync)
    result = asyncio.run(orchestrator.format_response("2+2", "4", "math"))
    assert result == "4"


def test_health_endpoint_includes_breakers(monkeypatch):
    monkeypatch.setattr(
        orchestrator.a2a_service,
        "health_snapshot",
        lambda: {
            "registry_url": "http://localhost:9300",
            "registry_breaker": {"failures": 0, "open": False},
            "agent_breakers": {},
        },
    )

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
    monkeypatch.setattr(orchestrator.a2a_service, "discover_agent", fake_discover_agent)

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
    monkeypatch.setattr(orchestrator.a2a_service, "discover_agent", fake_discover_agent)
    monkeypatch.setattr(orchestrator.a2a_service, "call_agent", fake_call_agent)

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
    monkeypatch.setattr(orchestrator.a2a_service, "discover_agent", fake_discover_agent)
    monkeypatch.setattr(orchestrator.a2a_service, "call_agent", fake_call_agent)
    monkeypatch.setattr(orchestrator, "format_response", fake_format_response)

    client = TestClient(orchestrator.app)
    response = client.post("/inference", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["agent"] == "General Assistant"
    assert body["response"] == "Hi from agent"
    assert body["route_source"] == "rule"
