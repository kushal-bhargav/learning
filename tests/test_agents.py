from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.agents.delivery_planner import DeliveryPlannerAgent
from src.agents.greeting_story import GreetingStoryAgent
from src.agents.llm import LLMProvider, select_provider
from src.agents.orchestrator import GiftSession
from src.agents.recipient_profiling import RecipientProfilingAgent
from src.agents.recommendation import RecommendationAgent
from src.agents.relationship_analysis import RelationshipAnalysisAgent


FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "long_distance_partners.json"


class FakeStructuredLLM:
    provider = LLMProvider.OLLAMA

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


@pytest.fixture
def session() -> GiftSession:
    return GiftSession(
        session_id="session-test",
        giver_id="person-maya",
        recipient_id="person-jordan",
        occasion_id="occasion-jordan-birthday-2026",
    )


def test_recipient_agent_uses_fixture_context_and_config(session: GiftSession) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    response = {"output": {"interests": [], "communication_style": "warm", "gift_history_summary": "Unknown"}, "confidence": 0.8, "rationale": "Stated evidence."}
    llm = FakeStructuredLLM(response)
    result = RecipientProfilingAgent(llm).run({"session": session, "stage_config": {"person": data["people"][1], "preferences": data["preferences"], "raw_notes": [memory["content"] for memory in data["memories"]]}})
    assert result["stage"] == "recipient_profiling"
    assert result["output"]["communication_style"] == "warm"
    assert llm.calls[0]["temperature"] == 0.3
    assert llm.calls[0]["schema"]["type"] == "object"


@pytest.mark.parametrize(
    ("agent_class", "stage_config", "response", "stage"),
    [
        (RelationshipAnalysisAgent, {"relationship": {"type": "partner", "closeness_score": 5}, "memories": []}, {"output": {"closeness_assessment": "close", "tone_guidance": "warm", "formality": "casual", "risk_flags": [], "agency_slider_default": 0.5}, "confidence": 0.9, "rationale": "Evidence."}, "relationship_analysis"),
        (RecommendationAgent, {"recipient_profile": {}, "relationship_guidance": {}, "occasion": {"budget_hint": "USD 60", "date": "2026-09-18"}}, {"output": {"recommendations": [{"rank": 1, "category": "art", "concept": "one", "evidence": [], "budget_fit": "yes", "artifact_type": "generated"}, {"rank": 2, "category": "book", "concept": "two", "evidence": [], "budget_fit": "yes", "artifact_type": "physical"}, {"rank": 3, "category": "event", "concept": "three", "evidence": [], "budget_fit": "yes", "artifact_type": "bundle"}]}, "confidence": 0.7, "rationale": "Fit."}, "recommendation"),
        (GreetingStoryAgent, {"relationship_guidance": {}, "occasion": {}, "memories": []}, {"output": {"message": "Happy birthday!", "memory_references": [], "tone": "warm"}, "confidence": 0.8, "rationale": "Tone."}, "greeting_story"),
    ],
)
def test_structured_agents_follow_common_contract(
    session: GiftSession,
    agent_class: type,
    stage_config: dict[str, Any],
    response: dict[str, Any],
    stage: str,
) -> None:
    result = agent_class(FakeStructuredLLM(response)).run(
        {"session": session, "stage_config": stage_config}
    )
    assert result["stage"] == stage
    assert isinstance(result["output"], dict)
    assert result["rationale"]


def test_smolagents_agents_use_real_schema_path_when_tool_wrapper_fails(
    monkeypatch: pytest.MonkeyPatch,
    session: GiftSession,
) -> None:
    monkeypatch.setenv("GMGI_FORCE_OLLAMA_AGENTS", "1")
    monkeypatch.setenv("GMGI_ALLOW_AGENT_FALLBACK", "0")
    monkeypatch.setattr(
        RelationshipAnalysisAgent,
        "_run_with_smolagents",
        lambda *_: (_ for _ in ()).throw(RuntimeError("tool wrapper failed")),
    )
    response = {
        "output": {
            "closeness_assessment": "close",
            "tone_guidance": "warm",
            "formality": "casual",
            "risk_flags": [],
            "agency_slider_default": 0.5,
        },
        "confidence": 0.9,
        "rationale": "Schema path used supplied relationship evidence.",
    }
    result = RelationshipAnalysisAgent(FakeStructuredLLM(response)).run(
        {"session": session, "stage_config": {"relationship": {"type": "friend", "closeness_score": 4}, "memories": []}}
    )
    assert result["stage"] == "relationship_analysis"
    assert result["output"]["closeness_assessment"] == "close"
    assert "schema-validated Ollama JSON path" in result["rationale"]


def test_delivery_planner_is_simulated_and_deterministic(session: GiftSession) -> None:
    result = DeliveryPlannerAgent().run({"session": session, "stage_config": {"artifact_type": "physical", "occasion": {"date": "2026-09-18"}}})
    assert result["output"]["status"] == "simulated"
    assert result["output"]["planned_send_date"] == "2026-09-11"
    assert "No shipment" in result["output"]["disclaimer"]
    assert "date_logistics_math" in result["output"]["skills_used"]
    assert result["output"]["prompt_version"] == "static"


def test_provider_selection_respects_explicit_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GMGI_LLM_PROVIDER", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert select_provider() == LLMProvider.OLLAMA
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    assert select_provider() == LLMProvider.CLAUDE
    assert select_provider("gemini") == LLMProvider.GEMINI


def test_prompt_and_temperature_live_in_config_files() -> None:
    config_dir = Path(__file__).parents[1] / "src" / "agents" / "configs"
    for name in ("recipient_profiling", "relationship_analysis", "recommendation", "greeting_story"):
        config = json.loads((config_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert isinstance(config["temperature"], float)
        assert "{context}" in config["prompt_template"]
        assert set(config["models"]) == {"ollama", "azure_openai", "openai", "gemini", "claude"}

from src.agents.gift_intent_reasoning import GiftIntentReasoningAgent
from src.agents.multi_agent_planning import MultiAgentPlanningAgent


def test_gift_intent_agent_heuristic_extracts_structured_intent(session: GiftSession) -> None:
    result = GiftIntentReasoningAgent().run(
        {
            "session": session,
            "stage_config": {
                "method": "heuristic",
                "recipient_profile": {"interests": [{"name": "ceramics", "confidence": 0.9}]},
                "relationship_guidance": {"tone_guidance": "warm and playful", "formality": "casual"},
                "relationship": {"type": "friend", "closeness_score": 4},
                "occasion": {"name": "Birthday", "date": "2026-09-18", "budget_hint": "USD 60-100", "formality": "casual"},
                "preferences": [{"value": "tea", "confidence": 1.0}],
                "memories": [{"content": "We laughed at a tiny tea shop."}],
            },
        }
    )
    assert result["stage"] == "gift_intent_reasoning"
    assert result["output"]["goal"]["gift_purpose"] == "celebrate relationship milestone"
    assert result["output"]["constraints"]["budget_hint"] == "USD 60-100"
    assert result["output"]["preferences"]
    assert result["output"]["goal"]["recommended_artifact_type"] == "greeting_card"
    assert result["output"]["visual_generation"]["artifact_type"] == "greeting_card"


def test_multi_agent_planner_outputs_bounded_executable_plan(session: GiftSession) -> None:
    result = MultiAgentPlanningAgent().run(
        {
            "session": session,
            "stage_config": {
                "method": "rule_constrained",
                "user_request": "Create a birthday gift",
                "intent": {"intent_summary": "Create a personal birthday gift", "clarifying_needs": []},
                "memory_signals": {"memory_count": 2, "preference_count": 3},
            },
        }
    )
    plan = result["output"]
    assert result["stage"] == "multi_agent_planning"
    assert "recommendation" in plan["agent_sequence"]
    assert "creative_generation" in plan["agent_sequence"]
    assert plan["fallback_plan"]["type"] == "current_staged_orchestration"
    assert all("agent" in step for step in plan["subtasks"])
