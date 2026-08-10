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


class FailingStructuredLLM:
    provider = LLMProvider.OLLAMA

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("model returned unusable structure")


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
    assert result["output"]["skills_declared"] == ["structured_recipient_extraction", "preference_signal_parser"]
    assert result["output"]["skills_used"] == result["output"]["skills_declared"]
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
    assert result["output"]["skills_declared"]
    assert result["output"]["skills_used"]
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


def test_recipient_profile_repairs_when_local_model_returns_invalid_shape(
    monkeypatch: pytest.MonkeyPatch,
    session: GiftSession,
) -> None:
    monkeypatch.setenv("GMGI_FORCE_OLLAMA_AGENTS", "1")
    monkeypatch.setenv("GMGI_ALLOW_AGENT_FALLBACK", "0")
    monkeypatch.setenv("GMGI_ALLOW_RECIPIENT_REPAIR", "1")
    monkeypatch.setattr(RecipientProfilingAgent, "_run_with_instructor", lambda *_: (_ for _ in ()).throw(RuntimeError("bad instructor output")))
    result = RecipientProfilingAgent(FailingStructuredLLM()).run(
        {
            "session": session,
            "stage_config": {
                "person": {"display_name": "Mira"},
                "preferences": [{"value": "tea", "confidence": 0.9}],
                "raw_notes": ["We laughed at a tiny tea shop."],
            },
        }
    )
    assert result["output"]["interests"][0]["name"] == "tea"
    assert result["output"]["skills_declared"] == ["structured_recipient_extraction", "preference_signal_parser"]
    assert result["output"]["skills_used"] == ["structured_recipient_extraction", "preference_signal_parser"]
    assert "preference_signal_parser" in result["rationale"]


def test_delivery_planner_is_simulated_and_deterministic(session: GiftSession) -> None:
    result = DeliveryPlannerAgent().run({"session": session, "stage_config": {"artifact_type": "physical", "occasion": {"date": "2026-09-18"}}})
    assert result["output"]["status"] == "simulated"
    assert result["output"]["planned_send_date"] == "2026-09-11"
    assert "No shipment" in result["output"]["disclaimer"]
    assert "date_logistics_math" in result["output"]["skills_used"]
    assert result["output"]["skills_declared"] == ["date_logistics_math", "simulated_delivery_structuring"]
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
    assert result["output"]["skills_declared"] == ["intent_classification", "constraint_extraction", "visual_artifact_mapping"]
    assert result["output"]["skills_used"] == result["output"]["skills_declared"]


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
    assert plan["skills_declared"] == ["task_decomposition", "dependency_planning", "fallback_plan_repair"]
    assert plan["skills_used"] == plan["skills_declared"]


def test_intent_and_planning_repair_llm_failures_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
    session: GiftSession,
) -> None:
    monkeypatch.setenv("GMGI_FORCE_OLLAMA_AGENTS", "1")
    monkeypatch.setenv("GMGI_ALLOW_AGENT_FALLBACK", "0")
    monkeypatch.setenv("GMGI_ALLOW_INTENT_REPAIR", "1")
    monkeypatch.setenv("GMGI_ALLOW_PLANNING_REPAIR", "1")

    intent = GiftIntentReasoningAgent(FailingStructuredLLM()).run(
        {
            "session": session,
            "stage_config": {
                "method": "llm_structured",
                "recipient_profile": {"interests": [{"name": "tea", "confidence": 0.9}]},
                "relationship_guidance": {"tone_guidance": "warm", "formality": "casual"},
                "relationship": {"type": "friend", "closeness_score": 4},
                "occasion": {"name": "Birthday", "date": "2026-12-18", "budget_hint": "USD 60-100"},
                "preferences": [{"value": "tea", "confidence": 1.0}],
                "memories": ["We found a tiny tea shop."],
            },
        }
    )
    assert intent["output"]["intent_summary"]
    assert "repaired with deterministic intent extractor" in intent["rationale"]

    plan = MultiAgentPlanningAgent(FailingStructuredLLM()).run(
        {
            "session": session,
            "stage_config": {
                "method": "llm_structured",
                "intent": intent["output"],
                "available_agents": [
                    "recipient_profiling",
                    "relationship_analysis",
                    "gift_intent_reasoning",
                    "multi_agent_planning",
                    "recommendation",
                ],
            },
        }
    )
    assert "recommendation" in plan["output"]["agent_sequence"]
    assert "repaired with bounded rule planner" in plan["rationale"]


def test_recommendation_repairs_when_tool_and_schema_paths_fail(
    monkeypatch: pytest.MonkeyPatch,
    session: GiftSession,
) -> None:
    monkeypatch.setenv("GMGI_FORCE_OLLAMA_AGENTS", "1")
    monkeypatch.setenv("GMGI_ALLOW_AGENT_FALLBACK", "0")
    monkeypatch.setenv("GMGI_ALLOW_RECOMMENDATION_REPAIR", "1")
    monkeypatch.setattr(RecommendationAgent, "_run_with_smolagents", lambda *_: (_ for _ in ()).throw(RuntimeError("tool wrapper failed")))
    result = RecommendationAgent(FailingStructuredLLM()).run(
        {
            "session": session,
            "stage_config": {
                "recipient_profile": {"interests": [{"name": "books", "confidence": 0.9}]},
                "relationship_guidance": {"tone_guidance": "warm"},
                "gift_intent": {
                    "goal": {"recommended_artifact_type": "greeting_card", "emotional_objective": "close"},
                    "visual_generation": {"artifact_type": "greeting_card"},
                    "preferences": [{"value": "travel"}],
                },
                "occasion": {"name": "Birthday", "budget_hint": "USD 60-100"},
                "preferences": [{"value": "books"}],
            },
        }
    )
    recs = result["output"]["recommendations"]
    assert len(recs) == 3
    assert [rec["rank"] for rec in recs] == [1, 2, 3]
    assert result["output"]["skills_declared"] == ["memory_context_lookup", "bandit_feedback_hint", "ranked_gift_reasoning"]
    assert result["output"]["skills_used"] == ["memory_context_lookup", "ranked_gift_reasoning"]
    assert "ranked_gift_reasoning" in result["rationale"]
