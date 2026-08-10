from __future__ import annotations

from src.agents.experience_store import Episode
from src.evals.structural import (
    constraint_satisfaction,
    dag_validity,
    evaluate_episode,
    provenance_traceability,
    schema_conformance,
)


def _outputs() -> dict:
    return {
        "recipient_profiling": {
            "interests": [{"name": "tea", "confidence": 0.9}],
            "communication_style": "warm",
            "gift_history_summary": "none supplied",
        },
        "relationship_analysis": {
            "closeness_assessment": "close",
            "tone_guidance": "warm",
            "formality": "casual",
            "risk_flags": [],
            "agency_slider_default": 0.5,
        },
        "gift_intent_reasoning": {
            "intent_summary": "Create a birthday gift",
            "occasion": {"name": "Birthday"},
            "goal": {"recommended_artifact_type": "greeting_card"},
            "constraints": {"budget_hint": "USD 60-100", "delivery_constraints": ["simulated delivery only"]},
            "preferences": [{"value": "tea", "confidence": 0.9}],
            "visual_generation": {"artifact_type": "greeting_card"},
            "open_questions": [],
            "clarifying_needs": [],
        },
        "multi_agent_planning": {
            "task_goal": "Create a birthday gift",
            "subtasks": [{"agent": "recipient_profiling"}],
            "agent_sequence": [
                "recipient_profiling",
                "relationship_analysis",
                "gift_intent_reasoning",
                "multi_agent_planning",
                "recommendation",
                "creative_generation",
                "greeting_story",
                "delivery_planner",
            ],
            "dependencies": [
                {"after": "recipient_profiling", "before": "relationship_analysis"},
                {"after": "relationship_analysis", "before": "gift_intent_reasoning"},
                {"after": "gift_intent_reasoning", "before": "multi_agent_planning"},
                {"after": "multi_agent_planning", "before": "recommendation"},
            ],
            "expected_outputs": [],
            "stop_conditions": [],
            "fallback_plan": {"type": "current_staged_orchestration"},
        },
        "recommendation": {
            "recommendations": [
                {"rank": 1, "category": "generated", "concept": "tea card", "evidence": ["tea"], "budget_fit": "fits USD 60-100", "artifact_type": "generated"},
                {"rank": 2, "category": "bundle", "concept": "tea set", "evidence": ["tea"], "budget_fit": "scalable", "artifact_type": "bundle"},
                {"rank": 3, "category": "physical", "concept": "mug", "evidence": ["tea"], "budget_fit": "within budget", "artifact_type": "physical"},
            ]
        },
        "greeting_story": {"message": "Happy birthday", "memory_references": [], "tone": "warm"},
        "delivery_planner": {
            "mode": "digital",
            "channel": "digital card",
            "planned_send_date": "2026-12-18",
            "occasion_date": "2026-12-18",
            "status": "simulated",
            "disclaimer": "No shipment, purchase, or external delivery was created.",
        },
    }


def test_phase1_metrics_score_valid_logged_outputs() -> None:
    outputs = _outputs()
    assert schema_conformance(outputs).passed is True
    assert dag_validity(outputs).passed is True
    assert constraint_satisfaction(outputs).passed is True
    provenance = provenance_traceability(outputs)
    assert provenance.score == 1.0


def test_dag_validity_flags_cycles_and_order_errors() -> None:
    outputs = _outputs()
    outputs["multi_agent_planning"]["dependencies"] = [{"after": "recommendation", "before": "recipient_profiling"}]
    outputs["multi_agent_planning"]["agent_sequence"] = ["recommendation", "recipient_profiling"]
    result = dag_validity(outputs)
    assert result.passed is False
    assert result.details["errors"]


def test_evaluate_episode_returns_report() -> None:
    episode = Episode(
        session_id="session-1",
        timestamp="2026-08-10T00:00:00Z",
        context_fingerprint="friend|casual|mid|abc123",
        agent_outputs=_outputs(),
        human_actions={},
        composite_reward=0.8,
    )
    report = evaluate_episode(episode)
    assert report["session_id"] == "session-1"
    assert report["overall_score"] is not None
