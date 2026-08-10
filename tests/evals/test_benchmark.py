from __future__ import annotations

from src.evals.benchmark import BenchmarkCase, load_cases, run_case


def test_default_benchmark_cases_are_loadable() -> None:
    cases = load_cases()
    assert len(cases) >= 3
    assert all(case.case_id for case in cases)
    assert all(case.custom_profile.get("recipient_name") for case in cases)


def test_benchmark_case_records_agent_errors_without_raising(monkeypatch) -> None:
    class FailingAgent:
        def run(self, _agent_input):
            raise RuntimeError("agent unavailable")

    class OkAgent:
        output = {}

        def run(self, _agent_input):
            return {"stage": "test", "output": self.output, "confidence": 0.8, "rationale": "test"}

    class RelationshipOkAgent(OkAgent):
        output = {"closeness_assessment": "moderate", "tone_guidance": "warm", "formality": "casual", "risk_flags": [], "agency_slider_default": 0.5}

    class IntentOkAgent(OkAgent):
        output = {
            "intent_summary": "Create a birthday gift",
            "occasion": {"name": "Birthday"},
            "goal": {"gift_purpose": "create a thoughtful gift"},
            "constraints": {"budget_hint": "USD 20-40", "delivery_constraints": ["simulated delivery only"]},
            "preferences": [{"value": "tea", "confidence": 1.0}],
            "visual_generation": {"artifact_type": "greeting_card"},
            "open_questions": [],
            "clarifying_needs": [],
        }

    class PlanningOkAgent(OkAgent):
        output = {
            "task_goal": "Create a gift",
            "subtasks": [{"agent": "recipient_profiling", "requires_human_review": True}],
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
                {"after": "recommendation", "before": "creative_generation"},
                {"after": "creative_generation", "before": "greeting_story"},
                {"after": "greeting_story", "before": "delivery_planner"},
            ],
            "expected_outputs": [],
            "stop_conditions": ["simulated delivery only"],
            "fallback_plan": {"type": "current_staged_orchestration"},
        }

    class RecommendationOkAgent(OkAgent):
        output = {
            "recommendations": [
                {"rank": 1, "category": "generated", "concept": "tea card", "evidence": ["tea"], "budget_fit": "fits", "artifact_type": "generated"},
                {"rank": 2, "category": "bundle", "concept": "tea ritual", "evidence": ["tea"], "budget_fit": "fits", "artifact_type": "bundle"},
                {"rank": 3, "category": "physical", "concept": "tea mug", "evidence": ["tea"], "budget_fit": "fits", "artifact_type": "physical"},
            ]
        }

    class GreetingOkAgent(OkAgent):
        output = {"message": "A warm birthday note about tea and care.", "memory_references": [], "tone": "warm"}

    class DeliveryOkAgent(OkAgent):
        output = {"mode": "digital", "channel": "digital card", "planned_send_date": "2026-12-01", "occasion_date": "2026-12-01", "status": "simulated", "disclaimer": "No shipment was created."}

    import src.evals.benchmark as benchmark

    monkeypatch.setattr(benchmark, "RecipientProfilingAgent", FailingAgent)
    monkeypatch.setattr(benchmark, "RelationshipAnalysisAgent", RelationshipOkAgent)
    monkeypatch.setattr(benchmark, "GiftIntentReasoningAgent", IntentOkAgent)
    monkeypatch.setattr(benchmark, "MultiAgentPlanningAgent", PlanningOkAgent)
    monkeypatch.setattr(benchmark, "RecommendationAgent", RecommendationOkAgent)
    monkeypatch.setattr(benchmark, "GreetingStoryAgent", GreetingOkAgent)
    monkeypatch.setattr(benchmark, "DeliveryPlannerAgent", DeliveryOkAgent)
    case = BenchmarkCase(
        case_id="error-case",
        custom_profile={
            "giver_name": "A",
            "recipient_name": "B",
            "relationship_type": "friend",
            "closeness_score": 3,
            "occasion_name": "Birthday",
            "occasion_date": "2026-12-01",
            "budget_hint": "USD 20-40",
            "preferences": ["tea"],
            "memories": ["They drink tea together."],
        },
        expected={"preferences": ["tea"], "occasion_name": "Birthday", "closeness_score": 3},
    )
    report = run_case(case, output_dir="experiments/evals/test-benchmark", include_creative=False, agency_slider=0.5, seed=1)
    assert any(trace["stage"] == "recipient_profiling" and trace["status"] == "error" for trace in report["agent_traces"])
    assert report["stage_reports"]["recipient_profiling"]["quality_score"] < 0.5
