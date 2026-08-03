from __future__ import annotations

import shutil
from pathlib import Path

from src.evaluation.intent_planning import (
    ExperimentCase,
    MethodVariant,
    compare_intent_planning_methods,
    evaluate_intent_output,
    evaluate_plan_output,
)


def test_component_metric_functions_score_structured_outputs() -> None:
    intent_metrics = evaluate_intent_output(
        {
            "intent_summary": "Birthday gift",
            "occasion": {"name": "Birthday"},
            "goal": {},
            "constraints": {"budget_hint": "USD 60", "delivery_constraints": []},
            "preferences": [],
            "open_questions": [],
            "clarifying_needs": [],
        },
        {"occasion_name": "Birthday", "constraints": ["budget_hint"]},
    )
    assert intent_metrics["structured_output_valid"] is True
    assert intent_metrics["occasion_exact_or_contains"] is True

    plan_metrics = evaluate_plan_output(
        {
            "task_goal": "Create gift",
            "subtasks": [{"agent": "recipient_profiling"}],
            "agent_sequence": ["recipient_profiling", "relationship_analysis", "gift_intent_reasoning", "multi_agent_planning", "recommendation"],
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
    )
    assert plan_metrics["structured_output_valid"] is True
    assert plan_metrics["executable_plan_rate"] is True


def test_comparison_framework_writes_component_and_overall_tables() -> None:
    case = ExperimentCase(
        case_id="tiny",
        custom_profile={
            "giver_name": "Asha",
            "recipient_name": "Mira",
            "relationship_type": "friend",
            "closeness_score": 4,
            "occasion_name": "Birthday",
            "occasion_date": "2026-12-18",
            "budget_hint": "USD 60-100",
            "formality": "casual",
            "preferences": ["tea"],
            "memories": ["We laughed at a tiny tea shop."],
        },
        expected_intent={"occasion_name": "Birthday", "constraints": ["budget_hint"]},
    )
    output_dir = Path(".test-tmp/intent-planning-eval")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    summary = compare_intent_planning_methods(
        [case],
        intent_variants=[MethodVariant(name="heuristic", method="heuristic")],
        planning_variants=[MethodVariant(name="rule", method="rule_constrained")],
        output_dir=output_dir,
        run_overall=True,
    )
    assert summary["component_rows"]
    assert summary["overall_rows"][0]["end_to_end_success"] is True
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "component_metrics.csv").exists()
    assert (output_dir / "overall_metrics.csv").exists()
    shutil.rmtree(output_dir)


