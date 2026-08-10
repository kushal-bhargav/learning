from __future__ import annotations

from src.evals.quality import creative_metrics, evaluate_outputs, recommendation_metrics


def _outputs() -> dict:
    return {
        "recipient_profiling": {
            "interests": [{"name": "ceramics", "confidence": 0.9}, {"name": "herbs", "confidence": 0.8}],
            "communication_style": "warm and practical",
            "gift_history_summary": "No gift history supplied.",
        },
        "relationship_analysis": {
            "closeness_assessment": "close",
            "tone_guidance": "Warm sibling tone with practical affection.",
            "formality": "casual",
            "risk_flags": [],
            "agency_slider_default": 0.45,
        },
        "gift_intent_reasoning": {
            "intent_summary": "Create a personalized housewarming gift for Asha.",
            "occasion": {"name": "Asha's housewarming", "date": "2026-11-12", "formality": "casual"},
            "goal": {"gift_purpose": "create a meaningful practical gift", "social_tone": "casual"},
            "constraints": {"budget_hint": "USD 50-90", "delivery_constraints": ["simulated delivery only"]},
            "preferences": [{"value": "ceramics", "confidence": 1.0}, {"value": "herbs", "confidence": 1.0}],
            "visual_generation": {"artifact_type": "greeting_card", "style_prompt": "ceramics and herbs"},
            "open_questions": [],
            "clarifying_needs": [],
        },
        "multi_agent_planning": {
            "task_goal": "Create a gift",
            "subtasks": [{"agent": stage, "requires_human_review": True} for stage in (
                "recipient_profiling",
                "relationship_analysis",
                "gift_intent_reasoning",
                "multi_agent_planning",
                "recommendation",
                "creative_generation",
                "greeting_story",
                "delivery_planner",
            )],
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
        },
        "recommendation": {
            "recommendations": [
                {"rank": 1, "category": "personalized art", "concept": "A ceramic herb marker card for Asha", "evidence": ["ceramics", "herbs"], "budget_fit": "Fits USD 50-90", "artifact_type": "generated"},
                {"rank": 2, "category": "experience", "concept": "A Sunday chai planting ritual", "evidence": ["herbs"], "budget_fit": "Scalable", "artifact_type": "bundle"},
                {"rank": 3, "category": "physical keepsake", "concept": "A small ceramic planter", "evidence": ["ceramics"], "budget_fit": "Within budget", "artifact_type": "physical"},
            ]
        },
        "creative_generation": {
            "artifact_path": "experiments/generated/real-card.png",
            "artifact_type": "generated",
            "media_type": "image/png",
            "width": 256,
            "height": 256,
            "agency_slider": 0.5,
            "seed": 2026,
            "diffusers_prompt": "ceramics and herbs housewarming card",
        },
        "greeting_story": {"message": "Asha, here is a warm housewarming wish for the kitchen you are making your own.", "memory_references": [], "tone": "warm"},
        "delivery_planner": {"mode": "digital", "channel": "digital card", "planned_send_date": "2026-11-12", "occasion_date": "2026-11-12", "status": "simulated", "disclaimer": "No shipment, purchase, or external delivery was created."},
    }


def test_quality_report_scores_real_task_behavior() -> None:
    expected = {
        "relationship_type": "sibling",
        "closeness_score": 4,
        "relationship_closeness": "close",
        "occasion_name": "Asha's housewarming",
        "occasion_date": "2026-11-12",
        "budget_hint": "USD 50-90",
        "preferences": ["ceramics", "herbs"],
    }
    report = evaluate_outputs(_outputs(), expected=expected, input_context={"preferences": [{"value": "ceramics"}, {"value": "herbs"}]})
    assert report["overall_quality_score"] > 0.8
    assert report["stage_reports"]["recipient_profiling"]["quality_score"] > 0.8
    assert report["stage_reports"]["recommendation"]["quality_score"] > 0.8


def test_creative_metrics_penalize_fake_placeholder_outputs() -> None:
    metrics = creative_metrics({"artifact_path": "experiments/generated/fake.png", "width": 16, "height": 16, "agency_slider": 0.5}, {}, {})
    scores = {metric.name: metric.score for metric in metrics}
    assert scores["artifact_not_marked_fake"] == 0.0
    assert scores["image_resolution_practical"] == 0.0


def test_recommendation_metrics_require_grounded_evidence() -> None:
    output = {
        "recommendations": [
            {"rank": 1, "concept": "Random luxury watch", "evidence": ["unseen preference"], "budget_fit": "expensive", "artifact_type": "physical"},
            {"rank": 2, "concept": "Random perfume", "evidence": ["unseen preference"], "budget_fit": "expensive", "artifact_type": "physical"},
            {"rank": 3, "concept": "Random tickets", "evidence": ["unseen preference"], "budget_fit": "expensive", "artifact_type": "physical"},
        ]
    }
    metrics = recommendation_metrics(output, {"preferences": ["ceramics"]}, {}, {})
    scores = {metric.name: metric.score for metric in metrics}
    assert scores["evidence_grounding_rate"] == 0.0
    assert scores["preference_coverage"] == 0.0
