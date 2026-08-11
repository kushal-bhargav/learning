from __future__ import annotations

import pytest

from src.agents.orchestrator import AgentOutput
from src.harness import (
    ConstraintVerifier,
    ExecutionPlan,
    HarnessConfig,
    HarnessController,
    HarnessRuntimeState,
    HarnessVerificationError,
    PlanStep,
    PlanValidationError,
    ToolPermissionError,
    UnsupportedHarnessModeError,
    VerificationResult,
)
from src.harness.config import DEFAULT_STAGE_ORDER


def test_default_controller_reproduces_fixed_stage_order() -> None:
    controller = HarnessController(HarnessConfig())
    completed: list[str] = []
    observed: list[str] = []

    while True:
        action = controller.next_action(completed_stages=completed)
        if action.action_type == "stop":
            break
        assert action.stage is not None
        observed.append(action.stage)
        completed.append(action.stage)

    assert tuple(observed) == DEFAULT_STAGE_ORDER
    assert controller.stop_reason == "fixed_stage_policy_complete"


def test_dynamic_router_selects_valid_next_action_and_respects_dependencies() -> None:
    controller = HarnessController(HarnessConfig(orchestration_mode="dynamic", routing_mode="dynamic"))

    first = controller.next_action(completed_stages=[])
    blocked = controller.next_action(completed_stages=["recipient_profiling", "gift_intent_reasoning"])

    assert first.stage == "recipient_profiling"
    assert first.policy_name == "dynamic_router"
    assert first.candidate_actions[0]["stage"] == "recipient_profiling"
    assert blocked.stage == "relationship_analysis"
    assert any(item["stage"] == "recommendation" and item["reason"] == "missing_preconditions" for item in blocked.rejected_actions)


def test_unsupported_dynamic_compatibility_value_still_fails_explicitly() -> None:
    with pytest.raises(UnsupportedHarnessModeError, match="unsupported_dynamic"):
        HarnessController(HarnessConfig(orchestration_mode="unsupported_dynamic"))


def test_authoritative_plan_controls_stage_order() -> None:
    plan = ExecutionPlan(steps=(PlanStep("recipient_profiling"), PlanStep("recommendation", dependencies=("recipient_profiling",))))
    controller = HarnessController(HarnessConfig(planner_mode="authoritative"), execution_plan=plan)

    first = controller.next_action(completed_stages=[])
    second = controller.next_action(completed_stages=["recipient_profiling"])

    assert first.stage == "recipient_profiling"
    assert first.reason == "authoritative_plan_next_step"
    assert second.stage == "recommendation"
    assert second.planner_decision(controller.config).decision == "authoritative"


def test_authoritative_plan_requires_valid_plan() -> None:
    with pytest.raises(PlanValidationError, match="unknown"):
        HarnessController(HarnessConfig(planner_mode="authoritative"), execution_plan=ExecutionPlan(steps=(PlanStep("unknown_agent"),)))

    with pytest.raises(PlanValidationError, match="missing dependency"):
        HarnessController(
            HarnessConfig(planner_mode="authoritative"),
            execution_plan=ExecutionPlan(steps=(PlanStep("recommendation", dependencies=("missing_stage",)),)),
        )

    with pytest.raises(PlanValidationError, match="cycle"):
        HarnessController(
            HarnessConfig(planner_mode="authoritative"),
            execution_plan=ExecutionPlan(
                steps=(
                    PlanStep("recipient_profiling", dependencies=("recommendation",)),
                    PlanStep("recommendation", dependencies=("recipient_profiling",)),
                )
            ),
        )


def test_authoritative_mode_without_plan_fails_explicitly() -> None:
    controller = HarnessController(HarnessConfig(planner_mode="authoritative"))

    with pytest.raises(PlanValidationError, match="requires"):
        controller.next_action(completed_stages=[])


def test_dynamic_authoritative_mode_can_start_before_plan_and_adopt_plan() -> None:
    controller = HarnessController(HarnessConfig(orchestration_mode="dynamic", routing_mode="dynamic", planner_mode="authoritative"))

    assert controller.next_action(completed_stages=[]).stage == "recipient_profiling"
    plan = controller.adopt_execution_plan_from_output(
        AgentOutput(
            stage="multi_agent_planning",
            output={"agent_sequence": ["recipient_profiling", "gift_intent_reasoning", "recommendation"]},
            confidence=1.0,
            rationale="reduced plan",
        )
    )
    assert [step.stage for step in plan.steps] == ["recipient_profiling", "gift_intent_reasoning", "recommendation"]
    next_action = controller.next_action(completed_stages=["recipient_profiling", "gift_intent_reasoning"])
    assert next_action.stage == "recommendation"


def test_tool_policy_denies_disallowed_tool() -> None:
    controller = HarnessController(HarnessConfig(tool_policy="deny_all_tools"))

    with pytest.raises(ToolPermissionError):
        controller.tool_policy.require_allowed(agent="DeliveryPlannerAgent", tool="date_logistics_math")


def test_tool_policy_allows_authorized_tool() -> None:
    controller = HarnessController(HarnessConfig())

    controller.tool_policy.require_allowed(agent="RelationshipAnalysisAgent", tool="query_memory_graph")


def test_controller_schema_gate_passes_and_fails() -> None:
    controller = HarnessController(HarnessConfig(verification_policy="controller_schema_gate"))
    schema = {
        "type": "object",
        "required": ["output", "confidence", "rationale"],
        "properties": {
            "output": {"type": "object", "required": ["message"], "properties": {"message": {"type": "string"}}},
            "confidence": {"type": ["number", "null"]},
            "rationale": {"type": ["string", "null"]},
        },
    }

    decision = controller.verify_output(
        stage="greeting_story",
        result=AgentOutput(stage="greeting_story", output={"message": "hi"}, confidence=1.0, rationale="ok"),
        output_schema=schema,
    )
    assert decision.decision == "passed"

    with pytest.raises(HarnessVerificationError):
        controller.verify_output(
            stage="greeting_story",
            result=AgentOutput(stage="greeting_story", output={}, confidence=1.0, rationale="bad"),
            output_schema=schema,
        )


def test_constraint_verifier_returns_retryable_budget_failure() -> None:
    verifier = ConstraintVerifier()

    result = verifier.verify(
        stage="recommendation",
        result=AgentOutput(
            stage="recommendation",
            output={"recommendations": [{"rank": 1, "concept": "Luxury watch", "budget_fit": "over budget", "price": "USD 900"}]},
            confidence=0.8,
            rationale="bad budget",
        ),
        stage_config={"occasion": {"budget_hint": "USD 20-40"}},
        agent_outputs={},
    )

    assert result.verdict == "FAIL_RETRYABLE"
    assert result.retryable is True
    assert "recommendation_0_price_exceeds_budget" in result.issues


def test_dynamic_router_changes_action_from_runtime_verification_observation() -> None:
    config = HarnessConfig(
        orchestration_mode="dynamic",
        routing_mode="dynamic",
        verification_policy="deterministic_constraints",
        retry_policy="controller_retry_once",
    )
    controller = HarnessController(config)
    completed_prefix = (
        "recipient_profiling",
        "relationship_analysis",
        "gift_intent_reasoning",
        "multi_agent_planning",
        "recommendation",
    )
    pass_state = HarnessRuntimeState(
        completed_stages=completed_prefix,
        verification_results={"recommendation": VerificationResult("recommendation", "PASS", "deterministic_constraints", "ok")},
        triggering_event="recommendation_verified",
    )
    fail_state = HarnessRuntimeState(
        completed_stages=completed_prefix[:-1],
        verification_results={"recommendation": VerificationResult("recommendation", "FAIL_RETRYABLE", "deterministic_constraints", "budget fail", ("budget",), True)},
        retries={"recommendation": 0},
        triggering_event="recommendation_verified",
    )

    continue_action = controller.next_action(completed_stages=completed_prefix, runtime_state=pass_state)
    retry_action = controller.next_action(completed_stages=completed_prefix[:-1], runtime_state=fail_state)

    assert continue_action.action_type == "agent"
    assert continue_action.stage == "creative_generation"
    assert retry_action.action_type == "retry"
    assert retry_action.stage == "recommendation"
    assert retry_action.routing_decision().details["candidate_actions"][0]["action_type"] == "retry"


def test_dynamic_router_stops_on_terminal_verification_failure() -> None:
    controller = HarnessController(HarnessConfig(orchestration_mode="dynamic", routing_mode="dynamic"))
    state = HarnessRuntimeState(
        completed_stages=("recipient_profiling", "relationship_analysis"),
        verification_results={"gift_intent_reasoning": VerificationResult("gift_intent_reasoning", "FAIL_NON_RETRYABLE", "deterministic_constraints", "terminal")},
        triggering_event="verification_complete",
    )

    action = controller.next_action(completed_stages=state.completed_stages, runtime_state=state)

    assert action.action_type == "stop"
    assert "FAIL_NON_RETRYABLE" in action.reason
    assert action.routing_decision().details["triggering_event"] == "verification:fail_non_retryable"


def test_controller_execution_limits_stop_dynamic_routing() -> None:
    controller = HarnessController(HarnessConfig(orchestration_mode="dynamic", routing_mode="dynamic", max_agent_invocations=2))
    action = controller.next_action(completed_stages=("recipient_profiling",), dependency_ids=("inv-1", "inv-2"))

    assert action.action_type == "stop"
    assert action.reason == "agent_invocation_limit"
    assert action.policy_name == "controller_execution_limits"


def test_controller_retry_and_fallback_policies_affect_execution() -> None:
    retry_controller = HarnessController(HarnessConfig(retry_policy="controller_retry_once"))
    action = retry_controller.next_action(completed_stages=[])
    attempts = {"count": 0}

    def flaky() -> AgentOutput:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return AgentOutput(stage=str(action.stage), output={"ok": True}, confidence=1.0, rationale="retried")

    result = retry_controller.execute_agent_action(action, flaky)
    assert attempts["count"] == 2
    assert result["output"]["ok"] is True

    fallback_controller = HarnessController(HarnessConfig(fallback_policy="skip_failed_stage"))
    fallback = fallback_controller.execute_agent_action(
        fallback_controller.next_action(completed_stages=[]),
        lambda: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    assert fallback["output"]["fallback"] == "skip_failed_stage"


def test_stop_before_delivery_alternative_changes_trajectory() -> None:
    controller = HarnessController(HarnessConfig(stopping_policy="stop_before_delivery"))
    completed = list(DEFAULT_STAGE_ORDER)
    completed.remove("delivery_planner")

    action = controller.next_action(completed_stages=completed)

    assert action.action_type == "stop"
    assert action.reason == "stopping_policy_stop_before_delivery"
