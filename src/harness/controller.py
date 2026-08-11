from __future__ import annotations

import time
from collections import defaultdict, deque
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence
from uuid import uuid4

from jsonschema import ValidationError, validate

from .config import HarnessConfig, default_harness_config
from .trace import DecisionTrace


AgentOutput = dict[str, Any]


class HarnessControllerError(RuntimeError):
    """Base error for controlled harness execution failures."""


class UnsupportedHarnessModeError(HarnessControllerError):
    """Raised when a configured harness mode is intentionally not implemented."""


class PlanValidationError(HarnessControllerError):
    """Raised when an authoritative execution plan is structurally unsafe."""


class ToolPermissionError(HarnessControllerError):
    """Raised when a tool call violates the active tool policy."""


class HarnessVerificationError(HarnessControllerError):
    """Raised when a controller verification gate rejects an output."""


VerificationVerdict = Literal["PASS", "FAIL_RETRYABLE", "FAIL_NON_RETRYABLE"]


@dataclass(frozen=True)
class VerificationResult:
    stage: str
    verdict: VerificationVerdict
    policy: str
    reason: str
    issues: tuple[str, ...] = ()
    retryable: bool = False

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def to_decision(self) -> DecisionTrace:
        return DecisionTrace(
            decision_type="verifier",
            decision=self.verdict.lower(),
            policy=self.policy,
            reason=self.reason,
            details={"stage": self.stage, "issues": list(self.issues), "retryable": self.retryable},
        )


@dataclass(frozen=True)
class HarnessRuntimeState:
    run_id: str | None = None
    case_id: str | None = None
    current_invocation_id: str | None = None
    completed_invocations: tuple[str, ...] = ()
    completed_stages: tuple[str, ...] = ()
    available_agents: tuple[str, ...] = ()
    available_tools: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    agent_outputs: Mapping[str, Any] = field(default_factory=dict)
    failures: Mapping[str, int] = field(default_factory=dict)
    retries: Mapping[str, int] = field(default_factory=dict)
    verification_results: Mapping[str, VerificationResult] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    memory_context: Mapping[str, Any] = field(default_factory=dict)
    remaining_budget: int | None = None
    triggering_event: str | None = None


@dataclass(frozen=True)
class PlanStep:
    stage: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[PlanStep, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExecutionPlan":
        raw_steps = payload.get("steps") or payload.get("agent_sequence") or payload.get("subtasks") or []
        steps: list[PlanStep] = []
        if raw_steps and all(isinstance(item, str) for item in raw_steps):
            steps = [PlanStep(stage=str(item)) for item in raw_steps]
        else:
            for item in raw_steps:
                if not isinstance(item, Mapping):
                    raise PlanValidationError(f"Invalid plan step: {item!r}")
                stage = str(item.get("stage") or item.get("agent") or "").strip()
                dependencies = item.get("dependencies") or item.get("depends_on") or ()
                steps.append(PlanStep(stage=stage, dependencies=tuple(str(dep) for dep in dependencies)))
        dependencies_payload = payload.get("dependencies") or []
        dependency_map: dict[str, list[str]] = defaultdict(list)
        for item in dependencies_payload:
            if isinstance(item, Mapping) and item.get("after") and item.get("before"):
                dependency_map[str(item["before"])].append(str(item["after"]))
        if dependency_map:
            steps = [PlanStep(stage=step.stage, dependencies=tuple([*step.dependencies, *dependency_map.get(step.stage, [])])) for step in steps]
        return cls(steps=tuple(steps))


@dataclass(frozen=True)
class NextAction:
    action_type: Literal["agent", "stop", "retry", "fallback", "verifier", "human"]
    stage: str | None = None
    reason: str = ""
    controller_decision_id: str = field(default_factory=lambda: f"ctrl-{uuid4().hex[:12]}")
    policy_name: str = "fixed_stage_policy"
    policy_version: str = "1"
    parent_invocation_id: str | None = None
    dependency_ids: tuple[str, ...] = ()
    candidate_actions: tuple[Mapping[str, Any], ...] = ()
    rejected_actions: tuple[Mapping[str, Any], ...] = ()
    triggering_event: str | None = None

    def routing_decision(self) -> DecisionTrace:
        return DecisionTrace(
            decision_type="routing",
            decision=self.stage or self.action_type,
            policy=self.policy_name,
            reason=self.reason,
            details={
                "controller_decision_id": self.controller_decision_id,
                "selected_action": self.action_type,
                "policy_version": self.policy_version,
                "candidate_actions": list(self.candidate_actions),
                "rejected_actions": list(self.rejected_actions),
                "triggering_event": self.triggering_event,
                "parent_invocation_id": self.parent_invocation_id,
            },
        )

    def planner_decision(self, config: HarnessConfig) -> DecisionTrace:
        return DecisionTrace(
            decision_type="planner",
            decision="advisory_only" if config.planner_mode == "advisory" else "authoritative",
            policy=f"planner_mode:{config.planner_mode}",
            reason="planner_output_is_advisory" if config.planner_mode == "advisory" else "authoritative_execution_plan",
            details={"controller_decision_id": self.controller_decision_id},
        )


class FixedStagePolicy:
    name = "fixed_stage_policy"
    version = "1"

    def __init__(self, stage_order: Sequence[str]) -> None:
        self.stage_order = tuple(stage_order)

    def next_action(self, completed_stages: Iterable[str], *, config: HarnessConfig, dependency_ids: Sequence[str] = ()) -> NextAction:
        if config.orchestration_mode != "fixed_stage":
            raise UnsupportedHarnessModeError(f"Unsupported orchestration_mode: {config.orchestration_mode}")
        if config.routing_mode != "static":
            raise UnsupportedHarnessModeError(f"Unsupported routing_mode: {config.routing_mode}")
        completed = set(completed_stages)
        for stage in self.stage_order:
            if stage not in completed:
                if config.stopping_policy == "stop_before_delivery" and stage == "delivery_planner":
                    return NextAction(
                        action_type="stop",
                        reason="stopping_policy_stop_before_delivery",
                        policy_name=self.name,
                        policy_version=self.version,
                        dependency_ids=tuple(dependency_ids),
                    )
                return NextAction(
                    action_type="agent",
                    stage=stage,
                    reason="fixed_stage_policy_next_stage",
                    policy_name=self.name,
                    policy_version=self.version,
                    dependency_ids=tuple(dependency_ids),
                )
        return NextAction(
            action_type="stop",
            reason="fixed_stage_policy_complete",
            policy_name=self.name,
            policy_version=self.version,
            dependency_ids=tuple(dependency_ids),
        )


DEFAULT_STAGE_PRECONDITIONS: dict[str, tuple[str, ...]] = {
    "recipient_profiling": (),
    "relationship_analysis": ("recipient_profiling",),
    "gift_intent_reasoning": ("recipient_profiling", "relationship_analysis"),
    "multi_agent_planning": ("recipient_profiling", "relationship_analysis", "gift_intent_reasoning"),
    "recommendation": ("recipient_profiling", "relationship_analysis", "gift_intent_reasoning", "multi_agent_planning"),
    "creative_generation": ("recommendation",),
    "greeting_story": ("relationship_analysis", "gift_intent_reasoning", "recommendation"),
    "delivery_planner": ("creative_generation", "greeting_story"),
}


class Router(ABC):
    name = "router"
    version = "1"

    @abstractmethod
    def next_action(
        self,
        state: HarnessRuntimeState,
        *,
        config: HarnessConfig,
        execution_plan: ExecutionPlan | None = None,
        dependency_ids: Sequence[str] = (),
    ) -> NextAction:
        raise NotImplementedError


class StaticRouter(Router):
    name = "static_router"
    version = "1"

    def __init__(self, stage_order: Sequence[str]) -> None:
        self.fixed_policy = FixedStagePolicy(stage_order)

    def next_action(
        self,
        state: HarnessRuntimeState,
        *,
        config: HarnessConfig,
        execution_plan: ExecutionPlan | None = None,
        dependency_ids: Sequence[str] = (),
    ) -> NextAction:
        policy = self.fixed_policy
        if config.planner_mode == "authoritative":
            if execution_plan is None:
                raise PlanValidationError("planner_mode='authoritative' requires a validated ExecutionPlan")
            policy = FixedStagePolicy(tuple(step.stage for step in execution_plan.steps))
        action = policy.next_action(state.completed_stages, config=config, dependency_ids=dependency_ids)
        return NextAction(
            action_type=action.action_type,
            stage=action.stage,
            reason="authoritative_plan_next_step" if config.planner_mode == "authoritative" and action.action_type == "agent" else action.reason,
            policy_name="authoritative_plan_policy" if config.planner_mode == "authoritative" else self.name,
            policy_version=self.version,
            parent_invocation_id=state.current_invocation_id,
            dependency_ids=action.dependency_ids,
            candidate_actions=({"action_type": action.action_type, "stage": action.stage, "reason": action.reason},),
            triggering_event=state.triggering_event,
        )


class DynamicRouter(Router):
    name = "dynamic_router"
    version = "1"

    def __init__(self, preconditions: Mapping[str, Sequence[str]] | None = None) -> None:
        self.preconditions = {stage: tuple(deps) for stage, deps in (preconditions or DEFAULT_STAGE_PRECONDITIONS).items()}

    def next_action(
        self,
        state: HarnessRuntimeState,
        *,
        config: HarnessConfig,
        execution_plan: ExecutionPlan | None = None,
        dependency_ids: Sequence[str] = (),
    ) -> NextAction:
        stage_order = _stage_order_for(config, execution_plan)
        completed = set(state.completed_stages)
        candidate_actions: list[dict[str, Any]] = []
        rejected_actions: list[dict[str, Any]] = []

        recovery = self._recovery_action(state, config, dependency_ids)
        if recovery is not None:
            return recovery

        for stage in stage_order:
            if stage in completed:
                rejected_actions.append({"action_type": "agent", "stage": stage, "reason": "already_completed"})
                continue
            if config.stopping_policy == "stop_before_delivery" and stage == "delivery_planner":
                return NextAction(
                    action_type="stop",
                    reason="stopping_policy_stop_before_delivery",
                    policy_name=self.name,
                    policy_version=self.version,
                    dependency_ids=tuple(dependency_ids),
                    candidate_actions=tuple(candidate_actions + [{"action_type": "stop", "stage": None, "reason": "stop_before_delivery"}]),
                    rejected_actions=tuple(rejected_actions),
                    triggering_event=state.triggering_event,
                )
            missing = [dependency for dependency in self.preconditions.get(stage, ()) if dependency in stage_order and dependency not in completed]
            if missing:
                rejected_actions.append({"action_type": "agent", "stage": stage, "reason": "missing_preconditions", "missing": missing})
                continue
            candidate_actions.append({"action_type": "agent", "stage": stage, "reason": "preconditions_satisfied"})

        if candidate_actions:
            selected = candidate_actions[0]
            return NextAction(
                action_type="agent",
                stage=str(selected["stage"]),
                reason="dynamic_router_selected_first_valid_action",
                policy_name=self.name,
                policy_version=self.version,
                parent_invocation_id=state.current_invocation_id,
                dependency_ids=tuple(dependency_ids),
                candidate_actions=tuple(candidate_actions),
                rejected_actions=tuple(rejected_actions),
                triggering_event=state.triggering_event,
            )

        terminal_reason = "dynamic_router_complete" if len(completed) >= len(stage_order) else "dynamic_router_no_routable_candidates"
        return NextAction(
            action_type="stop",
            reason=terminal_reason,
            policy_name=self.name,
            policy_version=self.version,
            parent_invocation_id=state.current_invocation_id,
            dependency_ids=tuple(dependency_ids),
            candidate_actions=({"action_type": "stop", "stage": None, "reason": terminal_reason},),
            rejected_actions=tuple(rejected_actions),
            triggering_event=state.triggering_event,
        )

    def _recovery_action(self, state: HarnessRuntimeState, config: HarnessConfig, dependency_ids: Sequence[str]) -> NextAction | None:
        for stage, result in reversed(list(state.verification_results.items())):
            if result.passed or stage in set(state.completed_stages):
                continue
            retries = int(state.retries.get(stage, 0))
            if result.verdict == "FAIL_RETRYABLE" and config.retry_policy == "controller_retry_once" and retries < 1:
                return NextAction(
                    action_type="retry",
                    stage=stage,
                    reason=f"{stage} failed verification and retry policy permits one retry",
                    policy_name=self.name,
                    policy_version=self.version,
                    parent_invocation_id=state.current_invocation_id,
                    dependency_ids=tuple(dependency_ids),
                    candidate_actions=(
                        {"action_type": "retry", "stage": stage, "reason": result.reason},
                        {"action_type": "fallback", "stage": stage, "reason": "available_if_retry_not_permitted"},
                        {"action_type": "stop", "stage": None, "reason": "available_for_terminal_failure"},
                    ),
                    triggering_event=f"verification:{result.verdict.lower()}",
                )
            if config.fallback_policy == "skip_failed_stage":
                return NextAction(
                    action_type="fallback",
                    stage=stage,
                    reason=f"{stage} failed verification; skip_failed_stage fallback selected",
                    policy_name=self.name,
                    policy_version=self.version,
                    parent_invocation_id=state.current_invocation_id,
                    dependency_ids=tuple(dependency_ids),
                    candidate_actions=({"action_type": "fallback", "stage": stage, "reason": result.reason},),
                    triggering_event=f"verification:{result.verdict.lower()}",
                )
            return NextAction(
                action_type="stop",
                stage=None,
                reason=f"{stage} failed verification with {result.verdict}",
                policy_name=self.name,
                policy_version=self.version,
                parent_invocation_id=state.current_invocation_id,
                dependency_ids=tuple(dependency_ids),
                candidate_actions=({"action_type": "stop", "stage": None, "reason": result.reason},),
                triggering_event=f"verification:{result.verdict.lower()}",
            )
        return None


class ToolPolicy:
    def __init__(self, config: HarnessConfig, allowed_tools: Mapping[str, Iterable[str]] | None = None) -> None:
        self.config = config
        self.allowed_tools = {agent: set(tools) for agent, tools in (allowed_tools or DEFAULT_ALLOWED_TOOLS).items()}

    def is_allowed(self, *, agent: str, tool: str) -> bool:
        if self.config.tool_policy == "deny_all_tools":
            return False
        if self.config.tool_policy != "agent_local_tools":
            raise UnsupportedHarnessModeError(f"Unsupported tool_policy: {self.config.tool_policy}")
        allowed = self.allowed_tools.get(agent)
        return allowed is not None and tool in allowed

    def require_allowed(self, *, agent: str, tool: str) -> None:
        if not self.is_allowed(agent=agent, tool=tool):
            raise ToolPermissionError(f"Tool {tool!r} is not allowed for agent {agent!r} under {self.config.tool_policy}")


DEFAULT_ALLOWED_TOOLS = {
    "RelationshipAnalysisAgent": {"query_memory_graph"},
    "RecommendationAgent": {"query_memory_graph", "bandit_feedback_hint"},
    "CreativeGenerationAgent": {"diffusers_image_generation", "clip_critic"},
    "DeliveryPlannerAgent": {"date_logistics_math"},
}


class HarnessController:
    """Execution-authoritative controller for the current GMGI harness."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        execution_plan: ExecutionPlan | None = None,
        allowed_tools: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.config = config or default_harness_config()
        self.execution_plan = execution_plan
        self.tool_policy = ToolPolicy(self.config, allowed_tools)
        self.static_router = StaticRouter(self.config.stage_order)
        self.dynamic_router = DynamicRouter()
        self.stop_reason: str | None = None
        self.verification_results: dict[str, VerificationResult] = {}
        self.retries: dict[str, int] = defaultdict(int)
        self.failures: dict[str, int] = defaultdict(int)
        self._validate_mode_shape()
        if self.config.planner_mode == "authoritative" and self.execution_plan is not None:
            self.validate_plan(self.execution_plan)

    def build_runtime_state(
        self,
        *,
        completed_stages: Iterable[str],
        dependency_ids: Sequence[str] = (),
        agent_outputs: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
        memory_context: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        case_id: str | None = None,
        triggering_event: str | None = None,
    ) -> HarnessRuntimeState:
        available_tools = {agent: tuple(sorted(tools)) for agent, tools in self.tool_policy.allowed_tools.items()}
        return HarnessRuntimeState(
            run_id=run_id,
            case_id=case_id,
            current_invocation_id=dependency_ids[-1] if dependency_ids else None,
            completed_invocations=tuple(dependency_ids),
            completed_stages=tuple(completed_stages),
            available_agents=tuple(self.config.stage_order),
            available_tools=available_tools,
            agent_outputs=dict(agent_outputs or {}),
            failures=dict(self.failures),
            retries=dict(self.retries),
            verification_results=dict(self.verification_results),
            constraints=dict(constraints or {}),
            memory_context=dict(memory_context or {}),
            triggering_event=triggering_event,
        )

    def next_action(
        self,
        *,
        completed_stages: Iterable[str],
        dependency_ids: Sequence[str] = (),
        runtime_state: HarnessRuntimeState | None = None,
        agent_outputs: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
        memory_context: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        case_id: str | None = None,
        triggering_event: str | None = None,
    ) -> NextAction:
        state = runtime_state or self.build_runtime_state(
            completed_stages=completed_stages,
            dependency_ids=dependency_ids,
            agent_outputs=agent_outputs,
            constraints=constraints,
            memory_context=memory_context,
            run_id=run_id,
            case_id=case_id,
            triggering_event=triggering_event,
        )
        limit_action = self._limit_action(state, dependency_ids)
        if limit_action is not None:
            self.stop_reason = limit_action.reason
            return limit_action
        if self.config.routing_mode == "dynamic" or self.config.orchestration_mode == "dynamic":
            action = self.dynamic_router.next_action(state, config=self.config, execution_plan=self.execution_plan, dependency_ids=dependency_ids)
        else:
            action = self.static_router.next_action(state, config=self.config, execution_plan=self.execution_plan, dependency_ids=dependency_ids)
        if action.action_type == "stop":
            self.stop_reason = action.reason
        return action

    def _limit_action(self, state: HarnessRuntimeState, dependency_ids: Sequence[str]) -> NextAction | None:
        controller_steps = len(state.completed_invocations)
        agent_invocations = len(state.completed_invocations)
        total_retries = sum(int(value) for value in state.retries.values())
        if controller_steps >= self.config.max_controller_steps:
            reason = "controller_step_limit"
        elif agent_invocations >= self.config.max_agent_invocations:
            reason = "agent_invocation_limit"
        elif total_retries >= self.config.max_total_retries:
            reason = "total_retry_limit"
        else:
            return None
        return NextAction(
            action_type="stop",
            reason=reason,
            policy_name="controller_execution_limits",
            policy_version="1",
            dependency_ids=tuple(dependency_ids),
            candidate_actions=({"action_type": "stop", "stage": None, "reason": reason},),
            triggering_event="execution_limit",
        )

    def execute_agent_action(self, action: NextAction, execute: Callable[[], AgentOutput]) -> AgentOutput:
        if action.action_type not in {"agent", "retry"} or not action.stage:
            raise HarnessControllerError(f"Cannot execute non-agent action: {action.action_type}")
        attempts = min(2 if self.config.retry_policy == "controller_retry_once" else 1, self.config.max_retries_per_action + 1)
        last_error: Exception | None = None
        for attempt_index in range(attempts):
            try:
                if action.action_type == "retry" or attempt_index > 0:
                    self.retries[action.stage] += 1
                return execute()
            except Exception as exc:
                last_error = exc
                self.failures[action.stage] += 1
                if self.config.fallback_policy == "skip_failed_stage":
                    return {
                        "stage": action.stage,
                        "output": {"error": str(exc), "error_type": type(exc).__name__, "fallback": "skip_failed_stage"},
                        "confidence": 0.0,
                        "rationale": "HarnessController skip_failed_stage fallback returned structured error output.",
                    }
                if self.config.retry_policy != "controller_retry_once":
                    break
                time.sleep(0)
        assert last_error is not None
        raise last_error

    def adopt_execution_plan_from_output(self, result: AgentOutput | Mapping[str, Any]) -> ExecutionPlan:
        output = _as_mapping(result.get("output") if isinstance(result, Mapping) else {})
        plan = ExecutionPlan.from_mapping(output)
        self.validate_plan(plan)
        self.execution_plan = plan
        return plan

    def verify_output(
        self,
        *,
        stage: str,
        result: AgentOutput,
        output_schema: Mapping[str, Any] | None = None,
        stage_config: Mapping[str, Any] | None = None,
        agent_outputs: Mapping[str, Any] | None = None,
    ) -> DecisionTrace:
        if self.config.verification_policy == "schema_validation_with_offline_evals":
            decision = DecisionTrace(
                decision_type="verifier",
                decision="not_available",
                policy=self.config.verification_policy,
                reason="Live controller verifier gate is disabled; existing agent/local validation and offline evals apply.",
            )
            self.verification_results[stage] = VerificationResult(stage, "PASS", self.config.verification_policy, "offline_verifier_not_live")
            return decision
        if self.config.verification_policy == "deterministic_constraints":
            verification = ConstraintVerifier().verify(stage=stage, result=result, stage_config=stage_config or {}, agent_outputs=agent_outputs or {})
            self.verification_results[stage] = verification
            return verification.to_decision()
        if self.config.verification_policy != "controller_schema_gate":
            raise UnsupportedHarnessModeError(f"Unsupported verification_policy: {self.config.verification_policy}")
        if not output_schema:
            decision = DecisionTrace(
                decision_type="verifier",
                decision="not_available",
                policy=self.config.verification_policy,
                reason="No output_schema is registered for this agent.",
            )
            self.verification_results[stage] = VerificationResult(stage, "PASS", self.config.verification_policy, "no_schema_registered")
            return decision
        output = result.get("output")
        if isinstance(output, Mapping):
            output = {
                key: value
                for key, value in output.items()
                if key not in {"prompt_version", "skills_used", "skills_declared"}
            }
        payload = {
            "output": output,
            "confidence": result.get("confidence"),
            "rationale": result.get("rationale"),
        }
        try:
            validate(instance=payload, schema=dict(output_schema))
        except ValidationError as exc:
            raise HarnessVerificationError(f"{stage} failed controller schema gate: {exc.message}") from exc
        verification = VerificationResult(stage, "PASS", self.config.verification_policy, "controller_schema_gate_passed")
        self.verification_results[stage] = verification
        return DecisionTrace(
            decision_type="verifier",
            decision="passed",
            policy=self.config.verification_policy,
            reason="controller_schema_gate_passed",
        )

    def validate_plan(self, plan: ExecutionPlan) -> None:
        stages = [step.stage for step in plan.steps]
        known = set(self.config.stage_order)
        if not stages:
            raise PlanValidationError("ExecutionPlan must contain at least one step")
        unknown = [stage for stage in stages if stage not in known]
        if unknown:
            raise PlanValidationError(f"ExecutionPlan references unknown stage(s): {unknown}")
        if len(set(stages)) != len(stages):
            raise PlanValidationError("ExecutionPlan must not contain duplicate stages")
        stage_set = set(stages)
        for step in plan.steps:
            missing = [dependency for dependency in step.dependencies if dependency not in stage_set]
            if missing:
                raise PlanValidationError(f"Step {step.stage!r} has missing dependency/dependencies: {missing}")
        _reject_cycles(plan)

    def tool_context(self, agent_name: str):
        from .trace import tool_policy_context

        return tool_policy_context(lambda tool_name: self.tool_policy.require_allowed(agent=agent_name, tool=tool_name))

    def _validate_mode_shape(self) -> None:
        if self.config.orchestration_mode == "unsupported_dynamic":
            raise UnsupportedHarnessModeError("orchestration_mode='unsupported_dynamic' is retained only for backwards compatibility tests")
        if self.config.stage_execution_mode != "service_run_stage":
            raise UnsupportedHarnessModeError(f"Unsupported stage_execution_mode: {self.config.stage_execution_mode}")


class ConstraintVerifier:
    """Deterministic semantic/business checks that do not call an LLM."""

    policy = "deterministic_constraints"

    def verify(
        self,
        *,
        stage: str,
        result: AgentOutput,
        stage_config: Mapping[str, Any],
        agent_outputs: Mapping[str, Any],
    ) -> VerificationResult:
        if stage == "recommendation":
            return self._verify_recommendation(result, stage_config)
        if stage == "creative_generation":
            return self._verify_creative(result)
        if stage == "delivery_planner":
            return self._verify_delivery(result, stage_config)
        return VerificationResult(stage, "PASS", self.policy, "no_deterministic_constraints_registered")

    def _verify_recommendation(self, result: AgentOutput, stage_config: Mapping[str, Any]) -> VerificationResult:
        output = _as_mapping(result.get("output"))
        recommendations = _as_sequence(output.get("recommendations"))
        issues: list[str] = []
        if not recommendations:
            issues.append("recommendation_missing_recommendations")
        for index, recommendation_value in enumerate(recommendations):
            recommendation = _as_mapping(recommendation_value)
            if not recommendation.get("concept"):
                issues.append(f"recommendation_{index}_missing_concept")
            if not recommendation.get("budget_fit"):
                issues.append(f"recommendation_{index}_missing_budget_fit")
        budget_hint = _budget_hint(stage_config)
        budget_limit = _upper_money_value(budget_hint)
        if budget_limit is not None:
            for index, recommendation_value in enumerate(recommendations):
                recommendation = _as_mapping(recommendation_value)
                explicit_price = _upper_money_value(recommendation.get("price") or recommendation.get("estimated_price") or recommendation.get("estimated_cost"))
                budget_fit = str(recommendation.get("budget_fit") or "")
                if explicit_price is not None and explicit_price > budget_limit:
                    issues.append(f"recommendation_{index}_price_exceeds_budget")
                if _budget_text_denies_fit(budget_fit):
                    issues.append(f"recommendation_{index}_budget_fit_denies_fit")
        if issues:
            return VerificationResult("recommendation", "FAIL_RETRYABLE", self.policy, "recommendation violated deterministic constraints", tuple(issues), retryable=True)
        return VerificationResult("recommendation", "PASS", self.policy, "recommendation satisfies deterministic constraints")

    def _verify_creative(self, result: AgentOutput) -> VerificationResult:
        output = _as_mapping(result.get("output"))
        width = int(output.get("width") or 0)
        height = int(output.get("height") or 0)
        if width < 128 or height < 128:
            return VerificationResult("creative_generation", "FAIL_RETRYABLE", self.policy, "creative artifact is below usable resolution", (f"image_resolution_{width}x{height}",), retryable=True)
        if not output.get("artifact_path"):
            return VerificationResult("creative_generation", "FAIL_RETRYABLE", self.policy, "creative artifact path is missing", ("missing_artifact_path",), retryable=True)
        return VerificationResult("creative_generation", "PASS", self.policy, "creative artifact satisfies deterministic constraints")

    def _verify_delivery(self, result: AgentOutput, stage_config: Mapping[str, Any]) -> VerificationResult:
        output = _as_mapping(result.get("output"))
        occasion = _as_mapping(stage_config.get("occasion"))
        issues: list[str] = []
        if not output:
            issues.append("delivery_output_missing")
        if occasion.get("date") and not any(key in output for key in ("delivery_window", "deadline", "latest_ship_date", "plan")):
            issues.append("delivery_timing_missing")
        if issues:
            return VerificationResult("delivery_planner", "FAIL_RETRYABLE", self.policy, "delivery plan missed deterministic logistics constraints", tuple(issues), retryable=True)
        return VerificationResult("delivery_planner", "PASS", self.policy, "delivery plan satisfies deterministic constraints")


def _stage_order_for(config: HarnessConfig, execution_plan: ExecutionPlan | None) -> tuple[str, ...]:
    if config.planner_mode == "authoritative":
        if execution_plan is None:
            return tuple(config.stage_order)
        return tuple(step.stage for step in execution_plan.steps)
    return tuple(config.stage_order)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _budget_hint(stage_config: Mapping[str, Any]) -> str:
    occasion = _as_mapping(stage_config.get("occasion"))
    intent = _as_mapping(stage_config.get("gift_intent"))
    constraints = _as_mapping(intent.get("constraints"))
    return str(stage_config.get("budget_hint") or stage_config.get("budget") or occasion.get("budget_hint") or constraints.get("budget_hint") or "")


def _upper_money_value(value: Any) -> float | None:
    matches = re.findall(r"\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    if not matches:
        return None
    return max(float(match) for match in matches)


def _budget_text_denies_fit(value: str) -> bool:
    lowered = value.lower()
    return any(phrase in lowered for phrase in ("over budget", "exceeds budget", "too expensive", "not within budget", "outside budget"))


def _reject_cycles(plan: ExecutionPlan) -> None:
    graph: dict[str, set[str]] = {step.stage: set(step.dependencies) for step in plan.steps}
    incoming = {stage: len(deps) for stage, deps in graph.items()}
    outgoing: dict[str, set[str]] = defaultdict(set)
    for stage, dependencies in graph.items():
        for dependency in dependencies:
            outgoing[dependency].add(stage)
    queue = deque(stage for stage, count in incoming.items() if count == 0)
    visited = 0
    while queue:
        stage = queue.popleft()
        visited += 1
        for child in outgoing.get(stage, set()):
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    if visited != len(graph):
        raise PlanValidationError("ExecutionPlan contains a dependency cycle")
