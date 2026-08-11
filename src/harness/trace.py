from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from traceback import format_exception_only
from typing import Any, Callable, Iterable, Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import HarnessConfig, default_harness_config


SECRET_TOKENS = ("api_key", "apikey", "token", "secret", "password", "credential", "authorization")


class ToolCallTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(default_factory=lambda: f"tool-{uuid4().hex[:12]}")
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    latency_seconds: float | None = None
    status: Literal["success", "error"] = "success"
    error_type: str | None = None
    error_message: str | None = None


class DecisionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_type: str
    decision: str
    policy: str | None = None
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RunEventTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage_name: str | None = None
    component: str | None = None
    status: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AgentInvocationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invocation_id: str = Field(default_factory=lambda: f"inv-{uuid4().hex[:12]}")
    run_id: str
    session_id: str
    case_id: str | None = None
    harness_id: str
    harness_version: int
    harness_config_hash: str

    stage_name: str
    agent_name: str
    agent_version: str = "static"
    sequence_number: int
    parent_invocation_id: str | None = None
    dependency_ids: list[str] = Field(default_factory=list)

    status: Literal["success", "error", "timeout", "retry", "skipped"] = "success"
    start_time: str
    end_time: str | None = None
    latency_seconds: float | None = None

    raw_agent_input: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    relevant_memory: Any = None
    constraints: Any = None

    model_provider: str | None = None
    model_name: str | None = None
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, Any] | None = None
    estimated_cost: float | None = None

    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    raw_agent_output: dict[str, Any] | None = None
    structured_output: dict[str, Any] | None = None
    validation_result: dict[str, Any] = Field(default_factory=dict)

    routing_decision: DecisionTrace
    planner_decision: DecisionTrace
    verifier_decision: DecisionTrace
    retry_count: int = 0
    fallback_used: bool = False
    fallback_reason: str | None = None

    error_type: str | None = None
    error_message: str | None = None
    error_reference: str | None = None

    @field_validator("latency_seconds")
    @classmethod
    def validate_latency(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("latency_seconds must be non-negative")
        return value


class AgentRunTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: f"run-{uuid4().hex[:12]}")
    case_id: str | None = None
    session_id: str
    harness_config: dict[str, Any]
    start_time: str
    end_time: str | None = None
    total_latency_seconds: float | None = None
    total_tokens: int | None = None
    total_cost: float | None = None
    final_status: Literal["running", "success", "error", "partial"] = "running"
    termination_reason: str | None = None
    events: list[RunEventTrace] = Field(default_factory=list)
    invocations: list[AgentInvocationTrace] = Field(default_factory=list)

    @property
    def harness_id(self) -> str:
        return str(self.harness_config["harness_id"])

    @property
    def harness_version(self) -> int:
        return int(self.harness_config["harness_version"])

    @property
    def harness_config_hash(self) -> str:
        return str(self.harness_config["config_hash"])

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")


class TraceRecorder:
    """Mutable run trace builder used around the existing fixed workflow."""

    def __init__(self, *, session_id: str, case_id: str | None = None, harness_config: HarnessConfig | None = None, run_id: str | None = None) -> None:
        self.harness_config = harness_config or default_harness_config()
        self._started = time.perf_counter()
        self.run_trace = AgentRunTrace(
            run_id=run_id or f"run-{uuid4().hex[:12]}",
            case_id=case_id,
            session_id=session_id,
            harness_config=self.harness_config.to_dict(),
            start_time=_now(),
        )
        self._active_invocation: AgentInvocationTrace | None = None
        self.record_event("run_started", component="trace_recorder")

    @property
    def run_id(self) -> str:
        return self.run_trace.run_id

    @property
    def invocations(self) -> list[AgentInvocationTrace]:
        return self.run_trace.invocations

    @contextlib.contextmanager
    def invocation(
        self,
        *,
        stage_name: str,
        agent_name: str,
        agent_version: str,
        raw_agent_input: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        relevant_memory: Any = None,
        constraints: Any = None,
        model_provider: str | None = None,
        model_name: str | None = None,
        model_parameters: Mapping[str, Any] | None = None,
        routing_decision: DecisionTrace | None = None,
        planner_decision: DecisionTrace | None = None,
        verifier_decision: DecisionTrace | None = None,
        retry_count: int = 0,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        parent_invocation_id: str | None = None,
        dependency_ids: Iterable[str] | None = None,
    ) -> Iterable[AgentInvocationTrace]:
        started = time.perf_counter()
        invocation = AgentInvocationTrace(
            run_id=self.run_trace.run_id,
            session_id=self.run_trace.session_id,
            case_id=self.run_trace.case_id,
            harness_id=self.harness_config.harness_id,
            harness_version=self.harness_config.harness_version,
            harness_config_hash=self.harness_config.config_hash,
            stage_name=stage_name,
            agent_name=agent_name,
            agent_version=agent_version,
            sequence_number=len(self.run_trace.invocations) + 1,
            parent_invocation_id=parent_invocation_id,
            dependency_ids=list(dependency_ids or []),
            start_time=_now(),
            raw_agent_input=sanitize(raw_agent_input),
            context=sanitize(context or {}),
            relevant_memory=sanitize(relevant_memory),
            constraints=sanitize(constraints),
            model_provider=model_provider,
            model_name=model_name,
            model_parameters=sanitize(model_parameters or {}),
            routing_decision=routing_decision or static_routing_decision(stage_name),
            planner_decision=planner_decision or advisory_planner_decision(),
            verifier_decision=verifier_decision or no_live_verifier_decision(),
            retry_count=max(0, int(retry_count)),
            fallback_used=bool(fallback_used),
            fallback_reason=fallback_reason,
        )
        previous = self._active_invocation
        self._active_invocation = invocation
        self.record_event("agent_started", stage_name=stage_name, component=agent_name)
        try:
            yield invocation
            invocation.status = "success"
        except Exception as exc:
            invocation.status = "error"
            invocation.error_type = type(exc).__name__
            invocation.error_message = str(exc)
            invocation.error_reference = "".join(format_exception_only(type(exc), exc)).strip()
            raise
        finally:
            invocation.end_time = _now()
            invocation.latency_seconds = time.perf_counter() - started
            self.run_trace.invocations.append(invocation)
            self.record_event(
                "agent_completed" if invocation.status == "success" else "agent_failed",
                stage_name=stage_name,
                component=agent_name,
                status=invocation.status,
                details={"latency_seconds": invocation.latency_seconds, "error_type": invocation.error_type},
            )
            self._active_invocation = previous

    def add_tool_call(self, tool_call: ToolCallTrace) -> None:
        if self._active_invocation is not None:
            self._active_invocation.tool_calls.append(tool_call)
        self.record_event(
            "tool_completed",
            component=tool_call.tool_name,
            status=tool_call.status,
            details={"latency_seconds": tool_call.latency_seconds, "error_type": tool_call.error_type},
        )

    def record_event(
        self,
        event_type: str,
        *,
        stage_name: str | None = None,
        component: str | None = None,
        status: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_trace.events.append(
            RunEventTrace(
                event_type=event_type,
                stage_name=stage_name,
                component=component,
                status=status,
                details=sanitize(details or {}),
            )
        )

    def record_timeout(
        self,
        *,
        stage_name: str,
        agent_name: str,
        agent_version: str = "static",
        raw_agent_input: Mapping[str, Any] | None = None,
        latency_seconds: float | None = None,
        error_message: str | None = None,
        dependency_ids: Iterable[str] | None = None,
    ) -> AgentInvocationTrace:
        invocation = AgentInvocationTrace(
            run_id=self.run_trace.run_id,
            session_id=self.run_trace.session_id,
            case_id=self.run_trace.case_id,
            harness_id=self.harness_config.harness_id,
            harness_version=self.harness_config.harness_version,
            harness_config_hash=self.harness_config.config_hash,
            stage_name=stage_name,
            agent_name=agent_name,
            agent_version=agent_version,
            sequence_number=len(self.run_trace.invocations) + 1,
            dependency_ids=list(dependency_ids or []),
            status="timeout",
            start_time=_now(),
            end_time=_now(),
            latency_seconds=latency_seconds,
            raw_agent_input=sanitize(raw_agent_input or {}),
            routing_decision=static_routing_decision(stage_name),
            planner_decision=advisory_planner_decision(),
            verifier_decision=no_live_verifier_decision(),
            error_type="TimeoutError",
            error_message=error_message,
        )
        self.run_trace.invocations.append(invocation)
        self.record_event(
            "run_timeout",
            stage_name=stage_name,
            component=agent_name,
            status="timeout",
            details={
                "timeout_stage": stage_name,
                "timeout_component": agent_name,
                "timeout_duration": latency_seconds,
                "timeout_type": "agent_timeout",
                "request_cancelled": False,
                "cancellation_supported": False,
                "cancellation_note": "Benchmark agent timeout uses a worker thread join; blocking provider calls may continue until their own client timeout.",
            },
        )
        return invocation

    def finish(self, final_status: Literal["success", "error", "partial"] | None = None, termination_reason: str | None = None) -> AgentRunTrace:
        self.run_trace.end_time = _now()
        self.run_trace.total_latency_seconds = time.perf_counter() - self._started
        if final_status is None:
            if any(invocation.status == "error" for invocation in self.run_trace.invocations):
                final_status = "partial"
            else:
                final_status = "success"
        self.run_trace.final_status = final_status
        self.run_trace.termination_reason = termination_reason or ("completed" if final_status == "success" else final_status)
        self.record_event(
            "run_completed" if final_status == "success" else "run_failed",
            component="trace_recorder",
            status=final_status,
            details={"termination_reason": self.run_trace.termination_reason},
        )
        return self.run_trace


_RECORDER: contextvars.ContextVar[TraceRecorder | None] = contextvars.ContextVar("gmgi_trace_recorder", default=None)
_TOOL_PERMISSION_CHECK: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar("gmgi_tool_permission_check", default=None)


def active_trace_recorder() -> TraceRecorder | None:
    return _RECORDER.get()


@contextlib.contextmanager
def trace_recorder_context(recorder: TraceRecorder | None):
    token = _RECORDER.set(recorder)
    try:
        yield
    finally:
        _RECORDER.reset(token)


@contextlib.contextmanager
def tool_policy_context(checker: Callable[[str], None] | None):
    token = _TOOL_PERMISSION_CHECK.set(checker)
    try:
        yield
    finally:
        _TOOL_PERMISSION_CHECK.reset(token)


def ensure_tool_allowed(tool_name: str) -> None:
    checker = _TOOL_PERMISSION_CHECK.get()
    if checker is not None:
        checker(tool_name)


def record_tool_call(
    tool_name: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    result: Any = None,
    latency_seconds: float | None = None,
    status: Literal["success", "error"] = "success",
    error: Exception | None = None,
) -> None:
    recorder = active_trace_recorder()
    if recorder is None:
        return
    recorder.add_tool_call(
        ToolCallTrace(
            tool_name=tool_name,
            arguments=sanitize(arguments or {}),
            result=sanitize(result),
            latency_seconds=latency_seconds,
            status="error" if error is not None else status,
            error_type=None if error is None else type(error).__name__,
            error_message=None if error is None else str(error),
        )
    )


def static_routing_decision(stage_name: str) -> DecisionTrace:
    return DecisionTrace(
        decision_type="routing",
        decision=stage_name,
        policy="static_stage_order",
        reason="Current Phase 1 tracing observes existing STAGES/_run_stage routing.",
    )


def advisory_planner_decision() -> DecisionTrace:
    return DecisionTrace(
        decision_type="planner",
        decision="advisory_only",
        policy="multi_agent_planning_output_not_execution_authoritative",
        reason="Planner output may inform downstream context but does not control runtime execution in Phase 1.",
    )


def no_live_verifier_decision() -> DecisionTrace:
    return DecisionTrace(
        decision_type="verifier",
        decision="not_available",
        policy="offline_evaluation_only",
        reason="Live semantic verifier gate is not implemented in Phase 1.",
    )


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(token in key_text.lower() for token in SECRET_TOKENS):
                result[key_text] = "[REDACTED]"
            else:
                result[key_text] = sanitize(item)
        return result
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if hasattr(value, "model_dump"):
        return sanitize(value.model_dump(mode="json"))
    if hasattr(value, "tolist"):
        return sanitize(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def payload_hash(value: Any) -> str:
    serialized = json.dumps(sanitize(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
