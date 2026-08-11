from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_STAGE_ORDER = (
    "recipient_profiling",
    "relationship_analysis",
    "gift_intent_reasoning",
    "multi_agent_planning",
    "recommendation",
    "creative_generation",
    "greeting_story",
    "delivery_planner",
)


class HarnessConfig(BaseModel):
    """First-class, immutable description of the current GMGI harness behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    harness_id: str = Field(default="gmgi_default", min_length=1)
    harness_version: int = Field(default=1, ge=1)
    description: str = "Default fixed-stage GMGI workflow with instrumentation."

    orchestration_mode: Literal["fixed_stage", "dynamic", "unsupported_dynamic"] = "fixed_stage"
    stage_execution_mode: Literal["service_run_stage"] = "service_run_stage"
    stage_order: tuple[str, ...] = DEFAULT_STAGE_ORDER

    planner_mode: Literal["advisory", "authoritative"] = "advisory"
    routing_mode: Literal["static", "dynamic"] = "static"
    memory_policy: Literal["fixture_graph_plus_experience_retrieval"] = "fixture_graph_plus_experience_retrieval"
    context_policy: Literal["stage_specific_static_context"] = "stage_specific_static_context"
    tool_policy: Literal["agent_local_tools", "deny_all_tools"] = "agent_local_tools"
    verification_policy: Literal["schema_validation_with_offline_evals", "controller_schema_gate", "deterministic_constraints"] = "schema_validation_with_offline_evals"
    retry_policy: Literal["agent_local_retries", "controller_retry_once"] = "agent_local_retries"
    fallback_policy: Literal["agent_local_fallbacks", "skip_failed_stage"] = "agent_local_fallbacks"
    stopping_policy: Literal["fixed_stage_completion_or_human_gate", "stop_before_delivery"] = "fixed_stage_completion_or_human_gate"
    budget_policy: Literal["local_retry_and_eval_timeout_only"] = "local_retry_and_eval_timeout_only"
    model_policy: Literal["provider_env_and_agent_config"] = "provider_env_and_agent_config"
    human_oversight_policy: Literal["proposal_review_accept_edit_regenerate_delegate"] = "proposal_review_accept_edit_regenerate_delegate"
    max_controller_steps: int = Field(default=64, ge=1)
    max_agent_invocations: int = Field(default=64, ge=1)
    max_retries_per_action: int = Field(default=1, ge=0)
    max_total_retries: int = Field(default=8, ge=0)

    @field_validator("stage_order")
    @classmethod
    def validate_stage_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("stage_order must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("stage_order must not contain duplicate stages")
        if any(not str(stage).strip() for stage in value):
            raise ValueError("stage names must not be empty")
        return value

    @property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @property
    def identity(self) -> str:
        return f"{self.harness_id}:v{self.harness_version}:{self.config_hash}"

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["config_hash"] = self.config_hash
        payload["identity"] = self.identity
        return payload


def default_harness_config() -> HarnessConfig:
    return HarnessConfig()
