from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import HarnessConfig, default_harness_config
from .controller import HarnessController, UnsupportedHarnessModeError


class HarnessCandidate(BaseModel):
    """Experimental identity and metadata for a concrete HarnessConfig."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    name: str = Field(min_length=1)
    description: str = ""
    config: HarnessConfig
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "gmgi_phase_4_registry"
    compatibility: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @property
    def candidate_hash(self) -> str:
        payload = {
            "candidate_id": self.candidate_id,
            "version": self.version,
            "config_hash": self.config.config_hash,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @property
    def identity(self) -> str:
        return f"{self.candidate_id}:v{self.version}:{self.config.config_hash}:{self.candidate_hash}"

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["candidate_hash"] = self.candidate_hash
        payload["identity"] = self.identity
        payload["config"] = self.config.to_dict()
        return payload


class HarnessRegistry:
    """Small explicit registry for manually defined harness candidates."""

    def __init__(self, candidates: list[HarnessCandidate] | None = None) -> None:
        self._candidates: dict[str, HarnessCandidate] = {}
        for candidate in candidates or []:
            self.register(candidate)

    def register(self, candidate: HarnessCandidate) -> None:
        if candidate.candidate_id in self._candidates:
            raise ValueError(f"Harness candidate already registered: {candidate.candidate_id}")
        validate_candidate(candidate)
        self._candidates[candidate.candidate_id] = candidate

    def get(self, candidate_id: str) -> HarnessCandidate:
        if candidate_id not in self._candidates:
            raise KeyError(f"Unknown harness candidate: {candidate_id}")
        return self._candidates[candidate_id]

    def exists(self, candidate_id: str) -> bool:
        return candidate_id in self._candidates

    def list(self, *, enabled_only: bool = True) -> list[HarnessCandidate]:
        candidates = list(self._candidates.values())
        if enabled_only:
            candidates = [candidate for candidate in candidates if candidate.enabled]
        return candidates


def validate_candidate(candidate: HarnessCandidate) -> None:
    config = candidate.config
    if not candidate.enabled:
        return
    if config.orchestration_mode == "unsupported_dynamic":
        raise UnsupportedHarnessModeError("unsupported_dynamic candidates cannot be registered for execution")
    if config.routing_mode == "dynamic" and config.orchestration_mode not in {"dynamic", "fixed_stage"}:
        raise UnsupportedHarnessModeError(f"Unsupported orchestration/routing combination: {config.orchestration_mode}/{config.routing_mode}")
    HarnessController(config)


def default_harness_candidates() -> list[HarnessCandidate]:
    default = default_harness_config()
    dynamic = HarnessConfig(
        harness_id="gmgi_dynamic_v1",
        harness_version=1,
        description="Dynamic router with advisory planning and default verifier behavior.",
        orchestration_mode="dynamic",
        routing_mode="dynamic",
    )
    dynamic_verified = HarnessConfig(
        harness_id="gmgi_dynamic_verified_v1",
        harness_version=1,
        description="Dynamic router with deterministic constraint verification.",
        orchestration_mode="dynamic",
        routing_mode="dynamic",
        verification_policy="deterministic_constraints",
    )
    conservative = HarnessConfig(
        harness_id="gmgi_conservative_reliability_v1",
        harness_version=1,
        description="Dynamic verified harness with one controller retry and structured skip fallback.",
        orchestration_mode="dynamic",
        routing_mode="dynamic",
        verification_policy="deterministic_constraints",
        retry_policy="controller_retry_once",
        fallback_policy="skip_failed_stage",
    )
    return [
        HarnessCandidate(
            candidate_id="gmgi_default",
            version=1,
            name="Fixed Default",
            description="Current fixed-stage GMGI behavior.",
            config=default,
            compatibility={"runtime": "service_run_stage", "default_behavior": True},
        ),
        HarnessCandidate(
            candidate_id="gmgi_dynamic_v1",
            version=1,
            name="Dynamic",
            description="Phase 3 dynamic routing without live deterministic verifier.",
            config=dynamic,
            compatibility={"runtime": "service_run_stage", "dynamic_routing": True},
        ),
        HarnessCandidate(
            candidate_id="gmgi_dynamic_verified_v1",
            version=1,
            name="Dynamic + Verification",
            description="Dynamic routing plus deterministic constraint verifier.",
            config=dynamic_verified,
            compatibility={"runtime": "service_run_stage", "dynamic_routing": True, "deterministic_verifier": True},
        ),
        HarnessCandidate(
            candidate_id="gmgi_conservative_reliability_v1",
            version=1,
            name="Conservative Reliability",
            description="Dynamic verified harness with controller retry and fallback policies.",
            config=conservative,
            compatibility={"runtime": "service_run_stage", "dynamic_routing": True, "recovery": "retry_then_fallback"},
        ),
    ]


def default_harness_registry() -> HarnessRegistry:
    return HarnessRegistry(default_harness_candidates())
