from __future__ import annotations

import pytest

from src.harness import HarnessCandidate, HarnessConfig, HarnessRegistry, default_harness_registry
from src.harness.controller import UnsupportedHarnessModeError


def test_default_harness_registry_contains_runtime_distinct_candidates() -> None:
    registry = default_harness_registry()
    candidates = registry.list()

    assert {candidate.candidate_id for candidate in candidates} >= {
        "gmgi_default",
        "gmgi_dynamic_v1",
        "gmgi_dynamic_verified_v1",
        "gmgi_conservative_reliability_v1",
    }
    configs = {candidate.candidate_id: candidate.config for candidate in candidates}
    assert configs["gmgi_default"].routing_mode == "static"
    assert configs["gmgi_dynamic_v1"].routing_mode == "dynamic"
    assert configs["gmgi_dynamic_verified_v1"].verification_policy == "deterministic_constraints"
    assert configs["gmgi_conservative_reliability_v1"].retry_policy == "controller_retry_once"


def test_candidate_identity_is_deterministic_and_config_sensitive() -> None:
    config = HarnessConfig(harness_id="candidate_test")
    first = HarnessCandidate(candidate_id="candidate_test", name="Candidate", config=config, created_at="2026-01-01T00:00:00Z")
    second = HarnessCandidate(candidate_id="candidate_test", name="Candidate", config=config, created_at="2027-01-01T00:00:00Z")
    changed = HarnessCandidate(
        candidate_id="candidate_test",
        name="Candidate",
        config=HarnessConfig(harness_id="candidate_test", routing_mode="dynamic", orchestration_mode="dynamic"),
        created_at="2026-01-01T00:00:00Z",
    )

    assert first.identity == second.identity
    assert first.identity != changed.identity


def test_registry_rejects_duplicate_and_unsupported_candidate() -> None:
    candidate = HarnessCandidate(candidate_id="one", name="One", config=HarnessConfig())
    registry = HarnessRegistry([candidate])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(candidate)

    unsupported = HarnessCandidate(
        candidate_id="bad",
        name="Bad",
        config=HarnessConfig(orchestration_mode="unsupported_dynamic"),
    )
    with pytest.raises(UnsupportedHarnessModeError):
        HarnessRegistry([unsupported])
