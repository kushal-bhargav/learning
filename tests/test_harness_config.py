from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.harness import HarnessConfig, default_harness_config
from src.harness.config import DEFAULT_STAGE_ORDER


def test_default_harness_config_has_stable_identity() -> None:
    first = default_harness_config()
    second = default_harness_config()

    assert first.harness_id == "gmgi_default"
    assert first.harness_version == 1
    assert first.stage_order == DEFAULT_STAGE_ORDER
    assert first.config_hash == second.config_hash
    assert first.identity == second.identity


def test_harness_config_round_trips_through_json_payload() -> None:
    config = HarnessConfig()
    restored = HarnessConfig(**config.model_dump(mode="json"))

    assert restored == config
    assert restored.to_dict()["identity"] == config.identity
    assert restored.to_dict()["config_hash"] == config.config_hash


def test_harness_config_rejects_unknown_and_invalid_values() -> None:
    dynamic_orchestration = HarnessConfig(orchestration_mode="dynamic")
    assert dynamic_orchestration.orchestration_mode == "dynamic"

    dynamic_routing = HarnessConfig(routing_mode="dynamic")
    assert dynamic_routing.routing_mode == "dynamic"

    with pytest.raises(ValidationError):
        HarnessConfig(stage_order=("recipient_profiling", "recipient_profiling"))

    with pytest.raises(ValidationError):
        HarnessConfig(extra_field=True)  # type: ignore[call-arg]


def test_harness_version_changes_identity() -> None:
    current = HarnessConfig(harness_version=1)
    next_version = HarnessConfig(harness_version=2)

    assert current.config_hash != next_version.config_hash
    assert current.identity != next_version.identity
