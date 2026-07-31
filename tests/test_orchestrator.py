import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.agents.orchestrator import AgentOrchestrator, GiftSession, HumanAction

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(GiftSession(session_id="session-1", giver_id="giver-1", recipient_id="recipient-1", occasion_id="occasion-1"))


def proposal(stage: str = "recipient_profiling") -> dict:
    return {"stage": stage, "output": {"tone": "warm", "items": ["art"]}, "confidence": 0.9, "rationale": "Grounded in preferences."}


def test_json_contract_and_deep_append_only_provenance() -> None:
    service = orchestrator()
    source = proposal()
    service.append_agent_output(source, timestamp=NOW)
    source["output"]["tone"] = "external mutation"
    entry = service.session.stage_log[0]
    payload = json.loads(service.session.model_dump_json())
    assert set(payload) == {"session_id", "giver_id", "recipient_id", "occasion_id", "stage_log"}
    assert payload["stage_log"][0]["output"] == {"tone": "warm", "items": ["art"]}
    with pytest.raises(TypeError):
        entry.output["tone"] = "overwrite"
    with pytest.raises(ValidationError):
        entry.output = {"tone": "overwrite"}
    with pytest.raises(ValidationError):
        service.session.stage_log = ()


@pytest.mark.parametrize("action", ["accept", "regenerate", "delegate"])
def test_human_actions_append_without_overwriting(action: str) -> None:
    service = orchestrator()
    service.append_agent_output(proposal(), timestamp=NOW)
    original = service.session.stage_log[0]
    service.apply_human_action(action, timestamp=NOW)
    assert service.session.stage_log[0] is original
    assert service.session.stage_log[0].human_action is None
    assert service.session.stage_log[1].human_action == HumanAction(action)
    assert service.session.stage_log[1].proposed_by == "human"


def test_edit_is_a_patch_beside_the_unchanged_proposal() -> None:
    service = orchestrator()
    service.append_agent_output(proposal(), timestamp=NOW)
    service.apply_human_action("edit", human_edit={"tone": "playful"}, timestamp=NOW)
    assert service.session.stage_log[0].output["tone"] == "warm"
    assert service.session.stage_log[1].human_edit == {"tone": "playful"}
    assert service.effective_output("recipient_profiling")["tone"] == "playful"


def test_regenerate_requires_same_stage_and_preserves_both_proposals() -> None:
    service = orchestrator()
    service.append_agent_output(proposal(), timestamp=NOW)
    service.apply_human_action("regenerate", timestamp=NOW)
    with pytest.raises(ValueError, match="same stage"):
        service.append_agent_output(proposal("relationship_analysis"))
    revised = proposal()
    revised["output"] = {"tone": "gentle", "items": ["art", "travel"]}
    service.append_agent_output(revised, timestamp=NOW)
    assert len(service.session.stage_log) == 3
    assert service.session.stage_log[0].output["tone"] == "warm"
    assert service.effective_output("recipient_profiling")["tone"] == "gentle"


def test_delegate_fast_forwards_and_errors_are_explicit() -> None:
    service = orchestrator()
    service.append_agent_output(proposal(), timestamp=NOW)
    service.apply_human_action("delegate", timestamp=NOW)
    service.append_agent_output(proposal("relationship_analysis"), timestamp=NOW)
    assert service.delegated and not service.awaiting_human_action
    assert service.session.stage_log[-1].human_action == HumanAction.DELEGATE
    errors = orchestrator()
    errors.record_error("creative_generation", "GPU out of memory", timestamp=NOW)
    assert errors.session.stage_log[-1].status == "error"
    assert errors.session.stage_log[-1].output["error"] == "GPU out of memory"


def test_pending_stage_and_action_arguments_are_validated() -> None:
    service = orchestrator()
    service.append_agent_output(proposal(), timestamp=NOW)
    with pytest.raises(RuntimeError, match="requires a human action"):
        service.append_agent_output(proposal("relationship_analysis"))
    with pytest.raises(ValueError, match="requires"):
        service.apply_human_action("edit")
    with pytest.raises(ValueError, match="only valid"):
        service.apply_human_action("accept", human_edit={"tone": "playful"})
