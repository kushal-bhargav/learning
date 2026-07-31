from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class HumanAction(StrEnum):
    ACCEPT = "accept"
    EDIT = "edit"
    REGENERATE = "regenerate"
    DELEGATE = "delegate"


class AgentInput(TypedDict):
    session: "GiftSession"
    stage_config: dict[str, Any]


class AgentOutput(TypedDict):
    stage: str
    output: dict[str, Any]
    confidence: float | None
    rationale: str | None


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


class StageLogEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    stage: str = Field(min_length=1)
    proposed_by: Literal["agent", "human"]
    output: Mapping[str, Any]
    human_action: HumanAction | None = None
    human_edit: Mapping[str, Any] | None = None
    confidence: float | None = None
    rationale: str | None = None
    timestamp: datetime
    status: Literal["pending", "completed", "error"] = "completed"

    @field_validator("output", "human_edit", mode="after")
    @classmethod
    def freeze_json(cls, value: Mapping[str, Any] | None) -> Any:
        return None if value is None else _freeze(value)

    @field_serializer("output", "human_edit")
    def serialize_json(self, value: Mapping[str, Any] | None) -> Any:
        return None if value is None else _thaw(value)


class GiftSession(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    giver_id: str = Field(min_length=1)
    recipient_id: str = Field(min_length=1)
    occasion_id: str = Field(min_length=1)
    stage_log: tuple[StageLogEntry, ...] = ()

    def append(self, entry: StageLogEntry) -> "GiftSession":
        """Return a new session with one immutable entry appended."""
        return self.model_copy(update={"stage_log": (*self.stage_log, entry)})


class AgentOrchestrator:
    """Sequential blackboard controller with append-only provenance."""

    def __init__(self, session: GiftSession) -> None:
        self._session = session

    @property
    def session(self) -> GiftSession:
        return self._session

    @property
    def delegated(self) -> bool:
        return any(
            entry.human_action == HumanAction.DELEGATE
            for entry in self._session.stage_log
        )

    @property
    def awaiting_human_action(self) -> bool:
        if self.delegated or not self._session.stage_log:
            return False
        return self._session.stage_log[-1].status == "pending"

    def append_agent_output(
        self,
        result: AgentOutput,
        *,
        timestamp: datetime | None = None,
    ) -> GiftSession:
        if self.awaiting_human_action:
            raise RuntimeError("The pending stage requires a human action")
        stage = result["stage"]
        if self._regeneration_stage() not in (None, stage):
            raise ValueError("Regeneration must rerun the same stage")
        action = HumanAction.DELEGATE if self.delegated else None
        entry = StageLogEntry(
            stage=stage,
            proposed_by="agent",
            output=result["output"],
            confidence=result.get("confidence"),
            rationale=result.get("rationale"),
            human_action=action,
            human_edit=None,
            timestamp=timestamp or datetime.now(timezone.utc),
            status="completed" if self.delegated else "pending",
        )
        self._session = self._session.append(entry)
        return self._session

    def apply_human_action(
        self,
        action: HumanAction | str,
        *,
        human_edit: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> GiftSession:
        action = HumanAction(action)
        proposal = self._pending_proposal()
        if action == HumanAction.EDIT and not human_edit:
            raise ValueError("The edit action requires a non-empty human_edit")
        if action != HumanAction.EDIT and human_edit is not None:
            raise ValueError("human_edit is only valid for the edit action")
        entry = StageLogEntry(
            stage=proposal.stage,
            proposed_by="human",
            output=proposal.output,
            human_action=action,
            human_edit=human_edit,
            confidence=proposal.confidence,
            rationale=proposal.rationale,
            timestamp=timestamp or datetime.now(timezone.utc),
            status="completed",
        )
        self._session = self._session.append(entry)
        return self._session

    def record_error(
        self,
        stage: str,
        error: Exception | str,
        *,
        timestamp: datetime | None = None,
    ) -> GiftSession:
        entry = StageLogEntry(
            stage=stage,
            proposed_by="agent",
            output={"error": str(error)},
            timestamp=timestamp or datetime.now(timezone.utc),
            status="error",
        )
        self._session = self._session.append(entry)
        return self._session

    def effective_output(self, stage: str) -> dict[str, Any]:
        for entry in reversed(self._session.stage_log):
            if entry.stage != stage:
                continue
            output = _thaw(entry.output)
            if entry.human_action == HumanAction.EDIT and entry.human_edit:
                output.update(_thaw(entry.human_edit))
            return output
        raise KeyError(f"No output recorded for stage {stage}")

    def _pending_proposal(self) -> StageLogEntry:
        if not self.awaiting_human_action:
            raise RuntimeError("There is no pending agent proposal")
        return self._session.stage_log[-1]

    def _regeneration_stage(self) -> str | None:
        if not self._session.stage_log:
            return None
        latest = self._session.stage_log[-1]
        if latest.human_action == HumanAction.REGENERATE:
            return latest.stage
        return None


