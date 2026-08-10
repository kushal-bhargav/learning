from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.agents.experience_store import Episode, ExperienceStore
from src.agents.orchestrator import AgentInput, AgentOutput, GiftSession


AgentFactory = Callable[[], Any]


@dataclass(frozen=True)
class ReplayResult:
    status: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "details": self.details}


def perturb_mapping(value: Mapping[str, Any], path: str, replacement: Any) -> dict[str, Any]:
    cloned = copy.deepcopy(dict(value))
    cursor: Any = cloned
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("path must not be empty")
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise KeyError(path)
        cursor = cursor.setdefault(part, {})
    if not isinstance(cursor, dict):
        raise KeyError(path)
    cursor[parts[-1]] = replacement
    return cloned


def diff_outputs(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_text = json.dumps(before, sort_keys=True, default=str)
    after_text = json.dumps(after, sort_keys=True, default=str)
    before_tokens = _tokens(before_text)
    after_tokens = _tokens(after_text)
    overlap = len(before_tokens & after_tokens)
    union = len(before_tokens | after_tokens)
    return {
        "changed": before != after,
        "token_jaccard": None if union == 0 else overlap / union,
        "before_size": len(before_text),
        "after_size": len(after_text),
    }


def run_self_consistency(
    agent_factory: AgentFactory,
    session: GiftSession,
    stage_config: Mapping[str, Any],
    *,
    runs: int = 3,
) -> dict[str, Any]:
    outputs: list[AgentOutput] = []
    for _ in range(max(1, runs)):
        agent = agent_factory()
        outputs.append(agent.run({"session": session, "stage_config": dict(stage_config)}))
    diffs = [
        diff_outputs(outputs[0]["output"], output["output"])
        for output in outputs[1:]
    ]
    similarities = [item["token_jaccard"] for item in diffs if item["token_jaccard"] is not None]
    return {
        "runs": len(outputs),
        "stable": all(not item["changed"] for item in diffs),
        "mean_token_jaccard": None if not similarities else sum(similarities) / len(similarities),
        "diffs": diffs,
    }


def run_counterfactual(
    agent_factory: AgentFactory,
    session: GiftSession,
    stage_config: Mapping[str, Any],
    *,
    perturbation_path: str,
    replacement: Any,
) -> dict[str, Any]:
    original_config = dict(stage_config)
    perturbed_config = perturb_mapping(original_config, perturbation_path, replacement)
    original = agent_factory().run({"session": session, "stage_config": original_config})
    candidate = agent_factory().run({"session": session, "stage_config": perturbed_config})
    return {
        "perturbation": {"path": perturbation_path, "replacement": replacement},
        "original_stage": original["stage"],
        "candidate_stage": candidate["stage"],
        "diff": diff_outputs(original["output"], candidate["output"]),
        "original_output": original["output"],
        "candidate_output": candidate["output"],
    }


def evaluate_store(path: str, *, limit: int | None = None) -> dict[str, Any]:
    store = ExperienceStore.load(path)
    episodes = store.episodes[-limit:] if limit else store.episodes
    return {
        "phase": 3,
        "episode_count": len(episodes),
        "summary": {
            "status": "not_run",
            "reason": "ExperienceStore episodes do not include raw per-stage AgentInput payloads required for safe replay.",
        },
        "episodes": [_insufficient_context(episode).to_dict() for episode in episodes],
    }


def _insufficient_context(episode: Episode) -> ReplayResult:
    return ReplayResult(
        status="insufficient_logged_context",
        details={
            "session_id": episode.session_id,
            "available_fields": ["agent_outputs", "human_actions", "context_fingerprint", "prompt_versions", "composite_reward"],
            "required_for_replay": ["session", "stage_config"],
        },
    )


def _tokens(value: str) -> set[str]:
    return {token for token in value.lower().replace("_", " ").split() if len(token) > 2}

