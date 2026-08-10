from __future__ import annotations

import json
from typing import Any, Mapping, Protocol

from src.agents.experience_store import Episode, ExperienceStore


class StructuredJudge(Protocol):
    def generate(self, **kwargs: Any) -> dict[str, Any]: ...


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fitness_for_purpose", "completeness", "unsupported_claim_risk", "rationale"],
    "properties": {
        "fitness_for_purpose": {"type": "number", "minimum": 0, "maximum": 1},
        "completeness": {"type": "number", "minimum": 0, "maximum": 1},
        "unsupported_claim_risk": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
}


DEFAULT_CONTRACTS = {
    "recipient_profiling": "Extract recipient interests, communication style, and gift history from supplied context only.",
    "relationship_analysis": "Assess relationship closeness, tone, formality, risk flags, and agency default from supplied context only.",
    "gift_intent_reasoning": "Infer gifting purpose, constraints, preferences, and visual artifact direction from upstream outputs.",
    "multi_agent_planning": "Produce a bounded executable plan using only available GMGI agents.",
    "recommendation": "Rank three gift concepts grounded in profile, relationship, intent, occasion, budget, and feedback hints.",
    "creative_generation": "Generate visual artifact metadata and prompt-grounded image outputs for the selected gift direction.",
    "greeting_story": "Generate an original greeting/story grounded in retrieved memories and tone guidance.",
    "delivery_planner": "Produce simulated delivery timing only, with no real purchase or shipment.",
}


def judge_purpose_alignment(
    agent_contract: str,
    input_context: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    judge: StructuredJudge,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    result = judge.generate(
        system_prompt=(
            "You are an isolated evaluator. Judge the agent output only against the declared contract "
            "and the actual input context. Do not use external gold answers."
        ),
        user_prompt=json.dumps(
            {"agent_contract": agent_contract, "input_context": input_context, "output": output},
            ensure_ascii=False,
            indent=2,
        ),
        schema=JUDGE_SCHEMA,
        temperature=0.0,
        model=model,
    )
    return {
        "fitness_for_purpose": float(result["fitness_for_purpose"]),
        "completeness": float(result["completeness"]),
        "unsupported_claim_risk": float(result["unsupported_claim_risk"]),
        "rationale": str(result["rationale"]),
    }


def evaluate_store(
    path: str,
    *,
    judge: StructuredJudge | None = None,
    model: str = "gpt-4o-mini",
    limit: int | None = None,
) -> dict[str, Any]:
    store = ExperienceStore.load(path)
    episodes = store.episodes[-limit:] if limit else store.episodes
    if judge is None:
        return {
            "phase": 4,
            "episode_count": len(episodes),
            "summary": {
                "status": "not_run",
                "reason": "Purpose-alignment judge calls are disabled. Pass --enable-judge to call the configured judge model.",
            },
            "episodes": [],
        }
    reports = [evaluate_episode(episode, judge=judge, model=model) for episode in episodes]
    scores = [
        stage["fitness_for_purpose"]
        for report in reports
        for stage in report["stages"]
    ]
    return {
        "phase": 4,
        "episode_count": len(reports),
        "summary": {"mean_fitness_for_purpose": None if not scores else sum(scores) / len(scores)},
        "episodes": reports,
    }


def evaluate_episode(episode: Episode, *, judge: StructuredJudge, model: str) -> dict[str, Any]:
    upstream: dict[str, Any] = {}
    stages = []
    for stage, output in episode.agent_outputs.items():
        contract = DEFAULT_CONTRACTS.get(stage, f"Evaluate stage {stage} against its supplied input.")
        result = judge_purpose_alignment(contract, upstream, output, judge=judge, model=model)
        stages.append({"stage": stage, **result})
        upstream[stage] = output
    return {"session_id": episode.session_id, "stages": stages}

