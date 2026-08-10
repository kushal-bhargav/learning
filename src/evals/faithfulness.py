from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from src.agents.experience_store import Episode, ExperienceStore


class StructuredVerifier(Protocol):
    def generate(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    supported: bool
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "supported": self.supported, "evidence": self.evidence}


def evaluate_faithfulness(
    input_context: Mapping[str, Any] | str,
    output: Mapping[str, Any] | str,
    *,
    verifier: StructuredVerifier | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    claims = decompose_claims(output)
    if not claims:
        return {"faithfulness_score": None, "unsupported_claims": [], "claim_checks": []}
    if verifier is not None:
        checks = _llm_verify(input_context, claims, verifier=verifier, model=model)
    else:
        checks = [_lexical_check(input_context, claim) for claim in claims]
    supported = sum(1 for check in checks if check.supported)
    unsupported = [check.claim for check in checks if not check.supported]
    return {
        "faithfulness_score": supported / len(checks),
        "unsupported_claims": unsupported,
        "claim_checks": [check.to_dict() for check in checks],
    }


def evaluate_store(path: str, *, limit: int | None = None) -> dict[str, Any]:
    store = ExperienceStore.load(path)
    episodes = store.episodes[-limit:] if limit else store.episodes
    reports = [evaluate_episode(episode) for episode in episodes]
    scores = [
        float(stage["faithfulness_score"])
        for report in reports
        for stage in report["stages"]
        if stage["faithfulness_score"] is not None
    ]
    return {
        "phase": 2,
        "episode_count": len(reports),
        "summary": {
            "mean_faithfulness": None if not scores else sum(scores) / len(scores),
            "checked_stage_count": len(scores),
        },
        "episodes": reports,
    }


def evaluate_episode(episode: Episode | Mapping[str, Any]) -> dict[str, Any]:
    item = episode if isinstance(episode, Episode) else Episode.from_mapping(episode)
    outputs = dict(item.agent_outputs)
    stages = []
    upstream: dict[str, Any] = {}
    for stage, output in outputs.items():
        context = {**upstream, "context_fingerprint": item.context_fingerprint}
        result = evaluate_faithfulness(context, output)
        stages.append({"stage": stage, **result})
        upstream[stage] = output
    return {"session_id": item.session_id, "stages": stages}


def decompose_claims(value: Mapping[str, Any] | Sequence[Any] | str) -> list[str]:
    claims: list[str] = []
    _collect_claims(value, claims)
    cleaned = []
    for claim in claims:
        text = " ".join(str(claim).split())
        if len(text) >= 3 and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _collect_claims(value: Any, claims: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"prompt_version", "skills_used", "artifact_path"}:
                continue
            if isinstance(child, (Mapping, list, tuple)):
                _collect_claims(child, claims)
            elif child is not None:
                claims.append(f"{key}: {child}")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_claims(child, claims)
    elif isinstance(value, str):
        for part in re.split(r"(?<=[.!?])\s+|\n+", value):
            if part.strip():
                claims.append(part.strip())


def _lexical_check(input_context: Mapping[str, Any] | str, claim: str) -> ClaimCheck:
    context_text = _normalize(json.dumps(input_context, ensure_ascii=False) if isinstance(input_context, Mapping) else input_context)
    claim_text = _normalize(claim)
    tokens = [token for token in claim_text.split() if len(token) > 2 and token not in _STOPWORDS]
    if not tokens:
        return ClaimCheck(claim, True, "claim has no content tokens")
    hits = [token for token in tokens if token in context_text]
    support = len(hits) / len(tokens)
    return ClaimCheck(claim, support >= 0.5, ", ".join(hits[:8]) or None)


def _llm_verify(input_context: Mapping[str, Any] | str, claims: Sequence[str], *, verifier: StructuredVerifier, model: str) -> list[ClaimCheck]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim", "supported", "evidence"],
                    "properties": {
                        "claim": {"type": "string"},
                        "supported": {"type": "boolean"},
                        "evidence": {"type": ["string", "null"]},
                    },
                },
            }
        },
    }
    result = verifier.generate(
        system_prompt="Verify whether each claim is entailed by the supplied input context. Do not use outside knowledge.",
        user_prompt=json.dumps({"input_context": input_context, "claims": list(claims)}, ensure_ascii=False, indent=2),
        schema=schema,
        temperature=0.0,
        model=model,
    )
    checks = []
    for item in result.get("claims", []):
        checks.append(ClaimCheck(str(item.get("claim", "")), bool(item.get("supported")), item.get("evidence")))
    return checks or [_lexical_check(input_context, claim) for claim in claims]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\s_-]", " ", value.lower())


_STOPWORDS = {
    "and",
    "are",
    "for",
    "the",
    "this",
    "that",
    "with",
    "from",
    "into",
    "must",
    "should",
    "would",
    "could",
    "will",
    "has",
    "have",
    "was",
    "were",
    "confidence",
}

