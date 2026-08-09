from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Episode:
    session_id: str
    timestamp: str
    context_fingerprint: str
    agent_outputs: dict[str, Any]
    human_actions: dict[str, str]
    composite_reward: float
    clip_score: float | None = None
    prompt_versions: dict[str, str] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Episode":
        return cls(
            session_id=str(payload["session_id"]),
            timestamp=str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            context_fingerprint=str(payload["context_fingerprint"]),
            agent_outputs=dict(payload.get("agent_outputs", {})),
            human_actions={str(k): str(v) for k, v in dict(payload.get("human_actions", {})).items()},
            composite_reward=float(payload.get("composite_reward", 0.0)),
            clip_score=None if payload.get("clip_score") is None else float(payload["clip_score"]),
            prompt_versions={str(k): str(v) for k, v in dict(payload.get("prompt_versions") or {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperienceStore:
    """Append-only JSONL store for session outcomes and self-improvement evidence."""

    def __init__(self, episodes: Iterable[Episode] | None = None, path: str | Path | None = None) -> None:
        self.episodes = list(episodes or [])
        self.path = None if path is None else Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceStore":
        destination = Path(path)
        episodes: list[Episode] = []
        if destination.exists():
            for line in destination.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    episodes.append(Episode.from_mapping(json.loads(line)))
        return cls(episodes, destination)

    def append(self, episode: Episode) -> None:
        if not 0.0 <= float(episode.composite_reward) <= 1.0:
            raise ValueError("composite_reward must be between 0 and 1")
        self.episodes.append(episode)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode.to_dict(), sort_keys=True, default=str) + "\n")

    def retrieve_similar(self, context_fp: str, *, top_k: int = 3) -> list[Episode]:
        target_parts = set(_fingerprint_parts(context_fp))
        scored = []
        for index, episode in enumerate(self.episodes):
            parts = set(_fingerprint_parts(episode.context_fingerprint))
            overlap = len(target_parts & parts)
            exact = 1 if episode.context_fingerprint == context_fp else 0
            scored.append((exact, overlap, episode.composite_reward, index, episode))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1], item[2], item[3]))
        return [episode for *_prefix, episode in scored[: max(0, top_k)]]

    def recent(self, *, n: int = 20) -> list[Episode]:
        return self.episodes[-max(0, n):]

    def save(self, path: str | Path | None = None) -> None:
        destination = Path(path) if path is not None else self.path
        if destination is None:
            raise ValueError("no path supplied for ExperienceStore.save")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for episode in self.episodes:
                handle.write(json.dumps(episode.to_dict(), sort_keys=True, default=str) + "\n")
        self.path = destination


def context_fingerprint(relationship: Mapping[str, Any] | None, occasion: Mapping[str, Any] | None) -> str:
    relationship = relationship or {}
    occasion = occasion or {}
    relationship_type = str(relationship.get("type") or "other").strip().lower() or "other"
    formality = str(occasion.get("formality") or "other").strip().lower() or "other"
    closeness = float(relationship.get("closeness_score") or 3.0)
    if closeness >= 4.0:
        closeness_bucket = "high"
    elif closeness >= 2.5:
        closeness_bucket = "mid"
    else:
        closeness_bucket = "low"
    readable = f"{relationship_type}|{formality}|{closeness_bucket}"
    digest = hashlib.sha1(readable.encode("utf-8")).hexdigest()[:10]
    return f"{readable}|{digest}"


def _fingerprint_parts(value: str) -> tuple[str, ...]:
    return tuple(part for part in str(value).split("|") if part)
