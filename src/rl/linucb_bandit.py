from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


RELATIONSHIP_TYPES = (
    "partner", "parent-child", "sibling", "friend", "colleague",
    "extended-family", "mentor", "other",
)
FORMALITY_LEVELS = ("casual", "semi-formal", "professional", "formal", "other")
AGENCY_BUCKETS = ("low", "mid", "high")


@dataclass(frozen=True, order=True)
class BanditAction:
    recommendation_category: str
    agency_bucket: str
    style_archetype: str

    def __post_init__(self) -> None:
        if not self.recommendation_category:
            raise ValueError("recommendation_category must not be empty")
        if self.agency_bucket not in AGENCY_BUCKETS:
            raise ValueError(f"agency_bucket must be one of {AGENCY_BUCKETS}")
        if not self.style_archetype:
            raise ValueError("style_archetype must not be empty")

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_value(cls, value: "BanditAction | Mapping[str, Any]") -> "BanditAction":
        return value if isinstance(value, cls) else cls(**value)


class ContextEncoder:
    """Encode the relationship context specified in section 05."""

    dimension = len(RELATIONSHIP_TYPES) + 1 + len(FORMALITY_LEVELS) + 1

    def encode(
        self,
        relationship_type: str,
        closeness_score: float,
        occasion_formality: str,
        agency_slider_default: float,
    ) -> NDArray[np.float64]:
        if relationship_type not in RELATIONSHIP_TYPES:
            relationship_type = "other"
        if occasion_formality not in FORMALITY_LEVELS:
            occasion_formality = "other"
        if not 1.0 <= closeness_score <= 5.0:
            raise ValueError("closeness_score must be between 1 and 5")
        if not 0.0 <= agency_slider_default <= 1.0:
            raise ValueError("agency_slider_default must be between 0 and 1")
        vector = np.zeros(self.dimension, dtype=np.float64)
        vector[RELATIONSHIP_TYPES.index(relationship_type)] = 1.0
        offset = len(RELATIONSHIP_TYPES)
        vector[offset] = (float(closeness_score) - 1.0) / 4.0
        offset += 1
        vector[offset + FORMALITY_LEVELS.index(occasion_formality)] = 1.0
        vector[-1] = float(agency_slider_default)
        return vector


def reward_from_feedback(
    rating: float,
    *,
    accept_count: int = 0,
    edit_count: int = 0,
    regenerate_count: int = 0,
    behavior_weight: float = 0.2,
) -> float:
    """Map a 1-5 rating and optional interaction signals to [0, 1]."""
    if not 1.0 <= rating <= 5.0:
        raise ValueError("rating must be between 1 and 5")
    if min(accept_count, edit_count, regenerate_count) < 0:
        raise ValueError("behavior counts must be non-negative")
    if not 0.0 <= behavior_weight <= 1.0:
        raise ValueError("behavior_weight must be between 0 and 1")
    explicit = (float(rating) - 1.0) / 4.0
    interactions = accept_count + edit_count + regenerate_count
    if interactions == 0 or behavior_weight == 0.0:
        return explicit
    implicit = (accept_count + 0.5 * edit_count) / interactions
    return float(np.clip((1.0 - behavior_weight) * explicit + behavior_weight * implicit, 0.0, 1.0))


class LinUCBBandit:
    """Independent-arm LinUCB with one linear model per discrete action."""

    def __init__(
        self,
        actions: Iterable[BanditAction | Mapping[str, Any]],
        context_dim: int,
        *,
        alpha: float = 0.25,
        regularization: float = 1.0,
    ) -> None:
        resolved = tuple(BanditAction.from_value(action) for action in actions)
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("actions must be non-empty and unique")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if alpha < 0 or regularization <= 0:
            raise ValueError("alpha must be non-negative and regularization positive")
        self.actions = resolved
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        self._a = {
            action: np.eye(self.context_dim, dtype=np.float64) * self.regularization
            for action in self.actions
        }
        self._b = {
            action: np.zeros(self.context_dim, dtype=np.float64)
            for action in self.actions
        }
        self.counts = {action: 0 for action in self.actions}

    def scores(self, context: Sequence[float] | NDArray[np.floating]) -> dict[BanditAction, float]:
        x = self._context(context)
        result: dict[BanditAction, float] = {}
        for action in self.actions:
            solved_x = np.linalg.solve(self._a[action], x)
            theta = np.linalg.solve(self._a[action], self._b[action])
            uncertainty = max(0.0, float(x @ solved_x))
            result[action] = float(x @ theta + self.alpha * np.sqrt(uncertainty))
        return result

    def select(self, context: Sequence[float] | NDArray[np.floating]) -> BanditAction:
        scores = self.scores(context)
        return max(self.actions, key=lambda action: scores[action])

    def update(
        self,
        action: BanditAction | Mapping[str, Any],
        context: Sequence[float] | NDArray[np.floating],
        reward: float,
    ) -> None:
        resolved = BanditAction.from_value(action)
        if resolved not in self._a:
            raise KeyError(f"unknown action: {resolved}")
        if not np.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be a finite value between 0 and 1")
        x = self._context(context)
        self._a[resolved] += np.outer(x, x)
        self._b[resolved] += float(reward) * x
        self.counts[resolved] += 1

    def parameters(self, action: BanditAction | Mapping[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        resolved = BanditAction.from_value(action)
        return self._a[resolved].copy(), self._b[resolved].copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "regularization": self.regularization,
            "arms": [
                {
                    "action": asdict(action),
                    "A": self._a[action].tolist(),
                    "b": self._b[action].tolist(),
                    "count": self.counts[action],
                }
                for action in self.actions
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinUCBBandit":
        arms = payload["arms"]
        policy = cls(
            [arm["action"] for arm in arms],
            int(payload["context_dim"]),
            alpha=float(payload["alpha"]),
            regularization=float(payload["regularization"]),
        )
        for arm, action in zip(arms, policy.actions):
            matrix = np.asarray(arm["A"], dtype=np.float64)
            vector = np.asarray(arm["b"], dtype=np.float64)
            if matrix.shape != (policy.context_dim, policy.context_dim) or vector.shape != (policy.context_dim,):
                raise ValueError("invalid saved arm dimensions")
            policy._a[action] = matrix
            policy._b[action] = vector
            policy.counts[action] = int(arm["count"])
        return policy

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinUCBBandit":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _context(self, context: Sequence[float] | NDArray[np.floating]) -> NDArray[np.float64]:
        vector = np.asarray(context, dtype=np.float64)
        if vector.shape != (self.context_dim,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"context must be a finite vector with shape ({self.context_dim},)")
        return vector


def log_session(
    path: str | Path,
    context: Sequence[float],
    action: BanditAction | Mapping[str, Any],
    reward: float,
    **metadata: Any,
) -> None:
    vector = np.asarray(context, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("context must be a finite one-dimensional vector")
    if not np.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError("reward must be a finite value between 0 and 1")
    record = {
        "context": vector.tolist(),
        "action": asdict(BanditAction.from_value(action)),
        "reward": float(reward),
        **metadata,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

