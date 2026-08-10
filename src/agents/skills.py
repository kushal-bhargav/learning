from __future__ import annotations

from typing import Any, Mapping, Sequence


def declared_skills(config: Mapping[str, Any]) -> list[str]:
    """Return the stage's configured skill names in declaration order."""
    skills = config.get("skills", [])
    if not isinstance(skills, list):
        return []
    return [str(skill) for skill in skills if str(skill).strip()]


def skills_used(config: Mapping[str, Any], active: Sequence[str] | None = None) -> list[str]:
    """Return active skills, preserving configured order and appending runtime-only skills."""
    declared = declared_skills(config)
    if active is None:
        return declared
    requested = [str(skill) for skill in active if str(skill).strip()]
    ordered = [skill for skill in declared if skill in requested]
    ordered.extend(skill for skill in requested if skill not in ordered)
    return ordered


def add_skill_metadata(output: dict[str, Any], config: Mapping[str, Any], *, active: Sequence[str] | None = None) -> dict[str, Any]:
    """Stamp skill metadata onto an agent output without changing existing payload fields."""
    output.setdefault("skills_declared", declared_skills(config))
    output.setdefault("skills_used", skills_used(config, active))
    return output
