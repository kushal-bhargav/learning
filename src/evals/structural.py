from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import ValidationError, validate

from src.agents.experience_store import Episode, ExperienceStore


CONFIG_DIR = Path(__file__).parents[1] / "agents" / "configs"
OBSERVABILITY_KEYS = {"prompt_version", "skills_used"}
LOCAL_STAGE_SCHEMAS: dict[str, dict[str, Any]] = {
    "creative_generation": {
        "type": "object",
        "required": ["output", "confidence", "rationale"],
        "properties": {
            "output": {
                "type": "object",
                "required": ["artifact_path", "media_type", "width", "height", "agency_slider", "seed"],
                "properties": {
                    "artifact_path": {"type": "string"},
                    "media_type": {"type": "string"},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1},
                    "agency_slider": {"type": "number", "minimum": 0, "maximum": 1},
                    "seed": {"type": "integer"},
                },
                "additionalProperties": True,
            },
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "rationale": {"type": ["string", "null"]},
        },
    },
    "delivery_planner": {
        "type": "object",
        "required": ["output", "confidence", "rationale"],
        "properties": {
            "output": {
                "type": "object",
                "required": ["mode", "channel", "planned_send_date", "occasion_date", "status", "disclaimer"],
                "properties": {
                    "mode": {"type": "string"},
                    "channel": {"type": "string"},
                    "planned_send_date": {"type": "string"},
                    "occasion_date": {"type": "string"},
                    "status": {"type": "string"},
                    "disclaimer": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "rationale": {"type": ["string", "null"]},
        },
    },
}
DEFAULT_STAGE_ORDER = (
    "recipient_profiling",
    "relationship_analysis",
    "gift_intent_reasoning",
    "multi_agent_planning",
    "recommendation",
    "creative_generation",
    "greeting_story",
    "delivery_planner",
)


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float | None
    passed: bool | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
        }


def evaluate_episode(episode: Episode | Mapping[str, Any], *, input_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    item = episode if isinstance(episode, Episode) else Episode.from_mapping(episode)
    outputs = dict(item.agent_outputs)
    results = [
        schema_conformance(outputs),
        constraint_satisfaction(outputs, input_context=input_context),
        dag_validity(outputs),
        provenance_traceability(outputs, input_context=input_context),
    ]
    scored = [result.score for result in results if result.score is not None]
    return {
        "session_id": item.session_id,
        "context_fingerprint": item.context_fingerprint,
        "metrics": [result.to_dict() for result in results],
        "overall_score": None if not scored else sum(scored) / len(scored),
    }


def evaluate_store(path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    store = ExperienceStore.load(path)
    episodes = store.episodes[-limit:] if limit else store.episodes
    reports = [evaluate_episode(episode) for episode in episodes]
    return {
        "phase": 1,
        "episode_count": len(reports),
        "summary": _summarize_reports(reports),
        "episodes": reports,
    }


def schema_conformance(agent_outputs: Mapping[str, Any]) -> MetricResult:
    stage_results: dict[str, dict[str, Any]] = {}
    valid_count = 0
    checked_count = 0
    for stage, output in agent_outputs.items():
        schema = _load_schema(stage)
        if schema is None:
            stage_results[stage] = {"checked": False, "valid": None, "error": "no schema config"}
            continue
        checked_count += 1
        core_output, ignored_keys = _core_output(output)
        payload = {"output": core_output, "confidence": None, "rationale": None}
        try:
            validate(instance=payload, schema=schema)
            stage_results[stage] = {"checked": True, "valid": True, "ignored_observability_keys": ignored_keys}
            valid_count += 1
        except ValidationError as exc:
            stage_results[stage] = {"checked": True, "valid": False, "error": exc.message, "ignored_observability_keys": ignored_keys}
    score = None if checked_count == 0 else valid_count / checked_count
    return MetricResult("schema_conformance", score, None if score is None else score == 1.0, {"stages": stage_results})


def constraint_satisfaction(agent_outputs: Mapping[str, Any], *, input_context: Mapping[str, Any] | None = None) -> MetricResult:
    constraints = _extract_constraints(agent_outputs, input_context)
    checks: list[dict[str, Any]] = []
    recommendations = _as_list(_get(agent_outputs, "recommendation", "recommendations"))
    delivery = _as_mapping(agent_outputs.get("delivery_planner"))
    intent_constraints = _as_mapping(_get(agent_outputs, "gift_intent_reasoning", "constraints"))

    budget_hint = str(constraints.get("budget_hint") or intent_constraints.get("budget_hint") or "")
    if budget_hint and recommendations:
        for index, recommendation in enumerate(recommendations):
            budget_fit = str(_as_mapping(recommendation).get("budget_fit") or "")
            checks.append(
                {
                    "name": f"recommendation_{index + 1}_budget_mentions_fit",
                    "passed": bool(budget_fit.strip()),
                    "constraint": budget_hint,
                }
            )

    exclusions = [str(item).lower() for item in _as_list(constraints.get("exclusions")) if str(item).strip()]
    if exclusions and recommendations:
        for index, recommendation in enumerate(recommendations):
            text = json.dumps(recommendation, ensure_ascii=False).lower()
            checks.append(
                {
                    "name": f"recommendation_{index + 1}_exclusions_absent",
                    "passed": not any(exclusion in text for exclusion in exclusions),
                    "constraint": exclusions,
                }
            )

    delivery_constraints = _as_list(intent_constraints.get("delivery_constraints"))
    if any("simulated" in str(item).lower() for item in delivery_constraints) and delivery:
        disclaimer = str(delivery.get("disclaimer") or "").lower()
        checks.append(
            {
                "name": "delivery_simulated_only",
                "passed": delivery.get("status") == "simulated" and ("no shipment" in disclaimer or "simulated" in disclaimer),
                "constraint": delivery_constraints,
            }
        )

    risk_flags = _as_list(_get(agent_outputs, "relationship_analysis", "risk_flags"))
    if risk_flags:
        checks.append({"name": "relationship_risk_flags_structured", "passed": all(isinstance(flag, str) for flag in risk_flags)})

    passed = sum(1 for check in checks if check["passed"])
    score = None if not checks else passed / len(checks)
    return MetricResult("constraint_satisfaction", score, None if score is None else score == 1.0, {"checks": checks})


def dag_validity(agent_outputs: Mapping[str, Any]) -> MetricResult:
    plan = _as_mapping(agent_outputs.get("multi_agent_planning"))
    sequence = [str(stage) for stage in _as_list(plan.get("agent_sequence"))]
    dependencies = [_as_mapping(item) for item in _as_list(plan.get("dependencies"))]
    known = set(sequence)
    errors: list[str] = []
    if not sequence:
        errors.append("missing agent_sequence")
    if len(sequence) != len(set(sequence)):
        errors.append("agent_sequence contains duplicates")
    for dependency in dependencies:
        after = str(dependency.get("after") or "")
        before = str(dependency.get("before") or "")
        if after not in known or before not in known:
            errors.append(f"unresolvable dependency: {after}->{before}")
    if _has_cycle(sequence, dependencies):
        errors.append("dependency graph contains a cycle")
    ordering_errors = _stage_order_errors(sequence)
    errors.extend(ordering_errors)
    passed = not errors
    return MetricResult("dag_validity", 1.0 if passed else 0.0, passed, {"sequence": sequence, "dependencies": dependencies, "errors": errors})


def provenance_traceability(agent_outputs: Mapping[str, Any], *, input_context: Mapping[str, Any] | None = None) -> MetricResult:
    upstream_text = _upstream_text(agent_outputs, input_context)
    items: list[dict[str, Any]] = []
    for recommendation in _as_list(_get(agent_outputs, "recommendation", "recommendations")):
        rec = _as_mapping(recommendation)
        for evidence in _as_list(rec.get("evidence")):
            evidence_text = str(evidence)
            items.append(
                {
                    "stage": "recommendation",
                    "field": "evidence",
                    "value": evidence_text,
                    "traceable": _contains(upstream_text, evidence_text),
                }
            )
    for reference in _as_list(_get(agent_outputs, "greeting_story", "memory_references")):
        ref_text = str(reference)
        items.append(
            {
                "stage": "greeting_story",
                "field": "memory_references",
                "value": ref_text,
                "traceable": _contains(upstream_text, ref_text),
            }
        )
    preferences = _as_list(_get(agent_outputs, "gift_intent_reasoning", "preferences"))
    for preference in preferences:
        pref = _as_mapping(preference)
        value = str(pref.get("value") or pref.get("name") or "")
        if value:
            items.append(
                {
                    "stage": "gift_intent_reasoning",
                    "field": "preferences",
                    "value": value,
                    "traceable": _contains(upstream_text, value),
                }
            )
    traceable = sum(1 for item in items if item["traceable"])
    score = None if not items else traceable / len(items)
    return MetricResult("provenance_traceability", score, None if score is None else score == 1.0, {"items": items})


def _load_schema(stage: str) -> dict[str, Any] | None:
    path = CONFIG_DIR / f"{stage}.json"
    if not path.exists():
        return LOCAL_STAGE_SCHEMAS.get(stage)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("output_schema")
    if isinstance(schema, dict):
        return schema
    return LOCAL_STAGE_SCHEMAS.get(stage)


def _core_output(output: Any) -> tuple[Any, list[str]]:
    if not isinstance(output, Mapping):
        return output, []
    ignored = sorted(key for key in output if key in OBSERVABILITY_KEYS)
    return {key: value for key, value in output.items() if key not in OBSERVABILITY_KEYS}, ignored


def _extract_constraints(agent_outputs: Mapping[str, Any], input_context: Mapping[str, Any] | None) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    for source in (input_context or {}, _as_mapping(_get(agent_outputs, "gift_intent_reasoning", "constraints"))):
        for key in ("budget_hint", "budget", "exclusions", "allergies", "format_requirements"):
            if key in source and key not in constraints:
                constraints[key] = source[key]
    return constraints


def _stage_order_errors(sequence: Sequence[str]) -> list[str]:
    positions = {stage: index for index, stage in enumerate(sequence)}
    errors: list[str] = []
    for left_index, before in enumerate(DEFAULT_STAGE_ORDER):
        for after in DEFAULT_STAGE_ORDER[left_index + 1 :]:
            if before in positions and after in positions and positions[before] > positions[after]:
                errors.append(f"stage order violation: {before} after {after}")
    return errors


def _has_cycle(sequence: Sequence[str], dependencies: Sequence[Mapping[str, Any]]) -> bool:
    graph = {stage: set() for stage in sequence}
    for dependency in dependencies:
        after = str(dependency.get("after") or "")
        before = str(dependency.get("before") or "")
        if after in graph and before in graph:
            graph[after].add(before)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in graph.get(node, ())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(stage) for stage in graph)


def _upstream_text(agent_outputs: Mapping[str, Any], input_context: Mapping[str, Any] | None) -> str:
    upstream = {
        "input_context": input_context or {},
        "recipient_profiling": agent_outputs.get("recipient_profiling", {}),
        "relationship_analysis": agent_outputs.get("relationship_analysis", {}),
        "gift_intent_reasoning": agent_outputs.get("gift_intent_reasoning", {}),
        "multi_agent_planning": agent_outputs.get("multi_agent_planning", {}),
    }
    return json.dumps(upstream, ensure_ascii=False).lower()


def _contains(haystack: str, needle: str) -> bool:
    normalized = str(needle).strip().lower()
    if not normalized:
        return False
    return normalized in haystack


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _get(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _summarize_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_metric: dict[str, list[float]] = {}
    pass_counts: dict[str, dict[str, int]] = {}
    for report in reports:
        for metric in report.get("metrics", []):
            name = str(metric["name"])
            if metric.get("score") is not None:
                by_metric.setdefault(name, []).append(float(metric["score"]))
            if metric.get("passed") is not None:
                counts = pass_counts.setdefault(name, {"passed": 0, "failed": 0})
                counts["passed" if metric["passed"] else "failed"] += 1
    return {
        name: {
            "mean_score": sum(scores) / len(scores),
            **pass_counts.get(name, {}),
        }
        for name, scores in sorted(by_metric.items())
    }
