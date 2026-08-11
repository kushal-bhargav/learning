from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agents.gift_intent_reasoning import GiftIntentReasoningAgent
from src.agents.multi_agent_planning import DEFAULT_AGENT_SEQUENCE, MultiAgentPlanningAgent
from src.agents.orchestrator import GiftSession
from src.api import AgencyConsoleService


@dataclass(frozen=True)
class MethodVariant:
    name: str
    method: str
    level: str = "component"


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    custom_profile: Mapping[str, Any]
    expected_intent: Mapping[str, Any] | None = None
    expected_plan: Mapping[str, Any] | None = None


class _EvalCreativeAgent:
    stage = "creative_generation"

    def run(self, agent_input: Mapping[str, Any]) -> dict[str, Any]:
        config = agent_input["stage_config"]
        return {
            "stage": self.stage,
            "output": {
                "artifact_path": "experiments/eval/generated/eval-placeholder.png",
                "artifact_type": "generated",
                "media_type": "image/png",
                "width": 16,
                "height": 16,
                "agency_slider": float(config.get("agency_slider", 0.5)),
                "seed": int(config.get("seed", 2026)),
            },
            "confidence": None,
            "rationale": "Evaluation placeholder avoids image generation dependency for overall workflow metrics.",
        }


def evaluate_intent_output(output: Mapping[str, Any], expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = {"intent_summary", "occasion", "goal", "constraints", "preferences", "open_questions", "clarifying_needs"}
    validity = required.issubset(output.keys())
    expected = expected or {}
    occasion_expected = expected.get("occasion_name")
    occasion_actual = str((output.get("occasion") or {}).get("name", "")).lower()
    occasion_match = None if not occasion_expected else str(occasion_expected).lower() in occasion_actual
    expected_constraints = set(expected.get("constraints", []))
    actual_constraints = set((output.get("constraints") or {}).keys()) | set(output.get("clarifying_needs", []))
    precision, recall, f1 = _prf(actual_constraints, expected_constraints)
    confidence = output.get("confidence")
    return {
        "structured_output_valid": validity,
        "occasion_exact_or_contains": occasion_match,
        "constraint_precision": precision,
        "constraint_recall": recall,
        "constraint_f1": f1,
        "clarification_count": len(output.get("clarifying_needs", [])),
        "confidence": confidence,
    }


def evaluate_plan_output(output: Mapping[str, Any], expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = {"task_goal", "subtasks", "agent_sequence", "dependencies", "expected_outputs", "stop_conditions", "fallback_plan"}
    validity = required.issubset(output.keys())
    sequence = [str(item) for item in output.get("agent_sequence", [])]
    expected_sequence = list((expected or {}).get("agent_sequence", DEFAULT_AGENT_SEQUENCE))
    ordering = _ordered_subsequence(expected_sequence, sequence)
    dependency_pairs = {(item.get("after"), item.get("before")) for item in output.get("dependencies", []) if isinstance(item, Mapping)}
    dependency_satisfaction = all((sequence[index - 1], sequence[index]) in dependency_pairs for index in range(1, len(sequence))) if len(sequence) > 1 else True
    coverage = len(set(sequence) & set(expected_sequence)) / max(1, len(set(expected_sequence)))
    return {
        "structured_output_valid": validity,
        "plan_completeness": coverage,
        "step_ordering_correct": ordering,
        "dependency_satisfaction": dependency_satisfaction,
        "executable_plan_rate": bool(validity and sequence and dependency_satisfaction),
        "fallback_present": bool(output.get("fallback_plan")),
        "subtask_count": len(output.get("subtasks", [])),
    }


def compare_intent_planning_methods(
    cases: Sequence[ExperimentCase],
    *,
    intent_variants: Sequence[MethodVariant],
    planning_variants: Sequence[MethodVariant],
    output_dir: str | Path = "experiments/intent_planning_eval",
    run_overall: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    component_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for case in cases:
        fixture = _fixture_from_profile(case.custom_profile)
        session = GiftSession(session_id=f"eval-{case.case_id}", giver_id=fixture["people"][0]["id"], recipient_id=fixture["people"][1]["id"], occasion_id=fixture["occasions"][0]["id"])
        base_config = {
            "session": session,
            "recipient_profile": {},
            "relationship_guidance": {},
            "relationship": fixture["relationships"][0],
            "occasion": fixture["occasions"][0],
            "memories": fixture.get("memories", []),
            "preferences": fixture.get("preferences", []),
        }
        for intent_variant in intent_variants:
            started = time.perf_counter()
            intent_result = GiftIntentReasoningAgent().run({"session": session, "stage_config": {**base_config, "method": intent_variant.method}})
            intent_latency = time.perf_counter() - started
            intent_metrics = evaluate_intent_output(intent_result["output"], case.expected_intent)
            component_rows.append({
                "case_id": case.case_id,
                "component": "intent",
                "method_variant": intent_variant.name,
                "latency_seconds": intent_latency,
                **intent_metrics,
            })
            for planning_variant in planning_variants:
                started = time.perf_counter()
                plan_result = MultiAgentPlanningAgent().run({
                    "session": session,
                    "stage_config": {
                        "method": planning_variant.method,
                        "user_request": f"Create a gift for {case.custom_profile.get('recipient_name', 'recipient')}",
                        "intent": intent_result["output"],
                        "recipient_profile": {},
                        "relationship_guidance": {},
                        "memory_signals": {"memory_count": len(fixture.get("memories", [])), "preference_count": len(fixture.get("preferences", []))},
                    },
                })
                plan_latency = time.perf_counter() - started
                plan_metrics = evaluate_plan_output(plan_result["output"], case.expected_plan)
                component_rows.append({
                    "case_id": case.case_id,
                    "component": "planning",
                    "method_variant": planning_variant.name,
                    "paired_intent_variant": intent_variant.name,
                    "latency_seconds": plan_latency,
                    **plan_metrics,
                })
                if run_overall:
                    overall_rows.append(_run_overall_case(case, intent_variant, planning_variant))
    summary = {
        "metadata": {
            "research_notes": [
                "Intent recognition is evaluated separately from plan execution to avoid hiding extraction errors in downstream success.",
                "Planning is evaluated as bounded hybrid decomposition because standalone LLM planning is brittle without external orchestration.",
                "Overall metrics include coordination and fallback signals, not only local agent validity.",
            ]
        },
        "component_rows": component_rows,
        "overall_rows": overall_rows,
        "component_summary": _aggregate(component_rows),
        "overall_summary": _aggregate(overall_rows),
    }
    (output_path / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output_path / "component_metrics.csv", component_rows)
    _write_csv(output_path / "overall_metrics.csv", overall_rows)
    return summary


def _run_overall_case(case: ExperimentCase, intent_variant: MethodVariant, planning_variant: MethodVariant) -> dict[str, Any]:
    service = AgencyConsoleService(generated_dir="experiments/eval/generated", bandit_log_path="experiments/eval/bandit_log.jsonl", bandit_state_path="experiments/eval/bandit_state.json")
    service._creative_agent = _EvalCreativeAgent()  # type: ignore[assignment]
    session = service.create_session(persona_id="custom-live", custom_profile=case.custom_profile, agency_slider=0.5)
    started = time.perf_counter()
    fallback_count = 0
    failure_reason = None
    try:
        for stage in service.STAGES if hasattr(service, "STAGES") else DEFAULT_AGENT_SEQUENCE:
            overrides = {}
            if stage == "gift_intent_reasoning":
                overrides["method"] = intent_variant.method
            if stage == "multi_agent_planning":
                overrides["method"] = planning_variant.method
            service.propose(session.session_id, stage, overrides)
            service.accept(session.session_id)
    except Exception as exc:
        failure_reason = str(exc)
    ledger = service.ledger(session.session_id)
    elapsed = time.perf_counter() - started
    counts = ledger["counts"]
    return {
        "case_id": case.case_id,
        "intent_method": intent_variant.name,
        "planning_method": planning_variant.name,
        "end_to_end_success": failure_reason is None and ledger["completed"],
        "plan_execution_success": failure_reason is None,
        "acceptance_rate": counts.get("accept", 0) / max(1, ledger["stage_count"]),
        "edit_rate": counts.get("edit", 0) / max(1, ledger["stage_count"]),
        "regeneration_rate": counts.get("regenerate", 0) / max(1, ledger["stage_count"]),
        "ledger_completeness": len(ledger["timeline"]) >= ledger["stage_count"],
        "fallback_count": fallback_count,
        "latency_seconds": elapsed,
        "failure_reason": failure_reason,
    }


def _fixture_from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    service = AgencyConsoleService()
    return service._custom_fixture(profile)


def _prf(actual: set[Any], expected: set[Any]) -> tuple[float | None, float | None, float | None]:
    if not expected:
        return None, None, None
    tp = len(actual & expected)
    precision = tp / max(1, len(actual))
    recall = tp / max(1, len(expected))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _ordered_subsequence(expected: Sequence[str], actual: Sequence[str]) -> bool:
    positions = {value: index for index, value in enumerate(actual)}
    present = [positions[value] for value in expected if value in positions]
    return present == sorted(present)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = str(row.get("component") or f"{row.get('intent_method')}+{row.get('planning_method')}" or row.get("method_variant"))
        grouped.setdefault(key, []).append(row)
    summary = {}
    for key, items in grouped.items():
        numeric: dict[str, list[float]] = {}
        for item in items:
            for metric, value in item.items():
                if isinstance(value, bool):
                    numeric.setdefault(metric, []).append(float(value))
                elif isinstance(value, (int, float)):
                    numeric.setdefault(metric, []).append(float(value))
        summary[key] = {metric: sum(values) / len(values) for metric, values in numeric.items() if values}
        summary[key]["n"] = len(items)
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
