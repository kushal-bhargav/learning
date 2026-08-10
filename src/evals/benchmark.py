from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from src.agents import (
    CreativeGenerationAgent,
    DeliveryPlannerAgent,
    GiftIntentReasoningAgent,
    GiftSession,
    GreetingStoryAgent,
    MultiAgentPlanningAgent,
    RecipientProfilingAgent,
    RecommendationAgent,
    RelationshipAnalysisAgent,
)
from src.agents.orchestrator import AgentOutput

from .quality import evaluate_outputs, summarize_quality_reports
from .structural import DEFAULT_STAGE_ORDER


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    custom_profile: Mapping[str, Any]
    expected: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(payload["case_id"]),
            custom_profile=dict(payload.get("custom_profile") or {}),
            expected=dict(payload.get("expected") or {}),
        )


def load_cases(path: str | Path | None = None) -> list[BenchmarkCase]:
    source = Path(path) if path else Path(__file__).with_name("benchmark_cases.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [BenchmarkCase.from_mapping(item) for item in payload.get("cases", [])]


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    output_dir: str | Path = "experiments/evals/benchmark",
    include_creative: bool = False,
    agency_slider: float = 0.5,
    seed: int = 2026,
    limit: int | None = None,
    stage_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected = list(cases[:limit] if limit else cases)
    case_reports = [
        run_case(
            case,
            output_dir=output_path,
            include_creative=include_creative,
            agency_slider=agency_slider,
            seed=seed,
            stage_timeout_seconds=stage_timeout_seconds,
        )
        for case in selected
    ]
    summary = {
        "phase": "benchmark",
        "case_count": len(case_reports),
        "include_creative": include_creative,
        "summary": summarize_quality_reports(case_reports),
        "cases": case_reports,
    }
    (output_path / "benchmark_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_rows(output_path / "benchmark_rows.csv", _flatten_rows(case_reports))
    return summary


def run_case(
    case: BenchmarkCase,
    *,
    output_dir: str | Path,
    include_creative: bool,
    agency_slider: float,
    seed: int,
    stage_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    fixture = _fixture_from_profile(case.custom_profile)
    giver = _person(fixture, "giver")
    recipient = _person(fixture, "recipient")
    relationship = fixture["relationships"][0]
    occasion = fixture["occasions"][0]
    memories = fixture.get("memories", [])
    preferences = fixture.get("preferences", [])
    session = GiftSession(
        session_id=f"benchmark-{case.case_id}",
        giver_id=giver["id"],
        recipient_id=recipient["id"],
        occasion_id=occasion["id"],
    )
    outputs: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    case_agency_slider = float(case.custom_profile.get("agency_slider", agency_slider))

    stage_configs: dict[str, dict[str, Any]] = {
        "recipient_profiling": {
            "person": recipient,
            "preferences": preferences,
            "raw_notes": [memory["content"] for memory in memories],
        },
        "relationship_analysis": {
            "relationship": relationship,
            "memories": memories,
            "occasion": occasion,
            "recipient_id": recipient["id"],
            "occasion_id": occasion["id"],
        },
    }
    timeout = stage_timeout_seconds if stage_timeout_seconds is not None else float(os.getenv("GMGI_EVAL_STAGE_TIMEOUT_SECONDS", "60"))
    _run_stage("recipient_profiling", RecipientProfilingAgent(), session, stage_configs["recipient_profiling"], outputs, traces, timeout)
    _run_stage("relationship_analysis", RelationshipAnalysisAgent(), session, stage_configs["relationship_analysis"], outputs, traces, timeout)

    stage_configs["gift_intent_reasoning"] = {
        "recipient_profile": outputs.get("recipient_profiling", {}),
        "relationship_guidance": outputs.get("relationship_analysis", {}),
        "relationship": relationship,
        "occasion": occasion,
        "memories": memories,
        "preferences": preferences,
        "budget_hint": occasion.get("budget_hint"),
    }
    _run_stage("gift_intent_reasoning", GiftIntentReasoningAgent(), session, stage_configs["gift_intent_reasoning"], outputs, traces, timeout)

    stage_configs["multi_agent_planning"] = {
        "user_request": f"Create a gift for {recipient.get('display_name', 'recipient')}",
        "recipient_profile": outputs.get("recipient_profiling", {}),
        "relationship_guidance": outputs.get("relationship_analysis", {}),
        "intent": outputs.get("gift_intent_reasoning", {}),
        "memory_signals": {"memory_count": len(memories), "preference_count": len(preferences)},
        "available_agents": list(DEFAULT_STAGE_ORDER),
    }
    _run_stage("multi_agent_planning", MultiAgentPlanningAgent(), session, stage_configs["multi_agent_planning"], outputs, traces, timeout)

    stage_configs["recommendation"] = {
        "recipient_profile": outputs.get("recipient_profiling", {}),
        "relationship_guidance": outputs.get("relationship_analysis", {}),
        "gift_intent": outputs.get("gift_intent_reasoning", {}),
        "execution_plan": outputs.get("multi_agent_planning", {}),
        "occasion": occasion,
        "preferences": preferences,
    }
    _run_stage("recommendation", RecommendationAgent(), session, stage_configs["recommendation"], outputs, traces, timeout)

    if include_creative:
        stage_configs["creative_generation"] = {
            "recipient_profile": outputs.get("recipient_profiling", {}),
            "relationship_guidance": outputs.get("relationship_analysis", {}),
            "gift_intent": outputs.get("gift_intent_reasoning", {}),
            "recommendation": outputs.get("recommendation", {}),
            "memories": memories,
            "preferences": preferences,
            "occasion": occasion,
            "agency_slider": case_agency_slider,
            "seed": seed,
            "output_dir": str(Path(output_dir) / "generated"),
        }
        _run_stage("creative_generation", CreativeGenerationAgent(), session, stage_configs["creative_generation"], outputs, traces, timeout)
    else:
        traces.append({"stage": "creative_generation", "status": "skipped", "latency_seconds": 0.0, "reason": "include_creative=False"})

    stage_configs["greeting_story"] = {
        "relationship_guidance": outputs.get("relationship_analysis", {}),
        "occasion": occasion,
        "memories": memories,
        "tone_guidance": outputs.get("relationship_analysis", {}).get("tone_guidance"),
        "giver_name": giver.get("display_name"),
        "recipient_name": recipient.get("display_name"),
    }
    _run_stage("greeting_story", GreetingStoryAgent(), session, stage_configs["greeting_story"], outputs, traces, timeout)

    stage_configs["delivery_planner"] = {
        "artifact_type": outputs.get("creative_generation", {}).get("artifact_type", "generated"),
        "occasion": occasion,
    }
    _run_stage("delivery_planner", DeliveryPlannerAgent(), session, stage_configs["delivery_planner"], outputs, traces, timeout)

    input_context = {
        "fixture": fixture,
        "recipient": recipient,
        "relationship": relationship,
        "occasion": occasion,
        "memories": memories,
        "preferences": preferences,
    }
    report = evaluate_outputs(outputs, expected=case.expected, input_context=input_context)
    report.update(
        {
            "case_id": case.case_id,
            "session_id": session.session_id,
            "agent_traces": traces,
            "input_context": input_context,
            "expected": dict(case.expected),
            "behavioral_metrics": {
                "composite_reward": None,
                "accept_rate": None,
                "edit_rate": None,
                "regenerate_rate": None,
                "delegate_rate": None,
                "completed_action_coverage": None,
            },
        }
    )
    return report


def _run_stage(
    stage: str,
    agent: Any,
    session: GiftSession,
    stage_config: Mapping[str, Any],
    outputs: dict[str, Any],
    traces: list[dict[str, Any]],
    timeout_seconds: float,
) -> None:
    started = time.perf_counter()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put(("ok", agent.run({"session": session, "stage_config": dict(stage_config)})))
        except Exception as exc:  # pragma: no cover - exercised through outer branch in tests
            result_queue.put(("error", exc))

    worker = threading.Thread(target=invoke, name=f"gmgi-eval-{stage}", daemon=True)
    worker.start()
    worker.join(timeout=max(0.1, timeout_seconds))
    if worker.is_alive():
        latency = time.perf_counter() - started
        message = f"stage exceeded {timeout_seconds:.1f}s timeout"
        outputs[stage] = {"error": message, "error_type": "TimeoutError"}
        traces.append({"stage": stage, "status": "timeout", "latency_seconds": latency, "error_type": "TimeoutError", "error": message})
        return

    status, value = result_queue.get()
    try:
        if status == "error":
            raise value
        result: AgentOutput = value
        latency = time.perf_counter() - started
        outputs[stage] = dict(result.get("output") or {})
        traces.append(
            {
                "stage": stage,
                "status": "ok",
                "latency_seconds": latency,
                "confidence": result.get("confidence"),
                "rationale_present": bool(result.get("rationale")),
            }
        )
    except Exception as exc:
        latency = time.perf_counter() - started
        outputs[stage] = {"error": str(exc), "error_type": type(exc).__name__}
        traces.append({"stage": stage, "status": "error", "latency_seconds": latency, "error_type": type(exc).__name__, "error": str(exc)[:2000]})


def _fixture_from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    suffix = uuid4().hex[:8]
    giver_id = f"person-giver-{suffix}"
    recipient_id = f"person-recipient-{suffix}"
    occasion_id = f"occasion-live-{suffix}"
    event_id = f"event-live-{suffix}"
    memories = [
        {
            "id": f"memory-live-{suffix}-{index + 1}",
            "modality": "text",
            "content": str(content),
            "timestamp": f"{profile.get('occasion_date', '2026-12-31')}T00:00:00Z",
            "event_id": event_id,
            "person_ids": [giver_id, recipient_id],
            "emotion_tag": _emotion_from_text(str(content)),
        }
        for index, content in enumerate(_list_field(profile.get("memories"))[:8])
    ]
    preferences = [
        {
            "id": f"preference-live-{suffix}-{index + 1}",
            "person_id": recipient_id,
            "category": "stated",
            "value": str(value),
            "confidence": 1.0,
            "source": "benchmark_case",
        }
        for index, value in enumerate(_list_field(profile.get("preferences"))[:12])
    ]
    return {
        "schema_version": "1.0",
        "persona_id": f"benchmark-{suffix}",
        "label": f"{profile.get('giver_name', 'Gift giver')} to {profile.get('recipient_name', 'Gift recipient')}",
        "synthetic": False,
        "people": [
            {"id": giver_id, "display_name": str(profile.get("giver_name") or "Gift giver"), "role": "giver"},
            {"id": recipient_id, "display_name": str(profile.get("recipient_name") or "Gift recipient"), "role": "recipient"},
        ],
        "relationships": [
            {
                "id": f"relationship-live-{suffix}",
                "person_a": giver_id,
                "person_b": recipient_id,
                "type": str(profile.get("relationship_type") or "other"),
                "closeness_score": max(1.0, min(5.0, float(profile.get("closeness_score", 3)))),
            }
        ],
        "occasions": [
            {
                "id": occasion_id,
                "name": str(profile.get("occasion_name") or "Gift occasion"),
                "date": str(profile.get("occasion_date") or "2026-12-31"),
                "budget_hint": str(profile.get("budget_hint") or "Flexible"),
                "formality": str(profile.get("formality") or "casual"),
            }
        ],
        "events": [{"id": event_id, "date": str(profile.get("occasion_date") or "2026-12-31"), "type": "benchmark-context", "participants": [giver_id, recipient_id]}],
        "memories": memories,
        "preferences": preferences,
    }


def _person(fixture: Mapping[str, Any], role: str) -> dict[str, Any]:
    return dict(next(person for person in fixture["people"] if person["role"] == role))


def _list_field(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.splitlines() if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _emotion_from_text(value: str) -> str:
    text = value.lower()
    if any(word in text for word in ("laughed", "jokes", "funny")):
        return "joy"
    if any(word in text for word in ("mentor", "helped", "grateful")):
        return "gratitude"
    if any(word in text for word in ("home", "kitchen", "chai")):
        return "warmth"
    return "nostalgia"


def _flatten_rows(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        case_id = str(report.get("case_id"))
        for stage, stage_report in dict(report.get("stage_reports") or {}).items():
            row = {"case_id": case_id, "stage": stage, "quality_score": stage_report.get("quality_score"), "status": stage_report.get("status")}
            for metric in stage_report.get("metrics", []):
                row[str(metric.get("name"))] = metric.get("score")
            rows.append(row)
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(json.dumps(row.get(field, ""), ensure_ascii=False) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GMGI benchmark-quality evals against real agent classes.")
    parser.add_argument("--case-file", default=None)
    parser.add_argument("--output-dir", default="experiments/evals/benchmark")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-creative", action="store_true")
    parser.add_argument("--agency-slider", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stage-timeout", type=float, default=None)
    args = parser.parse_args()
    report = run_benchmark(
        load_cases(args.case_file),
        output_dir=args.output_dir,
        include_creative=args.include_creative,
        agency_slider=args.agency_slider,
        seed=args.seed,
        limit=args.limit,
        stage_timeout_seconds=args.stage_timeout,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
