from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

import numpy as np

from src.agents import (
    CreativeGenerationAgent,
    DeliveryPlannerAgent,
    GiftIntentReasoningAgent,
    GreetingStoryAgent,
    MultiAgentPlanningAgent,
    RecipientProfilingAgent,
    RecommendationAgent,
    RelationshipAnalysisAgent,
)
from src.agents.orchestrator import GiftSession
from src.api.service import STAGES, AgencyConsoleService
from src.evaluation.intent_planning import ExperimentCase, evaluate_intent_output, evaluate_plan_output


def load_eval_cases(path: str | Path) -> list[ExperimentCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ExperimentCase(**case) for case in payload["cases"]]


def run_agent_quality_eval(
    cases: Sequence[ExperimentCase],
    *,
    output_dir: str | Path = "experiments/agent_quality_eval",
    include_creative: bool = False,
    creative_backend: str = "diffusers",
    agency_slider: float = 0.5,
    limit: int | None = None,
    require_ollama: bool = True,
) -> dict[str, Any]:
    if require_ollama:
        _assert_ollama_ready()
    os.environ["GMGI_USE_DEMO_AGENT_RESPONSES"] = "0"
    os.environ["GMGI_ALLOW_AGENT_FALLBACK"] = "0"
    os.environ["GMGI_FORCE_OLLAMA_AGENTS"] = "1"
    os.environ.setdefault("GMGI_INTENT_METHOD", "classifier_hybrid")
    os.environ.setdefault("GMGI_PLANNING_METHOD", "rule_constrained")
    os.environ.setdefault("GMGI_ALLOW_INTENT_REPAIR", "1")
    os.environ.setdefault("GMGI_ALLOW_PLANNING_REPAIR", "1")
    os.environ["GMGI_CREATIVE_BACKEND"] = creative_backend
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected_cases = list(cases[:limit] if limit else cases)
    for case in selected_cases:
        try:
            rows.extend(_run_case(case, output_path, include_creative=include_creative, creative_backend=creative_backend, agency_slider=agency_slider))
        except Exception as exc:
            rows.append(_error_row(case.case_id, "case_pipeline", exc))
    summary = {"rows": rows, "summary": _summarize(rows)}
    (output_path / "agent_quality.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_csv(output_path / "agent_quality.csv", rows)
    return summary


def _run_case(case: ExperimentCase, output_dir: Path, *, include_creative: bool, creative_backend: str, agency_slider: float) -> list[dict[str, Any]]:
    service = AgencyConsoleService(generated_dir=output_dir / "generated")
    fixture = service._custom_fixture(case.custom_profile)
    giver = next(person for person in fixture["people"] if person["role"] == "giver")
    recipient = next(person for person in fixture["people"] if person["role"] == "recipient")
    relationship = fixture["relationships"][0]
    occasion = fixture["occasions"][0]
    memories = fixture.get("memories", [])
    preferences = fixture.get("preferences", [])
    session = GiftSession(session_id=f"agent-eval-{case.case_id}", giver_id=giver["id"], recipient_id=recipient["id"], occasion_id=occasion["id"])
    rows: list[dict[str, Any]] = []
    outputs: dict[str, Mapping[str, Any]] = {}

    recipient_result = _timed_run(
        RecipientProfilingAgent(),
        session,
        {"person": recipient, "preferences": preferences, "raw_notes": [memory["content"] for memory in memories]},
    )
    outputs["recipient_profiling"] = recipient_result["output"]
    rows.append(_row(case.case_id, "recipient_profiling", recipient_result, _recipient_metrics(recipient_result["output"], preferences)))

    relationship_result = _timed_run(
        RelationshipAnalysisAgent(),
        session,
        {"relationship": relationship, "memories": memories, "occasion": occasion, "recipient_id": recipient["id"], "occasion_id": occasion["id"]},
    )
    outputs["relationship_analysis"] = relationship_result["output"]
    rows.append(_row(case.case_id, "relationship_analysis", relationship_result, _relationship_metrics(relationship_result["output"], relationship)))

    intent_result = _timed_run(
        GiftIntentReasoningAgent(),
        session,
        {
            "method": "llm_structured",
            "recipient_profile": outputs["recipient_profiling"],
            "relationship_guidance": outputs["relationship_analysis"],
            "relationship": relationship,
            "occasion": occasion,
            "memories": memories,
            "preferences": preferences,
            "budget_hint": occasion.get("budget_hint"),
            "method": os.getenv("GMGI_INTENT_METHOD", "classifier_hybrid"),
        },
    )
    outputs["gift_intent_reasoning"] = intent_result["output"]
    rows.append(_row(case.case_id, "gift_intent_reasoning", intent_result, evaluate_intent_output(intent_result["output"], case.expected_intent)))

    planning_result = _timed_run(
        MultiAgentPlanningAgent(),
        session,
        {
            "method": "llm_structured",
            "user_request": f"Create a gift for {recipient.get('display_name', 'recipient')}",
            "recipient_profile": outputs["recipient_profiling"],
            "relationship_guidance": outputs["relationship_analysis"],
            "intent": outputs["gift_intent_reasoning"],
            "memory_signals": {"memory_count": len(memories), "preference_count": len(preferences)},
            "available_agents": list(STAGES),
            "method": os.getenv("GMGI_PLANNING_METHOD", "rule_constrained"),
        },
    )
    outputs["multi_agent_planning"] = planning_result["output"]
    rows.append(_row(case.case_id, "multi_agent_planning", planning_result, evaluate_plan_output(planning_result["output"], case.expected_plan)))

    recommendation_result = _timed_run(
        RecommendationAgent(),
        session,
        {
            "recipient_profile": outputs["recipient_profiling"],
            "relationship_guidance": outputs["relationship_analysis"],
            "gift_intent": outputs["gift_intent_reasoning"],
            "execution_plan": outputs["multi_agent_planning"],
            "occasion": occasion,
            "preferences": preferences,
        },
    )
    outputs["recommendation"] = recommendation_result["output"]
    rows.append(_row(case.case_id, "recommendation", recommendation_result, _recommendation_metrics(recommendation_result["output"])))

    if include_creative:
        creative_config = _creative_config(service, fixture, recipient, occasion, relationship, outputs["gift_intent_reasoning"], output_dir, creative_backend, agency_slider)
        creative_result = _timed_run(CreativeGenerationAgent(), session, creative_config)
        outputs["creative_generation"] = creative_result["output"]
        rows.append(_row(case.case_id, "creative_generation", creative_result, _creative_metrics(creative_result["output"])))
    else:
        outputs["creative_generation"] = {"artifact_type": "generated", "gift_artifact_type": "not_run"}
        rows.append(_skipped_row(case.case_id, "creative_generation", "Set --include-creative or RUN_CREATIVE_EVAL=1 to run real Diffusers image generation."))

    greeting_result = _timed_run(
        GreetingStoryAgent(),
        session,
        {
            "relationship_guidance": outputs["relationship_analysis"],
            "occasion": occasion,
            "memories": memories,
            "tone_guidance": outputs["relationship_analysis"].get("tone_guidance"),
            "giver_name": giver.get("display_name"),
            "recipient_name": recipient.get("display_name"),
        },
    )
    outputs["greeting_story"] = greeting_result["output"]
    rows.append(_row(case.case_id, "greeting_story", greeting_result, _greeting_metrics(greeting_result["output"], memories)))

    delivery_result = _timed_run(
        DeliveryPlannerAgent(),
        session,
        {"artifact_type": outputs["creative_generation"].get("artifact_type", "generated"), "occasion": occasion},
    )
    rows.append(_row(case.case_id, "delivery_planner", delivery_result, _delivery_metrics(delivery_result["output"], occasion)))
    return rows


def _timed_run(agent: Any, session: GiftSession, stage_config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = agent.run({"session": session, "stage_config": stage_config})
    elapsed = time.perf_counter() - started
    return {**result, "latency_seconds": elapsed}


def _row(case_id: str, stage: str, result: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "stage": stage,
        "status": "ok",
        "latency_seconds": result.get("latency_seconds"),
        "confidence": result.get("confidence"),
        "rationale_present": bool(result.get("rationale")),
        "prompt_version": (result.get("output") or {}).get("prompt_version", "static"),
        "skills_used": ",".join((result.get("output") or {}).get("skills_used", [])),
        "quality_score": _quality_score(metrics),
        **{key: _metric_value(value) for key, value in metrics.items()},
    }



def _error_row(case_id: str, stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "stage": stage,
        "status": "error",
        "quality_score": 0.0,
        "error_type": type(exc).__name__,
        "error": str(exc)[:2000],
    }

def _skipped_row(case_id: str, stage: str, reason: str) -> dict[str, Any]:
    return {"case_id": case_id, "stage": stage, "status": "skipped", "skip_reason": reason, "quality_score": None}


def _recipient_metrics(output: Mapping[str, Any], preferences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    interests = output.get("interests", []) if isinstance(output.get("interests"), list) else []
    expected = {str(pref.get("value", "")).lower() for pref in preferences if pref.get("value")}
    actual = {str(item.get("name", "")).lower() for item in interests if isinstance(item, Mapping)}
    return {
        "structured_output_valid": all(key in output for key in ("interests", "communication_style", "gift_history_summary")),
        "interest_count": len(interests),
        "preference_recall": None if not expected else len(expected & actual) / len(expected),
        "communication_style_present": bool(output.get("communication_style")),
    }


def _relationship_metrics(output: Mapping[str, Any], relationship: Mapping[str, Any]) -> dict[str, Any]:
    expected = _closeness_bucket(float(relationship.get("closeness_score", 3)))
    return {
        "structured_output_valid": all(key in output for key in ("closeness_assessment", "tone_guidance", "formality", "risk_flags", "agency_slider_default")),
        "closeness_matches_bucket": output.get("closeness_assessment") == expected,
        "slider_in_range": _between(output.get("agency_slider_default"), 0, 1),
        "risk_flags_list": isinstance(output.get("risk_flags"), list),
    }


def _recommendation_metrics(output: Mapping[str, Any]) -> dict[str, Any]:
    recs = output.get("recommendations", []) if isinstance(output.get("recommendations"), list) else []
    valid_types = {"physical", "generated", "bundle"}
    return {
        "structured_output_valid": len(recs) == 3,
        "three_recommendations": len(recs) == 3,
        "rank_order_valid": [rec.get("rank") for rec in recs if isinstance(rec, Mapping)] == [1, 2, 3],
        "all_have_evidence": all(bool(rec.get("evidence")) for rec in recs if isinstance(rec, Mapping)),
        "all_have_budget_fit": all(bool(rec.get("budget_fit")) for rec in recs if isinstance(rec, Mapping)),
        "artifact_types_valid": all(rec.get("artifact_type") in valid_types for rec in recs if isinstance(rec, Mapping)),
    }


def _creative_metrics(output: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(output.get("artifact_path", "")))
    return {
        "artifact_exists": path.exists(),
        "image_dimensions_valid": int(output.get("width", 0) or 0) > 0 and int(output.get("height", 0) or 0) > 0,
        "clip_score": output.get("clip_score"),
        "prompt_present": bool(output.get("diffusers_prompt")),
        "artifact_type_present": bool(output.get("gift_artifact_type")),
    }


def _greeting_metrics(output: Mapping[str, Any], memories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    memory_ids = {str(memory.get("id")) for memory in memories if memory.get("id")}
    refs = output.get("memory_references", []) if isinstance(output.get("memory_references"), list) else []
    return {
        "structured_output_valid": all(key in output for key in ("message", "memory_references", "tone")),
        "message_nonempty": bool(str(output.get("message", "")).strip()),
        "tone_present": bool(output.get("tone")),
        "memory_refs_grounded": all(str(ref) in memory_ids for ref in refs) if refs else True,
    }


def _delivery_metrics(output: Mapping[str, Any], occasion: Mapping[str, Any]) -> dict[str, Any]:
    planned = output.get("planned_send_date")
    occasion_date = occasion.get("date")
    date_order_ok = False
    try:
        date_order_ok = date.fromisoformat(str(planned)) <= date.fromisoformat(str(occasion_date))
    except Exception:
        pass
    return {
        "structured_output_valid": all(key in output for key in ("mode", "channel", "planned_send_date", "occasion_date", "status", "disclaimer")),
        "simulated_only": output.get("status") == "simulated" and "No shipment" in str(output.get("disclaimer", "")),
        "planned_before_or_on_occasion": date_order_ok,
        "channel_present": bool(output.get("channel")),
    }


def _creative_config(service: AgencyConsoleService, fixture: Mapping[str, Any], recipient: Mapping[str, Any], occasion: Mapping[str, Any], relationship: Mapping[str, Any], intent: Mapping[str, Any], output_dir: Path, backend: str, agency_slider: float) -> dict[str, Any]:
    memory = (fixture.get("memories") or [{}])[0]
    context = service._context_embedding(type("Console", (), {"fixture": fixture, "fixture_path": None})(), recipient["id"], occasion["id"])
    visual = intent.get("visual_generation", {}) if isinstance(intent, Mapping) else {}
    goal = intent.get("goal", {}) if isinstance(intent, Mapping) else {}
    artifact_type = str(visual.get("artifact_type") or goal.get("recommended_artifact_type") or "greeting_card")
    return {
        "context_embedding": np.asarray(context, dtype=np.float32),
        "relationship_type": relationship.get("type", "other"),
        "emotion_tag": memory.get("emotion_tag", "joy"),
        "occasion": service._gan_occasion(occasion.get("name", "other")),
        "agency_slider": agency_slider,
        "human_style_ref": np.asarray(memory.get("embedding", context), dtype=np.float32),
        "seed": 2026,
        "gift_artifact_type": artifact_type,
        "visual_style_prompt": visual.get("style_prompt"),
        "output_dir": (output_dir / "generated").as_posix(),
        "filename": f"agent-quality-{artifact_type}.png",
        "generation_backend": backend,
        "diffusers_prompt": service._diffusers_prompt(relationship, occasion, memory, artifact_type, visual, agency_slider),
    }


def _closeness_bucket(value: float) -> str:
    if value >= 4.5:
        return "very close"
    if value >= 3.5:
        return "close"
    if value >= 2.0:
        return "moderate"
    return "low"


def _between(value: Any, low: float, high: float) -> bool:
    try:
        number = float(value)
    except Exception:
        return False
    return low <= number <= high


def _quality_score(metrics: Mapping[str, Any]) -> float | None:
    values = []
    for value in metrics.values():
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if 0.0 <= number <= 1.0:
                values.append(number)
    return None if not values else sum(values) / len(values)


def _metric_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stage in sorted({str(row["stage"]) for row in rows}):
        stage_rows = [row for row in rows if row["stage"] == stage]
        scores = [float(row["quality_score"]) for row in stage_rows if row.get("quality_score") is not None]
        summary[stage] = {
            "n": len(stage_rows),
            "ok": sum(row.get("status") == "ok" for row in stage_rows),
            "skipped": sum(row.get("status") == "skipped" for row in stage_rows),
            "mean_quality_score": None if not scores else sum(scores) / len(scores),
        }
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


def _assert_ollama_ready() -> None:
    host = os.getenv("GMGI_OLLAMA_HOST") or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = host.rstrip("/") + "/api/tags"
    try:
        with urlopen(url, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"status {response.status}")
    except Exception as exc:
        raise RuntimeError(f"Ollama is not reachable at {url}. Start Ollama before real agent evaluation.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real per-agent GMGI quality evals.")
    parser.add_argument("--config", default="eval/configs/intent_planning_eval.json")
    parser.add_argument("--output-dir", default="experiments/agent_quality_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-creative", action="store_true", help="Run real Diffusers/MemoryGAN creative generation instead of marking it skipped.")
    parser.add_argument("--creative-backend", default=os.getenv("GMGI_CREATIVE_BACKEND", "diffusers"))
    parser.add_argument("--agency-slider", type=float, default=0.5)
    parser.add_argument("--skip-ollama-check", action="store_true")
    args = parser.parse_args()
    summary = run_agent_quality_eval(
        load_eval_cases(args.config),
        output_dir=args.output_dir,
        include_creative=args.include_creative or os.getenv("RUN_CREATIVE_EVAL", "0") == "1",
        creative_backend=args.creative_backend,
        agency_slider=args.agency_slider,
        limit=args.limit,
        require_ollama=not args.skip_ollama_check,
    )
    print(json.dumps(summary["summary"], indent=2, default=str))
    print(f"Wrote {Path(args.output_dir) / 'agent_quality.json'}")


if __name__ == "__main__":
    main()
