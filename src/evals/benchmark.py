from __future__ import annotations

import argparse
import hashlib
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
from src.harness import HarnessConfig, HarnessController, NextAction, TraceRecorder, default_harness_config, trace_recorder_context

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
    harness_config: HarnessConfig | None = None,
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
            harness_config=harness_config,
        )
        for case in selected
    ]
    summary = {
        "phase": "benchmark",
        "case_count": len(case_reports),
        "include_creative": include_creative,
        "summary": summarize_quality_reports(case_reports),
        "artifact_analysis": _artifact_analysis(case_reports),
        "cases": case_reports,
    }
    (output_path / "benchmark_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_rows(output_path / "benchmark_rows.csv", _flatten_rows(case_reports))
    return summary


def compare_harness_runs(
    case: BenchmarkCase,
    *,
    harness_a: HarnessConfig,
    harness_b: HarnessConfig,
    output_dir: str | Path = "experiments/evals/harness_comparison",
    include_creative: bool = False,
    agency_slider: float = 0.5,
    seed: int = 2026,
    stage_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    report_a = run_case(
        case,
        output_dir=root / "harness_a",
        include_creative=include_creative,
        agency_slider=agency_slider,
        seed=seed,
        stage_timeout_seconds=stage_timeout_seconds,
        harness_config=harness_a,
    )
    report_b = run_case(
        case,
        output_dir=root / "harness_b",
        include_creative=include_creative,
        agency_slider=agency_slider,
        seed=seed,
        stage_timeout_seconds=stage_timeout_seconds,
        harness_config=harness_b,
    )
    comparison = {
        "case_id": case.case_id,
        "harness_a": _harness_run_summary(report_a),
        "harness_b": _harness_run_summary(report_b),
        "trajectory_changed": _trajectory(report_a) != _trajectory(report_b),
        "verifier_outcomes_changed": _verifier_outcomes(report_a) != _verifier_outcomes(report_b),
        "quality_delta": float(report_b.get("overall_score", 0.0) or 0.0) - float(report_a.get("overall_score", 0.0) or 0.0),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{case.case_id}_harness_comparison.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    return comparison


def run_case(
    case: BenchmarkCase,
    *,
    output_dir: str | Path,
    include_creative: bool,
    agency_slider: float,
    seed: int,
    stage_timeout_seconds: float | None = None,
    harness_config: HarnessConfig | None = None,
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
    effective_harness = harness_config or default_harness_config()
    trace_recorder = TraceRecorder(session_id=session.session_id, case_id=case.case_id, harness_config=effective_harness)
    controller = HarnessController(effective_harness)
    outputs: dict[str, Any] = {}
    traces: list[dict[str, Any]] = []
    completed_stages: list[str] = []
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
    _run_expected_stage(controller, completed_stages, "recipient_profiling", RecipientProfilingAgent(), session, stage_configs["recipient_profiling"], outputs, traces, timeout, trace_recorder)
    _run_expected_stage(controller, completed_stages, "relationship_analysis", RelationshipAnalysisAgent(), session, stage_configs["relationship_analysis"], outputs, traces, timeout, trace_recorder)

    stage_configs["gift_intent_reasoning"] = {
        "recipient_profile": outputs.get("recipient_profiling", {}),
        "relationship_guidance": outputs.get("relationship_analysis", {}),
        "relationship": relationship,
        "occasion": occasion,
        "memories": memories,
        "preferences": preferences,
        "budget_hint": occasion.get("budget_hint"),
    }
    _run_expected_stage(controller, completed_stages, "gift_intent_reasoning", GiftIntentReasoningAgent(), session, stage_configs["gift_intent_reasoning"], outputs, traces, timeout, trace_recorder)

    stage_configs["multi_agent_planning"] = {
        "user_request": f"Create a gift for {recipient.get('display_name', 'recipient')}",
        "recipient_profile": outputs.get("recipient_profiling", {}),
        "relationship_guidance": outputs.get("relationship_analysis", {}),
        "intent": outputs.get("gift_intent_reasoning", {}),
        "memory_signals": {"memory_count": len(memories), "preference_count": len(preferences)},
        "available_agents": list(DEFAULT_STAGE_ORDER),
    }
    _run_expected_stage(controller, completed_stages, "multi_agent_planning", MultiAgentPlanningAgent(), session, stage_configs["multi_agent_planning"], outputs, traces, timeout, trace_recorder)

    stage_configs["recommendation"] = {
        "recipient_profile": outputs.get("recipient_profiling", {}),
        "relationship_guidance": outputs.get("relationship_analysis", {}),
        "gift_intent": outputs.get("gift_intent_reasoning", {}),
        "execution_plan": outputs.get("multi_agent_planning", {}),
        "occasion": occasion,
        "preferences": preferences,
    }
    _run_expected_stage(controller, completed_stages, "recommendation", RecommendationAgent(), session, stage_configs["recommendation"], outputs, traces, timeout, trace_recorder)

    if include_creative:
        first_memory = memories[0] if memories else {}
        visual_generation = outputs.get("gift_intent_reasoning", {}).get("visual_generation", {})
        style_prompt = str(visual_generation.get("style_prompt") or ", ".join(str(item.get("value")) for item in preferences[:4] if item.get("value")) or "personalized gift visual")
        artifact_type = str(visual_generation.get("artifact_type") or "greeting_card")
        context_embedding = _context_embedding(memories, preferences)
        human_style_ref = _context_embedding([first_memory] if first_memory else [], preferences)
        stage_configs["creative_generation"] = {
            "context_embedding": context_embedding,
            "relationship_type": relationship.get("type", "other"),
            "emotion_tag": first_memory.get("emotion_tag", "joy"),
            "occasion": _visual_occasion(occasion.get("name", "other")),
            "recipient_profile": outputs.get("recipient_profiling", {}),
            "relationship_guidance": outputs.get("relationship_analysis", {}),
            "gift_intent": outputs.get("gift_intent_reasoning", {}),
            "recommendation": outputs.get("recommendation", {}),
            "memories": memories,
            "preferences": preferences,
            "occasion_context": occasion,
            "agency_slider": case_agency_slider,
            "human_style_ref": human_style_ref,
            "seed": seed,
            "output_dir": str(Path(output_dir) / "generated"),
            "filename": f"{session.session_id}.png",
            "generation_backend": os.getenv("GMGI_CREATIVE_BACKEND", "diffusers"),
            "gift_artifact_type": artifact_type,
            "visual_style_prompt": style_prompt,
            "negative_prompt": CreativeGenerationAgent.default_negative_prompt(),
            "diffusers_prompt": _diffusers_prompt(relationship, occasion, first_memory, artifact_type, style_prompt, case_agency_slider),
        }
        _run_expected_stage(controller, completed_stages, "creative_generation", _creative_agent(), session, stage_configs["creative_generation"], outputs, traces, timeout, trace_recorder)
    else:
        traces.append({"stage": "creative_generation", "status": "skipped", "latency_seconds": 0.0, "reason": "include_creative=False"})
        if controller.next_action(completed_stages=completed_stages).stage == "creative_generation":
            completed_stages.append("creative_generation")

    stage_configs["greeting_story"] = {
        "relationship_guidance": outputs.get("relationship_analysis", {}),
        "occasion": occasion,
        "memories": memories,
        "tone_guidance": outputs.get("relationship_analysis", {}).get("tone_guidance"),
        "giver_name": giver.get("display_name"),
        "recipient_name": recipient.get("display_name"),
    }
    _run_expected_stage(controller, completed_stages, "greeting_story", GreetingStoryAgent(), session, stage_configs["greeting_story"], outputs, traces, timeout, trace_recorder)

    stage_configs["delivery_planner"] = {
        "artifact_type": outputs.get("creative_generation", {}).get("artifact_type", "generated"),
        "occasion": occasion,
    }
    _run_expected_stage(controller, completed_stages, "delivery_planner", DeliveryPlannerAgent(), session, stage_configs["delivery_planner"], outputs, traces, timeout, trace_recorder)

    input_context = {
        "fixture": fixture,
        "recipient": recipient,
        "relationship": relationship,
        "occasion": occasion,
        "memories": memories,
        "preferences": preferences,
    }
    report = evaluate_outputs(outputs, expected=case.expected, input_context=input_context)
    if not include_creative:
        _mark_creative_unavailable(report)
    has_timeout = any(trace.get("status") == "timeout" for trace in traces)
    has_error = any(trace.get("status") in {"error", "fallback"} for trace in traces)
    final_status = "partial" if has_timeout or has_error else "success"
    termination_reason = "agent_timeout" if has_timeout else "stage_error_or_fallback" if has_error else "completed"
    report.update(
        {
            "case_id": case.case_id,
            "session_id": session.session_id,
            "agent_traces": traces,
            "run_trace": trace_recorder.finish(final_status=final_status, termination_reason=termination_reason).to_dict(),
            "run_id": trace_recorder.run_id,
            "harness_id": trace_recorder.harness_config.harness_id,
            "harness_version": trace_recorder.harness_config.harness_version,
            "harness_config_hash": trace_recorder.harness_config.config_hash,
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


def _mark_creative_unavailable(report: dict[str, Any]) -> None:
    stage_reports = _as_mapping(report.get("stage_reports"))
    if "creative_generation" in stage_reports:
        stage_reports["creative_generation"] = {
            "quality_score": None,
            "status": "skipped",
            "metrics": [
                {
                    "name": "creative_generation_skipped",
                    "score": None,
                    "passed": None,
                    "weight": 0.0,
                    "details": {"reason": "include_creative=False"},
                }
            ],
        }
    cross_scores = [
        _as_mapping(metric).get("score")
        for metric in _as_list(report.get("cross_component_metrics"))
        if _as_mapping(metric).get("score") is not None
    ]
    stage_scores = [
        _as_mapping(item).get("quality_score")
        for item in stage_reports.values()
        if _as_mapping(item).get("quality_score") is not None
    ]
    scores = [float(score) for score in [*stage_scores, *cross_scores] if score is not None]
    report["overall_quality_score"] = None if not scores else sum(scores) / len(scores)


def _run_expected_stage(
    controller: HarnessController,
    completed_stages: list[str],
    stage: str,
    agent: Any,
    session: GiftSession,
    stage_config: Mapping[str, Any],
    outputs: dict[str, Any],
    traces: list[dict[str, Any]],
    timeout_seconds: float,
    trace_recorder: TraceRecorder,
) -> None:
    action = controller.next_action(
        completed_stages=completed_stages,
        dependency_ids=[trace_recorder.invocations[-1].invocation_id] if trace_recorder.invocations else [],
        agent_outputs=outputs,
        constraints={"budget_hint": _as_mapping(stage_config.get("occasion")).get("budget_hint") or stage_config.get("budget_hint")},
        memory_context={"memory_count": len(_as_list(stage_config.get("memories"))), "preference_count": len(_as_list(stage_config.get("preferences")))},
        run_id=trace_recorder.run_id,
        case_id=trace_recorder.run_trace.case_id,
    )
    if action.action_type == "stop":
        traces.append({"stage": stage, "status": "skipped", "latency_seconds": 0.0, "reason": action.reason})
        return
    if action.stage != stage:
        raise RuntimeError(f"HarnessController selected {action.stage!r}; benchmark expected {stage!r}")
    try:
        result = controller.execute_agent_action(
            action,
            lambda: _run_stage(stage, agent, session, stage_config, outputs, traces, timeout_seconds, trace_recorder, controller, action),
        )
    except Exception:
        completed_stages.append(stage)
        return
    if result.get("output", {}).get("fallback") == "skip_failed_stage":
        outputs[stage] = dict(result.get("output") or {})
        traces.append({"stage": stage, "status": "fallback", "latency_seconds": 0.0, "reason": "skip_failed_stage"})
    if stage == "multi_agent_planning" and controller.config.planner_mode == "authoritative":
        controller.adopt_execution_plan_from_output(result)
    completed_stages.append(stage)


def _run_stage(
    stage: str,
    agent: Any,
    session: GiftSession,
    stage_config: Mapping[str, Any],
    outputs: dict[str, Any],
    traces: list[dict[str, Any]],
    timeout_seconds: float,
    trace_recorder: TraceRecorder,
    controller: HarnessController,
    action: NextAction,
) -> AgentOutput:
    started = time.perf_counter()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            agent_input = {"session": session, "stage_config": dict(stage_config)}
            with trace_recorder_context(trace_recorder):
                with controller.tool_context(type(agent).__name__):
                    with trace_recorder.invocation(
                        stage_name=stage,
                        agent_name=type(agent).__name__,
                        agent_version=str(getattr(agent, "prompt_version_id", "static")),
                        raw_agent_input=agent_input,
                        context={"stage": stage, "stage_config": dict(stage_config), "input_diagnostics": _input_diagnostics(agent_input, stage_config)},
                        relevant_memory={"memories": stage_config.get("memories", [])},
                        constraints={key: stage_config[key] for key in ("budget_hint", "agency_slider", "generation_backend") if key in stage_config},
                        model_provider=_model_provider(agent),
                        model_name=_model_name(agent, stage_config),
                        model_parameters=_model_parameters(agent, stage_config),
                        routing_decision=action.routing_decision(),
                        planner_decision=action.planner_decision(controller.config),
                        dependency_ids=list(action.dependency_ids),
                    ) as invocation:
                        result = agent.run(agent_input)
                        invocation.raw_agent_output = _jsonable(result)
                        invocation.structured_output = _jsonable(result.get("output", {}))
                        invocation.verifier_decision = controller.verify_output(
                            stage=stage,
                            result=result,
                            output_schema=_stage_output_schema(agent),
                            stage_config=stage_config,
                            agent_outputs=outputs,
                        )
                        invocation.validation_result = {"status": invocation.verifier_decision.decision, "policy": invocation.verifier_decision.policy}
            result_queue.put(("ok", result))
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
        trace_recorder.record_timeout(
            stage_name=stage,
            agent_name=type(agent).__name__,
            agent_version=str(getattr(agent, "prompt_version_id", "static")),
            raw_agent_input={"session": session, "stage_config": dict(stage_config)},
            latency_seconds=latency,
            error_message=message,
            dependency_ids=[trace_recorder.invocations[-1].invocation_id] if trace_recorder.invocations else [],
        )
        return AgentOutput(stage=stage, output=outputs[stage], confidence=0.0, rationale=message)

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
        return result
    except Exception as exc:
        latency = time.perf_counter() - started
        outputs[stage] = {"error": str(exc), "error_type": type(exc).__name__}
        traces.append({"stage": stage, "status": "error", "latency_seconds": latency, "error_type": type(exc).__name__, "error": str(exc)[:2000]})
        raise


def _model_provider(agent: Any) -> str | None:
    provider = getattr(getattr(agent, "llm", None), "provider", None)
    if provider is not None:
        return str(getattr(provider, "value", provider))
    return None


def _model_name(agent: Any, stage_config: Mapping[str, Any]) -> str | None:
    if stage_config.get("model"):
        return str(stage_config["model"])
    env_model = _agent_model_env(agent)
    if env_model:
        return env_model
    config = getattr(agent, "config", {})
    provider = getattr(getattr(getattr(agent, "llm", None), "provider", None), "value", None)
    if isinstance(config, Mapping) and provider:
        models = config.get("models")
        if isinstance(models, Mapping) and provider in models:
            return str(models[provider])
    return None


def _agent_model_env(agent: Any) -> str | None:
    env_by_agent = {
        "RecipientProfilingAgent": "GMGI_RECIPIENT_MODEL",
        "RelationshipAnalysisAgent": "GMGI_RELATIONSHIP_MODEL",
        "RecommendationAgent": "GMGI_RECOMMENDATION_MODEL",
        "GreetingStoryAgent": "GMGI_GREETING_MODEL",
        "DeliveryPlannerAgent": "GMGI_DELIVERY_MODEL",
    }
    specific = os.getenv(env_by_agent.get(type(agent).__name__, ""))
    return specific or os.getenv("GMGI_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL")


def _model_parameters(agent: Any, stage_config: Mapping[str, Any]) -> dict[str, Any]:
    config = getattr(agent, "config", {})
    parameters: dict[str, Any] = {}
    if isinstance(config, Mapping) and "temperature" in config:
        parameters["temperature"] = stage_config.get("temperature", config.get("temperature"))
    for key in ("max_steps", "num_predict", "num_inference_steps", "guidance_scale", "width", "height", "critique_max_retries"):
        if key in stage_config:
            parameters[key] = stage_config[key]
    parameters["request_timeout_seconds"] = _request_timeout(agent)
    parameters["endpoint_type"] = _endpoint_type(agent)
    parameters["retry_configuration"] = {
        "max_validation_retries": stage_config.get("max_validation_retries", _config_value(agent, "runtime_config", "max_validation_retries")),
        "critique_max_retries": stage_config.get("critique_max_retries"),
    }
    return parameters


def _request_timeout(agent: Any) -> Any:
    llm = getattr(agent, "llm", None)
    if getattr(llm, "timeout_seconds", None) is not None:
        return getattr(llm, "timeout_seconds")
    if type(agent).__name__ in {"RecipientProfilingAgent", "DeliveryPlannerAgent"}:
        return os.getenv("GMGI_OLLAMA_TIMEOUT_SECONDS", "30")
    if type(agent).__name__ in {"GreetingStoryAgent", "RelationshipAnalysisAgent", "RecommendationAgent"}:
        return os.getenv("GMGI_OLLAMA_TIMEOUT_SECONDS", "unknown")
    return "unknown"


def _endpoint_type(agent: Any) -> str:
    name = type(agent).__name__
    if name in {"RecipientProfilingAgent", "DeliveryPlannerAgent"}:
        return "ollama_openai_compatible"
    if name in {"RelationshipAnalysisAgent", "RecommendationAgent"}:
        return "smolagents_ollama_chat"
    if name == "GreetingStoryAgent":
        return "ollama_python_client"
    if getattr(getattr(agent, "llm", None), "provider", None) is not None:
        return "structured_http_llm"
    return "local_or_deterministic"


def _config_value(agent: Any, attr: str, key: str) -> Any:
    value = getattr(agent, attr, None)
    return value.get(key) if isinstance(value, Mapping) else None


def _input_diagnostics(agent_input: Mapping[str, Any], stage_config: Mapping[str, Any]) -> dict[str, Any]:
    serialized_input = json.dumps(_jsonable(agent_input), ensure_ascii=False, default=str)
    serialized_context = json.dumps(_jsonable(stage_config), ensure_ascii=False, default=str)
    memories = _as_list(stage_config.get("memories"))
    constraints = {key: stage_config[key] for key in ("budget_hint", "agency_slider", "generation_backend") if key in stage_config}
    return {
        "input_character_count": len(serialized_input),
        "approx_input_token_count": max(1, len(serialized_input) // 4),
        "context_character_count": len(serialized_context),
        "approx_context_token_count": max(1, len(serialized_context) // 4),
        "memory_item_count": len(memories),
        "memory_character_count": sum(len(str(item)) for item in memories),
        "constraint_character_count": len(json.dumps(_jsonable(constraints), ensure_ascii=False, default=str)),
    }


def _stage_output_schema(agent: Any) -> Mapping[str, Any] | None:
    config = getattr(agent, "config", {})
    if isinstance(config, Mapping) and isinstance(config.get("output_schema"), Mapping):
        return config["output_schema"]
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _context_embedding(memories: Sequence[Mapping[str, Any]], preferences: Sequence[Mapping[str, Any]], size: int = 512) -> list[float]:
    text = " ".join(
        [str(memory.get("content", "")) for memory in memories]
        + [str(preference.get("value", "")) for preference in preferences]
    )
    if not text.strip():
        return [0.0] * size
    buckets = [0.0] * size
    for index, char in enumerate(text.encode("utf-8")):
        buckets[index % size] += (char % 31) / 31.0
    total = sum(abs(value) for value in buckets) or 1.0
    return [value / total for value in buckets]


def _visual_occasion(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "birthday": ("birthday",),
        "anniversary": ("anniversary",),
        "graduation": ("graduation",),
        "housewarming": ("housewarming", "house warming"),
        "promotion": ("promotion",),
        "thank-you": ("thank-you", "thank you", "thanks"),
    }
    for canonical, variants in aliases.items():
        if any(variant in text for variant in variants):
            return canonical
    return "other"


def _diffusers_prompt(
    relationship: Mapping[str, Any],
    occasion: Mapping[str, Any],
    memory: Mapping[str, Any],
    artifact_type: str,
    style_prompt: str,
    agency_slider: float,
) -> str:
    anchor = "human-specified style details" if agency_slider < 0.5 else "memory-grounded emotional symbolism"
    return CreativeGenerationAgent.build_image_prompt(
        artifact_type=artifact_type,
        occasion=str(occasion.get("name", "special occasion")),
        relationship_type=str(relationship.get("type", "relationship")),
        emotion_tag=str(memory.get("emotion_tag", "joy")),
        style=str(style_prompt),
        agency_anchor=anchor,
    )


def _creative_agent() -> CreativeGenerationAgent:
    return CreativeGenerationAgent()


def _artifact_analysis(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for report in reports:
        creative = _as_mapping(_as_mapping(report.get("stage_reports")).get("creative_generation"))
        metrics = _as_list(creative.get("metrics"))
        path = ""
        width = 0
        height = 0
        for metric in metrics:
            item = _as_mapping(metric)
            details = _as_mapping(item.get("details"))
            if item.get("name") == "artifact_file_exists":
                path = str(details.get("artifact_path") or "")
            if item.get("name") == "image_resolution_practical":
                width = int(details.get("width") or 0)
                height = int(details.get("height") or 0)
        digest = _file_digest(Path(path)) if path else None
        artifacts.append({"case_id": report.get("case_id"), "artifact_path": path, "width": width, "height": height, "sha256": digest})

    hashes = [item["sha256"] for item in artifacts if item.get("sha256")]
    unique_hashes = len(set(hashes))
    generated = len(hashes)
    duplicate_collapse_rate = 0.0 if generated <= 1 else 1.0 - (unique_hashes / generated)
    return {
        "generated_artifact_count": generated,
        "unique_artifact_count": unique_hashes,
        "duplicate_collapse_rate": duplicate_collapse_rate,
        "artifacts": artifacts,
    }


def _harness_run_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    trace = _as_mapping(report.get("run_trace"))
    invocations = _as_list(trace.get("invocations"))
    return {
        "run_id": report.get("run_id"),
        "harness_id": report.get("harness_id"),
        "harness_version": report.get("harness_version"),
        "harness_config_hash": report.get("harness_config_hash"),
        "final_status": trace.get("final_status"),
        "trajectory": _trajectory(report),
        "verifier_outcomes": _verifier_outcomes(report),
        "retry_count": sum(int(_as_mapping(item).get("retry_count") or 0) for item in invocations),
        "failure_count": sum(1 for item in invocations if _as_mapping(item).get("status") in {"error", "timeout"}),
        "quality_score": report.get("overall_score"),
    }


def _trajectory(report: Mapping[str, Any]) -> list[str]:
    trace = _as_mapping(report.get("run_trace"))
    return [str(_as_mapping(item).get("stage_name")) for item in _as_list(trace.get("invocations"))]


def _verifier_outcomes(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace = _as_mapping(report.get("run_trace"))
    outcomes = []
    for item in _as_list(trace.get("invocations")):
        invocation = _as_mapping(item)
        verifier = _as_mapping(invocation.get("verifier_decision"))
        outcomes.append(
            {
                "stage": invocation.get("stage_name"),
                "decision": verifier.get("decision"),
                "policy": verifier.get("policy"),
            }
        )
    return outcomes


def _file_digest(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
