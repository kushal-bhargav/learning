from __future__ import annotations

import argparse
import json
import os
import queue
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from traceback import format_exception_only
from typing import Any, Literal, Mapping, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field

from src.harness import HarnessCandidate, HarnessConfig, default_harness_registry

from .benchmark import BenchmarkCase, load_cases, run_case


EVALUATION_VERSION = "gmgi_phase_5_harness_comparison_v1"
ExecutionStatus = Literal["SUCCESS", "PARTIAL_SUCCESS", "HARNESS_FAILURE", "INFRASTRUCTURE_FAILURE", "EVALUATION_FAILURE", "TIMEOUT", "CANCELLED"]


class HarnessExperimentManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(min_length=1)
    cases: list[str]
    candidates: list[str]
    case_file: str | None = None
    provider: str | None = None
    model: str | None = None
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = 2026
    state_isolation: Literal["isolated", "shared"] = "isolated"
    evaluation_version: str = EVALUATION_VERSION
    include_creative: bool = False
    agency_slider: float = 0.5
    stage_timeout_seconds: float | None = None
    execution_timeout_seconds: float | None = None
    repetitions: int = Field(default=1, ge=1)
    randomized_order: bool = False
    resume: bool = True
    report_path: str | None = None


class HarnessReplicationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(min_length=1)
    case_id: str
    case_file: str = "experiments/evals/ui_permutation_cases.json"
    candidates: list[str]
    repetitions: int = Field(default=3, ge=1)
    provider: str | None = None
    model: str | None = None
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    evaluation_version: str = EVALUATION_VERSION
    include_creative: bool = False
    agency_slider: float = 0.5
    state_isolation: Literal["isolated", "shared"] = "isolated"
    stage_timeout_seconds: float | None = None
    execution_timeout_seconds: float | None = None
    seed: int = 2026
    execution_order: Literal["interleaved", "grouped"] = "interleaved"
    resume: bool = True
    report_path: str = "experiments/evals/phase65_harness_replication_report.md"


class HarnessExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    case_id: str
    candidate_id: str
    candidate_identity: str
    candidate: dict[str, Any]
    harness_config: dict[str, Any]
    run_id: str
    trace: dict[str, Any]
    evaluation: dict[str, Any]
    timing: dict[str, Any]
    resource_usage: dict[str, Any]
    randomness: dict[str, Any]
    final_status: ExecutionStatus
    termination_reason: str | None = None
    failure_type: Literal["harness", "infrastructure", "evaluation", "none"] = "none"
    failure_categories: list[str] = Field(default_factory=list)
    progress_events: list[dict[str, Any]] = Field(default_factory=list)


def run_case_across_harnesses(
    case: BenchmarkCase,
    *,
    candidates: Sequence[HarnessCandidate],
    experiment_id: str,
    output_dir: str | Path = "experiments/evals/harness_comparison",
    include_creative: bool = False,
    agency_slider: float = 0.5,
    seed: int = 2026,
    stage_timeout_seconds: float | None = None,
    execution_timeout_seconds: float | None = None,
    repetitions: int = 1,
    randomized_order: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ordered = list(candidates)
    if randomized_order:
        random.Random(seed).shuffle(ordered)
    executions: list[HarnessExecutionResult] = []
    for candidate in ordered:
        for repetition in range(1, repetitions + 1):
            executions.append(
                execute_candidate_on_case(
                    case,
                    candidate=candidate,
                    experiment_id=experiment_id,
                    output_dir=root / "executions" / case.case_id / candidate.candidate_id / f"rep-{repetition}",
                    include_creative=include_creative,
                    agency_slider=agency_slider,
                    seed=seed + repetition - 1,
                    stage_timeout_seconds=stage_timeout_seconds,
                    execution_timeout_seconds=execution_timeout_seconds,
                    repetition=repetition,
                    resume=resume,
                )
            )
    comparison = compare_execution_results(executions)
    report = {
        "experiment_id": experiment_id,
        "case_id": case.case_id,
        "state_isolation": "isolated",
        "execution_order": [execution.candidate_id for execution in executions],
        "executions": [execution.model_dump(mode="json") for execution in executions],
        "comparison": comparison,
    }
    (root / f"{case.case_id}_comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def execute_candidate_on_case(
    case: BenchmarkCase,
    *,
    candidate: HarnessCandidate,
    experiment_id: str,
    output_dir: str | Path,
    include_creative: bool,
    agency_slider: float,
    seed: int,
    stage_timeout_seconds: float | None,
    execution_timeout_seconds: float | None = None,
    repetition: int = 1,
    resume: bool = True,
) -> HarnessExecutionResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "execution_result.json"
    if resume and result_path.exists():
        return HarnessExecutionResult(**json.loads(result_path.read_text(encoding="utf-8")))
    progress: list[dict[str, Any]] = [_event("execution_started", component="harness_comparison", details={"repetition": repetition})]
    started = time.perf_counter()
    report, run_error = _run_case_with_timeout(
        case,
        candidate=candidate,
        output_path=output_path,
        include_creative=include_creative,
        agency_slider=agency_slider,
        seed=seed,
        stage_timeout_seconds=stage_timeout_seconds,
        execution_timeout_seconds=execution_timeout_seconds,
        progress=progress,
    )
    elapsed = time.perf_counter() - started
    trace = dict(_as_mapping(report.get("run_trace")))
    trace["candidate_id"] = candidate.candidate_id
    trace["candidate_identity"] = candidate.identity
    trace.setdefault("events", [])
    trace["events"] = [*_as_list(trace.get("events")), *progress]
    status, failure_type, termination_reason = _status_from_report(report, run_error)
    result = HarnessExecutionResult(
        experiment_id=experiment_id,
        case_id=case.case_id,
        candidate_id=candidate.candidate_id,
        candidate_identity=candidate.identity,
        candidate=candidate.to_dict(),
        harness_config=candidate.config.to_dict(),
        run_id=str(report.get("run_id") or trace.get("run_id") or ""),
        trace=dict(trace),
        evaluation=_evaluation_summary(report),
        timing=_timing_summary(trace),
        resource_usage=_resource_summary(trace),
        randomness=_randomness_summary(seed=seed, agency_slider=agency_slider, include_creative=include_creative, repetition=repetition),
        final_status=status,
        termination_reason=termination_reason,
        failure_type=failure_type,
        failure_categories=_failure_categories(trace, run_error),
        progress_events=progress,
    )
    result.timing["execution_wrapper_latency_seconds"] = elapsed
    result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, default=str), encoding="utf-8")
    return result


def _run_case_with_timeout(
    case: BenchmarkCase,
    *,
    candidate: HarnessCandidate,
    output_path: Path,
    include_creative: bool,
    agency_slider: float,
    seed: int,
    stage_timeout_seconds: float | None,
    execution_timeout_seconds: float | None,
    progress: list[dict[str, Any]],
) -> tuple[dict[str, Any], BaseException | None]:
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            progress.append(_event("candidate_run_started", component=candidate.candidate_id))
            value = run_case(
                case,
                output_dir=output_path,
                include_creative=include_creative,
                agency_slider=agency_slider,
                seed=seed,
                stage_timeout_seconds=stage_timeout_seconds,
                harness_config=candidate.config,
            )
            progress.append(_event("candidate_run_completed", component=candidate.candidate_id, status="success"))
            result_queue.put(("ok", value))
        except Exception as exc:  # pragma: no cover - exercised through tests via monkeypatch
            progress.append(_event("candidate_run_failed", component=candidate.candidate_id, status="error", details={"error_type": type(exc).__name__, "error_message": str(exc)}))
            result_queue.put(("error", exc))

    worker = threading.Thread(target=invoke, name=f"gmgi-harness-{candidate.candidate_id}-{case.case_id}", daemon=True)
    worker.start()
    worker.join(timeout=execution_timeout_seconds)
    if worker.is_alive():
        message = f"candidate execution exceeded {execution_timeout_seconds:.1f}s timeout" if execution_timeout_seconds else "candidate execution timed out"
        error = TimeoutError(message)
        progress.append(
            _event(
                "run_timeout",
                component="harness_comparison",
                status="timeout",
                details={
                    "timeout_stage": None,
                    "timeout_component": "harness_comparison",
                    "timeout_duration": execution_timeout_seconds,
                    "timeout_type": "experiment_timeout",
                },
            )
        )
        return _timeout_report(case, candidate, output_path, message, progress), error
    status, value = result_queue.get()
    if status == "ok":
        return value, None
    return _error_report(case, candidate, output_path, value, progress), value


def _timeout_report(case: BenchmarkCase, candidate: HarnessCandidate, output_path: Path, message: str, progress: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trace = {
        "run_id": f"timeout-{candidate.candidate_id}-{case.case_id}",
        "case_id": case.case_id,
        "harness_config": candidate.config.to_dict(),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "total_latency_seconds": None,
        "final_status": "error",
        "termination_reason": "experiment_timeout",
        "events": list(progress),
        "invocations": [],
    }
    return {
        "case_id": case.case_id,
        "run_id": trace["run_id"],
        "harness_id": candidate.config.harness_id,
        "harness_version": candidate.config.harness_version,
        "harness_config_hash": candidate.config.config_hash,
        "overall_score": None,
        "stage_reports": {},
        "run_trace": trace,
        "error": {"error_type": "TimeoutError", "message": message, "failure_type": "infrastructure"},
    }


def _error_report(case: BenchmarkCase, candidate: HarnessCandidate, output_path: Path, error: BaseException, progress: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    category = _classify_exception(error)
    trace = {
        "run_id": f"error-{candidate.candidate_id}-{case.case_id}",
        "case_id": case.case_id,
        "harness_config": candidate.config.to_dict(),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "total_latency_seconds": None,
        "final_status": "error",
        "termination_reason": category,
        "events": list(progress),
        "invocations": [],
    }
    return {
        "case_id": case.case_id,
        "run_id": trace["run_id"],
        "harness_id": candidate.config.harness_id,
        "harness_version": candidate.config.harness_version,
        "harness_config_hash": candidate.config.config_hash,
        "overall_score": None,
        "stage_reports": {},
        "run_trace": trace,
        "error": {"error_type": type(error).__name__, "message": str(error), "failure_type": category, "error_reference": "".join(format_exception_only(type(error), error)).strip()},
    }


def compare_execution_results(executions: Sequence[HarnessExecutionResult]) -> dict[str, Any]:
    rows = []
    for execution in executions:
        resource = execution.resource_usage
        evaluation = execution.evaluation
        rows.append(
            {
                "case_id": execution.case_id,
                "candidate_id": execution.candidate_id,
                "repetition": execution.randomness.get("repetition"),
                "status": execution.final_status,
                "failure_type": execution.failure_type,
                "termination_reason": execution.termination_reason,
                "quality_comparison_eligible": _quality_comparison_eligible(execution),
                "quality": evaluation.get("overall_quality") if _quality_comparison_eligible(execution) else None,
                "constraint": evaluation.get("constraint_satisfaction"),
                "reliability": _reliability_score(execution),
                "latency": execution.timing.get("total_wall_clock_latency_seconds"),
                "tokens": resource.get("run_total_tokens"),
                "token_usage_status": resource.get("token_usage_status"),
                "cost": resource.get("total_cost"),
                "cost_status": resource.get("cost_status"),
                "invocations": resource.get("agent_invocation_count"),
                "tool_calls": resource.get("tool_call_count"),
                "retries": resource.get("retry_count"),
                "fallbacks": resource.get("fallback_count"),
                "verifications": resource.get("verification_count"),
                "failures": resource.get("failed_invocation_count"),
                "trajectory": _trajectory(execution.trace),
                "failure_categories": execution.failure_categories,
            }
        )
    return {
        "table": rows,
        "pareto": _pareto_analysis(rows),
        "trajectory_differences": _trajectory_differences(rows),
        "metrics_note": "Metrics remain separate; no single opaque harness score is produced.",
    }


def _quality_comparison_eligible(execution: HarnessExecutionResult) -> bool:
    if execution.final_status == "SUCCESS":
        return True
    if execution.final_status == "HARNESS_FAILURE":
        return bool(execution.evaluation.get("final_output_available"))
    return False


def benchmark_health_check(
    *,
    candidate: HarnessCandidate | None = None,
    case: BenchmarkCase | None = None,
    output_dir: str | Path = "experiments/evals/harness_health_check",
    stage_timeout_seconds: float | None = 10,
    execution_timeout_seconds: float | None = 60,
) -> dict[str, Any]:
    registry_candidate = candidate or default_harness_registry().get("gmgi_default")
    selected_case = case or load_cases()[0]
    provider = _provider_health()
    smoke = None
    if provider["reachable"]:
        smoke_result = execute_candidate_on_case(
            selected_case,
            candidate=registry_candidate,
            experiment_id="health-check",
            output_dir=Path(output_dir) / selected_case.case_id / registry_candidate.candidate_id,
            include_creative=False,
            agency_slider=0.5,
            seed=2026,
            stage_timeout_seconds=stage_timeout_seconds,
            execution_timeout_seconds=execution_timeout_seconds,
            resume=False,
        )
        smoke = {
            "status": smoke_result.final_status,
            "failure_type": smoke_result.failure_type,
            "termination_reason": smoke_result.termination_reason,
            "run_id": smoke_result.run_id,
        }
    allowed = bool(provider["reachable"] and smoke and smoke["status"] == "SUCCESS")
    report = {
        "provider": provider,
        "candidate": registry_candidate.candidate_id,
        "case_id": selected_case.case_id,
        "single_case_smoke": smoke,
        "harness_comparison_allowed": allowed,
        "recommendation": "run comparison" if allowed else "fix provider/model execution before large comparisons",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "health_check.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _provider_health() -> dict[str, Any]:
    import os

    provider = os.getenv("GMGI_LLM_PROVIDER", "ollama")
    host = os.getenv("GMGI_OLLAMA_HOST") or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    if provider != "ollama":
        return {"provider": provider, "reachable": "unknown", "reason": "preflight only implements cheap Ollama reachability check"}
    try:
        started = time.perf_counter()
        with urlopen(host.rstrip("/") + "/api/tags", timeout=2) as response:
            latency = time.perf_counter() - started
            payload = json.loads(response.read().decode("utf-8"))
            models = [str(item.get("name") or item.get("model")) for item in _as_list(payload.get("models")) if item]
            requested_model = os.getenv("GMGI_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL")
            model_available = requested_model in models if requested_model else None
            return {
                "provider": "ollama",
                "host": host,
                "reachable": response.status == 200 and model_available is not False,
                "latency_seconds": latency,
                "status": response.status,
                "requested_model": requested_model,
                "model_available": model_available,
                "available_models": models[:20],
                "error": None if model_available is not False else f"selected Ollama model is not installed: {requested_model}",
            }
    except (OSError, URLError) as exc:
        return {"provider": "ollama", "host": host, "reachable": False, "error_type": type(exc).__name__, "error": str(exc)}


def stage_latency_summary(executions: Sequence[HarnessExecutionResult]) -> dict[str, Any]:
    by_stage: dict[str, list[Mapping[str, Any]]] = {}
    for execution in executions:
        for invocation in _as_list(execution.trace.get("invocations")):
            item = _as_mapping(invocation)
            by_stage.setdefault(str(item.get("stage_name")), []).append(item)
    return {
        stage: {
            "invocation_count": len(items),
            "success_count": sum(1 for item in items if item.get("status") == "success"),
            "timeout_count": sum(1 for item in items if item.get("status") == "timeout"),
            "failure_count": sum(1 for item in items if item.get("status") == "error"),
            "min_latency": _min(item.get("latency_seconds") for item in items),
            "median_latency": _median(item.get("latency_seconds") for item in items),
            "max_latency": _max(item.get("latency_seconds") for item in items),
            "sample_size_note": "small_sample" if len(items) < 5 else "ok",
        }
        for stage, items in sorted(by_stage.items())
    }


def run_harness_experiment(
    *,
    cases: Sequence[BenchmarkCase],
    candidates: Sequence[HarnessCandidate],
    experiment_id: str | None = None,
    output_dir: str | Path = "experiments/evals/harness_comparison",
    limit: int | None = None,
    include_creative: bool = False,
    agency_slider: float = 0.5,
    seed: int = 2026,
    stage_timeout_seconds: float | None = None,
    randomized_order: bool = False,
    execution_timeout_seconds: float | None = None,
    repetitions: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    selected_cases = list(cases[:limit] if limit else cases)
    experiment = experiment_id or f"harness-exp-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    root = Path(output_dir)
    manifest = HarnessExperimentManifest(
        experiment_id=experiment,
        cases=[case.case_id for case in selected_cases],
        candidates=[candidate.candidate_id for candidate in candidates],
        provider=_selected_provider(),
        model=_selected_model(),
        model_parameters=_selected_model_parameters(),
        seed=seed,
        include_creative=include_creative,
        agency_slider=agency_slider,
        stage_timeout_seconds=stage_timeout_seconds,
        execution_timeout_seconds=execution_timeout_seconds,
        repetitions=repetitions,
        randomized_order=randomized_order,
        resume=resume,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    case_reports = [
        run_case_across_harnesses(
            case,
            candidates=candidates,
            experiment_id=experiment,
            output_dir=root,
            include_creative=include_creative,
            agency_slider=agency_slider,
            seed=seed,
            stage_timeout_seconds=stage_timeout_seconds,
            execution_timeout_seconds=execution_timeout_seconds,
            repetitions=repetitions,
            randomized_order=randomized_order,
            resume=resume,
        )
        for case in selected_cases
    ]
    summary = {
        "manifest": manifest.model_dump(mode="json"),
        "environment": _environment_snapshot(manifest, candidates),
        "candidate_definitions": candidate_definition_table(candidates),
        "candidate_differences": candidate_difference_audit(candidates),
        "case_selection": describe_case_selection(selected_cases),
        "case_count": len(case_reports),
        "candidate_count": len(candidates),
        "cases": case_reports,
        "aggregate": _aggregate(case_reports),
        "trajectory_analysis": trajectory_analysis(case_reports),
        "stage_latency": stage_latency_summary(
            [
                HarnessExecutionResult(**execution)
                for report in case_reports
                for execution in _as_list(report.get("executions"))
            ]
        ),
        "evaluation_stability": evaluation_stability_audit(),
    }
    (root / "harness_comparison_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_comparison_csv(root / "harness_comparison_rows.csv", [row for report in case_reports for row in report["comparison"]["table"]])
    write_phase6_report(root / "phase6_harness_benchmark_report.md", summary)
    return summary


def run_replication_experiment(
    manifest: HarnessReplicationManifest,
    *,
    output_dir: str | Path = "experiments/evals/harness_replication_phase65",
) -> dict[str, Any]:
    _apply_manifest_environment(manifest.model_dump(mode="json"))
    registry = default_harness_registry()
    candidates = [registry.get(candidate_id) for candidate_id in manifest.candidates]
    cases = [case for case in load_cases(manifest.case_file) if case.case_id == manifest.case_id]
    if not cases:
        raise ValueError(f"Replication case not found: {manifest.case_id}")
    case = cases[0]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    executions: list[HarnessExecutionResult] = []
    for repetition, candidate in replication_execution_order(candidates, repetitions=manifest.repetitions, mode=manifest.execution_order):
        executions.append(
            execute_candidate_on_case(
                case,
                candidate=candidate,
                experiment_id=manifest.experiment_id,
                output_dir=root / "executions" / case.case_id / candidate.candidate_id / f"rep-{repetition}",
                include_creative=manifest.include_creative,
                agency_slider=manifest.agency_slider,
                seed=manifest.seed + repetition - 1,
                stage_timeout_seconds=manifest.stage_timeout_seconds,
                execution_timeout_seconds=manifest.execution_timeout_seconds,
                repetition=repetition,
                resume=manifest.resume,
            )
        )
    report = analyze_replications(
        executions,
        manifest=manifest,
        case=case,
        candidates=candidates,
    )
    (root / "replication_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_phase65_report(manifest.report_path, report)
    return report


def replication_execution_order(
    candidates: Sequence[HarnessCandidate],
    *,
    repetitions: int,
    mode: Literal["interleaved", "grouped"] = "interleaved",
) -> list[tuple[int, HarnessCandidate]]:
    if mode == "grouped":
        return [(repetition, candidate) for candidate in candidates for repetition in range(1, repetitions + 1)]
    return [(repetition, candidate) for repetition in range(1, repetitions + 1) for candidate in candidates]


def analyze_replications(
    executions: Sequence[HarnessExecutionResult],
    *,
    manifest: HarnessReplicationManifest,
    case: BenchmarkCase,
    candidates: Sequence[HarnessCandidate],
) -> dict[str, Any]:
    rows = replication_per_run_rows(executions)
    stage_reliability = replication_stage_reliability(executions)
    stage_latency = replication_stage_latency(executions)
    aggregate = replication_aggregate(executions)
    trajectory = replication_trajectory_rows(executions)
    return {
        "manifest": manifest.model_dump(mode="json"),
        "environment": _environment_snapshot(
            HarnessExperimentManifest(
                experiment_id=manifest.experiment_id,
                cases=[manifest.case_id],
                candidates=manifest.candidates,
                case_file=manifest.case_file,
                provider=manifest.provider,
                model=manifest.model,
                model_parameters=manifest.model_parameters,
                seed=manifest.seed,
                state_isolation=manifest.state_isolation,
                evaluation_version=manifest.evaluation_version,
                include_creative=manifest.include_creative,
                agency_slider=manifest.agency_slider,
                stage_timeout_seconds=manifest.stage_timeout_seconds,
                execution_timeout_seconds=manifest.execution_timeout_seconds,
                repetitions=manifest.repetitions,
                randomized_order=False,
                resume=manifest.resume,
            ),
            candidates,
        ),
        "candidate_definitions": candidate_definition_table(candidates),
        "case": {"case_id": case.case_id, "custom_profile": dict(case.custom_profile)},
        "executions": [execution.model_dump(mode="json") for execution in executions],
        "per_run": rows,
        "stage_reliability": stage_reliability,
        "stage_latency": stage_latency,
        "aggregate": aggregate,
        "trajectory": trajectory,
        "within_harness_variance": within_harness_variance(executions),
        "harness_vs_runtime_attribution": attribution_summary(executions),
        "decision_gate": replication_decision_gate(executions),
        "meta_harness_readiness": "PARTIALLY READY",
    }


def _evaluation_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    stage_reports = _as_mapping(report.get("stage_reports"))
    structural = _as_mapping(report.get("structural_metrics"))
    overall_quality = report.get("overall_score")
    if overall_quality is None:
        overall_quality = report.get("overall_quality_score")
    return {
        "overall_quality": overall_quality,
        "stage_reports": stage_reports,
        "schema_conformance": structural.get("schema_conformance"),
        "constraint_satisfaction": _metric_score(stage_reports, "gift_intent_reasoning", "budget_constraint_preserved"),
        "DAG_validity": structural.get("dag_validity"),
        "provenance": structural.get("provenance_traceability"),
        "faithfulness": report.get("faithfulness"),
        "purpose_alignment": report.get("purpose_alignment"),
        "creative_quality": _as_mapping(stage_reports.get("creative_generation")).get("quality_score"),
        "human_behavior": report.get("behavioral_metrics"),
        "unavailable_metrics": [
            name
            for name, value in {
                "faithfulness": report.get("faithfulness"),
                "purpose_alignment": report.get("purpose_alignment"),
                "human_behavior": report.get("behavioral_metrics"),
            }.items()
            if value is None
        ],
    }


def _status_from_report(report: Mapping[str, Any], run_error: BaseException | None) -> tuple[ExecutionStatus, Literal["harness", "infrastructure", "evaluation", "none"], str | None]:
    if run_error is not None:
        failure_type = _classify_exception(run_error)
        if isinstance(run_error, TimeoutError):
            return "TIMEOUT", "infrastructure", "experiment_timeout"
        if failure_type == "evaluation":
            return "EVALUATION_FAILURE", "evaluation", type(run_error).__name__
        if failure_type == "harness":
            return "HARNESS_FAILURE", "harness", type(run_error).__name__
        return "INFRASTRUCTURE_FAILURE", "infrastructure", type(run_error).__name__
    trace = _as_mapping(report.get("run_trace"))
    final_status = str(trace.get("final_status") or "")
    termination_reason = str(trace.get("termination_reason") or "")
    if any(_as_mapping(invocation).get("status") == "timeout" for invocation in _as_list(trace.get("invocations"))):
        return "TIMEOUT", "infrastructure", termination_reason or "agent_timeout"
    if final_status == "success":
        return "SUCCESS", "none", termination_reason or "completed"
    if final_status == "partial":
        return "PARTIAL_SUCCESS", "infrastructure" if "timeout" in termination_reason else "harness", termination_reason or "partial"
    return "HARNESS_FAILURE", "harness", termination_reason or final_status or "unknown"


def _classify_exception(error: BaseException) -> Literal["harness", "infrastructure", "evaluation"]:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if isinstance(error, TimeoutError) or "timeout" in name or "timed out" in message:
        return "infrastructure"
    if any(token in message for token in ("connection refused", "network", "api", "ollama", "provider", "model loading", "filesystem", "permission")):
        return "infrastructure"
    if any(token in name for token in ("validation", "jsondecode")) or "evaluator" in message:
        return "evaluation"
    if any(token in message for token in ("routing", "plan", "dependency", "verification", "retry", "fallback", "constraint")):
        return "harness"
    return "infrastructure"


def _randomness_summary(*, seed: int, agency_slider: float, include_creative: bool, repetition: int) -> dict[str, Any]:
    return {
        "experiment_seed": seed,
        "repetition": repetition,
        "agency_slider": agency_slider,
        "include_creative": include_creative,
        "LLM_sampling": "stochastic_unless_temperature_zero_or_provider_forces_determinism",
        "image_generation": "stochastic_when_include_creative_true",
        "tool_results": "deterministic_for_local_fixture_tools",
        "retrieval": "deterministic_for_same_experience_store",
        "bandit": "state_dependent_if_shared_state_enabled",
        "stochastic_model_generation": True,
    }


def _event(
    event_type: str,
    *,
    component: str | None = None,
    status: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "status": status,
        "details": dict(details or {}),
    }


def evaluation_stability_audit() -> dict[str, dict[str, str]]:
    return {
        "schema_conformance": {"readiness": "READY", "stability": "deterministic", "reason": "Pure structural validation."},
        "constraint_satisfaction": {"readiness": "READY", "stability": "deterministic", "reason": "Deterministic field and budget checks where data is present."},
        "DAG_validity": {"readiness": "READY", "stability": "deterministic", "reason": "Pure plan-graph validation."},
        "provenance": {"readiness": "PARTIAL", "stability": "deterministic", "reason": "Depends on whether outputs contain evidence/provenance fields."},
        "faithfulness": {"readiness": "PARTIAL", "stability": "model_judge_dependent_or_unavailable", "reason": "Only stable when using deterministic/local verifier inputs."},
        "self_consistency": {"readiness": "PARTIAL", "stability": "stochastic", "reason": "Requires repeated model executions."},
        "purpose_alignment": {"readiness": "PARTIAL", "stability": "model_judge_dependent", "reason": "Judge availability and model choice affect score."},
        "agent_level_quality": {"readiness": "READY", "stability": "deterministic_given_outputs", "reason": "Current quality metrics are deterministic over captured outputs."},
        "overall_system_quality": {"readiness": "READY", "stability": "deterministic_given_outputs", "reason": "Aggregation is deterministic."},
        "creative_quality": {"readiness": "PARTIAL", "stability": "unavailable_when_no_image_generation", "reason": "Must distinguish creative input coverage from generated image quality."},
        "human_behavior": {"readiness": "NOT READY", "stability": "unavailable", "reason": "UI permutation data is not real human action data."},
        "latency": {"readiness": "READY", "stability": "environment_sensitive", "reason": "Measured directly but depends on runtime/provider conditions."},
        "tokens": {"readiness": "PARTIAL", "stability": "provider_dependent", "reason": "Only exact when provider exposes token usage."},
        "cost": {"readiness": "PARTIAL", "stability": "provider_dependent", "reason": "Only exact when pricing/cost metadata is configured."},
        "reliability": {"readiness": "READY", "stability": "deterministic_given_trace", "reason": "Derived from explicit status/failure taxonomy."},
    }


def audit_ui_permutation_dataset(path: str | Path = "experiments/evals/ui_permutation_cases.json") -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"path": str(source), "status": "missing"}
    payload = json.loads(source.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    profiles = [_as_mapping(case.get("custom_profile") if isinstance(case, Mapping) else {}) for case in cases]
    dimensions = {}
    for key in ("relationship_type", "occasion_name", "budget_hint", "formality", "agency_slider"):
        values = [str(profile.get(key, "")) for profile in profiles]
        dimensions[key] = {"unique_count": len(set(values)), "sample_values": sorted(set(values))[:20]}
    serialized = [json.dumps(profile, sort_keys=True, default=str) for profile in profiles]
    creative_required = sum(1 for profile in profiles if profile.get("creative_requirement") or profile.get("gift_artifact_type") or profile.get("agency_slider") is not None)
    invalid = sum(1 for profile in profiles if not profile.get("recipient_name"))
    return {
        "path": str(source),
        "case_count": len(cases),
        "dimensions": dimensions,
        "duplicate_cases": len(serialized) - len(set(serialized)),
        "invalid_cases_missing_recipient": invalid,
        "creative_or_style_cases": creative_required,
        "external_delivery_data_required": 0,
        "fast_benchmark_suitability": "smoke/check subsets preferred until real model-backed run succeeds",
    }


def candidate_definition_table(candidates: Sequence[HarnessCandidate]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        config = candidate.config
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "config_hash": config.config_hash,
                "orchestration": config.orchestration_mode,
                "routing": config.routing_mode,
                "planner": config.planner_mode,
                "verification": config.verification_policy,
                "retry": config.retry_policy,
                "fallback": config.fallback_policy,
            }
        )
    return rows


def candidate_difference_audit(candidates: Sequence[HarnessCandidate]) -> dict[str, Any]:
    rows = candidate_definition_table(candidates)
    pairwise = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            differing = [
                key
                for key in ("config_hash", "orchestration", "routing", "planner", "verification", "retry", "fallback")
                if left.get(key) != right.get(key)
            ]
            behavior_keys = [key for key in differing if key != "config_hash"]
            pairwise.append(
                {
                    "left": left["candidate_id"],
                    "right": right["candidate_id"],
                    "config_hash_differs": "config_hash" in differing,
                    "runtime_policy_differences": behavior_keys,
                    "behaviorally_distinct": bool(behavior_keys),
                }
            )
    return {"candidate_count": len(candidates), "rows": rows, "pairwise": pairwise}


def describe_case_selection(cases: Sequence[BenchmarkCase]) -> dict[str, Any]:
    dimensions: dict[str, dict[str, Any]] = {}
    for key in ("relationship_type", "occasion_name", "budget_hint", "formality", "agency_slider"):
        values = [str(case.custom_profile.get(key, "")) for case in cases]
        dimensions[key] = {"unique_count": len(set(values)), "values": sorted(set(values))}
    return {
        "case_ids": [case.case_id for case in cases],
        "case_count": len(cases),
        "dimensions": dimensions,
        "selection_note": "Small stratified UI-permutation subset; not statistically representative of all 12,600 cases.",
    }


def trajectory_analysis(case_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for report in case_reports:
        executions = [HarnessExecutionResult(**item) for item in _as_list(report.get("executions"))]
        case_rows = {}
        for execution in executions:
            trajectory = _trajectory(execution.trace)
            case_rows[execution.candidate_id] = trajectory
            rows.append(
                {
                    "case_id": execution.case_id,
                    "candidate_id": execution.candidate_id,
                    "trajectory": trajectory,
                    "trajectory_text": " -> ".join(_short_stage(stage) for stage in trajectory),
                    "trajectory_length": len(trajectory),
                    "unique_agents_used": len(set(trajectory)),
                    "divergence_trigger": None,
                }
            )
        divergence = _case_divergence(case_rows)
        for row in rows:
            if row["case_id"] == report.get("case_id"):
                row["divergence_trigger"] = divergence
    return {"rows": rows, "changed": any(row["divergence_trigger"] != "none" for row in rows)}


def replication_per_run_rows(executions: Sequence[HarnessExecutionResult]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": execution.case_id,
            "repetition": execution.randomness.get("repetition"),
            "candidate_id": execution.candidate_id,
            "status": execution.final_status,
            "failed_stage": first_failed_stage(execution),
            "latency": execution.timing.get("total_wall_clock_latency_seconds"),
            "quality": execution.evaluation.get("overall_quality") if _quality_comparison_eligible(execution) else None,
        }
        for execution in executions
    ]


def replication_stage_reliability(executions: Sequence[HarnessExecutionResult]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], dict[str, int]] = {}
    for execution in executions:
        seen_stages = set()
        for invocation in _as_list(execution.trace.get("invocations")):
            item = _as_mapping(invocation)
            stage = str(item.get("stage_name"))
            seen_stages.add(stage)
            bucket = totals.setdefault((execution.candidate_id, stage), {"n": 0, "success": 0, "timeout": 0, "failure": 0, "skipped": 0})
            bucket["n"] += 1
            status = str(item.get("status") or "")
            if status == "success":
                bucket["success"] += 1
            elif status == "timeout":
                bucket["timeout"] += 1
            elif status == "skipped":
                bucket["skipped"] += 1
            else:
                bucket["failure"] += 1
    rows = []
    for (candidate_id, stage), counts in sorted(totals.items()):
        n = counts["n"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "stage": stage,
                "n": n,
                "success_rate": _rate(counts["success"], n),
                "timeout_rate": _rate(counts["timeout"], n),
                "failure_rate": _rate(counts["failure"], n),
                "skipped_rate": _rate(counts["skipped"], n),
            }
        )
    return rows


def replication_stage_latency(executions: Sequence[HarnessExecutionResult]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], dict[str, list[float] | int]] = {}
    for execution in executions:
        for invocation in _as_list(execution.trace.get("invocations")):
            item = _as_mapping(invocation)
            key = (execution.candidate_id, str(item.get("stage_name")))
            bucket = values.setdefault(key, {"success_latencies": [], "failed": 0, "timeout": 0})
            if item.get("status") == "success" and item.get("latency_seconds") is not None:
                _as_list(bucket["success_latencies"]).append(float(item["latency_seconds"]))
            elif item.get("status") == "timeout":
                bucket["timeout"] = int(bucket["timeout"]) + 1
            else:
                bucket["failed"] = int(bucket["failed"]) + 1
    rows = []
    for (candidate_id, stage), bucket in sorted(values.items()):
        latencies = _as_list(bucket.get("success_latencies"))
        rows.append(
            {
                "candidate_id": candidate_id,
                "stage": stage,
                "successful_n": len(latencies),
                "failed_n": int(bucket.get("failed") or 0),
                "timeout_n": int(bucket.get("timeout") or 0),
                "min_latency": _min(latencies),
                "median_latency": _median(latencies),
                "mean_latency": _mean(latencies),
                "max_latency": _max(latencies),
            }
        )
    return rows


def replication_aggregate(executions: Sequence[HarnessExecutionResult]) -> dict[str, Any]:
    by_candidate: dict[str, list[HarnessExecutionResult]] = {}
    for execution in executions:
        by_candidate.setdefault(execution.candidate_id, []).append(execution)
    return {
        candidate_id: {
            "n": len(items),
            "success_rate": _rate(sum(1 for item in items if item.final_status == "SUCCESS"), len(items)),
            "timeout_rate": _rate(sum(1 for item in items if item.final_status == "TIMEOUT"), len(items)),
            "harness_failure_rate": _rate(sum(1 for item in items if item.failure_type == "harness"), len(items)),
            "infrastructure_failure_rate": _rate(sum(1 for item in items if item.failure_type == "infrastructure"), len(items)),
            "mean_quality": _mean(item.evaluation.get("overall_quality") for item in items if _quality_comparison_eligible(item)),
            "median_quality": _median(item.evaluation.get("overall_quality") for item in items if _quality_comparison_eligible(item)),
            "std_quality": _std(item.evaluation.get("overall_quality") for item in items if _quality_comparison_eligible(item)),
            "quality_n": sum(1 for item in items if _quality_comparison_eligible(item) and item.evaluation.get("overall_quality") is not None),
            "mean_latency": _mean(item.timing.get("total_wall_clock_latency_seconds") for item in items if item.final_status == "SUCCESS"),
            "median_latency": _median(item.timing.get("total_wall_clock_latency_seconds") for item in items if item.final_status == "SUCCESS"),
            "mean_invocations": _mean(item.resource_usage.get("agent_invocation_count") for item in items),
            "mean_retries": _mean(item.resource_usage.get("retry_count") for item in items),
            "mean_verifications": _mean(item.resource_usage.get("verification_count") for item in items),
            "mean_tool_calls": _mean(item.resource_usage.get("tool_call_count") for item in items),
            "tokens": "unavailable" if all(item.resource_usage.get("run_total_tokens") is None for item in items) else _mean(item.resource_usage.get("run_total_tokens") for item in items),
            "cost": "unknown" if all(item.resource_usage.get("total_cost") is None for item in items) else _mean(item.resource_usage.get("total_cost") for item in items),
        }
        for candidate_id, items in by_candidate.items()
    }


def replication_trajectory_rows(executions: Sequence[HarnessExecutionResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[HarnessExecutionResult]] = {}
    for execution in executions:
        grouped.setdefault((execution.case_id, int(execution.randomness.get("repetition") or 1)), []).append(execution)
    baseline_by_candidate: dict[str, tuple[str, ...]] = {}
    rows = []
    for execution in executions:
        trajectory = tuple(_trajectory(execution.trace))
        candidate_baseline = baseline_by_candidate.setdefault(execution.candidate_id, trajectory)
        paired = grouped.get((execution.case_id, int(execution.randomness.get("repetition") or 1)), [])
        other_trajectories = {item.candidate_id: tuple(_trajectory(item.trace)) for item in paired if item.candidate_id != execution.candidate_id}
        rows.append(
            {
                "case_id": execution.case_id,
                "candidate_id": execution.candidate_id,
                "repetition": execution.randomness.get("repetition"),
                "trajectory": list(trajectory),
                "trajectory_text": " -> ".join(_short_stage(stage) for stage in trajectory),
                "divergence_type": _divergence_type(execution, trajectory, candidate_baseline, other_trajectories),
            }
        )
    return rows


def _divergence_type(
    execution: HarnessExecutionResult,
    trajectory: tuple[str, ...],
    candidate_baseline: tuple[str, ...],
    other_trajectories: Mapping[str, tuple[str, ...]],
) -> str:
    if execution.failure_type == "infrastructure" or execution.final_status == "TIMEOUT":
        return "INFRASTRUCTURE_FAILURE"
    if trajectory != candidate_baseline:
        return "MODEL_VARIANCE"
    if any(other != trajectory for other in other_trajectories.values()):
        return "HARNESS_DECISION"
    return "NO_DIVERGENCE"


def within_harness_variance(executions: Sequence[HarnessExecutionResult]) -> dict[str, Any]:
    by_candidate: dict[str, list[HarnessExecutionResult]] = {}
    for execution in executions:
        by_candidate.setdefault(execution.candidate_id, []).append(execution)
    return {
        candidate_id: {
            "run_success_rate": _rate(sum(1 for item in items if item.final_status == "SUCCESS"), len(items)),
            "final_statuses": sorted({item.final_status for item in items}),
            "trajectory_variants": len({tuple(_trajectory(item.trace)) for item in items}),
            "quality_variance": _std(item.evaluation.get("overall_quality") for item in items if _quality_comparison_eligible(item)),
            "latency_variance": _std(item.timing.get("total_wall_clock_latency_seconds") for item in items if item.final_status == "SUCCESS"),
            "quality_n": sum(1 for item in items if _quality_comparison_eligible(item) and item.evaluation.get("overall_quality") is not None),
        }
        for candidate_id, items in by_candidate.items()
    }


def attribution_summary(executions: Sequence[HarnessExecutionResult]) -> dict[str, Any]:
    trajectories = replication_trajectory_rows(executions)
    harness_divergences = [row for row in trajectories if row["divergence_type"] == "HARNESS_DECISION"]
    model_divergences = [row for row in trajectories if row["divergence_type"] == "MODEL_VARIANCE"]
    infra_divergences = [row for row in trajectories if row["divergence_type"] == "INFRASTRUCTURE_FAILURE"]
    return {
        "harness_induced_divergence_count": len(harness_divergences),
        "model_runtime_divergence_count": len(model_divergences),
        "infrastructure_failure_count": len(infra_divergences),
        "interpretation": "attribute cautiously; repeated identical trajectories with different outcomes indicate model/runtime variance",
    }


def replication_decision_gate(executions: Sequence[HarnessExecutionResult]) -> dict[str, Any]:
    default_runs = [execution for execution in executions if execution.candidate_id == "gmgi_default"]
    if len(default_runs) < 3:
        return {"decision": "NOT_ENOUGH_DATA", "reason": "fewer than 3 default repetitions"}
    success_rate = _rate(sum(1 for item in default_runs if item.final_status == "SUCCESS"), len(default_runs)) or 0.0
    trajectory_variants = len({tuple(_trajectory(item.trace)) for item in default_runs})
    if success_rate < 0.67:
        return {"decision": "BASELINE UNSTABLE", "reason": f"default success_rate={success_rate:.3f}"}
    if trajectory_variants > 1:
        return {"decision": "BASELINE UNSTABLE", "reason": f"default trajectory_variants={trajectory_variants}"}
    dynamic_runs = [execution for execution in executions if execution.candidate_id != "gmgi_default"]
    if not dynamic_runs or any(len([item for item in dynamic_runs if item.candidate_id == candidate]) < len(default_runs) for candidate in {item.candidate_id for item in dynamic_runs}):
        return {"decision": "PRELIMINARY COMPARISON", "reason": "default stable but dynamic repetitions are incomplete"}
    if any(tuple(_trajectory(item.trace)) != tuple(_trajectory(default_runs[0].trace)) for item in dynamic_runs):
        return {"decision": "HARNESS EFFECTS OBSERVABLE", "reason": "repeated runs show trajectory differences across harnesses"}
    return {"decision": "PRELIMINARY COMPARISON", "reason": "baseline stable; no repeated trajectory-level harness effect yet"}


def first_failed_stage(execution: HarnessExecutionResult) -> str | None:
    for invocation in _as_list(execution.trace.get("invocations")):
        item = _as_mapping(invocation)
        if item.get("status") not in {"success", "skipped"}:
            return str(item.get("stage_name"))
    return None


def _case_divergence(trajectories: Mapping[str, Sequence[str]]) -> str:
    unique = {tuple(value) for value in trajectories.values()}
    if len(unique) <= 1:
        return "none"
    return "controller policy changed actual stage trajectory"


def _environment_snapshot(manifest: HarnessExperimentManifest, candidates: Sequence[HarnessCandidate]) -> dict[str, Any]:
    return {
        "provider": manifest.provider or _selected_provider(),
        "model": manifest.model or _selected_model(),
        "model_parameters": manifest.model_parameters or _selected_model_parameters(),
        "evaluation_version": manifest.evaluation_version,
        "state_isolation": manifest.state_isolation,
        "creative_enabled": manifest.include_creative,
        "seed": manifest.seed,
        "stage_timeout_seconds": manifest.stage_timeout_seconds,
        "execution_timeout_seconds": manifest.execution_timeout_seconds,
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
    }


def _selected_provider() -> str:
    import os

    return os.getenv("GMGI_LLM_PROVIDER", "ollama")


def _selected_model() -> str | None:
    import os

    return os.getenv("GMGI_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL")


def _selected_model_parameters() -> dict[str, Any]:
    keys = ("GMGI_OLLAMA_TIMEOUT_SECONDS", "GMGI_OLLAMA_NUM_PREDICT", "GMGI_OLLAMA_NUM_CTX", "GMGI_RECOMMENDATION_MAX_STEPS", "GMGI_GREETING_NUM_PREDICT")
    return {key: os.getenv(key) for key in keys if os.getenv(key) is not None}


def _apply_manifest_environment(manifest_payload: Mapping[str, Any]) -> None:
    provider = manifest_payload.get("provider")
    model = manifest_payload.get("model")
    if provider:
        existing = os.getenv("GMGI_LLM_PROVIDER")
        if existing and existing != provider:
            raise SystemExit(f"Frozen provider mismatch: manifest={provider!r}, environment={existing!r}")
        os.environ["GMGI_LLM_PROVIDER"] = str(provider)
    if model:
        existing = os.getenv("GMGI_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL")
        if existing and existing != model:
            raise SystemExit(f"Frozen model mismatch: manifest={model!r}, environment={existing!r}")
        os.environ["GMGI_OLLAMA_MODEL"] = str(model)
        os.environ["OLLAMA_MODEL"] = str(model)
    for key, value in _as_mapping(manifest_payload.get("model_parameters")).items():
        existing = os.getenv(str(key))
        if existing and existing != str(value):
            raise SystemExit(f"Frozen model parameter mismatch for {key}: manifest={value!r}, environment={existing!r}")
        os.environ[str(key)] = str(value)


def _timing_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    invocations = [_as_mapping(item) for item in _as_list(trace.get("invocations"))]
    tool_latency = sum(float(tool.get("latency_seconds") or 0.0) for invocation in invocations for tool in _as_list(invocation.get("tool_calls")))
    verifier_count = sum(1 for invocation in invocations if _as_mapping(invocation.get("verifier_decision")).get("decision") not in {None, "not_available"})
    return {
        "start_time": trace.get("start_time"),
        "end_time": trace.get("end_time"),
        "total_wall_clock_latency_seconds": trace.get("total_latency_seconds"),
        "sum_invocation_latency_seconds": sum(float(invocation.get("latency_seconds") or 0.0) for invocation in invocations),
        "sum_tool_latency_seconds": tool_latency,
        "model_latency_status": "unavailable",
        "routing_decision_latency_status": "unavailable",
        "planner_latency_status": "captured_as_agent_invocation" if any(invocation.get("stage_name") == "multi_agent_planning" for invocation in invocations) else "not_invoked",
        "verification_latency_status": "not_separately_measured" if verifier_count else "not_invoked",
    }


def _resource_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    invocations = [_as_mapping(item) for item in _as_list(trace.get("invocations"))]
    token_usages = [_as_mapping(invocation.get("token_usage")) for invocation in invocations if invocation.get("token_usage")]
    total_tokens = sum(int(usage.get("total_tokens") or 0) for usage in token_usages) if token_usages else None
    costs = [float(invocation.get("estimated_cost")) for invocation in invocations if invocation.get("estimated_cost") is not None]
    return {
        "agent_invocation_count": len(invocations),
        "tool_call_count": sum(len(_as_list(invocation.get("tool_calls"))) for invocation in invocations),
        "retry_count": sum(int(invocation.get("retry_count") or 0) for invocation in invocations),
        "fallback_count": sum(1 for invocation in invocations if bool(invocation.get("fallback_used"))),
        "verification_count": sum(1 for invocation in invocations if _as_mapping(invocation.get("verifier_decision")).get("decision") not in {None, "not_available"}),
        "routing_decision_count": sum(1 for invocation in invocations if invocation.get("routing_decision")),
        "planner_invocation_count": sum(1 for invocation in invocations if invocation.get("stage_name") == "multi_agent_planning"),
        "failed_invocation_count": sum(1 for invocation in invocations if invocation.get("status") in {"error", "timeout"}),
        "run_prompt_tokens": sum(int(usage.get("prompt_tokens") or 0) for usage in token_usages) if token_usages else None,
        "run_completion_tokens": sum(int(usage.get("completion_tokens") or 0) for usage in token_usages) if token_usages else None,
        "run_total_tokens": total_tokens,
        "token_usage_status": "available" if token_usages else "unavailable",
        "model_cost": sum(costs) if costs else None,
        "tool_cost": None,
        "verifier_cost": None,
        "total_cost": sum(costs) if costs else None,
        "cost_status": "available" if costs else "unknown",
    }


def _failure_categories(trace: Mapping[str, Any], run_error: BaseException | None = None) -> list[str]:
    categories: set[str] = set()
    if run_error is not None:
        if isinstance(run_error, TimeoutError):
            categories.add("experiment_timeout")
        categories.add(f"{_classify_exception(run_error)}_failure")
    for invocation in _as_list(trace.get("invocations")):
        item = _as_mapping(invocation)
        if item.get("status") == "timeout":
            categories.add("agent_timeout")
            categories.add("provider_timeout")
        if item.get("status") == "error":
            categories.add("model_error")
        verifier = _as_mapping(item.get("verifier_decision"))
        if str(verifier.get("decision") or "").startswith("fail"):
            categories.add("verification_failure")
            if any("budget" in str(issue).lower() for issue in _as_list(_as_mapping(verifier.get("details")).get("issues"))):
                categories.add("constraint_failure")
        if item.get("fallback_used"):
            categories.add("fallback_failure")
    timed_out_stages = {
        str(_as_mapping(invocation).get("stage_name"))
        for invocation in _as_list(trace.get("invocations"))
        if _as_mapping(invocation).get("status") == "timeout"
    }
    late_success = {
        str(_as_mapping(invocation).get("stage_name"))
        for invocation in _as_list(trace.get("invocations"))
        if _as_mapping(invocation).get("status") == "success"
    }
    if timed_out_stages & late_success:
        categories.add("late_completion_after_timeout")
    return sorted(categories)


def _reliability_score(execution: HarnessExecutionResult) -> float:
    resource = execution.resource_usage
    penalties = int(resource.get("failed_invocation_count") or 0) + int(resource.get("fallback_count") or 0)
    penalties += len([category for category in execution.failure_categories if category in {"verification_failure", "constraint_failure", "timeout"}])
    return max(0.0, 1.0 - 0.1 * penalties)


def _pareto_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible_rows = [
        row
        for row in rows
        if row.get("quality") is not None and row.get("reliability") is not None and row.get("latency") is not None
    ]
    dominated: list[str] = []
    for row in eligible_rows:
        candidate = str(row["candidate_id"])
        quality = float(row["quality"])
        reliability = float(row["reliability"])
        latency = float(row["latency"])
        for other in eligible_rows:
            if other is row:
                continue
            other_quality = float(other["quality"])
            other_reliability = float(other["reliability"])
            other_latency = float(other["latency"])
            no_worse = other_quality >= quality and other_reliability >= reliability and other_latency <= latency
            strictly_better = other_quality > quality or other_reliability > reliability or other_latency < latency
            if no_worse and strictly_better:
                dominated.append(candidate)
                break
    nondominated = sorted({str(row["candidate_id"]) for row in eligible_rows} - set(dominated))
    return {
        "dominated_candidates": sorted(set(dominated)),
        "non_dominated_candidates": nondominated,
        "excluded_rows": len(rows) - len(eligible_rows),
        "note": "Pareto uses only rows with available quality, reliability, and latency; missing values are excluded, not coerced to zero.",
    }


def _trajectory_differences(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trajectories = {str(row["candidate_id"]): row.get("trajectory") for row in rows}
    unique = {json.dumps(value, sort_keys=True) for value in trajectories.values()}
    return {"changed": len(unique) > 1, "trajectories": trajectories}


def _trajectory(trace: Mapping[str, Any]) -> list[str]:
    return [str(_as_mapping(item).get("stage_name")) for item in _as_list(trace.get("invocations"))]


def _metric_score(stage_reports: Mapping[str, Any], stage: str, metric_name: str) -> Any:
    for metric in _as_list(_as_mapping(stage_reports.get(stage)).get("metrics")):
        item = _as_mapping(metric)
        if item.get("name") == metric_name:
            return item.get("score")
    return None


def _aggregate(case_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for report in case_reports for row in _as_mapping(report.get("comparison")).get("table", [])]
    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
    return {
        candidate: {
            "n": len(items),
            "successful_runs": sum(1 for item in items if item.get("status") == "SUCCESS"),
            "partial_runs": sum(1 for item in items if item.get("status") == "PARTIAL_SUCCESS"),
            "failed_runs": sum(1 for item in items if item.get("status") in {"HARNESS_FAILURE", "INFRASTRUCTURE_FAILURE", "EVALUATION_FAILURE"}),
            "timeout_runs": sum(1 for item in items if item.get("status") == "TIMEOUT"),
            "success_rate": _rate(sum(1 for item in items if item.get("status") == "SUCCESS"), len(items)),
            "timeout_rate": _rate(sum(1 for item in items if item.get("status") == "TIMEOUT"), len(items)),
            "infrastructure_failure_rate": _rate(sum(1 for item in items if item.get("failure_type") == "infrastructure"), len(items)),
            "harness_failure_rate": _rate(sum(1 for item in items if item.get("failure_type") == "harness"), len(items)),
            "retry_rate": _rate(sum(1 for item in items if int(item.get("retries") or 0) > 0), len(items)),
            "fallback_rate": _rate(sum(1 for item in items if int(item.get("fallbacks") or 0) > 0), len(items)),
            "mean_quality": _mean(item.get("quality") for item in items),
            "median_quality": _median(item.get("quality") for item in items),
            "std_quality": _std(item.get("quality") for item in items),
            "min_quality": _min(item.get("quality") for item in items),
            "max_quality": _max(item.get("quality") for item in items),
            "mean_reliability": _mean(item.get("reliability") for item in items),
            "mean_latency": _mean(item.get("latency") for item in items),
            "median_latency": _median(item.get("latency") for item in items),
            "std_latency": _std(item.get("latency") for item in items),
            "mean_invocations": _mean(item.get("invocations") for item in items),
            "mean_retries": _mean(item.get("retries") for item in items),
            "mean_verifications": _mean(item.get("verifications") for item in items),
            "total_retries": sum(int(item.get("retries") or 0) for item in items),
            "token_usage_status": sorted({str(item.get("token_usage_status")) for item in items}),
            "cost_status": sorted({str(item.get("cost_status")) for item in items}),
        }
        for candidate, items in by_candidate.items()
    }


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _mean(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _median(values: Sequence[Any]) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    mid = len(numeric) // 2
    if len(numeric) % 2:
        return numeric[mid]
    return (numeric[mid - 1] + numeric[mid]) / 2.0


def _std(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) < 2:
        return 0.0 if numeric else None
    mean = sum(numeric) / len(numeric)
    return (sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1)) ** 0.5


def _min(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return min(numeric) if numeric else None


def _max(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return max(numeric) if numeric else None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = ["case_id", "candidate_id", "repetition", "status", "failure_type", "quality", "constraint", "reliability", "latency", "tokens", "cost", "invocations", "retries", "fallbacks", "verifications", "failures"]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(json.dumps(row.get(field, ""), ensure_ascii=False) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase6_report(path: str | Path, summary: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6 Harness Benchmark Report",
        "",
        "## Experiment Configuration",
        _markdown_kv(_as_mapping(summary.get("environment"))),
        "",
        "## Candidate Definitions",
        _markdown_table(
            _as_list(summary.get("candidate_definitions")),
            ["candidate_id", "config_hash", "orchestration", "routing", "planner", "verification", "retry", "fallback"],
        ),
        "",
        "## Case Selection",
        _markdown_kv(_as_mapping(summary.get("case_selection"))),
        "",
        "## Execution Results",
        _markdown_table(_comparison_rows(summary), ["case_id", "candidate_id", "status", "quality", "reliability", "latency", "tokens", "cost", "invocations", "retries", "verifications"]),
        "",
        "## Aggregate Results",
        _markdown_table(_aggregate_rows(summary), ["candidate_id", "n", "success_rate", "mean_quality", "mean_latency", "mean_invocations", "mean_retries", "mean_verifications"]),
        "",
        "## Trajectory Comparison",
        _markdown_table(_as_list(_as_mapping(summary.get("trajectory_analysis")).get("rows")), ["case_id", "candidate_id", "trajectory_text", "divergence_trigger"]),
        "",
        "## Failure Analysis",
        _markdown_table(_failure_rows(summary), ["case_id", "candidate_id", "status", "failure_type", "termination_reason", "failure_categories"]),
        "",
        "## Pareto Analysis",
        _markdown_kv(_combined_pareto(summary)),
        "",
        "## Limitations",
        "- Results are descriptive unless enough repetitions are present.",
        "- Missing token and cost values remain unavailable/unknown; they are not estimated.",
        "- Infrastructure failures are not quality failures.",
        "- Creative/image generation is excluded when `include_creative=false`.",
        "",
        "## Conclusions",
        "This report provides empirical traces and descriptive comparisons only. It does not declare a universal best harness and does not implement automatic harness selection.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase65_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6.5 Harness Replication Report",
        "",
        "## Experimental Configuration",
        _markdown_kv(_as_mapping(report.get("environment"))),
        "",
        "## Default Harness Replications",
        _markdown_table([row for row in _as_list(report.get("per_run")) if _as_mapping(row).get("candidate_id") == "gmgi_default"], ["case_id", "repetition", "candidate_id", "status", "failed_stage", "latency", "quality"]),
        "",
        "## Dynamic Harness Replications",
        _markdown_table([row for row in _as_list(report.get("per_run")) if _as_mapping(row).get("candidate_id") != "gmgi_default"], ["case_id", "repetition", "candidate_id", "status", "failed_stage", "latency", "quality"]),
        "",
        "## Stage Reliability",
        _markdown_table(_as_list(report.get("stage_reliability")), ["candidate_id", "stage", "success_rate", "timeout_rate", "failure_rate"]),
        "",
        "## Stage Latency",
        _markdown_table(_as_list(report.get("stage_latency")), ["candidate_id", "stage", "successful_n", "failed_n", "timeout_n", "min_latency", "median_latency", "mean_latency", "max_latency"]),
        "",
        "## Trajectory Comparison",
        _markdown_table(_as_list(report.get("trajectory")), ["case_id", "candidate_id", "repetition", "trajectory_text", "divergence_type"]),
        "",
        "## Harness-Induced Divergence",
        _markdown_kv(_as_mapping(report.get("harness_vs_runtime_attribution"))),
        "",
        "## Runtime-Induced Divergence",
        _markdown_kv(_as_mapping(report.get("within_harness_variance"))),
        "",
        "## Quality",
        _markdown_table(_aggregate_rows({"aggregate": report.get("aggregate", {})}), ["candidate_id", "n", "success_rate", "mean_quality", "mean_latency", "mean_invocations", "mean_retries", "mean_verifications"]),
        "",
        "## Reliability",
        _markdown_table(_replication_aggregate_rows(report), ["candidate_id", "n", "success_rate", "timeout_rate", "harness_failure_rate", "infrastructure_failure_rate"]),
        "",
        "## Efficiency",
        _markdown_table(_replication_aggregate_rows(report), ["candidate_id", "mean_latency", "mean_invocations", "mean_retries", "mean_verifications", "mean_tool_calls", "tokens", "cost"]),
        "",
        "## Limitations",
        "- Small-N replication is descriptive only.",
        "- Missing quality values are N/A, not zero.",
        "- Token and cost values remain unavailable unless provider traces expose them.",
        "- Creative/image generation is excluded when `include_creative=false`.",
        "",
        "## Meta-Harness Readiness",
        _markdown_kv({"decision_gate": _as_mapping(report.get("decision_gate")).get("decision"), "meta_harness_readiness": report.get("meta_harness_readiness")}),
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _replication_aggregate_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate_id, values in _as_mapping(report.get("aggregate")).items():
        item = _as_mapping(values)
        row = {"candidate_id": candidate_id}
        row.update(item)
        rows.append({key: _display(value) for key, value in row.items()})
    return rows


def _comparison_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for report in _as_list(summary.get("cases")):
        rows.extend(_as_list(_as_mapping(_as_mapping(report).get("comparison")).get("table")))
    return [{key: _display(value) for key, value in row.items()} for row in rows]


def _aggregate_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate_id, values in _as_mapping(summary.get("aggregate")).items():
        item = _as_mapping(values)
        rows.append(
            {
                "candidate_id": candidate_id,
                "n": item.get("n"),
                "success_rate": item.get("success_rate"),
                "mean_quality": item.get("mean_quality"),
                "mean_latency": item.get("mean_latency"),
                "mean_invocations": item.get("mean_invocations"),
                "mean_retries": item.get("mean_retries"),
                "mean_verifications": item.get("mean_verifications"),
            }
        )
    return [{key: _display(value) for key, value in row.items()} for row in rows]


def _failure_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for report in _as_list(summary.get("cases")):
        for execution in _as_list(_as_mapping(report).get("executions")):
            item = _as_mapping(execution)
            if item.get("final_status") == "SUCCESS":
                continue
            rows.append(
                {
                    "case_id": item.get("case_id"),
                    "candidate_id": item.get("candidate_id"),
                    "status": item.get("final_status"),
                    "failure_type": item.get("failure_type"),
                    "termination_reason": item.get("termination_reason"),
                    "failure_categories": ", ".join(str(value) for value in _as_list(item.get("failure_categories"))),
                }
            )
    return rows


def _combined_pareto(summary: Mapping[str, Any]) -> dict[str, Any]:
    dominated: set[str] = set()
    nondominated: set[str] = set()
    excluded = 0
    for report in _as_list(summary.get("cases")):
        pareto = _as_mapping(_as_mapping(_as_mapping(report).get("comparison")).get("pareto"))
        dominated.update(str(item) for item in _as_list(pareto.get("dominated_candidates")))
        nondominated.update(str(item) for item in _as_list(pareto.get("non_dominated_candidates")))
        excluded += int(pareto.get("excluded_rows") or 0)
    return {
        "non_dominated_candidates": sorted(nondominated),
        "dominated_candidates": sorted(dominated),
        "excluded_rows": excluded,
        "note": "Computed per case over available quality/reliability/latency only.",
    }


def _markdown_kv(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "N/A"
    return "\n".join(f"- `{key}`: {_display(value)}" for key, value in payload.items())


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "N/A"
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(_escape_markdown(_display(_as_mapping(row).get(field))) for field in fields) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _short_stage(stage: str) -> str:
    aliases = {
        "recipient_profiling": "recipient",
        "relationship_analysis": "relationship",
        "gift_intent_reasoning": "intent",
        "multi_agent_planning": "planning",
        "recommendation": "recommendation",
        "creative_generation": "creative",
        "greeting_story": "greeting",
        "delivery_planner": "delivery",
    }
    return aliases.get(stage, stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GMGI Phase 4 harness comparison experiment.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--case-file", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-dir", default="experiments/evals/harness_comparison")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidates", nargs="*", default=["gmgi_default", "gmgi_dynamic_v1", "gmgi_dynamic_verified_v1"])
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-creative", action="store_true")
    parser.add_argument("--randomized-order", action="store_true")
    parser.add_argument("--stage-timeout", type=float, default=None)
    parser.add_argument("--execution-timeout", type=float, default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--replication", action="store_true")
    args = parser.parse_args()
    manifest_payload = json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest else {}
    _apply_manifest_environment(manifest_payload)
    if args.replication:
        if not manifest_payload:
            raise SystemExit("--replication requires --manifest")
        manifest = HarnessReplicationManifest(**manifest_payload)
        report = run_replication_experiment(manifest, output_dir=args.output_dir)
        print(json.dumps(report["decision_gate"], indent=2, default=str))
        return
    registry = default_harness_registry()
    candidate_ids = manifest_payload.get("candidates") or ([args.candidate] if args.candidate else args.candidates)
    candidates = [registry.get(candidate_id) for candidate_id in candidate_ids]
    case_file = manifest_payload.get("case_file") or args.case_file
    cases = load_cases(case_file)
    case_ids = set(manifest_payload.get("cases") or [])
    if args.case_id:
        case_ids.add(args.case_id)
    if case_ids:
        cases = [case for case in cases if case.case_id in case_ids]
    if args.health_check:
        report = benchmark_health_check(
            candidate=candidates[0],
            case=cases[0] if cases else None,
            output_dir=args.output_dir,
            stage_timeout_seconds=manifest_payload.get("stage_timeout_seconds", args.stage_timeout),
            execution_timeout_seconds=manifest_payload.get("execution_timeout_seconds", args.execution_timeout),
        )
        print(json.dumps(report, indent=2, default=str))
        return
    if args.diagnostic:
        if not cases:
            raise SystemExit("No case selected for diagnostic run.")
        result = execute_candidate_on_case(
            cases[0],
            candidate=candidates[0],
            experiment_id=args.experiment_id or "diagnostic-single-case",
            output_dir=Path(args.output_dir) / "diagnostic" / cases[0].case_id / candidates[0].candidate_id,
            include_creative=bool(manifest_payload.get("include_creative", args.include_creative)),
            agency_slider=float(manifest_payload.get("agency_slider", 0.5)),
            seed=int(manifest_payload.get("seed", args.seed)),
            stage_timeout_seconds=manifest_payload.get("stage_timeout_seconds", args.stage_timeout),
            execution_timeout_seconds=manifest_payload.get("execution_timeout_seconds", args.execution_timeout),
            resume=not args.no_resume,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
        return
    report = run_harness_experiment(
        cases=cases,
        candidates=candidates,
        experiment_id=manifest_payload.get("experiment_id") or args.experiment_id,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=int(manifest_payload.get("seed", args.seed)),
        include_creative=bool(manifest_payload.get("include_creative", args.include_creative)),
        agency_slider=float(manifest_payload.get("agency_slider", 0.5)),
        stage_timeout_seconds=manifest_payload.get("stage_timeout_seconds", args.stage_timeout),
        execution_timeout_seconds=manifest_payload.get("execution_timeout_seconds", args.execution_timeout),
        repetitions=int(manifest_payload.get("repetitions", args.repetitions)),
        randomized_order=bool(manifest_payload.get("randomized_order", args.randomized_order)),
        resume=bool(manifest_payload.get("resume", not args.no_resume)),
    )
    if manifest_payload.get("report_path"):
        write_phase6_report(manifest_payload["report_path"], report)
    print(json.dumps(report["aggregate"], indent=2, default=str))


if __name__ == "__main__":
    main()
