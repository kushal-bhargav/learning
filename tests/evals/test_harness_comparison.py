from __future__ import annotations

import shutil
from pathlib import Path
import time

from src.evals.benchmark import BenchmarkCase
from src.evals import harness_comparison
from src.harness import default_harness_registry


def test_run_case_across_harnesses_persists_candidate_attributed_results(monkeypatch) -> None:
    output_dir = Path("experiments/tmp-tests/harness-comparison-case")
    shutil.rmtree(output_dir, ignore_errors=True)
    def fake_run_case(case, *, output_dir, include_creative, agency_slider, seed, stage_timeout_seconds, harness_config):
        dynamic = harness_config.routing_mode == "dynamic"
        invocations = [
            {
                "stage_name": "recipient_profiling",
                "status": "success",
                "latency_seconds": 0.1,
                "retry_count": 0,
                "fallback_used": False,
                "tool_calls": [],
                "routing_decision": {"decision": "recipient_profiling"},
                "verifier_decision": {"decision": "not_available", "policy": harness_config.verification_policy},
            },
            {
                "stage_name": "recommendation" if not dynamic else "creative_generation",
                "status": "success",
                "latency_seconds": 0.2,
                "retry_count": 1 if harness_config.retry_policy == "controller_retry_once" else 0,
                "fallback_used": False,
                "tool_calls": [{"tool_name": "query_memory_graph", "latency_seconds": 0.01}],
                "routing_decision": {"decision": "next"},
                "verifier_decision": {"decision": "pass" if harness_config.verification_policy == "deterministic_constraints" else "not_available", "policy": harness_config.verification_policy},
            },
        ]
        return {
            "case_id": case.case_id,
            "run_id": f"run-{harness_config.harness_id}",
            "harness_id": harness_config.harness_id,
            "harness_version": harness_config.harness_version,
            "harness_config_hash": harness_config.config_hash,
            "overall_score": 0.8 if dynamic else 0.7,
            "stage_reports": {"gift_intent_reasoning": {"metrics": [{"name": "budget_constraint_preserved", "score": 1.0}]}},
            "behavioral_metrics": None,
            "run_trace": {
                "run_id": f"run-{harness_config.harness_id}",
                "final_status": "success",
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-01-01T00:00:01Z",
                "total_latency_seconds": 0.3,
                "invocations": invocations,
            },
        }

    monkeypatch.setattr(harness_comparison, "run_case", fake_run_case)
    registry = default_harness_registry()
    case = BenchmarkCase(case_id="same-ui-case", custom_profile={"recipient_name": "Mira"}, expected={})
    report = harness_comparison.run_case_across_harnesses(
        case,
        candidates=[registry.get("gmgi_default"), registry.get("gmgi_dynamic_verified_v1")],
        experiment_id="phase4-test",
        output_dir=output_dir,
        seed=42,
    )

    assert report["state_isolation"] == "isolated"
    assert report["comparison"]["trajectory_differences"]["changed"] is True
    assert report["executions"][0]["trace"]["candidate_id"] == "gmgi_default"
    assert report["executions"][1]["trace"]["candidate_id"] == "gmgi_dynamic_verified_v1"
    assert report["comparison"]["table"][1]["quality"] == 0.8
    assert report["comparison"]["table"][1]["token_usage_status"] == "unavailable"
    assert report["comparison"]["table"][1]["cost_status"] == "unknown"
    assert (output_dir / "same-ui-case_comparison.json").exists()


def test_run_harness_experiment_writes_manifest_and_rows(monkeypatch) -> None:
    output_dir = Path("experiments/tmp-tests/harness-comparison-manifest")
    shutil.rmtree(output_dir, ignore_errors=True)
    def fake_run_case_across_harnesses(case, **kwargs):
        return {
            "experiment_id": kwargs["experiment_id"],
            "case_id": case.case_id,
            "comparison": {
                "table": [
                    {
                        "case_id": case.case_id,
                        "candidate_id": "gmgi_default",
                        "quality": 0.7,
                        "reliability": 1.0,
                        "latency": 0.2,
                        "invocations": 2,
                        "retries": 0,
                        "token_usage_status": "unavailable",
                        "cost_status": "unknown",
                    }
                ]
            },
        }

    monkeypatch.setattr(harness_comparison, "run_case_across_harnesses", fake_run_case_across_harnesses)
    registry = default_harness_registry()
    case = BenchmarkCase(case_id="case-one", custom_profile={}, expected={})
    report = harness_comparison.run_harness_experiment(
        cases=[case],
        candidates=[registry.get("gmgi_default")],
        experiment_id="manifest-test",
        output_dir=output_dir,
        limit=1,
        seed=123,
    )

    assert report["manifest"]["experiment_id"] == "manifest-test"
    assert report["aggregate"]["gmgi_default"]["mean_quality"] == 0.7
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "harness_comparison_rows.csv").exists()


def test_candidate_execution_timeout_is_infrastructure_timeout(monkeypatch) -> None:
    def slow_run_case(*_args, **_kwargs):
        time.sleep(1.0)
        return {}

    monkeypatch.setattr(harness_comparison, "run_case", slow_run_case)
    registry = default_harness_registry()
    case = BenchmarkCase(case_id="timeout-case", custom_profile={}, expected={})

    result = harness_comparison.execute_candidate_on_case(
        case,
        candidate=registry.get("gmgi_default"),
        experiment_id="timeout-test",
        output_dir="experiments/tmp-tests/harness-timeout",
        include_creative=False,
        agency_slider=0.5,
        seed=1,
        stage_timeout_seconds=1,
        execution_timeout_seconds=0.01,
        resume=False,
    )

    assert result.final_status == "TIMEOUT"
    assert result.failure_type == "infrastructure"
    assert "experiment_timeout" in result.failure_categories
    assert result.trace["termination_reason"] == "experiment_timeout"


def test_quality_is_unavailable_for_timeout_comparison_rows() -> None:
    registry = default_harness_registry()
    candidate = registry.get("gmgi_default")
    execution = harness_comparison.HarnessExecutionResult(
        experiment_id="eligibility",
        case_id="case",
        candidate_id=candidate.candidate_id,
        candidate_identity=candidate.identity,
        candidate=candidate.to_dict(),
        harness_config=candidate.config.to_dict(),
        run_id="run-timeout",
        trace={"invocations": []},
        evaluation={"overall_quality": 0.9},
        timing={},
        resource_usage={},
        randomness={},
        final_status="TIMEOUT",
        termination_reason="agent_timeout",
        failure_type="infrastructure",
        failure_categories=["agent_timeout"],
    )

    table = harness_comparison.compare_execution_results([execution])["table"]

    assert table[0]["quality_comparison_eligible"] is False
    assert table[0]["quality"] is None


def test_health_check_reports_provider_unavailable(monkeypatch) -> None:
    def fail_provider():
        return {"provider": "ollama", "reachable": False, "error": "connection refused"}

    monkeypatch.setattr(harness_comparison, "_provider_health", fail_provider)
    report = harness_comparison.benchmark_health_check(output_dir="experiments/tmp-tests/harness-health")

    assert report["provider"]["reachable"] is False
    assert report["single_case_smoke"] is None
    assert report["harness_comparison_allowed"] is False


def test_repetitions_are_aggregated(monkeypatch) -> None:
    def fake_run_case(case, *, output_dir, include_creative, agency_slider, seed, stage_timeout_seconds, harness_config):
        return {
            "case_id": case.case_id,
            "run_id": f"run-{seed}",
            "overall_score": float(seed),
            "stage_reports": {},
            "run_trace": {
                "run_id": f"run-{seed}",
                "final_status": "success",
                "termination_reason": "completed",
                "total_latency_seconds": float(seed),
                "invocations": [],
            },
        }

    monkeypatch.setattr(harness_comparison, "run_case", fake_run_case)
    registry = default_harness_registry()
    case = BenchmarkCase(case_id="rep-case", custom_profile={}, expected={})
    report = harness_comparison.run_harness_experiment(
        cases=[case],
        candidates=[registry.get("gmgi_default")],
        experiment_id="rep-test",
        output_dir="experiments/tmp-tests/harness-reps",
        seed=1,
        repetitions=3,
        resume=False,
    )

    aggregate = report["aggregate"]["gmgi_default"]
    assert aggregate["n"] == 3
    assert aggregate["successful_runs"] == 3
    assert aggregate["mean_quality"] == 2.0
    assert aggregate["median_latency"] == 2.0


def test_resume_reuses_existing_execution_result(monkeypatch) -> None:
    shutil.rmtree("experiments/tmp-tests/harness-resume", ignore_errors=True)
    calls = {"count": 0}

    def fake_run_case(case, *, output_dir, include_creative, agency_slider, seed, stage_timeout_seconds, harness_config):
        calls["count"] += 1
        return {
            "case_id": case.case_id,
            "run_id": "run-resume",
            "overall_score": 0.9,
            "stage_reports": {},
            "run_trace": {"run_id": "run-resume", "final_status": "success", "termination_reason": "completed", "total_latency_seconds": 0.1, "invocations": []},
        }

    monkeypatch.setattr(harness_comparison, "run_case", fake_run_case)
    registry = default_harness_registry()
    case = BenchmarkCase(case_id="resume-case", custom_profile={}, expected={})
    kwargs = {
        "case": case,
        "candidate": registry.get("gmgi_default"),
        "experiment_id": "resume-test",
        "output_dir": "experiments/tmp-tests/harness-resume",
        "include_creative": False,
        "agency_slider": 0.5,
        "seed": 1,
        "stage_timeout_seconds": 1,
        "resume": True,
    }
    first = harness_comparison.execute_candidate_on_case(**kwargs)
    second = harness_comparison.execute_candidate_on_case(**kwargs)

    assert first.run_id == second.run_id
    assert calls["count"] == 1


def test_candidate_reproducibility_identity_is_stable() -> None:
    registry = default_harness_registry()
    first = registry.get("gmgi_default")
    second = registry.get("gmgi_default")

    assert first.config.config_hash == second.config.config_hash
    assert first.identity == second.identity


def test_candidate_difference_audit_requires_runtime_policy_difference() -> None:
    registry = default_harness_registry()
    audit = harness_comparison.candidate_difference_audit(
        [
            registry.get("gmgi_default"),
            registry.get("gmgi_dynamic_v1"),
            registry.get("gmgi_dynamic_verified_v1"),
        ]
    )

    pairs = audit["pairwise"]
    assert all(pair["config_hash_differs"] for pair in pairs)
    assert any("routing" in pair["runtime_policy_differences"] for pair in pairs)
    assert any("verification" in pair["runtime_policy_differences"] for pair in pairs)


def test_case_selection_describes_stratification_dimensions() -> None:
    cases = [
        BenchmarkCase(case_id="a", custom_profile={"relationship_type": "partner", "occasion_name": "Birthday", "budget_hint": "Flexible", "formality": "casual", "agency_slider": 0.15}, expected={}),
        BenchmarkCase(case_id="b", custom_profile={"relationship_type": "sibling", "occasion_name": "Promotion", "budget_hint": "USD 25-45", "formality": "professional", "agency_slider": 0.5}, expected={}),
    ]

    summary = harness_comparison.describe_case_selection(cases)

    assert summary["case_ids"] == ["a", "b"]
    assert summary["dimensions"]["relationship_type"]["unique_count"] == 2
    assert "partner" in summary["dimensions"]["relationship_type"]["values"]


def test_trajectory_analysis_extracts_divergence() -> None:
    registry = default_harness_registry()
    default = registry.get("gmgi_default")
    dynamic = registry.get("gmgi_dynamic_v1")
    executions = []
    for candidate, stages in ((default, ["recipient_profiling", "recommendation"]), (dynamic, ["recipient_profiling", "relationship_analysis", "recommendation"])):
        executions.append(
            harness_comparison.HarnessExecutionResult(
                experiment_id="traj",
                case_id="case",
                candidate_id=candidate.candidate_id,
                candidate_identity=candidate.identity,
                candidate=candidate.to_dict(),
                harness_config=candidate.config.to_dict(),
                run_id=f"run-{candidate.candidate_id}",
                trace={"invocations": [{"stage_name": stage, "status": "success"} for stage in stages]},
                evaluation={"overall_quality": 0.8},
                timing={"total_wall_clock_latency_seconds": 1.0},
                resource_usage={},
                randomness={},
                final_status="SUCCESS",
            )
        )

    analysis = harness_comparison.trajectory_analysis([{"case_id": "case", "executions": [item.model_dump(mode="json") for item in executions]}])

    assert analysis["changed"] is True
    assert all(row["divergence_trigger"] != "none" for row in analysis["rows"])


def test_pareto_excludes_missing_quality_instead_of_zero() -> None:
    rows = [
        {"candidate_id": "a", "quality": None, "reliability": 1.0, "latency": 1.0},
        {"candidate_id": "b", "quality": 0.8, "reliability": 1.0, "latency": 2.0},
        {"candidate_id": "c", "quality": 0.9, "reliability": 1.0, "latency": 1.0},
    ]

    pareto = harness_comparison._pareto_analysis(rows)

    assert pareto["excluded_rows"] == 1
    assert "b" in pareto["dominated_candidates"]
    assert "c" in pareto["non_dominated_candidates"]


def test_phase6_report_writer_uses_unavailable_not_zero() -> None:
    summary = {
        "environment": {"model": "llama3.2:latest", "provider": "ollama"},
        "candidate_definitions": [{"candidate_id": "gmgi_default", "config_hash": "abc", "orchestration": "fixed_stage", "routing": "static", "planner": "advisory", "verification": "schema", "retry": "local", "fallback": "local"}],
        "case_selection": {"case_ids": ["case"], "case_count": 1},
        "cases": [
            {
                "comparison": {
                    "table": [
                        {
                            "case_id": "case",
                            "candidate_id": "gmgi_default",
                            "status": "TIMEOUT",
                            "quality": None,
                            "reliability": 0.8,
                            "latency": 10.0,
                            "tokens": None,
                            "cost": None,
                            "invocations": 3,
                            "retries": 0,
                            "verifications": 0,
                        }
                    ],
                    "pareto": {"dominated_candidates": [], "non_dominated_candidates": [], "excluded_rows": 1},
                },
                "executions": [],
            }
        ],
        "aggregate": {"gmgi_default": {"n": 1, "success_rate": 0.0, "mean_quality": None, "mean_latency": 10.0, "mean_invocations": 3.0, "mean_retries": 0.0, "mean_verifications": 0.0}},
        "trajectory_analysis": {"rows": [{"case_id": "case", "candidate_id": "gmgi_default", "trajectory_text": "recipient", "divergence_trigger": "none"}]},
    }

    output = Path("experiments/tmp-tests/phase6-report/report.md")
    shutil.rmtree(output.parent, ignore_errors=True)
    harness_comparison.write_phase6_report(output, summary)

    text = output.read_text(encoding="utf-8")
    assert "N/A" in text
    assert "| case | gmgi_default | TIMEOUT | N/A |" in text


def _execution(candidate_id: str, *, repetition: int, stages: list[tuple[str, str]], status: str = "SUCCESS", quality: float | None = 0.8) -> harness_comparison.HarnessExecutionResult:
    registry = default_harness_registry()
    candidate = registry.get(candidate_id)
    return harness_comparison.HarnessExecutionResult(
        experiment_id="replication",
        case_id="case",
        candidate_id=candidate.candidate_id,
        candidate_identity=candidate.identity,
        candidate=candidate.to_dict(),
        harness_config=candidate.config.to_dict(),
        run_id=f"run-{candidate_id}-{repetition}",
        trace={
            "invocations": [
                {
                    "stage_name": stage,
                    "status": stage_status,
                    "latency_seconds": float(index + 1),
                    "tool_calls": [],
                    "routing_decision": {"decision": stage},
                    "verifier_decision": {"decision": "not_available"},
                    "retry_count": 0,
                    "fallback_used": False,
                }
                for index, (stage, stage_status) in enumerate(stages)
            ]
        },
        evaluation={"overall_quality": quality},
        timing={"total_wall_clock_latency_seconds": 10.0 + repetition},
        resource_usage={"agent_invocation_count": len(stages), "retry_count": 0, "verification_count": 0, "tool_call_count": 0},
        randomness={"repetition": repetition},
        final_status=status,
        failure_type="none" if status == "SUCCESS" else "harness",
    )


def test_replication_execution_order_is_interleaved() -> None:
    registry = default_harness_registry()
    order = harness_comparison.replication_execution_order(
        [registry.get("gmgi_default"), registry.get("gmgi_dynamic_v1")],
        repetitions=2,
        mode="interleaved",
    )

    assert [(rep, candidate.candidate_id) for rep, candidate in order] == [
        (1, "gmgi_default"),
        (1, "gmgi_dynamic_v1"),
        (2, "gmgi_default"),
        (2, "gmgi_dynamic_v1"),
    ]


def test_replication_stage_reliability_rates() -> None:
    executions = [
        _execution("gmgi_default", repetition=1, stages=[("recommendation", "success")]),
        _execution("gmgi_default", repetition=2, stages=[("recommendation", "timeout")], status="TIMEOUT", quality=None),
        _execution("gmgi_default", repetition=3, stages=[("recommendation", "error")], status="PARTIAL_SUCCESS", quality=None),
    ]

    rows = harness_comparison.replication_stage_reliability(executions)
    recommendation = next(row for row in rows if row["stage"] == "recommendation")

    assert recommendation["success_rate"] == 1 / 3
    assert recommendation["timeout_rate"] == 1 / 3
    assert recommendation["failure_rate"] == 1 / 3


def test_replication_aggregate_keeps_missing_quality_unavailable() -> None:
    executions = [
        _execution("gmgi_default", repetition=1, stages=[("recipient_profiling", "success")], quality=0.9),
        _execution("gmgi_default", repetition=2, stages=[("recipient_profiling", "timeout")], status="TIMEOUT", quality=None),
    ]

    aggregate = harness_comparison.replication_aggregate(executions)["gmgi_default"]

    assert aggregate["quality_n"] == 1
    assert aggregate["mean_quality"] == 0.9
    assert aggregate["success_rate"] == 0.5


def test_replication_divergence_distinguishes_harness_and_runtime() -> None:
    executions = [
        _execution("gmgi_default", repetition=1, stages=[("recipient_profiling", "success"), ("recommendation", "success")]),
        _execution("gmgi_default", repetition=2, stages=[("recipient_profiling", "success"), ("recommendation", "success"), ("greeting_story", "success")]),
        _execution("gmgi_dynamic_v1", repetition=1, stages=[("recipient_profiling", "success"), ("recommendation", "success"), ("greeting_story", "success")]),
    ]

    rows = harness_comparison.replication_trajectory_rows(executions)

    default_second = next(row for row in rows if row["candidate_id"] == "gmgi_default" and row["repetition"] == 2)
    dynamic_first = next(row for row in rows if row["candidate_id"] == "gmgi_dynamic_v1")
    assert default_second["divergence_type"] == "MODEL_VARIANCE"
    assert dynamic_first["divergence_type"] == "HARNESS_DECISION"


def test_replication_decision_gate_detects_unstable_default() -> None:
    executions = [
        _execution("gmgi_default", repetition=1, stages=[("recommendation", "success")]),
        _execution("gmgi_default", repetition=2, stages=[("recommendation", "error")], status="PARTIAL_SUCCESS", quality=None),
        _execution("gmgi_default", repetition=3, stages=[("recommendation", "timeout")], status="TIMEOUT", quality=None),
    ]

    gate = harness_comparison.replication_decision_gate(executions)

    assert gate["decision"] == "BASELINE UNSTABLE"
