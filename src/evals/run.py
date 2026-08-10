from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.agents.llm import create_llm

from .benchmark import load_cases as load_benchmark_cases
from .benchmark import run_benchmark
from .faithfulness import evaluate_store as evaluate_faithfulness_store
from .judge import evaluate_store as evaluate_judge_store
from .permutations import generate_and_run as generate_and_run_permutations
from .permutations import write_ui_permutation_cases
from .quality import evaluate_store as evaluate_quality_store
from .replay import evaluate_store as evaluate_replay_store
from .structural import evaluate_store as evaluate_structural_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reference-free GMGI evals over logged sessions or benchmark cases.")
    parser.add_argument("--phase", choices=["1", "2", "3", "4", "quality", "benchmark", "ui-permutations", "all"], default="1", help="Evaluation phase to run.")
    parser.add_argument("--store", default="experiments/experience_store.jsonl", help="Path to ExperienceStore JSONL logs.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the most recent N episodes.")
    parser.add_argument("--enable-judge", action="store_true", help="Enable Phase 4 judge model calls.")
    parser.add_argument("--judge-provider", default=os.getenv("GMGI_EVAL_JUDGE_PROVIDER"), help="Optional provider for Phase 4 judge.")
    parser.add_argument("--judge-model", default=os.getenv("GMGI_EVAL_JUDGE_MODEL", "gpt-4o-mini"), help="Model name for Phase 4 judge.")
    parser.add_argument("--case-file", default=None, help="Optional benchmark case JSON file.")
    parser.add_argument("--output-dir", default="experiments/evals/benchmark", help="Benchmark report directory.")
    parser.add_argument("--include-creative", action="store_true", help="Run creative generation during benchmark eval.")
    parser.add_argument("--agency-slider", type=float, default=0.5, help="Agency slider value for benchmark creative generation.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for benchmark creative generation.")
    parser.add_argument("--stage-timeout", type=float, default=None, help="Per-agent benchmark timeout in seconds.")
    parser.add_argument("--max-cases", type=int, default=48, help="Maximum generated UI permutation cases.")
    parser.add_argument("--run-permutations", action="store_true", help="Run benchmark after generating UI permutation cases.")
    args = parser.parse_args()

    report = _run_phase(args.phase, args)
    rendered = json.dumps(report, indent=2, default=str)
    print(rendered)
    if args.output and args.phase != "ui-permutations":
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


def _run_phase(phase: str, args: argparse.Namespace) -> dict:
    if phase == "1":
        return evaluate_structural_store(args.store, limit=args.limit)
    if phase == "2":
        return evaluate_faithfulness_store(args.store, limit=args.limit)
    if phase == "3":
        return evaluate_replay_store(args.store, limit=args.limit)
    if phase == "4":
        judge = create_llm(args.judge_provider) if args.enable_judge else None
        return evaluate_judge_store(args.store, judge=judge, model=args.judge_model, limit=args.limit)
    if phase == "quality":
        return evaluate_quality_store(args.store, limit=args.limit)
    if phase == "benchmark":
        return run_benchmark(
            load_benchmark_cases(args.case_file),
            output_dir=args.output_dir,
            include_creative=args.include_creative,
            agency_slider=args.agency_slider,
            seed=args.seed,
            limit=args.limit,
            stage_timeout_seconds=args.stage_timeout,
        )
    if phase == "ui-permutations":
        case_output = args.output or "experiments/evals/ui_permutation_cases.json"
        if args.run_permutations:
            return generate_and_run_permutations(
                case_file=case_output,
                output_dir=args.output_dir,
                max_cases=args.max_cases,
                seed=args.seed,
                limit=args.limit,
                include_creative=args.include_creative,
                stage_timeout_seconds=args.stage_timeout,
            )
        payload = write_ui_permutation_cases(case_output, max_cases=args.max_cases, seed=args.seed)
        return {"phase": "ui-permutations", "case_file": case_output, **payload["metadata"]}
    reports = {
        "1": evaluate_structural_store(args.store, limit=args.limit),
        "2": evaluate_faithfulness_store(args.store, limit=args.limit),
        "3": evaluate_replay_store(args.store, limit=args.limit),
        "4": evaluate_judge_store(args.store, judge=create_llm(args.judge_provider) if args.enable_judge else None, model=args.judge_model, limit=args.limit),
        "quality": evaluate_quality_store(args.store, limit=args.limit),
    }
    return {"phase": "all", "reports": reports}


if __name__ == "__main__":
    main()
