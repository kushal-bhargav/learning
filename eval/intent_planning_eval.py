from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.intent_planning import ExperimentCase, MethodVariant, compare_intent_planning_methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GMGI gift-intent and multi-agent-planning method variants")
    parser.add_argument("--config", type=Path, default=Path("eval/configs/intent_planning_eval.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    cases = [
        ExperimentCase(
            case_id=item["case_id"],
            custom_profile=item["custom_profile"],
            expected_intent=item.get("expected_intent"),
            expected_plan=item.get("expected_plan"),
        )
        for item in config["cases"]
    ]
    intent_variants = [MethodVariant(**item) for item in config["intent_variants"]]
    planning_variants = [MethodVariant(**item) for item in config["planning_variants"]]
    summary = compare_intent_planning_methods(
        cases,
        intent_variants=intent_variants,
        planning_variants=planning_variants,
        output_dir=config.get("output_dir", "experiments/intent_planning_eval"),
        run_overall=bool(config.get("run_overall", True)),
    )
    print(json.dumps({"component_summary": summary["component_summary"], "overall_summary": summary["overall_summary"]}, indent=2))


if __name__ == "__main__":
    main()