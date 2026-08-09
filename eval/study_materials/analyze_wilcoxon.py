from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

MEASURES = ("authorship", "control", "satisfaction", "novelty", "trust")
CONDITIONS = ("ai_autonomous", "human_only", "negotiated_hybrid")
COMPARISONS = (
    ("negotiated_hybrid", "ai_autonomous"),
    ("negotiated_hybrid", "human_only"),
)
COUNT_COLUMNS = ("accept_count", "edit_count", "regenerate_count", "delegate_count")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no rows")
    required = {"participant_id", "condition", *MEASURES}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return rows


def to_float(value: str, *, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid numeric value for {column}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid non-finite value for {column}: {value!r}")
    return parsed


def pivot_scores(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        participant = row["participant_id"].strip()
        condition = row["condition"].strip()
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
        if condition in scores[participant]:
            raise ValueError(f"Duplicate row for participant {participant!r}, condition {condition!r}")
        scores[participant][condition] = {
            measure: to_float(row[measure], column=measure) for measure in MEASURES
        }
    return scores


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(array, [25, 75])
    return {
        "n": int(array.size),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "iqr": float(q3 - q1),
    }


def rank_biserial_effect(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    abs_values = np.abs(nonzero)
    order = np.argsort(abs_values)
    ranks = np.empty_like(abs_values, dtype=float)
    start = 0
    while start < abs_values.size:
        end = start + 1
        while end < abs_values.size and abs_values[order[end]] == abs_values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    return 0.0 if total == 0 else (positive - negative) / total


def paired_tests(scores: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for first, second in COMPARISONS:
        paired_ids = sorted(
            participant
            for participant, by_condition in scores.items()
            if first in by_condition and second in by_condition
        )
        for measure in MEASURES:
            first_values = np.asarray([scores[p][first][measure] for p in paired_ids], dtype=float)
            second_values = np.asarray([scores[p][second][measure] for p in paired_ids], dtype=float)
            differences = first_values - second_values
            if paired_ids and np.any(differences != 0):
                test = wilcoxon(first_values, second_values, zero_method="wilcox", alternative="two-sided", mode="auto")
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            else:
                statistic = 0.0
                p_value = 1.0
            results.append(
                {
                    "comparison": f"{first} vs {second}",
                    "measure": measure,
                    "n_pairs": len(paired_ids),
                    "mean_difference": float(differences.mean()) if paired_ids else None,
                    "median_difference": float(np.median(differences)) if paired_ids else None,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial_effect": rank_biserial_effect(differences) if paired_ids else None,
                }
            )
    return results


def behavioral_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float | int]]]:
    available = [column for column in COUNT_COLUMNS if column in rows[0]]
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    if not available:
        return summary
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"].strip()].append(row)
    for condition, condition_rows in by_condition.items():
        summary[condition] = {}
        for column in available:
            values = [to_float(row.get(column, "0") or "0", column=column) for row in condition_rows]
            summary[condition][column] = describe(values)
    return summary


def descriptive_tables(scores: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float | int]]]:
    descriptions: dict[str, dict[str, dict[str, float | int]]] = {}
    for condition in CONDITIONS:
        descriptions[condition] = {}
        for measure in MEASURES:
            values = [
                by_condition[condition][measure]
                for by_condition in scores.values()
                if condition in by_condition
            ]
            if values:
                descriptions[condition][measure] = describe(values)
    return descriptions


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Human Study Wilcoxon Analysis",
        "",
        f"Input rows: {report['rows']}",
        f"Participants: {report['participants']}",
        "",
        "## Paired Tests",
        "",
        "| Comparison | Measure | n | Mean diff | Median diff | W | p | Rank-biserial r |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["paired_tests"]:
        mean_diff = "NA" if row["mean_difference"] is None else f"{row['mean_difference']:.3f}"
        median_diff = "NA" if row["median_difference"] is None else f"{row['median_difference']:.3f}"
        effect = "NA" if row["rank_biserial_effect"] is None else f"{row['rank_biserial_effect']:.3f}"
        lines.append(
            f"| {row['comparison']} | {row['measure']} | {row['n_pairs']} | {mean_diff} | {median_diff} | {row['wilcoxon_statistic']:.3f} | {row['p_value']:.4f} | {effect} |"
        )
    lines.extend([
        "",
        "## Reporting Note",
        "",
        "Use these tests as small-n descriptive evidence. Report n, medians, spreads, p values, and effect sizes. Do not claim broad statistical significance without appropriate caveats.",
        "",
    ])
    return "\n".join(lines)


def analyze(input_path: Path) -> dict[str, Any]:
    rows = load_rows(input_path)
    scores = pivot_scores(rows)
    return {
        "input": str(input_path),
        "rows": len(rows),
        "participants": len(scores),
        "measures": list(MEASURES),
        "conditions": list(CONDITIONS),
        "descriptive": descriptive_tables(scores),
        "paired_tests": paired_tests(scores),
        "behavioral_summary": behavioral_summary(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GMGI human-study Likert CSV with paired Wilcoxon tests")
    parser.add_argument("csv", type=Path, help="Study CSV with one row per participant per condition")
    parser.add_argument("--json-output", type=Path, default=Path("eval/study_materials/wilcoxon_results.json"))
    parser.add_argument("--md-output", type=Path, default=Path("eval/study_materials/wilcoxon_results.md"))
    args = parser.parse_args()

    report = analyze(args.csv)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = markdown_report(report)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
