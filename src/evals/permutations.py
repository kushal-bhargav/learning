from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .benchmark import load_cases, run_benchmark


RELATIONSHIP_AXIS = [
    ("partner", 5.0),
    ("sibling", 4.0),
    ("friend", 3.5),
    ("parent-child", 4.0),
    ("colleague", 2.0),
    ("mentor", 3.0),
    ("extended-family", 2.5),
]

OCCASION_AXIS = [
    {"occasion_name": "Birthday", "occasion_date": "2026-09-18", "formality": "casual"},
    {"occasion_name": "Housewarming", "occasion_date": "2026-10-24", "formality": "casual"},
    {"occasion_name": "Promotion", "occasion_date": "2026-08-28", "formality": "professional"},
    {"occasion_name": "Graduation", "occasion_date": "2026-06-12", "formality": "semi-formal"},
    {"occasion_name": "Thank-you", "occasion_date": "2026-11-15", "formality": "professional"},
    {"occasion_name": "Anniversary", "occasion_date": "2026-12-03", "formality": "ceremonial"},
]

BUDGET_AXIS = ["Flexible", "USD 25-45", "USD 60-100", "USD 150-250"]

PREFERENCE_AXIS = [
    ["ceramics", "quiet mornings", "green", "handwritten notes"],
    ["coffee", "minimalist design", "desk plants"],
    ["mustard yellow", "urban sketching", "travel-poster illustration"],
    ["music", "cooking", "warm kitchen colors"],
    ["books", "vintage maps", "train rides"],
]

MEMORY_AXIS = [
    [
        "They once got lost finding a tiny tea shop.",
        "They always send photos of interesting doors.",
    ],
    [
        "They helped during a difficult project week.",
        "They keep a very organized notebook.",
    ],
    [
        "They laughed beside a yellow tram after getting lost in Lisbon.",
        "They cook together over video calls during long-distance weeks.",
    ],
    [
        "They kept a basil plant alive through three apartment moves.",
        "They make Sunday chai whenever they are in the same city.",
    ],
    [],
]

AGENCY_AXIS = [0.15, 0.5, 0.85]


def generate_ui_permutation_cases(
    *,
    max_cases: int = 48,
    seed: int = 2026,
    axes: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    axes = axes or {}
    relationships = list(axes.get("relationships", RELATIONSHIP_AXIS))
    occasions = list(axes.get("occasions", OCCASION_AXIS))
    budgets = list(axes.get("budgets", BUDGET_AXIS))
    preferences = list(axes.get("preferences", PREFERENCE_AXIS))
    memories = list(axes.get("memories", MEMORY_AXIS))
    agencies = list(axes.get("agency_sliders", AGENCY_AXIS))
    full_size = len(relationships) * len(occasions) * len(budgets) * len(preferences) * len(memories) * len(agencies)
    if max_cases >= full_size:
        selected = [
            (relationship, occasion, budget, preference, memory, agency)
            for relationship in relationships
            for occasion in occasions
            for budget in budgets
            for preference in preferences
            for memory in memories
            for agency in agencies
        ]
    else:
        selected = _balanced_cases(
            relationships,
            occasions,
            budgets,
            preferences,
            memories,
            agencies,
            max_cases=max_cases,
            seed=seed,
        )
    cases = []
    for index, item in enumerate(selected, start=1):
        relationship, occasion, budget, pref_bundle, memory_bundle, agency = item
        relationship_type, closeness_score = _relationship_pair(relationship)
        occasion_payload = dict(occasion)
        case_id = (
            f"ui_perm_{index:03d}_"
            f"{_slug(relationship_type)}_{_slug(occasion_payload['occasion_name'])}_"
            f"c{str(closeness_score).replace('.', '_')}_a{str(agency).replace('.', '_')}"
        )
        profile = {
            "giver_name": "Eval giver",
            "recipient_name": f"Eval recipient {index}",
            "relationship_type": relationship_type,
            "closeness_score": closeness_score,
            "occasion_name": occasion_payload["occasion_name"],
            "occasion_date": occasion_payload["occasion_date"],
            "budget_hint": str(budget),
            "formality": occasion_payload["formality"],
            "preferences": list(pref_bundle),
            "memories": list(memory_bundle),
            "agency_slider": float(agency),
        }
        cases.append(
            {
                "case_id": case_id,
                "custom_profile": profile,
                "expected": {
                    "relationship_type": relationship_type,
                    "closeness_score": closeness_score,
                    "relationship_closeness": _closeness_bucket(closeness_score),
                    "occasion_name": occasion_payload["occasion_name"],
                    "occasion_date": occasion_payload["occasion_date"],
                    "budget_hint": str(budget),
                    "preferences": list(pref_bundle),
                    "memory_count": len(memory_bundle),
                    "agency_slider": float(agency),
                    "must_preserve_constraints": [
                        str(budget),
                        occasion_payload["formality"],
                        "simulated delivery only",
                    ],
                },
            }
        )
    return {
        "metadata": {
            "source": "ui_input_permutation_generator",
            "ui_fields": [
                "giver_name",
                "recipient_name",
                "relationship_type",
                "closeness_score",
                "occasion_name",
                "occasion_date",
                "budget_hint",
                "formality",
                "preferences",
                "memories",
                "agency_slider",
            ],
            "max_cases": max_cases,
            "seed": seed,
            "full_factorial_size": full_size,
        },
        "cases": cases,
    }


def write_ui_permutation_cases(
    output: str | Path = "experiments/evals/ui_permutation_cases.json",
    *,
    max_cases: int = 48,
    seed: int = 2026,
) -> dict[str, Any]:
    payload = generate_ui_permutation_cases(max_cases=max_cases, seed=seed)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def generate_and_run(
    *,
    case_file: str | Path = "experiments/evals/ui_permutation_cases.json",
    output_dir: str | Path = "experiments/evals/ui_permutation_benchmark",
    max_cases: int = 48,
    seed: int = 2026,
    limit: int | None = None,
    include_creative: bool = False,
    stage_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    write_ui_permutation_cases(case_file, max_cases=max_cases, seed=seed)
    return run_benchmark(
        load_cases(case_file),
        output_dir=output_dir,
        include_creative=include_creative,
        seed=seed,
        limit=limit,
        stage_timeout_seconds=stage_timeout_seconds,
    )


def _spread(values: Iterable[tuple[Any, ...]], *, max_cases: int, seed: int) -> list[tuple[Any, ...]]:
    items = list(values)
    if max_cases >= len(items):
        return items
    stride = max(1, len(items) // max_cases)
    start = seed % stride
    selected = items[start::stride][:max_cases]
    return selected


def _balanced_cases(
    relationships: Sequence[Any],
    occasions: Sequence[Any],
    budgets: Sequence[Any],
    preferences: Sequence[Any],
    memories: Sequence[Any],
    agencies: Sequence[Any],
    *,
    max_cases: int,
    seed: int,
) -> list[tuple[Any, ...]]:
    if max_cases <= 0:
        return []
    lengths = [len(relationships), len(occasions), len(budgets), len(preferences), len(memories), len(agencies)]
    if any(length == 0 for length in lengths):
        return []
    steps = [1, 2, 3, 5, 7, 11]
    offsets = [seed % length for length in lengths]
    seen: set[tuple[int, int, int, int, int, int]] = set()
    selected: list[tuple[Any, ...]] = []
    index = 0
    while len(selected) < max_cases:
        key = tuple((offsets[pos] + index * steps[pos]) % lengths[pos] for pos in range(6))
        if key not in seen:
            seen.add(key)
            rel_i, occ_i, budget_i, pref_i, memory_i, agency_i = key
            selected.append((relationships[rel_i], occasions[occ_i], budgets[budget_i], preferences[pref_i], memories[memory_i], agencies[agency_i]))
        index += 1
        if index > max_cases * 20 and len(seen) == max_cases:
            break
    return selected


def _relationship_pair(value: Any) -> tuple[str, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return str(value[0]), float(value[1])
    return str(value), 3.0


def _closeness_bucket(value: float) -> str:
    if value >= 4.5:
        return "very close"
    if value >= 3.5:
        return "close"
    if value >= 2:
        return "moderate"
    return "low"


def _slug(value: Any) -> str:
    return "".join(char if char.isalnum() else "-" for char in str(value).lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UI-input permutation cases and optionally run the GMGI benchmark.")
    parser.add_argument("--output", default="experiments/evals/ui_permutation_cases.json")
    parser.add_argument("--max-cases", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", default="experiments/evals/ui_permutation_benchmark")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-creative", action="store_true")
    parser.add_argument("--stage-timeout", type=float, default=None)
    args = parser.parse_args()
    if args.run:
        report = generate_and_run(
            case_file=args.output,
            output_dir=args.output_dir,
            max_cases=args.max_cases,
            seed=args.seed,
            limit=args.limit,
            include_creative=args.include_creative,
            stage_timeout_seconds=args.stage_timeout,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        payload = write_ui_permutation_cases(args.output, max_cases=args.max_cases, seed=args.seed)
        print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
