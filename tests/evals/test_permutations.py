from __future__ import annotations

from src.evals.benchmark import BenchmarkCase
from src.evals.permutations import generate_ui_permutation_cases


def test_ui_permutation_cases_match_benchmark_schema() -> None:
    payload = generate_ui_permutation_cases(max_cases=6, seed=7)
    assert payload["metadata"]["full_factorial_size"] > 6
    assert len(payload["cases"]) == 6
    cases = [BenchmarkCase.from_mapping(item) for item in payload["cases"]]
    assert all(case.custom_profile["recipient_name"] for case in cases)
    assert all(case.expected["relationship_closeness"] in {"low", "moderate", "close", "very close"} for case in cases)


def test_ui_permutation_cases_cover_actual_ui_fields() -> None:
    payload = generate_ui_permutation_cases(max_cases=10, seed=2026)
    ui_fields = set(payload["metadata"]["ui_fields"])
    expected_fields = {
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
    }
    assert expected_fields <= ui_fields
    profiles = [case["custom_profile"] for case in payload["cases"]]
    assert len({profile["relationship_type"] for profile in profiles}) > 1
    assert len({profile["occasion_name"] for profile in profiles}) > 1
    assert len({profile["agency_slider"] for profile in profiles}) > 1
