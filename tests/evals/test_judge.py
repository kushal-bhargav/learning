from __future__ import annotations

from src.evals.judge import judge_purpose_alignment


class FakeJudge:
    def generate(self, **kwargs):
        return {
            "fitness_for_purpose": 0.8,
            "completeness": 0.7,
            "unsupported_claim_risk": 0.1,
            "rationale": "Grounded enough.",
        }


def test_purpose_alignment_judge_uses_isolated_structured_call() -> None:
    result = judge_purpose_alignment(
        "Rank gift concepts grounded in input.",
        {"preferences": ["tea"]},
        {"recommendations": [{"concept": "tea card"}]},
        judge=FakeJudge(),
        model="judge-model",
    )
    assert result["fitness_for_purpose"] == 0.8
    assert result["unsupported_claim_risk"] == 0.1
