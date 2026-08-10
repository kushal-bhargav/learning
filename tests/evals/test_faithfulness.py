from __future__ import annotations

from src.evals.faithfulness import decompose_claims, evaluate_faithfulness


def test_decompose_claims_extracts_nested_output_fields() -> None:
    claims = decompose_claims({"message": "Tea is preferred.", "metadata": {"tone": "warm"}})
    assert "message: Tea is preferred." in claims
    assert "tone: warm" in claims


def test_lexical_faithfulness_flags_unsupported_claims() -> None:
    result = evaluate_faithfulness(
        {"preferences": ["tea"], "occasion": "birthday"},
        {"concept": "A tea birthday card", "extra": "They love mountain climbing"},
    )
    assert result["faithfulness_score"] < 1.0
    assert any("mountain" in claim.lower() for claim in result["unsupported_claims"])

