from __future__ import annotations

from src.agents.orchestrator import GiftSession
from src.evals.replay import diff_outputs, perturb_mapping, run_counterfactual, run_self_consistency


class EchoAgent:
    stage = "echo"

    def run(self, agent_input):
        return {"stage": self.stage, "output": dict(agent_input["stage_config"]), "confidence": 1.0, "rationale": "echo"}


def test_perturb_mapping_updates_nested_field_without_mutating_original() -> None:
    original = {"relationship": {"closeness_score": 3}}
    perturbed = perturb_mapping(original, "relationship.closeness_score", 5)
    assert original["relationship"]["closeness_score"] == 3
    assert perturbed["relationship"]["closeness_score"] == 5


def test_counterfactual_and_self_consistency_harnesses_invoke_agent_in_isolation() -> None:
    session = GiftSession(session_id="s", giver_id="g", recipient_id="r", occasion_id="o")
    config = {"relationship": {"closeness_score": 3}, "budget": "USD 60"}
    counterfactual = run_counterfactual(lambda: EchoAgent(), session, config, perturbation_path="relationship.closeness_score", replacement=5)
    assert counterfactual["diff"]["changed"] is True
    stable = run_self_consistency(lambda: EchoAgent(), session, config, runs=3)
    assert stable["stable"] is True


def test_diff_outputs_reports_similarity() -> None:
    diff = diff_outputs({"a": "tea"}, {"a": "tea", "b": "card"})
    assert diff["changed"] is True
    assert diff["token_jaccard"] is not None

