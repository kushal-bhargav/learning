from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.agents.orchestrator import AgentInput, AgentOutput
from src.api import AgencyConsoleService, create_app


class _FakeGANConfig:
    context_dim = 8


class _FakeGAN:
    config = _FakeGANConfig()


class _FakeCreativeAgent:
    stage = "creative_generation"
    gan = _FakeGAN()

    def run(self, agent_input: AgentInput) -> AgentOutput:
        config = agent_input["stage_config"]
        return {
            "stage": self.stage,
            "output": {
                "artifact_path": "experiments/generated/fake.png",
                "artifact_type": "generated",
                "media_type": "image/png",
                "width": 16,
                "height": 16,
                "agency_slider": float(config["agency_slider"]),
                "seed": int(config["seed"]),
            },
            "confidence": None,
            "rationale": "Fake creative agent for API tests.",
        }


def _client() -> TestClient:
    root = Path("experiments/test-api")
    root.mkdir(parents=True, exist_ok=True)
    service = AgencyConsoleService(
        generated_dir=root / "generated",
        bandit_log_path=root / "bandit_log.jsonl",
        bandit_state_path=root / "bandit_state.json",
    )
    service._creative_agent = _FakeCreativeAgent()  # type: ignore[assignment]
    return TestClient(create_app(service))


def test_agency_console_session_stage_actions_ledger_and_feedback() -> None:
    client = _client()
    created = client.post("/sessions", json={"persona_id": "long-distance-partners", "agency_slider": 0.25}).json()
    session_id = created["session_id"]

    proposed = client.post(f"/sessions/{session_id}/stages/recipient_profiling/propose", json={}).json()
    entry = proposed["stage_log"][-1]
    assert entry["status"] == "pending"
    assert entry["rationale"]

    edited = client.post(
        f"/sessions/{session_id}/stages/recipient_profiling/edit",
        json={"human_edit": {"communication_style": "A little more playful."}},
    ).json()
    assert edited["stage_log"][-1]["human_action"] == "edit"

    client.post(f"/sessions/{session_id}/stages/relationship_analysis/propose", json={})
    client.post(f"/sessions/{session_id}/stages/relationship_analysis/accept")

    client.post(f"/sessions/{session_id}/stages/recommendation/propose", json={})
    delegated = client.post(f"/sessions/{session_id}/stages/recommendation/delegate").json()
    assert delegated["next_stage"] is None
    assert {entry["stage"] for entry in delegated["stage_log"]} >= {
        "creative_generation",
        "greeting_story",
        "delivery_planner",
    }

    ledger = client.get(f"/sessions/{session_id}/ledger").json()
    assert ledger["completed"] is True
    assert ledger["counts"]["edit"] == 1
    assert ledger["counts"]["delegate"] >= 1
    assert all("rationale" in item for item in ledger["timeline"])

    feedback = client.post(
        f"/sessions/{session_id}/feedback",
        json={
            "rating": 5,
            "authorship": "hybrid",
            "open_text": "Felt like co-creating.",
            "measures": {"agency": 5, "satisfaction": 5},
        },
    ).json()
    assert 0.0 <= feedback["reward"] <= 1.0
    assert feedback["action"]["agency_bucket"] == "low"


def test_regenerate_keeps_stage_pending_with_new_proposal() -> None:
    client = _client()
    session_id = client.post("/sessions", json={"persona_id": "long-distance-partners"}).json()["session_id"]
    client.post(f"/sessions/{session_id}/stages/recipient_profiling/propose", json={})
    regenerated = client.post(
        f"/sessions/{session_id}/stages/recipient_profiling/regenerate",
        json={"overrides": {}},
    ).json()
    assert [entry["human_action"] for entry in regenerated["stage_log"]] == [None, "regenerate", None]
    assert regenerated["stage_log"][-1]["stage"] == "recipient_profiling"
    assert regenerated["stage_log"][-1]["status"] == "pending"

