from __future__ import annotations

import os

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.agents.orchestrator import AgentInput, AgentOutput
from src.api import AgencyConsoleService, create_app
from src.harness import HarnessConfig


class _FakeCreativeAgent:
    stage = "creative_generation"

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
    os.environ["GMGI_USE_DEMO_AGENT_RESPONSES"] = "1"
    os.environ["GMGI_ALLOW_AGENT_FALLBACK"] = "1"
    os.environ.pop("GMGI_FORCE_OLLAMA_AGENTS", None)
    root = Path("experiments/test-api")
    root.mkdir(parents=True, exist_ok=True)
    service = AgencyConsoleService(
        generated_dir=root / "generated",
        bandit_log_path=root / "bandit_log.jsonl",
        bandit_state_path=root / "bandit_state.json",
        experience_store_path=root / "experience_store.jsonl",
    )
    service._creative_agent = _FakeCreativeAgent()  # type: ignore[assignment]
    return TestClient(create_app(service))


def test_agency_console_session_stage_actions_ledger_and_feedback() -> None:
    client = _client()
    created = client.post("/sessions", json={"persona_id": "long-distance-partners", "agency_slider": 0.25}).json()
    session_id = created["session_id"]
    assert created["run_id"].startswith("run-")
    assert created["harness"]["harness_id"] == "gmgi_default"

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

    client.post(f"/sessions/{session_id}/stages/gift_intent_reasoning/propose", json={})
    client.post(f"/sessions/{session_id}/stages/gift_intent_reasoning/accept")
    client.post(f"/sessions/{session_id}/stages/multi_agent_planning/propose", json={})
    client.post(f"/sessions/{session_id}/stages/multi_agent_planning/accept")

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


def test_session_trace_endpoint_exposes_raw_agent_input_and_harness_identity() -> None:
    client = _client()
    session_id = client.post("/sessions", json={"persona_id": "long-distance-partners"}).json()["session_id"]
    client.post(f"/sessions/{session_id}/stages/recipient_profiling/propose", json={})

    trace = client.get(f"/sessions/{session_id}/trace").json()
    assert trace["run_id"].startswith("run-")
    assert trace["case_id"] == "long-distance-partners"
    assert trace["harness_config"]["harness_id"] == "gmgi_default"
    assert trace["harness_config"]["config_hash"]
    assert len(trace["invocations"]) == 1

    invocation = trace["invocations"][0]
    assert invocation["stage_name"] == "recipient_profiling"
    assert invocation["agent_name"] == "RecipientProfilingAgent"
    assert invocation["raw_agent_input"]["session"]["session_id"] == session_id
    assert "stage_config" in invocation["raw_agent_input"]
    assert invocation["routing_decision"]["decision"] == "recipient_profiling"
    assert invocation["planner_decision"]["decision"] == "advisory_only"
    assert invocation["verifier_decision"]["decision"] == "not_available"


def test_controller_schema_gate_updates_trace_verifier_decision() -> None:
    os.environ["GMGI_USE_DEMO_AGENT_RESPONSES"] = "1"
    os.environ["GMGI_ALLOW_AGENT_FALLBACK"] = "1"
    os.environ.pop("GMGI_FORCE_OLLAMA_AGENTS", None)
    root = Path("experiments/test-api")
    service = AgencyConsoleService(
        generated_dir=root / "generated",
        bandit_log_path=root / "bandit_log.jsonl",
        bandit_state_path=root / "bandit_state.json",
        harness_config=HarnessConfig(harness_id="gmgi_schema_gate", verification_policy="controller_schema_gate"),
    )
    client = TestClient(create_app(service))
    session_id = client.post("/sessions", json={"persona_id": "long-distance-partners"}).json()["session_id"]
    client.post(f"/sessions/{session_id}/stages/recipient_profiling/propose", json={})

    trace = client.get(f"/sessions/{session_id}/trace").json()
    invocation = trace["invocations"][0]
    assert trace["harness_config"]["harness_id"] == "gmgi_schema_gate"
    assert invocation["verifier_decision"]["decision"] == "passed"
    assert invocation["validation_result"]["policy"] == "controller_schema_gate"


def test_stop_before_delivery_harness_changes_service_trajectory() -> None:
    os.environ["GMGI_USE_DEMO_AGENT_RESPONSES"] = "1"
    root = Path("experiments/test-api")
    service = AgencyConsoleService(
        generated_dir=root / "generated",
        bandit_log_path=root / "bandit_log.jsonl",
        bandit_state_path=root / "bandit_state.json",
        harness_config=HarnessConfig(harness_id="gmgi_stop_before_delivery", stopping_policy="stop_before_delivery"),
    )
    service._creative_agent = _FakeCreativeAgent()  # type: ignore[assignment]
    client = TestClient(create_app(service))
    session_id = client.post("/sessions", json={"persona_id": "long-distance-partners"}).json()["session_id"]

    for stage in (
        "recipient_profiling",
        "relationship_analysis",
        "gift_intent_reasoning",
        "multi_agent_planning",
        "recommendation",
        "creative_generation",
        "greeting_story",
    ):
        client.post(f"/sessions/{session_id}/stages/{stage}/propose", json={})
        client.post(f"/sessions/{session_id}/stages/{stage}/accept")

    session = client.get(f"/sessions/{session_id}").json()
    assert session["next_stage"] is None
    assert "delivery_planner" not in {entry["stage"] for entry in session["stage_log"]}
    assert session["harness"]["harness_id"] == "gmgi_stop_before_delivery"


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



def test_live_persona_default_and_custom_session_creation() -> None:
    client = _client()
    personas = client.get("/personas").json()
    assert personas == [{"persona_id": "custom-live", "label": "Create a live gifting context", "synthetic": False, "occasions": []}]

    created = client.post(
        "/sessions",
        json={
            "agency_slider": 0.4,
            "seed": 2026,
            "custom_profile": {
                "giver_name": "Asha",
                "recipient_name": "Mira",
                "relationship_type": "friend",
                "closeness_score": 4,
                "occasion_name": "Birthday",
                "occasion_date": "2026-12-18",
                "budget_hint": "USD 80",
                "formality": "casual",
                "preferences": ["ceramics", "green", "quiet mornings"],
                "memories": ["We got lost finding a tiny tea shop.", "She notices beautiful doors."],
            },
        },
    ).json()
    assert created["session_id"].startswith("live-")
    assert created["next_stage"] == "recipient_profiling"

def test_live_occasion_names_are_normalized_for_visual_generation() -> None:
    service = AgencyConsoleService()
    assert service._visual_occasion("Birthday dinner") == "birthday"
    assert service._visual_occasion("new home celebration") == "housewarming"
    assert service._visual_occasion("custom ritual") == "other"


def test_stage_agent_failures_return_structured_error_detail(monkeypatch) -> None:
    root = Path("experiments/test-api")
    root.mkdir(parents=True, exist_ok=True)
    service = AgencyConsoleService(
        generated_dir=root / "generated",
        bandit_log_path=root / "bandit_log.jsonl",
        bandit_state_path=root / "bandit_state.json",
        experience_store_path=root / "experience_store.jsonl",
    )
    client = TestClient(create_app(service))
    session_id = client.post("/sessions", json={"persona_id": "long-distance-partners"}).json()["session_id"]

    def fail_propose(*args, **kwargs):
        raise TypeError("tool wrapper failed")

    monkeypatch.setattr(service, "propose", fail_propose)
    response = client.post(f"/sessions/{session_id}/stages/relationship_analysis/propose", json={})
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["stage"] == "relationship_analysis"
    assert detail["error_type"] == "TypeError"
    assert detail["message"] == "tool wrapper failed"

from src.api.service import STAGES


def test_new_intent_and_planning_stages_are_in_default_orchestration() -> None:
    client = _client()
    created = client.post("/sessions", json={"persona_id": "long-distance-partners", "agency_slider": 0.25}).json()
    assert created["next_stage"] == "recipient_profiling"
    assert "gift_intent_reasoning" in STAGES
    assert "multi_agent_planning" in STAGES
    assert STAGES.index("relationship_analysis") < STAGES.index("gift_intent_reasoning") < STAGES.index("multi_agent_planning") < STAGES.index("recommendation")

    session_id = created["session_id"]
    for stage in ("recipient_profiling", "relationship_analysis", "gift_intent_reasoning", "multi_agent_planning"):
        proposed = client.post(f"/sessions/{session_id}/stages/{stage}/propose", json={}).json()
        assert proposed["stage_log"][-1]["stage"] == stage
        accepted = client.post(f"/sessions/{session_id}/stages/{stage}/accept").json()
        assert accepted["next_stage"] != stage
    assert accepted["next_stage"] == "recommendation"



