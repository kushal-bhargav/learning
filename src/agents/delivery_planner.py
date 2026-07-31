from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .orchestrator import AgentInput, AgentOutput


class DeliveryPlannerAgent:
    """Rule-based delivery simulation; it never contacts logistics providers."""

    stage = "delivery_planner"

    def __init__(self) -> None:
        path = Path(__file__).with_name("configs") / "delivery_planner.json"
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def run(self, agent_input: AgentInput) -> AgentOutput:
        stage_config = agent_input["stage_config"]
        artifact_type = stage_config.get("artifact_type", "generated")
        digital = artifact_type == "generated" or stage_config.get("delivery_mode") == "digital"
        lead_days = self.config["digital_lead_days" if digital else "physical_lead_days"]
        channel = self.config["default_channel" if digital else "physical_channel"]
        occasion_date = date.fromisoformat(stage_config["occasion"]["date"])
        planned_date = occasion_date - timedelta(days=lead_days)
        output: dict[str, Any] = {
            "mode": "digital" if digital else "physical",
            "channel": channel,
            "planned_send_date": planned_date.isoformat(),
            "occasion_date": occasion_date.isoformat(),
            "status": "simulated",
            "disclaimer": "No shipment, purchase, or external delivery was created.",
        }
        return AgentOutput(
            stage=self.stage,
            output=output,
            confidence=1.0,
            rationale=f"Used the configured {lead_days}-day simulated lead time.",
        )
