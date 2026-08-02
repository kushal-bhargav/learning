from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .orchestrator import AgentInput, AgentOutput


class _DeliveryPlanOutput(BaseModel):
    mode: str
    channel: str
    planned_send_date: str
    occasion_date: str
    status: str
    disclaimer: str


class _DeliveryPlanResponse(BaseModel):
    output: _DeliveryPlanOutput
    confidence: float | None = 1.0
    rationale: str | None = None


def _planned_delivery_fields(stage_config: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    artifact_type = stage_config.get("artifact_type", "generated")
    digital = artifact_type == "generated" or stage_config.get("delivery_mode") == "digital"
    lead_days = config["digital_lead_days" if digital else "physical_lead_days"]
    channel = config["default_channel" if digital else "physical_channel"]
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
    return output, int(lead_days)


class DeliveryPlannerAgent:
    """Simulated delivery planner with deterministic logistics math and optional Instructor structuring."""

    stage = "delivery_planner"

    def __init__(self) -> None:
        path = Path(__file__).with_name("configs") / "delivery_planner.json"
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def run(self, agent_input: AgentInput) -> AgentOutput:
        stage_config = agent_input["stage_config"]
        deterministic_output, lead_days = _planned_delivery_fields(stage_config, self.config)
        try:
            return self._run_with_instructor(stage_config, deterministic_output, lead_days)
        except Exception:
            return AgentOutput(
                stage=self.stage,
                output=deterministic_output,
                confidence=1.0,
                rationale=f"Used the configured {lead_days}-day simulated lead time.",
            )

    def _run_with_instructor(self, stage_config: dict[str, Any], deterministic_output: dict[str, Any], lead_days: int) -> AgentOutput:
        import instructor
        from openai import OpenAI

        model = stage_config.get("model") or os.getenv("GMGI_DELIVERY_MODEL") or os.getenv("GMGI_OLLAMA_MODEL")
        if not model:
            raise RuntimeError("Set GMGI_OLLAMA_MODEL or GMGI_DELIVERY_MODEL to enable Instructor delivery planning")
        model = str(model)
        client = instructor.from_openai(
            OpenAI(
                base_url=os.getenv("GMGI_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                timeout=float(os.getenv("GMGI_OLLAMA_TIMEOUT_SECONDS", "30")),
            ),
            mode=instructor.Mode.JSON,
        )
        prompt = (
            "Return a structured simulated delivery plan. Use the precomputed date/logistics fields exactly; "
            "do not create purchases, shipments, tracking numbers, or external integrations.\n"
            f"Precomputed fields: {json.dumps(deterministic_output, ensure_ascii=False)}\n"
            f"Lead days used by the logistics math tool: {lead_days}"
        )
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You structure deterministic gift-delivery simulation output and preserve all dates exactly."},
                {"role": "user", "content": prompt},
            ],
            response_model=_DeliveryPlanResponse,
            temperature=0.0,
        )
        payload = result.model_dump()
        payload["output"].update(deterministic_output)
        return AgentOutput(stage=self.stage, output=payload["output"], confidence=payload.get("confidence"), rationale=payload.get("rationale") or f"Used the configured {lead_days}-day simulated lead time.")