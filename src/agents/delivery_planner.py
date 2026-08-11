from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .orchestrator import AgentInput, AgentOutput
from .skills import add_skill_metadata
from src.harness import ensure_tool_allowed, record_tool_call


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

    def __init__(self, *, retriever: Any | None = None, prompt_versions_dir: str | Path | None = None) -> None:
        self.retriever = retriever
        self.config_dir = Path(__file__).with_name("configs")
        path = self.config_dir / "delivery_planner.json"
        self.config = json.loads(path.read_text(encoding="utf-8"))
        self.prompt_version_id = "static"
        self._load_prompt_version(prompt_versions_dir)

    def run(self, agent_input: AgentInput) -> AgentOutput:
        stage_config = agent_input["stage_config"]
        started = time.perf_counter()
        ensure_tool_allowed("date_logistics_math")
        deterministic_output, lead_days = _planned_delivery_fields(stage_config, self.config)
        record_tool_call(
            "date_logistics_math",
            arguments={"artifact_type": stage_config.get("artifact_type", "generated"), "occasion": stage_config.get("occasion", {})},
            result={"output": deterministic_output, "lead_days": lead_days},
            latency_seconds=time.perf_counter() - started,
        )
        add_skill_metadata(deterministic_output, self.config, active=["date_logistics_math", "simulated_delivery_structuring"])
        deterministic_output["prompt_version"] = self.prompt_version_id
        try:
            return self._run_with_instructor(stage_config, deterministic_output, lead_days)
        except Exception:
            return AgentOutput(
                stage=self.stage,
                output=deterministic_output,
                confidence=1.0,
                rationale=self._rationale(f"Used the configured {lead_days}-day simulated lead time."),
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
                {"role": "system", "content": self._system_prompt(stage_config)},
                {"role": "user", "content": prompt},
            ],
            response_model=_DeliveryPlanResponse,
            temperature=0.0,
        )
        payload = result.model_dump()
        payload["output"].update(deterministic_output)
        return AgentOutput(stage=self.stage, output=payload["output"], confidence=payload.get("confidence"), rationale=self._rationale(payload.get("rationale") or f"Used the configured {lead_days}-day simulated lead time."))

    def _system_prompt(self, stage_config: dict[str, Any]) -> str:
        prompt = str(self.config.get("system_prompt") or "You structure deterministic gift-delivery simulation output and preserve all dates exactly.")
        context_fp = str(stage_config.get("context_fingerprint") or "")
        if self.retriever is not None and context_fp:
            prompt = self.retriever.augment_system_prompt(self.stage, prompt, context_fp)
        return prompt

    def _load_prompt_version(self, prompt_versions_dir: str | Path | None) -> None:
        root_value = prompt_versions_dir or os.getenv("GMGI_PROMPT_VERSIONS")
        root = Path(root_value) if root_value else self.config_dir / "prompt_versions"
        latest = root / self.stage / "latest.json"
        if not latest.exists():
            return
        payload = json.loads(latest.read_text(encoding="utf-8"))
        prompt = payload.get("system_prompt") or payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            self.config["system_prompt"] = prompt
            self.prompt_version_id = str(payload.get("version_id") or latest.as_posix())

    def _rationale(self, rationale: str | None) -> str | None:
        if self.prompt_version_id == "static":
            return rationale
        prefix = f"prompt_version={self.prompt_version_id}"
        return f"{prefix}; {rationale}" if rationale else prefix

