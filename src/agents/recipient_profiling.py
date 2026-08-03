from __future__ import annotations

import json
import os
from typing import Any

from jsonschema import ValidationError, validate
from pydantic import BaseModel, Field

from .base import StructuredAgent
from .orchestrator import AgentInput, AgentOutput


class _Interest(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)


class _RecipientProfileOutput(BaseModel):
    interests: list[_Interest]
    communication_style: str
    gift_history_summary: str


class _RecipientProfileResponse(BaseModel):
    output: _RecipientProfileOutput
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None


class RecipientProfilingAgent(StructuredAgent):
    stage = "recipient_profiling"
    config_name = "recipient_profiling.json"

    def __init__(self, llm=None) -> None:
        self._explicit_llm = llm is not None
        super().__init__(llm)

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        return {
            "person": config["person"],
            "raw_notes": config.get("raw_notes", []),
            "preferences": config.get("preferences", []),
            "gift_history": config.get("gift_history", []),
        }

    def run(self, agent_input: AgentInput) -> AgentOutput:
        if self._explicit_llm and os.getenv("GMGI_FORCE_OLLAMA_AGENTS") != "1":
            return super().run(agent_input)
        try:
            return self._run_with_instructor(agent_input)
        except Exception:
            if os.getenv("GMGI_FORCE_OLLAMA_AGENTS") == "1" and os.getenv("GMGI_ALLOW_AGENT_FALLBACK") != "1":
                raise
            return super().run(agent_input)

    def _run_with_instructor(self, agent_input: AgentInput) -> AgentOutput:
        import instructor
        from openai import OpenAI

        stage_config = agent_input.get("stage_config", {})
        context = self.build_context(agent_input)
        prompt = self.config["prompt_template"].format(context=json.dumps(context, ensure_ascii=False, indent=2))
        model = str(stage_config.get("model") or os.getenv("GMGI_RECIPIENT_MODEL") or os.getenv("GMGI_OLLAMA_MODEL") or self.config["models"]["ollama"])
        temperature = float(stage_config.get("temperature", self.config["temperature"]))
        client = instructor.from_openai(
            OpenAI(
                base_url=os.getenv("GMGI_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                timeout=float(os.getenv("GMGI_OLLAMA_TIMEOUT_SECONDS", "30")),
            ),
            mode=instructor.Mode.JSON,
        )
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.config["system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            response_model=_RecipientProfileResponse,
            temperature=temperature,
        )
        payload = result.model_dump()
        try:
            validate(instance=payload, schema=self.config["output_schema"])
        except ValidationError as exc:
            raise ValueError(f"{self.stage} returned invalid structured output: {exc.message}") from exc
        return AgentOutput(stage=self.stage, output=payload["output"], confidence=payload["confidence"], rationale=payload["rationale"])