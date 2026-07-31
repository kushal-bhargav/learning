from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from .llm import StructuredLLM, create_llm
from .orchestrator import AgentInput, AgentOutput

CONFIG_DIR = Path(__file__).with_name("configs")


class StructuredAgent(ABC):
    stage: str
    config_name: str

    def __init__(self, llm: StructuredLLM | None = None) -> None:
        self.llm = llm or create_llm()
        self.config = json.loads((CONFIG_DIR / self.config_name).read_text(encoding="utf-8"))
        self.runtime_config = json.loads((CONFIG_DIR / "runtime.json").read_text(encoding="utf-8"))

    def run(self, agent_input: AgentInput) -> AgentOutput:
        stage_config = agent_input.get("stage_config", {})
        context = self.build_context(agent_input)
        prompt = self.config["prompt_template"].format(context=json.dumps(context, ensure_ascii=False, indent=2))
        provider_name = self.llm.provider.value
        model = stage_config.get("model", self.config["models"][provider_name])
        temperature = float(stage_config.get("temperature", self.config["temperature"]))
        max_retries = int(stage_config.get("max_validation_retries", self.runtime_config["max_validation_retries"]))
        result: dict[str, Any] | None = None
        for attempt in range(max_retries + 1):
            result = self.llm.generate(
                system_prompt=self.config["system_prompt"], user_prompt=prompt,
                schema=self.config["output_schema"], temperature=temperature, model=model,
            )
            try:
                validate(instance=result, schema=self.config["output_schema"])
                break
            except ValidationError as exc:
                if attempt == max_retries:
                    raise ValueError(f"{self.stage} returned invalid structured output: {exc.message}") from exc
                prompt = self.runtime_config["repair_prompt_template"].format(prompt=prompt, error=exc.message)
        assert result is not None
        return AgentOutput(stage=self.stage, output=result["output"], confidence=result["confidence"], rationale=result["rationale"])

    @abstractmethod
    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        """Build the stage-specific prompt context without mutating the session."""
