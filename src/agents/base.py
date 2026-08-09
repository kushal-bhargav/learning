from __future__ import annotations

import json
import os
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

    def __init__(self, llm: StructuredLLM | None = None, *, retriever: Any | None = None, prompt_versions_dir: str | Path | None = None) -> None:
        self.llm = llm or create_llm()
        self.retriever = retriever
        self.config = json.loads((CONFIG_DIR / self.config_name).read_text(encoding="utf-8"))
        self.runtime_config = json.loads((CONFIG_DIR / "runtime.json").read_text(encoding="utf-8"))
        self.prompt_version_id = "static"
        self._load_prompt_version(prompt_versions_dir)

    def run(self, agent_input: AgentInput) -> AgentOutput:
        stage_config = agent_input.get("stage_config", {})
        context = self.build_context(agent_input)
        prompt = self.config["prompt_template"].format(context=json.dumps(context, ensure_ascii=False, indent=2))
        provider_name = self.llm.provider.value
        model = stage_config.get("model", self.config["models"][provider_name])
        temperature = float(stage_config.get("temperature", self.config["temperature"]))
        max_retries = int(stage_config.get("max_validation_retries", self.runtime_config["max_validation_retries"]))
        system_prompt = self._system_prompt(agent_input)
        result: dict[str, Any] | None = None
        for attempt in range(max_retries + 1):
            result = self.llm.generate(
                system_prompt=system_prompt, user_prompt=prompt,
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
        rationale = result["rationale"]
        output = dict(result["output"])
        output.setdefault("prompt_version", self.prompt_version_id)
        output.setdefault("skills_used", list(self.config.get("skills", [])))
        if self.prompt_version_id != "static":
            prefix = f"prompt_version={self.prompt_version_id}"
            rationale = f"{prefix}; {rationale}" if rationale else prefix
        return AgentOutput(stage=self.stage, output=output, confidence=result["confidence"], rationale=rationale)

    def _system_prompt(self, agent_input: AgentInput) -> str:
        prompt = str(self.config["system_prompt"])
        stage_config = agent_input.get("stage_config", {})
        context_fp = str(stage_config.get("context_fingerprint") or "")
        if self.retriever is not None and context_fp:
            prompt = self.retriever.augment_system_prompt(self.stage, prompt, context_fp)
        return prompt

    def _load_prompt_version(self, prompt_versions_dir: str | Path | None) -> None:
        root_value = prompt_versions_dir or os.getenv("GMGI_PROMPT_VERSIONS")
        root = Path(root_value) if root_value else CONFIG_DIR / "prompt_versions"
        latest = root / self.stage / "latest.json"
        if not latest.exists():
            return
        payload = json.loads(latest.read_text(encoding="utf-8"))
        prompt = payload.get("system_prompt") or payload.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            self.config["system_prompt"] = prompt
            self.prompt_version_id = str(payload.get("version_id") or latest.as_posix())

    @abstractmethod
    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        """Build the stage-specific prompt context without mutating the session."""
