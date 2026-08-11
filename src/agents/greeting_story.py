from __future__ import annotations

import json
import os
from typing import Any

from jsonschema import ValidationError, validate

from .base import StructuredAgent
from .orchestrator import AgentInput, AgentOutput
from .skills import add_skill_metadata


def _extract_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Ollama did not return a JSON object")
    return json.loads(text[start : end + 1])


class GreetingStoryAgent(StructuredAgent):
    stage = "greeting_story"
    config_name = "greeting_story.json"

    def __init__(self, llm=None, *, retriever=None, prompt_versions_dir=None) -> None:
        self._explicit_llm = llm is not None
        super().__init__(llm, retriever=retriever, prompt_versions_dir=prompt_versions_dir)

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        memories = [
            {key: value for key, value in memory.items() if key != "embedding"}
            for memory in config.get("memories", [])[:2]
        ]
        return {
            "relationship_guidance": config["relationship_guidance"],
            "occasion": config["occasion"],
            "salient_memories": memories,
            "tone_guidance": config.get("tone_guidance"),
            "giver_name": config.get("giver_name"),
            "recipient_name": config.get("recipient_name"),
        }

    def run(self, agent_input: AgentInput) -> AgentOutput:
        if self._explicit_llm and os.getenv("GMGI_FORCE_OLLAMA_AGENTS") != "1":
            return super().run(agent_input)
        try:
            return self._run_with_ollama_chat(agent_input)
        except Exception:
            if os.getenv("GMGI_FORCE_OLLAMA_AGENTS") == "1" and os.getenv("GMGI_ALLOW_AGENT_FALLBACK") != "1":
                raise
            return super().run(agent_input)

    def _run_with_ollama_chat(self, agent_input: AgentInput) -> AgentOutput:
        from ollama import Client

        stage_config = agent_input.get("stage_config", {})
        context = self.build_context(agent_input)
        prompt = self.config["prompt_template"].format(context=json.dumps(context, ensure_ascii=False, indent=2))
        model = str(stage_config.get("model") or os.getenv("GMGI_GREETING_MODEL") or os.getenv("GMGI_OLLAMA_MODEL") or self.config["models"]["ollama"])
        client = Client(host=os.getenv("GMGI_OLLAMA_HOST", "http://localhost:11434"), timeout=float(os.getenv("GMGI_OLLAMA_TIMEOUT_SECONDS", "30")))
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": self._system_prompt(agent_input)},
                {"role": "user", "content": prompt + "\nReturn only JSON matching the provided schema. Keep the message under 90 words."},
            ],
            format=self.config["output_schema"],
            options={
                "temperature": float(stage_config.get("temperature", self.config["temperature"])),
                "num_predict": int(stage_config.get("num_predict", os.getenv("GMGI_GREETING_NUM_PREDICT", "220"))),
            },
            stream=False,
        )
        payload = _extract_json(response["message"]["content"])
        try:
            validate(instance=payload, schema=self.config["output_schema"])
        except ValidationError as exc:
            raise ValueError(f"{self.stage} returned invalid structured output: {exc.message}") from exc
        output = dict(payload["output"])
        output.setdefault("prompt_version", self.prompt_version_id)
        add_skill_metadata(output, self.config)
        return AgentOutput(stage=self.stage, output=output, confidence=payload["confidence"], rationale=payload["rationale"])




