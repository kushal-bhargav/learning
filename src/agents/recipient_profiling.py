from __future__ import annotations

import json
import os
from typing import Any

from jsonschema import ValidationError, validate
from pydantic import BaseModel, Field

from .base import StructuredAgent
from .orchestrator import AgentInput, AgentOutput
from .skills import add_skill_metadata


class Interest(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)


class RecipientProfileOutput(BaseModel):
    interests: list[Interest]
    communication_style: str
    gift_history_summary: str


class RecipientProfileResponse(BaseModel):
    output: RecipientProfileOutput
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None


class RecipientProfilingAgent(StructuredAgent):
    stage = "recipient_profiling"
    config_name = "recipient_profiling.json"

    def __init__(self, llm=None, *, retriever=None, prompt_versions_dir=None) -> None:
        self._explicit_llm = llm is not None
        super().__init__(llm, retriever=retriever, prompt_versions_dir=prompt_versions_dir)

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
                return self._run_with_schema_json(agent_input)
            return super().run(agent_input)

    def _run_with_instructor(self, agent_input: AgentInput) -> AgentOutput:
        import instructor
        from openai import OpenAI

        stage_config = agent_input.get("stage_config", {})
        context = self.build_context(agent_input)
        prompt = self.config["prompt_template"].format(context=json.dumps(context, ensure_ascii=False, indent=2))
        prompt += (
            "\n\nReturn JSON with this exact top-level shape and no class-name wrapper keys:\n"
            '{"output":{"interests":[{"name":"...","confidence":0.0}],"communication_style":"...","gift_history_summary":"..."},"confidence":0.0,"rationale":"..."}\n'
            "Each interests item must contain name and confidence directly. Do not emit Interest, _Interest, RecipientProfileOutput, or _RecipientProfileResponse keys."
        )
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
                {"role": "system", "content": self._system_prompt(agent_input)},
                {"role": "user", "content": prompt},
            ],
            response_model=RecipientProfileResponse,
            temperature=temperature,
        )
        payload = result.model_dump()
        try:
            validate(instance=payload, schema=self.config["output_schema"])
        except ValidationError as exc:
            raise ValueError(f"{self.stage} returned invalid structured output: {exc.message}") from exc
        output = dict(payload["output"])
        output.setdefault("prompt_version", self.prompt_version_id)
        add_skill_metadata(output, self.config)
        return AgentOutput(stage=self.stage, output=output, confidence=payload["confidence"], rationale=payload["rationale"])

    def _run_with_schema_json(self, agent_input: AgentInput) -> AgentOutput:
        try:
            result = super().run(agent_input)
            rationale = result.get("rationale")
            note = "instructor validation failed; used schema-validated Ollama JSON path"
            result["rationale"] = f"{note}; {rationale}" if rationale else note
            return result
        except Exception as schema_error:
            if os.getenv("GMGI_ALLOW_RECIPIENT_REPAIR", "1") != "1":
                raise
            return self._run_profile_repair(agent_input, schema_error)

    def _run_profile_repair(self, agent_input: AgentInput, schema_error: Exception) -> AgentOutput:
        context = self.build_context(agent_input)
        interests = []
        for item in context.get("preferences", []) or []:
            if isinstance(item, dict):
                name = item.get("value") or item.get("name") or item.get("category")
                confidence = float(item.get("confidence", 0.8) or 0.8)
            else:
                name = str(item)
                confidence = 0.8
            if name:
                interests.append({"name": str(name), "confidence": max(0.0, min(1.0, confidence))})
        notes = [str(note) for note in context.get("raw_notes", []) or [] if str(note).strip()]
        if not interests and notes:
            interests.append({"name": "memory-grounded personalization", "confidence": 0.65})
        output = {
            "interests": interests[:5],
            "communication_style": _communication_style_from_notes(notes),
            "gift_history_summary": _gift_history_summary(context.get("gift_history", []), notes),
            "prompt_version": self.prompt_version_id,
        }
        add_skill_metadata(output, self.config, active=["preference_signal_parser", "structured_recipient_extraction"])
        return AgentOutput(
            stage=self.stage,
            output=output,
            confidence=0.68,
            rationale=f"instructor and schema JSON profiling failed; repaired with preference_signal_parser; schema_error={type(schema_error).__name__}",
        )


def _communication_style_from_notes(notes: list[str]) -> str:
    text = " ".join(notes).lower()
    if any(word in text for word in ("professional", "colleague", "work", "launch")):
        return "Warm, concise, and professional."
    if any(word in text for word in ("laugh", "funny", "joke", "lost")):
        return "Warm, specific, and lightly playful."
    return "Warm, specific, and respectful."


def _gift_history_summary(gift_history: Any, notes: list[str]) -> str:
    if gift_history:
        return json.dumps(gift_history, ensure_ascii=False)
    if notes:
        return "No explicit gift history was supplied; use the provided memories as personalization evidence."
    return "No explicit gift history was supplied."




