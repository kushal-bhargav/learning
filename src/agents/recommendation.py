from __future__ import annotations

import json
import os
from typing import Any

from jsonschema import ValidationError, validate

from .base import StructuredAgent
from .orchestrator import AgentInput, AgentOutput


def _extract_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("agent did not return a JSON object")
    return json.loads(text[start : end + 1])


class RecommendationAgent(StructuredAgent):
    stage = "recommendation"
    config_name = "recommendation.json"

    def __init__(self, llm=None) -> None:
        self._explicit_llm = llm is not None
        super().__init__(llm)

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        return {
            "recipient_profile": config["recipient_profile"],
            "relationship_guidance": config["relationship_guidance"],
            "occasion": config["occasion"],
            "preferences": config.get("preferences", []),
            "budget": config.get("budget", config["occasion"].get("budget_hint")),
        }

    def run(self, agent_input: AgentInput) -> AgentOutput:
        if self._explicit_llm and os.getenv("GMGI_FORCE_OLLAMA_AGENTS") != "1":
            return super().run(agent_input)
        try:
            return self._run_with_smolagents(agent_input)
        except Exception:
            if os.getenv("GMGI_FORCE_OLLAMA_AGENTS") == "1" and os.getenv("GMGI_ALLOW_AGENT_FALLBACK") != "1":
                raise
            return super().run(agent_input)

    def _run_with_smolagents(self, agent_input: AgentInput) -> AgentOutput:
        from smolagents import LiteLLMModel, ToolCallingAgent, tool

        stage_config = agent_input.get("stage_config", {})
        context = self.build_context(agent_input)
        context_json = json.dumps(context, ensure_ascii=False, indent=2)

        @tool
        def query_memory_graph(topic: str) -> str:
            """Return supplied recipient, relationship, occasion, preference, and budget context for a topic."""
            lowered = topic.lower()
            if "recipient" in lowered or "profile" in lowered:
                return json.dumps(context.get("recipient_profile", {}), ensure_ascii=False)
            if "relationship" in lowered or "tone" in lowered:
                return json.dumps(context.get("relationship_guidance", {}), ensure_ascii=False)
            if "preference" in lowered or "interest" in lowered:
                return json.dumps(context.get("preferences", []), ensure_ascii=False)
            return json.dumps(context, ensure_ascii=False)

        @tool
        def bandit_feedback_hint(category: str, agency_bucket: str = "mid") -> str:
            """Return a conservative simulated feedback hint for a recommendation category and agency bucket."""
            return json.dumps(
                {
                    "category": category,
                    "agency_bucket": agency_bucket,
                    "hint": "Prefer evidence-backed personalized/generated concepts; no live bandit state is queried here.",
                },
                ensure_ascii=False,
            )

        model_name = str(stage_config.get("model") or os.getenv("GMGI_RECOMMENDATION_MODEL") or os.getenv("GMGI_OLLAMA_MODEL") or self.config["models"]["ollama"])
        model = LiteLLMModel(
            model_id=f"ollama_chat/{model_name}",
            api_base=os.getenv("GMGI_OLLAMA_HOST", "http://localhost:11434"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            num_ctx=int(os.getenv("GMGI_OLLAMA_NUM_CTX", "8192")),
        )
        agent = ToolCallingAgent(
            tools=[query_memory_graph, bandit_feedback_hint],
            model=model,
            max_steps=int(stage_config.get("max_steps", 6)),
        )
        prompt = (
            f"{self.config['system_prompt']}\n\n"
            f"Rank exactly three gift concepts for this JSON context:\n{context_json}\n\n"
            "Use the tools for evidence and feedback hints when helpful. Return only valid JSON matching this schema: "
            f"{json.dumps(self.config['output_schema'], ensure_ascii=False)}"
        )
        payload = _extract_json(agent.run(prompt))
        try:
            validate(instance=payload, schema=self.config["output_schema"])
        except ValidationError as exc:
            raise ValueError(f"{self.stage} returned invalid structured output: {exc.message}") from exc
        return AgentOutput(stage=self.stage, output=payload["output"], confidence=payload["confidence"], rationale=payload["rationale"])