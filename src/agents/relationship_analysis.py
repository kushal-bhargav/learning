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


class RelationshipAnalysisAgent(StructuredAgent):
    stage = "relationship_analysis"
    config_name = "relationship_analysis.json"

    def __init__(self, llm=None) -> None:
        self._explicit_llm = llm is not None
        super().__init__(llm)

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        memories = [
            {key: value for key, value in memory.items() if key != "embedding"}
            for memory in config.get("memories", [])
        ]
        return {
            "relationship": config["relationship"],
            "memories": memories,
            "occasion": config.get("occasion"),
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
        def query_memory_graph(query: str) -> str:
            """Return relationship, occasion, and supplied memory evidence relevant to the query."""
            lowered = query.lower()
            memories = context.get("memories", [])
            if "memory" in lowered or "evidence" in lowered:
                return json.dumps(memories, ensure_ascii=False)
            return json.dumps(context, ensure_ascii=False)

        model_name = str(stage_config.get("model") or os.getenv("GMGI_RELATIONSHIP_MODEL") or os.getenv("GMGI_OLLAMA_MODEL") or self.config["models"]["ollama"])
        model = LiteLLMModel(
            model_id=f"ollama_chat/{model_name}",
            api_base=os.getenv("GMGI_OLLAMA_HOST", "http://localhost:11434"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            num_ctx=int(os.getenv("GMGI_OLLAMA_NUM_CTX", "8192")),
        )
        agent = ToolCallingAgent(tools=[query_memory_graph], model=model, max_steps=int(stage_config.get("max_steps", 4)))
        prompt = (
            f"{self.config['system_prompt']}\n\n"
            f"Analyze this JSON context using query_memory_graph when you need evidence:\n{context_json}\n\n"
            "Return only valid JSON matching this schema: "
            f"{json.dumps(self.config['output_schema'], ensure_ascii=False)}"
        )
        payload = _extract_json(agent.run(prompt))
        try:
            validate(instance=payload, schema=self.config["output_schema"])
        except ValidationError as exc:
            raise ValueError(f"{self.stage} returned invalid structured output: {exc.message}") from exc
        return AgentOutput(stage=self.stage, output=payload["output"], confidence=payload["confidence"], rationale=payload["rationale"])