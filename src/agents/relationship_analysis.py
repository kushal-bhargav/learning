from __future__ import annotations

import json
import os
import time
from typing import Any

from networkx.readwrite import json_graph

from jsonschema import ValidationError, validate

from .base import StructuredAgent
from .orchestrator import AgentInput, AgentOutput
from .skills import add_skill_metadata
from src.harness import ensure_tool_allowed, record_tool_call


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

    def __init__(self, llm=None, *, memory_graph=None, retriever=None, prompt_versions_dir=None) -> None:
        self._explicit_llm = llm is not None
        self.memory_graph = memory_graph
        super().__init__(llm, retriever=retriever, prompt_versions_dir=prompt_versions_dir)

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
        except Exception as exc:
            if os.getenv("GMGI_FORCE_OLLAMA_AGENTS") == "1" and os.getenv("GMGI_ALLOW_AGENT_FALLBACK") != "1":
                return self._run_with_schema_json(agent_input, exc)
            return super().run(agent_input)

    def _run_with_smolagents(self, agent_input: AgentInput) -> AgentOutput:
        from smolagents import LiteLLMModel, ToolCallingAgent, tool

        stage_config = agent_input.get("stage_config", {})
        context = self.build_context(agent_input)
        context_json = json.dumps(context, ensure_ascii=False, indent=2)

        @tool
        def query_memory_graph(query: str) -> str:
            """Return relationship, occasion, and supplied memory evidence relevant to the query."""
            started = time.perf_counter()
            ensure_tool_allowed("query_memory_graph")
            try:
                if self.memory_graph is not None:
                    person_id = str(stage_config.get("recipient_id") or stage_config.get("person_id") or "")
                    occasion_id = stage_config.get("occasion_id")
                    if person_id:
                        graph = self.memory_graph.subgraph_for(person_id, None if occasion_id is None else str(occasion_id))
                        result = json.dumps(json_graph.node_link_data(graph, edges="edges"), ensure_ascii=False, default=str)
                        record_tool_call("query_memory_graph", arguments={"query": query, "person_id": person_id, "occasion_id": occasion_id}, result=result, latency_seconds=time.perf_counter() - started)
                        return result
                lowered = query.lower()
                memories = context.get("memories", [])
                result = json.dumps(memories if "memory" in lowered or "evidence" in lowered else context, ensure_ascii=False)
                record_tool_call("query_memory_graph", arguments={"query": query}, result=result, latency_seconds=time.perf_counter() - started)
                return result
            except Exception as exc:
                record_tool_call("query_memory_graph", arguments={"query": query}, latency_seconds=time.perf_counter() - started, error=exc)
                raise

        model_name = str(stage_config.get("model") or os.getenv("GMGI_RELATIONSHIP_MODEL") or os.getenv("GMGI_OLLAMA_MODEL") or self.config["models"]["ollama"])
        model = LiteLLMModel(
            model_id=f"ollama_chat/{model_name}",
            api_base=os.getenv("GMGI_OLLAMA_HOST", "http://localhost:11434"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            num_ctx=int(os.getenv("GMGI_OLLAMA_NUM_CTX", "8192")),
            timeout=float(os.getenv("GMGI_OLLAMA_TIMEOUT_SECONDS", "30")),
        )
        agent = ToolCallingAgent(tools=[query_memory_graph], model=model, max_steps=int(stage_config.get("max_steps", 4)))
        prompt = (
            f"{self._system_prompt(agent_input)}\n\n"
            f"Analyze this JSON context using query_memory_graph when you need evidence:\n{context_json}\n\n"
            "Return only valid JSON matching this schema: "
            f"{json.dumps(self.config['output_schema'], ensure_ascii=False)}"
        )
        payload = _extract_json(agent.run(prompt))
        try:
            validate(instance=payload, schema=self.config["output_schema"])
        except ValidationError as exc:
            raise ValueError(f"{self.stage} returned invalid structured output: {exc.message}") from exc
        output = dict(payload["output"])
        output.setdefault("prompt_version", self.prompt_version_id)
        add_skill_metadata(output, self.config)
        return AgentOutput(stage=self.stage, output=output, confidence=payload["confidence"], rationale=payload["rationale"])

    def _run_with_schema_json(self, agent_input: AgentInput, original_error: Exception) -> AgentOutput:
        result = super().run(agent_input)
        rationale = result.get("rationale")
        note = f"smolagents tool-calling failed; used schema-validated Ollama JSON path; original_error={type(original_error).__name__}"
        result["rationale"] = f"{note}; {rationale}" if rationale else note
        return result




