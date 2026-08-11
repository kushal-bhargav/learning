from __future__ import annotations

import json
import os
import time
from typing import Any

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


class RecommendationAgent(StructuredAgent):
    stage = "recommendation"
    config_name = "recommendation.json"

    def __init__(self, llm=None, *, bandit=None, retriever=None, prompt_versions_dir=None) -> None:
        self._explicit_llm = llm is not None
        self.bandit = bandit
        super().__init__(llm, retriever=retriever, prompt_versions_dir=prompt_versions_dir)

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        return {
            "recipient_profile": config["recipient_profile"],
            "relationship_guidance": config["relationship_guidance"],
            "gift_intent": config.get("gift_intent", {}),
            "execution_plan": config.get("execution_plan", {}),
            "occasion": config["occasion"],
            "preferences": config.get("preferences", []),
            "budget": config.get("budget", config["occasion"].get("budget_hint")),
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
        def query_memory_graph(topic: str) -> str:
            """Return supplied recipient, relationship, occasion, preference, and budget context for a topic."""
            started = time.perf_counter()
            ensure_tool_allowed("query_memory_graph")
            try:
                lowered = topic.lower()
                if "recipient" in lowered or "profile" in lowered:
                    result = json.dumps(context.get("recipient_profile", {}), ensure_ascii=False)
                elif "relationship" in lowered or "tone" in lowered:
                    result = json.dumps(context.get("relationship_guidance", {}), ensure_ascii=False)
                elif "preference" in lowered or "interest" in lowered:
                    result = json.dumps(context.get("preferences", []), ensure_ascii=False)
                else:
                    result = json.dumps(context, ensure_ascii=False)
                record_tool_call("query_memory_graph", arguments={"topic": topic}, result=result, latency_seconds=time.perf_counter() - started)
                return result
            except Exception as exc:
                record_tool_call("query_memory_graph", arguments={"topic": topic}, latency_seconds=time.perf_counter() - started, error=exc)
                raise

        @tool
        def bandit_feedback_hint(category: str, agency_bucket: str = "mid") -> str:
            """Return LinUCB feedback scores for compatible recommendation actions when bandit state is available."""
            started = time.perf_counter()
            ensure_tool_allowed("bandit_feedback_hint")
            try:
                if self.bandit is not None and stage_config.get("bandit_context") is not None:
                    scores = self.bandit.scores(stage_config["bandit_context"])
                    payload = [
                        {"action": action.__dict__, "ucb_score": score}
                        for action, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
                        if action.recommendation_category == category or action.agency_bucket == agency_bucket
                    ]
                    result = json.dumps({"source": "linucb", "scores": payload[:5]}, ensure_ascii=False)
                else:
                    result = json.dumps(
                        {
                            "category": category,
                            "agency_bucket": agency_bucket,
                            "hint": "No live bandit state is available yet; prefer evidence-backed personalized/generated concepts.",
                        },
                        ensure_ascii=False,
                    )
                record_tool_call("bandit_feedback_hint", arguments={"category": category, "agency_bucket": agency_bucket}, result=result, latency_seconds=time.perf_counter() - started)
                return result
            except Exception as exc:
                record_tool_call("bandit_feedback_hint", arguments={"category": category, "agency_bucket": agency_bucket}, latency_seconds=time.perf_counter() - started, error=exc)
                raise

        model_name = str(stage_config.get("model") or os.getenv("GMGI_RECOMMENDATION_MODEL") or os.getenv("GMGI_OLLAMA_MODEL") or self.config["models"]["ollama"])
        model = LiteLLMModel(
            model_id=f"ollama_chat/{model_name}",
            api_base=os.getenv("GMGI_OLLAMA_HOST", "http://localhost:11434"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            num_ctx=int(os.getenv("GMGI_OLLAMA_NUM_CTX", "8192")),
            timeout=float(os.getenv("GMGI_OLLAMA_TIMEOUT_SECONDS", "30")),
        )
        max_steps = int(os.getenv("GMGI_RECOMMENDATION_MAX_STEPS", str(stage_config.get("max_steps", 3))))
        agent = ToolCallingAgent(
            tools=[query_memory_graph, bandit_feedback_hint],
            model=model,
            max_steps=max_steps,
        )
        prompt = (
            f"{self._system_prompt(agent_input)}\n\n"
            f"Rank exactly three gift concepts for this JSON context:\n{context_json}\n\n"
            "Use each tool at most once, only if it changes the answer. Keep concepts concise. "
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
        if os.getenv("GMGI_ALLOW_RECOMMENDATION_REPAIR", "1") == "1":
            return self._run_ranked_repair(agent_input, original_error, None)
        try:
            result = super().run(agent_input)
            rationale = result.get("rationale")
            note = f"smolagents tool-calling failed; used schema-validated Ollama JSON path; original_error={type(original_error).__name__}"
            result["rationale"] = f"{note}; {rationale}" if rationale else note
            return result
        except Exception as schema_error:
            if os.getenv("GMGI_ALLOW_RECOMMENDATION_REPAIR", "1") != "1":
                raise
            return self._run_ranked_repair(agent_input, original_error, schema_error)

    def _run_ranked_repair(self, agent_input: AgentInput, original_error: Exception, schema_error: Exception | None) -> AgentOutput:
        context = self.build_context(agent_input)
        profile = context.get("recipient_profile", {}) if isinstance(context.get("recipient_profile"), dict) else {}
        intent = context.get("gift_intent", {}) if isinstance(context.get("gift_intent"), dict) else {}
        occasion = context.get("occasion", {}) if isinstance(context.get("occasion"), dict) else {}
        preferences = _preference_labels(context.get("preferences", []), profile, intent)
        evidence = preferences[:2] or [str(occasion.get("name") or "occasion context")]
        budget = str(context.get("budget") or occasion.get("budget_hint") or "the stated budget")
        visual = intent.get("visual_generation", {}) if isinstance(intent.get("visual_generation"), dict) else {}
        goal = intent.get("goal", {}) if isinstance(intent.get("goal"), dict) else {}
        artifact_hint = str(visual.get("artifact_type") or goal.get("recommended_artifact_type") or "greeting_card").replace("_", " ")
        primary_theme = ", ".join(preferences[:3]) if preferences else str(goal.get("emotional_objective") or "the supplied relationship context")
        recommendations = [
            {
                "rank": 1,
                "category": "generated keepsake",
                "concept": f"A personalized {artifact_hint} built around {primary_theme}.",
                "evidence": evidence,
                "budget_fit": f"Digital generation can stay within {budget}.",
                "artifact_type": "generated",
            },
            {
                "rank": 2,
                "category": "curated experience bundle",
                "concept": f"A small themed bundle that echoes {preferences[0] if preferences else 'the occasion'} and includes a generated note.",
                "evidence": evidence[:1],
                "budget_fit": f"Bundle size can be adjusted to {budget}.",
                "artifact_type": "bundle",
            },
            {
                "rank": 3,
                "category": "physical keepsake",
                "concept": f"A modest physical item paired with memory-grounded wording for {occasion.get('name', 'the occasion')}.",
                "evidence": evidence,
                "budget_fit": f"Choose a simple object so the total remains near {budget}.",
                "artifact_type": "physical",
            },
        ]
        output = {
            "recommendations": recommendations,
            "prompt_version": self.prompt_version_id,
        }
        add_skill_metadata(output, self.config, active=["ranked_gift_reasoning", "memory_context_lookup"])
        return AgentOutput(
            stage=self.stage,
            output=output,
            confidence=0.72,
            rationale=(
                "smolagents recommendation path failed; repaired with deterministic ranked_gift_reasoning "
                f"over supplied context; original_error={type(original_error).__name__}"
                + ("" if schema_error is None else f"; schema_error={type(schema_error).__name__}")
            ),
        )


def _preference_labels(preferences: Any, profile: dict[str, Any], intent: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for source in (preferences, profile.get("interests", []), intent.get("preferences", [])):
        for item in source or []:
            if isinstance(item, dict):
                value = item.get("value") or item.get("name") or item.get("category")
            else:
                value = item
            text = str(value or "").strip()
            if text and text not in labels:
                labels.append(text)
    return labels







