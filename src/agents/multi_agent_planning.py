from __future__ import annotations

import os
from typing import Any, Mapping

from .base import StructuredAgent
from .orchestrator import AgentInput, AgentOutput

DEFAULT_AGENT_SEQUENCE = [
    "recipient_profiling",
    "relationship_analysis",
    "gift_intent_reasoning",
    "multi_agent_planning",
    "recommendation",
    "creative_generation",
    "greeting_story",
    "delivery_planner",
]


class MultiAgentPlanningAgent(StructuredAgent):
    """Bounded hybrid planner; existing orchestrator remains the executor."""

    stage = "multi_agent_planning"
    config_name = "multi_agent_planning.json"

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        return {
            "user_request": config.get("user_request", "create a personalized gift"),
            "recipient_profile": config.get("recipient_profile", {}),
            "relationship_guidance": config.get("relationship_guidance", {}),
            "intent": config.get("intent", {}),
            "memory_signals": config.get("memory_signals", {}),
            "available_agents": config.get("available_agents", DEFAULT_AGENT_SEQUENCE),
            "method": config.get("method", self.config.get("default_method", "rule_constrained")),
        }

    def run(self, agent_input: AgentInput) -> AgentOutput:
        method = str(agent_input.get("stage_config", {}).get("method") or self.config.get("default_method", "rule_constrained"))
        if method in {"fixed_pipeline", "rule_constrained", "task_decomposition", "plan_repair", "memory_augmented"}:
            return self._run_rule_planner(agent_input, method=method)
        try:
            return super().run(agent_input)
        except Exception as exc:
            if os.getenv("GMGI_ALLOW_PLANNING_REPAIR", "1") == "1":
                result = self._run_rule_planner(agent_input, method="plan_repair")
                rationale = result.get("rationale")
                note = f"llm_structured planning failed; repaired with bounded rule planner; original_error={type(exc).__name__}"
                result["rationale"] = f"{note}; {rationale}" if rationale else note
                return result
            if os.getenv("GMGI_FORCE_OLLAMA_AGENTS") == "1" and os.getenv("GMGI_ALLOW_AGENT_FALLBACK") != "1":
                raise
            return self._run_rule_planner(agent_input, method="fallback_fixed_pipeline")

    def _run_rule_planner(self, agent_input: AgentInput, *, method: str) -> AgentOutput:
        context = self.build_context(agent_input)
        intent = dict(context.get("intent") or {})
        clarifying = list(intent.get("clarifying_needs") or [])
        available = [str(agent) for agent in context.get("available_agents", DEFAULT_AGENT_SEQUENCE)]
        sequence = [agent for agent in DEFAULT_AGENT_SEQUENCE if agent in available]
        if method == "fixed_pipeline":
            sequence = [agent for agent in sequence if agent not in {"gift_intent_reasoning", "multi_agent_planning"}]
        ask_first = bool(clarifying) and method in {"rule_constrained", "plan_repair"}
        subtasks = []
        if ask_first:
            subtasks.append({"id": "clarify", "agent": "human", "action": "ask_clarifying_questions", "needs": clarifying})
        for index, agent in enumerate(sequence, start=1):
            subtasks.append({"id": f"step_{index}", "agent": agent, "action": _action_for(agent), "requires_human_review": True})
        dependencies = [
            {"after": sequence[index - 1], "before": sequence[index], "type": "stage_output"}
            for index in range(1, len(sequence))
        ]
        expected_outputs = [{"agent": agent, "output": _expected_output(agent)} for agent in sequence]
        output = {
            "task_goal": _task_goal(intent, str(context.get("user_request", "create a personalized gift"))),
            "subtasks": subtasks,
            "agent_sequence": sequence,
            "dependencies": dependencies,
            "expected_outputs": expected_outputs,
            "stop_conditions": [
                "human rejects or edits a pending proposal",
                "required model/checkpoint is unavailable and fallback cannot satisfy the stage",
                "delivery remains simulated; no external purchase or shipping action is allowed",
            ],
            "fallback_plan": {
                "type": "current_staged_orchestration",
                "agent_sequence": DEFAULT_AGENT_SEQUENCE,
                "reason": "Use the established bounded pipeline if planner output is incomplete or invalid.",
            },
        }
        confidence = 0.84 if not ask_first else 0.72
        return AgentOutput(
            stage=self.stage,
            output=output,
            confidence=confidence,
            rationale=f"{method} planner produced a bounded executable plan; orchestration remains external and auditable.",
        )


def _task_goal(intent: Mapping[str, Any], user_request: str) -> str:
    summary = intent.get("intent_summary") if isinstance(intent, Mapping) else None
    return str(summary or user_request or "Create a personalized gift workflow")


def _action_for(agent: str) -> str:
    return {
        "recipient_profiling": "extract recipient signals",
        "relationship_analysis": "model relationship tone and risk",
        "gift_intent_reasoning": "infer gift intent and constraints",
        "multi_agent_planning": "create bounded execution plan",
        "recommendation": "rank gift concepts",
        "creative_generation": "generate visual artifact",
        "greeting_story": "write original message/story",
        "delivery_planner": "simulate delivery timing",
    }.get(agent, "run stage")


def _expected_output(agent: str) -> str:
    return {
        "recipient_profiling": "interests, communication style, gift history summary",
        "relationship_analysis": "closeness, tone guidance, risk flags",
        "gift_intent_reasoning": "goal, constraints, preferences, clarifying needs",
        "multi_agent_planning": "subtasks, dependencies, stop conditions, fallback plan",
        "recommendation": "ranked gift concepts",
        "creative_generation": "image artifact metadata",
        "greeting_story": "original greeting text",
        "delivery_planner": "simulated delivery plan",
    }.get(agent, "structured output")
