from __future__ import annotations

from typing import Any

from .base import StructuredAgent
from .orchestrator import AgentInput


class RecommendationAgent(StructuredAgent):
    stage = "recommendation"
    config_name = "recommendation.json"

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        return {
            "recipient_profile": config["recipient_profile"],
            "relationship_guidance": config["relationship_guidance"],
            "occasion": config["occasion"],
            "preferences": config.get("preferences", []),
            "budget": config.get("budget", config["occasion"].get("budget_hint")),
        }
