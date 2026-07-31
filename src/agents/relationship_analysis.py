from __future__ import annotations

from typing import Any

from .base import StructuredAgent
from .orchestrator import AgentInput


class RelationshipAnalysisAgent(StructuredAgent):
    stage = "relationship_analysis"
    config_name = "relationship_analysis.json"

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
