from __future__ import annotations

from typing import Any

from .base import StructuredAgent
from .orchestrator import AgentInput


class RecipientProfilingAgent(StructuredAgent):
    stage = "recipient_profiling"
    config_name = "recipient_profiling.json"

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        return {
            "person": config["person"],
            "raw_notes": config.get("raw_notes", []),
            "preferences": config.get("preferences", []),
            "gift_history": config.get("gift_history", []),
        }
