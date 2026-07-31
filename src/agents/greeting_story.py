from __future__ import annotations

from typing import Any

from .base import StructuredAgent
from .orchestrator import AgentInput


class GreetingStoryAgent(StructuredAgent):
    stage = "greeting_story"
    config_name = "greeting_story.json"

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
