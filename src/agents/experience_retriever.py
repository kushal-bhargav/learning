from __future__ import annotations

import json
from typing import Any

from .experience_store import Episode, ExperienceStore


class ExperienceRetriever:
    """Inject successful similar past episodes into agent prompts as compact few-shot guidance."""

    def __init__(self, store: ExperienceStore, *, top_k: int = 2) -> None:
        self.store = store
        self.top_k = int(top_k)

    def augment_system_prompt(self, agent_name: str, base_prompt: str, context_fp: str) -> str:
        examples = [
            episode
            for episode in self.store.retrieve_similar(context_fp, top_k=self.top_k * 3)
            if episode.human_actions.get(agent_name) == "accept" and episode.composite_reward >= 0.6
        ][: self.top_k]
        if not examples:
            return base_prompt
        return base_prompt + "\n\n" + self._format_few_shot(agent_name, examples)

    def _format_few_shot(self, agent_name: str, episodes: list[Episode]) -> str:
        lines = [
            "Successful prior examples for similar gifting contexts:",
            "Use these as light guidance only; preserve the current output schema and do not copy private details.",
        ]
        for index, episode in enumerate(episodes, start=1):
            output = episode.agent_outputs.get(agent_name, {})
            compact = _compact(output)
            lines.append(f"Example {index} reward={episode.composite_reward:.2f}: {compact}")
        return "\n".join(lines)


def _compact(value: Any, limit: int = 900) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."
