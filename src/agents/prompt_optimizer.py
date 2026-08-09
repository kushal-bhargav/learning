from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .experience_store import Episode, ExperienceStore
from .llm import StructuredLLM


class PromptOptimizerAgent:
    """Meta-agent that versions system prompts from low-performing session evidence."""

    TRIGGER_INTERVAL = 5
    MIN_EPISODES_NEEDED = 3
    IMPROVEMENT_THRESHOLD = 0.5

    def __init__(self, llm: StructuredLLM, store: ExperienceStore, config_dir: str | Path) -> None:
        self.llm = llm
        self.store = store
        self.config_dir = Path(config_dir)
        self.prompt_versions_dir = Path(
            self.config_dir if self.config_dir.name == "prompt_versions" else self.config_dir / "prompt_versions"
        )

    def maybe_run(self, session_count: int) -> dict[str, str] | None:
        if session_count % self.TRIGGER_INTERVAL != 0:
            return None
        if len(self.store.episodes) < self.MIN_EPISODES_NEEDED:
            return None
        return self.run()

    def run(self) -> dict[str, str]:
        episodes = self.store.recent(n=20)
        if len(episodes) < self.MIN_EPISODES_NEEDED:
            return {}
        rates = self._acceptance_rates(episodes)
        updated: dict[str, str] = {}
        for agent_name, rate in rates.items():
            if rate >= self.IMPROVEMENT_THRESHOLD:
                continue
            new_prompt = self._rewrite_prompt(agent_name, episodes)
            if new_prompt:
                version = self._save_versioned_prompt(agent_name, new_prompt, rate)
                updated[agent_name] = version
        return updated

    def _acceptance_rates(self, episodes: list[Episode]) -> dict[str, float]:
        totals: dict[str, int] = {}
        accepts: dict[str, int] = {}
        for episode in episodes:
            for agent, action in episode.human_actions.items():
                totals[agent] = totals.get(agent, 0) + 1
                if action == "accept":
                    accepts[agent] = accepts.get(agent, 0) + 1
        return {agent: accepts.get(agent, 0) / total for agent, total in totals.items() if total}

    def _rewrite_prompt(self, agent_name: str, episodes: list[Episode]) -> str | None:
        current = self._current_prompt(agent_name)
        if not current:
            return None
        failures = [episode for episode in episodes if episode.human_actions.get(agent_name) in {"edit", "regenerate"}]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["output", "confidence", "rationale"],
            "properties": {
                "output": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["system_prompt"],
                    "properties": {"system_prompt": {"type": "string"}},
                },
                "confidence": {"type": ["number", "null"]},
                "rationale": {"type": ["string", "null"]},
            },
        }
        user = (
            "Rewrite this GMGI agent system prompt to reduce repeated edits/regenerations. "
            "Preserve the agent role, conservative factual behavior, and existing output schema. "
            "Return only the improved system_prompt.\n\n"
            f"Agent: {agent_name}\nCurrent prompt:\n{current}\n\nFailure evidence:\n{self._failure_summary(agent_name, failures)}"
        )
        try:
            result = self.llm.generate(
                system_prompt="You are a careful prompt optimizer for a structured multi-agent gift intelligence system.",
                user_prompt=user,
                schema=schema,
                temperature=0.2,
                model=self._optimizer_model(),
            )
            prompt = str(result.get("output", {}).get("system_prompt", "")).strip()
        except Exception:
            return None
        return prompt if self._valid_prompt(agent_name, prompt) else None

    def _save_versioned_prompt(self, agent_name: str, prompt: str, acceptance_rate: float) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        old_prompt = self._current_prompt(agent_name) or ""
        version_id = f"{timestamp}-{hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:8]}"
        payload = {
            "version_id": version_id,
            "agent": agent_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "system_prompt": prompt,
            "acceptance_rate_before": acceptance_rate,
        }
        target_dir = self.prompt_versions_dir / agent_name
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{version_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (target_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.prompt_versions_dir.mkdir(parents=True, exist_ok=True)
        log_record = {
            "timestamp": payload["created_at"],
            "agent": agent_name,
            "old_hash": hashlib.sha1(old_prompt.encode("utf-8")).hexdigest()[:12],
            "new_hash": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
            "justification": f"acceptance_rate={acceptance_rate:.2f} below threshold {self.IMPROVEMENT_THRESHOLD:.2f}",
        }
        with (self.prompt_versions_dir / "PROMPT_CHANGE_LOG.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record, sort_keys=True) + "\n")
        return version_id

    def _current_prompt(self, agent_name: str) -> str | None:
        latest = self.prompt_versions_dir / agent_name / "latest.json"
        if latest.exists():
            payload = json.loads(latest.read_text(encoding="utf-8"))
            prompt = payload.get("system_prompt") or payload.get("prompt")
            if isinstance(prompt, str):
                return prompt
        static = self.config_dir / f"{agent_name}.json"
        if static.exists():
            payload = json.loads(static.read_text(encoding="utf-8"))
            prompt = payload.get("system_prompt")
            if isinstance(prompt, str):
                return prompt
        return None

    def _failure_summary(self, agent_name: str, episodes: list[Episode]) -> str:
        rows: list[Mapping[str, Any]] = []
        for episode in episodes[-8:]:
            rows.append({
                "reward": episode.composite_reward,
                "action": episode.human_actions.get(agent_name),
                "output": episode.agent_outputs.get(agent_name),
            })
        return json.dumps(rows, ensure_ascii=False, default=str)[:4000]

    def _optimizer_model(self) -> str:
        import os

        return os.getenv("GMGI_PROMPT_OPTIMIZER_MODEL") or os.getenv("GMGI_OLLAMA_MODEL") or "llama3.2:latest"

    @staticmethod
    def _valid_prompt(agent_name: str, prompt: str) -> bool:
        words = prompt.split()
        if not 50 <= len(words) <= 500:
            return False
        return agent_name.replace("_", " ").split()[0].lower() in prompt.lower() or "gift" in prompt.lower()
