from pathlib import Path
from uuid import uuid4

from src.agents.experience_store import Episode, ExperienceStore
from src.agents.llm import LLMProvider
from src.agents.prompt_optimizer import PromptOptimizerAgent


class FakeLLM:
    provider = LLMProvider.OLLAMA

    def generate(self, **kwargs):
        return {
            "output": {
                "system_prompt": "You are a gift recommendation agent that uses evidence carefully, asks for missing constraints, avoids invented private facts, preserves structured JSON output, ranks practical personalized ideas, explains uncertainty, keeps tone warm, respects simulated delivery boundaries, grounds every recommendation in supplied memories or preferences, avoids purchases or shipment claims, and clearly separates creative artifacts from physical gift suggestions."
            },
            "confidence": 0.8,
            "rationale": "test",
        }


def test_prompt_optimizer_writes_version_when_agent_underperforms():
    config_dir = Path("experiments/test-prompt-optimizer") / uuid4().hex / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "recommendation.json").write_text('{"system_prompt":"You are a gift recommendation agent with structured JSON output."}', encoding="utf-8")
    store = ExperienceStore([
        Episode("s1", "now", "partner|casual|high", {"recommendation": {"x": 1}}, {"recommendation": "regenerate"}, 0.2),
        Episode("s2", "now", "partner|casual|high", {"recommendation": {"x": 2}}, {"recommendation": "edit"}, 0.3),
        Episode("s3", "now", "partner|casual|high", {"recommendation": {"x": 3}}, {"recommendation": "accept"}, 0.9),
    ])
    result = PromptOptimizerAgent(FakeLLM(), store, config_dir).run()
    assert "recommendation" in result
    assert (config_dir / "prompt_versions" / "recommendation" / "latest.json").exists()
