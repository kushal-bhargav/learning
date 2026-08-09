from pathlib import Path
from typing import Any

from PIL import Image

from src.agents.creative_generation import CreativeGenerationAgent
from src.agents.orchestrator import GiftSession


class FakeRetriever:
    def augment_system_prompt(self, agent_name: str, base_prompt: str, context_fp: str) -> str:
        return base_prompt + "\nretrieved creative example"


class FakeGAN:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> Image.Image:
        self.kwargs = kwargs
        return Image.new("RGB", (16, 16), "gold")


def test_creative_generation_wraps_memory_gan_and_saves_artifact() -> None:
    session = GiftSession(
        session_id="session-test",
        giver_id="person-maya",
        recipient_id="person-jordan",
        occasion_id="occasion-jordan-birthday-2026",
    )
    gan = FakeGAN()
    result = CreativeGenerationAgent(gan).run({
        "session": session,
        "stage_config": {
            "context_embedding": [0.0] * 8,
            "relationship_type": "partner",
            "emotion_tag": "joy",
            "occasion": "birthday",
            "agency_slider": 0.5,
            "human_style_ref": [1.0] * 8,
            "seed": 7,
            "output_dir": "experiments/test-generated",
            "filename": "creative-agent-test.png",
            "context_fingerprint": "partner|casual|high|test",
        },
    })
    artifact = Path(result["output"]["artifact_path"])
    assert result["stage"] == "creative_generation"
    assert artifact.is_file()
    assert result["output"]["agency_slider"] == 0.5
    assert result["output"]["clip_score"] is None
    assert result["output"]["critique_retries"] == 0
    assert result["output"]["prompt_version"] == "static"
    assert "visual_prompt_builder" in result["output"]["skills_used"]
    assert gan.kwargs["seed"] == 7



def test_creative_prompt_uses_retriever_for_diffusers_prompt() -> None:
    agent = CreativeGenerationAgent(FakeGAN(), retriever=FakeRetriever())
    prompt = agent._diffusers_prompt({
        "relationship_type": "friend",
        "emotion_tag": "joy",
        "occasion": "birthday",
        "agency_slider": 0.7,
        "context_fingerprint": "friend|casual|high|test",
    })
    assert "retrieved creative example" in prompt
    assert "Visual prompt:" in prompt
