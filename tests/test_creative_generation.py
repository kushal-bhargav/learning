from pathlib import Path
from typing import Any

from PIL import Image

from src.agents.creative_generation import CreativeGenerationAgent
from src.agents.orchestrator import GiftSession


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
        },
    })
    artifact = Path(result["output"]["artifact_path"])
    assert result["stage"] == "creative_generation"
    assert artifact.is_file()
    assert result["output"]["agency_slider"] == 0.5
    assert gan.kwargs["seed"] == 7

