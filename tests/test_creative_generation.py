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
    assert "optional_memorygan_checkpoint_inference" in result["output"]["skills_used"]
    assert result["output"]["skills_declared"] == [
        "visual_prompt_builder",
        "diffusers_image_generation",
        "optional_memorygan_checkpoint_inference",
        "clip_critic",
    ]
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
    assert "You are the Creative Generation agent" not in prompt
    assert len(prompt.split()) <= 60


def test_creative_prompt_template_uses_artifact_specific_subjects() -> None:
    agent = CreativeGenerationAgent()
    card_prompt = agent._diffusers_prompt({
        "relationship_type": "partner",
        "emotion_tag": "nostalgia",
        "occasion": "birthday",
        "agency_slider": 0.35,
        "gift_artifact_type": "greeting_card",
        "visual_style_prompt": "ceramics, green, handwritten notes, tiny tea shop doorway",
    })
    assert "main subject: folded greeting card or card front" in card_prompt
    assert "clear card border" in card_prompt
    assert "no readable private words" in card_prompt
    assert "You are the Creative Generation agent" not in card_prompt

    mug_prompt = agent._diffusers_prompt({
        "relationship_type": "sibling",
        "emotion_tag": "gratitude",
        "occasion": "housewarming",
        "agency_slider": 0.8,
        "gift_artifact_type": "mug",
        "visual_style_prompt": "mustard yellow, urban sketching, shared coffee memory",
    })
    assert "main subject: giftable ceramic mug" in mug_prompt
    assert "single clear mug silhouette" in mug_prompt
    assert "polished personalized gift object" in mug_prompt


def test_default_negative_prompt_does_not_exclude_valid_gift_objects() -> None:
    negative_prompt = CreativeGenerationAgent.default_negative_prompt()
    assert "mug" not in negative_prompt
    assert "cup" not in negative_prompt
    assert "product photo" not in negative_prompt
    assert "watermark" in negative_prompt
