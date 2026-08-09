from pathlib import Path

from src.agents.experience_retriever import ExperienceRetriever
from src.agents.experience_store import Episode, ExperienceStore, context_fingerprint


def test_experience_store_roundtrip_and_similarity():
    path = Path("experiments/test-experience-store/experience.jsonl")
    if path.exists():
        path.unlink()
    fp = context_fingerprint({"type": "partner", "closeness_score": 5}, {"formality": "casual"})
    store = ExperienceStore.load(path)
    episode = Episode(
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        context_fingerprint=fp,
        agent_outputs={"recommendation": {"concept": "card"}},
        human_actions={"recommendation": "accept"},
        composite_reward=0.9,
        clip_score=0.7,
        prompt_versions={"recommendation": "static"},
    )
    store.append(episode)

    loaded = ExperienceStore.load(path)
    assert loaded.recent(n=1)[0].session_id == "s1"
    assert loaded.retrieve_similar(fp, top_k=1)[0].composite_reward == 0.9


def test_experience_retriever_augments_only_successful_accepts():
    fp = "partner|casual|high|abc"
    store = ExperienceStore([
        Episode("s1", "now", fp, {"greeting_story": {"message": "warm"}}, {"greeting_story": "accept"}, 0.8),
        Episode("s2", "now", fp, {"greeting_story": {"message": "bad"}}, {"greeting_story": "regenerate"}, 0.2),
    ])
    prompt = ExperienceRetriever(store).augment_system_prompt("greeting_story", "Base prompt", fp)
    assert "Base prompt" in prompt
    assert "Successful prior examples" in prompt
    assert "warm" in prompt
    assert "bad" not in prompt
