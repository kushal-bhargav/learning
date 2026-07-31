from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import (
    AgentOrchestrator,
    CreativeGenerationAgent,
    DeliveryPlannerAgent,
    GiftSession,
    GreetingStoryAgent,
    LLMProvider,
    RecipientProfilingAgent,
    RecommendationAgent,
    RelationshipAnalysisAgent,
)
from src.memory_graph.fixtures import load_fixture


class FixtureLLM:
    provider = LLMProvider.OLLAMA

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def generate(self, **_: Any) -> dict[str, Any]:
        return self.response


def padded(vector: np.ndarray, size: int) -> np.ndarray:
    return np.pad(vector, (0, max(0, size - vector.size)))[:size].astype(np.float32)


def append(service: AgentOrchestrator, agent: Any, stage_config: dict[str, Any]) -> None:
    service.append_agent_output(agent.run({"session": service.session, "stage_config": stage_config}))


def run_pipeline(fixture_path: Path, checkpoint: Path, output_path: Path) -> GiftSession:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    giver = next(person for person in data["people"] if person["role"] == "giver")
    recipient = next(person for person in data["people"] if person["role"] == "recipient")
    relationship = data["relationships"][0]
    occasion = data["occasions"][0]
    service = AgentOrchestrator(GiftSession(
        session_id="fixture-long-distance-partners-2026",
        giver_id=giver["id"],
        recipient_id=recipient["id"],
        occasion_id=occasion["id"],
    ))

    profile_response = {
        "output": {
            "interests": [
                {"name": "urban sketching", "confidence": 1.0},
                {"name": "travel-poster illustration", "confidence": 0.72},
                {"name": "mustard yellow", "confidence": 1.0},
            ],
            "communication_style": "Warm, playful, and memory-rich.",
            "gift_history_summary": "No prior gift history was supplied.",
        },
        "confidence": 0.9,
        "rationale": "Uses Jordan's stated and explicitly marked inferred preferences.",
    }
    append(service, RecipientProfilingAgent(FixtureLLM(profile_response)), {
        "person": recipient,
        "preferences": data["preferences"],
        "raw_notes": [memory["content"] for memory in data["memories"]],
    })
    service.apply_human_action("delegate")

    relationship_response = {
        "output": {
            "closeness_assessment": "very close",
            "tone_guidance": "Affectionate and playful without inventing private details.",
            "formality": "casual",
            "risk_flags": [],
            "agency_slider_default": 0.5,
        },
        "confidence": 0.95,
        "rationale": "The fixture gives a partner relationship with closeness 5/5.",
    }
    append(service, RelationshipAnalysisAgent(FixtureLLM(relationship_response)), {
        "relationship": relationship,
        "memories": data["memories"],
        "occasion": occasion,
    })

    recommendation_response = {
        "output": {"recommendations": [
            {
                "rank": 1,
                "category": "personalized art",
                "concept": "A mustard-yellow Lisbon travel-poster birthday card featuring a tram and shared-window motif.",
                "evidence": ["urban sketching", "mustard yellow", "Lisbon tram memory"],
                "budget_fit": "Digital generation fits the stated budget.",
                "artifact_type": "generated",
            },
            {
                "rank": 2,
                "category": "sketching",
                "concept": "A compact urban-sketching travel kit.",
                "evidence": ["urban sketching"],
                "budget_fit": "Can be sourced within USD 60-100.",
                "artifact_type": "physical",
            },
            {
                "rank": 3,
                "category": "shared experience",
                "concept": "A remote dinner-and-drawing birthday bundle.",
                "evidence": ["video dinner memory", "urban sketching"],
                "budget_fit": "Flexible within the stated budget.",
                "artifact_type": "bundle",
            },
        ]},
        "confidence": 0.91,
        "rationale": "The ranking connects explicit preferences and shared memories to the birthday.",
    }
    append(service, RecommendationAgent(FixtureLLM(recommendation_response)), {
        "recipient_profile": service.effective_output("recipient_profiling"),
        "relationship_guidance": service.effective_output("relationship_analysis"),
        "occasion": occasion,
        "preferences": data["preferences"],
    })

    graph = load_fixture(fixture_path)
    creative = CreativeGenerationAgent.from_checkpoint(str(checkpoint))
    context = padded(graph.context_embedding(recipient["id"], occasion["id"]), creative.gan.config.context_dim)
    human_style = padded(np.asarray(data["memories"][0]["embedding"]), creative.gan.config.context_dim)
    append(service, creative, {
        "context_embedding": context,
        "relationship_type": relationship["type"],
        "emotion_tag": data["memories"][0]["emotion_tag"],
        "occasion": "birthday",
        "agency_slider": 0.5,
        "human_style_ref": human_style,
        "seed": 2026,
        "output_dir": "experiments/generated",
        "filename": "fixture-long-distance-partners-2026.png",
    })

    greeting_response = {
        "output": {
            "message": "Happy birthday, Jordan. From Lisbon's yellow tram to our wonderfully mismatched video dinners, every shared detour feels like home. Here's to the next view we sketch together.",
            "memory_references": ["memory-lisbon-tram", "memory-video-dinner"],
            "tone": "warm and playful",
        },
        "confidence": 0.94,
        "rationale": "Uses two supplied memories and follows the close-partner tone guidance.",
    }
    append(service, GreetingStoryAgent(FixtureLLM(greeting_response)), {
        "relationship_guidance": service.effective_output("relationship_analysis"),
        "occasion": occasion,
        "memories": data["memories"],
        "tone_guidance": "warm and playful",
        "giver_name": giver["display_name"],
        "recipient_name": recipient["display_name"],
    })
    append(service, DeliveryPlannerAgent(), {
        "artifact_type": "generated",
        "occasion": occasion,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(service.session.model_dump_json(indent=2), encoding="utf-8")
    return service.session


def main() -> None:
    session = run_pipeline(
        Path("data/fixtures/long_distance_partners.json"),
        Path("experiments/run-002/checkpoint-000200.pt"),
        Path("experiments/fixture_pipeline_session.json"),
    )
    print(session.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

