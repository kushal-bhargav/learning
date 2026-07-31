from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from src.agents import (
    AgentOrchestrator,
    CreativeGenerationAgent,
    DeliveryPlannerAgent,
    GiftSession,
    GreetingStoryAgent,
    HumanAction,
    LLMProvider,
    RecipientProfilingAgent,
    RecommendationAgent,
    RelationshipAnalysisAgent,
)
from src.agents.orchestrator import AgentOutput
from src.memory_graph.fixtures import load_fixture
from src.rl import BanditAction, ContextEncoder, LinUCBBandit, log_session, reward_from_feedback


STAGES = (
    "recipient_profiling",
    "relationship_analysis",
    "recommendation",
    "creative_generation",
    "greeting_story",
    "delivery_planner",
)


class DemoStructuredLLM:
    """Deterministic fixture LLM for the poster/study demo backend."""

    provider = LLMProvider.OLLAMA

    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = dict(response)

    def generate(self, **_: Any) -> dict[str, Any]:
        return json.loads(json.dumps(self.response))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fixture_paths(root: Path = Path("data/fixtures")) -> dict[str, Path]:
    return {
        _load_json(path)["persona_id"]: path
        for path in sorted(root.glob("*.json"))
    }


def _padded(vector: Any, size: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    return np.pad(values, (0, max(0, size - values.size)))[:size].astype(np.float32)


def _agency_bucket(value: float) -> str:
    if value < 1 / 3:
        return "low"
    if value < 2 / 3:
        return "mid"
    return "high"


@dataclass
class ConsoleSession:
    orchestrator: AgentOrchestrator
    fixture: dict[str, Any]
    fixture_path: Path
    agency_slider: float
    seed: int
    budget_hint: str | None = None


class AgencyConsoleService:
    """In-memory application service backing the Agency Console FastAPI routes."""

    def __init__(
        self,
        *,
        fixture_root: str | Path = "data/fixtures",
        checkpoint_path: str | Path = "experiments/run-002/checkpoint-000200.pt",
        generated_dir: str | Path = "experiments/generated",
        bandit_log_path: str | Path = "experiments/bandit_log.jsonl",
        bandit_state_path: str | Path = "experiments/bandit_state.json",
    ) -> None:
        self.fixture_root = Path(fixture_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.generated_dir = Path(generated_dir)
        self.bandit_log_path = Path(bandit_log_path)
        self.bandit_state_path = Path(bandit_state_path)
        self.sessions: dict[str, ConsoleSession] = {}
        self._creative_agent: CreativeGenerationAgent | None = None
        self._encoder = ContextEncoder()

    def list_personas(self) -> list[dict[str, Any]]:
        personas = []
        for persona_id, path in _fixture_paths(self.fixture_root).items():
            data = _load_json(path)
            personas.append(
                {
                    "persona_id": persona_id,
                    "label": data.get("label", persona_id),
                    "synthetic": data.get("synthetic", True),
                    "occasions": data.get("occasions", []),
                }
            )
        return personas

    def create_session(
        self,
        *,
        persona_id: str,
        occasion_id: str | None = None,
        budget_hint: str | None = None,
        agency_slider: float | None = None,
        seed: int = 2026,
    ) -> GiftSession:
        fixtures = _fixture_paths(self.fixture_root)
        if persona_id not in fixtures:
            raise KeyError(f"Unknown fixture persona: {persona_id}")
        fixture_path = fixtures[persona_id]
        fixture = _load_json(fixture_path)
        giver = self._person(fixture, "giver")
        recipient = self._person(fixture, "recipient")
        occasion = self._occasion(fixture, occasion_id)
        session_id = f"{persona_id}-{uuid4().hex[:8]}"
        default_agency = 0.5 if agency_slider is None else float(agency_slider)
        if not 0.0 <= default_agency <= 1.0:
            raise ValueError("agency_slider must be between 0 and 1")
        session = GiftSession(
            session_id=session_id,
            giver_id=giver["id"],
            recipient_id=recipient["id"],
            occasion_id=occasion["id"],
        )
        self.sessions[session_id] = ConsoleSession(
            orchestrator=AgentOrchestrator(session),
            fixture=fixture,
            fixture_path=fixture_path,
            agency_slider=default_agency,
            seed=int(seed),
            budget_hint=budget_hint,
        )
        return session

    def get_session(self, session_id: str) -> GiftSession:
        return self._console(session_id).orchestrator.session

    def propose(self, session_id: str, stage: str, overrides: Mapping[str, Any] | None = None) -> GiftSession:
        if stage not in STAGES:
            raise KeyError(f"Unknown stage: {stage}")
        console = self._console(session_id)
        result = self._run_stage(console, stage, dict(overrides or {}))
        return console.orchestrator.append_agent_output(result)

    def accept(self, session_id: str) -> GiftSession:
        return self._console(session_id).orchestrator.apply_human_action(HumanAction.ACCEPT)

    def edit(self, session_id: str, human_edit: Mapping[str, Any]) -> GiftSession:
        return self._console(session_id).orchestrator.apply_human_action(
            HumanAction.EDIT,
            human_edit=human_edit,
        )

    def regenerate(
        self,
        session_id: str,
        *,
        stage: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> GiftSession:
        console = self._console(session_id)
        pending = console.orchestrator.session.stage_log[-1]
        target_stage = stage or pending.stage
        if target_stage != pending.stage:
            raise ValueError("Regenerate must target the currently pending stage")
        console.orchestrator.apply_human_action(HumanAction.REGENERATE)
        return self.propose(session_id, target_stage, overrides)

    def delegate(self, session_id: str) -> GiftSession:
        console = self._console(session_id)
        console.orchestrator.apply_human_action(HumanAction.DELEGATE)
        while True:
            next_stage = self.next_stage(session_id)
            if next_stage is None:
                return console.orchestrator.session
            self.propose(session_id, next_stage)

    def next_stage(self, session_id: str) -> str | None:
        session = self.get_session(session_id)
        completed = {
            entry.stage
            for entry in session.stage_log
            if entry.status == "completed" and entry.proposed_by == "human" and entry.human_action in {HumanAction.ACCEPT, HumanAction.EDIT, HumanAction.DELEGATE}
        }
        completed.update(
            entry.stage
            for entry in session.stage_log
            if entry.status == "completed" and entry.proposed_by == "agent" and entry.human_action == HumanAction.DELEGATE
        )
        for stage in STAGES:
            if stage not in completed:
                return stage
        return None

    def ledger(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        timeline = []
        counts = {"accept": 0, "edit": 0, "regenerate": 0, "delegate": 0, "pending": 0, "error": 0}
        for entry in session.stage_log:
            action = entry.human_action.value if entry.human_action else ("pending" if entry.status == "pending" else entry.status)
            if action in counts:
                counts[action] += 1
            timeline.append(
                {
                    "stage": entry.stage,
                    "actor": entry.proposed_by,
                    "action": action,
                    "status": entry.status,
                    "timestamp": entry.timestamp.isoformat(),
                    "rationale": entry.rationale,
                }
            )
        authored_by = "hybrid"
        if counts["delegate"] and not (counts["accept"] or counts["edit"] or counts["regenerate"]):
            authored_by = "ai"
        elif counts["edit"] or counts["accept"] or counts["regenerate"]:
            authored_by = "hybrid"
        return {
            "session_id": session.session_id,
            "timeline": timeline,
            "counts": counts,
            "authorship": authored_by,
            "stage_count": len(STAGES),
            "completed": self.next_stage(session_id) is None,
        }

    def submit_feedback(
        self,
        session_id: str,
        *,
        rating: float,
        authorship: str | None = None,
        open_text: str | None = None,
        measures: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        console = self._console(session_id)
        session = console.orchestrator.session
        accept_count = sum(entry.human_action == HumanAction.ACCEPT for entry in session.stage_log)
        edit_count = sum(entry.human_action == HumanAction.EDIT for entry in session.stage_log)
        regenerate_count = sum(entry.human_action == HumanAction.REGENERATE for entry in session.stage_log)
        reward = reward_from_feedback(
            rating,
            accept_count=accept_count,
            edit_count=edit_count,
            regenerate_count=regenerate_count,
        )
        relationship = console.fixture["relationships"][0]
        occasion = self._occasion(console.fixture, session.occasion_id)
        creative = self._safe_effective(console, "creative_generation")
        recommendation = self._safe_effective(console, "recommendation")
        selected = (recommendation.get("recommendations") or [{}])[0]
        agency_slider = float(creative.get("agency_slider", console.agency_slider))
        context = self._encoder.encode(
            relationship.get("type", "other"),
            float(relationship.get("closeness_score", 3)),
            occasion.get("formality", "other"),
            agency_slider,
        )
        action = BanditAction(
            recommendation_category=selected.get("category", "unknown"),
            agency_bucket=_agency_bucket(agency_slider),
            style_archetype=selected.get("artifact_type", "generated"),
        )
        policy = self._load_or_create_bandit(action)
        policy.update(action, context, reward)
        policy.save(self.bandit_state_path)
        log_session(
            self.bandit_log_path,
            context,
            action,
            reward,
            session_id=session_id,
            rating=rating,
            authorship=authorship,
            open_text=open_text,
            measures=dict(measures or {}),
            accept_count=accept_count,
            edit_count=edit_count,
            regenerate_count=regenerate_count,
        )
        return {"session_id": session_id, "reward": reward, "action": action.__dict__, "bandit_counts": {a.key: c for a, c in policy.counts.items()}}

    def _run_stage(self, console: ConsoleSession, stage: str, overrides: dict[str, Any]) -> AgentOutput:
        fixture = console.fixture
        giver = self._person(fixture, "giver")
        recipient = self._person(fixture, "recipient")
        relationship = fixture["relationships"][0]
        occasion = self._occasion(fixture, console.orchestrator.session.occasion_id)
        if console.budget_hint:
            occasion = {**occasion, "budget_hint": console.budget_hint}
        if stage == "recipient_profiling":
            agent = RecipientProfilingAgent(DemoStructuredLLM(self._recipient_response(fixture)))
            config = {"person": recipient, "preferences": fixture.get("preferences", []), "raw_notes": [m["content"] for m in fixture.get("memories", [])]}
        elif stage == "relationship_analysis":
            agent = RelationshipAnalysisAgent(DemoStructuredLLM(self._relationship_response(fixture)))
            config = {"relationship": relationship, "memories": fixture.get("memories", []), "occasion": occasion}
        elif stage == "recommendation":
            agent = RecommendationAgent(DemoStructuredLLM(self._recommendation_response(fixture)))
            config = {
                "recipient_profile": self._safe_effective(console, "recipient_profiling"),
                "relationship_guidance": self._safe_effective(console, "relationship_analysis"),
                "occasion": occasion,
                "preferences": fixture.get("preferences", []),
            }
        elif stage == "creative_generation":
            agent = self._creative()
            graph = load_fixture(console.fixture_path)
            context = _padded(graph.context_embedding(recipient["id"], occasion["id"]), agent.gan.config.context_dim)
            memory = fixture.get("memories", [{}])[0]
            agency_slider = float(overrides.get("agency_slider", console.agency_slider))
            console.agency_slider = agency_slider
            config = {
                "context_embedding": context,
                "relationship_type": relationship.get("type", "other"),
                "emotion_tag": memory.get("emotion_tag", "joy"),
                "occasion": occasion.get("name", "birthday"),
                "agency_slider": agency_slider,
                "human_style_ref": _padded(memory.get("embedding", context), agent.gan.config.context_dim),
                "seed": int(overrides.get("seed", console.seed)),
                "output_dir": self.generated_dir.as_posix(),
                "filename": f"{console.orchestrator.session.session_id}-{agency_slider:.2f}.png",
            }
        elif stage == "greeting_story":
            agent = GreetingStoryAgent(DemoStructuredLLM(self._greeting_response(fixture)))
            config = {
                "relationship_guidance": self._safe_effective(console, "relationship_analysis"),
                "occasion": occasion,
                "memories": fixture.get("memories", []),
                "tone_guidance": self._safe_effective(console, "relationship_analysis").get("tone_guidance"),
                "giver_name": giver.get("display_name"),
                "recipient_name": recipient.get("display_name"),
            }
        elif stage == "delivery_planner":
            agent = DeliveryPlannerAgent()
            config = {"artifact_type": self._safe_effective(console, "creative_generation").get("artifact_type", "generated"), "occasion": occasion}
        else:
            raise KeyError(stage)
        config.update(overrides)
        return agent.run({"session": console.orchestrator.session, "stage_config": config})

    def _creative(self) -> CreativeGenerationAgent:
        if self._creative_agent is None:
            self._creative_agent = CreativeGenerationAgent.from_checkpoint(str(self.checkpoint_path))
        return self._creative_agent

    def _load_or_create_bandit(self, action: BanditAction) -> LinUCBBandit:
        if self.bandit_state_path.exists():
            saved = LinUCBBandit.load(self.bandit_state_path)
            if action in saved.actions:
                return saved
            actions = [*saved.actions, action]
        else:
            actions = [action]
        return LinUCBBandit(actions, self._encoder.dimension)

    def _safe_effective(self, console: ConsoleSession, stage: str) -> dict[str, Any]:
        try:
            return console.orchestrator.effective_output(stage)
        except KeyError:
            return {}

    def _console(self, session_id: str) -> ConsoleSession:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session: {session_id}")
        return self.sessions[session_id]

    @staticmethod
    def _person(fixture: Mapping[str, Any], role: str) -> dict[str, Any]:
        return next(person for person in fixture["people"] if person["role"] == role)

    @staticmethod
    def _occasion(fixture: Mapping[str, Any], occasion_id: str | None) -> dict[str, Any]:
        occasions = fixture.get("occasions", [])
        if occasion_id is None:
            return dict(occasions[0])
        return dict(next(occasion for occasion in occasions if occasion["id"] == occasion_id))

    @staticmethod
    def _recipient_response(fixture: Mapping[str, Any]) -> dict[str, Any]:
        preferences = fixture.get("preferences", [])
        return {
            "output": {
                "interests": [{"name": item["value"], "confidence": item.get("confidence", 0.7)} for item in preferences[:3]],
                "communication_style": "Warm, specific, and grounded in shared memories.",
                "gift_history_summary": "No prior gift history was supplied in this fixture.",
            },
            "confidence": 0.88,
            "rationale": "Uses stated fixture preferences and supplied memory snippets only.",
        }

    @staticmethod
    def _relationship_response(fixture: Mapping[str, Any]) -> dict[str, Any]:
        relationship = fixture["relationships"][0]
        closeness = float(relationship.get("closeness_score", 3))
        label = "very close" if closeness >= 4.5 else "close" if closeness >= 3.5 else "moderate" if closeness >= 2 else "low"
        return {
            "output": {
                "closeness_assessment": label,
                "tone_guidance": "Affectionate and playful without inventing private details.",
                "formality": fixture.get("occasions", [{}])[0].get("formality", "casual"),
                "risk_flags": [],
                "agency_slider_default": 0.5,
            },
            "confidence": 0.92,
            "rationale": "Maps the fixture closeness score and occasion formality into conservative guidance.",
        }

    @staticmethod
    def _recommendation_response(fixture: Mapping[str, Any]) -> dict[str, Any]:
        preferences = [pref["value"] for pref in fixture.get("preferences", [])]
        memories = fixture.get("memories", [])
        memory = memories[0]["content"] if memories else "a shared memory"
        return {
            "output": {
                "recommendations": [
                    {
                        "rank": 1,
                        "category": "personalized art",
                        "concept": f"A generated keepsake illustration weaving {preferences[0] if preferences else 'a favorite style'} with {memory}",
                        "evidence": preferences[:2] + ([memories[0]["id"]] if memories else []),
                        "budget_fit": "Digital generation fits the supplied budget hint.",
                        "artifact_type": "generated",
                    },
                    {
                        "rank": 2,
                        "category": "experience",
                        "concept": "A small shared ritual kit tied to the occasion.",
                        "evidence": [m["id"] for m in memories[:2]],
                        "budget_fit": "Can be scaled to the stated budget.",
                        "artifact_type": "bundle",
                    },
                    {
                        "rank": 3,
                        "category": "physical keepsake",
                        "concept": "A modest physical object echoing the recipient's stated preferences.",
                        "evidence": preferences[:1],
                        "budget_fit": "Simulated only; no purchase is made.",
                        "artifact_type": "physical",
                    },
                ]
            },
            "confidence": 0.86,
            "rationale": "Ranks a generated concept first because the demo centers the Agency Console visual artifact.",
        }

    @staticmethod
    def _greeting_response(fixture: Mapping[str, Any]) -> dict[str, Any]:
        memories = fixture.get("memories", [])
        refs = [memory["id"] for memory in memories[:2]]
        return {
            "output": {
                "message": "Here is a small memory-made thing for you: warm, specific, and ours without needing to be grand.",
                "memory_references": refs,
                "tone": "warm and personal",
            },
            "confidence": 0.9,
            "rationale": "Uses supplied memory ids and keeps the message original and concise.",
        }

