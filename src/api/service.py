from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from src.agents import (
    AgentOrchestrator,
    CreativeGenerationAgent,
    DeliveryPlannerAgent,
    GiftIntentReasoningAgent,
    GiftSession,
    GreetingStoryAgent,
    HumanAction,
    LLMProvider,
    MultiAgentPlanningAgent,
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
    "gift_intent_reasoning",
    "multi_agent_planning",
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
    fixture_path: Path | None
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
        personas: list[dict[str, Any]] = [
            {
                "persona_id": "custom-live",
                "label": "Create a live gifting context",
                "synthetic": False,
                "occasions": [],
            }
        ]
        if os.getenv("GMGI_INCLUDE_SYNTHETIC_FIXTURES") == "1":
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
        custom_profile: Mapping[str, Any] | None = None,
    ) -> GiftSession:
        if custom_profile is not None or persona_id == "custom-live":
            fixture_path = None
            fixture = self._custom_fixture(custom_profile or {})
            persona_id = fixture["persona_id"]
            occasion_id = fixture["occasions"][0]["id"]
        else:
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
        demo_agents = os.getenv("GMGI_USE_DEMO_AGENT_RESPONSES", "0") == "1"
        if stage == "recipient_profiling":
            agent = RecipientProfilingAgent(DemoStructuredLLM(self._recipient_response(fixture)) if demo_agents else None)
            config = {"person": recipient, "preferences": fixture.get("preferences", []), "raw_notes": [m["content"] for m in fixture.get("memories", [])]}
        elif stage == "relationship_analysis":
            agent = RelationshipAnalysisAgent(DemoStructuredLLM(self._relationship_response(fixture)) if demo_agents else None)
            config = {"relationship": relationship, "memories": fixture.get("memories", []), "occasion": occasion}
        elif stage == "gift_intent_reasoning":
            agent = GiftIntentReasoningAgent(DemoStructuredLLM(self._intent_response(fixture, occasion)) if demo_agents else None)
            config = {
                "recipient_profile": self._safe_effective(console, "recipient_profiling"),
                "relationship_guidance": self._safe_effective(console, "relationship_analysis"),
                "relationship": relationship,
                "occasion": occasion,
                "memories": fixture.get("memories", []),
                "preferences": fixture.get("preferences", []),
                "budget_hint": occasion.get("budget_hint"),
                "method": os.getenv("GMGI_INTENT_METHOD", "heuristic"),
            }
        elif stage == "multi_agent_planning":
            intent = self._safe_effective(console, "gift_intent_reasoning") or self._intent_response(fixture, occasion)["output"]
            agent = MultiAgentPlanningAgent(DemoStructuredLLM(self._planning_response(fixture, intent)) if demo_agents else None)
            config = {
                "user_request": f"Create a gift for {recipient.get('display_name', 'the recipient')} for {occasion.get('name', 'the occasion')}",
                "recipient_profile": self._safe_effective(console, "recipient_profiling"),
                "relationship_guidance": self._safe_effective(console, "relationship_analysis"),
                "intent": intent,
                "memory_signals": {"memory_count": len(fixture.get("memories", [])), "preference_count": len(fixture.get("preferences", []))},
                "available_agents": list(STAGES),
                "method": os.getenv("GMGI_PLANNING_METHOD", "rule_constrained"),
            }
        elif stage == "recommendation":
            agent = RecommendationAgent(DemoStructuredLLM(self._recommendation_response(fixture)) if demo_agents else None)
            config = {
                "recipient_profile": self._safe_effective(console, "recipient_profiling"),
                "relationship_guidance": self._safe_effective(console, "relationship_analysis"),
                "gift_intent": self._safe_effective(console, "gift_intent_reasoning"),
                "execution_plan": self._safe_effective(console, "multi_agent_planning"),
                "occasion": occasion,
                "preferences": fixture.get("preferences", []),
            }
        elif stage == "creative_generation":
            agent = self._creative()
            context = _padded(self._context_embedding(console, recipient["id"], occasion["id"]), agent.gan.config.context_dim)
            memory = fixture.get("memories", [{}])[0]
            agency_slider = float(overrides.get("agency_slider", console.agency_slider))
            console.agency_slider = agency_slider
            config = {
                "context_embedding": context,
                "relationship_type": relationship.get("type", "other"),
                "emotion_tag": memory.get("emotion_tag", "joy"),
                "occasion": self._gan_occasion(occasion.get("name", "other")),
                "agency_slider": agency_slider,
                "human_style_ref": _padded(memory.get("embedding", context), agent.gan.config.context_dim),
                "seed": int(overrides.get("seed", console.seed)),
                "output_dir": self.generated_dir.as_posix(),
                "filename": f"{console.orchestrator.session.session_id}-{agency_slider:.2f}.png",
            }
        elif stage == "greeting_story":
            agent = GreetingStoryAgent(DemoStructuredLLM(self._greeting_response(fixture)) if demo_agents else None)
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

    def _context_embedding(self, console: ConsoleSession, recipient_id: str, occasion_id: str) -> np.ndarray:
        if console.fixture_path is not None:
            graph = load_fixture(console.fixture_path)
            return graph.context_embedding(recipient_id, occasion_id)
        vectors = []
        for memory in console.fixture.get("memories", []):
            if "embedding" in memory:
                vectors.append(np.asarray(memory["embedding"], dtype=np.float32))
        for preference in console.fixture.get("preferences", []):
            if "embedding" in preference:
                vectors.append(np.asarray(preference["embedding"], dtype=np.float32))
        if not vectors:
            seed_text = json.dumps(console.fixture, sort_keys=True)
            vectors.append(self._hash_embedding(seed_text))
        return np.mean(np.stack(vectors), axis=0, dtype=np.float32)

    def _custom_fixture(self, profile: Mapping[str, Any]) -> dict[str, Any]:
        giver_name = str(profile.get("giver_name") or "Gift giver").strip()
        recipient_name = str(profile.get("recipient_name") or "Gift recipient").strip()
        relationship_type = str(profile.get("relationship_type") or "other").strip() or "other"
        closeness_score = float(profile.get("closeness_score", 3))
        occasion_name = str(profile.get("occasion_name") or "Gift occasion").strip()
        occasion_date = str(profile.get("occasion_date") or "2026-12-31").strip()
        budget_hint = str(profile.get("budget_hint") or "Flexible").strip()
        formality = str(profile.get("formality") or "casual").strip() or "casual"
        raw_memories = profile.get("memories") or []
        if isinstance(raw_memories, str):
            raw_memories = [line.strip() for line in raw_memories.splitlines() if line.strip()]
        raw_preferences = profile.get("preferences") or []
        if isinstance(raw_preferences, str):
            raw_preferences = [item.strip() for item in raw_preferences.split(",") if item.strip()]
        suffix = uuid4().hex[:8]
        giver_id = f"person-giver-{suffix}"
        recipient_id = f"person-recipient-{suffix}"
        occasion_id = f"occasion-live-{suffix}"
        event_id = f"event-live-{suffix}"
        memories = [
            {
                "id": f"memory-live-{suffix}-{index + 1}",
                "modality": "text",
                "content": str(content),
                "embedding": self._hash_embedding(str(content)).tolist(),
                "embedding_source": "hash_text_runtime",
                "timestamp": f"{occasion_date}T00:00:00Z",
                "event_id": event_id,
                "person_ids": [giver_id, recipient_id],
                "emotion_tag": self._emotion_from_text(str(content)),
            }
            for index, content in enumerate(raw_memories[:8])
        ]
        preferences = [
            {
                "id": f"preference-live-{suffix}-{index + 1}",
                "person_id": recipient_id,
                "category": "stated",
                "value": str(value),
                "confidence": 1.0,
                "source": "user_provided",
                "embedding": self._hash_embedding(str(value)).tolist(),
            }
            for index, value in enumerate(raw_preferences[:12])
        ]
        return {
            "schema_version": "1.0",
            "persona_id": f"live-{suffix}",
            "label": f"{giver_name} to {recipient_name}",
            "synthetic": False,
            "people": [
                {"id": giver_id, "display_name": giver_name, "role": "giver"},
                {"id": recipient_id, "display_name": recipient_name, "role": "recipient"},
            ],
            "relationships": [
                {
                    "id": f"relationship-live-{suffix}",
                    "person_a": giver_id,
                    "person_b": recipient_id,
                    "type": relationship_type,
                    "closeness_score": max(1.0, min(5.0, closeness_score)),
                }
            ],
            "occasions": [
                {
                    "id": occasion_id,
                    "name": occasion_name,
                    "date": occasion_date,
                    "budget_hint": budget_hint,
                    "formality": formality,
                }
            ],
            "events": [
                {"id": event_id, "date": occasion_date, "type": "user-provided-context", "participants": [giver_id, recipient_id]}
            ],
            "memories": memories,
            "preferences": preferences,
        }

    @staticmethod
    def _gan_occasion(value: object) -> str:
        text = str(value or "").strip().lower()
        aliases = {
            "birthday": ("birthday", "bday", "birth day"),
            "anniversary": ("anniversary",),
            "graduation": ("graduation", "graduate", "commencement"),
            "housewarming": ("housewarming", "house warming", "new home"),
            "promotion": ("promotion", "promoted"),
            "holiday": ("holiday", "christmas", "diwali", "eid", "hanukkah", "new year"),
            "thank-you": ("thank-you", "thank you", "thanks", "gratitude"),
        }
        for canonical, needles in aliases.items():
            if any(needle in text for needle in needles):
                return canonical
        return "other"
    @staticmethod
    def _hash_embedding(text: str, dimensions: int = 8) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.frombuffer(digest[:dimensions], dtype=np.uint8).astype(np.float32)
        return values / 127.5 - 1.0

    @staticmethod
    def _emotion_from_text(text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ("funny", "laugh", "joke", "silly")):
            return "humor"
        if any(word in lowered for word in ("miss", "remember", "old", "first")):
            return "nostalgia"
        if any(word in lowered for word in ("thank", "grateful", "appreciate")):
            return "gratitude"
        if any(word in lowered for word in ("calm", "hard", "support", "comfort")):
            return "comfort"
        return "joy"
    def _creative(self) -> CreativeGenerationAgent:
        if self._creative_agent is None:
            self._creative_agent = CreativeGenerationAgent.from_checkpoint(str(self._resolve_checkpoint_path()))
        return self._creative_agent

    def _resolve_checkpoint_path(self) -> Path:
        require_full = os.getenv("GMGI_REQUIRE_FULL_GAN_CHECKPOINT", "0") == "1"
        explicit = os.getenv("GMGI_GAN_CHECKPOINT")
        if explicit:
            path = Path(explicit)
            if not path.exists():
                raise FileNotFoundError(f"GMGI_GAN_CHECKPOINT does not exist: {path}")
            if require_full:
                self._require_full_checkpoint(path)
            return path
        candidates = []
        if self.checkpoint_path.exists():
            candidates.append(self.checkpoint_path)
        candidates.extend(
            sorted(
                Path("experiments").glob("run-*/checkpoint-*.pt"),
                key=lambda path: (path.parent.name, path.name),
            )
        )
        for candidate in reversed(candidates):
            if require_full:
                try:
                    self._require_full_checkpoint(candidate)
                except ValueError:
                    continue
            return candidate
        if require_full:
            raise FileNotFoundError(
                "No full GAN checkpoint found. Train with `python -m src.gan.train --config src/gan/configs/train.json` "
                "or set GMGI_GAN_CHECKPOINT to a full 256px checkpoint. Smoke checkpoints are rejected."
            )
        raise FileNotFoundError(
            "No GAN checkpoint found. Train with `python -m src.gan.train --config src/gan/configs/train.json` "
            "or set GMGI_GAN_CHECKPOINT."
        )

    @staticmethod
    def _require_full_checkpoint(path: Path) -> None:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
        training_config = checkpoint.get("training_config", {}) if isinstance(checkpoint, dict) else {}
        resolution = int(model_config.get("resolution", 0) or 0)
        max_steps = int(training_config.get("max_steps", 0) or 0)
        metric_backend = str(training_config.get("metric_backend", ""))
        base_config = str(training_config.get("base_config", ""))
        if resolution < 256 or max_steps < 1000 or metric_backend == "clip" or "train_smoke" in base_config:
            raise ValueError(
                f"Checkpoint {path} looks like a smoke/pilot checkpoint "
                f"(resolution={resolution}, max_steps={max_steps}, metric_backend={metric_backend!r})."
            )

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
    def _intent_response(fixture: Mapping[str, Any], occasion: Mapping[str, Any] | None = None) -> dict[str, Any]:
        occasion = dict(occasion or (fixture.get("occasions", [{}])[0]))
        preferences = fixture.get("preferences", [])
        memories = fixture.get("memories", [])
        preference_items = [
            {"value": pref.get("value"), "confidence": pref.get("confidence", 0.8), "source": pref.get("source", "fixture")}
            for pref in preferences[:5]
        ]
        output = {
            "intent_summary": f"Create a personalized gift for {occasion.get('name', 'the occasion')} with a {occasion.get('formality', 'casual')} tone.",
            "occasion": {
                "name": occasion.get("name", "unspecified"),
                "date": occasion.get("date"),
                "formality": occasion.get("formality", "casual"),
                "urgency": "date-specified" if occasion.get("date") else "unknown",
            },
            "goal": {
                "gift_purpose": "create a meaningful personalized gift",
                "emotional_objective": "warm connection through memory-grounded personalization" if memories else "thoughtful personalization",
                "personalization_depth": "high" if memories else "medium",
                "social_tone": occasion.get("formality", "casual"),
            },
            "constraints": {
                "budget_hint": occasion.get("budget_hint", "Flexible"),
                "budget_sensitivity": "medium",
                "delivery_constraints": ["simulated delivery only"],
                "timing": "date-specified" if occasion.get("date") else "unknown",
            },
            "preferences": [item for item in preference_items if item.get("value")],
            "open_questions": [],
            "clarifying_needs": [],
        }
        return {
            "output": output,
            "confidence": 0.86,
            "rationale": "Deterministic demo intent derived from occasion, relationship context, memories, and stated preferences.",
        }

    @staticmethod
    def _planning_response(fixture: Mapping[str, Any], intent: Mapping[str, Any]) -> dict[str, Any]:
        sequence = list(STAGES)
        output = {
            "task_goal": str(intent.get("intent_summary") or "Create a personalized gift workflow"),
            "subtasks": [
                {"id": f"step_{index + 1}", "agent": agent, "action": "run_stage", "requires_human_review": True}
                for index, agent in enumerate(sequence)
            ],
            "agent_sequence": sequence,
            "dependencies": [
                {"after": sequence[index - 1], "before": sequence[index], "type": "stage_output"}
                for index in range(1, len(sequence))
            ],
            "expected_outputs": [
                {"agent": agent, "output": "structured stage output"}
                for agent in sequence
            ],
            "stop_conditions": [
                "human action is required for each proposal unless delegate is selected",
                "fallback to current staged orchestration if planner output is invalid",
                "delivery remains simulated only",
            ],
            "fallback_plan": {
                "type": "current_staged_orchestration",
                "agent_sequence": sequence,
                "reason": "Bounded default used by the existing Agency Console.",
            },
        }
        return {
            "output": output,
            "confidence": 0.84,
            "rationale": "Deterministic demo planner decomposes the GMGI workflow into bounded auditable stages.",
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