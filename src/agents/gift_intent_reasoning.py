from __future__ import annotations

import json
import os
from typing import Any, Mapping

from .base import StructuredAgent
from .orchestrator import AgentInput, AgentOutput


URGENCY_DAYS = 10


class GiftIntentReasoningAgent(StructuredAgent):
    """Dedicated intent layer: classify/extract intent before recommendation."""

    stage = "gift_intent_reasoning"
    config_name = "gift_intent_reasoning.json"

    def build_context(self, agent_input: AgentInput) -> dict[str, Any]:
        config = agent_input["stage_config"]
        return {
            "recipient_profile": config.get("recipient_profile", {}),
            "relationship_guidance": config.get("relationship_guidance", {}),
            "relationship": config.get("relationship", {}),
            "occasion": config.get("occasion", {}),
            "memories": config.get("memories", []),
            "preferences": config.get("preferences", []),
            "budget_hint": config.get("budget_hint") or config.get("occasion", {}).get("budget_hint"),
            "method": config.get("method", self.config.get("default_method", "heuristic")),
        }

    def run(self, agent_input: AgentInput) -> AgentOutput:
        method = str(agent_input.get("stage_config", {}).get("method") or self.config.get("default_method", "heuristic"))
        if method in {"heuristic", "rule_based", "classifier_hybrid", "abstain"}:
            return self._run_heuristic(agent_input, method=method)
        try:
            return super().run(agent_input)
        except Exception as exc:
            if os.getenv("GMGI_ALLOW_INTENT_REPAIR", "1") == "1":
                result = self._run_heuristic(agent_input, method="classifier_hybrid")
                rationale = result.get("rationale")
                note = f"llm_structured intent failed; repaired with deterministic intent extractor; original_error={type(exc).__name__}"
                result["rationale"] = f"{note}; {rationale}" if rationale else note
                return result
            if os.getenv("GMGI_FORCE_OLLAMA_AGENTS") == "1" and os.getenv("GMGI_ALLOW_AGENT_FALLBACK") != "1":
                raise
            return self._run_heuristic(agent_input, method="fallback_heuristic")

    def _run_heuristic(self, agent_input: AgentInput, *, method: str) -> AgentOutput:
        context = self.build_context(agent_input)
        occasion = dict(context.get("occasion") or {})
        preferences = _preference_values(context.get("preferences", []), context.get("recipient_profile", {}))
        relationship = dict(context.get("relationship") or {})
        relationship_guidance = dict(context.get("relationship_guidance") or {})
        memories = list(context.get("memories") or [])
        budget_hint = context.get("budget_hint") or "unspecified"
        formality = occasion.get("formality") or relationship_guidance.get("formality") or "casual"
        closeness = float(relationship.get("closeness_score", 3) or 3)
        personalization_depth = "high" if memories and closeness >= 3.5 else "medium" if memories or preferences else "low"
        urgency = _urgency_bucket(str(occasion.get("date", "")))
        clarifying_needs = []
        if not preferences:
            clarifying_needs.append("recipient_preferences")
        if budget_hint == "unspecified":
            clarifying_needs.append("budget")
        if not occasion.get("date"):
            clarifying_needs.append("occasion_date")
        if method == "abstain" and len(clarifying_needs) >= 2:
            confidence = 0.42
            open_questions = ["What budget range and timing should the system respect?", "Which recipient preferences matter most?"]
        else:
            confidence = 0.78 if clarifying_needs else 0.9
            open_questions = [f"Clarify {item.replace('_', ' ')}." for item in clarifying_needs]
        artifact_type = _recommended_artifact_type(occasion, preferences, relationship, memories)
        visual_style = _visual_style(artifact_type, preferences, memories)
        output = {
            "intent_summary": f"Create a {personalization_depth}-personalization gift for {occasion.get('name', 'the occasion')} with {formality} tone.",
            "occasion": {
                "name": occasion.get("name", "unspecified"),
                "date": occasion.get("date"),
                "formality": formality,
                "urgency": urgency,
            },
            "goal": {
                "gift_purpose": _purpose_from_occasion(str(occasion.get("name", ""))),
                "emotional_objective": _emotional_objective(memories, relationship_guidance),
                "social_tone": relationship_guidance.get("tone_guidance", formality),
                "personalization_depth": personalization_depth,
                "recommended_artifact_type": artifact_type,
                "visual_style": visual_style,
            },
            "constraints": {
                "budget_hint": budget_hint,
                "budget_sensitivity": _budget_sensitivity(str(budget_hint)),
                "delivery_constraints": _delivery_constraints(urgency, formality),
                "timing": urgency,
            },
            "preferences": preferences,
            "visual_generation": {
                "artifact_type": artifact_type,
                "style_prompt": visual_style,
            },
            "open_questions": open_questions,
            "clarifying_needs": clarifying_needs,
        }
        return AgentOutput(
            stage=self.stage,
            output=output,
            confidence=confidence,
            rationale=f"{method} intent extraction over occasion, relationship, preferences, and supplied memories.",
        )


def _preference_values(preferences: Any, profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for item in preferences or []:
        if isinstance(item, Mapping):
            values.append({"value": item.get("value") or item.get("name"), "confidence": item.get("confidence", 0.8), "source": item.get("source", "provided")})
        else:
            values.append({"value": str(item), "confidence": 0.8, "source": "provided"})
    for interest in profile.get("interests", []) if isinstance(profile, Mapping) else []:
        if isinstance(interest, Mapping):
            name = interest.get("name")
            if name and all(existing.get("value") != name for existing in values):
                values.append({"value": name, "confidence": interest.get("confidence", 0.7), "source": "recipient_profile"})
    return [item for item in values if item.get("value")]


def _purpose_from_occasion(name: str) -> str:
    lowered = name.lower()
    if "promotion" in lowered or "graduation" in lowered:
        return "celebrate achievement"
    if "thank" in lowered:
        return "express gratitude"
    if "house" in lowered or "home" in lowered:
        return "mark a transition"
    if "birthday" in lowered or "anniversary" in lowered:
        return "celebrate relationship milestone"
    return "create a meaningful personalized gift"


def _emotional_objective(memories: list[Any], guidance: Mapping[str, Any]) -> str:
    text = json.dumps(memories, ensure_ascii=False).lower()
    if "laugh" in text or "funny" in text:
        return "shared joy and humor"
    if "miss" in text or "remember" in text:
        return "nostalgia and closeness"
    if "thank" in text or "grateful" in text:
        return "gratitude"
    return str(guidance.get("closeness_assessment") or "warm connection")


def _budget_sensitivity(value: str) -> str:
    lowered = value.lower()
    if any(token in lowered for token in ("low", "under", "25", "30", "40")):
        return "high"
    if any(token in lowered for token in ("flex", "no limit", "premium")):
        return "low"
    return "medium"


def _urgency_bucket(date_text: str) -> str:
    # Keep local and deterministic: date distance is evaluated by the delivery planner at runtime.
    if not date_text:
        return "unknown"
    return "date-specified"


def _delivery_constraints(urgency: str, formality: str) -> list[str]:
    constraints = ["simulated delivery only"]
    if urgency == "unknown":
        constraints.append("ask for occasion date before real-world scheduling")
    if formality in {"professional", "ceremonial"}:
        constraints.append("avoid overly intimate language")
    return constraints


def _recommended_artifact_type(occasion: Mapping[str, Any], preferences: list[dict[str, Any]], relationship: Mapping[str, Any], memories: list[Any]) -> str:
    text = " ".join([
        str(occasion.get("name", "")),
        json.dumps(preferences, ensure_ascii=False),
        json.dumps(memories, ensure_ascii=False),
    ]).lower()
    if any(token in text for token in ("wrap", "wrapping", "pattern", "home", "housewarming")):
        return "gift_wrap"
    if any(token in text for token in ("graduation", "keepsake", "trophy", "robotics", "bicycle")):
        return "keepsake_print"
    if any(token in text for token in ("promotion", "office", "desk", "professional")) or relationship.get("type") == "colleague":
        return "gift_tag"
    return "greeting_card"


def _visual_style(artifact_type: str, preferences: list[dict[str, Any]], memories: list[Any]) -> str:
    preference_text = ", ".join(str(item.get("value")) for item in preferences if item.get("value"))
    memory_text = json.dumps(memories[:2], ensure_ascii=False).lower()
    base = artifact_type.replace("_", " ")
    if preference_text:
        return f"{base} using {preference_text} as visual cues"
    if "laugh" in memory_text or "funny" in memory_text:
        return f"playful {base} with light humor"
    if "thank" in memory_text or "grateful" in memory_text:
        return f"warm appreciative {base}"
    return f"personalized {base} with memory-inspired motifs"
