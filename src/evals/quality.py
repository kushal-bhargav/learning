from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agents.experience_store import Episode, ExperienceStore

from .structural import DEFAULT_STAGE_ORDER, dag_validity, schema_conformance


@dataclass(frozen=True)
class Score:
    name: str
    score: float | None
    passed: bool | None
    details: dict[str, Any]
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "weight": self.weight,
            "details": self.details,
        }


def evaluate_outputs(
    agent_outputs: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
    input_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = expected or {}
    input_context = input_context or {}
    stage_reports: dict[str, Any] = {}
    for stage in DEFAULT_STAGE_ORDER:
        output = _as_mapping(agent_outputs.get(stage))
        if not output and stage not in agent_outputs:
            stage_reports[stage] = {
                "quality_score": 0.0,
                "status": "missing",
                "metrics": [
                    Score("stage_present", 0.0, False, {"stage": stage}).to_dict(),
                ],
            }
            continue
        metrics = evaluate_stage(stage, output, expected=expected, agent_outputs=agent_outputs, input_context=input_context)
        stage_reports[stage] = _stage_report(metrics)
    cross = cross_component_metrics(agent_outputs, expected=expected, input_context=input_context)
    scores = [report["quality_score"] for report in stage_reports.values() if report.get("quality_score") is not None]
    cross_scores = [metric.score for metric in cross if metric.score is not None]
    overall_items = [*scores, *cross_scores]
    return {
        "stage_reports": stage_reports,
        "cross_component_metrics": [metric.to_dict() for metric in cross],
        "overall_quality_score": None if not overall_items else sum(overall_items) / len(overall_items),
    }


def evaluate_stage(
    stage: str,
    output: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
    agent_outputs: Mapping[str, Any] | None = None,
    input_context: Mapping[str, Any] | None = None,
) -> list[Score]:
    expected = expected or {}
    agent_outputs = agent_outputs or {}
    input_context = input_context or {}
    if "error" in output:
        return [
            Score(
                "agent_execution_success",
                0.0,
                False,
                {"error_type": output.get("error_type"), "error": output.get("error")},
                weight=4.0,
            )
        ]
    if stage == "recipient_profiling":
        return recipient_metrics(output, expected, input_context)
    if stage == "relationship_analysis":
        return relationship_metrics(output, expected)
    if stage == "gift_intent_reasoning":
        return intent_metrics(output, expected, input_context)
    if stage == "multi_agent_planning":
        return planning_metrics(output)
    if stage == "recommendation":
        return recommendation_metrics(output, expected, agent_outputs, input_context)
    if stage == "creative_generation":
        return creative_metrics(output, expected, agent_outputs)
    if stage == "greeting_story":
        return greeting_metrics(output, expected, agent_outputs, input_context)
    if stage == "delivery_planner":
        return delivery_metrics(output, expected)
    return [Score("known_stage", 0.0, False, {"stage": stage})]


def recipient_metrics(output: Mapping[str, Any], expected: Mapping[str, Any], input_context: Mapping[str, Any]) -> list[Score]:
    interests = [_as_mapping(item) for item in _as_list(output.get("interests"))]
    actual = {_norm(item.get("name")) for item in interests if item.get("name")}
    expected_preferences = {_norm(item) for item in _expected_list(expected, "preferences", input_context)}
    sensitive_hits = _sensitive_hits(output)
    recall = _recall(actual, expected_preferences)
    confidence_valid = all(_between(item.get("confidence"), 0, 1) for item in interests)
    return [
        _bool_score("profile_fields_complete", all(output.get(key) for key in ("interests", "communication_style", "gift_history_summary")), output),
        Score("stated_preference_recall", recall, None if recall is None else recall >= 0.67, {"expected": sorted(expected_preferences), "actual": sorted(actual)}, weight=2.0),
        _bool_score("interest_confidence_valid", confidence_valid, {"interest_count": len(interests)}),
        Score("no_unsupported_sensitive_traits", 1.0 if not sensitive_hits else 0.0, not sensitive_hits, {"sensitive_terms": sensitive_hits}, weight=2.0),
    ]


def relationship_metrics(output: Mapping[str, Any], expected: Mapping[str, Any]) -> list[Score]:
    expected_closeness = expected.get("relationship_closeness") or _bucket(expected.get("closeness_score"))
    actual = _norm(output.get("closeness_assessment"))
    tone = str(output.get("tone_guidance") or "").lower()
    risk_flags = output.get("risk_flags")
    closeness = None if not expected_closeness else float(actual == _norm(expected_closeness))
    conservative = "new" not in _norm(expected.get("relationship_type")) or any(term in tone for term in ("professional", "not overly intimate", "respect", "warm"))
    return [
        Score("closeness_bucket_match", closeness, None if closeness is None else closeness == 1.0, {"expected": expected_closeness, "actual": output.get("closeness_assessment")}, weight=2.0),
        _bool_score("tone_guidance_present", len(tone.split()) >= 3, {"tone_guidance": output.get("tone_guidance")}),
        _bool_score("formality_present", bool(output.get("formality")), {"formality": output.get("formality")}),
        _bool_score("risk_flags_structured", isinstance(risk_flags, list), {"risk_flags": risk_flags}),
        _bool_score("agency_slider_valid", _between(output.get("agency_slider_default"), 0, 1), {"agency_slider_default": output.get("agency_slider_default")}),
        _bool_score("social_boundary_respected", conservative, {"relationship_type": expected.get("relationship_type"), "tone_guidance": output.get("tone_guidance")}, weight=2.0),
    ]


def intent_metrics(output: Mapping[str, Any], expected: Mapping[str, Any], input_context: Mapping[str, Any]) -> list[Score]:
    constraints = _as_mapping(output.get("constraints"))
    goal = _as_mapping(output.get("goal"))
    occasion = _as_mapping(output.get("occasion"))
    actual_preferences = {_norm(_as_mapping(item).get("value")) for item in _as_list(output.get("preferences")) if _as_mapping(item).get("value")}
    expected_preferences = {_norm(item) for item in _expected_list(expected, "preferences", input_context)}
    pref_recall = _recall(actual_preferences, expected_preferences)
    expected_occ = _norm(expected.get("occasion_name") or _as_mapping(input_context.get("occasion")).get("name"))
    actual_occ = _norm(occasion.get("name"))
    occ_score = None if not expected_occ else float(expected_occ in actual_occ or actual_occ in expected_occ)
    budget_expected = _norm(expected.get("budget_hint") or _as_mapping(input_context.get("occasion")).get("budget_hint"))
    budget_actual = _norm(constraints.get("budget_hint"))
    return [
        _bool_score("intent_fields_complete", all(output.get(key) for key in ("intent_summary", "occasion", "goal", "constraints", "preferences")), output),
        Score("occasion_match", occ_score, None if occ_score is None else occ_score == 1.0, {"expected": expected_occ, "actual": actual_occ}, weight=2.0),
        Score("preference_recall", pref_recall, None if pref_recall is None else pref_recall >= 0.67, {"expected": sorted(expected_preferences), "actual": sorted(actual_preferences)}, weight=2.0),
        _bool_score("budget_constraint_preserved", not budget_expected or budget_expected in budget_actual or bool(budget_actual), {"expected": budget_expected, "actual": budget_actual}),
        _bool_score("delivery_constraint_present", "simulated" in json.dumps(constraints).lower(), {"constraints": constraints}),
        _bool_score("gift_goal_specific", len(str(goal.get("gift_purpose") or output.get("intent_summary") or "").split()) >= 4, {"goal": goal}),
    ]


def planning_metrics(output: Mapping[str, Any]) -> list[Score]:
    sequence = [str(item) for item in _as_list(output.get("agent_sequence"))]
    subtasks = [_as_mapping(item) for item in _as_list(output.get("subtasks"))]
    dependencies = [_as_mapping(item) for item in _as_list(output.get("dependencies"))]
    expected = list(DEFAULT_STAGE_ORDER)
    coverage = len(set(sequence) & set(expected)) / len(expected)
    order = _ordered_subsequence(expected, sequence)
    dependency_edges = {(item.get("after"), item.get("before")) for item in dependencies}
    dependency_chain = all((sequence[index - 1], sequence[index]) in dependency_edges for index in range(1, len(sequence))) if len(sequence) > 1 else False
    return [
        Score("agent_coverage", coverage, coverage == 1.0, {"expected": expected, "actual": sequence}, weight=2.0),
        _bool_score("stage_order_valid", order, {"sequence": sequence}, weight=2.0),
        _bool_score("dependencies_executable", dependency_chain, {"dependencies": dependencies}, weight=2.0),
        _bool_score("human_review_visible", any(bool(item.get("requires_human_review")) for item in subtasks), {"subtasks": subtasks}),
        _bool_score("fallback_plan_present", bool(output.get("fallback_plan")), {"fallback_plan": output.get("fallback_plan")}),
        _bool_score("stop_conditions_present", bool(_as_list(output.get("stop_conditions"))), {"stop_conditions": output.get("stop_conditions")}),
    ]


def recommendation_metrics(
    output: Mapping[str, Any],
    expected: Mapping[str, Any],
    agent_outputs: Mapping[str, Any],
    input_context: Mapping[str, Any],
) -> list[Score]:
    recs = [_as_mapping(item) for item in _as_list(output.get("recommendations"))]
    preference_text = " ".join(_expected_list(expected, "preferences", input_context))
    memory_text = _joined([input_context, agent_outputs.get("recipient_profiling"), agent_outputs.get("relationship_analysis"), agent_outputs.get("gift_intent_reasoning")])
    evidence_items = [str(item) for rec in recs for item in _as_list(rec.get("evidence"))]
    evidence_supported = [_has_overlap(item, memory_text) or _has_overlap(item, preference_text) for item in evidence_items]
    concept_text = " ".join(str(rec.get("concept") or "") for rec in recs)
    pref_recall = _token_recall(preference_text, concept_text + " " + " ".join(evidence_items))
    diversity = len({_norm(rec.get("artifact_type")) for rec in recs if rec.get("artifact_type")}) / 3
    return [
        _bool_score("three_ranked_recommendations", len(recs) == 3 and [rec.get("rank") for rec in recs] == [1, 2, 3], {"recommendation_count": len(recs)}, weight=2.0),
        Score("evidence_grounding_rate", None if not evidence_supported else sum(evidence_supported) / len(evidence_supported), None if not evidence_supported else all(evidence_supported), {"evidence": evidence_items}, weight=2.0),
        Score("preference_coverage", pref_recall, None if pref_recall is None else pref_recall >= 0.45, {"preference_text": preference_text, "concept_text": concept_text}, weight=2.0),
        _bool_score("budget_fit_present", recs and all(bool(rec.get("budget_fit")) for rec in recs), {"budget_fit": [rec.get("budget_fit") for rec in recs]}),
        Score("artifact_type_diversity", min(1.0, diversity), diversity >= 2 / 3, {"artifact_types": [rec.get("artifact_type") for rec in recs]}),
        _bool_score("no_external_purchase_claim", "purchased" not in concept_text.lower() and "ordered" not in concept_text.lower(), {"concept_text": concept_text}),
    ]


def creative_metrics(output: Mapping[str, Any], expected: Mapping[str, Any], agent_outputs: Mapping[str, Any]) -> list[Score]:
    path = Path(str(output.get("artifact_path") or ""))
    exists = bool(path.name) and path.exists()
    width = int(output.get("width", 0) or 0)
    height = int(output.get("height", 0) or 0)
    fake_marker = any(term in str(path).lower() for term in ("fake", "placeholder", "dummy", "mock"))
    image_valid = _image_magic_valid(path) if exists else False
    practical_resolution = width >= 128 and height >= 128
    prompt_text = " ".join(
        str(value)
        for value in (
            output.get("diffusers_prompt"),
            output.get("style_prompt"),
            _as_mapping(_as_mapping(agent_outputs.get("gift_intent_reasoning")).get("visual_generation")).get("style_prompt"),
        )
        if value
    )
    return [
        _bool_score("artifact_file_exists", exists, {"artifact_path": str(path)}, weight=2.0),
        _bool_score("artifact_not_marked_fake", not fake_marker, {"artifact_path": str(path)}, weight=2.0),
        _bool_score("image_file_header_valid", image_valid, {"artifact_path": str(path)}),
        _bool_score("image_resolution_practical", practical_resolution, {"width": width, "height": height}, weight=2.0),
        _bool_score("agency_slider_valid", _between(output.get("agency_slider"), 0, 1), {"agency_slider": output.get("agency_slider")}),
        _bool_score("conditioning_prompt_visible", bool(prompt_text.strip()) or bool(output.get("artifact_type")), {"prompt_text": prompt_text}),
    ]


def greeting_metrics(output: Mapping[str, Any], expected: Mapping[str, Any], agent_outputs: Mapping[str, Any], input_context: Mapping[str, Any]) -> list[Score]:
    message = str(output.get("message") or "")
    refs = [str(item) for item in _as_list(output.get("memory_references"))]
    context_text = _joined([input_context, agent_outputs])
    ref_support = [_has_overlap(ref, context_text) or ref in context_text for ref in refs]
    tone = str(output.get("tone") or "")
    forbidden = _contains_copyrighty_signal(message)
    return [
        _bool_score("message_present", len(message.split()) >= 6, {"message_length_words": len(message.split())}, weight=2.0),
        _bool_score("tone_present", bool(tone.strip()), {"tone": tone}),
        Score("memory_reference_grounding", None if not refs else sum(ref_support) / len(ref_support), None if not refs else all(ref_support), {"memory_references": refs}, weight=2.0),
        _bool_score("message_not_overlong", len(message.split()) <= 120, {"message_length_words": len(message.split())}),
        _bool_score("no_obvious_copied_lyrics_or_quote", not forbidden, {"flagged": forbidden}, weight=2.0),
        _bool_score("recipient_or_occasion_signal", _has_overlap(message, json.dumps(expected, ensure_ascii=False)) or _has_overlap(message, context_text), {"message": message[:200]}),
    ]


def delivery_metrics(output: Mapping[str, Any], expected: Mapping[str, Any]) -> list[Score]:
    occasion_date = str(output.get("occasion_date") or expected.get("occasion_date") or "")
    planned_date = str(output.get("planned_send_date") or "")
    date_ok = _date_lte(planned_date, occasion_date) if occasion_date and planned_date else None
    disclaimer = str(output.get("disclaimer") or "").lower()
    text = json.dumps(output, ensure_ascii=False).lower()
    return [
        _bool_score("simulated_delivery_only", output.get("status") == "simulated" and ("no shipment" in disclaimer or "simulated" in disclaimer), output, weight=2.0),
        Score("planned_before_or_on_occasion", None if date_ok is None else float(date_ok), date_ok, {"planned_send_date": planned_date, "occasion_date": occasion_date}),
        _bool_score("channel_present", bool(output.get("channel")), {"channel": output.get("channel")}),
        _bool_score("mode_present", bool(output.get("mode")), {"mode": output.get("mode")}),
        _bool_score("no_real_logistics_integration_claim", not any(term in text for term in ("tracking number", "label purchased", "charged card", "shipment booked")), {"output": output}, weight=2.0),
    ]


def cross_component_metrics(agent_outputs: Mapping[str, Any], *, expected: Mapping[str, Any], input_context: Mapping[str, Any]) -> list[Score]:
    schema = schema_conformance(agent_outputs)
    dag = dag_validity(agent_outputs)
    present = [stage for stage in DEFAULT_STAGE_ORDER if stage in agent_outputs]
    coverage = len(present) / len(DEFAULT_STAGE_ORDER)
    error_stages = [stage for stage, output in agent_outputs.items() if "error" in _as_mapping(output)]
    rec_text = json.dumps(agent_outputs.get("recommendation", {}), ensure_ascii=False)
    intent_text = json.dumps(agent_outputs.get("gift_intent_reasoning", {}), ensure_ascii=False)
    consistency = _token_recall(intent_text, rec_text)
    return [
        Score("pipeline_stage_coverage", coverage, coverage == 1.0, {"present": present, "expected": list(DEFAULT_STAGE_ORDER)}, weight=2.0),
        Score("schema_conformance", schema.score, schema.passed, schema.details, weight=2.0),
        Score("dag_validity", dag.score, dag.passed, dag.details, weight=2.0),
        _bool_score("no_error_stage_outputs", not error_stages, {"error_stages": error_stages}, weight=2.0),
        Score("intent_to_recommendation_consistency", consistency, None if consistency is None else consistency >= 0.2, {"intent_chars": len(intent_text), "recommendation_chars": len(rec_text)}),
    ]


def evaluate_episode(episode: Episode | Mapping[str, Any], *, expected: Mapping[str, Any] | None = None, input_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    item = episode if isinstance(episode, Episode) else Episode.from_mapping(episode)
    quality = evaluate_outputs(item.agent_outputs, expected=expected, input_context=input_context)
    actions = dict(item.human_actions)
    quality["session_id"] = item.session_id
    quality["context_fingerprint"] = item.context_fingerprint
    quality["behavioral_metrics"] = {
        "composite_reward": item.composite_reward,
        "accept_rate": _action_rate(actions, "accept"),
        "edit_rate": _action_rate(actions, "edit"),
        "regenerate_rate": _action_rate(actions, "regenerate"),
        "delegate_rate": _action_rate(actions, "delegate"),
        "completed_action_coverage": len(actions) / len(DEFAULT_STAGE_ORDER),
    }
    return quality


def evaluate_store(path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    store = ExperienceStore.load(path)
    episodes = store.episodes[-limit:] if limit else store.episodes
    reports = [evaluate_episode(episode) for episode in episodes]
    return {
        "phase": "quality",
        "episode_count": len(reports),
        "summary": summarize_quality_reports(reports),
        "episodes": reports,
    }


def summarize_quality_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage_scores: dict[str, list[float]] = {}
    cross_scores: dict[str, list[float]] = {}
    overall: list[float] = []
    behavior: dict[str, list[float]] = {}
    for report in reports:
        if isinstance(report.get("overall_quality_score"), (int, float)):
            overall.append(float(report["overall_quality_score"]))
        for stage, stage_report in _as_mapping(report.get("stage_reports")).items():
            if isinstance(stage_report.get("quality_score"), (int, float)):
                stage_scores.setdefault(stage, []).append(float(stage_report["quality_score"]))
        for metric in _as_list(report.get("cross_component_metrics")):
            item = _as_mapping(metric)
            if isinstance(item.get("score"), (int, float)):
                cross_scores.setdefault(str(item.get("name")), []).append(float(item["score"]))
        for key, value in _as_mapping(report.get("behavioral_metrics")).items():
            if isinstance(value, (int, float)):
                behavior.setdefault(key, []).append(float(value))
    return {
        "overall_quality_score": _mean(overall),
        "stage_quality": {key: _mean(values) for key, values in sorted(stage_scores.items())},
        "cross_component": {key: _mean(values) for key, values in sorted(cross_scores.items())},
        "behavioral": {key: _mean(values) for key, values in sorted(behavior.items())},
        "n": len(reports),
    }


def _stage_report(metrics: Sequence[Score]) -> dict[str, Any]:
    scored = [(metric.score, metric.weight) for metric in metrics if metric.score is not None]
    total_weight = sum(weight for _score, weight in scored)
    quality = None if total_weight == 0 else sum(float(score) * weight for score, weight in scored) / total_weight
    return {
        "status": "ok",
        "quality_score": quality,
        "metrics": [metric.to_dict() for metric in metrics],
    }


def _bool_score(name: str, value: bool, details: Mapping[str, Any], weight: float = 1.0) -> Score:
    return Score(name, 1.0 if value else 0.0, bool(value), dict(details), weight=weight)


def _expected_list(expected: Mapping[str, Any], key: str, input_context: Mapping[str, Any]) -> list[str]:
    direct = expected.get(key)
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return [str(item) for item in direct if str(item).strip()]
    if key == "preferences":
        prefs = input_context.get("preferences") or expected.get("expected_preferences") or []
        return [str(_as_mapping(item).get("value") or item) for item in _as_list(prefs) if str(_as_mapping(item).get("value") or item).strip()]
    return []


def _action_rate(actions: Mapping[str, str], action: str) -> float:
    if not actions:
        return 0.0
    return sum(1 for value in actions.values() if value == action) / len(actions)


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9'-]+", str(value).lower()) if len(token) > 2}


def _recall(actual: set[str], expected: set[str]) -> float | None:
    if not expected:
        return None
    return len(actual & expected) / len(expected)


def _token_recall(source: str, target: str) -> float | None:
    source_tokens = _tokens(source)
    if not source_tokens:
        return None
    return len(source_tokens & _tokens(target)) / len(source_tokens)


def _has_overlap(left: Any, right: Any) -> bool:
    left_tokens = _tokens(left)
    if not left_tokens:
        return False
    right_tokens = _tokens(right)
    return len(left_tokens & right_tokens) / len(left_tokens) >= 0.5


def _joined(values: Sequence[Any]) -> str:
    return " ".join(json.dumps(value, ensure_ascii=False, default=str) for value in values if value)


def _between(value: Any, low: float, high: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return low <= number <= high


def _bucket(value: Any) -> str | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score >= 4.5:
        return "very close"
    if score >= 3.5:
        return "close"
    if score >= 2:
        return "moderate"
    return "low"


def _ordered_subsequence(expected: Sequence[str], actual: Sequence[str]) -> bool:
    positions = {value: index for index, value in enumerate(actual)}
    present = [positions[value] for value in expected if value in positions]
    return present == sorted(present) and len(present) == len(expected)


def _sensitive_hits(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False, default=str).lower()
    terms = ("religion", "politics", "diagnosis", "income", "pregnant", "medical condition")
    return [term for term in terms if term in text]


def _contains_copyrighty_signal(message: str) -> bool:
    lowered = message.lower()
    return any(term in lowered for term in ("lyrics:", "to the tune of", "as sung by", "quote from"))


def _image_magic_valid(path: Path) -> bool:
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return False
    return header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8\xff") or header.startswith(b"GIF87a") or header.startswith(b"GIF89a")


def _date_lte(left: str, right: str) -> bool:
    try:
        return date.fromisoformat(left[:10]) <= date.fromisoformat(right[:10])
    except ValueError:
        return False


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)
