from __future__ import annotations

import pytest

from src.harness import TraceRecorder, record_tool_call, trace_recorder_context


def test_trace_recorder_captures_successful_invocation_and_tool_call() -> None:
    recorder = TraceRecorder(session_id="session-1", case_id="case-1")

    with trace_recorder_context(recorder):
        with recorder.invocation(
            stage_name="relationship_analysis",
            agent_name="RelationshipAnalysisAgent",
            agent_version="static",
            raw_agent_input={"session": {"id": "session-1"}, "stage_config": {"api_token": "secret-value"}},
            context={"stage": "relationship_analysis"},
            relevant_memory={"memories": ["shared tea memory"]},
            constraints={"budget_hint": "USD 20-40"},
            model_provider="ollama",
            model_name="llama3.1",
        ) as invocation:
            record_tool_call(
                "query_memory_graph",
                arguments={"query": "tea", "password": "should-not-leak"},
                result={"matches": 1},
                latency_seconds=0.01,
            )
            invocation.raw_agent_output = {"output": {"tone_guidance": "warm"}}
            invocation.structured_output = {"tone_guidance": "warm"}

    run_trace = recorder.finish()
    assert run_trace.case_id == "case-1"
    assert run_trace.harness_id == "gmgi_default"
    assert len(run_trace.invocations) == 1

    captured = run_trace.invocations[0]
    assert captured.status == "success"
    assert captured.raw_agent_input["stage_config"]["api_token"] == "[REDACTED]"
    assert captured.tool_calls[0].tool_name == "query_memory_graph"
    assert captured.tool_calls[0].arguments["password"] == "[REDACTED]"
    assert captured.tool_calls[0].result == {"matches": 1}
    assert captured.routing_decision.decision == "relationship_analysis"
    assert captured.verifier_decision.decision == "not_available"


def test_trace_recorder_captures_failed_invocation() -> None:
    recorder = TraceRecorder(session_id="session-1")

    with pytest.raises(RuntimeError):
        with recorder.invocation(
            stage_name="recipient_profiling",
            agent_name="RecipientProfilingAgent",
            agent_version="static",
            raw_agent_input={"stage_config": {}},
        ):
            raise RuntimeError("model unavailable")

    trace = recorder.finish()
    invocation = trace.invocations[0]
    assert invocation.status == "error"
    assert invocation.error_type == "RuntimeError"
    assert "model unavailable" in str(invocation.error_message)


def test_trace_recorder_records_timeout_event_and_dependencies() -> None:
    recorder = TraceRecorder(session_id="session-1", case_id="case-1")
    with recorder.invocation(
        stage_name="recipient_profiling",
        agent_name="RecipientProfilingAgent",
        agent_version="static",
        raw_agent_input={"stage_config": {}},
    ):
        pass

    previous = recorder.invocations[-1].invocation_id
    timeout = recorder.record_timeout(
        stage_name="relationship_analysis",
        agent_name="RelationshipAnalysisAgent",
        raw_agent_input={"stage_config": {"query": "memory"}},
        latency_seconds=1.5,
        error_message="stage exceeded 1.5s timeout",
        dependency_ids=[previous],
    )

    trace = recorder.finish()
    assert timeout.status == "timeout"
    assert timeout.dependency_ids == [previous]
    assert trace.invocations[1].sequence_number == 2
    assert trace.invocations[1].error_type == "TimeoutError"
    assert trace.total_latency_seconds is not None


def test_trace_recorder_preserves_parent_invocation_id() -> None:
    recorder = TraceRecorder(session_id="session-1")
    with recorder.invocation(
        stage_name="planner",
        agent_name="PlannerAgent",
        agent_version="static",
        raw_agent_input={"stage_config": {}},
    ):
        pass

    parent_id = recorder.invocations[0].invocation_id
    with recorder.invocation(
        stage_name="recommendation",
        agent_name="RecommendationAgent",
        agent_version="static",
        raw_agent_input={"stage_config": {}},
        parent_invocation_id=parent_id,
    ):
        pass

    trace = recorder.finish()
    assert trace.invocations[1].parent_invocation_id == parent_id
    assert trace.invocations[1].sequence_number == 2
