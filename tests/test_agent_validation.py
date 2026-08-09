from typing import Any

from src.agents.llm import LLMProvider
from src.agents.orchestrator import GiftSession
from src.agents.recipient_profiling import RecipientProfilingAgent


class RepairingLLM:
    provider = LLMProvider.OLLAMA

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {"output": {}, "confidence": 4, "rationale": "invalid"}
        return {
            "output": {
                "interests": [],
                "communication_style": "warm",
                "gift_history_summary": "Unknown",
            },
            "confidence": 0.8,
            "rationale": "Corrected to the schema.",
        }


def test_invalid_structured_output_is_repaired_using_configured_retry() -> None:
    llm = RepairingLLM()
    session = GiftSession(
        session_id="s", giver_id="g", recipient_id="r", occasion_id="o"
    )
    result = RecipientProfilingAgent(llm).run(
        {"session": session, "stage_config": {"person": {"id": "r"}}}
    )
    assert result["confidence"] == 0.8
    assert len(llm.calls) == 2
    assert "failed JSON Schema validation" in llm.calls[1]["user_prompt"]
