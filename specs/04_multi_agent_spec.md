# 04 â€” Multi-Agent Orchestration Spec

## Orchestration pattern
**Blackboard + sequential pipeline with revision loops**, not free-form agent chat. Rationale: keeps every decision attributable to exactly one agent or the human (required for the Agency Ledger, `08_demo_ui_spec.md`), and keeps the system debuggable within project time constraints.

Shared state object, `GiftSession` (JSON, versioned, append-only):
```json
{
  "session_id": "...",
  "giver_id": "...",
  "recipient_id": "...",
  "occasion_id": "...",
  "stage_log": [
    {
      "stage": "recipient_profiling",
      "proposed_by": "agent",
      "output": { "...": "..." },
      "human_action": "accept | edit | regenerate | delegate",
      "human_edit": { "...": "..." } ,
      "timestamp": "..."
    }
  ]
}
```
Each agent reads the current `GiftSession`, appends one `stage_log` entry, and returns control to the orchestrator/human before the next stage runs. No agent may edit a previous stage's `output` field directly (append corrections as new entries) â€” preserves provenance.

## Agents

### 1. Recipient Profiling Agent
- **Input**: `Person` node + raw free-text notes.
- **Output**: structured profile â€” interests (ranked, with confidence), communication style, gift-history summary.
- **Impl**: structured-output LLM call through the selectable provider layer (Ollama, Azure OpenAI, OpenAI, Gemini, or Claude), temperature low (~0.3) for consistency. Provider selection uses `GMGI_LLM_PROVIDER` when set, otherwise available credential environment variables, falling back to local Ollama.

### 2. Relationship Analysis Agent
- **Input**: `Relationship` node + `Memory` subgraph.
- **Output**: closeness assessment, relationship-appropriate tone/formality guidance, risk flags (e.g., "avoid overly intimate tone for a new-coworker relationship").
- **Impl**: LLM call, references the source paper's finding that social closeness governs willingness to accept AI recommendations â€” this agent's output directly gates how much agency the system should assume by default (feeds `agency_slider` default).

### 3. Memory Graph Agent
- Not LLM-based; thin wrapper around `02_memory_graph_spec.md` API (`context_embedding`, `subgraph_for`). Provides the pooled embedding all downstream agents/GAN condition on.

### 4. Recommendation Agent
- **Input**: profile + relationship guidance + budget/occasion.
- **Output**: ranked list of gift *categories/concepts* (not necessarily a generated artifact â€” may recommend a physical product idea *and/or* a generated-artifact concept).
- **Impl**: LLM call with retrieval-augmented context from the memory graph; optionally a simple content-based scoring function over a small demo product catalog (CSV) for the "recommend an existing item" path, kept separate from the generative path so the paper can compare them.

### 5. Creative Generation Agent
- Wraps `MemoryGAN.generate(...)` (spec 03). Produces the visual artifact given the conditioning vector and current `agency_slider`.

### 6. Greeting/Story Agent
- **Input**: relationship guidance, occasion, one or two salient `Memory` nodes, tone guidance.
- **Output**: short original message/caption/story (never reproduces real copyrighted text/lyrics â€” original generation only).
- **Impl**: structured-output call through the selectable LLM provider layer. Prompts, temperatures, provider-specific model names, and JSON schemas live in `src/agents/configs/*.json`, not in agent code.

### 7. Delivery Planner (stub)
- Rule-based, simulated only: outputs a mocked "delivery plan" (e.g., "digital card, send now" / "physical item, ship by <date>"). No real logistics integration in MVP â€” documented explicitly as future work.

### 8. Feedback/RL Agent
- Not conversational; consumes `stage_log` + post-session satisfaction rating and updates the bandit policy (spec 05).

## Agent IO contract (all agents)
```python
class AgentInput(TypedDict):
    session: GiftSession
    stage_config: dict          # per-stage knobs (e.g., temperature, agency_slider)

class AgentOutput(TypedDict):
    stage: str
    output: dict                 # stage-specific schema, documented per agent above
    confidence: float | None
    rationale: str | None        # short natural-language justification, shown to human in UI
```

## Human-in-the-loop contract
After each stage, orchestrator pauses and the Agency Console (spec 08) presents `output` + `rationale` with four actions:
- **Accept** â€” proceed as-is.
- **Edit** â€” human overwrites part of `output`; recorded as `human_edit`.
- **Regenerate** â€” same agent reruns with a nudged prompt/seed.
- **Delegate** â€” human explicitly defers full control to the agent for this and remaining stages (fast-forward mode) â€” this is itself a meaningful, logged agency choice, not a UI shortcut to hide.

## Failure handling
- Any agent call that fails (LLM error, GAN OOM, etc.) surfaces as a stage marked `"status": "error"` in the log with the raw error message, and the orchestrator offers retry/skip â€” never silently substitutes fabricated output.
