# Agent Harness Architecture Audit

## 1. Executive Summary

The current GMGI implementation is best classified as a **Configurable Multi-Agent Workflow**, not a full dynamic agent harness.

It has a real multi-agent runtime: eight specialized stages, structured agent contracts, schema validation, local/remote model selection, limited tool-calling agents, a memory graph, an episodic experience store, human approval/edit/regenerate/delegate actions, feedback-to-bandit updates, and an evaluation layer. However, the central execution policy is still mostly hardcoded in `AgencyConsoleService.STAGES` and `_run_stage(...)`. The planner produces a structural plan, but that plan is not the runtime authority for execution. Agents do not generally choose the next agent, revise the workflow, request tools through a shared permissions layer, manage budgets, or persist full traces of model/tool calls.

The strongest harness-like capabilities are:

- sequential multi-agent orchestration with append-only session ledger;
- per-stage structured I/O and schema validation;
- human-in-the-loop proposal review;
- episodic memory injection through `ExperienceRetriever`;
- limited tool use in `smolagents` relationship/recommendation paths;
- offline evaluation for schema, constraints, provenance, quality, faithfulness, judge scoring, and benchmark permutations.

The main blocker to claiming a proper agent harness is that **the runtime control layer is not adaptive**. The workflow is selected by code, not by an agent/harness policy. Planning, tool permissions, recovery, budget, tracing, and verification exist in pieces, but they are not unified into a harness that controls execution.

## 2. Current Architecture

### Entry Points

- FastAPI app: `src/api/app.py:create_app(...)`.
- Backend service/runtime: `src/api/service.py:AgencyConsoleService`.
- Frontend API client: `frontend/src/api.ts`.
- Frontend Agency Console: `frontend/src/main.tsx`.
- CLI evals: `src/evals/run.py`.
- Benchmark/permutation runner: `src/evals/benchmark.py`, `src/evals/permutations.py`.
- Offline fixture pipeline: `scripts/run_fixture_pipeline.py`.

### UI to Backend Flow

The UI defines the visible stage list in `frontend/src/main.tsx:11` and calls backend endpoints through `frontend/src/api.ts`. A user creates a session, then proposes a stage, then chooses accept/edit/regenerate/delegate. The frontend only enables the next stage when the backend response says `next_stage` matches that stage.

Backend routes are in `src/api/app.py`:

- `POST /sessions`: create session.
- `POST /sessions/{session_id}/stages/{stage}/propose`: run one stage.
- `POST /sessions/{session_id}/stages/{stage}/accept`: accept pending proposal.
- `POST /sessions/{session_id}/stages/{stage}/edit`: apply human JSON edit.
- `POST /sessions/{session_id}/stages/{stage}/regenerate`: rerun current pending stage.
- `POST /sessions/{session_id}/stages/{stage}/delegate`: delegate remaining stages.
- `GET /sessions/{session_id}/ledger`: ledger summary.
- `POST /sessions/{session_id}/feedback`: reward/bandit/experience update.

### Agent Definitions

The eight GMGI stages are:

1. `RecipientProfilingAgent`
2. `RelationshipAnalysisAgent`
3. `GiftIntentReasoningAgent`
4. `MultiAgentPlanningAgent`
5. `RecommendationAgent`
6. `CreativeGenerationAgent`
7. `GreetingStoryAgent`
8. `DeliveryPlannerAgent`

Most structured agents inherit from `src/agents/base.py:StructuredAgent`. Creative and delivery use custom classes because creative generation and delivery date math are not simple schema-only LLM calls.

### Agent Prompts and Schemas

Prompts, schemas, model names, temperatures, and declared skills live under `src/agents/configs/*.json`. `StructuredAgent.run(...)` loads `prompt_template`, `system_prompt`, `output_schema`, temperature, and provider-specific model mapping.

### Orchestration Code

`src/agents/orchestrator.py:AgentOrchestrator` is an append-only session controller. It does not select which agent to run. Selection happens in `src/api/service.py:AgencyConsoleService._run_stage(...)` through a hardcoded `if/elif` stage switch.

### Memory and State

- Working memory: `GiftSession.stage_log`, `ConsoleSession`, and `AgencyConsoleService.sessions`.
- Episodic memory: `ExperienceStore` appends post-feedback episodes to JSONL and `ExperienceRetriever` retrieves successful similar episodes into future system prompts.
- Semantic memory: fixture/live context is converted into a `MemoryGraph` for fixture sessions; live custom sessions use hashed embeddings and plain dictionaries.
- Bandit state: `LinUCBBandit` persists arms/counts in JSON.

### Evaluation

`src/evals` implements additive observability and evals. It evaluates logs and benchmark cases; it does not control live agent execution.

## 3. Runtime Execution Flow

Representative real UI request: a custom live session for a birthday gift, then stage-by-stage proposal.

```text
UI form
  -> frontend api.createSession(...)
  -> POST /sessions
  -> AgencyConsoleService.create_session(...)
  -> GiftSession + ConsoleSession stored in memory
  -> UI receives next_stage
  -> User clicks Generate proposal
  -> POST /sessions/{id}/stages/{stage}/propose
  -> AgencyConsoleService.propose(...)
  -> AgencyConsoleService._run_stage(...)
  -> selected Agent.run(...)
  -> AgentOutput
  -> AgentOrchestrator.append_agent_output(...)
  -> pending StageLogEntry
  -> UI displays proposal
  -> User accept/edit/regenerate/delegate
  -> AgentOrchestrator.apply_human_action(...) or regenerate/delegate path
  -> next_stage recomputed from hardcoded STAGES
```

For every transition:

| Transition | Controller | Deterministic or Agent-Controlled? | Context Passed | State Maintained | Failure Behavior |
|---|---|---|---|---|---|
| UI setup to session | Frontend + FastAPI route | Deterministic | Live form/profile or fixture ids | `ConsoleSession`, `GiftSession` | Pydantic/ValueError to 4xx |
| Stage proposal request | Frontend button gating via `next_stage` | Deterministic | `session_id`, `stage`, overrides | Session in service dict | route maps exceptions to 409/502 |
| Stage selection | `AgencyConsoleService._run_stage` | Hardcoded | fixture/live context plus upstream outputs | none beyond service session | exception bubbles to API |
| Agent execution | specific agent class | Mixed: LLM generation or deterministic fallback | `AgentInput(session, stage_config)` | prompt version, optional cached model/pipeline | schema retry/repair/fallback varies |
| Tool access | smolagents paths only for relationship/recommendation | Agent can choose among provided local tools inside `ToolCallingAgent` | context JSON and tool functions | tool outputs not centrally traced | fallback to schema/repair in some paths |
| Output append | `AgentOrchestrator` | Deterministic | `AgentOutput` | immutable `stage_log` entry | blocks if previous proposal pending |
| Human review | UI/user + orchestrator | Human-controlled | accept/edit/regenerate/delegate | human entries in `stage_log` | edit requires JSON/non-empty |
| Next step | `AgencyConsoleService.next_stage` | Deterministic | completed stages | no separate policy state | returns first uncompleted `STAGES` member |
| Feedback | user + service | Deterministic reward formula | rating, authorship, measures, stage log | bandit state, experience JSONL | 4xx on validation errors |

The `MultiAgentPlanningAgent` can output `agent_sequence`, `dependencies`, and `fallback_plan`, but `next_stage(...)` does not consume that plan. Therefore planning is observable and useful to later stages, but not execution-authoritative.

## 4. Harness Capability Matrix

| Harness Component | Status | Evidence | Missing Capability | Priority |
|---|---|---|---|---|
| Reasoning | PARTIALLY IMPLEMENTED | LLM/heuristic agents produce rationales and structured decisions. | No explicit reasoning trace used by controller; rationales do not drive routing. | P0 |
| Planning | PARTIALLY IMPLEMENTED | `MultiAgentPlanningAgent` emits plan/dependencies. | Plan is not executed by orchestrator; no adaptive replanning. | P0 |
| Search | PARTIALLY IMPLEMENTED | Memory graph/query tools and episodic retrieval. | No external/product/web search; limited adaptive RAG. | P1 |
| Reflection | PARTIALLY IMPLEMENTED | Creative CLIP critique can refine prompt/regenerate. | Reflection limited to creative image loop; no general generate-critique-improve harness. | P1 |
| Verification | PARTIALLY IMPLEMENTED | JSON schema validation, eval metrics, CLIP critique, MemoryGAN size validation. | Semantic verification not in live gate for most stages. | P0 |
| Tools | PARTIALLY IMPLEMENTED | smolagents tools for memory and bandit; date math; image generation. | No shared tool registry/permission runtime; tool calls not centrally traced. | P0 |
| APIs | PARTIALLY IMPLEMENTED | Ollama/OpenAI/Azure/Gemini/Claude, Diffusers/HF, FastAPI. | Agents cannot dynamically call arbitrary APIs; API choice mostly configured. | P1 |
| Code | NOT IMPLEMENTED | No agent code execution loop. | Code execution/sandbox/tool-result iteration. | P2 |
| Environment | PARTIALLY IMPLEMENTED | Filesystem artifacts, checkpoints, JSONL state, local Ollama, generated images. | No environment abstraction or permissions policy per agent. | P1 |
| Human | IMPLEMENTED | accept/edit/regenerate/delegate/feedback in UI and API. | No granular approval policy by action/tool risk. | P0 |
| Working Memory | IMPLEMENTED | `GiftSession.stage_log`, upstream outputs via `_safe_effective`. | No summarization/compression. | P0 |
| Episodic Memory | PARTIALLY IMPLEMENTED | `ExperienceStore` + `ExperienceRetriever` influence prompts after feedback. | Only post-feedback accepted examples; raw stage inputs absent. | P1 |
| Semantic Memory | PARTIALLY IMPLEMENTED | `MemoryGraph` fixtures/live context. | Not a durable evolving knowledge base for live users. | P1 |
| Context Management | PARTIALLY IMPLEMENTED | Stage-specific `build_context`, context fingerprints, few-shot retrieval. | No token budgeting, adaptive context selection, or summarization. | P0 |
| Sequential | IMPLEMENTED | `STAGES` order and `next_stage`. | None for static flow. | P0 |
| DAG | IMPLICIT / EMERGENT | Planner/evals produce/check dependencies. | Runtime does not execute DAG. | P1 |
| Dynamic | PARTIALLY IMPLEMENTED | human actions, delegate, regenerate, env-config methods/backend. | Agents cannot alter route; confidence/errors do not replan. | P0 |
| Router | PARTIALLY IMPLEMENTED | `select_provider`, `_creative` backend choice, hardcoded stage switch. | No policy router selecting agents/tools/workflows by context. | P0 |
| Hierarchical | IMPLICIT / EMERGENT | Planning stage resembles manager output. | Planner does not control sub-agents. | P2 |
| Parallel | NOT IMPLEMENTED | Runtime stage execution is serial; eval uses thread timeout per stage but not parallel agents. | Concurrent independent stage execution. | P2 |
| Multi-Agent | IMPLEMENTED | Eight specialized agents exchange outputs through shared session context. | Coordination is blackboard/static, not conversational multi-agent deliberation. | P0 |
| Budget | PARTIALLY IMPLEMENTED | max validation retries, smolagents max_steps, eval timeouts. | No token/cost/tool-call budget at harness level. | P0 |
| Stopping | PARTIALLY IMPLEMENTED | fixed stage completion, pending human gate, delegate loop ends at no `next_stage`. | No evaluator/confidence/convergence stopping policy. | P0 |
| Model Choice | PARTIALLY IMPLEMENTED | env/provider/model selection and per-stage config. | No adaptive model choice by complexity/cost/failure. | P1 |
| Retry | IMPLEMENTED | schema validation retry/repair, creative critique retries, regenerate endpoint. | Retry strategy is local, not centrally governed. | P1 |
| Recovery | PARTIALLY IMPLEMENTED | deterministic repair/fallback paths for several agents. | No diagnosis-driven alternative tool/workflow recovery. | P1 |
| Checkpoint | PARTIALLY IMPLEMENTED | session logs, experience store, bandit state, prompt versions, MemoryGAN checkpoints. | In-memory active sessions are not resumable after process restart. | P1 |
| Fallback | IMPLEMENTED | provider fallbacks/repairs, demo responses, deterministic repairs, diffusers vs MemoryGAN. | Some fallback can mask real failures unless strict env flags are used. | P1 |
| Permissions | PARTIALLY IMPLEMENTED | Tools are scoped by code to agents; artifact path safety. | No formal permissions engine or policy audit. | P0 |
| Guardrails | PARTIALLY IMPLEMENTED | Pydantic requests, schema validation, simulated delivery-only, no path traversal, prompt safety constraints. | Limited input moderation and semantic/safety guardrails. | P0 |
| Human Oversight | IMPLEMENTED | proposals wait for human action unless delegated. | No differentiated approval levels. | P0 |
| Audit | PARTIALLY IMPLEMENTED | ledger, rationales, skills metadata, experience store. | Missing full raw inputs, prompts, model responses, tool calls/results, costs. | P0 |
| Tracing | PARTIALLY IMPLEMENTED | stage log, eval benchmark traces with latency/errors. | No live full trace of model/tool invocations. | P0 |
| Evaluation | IMPLEMENTED | `src/evals` phases plus benchmark/permutations. | Harness control decisions are weakly evaluated because runtime lacks adaptive control traces. | P1 |
| Cost | NOT IMPLEMENTED | No token or monetary cost fields found. | model/tool token accounting and estimated cost. | P1 |
| Latency | PARTIALLY IMPLEMENTED | benchmark stage latency; not live ledger. | live per-agent/tool/model latency and retry overhead. | P1 |
| Failure Analysis | PARTIALLY IMPLEMENTED | API 502 details, benchmark error traces. | Missing input/tool/prompt context for root-cause trace. | P1 |

## 5. Cognition Audit

### 5.1 Reasoning - PARTIALLY IMPLEMENTED

Evidence:

- `StructuredAgent.run(...)` invokes LLMs with prompts and schemas.
- `GiftIntentReasoningAgent` performs deterministic intent extraction when heuristic methods are selected.
- `RecommendationAgent` ranks concepts from profile, relationship, intent, plan, preferences, budget.
- Outputs carry `rationale` and confidence fields.

Runtime path:

- `AgencyConsoleService._run_stage(...)` constructs stage configs and calls `agent.run(...)`.
- Downstream stages consume structured outputs via `_safe_effective(...)`.

Limitations:

- Reasoning is not represented as a separate state object.
- Rationale is recorded but not used to decide next stage, tool permissions, recovery, or stopping.
- There is no chain-level reasoning trace or deliberation controller.

### 5.2 Planning - PARTIALLY IMPLEMENTED

Evidence:

- `MultiAgentPlanningAgent._run_rule_planner(...)` emits `task_goal`, `subtasks`, `agent_sequence`, `dependencies`, `expected_outputs`, `stop_conditions`, and `fallback_plan`.
- `src/evals/structural.py:dag_validity(...)` evaluates the emitted plan.

Runtime path:

- The planning output is passed into recommendation as `execution_plan`.
- Runtime `next_stage(...)` still follows `STAGES`, not the planner output.

Limitations:

- Planner cannot route execution.
- Planner cannot change stage order at runtime.
- Plan cannot trigger or skip agents.
- No replanning after failure/tool result/user edit.

### 5.3 Search - PARTIALLY IMPLEMENTED

Evidence:

- `ExperienceRetriever.augment_system_prompt(...)` retrieves similar accepted high-reward episodes by `context_fingerprint`.
- `RelationshipAnalysisAgent` exposes `query_memory_graph` to smolagents.
- `RecommendationAgent` exposes `query_memory_graph` and `bandit_feedback_hint`.
- `MemoryGraph.subgraph_for(...)` queries fixture graph context.

Limitations:

- No live product/database/web search.
- Episodic retrieval is fixed by context fingerprint overlap, not agent-selected.
- Tool results are not centrally traced.
- Live custom sessions do not build a durable graph object for tool querying.

### 5.4 Reflection - PARTIALLY IMPLEMENTED

Evidence:

- `CreativeGenerationAgent` can generate an image, score with CLIP, refine the prompt, and retry up to `GMGI_CRITIQUE_MAX_RETRIES`.

Limitations:

- Reflection exists only in creative generation.
- No reflection for recipient profile, relationship, intent, planning, recommendation, story, or delivery.
- CLIP score is an alignment proxy, not full semantic critique.

### 5.5 Verification - PARTIALLY IMPLEMENTED

Evidence:

- JSON schema validation in `StructuredAgent.run(...)`, recipient/relationship/recommendation/greeting custom paths, and evals.
- `src/evals/structural.py` checks schema, constraints, DAG validity, provenance.
- `src/evals/quality.py` checks stage quality and creative artifact usability.
- Creative generation validates MemoryGAN image size.

Limitations:

- Most verification is offline, not live-gating.
- Semantic/factual verification is not enforced during the live pipeline.
- Evals cannot replay because logged episodes lack raw per-stage `AgentInput`.

## 6. Action Audit

### 6.1 Tools - PARTIALLY IMPLEMENTED

Tools available at runtime:

| Tool | Agent | Dynamic? | Fed Back Into Reasoning? | Evidence |
|---|---|---|---|---|
| `query_memory_graph` | RelationshipAnalysisAgent | Yes, inside smolagents | Yes, into tool-calling agent response | `relationship_analysis.py` |
| `query_memory_graph` | RecommendationAgent | Yes, inside smolagents | Yes | `recommendation.py` |
| `bandit_feedback_hint` | RecommendationAgent | Yes, inside smolagents | Yes | `recommendation.py` |
| `date_logistics_math` | DeliveryPlannerAgent | No, deterministic precompute | Yes, Instructor structures around result | `delivery_planner.py` |
| `visual_prompt_builder` | CreativeGenerationAgent | No, deterministic template | Yes, prompt drives image generation | `creative_generation.py` |
| `diffusers_image_generation` | CreativeGenerationAgent | No, backend selected by config/env | Output artifact | `creative_generation.py` |
| `clip_critic` | CreativeGenerationAgent | No, conditional loop | Yes, can refine prompt | `creative_generation.py` |
| `ExperienceRetriever` | most agents via base/system prompt | No, fixed injection | Yes, prompt augmentation | `experience_retriever.py` |

Limitations:

- There is no shared tool registry or permissions engine.
- Tool calls/results are not logged in live traces.
- Tool availability is hardcoded in each agent.

### 6.2 APIs - PARTIALLY IMPLEMENTED

External APIs/libraries:

- Ollama local API via `/api/chat` and OpenAI-compatible `/v1`.
- OpenAI, Azure OpenAI, Gemini, Claude through `HTTPStructuredLLM`.
- Diffusers/Hugging Face model loading.
- `open_clip` for image-text scoring.

Agents cannot dynamically choose arbitrary APIs. Provider/model choice is config/env-driven.

### 6.3 Code - NOT IMPLEMENTED

No agent can write, run, inspect, and iterate on code as part of the GMGI runtime.

### 6.4 Environment - PARTIALLY IMPLEMENTED

Agents interact indirectly with:

- filesystem for generated artifacts;
- JSONL experience store;
- bandit state file;
- prompt version files;
- local Ollama/Diffusers model environment.

There is no environment abstraction with per-agent capabilities.

### 6.5 Human - IMPLEMENTED

Human actions:

- accept;
- edit;
- regenerate;
- delegate;
- submit feedback and authorship/satisfaction measures.

Evidence:

- `HumanAction` enum in `orchestrator.py`.
- endpoints in `app.py`.
- UI buttons in `frontend/src/main.tsx`.

## 7. State Audit

### 7.1 Working Memory - IMPLEMENTED

Evidence:

- `GiftSession.stage_log` records stage outputs and human actions.
- `AgencyConsoleService._safe_effective(...)` retrieves upstream outputs.
- `ConsoleSession` stores fixture, orchestrator, agency slider, seed, budget hint.

Limitations:

- Active sessions are held in process memory.
- There is no context compression/summarization.

### 7.2 Episodic Memory - PARTIALLY IMPLEMENTED

Evidence:

- `ExperienceStore.append(...)` persists episodes after feedback.
- `ExperienceRetriever` injects successful prior outputs into prompts.

Limitations:

- Episodic memory is only updated after feedback.
- Retrieval is simple fingerprint overlap.
- Raw per-stage inputs/prompts/tool calls are not stored, so replay is incomplete.

### 7.3 Semantic Memory - PARTIALLY IMPLEMENTED

Evidence:

- `MemoryGraph` stores people, relationships, occasions, events, memories, preferences.
- Fixture sessions can load graph context through `load_fixture(...)`.

Limitations:

- Live custom profile data is not persisted as an evolving semantic memory graph.
- Memory graph is not a general durable user profile store.

### 7.4 Context Management - PARTIALLY IMPLEMENTED

Evidence:

- Each agent has `build_context(...)`.
- Service composes stage-specific configs from fixture/live context and upstream outputs.
- `ExperienceRetriever` provides compact retrieved examples.
- Creative prompt caps words through `GMGI_IMAGE_PROMPT_MAX_WORDS`.

Limitations:

- No token budget or context-window accounting.
- Context is mostly static and hand-authored per stage.
- No adaptive summarizer or retrieval planner.

## 8. Orchestration Audit

### 8.1 Sequential - IMPLEMENTED

Evidence:

- `STAGES` tuple in `src/api/service.py`.
- `next_stage(...)` returns the first uncompleted stage.
- UI disables stages except current `next_stage`.

### 8.2 DAG - IMPLICIT / EMERGENT

Evidence:

- `MultiAgentPlanningAgent` emits dependencies.
- `src/evals/structural.py:dag_validity(...)` checks plan dependencies.

Limitation:

- The actual runtime does not execute a DAG; it executes a fixed sequence.

### 8.3 Dynamic - PARTIALLY IMPLEMENTED

Dynamic elements:

- Human can edit/regenerate/delegate.
- `agency_slider` changes creative output.
- Environment flags select methods/backends/providers.
- Planner can include clarifying human subtasks in output.

Not dynamic:

- Agent outputs do not choose next stage.
- Confidence does not route to verifier/retry/escalation.
- Planner output does not alter execution.

### 8.4 Router - PARTIALLY IMPLEMENTED

Evidence:

- `_run_stage(...)` routes stage name to agent class.
- `select_provider(...)` selects LLM provider based on env/credentials.
- `_creative(...)` routes Diffusers vs MemoryGAN.

Limitation:

- Routing is mostly hardcoded or environment-configured, not adaptive policy.

### 8.5 Hierarchical - IMPLICIT / EMERGENT

The planning agent resembles a manager/planner, but it does not command sub-agents. Therefore hierarchical control is not implemented at runtime.

### 8.6 Parallel - NOT IMPLEMENTED

Runtime agent execution is serial. Benchmark uses a worker thread to enforce timeouts, not to parallelize independent agents.

### 8.7 Multi-Agent - IMPLEMENTED

There are eight specialized agents. They communicate through blackboard-style upstream outputs in `GiftSession.stage_log` and service-composed configs. The communication is structured and one-directional along the pipeline.

## 9. Control Audit

### 9.1 Budget - PARTIALLY IMPLEMENTED

Evidence:

- `max_validation_retries` in runtime config.
- smolagents `max_steps`.
- benchmark `stage_timeout`.
- creative `critique_max_retries`.

Missing:

- token budget;
- cost budget;
- total execution budget;
- per-tool call budget;
- budget-aware routing.

### 9.2 Stopping - PARTIALLY IMPLEMENTED

Implemented:

- fixed stop after all stages complete;
- pending human gate;
- delegate loop stops when `next_stage` is `None`;
- creative loop stops on CLIP threshold or retry limit.

Missing:

- evaluator-driven stop;
- confidence/convergence stop;
- planner-controlled stop.

### 9.3 Routing - PARTIALLY IMPLEMENTED

Routing is hardcoded by `STAGES` and `_run_stage`, plus env/config switches for model/backend/method.

### 9.4 Model Choice - PARTIALLY IMPLEMENTED

`HTTPStructuredLLM.select_provider(...)` and per-agent config model mappings support provider/model choice. It is not adaptive by task complexity, latency, cost, or failure history.

## 10. Reliability Audit

### 10.1 Retry - IMPLEMENTED

Evidence:

- `StructuredAgent.run(...)` retries schema validation and modifies prompt using `repair_prompt_template`.
- Creative generation retries after CLIP critique.
- Human regenerate reruns the pending stage.

### 10.2 Recovery - PARTIALLY IMPLEMENTED

Evidence:

- Recipient profiling has schema JSON and deterministic repair.
- Intent/planning/recommendation have deterministic repair/fallback modes.
- Delivery falls back to deterministic plan if Instructor path fails.
- API returns structured 502 details.

Limitations:

- Recovery is local to agents and env flags.
- No central failure diagnosis or alternative tool routing.

### 10.3 Checkpoint - PARTIALLY IMPLEMENTED

Evidence:

- Append-only stage logs exist inside session object.
- Experience store and bandit state persist to disk.
- Prompt versions persist to disk.
- MemoryGAN checkpoints can be loaded.

Limitations:

- Active UI sessions are not persisted/resumable across backend process restart.
- Raw inputs/prompts/tool calls are not checkpointed.

### 10.4 Fallback - IMPLEMENTED

Evidence:

- Provider fallbacks are possible through `select_provider`.
- Demo fixture LLM exists behind env flag.
- Agents have deterministic repair paths.
- Creative can use Diffusers by default or MemoryGAN if explicitly configured.

Limitation:

- Fallbacks are not centrally governed and can obscure failures unless strict flags disable them.

## 11. Governance Audit

### 11.1 Permissions - PARTIALLY IMPLEMENTED

Permissions are implicit in code:

- relationship agent only receives memory graph tool;
- recommendation receives memory and bandit tools;
- delivery has no real shipping API;
- artifacts endpoint blocks absolute/parent paths.

There is no formal permissions model, policy file, or runtime enforcement layer.

### 11.2 Guardrails - PARTIALLY IMPLEMENTED

Evidence:

- Pydantic request validation.
- JSON schema validation.
- delivery planner forbids real purchase/shipping.
- artifact path safety check.
- creative negative prompt avoids logos/private text.
- MemoryGAN tiny-artifact validation.

Missing:

- semantic safety/factual guardrails for live LLM outputs;
- policy classification;
- risk-based human approval.

### 11.3 Human Oversight - IMPLEMENTED

The default flow requires human action after each proposal unless delegated. UI supports accept/edit/regenerate/delegate and feedback.

### 11.4 Audit - PARTIALLY IMPLEMENTED

Available:

- stage, actor, output, confidence, rationale, timestamp, status;
- human action/edit;
- feedback reward/bandit action;
- prompt versions in `ExperienceStore`;
- skills metadata in most real agent paths.

Missing:

- raw stage configs;
- prompts;
- model/provider call metadata;
- token/cost;
- tool calls/results;
- retry trace;
- full failure context.

## 12. Observability Audit

### 12.1 Tracing - PARTIALLY IMPLEMENTED

Live trace has stage ledger but not full model/tool trace. Benchmark traces include per-stage status, latency, confidence, and error type. `src/evals/replay.py` explicitly reports that raw `AgentInput` payloads are missing for replay.

### 12.2 Evaluation - IMPLEMENTED

Current evals map as follows:

| Eval | Harness Area Tested | Limitation |
|---|---|---|
| schema conformance | verification | output-only |
| constraint satisfaction | verification/governance | limited semantic coverage |
| DAG validity | planning/orchestration | checks plan output, not runtime execution |
| provenance traceability | audit/grounding | text-match style heuristic |
| faithfulness | verification | reference-free claim extraction; optional verifier |
| self-consistency/counterfactual | reliability/state | blocked on missing raw inputs for logged replay |
| purpose alignment judge | governance/verification | disabled unless judge model is enabled |
| quality benchmark | agent-level output quality | controlled synthetic/live-like cases |
| UI permutations | coverage over UI input axes | mainly final pipeline behavior, not harness internals |

### 12.3 Cost - NOT IMPLEMENTED

No token count, model call count, tool call count, or monetary estimates are logged.

### 12.4 Latency - PARTIALLY IMPLEMENTED

Benchmark logs per-stage latency and timeout. Live session ledger does not include latency or retry overhead.

### 12.5 Failure Analysis - PARTIALLY IMPLEMENTED

The API can identify stage and exception type. Benchmark traces capture stage/error/latency. But without raw inputs/prompts/tool traces, failure analysis is incomplete.

## 13. Workflow vs Agent Harness Assessment

Most accurate classification: **Configurable Multi-Agent Workflow**.

It is more than a static pipeline because:

- model/provider/backend/methods are configurable;
- human actions can pause, edit, regenerate, or delegate;
- episodic memory can influence prompts;
- selected agents can use tools;
- feedback updates a bandit policy;
- evals and prompt versions exist.

It is not yet a dynamic agent harness because:

- the runtime stage order is hardcoded;
- planner output does not control execution;
- there is no central policy for routing, tools, permissions, budget, recovery, or verification;
- tool/model calls are not fully traced;
- agents cannot request other agents or stop/skip workflow steps.

It is not a full agent harness because adaptive cognition, state, action, control, reliability, governance, and observability are not integrated into one runtime control layer.

## 14. GMGI Agent Mapping

| Stage | Actual Agent? | Runtime Tool Use | Memory Access | Can Affect Later Stages? | Can Control Routing? | Retry/Recovery | Output Verified? | Persisted? |
|---|---|---|---|---|---|---|---|---|
| Recipient Profiling | Yes | Instructor structured extraction; no tools | prompt retrieval via retriever | Yes, profile used downstream | No | Instructor/schema/repair | schema | stage log/experience |
| Relationship Analysis | Yes | smolagents `query_memory_graph` | fixture memory graph + retriever | Yes, tone/formality used downstream | No | schema fallback | schema | stage log/experience |
| Gift Intent Reasoning | Agent-like hybrid | deterministic or LLM structured | upstream outputs/retriever | Yes, artifact type/style/constraints | No | deterministic repair | schema if LLM; offline eval | stage log/experience |
| Multi-Agent Planning | Agent-like planner | no runtime tools | upstream outputs/retriever | Yes, plan visible to recommendation | No, despite producing plan | deterministic repair | schema/DAG eval | stage log/experience |
| Recommendation | Yes | smolagents memory and bandit tools | upstream outputs, bandit, retriever | Yes, selected concept informs feedback and creative indirectly | No | deterministic ranked repair | schema/offline quality | stage log/experience |
| Creative Generation | Yes, multimodal generator | Diffusers/MemoryGAN, optional CLIP critic | context embedding, style, retriever | Yes, artifact used in UI/feedback | No | CLIP retries; MemoryGAN validation | size/artifact eval | image file/stage log |
| Greeting Story | Yes | Ollama chat, no tools | memories/upstream tone/retriever | Yes, final message | No | fallback to structured base | schema | stage log/experience |
| Delivery Planner | Hybrid deterministic tool + optional LLM structurer | date logistics math | occasion/artifact type | Final output only | No | deterministic fallback | local schema/eval | stage log/experience |

Answers to specific questions:

1. Actual agents: recipient, relationship, recommendation, creative, greeting; intent/planning are hybrid agent-like; delivery is deterministic tool plus optional structured LLM.
2. Merely pipeline stages: none are only empty placeholders, but delivery/planning are not autonomous execution controllers.
3. Controlled by harness: proposal gating, stage order, ledger, human actions, feedback persistence.
4. Hardcoded: `STAGES`, `_run_stage` selection, dependency order, most context construction.
5. Agents that decide subsequent execution: none at runtime.
6. Agents that call tools: relationship, recommendation, creative, delivery.
7. Agents with memory: all can receive episodic prompt retrieval; relationship can query memory graph; recommendation can query context/bandit.
8. Shared state: `GiftSession.stage_log`, fixture/live context, experience store, bandit state.
9. Agents that can retry: structured base agents, creative; human-triggered regenerate is external.
10. Agents that can request another agent: none.
11. Agents that can stop execution: none; planner can write stop conditions but cannot enforce them.
12. Agents that can trigger another agent: none; delegate loop is service-controlled.
13. Outputs influencing routing: only completion status affects `next_stage`; content does not route.
14. Outputs verified: schemas for structured agents; creative image metadata/artifact offline; delivery local schema; semantic checks mainly offline.
15. Outputs persisted: stage log in memory during session; experience JSONL after feedback; generated artifacts on disk.

## 15. UI Permutation Evaluation Coverage

The UI permutation dataset is useful for coverage over input combinations, but it primarily evaluates pipeline output behavior. It does not fully evaluate harness behavior because the logs do not capture enough internal trajectory data.

| Harness Capability | Tested by UI permutations? | Direct / Indirect | Additional runtime data needed? |
|---|---|---|---|
| Reasoning | Partly | Indirect through output quality/rationale | raw prompts, model responses, rationale usage |
| Planning | Partly | Direct plan output; indirect final behavior | whether plan controlled execution |
| Dynamic routing | No | N/A | routing decisions, policy inputs/outputs |
| Memory | Partly | Indirect via output grounding | retrieved episodes, graph query calls/results |
| Tool use | Partly | Indirect through outputs/skills metadata | tool call trace, args, results, latency |
| Reflection | Partly | Creative only | critique scores per retry, prompt deltas |
| Verification | Partly | Direct offline metrics | live verifier decisions/gating |
| Reliability | Partly | benchmark timeouts/errors | retry attempts, fallback reason, recovery route |
| Recovery | Partly | indirect fallback output/rationale | original failure, chosen fallback, success/failure |
| Control | Weakly | stage order and timeouts | budget, stopping, model/tool selection trace |
| Governance | Partly | human actions, simulated delivery, constraints | permissions decisions, policy violations |
| Observability | Partly | reports/traces | complete live trace with inputs/prompts/tools/cost |

The 12,600 UI cases can exercise relationships, occasions, budgets, preferences, memories, and agency slider combinations. They do not by themselves prove agent harness behavior because the execution route is essentially fixed. To evaluate a harness, each case would need trajectory logging: selected agent, selected model, selected tools, raw `stage_config`, prompt version, prompt hash, tool calls/results, retry/fallback path, verifier decisions, latency, and cost.

## 16. Missing Capabilities

Most important missing capabilities:

1. Execution-authoritative planner/router.
2. Shared tool registry with per-agent permissions and call tracing.
3. Full live trace format that stores raw stage inputs, prompt hashes, model metadata, tool calls/results, retries, fallbacks, latency, and cost.
4. Runtime verifier gates for high-impact outputs.
5. Adaptive recovery policy driven by error type, confidence, verifier results, or budget.
6. Persistent resumable session checkpointing.
7. Context budget management and adaptive retrieval.
8. Harness-level cost/token accounting.

## 17. Recommended Priorities

Minimum changes required to become a stronger agent harness:

1. **P0: Add a HarnessController around `AgencyConsoleService._run_stage`.** It should own stage selection, planner consumption, routing decisions, stopping rules, and fallback policy.
2. **P0: Persist full `AgentInvocationTrace`.** Include stage input, upstream dependencies, prompt version/hash, model/provider, skills/tools declared/used, tool args/results, retries, latency, errors, verifier decisions.
3. **P0: Make `MultiAgentPlanningAgent` execution-authoritative in controlled mode.** Start with allowing the planner to skip/insert/reorder only among approved GMGI stages.
4. **P0: Add a tool registry/permission map.** Keep current tools, but register which agent can call what, with audit records.
5. **P1: Add live verifier gates.** Use existing eval logic first for schemas/constraints/provenance; later add semantic verifier/judge gates.
6. **P1: Add budgets.** Track max stages, max tool calls, max retries, max latency, estimated token/cost.
7. **P1: Make logged sessions replayable.** Store enough raw invocation data for counterfactual and self-consistency evals.

## 18. Risks / Architectural Concerns

- The word "planner" may overstate current runtime behavior because the plan is not executed by the orchestrator.
- `ExperienceStore` is useful episodic memory, but it currently omits raw invocation inputs needed for replay and failure analysis.
- The service finalizer backfills `skills_used` but not always `skills_declared` for raw/demo outputs, so skill observability may be inconsistent in demo paths.
- Fallback paths improve demo robustness but can mask real agent/tool failures unless strict flags are set.
- UI permutation evals can produce high output scores even if harness-level routing, tool selection, and recovery are not tested.
- Active sessions are in-memory and can be lost on backend restart.
- There is no token/cost visibility, which matters for claiming robust harness control.

## 19. Evidence / File References

| Area | File / Function |
|---|---|
| FastAPI routes | `src/api/app.py:create_app`, route functions around `/sessions`, `/stages`, `/feedback` |
| Stage order | `src/api/service.py:STAGES` |
| Runtime stage selection | `src/api/service.py:AgencyConsoleService._run_stage` |
| Next-stage logic | `src/api/service.py:AgencyConsoleService.next_stage` |
| Delegation loop | `src/api/service.py:AgencyConsoleService.delegate` |
| Feedback/bandit/experience update | `src/api/service.py:AgencyConsoleService.submit_feedback` |
| Session ledger | `src/agents/orchestrator.py:AgentOrchestrator`, `StageLogEntry`, `GiftSession` |
| Structured agent base | `src/agents/base.py:StructuredAgent.run` |
| LLM providers | `src/agents/llm.py:HTTPStructuredLLM`, `select_provider` |
| Skills metadata | `src/agents/skills.py` |
| Recipient profiling | `src/agents/recipient_profiling.py` |
| Relationship tool calling | `src/agents/relationship_analysis.py:_run_with_smolagents` |
| Intent reasoning | `src/agents/gift_intent_reasoning.py` |
| Planner | `src/agents/multi_agent_planning.py` |
| Recommendation tool calling/bandit hint | `src/agents/recommendation.py:_run_with_smolagents` |
| Creative generation/reflection | `src/agents/creative_generation.py` |
| Greeting generation | `src/agents/greeting_story.py` |
| Delivery planning | `src/agents/delivery_planner.py` |
| Memory graph | `src/memory_graph/graph.py`, `src/memory_graph/fixtures.py` |
| Episodic memory | `src/agents/experience_store.py`, `src/agents/experience_retriever.py` |
| Bandit | `src/rl/linucb_bandit.py` |
| Structural eval | `src/evals/structural.py` |
| Faithfulness eval | `src/evals/faithfulness.py` |
| Replay eval limitation | `src/evals/replay.py:evaluate_store` |
| Judge eval | `src/evals/judge.py` |
| Quality eval | `src/evals/quality.py` |
| Benchmark trace/eval | `src/evals/benchmark.py` |
| UI permutations | `src/evals/permutations.py`, `experiments/evals/UI_PERMUTATION_USE_CASES.md` |
| Frontend stage flow | `frontend/src/main.tsx` |
| Frontend API calls | `frontend/src/api.ts` |

## 20. Meta-Harness / Auto-Harness Audit

### 20.1 Definition

For this audit, a **meta-harness** is an outer-loop system that optimizes the harness itself, not just an agent output. The relevant reference point is the 2026 Meta-Harness paper, whose abstract defines a harness as the code controlling what information is stored, retrieved, and presented to the model, and describes Meta-Harness as an outer-loop system that searches over harness code using source code, scores, and execution traces from prior candidates.

An **adaptive auto-harness** is stronger still: it should sustain improvement over open-ended task streams through harness construction/adaptation, solve-time routing across harness variants, stateful evolution, and human steering. The Adaptive Auto-Harness abstract explicitly mentions optimizing prompts, skills, tools, memories, and supporting infrastructure from execution feedback, then adds harness trees, solve-time routing, and a stateful multi-agent evolver.

Critical distinction for GMGI:

```text
Evaluation output      != meta-harness
Bandit recommendation  != meta-harness
Prompt versioning      != meta-harness by itself
Planner output         != harness control unless it changes runtime execution
```

The current GMGI system has several adaptive mechanisms, but none currently performs end-to-end harness candidate generation, execution, trace collection, comparison, selection, and retention.

### 20.2 Existing Adaptive Mechanisms

Repository search found the following mechanisms that are adaptive or optimization-adjacent:

| Mechanism | Runtime Path | What It Consumes | What It Changes | Current Level |
|---|---|---|---|---|
| LinUCB feedback bandit | `AgencyConsoleService.submit_feedback` -> `LinUCBBandit.update/save`; `RecommendationAgent.bandit_feedback_hint` reads scores | rating, accept/edit/regenerate counts, optional CLIP score, context vector, selected action | recommendation policy state used as a hint to recommendation agent | Level 1 - Adaptive Agent |
| Episodic memory retrieval | `ExperienceStore.append` after feedback; `ExperienceRetriever.augment_system_prompt` during agent construction/execution | previous episodes with context fingerprint, human action, composite reward | system prompt context / few-shot guidance | Level 1 - Adaptive Agent |
| PromptOptimizerAgent | `PromptOptimizerAgent.run/maybe_run`, tested in `tests/test_prompt_optimizer.py` | recent episodes, human actions, acceptance rates, failed outputs | writes versioned system prompts under `prompt_versions` | Level 1.5 - Agent prompt optimization, not active meta-harness |
| Prompt version loading | `StructuredAgent._load_prompt_version`, `CreativeGenerationAgent._load_prompt_version`, `DeliveryPlannerAgent._load_prompt_version` | `latest.json` prompt version files | agent system prompt at instantiation | Level 1 - Adaptive Agent if optimizer/manual process writes versions |
| Creative CLIP critique loop | `CreativeGenerationAgent.run` -> `_critique_with_clip` -> `_refine_prompt` | generated image and prompt | prompt for retry within same creative stage | Level 1 - Adaptive Agent |
| Human regeneration/editing | UI/API `regenerate`, `edit`, `delegate` -> orchestrator | human action | stage output or rerun of same stage | Human-in-loop workflow control, not meta-harness |
| Offline evals and UI permutations | `src/evals/run.py`, `benchmark.py`, `permutations.py` | logs/cases/outputs | JSON reports, CSV rows, generated artifacts | Evaluation substrate only |

No mechanism currently generates a new harness candidate, executes it as an alternative harness, compares candidates, and promotes a better harness configuration or code path.

### 20.3 What Is Actually Being Optimized?

| Mechanism | Model Weights | Prompt / Instructions | Agent Policy | Recommendation Policy | Memory Contents | Memory Retrieval | Tool Selection | Agent Routing | Stage Ordering | Planner | Verifier | Retry Policy | Budget Policy | Model Selection | Harness Config | Harness Code | Entire Harness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LinUCB feedback bandit | No | No | Partly, as recommendation hint | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No |
| ExperienceStore + ExperienceRetriever | No | Prompt context only | Partly | Indirect | Yes, appends episodes | Fixed heuristic retrieval | No | No | No | No | No | No | No | No | No | No | No |
| PromptOptimizerAgent | No | Yes | Indirect | No | No | No | No | No | No | No | No | No | No | No | No, except prompt file location | No | No |
| Prompt version loading | No | Yes | Indirect | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Creative CLIP critique | No | Local retry prompt | Yes, creative only | No | No | No | No | No | No | No | Local image verifier proxy | Yes, creative only | No | No | No | No | No |
| UI permutations / evals | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |

Conclusion: the current optimization targets are **agent behavior**, **recommendation hints**, **prompt text**, and **retrieved context**. The system does **not** optimize stage order, routing policy, tool permissions, verifier policy, retry policy, budget policy, harness code, or an entire harness candidate.

### 20.4 Agent-Level vs Harness-Level Adaptation

The adaptive mechanisms classify as follows:

| Mechanism | Classification | Reason |
|---|---|---|
| LinUCB feedback | Agent-level learning | It updates a recommendation action policy and is exposed to RecommendationAgent as a tool hint. It does not change orchestration, routing, tools, prompts, planner, verifier, or stage execution. |
| ExperienceRetriever | Agent-level adaptation | It augments prompts with successful examples for similar contexts. It does not alter harness control. |
| PromptOptimizerAgent | Agent prompt optimization; not active harness optimization | It can rewrite system prompts from low acceptance evidence. It does not search over harness configs/code and is not currently invoked by `AgencyConsoleService`. |
| Creative CLIP critique | Agent-level self-refinement | It changes only the creative prompt inside one stage. |
| Human edit/regenerate/delegate | Human-in-loop workflow control | Human can alter output or continue execution, but the harness does not learn a new policy from the action in real time except via later feedback storage. |
| Eval benchmark/permutations | Observability substrate | Scores outputs and traces limited stage status/latency, but does not change runtime harness. |

Bandit audit answers:

1. What is it optimizing? A LinUCB policy over `BanditAction(recommendation_category, agency_bucket, style_archetype)`.
2. What state does it maintain? Per-arm matrices/vectors/counts in `LinUCBBandit`; persisted as JSON by `policy.save`.
3. What feedback does it consume? `reward_from_feedback(...)` computed from rating, accept/edit/regenerate counts, and optional `clip_score`.
4. Does it change the harness? No.
5. Does it change agent behavior? Partially, by providing a score hint to RecommendationAgent if a bandit state exists.
6. Does it change routing? No.
7. Does it change prompts? No.
8. Does it change tool selection? No.
9. Does it change stage execution? No.
10. Does it change planner behavior? No.
11. Does it change verifier behavior? No.
12. Does it persist learning across tasks? Yes, through `bandit_state_path`, if reused.
13. Does it persist learning across sessions? Yes.
14. Does it learn from execution trajectories? Weakly. It uses aggregate human action counts and reward, not full trajectories.
15. Can it compare alternative harness configurations? No.

Classification: **Agent-level learning**, not harness-level adaptation and not meta-harness optimization.

### 20.5 Meta-Harness Capability Matrix

| Capability | Status | Evidence | Gap |
|---|---|---|---|
| Harness candidate representation | PARTIAL | Env vars/config files/prompts can represent some choices. | No single `HarnessConfig` object covering stages, routing, tools, verifiers, budgets, recovery. |
| Harness selection | NONE | Runtime uses hardcoded `STAGES` and `_run_stage`. | No selector across candidate harnesses. |
| Harness configuration optimization | NONE | PromptOptimizer only writes prompts. | No optimizer over harness-level config. |
| Harness code optimization | NONE | No candidate code generation/execution loop. | No source-editing proposer, sandbox, candidate evaluation, or promotion. |
| Experience-driven improvement | PARTIAL | ExperienceStore, retriever, bandit, prompt optimizer. | Improvements affect prompts/recommendations, not harness policy. |
| Cross-session learning | PARTIAL | Bandit and experience store persist across sessions. | Learned signal is narrow and not trajectory-complete. |
| Cross-task learning | PARTIAL | Context fingerprint retrieval and shared bandit state can generalize weakly. | No task distribution model or harness selection by task. |
| Routing across harnesses | NONE | No harness variants. | Need harness tree/registry and solve-time routing. |
| Policy/budget at harness level | NONE | Local retry/timeouts only. | No global token/cost/tool/stage budget policy. |
| Automatic harness evolution | NONE | PromptOptimizer evolves prompts only. | No evolver modifying tools/memory/routing/verifier/stage policy. |
| Candidate trace collection | PARTIAL | Benchmark `agent_traces` has stage status/latency/errors. | Live traces lack raw inputs, prompts, tool calls/results, model metadata, retry/fallback details. |
| Candidate comparison | PARTIAL | Eval reports can compare output scores manually. | No automated A/B harness comparison on identical cases. |
| Regression control | PARTIAL | UI permutation benchmark can run many cases. | No candidate promotion gate or historical baseline tracking. |

### 20.6 12,600 UI Permutation Dataset as Optimization Substrate

The UI permutation dataset is a strong **input coverage substrate**. It spans relationship type, closeness, occasion, budget, preferences, memories, and agency slider. It can help evaluate output quality and some behavior across a broad case space.

Current capture ability:

| Required Optimization Signal | Currently Captured? | Notes |
|---|---|---|
| Input | Yes | Benchmark cases include custom profile fields. |
| Harness configuration | No | No unified harness config ID or variant snapshot. |
| Planner decision | Partly | Planner output is captured, but it does not control execution. |
| Routing decision | No | Routing is hardcoded; no routing decision object exists. |
| Agent invocation | Partly | Benchmark traces stage status/latency/confidence. |
| Agent input | Partly in benchmark, no in ExperienceStore | Benchmark keeps `input_context`; logged episodes omit raw per-stage `stage_config`. |
| Model output | Partly | Final structured output captured; raw model response omitted. |
| Tool calls | No | Tool args/calls/results are not traced. |
| Tool results | No | Only effects may appear in final output. |
| Verifier result | Partly offline | Eval metrics exist after execution; live verifier gates absent. |
| Retry/recovery | Partly | Rationale may mention repair; no structured retry trace. |
| Final output | Yes | Agent outputs and artifacts are recorded. |
| Quality metrics | Yes | `quality`, `benchmark`, `structural`, judge/faithfulness optional. |
| Cost | No | No token or monetary accounting. |
| Latency | Partly | Benchmark latency exists; live latency absent. |

Answer: **No, the current system does not have enough information to compare different harness configurations rigorously.** It can compare output quality for one current workflow over many cases, and it can manually compare reports from separately run experiments, but it lacks candidate identity, harness configuration snapshots, execution-authoritative routing decisions, full traces, cost, and candidate promotion/regression logic.

### 20.7 Missing Infrastructure for Meta-Harness

Minimum missing pieces:

1. `HarnessConfig` data model covering stage order, agent selection, model selection, tool permissions, memory policy, verifier policy, retry/fallback policy, budget, reflection, and human approval policy.
2. `HarnessCandidate` registry with immutable candidate IDs and versioned configs/code references.
3. Execution runner that can run candidate A and candidate B on identical UI cases.
4. Full `AgentInvocationTrace`:
   - case ID;
   - harness candidate ID;
   - stage;
   - raw `stage_config`;
   - upstream dependencies;
   - prompt version/hash;
   - provider/model;
   - tool calls/args/results;
   - verifier results;
   - retries/fallbacks;
   - latency;
   - token/cost estimates;
   - final output/artifacts.
5. Harness-level evaluator that scores not only final output, but routing, cost, latency, recovery, and governance.
6. Candidate comparison and promotion policy.
7. Regression suite over UI permutations with baseline tracking.
8. Optional evolver/proposer that can generate new harness configs safely before any code-generation loop.
9. Human review gate for accepting harness changes.

### 20.8 Meta-Harness Readiness Assessment

| Readiness Area | Current Status | Assessment |
|---|---|---|
| Candidate representation | PARTIAL | Some variables live in configs/env vars, but no unified harness config. |
| Execution | PARTIAL | Benchmark can run many cases, but not multiple harness candidates as first-class objects. |
| Evaluation | PARTIAL | Good output evals exist; harness-level evals are incomplete. |
| Trace collection | PARTIAL | Benchmark traces are useful but too shallow; ExperienceStore explicitly lacks raw `AgentInput`. |
| Comparison | PARTIAL | Manual report comparison possible; no automatic A/B harness candidate comparison. |
| Persistence | PARTIAL | Experiences, bandit, prompt versions, eval reports persist; harness candidates do not. |
| Optimization | NONE | No search/controller over harness configurations or code. |
| Regression control | PARTIAL | UI permutations can support it, but no promotion/regression gate exists. |

Final classification:

```text
Current Agent Architecture:
PARTIAL

Current Harness:
PARTIAL

Current Adaptive Harness:
PARTIAL

Current Meta-Harness:
NONE
```

More precise interpretation:

- Agent architecture is **PARTIAL** because real specialized agents and tools exist, but autonomy is bounded.
- Harness is **PARTIAL** because the runtime controls state, sequence, human actions, fallbacks, and some observability, but not adaptive execution.
- Adaptive harness is **PARTIAL** only in the weak sense that prompts/context/recommendation hints can adapt. Harness-level routing/tool/verifier/budget policy does not adapt.
- Meta-harness is **NONE** because there is no outer-loop harness candidate optimization.

Minimum architectural evolution:

| Transition | Required Components | Required Data / Traces | Required Evaluation | Architectural Changes | Research Value | Engineering Complexity |
|---|---|---|---|---|---|---|
| Current -> Dynamic Agent Harness | `HarnessController`, execution-authoritative planner/router, tool permission registry, live verifier gates | stage inputs, selected route, tool calls/results, verifier decisions, retries | route correctness, verifier pass rate, recovery success | move stage selection from `_run_stage` hardcode into controller policy | Shows GMGI as real adaptive agent harness | Medium |
| Dynamic Harness -> Adaptive Agent Harness | `HarnessConfig`, policy state, budget model, adaptive memory/retrieval policy, model/tool routing policy | outcomes by policy decision, reward, latency, cost, failures | policy regret, quality/cost tradeoff, robustness | make routing/tools/verifiers/budgets configurable and learnable | Demonstrates learning control layer over agents | Medium-high |
| Adaptive Harness -> Meta-Harness / Harness Optimization | candidate registry, candidate runner, optimizer/evolver, promotion gate, regression suite | full traces per candidate and case, candidate config/code version, scores/costs | A/B candidate comparison, held-out UI permutations, regression thresholds | add outer-loop system that proposes/evaluates/selects harness candidates | Strong research contribution aligned with Meta-Harness/Auto-Harness literature | High |

The lowest-risk path is to start with **config-only harness candidates**, not code-generating harness evolution. Once `HarnessConfig + AgentInvocationTrace + candidate benchmark comparison` exist, GMGI can support real harness optimization experiments without unsafe automatic code mutation.

## Phase 1 Implementation: HarnessConfig and AgentInvocationTrace

This phase makes the current harness measurable without changing the runtime harness itself. The authoritative execution path is still the existing fixed `STAGES` sequence and `AgencyConsoleService._run_stage(...)`; no dynamic routing, execution-authoritative planner, harness optimizer, or meta-harness was introduced.

### Architecture

New harness observability code lives in `src/harness/`:

- `HarnessConfig` describes the effective current harness as an immutable configuration object.
- `TraceRecorder` builds one `AgentRunTrace` per UI/eval run.
- `AgentInvocationTrace` records each actual agent call made by the existing workflow.
- `record_tool_call(...)` lets existing tool-using agents attach tool-call traces to the active invocation.

The API service owns one `TraceRecorder` per live console session. The eval benchmark owns one `TraceRecorder` per benchmark case. Both surfaces expose the same identity tuple:

```text
case_id + harness_id + harness_version + harness_config_hash + run_id
```

### HarnessConfig Schema

| Field | Type | Current Value | Runtime Effect | Future Purpose |
|---|---|---|---|---|
| `harness_id` | string | `gmgi_default` | Observational identity | Compare named harness candidates |
| `harness_version` | int | `1` | Observational identity | Version harness candidates |
| `description` | string | default fixed workflow description | Observational | Human-readable experiment metadata |
| `orchestration_mode` | literal | `fixed_stage` | Describes current behavior | Compare DAG/dynamic variants |
| `stage_execution_mode` | literal | `service_run_stage` | Describes current behavior | Compare controller-based execution later |
| `stage_order` | tuple | current eight stages | Documents current order | Compare stage-order variants later |
| `planner_mode` | literal | `advisory` | Describes current planner authority | Enable execution-authoritative planner experiments later |
| `routing_mode` | literal | `static` | Describes current routing | Enable dynamic routing experiments later |
| `memory_policy` | literal | `fixture_graph_plus_experience_retrieval` | Describes current memory use | Compare memory strategies |
| `context_policy` | literal | `stage_specific_static_context` | Describes current context assembly | Compare context packing policies |
| `tool_policy` | literal | `agent_local_tools` | Describes current tool ownership | Compare centralized tool registries later |
| `verification_policy` | literal | `schema_validation_with_offline_evals` | Describes current validation/eval split | Compare live verifier gates later |
| `retry_policy` | literal | `agent_local_retries` | Describes current retry behavior | Compare harness-level retry policies later |
| `fallback_policy` | literal | `agent_local_fallbacks` | Describes current fallback behavior | Compare explicit recovery policies later |
| `stopping_policy` | literal | `fixed_stage_completion_or_human_gate` | Describes current stop behavior | Compare adaptive stopping policies |
| `budget_policy` | literal | `local_retry_and_eval_timeout_only` | Describes current budget behavior | Add budget-aware control later |
| `model_policy` | literal | `provider_env_and_agent_config` | Describes current model selection | Compare model routing policies |
| `human_oversight_policy` | literal | `proposal_review_accept_edit_regenerate_delegate` | Describes UI control loop | Compare human-gating variants |

The stable configuration identity is deterministic: `HarnessConfig.config_hash` is computed from the canonical JSON configuration, and `HarnessConfig.identity` combines `harness_id`, `harness_version`, and the hash.

### AgentInvocationTrace Schema

Each invocation records:

- run/session/case/harness identity;
- stage name, agent name, agent version, invocation ID, sequence number, parent/dependency IDs;
- raw `AgentInput`, including the per-stage `session` and `stage_config`;
- context, relevant memory, and constraints extracted for easier eval consumption;
- provider/model metadata when available;
- tool calls with redacted arguments/results/errors;
- raw and structured agent outputs;
- validation status available at the current layer;
- routing/planner/verifier decision observations;
- retry/fallback observations when available;
- status, latency, and error details.

Secrets are redacted by key name for tokens, passwords, credentials, authorization headers, and related fields.

### Runtime Integration

The FastAPI service exposes trace data through:

```text
GET /sessions/{session_id}/trace
```

Normal session responses also include `run_id` and `harness` metadata so UI sessions can be tied back to a run trace. The service wraps the existing `agent.run({"session": ..., "stage_config": ...})` call; the call shape and stage order remain unchanged.

Tool-call tracing is additive inside the agents that already call tools:

- relationship memory graph query;
- recommendation memory graph query;
- recommendation bandit feedback hint;
- delivery date/logistics helper;
- creative diffusers generation;
- creative CLIP critique when available.

### Evaluation Integration

`src/evals/benchmark.py` now stores a run-level trace inside every case report as `run_trace`, plus top-level `run_id`, `harness_id`, `harness_version`, and `harness_config_hash`. Existing quality/eval semantics are unchanged; future evaluators can consume the trace without replaying the live app.

### Captured vs Intentionally Not Captured

Captured:

- raw per-stage `AgentInput`;
- actual invocation order;
- model/provider names where available from current config/env;
- tool arguments/results for explicitly wrapped local tools;
- output payloads;
- latency;
- errors and eval timeouts;
- static routing and advisory planner observations;
- explicit `not_available` verifier observation;
- harness/run/case identity.

Intentionally not captured yet:

- token usage and monetary cost, because current Ollama/local calls do not expose reliable accounting here;
- full provider request/response envelopes, to avoid changing agent invocation internals;
- execution-authoritative planner decisions, because the planner is not authoritative in the current architecture;
- live verifier pass/fail decisions, because live verifier gates do not exist yet;
- harness-level retries/fallbacks, because retries/fallbacks remain agent-local.

### Trace Completeness Audit

| Question | Answer | Notes |
|---|---|---|
| What UI case produced this run? | Yes | `case_id` is stored for benchmark and live fixture/custom sessions. |
| Which harness configuration was used? | Yes | Full `harness_config` plus stable hash. |
| Which agents ran? | Yes | `agent_name` per invocation. |
| In what order? | Yes | `sequence_number` and invocation list order. |
| What was the raw input to each agent? | Yes | `raw_agent_input` preserves `session` and `stage_config`. |
| What context did each agent receive? | Yes | Raw input plus extracted `context`. |
| What memory was injected? | Yes | Raw input plus extracted `relevant_memory`. |
| Which tools were called? | Partly | Wrapped local tool functions are traced. Framework-internal model/tool loops may not expose every intermediate step. |
| What arguments were passed? | Yes for wrapped tools | Secret-like fields are redacted. |
| What did tools return? | Yes for wrapped tools | Results are sanitized for JSON. |
| Which model was used? | Partly | Provider/model names are captured when available from agent config/env. |
| How long did each invocation take? | Yes | `latency_seconds`. |
| How many tokens were consumed? | No | Current local provider path does not expose reliable token accounting. |
| What was the estimated cost? | No | Local Ollama/diffusers paths do not have cost accounting. |
| Did an error occur? | Yes | Invocation status and error fields. |
| Was a retry performed? | Partly | Retry count can be recorded when exposed; internal framework retries may not be fully observable. |
| Was fallback used? | Partly | Field exists; only observable fallbacks can be recorded. |
| Was a verifier invoked? | Yes as absence | `verifier_decision=not_available` for current phase. |
| What was the final output? | Yes | `raw_agent_output` and `structured_output`. |
| Can we reconstruct the complete trajectory? | Mostly | Agent order/input/output/tool traces are available; provider-internal token/cost/intermediate loops remain unavailable. |

### Readiness

GMGI can now run the same UI case under two different future `HarnessConfig` objects and compare identity, invocation order, inputs, outputs, tool traces, latency, and evaluation results. What remains missing for full meta-harness optimization is an execution-authoritative harness/controller, explicit candidate registry, token/cost accounting, live verifier gates, and harness-level retry/fallback policies.

## Phase 2 Implementation: HarnessController

### 20.1 Controller Architecture

Phase 2 introduces `src/harness/controller.py` as the execution-authoritative layer for the current GMGI workflow. It reuses the Phase 1 `HarnessConfig`, `TraceRecorder`, `AgentRunTrace`, and `AgentInvocationTrace` types rather than creating a second tracing system.

Implemented controller components:

- `HarnessController`: owns runtime policy decisions.
- `NextAction`: explicit controller action representation for `agent`, `stop`, `retry`, and `fallback` outcomes.
- `FixedStagePolicy`: reproduces the existing eight-stage workflow by default.
- `ExecutionPlan` / `PlanStep`: structured representation for authoritative planner experiments.
- `ToolPolicy`: answers whether an agent may call a named tool.
- Controller schema gate: optional live schema verification using the existing agent output schemas.

Not implemented in this phase:

- dynamic router;
- semantic verifier;
- harness optimizer;
- automatic harness candidate search;
- code-generating meta-harness.

### 20.2 Execution Authority

Before Phase 2, runtime stage selection was effectively:

```text
STAGES -> next_stage(...) -> _run_stage(...)
```

After Phase 2, runtime stage selection is:

```text
HarnessController -> policy -> NextAction -> _run_stage(...)
```

`_run_stage(...)` still constructs the exact same agents and `AgentInput` payloads and still performs one stage execution. It no longer decides the workflow. The service now asks the controller for the next action, verifies that the requested UI stage matches the controller-selected stage, and then executes that action.

Default behavior remains the fixed-stage workflow:

```text
recipient_profiling
relationship_analysis
gift_intent_reasoning
multi_agent_planning
recommendation
creative_generation
greeting_story
delivery_planner
```

### 20.3 HarnessConfig Runtime Effects

| Configuration | Runtime Active? | Evidence |
|---|---|---|
| `orchestration_mode` | Yes | `fixed_stage` is required; unsupported values fail explicitly. |
| `planner_mode` | Yes | `advisory` uses fixed policy; `authoritative` requires and executes a validated `ExecutionPlan`. |
| `routing_mode` | Yes | `static` runs; `dynamic` is accepted as configured-but-unsupported and raises a controlled error. |
| `tool_policy` | Yes | `agent_local_tools` allows registered local tools; `deny_all_tools` denies before tool execution. |
| `verification_policy` | Yes | default observes existing validation/offline evals; `controller_schema_gate` validates outputs live. |
| `retry_policy` | Yes | default preserves agent-local retry behavior; `controller_retry_once` retries a failed controller action once. |
| `fallback_policy` | Yes | default preserves agent-local fallbacks; `skip_failed_stage` returns structured fallback output on action failure. |
| `stopping_policy` | Yes | default stops after final stage; `stop_before_delivery` produces a shorter trajectory. |

### 20.4 Planner Authority

The default planner mode remains `advisory`, so the Multi-Agent Planning stage can produce planning output but does not alter the default runtime trajectory.

The controller now supports `planner_mode="authoritative"` when supplied an `ExecutionPlan`. The plan is validated before execution. The controller rejects:

- unknown stages;
- duplicate stages;
- dependencies on missing stages;
- dependency cycles.

This is intentionally minimal. It proves the controller can execute a structural plan, but it does not yet ask an LLM planner to authoritatively produce that plan in the live UI.

### 20.5 Routing

Routing is now explicit and controller-owned. The implemented router is static fixed-stage routing through `FixedStagePolicy`.

`routing_mode="dynamic"` is intentionally not implemented. If configured, the controller raises `UnsupportedHarnessModeError` instead of silently pretending dynamic routing exists.

### 20.6 Tool Permissions

Tool permission enforcement is controller-owned through `ToolPolicy`.

Default allowed tools:

| Agent | Allowed Tools |
|---|---|
| `RelationshipAnalysisAgent` | `query_memory_graph` |
| `RecommendationAgent` | `query_memory_graph`, `bandit_feedback_hint` |
| `CreativeGenerationAgent` | `diffusers_image_generation`, `clip_critic` |
| `DeliveryPlannerAgent` | `date_logistics_math` |

Tool-using agents call `ensure_tool_allowed(...)` before doing the tool work, then record the tool call through the existing Phase 1 trace helper.

### 20.7 Verification

The default verification policy remains honest about the current system:

```text
schema_validation_with_offline_evals
```

This records that live semantic verifier gates are not active by default.

The minimal alternative verifier is:

```text
controller_schema_gate
```

It validates the agent payload against existing output schemas after stripping known observability metadata (`prompt_version`, `skills_used`, `skills_declared`). This is schema verification only, not semantic faithfulness verification.

### 20.8 Retry / Recovery

The default retry/fallback behavior remains agent-local:

```text
retry_policy = agent_local_retries
fallback_policy = agent_local_fallbacks
```

Controller-owned alternatives now exist:

- `controller_retry_once`: retries a failed action one additional time.
- `skip_failed_stage`: converts a failed action into structured fallback output.

These policies are intentionally small and deterministic.

### 20.9 Stopping

Stopping is now controller-owned.

Default:

```text
fixed_stage_completion_or_human_gate
```

Alternative:

```text
stop_before_delivery
```

The alternative stops after `greeting_story` and before `delivery_planner`, producing a real trajectory difference for harness comparison without changing agents or prompts.

### 20.10 Trace Integration

Every service and benchmark stage invocation still produces Phase 1 `AgentInvocationTrace` records. Controller decisions are written into existing trace fields:

- `routing_decision`: selected `NextAction`, policy, decision ID, reason;
- `planner_decision`: advisory vs authoritative mode;
- `verifier_decision`: not available, schema pass, or schema failure;
- `dependency_ids`: previous invocation dependency chain;
- tool calls: still captured through the existing `ToolCallTrace` list.

No hidden chain-of-thought is recorded. Only decision metadata is stored.

### 20.11 Alternative Harness Demonstration

Minimal comparison:

```text
Same UI case
├── Harness A: gmgi_default
│   └── fixed-stage route includes delivery_planner
└── Harness B: gmgi_candidate_a / stop_before_delivery
    └── fixed-stage route stops before delivery_planner
```

The benchmark test `test_benchmark_case_accepts_custom_harness_config` verifies that the same case can be evaluated under a custom `HarnessConfig`, receives a separate harness identity, and produces a different trajectory with no `delivery_planner` invocation.

### 20.12 Remaining Harness Gaps

Current classification after Phase 2:

```text
Current Agent Architecture: CONFIGURABLE MULTI-AGENT WORKFLOW
Current Harness:            PARTIAL EXECUTION-AUTHORITATIVE HARNESS
Current Adaptive Harness:   PARTIAL
Current Meta-Harness:       NONE
```

Before / after assessment:

| Category | Before Phase 2 | After Phase 2 |
|---|---|---|
| Cognition | Specialized agents, planner advisory | Same agents; planner can be structurally authoritative when a validated plan is supplied |
| Action | Service called stages directly | Controller emits `NextAction`, then service executes |
| State | Session/orchestrator state plus traces | Same state plus controller stop reason and policy decisions |
| Orchestration | Hardcoded fixed stage order | Controller-owned fixed-stage policy by default |
| Control | Mostly service/orchestrator | Controller owns routing, stopping, tool policy, verifier gate, retry/fallback policy |
| Reliability | Agent-local retries/fallbacks | Agent-local by default, optional controller retry/fallback |
| Governance | No tool permission gate | Explicit tool allow/deny policy |
| Observability | Phase 1 traces | Phase 1 traces plus controller decision metadata |

Direct answers:

1. Is the planner now execution-authoritative? **Partially.** The controller supports authoritative `ExecutionPlan` execution, but the live UI still defaults to advisory planning.
2. Is routing now controller-owned? **Yes**, for static routing; dynamic routing is explicitly unsupported.
3. Is tool permission enforcement controller-owned? **Yes**, for wrapped local tools.
4. Are verification gates controller-owned? **Partially.** Schema gate exists; semantic verifier does not.
5. Are retry/recovery policies controller-owned? **Partially.** Minimal controller retry/fallback policies exist; default remains agent-local.
6. Is stopping controller-owned? **Yes.**
7. Does `HarnessConfig` affect runtime? **Yes.**
8. Can two harness configurations produce different execution trajectories? **Yes.**
9. Can the same UI case be replayed under multiple harnesses? **Yes, through benchmark/controller configuration.**
10. Can traces be compared between harnesses? **Yes, with `case_id + harness_id + harness_version + run_id`.**

Meta-harness readiness: **Partially Ready**.

The system can now execute a case under a chosen harness config, collect traces, evaluate, and compare results. It still lacks a candidate registry, automatic candidate generation, optimizer/search, promotion policy, token/cost accounting, semantic verifier gates, and production-grade dynamic routing.

## Phase 3 Implementation: Dynamic Agent Harness

### 23.1 Runtime State

Phase 3 adds `HarnessRuntimeState` in `src/harness/controller.py`. The controller can now reason over completed stages, completed invocation ids, available agents/tools, prior outputs, failures, retry counts, verification results, constraints, memory context, and the triggering event. This state is passed from both the FastAPI service and the benchmark runner without changing agent invocation signatures.

### 23.2 Router Architecture

Routing is formalized behind a controller-owned `Router` abstraction:

```text
HarnessController
  -> HarnessRuntimeState
  -> StaticRouter or DynamicRouter
  -> NextAction
```

Agents still produce outputs. They do not mutate the workflow or bypass the harness.

### 23.3 Static vs Dynamic Routing

`StaticRouter` reproduces the existing fixed stage behavior for the default harness:

```text
orchestration_mode=fixed_stage
routing_mode=static
planner_mode=advisory
```

`DynamicRouter` is activated when either `orchestration_mode=dynamic` or `routing_mode=dynamic`. It selects from valid candidate actions using runtime state, explicit preconditions, verifier results, retry counts, and stopping policy.

### 23.4 Authoritative Planning

The controller can now adopt a real `ExecutionPlan` from the `multi_agent_planning` stage output through `adopt_execution_plan_from_output(...)`. In dynamic authoritative mode, execution can start with the default safe prefix, then the validated planner output can influence subsequent routing. Invalid plans are still rejected by the existing validator.

### 23.5 Dependency-Aware Execution

The dynamic router has explicit stage preconditions. For example, `recommendation` requires recipient profiling, relationship analysis, gift intent reasoning, and multi-agent planning unless the authoritative plan provides a valid reduced route. The router records rejected actions with `missing_preconditions` instead of silently following a positional stage list.

### 23.6 Semantic Verification

Phase 3 adds a deterministic `ConstraintVerifier` separate from the existing schema gate. It currently checks business/semantic constraints where they can be verified without an LLM:

- recommendation structure and budget fit;
- creative artifact path and minimum usable image resolution;
- delivery timing/logistics presence when an occasion date is known.

The existing `controller_schema_gate` behavior is preserved.

### 23.7 Dynamic Recovery

Verifier results are stored as `VerificationResult` values with `PASS`, `FAIL_RETRYABLE`, or `FAIL_NON_RETRYABLE`. The dynamic router can route a retry when a stage has a retryable verification failure and `retry_policy=controller_retry_once`; otherwise it can select fallback or stop according to configured policy.

### 23.8 Dynamic Stopping

Stopping is now available in dynamic routing for:

- completion of all routable stages;
- configured `stop_before_delivery`;
- verifier terminal failure or no routable candidates.

The selected stop action carries a structured reason.

### 23.9 Dynamic Decision Tracing

`NextAction.routing_decision()` now records:

- controller decision id;
- selected action;
- candidate actions;
- rejected actions;
- parent invocation id;
- triggering event;
- policy name/version;
- concise decision reason.

This keeps dynamic routing auditable without recording hidden reasoning.

### 23.10 Same-Case Harness Comparison

`src/evals/benchmark.py` now exposes `compare_harness_runs(...)`, which executes one benchmark case under two `HarnessConfig` objects and compares trajectory, final status, verifier outcomes, retries, failures, and quality score delta.

The Phase 3 controller tests demonstrate the critical property:

```text
same completed prefix
+ verifier PASS
-> next action = creative_generation

same completed prefix
+ recommendation budget FAIL_RETRYABLE
-> next action = retry recommendation
```

This is runtime-observation-driven behavior, not a predefined alternate stage sequence.

### 23.11 Remaining Harness Gaps

Current classification after Phase 3:

```text
Current Agent Architecture: DYNAMIC AGENT HARNESS
Current Meta-Harness:       NOT IMPLEMENTED
```

The implementation now satisfies the core dynamic harness tests: controller-owned execution, runtime-state-dependent routing, planner adoption, verification-influenced routing, failure/retry routing, dynamic stopping, and traceable decisions.

Remaining gaps:

- semantic verification is deterministic only; no separate LLM judge is wired into live control;
- dynamic tool selection is limited to permission enforcement around existing agent-local tools;
- no parallel execution;
- no token/cost accounting;
- no checkpoint/resume for active in-memory UI sessions;
- fallback repair is still simple;
- no harness candidate registry;
- no optimizer/search/meta-harness.

Meta-harness readiness: **Ready for Harness Comparison**.

The infrastructure can now run the same UI/benchmark case under multiple harness configs, collect traces, evaluate outputs, and compare behavior. It is not yet ready for automated meta-harness optimization because candidate generation, promotion criteria, cost accounting, and safe optimizer/search loops remain absent.

## Phase 4 Implementation: Harness Comparison Infrastructure

### 24.1 Harness Candidate Model

Phase 4 adds `HarnessCandidate` in `src/harness/candidates.py`.

The distinction is:

- `HarnessConfig`: runtime behavior knobs consumed by `HarnessController`.
- `HarnessCandidate`: experimental identity, metadata, compatibility notes, enabled status, and one concrete `HarnessConfig`.
- `HarnessExecutionResult`: one candidate executed against one case with trace, evaluation, timing, resource usage, randomness, and status.
- `AgentRunTrace`: low-level execution trajectory for one run.
- `EvaluationResult`: existing benchmark/eval metrics wrapped without replacing the evaluator.

### 24.2 Harness Registry

`HarnessRegistry` supports `register(...)`, `get(...)`, `list(...)`, and `exists(...)`. It validates candidates before registration and fails clearly for unsupported runtime modes.

Initial manually defined candidates:

| Candidate | Runtime Difference |
|---|---|
| `gmgi_default` | Fixed-stage, static routing, advisory planner, offline/default verification. |
| `gmgi_dynamic_v1` | Dynamic orchestration and dynamic routing. |
| `gmgi_dynamic_verified_v1` | Dynamic routing plus deterministic constraint verifier. |
| `gmgi_conservative_reliability_v1` | Dynamic verified harness plus controller retry-once and skip-stage fallback. |

### 24.3 Candidate Identity

Candidate identity is deterministic:

```text
candidate_id + version + HarnessConfig.config_hash -> candidate_hash
candidate_id:v{version}:{config_hash}:{candidate_hash}
```

Changing runtime configuration changes the identity. Creation timestamps do not affect identity.

### 24.4 Execution Result Model

`src/evals/harness_comparison.py` adds:

- `HarnessExperimentManifest`
- `HarnessExecutionResult`
- `run_case_across_harnesses(...)`
- `run_harness_experiment(...)`
- `compare_execution_results(...)`

Each execution result stores experiment id, case id, candidate id/identity, harness config, run id, candidate-attributed trace, evaluation summary, timing, resource usage, randomness settings, status, and failure categories.

### 24.5 Latency Accounting

Measured:

- run start/end timestamps from `AgentRunTrace`;
- total wall-clock latency from `TraceRecorder.finish(...)`;
- sum of invocation latencies;
- sum of recorded tool latencies.

Not separately measured yet:

- model-only latency;
- routing-decision latency;
- verifier-only latency.

Those are marked with explicit status values rather than estimated.

### 24.6 Token Accounting

The comparison layer aggregates exact token usage when `AgentInvocationTrace.token_usage` is populated. For current local/Ollama/diffusers paths, exact token usage is usually unavailable, so the report uses:

```text
token_usage_status = unavailable
```

No token estimates are fabricated.

### 24.7 Cost Accounting

The comparison layer aggregates `AgentInvocationTrace.estimated_cost` only when present. Current local providers normally do not expose reliable cost, so the report uses:

```text
cost_status = unknown
total_cost = null
```

No temporary model pricing is hardcoded.

### 24.8 Evaluation Normalization

Existing benchmark outputs are wrapped into a normalized evaluation summary containing available values for overall quality, stage reports, constraint preservation, creative quality, human behavior, and placeholders for unavailable metrics. The existing evaluator remains the source of truth.

### 24.9 Same-Case Comparison

`run_case_across_harnesses(...)` runs the exact same `BenchmarkCase` object against each supplied `HarnessCandidate`. The only intended independent variable is the candidate.

The comparison table preserves individual dimensions:

```text
quality, constraint, reliability, latency, tokens, cost,
invocations, tool calls, retries, fallbacks, verifications, failures
```

No single harness score is produced.

### 24.10 State Isolation

Default comparison uses isolated state:

- each candidate run receives a fresh benchmark session;
- output directories are separated by case and candidate;
- candidate traces/results are persisted independently;
- no experience store or bandit state is intentionally shared by the comparison runner.

Model/provider caches may still exist outside the experiment process, so the manifest records stochasticity and optional randomized order. The current framework documents this rather than claiming perfect determinism.

### 24.11 Experiment Manifest

`eval/configs/harness_comparison_phase4.json` provides a reproducible manifest:

```json
{
  "experiment_id": "phase4_initial_harness_comparison",
  "candidates": ["gmgi_default", "gmgi_dynamic_v1", "gmgi_dynamic_verified_v1"],
  "seed": 2026,
  "state_isolation": "isolated",
  "evaluation_version": "gmgi_phase_4_harness_comparison_v1"
}
```

The CLI accepts `--manifest` and writes `manifest.json`, per-case comparison JSON, per-candidate execution results, aggregate JSON, and CSV rows.

### 24.12 Initial Harness Comparison

Command:

```text
python -m src.evals.harness_comparison --manifest eval/configs/harness_comparison_phase4.json --output-dir experiments/evals/phase4_initial
```

In the current local environment, a live run with real model-backed agent paths timed out after 180 seconds. The comparison infrastructure and tests are implemented; producing the table requires a ready Ollama/local model runtime or tighter stage timeouts for the chosen cases.

### 24.13 Pareto Analysis

The comparison report includes Pareto-dominated candidates using measured quality/reliability as maximize dimensions and latency as the minimize dimension. Unknown cost/tokens are not used in Pareto dominance.

### 24.14 Remaining Gaps

Current classification remains:

```text
Dynamic Agent Harness
```

Meta-harness status:

```text
Ready for Meta-Harness
```

Reason: the infrastructure now supports:

```text
candidate -> execute -> trace -> evaluate -> compare
```

without changing code for every manually supplied candidate. It is still not a meta-harness because candidate generation, automatic selection, optimization/search, promotion gates, and learning loops are intentionally absent.

Remaining gaps:

- exact token/cost accounting depends on provider support;
- model-only/routing/verifier latency is not separately timed;
- no automatic candidate generation or optimizer;
- no online harness adaptation;
- no candidate promotion policy;
- no perfect isolation from external model caches/rate limits;
- no large 5-20 case live comparison table yet because local model-backed execution timed out in this environment.

## Phase 5 Implementation: Empirical Harness Benchmark

### 25.1 Timeout / Execution Reliability

Phase 5 adds explicit empirical-run reliability handling around the Phase 4 comparison runner:

- per-stage timeout remains in `src/evals/benchmark.py`;
- per-candidate execution timeout is added in `src/evals/harness_comparison.py`;
- final execution status is standardized as `SUCCESS`, `PARTIAL_SUCCESS`, `HARNESS_FAILURE`, `INFRASTRUCTURE_FAILURE`, `EVALUATION_FAILURE`, `TIMEOUT`, or `CANCELLED`;
- partial traces are preserved and persisted even when a candidate times out or errors;
- `AgentRunTrace` now records operational events such as `run_started`, `agent_started`, `agent_completed`, `run_timeout`, and `run_failed`.

Root cause of the Phase 4 180-second timeout:

The runner was not stuck in dynamic routing or trace serialization. It multiplied model-backed per-stage waits across candidates. In a bounded real run, the slow components were model-backed stages:

- `RecipientProfilingAgent`
- `RelationshipAnalysisAgent`
- `RecommendationAgent`
- `GreetingStoryAgent`

Deterministic stages completed quickly. Therefore the timeout is currently an infrastructure/provider latency issue, not evidence that one harness architecture is low quality.

### 25.2 Failure Taxonomy

Failures are classified into:

- harness failures: routing, plan, dependency, verification, retry, fallback, constraint;
- infrastructure failures: provider/model/network/filesystem/tool infrastructure/timeouts;
- evaluation failures: evaluator exceptions, malformed evaluator input, judge unavailable.

Timeouts are explicitly reported as infrastructure failures unless a controller execution limit is the cause.

### 25.3 Candidate Isolation

Candidate executions are isolated by default:

```text
executions/{case_id}/{candidate_id}/rep-{n}/execution_result.json
```

Each candidate/repetition receives a separate benchmark run, output directory, run id, trace, and result file. The runner supports resume by reusing valid completed execution result files instead of rerunning them.

Known shared/external state:

- provider/model cache may be shared by the local runtime;
- external Ollama/provider availability is shared;
- benchmark comparison does not intentionally share bandit/experience/session state.

### 25.4 Reproducibility

Every execution records:

- experiment id;
- case id;
- candidate id and deterministic candidate identity;
- harness config hash;
- run id;
- evaluation version;
- seed/repetition;
- model/provider metadata when available in invocation traces;
- execution mode and state isolation mode through the manifest;
- timestamps and operational events.

### 25.5 Stochasticity

Current stochastic components:

- LLM generation unless temperature/provider configuration forces determinism;
- image generation when creative runs are enabled;
- external provider latency/availability;
- bandit behavior if shared state is intentionally enabled.

Local fixture tools and deterministic eval aggregations are stable given the same captured outputs.

### 25.6 Evaluation Stability

Readiness classification:

| Metric | Readiness |
|---|---|
| schema conformance | READY |
| constraint satisfaction | READY |
| DAG validity | READY |
| provenance | PARTIAL |
| faithfulness | PARTIAL |
| self-consistency | PARTIAL |
| purpose alignment | PARTIAL |
| agent-level quality | READY |
| overall system quality | READY |
| creative quality | PARTIAL |
| human behavior | NOT READY |
| latency | READY |
| tokens | PARTIAL |
| cost | PARTIAL |
| reliability | READY |

Human behavior metrics are not fabricated from UI permutations; they remain unavailable without real accept/edit/regenerate/delegate data.

### 25.7 Benchmark Stratification

The UI permutation dataset audit found:

- cases: 12,600;
- relationships: 7;
- occasions: 6;
- budgets: 4;
- formality values: 4;
- agency slider values: 3;
- duplicate cases: 0;
- invalid missing-recipient cases: 0;
- creative/style cases: 12,600;
- external delivery data required: 0.

The full dataset should not be used until small real model-backed comparisons complete reliably. Smoke/check subsets are preferred first.

### 25.8 Real Model-Backed Results

Bounded command:

```text
python -m src.evals.harness_comparison --limit 1 --candidates gmgi_default gmgi_dynamic_v1 --experiment-id phase5-real-bounded --output-dir experiments/evals/phase5_real_bounded --stage-timeout 2 --execution-timeout 12 --repetitions 1 --no-resume
```

Observed rows:

| Case | Harness | Rep | Status | Failure Type | Quality | Reliability | Latency | Tokens | Cost |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| close_partner_birthday_memory_art | gmgi_default | 1 | TIMEOUT | infrastructure | n/a | 0.6 | 8.397s | unavailable | unknown |
| close_partner_birthday_memory_art | gmgi_dynamic_v1 | 1 | TIMEOUT | infrastructure | n/a | 0.6 | 8.230s | unavailable | unknown |

These are real bounded executions, but they did not produce successful quality-comparable model outputs because multiple model-backed stages exceeded the intentionally short per-stage timeout.

### 25.9 Harness Comparison Results

Aggregate bounded result:

| Candidate | N | Success | Timeout | Mean Quality | Mean Latency | Tokens | Cost | Retries |
|---|---:|---:|---:|---:|---:|---|---|---:|
| gmgi_default | 1 | 0 | 1 | n/a | 8.397s | unavailable | unknown | 0 |
| gmgi_dynamic_v1 | 1 | 0 | 1 | n/a | 8.230s | unavailable | unknown | 0 |

This is not enough to claim a better harness. It only proves the runner can preserve and classify bounded failures independently.

### 25.10 Trajectory Analysis

Both bounded traces reached seven invocation records and preserved the timed-out stages. Example timeout stages:

```text
recipient_profiling -> timeout
relationship_analysis -> timeout
gift_intent_reasoning -> success
multi_agent_planning -> success
recommendation -> timeout
greeting_story -> timeout
delivery_planner -> success
```

This indicates provider/model-stage latency rather than a dynamic routing loop.

### 25.11 Pareto Analysis

Pareto analysis is implemented over quality, reliability, and latency, but the bounded run has no successful quality values. Therefore no statistically meaningful Pareto conclusion should be drawn from this run.

### 25.12 Evaluation Readiness

Harness comparison readiness: **PARTIALLY READY**.

The infrastructure now supports same-case, multi-candidate, repeated, resumable, partially failing runs with failure attribution. However, real model-backed benchmark success is not yet reliable in this environment.

Meta-harness readiness: **PARTIALLY READY**.

The code can perform candidate -> execute -> trace -> evaluate -> compare, but automatic meta-harness optimization should wait until real executions complete reliably and evaluation metrics are stable across repetitions.

### 25.13 Remaining Gaps

- Need a real model runtime profile where core LLM stages complete within practical timeouts.
- Need successful replicated runs before interpreting quality differences.
- Need broader 5-case x 2-candidate x 3-repetition benchmark after provider reliability improves.
- Need exact token/cost capture from providers where possible.
- Need human behavior data from actual UI sessions.
- Need creative/image quality benchmarks only when image generation is enabled and completes.

## 26. Phase 5.5: Real Model Execution Stabilization

### 26.1 Root Cause

The bounded harness failures are caused by slow real model-backed agent execution, not by the UI tunnel or by evaluator scoring. The cheap Ollama provider health check reached `/api/tags` successfully, but the single-case smoke run still timed out. This means the provider process was reachable while one or more model generations exceeded the harness stage timeout.

Observed smoke health check:

| Check | Result |
|---|---|
| Provider | Ollama |
| Host | `http://127.0.0.1:11434` |
| Health endpoint | `/api/tags` |
| Status | 200 |
| Health latency | 0.085s |
| Single-case smoke | TIMEOUT |
| Failure type | infrastructure |
| Recommendation | fix provider/model execution before large comparisons |

### 26.2 Timeout Hierarchy

The current timeout hierarchy is:

| Layer | Owner | Current behavior |
|---|---|---|
| Provider/API request timeout | agent client libraries | `GMGI_OLLAMA_TIMEOUT_SECONDS` now controls the base structured LLM path, instructor/OpenAI-compatible paths, smolagents LiteLLM paths, and the Ollama Python client where those clients expose a timeout. Default remains backward-compatible at 120s for the base structured path. |
| Agent/stage timeout | `src/evals/benchmark.py:_run_stage` | A worker thread is joined for `stage_timeout_seconds`. If it is still alive, the stage is recorded as timeout and benchmark execution proceeds. |
| Tool timeout | local agent tools | No separate tool timeout. Local tools are synchronous and traced. Observed date/logistics math is sub-millisecond. |
| Controller limits | `src/harness/controller.py` | Limits steps, agent invocations, retries, and fallback behavior. These are execution-count policies, not wall-clock cancellation. |
| Candidate execution timeout | `src/evals/harness_comparison.py:_run_case_with_timeout` | Joins the whole candidate run worker for `execution_timeout_seconds`, then records experiment timeout if still alive. |
| Experiment timeout | CLI/shell/runtime | No global multi-case wall-clock budget beyond per-candidate timeout and external process timeout. |

### 26.3 Cancellation Lifecycle

Cancellation is not currently hard cancellation. The benchmark stage timeout uses a daemon worker thread and `join(timeout=...)`. If a provider request blocks past the stage timeout, the harness records an `agent_timeout`, but the underlying request can continue until its own client timeout or completion.

This was observed directly in traces:

| Stage | Harness timeout | Later provider outcome |
|---|---:|---|
| recipient_profiling | 15.0s | later success at 22.7s |
| relationship_analysis | 15.0s | later success at 36.1s |
| recommendation | 75.0s | later error at 120.0s before timeout alignment |

Trace timeout events now explicitly report `request_cancelled=false` and `cancellation_supported=false` so benchmark output does not imply that the provider request was terminated.

### 26.4 Stage Latency Distribution

Small-sample real execution observations:

| Run | recipient_profiling | relationship_analysis | gift_intent_reasoning | multi_agent_planning | recommendation | greeting_story | delivery_planner |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15s diagnostic | late success 22.7s | late success 36.1s | 0.002s | 0.002s | timeout 15.0s | timeout 15.0s | 0.002s |
| 45s reliable-profile attempt | success 23.7s | success 34.1s | 0.003s | 0.002s | timeout 45.0s | timeout 45.0s | 0.002s |
| 75s aligned diagnostic | success 26.2s | success 30.5s | 0.002s | 0.002s | timeout 75.0s | timeout 75.0s | 0.004s |
| calibrated one-case baseline | success 29.8s | success 29.3s | 0.002s | 0.004s | success 117.2s | success 38.2s | 0.002s |

The deterministic intent, planning, and delivery stages are fast. The slow boundary is real LLM/tool-agent generation, especially recommendation and greeting generation in this environment.

### 26.5 Prompt and Context Diagnostics

Agent invocation traces now include lightweight input diagnostics:

- input character count;
- approximate input token count;
- context character count;
- approximate context token count;
- memory item count;
- memory character count;
- constraint character count.

These are captured under each invocation's `context.input_diagnostics`. They are intentionally approximate and dependency-free.

### 26.6 Concurrency Findings

The harness comparison runner executes candidates sequentially. However, stage timeouts can create accidental overlap because the timed-out provider worker is not cancelled. After a timeout, the next stage may begin while the previous provider call is still running in the background. This can starve local Ollama on CPU/limited GPU runtimes and make later stages slower.

The safe interpretation is that intended benchmark concurrency is 1, but effective provider concurrency can exceed 1 after a timeout until cancellable provider execution is implemented.

### 26.7 Benchmark Eligibility

Quality comparison is now marked unavailable for infrastructure timeouts and evaluation failures. A row is quality-comparison eligible only when:

- the candidate status is `SUCCESS`; or
- the status is `HARNESS_FAILURE` and a final output is explicitly available.

Timeout rows keep reliability/latency/failure metadata, but their quality value is `null` so incomplete executions are not treated as low-quality successful outputs.

### 26.8 Reliable Profile

`eval/configs/harness_comparison_phase55_reliable.json` is a conservative single-case profile for validating the default harness before comparing candidates. It disables creative generation and uses longer wall-clock budgets than the intentionally short bounded smoke run.

The calibrated successful local command used:

```text
GMGI_LLM_PROVIDER=ollama
GMGI_OLLAMA_MODEL=llama3.2:latest
GMGI_OLLAMA_TIMEOUT_SECONDS=160
GMGI_OLLAMA_NUM_PREDICT=384
GMGI_RECOMMENDATION_MAX_STEPS=2
GMGI_GREETING_NUM_PREDICT=160
python -m src.evals.harness_comparison --diagnostic --candidate gmgi_default --case-id close_partner_birthday_memory_art --stage-timeout 180 --execution-timeout 500 --no-resume
```

The in-process baseline completed with `final_status=success` and `termination_reason=completed`. This validates the default harness for one case when the selected Ollama model is installed and output length is large enough to avoid truncated JSON.

The profile should still be considered a stabilization profile, not proof of a better harness. Candidate comparison should start only after the default smoke/baseline succeeds in the target runtime.

### 26.9 Readiness

Current readiness: **READY for one-case default baseline validation; NOT READY for harness optimization**.

The infrastructure can now classify real failures and has produced one successful default real-model baseline. The next gate is a successful same-case run for each candidate before interpreting quality deltas or attempting automatic harness optimization.

## Phase 6 Implementation: Empirical Harness Benchmark

### 26.1 Experimental Design

Phase 6 uses human-selected `HarnessCandidate` values as the independent variable. The benchmark keeps UI case, model/provider, evaluation version, creative setting, state isolation, seed, and timeout profile fixed across candidates.

No meta-harness, automatic candidate generation, automatic harness selection, or optimization loop is implemented.

### 26.2 Frozen Variables

The Phase 6 manifest is `eval/configs/harness_benchmark_phase6.json`.

Frozen defaults:

| Variable | Value |
|---|---|
| provider | ollama |
| model | llama3.2:latest |
| creative generation | false |
| state isolation | isolated |
| repetitions | 1 |
| stage timeout | 180s |
| execution timeout | 500s |
| evaluation version | gmgi_phase_5_harness_comparison_v1 |

The CLI applies manifest model settings and refuses mismatched environment values.

### 26.3 Harness Candidates

| Candidate | Orchestration | Routing | Planner | Verification | Retry | Fallback |
|---|---|---|---|---|---|---|
| gmgi_default | fixed_stage | static | advisory | schema_validation_with_offline_evals | agent_local_retries | agent_local_fallbacks |
| gmgi_dynamic_v1 | dynamic | dynamic | advisory | schema_validation_with_offline_evals | agent_local_retries | agent_local_fallbacks |
| gmgi_dynamic_verified_v1 | dynamic | dynamic | advisory | deterministic_constraints | agent_local_retries | agent_local_fallbacks |

Candidate difference auditing now reports config-hash differences and runtime-policy differences separately.

### 26.4 Case Selection

Five UI permutation cases are selected from `experiments/evals/ui_permutation_cases.json`:

| Case ID | Relationship | Occasion | Budget | Formality | Agency |
|---|---|---|---|---|---|
| ui_perm_001_partner_birthday_c5_0_a0_15 | partner | Birthday | Flexible | casual | 0.15 |
| ui_perm_2477_sibling_promotion_c4_0_a0_5 | sibling | Promotion | USD 25-45 | professional | 0.5 |
| ui_perm_4653_friend_graduation_c3_5_a0_85 | friend | Graduation | USD 60-100 | semi-formal | 0.85 |
| ui_perm_8926_colleague_anniversary_c2_0_a0_15 | colleague | Anniversary | USD 150-250 | ceremonial | 0.15 |
| ui_perm_5701_parent-child_housewarming_c4_0_a0_15 | parent-child | Housewarming | Flexible | casual | 0.15 |

This is a small stratified subset, not full coverage of all 12,600 permutations.

### 26.5 Replication Strategy

The manifest starts at 1 repetition. The intended scale-up is:

1. baseline reproduction;
2. one same-case, three-harness comparison;
3. five cases x one repetition;
4. five cases x three harnesses x three repetitions only after stability is demonstrated.

The local run did not satisfy the stability gate, so the 5-case expansion was not executed.

### 26.6 Quality Comparison

Non-creative runs now mark creative quality as unavailable when `include_creative=false`; missing creative artifacts are not scored as zero. Quality comparison remains unavailable for timeout/infrastructure failures.

### 26.7 Reliability Comparison

Reliability analysis separates success, timeout, infrastructure failure, harness failure, retry, and fallback rates. Infrastructure/model failures are not interpreted as lower harness quality.

### 26.8 Efficiency Comparison

The runner captures wall-clock latency, invocation count, tool calls, routing decisions, verification count, retries, and fallbacks. Token/cost values remain unavailable for the local Ollama path unless the provider exposes them.

### 26.9 Trajectory Comparison

Trajectory analysis extracts actual invocation sequences from traces and reports whether trajectories changed. In the first compact same-case run, stage sequence did not change; outcome divergence occurred inside stages.

### 26.10 Failure Analysis

First compact same-case observations on `ui_perm_001_partner_birthday_c5_0_a0_15`:

| Harness | Status | Latency | Failed Stage |
|---|---|---:|---|
| gmgi_default | partial | 283.179s | recommendation |
| gmgi_dynamic_v1 | success | 246.596s | none |
| gmgi_dynamic_verified_v1 | partial | 186.920s | relationship_analysis |

This is descriptive evidence only. The default baseline did not reproduce on the first UI permutation case in this environment.

### 26.11 Pareto Analysis

Pareto analysis now excludes rows where quality, reliability, or latency is unavailable. Missing values are not coerced to zero. No meaningful Pareto winner is reported for the compact same-case run.

### 26.12 Creative Benchmark

Creative/image benchmarking remains separate. It was not run in Phase 6 because the non-creative benchmark is not stable enough yet.

### 26.13 Observed Harness Trade-offs

Observed so far:

- `gmgi_dynamic_v1` completed the compact same-case run.
- `gmgi_default` and `gmgi_dynamic_verified_v1` had model/stage failures in that run.
- Dynamic routing did not produce a different stage trajectory in the compact run.
- Verification did not help when a stage failed before usable output existed.

No universal winner is declared.

### 26.14 Limitations

- Only one compact same-case comparison was run.
- Full persisted traces were avoided locally because disk space was tight.
- Token and cost metrics are unavailable.
- The 5-case benchmark is prepared but not executed.
- Results are not statistically significant.

### 26.15 Meta-Harness Prerequisites

Meta-harness readiness: **PARTIALLY READY**.

The system has candidate definitions, same-case execution, isolated state, descriptive measurements, trajectory analysis, and report generation. It is not ready for automatic harness selection until same-case and 5-case comparisons complete reproducibly with full persisted traces.

## Phase 6.5 Implementation: Controlled Harness Replication

### 26.16 Experimental Replication Design

Phase 6.5 freezes one diagnostic UI case, provider/model settings, timeout profile, state isolation, creative setting, schemas, prompts, and evaluation version. The independent variable remains `HarnessCandidate`.

Diagnostic case:

```text
ui_perm_001_partner_birthday_c5_0_a0_15
```

Frozen runtime:

```text
provider = ollama
model = llama3.2:latest
creative = false
GMGI_OLLAMA_TIMEOUT_SECONDS = 160
GMGI_OLLAMA_NUM_PREDICT = 384
GMGI_OLLAMA_NUM_CTX = 4096
GMGI_RECOMMENDATION_MAX_STEPS = 2
GMGI_GREETING_NUM_PREDICT = 160
```

The replication manifest is `eval/configs/harness_replication_phase65.json`.

### 26.17 Within-Harness Variance

Default harness replications:

| Rep | Status | Latency | Quality |
|---:|---|---:|---:|
| 1 | success | 218.582s | 0.848 |
| 2 | success | 204.359s | 0.840 |
| 3 | success | 181.353s | 0.841 |

Dynamic harness replications:

| Rep | Status | Latency | Quality |
|---:|---|---:|---:|
| 1 | success | 247.302s | 0.808 |
| 2 | success | 225.731s | 0.845 |
| 3 | partial | 29.436s | N/A |

### 26.18 Stage-Level Reliability

`gmgi_default` was stable in this N=3 sample: all non-creative stages succeeded in all three runs.

`gmgi_dynamic_v1` succeeded in 2/3 runs. Repetition 3 failed early in recipient profiling and later model-backed stages, while deterministic stages still completed.

### 26.19 Model/Runtime Variance

Runtime/model variance remains present. The same dynamic harness and same input produced two completed runs and one partial run. Since the dynamic trajectory did not differ structurally, the failure is attributed to model/runtime variance rather than a harness routing decision.

### 26.20 Harness-Induced Divergence

Observed harness-induced divergence count: `0`.

No repeated run showed a dynamic-router-selected alternate stage path such as retry, skip, or verifier-triggered rerouting.

### 26.21 Trajectory Reproducibility

All successful and partial runs followed the same stage sequence:

```text
recipient -> relationship -> intent -> planning -> recommendation -> greeting -> delivery
```

The outcome changed, but the trajectory did not.

### 26.22 Quality Reproducibility

Quality is computed only for completed executions:

| Harness | Quality N | Mean Quality | Median Quality |
|---|---:|---:|---:|
| gmgi_default | 3 | 0.843 | 0.841 |
| gmgi_dynamic_v1 | 2 | 0.827 | 0.827 |

The partial dynamic run is not treated as quality zero.

### 26.23 Reliability Comparison

| Harness | N | Success Rate | Timeout Rate | Failure Rate |
|---|---:|---:|---:|---:|
| gmgi_default | 3 | 1.000 | 0.000 | 0.000 |
| gmgi_dynamic_v1 | 3 | 0.667 | 0.000 | 0.333 |

### 26.24 Efficiency Comparison

Successful executions only:

| Harness | Successful N | Mean Latency | Median Latency | Mean Invocations |
|---|---:|---:|---:|---:|
| gmgi_default | 3 | 201.431s | 204.359s | 7.000 |
| gmgi_dynamic_v1 | 2 | 236.517s | 236.517s | 7.000 |

Token and cost metrics remain unavailable/unknown for the local Ollama path.

### 26.25 Decision Gate

Decision: **PRELIMINARY COMPARISON**.

Default is sufficiently reproducible for this one diagnostic case at N=3. Dynamic comparison remains preliminary because one of three dynamic runs failed due to model/runtime variance and no harness-induced trajectory divergence was observed.

### 26.26 Remaining Confounders

- Local Ollama generation remains slow and stochastic.
- No token/cost accounting is available.
- Replication used one UI case only.
- Dynamic-verified was not replicated in Phase 6.5.
- Creative/image generation remains excluded.

Meta-harness readiness after Phase 6.5: **READY FOR SYSTEMATIC HARNESS BENCHMARK**, not ready for Meta-Harness.
