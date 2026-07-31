# 01 — System Architecture Spec

## High-level pipeline
```
                    ┌─────────────────────────┐
 (photos, chat       │   Ingestion Layer        │
  logs, dates,        │  (text/photo/date        │
  free-text notes) →  │   normalizers)           │
                    └────────────┬─────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │   Memory Graph Service   │  (spec 02)
                    │  Person/Event/Photo/     │
                    │  Preference/Emotion nodes│
                    └────────────┬─────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │  Agent Orchestrator      │  (spec 04)
                    │  (blackboard + router)   │
                    └───┬───┬───┬───┬───┬─────┘
       ┌─────────────┐  │   │   │   │   │  ┌───────────────┐
       │ Recipient    │←─┘   │   │   │   └─→│ Delivery       │
       │ Profiling    │      │   │   │      │ Planner (stub)│
       └─────────────┘      │   │   │      └───────────────┘
       ┌─────────────┐      │   │   │
       │ Relationship │←─────┘   │   │
       │ Analysis     │          │   │
       └─────────────┘          │   │
       ┌─────────────┐          │   │
       │ Recommend.   │←─────────┘   │
       │ Agent        │              │
       └─────────────┘              │
       ┌─────────────┐              │
       │ Creative Gen │←─────────────┘
       │ Agent (GAN)  │  (spec 03)
       └──────┬──────┘
              ↓
       ┌─────────────┐
       │ Greeting/    │  (LLM, e.g. Claude via API)
       │ Story Agent  │
       └──────┬──────┘
              ↓
       ┌─────────────────────────┐
       │  Human Review / Agency   │  ← human accepts/edits/
       │  Console (demo UI)       │    overrides/delegates
       └────────────┬────────────┘
                     ↓
       ┌─────────────────────────┐
       │  Feedback → RL/Bandit    │  (spec 05)
       │  updates agent/GAN prefs │
       └─────────────────────────┘
```

## Components

### 1. Ingestion Layer
- Accepts free-text notes, a small set of uploaded photos, key dates, and optional short chat excerpts (all synthetic/consented demo data — no scraping real private data, see `09_engineering_conventions.md` § data ethics).
- Normalizes into: `Person`, `Event`, `Photo(embedding)`, `TextSnippet(embedding)`.
- Implementation: Python module, CLIP for photo embeddings, a text embedding model for snippets.

### 2. Memory Graph Service
- Owns the knowledge graph (see `02_memory_graph_spec.md`).
- Exposes a small internal API: `add_node`, `add_edge`, `query_subgraph(person_id)`, `embed_context(person_id, occasion)`.
- Storage: `networkx` + JSON snapshot for demo scale (a handful of people, tens of memories). Document a Neo4j upgrade path but do not build it for the MVP — YAGNI.

### 3. Agent Orchestrator
- A lightweight blackboard pattern: each agent reads from and writes to a shared `GiftSession` state object, not to each other directly.
- Agents are implemented as functions/classes with a strict IO contract (see `04_multi_agent_spec.md`); reasoning-heavy agents (Recipient Profiling, Relationship Analysis, Recommendation, Greeting/Story) call an LLM (Claude) with structured-output prompts; the Creative Generation agent calls the local GAN; Delivery Planner is a rule-based stub.
- Orchestration is **sequential with revision loops**, not free-form multi-agent chat — this keeps the system debuggable and keeps the "agency ledger" well-defined (one decision per stage, attributable to human or agent).

### 4. Creative Generation Agent (GAN core)
- Wraps the trained conditional GAN (`03_gan_model_spec.md`) behind a simple `generate(conditioning_vector, agency_slider) -> image`.
- `agency_slider ∈ [0,1]`: 0 = maximally constrained by human-specified style tags (low AI creative agency), 1 = maximally sampled from the GAN prior conditioned only on the memory embedding (high AI creative agency). Implemented as an interpolation weight between a human style-embedding and the memory-graph embedding fed into the generator's mapping network.

### 5. Greeting/Story Agent
- LLM call (Claude, via Anthropic API) conditioned on the memory graph subgraph + relationship summary + occasion, producing a short message/story/caption to accompany the generated visual.
- Copyright discipline: never reproduce real copyrighted song lyrics/poems; all generated text must be original.

### 6. Human Review / Agency Console
- The demo UI surface (spec `08_demo_ui_spec.md`) where each stage's proposed decision is shown with **Accept / Edit / Regenerate / Delegate-to-AI** controls. Every action is timestamped and stored as an `AgencyEvent`.

### 7. Feedback → RL/Bandit
- Consumes `AgencyEvent`s and a post-hoc satisfaction rating to update a small contextual bandit over (style-agency-slider bucket, recommendation category) — see `05_rl_feedback_spec.md`.

## Tech stack (MVP)
| Layer | Choice | Reason |
|---|---|---|
| Backend API | Python, FastAPI | fast to scaffold, async-friendly |
| Graph | networkx + JSON | zero infra for demo scale |
| Embeddings | CLIP (image), a small sentence-embedding model (text) | off-the-shelf, no training needed |
| Reasoning agents | Claude API (function-calling / structured output) | strong instruction following, matches "Claude Code" workflow |
| Generative core | PyTorch, GAN (spec 03) | explicit project constraint |
| RL | contextual bandit (e.g., `vowpalwabbit` or a ~50-line custom LinUCB) | scoped down from full RL, still a real learning signal |
| Frontend demo | React + Vite, single-page "Agency Console" | matches Artifact-style rapid iteration |
| Experiment tracking | simple CSV/JSON logs + matplotlib; Weights & Biases optional | avoid infra overhead for a 2–6 page paper |

## Data flow contracts
All inter-stage payloads are plain JSON with a versioned schema documented inline in `04_multi_agent_spec.md`. No stage may silently mutate another stage's fields — new fields are appended, originals preserved, so the Agency Ledger can reconstruct full provenance.

## Non-functional requirements
- Runs on a single consumer GPU (≥8GB VRAM) for GAN training and inference; CPU-only inference acceptable if slow.
- Deterministic seeds for all stochastic components so demo runs are reproducible for reviewers.
- No hard dependency on paid third-party gifting APIs — Delivery Planner is simulated.
