# 08 — Demo UI Spec: "The Agency Console"

## Purpose
A single-page interactive demo, suitable both for **live use in the human study** (spec 06) and for **poster/large-screen presentation** at NeurIPS. This is a first-class deliverable, not an afterthought — Creative AI Track reviewers/attendees interact with it directly.

## Core screens

### 1. Session Setup
- Pick or create a synthetic persona (giver + recipient) from the fixture set (`02_memory_graph_spec.md` § test fixtures), occasion, budget hint.
- Optional: upload/select a few demo memory snippets (photo thumbnails, short text notes) — pre-seeded for demo/study, not live personal data collection at a poster.

### 2. Agency Console (main screen)
For each pipeline stage (spec 04), a card shows:
- Stage name + short label of which sub-system produced it ("Relationship Analysis · LLM agent").
- The proposed output (profile text / recommendation list / generated image / generated message).
- The agent's one-line `rationale`.
- Four buttons: **Accept**, **Edit**, **Regenerate**, **Delegate rest to AI**.
- A visible **Agency Slider** (0–1) specifically on the Creative Generation stage, with the live regenerated image updating as the slider moves (this is the single most important interactive element for the Agency theme — make it feel tactile and immediate).

### 3. Agency Ledger (summary view)
- After a session completes, show a compact timeline: which stages were human-authored vs. AI-authored vs. hybrid, rendered as a simple colored bar (e.g., green=accept, amber=edit, blue=regenerate, purple=delegate).
- This view doubles as the artifact shown on the poster and as the per-session log exported for the study (spec 06).

### 4. Feedback screen
- Post-session Likert questions (spec 06 measures) + the open-ended authorship question.
- Submits to the bandit's reward signal (spec 05).

## Interaction principles
- **Always show the "why"** — every AI proposal ships with its one-line rationale; hiding this would undercut the whole Agency argument.
- **Never auto-advance without a visible action** in study mode (auto-advance only allowed after explicit "Delegate rest to AI").
- **Fast enough to feel live**: GAN inference must return in a few seconds max on the demo hardware; if slower, pre-cache a few conditioning vectors for the live poster demo and clearly label pre-cached vs. live-generated runs to avoid misleading the audience.

## Implementation
- React + Vite single-page app, calling the FastAPI backend (`01_architecture_spec.md`).
- No backend calls needed for pure UI state (slider dragging before "commit") — debounce slider movement, only call `generate()` when the user pauses or hits "regenerate," to keep the GPU from being hammered during a live demo.
- Keep visual design simple, warm, personal (this is a gifting product) — see `frontend-design` conventions if building with Claude Code's frontend skill; avoid a generic dashboard look.

## Poster/large-screen adaptation
- A read-only "replay mode": step through a pre-recorded session's Agency Ledger + generated artifacts, for unattended large-screen looping when no presenter is at the booth.
- Export a single static hero image: the interpolation strip (spec 03) + Agency Ledger bar, for the printed poster itself.
