# 00 — Project Overview & Venue Framing

## Project
**Gift Intelligence: Negotiated Agency in Multi-Agent, GAN-Grounded Human–AI Gift Co-Creation**
(working title — finalize once results exist; see `07_paper_writing_spec.md` for title guidance)

Internal codename: **GMGI** (Generative Multi-Agent Gift Intelligence).

## Target venue
- **NeurIPS 2026 Creative AI Track**, 4th year, theme: **Agency**.
- Non-archival: accepted work is posted on OpenReview and presented as an onsite poster / large-screen display, but is **not** part of the main proceedings. This lowers the bar for "novelty vs. a top-tier archival paper" and raises the bar for **demonstrability and clarity of the human/AI agency argument**.
- Papers: 2–6 pages **excluding references**, official template, at least one author must register and attend.
- Prior years (2025, theme "Humanity") accepted a mix of research papers *and* artworks/interactive systems — scent-from-memory devices, live human/AI musical duets, robotic assembly from language, etc. Lesson: **a working, demoable artifact matters as much as the write-up.** Build the demo first, write second.

## Why "Agency" fits gifting
Gift-giving is inherently an agency-laden ritual: the giver expresses intent, the recipient interprets it, and the object itself carries delegated meaning. Inserting an AI system into that loop forces an explicit question the track wants answered:

> **When an AI plans, designs, and generates a gift, whose creative agency does the artifact express — the giver's, the AI's, or a negotiated hybrid? And can we make that negotiation visible and controllable rather than hidden?**

This reframes GMGI from "a gift recommender" (RecSys framing, not distinctive enough for Creative AI) into **an agency-instrumented co-creation system**: every stage of the pipeline exposes a control surface where the human can accept, edit, override, or delegate a decision back to the AI, and we log + visualize that negotiation as part of the contribution itself (see `08_demo_ui_spec.md`, "Agency Ledger" and "Agency Slider").

## Core research contributions (paper-sized, not product-sized)
1. **A pipeline formulation of gifting as staged, revocable agency-delegation** (Recipient Understanding → Relationship Modeling → Memory Graph → Intent Reasoning → Multi-Agent Planning → Generation → Feedback), instrumented so each stage records *who* (human/agent) authored the final decision.
2. **A conditional GAN generative core** ("MemoryGAN") that turns a relationship-and-memory embedding into a personalized visual gift artifact (illustration / greeting-card art / motif for physical gift wrap), with an explicit *style-agency* control knob at inference time (see `03_gan_model_spec.md`).
3. **An "Agency Ledger" evaluation protocol**: a small human study measuring perceived authorship/control/satisfaction across 3 conditions (AI-autonomous, human-only, negotiated hybrid), reported alongside standard generative-quality metrics (FID/KID/CLIPScore).
4. A working interactive demo suitable for a poster/large-screen presentation.

## Explicit non-goals (MVP scope control)
- **Not** building a production e-commerce gifting product, delivery logistics, or payment flow — `Delivery Planner` agent is a stub/simulated agent only.
- **Not** training a full RLHF pipeline — reward-model / RL is scoped down to a contextual bandit over a small discrete style/agency action space (see `05_rl_feedback_spec.md`).
- **Not** training a large text-to-image diffusion model from scratch — the GAN is deliberately small (StyleGAN2-ADA / Lightweight-GAN class of architecture) so it can be trained on a laptop-class GPU or a single cloud GPU within the project timeline, in keeping with the "we will use GAN architecture" constraint and Creative-AI-track compute norms.
- **Not** claiming SOTA generative fidelity — the contribution is the **agency-instrumented pipeline + evaluation protocol**, with the GAN as one grounded, reproducible generative component.

## Success criteria
- [ ] End-to-end demo: text/photo memory input → generated gift concept (recommendation + generated illustration + generated message) in under ~10s per stage, runnable live.
- [ ] GAN trained to a stable FID on a chosen small dataset (target documented in `03_gan_model_spec.md`).
- [ ] Small user study (n ≥ 12–20) comparing the three agency conditions, with at least one statistically-discussable trend (t-test/Wilcoxon, effect size reported honestly even if not significant — track values honesty over hype).
- [ ] 2–6 page paper drafted against the official template, with required "role of AI/ML" and "how the theme is addressed" sections (`07_paper_writing_spec.md`).
- [ ] Poster + live/video demo asset ready for onsite/large-screen presentation.

## Repo map (see `09_engineering_conventions.md` for details)
```
gmgi/
├── specs/                # this folder — source of truth, read before coding
├── src/
│   ├── memory_graph/
│   ├── agents/
│   ├── gan/
│   ├── rl/
│   └── api/
├── demo/                 # frontend for the interactive poster demo
├── experiments/          # training runs, configs, logs, checkpoints (gitignored weights)
├── eval/                 # metrics scripts + user study materials
├── paper/                # NeurIPS Creative AI template + drafts
└── data/                 # raw/processed (gitignored large files)
```

## How to use this spec folder
This is a **spec-driven vibe coding** project: every file in `specs/` is treated as the contract. Before writing or changing code, read the relevant spec file(s). If an implementation needs to diverge from a spec, **update the spec in the same change**, don't let code and spec drift. See `prompts_for_claude_code.md` for the exact sequence of prompts to drive this with Claude Code.
