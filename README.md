# GMGI — Gift Intelligence

**Gift Intelligence: Negotiated Agency in Multi-Agent, GAN-Grounded Human–AI Gift Co-Creation** is a research prototype for the NeurIPS 2026 Creative AI Track. It explores a central question: when AI helps plan and create a gift, does the result express the giver's agency, the AI's agency, or a negotiated hybrid?

GMGI treats gifting as a staged, revocable delegation process. A memory graph grounds recipient and relationship context; a sequential multi-agent pipeline proposes profiles, recommendations, visual concepts, messages, and a simulated delivery plan; and a conditional **MemoryGAN** creates personalized visual artifacts. At every stage, the human can accept, edit, regenerate, or delegate. Those choices become an **Agency Ledger** that makes authorship and control visible.

The generative core includes an agency slider that interpolates between human-directed and AI-inferred style representations. A scoped contextual bandit learns default conditioning preferences without modifying GAN weights. Evaluation combines FID, KID, CLIPScore, and LPIPS-based measures with a small human study comparing AI-autonomous, human-only, and negotiated-hybrid conditions.

This is deliberately a research-sized system: it is not an e-commerce platform, does not implement real delivery or payments, does not attempt full RLHF, and does not claim state-of-the-art image fidelity. Its contribution is the agency-instrumented co-creation pipeline, its auditable controls, and the evaluation protocol around perceived authorship and control.

## Repository layout

- `specs/` — source-of-truth project specifications
- `src/memory_graph/` — relationship and memory graph service
- `src/agents/` — blackboard-based sequential agents and orchestrator
- `src/gan/` — conditional GAN training and inference
- `src/rl/` — contextual-bandit feedback layer
- `src/api/` — FastAPI backend
- `demo/` — React and Vite Agency Console
- `experiments/` — run configurations, logs, and data documentation
- `eval/` — generative metrics and human-study materials
- `paper/` — paper drafts, template, and figures
- `data/` — local datasets and small fixtures
- `tests/` — automated tests

Implementation has not begun; this revision establishes the spec-defined scaffold only.
