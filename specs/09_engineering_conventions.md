# 09 — Engineering Conventions & Spec-Driven Workflow

## The core rule
**Specs are the source of truth.** Before implementing or changing any module, read the relevant `specs/*.md` file(s). If reality forces a divergence (a library doesn't support something, a hyperparameter needs changing, a design choice turns out wrong), **update the spec in the same commit/PR as the code change** — never let them drift silently. A spec update with no matching code change, or code with no matching spec update, should read as incomplete work.

## Repo structure (authoritative)
```
gmgi/
├── specs/                      # this folder
├── src/
│   ├── memory_graph/            # spec 02
│   ├── agents/                  # spec 04
│   │   ├── recipient_profiling.py
│   │   ├── relationship_analysis.py
│   │   ├── recommendation.py
│   │   ├── creative_generation.py
│   │   ├── greeting_story.py
│   │   ├── delivery_planner.py
│   │   └── orchestrator.py
│   ├── gan/                     # spec 03
│   │   ├── models.py            # generator/discriminator defs
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── configs/
│   ├── rl/                      # spec 05
│   │   └── linucb_bandit.py
│   └── api/                     # FastAPI app, spec 01
├── demo/                        # spec 08, React+Vite
├── experiments/                 # configs, logs, DATA_CARD.md, checkpoints (gitignored)
├── eval/                        # spec 06 — metrics + study materials
│   ├── gan_metrics.py
│   ├── bandit_offline_eval.py
│   └── study_materials/
├── paper/                       # spec 07 — template + drafts + figures
├── data/                        # gitignored raw/processed data + fixtures
├── tests/
├── requirements.txt / pyproject.toml
└── README.md
```

## Definition of Done (per feature/module)
1. Spec file read and, if needed, updated.
2. Code implements the documented interface exactly (function signatures in specs are contracts, e.g. `MemoryGAN.generate(...)` in spec 03).
3. At least a minimal test (unit test for pure logic, smoke test for GAN/LLM calls using a fixture persona from spec 02).
4. Any new hyperparameter/config value actually used is written back into the spec's tables (specs must reflect the *real* run configuration, not just a hypothetical starting point).
5. If it touches the demo UI, the Agency Console principles in spec 08 are respected (rationale shown, actions logged).

## Coding standards
- Python 3.11+, type hints everywhere, `TypedDict`/`pydantic` for the JSON contracts defined in specs 04/05.
- One agent = one file, one class, matching the IO contract in spec 04 exactly — no agent calls another agent directly; everything goes through the orchestrator + `GiftSession` blackboard.
- GAN training/inference code isolated in `src/gan/`, no orchestrator imports inside it (keep it usable standalone for the eval scripts).
- No secrets in code; API keys via environment variables only.
- Commit messages: `[spec-02] add context_embedding pooling`, `[gan] tune r1_gamma to 0.8 per experiments/run-014`, etc. — tag which spec a change relates to.

## Data & compute discipline
- All datasets/checkpoints gitignored; only configs, code, and small fixture JSONs are committed.
- `experiments/DATA_CARD.md` documents dataset source + license for anything used to train the GAN (spec 03) — mandatory before any training run, not an afterthought before submission.
- Every training run gets a config file + a log with final FID/KID — no "I just remember it was good" results.

## Testing priorities (given limited timeline)
1. Memory graph construction + `context_embedding` (deterministic, easy to unit test).
2. GAN `infer.py` interface + agency-slider interpolation (shape/determinism tests; visual QA by eye, not automated).
3. Orchestrator stage sequencing + `stage_log` provenance integrity (an automated test that no stage overwrites a previous `output`).
4. Bandit update math (a toy synthetic-reward test that LinUCB converges to the better arm).

## When in doubt
Prefer the smaller, more honestly-scoped implementation over a more impressive-sounding one that won't be finished and evaluated in time — the paper's credibility rests on what was actually built and measured (see `00_project_overview.md` non-goals).
