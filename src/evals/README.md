# GMGI Reference-Free Evals

This directory contains additive observability code. It reads existing logged
sessions and structured outputs; it does not change agent prompts, schemas,
tool definitions, invocation, or orchestrator flow.

## Phase 1: Structural / Deterministic Metrics

Implemented in `src.evals.structural`.

Metrics:

- `schema_conformance`: validates each logged stage output against that stage's existing config schema.
- `constraint_satisfaction`: checks literal constraints visible in logged outputs, such as budget-fit fields, exclusions when supplied by an external input context, simulated delivery constraints, and structured risk flags.
- `dag_validity`: validates the logged `multi_agent_planning` sequence and dependencies for duplicate stages, unresolvable dependencies, cycles, and known stage-order violations.
- `provenance_traceability`: checks whether recommendation evidence, greeting memory references, and intent preferences can be traced to upstream logged outputs or optional supplied input context.

Run:

```bash
python -m src.evals.run --phase 1 --store experiments/experience_store.jsonl
```

Optional JSON output:

```bash
python -m src.evals.run --phase 1 --store experiments/experience_store.jsonl --output experiments/evals/phase1.json
```

Limit to recent episodes:

```bash
python -m src.evals.run --phase 1 --limit 10
```

## Current Logging Boundary

`ExperienceStore` currently contains final `agent_outputs`, `human_actions`,
reward, prompt versions, and context fingerprint. It does not store every
agent's raw input context or tool-call transcript. Because this phase is not
allowed to modify live logging, metrics that require those fields are computed
only when the caller supplies `input_context` directly to the pure functions.

This means Phase 1 fully covers schema conformance and DAG validity from the
existing store, partially covers constraint satisfaction and provenance from
logged outputs, and leaves richer tool-call provenance for a later phase if
non-invasive logging is explicitly approved.

## Phase 2: Faithfulness / Groundedness

Implemented in `src.evals.faithfulness`.

Default mode is deterministic and dependency-free:

- Decompose structured outputs into simple atomic claims.
- Treat upstream logged outputs and optional supplied input context as the reference.
- Mark a claim supported when enough claim tokens are present in the context.

Run:

```bash
python -m src.evals.run --phase 2 --store experiments/experience_store.jsonl --limit 10
```

The module also exposes `evaluate_faithfulness(input_context, output, verifier=...)`
for an optional structured NLI/LLM verifier. The CLI does not call an LLM for
Phase 2 by default.

## Phase 3: Self-Consistency / Counterfactual Sensitivity

Implemented in `src.evals.replay`.

Reusable harness functions:

- `run_self_consistency(agent_factory, session, stage_config, runs=3)`
- `run_counterfactual(agent_factory, session, stage_config, perturbation_path=..., replacement=...)`
- `diff_outputs(before, after)`
- `perturb_mapping(value, path, replacement)`

Run against the current store:

```bash
python -m src.evals.run --phase 3 --store experiments/experience_store.jsonl
```

Current store-only output is intentionally marked `insufficient_logged_context`
because `ExperienceStore` does not store raw per-stage `AgentInput` payloads.
The harness is ready for cloned/replayed inputs supplied by an eval caller, but
the live pipeline is untouched.

## Phase 4: Purpose-Alignment Judge

Implemented in `src.evals.judge`.

The judge is isolated from the agents under evaluation. It sees only:

1. The stage contract.
2. The session input/upstream context available to that stage.
3. The stage output.

No gold answer is supplied.

Safe default:

```bash
python -m src.evals.run --phase 4 --store experiments/experience_store.jsonl
```

This reports that judge calls are disabled.

To enable judge calls with the existing structured LLM provider:

```bash
python -m src.evals.run --phase 4 --enable-judge --judge-provider openai --judge-model gpt-4o-mini
```

You can also use environment variables:

```bash
GMGI_EVAL_JUDGE_PROVIDER=openai GMGI_EVAL_JUDGE_MODEL=gpt-4o-mini python -m src.evals.run --phase 4 --enable-judge
```

## Run All Safe Phases

```bash
python -m src.evals.run --phase all --store experiments/experience_store.jsonl --limit 10
```

Phase 4 remains disabled inside `all` unless `--enable-judge` is provided.

## AI-System Quality Evals

Implemented in `src.evals.quality` and `src.evals.benchmark`.

These are the practical engineering evals for the GMGI system, beyond smoke
tests. They score each stage and the overall system on task behavior:

- Recipient Profiling: preference extraction recall, confidence validity, no unsupported sensitive traits.
- Relationship Analysis: closeness bucket match, tone/formality guidance, risk-flag structure, social-boundary respect.
- Gift Intent Reasoning: occasion match, preference recall, budget/delivery constraint preservation, gift-goal specificity.
- Multi-Agent Planning: full agent coverage, correct order, executable dependencies, fallback and human-review visibility.
- Recommendation: ranked list validity, evidence grounding, preference coverage, budget awareness, artifact-type diversity.
- Creative Generation: artifact file existence, non-placeholder path, image header validity, practical resolution, slider validity.
- Greeting/Story: message quality heuristics, tone presence, memory-reference grounding, no obvious copied lyrics/quotes.
- Delivery Planner: simulated-only behavior, planned date before/on occasion, no real logistics claims.
- Cross-component: schema conformance, DAG validity, pipeline coverage, no error outputs, intent-to-recommendation consistency.

Run quality metrics over real logged sessions:

```bash
python -m src.evals.run --phase quality --store experiments/experience_store.jsonl --limit 20 --output experiments/evals/logged_quality.json
```

Run the curated benchmark against the real agent classes:

```bash
python -m src.evals.run --phase benchmark --limit 3 --output-dir experiments/evals/benchmark
```

Use a per-stage timeout in Colab to prevent slow model calls from looking like
a notebook hang:

```bash
python -m src.evals.run --phase benchmark --limit 3 --stage-timeout 45 --output-dir experiments/evals/benchmark
```

Run benchmark including the Creative Generation agent:

```bash
python -m src.evals.run --phase benchmark --include-creative --stage-timeout 120 --limit 3 --output-dir experiments/evals/benchmark_creative
```

The benchmark does not substitute demo outputs. If Ollama, a model, Diffusers,
or a checkpoint is unavailable, that stage records an error and receives a low
score. This is intentional: eval results should expose system readiness, not
hide it.

Default benchmark cases live in:

```text
src/evals/benchmark_cases.json
```

You can pass a custom benchmark set:

```bash
python -m src.evals.run --phase benchmark --case-file path/to/cases.json
```

## UI-Permutation Dataset

Implemented in `src.evals.permutations`.

This creates a deterministic benchmark dataset from the same inputs allowed in
the Agency Console setup screen:

- `giver_name`
- `recipient_name`
- `relationship_type`
- `closeness_score`
- `occasion_name`
- `occasion_date`
- `budget_hint`
- `formality`
- `preferences`
- `memories`
- `agency_slider`

Generate the case file only:

```bash
python -m src.evals.run --phase ui-permutations --max-cases 48 --output experiments/evals/ui_permutation_cases.json
```

Generate and evaluate a limited sample:

```bash
python -m src.evals.run --phase ui-permutations --run-permutations --max-cases 48 --limit 8 --stage-timeout 45 --output experiments/evals/ui_permutation_cases.json --output-dir experiments/evals/ui_permutation_benchmark
```

Run the generated case file directly through the benchmark:

```bash
python -m src.evals.run --phase benchmark --case-file experiments/evals/ui_permutation_cases.json --limit 8 --stage-timeout 45 --output-dir experiments/evals/ui_permutation_benchmark
```
