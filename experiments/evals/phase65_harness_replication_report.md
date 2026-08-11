# Phase 6.5 Harness Replication Report

## Experimental Configuration

| Field | Value |
|---|---|
| experiment_id | phase65_controlled_harness_replication_v1 |
| case_id | ui_perm_001_partner_birthday_c5_0_a0_15 |
| provider | ollama |
| model | llama3.2:latest |
| creative generation | disabled |
| state isolation | isolated |
| repetitions | 3 per harness |
| execution order | controlled replication; default gate first, then dynamic |
| stage timeout | 180s |
| execution timeout | 500s |

Frozen model parameters:

| Parameter | Value |
|---|---|
| GMGI_OLLAMA_TIMEOUT_SECONDS | 160 |
| GMGI_OLLAMA_NUM_PREDICT | 384 |
| GMGI_OLLAMA_NUM_CTX | 4096 |
| GMGI_RECOMMENDATION_MAX_STEPS | 2 |
| GMGI_GREETING_NUM_PREDICT | 160 |

## Default Harness Replications

| Case | Repetition | Harness | Status | Failed Stage | Latency | Quality |
|---|---:|---|---|---|---:|---:|
| ui_perm_001_partner_birthday_c5_0_a0_15 | 1 | gmgi_default | SUCCESS | N/A | 218.582s | 0.848 |
| ui_perm_001_partner_birthday_c5_0_a0_15 | 2 | gmgi_default | SUCCESS | N/A | 204.359s | 0.840 |
| ui_perm_001_partner_birthday_c5_0_a0_15 | 3 | gmgi_default | SUCCESS | N/A | 181.353s | 0.841 |

Default classification: **HIGHLY STABLE for this small N=3 diagnostic case**.

Evidence: all 3 runs completed, all required non-creative stages succeeded, and the stage trajectory was identical.

## Dynamic Harness Replications

| Case | Repetition | Harness | Status | Failed Stage | Latency | Quality |
|---|---:|---|---|---|---:|---:|
| ui_perm_001_partner_birthday_c5_0_a0_15 | 1 | gmgi_dynamic_v1 | SUCCESS | N/A | 247.302s | 0.808 |
| ui_perm_001_partner_birthday_c5_0_a0_15 | 2 | gmgi_dynamic_v1 | SUCCESS | N/A | 225.731s | 0.845 |
| ui_perm_001_partner_birthday_c5_0_a0_15 | 3 | gmgi_dynamic_v1 | PARTIAL | recipient_profiling | 29.436s | N/A |

Dynamic classification: **UNSTABLE in this N=3 diagnostic sample**.

Evidence: 2/3 runs completed. The third run failed early across multiple model-backed stages.

## Stage Reliability

| Harness | Stage | Success Rate | Timeout Rate | Failure Rate |
|---|---|---:|---:|---:|
| gmgi_default | recipient_profiling | 1.000 | 0.000 | 0.000 |
| gmgi_default | relationship_analysis | 1.000 | 0.000 | 0.000 |
| gmgi_default | gift_intent_reasoning | 1.000 | 0.000 | 0.000 |
| gmgi_default | multi_agent_planning | 1.000 | 0.000 | 0.000 |
| gmgi_default | recommendation | 1.000 | 0.000 | 0.000 |
| gmgi_default | greeting_story | 1.000 | 0.000 | 0.000 |
| gmgi_default | delivery_planner | 1.000 | 0.000 | 0.000 |
| gmgi_dynamic_v1 | recipient_profiling | 0.667 | 0.000 | 0.333 |
| gmgi_dynamic_v1 | relationship_analysis | 0.667 | 0.000 | 0.333 |
| gmgi_dynamic_v1 | gift_intent_reasoning | 1.000 | 0.000 | 0.000 |
| gmgi_dynamic_v1 | multi_agent_planning | 1.000 | 0.000 | 0.000 |
| gmgi_dynamic_v1 | recommendation | 0.667 | 0.000 | 0.333 |
| gmgi_dynamic_v1 | greeting_story | 0.667 | 0.000 | 0.333 |
| gmgi_dynamic_v1 | delivery_planner | 1.000 | 0.000 | 0.000 |

## Stage Latency

Successful invocation latencies only:

| Harness | Stage | Successful N | Failed N | Min | Median | Mean | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| gmgi_default | recipient_profiling | 3 | 0 | 25.5s | 31.4s | 31.8s | 38.5s |
| gmgi_default | relationship_analysis | 3 | 0 | 26.5s | 30.6s | 30.0s | 32.8s |
| gmgi_default | recommendation | 3 | 0 | 104.1s | 111.1s | 109.7s | 114.0s |
| gmgi_default | greeting_story | 3 | 0 | 25.1s | 28.9s | 29.8s | 35.4s |
| gmgi_dynamic_v1 | recipient_profiling | 2 | 1 | 30.9s | 31.4s | 31.4s | 31.9s |
| gmgi_dynamic_v1 | relationship_analysis | 2 | 1 | 31.7s | 32.2s | 32.2s | 32.6s |
| gmgi_dynamic_v1 | recommendation | 2 | 1 | 129.2s | 134.7s | 134.7s | 140.1s |
| gmgi_dynamic_v1 | greeting_story | 2 | 1 | 33.9s | 38.3s | 38.3s | 42.6s |

Deterministic intent, planning, and delivery stages remained near-zero latency in all successful runs.

## Trajectory Comparison

| Case | Harness | Repetition | Trajectory | Divergence Type |
|---|---|---:|---|---|
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_default | 1 | recipient -> relationship -> intent -> planning -> recommendation -> greeting -> delivery | NO_DIVERGENCE |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_default | 2 | recipient -> relationship -> intent -> planning -> recommendation -> greeting -> delivery | NO_DIVERGENCE |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_default | 3 | recipient -> relationship -> intent -> planning -> recommendation -> greeting -> delivery | NO_DIVERGENCE |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_dynamic_v1 | 1 | recipient -> relationship -> intent -> planning -> recommendation -> greeting -> delivery | NO_DIVERGENCE |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_dynamic_v1 | 2 | recipient -> relationship -> intent -> planning -> recommendation -> greeting -> delivery | NO_DIVERGENCE |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_dynamic_v1 | 3 | recipient -> relationship -> intent -> planning -> recommendation -> greeting -> delivery | MODEL_VARIANCE |

No harness-induced stage-sequence divergence was observed. The dynamic harness did not route to a different stage sequence in this diagnostic sample.

## Harness-Induced Divergence

Harness-induced divergence count: `0`.

The observed candidate differences are not yet attributable to dynamic routing decisions because the stage trajectory stayed the same. The dynamic failure in repetition 3 is better classified as model/runtime variance.

## Runtime-Induced Divergence

Runtime/model variance remains present:

- Same harness and same input produced one dynamic failure in repetition 3.
- The default harness was stable in this N=3 sample.
- The dynamic failure occurred before a meaningful alternate harness decision could explain the outcome.

## Quality

Only completed executions are included in quality aggregation.

| Harness | Quality N | Mean Quality | Median Quality |
|---|---:|---:|---:|
| gmgi_default | 3 | 0.843 | 0.841 |
| gmgi_dynamic_v1 | 2 | 0.827 | 0.827 |

The partial dynamic run is not counted as quality zero.

## Reliability

| Harness | N | Success Rate | Timeout Rate | Failure Rate |
|---|---:|---:|---:|---:|
| gmgi_default | 3 | 1.000 | 0.000 | 0.000 |
| gmgi_dynamic_v1 | 3 | 0.667 | 0.000 | 0.333 |

## Efficiency

Successful executions only for latency:

| Harness | Successful N | Mean Latency | Median Latency | Mean Invocations | Tokens | Cost |
|---|---:|---:|---:|---:|---|---|
| gmgi_default | 3 | 201.431s | 204.359s | 7.000 | unavailable | unknown |
| gmgi_dynamic_v1 | 2 | 236.517s | 236.517s | 7.000 | unavailable | unknown |

## Limitations

- N=3 is still small.
- Runs were compact in-process replications to avoid large trace writes while local disk was tight.
- Token and cost metrics are unavailable from the local Ollama path.
- Creative/image generation was intentionally disabled.
- Dynamic-verified was not replicated in Phase 6.5 because the methodology gate first compares default vs dynamic.

## Decision Gate

Decision: **PRELIMINARY COMPARISON**.

Reason: default baseline was stable at N=3, but dynamic comparison has only small-N evidence and one runtime/model failure. The correct next step is a persisted 5-case x 3-harness x 1-repetition benchmark, then replications if stable.

## Meta-Harness Readiness

Status: **READY FOR SYSTEMATIC HARNESS BENCHMARK**, not ready for Meta-Harness.

We can now separate harness-induced divergence from runtime-induced divergence, but we do not yet have enough stable repeated evidence to automate harness selection.
