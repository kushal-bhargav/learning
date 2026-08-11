# Phase 6 Harness Benchmark Report

## Experiment Configuration

| Field | Value |
|---|---|
| experiment_id | phase6_empirical_harness_benchmark_v1 |
| provider | ollama |
| model | llama3.2:latest |
| creative generation | disabled |
| state isolation | isolated |
| repetitions | 1 planned initially |
| stage timeout | 180s |
| execution timeout | 500s |
| evaluation version | gmgi_phase_5_harness_comparison_v1 |

Frozen model parameters:

| Parameter | Value |
|---|---|
| GMGI_OLLAMA_TIMEOUT_SECONDS | 160 |
| GMGI_OLLAMA_NUM_PREDICT | 384 |
| GMGI_OLLAMA_NUM_CTX | 4096 |
| GMGI_RECOMMENDATION_MAX_STEPS | 2 |
| GMGI_GREETING_NUM_PREDICT | 160 |

## Candidate Definitions

| Candidate | Orchestration | Routing | Planner | Verification | Retry | Fallback |
|---|---|---|---|---|---|---|
| gmgi_default | fixed_stage | static | advisory | schema_validation_with_offline_evals | agent_local_retries | agent_local_fallbacks |
| gmgi_dynamic_v1 | dynamic | dynamic | advisory | schema_validation_with_offline_evals | agent_local_retries | agent_local_fallbacks |
| gmgi_dynamic_verified_v1 | dynamic | dynamic | advisory | deterministic_constraints | agent_local_retries | agent_local_fallbacks |

These candidates differ in runtime policy, not just metadata: routing changes between default and dynamic candidates, and verification changes between dynamic and dynamic-verified.

## Case Selection

The Phase 6 manifest prepares five stratified UI permutation cases from `experiments/evals/ui_permutation_cases.json`:

| Case ID | Relationship | Occasion | Budget | Formality | Agency |
|---|---|---|---|---|---|
| ui_perm_001_partner_birthday_c5_0_a0_15 | partner | Birthday | Flexible | casual | 0.15 |
| ui_perm_2477_sibling_promotion_c4_0_a0_5 | sibling | Promotion | USD 25-45 | professional | 0.5 |
| ui_perm_4653_friend_graduation_c3_5_a0_85 | friend | Graduation | USD 60-100 | semi-formal | 0.85 |
| ui_perm_8926_colleague_anniversary_c2_0_a0_15 | colleague | Anniversary | USD 150-250 | ceremonial | 0.15 |
| ui_perm_5701_parent-child_housewarming_c4_0_a0_15 | parent-child | Housewarming | Flexible | casual | 0.15 |

This covers five relationship types, five occasions, four budget regimes, four formality regimes, and three agency slider values. It is a small stratified subset, not a statistically representative sample of all 12,600 UI permutations.

## Baseline Gate

Phase 5.6 established one successful real-model baseline on `close_partner_birthday_memory_art`:

| Stage | Status | Latency |
|---|---|---:|
| recipient_profiling | SUCCESS | 29.807s |
| relationship_analysis | SUCCESS | 29.251s |
| gift_intent_reasoning | SUCCESS | 0.002s |
| multi_agent_planning | SUCCESS | 0.004s |
| recommendation | SUCCESS | 117.169s |
| greeting_story | SUCCESS | 38.152s |
| delivery_planner | SUCCESS | 0.002s |

Final status: `SUCCESS`.

## First Same-Case Execution Results

A compact real same-case run was executed in-process for `ui_perm_001_partner_birthday_c5_0_a0_15` using the same model/runtime settings. Creative generation was disabled.

| Case | Harness | Status | Quality | Reliability | Latency | Tokens | Cost | Invocations | Retries | Verifications |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_default | PARTIAL | N/A | N/A | 283.179s | unavailable | unknown | 7 | N/A | N/A |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_dynamic_v1 | SUCCESS | N/A | N/A | 246.596s | unavailable | unknown | 7 | N/A | N/A |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_dynamic_verified_v1 | PARTIAL | N/A | N/A | 186.920s | unavailable | unknown | 7 | N/A | N/A |

Quality is listed as `N/A` because this compact run did not persist the full comparison object, and infrastructure/stage failures must not be converted to zero quality.

## Aggregate Results

| Harness | N | Success Rate | Mean Quality | Mean Latency | Mean Invocations | Mean Retries | Mean Verifications |
|---|---:|---:|---:|---:|---:|---:|---:|
| gmgi_default | 1 | 0.000 | N/A | 283.179s | 7.000 | N/A | N/A |
| gmgi_dynamic_v1 | 1 | 1.000 | N/A | 246.596s | 7.000 | N/A | N/A |
| gmgi_dynamic_verified_v1 | 1 | 0.000 | N/A | 186.920s | 7.000 | N/A | N/A |

With `N=1`, these are observations only. They are not statistically significant.

## Trajectory Comparison

| Case | Harness | Trajectory | Divergence Trigger |
|---|---|---|---|
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_default | recipient:success -> relationship:success -> intent:success -> planning:success -> recommendation:error -> greeting:success -> delivery:success | recommendation model output failed |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_dynamic_v1 | recipient:success -> relationship:success -> intent:success -> planning:success -> recommendation:success -> greeting:success -> delivery:success | none |
| ui_perm_001_partner_birthday_c5_0_a0_15 | gmgi_dynamic_verified_v1 | recipient:success -> relationship:error -> intent:success -> planning:success -> recommendation:success -> greeting:success -> delivery:success | relationship model output failed before verification could help |

The actual stage sequence did not change; the observed divergence was in stage outcomes. In this run, dynamic routing did not add or remove stages.

## Failure Analysis

| Harness | Failure Type | Failed Stage | Interpretation |
|---|---|---|---|
| gmgi_default | model/stage failure | recommendation | Not attributable to harness quality; the model failed to produce a usable recommendation output in this repetition. |
| gmgi_dynamic_v1 | none | none | Completed the same case successfully in this repetition. |
| gmgi_dynamic_verified_v1 | model/stage failure | relationship_analysis | Verification did not prevent this because the failure occurred before usable relationship output existed. |

Infrastructure/model instability is still present. Therefore the 5-case expansion was intentionally not run in this local environment.

## Pareto Analysis

Pareto analysis is not meaningful yet because quality/reliability values were unavailable for the compact run and the baseline gate was not stable enough to expand. Missing metrics were not coerced to zero.

## Creative Results

Not run in Phase 6. Creative/image benchmarking remains separate and should only be enabled after the non-creative benchmark is stable.

## Limitations

- Only one same-case comparison was attempted.
- The default baseline did not reproduce on the first UI permutation case in this environment.
- Token and cost metrics are unavailable from the local Ollama path.
- The comparison is descriptive, not statistically significant.
- Creative generation is excluded.

## Conclusions

Dynamic routing did not yet show a trajectory change on this case. `gmgi_dynamic_v1` succeeded where the default and dynamic-verified candidates had model/stage failures, but this cannot be interpreted as a stable quality advantage without successful replications.

Verification did not help in this run because the verified candidate failed in `relationship_analysis` before verifier policy could improve downstream behavior.

The correct next step is to improve runtime stability/reproducibility, rerun the one-case three-harness comparison with persisted full traces, and only then expand to the prepared 5-case manifest.

## Meta-Harness Readiness

Status: **PARTIALLY READY**.

The system has candidates, same-case execution, isolated state, trajectory capture, failure attribution, and report generation. It is not ready for automatic harness selection because empirical comparisons are still too affected by model/runtime instability.
