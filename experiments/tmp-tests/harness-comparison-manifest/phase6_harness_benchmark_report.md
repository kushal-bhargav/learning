# Phase 6 Harness Benchmark Report

## Experiment Configuration
- `provider`: ollama
- `model`: N/A
- `model_parameters`: {}
- `evaluation_version`: gmgi_phase_5_harness_comparison_v1
- `state_isolation`: isolated
- `creative_enabled`: False
- `seed`: 123
- `stage_timeout_seconds`: N/A
- `execution_timeout_seconds`: N/A
- `candidate_ids`: ["gmgi_default"]

## Candidate Definitions
| candidate_id | config_hash | orchestration | routing | planner | verification | retry | fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gmgi_default | b3f074b0af45a34a | fixed_stage | static | advisory | schema_validation_with_offline_evals | agent_local_retries | agent_local_fallbacks |

## Case Selection
- `case_ids`: ["case-one"]
- `case_count`: 1
- `dimensions`: {"relationship_type": {"unique_count": 1, "values": [""]}, "occasion_name": {"unique_count": 1, "values": [""]}, "budget_hint": {"unique_count": 1, "values": [""]}, "formality": {"unique_count": 1, "values": [""]}, "agency_slider": {"unique_count": 1, "values": [""]}}
- `selection_note`: Small stratified UI-permutation subset; not statistically representative of all 12,600 cases.

## Execution Results
| case_id | candidate_id | status | quality | reliability | latency | tokens | cost | invocations | retries | verifications |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case-one | gmgi_default | N/A | 0.700 | 1.000 | 0.200 | N/A | N/A | 2 | 0 | N/A |

## Aggregate Results
| candidate_id | n | success_rate | mean_quality | mean_latency | mean_invocations | mean_retries | mean_verifications |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gmgi_default | 1 | 0.000 | 0.700 | 0.200 | 2.000 | 0.000 | N/A |

## Trajectory Comparison
N/A

## Failure Analysis
N/A

## Pareto Analysis
- `non_dominated_candidates`: []
- `dominated_candidates`: []
- `excluded_rows`: 0
- `note`: Computed per case over available quality/reliability/latency only.

## Limitations
- Results are descriptive unless enough repetitions are present.
- Missing token and cost values remain unavailable/unknown; they are not estimated.
- Infrastructure failures are not quality failures.
- Creative/image generation is excluded when `include_creative=false`.

## Conclusions
This report provides empirical traces and descriptive comparisons only. It does not declare a universal best harness and does not implement automatic harness selection.
