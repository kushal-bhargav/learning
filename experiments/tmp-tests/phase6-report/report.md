# Phase 6 Harness Benchmark Report

## Experiment Configuration
- `model`: llama3.2:latest
- `provider`: ollama

## Candidate Definitions
| candidate_id | config_hash | orchestration | routing | planner | verification | retry | fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gmgi_default | abc | fixed_stage | static | advisory | schema | local | local |

## Case Selection
- `case_ids`: ["case"]
- `case_count`: 1

## Execution Results
| case_id | candidate_id | status | quality | reliability | latency | tokens | cost | invocations | retries | verifications |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case | gmgi_default | TIMEOUT | N/A | 0.800 | 10.000 | N/A | N/A | 3 | 0 | 0 |

## Aggregate Results
| candidate_id | n | success_rate | mean_quality | mean_latency | mean_invocations | mean_retries | mean_verifications |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gmgi_default | 1 | 0.000 | N/A | 10.000 | 3.000 | 0.000 | 0.000 |

## Trajectory Comparison
| case_id | candidate_id | trajectory_text | divergence_trigger |
| --- | --- | --- | --- |
| case | gmgi_default | recipient | none |

## Failure Analysis
N/A

## Pareto Analysis
- `non_dominated_candidates`: []
- `dominated_candidates`: []
- `excluded_rows`: 1
- `note`: Computed per case over available quality/reliability/latency only.

## Limitations
- Results are descriptive unless enough repetitions are present.
- Missing token and cost values remain unavailable/unknown; they are not estimated.
- Infrastructure failures are not quality failures.
- Creative/image generation is excluded when `include_creative=false`.

## Conclusions
This report provides empirical traces and descriptive comparisons only. It does not declare a universal best harness and does not implement automatic harness selection.
