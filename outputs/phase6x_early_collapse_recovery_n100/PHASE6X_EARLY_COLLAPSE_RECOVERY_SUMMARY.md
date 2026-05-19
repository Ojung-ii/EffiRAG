# PHASE6X Early-Collapse Recovery Summary

## 1. Goal
Test one dataset-agnostic reserve candidate (`reserve=1`) to reduce Phase1/Phase2 early collapse without changing ABR or context budget.

## 2. Artifact Validation
- scheduled_runs: 0
- completed_runs: 0
- incomplete_attempts: 0
- extra_not_in_manifest: 0
- multi_attempt_run_names: 0
- parse_errors: 0

## 3. Paired Main Results
| dataset | profile | EM | F1 | Recall@5 | SF_recall | SF_precision | SF_f1 | avg_context_tokens | F1_per_1k_context_tokens | answer_string_hit | answer_bearing_density | retrieval_ms | generation_ms | total_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | unified_acr_rcedr_v12 | 0.3500 | 0.4577 | 0.4565 | 0.4815 | 0.1367 | 0.2129 | 542.03 | 0.8444 | 0.5900 | 0.0142 | 6504.37 | 84.09 | 6605.94 |
| hotpotqa | unified_acr_rcedr_v12_early_collapse_recovery | 0.4100 | 0.5007 | 0.4637 | 0.4870 | 0.1373 | 0.2142 | 541.93 | 0.9239 | 0.6000 | 0.0146 | 6335.41 | 86.14 | 6439.09 |
| 2wikimultihopqa | unified_acr_rcedr_v12 | 0.2800 | 0.2920 | 0.4000 | 0.4225 | 0.1286 | 0.1972 | 503.51 | 0.5799 | 0.3700 | 0.0144 | 3963.93 | 101.02 | 4080.05 |
| 2wikimultihopqa | unified_acr_rcedr_v12_early_collapse_recovery | 0.3200 | 0.3433 | 0.4000 | 0.4300 | 0.1308 | 0.2006 | 498.59 | 0.6886 | 0.3800 | 0.0148 | 3989.24 | 94.90 | 4098.84 |

## 4. Delta vs Baseline
| dataset | ΔEM | ΔF1 | ΔSF_recall | ΔSF_precision | Δavg_context_tokens | ΔF1_per_1k_context_tokens | Δanswer_string_hit | Δanswer_bearing_density | Δretrieval_ms | Δtotal_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | 0.0600 | 0.0430 | 0.0055 | 0.0007 | -0.10 | 0.0795 | 0.0100 | 0.0004 | -168.96 | -166.84 |
| 2wikimultihopqa | 0.0400 | 0.0513 | 0.0075 | 0.0022 | -4.92 | 0.1087 | 0.0100 | 0.0004 | 25.31 | 18.79 |

## 5. Stagewise Evidence Transfer
- hotpotqa baseline deltas: {'S1_to_S2_SF_delta': -0.34933333333333316, 'S2_to_S3_SF_delta': -0.04916666666666658, 'S3_to_S4_SF_delta': 0.08833333333333337, 'S4_to_S5_SF_delta': -0.1133333333333334, 'S1_to_S2_answer_hit_delta': -0.30999999999999994, 'S2_to_S3_answer_hit_delta': -0.010000000000000009, 'S3_to_S4_answer_hit_delta': 0.13, 'S4_to_S5_answer_hit_delta': -0.12}
- hotpotqa variant deltas: {'S1_to_S2_SF_delta': -0.34933333333333316, 'S2_to_S3_SF_delta': -0.04033333333333322, 'S3_to_S4_SF_delta': 0.0808333333333332, 'S4_to_S5_SF_delta': -0.10916666666666658, 'S1_to_S2_answer_hit_delta': -0.30999999999999994, 'S2_to_S3_answer_hit_delta': -0.010000000000000009, 'S3_to_S4_answer_hit_delta': 0.10999999999999999, 'S4_to_S5_answer_hit_delta': -0.08999999999999997}
- 2wikimultihopqa baseline deltas: {'S1_to_S2_SF_delta': -0.12, 'S2_to_S3_SF_delta': -0.08750000000000002, 'S3_to_S4_SF_delta': 0.10999999999999999, 'S4_to_S5_SF_delta': -0.16749999999999998, 'S1_to_S2_answer_hit_delta': -0.13, 'S2_to_S3_answer_hit_delta': -0.11000000000000004, 'S3_to_S4_answer_hit_delta': 0.13000000000000006, 'S4_to_S5_answer_hit_delta': -0.13000000000000006}
- 2wikimultihopqa variant deltas: {'S1_to_S2_SF_delta': -0.12, 'S2_to_S3_SF_delta': -0.09500000000000003, 'S3_to_S4_SF_delta': 0.12000000000000005, 'S4_to_S5_SF_delta': -0.16250000000000003, 'S1_to_S2_answer_hit_delta': -0.13, 'S2_to_S3_answer_hit_delta': -0.10000000000000003, 'S3_to_S4_answer_hit_delta': 0.13000000000000006, 'S4_to_S5_answer_hit_delta': -0.13000000000000006}

## 6. Phase1 Reserve Diagnostics
| dataset | reserve_trigger_rate | reserve_phase2_survival_rate | reserve_sentence_candidate_rate | reserve_ABR_selected_rate | reserve_rendered_rate |
|---|---:|---:|---:|---:|---:|
| hotpotqa | 0.1800 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2wikimultihopqa | 0.1600 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 7. Candidate Ceiling Recovery
See `early_collapse_stagewise_metrics.csv` for S1~S6 per-profile aggregates and transfer deltas.

## 8. Token/Latency Guardrail
- avg_delta_f1: 0.0472
- max_dataset_drop_f1: 0.0430
- avg_delta_context_tokens: -2.51
- avg_delta_retrieval_ms: -71.82

## 9. Improved Examples
- `improved_examples_top20.md` (16 rows)

## 10. Regression Examples
- `regression_examples_top20.md` (5 rows)

## 11. Decision
- strong_accept
