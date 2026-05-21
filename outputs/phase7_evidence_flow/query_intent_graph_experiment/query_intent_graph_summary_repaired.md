# Query-Intent Graph Summary (Repaired)

## Main Result Table
| dataset | profile | variant | EM | F1 | SF-R | SF-P | avg_tokens | retrieval_ms | total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | baseline_bq | 0.0600 | 0.0717 | 0.3025 | 0.0900 | 434.68 | 7798.29 | 7998.57 |
| 2wikimultihopqa | balanced_384_8 | intent_p1 | 0.0800 | 0.0917 | 0.2900 | 0.0875 | 439.15 | 7887.01 | 8077.60 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq | 0.0900 | 0.1017 | 0.2950 | 0.0887 | 436.92 | 8099.35 | 8290.48 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq_rq | 0.0700 | 0.0817 | 0.3000 | 0.0900 | 427.56 | 8223.60 | 8414.65 |
| 2wikimultihopqa | legacy_512_10 | baseline_bq | 0.0600 | 0.0745 | 0.3300 | 0.0790 | 508.38 | 7648.33 | 7848.43 |
| 2wikimultihopqa | legacy_512_10 | intent_p1 | 0.0600 | 0.0745 | 0.3200 | 0.0770 | 513.23 | 8318.88 | 8521.34 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq | 0.0600 | 0.0790 | 0.3100 | 0.0750 | 510.92 | 8410.22 | 8616.79 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq_rq | 0.0600 | 0.0745 | 0.3100 | 0.0750 | 506.89 | 8062.23 | 8256.90 |
| hotpotqa | balanced_384_8 | baseline_bq | 0.2400 | 0.2956 | 0.4507 | 0.1437 | 453.80 | 13808.90 | 14129.61 |
| hotpotqa | balanced_384_8 | intent_p1 | 0.2400 | 0.2996 | 0.4457 | 0.1412 | 454.83 | 15374.02 | 15731.18 |
| hotpotqa | balanced_384_8 | intent_p1_aq | 0.2200 | 0.2856 | 0.4423 | 0.1412 | 447.07 | 15160.77 | 15516.72 |
| hotpotqa | balanced_384_8 | intent_p1_aq_rq | 0.2300 | 0.2896 | 0.4565 | 0.1450 | 446.67 | 15263.52 | 15617.21 |
| hotpotqa | legacy_512_10 | baseline_bq | 0.2600 | 0.3253 | 0.5090 | 0.1300 | 533.93 | 14546.00 | 14899.71 |
| hotpotqa | legacy_512_10 | intent_p1 | 0.2800 | 0.3453 | 0.5032 | 0.1280 | 533.46 | 14431.06 | 14766.92 |
| hotpotqa | legacy_512_10 | intent_p1_aq | 0.2700 | 0.3353 | 0.4823 | 0.1230 | 531.36 | 13401.50 | 13713.87 |
| hotpotqa | legacy_512_10 | intent_p1_aq_rq | 0.1600 | 0.1919 | 0.3042 | 0.0855 | 573.80 | 3534.06 | 3614.37 |

## Oracle Gap Table
| dataset | profile | variant | phase1_partial | phase1_full | phase1_gold_recall | selected_partial | selected_full | selected_gold_recall | rendered_partial | rendered_full | rendered_gold_recall | selected_F1 | gold_oracle_F1 | oracle_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | baseline_bq | 0.8300 | 0.1900 | 0.4875 | 0.6600 | 0.0400 | 0.3025 | 0.6600 | 0.0400 | 0.3025 | 0.0124 | 0.3907 | 0.3783 |
| 2wikimultihopqa | balanced_384_8 | intent_p1 | 0.8300 | 0.1900 | 0.4875 | 0.6300 | 0.0400 | 0.2900 | 0.6300 | 0.0400 | 0.2900 | 0.0127 | 0.3883 | 0.3756 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq | 0.8300 | 0.1900 | 0.4875 | 0.6400 | 0.0400 | 0.2950 | 0.6400 | 0.0400 | 0.2950 | 0.0128 | 0.3816 | 0.3688 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq_rq | 0.8300 | 0.1900 | 0.4875 | 0.6500 | 0.0400 | 0.3000 | 0.6500 | 0.0400 | 0.3000 | 0.0125 | 0.3841 | 0.3716 |
| 2wikimultihopqa | legacy_512_10 | baseline_bq | 0.8300 | 0.1900 | 0.4875 | 0.6800 | 0.0600 | 0.3300 | 0.6800 | 0.0600 | 0.3300 | 0.0155 | 0.3907 | 0.3752 |
| 2wikimultihopqa | legacy_512_10 | intent_p1 | 0.8300 | 0.1900 | 0.4875 | 0.6600 | 0.0600 | 0.3200 | 0.6600 | 0.0600 | 0.3200 | 0.0155 | 0.3876 | 0.3722 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq | 0.8300 | 0.1900 | 0.4875 | 0.6500 | 0.0500 | 0.3100 | 0.6500 | 0.0500 | 0.3100 | 0.0202 | 0.3907 | 0.3705 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq_rq | 0.8300 | 0.1900 | 0.4875 | 0.6600 | 0.0400 | 0.3100 | 0.6600 | 0.0400 | 0.3100 | 0.0119 | 0.3800 | 0.3681 |
| hotpotqa | balanced_384_8 | baseline_bq | 0.9400 | 0.6100 | 0.7708 | 0.7500 | 0.1500 | 0.4507 | 0.7500 | 0.1500 | 0.4507 | 0.0732 | 0.4971 | 0.4240 |
| hotpotqa | balanced_384_8 | intent_p1 | 0.9400 | 0.6100 | 0.7708 | 0.7500 | 0.1500 | 0.4457 | 0.7500 | 0.1500 | 0.4457 | 0.0732 | 0.4971 | 0.4240 |
| hotpotqa | balanced_384_8 | intent_p1_aq | 0.9400 | 0.6100 | 0.7708 | 0.7400 | 0.1400 | 0.4423 | 0.7400 | 0.1400 | 0.4423 | 0.0841 | 0.5049 | 0.4208 |
| hotpotqa | balanced_384_8 | intent_p1_aq_rq | 0.9400 | 0.6100 | 0.7708 | 0.7600 | 0.1500 | 0.4565 | 0.7600 | 0.1500 | 0.4565 | 0.0774 | 0.4777 | 0.4003 |
| hotpotqa | legacy_512_10 | baseline_bq | 0.9400 | 0.6100 | 0.7708 | 0.8000 | 0.2100 | 0.5090 | 0.8000 | 0.2100 | 0.5090 | 0.0882 | 0.4522 | 0.3641 |
| hotpotqa | legacy_512_10 | intent_p1 | 0.9400 | 0.6100 | 0.7708 | 0.8000 | 0.2100 | 0.5032 | 0.8000 | 0.2100 | 0.5032 | 0.0906 | 0.4509 | 0.3603 |
| hotpotqa | legacy_512_10 | intent_p1_aq | 0.9400 | 0.6100 | 0.7708 | 0.7700 | 0.1900 | 0.4823 | 0.7700 | 0.1900 | 0.4823 | 0.0857 | 0.4326 | 0.3469 |
| hotpotqa | legacy_512_10 | intent_p1_aq_rq | 0.2900 | 0.2500 | 0.2675 | 0.6700 | 0.1600 | 0.3990 | 0.6700 | 0.1600 | 0.3990 | 0.0572 | 0.5368 | 0.4796 |

## Failure Attribution Table
| dataset | profile | variant | n_failed | PHASE1_NOT_FOUND | SEMANTIC_ANCHOR_FAILED | PRUNING_FAILED | FINAL_SELECTION_FAILED | RENDERING_ID_MISMATCH | SELECTED_NOT_SUFFICIENT | QA_PROMPT_FAILED_WITH_GOLD_CONTEXT | QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT | OTHER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | baseline_bq | 94 | 17 | 17 | 5 | 11 | 0 | 21 | 31 | 0 | 0 |
| 2wikimultihopqa | balanced_384_8 | intent_p1 | 92 | 17 | 17 | 3 | 16 | 0 | 19 | 30 | 0 | 0 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq | 91 | 17 | 17 | 3 | 15 | 0 | 16 | 32 | 0 | 0 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq_rq | 93 | 17 | 17 | 3 | 15 | 0 | 18 | 32 | 0 | 0 |
| 2wikimultihopqa | legacy_512_10 | baseline_bq | 94 | 17 | 17 | 5 | 8 | 0 | 16 | 35 | 2 | 0 |
| 2wikimultihopqa | legacy_512_10 | intent_p1 | 94 | 17 | 17 | 3 | 12 | 0 | 14 | 36 | 2 | 0 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq | 94 | 17 | 17 | 3 | 13 | 0 | 14 | 34 | 1 | 0 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq_rq | 94 | 17 | 17 | 3 | 14 | 0 | 14 | 34 | 0 | 0 |
| hotpotqa | balanced_384_8 | baseline_bq | 76 | 5 | 5 | 5 | 12 | 0 | 19 | 21 | 3 | 6 |
| hotpotqa | balanced_384_8 | intent_p1 | 76 | 5 | 5 | 4 | 13 | 0 | 17 | 21 | 3 | 8 |
| hotpotqa | balanced_384_8 | intent_p1_aq | 78 | 5 | 5 | 4 | 14 | 0 | 15 | 22 | 3 | 10 |
| hotpotqa | balanced_384_8 | intent_p1_aq_rq | 77 | 5 | 5 | 4 | 13 | 0 | 15 | 22 | 3 | 10 |
| hotpotqa | legacy_512_10 | baseline_bq | 74 | 5 | 5 | 5 | 8 | 0 | 19 | 22 | 5 | 5 |
| hotpotqa | legacy_512_10 | intent_p1 | 72 | 5 | 5 | 4 | 8 | 0 | 17 | 23 | 4 | 6 |
| hotpotqa | legacy_512_10 | intent_p1_aq | 73 | 5 | 5 | 4 | 11 | 0 | 15 | 25 | 2 | 6 |
| hotpotqa | legacy_512_10 | intent_p1_aq_rq | 85 | 61 | 23 | 0 | 3 | 0 | 3 | 6 | 4 | 0 |

## Candidate Source Contribution Table
| dataset | profile | variant | entity_source_gold_count | entity_source_gold_hit_rate | relation_source_gold_count | relation_source_gold_hit_rate | answer_type_source_gold_count | answer_type_source_gold_hit_rate | graph_flow_gold_count | graph_flow_gold_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | baseline_bq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2wikimultihopqa | balanced_384_8 | intent_p1 | 0.9900 | 0.7900 | 0.0500 | 0.0400 | 0.3000 | 0.2400 | 1.0300 | 0.8000 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq | 0.9900 | 0.7900 | 0.0500 | 0.0400 | 0.3000 | 0.2400 | 1.0300 | 0.8000 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq_rq | 0.9900 | 0.7900 | 0.0500 | 0.0400 | 0.3000 | 0.2400 | 1.0300 | 0.8000 |
| 2wikimultihopqa | legacy_512_10 | baseline_bq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2wikimultihopqa | legacy_512_10 | intent_p1 | 0.9900 | 0.7900 | 0.0500 | 0.0400 | 0.3000 | 0.2400 | 1.0300 | 0.8000 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq | 0.9900 | 0.7900 | 0.0500 | 0.0400 | 0.3000 | 0.2400 | 1.0300 | 0.8000 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq_rq | 0.9900 | 0.7900 | 0.0500 | 0.0400 | 0.3000 | 0.2400 | 1.0300 | 0.8000 |
| hotpotqa | balanced_384_8 | baseline_bq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | balanced_384_8 | intent_p1 | 1.5500 | 0.8900 | 0.0200 | 0.0200 | 0.3600 | 0.2300 | 1.8000 | 0.9000 |
| hotpotqa | balanced_384_8 | intent_p1_aq | 1.5500 | 0.8900 | 0.0200 | 0.0200 | 0.3600 | 0.2300 | 1.8000 | 0.9000 |
| hotpotqa | balanced_384_8 | intent_p1_aq_rq | 1.5500 | 0.8900 | 0.0200 | 0.0200 | 0.3600 | 0.2300 | 1.8000 | 0.9000 |
| hotpotqa | legacy_512_10 | baseline_bq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | legacy_512_10 | intent_p1 | 1.5500 | 0.8900 | 0.0200 | 0.0200 | 0.3600 | 0.2300 | 1.8000 | 0.9000 |
| hotpotqa | legacy_512_10 | intent_p1_aq | 1.5500 | 0.8900 | 0.0200 | 0.0200 | 0.3600 | 0.2300 | 1.8000 | 0.9000 |
| hotpotqa | legacy_512_10 | intent_p1_aq_rq | 1.2400 | 0.7600 | 0.0300 | 0.0300 | 0.2800 | 0.1600 | 1.5800 | 0.7700 |

## Timing Table
| dataset | profile | variant | intent_ms | semantic_anchor_ms | local_graph_ms | corridor_ms | feature_Aq_ms | feature_Rq_ms | retrieval_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | baseline_bq | 36.26 | 1025.82 | 3158.92 | 460.43 | 515.18 | 13.40 | 7798.29 |
| 2wikimultihopqa | balanced_384_8 | intent_p1 | 426.61 | 1007.37 | 3004.49 | 408.22 | 477.47 | 16.05 | 7887.01 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq | 439.19 | 1042.37 | 3116.23 | 432.08 | 495.14 | 16.73 | 8099.35 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq_rq | 427.33 | 1039.88 | 3068.14 | 433.08 | 491.57 | 17.11 | 8223.60 |
| 2wikimultihopqa | legacy_512_10 | baseline_bq | 34.94 | 1013.56 | 2985.74 | 449.22 | 504.86 | 19.15 | 7648.33 |
| 2wikimultihopqa | legacy_512_10 | intent_p1 | 429.28 | 1047.58 | 3095.00 | 451.26 | 496.74 | 25.41 | 8318.88 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq | 439.61 | 1059.92 | 3132.94 | 445.53 | 513.29 | 24.77 | 8410.22 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq_rq | 420.87 | 1031.13 | 3004.61 | 434.97 | 484.13 | 26.63 | 8062.23 |
| hotpotqa | balanced_384_8 | baseline_bq | 63.29 | 1674.22 | 5736.15 | 729.15 | 904.82 | 12.63 | 13808.90 |
| hotpotqa | balanced_384_8 | intent_p1 | 708.67 | 1652.21 | 6074.97 | 821.16 | 1019.85 | 19.27 | 15374.02 |
| hotpotqa | balanced_384_8 | intent_p1_aq | 693.90 | 1633.18 | 6004.68 | 805.16 | 1002.56 | 18.44 | 15160.77 |
| hotpotqa | balanced_384_8 | intent_p1_aq_rq | 727.00 | 1706.95 | 6000.82 | 799.41 | 1012.26 | 20.32 | 15263.52 |
| hotpotqa | legacy_512_10 | baseline_bq | 68.69 | 1673.17 | 6046.65 | 764.86 | 991.12 | 19.17 | 14546.00 |
| hotpotqa | legacy_512_10 | intent_p1 | 658.13 | 1636.94 | 5764.87 | 715.03 | 921.73 | 23.47 | 14431.06 |
| hotpotqa | legacy_512_10 | intent_p1_aq | 710.95 | 1682.71 | 5783.52 | 739.59 | 935.83 | 26.09 | 13401.50 |
| hotpotqa | legacy_512_10 | intent_p1_aq_rq | 185.20 | 450.04 | 1499.96 | 190.96 | 233.11 | 22.94 | 3534.06 |

## Sanity & Coverage Source
| dataset | profile | variant | num_queries_with_gold_support | num_queries_missing_gold_support | num_queries_with_phase1_candidates | num_queries_with_selected_context | num_queries_with_rendered_context | coverage_source | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | baseline_bq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| 2wikimultihopqa | balanced_384_8 | intent_p1 | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq_rq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| 2wikimultihopqa | legacy_512_10 | baseline_bq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| 2wikimultihopqa | legacy_512_10 | intent_p1 | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq_rq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| hotpotqa | balanced_384_8 | baseline_bq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| hotpotqa | balanced_384_8 | intent_p1 | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| hotpotqa | balanced_384_8 | intent_p1_aq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| hotpotqa | balanced_384_8 | intent_p1_aq_rq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| hotpotqa | legacy_512_10 | baseline_bq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| hotpotqa | legacy_512_10 | intent_p1 | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| hotpotqa | legacy_512_10 | intent_p1_aq | 100 | 0 | 100 | 100 | 100 | diagnostics_fallback | - |
| hotpotqa | legacy_512_10 | intent_p1_aq_rq | 100 | 0 | 29 | 100 | 100 | diagnostics_fallback | selected_partial_gt_phase1_partial |

## Warnings
- hotpotqa/legacy_512_10/intent_p1_aq_rq: selected_partial_gt_phase1_partial