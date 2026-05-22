# Phase8 PAMAE Entity Seeding Summary Repaired

## Main Table

| dataset | profile | variant | status | num_queries | EM | F1 | SF-R | SF-P | SF-F1 | avg_context_tokens | retrieval_ms | total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | COMPLETE | 100 | 0.0500 | 0.0529 | 0.0225 | 0.0063 | 0.0097 | 536.0700 | 21118.1878 | 21380.5359 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | COMPLETE | 100 | 0.0700 | 0.0751 | 0.2375 | 0.0688 | 0.1050 | 474.3800 | 25024.7380 | 25302.7599 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | COMPLETE | 100 | 0.1200 | 0.1317 | 0.3425 | 0.1037 | 0.1563 | 445.3600 | 15344.7971 | 15755.9385 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | COMPLETE | 100 | 0.0500 | 0.0551 | 0.0225 | 0.0050 | 0.0081 | 642.5700 | 20713.1164 | 20964.9706 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | COMPLETE | 100 | 0.0800 | 0.0854 | 0.2425 | 0.0560 | 0.0898 | 561.4000 | 25158.7831 | 25449.7509 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | COMPLETE | 100 | 0.0800 | 0.0950 | 0.3725 | 0.0890 | 0.1414 | 521.6200 | 15469.2437 | 15880.7765 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | COMPLETE | 100 | 0.0400 | 0.0608 | 0.0600 | 0.0187 | 0.0281 | 548.6900 | 35980.1614 | 36360.1573 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | MISSING | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | balanced_384_8 | source_balanced_128 | COMPLETE | 100 | 0.2300 | 0.2942 | 0.4523 | 0.1425 | 0.2133 | 461.0100 | 24289.6216 | 24836.1877 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | COMPLETE | 100 | 0.0500 | 0.0708 | 0.0600 | 0.0150 | 0.0237 | 648.9400 | 36169.6391 | 36554.3341 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | PARTIAL | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 543.0000 | 72163.1359 | 73329.6133 |
| hotpotqa | legacy_512_10 | source_balanced_128 | COMPLETE | 100 | 0.2500 | 0.3172 | 0.5040 | 0.1260 | 0.1988 | 545.4700 | 24257.6635 | 24813.7115 |

## Repaired Proposal Table

| dataset | profile | variant | status | num_queries | normalized_candidate_gold_recall | normalized_selected_gold_recall | normalized_rendered_gold_recall | candidate_oracle_F1 | selected_context_F1 | chain_unit_oracle_feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | COMPLETE | 100 | 0.0325 | 0.0225 | 0.0225 | 0.0440 | 0.0097 | 0.0000 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | COMPLETE | 100 | 0.3625 | 0.2375 | 0.2375 | 0.4372 | 0.1050 | 0.0800 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | COMPLETE | 100 | 0.5475 | 0.3425 | 0.3425 | 0.6397 | 0.1563 | 0.0600 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | COMPLETE | 100 | 0.0325 | 0.0225 | 0.0225 | 0.0440 | 0.0081 | 0.0000 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | COMPLETE | 100 | 0.3625 | 0.2425 | 0.2425 | 0.4372 | 0.0898 | 0.0800 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | COMPLETE | 100 | 0.5475 | 0.3725 | 0.3725 | 0.6397 | 0.1414 | 0.0600 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | COMPLETE | 100 | 0.0665 | 0.0600 | 0.0600 | 0.0823 | 0.0281 | 0.0200 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | MISSING | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | balanced_384_8 | source_balanced_128 | COMPLETE | 100 | 0.7433 | 0.4523 | 0.4523 | 0.8002 | 0.2133 | 0.2500 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | COMPLETE | 100 | 0.0665 | 0.0600 | 0.0600 | 0.0823 | 0.0237 | 0.0200 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | PARTIAL | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | legacy_512_10 | source_balanced_128 | COMPLETE | 100 | 0.7433 | 0.5040 | 0.5040 | 0.8002 | 0.1988 | 0.2500 |

## Phase8 Seed Table

| dataset | profile | variant | entity_universe_size | gold_entity_hit_rate_eval_only | seed_gold_hit_rate_eval_only | seed_evidence_gold_hit_rate_eval_only | num_changed_seeds | before_refine_seed_evidence_hit | after_refine_seed_evidence_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 2000.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 2000.0000 | 0.0000 | 0.0000 | 0.0000 | 4.9900 | 0.0000 | 0.0000 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 2000.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 2000.0000 | 0.0000 | 0.0000 | 0.0000 | 4.9900 | 0.0000 | 0.0000 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 2000.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | balanced_384_8 | source_balanced_128 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 2000.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 2000.0000 | 0.0000 | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 |
| hotpotqa | legacy_512_10 | source_balanced_128 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Timing Table

| dataset | profile | variant | entity_universe_ms | sampling_ms | kmedoids_ms | seed_selection_ms | refinement_ms | evidence_proposal_ms | retrieval_ms | total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 3053.1479 | 1651.4936 | 1639.8560 | 664.6839 | 259.8904 | 2281.6021 | 21118.1878 | 21380.5359 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 2906.4565 | 1814.2138 | 1801.5453 | 723.9981 | 1511.9678 | 3434.5648 | 25024.7380 | 25302.7599 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15344.7971 | 15755.9385 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 2935.6602 | 1738.4477 | 1725.7087 | 685.8017 | 271.0581 | 2296.1980 | 20713.1164 | 20964.9706 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 2966.1339 | 1689.5240 | 1676.9227 | 690.9465 | 1475.0237 | 3548.3968 | 25158.7831 | 25449.7509 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15469.2437 | 15880.7765 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 4544.5553 | 1665.4302 | 1653.4578 | 668.6663 | 261.2972 | 4408.0646 | 35980.1614 | 36360.1573 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | balanced_384_8 | source_balanced_128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24289.6216 | 24836.1877 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 4690.6177 | 1786.1395 | 1773.1556 | 709.9138 | 281.0666 | 4511.9614 | 36169.6391 | 36554.3341 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 4560.5199 | 1604.9105 | 1594.7569 | 720.1384 | 3938.2122 | 5812.1430 | 72163.1359 | 73329.6133 |
| hotpotqa | legacy_512_10 | source_balanced_128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24257.6635 | 24813.7115 |

## Trace Completeness

| dataset | profile | variant | expected_num_queries | actual_num_queries | status |
| --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 100 | COMPLETE |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 100 | COMPLETE |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 100 | COMPLETE |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 100 | COMPLETE |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 100 | COMPLETE |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 100 | COMPLETE |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 100 | COMPLETE |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 0 | MISSING |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 100 | COMPLETE |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 100 | COMPLETE |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 1 | PARTIAL |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 100 | COMPLETE |

Missing or partial traces are status-coded and are not silently interpreted as zeros.

