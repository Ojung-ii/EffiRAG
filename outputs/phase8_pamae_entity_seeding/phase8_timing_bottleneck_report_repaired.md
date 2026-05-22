# Phase8 Timing Bottleneck Report Repaired

Means, p50, p95, and max are reported in the JSON for every stage. This markdown table includes mean timings and the largest Phase8-specific stage.

| dataset | profile | variant | status | num_queries | entity_universe_ms_mean | sampling_ms_mean | kmedoids_ms_mean | seed_selection_ms_mean | refinement_ms_mean | evidence_proposal_ms_mean | feature_construction_ms_mean | selection_ms_mean | generation_ms_mean | total_retrieval_ms_mean | total_ms_mean | largest_phase8_stage | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | COMPLETE | 100 | 3053.1479 | 1651.4936 | 1639.8560 | 664.6839 | 259.8904 | 2281.6021 | 519.8867 | 48.8243 | 82.3221 | 21118.1878 | 21380.5359 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | COMPLETE | 100 | 2906.4565 | 1814.2138 | 1801.5453 | 723.9981 | 1511.9678 | 3434.5648 | 705.7424 | 68.6282 | 68.8501 | 25024.7380 | 25302.7599 | evidence_proposal_ms | Evidence proposal is the largest Phase8 stage. |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | COMPLETE | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 730.2614 | 68.1681 | 217.1810 | 15344.7971 | 15755.9385 | N/A | Baseline timing; Phase8 stages should be zero. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | COMPLETE | 100 | 2935.6602 | 1738.4477 | 1725.7087 | 685.8017 | 271.0581 | 2296.1980 | 522.7841 | 53.8677 | 73.7303 | 20713.1164 | 20964.9706 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | COMPLETE | 100 | 2966.1339 | 1689.5240 | 1676.9227 | 690.9465 | 1475.0237 | 3548.3968 | 717.4380 | 77.0351 | 77.1176 | 25158.7831 | 25449.7509 | evidence_proposal_ms | Evidence proposal is the largest Phase8 stage. |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | COMPLETE | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 775.0696 | 80.8277 | 206.4730 | 15469.2437 | 15880.7765 | N/A | Baseline timing; Phase8 stages should be zero. |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | COMPLETE | 100 | 4544.5553 | 1665.4302 | 1653.4578 | 668.6663 | 261.2972 | 4408.0646 | 984.5577 | 82.5270 | 65.8371 | 35980.1614 | 36360.1573 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | MISSING | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | balanced_384_8 | source_balanced_128 | COMPLETE | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1504.7662 | 112.2913 | 175.5726 | 24289.6216 | 24836.1877 | N/A | Baseline timing; Phase8 stages should be zero. |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | COMPLETE | 100 | 4690.6177 | 1786.1395 | 1773.1556 | 709.9138 | 281.0666 | 4511.9614 | 996.1127 | 85.0658 | 73.0515 | 36169.6391 | 36554.3341 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | PARTIAL | 1 | 4560.5199 | 1604.9105 | 1594.7569 | 720.1384 | 3938.2122 | 5812.1430 | 1349.4523 | 117.8569 | 781.7042 | 72163.1359 | 73329.6133 | evidence_proposal_ms | Evidence proposal is the largest Phase8 stage. |
| hotpotqa | legacy_512_10 | source_balanced_128 | COMPLETE | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1484.8971 | 120.5532 | 182.3595 | 24257.6635 | 24813.7115 | N/A | Baseline timing; Phase8 stages should be zero. |

## Stage Shares

| dataset | profile | variant | num_queries | entity_universe_ms_share_of_retrieval | sampling_ms_share_of_retrieval | kmedoids_ms_share_of_retrieval | seed_selection_ms_share_of_retrieval | refinement_ms_share_of_retrieval | evidence_proposal_ms_share_of_retrieval | feature_construction_ms_share_of_retrieval | selection_ms_share_of_retrieval | rendering_ms_share_of_retrieval | generation_ms_share_of_retrieval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 0.1446 | 0.0782 | 0.0777 | 0.0315 | 0.0123 | 0.1080 | 0.0246 | 0.0023 | 0.0000 | 0.0039 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 0.1161 | 0.0725 | 0.0720 | 0.0289 | 0.0604 | 0.1372 | 0.0282 | 0.0027 | 0.0000 | 0.0028 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0476 | 0.0044 | 0.0000 | 0.0142 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 0.1417 | 0.0839 | 0.0833 | 0.0331 | 0.0131 | 0.1109 | 0.0252 | 0.0026 | 0.0000 | 0.0036 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 0.1179 | 0.0672 | 0.0667 | 0.0275 | 0.0586 | 0.1410 | 0.0285 | 0.0031 | 0.0000 | 0.0031 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0501 | 0.0052 | 0.0000 | 0.0133 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 0.1263 | 0.0463 | 0.0460 | 0.0186 | 0.0073 | 0.1225 | 0.0274 | 0.0023 | 0.0000 | 0.0018 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0620 | 0.0046 | 0.0000 | 0.0072 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 0.1297 | 0.0494 | 0.0490 | 0.0196 | 0.0078 | 0.1247 | 0.0275 | 0.0024 | 0.0000 | 0.0020 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 1 | 0.0632 | 0.0222 | 0.0221 | 0.0100 | 0.0546 | 0.0805 | 0.0187 | 0.0016 | 0.0000 | 0.0108 |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0612 | 0.0050 | 0.0000 | 0.0075 |

## Explicit Answers

### 2wikimultihopqa/balanced_384_8/pamae_seed_k5

- Is entity_universe dominant? True
- Is kmedoids too expensive? True
- Is evidence_proposal too expensive? True
- Does refinement add meaningful overhead? False
- Optimize first if Phase8 is retained: entity_universe_ms

### 2wikimultihopqa/balanced_384_8/pamae_seed_k5_refine

- Is entity_universe dominant? False
- Is kmedoids too expensive? True
- Is evidence_proposal too expensive? True
- Does refinement add meaningful overhead? True
- Optimize first if Phase8 is retained: evidence_proposal_ms

### 2wikimultihopqa/legacy_512_10/pamae_seed_k5

- Is entity_universe dominant? True
- Is kmedoids too expensive? True
- Is evidence_proposal too expensive? True
- Does refinement add meaningful overhead? False
- Optimize first if Phase8 is retained: entity_universe_ms

### 2wikimultihopqa/legacy_512_10/pamae_seed_k5_refine

- Is entity_universe dominant? False
- Is kmedoids too expensive? True
- Is evidence_proposal too expensive? True
- Does refinement add meaningful overhead? True
- Optimize first if Phase8 is retained: evidence_proposal_ms

### hotpotqa/balanced_384_8/pamae_seed_k5

- Is entity_universe dominant? True
- Is kmedoids too expensive? True
- Is evidence_proposal too expensive? True
- Does refinement add meaningful overhead? False
- Optimize first if Phase8 is retained: entity_universe_ms

### hotpotqa/balanced_384_8/pamae_seed_k5_refine

- Is entity_universe dominant? True
- Is kmedoids too expensive? False
- Is evidence_proposal too expensive? False
- Does refinement add meaningful overhead? False
- Optimize first if Phase8 is retained: entity_universe_ms

### hotpotqa/legacy_512_10/pamae_seed_k5

- Is entity_universe dominant? True
- Is kmedoids too expensive? True
- Is evidence_proposal too expensive? True
- Does refinement add meaningful overhead? False
- Optimize first if Phase8 is retained: entity_universe_ms

### hotpotqa/legacy_512_10/pamae_seed_k5_refine

- Is entity_universe dominant? False
- Is kmedoids too expensive? True
- Is evidence_proposal too expensive? True
- Does refinement add meaningful overhead? True
- Optimize first if Phase8 is retained: evidence_proposal_ms

