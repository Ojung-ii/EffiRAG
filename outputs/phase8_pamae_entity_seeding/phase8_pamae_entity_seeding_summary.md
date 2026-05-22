# Phase8 PAMAE Entity Seeding Summary

| dataset | profile | variant | num_queries | em | f1 | sf_recall | sf_precision | phase1_full_gold_coverage | candidate_gold_recall | candidate_oracle_F1 | chain_unit_oracle_feasible | retrieval_ms | total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 0.0500 | 0.0529 | 0.0225 | 0.0063 | 0.0000 | 0.0325 | 0.0000 | 0.0000 | 21118.1878 | 21380.5359 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 0.0700 | 0.0751 | 0.2375 | 0.0688 | 0.1500 | 0.3625 | 0.0000 | 0.0800 | 25024.7380 | 25302.7599 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 0.1000 | 0.1099 | 0.3425 | 0.1037 | 0.2700 | 0.5475 | 0.0000 | 0.0600 | 15344.7971 | 15755.9385 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 0.0500 | 0.0551 | 0.0225 | 0.0050 | 0.0000 | 0.0325 | 0.0000 | 0.0000 | 20713.1164 | 20964.9706 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 0.0800 | 0.0854 | 0.2425 | 0.0560 | 0.1500 | 0.3625 | 0.0000 | 0.0800 | 25158.7831 | 25449.7509 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 0.0700 | 0.0832 | 0.3725 | 0.0890 | 0.2700 | 0.5475 | 0.0000 | 0.0600 | 15469.2437 | 15880.7765 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 0.0400 | 0.0608 | 0.0600 | 0.0187 | 0.0200 | 0.0665 | 0.0000 | 0.0200 | 35980.1614 | 36360.1573 |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 0.2000 | 0.2587 | 0.4523 | 0.1425 | 0.5600 | 0.7433 | 0.0000 | 0.2500 | 24289.6216 | 24836.1877 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 0.0500 | 0.0708 | 0.0600 | 0.0150 | 0.0200 | 0.0665 | 0.0000 | 0.0200 | 36169.6391 | 36554.3341 |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 0.2300 | 0.2929 | 0.5040 | 0.1260 | 0.5600 | 0.7433 | 0.0000 | 0.2500 | 24257.6635 | 24813.7115 |

## Timing

| dataset | profile | variant | entity_universe_ms | sampling_ms | kmedoids_ms | seed_selection_ms | refinement_ms | evidence_proposal_ms | retrieval_ms | generation_ms | total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 3053.1479 | 1651.4936 | 1639.8560 | 664.6839 | 259.8904 | 2281.6021 | 21118.1878 | 82.3221 | 21380.5359 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 2906.4565 | 1814.2138 | 1801.5453 | 723.9981 | 1511.9678 | 3434.5648 | 25024.7380 | 68.8501 | 25302.7599 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15344.7971 | 217.1810 | 15755.9385 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 2935.6602 | 1738.4477 | 1725.7087 | 685.8017 | 271.0581 | 2296.1980 | 20713.1164 | 73.7303 | 20964.9706 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 2966.1339 | 1689.5240 | 1676.9227 | 690.9465 | 1475.0237 | 3548.3968 | 25158.7831 | 77.1176 | 25449.7509 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15469.2437 | 206.4730 | 15880.7765 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 4544.5553 | 1665.4302 | 1653.4578 | 668.6663 | 261.2972 | 4408.0646 | 35980.1614 | 65.8371 | 36360.1573 |
| hotpotqa | balanced_384_8 | source_balanced_128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24289.6216 | 175.5726 | 24836.1877 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 4690.6177 | 1786.1395 | 1773.1556 | 709.9138 | 281.0666 | 4511.9614 | 36169.6391 | 73.0515 | 36554.3341 |
| hotpotqa | legacy_512_10 | source_balanced_128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24257.6635 | 182.3595 | 24813.7115 |
