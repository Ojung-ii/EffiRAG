# Phase8 PAMAE Entity Seeding Report

## Questions

1. Did PAMAE-inspired seeding improve Phase1 full-chain coverage?
2. Did it improve candidate oracle feasibility?
3. Did one-step refinement improve over raw seeding?
4. Did final F1 improve?
5. Did SF-P collapse?
6. Did retrieval latency remain acceptable?
7. Were medoid seeds query-relevant?
8. Did medoid-linked evidence recover gold/equivalent evidence?
9. Should PAMAE-inspired proposal replace Phase7 source-balanced proposal?
10. If not, is the bottleneck U_q construction, medoid seeding, refinement, evidence mapping, or final selection?

## Run Table

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

## Notes

This report is generated from completed run artifacts. Interpret missing variants as not yet run.
