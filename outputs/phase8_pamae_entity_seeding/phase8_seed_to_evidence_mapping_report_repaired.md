# Phase8 Seed-to-Evidence Mapping Report Repaired

This repaired report uses normalized Phase7 candidate/selected ids for candidate and selected recall. Source-balanced rows are included for comparison, but Phase8-specific interpretations apply only to pamae variants.

| dataset | profile | variant | num_queries | num_seed_atoms_mean | num_seed_atoms_p50 | num_seed_atoms_p95 | num_seed_carriers_mean | num_seed_carriers_p50 | num_seed_carriers_p95 | num_anchor_seed_path_atoms_mean | num_seed_seed_path_atoms_mean | num_same_title_support_atoms_mean | num_same_carrier_support_atoms_mean | num_final_evidence_candidates_mean | num_final_evidence_candidates_p50 | num_final_evidence_candidates_p95 | candidate_gold_partial_rate_eval_only | candidate_gold_full_rate_eval_only | candidate_gold_recall_mean_eval_only | candidate_oracle_F1_mean_eval_only | chain_unit_oracle_feasible_mean_eval_only | selected_pamae_source_rate | interpretation | candidate_gold_recall_eval_only | selected_gold_recall_eval_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 9.7600 | 10.0000 | 13.0000 | 14.3200 | 14.0000 | 21.0000 | 3.3800 | 0.0000 | 16.1600 | 5.0300 | 48.6500 | 47.5000 | 76.1000 | 0.0700 | 0.0000 | 0.0325 | 0.0000 | 0.0000 | 1.0000 | Phase8 emits candidates, but gold/equivalent coverage is very low; seed-to-evidence mapping is weak. | 0.0325 | 0.0225 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 19.7400 | 20.0000 | 20.0000 | 19.4300 | 20.0000 | 25.0000 | 10.5500 | 0.1900 | 43.4300 | 14.5200 | 107.8600 | 108.0000 | 134.0000 | 0.6100 | 0.1500 | 0.3625 | 0.0000 | 0.0800 | 1.0000 | Seed-to-evidence mapping produces selected Phase8 candidates. | 0.3625 | 0.2375 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8400 | 0.2700 | 0.5475 | 0.0000 | 0.0600 | 0.0000 | Source-balanced candidates have nonzero candidate recall; prior zero summary was a diagnostic extraction issue. | 0.5475 | 0.3425 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 9.7600 | 10.0000 | 13.0000 | 14.3200 | 14.0000 | 21.0000 | 3.3800 | 0.0000 | 16.1600 | 5.0300 | 48.6500 | 47.5000 | 76.1000 | 0.0700 | 0.0000 | 0.0325 | 0.0000 | 0.0000 | 1.0000 | Phase8 emits candidates, but gold/equivalent coverage is very low; seed-to-evidence mapping is weak. | 0.0325 | 0.0225 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 19.7400 | 20.0000 | 20.0000 | 19.4300 | 20.0000 | 25.0000 | 10.5500 | 0.1900 | 43.4300 | 14.5200 | 107.8600 | 108.0000 | 134.0000 | 0.6100 | 0.1500 | 0.3625 | 0.0000 | 0.0800 | 1.0000 | Seed-to-evidence mapping produces selected Phase8 candidates. | 0.3625 | 0.2425 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8400 | 0.2700 | 0.5475 | 0.0000 | 0.0600 | 0.0000 | Source-balanced candidates have nonzero candidate recall; prior zero summary was a diagnostic extraction issue. | 0.5475 | 0.3725 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 11.0300 | 11.0000 | 15.0500 | 14.5700 | 14.0000 | 20.0000 | 3.5800 | 0.0000 | 17.3700 | 3.0500 | 49.6000 | 48.0000 | 73.1000 | 0.1200 | 0.0200 | 0.0665 | 0.0000 | 0.0200 | 1.0000 | Phase8 emits candidates, but gold/equivalent coverage is very low; seed-to-evidence mapping is weak. | 0.0665 | 0.0600 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Seed-to-index mapping is broken: seed atoms/carriers are near zero. | 0.0000 | 0.0000 |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9200 | 0.5600 | 0.7433 | 0.0000 | 0.2500 | 0.0000 | Source-balanced candidates have nonzero candidate recall; prior zero summary was a diagnostic extraction issue. | 0.7433 | 0.4523 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 11.0300 | 11.0000 | 15.0500 | 14.5700 | 14.0000 | 20.0000 | 3.5800 | 0.0000 | 17.3700 | 3.0500 | 49.6000 | 48.0000 | 73.1000 | 0.1200 | 0.0200 | 0.0665 | 0.0000 | 0.0200 | 1.0000 | Phase8 emits candidates, but gold/equivalent coverage is very low; seed-to-evidence mapping is weak. | 0.0665 | 0.0600 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 1 | 20.0000 | 20.0000 | 20.0000 | 19.0000 | 19.0000 | 19.0000 | 17.0000 | 0.0000 | 44.0000 | 7.0000 | 107.0000 | 107.0000 | 107.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | Phase8 emits candidates, but gold/equivalent coverage is very low; seed-to-evidence mapping is weak. | 0.0000 | 0.0000 |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9200 | 0.5600 | 0.7433 | 0.0000 | 0.2500 | 0.0000 | Source-balanced candidates have nonzero candidate recall; prior zero summary was a diagnostic extraction issue. | 0.7433 | 0.5040 |

## Explicit Answers

- Is seed-to-evidence mapping actually failing? For raw `pamae_seed_k5`, yes: normalized candidate recall is very low despite nonzero Phase8 candidate counts.
- Or was SEED_TO_EVIDENCE_MISS inflated by broken diagnostics? The original 100/100 attribution was inflated by broken Phase8 trace logging, but repaired recall still shows raw k5 is weak.
- Do Phase8 evidence candidates reach Phase7 selection? Yes; selected evidence carries Phase8 atom source tags in pamae variants.
- Does refinement improve seed-to-evidence mapping? On completed 2Wiki refine runs, candidate recall improves materially; Hotpot refine is partial/missing and should not be interpreted.

## Representative Cases

### u_q_contains_gold_but_seeds_miss

No cases found.

### relevant_seeds_but_evidence_candidates_missing

No cases found.

### seeds_exist_but_low_final_candidates

No cases found.

### final_candidates_exist_but_selected_pamae_zero

No cases found.

### seed_evidence_hit_but_selected_gold_false

No cases found.

### pamae_candidates_exist_but_all_filtered

No cases found.

### refinement_improves_seed_evidence_hit

No cases found.

### refinement_causes_drift

No cases found.

