# Phase8 Root-Cause Report Repaired

## Verdict

Current pamae_seed_k5 implementation:
  reject as replacement

PAMAE-inspired idea:
  still viable as augmentation

Most likely root cause:
  raw k5 replacement loses source-balanced coverage; U_q/seed gold diagnostics are weak, and seed-to-evidence candidate recall is low before refinement

Next required action:
  design source_balanced_128 plus pamae_seed_k5_refine augmentation after repairing diagnostics; do not run PAMAE-only replacement again

## Run-Level Diagnosis

| dataset | profile | variant | status | num_queries | normalized_candidate_gold_recall | normalized_selected_gold_recall | F1 | SF-R | retrieval_ms | u_q_failure | seed_selection_failure | refinement_drift | seed_to_evidence_failure | candidate_transfer_selection_failure | pamae_seed_k5_refine_useful_signal | largest_time_bottleneck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | COMPLETE | 100 | 0.0325 | 0.0225 | 0.0529 | 0.0225 | 21118.1878 | True | True | False | True | False | False | entity_universe_ms |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | COMPLETE | 100 | 0.3625 | 0.2375 | 0.0751 | 0.2375 | 25024.7380 | True | False | False | False | True | True | evidence_proposal_ms |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | COMPLETE | 100 | 0.5475 | 0.3425 | 0.1317 | 0.3425 | 15344.7971 | False | False | False | False | False | False | N/A |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | COMPLETE | 100 | 0.0325 | 0.0225 | 0.0551 | 0.0225 | 20713.1164 | True | True | False | True | False | False | entity_universe_ms |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | COMPLETE | 100 | 0.3625 | 0.2425 | 0.0854 | 0.2425 | 25158.7831 | True | False | False | False | True | True | evidence_proposal_ms |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | COMPLETE | 100 | 0.5475 | 0.3725 | 0.0950 | 0.3725 | 15469.2437 | False | False | False | False | False | False | N/A |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | COMPLETE | 100 | 0.0665 | 0.0600 | 0.0608 | 0.0600 | 35980.1614 | True | True | False | True | False | False | entity_universe_ms |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | MISSING | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | False | False | False | False | False | entity_universe_ms |
| hotpotqa | balanced_384_8 | source_balanced_128 | COMPLETE | 100 | 0.7433 | 0.4523 | 0.2942 | 0.4523 | 24289.6216 | False | False | False | False | False | False | N/A |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | COMPLETE | 100 | 0.0665 | 0.0600 | 0.0708 | 0.0600 | 36169.6391 | True | True | False | True | False | False | entity_universe_ms |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | PARTIAL | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 72163.1359 | False | False | False | False | False | False | evidence_proposal_ms |
| hotpotqa | legacy_512_10 | source_balanced_128 | COMPLETE | 100 | 0.7433 | 0.5040 | 0.3172 | 0.5040 | 24257.6635 | False | False | False | False | False | False | N/A |

## Questions

| Question | Answer |
| --- | --- |
| Did Phase8 actually fail at U_q construction? | Yes for pamae variants by eval-only gold-entity hit, though gold-entity diagnostics may be incomplete. |
| Did Phase8 actually fail at medoid seed selection? | Yes for raw k5: seed gold/evidence hit is near zero and candidate recall is very low. |
| Did Phase8 actually fail at refinement? | No clear drift; completed 2Wiki refine runs show useful candidate recall signal. |
| Did Phase8 actually fail at seed-to-evidence mapping? | Raw k5 is weak, but previous 100/100 SEED_TO_EVIDENCE_MISS was inflated by broken diagnostics. |
| Were previous candidate/oracle diagnostics broken? | Yes. Source-balanced candidate recall is nonzero after normalization. |
| Was source-balanced incorrectly attributed with Phase8 failures? | Yes; repaired taxonomy prevents source-balanced from receiving Phase8-specific labels. |
| Is pamae_seed_k5 genuinely weak after repaired diagnostics? | Yes. Hotpot normalized candidate recall is about 0.0665 and 2Wiki raw k5 is similarly low. |
| Does pamae_seed_k5_refine show useful signal, especially on 2Wiki? | Yes. Completed 2Wiki refine runs show much higher normalized candidate recall than raw k5. |
| Recommended next experiment | source_balanced_128 union pamae_seed_k5_refine augmentation, not PAMAE-only replacement. |

