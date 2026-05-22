# Phase8 PAMAE Failure Analysis Repaired

This report is variant-aware: source-balanced baselines cannot receive Phase8-specific failure labels.

| dataset | profile | variant | category | count | rate | num_queries |
| --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | UQ_ENTITY_MISS | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | SEED_SELECTION_MISS | 93 | 0.9300 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | REFINEMENT_DRIFT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | SEED_TO_EVIDENCE_MISS | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | CANDIDATE_CHAIN_INFEASIBLE | 7 | 0.0700 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | FINAL_SELECTION_FAILED | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | SELECTED_NOT_SUFFICIENT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | QA_PROMPT_FAILED | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | OTHER | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | UQ_ENTITY_MISS | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | SEED_SELECTION_MISS | 39 | 0.3900 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | REFINEMENT_DRIFT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | SEED_TO_EVIDENCE_MISS | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | CANDIDATE_CHAIN_INFEASIBLE | 53 | 0.5300 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | FINAL_SELECTION_FAILED | 7 | 0.0700 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | SELECTED_NOT_SUFFICIENT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | QA_PROMPT_FAILED | 1 | 0.0100 | 100 |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | OTHER | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | PHASE1_NOT_FOUND | 16 | 0.1600 | 100 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | FINAL_SELECTION_FAILED | 38 | 0.3800 | 100 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | SELECTED_NOT_SUFFICIENT | 35 | 0.3500 | 100 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | QA_PROMPT_FAILED_WITH_GOLD_CONTEXT | 8 | 0.0800 | 100 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | QA_PROMPT_FAILED | 0 | 0.0000 | 100 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | OTHER | 3 | 0.0300 | 100 |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | N/A_PHASE8_STAGE | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | UQ_ENTITY_MISS | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | SEED_SELECTION_MISS | 93 | 0.9300 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | REFINEMENT_DRIFT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | SEED_TO_EVIDENCE_MISS | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | CANDIDATE_CHAIN_INFEASIBLE | 7 | 0.0700 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | FINAL_SELECTION_FAILED | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | SELECTED_NOT_SUFFICIENT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | QA_PROMPT_FAILED | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | OTHER | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | UQ_ENTITY_MISS | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | SEED_SELECTION_MISS | 39 | 0.3900 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | REFINEMENT_DRIFT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | SEED_TO_EVIDENCE_MISS | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | CANDIDATE_CHAIN_INFEASIBLE | 53 | 0.5300 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | FINAL_SELECTION_FAILED | 7 | 0.0700 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | SELECTED_NOT_SUFFICIENT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | QA_PROMPT_FAILED | 1 | 0.0100 | 100 |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | OTHER | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | PHASE1_NOT_FOUND | 16 | 0.1600 | 100 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | FINAL_SELECTION_FAILED | 35 | 0.3500 | 100 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | SELECTED_NOT_SUFFICIENT | 36 | 0.3600 | 100 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | QA_PROMPT_FAILED_WITH_GOLD_CONTEXT | 10 | 0.1000 | 100 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | QA_PROMPT_FAILED | 0 | 0.0000 | 100 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | OTHER | 3 | 0.0300 | 100 |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | N/A_PHASE8_STAGE | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | UQ_ENTITY_MISS | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | SEED_SELECTION_MISS | 88 | 0.8800 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | REFINEMENT_DRIFT | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | SEED_TO_EVIDENCE_MISS | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | CANDIDATE_CHAIN_INFEASIBLE | 10 | 0.1000 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | FINAL_SELECTION_FAILED | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | SELECTED_NOT_SUFFICIENT | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | QA_PROMPT_FAILED | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | OTHER | 2 | 0.0200 | 100 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | UQ_ENTITY_MISS | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | SEED_SELECTION_MISS | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | REFINEMENT_DRIFT | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | SEED_TO_EVIDENCE_MISS | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | CANDIDATE_CHAIN_INFEASIBLE | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | FINAL_SELECTION_FAILED | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | SELECTED_NOT_SUFFICIENT | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | QA_PROMPT_FAILED | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | OTHER | 0 | 0.0000 | 0 |
| hotpotqa | balanced_384_8 | source_balanced_128 | PHASE1_NOT_FOUND | 8 | 0.0800 | 100 |
| hotpotqa | balanced_384_8 | source_balanced_128 | FINAL_SELECTION_FAILED | 52 | 0.5200 | 100 |
| hotpotqa | balanced_384_8 | source_balanced_128 | SELECTED_NOT_SUFFICIENT | 24 | 0.2400 | 100 |
| hotpotqa | balanced_384_8 | source_balanced_128 | QA_PROMPT_FAILED_WITH_GOLD_CONTEXT | 6 | 0.0600 | 100 |
| hotpotqa | balanced_384_8 | source_balanced_128 | QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | source_balanced_128 | QA_PROMPT_FAILED | 0 | 0.0000 | 100 |
| hotpotqa | balanced_384_8 | source_balanced_128 | OTHER | 10 | 0.1000 | 100 |
| hotpotqa | balanced_384_8 | source_balanced_128 | N/A_PHASE8_STAGE | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | UQ_ENTITY_MISS | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | SEED_SELECTION_MISS | 88 | 0.8800 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | REFINEMENT_DRIFT | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | SEED_TO_EVIDENCE_MISS | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | CANDIDATE_CHAIN_INFEASIBLE | 10 | 0.1000 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | FINAL_SELECTION_FAILED | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | SELECTED_NOT_SUFFICIENT | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | QA_PROMPT_FAILED | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | OTHER | 2 | 0.0200 | 100 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | UQ_ENTITY_MISS | 0 | 0.0000 | 1 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | SEED_SELECTION_MISS | 1 | 1.0000 | 1 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | REFINEMENT_DRIFT | 0 | 0.0000 | 1 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | SEED_TO_EVIDENCE_MISS | 0 | 0.0000 | 1 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | CANDIDATE_CHAIN_INFEASIBLE | 0 | 0.0000 | 1 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | FINAL_SELECTION_FAILED | 0 | 0.0000 | 1 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | SELECTED_NOT_SUFFICIENT | 0 | 0.0000 | 1 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | QA_PROMPT_FAILED | 0 | 0.0000 | 1 |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | OTHER | 0 | 0.0000 | 1 |
| hotpotqa | legacy_512_10 | source_balanced_128 | PHASE1_NOT_FOUND | 8 | 0.0800 | 100 |
| hotpotqa | legacy_512_10 | source_balanced_128 | FINAL_SELECTION_FAILED | 45 | 0.4500 | 100 |
| hotpotqa | legacy_512_10 | source_balanced_128 | SELECTED_NOT_SUFFICIENT | 27 | 0.2700 | 100 |
| hotpotqa | legacy_512_10 | source_balanced_128 | QA_PROMPT_FAILED_WITH_GOLD_CONTEXT | 7 | 0.0700 | 100 |
| hotpotqa | legacy_512_10 | source_balanced_128 | QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | source_balanced_128 | QA_PROMPT_FAILED | 0 | 0.0000 | 100 |
| hotpotqa | legacy_512_10 | source_balanced_128 | OTHER | 13 | 0.1300 | 100 |
| hotpotqa | legacy_512_10 | source_balanced_128 | N/A_PHASE8_STAGE | 0 | 0.0000 | 100 |

## Checks

- Source-balanced rows use `PHASE1_NOT_FOUND`, final-selection, selected-context, QA, or `OTHER` labels only.
- Phase8-specific labels are reserved for pamae variants.
- Candidate/selected/rendered recall values are normalized before attribution.

