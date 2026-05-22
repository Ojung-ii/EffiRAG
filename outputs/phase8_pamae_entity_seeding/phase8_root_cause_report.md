# Phase8 Root-Cause Report

## Verdict

Current pamae_seed_k5 implementation:
  reject as mainline.

PAMAE-inspired idea:
  inconclusive.

Most likely root cause:
  implementation/integration failure in seed-to-evidence mapping, with a separate diagnostic logging/id-normalization bug in the Phase8 summary trace.

Next required fix:
  repair diagnostics first, then repair seed-to-evidence mapping. Do not run more full experiments until candidate oracle metrics and Phase8 candidate transfer are trustworthy.

## Run-Level Verdicts

| dataset | profile | variant | num_queries | partial_run | u_q_miss | seed_selection_miss | refinement_drift | seed_to_evidence_miss | diagnostic_id_normalization_broken | phase8_replaced_source_balanced | generated_candidates_not_selected | largest_time_bottleneck | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | False | True | True | False | True | True | True | False | entity_universe_ms | Fix seed-to-evidence mapping before any more full experiments |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | False | True | True | False | False | True | True | False | evidence_proposal_ms | Fix U_q construction and gold-entity diagnostics |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | False | False | False | False | False | True | False | False | entity_universe_ms | Fix diagnostic id normalization |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | False | True | True | False | True | True | True | False | entity_universe_ms | Fix seed-to-evidence mapping before any more full experiments |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | False | True | True | False | False | True | True | False | evidence_proposal_ms | Fix U_q construction and gold-entity diagnostics |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | False | False | False | False | False | True | False | False | entity_universe_ms | Fix diagnostic id normalization |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | False | True | True | False | True | True | True | False | entity_universe_ms | Fix seed-to-evidence mapping before any more full experiments |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0 | True | False | False | False | False | False | True | False | entity_universe_ms | Partial run; do not interpret as evidence |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | False | False | False | False | False | True | False | False | entity_universe_ms | Fix diagnostic id normalization |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | False | True | True | False | True | True | True | False | entity_universe_ms | Fix seed-to-evidence mapping before any more full experiments |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 1 | True | False | False | False | False | False | True | False | evidence_proposal_ms | Partial run; do not interpret as evidence |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | False | False | False | False | False | True | False | False | entity_universe_ms | Fix diagnostic id normalization |

## Decision Table

| Condition | Evidence | Recommended action |
| --- | --- | --- |
| 2wikimultihopqa/balanced_384_8 Phase8 candidates miss gold evidence | Final evidence candidates exist, but candidate_gold_recall remains near zero. | Fix seed-to-evidence mapping; inspect entity-id to sentence/carrier joins. |
| 2wikimultihopqa/balanced_384_8 U_q gold hit near zero | Large entity universe with zero eval-only gold entity hit. | Check U_q construction and gold-entity diagnostic mapping. |
| 2wikimultihopqa/balanced_384_8 runtime bottleneck | Largest Phase8 stage is entity_universe_ms. | Optimize entity_universe_ms after correctness is repaired. |
| 2wikimultihopqa/balanced_384_8 U_q gold hit near zero | Large entity universe with zero eval-only gold entity hit. | Check U_q construction and gold-entity diagnostic mapping. |
| 2wikimultihopqa/balanced_384_8 runtime bottleneck | Largest Phase8 stage is evidence_proposal_ms. | Optimize evidence_proposal_ms after correctness is repaired. |
| 2wikimultihopqa/balanced_384_8 source-balanced oracle appears zero in phase8 trace | Phase7 candidate recall is nonzero but phase8_query_trace logs zero. | Fix diagnostic id normalization/logging; do not treat source-balanced oracle as zero. |
| 2wikimultihopqa/legacy_512_10 Phase8 candidates miss gold evidence | Final evidence candidates exist, but candidate_gold_recall remains near zero. | Fix seed-to-evidence mapping; inspect entity-id to sentence/carrier joins. |
| 2wikimultihopqa/legacy_512_10 U_q gold hit near zero | Large entity universe with zero eval-only gold entity hit. | Check U_q construction and gold-entity diagnostic mapping. |
| 2wikimultihopqa/legacy_512_10 runtime bottleneck | Largest Phase8 stage is entity_universe_ms. | Optimize entity_universe_ms after correctness is repaired. |
| 2wikimultihopqa/legacy_512_10 U_q gold hit near zero | Large entity universe with zero eval-only gold entity hit. | Check U_q construction and gold-entity diagnostic mapping. |
| 2wikimultihopqa/legacy_512_10 runtime bottleneck | Largest Phase8 stage is evidence_proposal_ms. | Optimize evidence_proposal_ms after correctness is repaired. |
| 2wikimultihopqa/legacy_512_10 source-balanced oracle appears zero in phase8 trace | Phase7 candidate recall is nonzero but phase8_query_trace logs zero. | Fix diagnostic id normalization/logging; do not treat source-balanced oracle as zero. |
| hotpotqa/balanced_384_8 Phase8 candidates miss gold evidence | Final evidence candidates exist, but candidate_gold_recall remains near zero. | Fix seed-to-evidence mapping; inspect entity-id to sentence/carrier joins. |
| hotpotqa/balanced_384_8 U_q gold hit near zero | Large entity universe with zero eval-only gold entity hit. | Check U_q construction and gold-entity diagnostic mapping. |
| hotpotqa/balanced_384_8 runtime bottleneck | Largest Phase8 stage is entity_universe_ms. | Optimize entity_universe_ms after correctness is repaired. |
| hotpotqa/balanced_384_8/pamae_seed_k5_refine partial run | Only 0 query rows are available. | Do not use this run for method conclusions. |
| hotpotqa/balanced_384_8 source-balanced oracle appears zero in phase8 trace | Phase7 candidate recall is nonzero but phase8_query_trace logs zero. | Fix diagnostic id normalization/logging; do not treat source-balanced oracle as zero. |
| hotpotqa/legacy_512_10 Phase8 candidates miss gold evidence | Final evidence candidates exist, but candidate_gold_recall remains near zero. | Fix seed-to-evidence mapping; inspect entity-id to sentence/carrier joins. |
| hotpotqa/legacy_512_10 U_q gold hit near zero | Large entity universe with zero eval-only gold entity hit. | Check U_q construction and gold-entity diagnostic mapping. |
| hotpotqa/legacy_512_10 runtime bottleneck | Largest Phase8 stage is entity_universe_ms. | Optimize entity_universe_ms after correctness is repaired. |
| hotpotqa/legacy_512_10/pamae_seed_k5_refine partial run | Only 1 query rows are available. | Do not use this run for method conclusions. |
| hotpotqa/legacy_512_10 source-balanced oracle appears zero in phase8 trace | Phase7 candidate recall is nonzero but phase8_query_trace logs zero. | Fix diagnostic id normalization/logging; do not treat source-balanced oracle as zero. |

## Explicit Answers

- Did Phase8 fail because U_q missed relevant entities? yes.
- Did Phase8 fail because medoid seed selection missed relevant entities? yes.
- Did Phase8 fail because refinement caused drift? no.
- Did Phase8 fail because seed-to-evidence mapping failed? yes.
- Did Phase8 fail because diagnostics/id normalization is broken? yes.
- Did Phase8 fail because PAMAE proposal replaced source-balanced candidates? yes.
- Did Phase8 fail because generated candidates were not selected? no.
- What is the largest time bottleneck? entity_universe_ms, evidence_proposal_ms.
- Should Phase8 be discarded, repaired, augmented, or postponed? Reject current PAMAE-only implementation as mainline; repair diagnostics and seed-to-evidence mapping, and consider source-balanced plus PAMAE augmentation only after that.

## Failure Type Distinction

- Implementation failure: seed-to-evidence mapping emits many candidates but they do not cover gold/equivalent evidence.
- Diagnostic failure: source-balanced candidate oracle metrics were written as zero in `phase8_query_trace` despite nonzero Phase7 candidate recall.
- Method failure: not proven; PAMAE-only replacement is currently not viable, but the idea remains inconclusive because diagnostics and mapping are faulty.
- Integration failure: Phase8 replaces the Phase7 source-balanced proposal path in pamae variants, so useful baseline candidates are not preserved.
- Timing bottleneck: entity universe construction and evidence proposal dominate Phase8 added cost, with sampling/k-medoids also material.

