# Canonical Mainline Pruning (Phase-3.5)

## Stage-0 Reference Freeze
- phase-3 result path: `outputs/canonical_pipeline_simplification_phase3/20260508_193149`
- pruning reference profile: `canonical_copy_span`
- target datasets (this phase): `hotpotqa`, `2wikimultihopqa`
- excluded datasets (deferred): `musique`, `popqa`

## Baseline Facts (Before Pruning)
1. `canonical_copy_span` is the accepted phase-3 canonical path.
2. Stage-B parity passed on HotpotQA / 2WikiMultiHopQA.
3. Stage-C ablations on HotpotQA / 2WikiMultiHopQA reported delta `0.0` for:
   - Recall@1/5/10
   - EM/F1
   - avg_context_tokens
   - retrieval_ms
   - equivalent_evidence_coverage
4. `canonical_no_dataset_guard` also showed numeric parity, but guard removal is out-of-scope for phase-3.5.

## Motivation
Phase-3 introduced canonical/legacy boundaries.  
Phase-3.5 prunes no-op optional heuristic branches from the canonical mainline so the implementation itself becomes lighter.

## Preserved Core
The following are preserved:
1. proposal graph construction and proposal union
2. stochastic multi-run PPR
3. bounded local refinement
4. copy-span prompt/interface
5. dataset guard behavior
6. bridge candidate induction
7. redundancy penalty
8. answer support pinning

## Pruned From Canonical Path
1. seed optional score terms:
   - `seed_score_bridge_weight`
   - `seed_score_chunk_grounding_weight`
2. run optional score terms:
   - `run_score_entity_chunk_grounding_weight`
   - `run_score_anchor_dispersion_penalty`
   - `run_score_pair_coverage_weight`
   - `run_score_bridge_completeness_weight`
3. render optional terms (canonical path):
   - `zeta_query`
   - `xi_locality`
   - legacy-only bridge boost / role-balance side heuristics

## Seed Scoring Boundary Clarification
1. `seed_score_*` terms:
   - candidate-level scoring terms used to rank individual seed candidates
2. `seed_objective_*` terms:
   - seed-set objective terms used to select a compact and coverage-preserving seed set

Phase-3.5 pruning removed optional candidate-level boosts from canonical scoring.
It did **not** remove seed-set objective terms, because those terms belong to the
set-selection objective rather than per-candidate heuristic boosts.

## Not Pruned In This Phase
1. dataset-specific guarded answer preserve
2. bridge candidate induction
3. redundancy control
4. connector role signal
5. answer support pinning
6. copy-span instruction interface

## Evaluation Policy
1. parity first: before-pruning canonical vs after-pruning canonical on HotpotQA/2Wiki
2. pruning acceptance only when required parity metrics and hashes are preserved
3. MuSiQue/PopQA recovery is deferred to:
   - **Phase-4: Cross-Dataset Generalization and Performance Recovery**
