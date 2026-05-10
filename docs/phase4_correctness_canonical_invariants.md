# Phase 4: Correctness and Canonical-Invariant Refactoring

## Purpose
Phase 4 focuses on trustworthiness and reviewer-facing clarity:
1. preserve validated canonical/champion behavior
2. remove correctness risks
3. make canonical invariants explicit in code, not only in documents

## Three Refactoring Principles
1. Performance preservation first.
2. Dataset generality second.
3. Avoid heuristic-looking design.

## What Changed
1. PPR graph-structure cache safety:
   - replaced id-only cache behavior with weakref-safe object identity caching
   - fallback cache validates `weakref(g) is current_graph` before reuse
2. Canonical pruning invariant enforcement:
   - in `canonical_copy_span`, all `CANONICAL_PRUNED_WEIGHT_FIELDS` are forced to `0.0`
   - overridden fields are recorded in objective diagnostics
3. Config safety:
   - unknown config keys now emit warnings by default
   - strict mode can raise `ValueError`
4. Canonical path naming audit:
   - warn (or strict-fail) when config path includes `canonical` but objective mode is `baseline`
5. Temporary config mutation safety:
   - local `ppr_parallel_workers` override now protected by `try/finally`

## What Did Not Change
1. No new scoring weights were added.
2. No retrieval/rendering formula redesign was introduced.
3. No prompt/template/generation-interface rewrite was introduced.
4. Dataset behavior mapping remains compatibility-preserving.

## Canonical Core vs Dataset Guard
Canonical retrieval core is dataset-agnostic:
1. query-conditioned proposal graph
2. stochastic multi-run PPR
3. compact run scoring
4. bounded local refinement
5. copy-span rendering/generation interface

Dataset-specific guard/adaptation layer:
1. Hotpot guarded answer-preserve compatibility mode
2. dataset-specific supporting-fact/evaluator matching conventions

This separation is explicit in `canonical_effective_reference_mode(...)` and
guard helper logic, while preserving validated behavior.

## Seed Scoring Boundary
1. `seed_score_*`:
   - candidate-level seed utility terms
2. `seed_objective_*`:
   - seed-set objective terms for compact/coverage-preserving set selection

Phase 4 keeps seed-set objective terms intact.
Only pruned candidate-level optional terms are enforced out of canonical scoring.

## Expected Parity Checks
1. unit-test parity for retrieval/canonical/pipeline invariants
2. small HotpotQA/2Wiki smoke parity
3. if deviations occur after cache fix, report stale-cache contamination risk as a correctness note
