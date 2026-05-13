# Phase 5A: Score Simplification Diagnostics

## 1. Goal
Phase 5A is a diagnostic-only refactoring phase.

Primary question:
- Which score components appear redundant, and which can be safely ablated later without harming retrieval/QA quality?

## 2. Why Diagnostic-Only
This phase does **not** simplify canonical defaults.

Rules:
1. default canonical/champion path must remain unchanged
2. all new ablation flags default to `False`
3. no new heuristic weights are introduced
4. no score component is removed from default behavior in this phase

## 3. Performance Preservation Principle
When all Phase5A flags are off:
- retrieval decisions should remain unchanged
- selected context ordering should remain unchanged
- output-level behavior should match existing canonical behavior

Any behavior-changing logic is enabled only by explicit ablation flags.

## 4. Why Redundancy Is Suspected
Semantic/anchor/bridge signals are used across multiple stages.
Potential overlap exists between:
1. pair shortlist semantic term vs upstream semantic filtering
2. pair bridge term vs structural path signals
3. final embedding rerank vs retrieval score ordering

Phase 5A adds diagnostics/ablation switches to measure overlap empirically.

## 5. Phase5A Scope
Implemented diagnostic targets:
1. pair semantic ablation (`ablation_no_pair_semantic`)
2. pair bridge ablation (`ablation_no_pair_bridge`)
3. final text rerank ablation (`ablation_no_final_text_rerank`)
4. compact score component trace (`score_component_trace_enabled`)
5. baseline-vs-variant decision overlap script

## 6. Deferred Items (Phase 5B/5C)
Deferred candidates:
1. local refinement semantic ablation
2. top1 correction ablation
3. run pre-score semantic ablation
4. run semantic coverage ablation
5. anchor alignment ablation in pair scoring

These are placeholders only in Phase 5A and are not wired as default behavior changes.

## 7. Default Behavior Invariance
Default preserved components (no ablation flags):
1. semantic proposal lookup
2. stochastic PPR aggregate behavior
3. greedy seed-set objective
4. structural connectivity and bounded local refinement
5. base retrieval score pipeline
6. final rendering and QA interface

## 8. How to Run Ablations
Example (diagnostic-only):

```bash
python -m effirag.run_rag \
  --config configs/canonical/rag_canonical_copy_span_phase4.yaml \
  --dataset hotpotqa \
  --limit 100 \
  --output-dir outputs/phase5a_diag/no_pair_semantic \
  --ablation-no-pair-semantic true \
  --score-component-trace-enabled true
```

Default no-flag control:

```bash
python -m effirag.run_rag \
  --config configs/canonical/rag_canonical_copy_span_phase4.yaml \
  --dataset hotpotqa \
  --limit 100 \
  --output-dir outputs/phase5a_diag/no_flag
```

Decision overlap comparison:

```bash
python scripts/compare_phase5_score_ablation.py \
  --baseline <baseline_rag_query_results.jsonl> \
  --variant <variant_rag_query_results.jsonl> \
  --output phase5_score_ablation_compare.json \
  --markdown-output phase5_score_ablation_compare.md
```

## 9. How to Interpret Decision Overlap
Focus on:
1. selected_text_top1_match_rate
2. selected_text_topk_jaccard_mean
3. seed overlap and pair overlap
4. EM/F1 deltas
5. retrieval latency delta
6. context token delta

Interpretation:
- high overlap + near-zero EM/F1 delta: candidate for later removal
- dataset-specific drop: keep as optional or adapter behavior
- consistent multi-dataset drop: keep component in canonical path

## 10. Removal Criteria for Later Phase
A component should be considered removable only after larger-scale validation (e.g., n=1000) confirms:
1. no consistent EM/F1 degradation on HotpotQA/2Wiki
2. MuSiQue/PopQA checks completed
3. stable or improved retrieval latency
4. stable selected evidence overlap
5. no new instability in ranking/tie behavior
