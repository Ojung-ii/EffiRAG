# Copy-Span Mainline Profile

## Mainline Definition
`copy-span (champion_reconfirm)` is the frozen mainline profile for current SOTA reporting.

- profile: `copy-span`
- canonical variant: `champion_reconfirm`
- lock round root: `outputs/light_separator_lockin_round/20260428_004715`

Main reporting must remain aligned to `copy-span (champion_reconfirm)`.
`light_separator_render` remains auxiliary/analysis-only and is not the mainline profile.

## Locked Values
Phase-1 keeps the active score contract unchanged.

Locked retrieval/seed weights:

- `run_score_semantic_weight=0.30`
- `run_score_anchor_weight=0.20`
- `run_score_structure_weight=0.25`
- `run_score_bridge_weight=0.15`
- `run_score_redundancy_weight=0.10`
- `seed_score_semantic_weight=0.35`
- `seed_score_graph_weight=0.45`
- `seed_score_anchor_weight=0.20`
- `embedding_weight=0.35`

Locked render weights:

- `alpha=1.0`
- `beta=0.35`
- `gamma_main=0.45`
- `delta_support=0.20`
- `eta_connector=0.20`
- `zeta_query=0.12`
- `xi_locality=0.08`
- `lambda_redundancy=0.25`

Locked budget/shape:

- `max_context_sentences=8`
- `max_total_sentences=8`
- `top_corridors=3`
- `delivery_mode=sentence_compressed`
- `render_mode=corridor_aware_flat`
- `order_strategy=score`

## Dataset-Specific Behavior
Dataset-level locks preserved by the mainline contract:

- `hotpotqa`: `retrieval_objective_mode=p3_answer_preserve_guarded_hotpot`, `answer_support_pinning_enabled=true`, `final_top_slice_reorder_enabled=false`, `precomputed_retrieval_strict=true`
- `2wikimultihopqa`: `retrieval_objective_mode=baseline`, `answer_support_pinning_enabled=false`, `final_top_slice_reorder_enabled=true`, `precomputed_retrieval_strict=true`
- `musique`: `retrieval_objective_mode=baseline`, `answer_support_pinning_enabled=false`, `final_top_slice_reorder_enabled=false`, `precomputed_retrieval_strict=false`
- `popqa`: `retrieval_objective_mode=baseline`, `answer_support_pinning_enabled=false`, `final_top_slice_reorder_enabled=false`, `precomputed_retrieval_strict=false`

## Disabled Experimental Modules
These modules are not part of the mainline result and remain disabled:

- `top1_correction_enabled=false`
- `run_light_rerank_enabled=false`
- `corridor_compact_shaping_enabled=false`
- `corridor_bridge_purity_shaping_enabled=false`
- `chunk_grounding_enabled=false`
- `hybrid_anchor_recall_enabled=false`
- `role_aware_chunk_scoring_enabled=false`
- `coverage_selection_enabled=false`

## Deprecated / Legacy Knobs
Phase-1 preserves backward compatibility for legacy knobs and does not delete them:

- `semantic_chunk_support_expand_per_entity`
- `semantic_chunk_support_expand_total`
- `proposal_adaptive_budget`
- `proposal_chunk_rank_cap`
- `proposal_reserve_graph_mix_topn`
- `proposal_sparse_subgraph_build`
- `proposal_subgraph_anchor_neighbor_cap`
- `proposal_subgraph_support_cap`
- `proposal_subgraph_connector_cap`

These fields are retained for compatibility and audit visibility only. Their presence should be treated as a warning in profile audit output, not a lock failure.

## Invariance Validation
Phase-1 invariance checks:

1. `PYTHONPATH=. python -m py_compile effirag/*.py`
2. `PYTHONPATH=. pytest -q`
3. `PYTHONPATH=. python scripts/audit_copy_span_profile.py --config-root outputs/light_separator_lockin_round/20260428_004715/configs/s0/champion_reconfirm --out-dir outputs/refactor_audit/copy_span_profile`

Audit outputs:

- `outputs/refactor_audit/copy_span_profile/copy_span_profile_audit.md`
- `outputs/refactor_audit/copy_span_profile/copy_span_profile_audit.json`

## Phase-2 Scope Exclusion
Phase-1 is engineering hygiene for reproducibility and clarity only.

- no active-weight retuning
- no retrieval/render equation redesign
- no grouped/minimal scoring migration
- no dataset toggle simplification

Grouped/minimal methodology refactoring is explicitly deferred to Phase-2 on a separate branch.
