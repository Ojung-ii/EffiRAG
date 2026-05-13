from __future__ import annotations

from typing import Dict


COPY_SPAN_PROFILE_NAME = "copy-span"
COPY_SPAN_CANONICAL_VARIANT = "champion_reconfirm"
COPY_SPAN_LOCK_ROUND_ROOT = "outputs/light_separator_lockin_round/20260428_004715"

COPY_SPAN_RETRIEVAL_WEIGHTS = {
    "run_score_semantic_weight": 0.30,
    "run_score_anchor_weight": 0.20,
    "run_score_structure_weight": 0.25,
    "run_score_bridge_weight": 0.15,
    "run_score_redundancy_weight": 0.10,
    "seed_score_semantic_weight": 0.35,
    "seed_score_graph_weight": 0.45,
    "seed_score_anchor_weight": 0.20,
    "embedding_weight": 0.35,
}

COPY_SPAN_RENDER_WEIGHTS = {
    "alpha": 1.0,
    "beta": 0.35,
    "gamma_main": 0.45,
    "delta_support": 0.20,
    "eta_connector": 0.20,
    "zeta_query": 0.12,
    "xi_locality": 0.08,
    "lambda_redundancy": 0.25,
}

COPY_SPAN_BUDGET_AND_SHAPE = {
    "max_context_sentences": 8,
    "max_total_sentences": 8,
    "top_corridors": 3,
    "delivery_mode": "sentence_compressed",
    "render_mode": "corridor_aware_flat",
    "order_strategy": "score",
}

COPY_SPAN_DISABLED_MODULES = {
    "top1_correction_enabled": False,
    "run_light_rerank_enabled": False,
    "corridor_compact_shaping_enabled": False,
    "corridor_bridge_purity_shaping_enabled": False,
    "chunk_grounding_enabled": False,
    "hybrid_anchor_recall_enabled": False,
    "role_aware_chunk_scoring_enabled": False,
    "coverage_selection_enabled": False,
}

# Legacy compatibility settings for reproducing previous SOTA runs.
# These settings may contain dataset-specific behavior and must not be used
# as the paper's strict unified main method.
COPY_SPAN_DATASET_LOCKS = {
    "hotpotqa": {
        "retrieval_objective_mode": "p3_answer_preserve_guarded_hotpot",
        "answer_support_pinning_enabled": True,
        "final_top_slice_reorder_enabled": False,
        "precomputed_retrieval_strict": True,
    },
    "2wikimultihopqa": {
        "retrieval_objective_mode": "baseline",
        "answer_support_pinning_enabled": False,
        "final_top_slice_reorder_enabled": True,
        "precomputed_retrieval_strict": True,
    },
    "musique": {
        "retrieval_objective_mode": "baseline",
        "answer_support_pinning_enabled": False,
        "final_top_slice_reorder_enabled": False,
        "precomputed_retrieval_strict": False,
    },
    "popqa": {
        "retrieval_objective_mode": "baseline",
        "answer_support_pinning_enabled": False,
        "final_top_slice_reorder_enabled": False,
        "precomputed_retrieval_strict": False,
    },
}

# Phase-1 registry for deprecated/legacy config knobs. These are preserved for
# backward compatibility and audit visibility; runtime behavior is unchanged.
DEPRECATED_OR_LEGACY_CONFIG_FIELDS = {
    "semantic_chunk_support_expand_per_entity",
    "semantic_chunk_support_expand_total",
    "proposal_adaptive_budget",
    "proposal_chunk_rank_cap",
    "proposal_reserve_graph_mix_topn",
    "proposal_sparse_subgraph_build",
    "proposal_subgraph_anchor_neighbor_cap",
    "proposal_subgraph_support_cap",
    "proposal_subgraph_connector_cap",
}


def _normalize_copy_span_dataset_name(dataset: str) -> str:
    raw = str(dataset or "").strip().lower()
    aliases = {
        "hotpotqa": "hotpotqa",
        "hotpot": "hotpotqa",
        "2wikimultihopqa": "2wikimultihopqa",
        "2wiki": "2wikimultihopqa",
        "wikimultihopqa": "2wikimultihopqa",
        "musique": "musique",
        "popqa": "popqa",
    }
    return aliases.get(raw, raw)


def expected_copy_span_config(dataset: str) -> Dict[str, object]:
    """Return the expected raw config contract for a copy-span dataset."""
    dataset_key = _normalize_copy_span_dataset_name(dataset)
    if dataset_key not in COPY_SPAN_DATASET_LOCKS:
        supported = ", ".join(sorted(COPY_SPAN_DATASET_LOCKS.keys()))
        raise ValueError(f"Unsupported copy-span dataset '{dataset}'. Supported: {supported}")

    expected: Dict[str, object] = {}
    expected.update(COPY_SPAN_RETRIEVAL_WEIGHTS)
    expected.update(COPY_SPAN_RENDER_WEIGHTS)
    expected.update(COPY_SPAN_BUDGET_AND_SHAPE)
    expected.update(COPY_SPAN_DISABLED_MODULES)
    expected.update(COPY_SPAN_DATASET_LOCKS[dataset_key])
    expected["canonical_variant_name"] = COPY_SPAN_CANONICAL_VARIANT
    return expected
