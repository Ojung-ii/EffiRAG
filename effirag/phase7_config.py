from dataclasses import dataclass
from typing import Any


@dataclass
class Phase7Config:
    method_name: str = "phase7_evidence_flow"
    enabled: bool = True
    variant: str = "full"
    enable_phase2_refinement: bool = False
    objective_mode: str = "normalized_equal_weight"
    lambda_bridge: float = 1.0
    lambda_decay: float = 1.0
    lambda_bq: float = 1.0
    mu_redundancy: float = 1.0
    conditional_redundancy_enabled: bool = False
    require_sentence_layer: bool = True
    build_sentence_layer: bool = True
    build_carrier_layer: bool = True
    link_sentence_to_carrier: bool = True
    link_sentence_to_entities: bool = True
    atom_unit: str = "sentence"
    carrier_unit: str = "sentence_window"
    carrier_chunk_size_sentences: int = 3
    carrier_chunk_stride_sentences: int = 1

    semantic_anchor_top_t: int = 128
    candidate_top_m: int = 96
    flow_alpha: float = 0.15
    max_flow_iterations: int = 30
    flow_tolerance: float = 1.0e-6

    max_selected_atoms: int = 8
    max_context_tokens: int = 520
    min_positive_gain: float = 0.0

    bottleneck_total_retrieval_ms: float = 500.0
    bottleneck_graph_flow_ms: float = 150.0
    bottleneck_local_graph_build_ms: float = 150.0
    bottleneck_feature_extraction_ms: float = 150.0
    bottleneck_marginal_selection_ms: float = 100.0
    bottleneck_local_graph_nodes: int = 3000
    bottleneck_local_graph_edges: int = 20000
    bottleneck_candidate_atoms: int = 256
    allow_empty_candidates_for_debug: bool = False
    diagnostics_enabled: bool = False
    diagnostics_max_examples_to_dump: int = 100
    diagnostics_dump_text: bool = True
    diagnostics_dump_context: bool = True
    diagnostics_dump_scores: bool = True
    diagnostics_fail_on_unit_mismatch: bool = False

    corridor_enabled: bool = False
    corridor_max_anchors: int = 4
    corridor_max_seeds: int = 16
    corridor_max_hops: int = 3
    corridor_max_paths_per_pair: int = 1
    corridor_degree_cap: int = 100
    corridor_max_pairs: int = 64

    anchor_decay_enabled: bool = False
    anchor_decay_gamma: float = 0.7
    anchor_decay_max_hops: int = 4
    query_intent_enabled: bool = False
    intent_phase1_enabled: bool = False
    intent_max_relation_candidates: int = 32
    intent_max_answer_type_candidates: int = 32
    intent_max_entity_candidates: int = 32
    intent_candidate_top_m: int = 128


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def phase7_config_from_cfg(cfg: Any) -> Phase7Config:
    method_name = str(getattr(cfg, "method", "phase7_evidence_flow") or "phase7_evidence_flow")
    retrieval_method = str(getattr(cfg, "retrieval_method", "") or "").strip().lower()
    variant = str(getattr(cfg, "phase7_variant", "full") or "full").strip().lower()
    if variant not in {
        "full",
        "phase1_only",
        "no_phase2_refinement",
        "phase2_budget_x2",
        "phase2_budget_x3",
        "legacy_compatible_render",
    }:
        variant = "full"
    objective_mode = str(getattr(cfg, "phase7_objective_mode", "normalized_equal_weight") or "normalized_equal_weight").strip().lower()
    if objective_mode not in {
        "normalized_equal_weight",
        "a_only",
        "a_plus_b",
        "a_minus_r",
        "a_plus_b_minus_r",
        "a_plus_decay_minus_r",
        "a_plus_bq_minus_r",
        "a_plus_bq_minus_rq",
        "aq_plus_bq_minus_r",
        "aq_plus_bq_minus_rq",
    }:
        objective_mode = "normalized_equal_weight"
    phase7_enabled = _to_bool(
        getattr(cfg, "phase7_enabled", method_name == "phase7_evidence_flow" or retrieval_method == "phase7_evidence_flow"),
        method_name == "phase7_evidence_flow" or retrieval_method == "phase7_evidence_flow",
    )
    atom_unit = str(getattr(cfg, "phase7_atom_unit", "sentence") or "sentence").strip().lower()
    return Phase7Config(
        method_name=method_name,
        enabled=phase7_enabled,
        variant=variant,
        enable_phase2_refinement=_to_bool(
            getattr(cfg, "phase7_enable_phase2_refinement", False),
            False,
        ),
        objective_mode=str(objective_mode),
        lambda_bridge=max(0.0, _to_float(getattr(cfg, "phase7_lambda_bridge", 1.0), 1.0)),
        lambda_decay=max(0.0, _to_float(getattr(cfg, "phase7_lambda_decay", 1.0), 1.0)),
        lambda_bq=max(0.0, _to_float(getattr(cfg, "phase7_lambda_bq", 1.0), 1.0)),
        mu_redundancy=max(0.0, _to_float(getattr(cfg, "phase7_mu_redundancy", 1.0), 1.0)),
        conditional_redundancy_enabled=_to_bool(
            getattr(cfg, "phase7_conditional_redundancy_enabled", False),
            False,
        ),
        require_sentence_layer=_to_bool(
            getattr(cfg, "phase7_require_sentence_layer", atom_unit == "sentence"),
            atom_unit == "sentence",
        ),
        build_sentence_layer=_to_bool(getattr(cfg, "phase7_build_sentence_layer", True), True),
        build_carrier_layer=_to_bool(getattr(cfg, "phase7_build_carrier_layer", True), True),
        link_sentence_to_carrier=_to_bool(getattr(cfg, "phase7_link_sentence_to_carrier", True), True),
        link_sentence_to_entities=_to_bool(getattr(cfg, "phase7_link_sentence_to_entities", True), True),
        atom_unit=atom_unit,
        carrier_unit=str(getattr(cfg, "phase7_carrier_unit", "sentence_window") or "sentence_window").strip().lower(),
        carrier_chunk_size_sentences=max(
            1, _to_int(getattr(cfg, "phase7_carrier_chunk_size_sentences", 3), 3)
        ),
        carrier_chunk_stride_sentences=max(
            1, _to_int(getattr(cfg, "phase7_carrier_chunk_stride_sentences", 1), 1)
        ),
        semantic_anchor_top_t=max(
            1, _to_int(getattr(cfg, "phase7_semantic_anchor_top_t", 128), 128)
        ),
        candidate_top_m=max(1, _to_int(getattr(cfg, "phase7_candidate_top_m", 96), 96)),
        flow_alpha=max(0.01, min(0.99, _to_float(getattr(cfg, "phase7_flow_alpha", 0.15), 0.15))),
        max_flow_iterations=max(
            1, _to_int(getattr(cfg, "phase7_max_flow_iterations", 30), 30)
        ),
        flow_tolerance=max(
            1.0e-12, _to_float(getattr(cfg, "phase7_flow_tolerance", 1.0e-6), 1.0e-6)
        ),
        max_selected_atoms=max(
            1, _to_int(getattr(cfg, "phase7_max_selected_atoms", 8), 8)
        ),
        max_context_tokens=max(
            32, _to_int(getattr(cfg, "phase7_max_context_tokens", 520), 520)
        ),
        min_positive_gain=_to_float(
            getattr(cfg, "phase7_min_positive_gain", 0.0), 0.0
        ),
        bottleneck_total_retrieval_ms=_to_float(
            getattr(cfg, "phase7_bottleneck_total_retrieval_ms", 500), 500.0
        ),
        bottleneck_graph_flow_ms=_to_float(
            getattr(cfg, "phase7_bottleneck_graph_flow_ms", 150), 150.0
        ),
        bottleneck_local_graph_build_ms=_to_float(
            getattr(cfg, "phase7_bottleneck_local_graph_build_ms", 150), 150.0
        ),
        bottleneck_feature_extraction_ms=_to_float(
            getattr(cfg, "phase7_bottleneck_feature_extraction_ms", 150), 150.0
        ),
        bottleneck_marginal_selection_ms=_to_float(
            getattr(cfg, "phase7_bottleneck_marginal_selection_ms", 100), 100.0
        ),
        bottleneck_local_graph_nodes=max(
            1, _to_int(getattr(cfg, "phase7_bottleneck_local_graph_nodes", 3000), 3000)
        ),
        bottleneck_local_graph_edges=max(
            1, _to_int(getattr(cfg, "phase7_bottleneck_local_graph_edges", 20000), 20000)
        ),
        bottleneck_candidate_atoms=max(
            1, _to_int(getattr(cfg, "phase7_bottleneck_candidate_atoms", 256), 256)
        ),
        allow_empty_candidates_for_debug=_to_bool(
            getattr(cfg, "phase7_allow_empty_candidates_for_debug", False),
            False,
        ),
        diagnostics_enabled=_to_bool(
            getattr(cfg, "phase7_diagnostics_enabled", False),
            False,
        ),
        diagnostics_max_examples_to_dump=max(
            1, _to_int(getattr(cfg, "phase7_diagnostics_max_examples_to_dump", 100), 100)
        ),
        diagnostics_dump_text=_to_bool(
            getattr(cfg, "phase7_diagnostics_dump_text", True),
            True,
        ),
        diagnostics_dump_context=_to_bool(
            getattr(cfg, "phase7_diagnostics_dump_context", True),
            True,
        ),
        diagnostics_dump_scores=_to_bool(
            getattr(cfg, "phase7_diagnostics_dump_scores", True),
            True,
        ),
        diagnostics_fail_on_unit_mismatch=_to_bool(
            getattr(cfg, "phase7_diagnostics_fail_on_unit_mismatch", False),
            False,
        ),
        corridor_enabled=_to_bool(
            getattr(cfg, "phase7_corridor_enabled", False),
            False,
        ),
        corridor_max_anchors=max(1, _to_int(getattr(cfg, "phase7_corridor_max_anchors", 4), 4)),
        corridor_max_seeds=max(1, _to_int(getattr(cfg, "phase7_corridor_max_seeds", 16), 16)),
        corridor_max_hops=max(1, _to_int(getattr(cfg, "phase7_corridor_max_hops", 3), 3)),
        corridor_max_paths_per_pair=max(
            1, _to_int(getattr(cfg, "phase7_corridor_max_paths_per_pair", 1), 1)
        ),
        corridor_degree_cap=max(1, _to_int(getattr(cfg, "phase7_corridor_degree_cap", 100), 100)),
        corridor_max_pairs=max(1, _to_int(getattr(cfg, "phase7_corridor_max_pairs", 64), 64)),
        anchor_decay_enabled=_to_bool(
            getattr(cfg, "phase7_anchor_decay_enabled", False),
            False,
        ),
        anchor_decay_gamma=max(0.0, _to_float(getattr(cfg, "phase7_anchor_decay_gamma", 0.7), 0.7)),
        anchor_decay_max_hops=max(
            1, _to_int(getattr(cfg, "phase7_anchor_decay_max_hops", 4), 4)
        ),
        query_intent_enabled=_to_bool(
            getattr(cfg, "phase7_query_intent_enabled", False),
            False,
        ),
        intent_phase1_enabled=_to_bool(
            getattr(cfg, "phase7_intent_phase1_enabled", False),
            False,
        ),
        intent_max_relation_candidates=max(
            1, _to_int(getattr(cfg, "phase7_intent_max_relation_candidates", 32), 32)
        ),
        intent_max_answer_type_candidates=max(
            1, _to_int(getattr(cfg, "phase7_intent_max_answer_type_candidates", 32), 32)
        ),
        intent_max_entity_candidates=max(
            1, _to_int(getattr(cfg, "phase7_intent_max_entity_candidates", 32), 32)
        ),
        intent_candidate_top_m=max(
            1, _to_int(getattr(cfg, "phase7_intent_candidate_top_m", 128), 128)
        ),
    )
