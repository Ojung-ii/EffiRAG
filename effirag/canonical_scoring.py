from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping


CANONICAL_SEED_CORE_WEIGHT_FIELDS = (
    "seed_score_semantic_weight",
    "seed_score_graph_weight",
    "seed_score_anchor_weight",
)

CANONICAL_SEED_PRUNED_WEIGHT_FIELDS = (
    "seed_score_bridge_weight",
    "seed_score_chunk_grounding_weight",
)

CANONICAL_RUN_CORE_WEIGHT_FIELDS = (
    "run_score_semantic_weight",
    "run_score_anchor_weight",
    "run_score_structure_weight",
    "run_score_bridge_weight",
    "run_score_redundancy_weight",
)

CANONICAL_RUN_PRUNED_WEIGHT_FIELDS = (
    "run_score_pair_coverage_weight",
    "run_score_bridge_completeness_weight",
    "run_score_entity_chunk_grounding_weight",
    "run_score_anchor_dispersion_penalty",
)

CANONICAL_RENDER_CORE_WEIGHT_FIELDS = (
    "alpha",
    "beta",
    "gamma_main",
    "delta_support",
    "eta_connector",
    "lambda_redundancy",
)

CANONICAL_RENDER_PRUNED_WEIGHT_FIELDS = (
    "zeta_query",
    "xi_locality",
)

CANONICAL_PRUNED_WEIGHT_FIELDS = (
    *CANONICAL_SEED_PRUNED_WEIGHT_FIELDS,
    *CANONICAL_RUN_PRUNED_WEIGHT_FIELDS,
    *CANONICAL_RENDER_PRUNED_WEIGHT_FIELDS,
)

CANONICAL_CORE_WEIGHT_FIELDS = (
    *CANONICAL_SEED_CORE_WEIGHT_FIELDS,
    *CANONICAL_RUN_CORE_WEIGHT_FIELDS,
    *CANONICAL_RENDER_CORE_WEIGHT_FIELDS,
)


def _as_nonneg_float(value, default: float = 0.0) -> float:
    try:
        vv = float(value)
    except Exception:
        vv = float(default)
    return max(0.0, vv)


def canonical_seed_weight_bundle(cfg) -> Dict[str, float]:
    w_sem = _as_nonneg_float(getattr(cfg, "seed_score_semantic_weight", 0.30), 0.30)
    w_graph = _as_nonneg_float(getattr(cfg, "seed_score_graph_weight", 0.50), 0.50)
    w_anchor = _as_nonneg_float(getattr(cfg, "seed_score_anchor_weight", 0.20), 0.20)
    total = w_sem + w_graph + w_anchor
    if total <= 0.0:
        w_sem, w_graph, w_anchor = 0.0, 1.0, 0.0
        total = 1.0
    return {
        "semantic": float(w_sem / total),
        "graph": float(w_graph / total),
        "anchor": float(w_anchor / total),
    }


def canonical_seed_score(
    *,
    graph_score: float,
    semantic_score: float,
    anchor_score: float,
    cfg,
) -> float:
    ww = canonical_seed_weight_bundle(cfg)
    return float(
        ww["graph"] * float(graph_score)
        + ww["semantic"] * float(semantic_score)
        + ww["anchor"] * float(anchor_score)
    )


def canonical_seed_score_map(
    *,
    candidates: Iterable[str],
    graph_norm: Mapping[str, float],
    semantic_norm: Mapping[str, float],
    anchor_norm: Mapping[str, float],
    cfg,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ww = canonical_seed_weight_bundle(cfg)
    w_graph = float(ww["graph"])
    w_sem = float(ww["semantic"])
    w_anchor = float(ww["anchor"])
    for node in candidates:
        out[node] = float(
            w_graph * float(graph_norm.get(node, 0.0))
            + w_sem * float(semantic_norm.get(node, 0.0))
            + w_anchor * float(anchor_norm.get(node, 0.0))
        )
    return out


def canonical_run_weight_bundle(cfg) -> Dict[str, float]:
    w_sem = _as_nonneg_float(getattr(cfg, "run_score_semantic_weight", 0.30), 0.30)
    w_anchor = _as_nonneg_float(getattr(cfg, "run_score_anchor_weight", 0.20), 0.20)
    w_struct = _as_nonneg_float(getattr(cfg, "run_score_structure_weight", 0.25), 0.25)
    w_bridge = _as_nonneg_float(getattr(cfg, "run_score_bridge_weight", 0.15), 0.15)
    w_redundancy = _as_nonneg_float(getattr(cfg, "run_score_redundancy_weight", 0.10), 0.10)
    total = w_sem + w_anchor + w_struct + w_bridge + w_redundancy
    if total <= 0.0:
        w_sem, w_anchor, w_struct, w_bridge, w_redundancy = 0.30, 0.20, 0.25, 0.15, 0.10
        total = 1.0
    return {
        "semantic": float(w_sem / total),
        "anchor": float(w_anchor / total),
        "structure": float(w_struct / total),
        "bridge": float(w_bridge / total),
        "redundancy": float(w_redundancy / total),
    }


def canonical_run_score(
    *,
    semantic: float,
    anchor: float,
    structure: float,
    bridge: float,
    redundancy: float,
    cfg,
) -> float:
    ww = canonical_run_weight_bundle(cfg)
    return float(
        ww["semantic"] * float(semantic)
        + ww["anchor"] * float(anchor)
        + ww["structure"] * float(structure)
        + ww["bridge"] * float(bridge)
        - ww["redundancy"] * float(redundancy)
    )


def canonical_run_surrogate_loss(
    *,
    semantic: float,
    anchor: float,
    structure: float,
    bridge: float,
    redundancy: float,
    cfg,
    dispersion_penalty: float = 0.0,
) -> float:
    ww = canonical_run_weight_bundle(cfg)
    return float(
        ww["semantic"] * (1.0 - float(semantic))
        + ww["anchor"] * (1.0 - float(anchor))
        + ww["structure"] * (1.0 - float(structure))
        + ww["bridge"] * (1.0 - float(bridge))
        + ww["redundancy"] * float(redundancy)
        + 0.25 * max(0.0, float(dispersion_penalty))
    )


def canonical_sentence_score(
    *,
    feat: Mapping[str, float],
    alpha: float,
    beta: float,
    gamma_main: float,
    delta_support: float,
    eta_connector: float,
    lambda_redundancy: float,
    redundancy: float,
    title_repeats: float = 0.0,
) -> float:
    base_retrieval_score = float(feat.get("base_retrieval_score", 0.0))
    corridor_score = float(feat.get("best_corridor_score", 0.0))
    is_main = 1.0 if bool(feat.get("is_main_candidate")) else 0.0
    is_support = 1.0 if bool(feat.get("is_support_candidate")) else 0.0
    is_connector = 1.0 if bool(feat.get("is_connector_adjacent")) else 0.0

    retrieval_confidence = float(alpha) * base_retrieval_score + float(beta) * corridor_score
    evidence_role_score = (
        float(gamma_main) * is_main
        + float(delta_support) * is_support
        + float(eta_connector) * is_connector
    )
    compactness_penalty = float(lambda_redundancy) * float(redundancy) + (
        0.20 * max(0.0, float(lambda_redundancy)) * float(max(0.0, float(title_repeats)))
    )
    return float(retrieval_confidence + evidence_role_score - compactness_penalty)
