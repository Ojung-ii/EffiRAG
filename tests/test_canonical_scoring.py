import math

from effirag.canonical_scoring import (
    canonical_run_score,
    canonical_run_weight_bundle,
    canonical_seed_score_map,
    canonical_sentence_score,
)
from effirag.config import RagConfig


def test_canonical_seed_score_ignores_optional_seed_fields():
    cfg = RagConfig()
    cfg.seed_score_semantic_weight = 0.35
    cfg.seed_score_graph_weight = 0.45
    cfg.seed_score_anchor_weight = 0.20
    cfg.seed_score_bridge_weight = 100.0
    cfg.seed_score_chunk_grounding_weight = 100.0

    out = canonical_seed_score_map(
        candidates=["n0"],
        graph_norm={"n0": 0.8},
        semantic_norm={"n0": 0.6},
        anchor_norm={"n0": 0.4},
        cfg=cfg,
    )
    expected = (0.45 * 0.8) + (0.35 * 0.6) + (0.20 * 0.4)
    assert math.isclose(float(out["n0"]), float(expected), rel_tol=0.0, abs_tol=1e-12)


def test_canonical_run_score_ignores_optional_run_fields():
    cfg = RagConfig()
    cfg.run_score_semantic_weight = 0.30
    cfg.run_score_anchor_weight = 0.20
    cfg.run_score_structure_weight = 0.25
    cfg.run_score_bridge_weight = 0.15
    cfg.run_score_redundancy_weight = 0.10
    cfg.run_score_pair_coverage_weight = 7.0
    cfg.run_score_bridge_completeness_weight = 7.0
    cfg.run_score_entity_chunk_grounding_weight = 7.0
    cfg.run_score_anchor_dispersion_penalty = 7.0

    score = canonical_run_score(
        semantic=1.0,
        anchor=0.5,
        structure=0.25,
        bridge=0.75,
        redundancy=0.2,
        cfg=cfg,
    )
    expected = (0.30 * 1.0) + (0.20 * 0.5) + (0.25 * 0.25) + (0.15 * 0.75) - (0.10 * 0.2)
    assert math.isclose(float(score), float(expected), rel_tol=0.0, abs_tol=1e-12)


def test_canonical_run_weight_bundle_has_required_core_terms():
    cfg = RagConfig()
    ww = canonical_run_weight_bundle(cfg)
    assert set(ww.keys()) == {"semantic", "anchor", "structure", "bridge", "redundancy"}
    assert math.isclose(sum(float(v) for v in ww.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_canonical_sentence_score_ignores_query_locality_bonus_terms():
    feat = {
        "base_retrieval_score": 0.7,
        "best_corridor_score": 0.2,
        "is_main_candidate": True,
        "is_support_candidate": False,
        "is_connector_adjacent": True,
        "query_overlap_score": 99.0,
        "locality_score": 99.0,
    }
    score = canonical_sentence_score(
        feat=feat,
        alpha=1.0,
        beta=0.5,
        gamma_main=0.2,
        delta_support=0.1,
        eta_connector=0.3,
        lambda_redundancy=0.25,
        redundancy=0.4,
        title_repeats=2.0,
    )
    expected = (1.0 * 0.7) + (0.5 * 0.2) + 0.2 + 0.3 - (0.25 * 0.4) - (0.20 * 0.25 * 2.0)
    assert math.isclose(float(score), float(expected), rel_tol=0.0, abs_tol=1e-12)
