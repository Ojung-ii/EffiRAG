from effirag.config import RetrievalConfig
from effirag.retrieval import _select_phase1_diversity_reserve
from effirag.unified_copy_span_policy import unified_profile_dir


def _run(run_id, seeds, hybrid=0.5, semantic=0.5, graph=0.5, bridge=0.5, grounding=0.0):
    return {
        "run_id": int(run_id),
        "seeds": set(seeds),
        "hybrid_run_score": float(hybrid),
        "run_score_components": {
            "semantic_coverage": float(semantic),
            "structural_connectivity": float(graph),
            "bridge_path_completeness": float(bridge),
            "entity_chunk_grounding_score": float(grounding),
            "cheap_pre_components": {
                "max_semantic_seed": float(semantic),
                "graph_seed_mass": float(graph),
                "bridge_proxy": float(bridge),
            },
        },
    }


def test_baseline_unchanged_when_reserve_disabled():
    cfg = RetrievalConfig()
    cfg.phase1_diversity_reserve_enabled = False
    ranked = [_run(1, {"a", "b"}), _run(2, {"c", "d"})]
    shortlisted = [ranked[0]]
    reserve, diag = _select_phase1_diversity_reserve(ranked, shortlisted, cfg)
    assert reserve is None
    assert diag["enabled"] is False
    assert diag["added"] is False


def test_reserve_count_is_single_and_nonredundant():
    cfg = RetrievalConfig()
    cfg.phase1_diversity_reserve_enabled = True
    cfg.phase1_diversity_reserve_count = 1
    cfg.phase1_diversity_reserve_mode = "nonredundant_best"
    ranked = [
        _run(1, {"s1", "s2"}, hybrid=0.9),
        _run(2, {"s1", "s2"}, hybrid=0.8),  # high overlap -> skip
        _run(3, {"s3", "s4"}, hybrid=0.7),  # nonredundant -> select
    ]
    shortlisted = [ranked[0]]
    reserve, diag = _select_phase1_diversity_reserve(ranked, shortlisted, cfg)
    assert reserve is not None
    assert int(reserve["run_id"]) == 3
    assert diag["added"] is True
    assert int(diag["added_run_id"]) == 3


def test_reserve_selection_ignores_gold_fields():
    cfg = RetrievalConfig()
    cfg.phase1_diversity_reserve_enabled = True
    ranked = [
        _run(1, {"a", "b"}, hybrid=0.8),
        {**_run(2, {"c", "d"}, hybrid=0.7), "gold_support": True, "gold_answer": "x"},
    ]
    shortlisted = [ranked[0]]
    reserve, diag = _select_phase1_diversity_reserve(ranked, shortlisted, cfg)
    assert reserve is not None
    assert int(reserve["run_id"]) == 2
    assert diag["added"] is True


def test_retrieval_config_reserve_defaults():
    cfg = RetrievalConfig()
    assert cfg.phase1_diversity_reserve_enabled is False
    assert cfg.phase1_diversity_reserve_count == 1
    assert cfg.phase1_diversity_reserve_mode == "nonredundant_best"
    assert cfg.phase1_reserve_allow_phase2_refinement is True


def test_unified_profile_alias_for_early_collapse_recovery():
    assert (
        unified_profile_dir("unified_acr_rcedr_v12_early_collapse_recovery")
        == "unified_acr_rcedr_v12_early_collapse_recovery"
    )

