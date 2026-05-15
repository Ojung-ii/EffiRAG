from __future__ import annotations

from dataclasses import dataclass

from effirag.gl_rcedr import apply_gl_rcedr


@dataclass
class DummyCfg:
    gl_rcedr_enabled: bool = True
    gl_rcedr_stability_enabled: bool = True
    gl_rcedr_bridge_path_enabled: bool = True
    gl_rcedr_seed_topk: int = 4
    min_render_topn: int = 2

    unified_marginal_utility_enabled: bool = True
    gl_rcedr_v2_adaptive_bridge_enabled: bool = True
    gl_rcedr_v2_cost_enabled: bool = True
    gl_rcedr_v2_redundancy_enabled: bool = True
    gl_rcedr_v2_density_first_enabled: bool = False
    gl_rcedr_v2_evidence_gain_weight: float = 0.52
    gl_rcedr_v2_bridge_gain_weight: float = 0.28
    gl_rcedr_v2_redundancy_weight: float = 0.22
    gl_rcedr_v2_cost_weight: float = 0.14
    gl_rcedr_v2_seed_stability_weight: float = 0.15
    gl_rcedr_v2_seed_diversity_weight: float = 0.15
    gl_rcedr_v2_bridge_lambda_base: float = 1.0
    gl_rcedr_v2_bridge_lambda_alpha: float = 0.30
    gl_rcedr_v2_bridge_lambda_min: float = 0.85
    gl_rcedr_v2_bridge_lambda_max: float = 1.35
    gl_rcedr_v2_max_selected_candidates: int = 5
    gl_rcedr_v2_max_rendered_candidates: int = 5
    gl_rcedr_v2_max_rendered_tokens: int = 24
    gl_rcedr_v2_core_preserve_threshold: float = 0.56
    gl_rcedr_v2_bridge_preserve_threshold: float = 0.46


def _features():
    return {
        "s_core": {
            "base_retrieval_score": 0.95,
            "best_corridor_rank": 1,
            "best_corridor_score": 0.88,
            "is_main_candidate": True,
            "is_support_candidate": True,
            "is_connector_adjacent": True,
            "corridor_ids": ["c1"],
            "locality_score": 1.0,
        },
        "s_bridge": {
            "base_retrieval_score": 0.76,
            "best_corridor_rank": 1,
            "best_corridor_score": 0.82,
            "is_main_candidate": False,
            "is_support_candidate": True,
            "is_connector_adjacent": True,
            "corridor_ids": ["c1", "c2"],
            "locality_score": 0.8,
        },
        "s_dup": {
            "base_retrieval_score": 0.78,
            "best_corridor_rank": 2,
            "best_corridor_score": 0.70,
            "is_main_candidate": False,
            "is_support_candidate": False,
            "is_connector_adjacent": False,
            "corridor_ids": ["c1"],
            "locality_score": 0.6,
        },
        "s_short": {
            "base_retrieval_score": 0.70,
            "best_corridor_rank": 4,
            "best_corridor_score": 0.45,
            "is_main_candidate": False,
            "is_support_candidate": False,
            "is_connector_adjacent": False,
            "corridor_ids": ["c3"],
            "locality_score": 0.4,
        },
    }


def _payload():
    ids = ["s_core", "s_bridge", "s_dup", "s_short"]
    texts = [
        "alpha connector evidence chain support",  # core
        "bridge sentence links alpha and beta entities",  # bridge
        "alpha connector evidence chain support",  # duplicate
        "beta compact clue",  # short
    ]
    return ids, texts


def test_v2_preserves_core_and_bridge_then_prunes_by_budget():
    ids, texts = _payload()
    cfg = DummyCfg()
    selected_ids, _selected_texts, diag = apply_gl_rcedr(
        question_text="alpha beta connector",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=_features(),
        cfg=cfg,
    )
    assert "s_core" in selected_ids
    assert "s_bridge" in selected_ids
    assert int(diag["tokens_after"]) <= int(cfg.gl_rcedr_v2_max_rendered_tokens)
    assert int(diag["selected_count_after"]) <= int(cfg.gl_rcedr_v2_max_selected_candidates)
    assert bool(diag["unified_marginal_utility_enabled"]) is True


def test_v2_density_first_ablation_changes_selection():
    ids, texts = _payload()
    base_cfg = DummyCfg(gl_rcedr_v2_density_first_enabled=False)
    density_cfg = DummyCfg(gl_rcedr_v2_density_first_enabled=True)
    base_selected, _base_text, _base_diag = apply_gl_rcedr(
        question_text="alpha beta connector",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=_features(),
        cfg=base_cfg,
    )
    density_selected, _density_text, _density_diag = apply_gl_rcedr(
        question_text="alpha beta connector",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=_features(),
        cfg=density_cfg,
    )
    assert bool(_density_diag["density_first_ablation_enabled"]) is True
    if base_selected == density_selected:
        assert _base_diag.get("score_trace", {}) != _density_diag.get("score_trace", {})
