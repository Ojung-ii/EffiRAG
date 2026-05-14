from __future__ import annotations

from dataclasses import dataclass

from effirag.gl_rcedr import apply_gl_rcedr


@dataclass
class DummyCfg:
    gl_rcedr_enabled: bool = True
    gl_rcedr_dynamic_control_enabled: bool = True
    gl_rcedr_stability_enabled: bool = True
    gl_rcedr_bridge_path_enabled: bool = True
    gl_rcedr_density_first_ablation_enabled: bool = False
    gl_rcedr_view_count: int = 6
    gl_rcedr_seed_topk: int = 4
    gl_rcedr_recall_weight: float = 0.34
    gl_rcedr_bridge_path_weight: float = 0.22
    gl_rcedr_diversity_weight: float = 0.16
    gl_rcedr_stability_weight: float = 0.12
    gl_rcedr_redundancy_weight: float = 0.16
    gl_rcedr_cost_weight: float = 0.12
    gl_rcedr_max_selected_candidates: int = 6
    gl_rcedr_max_rendered_candidates: int = 6
    gl_rcedr_max_rendered_tokens: int = 24
    gl_rcedr_core_preserve_threshold: float = 0.56
    gl_rcedr_bridge_preserve_threshold: float = 0.46
    gl_rcedr_coverage_preserve_threshold: float = 0.38
    min_render_topn: int = 2


def _features():
    return {
        "s_core": {
            "base_retrieval_score": 0.95,
            "best_corridor_rank": 1,
            "best_corridor_score": 0.85,
            "is_main_candidate": True,
            "is_support_candidate": True,
            "is_connector_adjacent": True,
            "corridor_ids": ["c1"],
            "locality_score": 1.0,
        },
        "s_bridge": {
            "base_retrieval_score": 0.74,
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
            "best_corridor_score": 0.72,
            "is_main_candidate": False,
            "is_support_candidate": False,
            "is_connector_adjacent": False,
            "corridor_ids": ["c1"],
            "locality_score": 0.6,
        },
        "s_short": {
            "base_retrieval_score": 0.72,
            "best_corridor_rank": 3,
            "best_corridor_score": 0.55,
            "is_main_candidate": False,
            "is_support_candidate": False,
            "is_connector_adjacent": False,
            "corridor_ids": ["c3"],
            "locality_score": 0.5,
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


def test_core_and_bridge_evidence_are_preserved_before_density_pruning():
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
    assert int(diag["tokens_after"]) <= int(cfg.gl_rcedr_max_rendered_tokens)
    assert int(diag["selected_count_after"]) <= int(cfg.gl_rcedr_max_selected_candidates)
    assert int(diag["selected_core_count"]) >= 1


def test_density_first_ablation_changes_selection_behavior():
    ids, texts = _payload()
    base_cfg = DummyCfg(gl_rcedr_density_first_ablation_enabled=False)
    ablation_cfg = DummyCfg(gl_rcedr_density_first_ablation_enabled=True)

    selected_base, _texts_base, _diag_base = apply_gl_rcedr(
        question_text="alpha beta connector",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=_features(),
        cfg=base_cfg,
    )
    selected_ablation, _texts_ablation, _diag_ablation = apply_gl_rcedr(
        question_text="alpha beta connector",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=_features(),
        cfg=ablation_cfg,
    )
    assert selected_base != selected_ablation


def test_no_bridge_path_ablation_can_drop_bridge_candidate():
    ids, texts = _payload()
    base_cfg = DummyCfg(gl_rcedr_bridge_path_enabled=True)
    no_bridge_cfg = DummyCfg(gl_rcedr_bridge_path_enabled=False)

    selected_base, _texts_base, diag_base = apply_gl_rcedr(
        question_text="alpha beta connector",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=_features(),
        cfg=base_cfg,
    )
    selected_no_bridge, _texts_no_bridge, diag_no_bridge = apply_gl_rcedr(
        question_text="alpha beta connector",
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=_features(),
        cfg=no_bridge_cfg,
    )
    assert "s_bridge" in selected_base
    assert bool(diag_base["bridge_path_enabled"]) is True
    assert bool(diag_no_bridge["bridge_path_enabled"]) is False
    assert selected_base != selected_no_bridge or bool(diag_no_bridge["applied"]) is True
