from effirag.config import RagConfig, dataclass_from_dict
from effirag.retrieval import _apply_connector_objective_profile, _resolve_retrieval_objective_flags
from effirag.unified_copy_span_policy import apply_unified_profile, resolve_unified_copy_span_mode


def test_unified_resolver_ignores_hotpot_dataset_name():
    assert resolve_unified_copy_span_mode("hotpotqa") == "baseline"
    assert resolve_unified_copy_span_mode("2wikimultihopqa") == "baseline"


def test_hotpotqa_unified_profile_does_not_enable_hotpot_guard_at_runtime():
    profiles = [
        "unified_large",
        "unified_medium",
        "unified_compact",
        "unified_large_lightsep",
        "unified_large_reorder_all",
        "unified_large_lightsep_reorder_all",
        "unified_medium_lightsep_reorder_all",
        "unified_dynamic_compact_v1",
        "unified_dynamic_compact_v2",
        "unified_dynamic_contextual_compact_v3",
        "unified_candidate_recall_boost_v1",
        "unified_candidate_recall_boost_dynamic_v1",
        "unified_candidate_recall_boost_density_rerank_v1",
    ]

    for profile in profiles:
        raw_cfg = apply_unified_profile({}, "hotpotqa", profile)
        cfg = dataclass_from_dict(RagConfig, raw_cfg, warn_unknown_keys=False)

        flags = _resolve_retrieval_objective_flags(cfg)
        flags, diag = _apply_connector_objective_profile(cfg, flags)

        assert cfg.retrieval_objective_mode == "baseline"
        assert diag["effective_mode"] == "baseline"
        assert flags["corridor_answer_preserve_guarded_hotpot"] is False
        assert cfg.corridor_answer_preserve_guarded_hotpot_enabled is False
        assert cfg.corridor_answer_preserve_enabled is False
        assert cfg.answer_support_pinning_enabled is False
        if profile in {
            "unified_dynamic_compact_v1",
            "unified_dynamic_compact_v2",
            "unified_candidate_recall_boost_dynamic_v1",
        }:
            assert cfg.dynamic_compact_selection_enabled is True
            assert cfg.coverage_gain_enabled is True
            assert cfg.bridge_preserve_enabled is True
        if profile in {
            "unified_candidate_recall_boost_v1",
            "unified_candidate_recall_boost_dynamic_v1",
            "unified_candidate_recall_boost_density_rerank_v1",
        }:
            assert cfg.bridge_candidate_induction_enabled is True
            assert cfg.path_candidate_expansion_enabled is True
            assert cfg.anchor_expansion_enabled is True
            assert cfg.candidate_diversity_enabled is True
            assert cfg.entity_diversity_enabled is True
            assert cfg.source_diversity_enabled is True
            assert cfg.dataset_specific_branch_enabled is False
        if profile == "unified_candidate_recall_boost_density_rerank_v1":
            assert cfg.evidence_density_rerank_enabled is True
            assert cfg.bridge_path_utility_enabled is True
            assert cfg.coverage_gain_enabled is True
            assert cfg.redundancy_penalty_enabled is True
            assert cfg.token_cost_penalty_enabled is True
            assert cfg.density_budget_awareness_enabled is True
        if profile == "unified_dynamic_compact_v2":
            assert cfg.selector_aware_render_enabled is True
            assert cfg.render_selected_only is True
        if profile == "unified_dynamic_contextual_compact_v3":
            assert cfg.selector_aware_render_enabled is True
            assert cfg.render_selected_only is False
            assert cfg.render_selected_centered is True
            assert cfg.render_contextual_expansion_enabled is True
            assert cfg.render_bridge_context_enabled is True
            assert cfg.render_path_context_enabled is True
        if "reorder_all" in profile:
            assert cfg.final_top_slice_reorder_enabled is True
        else:
            assert cfg.final_top_slice_reorder_enabled is False
