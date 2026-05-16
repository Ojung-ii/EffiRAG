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
        "unified_gl_rcedr_v1",
        "unified_gl_rcedr_v1_sentence_contract",
        "unified_gl_rcedr_v1_sentence_contract_no_item_cap",
        "unified_gl_rcedr_v1_sentence_contract_metadata_on",
        "unified_gl_rcedr_v1_sentence_contract_span40",
        "unified_gl_rcedr_v1_support_span_contract",
        "unified_gl_rcedr_v1_support_span_contract_span40",
        "unified_gl_rcedr_v1_support_span_contract_no_cap",
        "unified_gl_rcedr_v1_support_span_contract_no_bridge_signal",
        "unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal",
        "unified_gl_rcedr_v1_adaptive_support_span",
        "unified_gl_rcedr_v1_adaptive_support_span_weak_bridge",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
        "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty",
        "unified_gl_rcedr_v1_adaptive_support_span_span40",
        "unified_gl_rcedr_no_dynamic_control",
        "unified_gl_rcedr_no_stability",
        "unified_gl_rcedr_no_bridge_path",
        "unified_gl_rcedr_density_first_ablation",
        "unified_gl_rcedr_v2",
        "unified_gl_rcedr_v2_no_adaptive_bridge",
        "unified_gl_rcedr_v2_no_cost",
        "unified_gl_rcedr_v2_no_redundancy",
        "unified_gl_rcedr_v2_density_first",
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
            "unified_gl_rcedr_v1",
            "unified_gl_rcedr_v1_sentence_contract",
            "unified_gl_rcedr_v1_sentence_contract_no_item_cap",
            "unified_gl_rcedr_v1_sentence_contract_metadata_on",
            "unified_gl_rcedr_v1_sentence_contract_span40",
            "unified_gl_rcedr_v1_support_span_contract",
            "unified_gl_rcedr_v1_support_span_contract_span40",
            "unified_gl_rcedr_v1_support_span_contract_no_cap",
            "unified_gl_rcedr_v1_support_span_contract_no_bridge_signal",
            "unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal",
            "unified_gl_rcedr_v1_adaptive_support_span",
            "unified_gl_rcedr_v1_adaptive_support_span_weak_bridge",
            "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
            "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty",
            "unified_gl_rcedr_v1_adaptive_support_span_span40",
            "unified_gl_rcedr_no_dynamic_control",
            "unified_gl_rcedr_no_stability",
            "unified_gl_rcedr_no_bridge_path",
            "unified_gl_rcedr_density_first_ablation",
            "unified_gl_rcedr_v2",
            "unified_gl_rcedr_v2_no_adaptive_bridge",
            "unified_gl_rcedr_v2_no_cost",
            "unified_gl_rcedr_v2_no_redundancy",
            "unified_gl_rcedr_v2_density_first",
        }:
            assert cfg.bridge_candidate_induction_enabled is True
            assert cfg.path_candidate_expansion_enabled is True
            assert cfg.anchor_expansion_enabled is True
            assert cfg.candidate_diversity_enabled is True
            assert cfg.entity_diversity_enabled is True
            assert cfg.source_diversity_enabled is True
            assert cfg.dataset_specific_branch_enabled is False
        if profile in {
            "unified_gl_rcedr_v1",
            "unified_gl_rcedr_v1_sentence_contract",
            "unified_gl_rcedr_v1_sentence_contract_no_item_cap",
            "unified_gl_rcedr_v1_sentence_contract_metadata_on",
            "unified_gl_rcedr_v1_sentence_contract_span40",
            "unified_gl_rcedr_v1_support_span_contract",
            "unified_gl_rcedr_v1_support_span_contract_span40",
            "unified_gl_rcedr_v1_support_span_contract_no_cap",
            "unified_gl_rcedr_v1_support_span_contract_no_bridge_signal",
            "unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal",
            "unified_gl_rcedr_v1_adaptive_support_span",
            "unified_gl_rcedr_v1_adaptive_support_span_weak_bridge",
            "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
            "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty",
            "unified_gl_rcedr_v1_adaptive_support_span_span40",
            "unified_gl_rcedr_no_dynamic_control",
            "unified_gl_rcedr_no_stability",
            "unified_gl_rcedr_no_bridge_path",
            "unified_gl_rcedr_density_first_ablation",
            "unified_gl_rcedr_v2",
            "unified_gl_rcedr_v2_no_adaptive_bridge",
            "unified_gl_rcedr_v2_no_cost",
            "unified_gl_rcedr_v2_no_redundancy",
            "unified_gl_rcedr_v2_density_first",
        }:
            assert cfg.gl_rcedr_enabled is True
            assert cfg.answer_support_pinning_enabled is False
            assert cfg.corridor_answer_preserve_guarded_hotpot_enabled is False
            assert cfg.oracle_support_injection_enabled is False
            assert cfg.dataset_specific_branch_enabled is False
        if profile in {
            "unified_gl_rcedr_v2",
            "unified_gl_rcedr_v2_no_adaptive_bridge",
            "unified_gl_rcedr_v2_no_cost",
            "unified_gl_rcedr_v2_no_redundancy",
            "unified_gl_rcedr_v2_density_first",
        }:
            assert cfg.unified_marginal_utility_enabled is True
            assert cfg.gl_rcedr_v2_evidence_gain_weight == 0.52
            assert cfg.gl_rcedr_v2_bridge_gain_weight == 0.28
            assert cfg.gl_rcedr_v2_redundancy_weight == 0.22
            assert cfg.gl_rcedr_v2_cost_weight == 0.14
            assert cfg.gl_rcedr_v2_max_selected_candidates == 24
            assert cfg.gl_rcedr_v2_max_rendered_candidates == 24
            assert cfg.gl_rcedr_v2_max_rendered_tokens == 360
        if profile == "unified_gl_rcedr_v2":
            assert cfg.gl_rcedr_v2_adaptive_bridge_enabled is True
            assert cfg.gl_rcedr_v2_cost_enabled is True
            assert cfg.gl_rcedr_v2_redundancy_enabled is True
            assert cfg.gl_rcedr_v2_density_first_enabled is False
        if profile == "unified_gl_rcedr_v2_no_adaptive_bridge":
            assert cfg.gl_rcedr_v2_adaptive_bridge_enabled is False
            assert cfg.gl_rcedr_v2_cost_enabled is True
            assert cfg.gl_rcedr_v2_redundancy_enabled is True
            assert cfg.gl_rcedr_v2_density_first_enabled is False
        if profile == "unified_gl_rcedr_v2_no_cost":
            assert cfg.gl_rcedr_v2_adaptive_bridge_enabled is True
            assert cfg.gl_rcedr_v2_cost_enabled is False
            assert cfg.gl_rcedr_v2_redundancy_enabled is True
            assert cfg.gl_rcedr_v2_density_first_enabled is False
        if profile == "unified_gl_rcedr_v2_no_redundancy":
            assert cfg.gl_rcedr_v2_adaptive_bridge_enabled is True
            assert cfg.gl_rcedr_v2_cost_enabled is True
            assert cfg.gl_rcedr_v2_redundancy_enabled is False
            assert cfg.gl_rcedr_v2_density_first_enabled is False
        if profile == "unified_gl_rcedr_v2_density_first":
            assert cfg.gl_rcedr_v2_adaptive_bridge_enabled is True
            assert cfg.gl_rcedr_v2_cost_enabled is True
            assert cfg.gl_rcedr_v2_redundancy_enabled is True
            assert cfg.gl_rcedr_v2_density_first_enabled is True
        if profile == "unified_gl_rcedr_v1":
            assert cfg.gl_rcedr_dynamic_control_enabled is True
            assert cfg.gl_rcedr_stability_enabled is True
            assert cfg.gl_rcedr_bridge_path_enabled is True
            assert cfg.gl_rcedr_density_first_ablation_enabled is False
        if profile == "unified_gl_rcedr_v1_sentence_contract":
            assert cfg.sentence_contract_render_enabled is True
            assert cfg.sentence_contract_max_item_tokens == 32
            assert cfg.sentence_contract_metadata_pruning is True
            assert cfg.sentence_contract_chunk_expansion_allowed is False
            assert cfg.sentence_contract_preserve_selected_items is True
        if profile == "unified_gl_rcedr_v1_sentence_contract_no_item_cap":
            assert cfg.sentence_contract_render_enabled is True
            assert cfg.sentence_contract_max_item_tokens is None
            assert cfg.sentence_contract_metadata_pruning is True
            assert cfg.sentence_contract_chunk_expansion_allowed is False
            assert cfg.sentence_contract_preserve_selected_items is True
        if profile == "unified_gl_rcedr_v1_sentence_contract_metadata_on":
            assert cfg.sentence_contract_render_enabled is True
            assert cfg.sentence_contract_max_item_tokens == 32
            assert cfg.sentence_contract_metadata_pruning is False
            assert cfg.sentence_contract_chunk_expansion_allowed is False
            assert cfg.sentence_contract_preserve_selected_items is True
        if profile == "unified_gl_rcedr_v1_sentence_contract_span40":
            assert cfg.sentence_contract_render_enabled is True
            assert cfg.sentence_contract_max_item_tokens == 40
            assert cfg.sentence_contract_metadata_pruning is True
            assert cfg.sentence_contract_chunk_expansion_allowed is False
            assert cfg.sentence_contract_preserve_selected_items is True
        if profile == "unified_gl_rcedr_v1_support_span_contract":
            assert cfg.sentence_contract_render_enabled is True
            assert cfg.support_span_contract_enabled is True
            assert cfg.support_span_max_item_tokens == 48
            assert cfg.support_span_use_bridge_signal is True
            assert cfg.support_span_use_query_entity_signal is True
            assert cfg.support_span_use_anchor_entity_signal is True
        if profile == "unified_gl_rcedr_v1_support_span_contract_span40":
            assert cfg.support_span_contract_enabled is True
            assert cfg.support_span_max_item_tokens == 40
        if profile == "unified_gl_rcedr_v1_support_span_contract_no_cap":
            assert cfg.support_span_contract_enabled is True
            assert cfg.support_span_max_item_tokens is None
        if profile == "unified_gl_rcedr_v1_support_span_contract_no_bridge_signal":
            assert cfg.support_span_contract_enabled is True
            assert cfg.support_span_use_bridge_signal is False
        if profile == "unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal":
            assert cfg.support_span_contract_enabled is True
            assert cfg.support_span_use_query_entity_signal is False
            assert cfg.support_span_use_anchor_entity_signal is False
        if profile == "unified_gl_rcedr_v1_adaptive_support_span":
            assert cfg.adaptive_support_span_enabled is True
            assert cfg.adaptive_support_span_hard_cap_enabled is False
            assert cfg.adaptive_support_span_use_bridge_signal is True
            assert cfg.adaptive_support_span_bridge_mode == "conditional"
        if profile == "unified_gl_rcedr_v1_adaptive_support_span_weak_bridge":
            assert cfg.adaptive_support_span_enabled is True
            assert cfg.adaptive_support_span_bridge_mode == "weak"
        if profile == "unified_gl_rcedr_v1_adaptive_support_span_no_bridge":
            assert cfg.adaptive_support_span_enabled is True
            assert cfg.adaptive_support_span_use_bridge_signal is False
        if profile == "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty":
            assert cfg.adaptive_support_span_enabled is True
            assert cfg.adaptive_support_span_soft_length_penalty_enabled is False
        if profile == "unified_gl_rcedr_v1_adaptive_support_span_span40":
            assert cfg.adaptive_support_span_enabled is True
            assert cfg.adaptive_support_span_hard_cap_enabled is True
            assert cfg.adaptive_support_span_max_item_tokens == 40
        if profile == "unified_gl_rcedr_no_dynamic_control":
            assert cfg.gl_rcedr_dynamic_control_enabled is False
            assert cfg.gl_rcedr_stability_enabled is True
            assert cfg.gl_rcedr_bridge_path_enabled is True
        if profile == "unified_gl_rcedr_no_stability":
            assert cfg.gl_rcedr_dynamic_control_enabled is True
            assert cfg.gl_rcedr_stability_enabled is False
            assert cfg.gl_rcedr_bridge_path_enabled is True
        if profile == "unified_gl_rcedr_no_bridge_path":
            assert cfg.gl_rcedr_dynamic_control_enabled is True
            assert cfg.gl_rcedr_stability_enabled is True
            assert cfg.gl_rcedr_bridge_path_enabled is False
        if profile == "unified_gl_rcedr_density_first_ablation":
            assert cfg.gl_rcedr_density_first_ablation_enabled is True
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
