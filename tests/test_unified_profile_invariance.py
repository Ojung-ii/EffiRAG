from effirag.unified_copy_span_policy import (
    DATASET_ORDER,
    UNIFIED_ENHANCED_PROFILE_NAMES,
    UNIFIED_BUDGET_CONFIG_KEYS,
    UNIFIED_CLEAN_PROFILE_NAMES,
    UNIFIED_DYNAMIC_PROFILE_NAMES,
    UNIFIED_RENDERING_LOCK,
    apply_unified_profile,
    unified_budget_signature,
    unified_method_signature,
)


PROFILES = [
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

REORDER_ALL_PROFILES = {
    "unified_large_reorder_all",
    "unified_large_lightsep_reorder_all",
    "unified_medium_lightsep_reorder_all",
}

LIGHTSEP_PROFILES = {
    "unified_large",
    "unified_medium",
    "unified_compact",
    "unified_large_lightsep",
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
    "unified_gl_rcedr_no_dynamic_control",
    "unified_gl_rcedr_no_stability",
    "unified_gl_rcedr_no_bridge_path",
    "unified_gl_rcedr_density_first_ablation",
    "unified_gl_rcedr_v2",
    "unified_gl_rcedr_v2_no_adaptive_bridge",
    "unified_gl_rcedr_v2_no_cost",
    "unified_gl_rcedr_v2_no_redundancy",
    "unified_gl_rcedr_v2_density_first",
}

DYNAMIC_PROFILES = {
    "unified_dynamic_compact_v1",
    "unified_dynamic_compact_v2",
    "unified_dynamic_contextual_compact_v3",
    "unified_candidate_recall_boost_dynamic_v1",
}

CANDIDATE_RECALL_BOOST_PROFILES = {
    "unified_candidate_recall_boost_v1",
    "unified_candidate_recall_boost_dynamic_v1",
    "unified_candidate_recall_boost_density_rerank_v1",
    "unified_gl_rcedr_v1",
    "unified_gl_rcedr_v1_sentence_contract",
    "unified_gl_rcedr_v1_sentence_contract_no_item_cap",
    "unified_gl_rcedr_v1_sentence_contract_metadata_on",
    "unified_gl_rcedr_v1_sentence_contract_span40",
    "unified_gl_rcedr_no_dynamic_control",
    "unified_gl_rcedr_no_stability",
    "unified_gl_rcedr_no_bridge_path",
    "unified_gl_rcedr_density_first_ablation",
    "unified_gl_rcedr_v2",
    "unified_gl_rcedr_v2_no_adaptive_bridge",
    "unified_gl_rcedr_v2_no_cost",
    "unified_gl_rcedr_v2_no_redundancy",
    "unified_gl_rcedr_v2_density_first",
}

GL_RCEDR_PROFILES = {
    "unified_gl_rcedr_v1",
    "unified_gl_rcedr_v1_sentence_contract",
    "unified_gl_rcedr_v1_sentence_contract_no_item_cap",
    "unified_gl_rcedr_v1_sentence_contract_metadata_on",
    "unified_gl_rcedr_v1_sentence_contract_span40",
    "unified_gl_rcedr_no_dynamic_control",
    "unified_gl_rcedr_no_stability",
    "unified_gl_rcedr_no_bridge_path",
    "unified_gl_rcedr_density_first_ablation",
    "unified_gl_rcedr_v2",
    "unified_gl_rcedr_v2_no_adaptive_bridge",
    "unified_gl_rcedr_v2_no_cost",
    "unified_gl_rcedr_v2_no_redundancy",
    "unified_gl_rcedr_v2_density_first",
}

GL_RCEDR_V2_PROFILES = {
    "unified_gl_rcedr_v2",
    "unified_gl_rcedr_v2_no_adaptive_bridge",
    "unified_gl_rcedr_v2_no_cost",
    "unified_gl_rcedr_v2_no_redundancy",
    "unified_gl_rcedr_v2_density_first",
}


def _expected_reorder(profile):
    return profile in REORDER_ALL_PROFILES


def test_unified_method_settings_are_identical_across_datasets():
    for profile in PROFILES:
        configs = {
            dataset: apply_unified_profile({}, dataset, profile)
            for dataset in DATASET_ORDER
        }
        first = unified_method_signature(configs[DATASET_ORDER[0]])
        for dataset, cfg in configs.items():
            assert cfg["retrieval_objective_mode"] == "baseline"
            for key, expected in UNIFIED_RENDERING_LOCK.items():
                if key == "final_top_slice_reorder_enabled":
                    expected = _expected_reorder(profile)
                assert cfg[key] is expected, f"{profile}.{dataset}.{key} drifted"
            assert unified_method_signature(cfg) == first, f"method signature drifted for {dataset}"


def test_unified_budget_is_profile_selected_not_dataset_selected():
    observed_by_profile = {}
    for profile in PROFILES:
        configs = {
            dataset: apply_unified_profile({}, dataset, profile)
            for dataset in DATASET_ORDER
        }
        first = unified_budget_signature(configs[DATASET_ORDER[0]])
        for dataset, cfg in configs.items():
            assert unified_budget_signature(cfg) == first, f"budget drifted for {profile}.{dataset}"
            for key in UNIFIED_BUDGET_CONFIG_KEYS:
                assert int(cfg[key]) == int(first[key])
        observed_by_profile[profile] = tuple(first[key] for key in UNIFIED_BUDGET_CONFIG_KEYS)

    assert observed_by_profile["unified_large"] == (45, 24, 24)
    assert observed_by_profile["unified_medium"] == (36, 18, 18)
    assert observed_by_profile["unified_compact"] == (30, 15, 15)
    assert observed_by_profile["unified_large_lightsep"] == (45, 24, 24)
    assert observed_by_profile["unified_large_reorder_all"] == (45, 24, 24)
    assert observed_by_profile["unified_large_lightsep_reorder_all"] == (45, 24, 24)
    assert observed_by_profile["unified_medium_lightsep_reorder_all"] == (36, 18, 18)
    assert observed_by_profile["unified_dynamic_compact_v1"] == (45, 24, 24)
    assert observed_by_profile["unified_dynamic_compact_v2"] == (45, 24, 24)
    assert observed_by_profile["unified_dynamic_contextual_compact_v3"] == (45, 24, 24)
    assert observed_by_profile["unified_candidate_recall_boost_v1"] == (45, 24, 24)
    assert observed_by_profile["unified_candidate_recall_boost_dynamic_v1"] == (45, 24, 24)
    assert observed_by_profile["unified_candidate_recall_boost_density_rerank_v1"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v1"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v1_sentence_contract"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v1_sentence_contract_no_item_cap"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v1_sentence_contract_metadata_on"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v1_sentence_contract_span40"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_no_dynamic_control"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_no_stability"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_no_bridge_path"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_density_first_ablation"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v2"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v2_no_adaptive_bridge"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v2_no_cost"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v2_no_redundancy"] == (45, 24, 24)
    assert observed_by_profile["unified_gl_rcedr_v2_density_first"] == (45, 24, 24)


def test_enhanced_profile_policy_flags_are_global():
    assert {
        "copy_span_instruction_unified_large_lightsep",
        "copy_span_instruction_unified_large_reorder_all",
        "copy_span_instruction_unified_large_lightsep_reorder_all",
        "copy_span_instruction_unified_medium_lightsep_reorder_all",
        "copy_span_instruction_unified_candidate_recall_boost_v1",
        "copy_span_instruction_unified_candidate_recall_boost_density_rerank_v1",
    }.issubset(set(UNIFIED_ENHANCED_PROFILE_NAMES))
    assert {
        "copy_span_instruction_unified_large",
        "copy_span_instruction_unified_medium",
        "copy_span_instruction_unified_compact",
    }.issubset(set(UNIFIED_CLEAN_PROFILE_NAMES))
    assert {
        "copy_span_instruction_unified_dynamic_compact_v1",
        "copy_span_instruction_unified_dynamic_compact_v2",
        "copy_span_instruction_unified_dynamic_contextual_compact_v3",
        "copy_span_instruction_unified_candidate_recall_boost_dynamic_v1",
        "copy_span_instruction_unified_gl_rcedr_v1",
        "copy_span_instruction_unified_gl_rcedr_v1_sentence_contract",
        "copy_span_instruction_unified_gl_rcedr_v1_sentence_contract_no_item_cap",
        "copy_span_instruction_unified_gl_rcedr_v1_sentence_contract_metadata_on",
        "copy_span_instruction_unified_gl_rcedr_v1_sentence_contract_span40",
        "copy_span_instruction_unified_gl_rcedr_no_dynamic_control",
        "copy_span_instruction_unified_gl_rcedr_no_stability",
        "copy_span_instruction_unified_gl_rcedr_no_bridge_path",
        "copy_span_instruction_unified_gl_rcedr_density_first_ablation",
        "copy_span_instruction_unified_gl_rcedr_v2",
        "copy_span_instruction_unified_gl_rcedr_v2_no_adaptive_bridge",
        "copy_span_instruction_unified_gl_rcedr_v2_no_cost",
        "copy_span_instruction_unified_gl_rcedr_v2_no_redundancy",
        "copy_span_instruction_unified_gl_rcedr_v2_density_first",
    }.issubset(set(UNIFIED_DYNAMIC_PROFILE_NAMES))

    for profile in PROFILES:
        configs = {
            dataset: apply_unified_profile({}, dataset, profile)
            for dataset in DATASET_ORDER
        }
        first = configs[DATASET_ORDER[0]]
        for dataset, cfg in configs.items():
            assert cfg["retrieval_objective_mode"] == "baseline"
            assert cfg["answer_support_pinning_enabled"] is False
            assert cfg["corridor_answer_preserve_guarded_hotpot_enabled"] is False
            assert cfg["oracle_support_injection_enabled"] is False
            assert cfg["final_top_slice_reorder_enabled"] is _expected_reorder(profile)
            assert cfg["order_strategy"] == first["order_strategy"]
            assert cfg["prompt_variant"] == first["prompt_variant"]
            assert cfg["dynamic_compact_selection_enabled"] is (profile in DYNAMIC_PROFILES)
            if profile in CANDIDATE_RECALL_BOOST_PROFILES:
                assert cfg["bridge_candidate_induction_enabled"] is True
                assert cfg["path_candidate_expansion_enabled"] is True
                assert cfg["anchor_expansion_enabled"] is True
                assert cfg["candidate_diversity_enabled"] is True
                assert cfg["entity_diversity_enabled"] is True
                assert cfg["source_diversity_enabled"] is True
                assert cfg["max_bridge_candidates"] == 24
                assert cfg["max_path_candidates"] == 24
                assert cfg["max_anchor_expansion_hops"] == 1
                assert cfg["max_expanded_candidates"] == 64
                assert cfg["candidate_dedup_enabled"] is True
                assert cfg["dataset_specific_branch_enabled"] is False
            if profile in GL_RCEDR_PROFILES:
                assert cfg["gl_rcedr_enabled"] is True
                assert cfg["retrieval_objective_mode"] == "baseline"
                assert cfg["answer_support_pinning_enabled"] is False
                assert cfg["corridor_answer_preserve_guarded_hotpot_enabled"] is False
                assert cfg["oracle_support_injection_enabled"] is False
                assert cfg["dataset_specific_branch_enabled"] is False
                assert cfg["gl_rcedr_view_count"] == 6
                assert cfg["gl_rcedr_seed_topk"] == 6
                assert cfg["gl_rcedr_recall_weight"] == 0.34
                assert cfg["gl_rcedr_bridge_path_weight"] == 0.22
                assert cfg["gl_rcedr_diversity_weight"] == 0.16
                assert cfg["gl_rcedr_stability_weight"] == 0.12
                assert cfg["gl_rcedr_redundancy_weight"] == 0.16
                assert cfg["gl_rcedr_cost_weight"] == 0.12
                assert cfg["gl_rcedr_max_selected_candidates"] == 24
                assert cfg["gl_rcedr_max_rendered_candidates"] == 24
                assert cfg["gl_rcedr_max_rendered_tokens"] == 360
                assert cfg["gl_rcedr_core_preserve_threshold"] == 0.56
                assert cfg["gl_rcedr_bridge_preserve_threshold"] == 0.46
                assert cfg["gl_rcedr_coverage_preserve_threshold"] == 0.38
            if profile == "unified_gl_rcedr_v1":
                assert cfg["gl_rcedr_dynamic_control_enabled"] is True
                assert cfg["gl_rcedr_stability_enabled"] is True
                assert cfg["gl_rcedr_bridge_path_enabled"] is True
                assert cfg["gl_rcedr_density_first_ablation_enabled"] is False
            if profile == "unified_gl_rcedr_v1_sentence_contract":
                assert cfg["gl_rcedr_dynamic_control_enabled"] is True
                assert cfg["gl_rcedr_stability_enabled"] is True
                assert cfg["gl_rcedr_bridge_path_enabled"] is True
                assert cfg["gl_rcedr_density_first_ablation_enabled"] is False
                assert cfg["sentence_contract_render_enabled"] is True
                assert cfg["sentence_contract_max_item_tokens"] == 32
                assert cfg["sentence_contract_metadata_pruning"] is True
                assert cfg["sentence_contract_chunk_expansion_allowed"] is False
                assert cfg["sentence_contract_preserve_selected_items"] is True
                assert cfg["sentence_contract_minimal_span_fallback"] is True
                assert cfg["sentence_contract_log_diagnostics"] is True
            if profile == "unified_gl_rcedr_v1_sentence_contract_no_item_cap":
                assert cfg["sentence_contract_render_enabled"] is True
                assert cfg["sentence_contract_max_item_tokens"] is None
                assert cfg["sentence_contract_metadata_pruning"] is True
                assert cfg["sentence_contract_chunk_expansion_allowed"] is False
                assert cfg["sentence_contract_preserve_selected_items"] is True
                assert cfg["sentence_contract_minimal_span_fallback"] is True
            if profile == "unified_gl_rcedr_v1_sentence_contract_metadata_on":
                assert cfg["sentence_contract_render_enabled"] is True
                assert cfg["sentence_contract_max_item_tokens"] == 32
                assert cfg["sentence_contract_metadata_pruning"] is False
                assert cfg["sentence_contract_chunk_expansion_allowed"] is False
                assert cfg["sentence_contract_preserve_selected_items"] is True
                assert cfg["sentence_contract_minimal_span_fallback"] is True
            if profile == "unified_gl_rcedr_v1_sentence_contract_span40":
                assert cfg["sentence_contract_render_enabled"] is True
                assert cfg["sentence_contract_max_item_tokens"] == 40
                assert cfg["sentence_contract_metadata_pruning"] is True
                assert cfg["sentence_contract_chunk_expansion_allowed"] is False
                assert cfg["sentence_contract_preserve_selected_items"] is True
                assert cfg["sentence_contract_minimal_span_fallback"] is True
            if profile == "unified_gl_rcedr_no_dynamic_control":
                assert cfg["gl_rcedr_dynamic_control_enabled"] is False
                assert cfg["gl_rcedr_stability_enabled"] is True
                assert cfg["gl_rcedr_bridge_path_enabled"] is True
                assert cfg["gl_rcedr_density_first_ablation_enabled"] is False
            if profile == "unified_gl_rcedr_no_stability":
                assert cfg["gl_rcedr_dynamic_control_enabled"] is True
                assert cfg["gl_rcedr_stability_enabled"] is False
                assert cfg["gl_rcedr_bridge_path_enabled"] is True
                assert cfg["gl_rcedr_density_first_ablation_enabled"] is False
            if profile == "unified_gl_rcedr_no_bridge_path":
                assert cfg["gl_rcedr_dynamic_control_enabled"] is True
                assert cfg["gl_rcedr_stability_enabled"] is True
                assert cfg["gl_rcedr_bridge_path_enabled"] is False
                assert cfg["gl_rcedr_density_first_ablation_enabled"] is False
            if profile == "unified_gl_rcedr_density_first_ablation":
                assert cfg["gl_rcedr_dynamic_control_enabled"] is True
                assert cfg["gl_rcedr_stability_enabled"] is True
                assert cfg["gl_rcedr_bridge_path_enabled"] is True
                assert cfg["gl_rcedr_density_first_ablation_enabled"] is True
            if profile in GL_RCEDR_V2_PROFILES:
                assert cfg["unified_marginal_utility_enabled"] is True
                assert cfg["gl_rcedr_v2_evidence_gain_weight"] == 0.52
                assert cfg["gl_rcedr_v2_bridge_gain_weight"] == 0.28
                assert cfg["gl_rcedr_v2_redundancy_weight"] == 0.22
                assert cfg["gl_rcedr_v2_cost_weight"] == 0.14
                assert cfg["gl_rcedr_v2_seed_stability_weight"] == 0.15
                assert cfg["gl_rcedr_v2_seed_diversity_weight"] == 0.15
                assert cfg["gl_rcedr_v2_bridge_lambda_base"] == 1.0
                assert cfg["gl_rcedr_v2_bridge_lambda_alpha"] == 0.30
                assert cfg["gl_rcedr_v2_bridge_lambda_min"] == 0.85
                assert cfg["gl_rcedr_v2_bridge_lambda_max"] == 1.35
                assert cfg["gl_rcedr_v2_max_selected_candidates"] == 24
                assert cfg["gl_rcedr_v2_max_rendered_candidates"] == 24
                assert cfg["gl_rcedr_v2_max_rendered_tokens"] == 360
                assert cfg["gl_rcedr_v2_core_preserve_threshold"] == 0.56
                assert cfg["gl_rcedr_v2_bridge_preserve_threshold"] == 0.46
            if profile == "unified_gl_rcedr_v2":
                assert cfg["gl_rcedr_v2_adaptive_bridge_enabled"] is True
                assert cfg["gl_rcedr_v2_cost_enabled"] is True
                assert cfg["gl_rcedr_v2_redundancy_enabled"] is True
                assert cfg["gl_rcedr_v2_density_first_enabled"] is False
            if profile == "unified_gl_rcedr_v2_no_adaptive_bridge":
                assert cfg["gl_rcedr_v2_adaptive_bridge_enabled"] is False
                assert cfg["gl_rcedr_v2_cost_enabled"] is True
                assert cfg["gl_rcedr_v2_redundancy_enabled"] is True
                assert cfg["gl_rcedr_v2_density_first_enabled"] is False
            if profile == "unified_gl_rcedr_v2_no_cost":
                assert cfg["gl_rcedr_v2_adaptive_bridge_enabled"] is True
                assert cfg["gl_rcedr_v2_cost_enabled"] is False
                assert cfg["gl_rcedr_v2_redundancy_enabled"] is True
                assert cfg["gl_rcedr_v2_density_first_enabled"] is False
            if profile == "unified_gl_rcedr_v2_no_redundancy":
                assert cfg["gl_rcedr_v2_adaptive_bridge_enabled"] is True
                assert cfg["gl_rcedr_v2_cost_enabled"] is True
                assert cfg["gl_rcedr_v2_redundancy_enabled"] is False
                assert cfg["gl_rcedr_v2_density_first_enabled"] is False
            if profile == "unified_gl_rcedr_v2_density_first":
                assert cfg["gl_rcedr_v2_adaptive_bridge_enabled"] is True
                assert cfg["gl_rcedr_v2_cost_enabled"] is True
                assert cfg["gl_rcedr_v2_redundancy_enabled"] is True
                assert cfg["gl_rcedr_v2_density_first_enabled"] is True
            if profile in DYNAMIC_PROFILES:
                assert cfg["coverage_gain_enabled"] is True
                assert cfg["redundancy_penalty_enabled"] is True
                assert cfg["bridge_preserve_enabled"] is True
                assert cfg["path_preserve_enabled"] is True
                assert cfg["adaptive_stop_enabled"] is True
                assert cfg["max_render_topn"] == 24
            if profile == "unified_dynamic_compact_v1":
                assert cfg["min_render_topn"] == 6
                assert cfg["target_prompt_tokens"] == 600
                assert cfg["max_prompt_tokens"] == 700
                assert cfg["selector_aware_render_enabled"] is False
                assert cfg["render_selected_only"] is False
            if profile == "unified_dynamic_compact_v2":
                assert cfg["min_render_topn"] == 6
                assert cfg["target_prompt_tokens"] == 550
                assert cfg["max_prompt_tokens"] == 650
                assert cfg["selector_aware_render_enabled"] is True
                assert cfg["render_selected_only"] is True
                assert cfg["render_include_neighbor_sentences"] is False
                assert cfg["render_include_corridor_headers"] is False
                assert cfg["render_include_source_titles"] == "minimal"
                assert cfg["render_include_metadata"] == "minimal"
                assert cfg["render_deduplicate_selected_text"] is True
                assert cfg["render_enforce_actual_prompt_budget"] is True
            if profile == "unified_dynamic_contextual_compact_v3":
                assert cfg["min_render_topn"] == 8
                assert cfg["target_prompt_tokens"] == 750
                assert cfg["max_prompt_tokens"] == 850
                assert cfg["selector_aware_render_enabled"] is True
                assert cfg["render_selected_only"] is False
                assert cfg["render_selected_centered"] is True
                assert cfg["render_contextual_expansion_enabled"] is True
                assert cfg["render_conditional_neighbor_sentences"] is True
                assert cfg["render_bridge_context_enabled"] is True
                assert cfg["render_path_context_enabled"] is True
                assert cfg["render_include_source_titles"] == "minimal"
                assert cfg["render_include_metadata"] == "minimal"
                assert cfg["render_deduplicate_selected_text"] is True
                assert cfg["render_deduplicate_context_text"] is True
                assert cfg["render_enforce_actual_prompt_budget"] is True
                assert cfg["max_neighbors_per_selected"] == 1
                assert cfg["max_context_sentences_per_selected"] == 1
                assert cfg["max_bridge_context_sentences"] == 2
                assert cfg["max_path_context_sentences"] == 2
            if profile == "unified_candidate_recall_boost_dynamic_v1":
                assert cfg["min_render_topn"] == 6
                assert cfg["target_prompt_tokens"] == 600
                assert cfg["max_prompt_tokens"] == 700
                assert cfg["selector_aware_render_enabled"] is False
                assert cfg["render_selected_only"] is False
            if profile == "unified_candidate_recall_boost_density_rerank_v1":
                assert cfg["evidence_density_rerank_enabled"] is True
                assert cfg["bridge_path_utility_enabled"] is True
                assert cfg["coverage_gain_enabled"] is True
                assert cfg["redundancy_penalty_enabled"] is True
                assert cfg["token_cost_penalty_enabled"] is True
                assert cfg["density_budget_awareness_enabled"] is True
                assert cfg["density_semantic_weight"] == 0.42
                assert cfg["density_bridge_path_weight"] == 0.24
                assert cfg["density_coverage_weight"] == 0.24
                assert cfg["density_redundancy_weight"] == 0.18
                assert cfg["density_token_cost_weight"] == 0.22
                assert cfg["max_selected_candidates"] == 24
                assert cfg["max_rendered_candidates"] == 24
                assert cfg["max_rendered_tokens"] == 340


def test_enhanced_profile_interfaces_are_explicit():
    for profile in LIGHTSEP_PROFILES:
        cfg = apply_unified_profile({}, "hotpotqa", profile)
        assert cfg["order_strategy"] == "score+light_separator_render"
        assert cfg["prompt_variant"] == "light_separator_copy_span_instruction"

    reorder_only = apply_unified_profile({}, "hotpotqa", "unified_large_reorder_all")
    assert reorder_only["order_strategy"] == "score"
    assert reorder_only["prompt_variant"] == "default"


def test_dynamic_compact_profile_has_no_dataset_specific_method_drift():
    forbidden_drift_keys = [
        "retrieval_objective_mode",
        "bridge_candidate_induction_enabled",
        "path_candidate_expansion_enabled",
        "anchor_expansion_enabled",
        "candidate_diversity_enabled",
        "entity_diversity_enabled",
        "source_diversity_enabled",
        "max_bridge_candidates",
        "max_path_candidates",
        "max_anchor_expansion_hops",
        "max_expanded_candidates",
        "candidate_dedup_enabled",
        "dataset_specific_branch_enabled",
        "answer_support_pinning_enabled",
        "final_top_slice_reorder_enabled",
        "corridor_answer_preserve_guarded_hotpot_enabled",
        "oracle_support_injection_enabled",
        "dynamic_compact_selection_enabled",
        "coverage_gain_enabled",
        "redundancy_penalty_enabled",
        "bridge_preserve_enabled",
        "path_preserve_enabled",
        "adaptive_stop_enabled",
        "max_render_topn",
        "min_render_topn",
        "target_prompt_tokens",
        "max_prompt_tokens",
        "coverage_gain_threshold",
        "bridge_score_threshold",
        "redundancy_threshold",
        "marginal_gain_threshold",
        "evidence_density_rerank_enabled",
        "bridge_path_utility_enabled",
        "token_cost_penalty_enabled",
        "density_budget_awareness_enabled",
        "density_semantic_weight",
        "density_bridge_path_weight",
        "density_coverage_weight",
        "density_redundancy_weight",
        "density_token_cost_weight",
        "max_selected_candidates",
        "max_rendered_candidates",
        "max_rendered_tokens",
        "gl_rcedr_enabled",
        "gl_rcedr_dynamic_control_enabled",
        "gl_rcedr_stability_enabled",
        "gl_rcedr_bridge_path_enabled",
        "gl_rcedr_density_first_ablation_enabled",
        "gl_rcedr_view_count",
        "gl_rcedr_seed_topk",
        "gl_rcedr_recall_weight",
        "gl_rcedr_bridge_path_weight",
        "gl_rcedr_diversity_weight",
        "gl_rcedr_stability_weight",
        "gl_rcedr_redundancy_weight",
        "gl_rcedr_cost_weight",
        "gl_rcedr_max_selected_candidates",
        "gl_rcedr_max_rendered_candidates",
        "gl_rcedr_max_rendered_tokens",
        "gl_rcedr_core_preserve_threshold",
        "gl_rcedr_bridge_preserve_threshold",
        "gl_rcedr_coverage_preserve_threshold",
        "unified_marginal_utility_enabled",
        "gl_rcedr_v2_adaptive_bridge_enabled",
        "gl_rcedr_v2_cost_enabled",
        "gl_rcedr_v2_redundancy_enabled",
        "gl_rcedr_v2_density_first_enabled",
        "gl_rcedr_v2_evidence_gain_weight",
        "gl_rcedr_v2_bridge_gain_weight",
        "gl_rcedr_v2_redundancy_weight",
        "gl_rcedr_v2_cost_weight",
        "gl_rcedr_v2_seed_stability_weight",
        "gl_rcedr_v2_seed_diversity_weight",
        "gl_rcedr_v2_bridge_lambda_base",
        "gl_rcedr_v2_bridge_lambda_alpha",
        "gl_rcedr_v2_bridge_lambda_min",
        "gl_rcedr_v2_bridge_lambda_max",
        "gl_rcedr_v2_max_selected_candidates",
        "gl_rcedr_v2_max_rendered_candidates",
        "gl_rcedr_v2_max_rendered_tokens",
        "gl_rcedr_v2_core_preserve_threshold",
        "gl_rcedr_v2_bridge_preserve_threshold",
        "order_strategy",
        "prompt_variant",
        "top_corridors",
        "max_sentences",
        "max_context_sentences",
        "selector_aware_render_enabled",
        "render_selected_only",
        "render_selected_centered",
        "render_contextual_expansion_enabled",
        "render_conditional_neighbor_sentences",
        "render_bridge_context_enabled",
        "render_path_context_enabled",
        "render_include_neighbor_sentences",
        "render_include_corridor_headers",
        "render_include_source_titles",
        "render_include_metadata",
        "render_deduplicate_selected_text",
        "render_deduplicate_context_text",
        "render_enforce_actual_prompt_budget",
        "max_neighbors_per_selected",
        "max_context_sentences_per_selected",
        "max_bridge_context_sentences",
        "max_path_context_sentences",
    ]
    for profile in DYNAMIC_PROFILES.union(GL_RCEDR_V2_PROFILES):
        configs = {
            dataset: apply_unified_profile({}, dataset, profile)
            for dataset in DATASET_ORDER
        }
        first = configs[DATASET_ORDER[0]]
        for dataset, cfg in configs.items():
            for key in forbidden_drift_keys:
                assert cfg[key] == first[key], f"{profile}.{key} drifted for {dataset}"
