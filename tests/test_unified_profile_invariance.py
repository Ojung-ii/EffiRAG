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
}

DYNAMIC_PROFILES = {
    "unified_dynamic_compact_v1",
    "unified_dynamic_compact_v2",
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


def test_enhanced_profile_policy_flags_are_global():
    assert {
        "copy_span_instruction_unified_large_lightsep",
        "copy_span_instruction_unified_large_reorder_all",
        "copy_span_instruction_unified_large_lightsep_reorder_all",
        "copy_span_instruction_unified_medium_lightsep_reorder_all",
    }.issubset(set(UNIFIED_ENHANCED_PROFILE_NAMES))
    assert {
        "copy_span_instruction_unified_large",
        "copy_span_instruction_unified_medium",
        "copy_span_instruction_unified_compact",
    }.issubset(set(UNIFIED_CLEAN_PROFILE_NAMES))
    assert {
        "copy_span_instruction_unified_dynamic_compact_v1",
        "copy_span_instruction_unified_dynamic_compact_v2",
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
            if profile in DYNAMIC_PROFILES:
                assert cfg["coverage_gain_enabled"] is True
                assert cfg["redundancy_penalty_enabled"] is True
                assert cfg["bridge_preserve_enabled"] is True
                assert cfg["path_preserve_enabled"] is True
                assert cfg["adaptive_stop_enabled"] is True
                assert cfg["min_render_topn"] == 6
                assert cfg["max_render_topn"] == 24
            if profile == "unified_dynamic_compact_v1":
                assert cfg["target_prompt_tokens"] == 600
                assert cfg["max_prompt_tokens"] == 700
                assert cfg["selector_aware_render_enabled"] is False
                assert cfg["render_selected_only"] is False
            if profile == "unified_dynamic_compact_v2":
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
        "order_strategy",
        "prompt_variant",
        "top_corridors",
        "max_sentences",
        "max_context_sentences",
        "selector_aware_render_enabled",
        "render_selected_only",
        "render_include_neighbor_sentences",
        "render_include_corridor_headers",
        "render_include_source_titles",
        "render_include_metadata",
        "render_deduplicate_selected_text",
        "render_enforce_actual_prompt_budget",
    ]
    for profile in DYNAMIC_PROFILES:
        configs = {
            dataset: apply_unified_profile({}, dataset, profile)
            for dataset in DATASET_ORDER
        }
        first = configs[DATASET_ORDER[0]]
        for dataset, cfg in configs.items():
            for key in forbidden_drift_keys:
                assert cfg[key] == first[key], f"{profile}.{key} drifted for {dataset}"
