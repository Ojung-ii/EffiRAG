from effirag.unified_copy_span_policy import (
    DATASET_ORDER,
    UNIFIED_ENHANCED_PROFILE_NAMES,
    UNIFIED_BUDGET_CONFIG_KEYS,
    UNIFIED_CLEAN_PROFILE_NAMES,
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


def test_enhanced_profile_interfaces_are_explicit():
    for profile in LIGHTSEP_PROFILES:
        cfg = apply_unified_profile({}, "hotpotqa", profile)
        assert cfg["order_strategy"] == "score+light_separator_render"
        assert cfg["prompt_variant"] == "light_separator_copy_span_instruction"

    reorder_only = apply_unified_profile({}, "hotpotqa", "unified_large_reorder_all")
    assert reorder_only["order_strategy"] == "score"
    assert reorder_only["prompt_variant"] == "default"
