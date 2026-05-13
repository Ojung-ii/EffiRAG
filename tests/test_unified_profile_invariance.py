from effirag.unified_copy_span_policy import (
    DATASET_ORDER,
    UNIFIED_BUDGET_CONFIG_KEYS,
    UNIFIED_RENDERING_LOCK,
    apply_unified_profile,
    unified_budget_signature,
    unified_method_signature,
)


PROFILES = ["unified_large", "unified_medium", "unified_compact"]


def test_unified_method_settings_are_identical_across_datasets():
    for profile in PROFILES:
        configs = {
            dataset: apply_unified_profile({}, dataset, profile)
            for dataset in DATASET_ORDER
        }
        first = unified_method_signature(configs[DATASET_ORDER[0]])
        for dataset, cfg in configs.items():
            assert cfg["retrieval_objective_mode"] in {"baseline", "canonical_copy_span_unified"}
            for key, expected in UNIFIED_RENDERING_LOCK.items():
                assert cfg[key] is expected, f"{dataset}.{key} drifted"
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
