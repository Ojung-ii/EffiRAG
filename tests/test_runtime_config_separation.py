from effirag.unified_copy_span_policy import (
    UNIFIED_METHOD_SETTING_KEYS,
    apply_unified_profile,
    unified_method_signature,
)


def test_precomputed_retrieval_strict_is_not_a_unified_method_lock():
    assert "precomputed_retrieval_strict" not in UNIFIED_METHOD_SETTING_KEYS
    assert "precomputed_retrieval_path" not in UNIFIED_METHOD_SETTING_KEYS


def test_runtime_precomputed_fields_do_not_change_unified_method_signature():
    profiles = [
        "unified_medium",
        "unified_large_lightsep",
        "unified_large_reorder_all",
        "unified_large_lightsep_reorder_all",
        "unified_medium_lightsep_reorder_all",
        "unified_dynamic_compact_v1",
        "unified_dynamic_compact_v2",
        "unified_dynamic_contextual_compact_v3",
        "unified_candidate_recall_boost_v1",
        "unified_candidate_recall_boost_dynamic_v1",
    ]

    for profile in profiles:
        cfg_a = apply_unified_profile({}, "hotpotqa", profile)
        cfg_b = dict(cfg_a)
        cfg_b["precomputed_retrieval_path"] = "outputs/some/replayed/query_results.jsonl"
        cfg_b["precomputed_retrieval_strict"] = True

        assert unified_method_signature(cfg_a) == unified_method_signature(cfg_b)
