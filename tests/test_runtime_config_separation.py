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

    for profile in profiles:
        cfg_a = apply_unified_profile({}, "hotpotqa", profile)
        cfg_b = dict(cfg_a)
        cfg_b["precomputed_retrieval_path"] = "outputs/some/replayed/query_results.jsonl"
        cfg_b["precomputed_retrieval_strict"] = True

        assert unified_method_signature(cfg_a) == unified_method_signature(cfg_b)
