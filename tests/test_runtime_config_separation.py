from effirag.unified_copy_span_policy import (
    UNIFIED_METHOD_SETTING_KEYS,
    apply_unified_profile,
    unified_method_signature,
)


def test_precomputed_retrieval_strict_is_not_a_unified_method_lock():
    assert "precomputed_retrieval_strict" not in UNIFIED_METHOD_SETTING_KEYS
    assert "precomputed_retrieval_path" not in UNIFIED_METHOD_SETTING_KEYS


def test_runtime_precomputed_fields_do_not_change_unified_method_signature():
    cfg_a = apply_unified_profile({}, "hotpotqa", "unified_medium")
    cfg_b = dict(cfg_a)
    cfg_b["precomputed_retrieval_path"] = "outputs/some/replayed/query_results.jsonl"
    cfg_b["precomputed_retrieval_strict"] = True

    assert unified_method_signature(cfg_a) == unified_method_signature(cfg_b)
