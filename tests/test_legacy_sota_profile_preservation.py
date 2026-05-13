from effirag.legacy_sota_policy import (
    LEGACY_COPY_SPAN_DATASET_LOCKS,
    legacy_sota_copy_span_locks,
    resolve_legacy_sota_copy_span_mode,
)


def test_hotpotqa_legacy_guarded_answer_preserve_is_preserved():
    locks = legacy_sota_copy_span_locks("hotpotqa")
    assert resolve_legacy_sota_copy_span_mode("hotpotqa") == "p3_answer_preserve_guarded_hotpot"
    assert locks["retrieval_objective_mode"] == "p3_answer_preserve_guarded_hotpot"
    assert locks["answer_support_pinning_enabled"] is True
    assert locks["final_top_slice_reorder_enabled"] is False


def test_2wiki_legacy_final_top_slice_reorder_is_preserved():
    locks = legacy_sota_copy_span_locks("2wikimultihopqa")
    assert locks["retrieval_objective_mode"] == "baseline"
    assert locks["answer_support_pinning_enabled"] is False
    assert locks["final_top_slice_reorder_enabled"] is True


def test_musique_popqa_legacy_compact_behavior_is_preserved():
    for dataset in ("musique", "popqa"):
        locks = legacy_sota_copy_span_locks(dataset)
        assert locks["retrieval_objective_mode"] == "baseline"
        assert locks["answer_support_pinning_enabled"] is False
        assert locks["final_top_slice_reorder_enabled"] is False
        assert locks["precomputed_retrieval_strict"] is False


def test_legacy_dataset_locks_are_explicitly_named_legacy():
    assert set(LEGACY_COPY_SPAN_DATASET_LOCKS) == {
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "popqa",
    }
