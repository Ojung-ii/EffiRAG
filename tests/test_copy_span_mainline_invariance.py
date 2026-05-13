import inspect

import effirag.retrieval as retrieval
from effirag.profiles import (
    COPY_SPAN_BUDGET_AND_SHAPE,
    COPY_SPAN_DATASET_LOCKS,
    COPY_SPAN_DISABLED_MODULES,
    COPY_SPAN_RENDER_WEIGHTS,
    COPY_SPAN_RETRIEVAL_WEIGHTS,
    expected_copy_span_config,
)


def test_copy_span_expected_config_contains_locked_weights():
    expected = expected_copy_span_config("hotpotqa")
    for key, value in COPY_SPAN_RETRIEVAL_WEIGHTS.items():
        assert expected[key] == value, f"retrieval weight mismatch for '{key}'"
    for key, value in COPY_SPAN_RENDER_WEIGHTS.items():
        assert expected[key] == value, f"render weight mismatch for '{key}'"


def test_copy_span_expected_config_contains_dataset_locks():
    for dataset, dataset_locks in COPY_SPAN_DATASET_LOCKS.items():
        expected = expected_copy_span_config(dataset)
        for key, value in dataset_locks.items():
            assert expected[key] == value, f"dataset lock mismatch for '{dataset}.{key}'"


def test_copy_span_disabled_modules_are_false():
    expected = expected_copy_span_config("hotpotqa")
    for key in COPY_SPAN_DISABLED_MODULES:
        assert expected[key] is False, f"disabled module must remain false: '{key}'"


def test_copy_span_order_strategy_is_score():
    expected = expected_copy_span_config("hotpotqa")
    assert expected["order_strategy"] == "score"
    assert COPY_SPAN_BUDGET_AND_SHAPE["order_strategy"] == "score"


def test_hotpotqa_objective_profile_is_guarded_answer_preserve():
    expected = expected_copy_span_config("hotpotqa")
    assert expected["retrieval_objective_mode"] == "p3_answer_preserve_guarded_hotpot"


def test_2wiki_final_top_slice_reorder_is_enabled():
    expected = expected_copy_span_config("2wikimultihopqa")
    assert expected["final_top_slice_reorder_enabled"] is True


def test_musique_popqa_precomputed_strict_is_false():
    musique_expected = expected_copy_span_config("musique")
    popqa_expected = expected_copy_span_config("popqa")
    assert musique_expected["precomputed_retrieval_strict"] is False
    assert popqa_expected["precomputed_retrieval_strict"] is False


def test_retrieval_objective_profile_call_order_is_preserved():
    src = inspect.getsource(retrieval.run_graphrag_core)
    shared_budget_call = "shared_budget_diag = _apply_shared_budget_profile_once(cfg)"
    resolve_objective_call = "objective_flags = _resolve_retrieval_objective_flags(cfg)"
    apply_profile_call = "objective_flags, objective_profile_diag = _apply_connector_objective_profile("
    i_shared = src.find(shared_budget_call)
    i_resolve = src.find(resolve_objective_call)
    i_apply = src.find(apply_profile_call)
    assert i_shared >= 0, "missing shared budget profile call in run_graphrag_core"
    assert i_resolve >= 0, "missing retrieval objective resolution call in run_graphrag_core"
    assert i_apply >= 0, "missing connector objective profile call in run_graphrag_core"
    assert i_shared < i_resolve < i_apply, (
        "retrieval objective pipeline call order changed: "
        "_apply_shared_budget_profile_once -> _resolve_retrieval_objective_flags -> "
        "_apply_connector_objective_profile must remain in this order"
    )
