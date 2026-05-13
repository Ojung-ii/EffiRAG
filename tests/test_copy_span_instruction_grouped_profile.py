import math

import pytest

from effirag.grouped_profiles import (
    COPY_SPAN_INSTRUCTION_GROUPED_V0,
    COPY_SPAN_INSTRUCTION_GROUPED_V1,
    COPY_SPAN_INSTRUCTION_MINIMAL_V1,
    INSTRUCTION_GROUPED_PROFILE_NAMES,
    REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
    apply_grouped_profile,
    instruction_reference_contract,
)
from effirag.profiles import COPY_SPAN_DATASET_LOCKS, expected_copy_span_config


def _assert_close_dict(lhs, rhs, keys, tol=1e-9):
    for key in keys:
        assert key in lhs, f"missing key in lhs: {key}"
        assert key in rhs, f"missing key in rhs: {key}"
        assert math.isclose(float(lhs[key]), float(rhs[key]), rel_tol=0.0, abs_tol=tol), (
            f"value mismatch for key={key}: lhs={lhs[key]} rhs={rhs[key]}"
        )


def test_instruction_v0_expands_to_reference_weights():
    cfg = apply_grouped_profile({}, "hotpotqa", COPY_SPAN_INSTRUCTION_GROUPED_V0)
    ref = instruction_reference_contract("hotpotqa")
    keys = [
        "seed_score_semantic_weight",
        "seed_score_graph_weight",
        "seed_score_anchor_weight",
        "run_score_semantic_weight",
        "run_score_anchor_weight",
        "run_score_structure_weight",
        "run_score_bridge_weight",
        "run_score_redundancy_weight",
        "embedding_weight",
        "alpha",
        "beta",
        "gamma_main",
        "delta_support",
        "eta_connector",
        "zeta_query",
        "xi_locality",
        "lambda_redundancy",
    ]
    _assert_close_dict(cfg, ref, keys)


def test_instruction_v0_preserves_order_strategy():
    cfg = apply_grouped_profile({}, "hotpotqa", COPY_SPAN_INSTRUCTION_GROUPED_V0)
    assert cfg["order_strategy"] == "score+light_separator_render"


def test_instruction_v0_preserves_prompt_variant():
    cfg = apply_grouped_profile({}, "hotpotqa", COPY_SPAN_INSTRUCTION_GROUPED_V0)
    assert cfg["prompt_variant"] == "light_separator_copy_span_instruction"
    assert cfg["prompt_variant_label"] == "light_separator_copy_span_instruction"


def test_instruction_v0_preserves_added_instruction():
    cfg = apply_grouped_profile({}, "hotpotqa", COPY_SPAN_INSTRUCTION_GROUPED_V0)
    assert cfg["added_instruction"] == "copy_span_only"


def test_instruction_v0_preserves_dataset_locks():
    for dataset in ("hotpotqa", "2wikimultihopqa"):
        cfg = apply_grouped_profile({}, dataset, COPY_SPAN_INSTRUCTION_GROUPED_V0)
        for key, value in COPY_SPAN_DATASET_LOCKS[dataset].items():
            assert cfg.get(key) == value, f"dataset lock mismatch: {dataset}.{key}"


def test_instruction_grouped_does_not_modify_champion_reconfirm():
    _ = apply_grouped_profile({}, "hotpotqa", COPY_SPAN_INSTRUCTION_GROUPED_V1)
    champion = expected_copy_span_config("hotpotqa")
    assert champion["order_strategy"] == "score"
    assert champion["canonical_variant_name"] == "champion_reconfirm"


def test_instruction_grouped_uses_explicit_reference_profile():
    with pytest.raises(ValueError):
        apply_grouped_profile(
            {},
            "hotpotqa",
            COPY_SPAN_INSTRUCTION_GROUPED_V0,
            reference_profile="champion_reconfirm",
        )
    cfg = apply_grouped_profile(
        {},
        "hotpotqa",
        REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
        reference_profile=REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
    )
    assert cfg["reference_profile"] == "light_separator_copy_span_instruction"


def test_instruction_redundancy_orientation_is_preserved():
    cfg = apply_grouped_profile({}, "hotpotqa", COPY_SPAN_INSTRUCTION_MINIMAL_V1)
    assert float(cfg["run_score_redundancy_weight"]) > 0.0
    assert float(cfg["lambda_redundancy"]) > 0.0


def test_instruction_profile_names_are_not_ambiguous_with_champion_grouped():
    old_names = {
        "copy_span_grouped_v0",
        "copy_span_grouped_v1",
        "copy_span_minimal_v1",
        "copy_span_grouped_no_dataset_toggle",
    }
    current = set(INSTRUCTION_GROUPED_PROFILE_NAMES)
    assert current.isdisjoint(old_names)
    for name in current:
        assert name.startswith("copy_span_instruction_")
