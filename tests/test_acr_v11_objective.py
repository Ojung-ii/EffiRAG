from __future__ import annotations

import inspect

import effirag.answerability_selection as acr
from effirag.unified_copy_span_policy import DATASET_ORDER, apply_unified_profile


def test_acr_v11_main_objective_is_dataset_invariant_and_guarded():
    first = apply_unified_profile({}, DATASET_ORDER[0], "unified_acr_v11")
    for dataset in DATASET_ORDER:
        cfg = apply_unified_profile({}, dataset, "unified_acr_v11")
        assert cfg["gl_rcedr_enabled"] is True
        assert cfg["answerability_selection_enabled"] is True
        assert cfg["answerability_selection_mode"] == "greedy"
        assert cfg["answerability_selection_use_answerability"] is True
        assert cfg["answerability_selection_use_structure"] is True
        assert cfg["answerability_selection_use_cost_penalty"] is True
        assert cfg["answerability_selection_use_noise_penalty"] is False
        assert cfg["answerability_selection_ordering_enabled"] is False
        assert cfg["answer_support_pinning_enabled"] is False
        assert cfg["corridor_answer_preserve_guarded_hotpot_enabled"] is False
        assert cfg["oracle_support_injection_enabled"] is False
        assert cfg["dataset_specific_branch_enabled"] is False
        assert cfg["answerability_selection_enabled"] is first["answerability_selection_enabled"]
        assert cfg["answerability_selection_mode"] == first["answerability_selection_mode"]
        assert cfg["answerability_selection_use_answerability"] is first[
            "answerability_selection_use_answerability"
        ]
        assert cfg["answerability_selection_use_structure"] is first["answerability_selection_use_structure"]
        assert cfg["answerability_selection_use_cost_penalty"] is first["answerability_selection_use_cost_penalty"]
        assert cfg["answerability_selection_use_noise_penalty"] is first["answerability_selection_use_noise_penalty"]
        assert cfg["answerability_selection_ordering_enabled"] is first[
            "answerability_selection_ordering_enabled"
        ]


def test_acr_v11_selection_code_does_not_use_dataset_name():
    src = inspect.getsource(acr.apply_answerability_constrained_selection).lower()
    for banned in ("hotpotqa", "2wikimultihopqa", "musique", "popqa"):
        assert banned not in src
