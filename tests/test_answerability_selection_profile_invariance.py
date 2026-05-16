from effirag.unified_copy_span_policy import DATASET_ORDER, apply_unified_profile


PHASE6P_PROFILES = [
    "unified_acr_v1",
    "unified_acr_v1_no_answerability",
    "unified_acr_v1_no_structure",
    "unified_acr_v1_no_noise_penalty",
    "unified_acr_v1_no_cost",
    "unified_acr_v1_no_ordering",
    "unified_acr_v1_beam3",
]


def test_phase6p_acr_profiles_are_dataset_invariant_and_guarded():
    for profile in PHASE6P_PROFILES:
        first = apply_unified_profile({}, DATASET_ORDER[0], profile)
        for dataset in DATASET_ORDER:
            cfg = apply_unified_profile({}, dataset, profile)
            assert cfg["gl_rcedr_enabled"] is True
            assert cfg["answerability_selection_enabled"] is True
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
            assert cfg["answerability_selection_use_noise_penalty"] is first[
                "answerability_selection_use_noise_penalty"
            ]
            assert cfg["answerability_selection_use_cost_penalty"] is first[
                "answerability_selection_use_cost_penalty"
            ]


def test_phase6p_acr_profile_specific_overrides():
    main_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_v1")
    no_answer_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_v1_no_answerability")
    no_structure_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_v1_no_structure")
    no_noise_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_v1_no_noise_penalty")
    no_cost_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_v1_no_cost")
    no_order_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_v1_no_ordering")
    beam_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_v1_beam3")

    assert main_cfg["answerability_selection_enabled"] is True
    assert main_cfg["answerability_selection_mode"] == "greedy"
    assert main_cfg["answerability_selection_use_answerability"] is True
    assert main_cfg["answerability_selection_use_structure"] is True
    assert main_cfg["answerability_selection_use_noise_penalty"] is True
    assert main_cfg["answerability_selection_use_cost_penalty"] is True
    assert main_cfg["answerability_selection_ordering_enabled"] is True

    assert no_answer_cfg["answerability_selection_use_answerability"] is False
    assert no_structure_cfg["answerability_selection_use_structure"] is False
    assert no_noise_cfg["answerability_selection_use_noise_penalty"] is False
    assert no_cost_cfg["answerability_selection_use_cost_penalty"] is False
    assert no_order_cfg["answerability_selection_ordering_enabled"] is False

    assert beam_cfg["answerability_selection_mode"] == "beam3"
    assert beam_cfg["answerability_selection_beam_size"] == 3
