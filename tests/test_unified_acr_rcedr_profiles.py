from __future__ import annotations

from effirag.unified_copy_span_policy import DATASET_ORDER, apply_unified_profile


PHASE6S_PROFILES = [
    "unified_acr_rcedr_v12",
    "unified_acr_rcedr_v12_answerability_only",
    "unified_acr_rcedr_v12_no_bridge",
    "unified_acr_rcedr_v12_no_redundancy",
    "unified_acr_rcedr_v12_beam3",
    "unified_acr_rcedr_v12_no_sentence_rerank",
    "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
]


def test_unified_acr_rcedr_profiles_are_dataset_invariant_and_guarded():
    for profile in PHASE6S_PROFILES:
        first = apply_unified_profile({}, DATASET_ORDER[0], profile)
        for dataset in DATASET_ORDER:
            cfg = apply_unified_profile({}, dataset, profile)
            assert cfg["gl_rcedr_enabled"] is False
            assert cfg["answerability_selection_enabled"] is False
            assert cfg["unified_acr_rcedr_enabled"] is True
            assert cfg["evidence_density_rerank_enabled"] is False
            assert cfg["top1_correction_enabled"] is False
            assert cfg["answer_support_pinning_enabled"] is False
            assert cfg["corridor_answer_preserve_guarded_hotpot_enabled"] is False
            assert cfg["oracle_support_injection_enabled"] is False
            assert cfg["dataset_specific_branch_enabled"] is False
            assert cfg["unified_acr_rcedr_mode"] == first["unified_acr_rcedr_mode"]
            assert cfg["unified_acr_rcedr_use_answerability_gain"] is first[
                "unified_acr_rcedr_use_answerability_gain"
            ]
            assert cfg["unified_acr_rcedr_use_bridge_gain"] is first["unified_acr_rcedr_use_bridge_gain"]
            assert cfg["unified_acr_rcedr_use_redundancy_penalty"] is first[
                "unified_acr_rcedr_use_redundancy_penalty"
            ]
            assert cfg["unified_acr_rcedr_use_cost_penalty"] is first["unified_acr_rcedr_use_cost_penalty"]


def test_unified_acr_rcedr_profile_overrides_match_phase6s_contract():
    main_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_rcedr_v12")
    answerability_only_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_rcedr_v12_answerability_only")
    no_bridge_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_rcedr_v12_no_bridge")
    no_redundancy_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_rcedr_v12_no_redundancy")
    beam_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_rcedr_v12_beam3")
    no_rerank_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_rcedr_v12_no_sentence_rerank")
    beam_no_rerank_cfg = apply_unified_profile(
        {},
        "hotpotqa",
        "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
    )

    assert main_cfg["unified_acr_rcedr_mode"] == "greedy"
    assert main_cfg["unified_acr_rcedr_use_answerability_gain"] is True
    assert main_cfg["unified_acr_rcedr_use_bridge_gain"] is True
    assert main_cfg["unified_acr_rcedr_use_redundancy_penalty"] is True
    assert main_cfg["unified_acr_rcedr_use_cost_penalty"] is False

    assert answerability_only_cfg["unified_acr_rcedr_use_bridge_gain"] is False
    assert answerability_only_cfg["unified_acr_rcedr_use_redundancy_penalty"] is False
    assert answerability_only_cfg["unified_acr_rcedr_use_answerability_gain"] is True

    assert no_bridge_cfg["unified_acr_rcedr_use_bridge_gain"] is False
    assert no_bridge_cfg["unified_acr_rcedr_use_redundancy_penalty"] is True

    assert no_redundancy_cfg["unified_acr_rcedr_use_redundancy_penalty"] is False
    assert no_redundancy_cfg["unified_acr_rcedr_use_bridge_gain"] is True

    assert beam_cfg["unified_acr_rcedr_mode"] == "beam"
    assert beam_cfg["unified_acr_rcedr_beam_size"] == 3

    assert no_rerank_cfg["sentence_rerank_enabled"] is False

    assert beam_no_rerank_cfg["unified_acr_rcedr_mode"] == "beam"
    assert beam_no_rerank_cfg["unified_acr_rcedr_beam_size"] == 3
    assert beam_no_rerank_cfg["sentence_rerank_enabled"] is False
