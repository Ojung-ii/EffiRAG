from effirag.unified_copy_span_policy import DATASET_ORDER, apply_unified_profile


PHASE6M_PROFILES = [
    "unified_gl_rcedr_v1_adaptive_support_span",
    "unified_gl_rcedr_v1_adaptive_support_span_weak_bridge",
    "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
    "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty",
    "unified_gl_rcedr_v1_adaptive_support_span_span40",
]


def test_phase6m_adaptive_profiles_are_dataset_invariant_and_guarded():
    for profile in PHASE6M_PROFILES:
        first = apply_unified_profile({}, DATASET_ORDER[0], profile)
        for dataset in DATASET_ORDER:
            cfg = apply_unified_profile({}, dataset, profile)
            assert cfg["gl_rcedr_enabled"] is True
            assert cfg["adaptive_support_span_enabled"] is True
            assert cfg["answer_support_pinning_enabled"] is False
            assert cfg["corridor_answer_preserve_guarded_hotpot_enabled"] is False
            assert cfg["oracle_support_injection_enabled"] is False
            assert cfg["dataset_specific_branch_enabled"] is False
            assert cfg["adaptive_support_span_enabled"] is first["adaptive_support_span_enabled"]
            assert cfg["adaptive_support_span_hard_cap_enabled"] is first["adaptive_support_span_hard_cap_enabled"]
            assert cfg["adaptive_support_span_use_bridge_signal"] is first["adaptive_support_span_use_bridge_signal"]
            assert cfg["adaptive_support_span_bridge_mode"] == first["adaptive_support_span_bridge_mode"]
            assert cfg["adaptive_support_span_soft_length_penalty_enabled"] is first[
                "adaptive_support_span_soft_length_penalty_enabled"
            ]


def test_phase6m_adaptive_profile_specific_overrides():
    main_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_adaptive_support_span")
    weak_bridge_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_adaptive_support_span_weak_bridge")
    no_bridge_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_adaptive_support_span_no_bridge")
    no_length_cfg = apply_unified_profile(
        {}, "hotpotqa", "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty"
    )
    span40_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_adaptive_support_span_span40")

    assert main_cfg["sentence_contract_render_enabled"] is True
    assert main_cfg["support_span_contract_enabled"] is False
    assert main_cfg["adaptive_support_span_enabled"] is True
    assert main_cfg["adaptive_support_span_hard_cap_enabled"] is False
    assert main_cfg["adaptive_support_span_soft_length_penalty_enabled"] is True
    assert main_cfg["adaptive_support_span_use_query_entity_signal"] is True
    assert main_cfg["adaptive_support_span_use_anchor_entity_signal"] is True
    assert main_cfg["adaptive_support_span_use_bridge_signal"] is True
    assert main_cfg["adaptive_support_span_bridge_mode"] == "conditional"

    assert weak_bridge_cfg["adaptive_support_span_enabled"] is True
    assert weak_bridge_cfg["adaptive_support_span_bridge_mode"] == "weak"

    assert no_bridge_cfg["adaptive_support_span_enabled"] is True
    assert no_bridge_cfg["adaptive_support_span_use_bridge_signal"] is False

    assert no_length_cfg["adaptive_support_span_enabled"] is True
    assert no_length_cfg["adaptive_support_span_soft_length_penalty_enabled"] is False

    assert span40_cfg["adaptive_support_span_enabled"] is True
    assert span40_cfg["adaptive_support_span_hard_cap_enabled"] is True
    assert span40_cfg["adaptive_support_span_max_item_tokens"] == 40
