from effirag.unified_copy_span_policy import DATASET_ORDER, apply_unified_profile


PHASE6L_PROFILES = [
    "unified_gl_rcedr_v1_support_span_contract",
    "unified_gl_rcedr_v1_support_span_contract_span40",
    "unified_gl_rcedr_v1_support_span_contract_no_cap",
    "unified_gl_rcedr_v1_support_span_contract_no_bridge_signal",
    "unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal",
]


def test_phase6l_support_span_profiles_are_dataset_invariant_and_guarded():
    for profile in PHASE6L_PROFILES:
        first = apply_unified_profile({}, DATASET_ORDER[0], profile)
        for dataset in DATASET_ORDER:
            cfg = apply_unified_profile({}, dataset, profile)
            assert cfg["gl_rcedr_enabled"] is True
            assert cfg["support_span_contract_enabled"] is True
            assert cfg["answer_support_pinning_enabled"] is False
            assert cfg["corridor_answer_preserve_guarded_hotpot_enabled"] is False
            assert cfg["oracle_support_injection_enabled"] is False
            assert cfg["dataset_specific_branch_enabled"] is False
            assert cfg["gl_rcedr_dynamic_control_enabled"] is first["gl_rcedr_dynamic_control_enabled"]
            assert cfg["gl_rcedr_stability_enabled"] is first["gl_rcedr_stability_enabled"]
            assert cfg["gl_rcedr_bridge_path_enabled"] is first["gl_rcedr_bridge_path_enabled"]
            assert cfg["support_span_max_item_tokens"] == first["support_span_max_item_tokens"]
            assert cfg["support_span_use_bridge_signal"] is first["support_span_use_bridge_signal"]
            assert cfg["support_span_use_query_entity_signal"] is first["support_span_use_query_entity_signal"]
            assert cfg["support_span_use_anchor_entity_signal"] is first["support_span_use_anchor_entity_signal"]


def test_phase6l_support_span_profile_specific_overrides():
    main_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_support_span_contract")
    span40_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_support_span_contract_span40")
    no_cap_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_support_span_contract_no_cap")
    no_bridge_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_support_span_contract_no_bridge_signal")
    no_query_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_support_span_contract_no_query_entity_signal")

    assert main_cfg["sentence_contract_render_enabled"] is True
    assert main_cfg["support_span_contract_enabled"] is True
    assert main_cfg["support_span_max_item_tokens"] == 48
    assert main_cfg["support_span_metadata_pruning"] is True
    assert main_cfg["support_span_chunk_expansion_allowed"] is False
    assert main_cfg["support_span_preserve_selected_items"] is True

    assert span40_cfg["support_span_contract_enabled"] is True
    assert span40_cfg["support_span_max_item_tokens"] == 40

    assert no_cap_cfg["support_span_contract_enabled"] is True
    assert no_cap_cfg["support_span_max_item_tokens"] is None

    assert no_bridge_cfg["support_span_contract_enabled"] is True
    assert no_bridge_cfg["support_span_use_bridge_signal"] is False

    assert no_query_cfg["support_span_contract_enabled"] is True
    assert no_query_cfg["support_span_use_query_entity_signal"] is False
    assert no_query_cfg["support_span_use_anchor_entity_signal"] is False
