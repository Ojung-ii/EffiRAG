from effirag.unified_copy_span_policy import DATASET_ORDER, apply_unified_profile


PHASE6K_PROFILES = [
    "unified_gl_rcedr_v1_sentence_contract",
    "unified_gl_rcedr_v1_sentence_contract_no_item_cap",
    "unified_gl_rcedr_v1_sentence_contract_metadata_on",
    "unified_gl_rcedr_v1_sentence_contract_span40",
]


def test_phase6k_sentence_contract_profiles_are_dataset_invariant_and_guarded():
    for profile in PHASE6K_PROFILES:
        first = apply_unified_profile({}, DATASET_ORDER[0], profile)
        for dataset in DATASET_ORDER:
            cfg = apply_unified_profile({}, dataset, profile)
            assert cfg["gl_rcedr_enabled"] is True
            assert cfg["sentence_contract_render_enabled"] is True
            assert cfg["answer_support_pinning_enabled"] is False
            assert cfg["corridor_answer_preserve_guarded_hotpot_enabled"] is False
            assert cfg["oracle_support_injection_enabled"] is False
            assert cfg["dataset_specific_branch_enabled"] is False
            assert cfg["gl_rcedr_dynamic_control_enabled"] is first["gl_rcedr_dynamic_control_enabled"]
            assert cfg["gl_rcedr_stability_enabled"] is first["gl_rcedr_stability_enabled"]
            assert cfg["gl_rcedr_bridge_path_enabled"] is first["gl_rcedr_bridge_path_enabled"]


def test_phase6k_sentence_contract_profile_specific_overrides():
    main_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_sentence_contract")
    no_cap_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_sentence_contract_no_item_cap")
    metadata_on_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_sentence_contract_metadata_on")
    span40_cfg = apply_unified_profile({}, "hotpotqa", "unified_gl_rcedr_v1_sentence_contract_span40")

    assert main_cfg["sentence_contract_max_item_tokens"] == 32
    assert main_cfg["sentence_contract_metadata_pruning"] is True
    assert main_cfg["sentence_contract_chunk_expansion_allowed"] is False
    assert main_cfg["sentence_contract_preserve_selected_items"] is True
    assert main_cfg["sentence_contract_minimal_span_fallback"] is True

    assert no_cap_cfg["sentence_contract_max_item_tokens"] is None
    assert no_cap_cfg["sentence_contract_metadata_pruning"] is True

    assert metadata_on_cfg["sentence_contract_max_item_tokens"] == 32
    assert metadata_on_cfg["sentence_contract_metadata_pruning"] is False

    assert span40_cfg["sentence_contract_max_item_tokens"] == 40
    assert span40_cfg["sentence_contract_metadata_pruning"] is True
