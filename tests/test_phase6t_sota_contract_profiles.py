from __future__ import annotations

from effirag.unified_copy_span_policy import DATASET_ORDER, apply_unified_profile


PHASE6T_PROFILES = [
    "unified_acr_rcedr_v12_sota_contract_unified",
    "unified_acr_rcedr_v12_sota_contract_light",
]


def test_phase6t_sota_contract_profiles_are_dataset_invariant() -> None:
    expected = {
        "prompt_variant": "light_separator_copy_span_instruction",
        "order_strategy": "score+light_separator_render",
        "render_mode": "corridor_aware_flat",
        "delivery_mode": "sentence_compressed",
        "max_context_sentences": 8,
        "max_total_sentences": 8,
        "max_sentences": 8,
        "target_prompt_tokens": 600,
        "max_prompt_tokens": 700,
        "top_corridors": 3,
        "max_corridors_in_context": 3,
        "bridge_candidate_induction_enabled": False,
    }
    for profile in PHASE6T_PROFILES:
        first = apply_unified_profile({}, DATASET_ORDER[0], profile)
        for dataset in DATASET_ORDER:
            cfg = apply_unified_profile({}, dataset, profile)
            assert cfg["gl_rcedr_enabled"] is False
            assert cfg["answerability_selection_enabled"] is False
            assert cfg["unified_acr_rcedr_enabled"] is True
            assert cfg["answer_support_pinning_enabled"] is False
            assert cfg["corridor_answer_preserve_guarded_hotpot_enabled"] is False
            assert cfg["final_top_slice_reorder_enabled"] is False
            for key, value in expected.items():
                assert cfg[key] == value
                assert cfg[key] == first[key]


def test_phase6t_light_profile_cost_settings() -> None:
    unified_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_rcedr_v12_sota_contract_unified")
    light_cfg = apply_unified_profile({}, "hotpotqa", "unified_acr_rcedr_v12_sota_contract_light")

    assert unified_cfg["max_anchors"] == 4
    assert unified_cfg["samples_per_anchor"] == 4

    assert light_cfg["max_anchors"] == 3
    assert light_cfg["samples_per_anchor"] == 3
    assert light_cfg["ppr_mc_walks"] == 256
    assert light_cfg["corridor_top_bc"] == 10
    assert light_cfg["ppr_mc_walks"] != unified_cfg.get("ppr_mc_walks", light_cfg["ppr_mc_walks"] + 1)
    assert light_cfg["corridor_top_bc"] != unified_cfg.get(
        "corridor_top_bc",
        light_cfg["corridor_top_bc"] + 1,
    )
