from __future__ import annotations

from pathlib import Path

import yaml

from scripts.audit_phase6t_sota_contract import DEFAULT_MAIN_PROFILE, run_audit


def _write_cfg(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_cfg() -> dict:
    return {
        "unified_acr_rcedr_enabled": True,
        "gl_rcedr_enabled": False,
        "answerability_selection_enabled": False,
        "unified_acr_rcedr_hard_token_budget_enabled": True,
        "unified_acr_rcedr_use_answerability_gain": True,
        "unified_acr_rcedr_use_bridge_gain": True,
        "unified_acr_rcedr_use_redundancy_penalty": True,
        "unified_acr_rcedr_use_cost_penalty": False,
        "answer_support_pinning_enabled": False,
        "corridor_answer_preserve_guarded_hotpot_enabled": False,
        "final_top_slice_reorder_enabled": False,
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


def test_phase6t_audit_passes_for_valid_contract(tmp_path: Path) -> None:
    config_root = tmp_path / "cfgs"
    profile_dir = config_root / "unified_acr_rcedr_v12_sota_contract_unified"
    for dataset in ["hotpotqa", "2wikimultihopqa"]:
        _write_cfg(profile_dir / f"{dataset}.yaml", _base_cfg())

    report = run_audit(
        profile=DEFAULT_MAIN_PROFILE,
        datasets=["hotpotqa", "2wikimultihopqa"],
        config_root=config_root,
    )
    assert report["status"] == "pass"
    assert report["unified_selector_active"] is True
    assert report["old_cascade_active"] is False
    assert report["bridge_induction_disabled"] is True


def test_phase6t_audit_fails_with_forbidden_dataset_guard(tmp_path: Path) -> None:
    config_root = tmp_path / "cfgs"
    profile_dir = config_root / "unified_acr_rcedr_v12_sota_contract_unified"
    cfg = _base_cfg()
    cfg["corridor_answer_preserve_guarded_hotpot_enabled"] = True
    _write_cfg(profile_dir / "hotpotqa.yaml", cfg)
    _write_cfg(profile_dir / "2wikimultihopqa.yaml", _base_cfg())

    report = run_audit(
        profile=DEFAULT_MAIN_PROFILE,
        datasets=["hotpotqa", "2wikimultihopqa"],
        config_root=config_root,
    )
    assert report["status"] == "fail"
    assert report["hotpot_guard_enabled"] is True
