from __future__ import annotations

from pathlib import Path

import yaml

from scripts.audit_phase6s_methodology import run_audit


def _write_cfg(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _base_cfg() -> dict:
    return {
        "unified_acr_rcedr_enabled": True,
        "unified_acr_rcedr_mode": "greedy",
        "unified_acr_rcedr_beam_size": 3,
        "unified_acr_rcedr_max_atoms": 8,
        "unified_acr_rcedr_max_tokens": 220,
        "unified_acr_rcedr_hard_token_budget_enabled": True,
        "unified_acr_rcedr_use_answerability_gain": True,
        "unified_acr_rcedr_use_bridge_gain": True,
        "unified_acr_rcedr_use_redundancy_penalty": True,
        "unified_acr_rcedr_use_cost_penalty": False,
        "unified_acr_rcedr_lambda_bridge": 0.28,
        "unified_acr_rcedr_mu_redundancy": 0.22,
        "unified_acr_rcedr_answerability_weight": 1.0,
        "unified_acr_rcedr_atom_span_max_sentences": 2,
        "unified_acr_rcedr_length_penalty_weight": 0.04,
        "unified_acr_rcedr_log_diagnostics": True,
        "gl_rcedr_enabled": False,
        "answerability_selection_enabled": False,
    }


def test_methodology_audit_passes_core_checks(tmp_path: Path) -> None:
    config_root = tmp_path / "cfgs"
    profile = "unified_acr_rcedr_v12"
    datasets = ["hotpotqa", "2wikimultihopqa"]
    for dataset in datasets:
        _write_cfg(config_root / profile / f"{dataset}.yaml", _base_cfg())

    report = run_audit(
        main_profile=profile,
        config_root=config_root,
        datasets=datasets,
    )
    assert report["status"] == "pass"
    assert report["unified_selector_active"] is True
    assert report["old_cascade_active"] is False
    assert report["hard_token_budget_enabled"] is True
    assert report["new_score_terms_detected"] is False


def test_methodology_audit_fails_when_old_cascade_is_active(tmp_path: Path) -> None:
    config_root = tmp_path / "cfgs"
    profile = "unified_acr_rcedr_v12"
    cfg = _base_cfg()
    cfg["gl_rcedr_enabled"] = True
    _write_cfg(config_root / profile / "hotpotqa.yaml", cfg)

    report = run_audit(
        main_profile=profile,
        config_root=config_root,
        datasets=["hotpotqa"],
    )
    assert report["status"] == "fail"
    assert report["old_cascade_active"] is True
    assert any("old cascade active" in err for err in report["errors"])


def test_methodology_audit_fails_on_unknown_unified_terms(tmp_path: Path) -> None:
    config_root = tmp_path / "cfgs"
    profile = "unified_acr_rcedr_v12"
    cfg = _base_cfg()
    cfg["unified_acr_rcedr_new_magic_weight"] = 0.7
    _write_cfg(config_root / profile / "hotpotqa.yaml", cfg)

    report = run_audit(
        main_profile=profile,
        config_root=config_root,
        datasets=["hotpotqa"],
    )
    assert report["status"] == "fail"
    assert report["new_score_terms_detected"] is True
