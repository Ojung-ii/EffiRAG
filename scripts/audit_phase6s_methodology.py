#!/usr/bin/env python3
"""Phase-6S methodology audit gate for full-run launches."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa", "popqa", "musique"]
DEFAULT_MAIN_PROFILE = "unified_acr_rcedr_v12"
DEFAULT_CONFIG_ROOT = Path("configs/main_config/copy_span_instruction_unified")

# Expected unified selector terms for Phase-6S main method.
ALLOWED_UNIFIED_KEYS = {
    "unified_acr_rcedr_enabled",
    "unified_acr_rcedr_mode",
    "unified_acr_rcedr_beam_size",
    "unified_acr_rcedr_max_atoms",
    "unified_acr_rcedr_max_tokens",
    "unified_acr_rcedr_hard_token_budget_enabled",
    "unified_acr_rcedr_use_answerability_gain",
    "unified_acr_rcedr_use_bridge_gain",
    "unified_acr_rcedr_use_redundancy_penalty",
    "unified_acr_rcedr_use_cost_penalty",
    "unified_acr_rcedr_lambda_bridge",
    "unified_acr_rcedr_mu_redundancy",
    "unified_acr_rcedr_answerability_weight",
    "unified_acr_rcedr_atom_span_max_sentences",
    "unified_acr_rcedr_length_penalty_weight",
    "unified_acr_rcedr_log_diagnostics",
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _collect_profile_cfgs(profile_root: Path, datasets: Sequence[str]) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    cfgs: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for dataset in datasets:
        p = profile_root / f"{dataset}.yaml"
        if not p.exists():
            missing.append(str(p))
            continue
        cfg = _read_yaml(p)
        if not cfg:
            missing.append(str(p))
            continue
        cfgs[dataset] = cfg
    return cfgs, missing


def _find_unknown_unified_terms(cfg: Mapping[str, Any]) -> List[str]:
    unknown: List[str] = []
    for key in cfg.keys():
        k = _safe_text(key)
        if not k.startswith("unified_acr_rcedr_"):
            continue
        if k not in ALLOWED_UNIFIED_KEYS:
            unknown.append(k)
    return sorted(set(unknown))


def _mk_markdown(audit: Mapping[str, Any]) -> str:
    details = list(audit.get("per_dataset", []) or [])
    warnings = list(audit.get("warnings", []) or [])
    errors = list(audit.get("errors", []) or [])

    lines: List[str] = []
    lines.append("# Phase-6S Methodology Audit")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append(f"| main_profile | {_safe_text(audit.get('main_profile'))} |")
    lines.append(f"| dataset_specific_main_profile | {bool(audit.get('dataset_specific_main_profile', False))} |")
    lines.append(f"| unified_selector_active | {bool(audit.get('unified_selector_active', False))} |")
    lines.append(f"| old_cascade_active | {bool(audit.get('old_cascade_active', False))} |")
    lines.append(f"| hard_token_budget_enabled | {bool(audit.get('hard_token_budget_enabled', False))} |")
    lines.append(f"| repeated_cost_penalty_warning | {bool(audit.get('repeated_cost_penalty_warning', False))} |")
    lines.append(f"| new_score_terms_detected | {bool(audit.get('new_score_terms_detected', False))} |")
    lines.append(f"| ablation_profiles_used_as_main | {bool(audit.get('ablation_profiles_used_as_main', False))} |")
    lines.append(f"| status | {_safe_text(audit.get('status'))} |")
    lines.append("")

    lines.append("## Dataset Checks")
    lines.append("")
    lines.append(
        "| dataset | unified_selector_enabled | gl_rcedr_enabled | answerability_selection_enabled | hard_budget | use_cost_penalty | length_penalty_weight | unknown_unified_terms |"
    )
    lines.append("|---|---|---|---|---|---|---:|---|")
    for row in details:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                bool(row.get("unified_selector_enabled", False)),
                bool(row.get("gl_rcedr_enabled", False)),
                bool(row.get("answerability_selection_enabled", False)),
                bool(row.get("hard_token_budget_enabled", False)),
                bool(row.get("use_cost_penalty", False)),
                f"{_safe_float(row.get('length_penalty_weight', 0.0), 0.0):.4f}",
                ", ".join(list(row.get("unknown_unified_terms", []) or [])) or "-",
            )
        )
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Errors")
    lines.append("")
    if errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run_audit(
    *,
    main_profile: str,
    config_root: Path,
    datasets: Sequence[str],
) -> Dict[str, Any]:
    profile_root = config_root / main_profile
    cfgs, missing = _collect_profile_cfgs(profile_root, datasets)

    errors: List[str] = []
    warnings: List[str] = []
    per_dataset: List[Dict[str, Any]] = []

    if not profile_root.exists():
        errors.append(f"profile_dir_missing:{profile_root}")

    if missing:
        errors.extend([f"dataset_cfg_missing:{m}" for m in missing])

    unified_selector_active = True
    old_cascade_active = False
    hard_token_budget_enabled = True
    repeated_cost_penalty_warning = False
    new_score_terms_detected = False
    dataset_specific_main_profile = False
    ablation_profiles_used_as_main = False

    for dataset in datasets:
        cfg = cfgs.get(dataset, {})
        if not cfg:
            unified_selector_active = False
            hard_token_budget_enabled = False
            continue

        unified_on = _safe_bool(cfg.get("unified_acr_rcedr_enabled"), False)
        gl_on = _safe_bool(cfg.get("gl_rcedr_enabled"), False)
        acr_on = _safe_bool(cfg.get("answerability_selection_enabled"), False)
        hard_budget_on = _safe_bool(cfg.get("unified_acr_rcedr_hard_token_budget_enabled"), False)
        use_cost_penalty = _safe_bool(cfg.get("unified_acr_rcedr_use_cost_penalty"), False)
        length_penalty_weight = _safe_float(cfg.get("unified_acr_rcedr_length_penalty_weight"), 0.0)
        use_answerability_gain = _safe_bool(cfg.get("unified_acr_rcedr_use_answerability_gain"), False)
        use_bridge_gain = _safe_bool(cfg.get("unified_acr_rcedr_use_bridge_gain"), False)
        use_redundancy_penalty = _safe_bool(cfg.get("unified_acr_rcedr_use_redundancy_penalty"), False)
        unknown_unified_terms = _find_unknown_unified_terms(cfg)

        if not unified_on:
            unified_selector_active = False
        if gl_on or acr_on:
            old_cascade_active = True
        if not hard_budget_on:
            hard_token_budget_enabled = False
        if use_cost_penalty:
            repeated_cost_penalty_warning = True
            warnings.append(f"{dataset}: unified_acr_rcedr_use_cost_penalty=true")
        if length_penalty_weight > 0.0:
            repeated_cost_penalty_warning = True
            warnings.append(f"{dataset}: unified_acr_rcedr_length_penalty_weight={length_penalty_weight:.4f}")
        if unknown_unified_terms:
            new_score_terms_detected = True
            warnings.append(f"{dataset}: unknown unified terms detected -> {', '.join(unknown_unified_terms)}")
        if not (use_answerability_gain and use_bridge_gain and use_redundancy_penalty):
            errors.append(
                f"{dataset}: expected objective terms missing "
                f"(answerability={use_answerability_gain}, bridge={use_bridge_gain}, redundancy={use_redundancy_penalty})"
            )

        per_dataset.append(
            {
                "dataset": dataset,
                "unified_selector_enabled": unified_on,
                "gl_rcedr_enabled": gl_on,
                "answerability_selection_enabled": acr_on,
                "hard_token_budget_enabled": hard_budget_on,
                "use_cost_penalty": use_cost_penalty,
                "length_penalty_weight": length_penalty_weight,
                "use_answerability_gain": use_answerability_gain,
                "use_bridge_gain": use_bridge_gain,
                "use_redundancy_penalty": use_redundancy_penalty,
                "unknown_unified_terms": unknown_unified_terms,
            }
        )

    if not unified_selector_active:
        errors.append("unified selector inactive for one or more datasets")
    if old_cascade_active:
        errors.append("old cascade active (gl_rcedr_enabled or answerability_selection_enabled)")
    if not hard_token_budget_enabled:
        errors.append("hard token budget disabled for one or more datasets")
    if new_score_terms_detected:
        errors.append("new unified score term(s) detected")

    status = "pass" if not errors else "fail"

    audit: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "main_profile": main_profile,
        "datasets": list(datasets),
        "dataset_specific_main_profile": dataset_specific_main_profile,
        "old_cascade_active": old_cascade_active,
        "unified_selector_active": unified_selector_active,
        "hard_token_budget_enabled": hard_token_budget_enabled,
        "repeated_cost_penalty_warning": repeated_cost_penalty_warning,
        "new_score_terms_detected": new_score_terms_detected,
        "ablation_profiles_used_as_main": ablation_profiles_used_as_main,
        "warnings": warnings,
        "errors": errors,
        "per_dataset": per_dataset,
        "status": status,
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase-6S methodology gate before full-run.")
    parser.add_argument("--main-profile", default=DEFAULT_MAIN_PROFILE)
    parser.add_argument("--check-configs", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--out-root", default="", help="If set, write outputs under this root.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--allow-fail", action="store_true", help="Exit 0 even when status=fail.")
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve() if _safe_text(args.out_root) else None
    if out_root is not None:
        output_json = Path(_safe_text(args.output_json) or str(out_root / "phase6s_methodology_audit.json")).resolve()
        output_md = Path(_safe_text(args.output_md) or str(out_root / "PHASE6S_METHODOLOGY_AUDIT.md")).resolve()
    else:
        output_json = Path(_safe_text(args.output_json) or "phase6s_methodology_audit.json").resolve()
        output_md = Path(_safe_text(args.output_md) or "PHASE6S_METHODOLOGY_AUDIT.md").resolve()

    datasets = [_safe_text(d) for d in args.datasets if _safe_text(d)]
    audit = run_audit(
        main_profile=_safe_text(args.main_profile) or DEFAULT_MAIN_PROFILE,
        config_root=Path(_safe_text(args.check_configs) or str(DEFAULT_CONFIG_ROOT)).resolve(),
        datasets=datasets or list(DEFAULT_DATASETS),
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_mk_markdown(audit) + "\n", encoding="utf-8")

    print(str(output_json))
    print(str(output_md))
    print(f"status={_safe_text(audit.get('status'))}")

    if _safe_text(audit.get("status")) != "pass" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
