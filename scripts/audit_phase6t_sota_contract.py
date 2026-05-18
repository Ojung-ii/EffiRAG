#!/usr/bin/env python3
"""Phase-6T audit gate for non-heuristic SOTA-contract alignment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from effirag.unified_copy_span_policy import unified_profile_dir


DEFAULT_MAIN_PROFILE = "unified_acr_rcedr_v12_sota_contract_unified"
DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa"]
DEFAULT_CONFIG_ROOT = Path("configs/main_config/copy_span_instruction_unified")

EXPECTED_CONTRACT = {
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


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _normalize_datasets(raw_values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in raw_values:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _mk_md(audit: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Phase-6T SOTA Contract Audit")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    for key in [
        "main_profile",
        "dataset_specific_main_profile",
        "hotpot_guard_enabled",
        "answer_support_pinning_enabled",
        "top_slice_reorder_enabled",
        "prompt_contract_ok",
        "render_contract_ok",
        "budget_contract_ok",
        "bridge_induction_disabled",
        "unified_selector_active",
        "old_cascade_active",
        "status",
    ]:
        lines.append(f"| {key} | {audit.get(key)} |")
    lines.append("")
    lines.append("## Dataset Checks")
    lines.append("")
    lines.append(
        "| dataset | unified_acr_rcedr_enabled | gl_rcedr_enabled | answerability_selection_enabled | "
        "prompt_variant | order_strategy | render_mode | delivery_mode | max_context_sentences | max_total_sentences | "
        "max_sentences | target_prompt_tokens | max_prompt_tokens | top_corridors | max_corridors_in_context | "
        "bridge_candidate_induction_enabled |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for row in list(audit.get("per_dataset", []) or []):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                bool(row.get("unified_acr_rcedr_enabled", False)),
                bool(row.get("gl_rcedr_enabled", False)),
                bool(row.get("answerability_selection_enabled", False)),
                _safe_text(row.get("prompt_variant")),
                _safe_text(row.get("order_strategy")),
                _safe_text(row.get("render_mode")),
                _safe_text(row.get("delivery_mode")),
                _safe_int(row.get("max_context_sentences", 0), 0),
                _safe_int(row.get("max_total_sentences", 0), 0),
                _safe_int(row.get("max_sentences", 0), 0),
                _safe_int(row.get("target_prompt_tokens", 0), 0),
                _safe_int(row.get("max_prompt_tokens", 0), 0),
                _safe_int(row.get("top_corridors", 0), 0),
                _safe_int(row.get("max_corridors_in_context", 0), 0),
                bool(row.get("bridge_candidate_induction_enabled", False)),
            )
        )
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    warnings = list(audit.get("warnings", []) or [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Errors")
    lines.append("")
    errors = list(audit.get("errors", []) or [])
    if errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run_audit(
    *,
    profile: str,
    datasets: Sequence[str],
    config_root: Path,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    per_dataset: List[Dict[str, Any]] = []

    dataset_specific_main_profile = False
    hotpot_guard_enabled = False
    answer_support_pinning_enabled = False
    top_slice_reorder_enabled = False
    prompt_contract_ok = True
    render_contract_ok = True
    budget_contract_ok = True
    bridge_induction_disabled = True
    unified_selector_active = True
    old_cascade_active = False

    if profile != DEFAULT_MAIN_PROFILE:
        warnings.append(f"main profile differs from default candidate: {profile}")

    profile_path = config_root / unified_profile_dir(profile)
    if not profile_path.exists():
        errors.append(f"profile_dir_missing:{profile_path}")

    for dataset in datasets:
        cfg_path = profile_path / f"{dataset}.yaml"
        cfg = _read_yaml(cfg_path)
        if not cfg:
            errors.append(f"dataset_cfg_missing:{cfg_path}")
            unified_selector_active = False
            prompt_contract_ok = False
            render_contract_ok = False
            budget_contract_ok = False
            bridge_induction_disabled = False
            continue

        unified_on = _safe_bool(cfg.get("unified_acr_rcedr_enabled"), False)
        gl_on = _safe_bool(cfg.get("gl_rcedr_enabled"), False)
        acr_on = _safe_bool(cfg.get("answerability_selection_enabled"), False)
        pinning_on = _safe_bool(cfg.get("answer_support_pinning_enabled"), False)
        guarded_hotpot_on = _safe_bool(cfg.get("corridor_answer_preserve_guarded_hotpot_enabled"), False)
        reorder_on = _safe_bool(cfg.get("final_top_slice_reorder_enabled"), False)
        bridge_induction_on = _safe_bool(cfg.get("bridge_candidate_induction_enabled"), False)
        hard_budget_on = _safe_bool(cfg.get("unified_acr_rcedr_hard_token_budget_enabled"), False)

        if not unified_on:
            unified_selector_active = False
        if gl_on or acr_on:
            old_cascade_active = True
        if not hard_budget_on:
            errors.append(f"{dataset}: unified_acr_rcedr_hard_token_budget_enabled=false")

        if pinning_on:
            answer_support_pinning_enabled = True
            errors.append(f"{dataset}: answer_support_pinning_enabled=true")
        if guarded_hotpot_on:
            hotpot_guard_enabled = True
            errors.append(f"{dataset}: corridor_answer_preserve_guarded_hotpot_enabled=true")
        if reorder_on:
            top_slice_reorder_enabled = True
            errors.append(f"{dataset}: final_top_slice_reorder_enabled=true")
        if bridge_induction_on:
            bridge_induction_disabled = False
            errors.append(f"{dataset}: bridge_candidate_induction_enabled=true")

        for key in ["prompt_variant", "order_strategy"]:
            expected = EXPECTED_CONTRACT[key]
            if cfg.get(key) != expected:
                prompt_contract_ok = False
                errors.append(f"{dataset}: {key}={cfg.get(key)!r} expected={expected!r}")
        for key in ["render_mode", "delivery_mode"]:
            expected = EXPECTED_CONTRACT[key]
            if cfg.get(key) != expected:
                render_contract_ok = False
                errors.append(f"{dataset}: {key}={cfg.get(key)!r} expected={expected!r}")
        for key in [
            "max_context_sentences",
            "max_total_sentences",
            "max_sentences",
            "target_prompt_tokens",
            "max_prompt_tokens",
            "top_corridors",
            "max_corridors_in_context",
        ]:
            expected = EXPECTED_CONTRACT[key]
            actual = _safe_int(cfg.get(key), -999999)
            if actual != expected:
                budget_contract_ok = False
                errors.append(f"{dataset}: {key}={actual} expected={expected}")

        per_dataset.append(
            {
                "dataset": dataset,
                "unified_acr_rcedr_enabled": unified_on,
                "gl_rcedr_enabled": gl_on,
                "answerability_selection_enabled": acr_on,
                "prompt_variant": cfg.get("prompt_variant"),
                "order_strategy": cfg.get("order_strategy"),
                "render_mode": cfg.get("render_mode"),
                "delivery_mode": cfg.get("delivery_mode"),
                "max_context_sentences": _safe_int(cfg.get("max_context_sentences"), 0),
                "max_total_sentences": _safe_int(cfg.get("max_total_sentences"), 0),
                "max_sentences": _safe_int(cfg.get("max_sentences"), 0),
                "target_prompt_tokens": _safe_int(cfg.get("target_prompt_tokens"), 0),
                "max_prompt_tokens": _safe_int(cfg.get("max_prompt_tokens"), 0),
                "top_corridors": _safe_int(cfg.get("top_corridors"), 0),
                "max_corridors_in_context": _safe_int(cfg.get("max_corridors_in_context"), 0),
                "bridge_candidate_induction_enabled": bridge_induction_on,
            }
        )

    if not unified_selector_active:
        errors.append("unified_selector_active=false")
    if old_cascade_active:
        errors.append("old GL-RCEDR->ACR cascade appears active")

    audit: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "main_profile": profile,
        "datasets": list(datasets),
        "dataset_specific_main_profile": dataset_specific_main_profile,
        "hotpot_guard_enabled": hotpot_guard_enabled,
        "answer_support_pinning_enabled": answer_support_pinning_enabled,
        "top_slice_reorder_enabled": top_slice_reorder_enabled,
        "prompt_contract_ok": prompt_contract_ok,
        "render_contract_ok": render_contract_ok,
        "budget_contract_ok": budget_contract_ok,
        "bridge_induction_disabled": bridge_induction_disabled,
        "unified_selector_active": unified_selector_active,
        "old_cascade_active": old_cascade_active,
        "warnings": warnings,
        "errors": errors,
        "per_dataset": per_dataset,
        "status": "pass" if not errors else "fail",
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase-6T SOTA-contract alignment profile.")
    parser.add_argument("--profile", default=DEFAULT_MAIN_PROFILE)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--check-configs", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--out-root", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    datasets = _normalize_datasets(args.datasets) or list(DEFAULT_DATASETS)
    config_root = Path(_safe_text(args.check_configs) or str(DEFAULT_CONFIG_ROOT)).resolve()
    profile = _safe_text(args.profile) or DEFAULT_MAIN_PROFILE

    out_root = Path(_safe_text(args.out_root)).resolve() if _safe_text(args.out_root) else None
    if out_root is not None:
        output_json = Path(
            _safe_text(args.output_json) or str(out_root / "phase6t_sota_contract_audit.json")
        ).resolve()
        output_md = Path(
            _safe_text(args.output_md) or str(out_root / "PHASE6T_SOTA_CONTRACT_AUDIT.md")
        ).resolve()
    else:
        output_json = Path(_safe_text(args.output_json) or "phase6t_sota_contract_audit.json").resolve()
        output_md = Path(_safe_text(args.output_md) or "PHASE6T_SOTA_CONTRACT_AUDIT.md").resolve()

    audit = run_audit(profile=profile, datasets=datasets, config_root=config_root)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_mk_md(audit) + "\n", encoding="utf-8")

    print(str(output_json))
    print(str(output_md))
    print(f"status={_safe_text(audit.get('status'))}")

    if _safe_text(audit.get("status")) != "pass" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

