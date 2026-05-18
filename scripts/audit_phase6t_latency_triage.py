#!/usr/bin/env python3
"""Audit Phase-6T latency triage profiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from effirag.unified_copy_span_policy import unified_profile_dir


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa"]
DEFAULT_PROFILES = [
    "unified_acr_rcedr_v12_sota_contract_unified",
    "unified_acr_rcedr_v12_sota_contract_rerank20",
    "unified_acr_rcedr_v12_sota_contract_search3x3",
    "unified_acr_rcedr_v12_sota_contract_fast",
    "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
    "unified_acr_rcedr_v12_sota_contract_fast_rerank5",
    "unified_acr_rcedr_v12_sota_contract_fast_proposal_ultralight",
]
DEFAULT_CONFIG_ROOT = Path("configs/main_config/copy_span_instruction_unified")

COMMON_EXPECTED = {
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

PROFILE_SPECIFIC_EXPECTED = {
    "unified_acr_rcedr_v12_sota_contract_unified": {},
    "unified_acr_rcedr_v12_sota_contract_rerank20": {
        "embedding_rerank_topn": 20,
    },
    "unified_acr_rcedr_v12_sota_contract_search3x3": {
        "max_anchors": 3,
        "samples_per_anchor": 3,
    },
    "unified_acr_rcedr_v12_sota_contract_fast": {
        "embedding_rerank_topn": 20,
        "max_anchors": 3,
        "samples_per_anchor": 3,
        "ppr_mc_walks": 256,
        "corridor_top_bc": 10,
        "ppr_subgraph_max_nodes": 15000,
    },
    "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank": {
        "embedding_rerank_topn": 0,
        "sentence_rerank_enabled": False,
        "max_anchors": 3,
        "samples_per_anchor": 3,
        "ppr_mc_walks": 256,
        "corridor_top_bc": 10,
        "ppr_subgraph_max_nodes": 15000,
    },
    "unified_acr_rcedr_v12_sota_contract_fast_rerank5": {
        "embedding_rerank_topn": 5,
        "max_anchors": 3,
        "samples_per_anchor": 3,
        "ppr_mc_walks": 256,
        "corridor_top_bc": 10,
        "ppr_subgraph_max_nodes": 15000,
    },
    "unified_acr_rcedr_v12_sota_contract_fast_proposal_ultralight": {
        "embedding_rerank_topn": 20,
        "max_anchors": 2,
        "samples_per_anchor": 2,
        "ppr_mc_walks": 128,
        "corridor_top_bc": 6,
        "ppr_subgraph_max_nodes": 8000,
    },
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


def _tokenize(raw_values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in raw_values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def run_audit(
    *,
    profiles: Sequence[str],
    datasets: Sequence[str],
    config_root: Path,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    per_profile_dataset: List[Dict[str, Any]] = []

    seen_profiles: List[str] = []
    seen_set = set()
    for profile in profiles:
        name = _safe_text(profile)
        if not name:
            continue
        if name in seen_set:
            continue
        seen_set.add(name)
        seen_profiles.append(name)

    for profile in seen_profiles:
        if profile not in PROFILE_SPECIFIC_EXPECTED:
            errors.append(f"unsupported_profile:{profile}")
            continue

        profile_dir = config_root / unified_profile_dir(profile)
        if not profile_dir.exists():
            errors.append(f"profile_dir_missing:{profile_dir}")
            continue

        for dataset in datasets:
            cfg_path = profile_dir / f"{dataset}.yaml"
            cfg = _read_yaml(cfg_path)
            if not cfg:
                errors.append(f"dataset_cfg_missing:{cfg_path}")
                continue

            row: Dict[str, Any] = {
                "profile": profile,
                "dataset": dataset,
                "cfg_path": str(cfg_path),
                "unified_acr_rcedr_enabled": _safe_bool(cfg.get("unified_acr_rcedr_enabled"), False),
                "gl_rcedr_enabled": _safe_bool(cfg.get("gl_rcedr_enabled"), False),
                "answerability_selection_enabled": _safe_bool(cfg.get("answerability_selection_enabled"), False),
                "answer_support_pinning_enabled": _safe_bool(cfg.get("answer_support_pinning_enabled"), False),
                "corridor_answer_preserve_guarded_hotpot_enabled": _safe_bool(
                    cfg.get("corridor_answer_preserve_guarded_hotpot_enabled"), False
                ),
                "final_top_slice_reorder_enabled": _safe_bool(cfg.get("final_top_slice_reorder_enabled"), False),
            }

            if not row["unified_acr_rcedr_enabled"]:
                errors.append(f"{profile}/{dataset}: unified_acr_rcedr_enabled=false")
            if row["gl_rcedr_enabled"] or row["answerability_selection_enabled"]:
                errors.append(f"{profile}/{dataset}: old_cascade_active")
            if row["answer_support_pinning_enabled"]:
                errors.append(f"{profile}/{dataset}: answer_support_pinning_enabled=true")
            if row["corridor_answer_preserve_guarded_hotpot_enabled"]:
                errors.append(f"{profile}/{dataset}: corridor_answer_preserve_guarded_hotpot_enabled=true")
            if row["final_top_slice_reorder_enabled"]:
                errors.append(f"{profile}/{dataset}: final_top_slice_reorder_enabled=true")

            for key, expected in COMMON_EXPECTED.items():
                actual = cfg.get(key)
                row[key] = actual
                if actual != expected:
                    errors.append(f"{profile}/{dataset}: {key}={actual!r} expected={expected!r}")

            for key, expected in PROFILE_SPECIFIC_EXPECTED[profile].items():
                actual = cfg.get(key)
                row[key] = actual
                if actual != expected:
                    errors.append(f"{profile}/{dataset}: {key}={actual!r} expected={expected!r}")

            # Keep objective contract explicit in audit output.
            row["unified_acr_rcedr_use_answerability_gain"] = _safe_bool(
                cfg.get("unified_acr_rcedr_use_answerability_gain"), False
            )
            row["unified_acr_rcedr_use_bridge_gain"] = _safe_bool(cfg.get("unified_acr_rcedr_use_bridge_gain"), False)
            row["unified_acr_rcedr_use_redundancy_penalty"] = _safe_bool(
                cfg.get("unified_acr_rcedr_use_redundancy_penalty"), False
            )
            row["unified_acr_rcedr_hard_token_budget_enabled"] = _safe_bool(
                cfg.get("unified_acr_rcedr_hard_token_budget_enabled"), False
            )
            if not row["unified_acr_rcedr_hard_token_budget_enabled"]:
                errors.append(f"{profile}/{dataset}: unified_acr_rcedr_hard_token_budget_enabled=false")

            per_profile_dataset.append(row)

    status = "pass" if not errors else "fail"
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profiles": seen_profiles,
        "datasets": list(datasets),
        "config_root": str(config_root),
        "errors": errors,
        "warnings": warnings,
        "per_profile_dataset": per_profile_dataset,
        "status": status,
    }


def _mk_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Phase-6T Latency Triage Audit")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append(f"| status | {_safe_text(report.get('status'))} |")
    lines.append(f"| profiles | {', '.join(report.get('profiles', []) or [])} |")
    lines.append(f"| datasets | {', '.join(report.get('datasets', []) or [])} |")
    lines.append("")

    lines.append("## Per Profile/Dataset Checks")
    lines.append("")
    lines.append(
        "| profile | dataset | unified_selector | gl_rcedr | answerability_selection | "
        "prompt_variant | order_strategy | render_mode | delivery_mode | "
        "max_context | max_total | max_sentences | top_corridors | max_corridors | bridge_induction | "
        "target_prompt_tokens | max_prompt_tokens | rerank_topn | max_anchors | samples_per_anchor | "
        "ppr_mc_walks | corridor_top_bc | ppr_subgraph_max_nodes |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in list(report.get("per_profile_dataset", []) or []):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("profile")),
                _safe_text(row.get("dataset")),
                bool(row.get("unified_acr_rcedr_enabled")),
                bool(row.get("gl_rcedr_enabled")),
                bool(row.get("answerability_selection_enabled")),
                _safe_text(row.get("prompt_variant")),
                _safe_text(row.get("order_strategy")),
                _safe_text(row.get("render_mode")),
                _safe_text(row.get("delivery_mode")),
                _safe_int(row.get("max_context_sentences"), 0),
                _safe_int(row.get("max_total_sentences"), 0),
                _safe_int(row.get("max_sentences"), 0),
                _safe_int(row.get("top_corridors"), 0),
                _safe_int(row.get("max_corridors_in_context"), 0),
                bool(row.get("bridge_candidate_induction_enabled")),
                _safe_int(row.get("target_prompt_tokens"), 0),
                _safe_int(row.get("max_prompt_tokens"), 0),
                _safe_int(row.get("embedding_rerank_topn"), 0),
                _safe_int(row.get("max_anchors"), 0),
                _safe_int(row.get("samples_per_anchor"), 0),
                _safe_int(row.get("ppr_mc_walks"), 0),
                _safe_int(row.get("corridor_top_bc"), 0),
                _safe_int(row.get("ppr_subgraph_max_nodes"), 0),
            )
        )
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    warnings = list(report.get("warnings", []) or [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Errors")
    lines.append("")
    errors = list(report.get("errors", []) or [])
    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase-6T latency triage profiles.")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--check-configs", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--out-root", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    profiles = _tokenize(args.profiles) or list(DEFAULT_PROFILES)
    datasets = _tokenize(args.datasets) or list(DEFAULT_DATASETS)
    config_root = Path(_safe_text(args.check_configs) or str(DEFAULT_CONFIG_ROOT)).resolve()

    out_root = Path(_safe_text(args.out_root)).resolve() if _safe_text(args.out_root) else None
    if out_root is not None:
        output_json = Path(
            _safe_text(args.output_json) or str(out_root / "phase6t_latency_triage_audit.json")
        ).resolve()
        output_md = Path(
            _safe_text(args.output_md) or str(out_root / "PHASE6T_LATENCY_TRIAGE_AUDIT.md")
        ).resolve()
    else:
        output_json = Path(_safe_text(args.output_json) or "phase6t_latency_triage_audit.json").resolve()
        output_md = Path(_safe_text(args.output_md) or "PHASE6T_LATENCY_TRIAGE_AUDIT.md").resolve()

    report = run_audit(profiles=profiles, datasets=datasets, config_root=config_root)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_mk_markdown(report) + "\n", encoding="utf-8")

    print(str(output_json))
    print(str(output_md))
    print(f"status={_safe_text(report.get('status'))}")
    if _safe_text(report.get("status")) != "pass" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
