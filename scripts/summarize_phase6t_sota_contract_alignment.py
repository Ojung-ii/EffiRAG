#!/usr/bin/env python3
"""Summarize Phase-6T SOTA-contract alignment runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from validate_phase6s_artifacts import collect_phase6s_artifacts


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa"]
DEFAULT_PROFILES = [
    "unified_large",
    "legacy_sota",
    "unified_acr_rcedr_v12",
    "unified_acr_rcedr_v12_sota_contract_unified",
    "unified_acr_rcedr_v12_sota_contract_light",
]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _fmt(value: Any, nd: int = 4) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def _tokenize(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _load_methodology_audit(out_root: Path) -> Dict[str, Any]:
    path = out_root / "phase6t_sota_contract_audit.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _build_rows(
    report: Mapping[str, Any],
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_datasets = set(datasets)
    allowed_profiles = set(profiles)
    rows: List[Dict[str, Any]] = []
    for run in list(report.get("completed_runs", []) or []):
        dataset = _safe_text(run.get("dataset"))
        profile = _safe_text(run.get("profile"))
        if dataset not in allowed_datasets:
            continue
        if profile not in allowed_profiles:
            continue
        summary = dict(run.get("summary", {}) or {})
        em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
        f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
        prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
        completion_tokens = _safe_float(summary.get("completion_tokens_avg", 0.0), 0.0)
        retrieval_ms = _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0)
        generation_ms = _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0)
        total_ms = _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0)

        sf_recall = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
        sf_precision = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
        bridge_noise_ratio = _safe_float(
            summary.get("bridge_noise_ratio", summary.get("stagewise_bridge_noise_ratio", 0.0)),
            0.0,
        )
        abgf = _safe_float(
            summary.get(
                "answer_present_but_generation_fail",
                summary.get("stagewise_answer_present_but_generation_fail", 0.0),
            ),
            0.0,
        )

        rows.append(
            {
                "run_name": _safe_text(run.get("run_name")),
                "dataset": dataset,
                "profile": profile,
                "EM": em,
                "F1": f1,
                "avg_prompt_tokens": prompt_tokens,
                "completion_tokens_avg": completion_tokens,
                "F1_per_1k_prompt": (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0,
                "supporting_fact_recall": sf_recall,
                "supporting_fact_precision": sf_precision,
                "bridge_noise_ratio": bridge_noise_ratio,
                "answer_present_but_generation_fail": abgf,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "qa_num_queries": _safe_int(summary.get("qa_executed_samples", 0), 0),
                "summary_path": _safe_text(run.get("summary_path")),
            }
        )
    rows.sort(key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("profile")), _safe_text(x.get("run_name"))))
    return rows


def _contract_alignment_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_pair = {
        (_safe_text(row.get("dataset")), _safe_text(row.get("profile"))): dict(row)
        for row in rows
    }
    for dataset in DEFAULT_DATASETS:
        current = by_pair.get((dataset, "unified_acr_rcedr_v12"))
        contract = by_pair.get((dataset, "unified_acr_rcedr_v12_sota_contract_unified"))
        light = by_pair.get((dataset, "unified_acr_rcedr_v12_sota_contract_light"))
        for row, variant in [(current, "current_v12"), (contract, "contract_unified"), (light, "contract_light")]:
            if row is None:
                continue
            delta_f1 = None
            delta_tok = None
            delta_f1k = None
            if current is not None:
                delta_f1 = _safe_float(row.get("F1"), 0.0) - _safe_float(current.get("F1"), 0.0)
                delta_tok = _safe_float(row.get("avg_prompt_tokens"), 0.0) - _safe_float(
                    current.get("avg_prompt_tokens"), 0.0
                )
                delta_f1k = _safe_float(row.get("F1_per_1k_prompt"), 0.0) - _safe_float(
                    current.get("F1_per_1k_prompt"), 0.0
                )
            out.append(
                {
                    **dict(row),
                    "variant": variant,
                    "delta_F1_vs_current_v12": delta_f1,
                    "delta_prompt_tokens_vs_current_v12": delta_tok,
                    "delta_F1_per_1k_vs_current_v12": delta_f1k,
                }
            )
    return out


def _row(rows: Sequence[Mapping[str, Any]], dataset: str, profile: str) -> Dict[str, Any] | None:
    for row in rows:
        if _safe_text(row.get("dataset")) == dataset and _safe_text(row.get("profile")) == profile:
            return dict(row)
    return None


def _mk_md(
    *,
    rows: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    methodology_audit: Mapping[str, Any],
) -> str:
    counts = dict(report.get("counts", {}) or {})
    lines: List[str] = []
    lines.append("# Phase-6T SOTA-Contract Alignment Summary")
    lines.append("")
    lines.append("## 1. Executed QA Results")
    lines.append("")
    lines.append(
        "| dataset | profile | EM | F1 | avg_prompt_tokens | F1_per_1k_prompt | "
        "supporting_fact_recall | supporting_fact_precision | bridge_noise_ratio | "
        "answer_present_but_generation_fail | retrieval_ms | total_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("EM", 0.0)),
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_prompt_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_prompt", 0.0)),
                _fmt(row.get("supporting_fact_recall", 0.0)),
                _fmt(row.get("supporting_fact_precision", 0.0)),
                _fmt(row.get("bridge_noise_ratio", 0.0)),
                _fmt(row.get("answer_present_but_generation_fail", 0.0)),
                _fmt(row.get("retrieval_ms", 0.0), 1),
                _fmt(row.get("total_ms", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 2. Contract Alignment Comparison")
    lines.append("")
    lines.append(
        "| dataset | variant | profile | F1 | avg_prompt_tokens | F1_per_1k_prompt | "
        "delta_F1_vs_current_v12 | delta_prompt_tokens_vs_current_v12 | delta_F1_per_1k_vs_current_v12 |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in contract_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("variant")),
                _safe_text(row.get("profile")),
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_prompt_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_prompt", 0.0)),
                _fmt(row.get("delta_F1_vs_current_v12", "n/a")),
                _fmt(row.get("delta_prompt_tokens_vs_current_v12", "n/a"), 1),
                _fmt(row.get("delta_F1_per_1k_vs_current_v12", "n/a")),
            )
        )
    lines.append("")

    lines.append("## 3. Legacy vs Current vs Contract-v12")
    lines.append("")
    lines.append("| dataset | profile | F1 | avg_prompt_tokens | supporting_fact_precision | retrieval_ms |")
    lines.append("|---|---|---:|---:|---:|---:|")
    compare_profiles = [
        "legacy_sota",
        "unified_large",
        "unified_acr_rcedr_v12",
        "unified_acr_rcedr_v12_sota_contract_unified",
        "unified_acr_rcedr_v12_sota_contract_light",
    ]
    for dataset in DEFAULT_DATASETS:
        for profile in compare_profiles:
            row = _row(rows, dataset, profile)
            if row is None:
                continue
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("avg_prompt_tokens", 0.0), 1),
                    _fmt(row.get("supporting_fact_precision", 0.0)),
                    _fmt(row.get("retrieval_ms", 0.0), 1),
                )
            )
    lines.append("")

    lines.append("## 4. Token Efficiency")
    lines.append("")
    lines.append("| dataset | profile | avg_prompt_tokens | F1_per_1k_prompt |")
    lines.append("|---|---|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("avg_prompt_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_prompt", 0.0)),
            )
        )
    lines.append("")

    lines.append("## 5. Latency Summary")
    lines.append("")
    lines.append("| dataset | profile | retrieval_ms | generation_ms | total_ms |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("retrieval_ms", 0.0), 1),
                _fmt(row.get("generation_ms", 0.0), 1),
                _fmt(row.get("total_ms", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 6. Supporting Fact / Evidence Precision")
    lines.append("")
    lines.append(
        "| dataset | profile | supporting_fact_recall | supporting_fact_precision | bridge_noise_ratio | "
        "answer_present_but_generation_fail |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("supporting_fact_recall", 0.0)),
                _fmt(row.get("supporting_fact_precision", 0.0)),
                _fmt(row.get("bridge_noise_ratio", 0.0)),
                _fmt(row.get("answer_present_but_generation_fail", 0.0)),
            )
        )
    lines.append("")

    lines.append("## 7. Methodology Audit Result")
    lines.append("")
    if methodology_audit:
        lines.append("| item | value |")
        lines.append("|---|---|")
        for key in [
            "status",
            "main_profile",
            "dataset_specific_main_profile",
            "unified_selector_active",
            "old_cascade_active",
            "prompt_contract_ok",
            "render_contract_ok",
            "budget_contract_ok",
            "bridge_induction_disabled",
        ]:
            lines.append(f"| {key} | {methodology_audit.get(key)} |")
    else:
        lines.append("Methodology audit file not found: `phase6t_sota_contract_audit.json`")
    lines.append("")

    lines.append("## 8. Artifact Consistency")
    lines.append("")
    lines.append("| item | count |")
    lines.append("|---|---:|")
    lines.append(f"| scheduled_runs | {int(counts.get('scheduled_runs', 0))} |")
    lines.append(f"| run_names_with_attempts | {int(counts.get('run_names_with_attempts', 0))} |")
    lines.append(f"| completed_run_names | {int(counts.get('completed_run_names', 0))} |")
    lines.append(f"| incomplete_attempts | {int(counts.get('incomplete_attempts', 0))} |")
    lines.append(f"| extra_not_in_manifest | {int(counts.get('extra_not_in_manifest', 0))} |")
    lines.append(f"| multi_attempt_run_names | {int(counts.get('multi_attempt_run_names', 0))} |")
    lines.append(f"| parse_errors | {int(counts.get('parse_errors', 0))} |")
    lines.append("")

    lines.append("## 9. Decision Checklist")
    lines.append("")
    lines.append("- `v12_sota_contract_unified` F1 vs `v12` improves or not?")
    lines.append("- Prompt-token efficiency is maintained or improved?")
    lines.append("- Supporting-fact precision and bridge noise move in the right direction?")
    lines.append("- Latency impact acceptable, or light profile needed for cost control?")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6T SOTA-contract alignment runs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    output_md = Path(
        _safe_text(args.output_md) or str(out_root / "PHASE6T_SOTA_CONTRACT_ALIGNMENT_SUMMARY.md")
    ).resolve()
    output_json = Path(
        _safe_text(args.output_json) or str(out_root / "phase6t_sota_contract_alignment_summary.json")
    ).resolve()

    datasets = _tokenize(args.datasets) or list(DEFAULT_DATASETS)
    profiles = _tokenize(args.profiles) or list(DEFAULT_PROFILES)

    report = collect_phase6s_artifacts(out_root)
    methodology_audit = _load_methodology_audit(out_root)
    rows = _build_rows(report, datasets=datasets, profiles=profiles)
    contract_rows = _contract_alignment_rows(rows)
    markdown = _mk_md(
        rows=rows,
        contract_rows=contract_rows,
        report=report,
        methodology_audit=methodology_audit,
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown + "\n", encoding="utf-8")

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "output_md": str(output_md),
        "output_json": str(output_json),
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "rows": rows,
        "contract_alignment_rows": contract_rows,
        "methodology_audit": methodology_audit,
        "report": report,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(output_md))


if __name__ == "__main__":
    main()

