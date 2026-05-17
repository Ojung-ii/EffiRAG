#!/usr/bin/env python3
"""Create a single Phase-6P summary markdown by joining QA and retrieval audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.compare_qa_pareto_profiles import (
    _collect_qa_summary_rows,
    _find_retrieval_aggregate_csv,
    _pareto_frontier,
    _prepare_retrieval_index,
    _read_csv,
    _safe_float,
    _safe_int,
    _safe_text,
    _summarize_qa_rows,
    _tokens,
    _write_csv,
)


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa"]
DEFAULT_PROFILES = [
    "legacy_sota",
    "unified_large",
    "unified_gl_rcedr_v1",
    "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
    "unified_acr_v1",
    "unified_acr_v1_no_answerability",
    "unified_acr_v1_no_structure",
    "unified_acr_v1_no_noise_penalty",
    "unified_acr_v1_no_cost",
    "unified_acr_v1_no_ordering",
    "unified_acr_v1_beam3",
]


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "0.0000"


def _profile_role(profile: str) -> str:
    role_map = {
        "legacy_sota": "teacher/reference",
        "unified_large": "primary baseline",
        "unified_gl_rcedr_v1": "GL-RCEDR baseline",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge": "best prior compact baseline",
        "unified_acr_v1": "main ACR candidate",
        "unified_acr_v1_no_answerability": "ablation",
        "unified_acr_v1_no_structure": "ablation",
        "unified_acr_v1_no_noise_penalty": "ablation",
        "unified_acr_v1_no_cost": "ablation",
        "unified_acr_v1_no_ordering": "ablation",
        "unified_acr_v1_beam3": "beam-search ablation",
    }
    return role_map.get(profile, "candidate")


def _build_markdown(
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
    merged_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
) -> str:
    by_key = {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): dict(r) for r in merged_rows}
    frontier_set = {
        (_safe_text(r.get("dataset")), _safe_text(r.get("profile")), _safe_text(r.get("frontier_name")))
        for r in frontier_rows
    }

    lines: List[str] = []
    lines.append("# Phase-6P ACR Summary")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report summarizes QA and retrieval efficiency for ACR candidates to verify the QA-token Pareto objective."
    )
    lines.append("")
    lines.append("## 2. Compared Profiles")
    lines.append("")
    lines.append("| profile | role |")
    lines.append("|---|---|")
    for profile in profiles:
        lines.append(f"| {profile} | {_profile_role(profile)} |")
    lines.append("")

    lines.append("## 3. QA Metrics")
    lines.append("")
    lines.append(
        "| dataset | profile | EM | F1 | prompt_tokens_avg | completion_tokens_avg | retrieval_ms | generation_ms | total_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("EM", 0.0)),
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("prompt_tokens_avg", 0.0), 1),
                    _fmt(row.get("completion_tokens_avg", 0.0), 1),
                    _fmt(row.get("retrieval_ms", 0.0), 1),
                    _fmt(row.get("generation_ms", 0.0), 1),
                    _fmt(row.get("total_ms", 0.0), 1),
                )
            )
    lines.append("")

    lines.append("## 4. Retrieval + Density")
    lines.append("")
    lines.append(
        "| dataset | profile | rendered_sf_R | rendered_sf_P | rendered_tokens | rendered_sf_F1_per_1k_tokens | render_drop_rate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("rendered_sf_R", 0.0)),
                    _fmt(row.get("rendered_sf_P", 0.0)),
                    _fmt(row.get("rendered_tokens", 0.0), 1),
                    _fmt(row.get("rendered_sf_F1_per_1k_tokens", 0.0)),
                    _fmt(row.get("render_drop_rate", 0.0)),
                )
            )
    lines.append("")

    lines.append("## 5. Pareto Status (QA F1 vs prompt_tokens)")
    lines.append("")
    lines.append("| dataset | profile | F1 | prompt_tokens_avg | F1_per_1k_prompt_tokens | pareto_status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for dataset in datasets:
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            key = (dataset, profile, "qa_f1_vs_prompt_tokens")
            status = "pareto" if key in frontier_set else "dominated"
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("prompt_tokens_avg", 0.0), 1),
                    _fmt(row.get("sf_F1_per_1k_prompt_tokens", 0.0)),
                    status,
                )
            )
    lines.append("")
    lines.append("## 6. Recommendation")
    lines.append("")
    lines.append("- Fill after reviewing gate outcomes.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6P ACR results from QA and retrieval outputs.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--qa-root", required=True, help="Root containing qa_runs/*/run_manifest.json")
    parser.add_argument(
        "--retrieval-audit-root",
        required=True,
        help="Root containing aggregate_by_profile_dataset.csv (or /audit subdir).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-md-name", default="PHASE6P_ACR_SUMMARY.md")
    args = parser.parse_args()

    datasets = _tokens(args.datasets)
    profiles = _tokens(args.profiles)
    qa_root = Path(args.qa_root).resolve()
    retrieval_root = Path(args.retrieval_audit_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_raw_rows = _collect_qa_summary_rows(input_root=qa_root, datasets=datasets)
    qa_rows = _summarize_qa_rows(qa_rows=qa_raw_rows, datasets=datasets, profiles=profiles)

    retrieval_rows: List[Dict[str, Any]] = []
    retrieval_csv_path = ""
    try:
        retrieval_csv = _find_retrieval_aggregate_csv(retrieval_root)
        retrieval_rows = _read_csv(retrieval_csv)
        retrieval_csv_path = str(retrieval_csv)
    except Exception:
        retrieval_rows = []
        retrieval_csv_path = ""
    retrieval_index = _prepare_retrieval_index(retrieval_rows)

    merged_rows: List[Dict[str, Any]] = []
    for row in qa_rows:
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))
        r = dict(retrieval_index.get((dataset, profile), {}) or {})
        merged_rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "EM": _safe_float(row.get("EM", 0.0), 0.0),
                "F1": _safe_float(row.get("F1", 0.0), 0.0),
                "prompt_tokens_avg": _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0),
                "completion_tokens_avg": _safe_float(row.get("completion_tokens_avg", 0.0), 0.0),
                "retrieval_ms": _safe_float(row.get("retrieval_ms", 0.0), 0.0),
                "generation_ms": _safe_float(row.get("generation_ms", 0.0), 0.0),
                "total_ms": _safe_float(row.get("total_ms", 0.0), 0.0),
                "sf_F1_per_1k_prompt_tokens": _safe_float(row.get("sf_F1_per_1k_prompt_tokens", 0.0), 0.0),
                "qa_num_runs": _safe_int(row.get("qa_num_runs", 0), 0),
                "qa_num_queries": _safe_int(row.get("qa_num_queries", 0), 0),
                "qa_available": bool(row.get("qa_available", False)),
                "rendered_sf_R": _safe_float(r.get("rendered_sf_R", 0.0), 0.0),
                "rendered_sf_P": _safe_float(r.get("rendered_sf_P", 0.0), 0.0),
                "rendered_sf_F1": _safe_float(r.get("rendered_sf_F1", 0.0), 0.0),
                "rendered_sf_F1_per_1k_tokens": _safe_float(r.get("rendered_sf_F1_per_1k_tokens", 0.0), 0.0),
                "rendered_tokens": _safe_float(r.get("rendered_tokens", r.get("rendered_tokens_avg", 0.0)), 0.0),
                "render_drop_rate": _safe_float(r.get("render_drop_rate", r.get("selector_drop_rate", 0.0)), 0.0),
            }
        )

    frontier_rows: List[Dict[str, Any]] = []
    frontier_specs = [
        ("qa_f1_vs_prompt_tokens", ["F1"], ["prompt_tokens_avg"]),
        ("qa_f1_vs_rendered_tokens", ["F1"], ["rendered_tokens"]),
    ]
    for dataset in datasets:
        ds_rows = [dict(r) for r in merged_rows if _safe_text(r.get("dataset")) == dataset]
        for frontier_name, maximize, minimize in frontier_specs:
            base_rows = [r for r in ds_rows if bool(r.get("qa_available", False)) and _safe_int(r.get("qa_num_queries", 0), 0) > 0]
            if not base_rows:
                continue
            frontier = _pareto_frontier(base_rows, maximize=maximize, minimize=minimize)
            for row in frontier:
                frontier_rows.append(
                    {
                        "dataset": dataset,
                        "profile": _safe_text(row.get("profile")),
                        "frontier_name": frontier_name,
                        "maximize": ",".join(maximize),
                        "minimize": ",".join(minimize),
                        "F1": _safe_float(row.get("F1", 0.0), 0.0),
                        "prompt_tokens_avg": _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0),
                        "rendered_tokens": _safe_float(row.get("rendered_tokens", 0.0), 0.0),
                        "sf_F1_per_1k_prompt_tokens": _safe_float(row.get("sf_F1_per_1k_prompt_tokens", 0.0), 0.0),
                    }
                )

    qa_pareto_csv = output_dir / "qa_pareto_by_profile.csv"
    frontier_csv = output_dir / "pareto_frontier.csv"
    summary_md = output_dir / str(args.summary_md_name)

    qa_fields = [
        "dataset",
        "profile",
        "qa_num_runs",
        "qa_num_queries",
        "qa_available",
        "EM",
        "F1",
        "prompt_tokens_avg",
        "completion_tokens_avg",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "sf_F1_per_1k_prompt_tokens",
        "rendered_sf_R",
        "rendered_sf_P",
        "rendered_sf_F1",
        "rendered_sf_F1_per_1k_tokens",
        "rendered_tokens",
        "render_drop_rate",
    ]
    frontier_fields = [
        "dataset",
        "profile",
        "frontier_name",
        "maximize",
        "minimize",
        "F1",
        "prompt_tokens_avg",
        "rendered_tokens",
        "sf_F1_per_1k_prompt_tokens",
    ]
    _write_csv(qa_pareto_csv, merged_rows, qa_fields)
    _write_csv(frontier_csv, frontier_rows, frontier_fields)

    md = _build_markdown(
        datasets=datasets,
        profiles=profiles,
        merged_rows=merged_rows,
        frontier_rows=frontier_rows,
    )
    summary_md.write_text(md + "\n", encoding="utf-8")

    run_manifest = {
        "datasets": list(datasets),
        "profiles": list(profiles),
        "qa_root": str(qa_root),
        "retrieval_audit_root": str(retrieval_root),
        "retrieval_aggregate_csv": retrieval_csv_path,
        "output_dir": str(output_dir),
        "summary_md": str(summary_md),
        "num_qa_rows_raw": int(len(qa_raw_rows)),
        "num_qa_rows_merged": int(len(merged_rows)),
        "num_frontier_rows": int(len(frontier_rows)),
    }
    (output_dir / "phase6p_summary_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(str(summary_md))


if __name__ == "__main__":
    main()
