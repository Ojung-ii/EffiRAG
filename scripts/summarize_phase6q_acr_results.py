#!/usr/bin/env python3
"""Summarize Phase-6Q ACR v1.1 QA smoke outputs into one markdown report."""

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
    "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
    "unified_acr_v1",
    "unified_acr_v1_no_noise_penalty",
    "unified_acr_v1_no_ordering",
    "unified_acr_v1_beam3",
    "unified_acr_v11",
    "unified_acr_v11_no_structure",
    "unified_acr_v11_no_cost",
    "unified_acr_v11_ordered",
    "unified_acr_v11_beam3",
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
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge": "phase6m compact reference",
        "unified_acr_v1": "ACR v1 main",
        "unified_acr_v1_no_noise_penalty": "v1 ablation",
        "unified_acr_v1_no_ordering": "v1 ablation",
        "unified_acr_v1_beam3": "v1 beam-search ablation",
        "unified_acr_v11": "ACR v1.1 main",
        "unified_acr_v11_no_structure": "v1.1 ablation",
        "unified_acr_v11_no_cost": "v1.1 ablation",
        "unified_acr_v11_ordered": "v1.1 ablation",
        "unified_acr_v11_beam3": "v1.1 beam-search ablation",
    }
    return role_map.get(profile, "candidate")


def _build_index(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): dict(r) for r in rows}


def _delta_rows(
    rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    profiles: Sequence[str],
    baseline_profile: str = "unified_large",
) -> List[Dict[str, Any]]:
    by_key = _build_index(rows)
    out: List[Dict[str, Any]] = []
    for dataset in datasets:
        base = by_key.get((dataset, baseline_profile), {})
        base_f1 = _safe_float(base.get("F1", 0.0), 0.0)
        base_prompt = _safe_float(base.get("prompt_tokens_avg", 0.0), 0.0)
        base_f1k = _safe_float(base.get("sf_F1_per_1k_prompt_tokens", 0.0), 0.0)
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            out.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "delta_F1_vs_unified_large": _safe_float(row.get("F1", 0.0), 0.0) - base_f1,
                    "delta_prompt_tokens_vs_unified_large": _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0)
                    - base_prompt,
                    "delta_F1_per_1k_vs_unified_large": _safe_float(row.get("sf_F1_per_1k_prompt_tokens", 0.0), 0.0)
                    - base_f1k,
                }
            )
    return out


def _gate_status(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    by_key = _build_index(rows)

    def _f(dataset: str, profile: str, key: str) -> float:
        return _safe_float(by_key.get((dataset, profile), {}).get(key, 0.0), 0.0)

    hp = "hotpotqa"
    tw = "2wikimultihopqa"
    main = "unified_acr_v11"
    large = "unified_large"
    acr_v1 = "unified_acr_v1"

    c1 = _f(hp, main, "F1") >= _f(hp, large, "F1") and _f(tw, main, "F1") >= _f(tw, large, "F1")
    c2 = _f(hp, main, "prompt_tokens_avg") <= _f(hp, large, "prompt_tokens_avg") and _f(
        tw, main, "prompt_tokens_avg"
    ) <= _f(tw, large, "prompt_tokens_avg")
    c3 = _f(hp, main, "sf_F1_per_1k_prompt_tokens") > _f(hp, large, "sf_F1_per_1k_prompt_tokens") and _f(
        tw, main, "sf_F1_per_1k_prompt_tokens"
    ) > _f(tw, large, "sf_F1_per_1k_prompt_tokens")
    c4 = _f(hp, main, "F1") > _f(hp, acr_v1, "F1")
    c5 = _f(tw, main, "F1") >= (_f(tw, acr_v1, "F1") - 0.02)

    return [
        {
            "criterion": "F1 >= unified_large on both datasets",
            "pass_fail": "pass" if c1 else "fail",
            "note": f"hotpot={_fmt(_f(hp, main, 'F1'))} vs {_fmt(_f(hp, large, 'F1'))}, 2wiki={_fmt(_f(tw, main, 'F1'))} vs {_fmt(_f(tw, large, 'F1'))}",
        },
        {
            "criterion": "prompt tokens <= unified_large",
            "pass_fail": "pass" if c2 else "fail",
            "note": f"hotpot={_fmt(_f(hp, main, 'prompt_tokens_avg'), 1)} vs {_fmt(_f(hp, large, 'prompt_tokens_avg'), 1)}, 2wiki={_fmt(_f(tw, main, 'prompt_tokens_avg'), 1)} vs {_fmt(_f(tw, large, 'prompt_tokens_avg'), 1)}",
        },
        {
            "criterion": "F1/token > unified_large",
            "pass_fail": "pass" if c3 else "fail",
            "note": f"hotpot={_fmt(_f(hp, main, 'sf_F1_per_1k_prompt_tokens'))} vs {_fmt(_f(hp, large, 'sf_F1_per_1k_prompt_tokens'))}, 2wiki={_fmt(_f(tw, main, 'sf_F1_per_1k_prompt_tokens'))} vs {_fmt(_f(tw, large, 'sf_F1_per_1k_prompt_tokens'))}",
        },
        {
            "criterion": "HotpotQA improved over ACR v1",
            "pass_fail": "pass" if c4 else "fail",
            "note": f"{_fmt(_f(hp, main, 'F1'))} vs {_fmt(_f(hp, acr_v1, 'F1'))}",
        },
        {
            "criterion": "2Wiki not worse than ACR v1 by more than 0.02",
            "pass_fail": "pass" if c5 else "fail",
            "note": f"{_fmt(_f(tw, main, 'F1'))} vs {_fmt(_f(tw, acr_v1, 'F1'))} (threshold={_fmt(_f(tw, acr_v1, 'F1') - 0.02)})",
        },
    ]


def _build_markdown(
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
    merged_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
) -> str:
    by_key = _build_index(merged_rows)
    frontier_set = {
        (_safe_text(r.get("dataset")), _safe_text(r.get("profile")), _safe_text(r.get("frontier_name")))
        for r in frontier_rows
    }
    deltas = _delta_rows(merged_rows, datasets, profiles)
    gate = _gate_status(merged_rows)
    ablation_profiles = [
        "unified_acr_v11",
        "unified_acr_v11_no_structure",
        "unified_acr_v11_no_cost",
        "unified_acr_v11_ordered",
        "unified_acr_v11_beam3",
    ]

    lines: List[str] = []
    lines.append("# Phase-6Q ACR v1.1 QA Smoke Results")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report validates whether ACR v1.1 objective simplification improves QA-token Pareto trade-offs while keeping answerability-constrained compact retrieval."
    )
    lines.append("")
    lines.append("## 2. Compared Profiles")
    lines.append("")
    lines.append("| profile | role |")
    lines.append("|---|---|")
    for profile in profiles:
        lines.append(f"| {profile} | {_profile_role(profile)} |")
    lines.append("")

    lines.append("## 3. QA Results")
    lines.append("")
    lines.append("| dataset | profile | EM | F1 | prompt_tokens | F1_per_1k_prompt | total_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("EM", 0.0)),
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("prompt_tokens_avg", 0.0), 1),
                    _fmt(row.get("sf_F1_per_1k_prompt_tokens", 0.0)),
                    _fmt(row.get("total_ms", 0.0), 1),
                )
            )
    lines.append("")

    lines.append("## 4. Baseline-relative Comparison")
    lines.append("")
    lines.append("| dataset | profile | delta_F1_vs_unified_large | delta_prompt_tokens_vs_unified_large | delta_F1_per_1k_vs_unified_large |")
    lines.append("|---|---|---:|---:|---:|")
    for row in deltas:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("delta_F1_vs_unified_large", 0.0)),
                _fmt(row.get("delta_prompt_tokens_vs_unified_large", 0.0), 1),
                _fmt(row.get("delta_F1_per_1k_vs_unified_large", 0.0)),
            )
        )
    lines.append("")

    lines.append("## 5. ACR Component Ablation")
    lines.append("")
    lines.append("| profile | interpretation | HotpotQA F1 | 2Wiki F1 | conclusion |")
    lines.append("|---|---|---:|---:|---|")
    interpretations = {
        "unified_acr_v11": "main (A+S-C, no explicit N, no forced ordering)",
        "unified_acr_v11_no_structure": "drop structure term S",
        "unified_acr_v11_no_cost": "drop cost term C",
        "unified_acr_v11_ordered": "re-enable structure-aware ordering",
        "unified_acr_v11_beam3": "beam3 set search",
    }
    for profile in ablation_profiles:
        hp = by_key.get(("hotpotqa", profile), {})
        tw = by_key.get(("2wikimultihopqa", profile), {})
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                profile,
                interpretations.get(profile, "ablation"),
                _fmt(hp.get("F1", 0.0)),
                _fmt(tw.get("F1", 0.0)),
                "TBD",
            )
        )
    lines.append("")

    lines.append("## 6. Success Gate")
    lines.append("")
    lines.append("| criterion | pass/fail | note |")
    lines.append("|---|---|---|")
    for row in gate:
        lines.append(
            f"| {row['criterion']} | {row['pass_fail']} | {row['note']} |"
        )
    lines.append("")

    lines.append("## 7. Decision")
    lines.append("")
    lines.append("- Is unified_acr_v11 the main candidate?")
    lines.append("- Is beam3 worth the cost?")
    lines.append("- Should ordering remain disabled?")
    lines.append("- Should noise penalty remain removed?")
    lines.append("- Next phase recommendation.")
    lines.append("")

    lines.append("## Pareto (QA F1 vs prompt_tokens)")
    lines.append("")
    lines.append("| dataset | profile | pareto_status |")
    lines.append("|---|---|---|")
    for dataset in datasets:
        for profile in profiles:
            key = (dataset, profile, "qa_f1_vs_prompt_tokens")
            lines.append(f"| {dataset} | {profile} | {'pareto' if key in frontier_set else 'dominated'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6Q ACR v1.1 QA smoke outputs.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--qa-root", required=True, help="Root containing qa_runs/*/run_manifest.json")
    parser.add_argument(
        "--retrieval-audit-root",
        required=True,
        help="Root containing aggregate_by_profile_dataset.csv (or /audit subdir).",
    )
    parser.add_argument(
        "--pareto-root",
        default="",
        help="Optional root containing pareto_frontier.csv from compare_qa_pareto_profiles.py",
    )
    parser.add_argument("--output", required=True, help="Output markdown file path.")
    args = parser.parse_args()

    datasets = _tokens(args.datasets)
    profiles = _tokens(args.profiles)
    qa_root = Path(args.qa_root).resolve()
    retrieval_root = Path(args.retrieval_audit_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    if str(args.pareto_root or "").strip():
        frontier_csv = Path(args.pareto_root).resolve() / "pareto_frontier.csv"
        if frontier_csv.exists():
            frontier_rows = _read_csv(frontier_csv)
    if not frontier_rows:
        for dataset in datasets:
            ds_rows = [dict(r) for r in merged_rows if _safe_text(r.get("dataset")) == dataset]
            base_rows = [
                r
                for r in ds_rows
                if bool(r.get("qa_available", False)) and _safe_int(r.get("qa_num_queries", 0), 0) > 0
            ]
            if not base_rows:
                continue
            frontier = _pareto_frontier(base_rows, maximize=["F1"], minimize=["prompt_tokens_avg"])
            for row in frontier:
                frontier_rows.append(
                    {
                        "dataset": dataset,
                        "profile": _safe_text(row.get("profile")),
                        "frontier_name": "qa_f1_vs_prompt_tokens",
                    }
                )

    qa_csv = output_path.parent / "qa_pareto_by_profile.csv"
    frontier_csv = output_path.parent / "pareto_frontier.csv"
    _write_csv(
        qa_csv,
        merged_rows,
        [
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
        ],
    )
    _write_csv(
        frontier_csv,
        frontier_rows,
        [
            "dataset",
            "profile",
            "frontier_name",
            "maximize",
            "minimize",
            "F1",
            "prompt_tokens_avg",
            "rendered_tokens",
            "sf_F1_per_1k_prompt_tokens",
            "rendered_sf_R",
            "rendered_sf_F1_per_1k_tokens",
        ],
    )

    output_path.write_text(
        _build_markdown(
            datasets=datasets,
            profiles=profiles,
            merged_rows=merged_rows,
            frontier_rows=frontier_rows,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "datasets": list(datasets),
        "profiles": list(profiles),
        "qa_root": str(qa_root),
        "retrieval_audit_root": str(retrieval_root),
        "retrieval_aggregate_csv": retrieval_csv_path,
        "pareto_root": str(args.pareto_root or ""),
        "output_summary_md": str(output_path),
        "output_qa_csv": str(qa_csv),
        "output_frontier_csv": str(frontier_csv),
        "num_qa_rows_raw": int(len(qa_raw_rows)),
        "num_qa_rows_merged": int(len(merged_rows)),
        "num_frontier_rows": int(len(frontier_rows)),
    }
    (output_path.parent / "phase6q_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
