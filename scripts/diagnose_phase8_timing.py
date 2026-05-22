#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from effirag.phase8_diagnostics import group_runs, iter_queries, load_runs, md_table, safe_float, stats, write_json, write_md


STAGES = [
    "entity_universe_ms",
    "sampling_ms",
    "kmedoids_ms",
    "seed_selection_ms",
    "refinement_ms",
    "evidence_proposal_ms",
    "feature_construction_ms",
    "selection_ms",
    "rendering_ms",
    "generation_ms",
    "total_retrieval_ms",
    "total_ms",
]


def _stage_value(query: Dict[str, Any], stage: str) -> float:
    timing = query.get("timing", {})
    if isinstance(timing, dict):
        return safe_float(timing.get(stage, 0.0), 0.0)
    return 0.0


def _summarize_group(dataset: str, profile: str, variant: str, group: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries = list(iter_queries(group))
    row: Dict[str, Any] = {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "num_queries": len(queries),
    }
    retrieval_vals = [_stage_value(q, "total_retrieval_ms") for q in queries]
    retrieval_mean = stats(retrieval_vals)["mean"]
    for stage in STAGES:
        st = stats(_stage_value(q, stage) for q in queries)
        row[f"{stage}_mean"] = st["mean"]
        row[f"{stage}_p50"] = st["p50"]
        row[f"{stage}_p95"] = st["p95"]
        row[f"{stage}_max"] = st["max"]
        if stage not in {"total_retrieval_ms", "total_ms"}:
            row[f"{stage}_share_of_retrieval"] = st["mean"] / retrieval_mean if retrieval_mean > 0 else 0.0
    pamae_stages = ["entity_universe_ms", "sampling_ms", "kmedoids_ms", "seed_selection_ms", "refinement_ms", "evidence_proposal_ms"]
    row["largest_phase8_stage"] = max(pamae_stages, key=lambda s: row.get(f"{s}_mean", 0.0))
    row["largest_phase8_stage_ms"] = row.get(f"{row['largest_phase8_stage']}_mean", 0.0)
    row["interpretation"] = _interpret(row)
    return row


def _interpret(row: Dict[str, Any]) -> str:
    if row.get("variant") == "source_balanced_128":
        return "Baseline timing; Phase8 stages should be zero."
    largest = row.get("largest_phase8_stage", "")
    if largest == "entity_universe_ms":
        return "Entity universe construction is the largest Phase8 stage."
    if largest == "evidence_proposal_ms":
        return "Evidence proposal is the largest Phase8 stage."
    if largest in {"sampling_ms", "kmedoids_ms"}:
        return "Sampling/k-medoids distance computation is the largest Phase8 cost."
    return f"Largest Phase8 stage: {largest}."


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Phase8 timing bottlenecks.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = load_runs(root)
    rows = []
    for (dataset, profile, variant), group in sorted(group_runs(runs).items()):
        rows.append(_summarize_group(dataset, profile, variant, group))
    write_json(root / "phase8_timing_bottleneck.json", {"runs": rows})

    keys = ["dataset", "profile", "variant", "num_queries"]
    for stage in STAGES:
        keys.extend([f"{stage}_mean", f"{stage}_p50", f"{stage}_p95", f"{stage}_max"])
    keys.extend(["largest_phase8_stage", "largest_phase8_stage_ms", "interpretation"])
    compact_keys = [
        "dataset",
        "profile",
        "variant",
        "num_queries",
        "entity_universe_ms_mean",
        "sampling_ms_mean",
        "kmedoids_ms_mean",
        "seed_selection_ms_mean",
        "refinement_ms_mean",
        "evidence_proposal_ms_mean",
        "feature_construction_ms_mean",
        "selection_ms_mean",
        "generation_ms_mean",
        "total_retrieval_ms_mean",
        "total_ms_mean",
        "largest_phase8_stage",
        "interpretation",
    ]
    lines = [
        "# Phase8 Timing Bottleneck Report",
        "",
        "## Compact Timing",
        "",
        md_table(rows, compact_keys),
        "",
        "## Detailed Timing",
        "",
        md_table(rows, keys),
        "",
        "## Explicit Answers",
        "",
    ]
    for row in rows:
        if row.get("variant") == "source_balanced_128":
            continue
        label = f"{row['dataset']}/{row['profile']}/{row['variant']}"
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Entity universe dominant: {row['largest_phase8_stage'] == 'entity_universe_ms'}",
                f"- K-medoids expensive: {(row['sampling_ms_mean'] + row['kmedoids_ms_mean']) > row['entity_universe_ms_mean'] or (row['sampling_ms_mean'] + row['kmedoids_ms_mean']) > row['evidence_proposal_ms_mean']}",
                f"- Evidence proposal expensive: {row['evidence_proposal_ms_mean'] > 2000.0}",
                f"- Refinement meaningful cost: {row['refinement_ms_mean'] > 1000.0}",
                f"- Optimize first: {row['largest_phase8_stage']}",
                "",
            ]
        )
    write_md(root / "phase8_timing_bottleneck_report.md", lines)
    print(root / "phase8_timing_bottleneck_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
