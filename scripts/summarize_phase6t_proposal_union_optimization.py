#!/usr/bin/env python3
"""Compare baseline vs optimized proposal-union profiling outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

DEFAULT_STAGES = [
    "proposal_total_ms",
    "proposal_union_total_ms",
    "raw_semantic_entity_fetch_ms",
    "raw_semantic_chunk_fetch_ms",
    "graph_reserve_fetch_ms",
    "entity_to_chunk_expand_ms",
    "chunk_candidate_lookup_ms",
    "candidate_materialization_ms",
    "chunk_text_lookup_ms",
    "metadata_lookup_ms",
    "token_count_lookup_ms",
    "score_merge_ms",
    "dedup_ms",
    "sort_topk_ms",
    "early_pruning_ms",
    "candidate_filter_ms",
    "python_loop_overhead_ms",
    "unattributed_proposal_ms",
]


def _safe_text(v: Any) -> str:
    return str(v or "").strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return obj


def _extract_single_run(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(payload.get("rows", []) or [])
    runs = list(payload.get("run_profiles", []) or [])
    if not rows or not runs:
        raise ValueError("Missing rows/run_profiles in profile summary JSON.")
    row = dict(rows[0])
    run = dict(runs[0])
    stage = dict(run.get("stage_metrics", {}) or {})
    return {
        "row": row,
        "run": run,
        "stage": stage,
    }


def _stage_avg(stage: Mapping[str, Any], key: str) -> float:
    bucket = dict((stage.get(key, {}) or {}))
    return _safe_float(bucket.get("avg", 0.0), 0.0)


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize proposal-union optimization before/after.")
    ap.add_argument("--baseline-out-root", required=True)
    ap.add_argument("--optimized-out-root", required=True)
    ap.add_argument("--output-md", default="")
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    baseline_root = Path(_safe_text(args.baseline_out_root)).resolve()
    optimized_root = Path(_safe_text(args.optimized_out_root)).resolve()

    baseline_json = baseline_root / "proposal_union_profile_summary.json"
    optimized_json = optimized_root / "proposal_union_profile_summary.json"

    base_payload = _read_json(baseline_json)
    opt_payload = _read_json(optimized_json)

    base = _extract_single_run(base_payload)
    opt = _extract_single_run(opt_payload)

    base_row = dict(base["row"])
    opt_row = dict(opt["row"])
    base_stage = dict(base["stage"])
    opt_stage = dict(opt["stage"])

    table_rows: List[Dict[str, Any]] = []
    proposal_before = _stage_avg(base_stage, "proposal_total_ms")
    proposal_after = _stage_avg(opt_stage, "proposal_total_ms")
    for key in DEFAULT_STAGES:
        before = _stage_avg(base_stage, key)
        after = _stage_avg(opt_stage, key)
        delta = after - before
        delta_pct = (delta / before * 100.0) if before > 0 else 0.0
        table_rows.append(
            {
                "stage": key,
                "baseline_avg_ms": before,
                "optimized_avg_ms": after,
                "delta_ms": delta,
                "delta_percent": delta_pct,
                "share_of_proposal_before": (before / proposal_before) if proposal_before > 0 else 0.0,
                "share_of_proposal_after": (after / proposal_after) if proposal_after > 0 else 0.0,
            }
        )

    md_path = Path(
        _safe_text(args.output_md)
        or str(optimized_root / "PHASE6T_PROPOSAL_UNION_OPTIMIZATION_SUMMARY.md")
    ).resolve()
    json_path = Path(
        _safe_text(args.output_json)
        or str(optimized_root / "proposal_union_optimization_summary.json")
    ).resolve()

    lines: List[str] = []
    lines.append("# Phase-6T Proposal-Union Optimization Summary")
    lines.append("")
    lines.append("## 1. Profiling Setup")
    lines.append("")
    lines.append(f"- Baseline root: `{baseline_root}`")
    lines.append(f"- Optimized root: `{optimized_root}`")
    lines.append("")

    lines.append("## 2. Baseline Latency")
    lines.append("")
    lines.append(f"- retrieval_ms(avg): {_fmt(_stage_avg(base_stage, 'retrieval_ms'), 2)}")
    lines.append(f"- proposal_total_ms(avg): {_fmt(_stage_avg(base_stage, 'proposal_total_ms'), 2)}")
    lines.append(f"- proposal_union_total_ms(avg): {_fmt(_stage_avg(base_stage, 'proposal_union_total_ms'), 2)}")
    lines.append("")

    lines.append("## 3. Proposal Union Breakdown")
    lines.append("")
    lines.append("| stage | baseline_avg_ms | optimized_avg_ms | delta_ms | delta_percent | share_of_proposal_before | share_of_proposal_after |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in table_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("stage")),
                _fmt(row.get("baseline_avg_ms"), 2),
                _fmt(row.get("optimized_avg_ms"), 2),
                _fmt(row.get("delta_ms"), 2),
                _fmt(row.get("delta_percent"), 2),
                _fmt(row.get("share_of_proposal_before"), 4),
                _fmt(row.get("share_of_proposal_after"), 4),
            )
        )
    lines.append("")

    lines.append("## 4. Dominant Bottleneck")
    lines.append("")
    lines.append("- Compare the largest `share_of_proposal_before` stage with after-change value.")
    lines.append("")

    lines.append("## 5. Optimization Applied")
    lines.append("")
    lines.append("- This summary compares baseline vs one targeted optimization run.")
    lines.append("")

    lines.append("## 6. Before/After Comparison")
    lines.append("")
    lines.append("| metric | baseline | optimized | delta |")
    lines.append("|---|---:|---:|---:|")
    retrieval_before = _stage_avg(base_stage, "retrieval_ms")
    retrieval_after = _stage_avg(opt_stage, "retrieval_ms")
    lines.append(f"| retrieval_ms(avg) | {_fmt(retrieval_before,2)} | {_fmt(retrieval_after,2)} | {_fmt(retrieval_after-retrieval_before,2)} |")
    lines.append(f"| proposal_total_ms(avg) | {_fmt(proposal_before,2)} | {_fmt(proposal_after,2)} | {_fmt(proposal_after-proposal_before,2)} |")
    lines.append(
        f"| proposal_union_total_ms(avg) | {_fmt(_stage_avg(base_stage, 'proposal_union_total_ms'),2)} | {_fmt(_stage_avg(opt_stage, 'proposal_union_total_ms'),2)} | {_fmt(_stage_avg(opt_stage, 'proposal_union_total_ms')-_stage_avg(base_stage, 'proposal_union_total_ms'),2)} |"
    )
    lines.append("")

    lines.append("## 7. QA/Evidence Stability")
    lines.append("")
    lines.append("| metric | baseline | optimized | delta |")
    lines.append("|---|---:|---:|---:|")
    for key in ("EM", "F1", "prompt_tokens_avg", "supporting_fact_recall", "supporting_fact_precision"):
        b = _safe_float(base_row.get(key, 0.0), 0.0)
        o = _safe_float(opt_row.get(key, 0.0), 0.0)
        lines.append(f"| {key} | {_fmt(b,4)} | {_fmt(o,4)} | {_fmt(o-b,4)} |")
    lines.append("")

    lines.append("## 8. Recommendation")
    lines.append("")
    lines.append("- Promote the optimization if proposal_union_total_ms drops materially and F1/SF metrics remain stable.")
    lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_out_root": str(baseline_root),
        "optimized_out_root": str(optimized_root),
        "table_rows": table_rows,
        "baseline_row": base_row,
        "optimized_row": opt_row,
    }
    _write_json(json_path, payload)

    print(str(md_path))


if __name__ == "__main__":
    main()
