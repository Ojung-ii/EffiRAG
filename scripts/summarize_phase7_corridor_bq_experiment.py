#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = str(line or "").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _git(args: List[str], default: str = "") -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip() or default
    except Exception:
        return default


def _percentiles(vals: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in list(vals or [])], dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "null"
    return f"{_safe_float(x, 0.0):.{nd}f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    s = "| " + " | ".join(["---"] * len(headers)) + " |"
    b = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([h, s] + b)


def _discover_summary_rows(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(root.glob("*/*/*/diagnostic_summary.json")):
        d = _read_json(p, {})
        if not isinstance(d, dict):
            continue
        profile = p.parts[-4]
        variant = p.parts[-3]
        dataset = p.parts[-2]
        row = dict(d)
        row["profile"] = profile
        row["variant_name"] = variant
        row["dataset"] = dataset
        row["run_dir"] = str(p.parent.resolve())
        out.append(row)
    return out


def _compute_stage_summary(root: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    by_stage: Dict[str, List[float]] = defaultdict(list)
    for p in sorted(root.glob("*/*/*/phase7_stage_timing.jsonl")):
        rows = _read_jsonl(p)
        for r in rows:
            st = str(r.get("stage", "") or "")
            if not st:
                continue
            by_stage[st].append(_safe_float(r.get("elapsed_ms", 0.0), 0.0))
    stats = {k: _percentiles(v) for k, v in sorted(by_stage.items())}
    flags: List[str] = []
    if _safe_float(stats.get("corridor_extraction", {}).get("p95", 0.0), 0.0) > 1000.0:
        flags.append("corridor_extraction_p95_gt_1000ms")
    if _safe_float(stats.get("anchor_distance_bfs", {}).get("p95", 0.0), 0.0) > 500.0:
        flags.append("anchor_distance_bfs_p95_gt_500ms")
    if _safe_float(stats.get("marginal_selection", {}).get("p95", 0.0), 0.0) > 500.0:
        flags.append("marginal_selection_p95_gt_500ms")
    return stats, flags


def _sf_f1(r: float, p: float) -> float:
    rr = _safe_float(r, 0.0)
    pp = _safe_float(p, 0.0)
    if (rr + pp) <= 0.0:
        return 0.0
    return float(2.0 * rr * pp / (rr + pp))


def _index_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in rows:
        out[(str(r.get("profile", "")), str(r.get("variant_name", "")), str(r.get("dataset", "")))] = r
    return out


def _mean(vals: Iterable[float]) -> float:
    seq = [float(v) for v in list(vals or [])]
    return float(sum(seq) / len(seq)) if seq else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Phase7 corridor-Bq experiment outputs.")
    ap.add_argument("--root", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = _discover_summary_rows(root)
    if not rows:
        raise FileNotFoundError(f"No diagnostic_summary.json found under: {root}")

    stage_stats, stage_flags = _compute_stage_summary(root)

    # 1) corridor_bq_summary.md
    main_rows: List[List[str]] = []
    survival_rows: List[List[str]] = []
    quality_rows: List[List[str]] = []

    for r in sorted(rows, key=lambda x: (str(x.get("profile", "")), str(x.get("variant_name", "")), str(x.get("dataset", "")))):
        sf_r = _safe_float(r.get("SF_recall", 0.0), 0.0)
        sf_p = _safe_float(r.get("SF_precision", 0.0), 0.0)
        sf_f1 = _sf_f1(sf_r, sf_p)
        main_rows.append(
            [
                str(r.get("profile", "")),
                str(r.get("variant_name", "")),
                str(r.get("dataset", "")),
                _fmt(r.get("EM", 0.0), 4),
                _fmt(r.get("F1", 0.0), 4),
                _fmt(sf_r, 4),
                _fmt(sf_p, 4),
                _fmt(sf_f1, 4),
                _fmt(r.get("avg_context_tokens", 0.0), 2),
                _fmt(r.get("F1_per_1k", 0.0), 4),
                _fmt(r.get("avg_retrieval_ms", 0.0), 2),
                _fmt(r.get("avg_total_ms", 0.0), 2),
            ]
        )
        survival_rows.append(
            [
                str(r.get("profile", "")),
                str(r.get("variant_name", "")),
                str(r.get("dataset", "")),
                _fmt(r.get("phase1_gold_hit_rate", 0.0), 4),
                _fmt(r.get("corridor_gold_hit_rate", 0.0), 4),
                _fmt(r.get("selected_gold_hit_rate", 0.0), 4),
                _fmt(r.get("rendered_gold_hit_rate", 0.0), 4),
                _fmt(r.get("selected_to_rendered_match_rate", 0.0), 4),
            ]
        )
        quality_rows.append(
            [
                str(r.get("profile", "")),
                str(r.get("variant_name", "")),
                str(r.get("dataset", "")),
                _fmt(r.get("gold_avg_Bq", None), 4) if r.get("gold_avg_Bq", None) is not None else "null",
                _fmt(r.get("distractor_avg_Bq", None), 4) if r.get("distractor_avg_Bq", None) is not None else "null",
                _fmt(r.get("gold_rank_by_Bq", None), 2) if r.get("gold_rank_by_Bq", None) is not None else "null",
                _fmt(r.get("paths_found_avg", None), 2) if r.get("paths_found_avg", None) is not None else "null",
                _fmt(r.get("corridor_ms_p95", 0.0), 2),
            ]
        )

    summary_md: List[str] = []
    summary_md.append("# Corridor-Bq Experiment Summary")
    summary_md.append("")
    summary_md.append("## Main Result Table")
    summary_md.append(
        _table(
            [
                "profile",
                "variant",
                "dataset",
                "EM",
                "F1",
                "SF-R",
                "SF-P",
                "SF-F1",
                "avg_context_tokens",
                "F1_per_1k",
                "retrieval_ms",
                "total_ms",
            ],
            main_rows,
        )
    )
    summary_md.append("")
    summary_md.append("## Evidence Survival")
    summary_md.append(
        _table(
            [
                "profile",
                "variant",
                "dataset",
                "phase1_hit",
                "corridor_gold_hit",
                "selected_hit",
                "rendered_hit",
                "selected_to_rendered_match",
            ],
            survival_rows,
        )
    )
    summary_md.append("")
    summary_md.append("## Corridor Quality")
    summary_md.append(
        _table(
            [
                "profile",
                "variant",
                "dataset",
                "gold_avg_Bq",
                "distractor_avg_Bq",
                "gold_rank_by_Bq",
                "paths_found_avg",
                "corridor_ms_p95",
            ],
            quality_rows,
        )
    )
    (root / "corridor_bq_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    # 2) stage_timing_summary.md
    stage_rows = []
    for stage in sorted(stage_stats.keys()):
        s = stage_stats.get(stage, {}) or {}
        stage_rows.append(
            [
                stage,
                _fmt(s.get("mean", 0.0), 2),
                _fmt(s.get("p50", 0.0), 2),
                _fmt(s.get("p90", 0.0), 2),
                _fmt(s.get("p95", 0.0), 2),
                _fmt(s.get("p99", 0.0), 2),
            ]
        )
    stage_md = []
    stage_md.append("# Stage Timing Summary")
    stage_md.append("")
    stage_md.append(_table(["stage", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"], stage_rows))
    stage_md.append("")
    stage_md.append("## Bottleneck Flags")
    if stage_flags:
        for f in stage_flags:
            stage_md.append(f"- {f}")
    else:
        stage_md.append("- none")
    (root / "stage_timing_summary.md").write_text("\n".join(stage_md), encoding="utf-8")

    # 3) phase7_corridor_bq_report.md
    idx = _index_rows(rows)

    def _cmp(profile: str, dataset: str, left: str, right: str, key: str) -> Optional[float]:
        l = idx.get((profile, left, dataset))
        r = idx.get((profile, right, dataset))
        if not l or not r:
            return None
        return float(_safe_float(r.get(key, 0.0), 0.0) - _safe_float(l.get(key, 0.0), 0.0))

    deltas_bq_selected = []
    deltas_bq_sfr = []
    deltas_rq_selected = []
    deltas_rq_f1 = []
    distractor_higher = 0
    distractor_total = 0

    profiles = sorted({str(r.get("profile", "")) for r in rows})
    datasets = sorted({str(r.get("dataset", "")) for r in rows})
    for p in profiles:
        for ds in datasets:
            d1 = _cmp(p, ds, "baseline_a_minus_r", "corridor_a_plus_bq_minus_r", "selected_gold_hit_rate")
            d2 = _cmp(p, ds, "baseline_a_minus_r", "corridor_a_plus_bq_minus_r", "SF_recall")
            if d1 is not None:
                deltas_bq_selected.append(d1)
            if d2 is not None:
                deltas_bq_sfr.append(d2)

            d3 = _cmp(p, ds, "corridor_a_plus_bq_minus_r", "corridor_conditional_r_a_plus_bq_minus_rq", "selected_gold_hit_rate")
            d4 = _cmp(p, ds, "corridor_a_plus_bq_minus_r", "corridor_conditional_r_a_plus_bq_minus_rq", "F1")
            if d3 is not None:
                deltas_rq_selected.append(d3)
            if d4 is not None:
                deltas_rq_f1.append(d4)

            rr = idx.get((p, "corridor_a_plus_bq_minus_r", ds))
            if rr:
                g = rr.get("gold_avg_Bq", None)
                d = rr.get("distractor_avg_Bq", None)
                if g is not None and d is not None:
                    distractor_total += 1
                    if _safe_float(d, 0.0) >= _safe_float(g, 0.0):
                        distractor_higher += 1

    bq_help = (_mean(deltas_bq_selected) > 0.0 and _mean(deltas_bq_sfr) > 0.0) if deltas_bq_selected and deltas_bq_sfr else False
    rq_help = (_mean(deltas_rq_selected) > 0.0 and _mean(deltas_rq_f1) >= 0.0) if deltas_rq_selected and deltas_rq_f1 else False
    bq_distractor_risk = (distractor_higher > 0 and distractor_total > 0)

    audit_log = root / "logs" / "audit_phase7_method.log"
    audit_json = _read_json(audit_log, {}) if audit_log.exists() else {}
    audit_status = str((audit_json or {}).get("status", "UNKNOWN")) if isinstance(audit_json, dict) else "UNKNOWN"

    failure_dist_json = root / "failure_case_analysis.json"
    failure_dist = _read_json(failure_dist_json, {}) if failure_dist_json.exists() else {}

    report: List[str] = []
    report.append("# Phase7 Corridor-Bq Report")
    report.append("")
    report.append(f"- branch: {_git(['git', 'branch', '--show-current'], '')}")
    report.append(f"- commit: {_git(['git', 'rev-parse', 'HEAD'], '')}")
    report.append(f"- output_root: {root}")
    report.append("")
    report.append("## Method Summary")
    report.append("- Query-conditioned sentence-atom candidate pool")
    report.append("- Phase II feature extraction only")
    report.append("- Single marginal selector with A/D/Bq/R/Rq objective modes")
    report.append("- Deterministic rendering")
    report.append("")
    report.append("## Anti-Heuristic Audit")
    report.append(f"- status: {audit_status}")
    report.append("")
    report.append("## Configs Used")
    report.append("- profiles: balanced_384_8, legacy_512_10")
    report.append("- variants: baseline_a_minus_r, anchor_decay_a_plus_d_minus_r, corridor_a_plus_bq_minus_r, corridor_conditional_r_a_plus_bq_minus_rq")
    report.append("")
    report.append("## Main Result Table")
    report.append(_table(["profile", "variant", "dataset", "EM", "F1", "SF-R", "SF-P", "SF-F1", "avg_context_tokens", "F1_per_1k", "retrieval_ms", "total_ms"], main_rows))
    report.append("")
    report.append("## Evidence Survival Table")
    report.append(_table(["profile", "variant", "dataset", "phase1_hit", "corridor_gold_hit", "selected_hit", "rendered_hit", "selected_to_rendered_match"], survival_rows))
    report.append("")
    report.append("## Corridor Quality Table")
    report.append(_table(["profile", "variant", "dataset", "gold_avg_Bq", "distractor_avg_Bq", "gold_rank_by_Bq", "paths_found_avg", "corridor_ms_p95"], quality_rows))
    report.append("")
    report.append("## Stage Timing")
    report.append(_table(["stage", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"], stage_rows))
    report.append("")
    report.append("## Stage Bottleneck Flags")
    if stage_flags:
        for f in stage_flags:
            report.append(f"- {f}")
    else:
        report.append("- none")
    report.append("")
    report.append("## Failure Subtype Distribution")
    if isinstance(failure_dist, dict) and failure_dist:
        for k in sorted(failure_dist.keys()):
            report.append(f"- {k}: {failure_dist[k]}")
    else:
        report.append("- not available (run analyze_phase7_failures.py)")
    report.append("")
    report.append("## Recommendation")
    if bq_help and not bq_distractor_risk:
        report.append("- keep Bq")
    elif bq_help and bq_distractor_risk:
        report.append("- revise Bq")
    else:
        report.append("- reject Bq")
    if rq_help:
        report.append("- test Rq")
    report.append("- proceed to Aq/Bq/Rq only after preserving anti-heuristic constraints")
    report.append("")
    report.append("## Explicit Answers")
    report.append(f"- Did Bq improve support-chain survival?: {'YES' if bq_help else 'NO/UNCLEAR'}")
    report.append(f"- Did Bq select distractor corridors?: {'YES (risk observed)' if bq_distractor_risk else 'NO clear evidence'}")
    report.append(f"- Did Rq prevent useful corridor evidence from being dropped?: {'YES' if rq_help else 'NO/UNCLEAR'}")
    report.append(f"- Is corridor extraction a bottleneck?: {'YES' if 'corridor_extraction_p95_gt_1000ms' in stage_flags else 'NO'}")
    report.append(f"- Is method still non-heuristic and dataset-agnostic?: {'YES' if audit_status == 'PASS' else 'CHECK FAILED'}")

    (root / "phase7_corridor_bq_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[corridor-bq] wrote {root / 'corridor_bq_summary.md'}")
    print(f"[corridor-bq] wrote {root / 'stage_timing_summary.md'}")
    print(f"[corridor-bq] wrote {root / 'phase7_corridor_bq_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
