#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_STAGES = [
    "index_validation",
    "query_parse",
    "query_embedding",
    "query_entity_extraction",
    "query_constraint_extraction",
    "semantic_anchor_retrieval",
    "entity_anchor_resolution",
    "local_graph_build",
    "graph_flow",
    "phase1_candidate_assembly",
    "safe_pruning_invalid",
    "safe_pruning_dedup",
    "safe_pruning_dominance",
    "feature_A_scoring",
    "feature_anchor_distance",
    "corridor_anchor_selection",
    "corridor_seed_selection",
    "corridor_extraction",
    "feature_Bq_scoring",
    "feature_R_scoring",
    "marginal_selection",
    "rendering",
    "prompt_construction",
    "qa_generation",
    "evaluation",
    "oracle_context_construction",
    "oracle_qa_replay",
    "total_retrieval",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _pct(values: Iterable[float]) -> Dict[str, float]:
    seq = [float(v) for v in list(values or [])]
    if not seq:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(seq, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def _fmt2(x: float) -> str:
    return f"{float(x):.2f}"


def _iter_stage_rows(root: Path) -> Iterable[Dict[str, Any]]:
    for p in sorted(root.rglob("phase7_stage_timing.jsonl")):
        for row in _read_jsonl(p):
            out = dict(row or {})
            out["_path"] = str(p)
            yield out


def _bottleneck_flag(stage: str, p95: float) -> bool:
    if stage == "query_embedding":
        return bool(p95 > 1000.0)
    if stage == "semantic_anchor_retrieval":
        return bool(p95 > 1000.0)
    if stage == "graph_flow":
        return bool(p95 > 2000.0)
    if stage == "corridor_extraction":
        return bool(p95 > 1000.0)
    if stage in {"feature_A_scoring", "feature_Bq_scoring", "feature_R_scoring"}:
        return bool(p95 > 1000.0)
    if stage == "marginal_selection":
        return bool(p95 > 500.0)
    if stage == "rendering":
        return bool(p95 > 300.0)
    if stage == "qa_generation":
        return bool(p95 > 3000.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize stage timing for Phase7 runs.")
    ap.add_argument("--root", type=str, required=True, help="Run root containing one or more phase7_stage_timing.jsonl files")
    ap.add_argument("--stages", type=str, default=",".join(DEFAULT_STAGES), help="Comma-separated stage names to include first.")
    ap.add_argument("--out", type=str, default="outputs/phase7_evidence_flow/oracle_gap_decomposition/phase7_stage_timing_summary.md")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    wanted = [s.strip() for s in str(args.stages).split(",") if s.strip()]

    grouped: Dict[str, List[float]] = defaultdict(list)
    by_query_stage: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)

    for row in _iter_stage_rows(root):
        stage = str(row.get("stage", "") or "")
        qid = str(row.get("query_id", "") or "")
        ds = str(row.get("dataset", "") or "")
        if not stage:
            continue
        elapsed = _safe_float(row.get("elapsed_ms", 0.0), 0.0)
        grouped[stage].append(elapsed)
        if qid:
            by_query_stage[(ds, qid)][stage] = elapsed

    stage_order = list(wanted)
    for stage in sorted(grouped.keys()):
        if stage not in stage_order:
            stage_order.append(stage)

    rows_md: List[List[str]] = []
    stats_json: Dict[str, Any] = {}
    for stage in stage_order:
        vals = list(grouped.get(stage, []) or [])
        if not vals:
            continue
        stat = _pct(vals)
        flag = _bottleneck_flag(stage, float(stat["p95"]))
        stats_json[stage] = dict(stat)
        stats_json[stage]["bottleneck_flag"] = bool(flag)
        rows_md.append(
            [
                stage,
                _fmt2(stat["mean"]),
                _fmt2(stat["p50"]),
                _fmt2(stat["p90"]),
                _fmt2(stat["p95"]),
                _fmt2(stat["p99"]),
                _fmt2(stat["max"]),
                "true" if flag else "false",
            ]
        )

    per_query_rows_md: List[List[str]] = []
    for (ds, qid), stage_map in sorted(by_query_stage.items(), key=lambda x: (x[0][0], x[0][1])):
        if not stage_map:
            continue
        slow_stage, slow_ms = max(stage_map.items(), key=lambda kv: kv[1])
        total_retrieval = _safe_float(stage_map.get("total_retrieval", 0.0), 0.0)
        total_ms = _safe_float(stage_map.get("total_query", stage_map.get("total_retrieval", 0.0)), 0.0)
        per_query_rows_md.append(
            [
                f"{ds}:{qid}",
                str(slow_stage),
                _fmt2(slow_ms),
                _fmt2(total_retrieval),
                _fmt2(total_ms),
            ]
        )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md: List[str] = []
    md.append("# Phase7 Stage Timing Summary")
    md.append("")
    md.append(f"- root: `{root}`")
    md.append("")
    md.append(_table(["stage", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "bottleneck_flag"], rows_md))
    md.append("")
    md.append("## Per-Query Slowest Stage")
    md.append(_table(["query_id", "slowest_stage", "slowest_ms", "total_retrieval_ms", "total_ms"], per_query_rows_md[:200]))
    out_path.write_text("\n".join(md), encoding="utf-8")

    flags = {
        "query_embedding_p95_gt_1000ms": bool(stats_json.get("query_embedding", {}).get("p95", 0.0) > 1000.0),
        "semantic_anchor_p95_gt_1000ms": bool(stats_json.get("semantic_anchor_retrieval", {}).get("p95", 0.0) > 1000.0),
        "graph_flow_p95_gt_2000ms": bool(stats_json.get("graph_flow", {}).get("p95", 0.0) > 2000.0),
        "corridor_extraction_p95_gt_1000ms": bool(stats_json.get("corridor_extraction", {}).get("p95", 0.0) > 1000.0),
        "feature_scoring_p95_gt_1000ms": bool(
            max(
                stats_json.get("feature_A_scoring", {}).get("p95", 0.0),
                stats_json.get("feature_Bq_scoring", {}).get("p95", 0.0),
                stats_json.get("feature_R_scoring", {}).get("p95", 0.0),
            )
            > 1000.0
        ),
        "marginal_selection_p95_gt_500ms": bool(stats_json.get("marginal_selection", {}).get("p95", 0.0) > 500.0),
        "rendering_p95_gt_300ms": bool(stats_json.get("rendering", {}).get("p95", 0.0) > 300.0),
        "qa_generation_p95_gt_3000ms": bool(stats_json.get("qa_generation", {}).get("p95", 0.0) > 3000.0),
    }

    (out_path.parent / (out_path.stem + ".json")).write_text(
        json.dumps(
            {
                "root": str(root),
                "stage_stats": stats_json,
                "flags": flags,
                "per_query_slowest": per_query_rows_md,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"root": str(root), "flags": flags}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
