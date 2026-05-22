#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import md_table, mean, normalized_query_diagnostics, read_json, read_jsonl, safe_float, stats, write_json, write_md


TIMING_KEYS = [
    "query_embed_ms",
    "phase8_pamae_proposal_ms",
    "phase8_chunk_medoid_proposal_ms",
    "phase8_local_graph_build_ms",
    "phase8_graph_flow_ms",
    "source_balanced_union_ms",
    "feature_extraction_ms",
    "corridor_extraction_ms",
    "marginal_selection_ms",
    "retrieval_total_ms",
    "generation_ms",
    "total_ms",
]


def _query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("query_id", row.get("sample_id", row.get("qid", ""))) or "")


def _dedupe(rows: Iterable[Mapping[str, Any]], run_id: str = "") -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    raw = [dict(r) for r in list(rows or []) if isinstance(r, Mapping)]
    if run_id and any(str(r.get("run_id", "")) for r in raw):
        raw = [r for r in raw if str(r.get("run_id", "")) == str(run_id)]
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        qid = _query_id(row)
        if qid:
            out[qid] = dict(row)
    return out, {"raw_rows": len(raw), "unique_queries": len(out), "duplicate_rows": max(0, len(raw) - len(out))}


def _run_dirs(root: Path) -> List[Path]:
    found = set()
    for name in ("rag_summary.json", "rag_query_results.jsonl", "phase7_query_trace.jsonl"):
        for path in root.glob(f"*/*/*/*/{name}"):
            found.add(path.parent)
    return sorted(found)


def _parts(root: Path, run_dir: Path) -> Tuple[str, str, str, str]:
    parts = list(run_dir.relative_to(root).parts)
    dataset = parts[0] if len(parts) > 0 else ""
    profile = parts[1] if len(parts) > 1 else ""
    variant = parts[2] if len(parts) > 2 else ""
    run_id = parts[3] if len(parts) > 3 else ""
    return dataset, profile, variant, run_id


def _load_run(root: Path, run_dir: Path, expected_limit: int) -> Dict[str, Any]:
    dataset, profile, variant, run_id = _parts(root, run_dir)
    summary = read_json(run_dir / "rag_summary.json")
    result, result_info = _dedupe(read_jsonl(run_dir / "rag_query_results.jsonl"), run_id)
    p7, p7_info = _dedupe(read_jsonl(run_dir / "phase7_query_trace.jsonl"), run_id)
    p7diag, p7diag_info = _dedupe(read_jsonl(run_dir / "phase7_diagnostics.jsonl"), run_id)
    qids = sorted(set(result) | set(p7) | set(p7diag))
    queries = [
        {
            "query_id": qid,
            "result": result.get(qid, {}),
            "phase7": p7.get(qid, {}),
            "phase7_diag": p7diag.get(qid, {}),
        }
        for qid in qids
    ]
    expected = int(summary.get("num_queries", summary.get("count", expected_limit or 0)) or expected_limit or len(queries))
    warnings = []
    for name, info in {
        "rag_query_results.jsonl": result_info,
        "phase7_query_trace.jsonl": p7_info,
        "phase7_diagnostics.jsonl": p7diag_info,
    }.items():
        if int(info.get("duplicate_rows", 0)) > 0:
            warnings.append(f"{name}: duplicate_rows={info['duplicate_rows']}")
    if expected and len(queries) != expected:
        warnings.append(f"expected={expected} actual={len(queries)}")
    status = "COMPLETE" if queries and not warnings else ("MISSING" if not queries else "COMPLETE_WITH_WARNINGS")
    timestamp = str(summary.get("run_timestamp", "") or run_id)
    return {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "run_id": run_id,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "expected_num_queries": expected,
        "actual_num_queries": len(queries),
        "status": status,
        "warnings": warnings,
        "queries": queries,
    }


def _latest_run(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    complete = [r for r in group if str(r.get("status", "")).startswith("COMPLETE")]
    pool = complete or group
    return sorted(pool, key=lambda r: (str(r.get("timestamp", "")), str(r.get("run_id", "")), str(r.get("run_dir", ""))))[-1]


def _select_runs(runs: List[Dict[str, Any]], latest_only: bool) -> List[Dict[str, Any]]:
    if not latest_only:
        return runs
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["dataset"], run["profile"], run["variant"]), []).append(run)
    return [_latest_run(group) for group in grouped.values()]


def _metric(query: Mapping[str, Any], key: str) -> float:
    result = query.get("result")
    metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
    aliases = {
        "EM": "exact_match",
        "F1": "f1",
        "SF-R": "supporting_fact_recall",
        "SF-P": "supporting_fact_precision",
        "SF-F1": "supporting_fact_f1",
    }
    return safe_float(metrics.get(aliases.get(key, key), 0.0) if isinstance(metrics, Mapping) else 0.0, 0.0)


def _diag(query: Mapping[str, Any]) -> Dict[str, Any]:
    p7 = query.get("phase7")
    return p7 if isinstance(p7, dict) else {}


def _source_balance(query: Mapping[str, Any]) -> Dict[str, Any]:
    diag = _diag(query)
    val = diag.get("phase7_source_balance_diagnostics")
    return dict(val) if isinstance(val, dict) else {}


def _latency(query: Mapping[str, Any], key: str) -> float:
    result = query.get("result")
    if isinstance(result, Mapping):
        eff = result.get("efficiency")
        if key == "generation_ms" and isinstance(eff, Mapping):
            return safe_float(eff.get("generation_ms", eff.get("generation_latency_ms", 0.0)), 0.0)
        if key == "total_ms" and isinstance(eff, Mapping):
            return safe_float(eff.get("total_latency_ms", 0.0), 0.0)
        retrieval = result.get("retrieval")
        if isinstance(retrieval, Mapping):
            diagnostics = retrieval.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                latency = diagnostics.get("latency_breakdown_ms")
                if isinstance(latency, Mapping):
                    return safe_float(latency.get(key, 0.0), 0.0)
        latency = result.get("latency_breakdown_ms")
        if isinstance(latency, Mapping):
            return safe_float(latency.get(key, 0.0), 0.0)
    diag = _diag(query)
    latency = diag.get("latency_breakdown_ms")
    if isinstance(latency, Mapping):
        return safe_float(latency.get(key, 0.0), 0.0)
    return 0.0


def _summarize(run: Mapping[str, Any]) -> Dict[str, Any]:
    queries = list(run.get("queries", []) or [])
    nds = [normalized_query_diagnostics(q) for q in queries]
    sbs = [_source_balance(q) for q in queries]
    row: Dict[str, Any] = {
        "dataset": run["dataset"],
        "profile": run["profile"],
        "variant": run["variant"],
        "run_id": run["run_id"],
        "status": run["status"],
        "num_queries": len(queries),
        "expected_num_queries": run.get("expected_num_queries", len(queries)),
        "trace_warning_count": len(run.get("warnings", []) or []),
        "EM": mean(_metric(q, "EM") for q in queries),
        "F1": mean(_metric(q, "F1") for q in queries),
        "SF-R": mean(_metric(q, "SF-R") for q in queries),
        "SF-P": mean(_metric(q, "SF-P") for q in queries),
        "SF-F1": mean(_metric(q, "SF-F1") for q in queries),
        "normalized_candidate_gold_recall": mean(d["normalized_candidate_gold_recall"] for d in nds if d["normalized_candidate_gold_recall"] is not None),
        "normalized_selected_gold_recall": mean(d["normalized_selected_gold_recall"] for d in nds if d["normalized_selected_gold_recall"] is not None),
        "normalized_rendered_gold_recall": mean(d["normalized_rendered_gold_recall"] for d in nds if d["normalized_rendered_gold_recall"] is not None),
        "candidate_oracle_F1": mean(d["candidate_oracle_F1"] for d in nds if d["candidate_oracle_F1"] is not None),
        "phase8_unique_added": mean(safe_float(sb.get("phase8_unique_added_count", 0.0), 0.0) for sb in sbs),
        "phase8_duplicate_count": mean(safe_float(sb.get("phase8_duplicate_count", 0.0), 0.0) for sb in sbs),
        "augmented_candidate_count": mean(safe_float(sb.get("augmented_candidate_count", 0.0), 0.0) for sb in sbs),
    }
    for key in TIMING_KEYS:
        s = stats(_latency(q, key) for q in queries)
        row[key] = s["mean"]
        row[f"{key}_p95"] = s["p95"]
    return row


def _delta_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        if row["variant"] == "source_balanced_128":
            baseline[(row["dataset"], row["profile"])] = row
    out = []
    for row in rows:
        base = baseline.get((row["dataset"], row["profile"]))
        if not base or row["variant"] == "source_balanced_128":
            continue
        out.append(
            {
                "dataset": row["dataset"],
                "profile": row["profile"],
                "variant": row["variant"],
                "delta_candidate_recall": row["normalized_candidate_gold_recall"] - base["normalized_candidate_gold_recall"],
                "delta_selected_recall": row["normalized_selected_gold_recall"] - base["normalized_selected_gold_recall"],
                "delta_F1": row["F1"] - base["F1"],
                "delta_SF-R": row["SF-R"] - base["SF-R"],
                "retrieval_ms_ratio": row["retrieval_total_ms"] / float(base["retrieval_total_ms"] or 1.0),
                "phase8_unique_added": row["phase8_unique_added"],
            }
        )
    return out


def _md(rows: List[Dict[str, Any]]) -> List[str]:
    main_keys = ["dataset", "profile", "variant", "status", "num_queries", "EM", "F1", "SF-R", "SF-P", "SF-F1", "normalized_candidate_gold_recall", "normalized_selected_gold_recall", "candidate_oracle_F1", "retrieval_total_ms"]
    aug_keys = ["dataset", "profile", "variant", "phase8_unique_added", "phase8_duplicate_count", "augmented_candidate_count", "phase8_pamae_proposal_ms", "phase8_chunk_medoid_proposal_ms", "phase8_local_graph_build_ms", "phase8_graph_flow_ms"]
    delta_keys = ["dataset", "profile", "variant", "delta_candidate_recall", "delta_selected_recall", "delta_F1", "delta_SF-R", "retrieval_ms_ratio", "phase8_unique_added"]
    return [
        "# Phase8 Phase7 Augmentation Summary",
        "",
        "Phase8 is treated as an augmentation source on top of the Phase7 source-balanced backbone. Final selection/rendering/QA remain Phase7.",
        "",
        "## Main Metrics",
        "",
        md_table(rows, main_keys),
        "",
        "## Augmentation Diagnostics",
        "",
        md_table(rows, aug_keys),
        "",
        "## Delta Vs Source-Balanced Reference",
        "",
        md_table(_delta_rows(rows), delta_keys),
        "",
        "## Decision Rule",
        "",
        "- Keep augmentation only if candidate recall or selected recall improves without material F1/SF-P drop and retrieval_ms_ratio stays near 1.30 or below.",
        "- If candidate recall does not improve, Phase8 proposal is not adding useful evidence and should be redesigned rather than scaled.",
        "- If candidate recall improves but selected recall/F1 do not, the bottleneck is Phase7 selection interaction, not proposal generation.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase8-on-Phase7 augmentation runs.")
    parser.add_argument("--root", default="outputs/phase8_phase7_augmentation")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--expected-limit", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = [_load_run(root, d, int(args.expected_limit or 0)) for d in _run_dirs(root)]
    selected = _select_runs(runs, latest_only=bool(args.latest_only))
    rows = [_summarize(r) for r in sorted(selected, key=lambda x: (x["dataset"], x["profile"], x["variant"], x["run_id"]))]
    write_json(root / "phase8_phase7_augmentation_summary.json", {"runs": rows, "selected_runs": selected})
    write_md(root / "phase8_phase7_augmentation_summary.md", _md(rows))
    print(root / "phase8_phase7_augmentation_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
