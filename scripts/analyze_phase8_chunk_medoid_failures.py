#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import (
    md_table,
    metric_from_result,
    normalized_query_diagnostics,
    read_json,
    read_jsonl,
    safe_float,
    write_json,
    write_md,
)


ACTIVE_VARIANTS = {"chunk_pamae_k5"}
CHUNK_CATEGORIES = [
    "CHUNK_UNIVERSE_MISS",
    "CHUNK_SEED_MISS",
    "CHUNK_TO_EVIDENCE_MISS",
    "CANDIDATE_CHAIN_INFEASIBLE",
    "FINAL_SELECTION_FAILED",
    "SELECTED_NOT_SUFFICIENT",
    "QA_PROMPT_FAILED",
    "OTHER",
]
TRACE_NAMES = [
    "rag_query_results.jsonl",
    "phase7_query_trace.jsonl",
    "phase7_diagnostics.jsonl",
    "phase8_chunk_query_trace.jsonl",
    "phase8_chunk_seed_trace.jsonl",
    "phase8_chunk_evidence_trace.jsonl",
]


def _run_dirs(root: Path) -> List[Path]:
    found = set()
    for name in ["rag_summary.json", *TRACE_NAMES]:
        for pattern in (f"*/*/*/{name}", f"*/*/*/*/{name}"):
            for path in root.glob(pattern):
                found.add(path.parent)
    return sorted(found)


def _parts(root: Path, run_dir: Path) -> Tuple[str, str, str, str, str]:
    parts = list(run_dir.relative_to(root).parts)
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], parts[3], "isolated"
    profile = parts[0] if len(parts) > 0 else ""
    variant = parts[1] if len(parts) > 1 else ""
    dataset = parts[2] if len(parts) > 2 else ""
    return dataset, profile, variant, "", "legacy"


def _query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("query_id", row.get("sample_id", row.get("qid", ""))) or "")


def _dedupe_rows(rows: Iterable[Mapping[str, Any]], *, run_id: str = "", latest_timestamp_only: bool = False) -> Dict[str, Dict[str, Any]]:
    raw = [dict(r) for r in list(rows or []) if isinstance(r, Mapping)]
    if run_id and any(str(r.get("run_id", "")) for r in raw):
        raw = [r for r in raw if str(r.get("run_id", "")) == str(run_id)]
    if latest_timestamp_only:
        timestamps = sorted({str(r.get("run_timestamp", r.get("timestamp", "")) or "") for r in raw if str(r.get("run_timestamp", r.get("timestamp", "")) or "")})
        if timestamps:
            latest = timestamps[-1]
            raw = [r for r in raw if str(r.get("run_timestamp", r.get("timestamp", "")) or "") in {"", latest}]
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        qid = _query_id(row)
        if qid:
            out[qid] = dict(row)
    return out


def _load_run(root: Path, run_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    dataset, profile, variant, run_id, layout = _parts(root, run_dir)
    summary = read_json(run_dir / "rag_summary.json")
    result_rows = _dedupe_rows(read_jsonl(run_dir / "rag_query_results.jsonl"), run_id=run_id, latest_timestamp_only=bool(args.latest_only))
    timestamps = sorted({str(r.get("run_timestamp", "") or "") for r in result_rows.values() if str(r.get("run_timestamp", "") or "")})
    timestamp = timestamps[-1] if timestamps else str(summary.get("run_timestamp", "") or "")
    effective_run_id = run_id or timestamp or run_dir.name
    maps = {
        "result": result_rows,
        "phase7": _dedupe_rows(read_jsonl(run_dir / "phase7_query_trace.jsonl"), run_id=effective_run_id if layout == "isolated" else ""),
        "phase7_diag": _dedupe_rows(read_jsonl(run_dir / "phase7_diagnostics.jsonl"), run_id=effective_run_id if layout == "isolated" else ""),
        "chunk_query": _dedupe_rows(read_jsonl(run_dir / "phase8_chunk_query_trace.jsonl"), run_id=effective_run_id if layout == "isolated" else ""),
        "chunk_seed": _dedupe_rows(read_jsonl(run_dir / "phase8_chunk_seed_trace.jsonl"), run_id=effective_run_id if layout == "isolated" else ""),
        "chunk_evidence": _dedupe_rows(read_jsonl(run_dir / "phase8_chunk_evidence_trace.jsonl"), run_id=effective_run_id if layout == "isolated" else ""),
    }
    qids = set()
    for mapping in maps.values():
        qids.update(mapping.keys())
    expected = int(summary.get("num_queries", summary.get("count", args.expected_limit or 0)) or (args.expected_limit or 0))
    queries = [
        {
            "query_id": qid,
            "result": maps["result"].get(qid, {}),
            "phase7": maps["phase7"].get(qid, {}),
            "phase7_diag": maps["phase7_diag"].get(qid, {}),
            "chunk_query": maps["chunk_query"].get(qid, {}),
            "chunk_seed": maps["chunk_seed"].get(qid, {}),
            "chunk_evidence": maps["chunk_evidence"].get(qid, {}),
        }
        for qid in sorted(qids)
    ]
    status = "COMPLETE" if (not expected or len(queries) == expected) else "PARTIAL"
    if not queries:
        status = "MISSING"
    return {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "run_id": effective_run_id,
        "timestamp": timestamp or effective_run_id,
        "run_dir": str(run_dir),
        "expected_num_queries": expected or len(queries),
        "actual_num_queries": len(queries),
        "status": status,
        "queries": queries,
    }


def _latest_run(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    complete = [r for r in group if str(r.get("status", "")).startswith("COMPLETE")]
    pool = complete or list(group)
    return sorted(pool, key=lambda r: (str(r.get("timestamp", "")), str(r.get("run_id", "")), str(r.get("run_dir", ""))))[-1]


def _select_runs(runs: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    runs = [r for r in runs if str(r.get("variant", "")) in ACTIVE_VARIANTS]
    if not args.latest_only:
        return runs
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((str(run["dataset"]), str(run["profile"]), str(run["variant"])), []).append(run)
    return [_latest_run(group) for group in grouped.values()]


def _seed_section(query: Mapping[str, Any], key: str) -> Dict[str, Any]:
    seed = query.get("chunk_seed")
    if isinstance(seed, dict):
        val = seed.get(key)
        if isinstance(val, dict):
            return val
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        diag = p7.get("phase8_chunk_diagnostics")
        if isinstance(diag, dict):
            val = diag.get(key)
            if isinstance(val, dict):
                return val
    return {}


def _evidence_section(query: Mapping[str, Any]) -> Dict[str, Any]:
    ev = query.get("chunk_evidence")
    if isinstance(ev, dict):
        val = ev.get("chunk_medoid_evidence_proposal")
        if isinstance(val, dict):
            return val
    p7 = query.get("phase7")
    if isinstance(p7, dict):
        diag = p7.get("phase8_chunk_diagnostics")
        if isinstance(diag, dict):
            val = diag.get("chunk_medoid_evidence_proposal")
            if isinstance(val, dict):
                return val
    return {}


def _question(query: Mapping[str, Any]) -> str:
    for key in ("phase7", "chunk_query", "phase7_diag", "result"):
        row = query.get(key)
        if isinstance(row, dict) and row.get("question"):
            return str(row.get("question"))
    return ""


def _label(query: Mapping[str, Any]) -> str:
    diag = normalized_query_diagnostics(query)
    candidate_recall = safe_float(diag.get("normalized_candidate_gold_recall"), 0.0)
    selected_recall = safe_float(diag.get("normalized_selected_gold_recall"), 0.0)
    rendered_recall = safe_float(diag.get("normalized_rendered_gold_recall"), 0.0)
    f1 = metric_from_result(query, "f1")
    sf_recall = metric_from_result(query, "sf_recall")
    chunk_query = query.get("chunk_query") if isinstance(query.get("chunk_query"), dict) else {}
    seed_sel = _seed_section(query, "chunk_seed_selection")
    evidence = _evidence_section(query)
    if safe_float(chunk_query.get("chunk_universe_size", 0.0), 0.0) <= 0.0:
        return "CHUNK_UNIVERSE_MISS"
    if safe_float(seed_sel.get("best_seed_relevance", 0.0), 0.0) <= 0.05 and candidate_recall <= 0.10:
        return "CHUNK_SEED_MISS"
    if safe_float(evidence.get("num_final_evidence_candidates", 0.0), 0.0) <= 0.0 or candidate_recall <= 0.10:
        return "CHUNK_TO_EVIDENCE_MISS"
    feasible = diag.get("chain_unit_oracle_feasible")
    if feasible is False and candidate_recall > 0.0:
        return "CANDIDATE_CHAIN_INFEASIBLE"
    if selected_recall + 1e-9 < candidate_recall:
        return "FINAL_SELECTION_FAILED"
    if rendered_recall < 1.0 and sf_recall < 1.0:
        return "SELECTED_NOT_SUFFICIENT"
    if rendered_recall > 0.0 and f1 < 0.5:
        return "QA_PROMPT_FAILED"
    return "OTHER"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase8 chunk-medoid Phase8-only probe failures.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--dedupe-query-id", action="store_true")
    parser.add_argument("--expected-limit", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = _select_runs([_load_run(root, d, args) for d in _run_dirs(root)], args)
    summaries = []
    rows = []
    examples = defaultdict(list)
    for run in sorted(runs, key=lambda r: (r["dataset"], r["profile"], r["variant"], r["run_id"])):
        counter: Counter = Counter()
        for query in list(run.get("queries", []) or []):
            label = _label(query)
            counter[label] += 1
            diag = normalized_query_diagnostics(query)
            row = {
                "dataset": run["dataset"],
                "profile": run["profile"],
                "variant": run["variant"],
                "run_id": run["run_id"],
                "query_id": query.get("query_id", ""),
                "category": label,
                "question": _question(query),
                "normalized_candidate_gold_recall": diag.get("normalized_candidate_gold_recall"),
                "normalized_selected_gold_recall": diag.get("normalized_selected_gold_recall"),
                "F1": metric_from_result(query, "f1"),
                "SF-R": metric_from_result(query, "sf_recall"),
            }
            rows.append(row)
            key = (run["dataset"], run["profile"], run["variant"], label)
            if len(examples[key]) < 5:
                examples[key].append(row)
        total = int(len(run.get("queries", []) or []))
        for category in CHUNK_CATEGORIES:
            count = int(counter.get(category, 0))
            summaries.append(
                {
                    "dataset": run["dataset"],
                    "profile": run["profile"],
                    "variant": run["variant"],
                    "run_id": run["run_id"],
                    "category": category,
                    "count": count,
                    "rate": float(count / total) if total else 0.0,
                    "num_queries": total,
                }
            )

    payload = {"summary": summaries, "rows": rows, "examples": {"/".join(k): v for k, v in examples.items()}}
    write_json(root / "phase8_chunk_medoid_failure_analysis.json", payload)
    write_json(root / "phase8_chunk_probe_failure_analysis.json", payload)
    lines = [
        "# Phase8 Chunk Probe Failure Analysis",
        "",
        "This failure analysis is Phase8 chunk-only. Source-balanced and bridge-refine variants are excluded unless a separate historical report is requested.",
        "",
        md_table(summaries, ["dataset", "profile", "variant", "run_id", "category", "count", "rate", "num_queries"]),
        "",
    ]
    write_md(root / "phase8_chunk_medoid_failure_analysis.md", lines)
    write_md(root / "phase8_chunk_probe_failure_analysis.md", lines)
    print(root / "phase8_chunk_probe_failure_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
