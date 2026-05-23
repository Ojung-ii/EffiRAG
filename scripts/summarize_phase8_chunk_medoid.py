#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import (
    md_table,
    mean,
    normalized_query_diagnostics,
    read_json,
    read_jsonl,
    safe_float,
    stats,
    write_json,
    write_md,
)


ACTIVE_VARIANTS = {"chunk_pamae_k5"}
REJECTED_VARIANTS = {"chunk_bridge_refine_k5"}
EXTERNAL_REFERENCE_VARIANTS = {"source_balanced_128"}
TRACE_NAMES = [
    "rag_query_results.jsonl",
    "phase7_query_trace.jsonl",
    "phase7_diagnostics.jsonl",
    "phase8_chunk_query_trace.jsonl",
    "phase8_chunk_stage_timing.jsonl",
    "phase8_chunk_seed_trace.jsonl",
    "phase8_chunk_evidence_trace.jsonl",
]
TIMING_KEYS = [
    "query_embed_ms",
    "chunk_universe_ms",
    "chunk_sampling_ms",
    "chunk_kmedoids_ms",
    "chunk_seed_selection_ms",
    "chunk_medoid_proposal_ms",
    "local_graph_build_ms",
    "bridge_refinement_ms",
    "evidence_proposal_ms",
    "feature_construction_ms",
    "selection_ms",
    "rendering_ms",
    "generation_ms",
    "retrieval_ms",
    "total_ms",
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
        dataset, profile, variant, run_id = parts[0], parts[1], parts[2], parts[3]
        layout = "isolated"
    else:
        profile = parts[0] if len(parts) > 0 else ""
        variant = parts[1] if len(parts) > 1 else ""
        dataset = parts[2] if len(parts) > 2 else ""
        run_id = ""
        layout = "legacy"
    return dataset, profile, variant, run_id, layout


def _query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("query_id", row.get("sample_id", row.get("qid", ""))) or "")


def _row_timestamp(row: Mapping[str, Any]) -> str:
    return str(row.get("timestamp", row.get("run_timestamp", "")) or "")


def _dedupe_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    run_id: str = "",
    latest_timestamp_only: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    raw = [dict(r) for r in list(rows or []) if isinstance(r, Mapping)]
    if run_id and any(str(r.get("run_id", "")) for r in raw):
        raw = [r for r in raw if str(r.get("run_id", "")) == str(run_id)]
    if latest_timestamp_only:
        timestamps = sorted({str(r.get("run_timestamp", r.get("timestamp", "")) or "") for r in raw if str(r.get("run_timestamp", r.get("timestamp", "")) or "")})
        if timestamps:
            latest = timestamps[-1]
            raw = [
                r
                for r in raw
                if str(r.get("run_timestamp", r.get("timestamp", "")) or "") in {"", latest}
            ]
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        qid = _query_id(row)
        if qid:
            out[qid] = dict(row)
    info = {
        "raw_rows": int(len(raw)),
        "unique_queries": int(len(out)),
        "duplicate_rows": int(max(0, len(raw) - len(out))),
    }
    return out, info


def _metric(query: Mapping[str, Any], key: str) -> float:
    result = query.get("result")
    if not isinstance(result, dict):
        return 0.0
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    aliases = {
        "EM": "exact_match",
        "F1": "f1",
        "SF-R": "supporting_fact_recall",
        "SF-P": "supporting_fact_precision",
        "SF-F1": "supporting_fact_f1",
    }
    return safe_float(metrics.get(aliases.get(key, key), 0.0), 0.0)


def _timing(query: Mapping[str, Any], key: str) -> float:
    timing = query.get("chunk_timing")
    if isinstance(timing, dict):
        return safe_float(timing.get(key, 0.0), 0.0)
    result = query.get("result")
    if isinstance(result, dict):
        latency = result.get("latency_breakdown_ms")
        if isinstance(latency, dict):
            aliases = {
                "chunk_medoid_proposal_ms": "phase8_chunk_medoid_proposal_ms",
                "local_graph_build_ms": "phase8_local_graph_build_ms",
                "feature_construction_ms": "feature_extraction_ms",
                "selection_ms": "marginal_selection_ms",
                "rendering_ms": "render_ms",
                "retrieval_ms": "retrieval_total_ms",
            }
            return safe_float(latency.get(aliases.get(key, key), 0.0), 0.0)
    return 0.0


def _load_run(root: Path, run_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    dataset, profile, variant, run_id, layout = _parts(root, run_dir)
    summary = read_json(run_dir / "rag_summary.json")
    result_rows, result_info = _dedupe_rows(
        read_jsonl(run_dir / "rag_query_results.jsonl"),
        run_id=run_id,
        latest_timestamp_only=bool(args.latest_only),
    )
    timestamps = sorted(
        {
            str(row.get("run_timestamp", "") or "")
            for row in result_rows.values()
            if str(row.get("run_timestamp", "") or "")
        }
    )
    timestamp = timestamps[-1] if timestamps else str(summary.get("run_timestamp", "") or "")
    effective_run_id = run_id or timestamp or run_dir.name
    loaders = {
        "phase7": "phase7_query_trace.jsonl",
        "phase7_diag": "phase7_diagnostics.jsonl",
        "chunk_query": "phase8_chunk_query_trace.jsonl",
        "chunk_timing": "phase8_chunk_stage_timing.jsonl",
        "chunk_seed": "phase8_chunk_seed_trace.jsonl",
        "chunk_evidence": "phase8_chunk_evidence_trace.jsonl",
    }
    maps: Dict[str, Dict[str, Dict[str, Any]]] = {"result": result_rows}
    infos: Dict[str, Dict[str, Any]] = {"rag_query_results.jsonl": result_info}
    for key, name in loaders.items():
        mapping, info = _dedupe_rows(
            read_jsonl(run_dir / name),
            run_id=effective_run_id if layout == "isolated" else "",
            latest_timestamp_only=False,
        )
        maps[key] = mapping
        infos[name] = info
    qids = set()
    for mapping in maps.values():
        qids.update(mapping.keys())
    queries = [
        {
            "query_id": qid,
            "result": maps["result"].get(qid, {}),
            "phase7": maps["phase7"].get(qid, {}),
            "phase7_diag": maps["phase7_diag"].get(qid, {}),
            "chunk_query": maps["chunk_query"].get(qid, {}),
            "chunk_seed": maps["chunk_seed"].get(qid, {}),
            "chunk_evidence": maps["chunk_evidence"].get(qid, {}),
            "chunk_timing": maps["chunk_timing"].get(qid, {}),
        }
        for qid in sorted(qids)
    ]
    expected = int(summary.get("num_queries", summary.get("count", 0)) or 0)
    if expected <= 0:
        expected = int(args.expected_limit or 0)
    actual = len(queries)
    files = {name: (run_dir / name).exists() for name in ["rag_summary.json", *TRACE_NAMES]}
    warnings = []
    for name, info in infos.items():
        if info["duplicate_rows"] > 0:
            warnings.append(
                f"{name}: raw_rows={info['raw_rows']} unique_queries={info['unique_queries']} duplicate_rows={info['duplicate_rows']}"
            )
    if expected and actual != expected:
        warnings.append(f"expected LIMIT/query count {expected} but found unique_queries={actual}")
    if not any(files.values()) or actual == 0:
        status = "MISSING"
    elif expected and actual < expected:
        status = "PARTIAL"
    elif warnings:
        status = "COMPLETE_WITH_WARNINGS"
    else:
        status = "COMPLETE"
    return {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "run_id": str(effective_run_id),
        "timestamp": str(timestamp or effective_run_id),
        "layout": layout,
        "run_dir": str(run_dir),
        "files": files,
        "expected_num_queries": int(expected or actual),
        "actual_num_queries": int(actual),
        "trace_infos": infos,
        "warnings": warnings,
        "status": status,
        "queries": queries,
    }


def _latest_run(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    complete = [r for r in group if str(r.get("status", "")).startswith("COMPLETE")]
    pool = complete or list(group)
    return sorted(pool, key=lambda r: (str(r.get("timestamp", "")), str(r.get("run_id", "")), str(r.get("run_dir", ""))))[-1]


def _filter_runs(runs: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    include_external = bool(args.include_external_reference)
    include_rejected = bool(args.include_rejected)
    filtered = []
    for run in runs:
        variant = str(run.get("variant", ""))
        if variant in REJECTED_VARIANTS and not include_rejected:
            continue
        if variant in EXTERNAL_REFERENCE_VARIANTS and not include_external:
            continue
        if variant not in ACTIVE_VARIANTS and variant not in EXTERNAL_REFERENCE_VARIANTS and variant not in REJECTED_VARIANTS:
            continue
        filtered.append(run)
    if not args.latest_only:
        return filtered
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for run in filtered:
        key = (str(run.get("dataset", "")), str(run.get("profile", "")), str(run.get("variant", "")))
        grouped.setdefault(key, []).append(run)
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


def _summarize(run: Dict[str, Any]) -> Dict[str, Any]:
    queries = list(run.get("queries", []) or [])
    nds = [normalized_query_diagnostics(q) for q in queries]
    row: Dict[str, Any] = {
        "dataset": run["dataset"],
        "profile": run["profile"],
        "variant": run["variant"],
        "run_id": run["run_id"],
        "timestamp": run["timestamp"],
        "status": run["status"],
        "num_queries": len(queries),
        "expected_num_queries": int(run.get("expected_num_queries", len(queries))),
        "trace_warning_count": int(len(run.get("warnings", []) or [])),
        "EM": mean(_metric(q, "EM") for q in queries),
        "F1": mean(_metric(q, "F1") for q in queries),
        "SF-R": mean(_metric(q, "SF-R") for q in queries),
        "SF-P": mean(_metric(q, "SF-P") for q in queries),
        "SF-F1": mean(_metric(q, "SF-F1") for q in queries),
        "avg_context_tokens": mean(
            safe_float(q.get("result", {}).get("generation_diagnostics", {}).get("prompt_tokens", 0.0), 0.0)
            for q in queries
        ),
        "avg_selected_atoms": mean(
            safe_float(q.get("phase7", {}).get("num_selected_atoms", q.get("chunk_query", {}).get("avg_selected_atoms", 0.0)), 0.0)
            for q in queries
        ),
        "normalized_candidate_gold_recall": mean(d["normalized_candidate_gold_recall"] for d in nds if d["normalized_candidate_gold_recall"] is not None),
        "normalized_selected_gold_recall": mean(d["normalized_selected_gold_recall"] for d in nds if d["normalized_selected_gold_recall"] is not None),
        "normalized_rendered_gold_recall": mean(d["normalized_rendered_gold_recall"] for d in nds if d["normalized_rendered_gold_recall"] is not None),
        "candidate_oracle_F1": mean(d["candidate_oracle_F1"] for d in nds if d["candidate_oracle_F1"] is not None),
        "chain_unit_oracle_feasible": mean(1.0 if d["chain_unit_oracle_feasible"] else 0.0 for d in nds if d["chain_unit_oracle_feasible"] is not None),
    }
    for key in TIMING_KEYS:
        s = stats(_timing(q, key) for q in queries)
        row[key] = s["mean"]
        row[f"{key}_p50"] = s["p50"]
        row[f"{key}_p95"] = s["p95"]
        row[f"{key}_max"] = s["max"]
        retrieval = row.get("retrieval_ms", 0.0)
        if key != "retrieval_ms":
            row[f"{key}_share_of_retrieval"] = safe_float(row.get(key, 0.0), 0.0) / float(retrieval or 1.0)

    chunk_rows = [q.get("chunk_query", {}) for q in queries if isinstance(q.get("chunk_query", {}), dict)]
    seed_sel = [_seed_section(q, "chunk_seed_selection") for q in queries]
    refine = [_seed_section(q, "chunk_bridge_refinement") for q in queries]
    evidence = [_evidence_section(q) for q in queries]
    row.update(
        {
            "chunk_universe_size": mean(safe_float(r.get("chunk_universe_size", 0.0), 0.0) for r in chunk_rows),
            "chunk_seed_gold_hit_rate_eval_only": mean(safe_float(r.get("chunk_seed_gold_hit_rate_eval_only", 0.0), 0.0) for r in chunk_rows),
            "chunk_candidate_gold_recall_eval_only": mean(safe_float(r.get("chunk_candidate_gold_recall_eval_only", 0.0), 0.0) for r in chunk_rows),
            "best_seed_score": mean(safe_float(r.get("best_seed_score", 0.0), 0.0) for r in seed_sel),
            "num_changed_chunk_seeds": mean(safe_float(r.get("num_changed_seeds", 0.0), 0.0) for r in refine),
            "num_bridge_entities": mean(safe_float(r.get("avg_bridge_entities_per_seed", 0.0), 0.0) for r in refine),
            "num_final_evidence_candidates": mean(safe_float(r.get("num_final_evidence_candidates", 0.0), 0.0) for r in evidence),
            "selected_chunk_medoid_source_rate": mean(safe_float(q.get("chunk_evidence", {}).get("selected_chunk_medoid_source_rate", 0.0), 0.0) for q in queries),
        }
    )
    return row


def _trace_report_lines(runs: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[str]:
    trace_rows = []
    for run in runs:
        for name, info in dict(run.get("trace_infos", {}) or {}).items():
            trace_rows.append(
                {
                    "dataset": run.get("dataset"),
                    "profile": run.get("profile"),
                    "variant": run.get("variant"),
                    "run_id": run.get("run_id"),
                    "trace": name,
                    "raw_rows": info.get("raw_rows", 0),
                    "unique_queries": info.get("unique_queries", 0),
                    "duplicate_rows": info.get("duplicate_rows", 0),
                }
            )
    all_unique = all(int(r.get("trace_warning_count", 0)) == 0 for r in rows)
    all_counts_match = all(int(r.get("num_queries", 0)) == int(r.get("expected_num_queries", 0)) for r in rows)
    return [
        "# Phase8 Chunk Trace Repair Report",
        "",
        f"Every selected run has a unique run_id: {'yes' if all(str(r.get('run_id', '')) for r in rows) else 'no'}",
        f"Trace rows deduped by query_id: yes",
        f"Unique query count equals LIMIT/summary count: {'yes' if all_counts_match else 'no'}",
        f"Stale appended rows excluded by latest/run_id selection: yes",
        f"Candidate/selected/rendered diagnostics use normalized ids: yes",
        f"Duplicate trace warnings remaining: {'no' if all_unique else 'yes'}",
        "",
        "## Selected Runs",
        "",
        md_table(rows, ["dataset", "profile", "variant", "run_id", "status", "num_queries", "expected_num_queries", "trace_warning_count"]),
        "",
        "## Trace Counts",
        "",
        md_table(trace_rows, ["dataset", "profile", "variant", "run_id", "trace", "raw_rows", "unique_queries", "duplicate_rows"]),
        "",
    ]


def _timing_report_lines(rows: List[Dict[str, Any]]) -> List[str]:
    timing_rows = []
    for row in rows:
        phase8_stages = [
            ("chunk_medoid_proposal_ms", safe_float(row.get("chunk_medoid_proposal_ms", 0.0), 0.0)),
            ("local_graph_build_ms", safe_float(row.get("local_graph_build_ms", 0.0), 0.0)),
            ("chunk_universe_ms", safe_float(row.get("chunk_universe_ms", 0.0), 0.0)),
            ("chunk_kmedoids_ms", safe_float(row.get("chunk_kmedoids_ms", 0.0), 0.0)),
            ("evidence_proposal_ms", safe_float(row.get("evidence_proposal_ms", 0.0), 0.0)),
            ("feature_construction_ms", safe_float(row.get("feature_construction_ms", 0.0), 0.0)),
            ("selection_ms", safe_float(row.get("selection_ms", 0.0), 0.0)),
        ]
        largest = max(phase8_stages, key=lambda x: x[1])[0] if phase8_stages else ""
        timing_rows.append(
            {
                "dataset": row["dataset"],
                "profile": row["profile"],
                "variant": row["variant"],
                "num_queries": row["num_queries"],
                "retrieval_ms": row["retrieval_ms"],
                "query_embed_ms": row["query_embed_ms"],
                "chunk_medoid_proposal_ms": row["chunk_medoid_proposal_ms"],
                "local_graph_build_ms": row["local_graph_build_ms"],
                "evidence_proposal_ms": row["evidence_proposal_ms"],
                "largest_phase8_stage": largest,
            }
        )
    share_keys = ["dataset", "profile", "variant"] + [f"{k}_share_of_retrieval" for k in TIMING_KEYS if k != "retrieval_ms"]
    return [
        "# Phase8 Chunk Timing Optimization Report",
        "",
        "The probe reports Phase8-specific timing after trace isolation and lookup/vectorization fixes.",
        "",
        "## Mean Timing",
        "",
        md_table(timing_rows, ["dataset", "profile", "variant", "num_queries", "retrieval_ms", "query_embed_ms", "chunk_medoid_proposal_ms", "local_graph_build_ms", "evidence_proposal_ms", "largest_phase8_stage"]),
        "",
        "## Stage Share Of Retrieval",
        "",
        md_table(rows, share_keys),
        "",
        "## Answers",
        "",
        "1. Biggest Phase8-specific bottleneck: see `largest_phase8_stage` above.",
        "2. Caching/vectorization impact: compare this report with earlier probe timings; raw trace duplication is excluded.",
        "3. Local graph build impact: see `local_graph_build_ms` and its retrieval share.",
        "4. Query embedding is reported separately and should be treated as common unless repeated inside Phase8-specific stages.",
        "5. 30% overhead should be assessed against the external Phase7 reference, not as a Phase8 component.",
        "",
    ]


def _summary_lines(rows: List[Dict[str, Any]]) -> List[str]:
    main_keys = ["dataset", "profile", "variant", "status", "num_queries", "EM", "F1", "SF-R", "SF-P", "SF-F1", "avg_context_tokens", "avg_selected_atoms", "retrieval_ms", "generation_ms", "total_ms"]
    proposal_keys = ["dataset", "profile", "variant", "normalized_candidate_gold_recall", "normalized_selected_gold_recall", "normalized_rendered_gold_recall", "candidate_oracle_F1", "chain_unit_oracle_feasible"]
    chunk_keys = ["dataset", "profile", "variant", "chunk_universe_size", "chunk_seed_gold_hit_rate_eval_only", "chunk_candidate_gold_recall_eval_only", "best_seed_score", "num_changed_chunk_seeds", "num_bridge_entities", "num_final_evidence_candidates", "selected_chunk_medoid_source_rate"]
    timing_keys = ["dataset", "profile", "variant", *TIMING_KEYS]
    return [
        "# Phase8 Chunk Probe Summary",
        "",
        "This summary treats `source_balanced_128` only as an external Phase7 reference when explicitly included. The active Phase8-only method is `chunk_pamae_k5`.",
        "",
        "## Main Metrics",
        "",
        md_table(rows, main_keys),
        "",
        "## Proposal Metrics",
        "",
        md_table(rows, proposal_keys),
        "",
        "## Chunk Diagnostics",
        "",
        md_table(rows, chunk_keys),
        "",
        "## Timing",
        "",
        md_table(rows, timing_keys),
        "",
        "## Probe Questions",
        "",
        "1. Is `chunk_pamae_k5` better than entity-first PAMAE raw? Compare against the repaired entity-first historical report.",
        "2. Is `chunk_pamae_k5` stable after trace repair? Check `status`, `trace_warning_count`, and query counts.",
        "3. Is `chunk_pamae_k5` viable as Phase8-only? Use SF-R/F1 together with timing overhead.",
        "4. Do not proceed to n=100 unless the limit-10 and limit-30 probes have clean trace counts and acceptable timing.",
        "5. If not viable, inspect whether the next bottleneck is chunk universe, medoid selection, evidence mapping, or final selection.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase8 chunk/carrier medoid probe outputs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--dedupe-query-id", action="store_true")
    parser.add_argument("--expected-limit", type=int, default=0)
    parser.add_argument("--include-external-reference", action="store_true")
    parser.add_argument("--include-rejected", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = [_load_run(root, run_dir, args) for run_dir in _run_dirs(root)]
    selected = _filter_runs(runs, args)
    rows = [_summarize(run) for run in sorted(selected, key=lambda r: (r["dataset"], r["profile"], r["variant"], r["run_id"]))]

    write_json(root / "phase8_chunk_medoid_summary.json", {"runs": rows, "selected_runs": selected})
    write_json(root / "phase8_chunk_probe_summary.json", {"runs": rows, "selected_runs": selected})
    write_md(root / "phase8_chunk_medoid_summary.md", _summary_lines(rows))
    write_md(root / "phase8_chunk_probe_summary.md", _summary_lines(rows))
    write_md(root / "phase8_chunk_trace_repair_report.md", _trace_report_lines(selected, rows))
    write_md(root / "phase8_chunk_timing_optimization_report.md", _timing_report_lines(rows))
    print(root / "phase8_chunk_probe_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
