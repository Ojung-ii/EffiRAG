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
    rows_by_query,
    safe_float,
    stats,
    write_json,
    write_md,
)


VARIANTS = ["source_balanced_128", "chunk_pamae_k5", "chunk_bridge_refine_k5"]
TIMING_KEYS = [
    "chunk_universe_ms",
    "chunk_sampling_ms",
    "chunk_kmedoids_ms",
    "chunk_seed_selection_ms",
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
    names = [
        "rag_summary.json",
        "rag_query_results.jsonl",
        "phase7_query_trace.jsonl",
        "phase8_chunk_query_trace.jsonl",
        "phase8_chunk_stage_timing.jsonl",
        "phase8_chunk_seed_trace.jsonl",
        "phase8_chunk_evidence_trace.jsonl",
    ]
    for name in names:
        for path in root.glob(f"*/*/*/{name}"):
            found.add(path.parent)
    return sorted(found)


def _parts(root: Path, run_dir: Path) -> Tuple[str, str, str]:
    parts = list(run_dir.relative_to(root).parts)
    profile = parts[0] if len(parts) > 0 else ""
    variant = parts[1] if len(parts) > 1 else ""
    dataset = parts[2] if len(parts) > 2 else ""
    return dataset, profile, variant


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
    return 0.0


def _load_run(root: Path, run_dir: Path) -> Dict[str, Any]:
    dataset, profile, variant = _parts(root, run_dir)
    result_rows = rows_by_query(read_jsonl(run_dir / "rag_query_results.jsonl"))
    phase7_rows = rows_by_query(read_jsonl(run_dir / "phase7_query_trace.jsonl"))
    phase7_diag_rows = rows_by_query(read_jsonl(run_dir / "phase7_diagnostics.jsonl"))
    chunk_rows = rows_by_query(read_jsonl(run_dir / "phase8_chunk_query_trace.jsonl"))
    seed_rows = rows_by_query(read_jsonl(run_dir / "phase8_chunk_seed_trace.jsonl"))
    evidence_rows = rows_by_query(read_jsonl(run_dir / "phase8_chunk_evidence_trace.jsonl"))
    timing_rows = rows_by_query(read_jsonl(run_dir / "phase8_chunk_stage_timing.jsonl"))
    qids = set()
    for mapping in [result_rows, phase7_rows, phase7_diag_rows, chunk_rows, seed_rows, evidence_rows, timing_rows]:
        qids.update(mapping.keys())
    queries = []
    for qid in sorted(qids):
        queries.append(
            {
                "query_id": qid,
                "result": result_rows.get(qid, {}),
                "phase7": phase7_rows.get(qid, {}),
                "phase7_diag": phase7_diag_rows.get(qid, {}),
                "chunk_query": chunk_rows.get(qid, {}),
                "chunk_seed": seed_rows.get(qid, {}),
                "chunk_evidence": evidence_rows.get(qid, {}),
                "chunk_timing": timing_rows.get(qid, {}),
            }
        )
    files = {
        "rag_summary.json": (run_dir / "rag_summary.json").exists(),
        "rag_query_results.jsonl": (run_dir / "rag_query_results.jsonl").exists(),
        "phase7_query_trace.jsonl": (run_dir / "phase7_query_trace.jsonl").exists(),
        "phase8_chunk_query_trace.jsonl": (run_dir / "phase8_chunk_query_trace.jsonl").exists(),
        "phase8_chunk_stage_timing.jsonl": (run_dir / "phase8_chunk_stage_timing.jsonl").exists(),
        "phase8_chunk_seed_trace.jsonl": (run_dir / "phase8_chunk_seed_trace.jsonl").exists(),
        "phase8_chunk_evidence_trace.jsonl": (run_dir / "phase8_chunk_evidence_trace.jsonl").exists(),
    }
    summary = read_json(run_dir / "rag_summary.json")
    expected = int(summary.get("num_queries", summary.get("count", 100)) or 100)
    actual = len(queries)
    if not any(files.values()) or actual == 0:
        status = "MISSING"
    elif actual < min(expected, 100):
        status = "PARTIAL"
    elif files["rag_summary.json"] and int(summary.get("num_queries", summary.get("count", actual)) or actual) != actual:
        status = "SUMMARY_MISMATCH"
    else:
        status = "COMPLETE"
    return {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "run_dir": str(run_dir),
        "files": files,
        "expected_num_queries": expected,
        "actual_num_queries": actual,
        "status": status,
        "queries": queries,
    }


def _iter_queries(group: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for run in group:
        for query in list(run.get("queries", []) or []):
            yield query


def _group_runs(runs: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for run in runs:
        key = (str(run.get("dataset", "")), str(run.get("profile", "")), str(run.get("variant", "")))
        groups.setdefault(key, []).append(run)
    return groups


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


def _summarize(dataset: str, profile: str, variant: str, group: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries = list(_iter_queries(group))
    nds = [normalized_query_diagnostics(q) for q in queries]
    row: Dict[str, Any] = {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "num_queries": len(queries),
        "expected_num_queries": max([int(r.get("expected_num_queries", 100)) for r in group] or [100]),
        "actual_num_queries": sum(int(r.get("actual_num_queries", 0)) for r in group),
        "status": ",".join(sorted({str(r.get("status", "MISSING")) for r in group})),
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
            safe_float(q.get("phase7", {}).get("num_selected_atoms", 0.0), 0.0)
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

    if variant in {"chunk_pamae_k5", "chunk_bridge_refine_k5"}:
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
    else:
        row.update(
            {
                "chunk_universe_size": "N/A",
                "chunk_seed_gold_hit_rate_eval_only": "N/A",
                "chunk_candidate_gold_recall_eval_only": "N/A",
                "best_seed_score": "N/A",
                "num_changed_chunk_seeds": "N/A",
                "num_bridge_entities": "N/A",
                "num_final_evidence_candidates": "N/A",
                "selected_chunk_medoid_source_rate": "N/A",
            }
        )
    return row


def _variant_lookup(rows: List[Dict[str, Any]], dataset: str, profile: str, variant: str) -> Dict[str, Any]:
    for row in rows:
        if row["dataset"] == dataset and row["profile"] == profile and row["variant"] == variant:
            return row
    return {}


def _report_lines(rows: List[Dict[str, Any]]) -> List[str]:
    comparisons = []
    keys = sorted({(r["dataset"], r["profile"]) for r in rows})
    for dataset, profile in keys:
        base = _variant_lookup(rows, dataset, profile, "source_balanced_128")
        raw = _variant_lookup(rows, dataset, profile, "chunk_pamae_k5")
        ref = _variant_lookup(rows, dataset, profile, "chunk_bridge_refine_k5")
        if not raw:
            continue
        comparisons.append(
            {
                "dataset": dataset,
                "profile": profile,
                "chunk_vs_base_F1_delta": safe_float(raw.get("F1", 0.0)) - safe_float(base.get("F1", 0.0)),
                "chunk_vs_base_SFR_delta": safe_float(raw.get("SF-R", 0.0)) - safe_float(base.get("SF-R", 0.0)),
                "refine_vs_raw_F1_delta": safe_float(ref.get("F1", 0.0)) - safe_float(raw.get("F1", 0.0)) if ref else 0.0,
                "refine_vs_raw_candidate_recall_delta": safe_float(ref.get("normalized_candidate_gold_recall", 0.0)) - safe_float(raw.get("normalized_candidate_gold_recall", 0.0)) if ref else 0.0,
                "chunk_retrieval_ms": safe_float(raw.get("retrieval_ms", 0.0)),
                "base_retrieval_ms": safe_float(base.get("retrieval_ms", 0.0)),
            }
        )
    return [
        "# Phase8 Chunk Medoid Report",
        "",
        "This report is generated from existing `outputs/phase8_chunk_medoid` traces. Missing runs are not treated as zero-quality results.",
        "",
        "## Comparison",
        "",
        md_table(
            comparisons,
            [
                "dataset",
                "profile",
                "chunk_vs_base_F1_delta",
                "chunk_vs_base_SFR_delta",
                "refine_vs_raw_F1_delta",
                "refine_vs_raw_candidate_recall_delta",
                "chunk_retrieval_ms",
                "base_retrieval_ms",
            ],
        ),
        "",
        "## Questions",
        "",
        "1. Did chunk-first medoids improve over entity-first PAMAE? Compare this table with repaired entity-first Phase8 reports.",
        "2. Did chunk-first medoids recover candidate recall / SF-R? See `normalized_candidate_gold_recall` and `SF-R` in the summary.",
        "3. Did bridge-entity refinement improve over raw chunk medoids? See `refine_vs_raw_*` deltas.",
        "4. Did final F1 improve? See `F1` and `chunk_vs_base_F1_delta`.",
        "5. Did SF-P collapse? See `SF-P` in the main summary table.",
        "6. Was runtime acceptable? Compare retrieval_ms against source-balanced baselines.",
        "7. Did CHUNK_TO_EVIDENCE_MISS decrease? See failure analysis after running `analyze_phase8_chunk_medoid_failures.py`.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase8 chunk/carrier medoid experiment outputs.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = [_load_run(root, run_dir) for run_dir in _run_dirs(root)]
    rows = []
    for (dataset, profile, variant), group in sorted(_group_runs(runs).items()):
        rows.append(_summarize(dataset, profile, variant, group))

    write_json(root / "phase8_chunk_medoid_summary.json", {"runs": rows})
    main_keys = ["dataset", "profile", "variant", "status", "num_queries", "EM", "F1", "SF-R", "SF-P", "SF-F1", "avg_context_tokens", "avg_selected_atoms", "retrieval_ms", "generation_ms", "total_ms"]
    proposal_keys = ["dataset", "profile", "variant", "normalized_candidate_gold_recall", "normalized_selected_gold_recall", "normalized_rendered_gold_recall", "candidate_oracle_F1", "chain_unit_oracle_feasible"]
    chunk_keys = ["dataset", "profile", "variant", "chunk_universe_size", "chunk_seed_gold_hit_rate_eval_only", "chunk_candidate_gold_recall_eval_only", "best_seed_score", "num_changed_chunk_seeds", "num_bridge_entities", "num_final_evidence_candidates", "selected_chunk_medoid_source_rate"]
    timing_keys = ["dataset", "profile", "variant", *TIMING_KEYS]
    lines = [
        "# Phase8 Chunk Medoid Summary",
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
    ]
    write_md(root / "phase8_chunk_medoid_summary.md", lines)
    write_md(root / "phase8_chunk_medoid_report.md", _report_lines(rows))
    print(root / "phase8_chunk_medoid_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
