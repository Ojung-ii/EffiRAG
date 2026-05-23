#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import (
    entity_universe,
    group_runs,
    is_phase8_variant,
    iter_queries,
    load_runs,
    md_table,
    mean,
    metric_from_result,
    normalized_query_diagnostics,
    phase8_enabled,
    safe_float,
    seed_best,
    seed_refinement,
    stats,
    write_json,
    write_md,
)


TIMING_KEYS = [
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


def _timing(query: Dict[str, Any], key: str) -> float:
    row = query.get("timing")
    if not isinstance(row, dict):
        return 0.0
    return safe_float(row.get(key, 0.0), 0.0)


def _run_status(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = sorted({str(r.get("status", "MISSING")) for r in group})
    expected = max([int(r.get("expected_num_queries", 100)) for r in group] or [100])
    actual = sum(int(r.get("actual_num_queries", 0)) for r in group)
    files = {}
    for run in group:
        files[str(run.get("run_dir", ""))] = {
            "status": run.get("status", "MISSING"),
            "expected_num_queries": run.get("expected_num_queries", expected),
            "actual_num_queries": run.get("actual_num_queries", 0),
            "rag_summary_exists": bool(run.get("files", {}).get("rag_summary.json", False)),
            "query_trace_exists": bool(run.get("files", {}).get("phase8_query_trace.jsonl", False)),
            "seed_trace_exists": bool(run.get("files", {}).get("phase8_seed_trace.jsonl", False)),
            "evidence_trace_exists": bool(run.get("files", {}).get("phase8_evidence_trace.jsonl", False)),
            "stage_timing_exists": bool(run.get("files", {}).get("phase8_stage_timing.jsonl", False)),
        }
    if "COMPLETE" in statuses and len(statuses) == 1:
        status = "COMPLETE"
    elif actual == 0:
        status = "MISSING"
    elif actual < expected:
        status = "PARTIAL"
    elif "SUMMARY_MISMATCH" in statuses:
        status = "SUMMARY_MISMATCH"
    else:
        status = statuses[0] if statuses else "MISSING"
    return {"expected_num_queries": expected, "actual_num_queries": actual, "status": status, "trace_status": files}


def _summarize(dataset: str, profile: str, variant: str, group: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries = list(iter_queries(group))
    status = _run_status(group)
    nds = [normalized_query_diagnostics(q) for q in queries]
    row: Dict[str, Any] = {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "num_queries": len(queries),
        **status,
        "EM": mean(metric_from_result(q, "em") for q in queries),
        "F1": mean(metric_from_result(q, "f1") for q in queries),
        "SF-R": mean(metric_from_result(q, "sf_recall") for q in queries),
        "SF-P": mean(metric_from_result(q, "sf_precision") for q in queries),
        "SF-F1": mean(metric_from_result(q, "sf_f1") for q in queries),
        "avg_context_tokens": mean(
            safe_float(q.get("result", {}).get("generation_diagnostics", {}).get("prompt_tokens", 0.0), 0.0)
            for q in queries
        ),
        "normalized_candidate_gold_recall": mean(d["normalized_candidate_gold_recall"] for d in nds if d["normalized_candidate_gold_recall"] is not None),
        "normalized_selected_gold_recall": mean(d["normalized_selected_gold_recall"] for d in nds if d["normalized_selected_gold_recall"] is not None),
        "normalized_rendered_gold_recall": mean(d["normalized_rendered_gold_recall"] for d in nds if d["normalized_rendered_gold_recall"] is not None),
        "candidate_oracle_F1": mean(d["candidate_oracle_F1"] for d in nds if d["candidate_oracle_F1"] is not None),
        "selected_context_F1": mean(d["selected_context_F1"] for d in nds if d["selected_context_F1"] is not None),
        "chain_unit_oracle_feasible": mean(1.0 if d["chain_unit_oracle_feasible"] else 0.0 for d in nds if d["chain_unit_oracle_feasible"] is not None),
    }
    for key in TIMING_KEYS:
        vals = [_timing(q, key) for q in queries]
        s = stats(vals)
        out_key = "retrieval_ms" if key == "total_retrieval_ms" else key
        row[out_key] = s["mean"]
        row[f"{out_key}_p50"] = s["p50"]
        row[f"{out_key}_p95"] = s["p95"]
        row[f"{out_key}_max"] = s["max"]

    if is_phase8_variant(variant):
        universes = [entity_universe(q) for q in queries]
        bests = [seed_best(q) for q in queries]
        refs = [seed_refinement(q) for q in queries]
        row.update(
            {
                "entity_universe_size": mean(u.get("num_entities", 0.0) for u in universes),
                "gold_entity_hit_rate_eval_only": mean(1.0 if u.get("gold_entity_hit_eval_only") else 0.0 for u in universes),
                "seed_gold_hit_rate_eval_only": mean(1.0 if b.get("seed_gold_hit_eval_only") else 0.0 for b in bests),
                "seed_evidence_gold_hit_rate_eval_only": mean(1.0 if b.get("seed_evidence_gold_hit_eval_only") else 0.0 for b in bests),
                "num_changed_seeds": mean(r.get("num_changed_seeds", 0.0) for r in refs),
                "before_refine_seed_evidence_hit": mean(1.0 if r.get("before_seed_evidence_gold_hit_eval_only") else 0.0 for r in refs),
                "after_refine_seed_evidence_hit": mean(1.0 if r.get("after_seed_evidence_gold_hit_eval_only") else 0.0 for r in refs),
            }
        )
    else:
        row.update(
            {
                "entity_universe_size": "N/A",
                "gold_entity_hit_rate_eval_only": "N/A",
                "seed_gold_hit_rate_eval_only": "N/A",
                "seed_evidence_gold_hit_rate_eval_only": "N/A",
                "num_changed_seeds": "N/A",
                "before_refine_seed_evidence_hit": "N/A",
                "after_refine_seed_evidence_hit": "N/A",
            }
        )
    row["phase8_enabled_rate"] = mean(1.0 if phase8_enabled(q.get("phase7", {})) else 0.0 for q in queries)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Repaired Phase8 PAMAE summary with normalized diagnostics.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for (dataset, profile, variant), group in sorted(group_runs(load_runs(root)).items()):
        rows.append(_summarize(dataset, profile, variant, group))

    write_json(root / "phase8_pamae_entity_seeding_summary_repaired.json", {"runs": rows})
    main_keys = [
        "dataset",
        "profile",
        "variant",
        "status",
        "num_queries",
        "EM",
        "F1",
        "SF-R",
        "SF-P",
        "SF-F1",
        "avg_context_tokens",
        "retrieval_ms",
        "total_ms",
    ]
    proposal_keys = [
        "dataset",
        "profile",
        "variant",
        "status",
        "num_queries",
        "normalized_candidate_gold_recall",
        "normalized_selected_gold_recall",
        "normalized_rendered_gold_recall",
        "candidate_oracle_F1",
        "selected_context_F1",
        "chain_unit_oracle_feasible",
    ]
    seed_keys = [
        "dataset",
        "profile",
        "variant",
        "entity_universe_size",
        "gold_entity_hit_rate_eval_only",
        "seed_gold_hit_rate_eval_only",
        "seed_evidence_gold_hit_rate_eval_only",
        "num_changed_seeds",
        "before_refine_seed_evidence_hit",
        "after_refine_seed_evidence_hit",
    ]
    timing_keys = [
        "dataset",
        "profile",
        "variant",
        "entity_universe_ms",
        "sampling_ms",
        "kmedoids_ms",
        "seed_selection_ms",
        "refinement_ms",
        "evidence_proposal_ms",
        "retrieval_ms",
        "total_ms",
    ]
    status_keys = [
        "dataset",
        "profile",
        "variant",
        "expected_num_queries",
        "actual_num_queries",
        "status",
    ]
    lines = [
        "# Phase8 PAMAE Entity Seeding Summary Repaired",
        "",
        "## Main Table",
        "",
        md_table(rows, main_keys),
        "",
        "## Repaired Proposal Table",
        "",
        md_table(rows, proposal_keys),
        "",
        "## Phase8 Seed Table",
        "",
        md_table(rows, seed_keys),
        "",
        "## Timing Table",
        "",
        md_table(rows, timing_keys),
        "",
        "## Trace Completeness",
        "",
        md_table(rows, status_keys),
        "",
        "Missing or partial traces are status-coded and are not silently interpreted as zeros.",
        "",
    ]
    write_md(root / "phase8_pamae_entity_seeding_summary_repaired.md", lines)
    print(root / "phase8_pamae_entity_seeding_summary_repaired.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
