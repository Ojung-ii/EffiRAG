#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import (
    md_table,
    metric_from_result,
    normalized_query_diagnostics,
    read_jsonl,
    rows_by_query,
    safe_float,
    write_json,
    write_md,
)


CHUNK_CATEGORIES = [
    "CHUNK_UNIVERSE_MISS",
    "CHUNK_SEED_MISS",
    "BRIDGE_ENTITY_MISS",
    "BRIDGE_REFINEMENT_DRIFT",
    "CHUNK_TO_EVIDENCE_MISS",
    "CANDIDATE_CHAIN_INFEASIBLE",
    "FINAL_SELECTION_FAILED",
    "SELECTED_NOT_SUFFICIENT",
    "QA_PROMPT_FAILED",
    "OTHER",
]

BASELINE_CATEGORIES = [
    "PHASE1_NOT_FOUND",
    "FINAL_SELECTION_FAILED",
    "SELECTED_NOT_SUFFICIENT",
    "QA_PROMPT_FAILED",
    "OTHER",
]


def _run_dirs(root: Path) -> List[Path]:
    found = set()
    for name in [
        "rag_query_results.jsonl",
        "phase7_query_trace.jsonl",
        "phase8_chunk_query_trace.jsonl",
        "phase8_chunk_seed_trace.jsonl",
        "phase8_chunk_evidence_trace.jsonl",
    ]:
        for path in root.glob(f"*/*/*/{name}"):
            found.add(path.parent)
    return sorted(found)


def _parts(root: Path, run_dir: Path) -> Tuple[str, str, str]:
    parts = list(run_dir.relative_to(root).parts)
    profile = parts[0] if len(parts) > 0 else ""
    variant = parts[1] if len(parts) > 1 else ""
    dataset = parts[2] if len(parts) > 2 else ""
    return dataset, profile, variant


def _load_queries(run_dir: Path) -> List[Dict[str, Any]]:
    result_rows = rows_by_query(read_jsonl(run_dir / "rag_query_results.jsonl"))
    p7_rows = rows_by_query(read_jsonl(run_dir / "phase7_query_trace.jsonl"))
    p7_diag_rows = rows_by_query(read_jsonl(run_dir / "phase7_diagnostics.jsonl"))
    chunk_rows = rows_by_query(read_jsonl(run_dir / "phase8_chunk_query_trace.jsonl"))
    seed_rows = rows_by_query(read_jsonl(run_dir / "phase8_chunk_seed_trace.jsonl"))
    evidence_rows = rows_by_query(read_jsonl(run_dir / "phase8_chunk_evidence_trace.jsonl"))
    qids = set()
    for mapping in [result_rows, p7_rows, p7_diag_rows, chunk_rows, seed_rows, evidence_rows]:
        qids.update(mapping.keys())
    return [
        {
            "query_id": qid,
            "result": result_rows.get(qid, {}),
            "phase7": p7_rows.get(qid, {}),
            "phase7_diag": p7_diag_rows.get(qid, {}),
            "chunk_query": chunk_rows.get(qid, {}),
            "chunk_seed": seed_rows.get(qid, {}),
            "chunk_evidence": evidence_rows.get(qid, {}),
        }
        for qid in sorted(qids)
    ]


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


def _chunk_label(variant: str, query: Mapping[str, Any]) -> str:
    diag = normalized_query_diagnostics(query)
    candidate_recall = safe_float(diag.get("normalized_candidate_gold_recall"), 0.0)
    selected_recall = safe_float(diag.get("normalized_selected_gold_recall"), 0.0)
    rendered_recall = safe_float(diag.get("normalized_rendered_gold_recall"), 0.0)
    f1 = metric_from_result(query, "f1")
    sf_recall = metric_from_result(query, "sf_recall")

    if variant == "source_balanced_128":
        if candidate_recall <= 0.0:
            return "PHASE1_NOT_FOUND"
        if selected_recall + 1e-9 < candidate_recall:
            return "FINAL_SELECTION_FAILED"
        if rendered_recall < 1.0 and sf_recall < 1.0:
            return "SELECTED_NOT_SUFFICIENT"
        if rendered_recall > 0.0 and f1 < 0.5:
            return "QA_PROMPT_FAILED"
        return "OTHER"

    chunk_query = query.get("chunk_query") if isinstance(query.get("chunk_query"), dict) else {}
    seed_sel = _seed_section(query, "chunk_seed_selection")
    refine = _seed_section(query, "chunk_bridge_refinement")
    evidence = _evidence_section(query)
    if safe_float(chunk_query.get("chunk_universe_size", 0.0), 0.0) <= 0.0:
        return "CHUNK_UNIVERSE_MISS"
    if safe_float(seed_sel.get("best_seed_relevance", 0.0), 0.0) <= 0.05 and candidate_recall <= 0.10:
        return "CHUNK_SEED_MISS"
    if safe_float(refine.get("avg_bridge_entities_per_seed", 0.0), 0.0) <= 0.0 and "bridge_refine" in variant:
        return "BRIDGE_ENTITY_MISS"
    if safe_float(refine.get("num_changed_seeds", 0.0), 0.0) > 0.0:
        before = safe_float(refine.get("before_candidate_gold_recall_eval_only", 0.0), 0.0)
        after = safe_float(refine.get("after_candidate_gold_recall_eval_only", 0.0), 0.0)
        if after + 1e-9 < before:
            return "BRIDGE_REFINEMENT_DRIFT"
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
    parser = argparse.ArgumentParser(description="Analyze Phase8 chunk medoid failures.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    summaries = []
    rows = []
    examples = defaultdict(list)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for run_dir in _run_dirs(root):
        dataset, profile, variant = _parts(root, run_dir)
        grouped.setdefault((dataset, profile, variant), []).extend(_load_queries(run_dir))

    for (dataset, profile, variant), queries in sorted(grouped.items()):
        counter: Counter = Counter()
        allowed = BASELINE_CATEGORIES if variant == "source_balanced_128" else CHUNK_CATEGORIES
        for query in queries:
            label = _chunk_label(variant, query)
            counter[label] += 1
            diag = normalized_query_diagnostics(query)
            row = {
                "dataset": dataset,
                "profile": profile,
                "variant": variant,
                "query_id": query.get("query_id", ""),
                "category": label,
                "question": _question(query),
                "normalized_candidate_gold_recall": diag.get("normalized_candidate_gold_recall"),
                "normalized_selected_gold_recall": diag.get("normalized_selected_gold_recall"),
                "F1": metric_from_result(query, "f1"),
                "SF-R": metric_from_result(query, "sf_recall"),
            }
            rows.append(row)
            key = (dataset, profile, variant, label)
            if len(examples[key]) < 5:
                examples[key].append(row)
        total = len(queries)
        for category in allowed:
            count = int(counter.get(category, 0))
            summaries.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "variant": variant,
                    "category": category,
                    "count": count,
                    "rate": float(count / total) if total else 0.0,
                    "num_queries": total,
                }
            )

    write_json(
        root / "phase8_chunk_medoid_failure_analysis.json",
        {"summary": summaries, "rows": rows, "examples": {"/".join(k): v for k, v in examples.items()}},
    )
    lines = [
        "# Phase8 Chunk Medoid Failure Analysis",
        "",
        md_table(summaries, ["dataset", "profile", "variant", "category", "count", "rate", "num_queries"]),
        "",
        "Failure labels are variant-aware: source-balanced baselines are not assigned chunk-medoid stages.",
        "",
    ]
    write_md(root / "phase8_chunk_medoid_failure_analysis.md", lines)
    print(root / "phase8_chunk_medoid_failure_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
