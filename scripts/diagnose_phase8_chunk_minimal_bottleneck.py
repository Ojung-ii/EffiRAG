#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import md_table, mean, read_json, read_jsonl, safe_float, stats, write_json, write_md


TIMING_KEYS = [
    "query_embed_ms",
    "chunk_universe_ms",
    "chunk_sampling_ms",
    "chunk_kmedoids_ms",
    "chunk_seed_selection_ms",
    "chunk_medoid_proposal_ms",
    "local_graph_build_ms",
    "evidence_proposal_ms",
    "feature_construction_ms",
    "selection_ms",
    "rendering_ms",
    "generation_ms",
    "retrieval_ms",
    "total_ms",
]

FAILURE_CATEGORIES = [
    "CHUNK_UNIVERSE_MISS",
    "CHUNK_SEED_MISS",
    "SEED_CARRIER_ATOM_MAPPING_MISS",
    "FINAL_CANDIDATE_EVIDENCE_MISS",
    "FINAL_SELECTION_FAILED",
    "SELECTED_NOT_SUFFICIENT",
    "QA_PROMPT_FAILED",
    "OTHER",
]


def _query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("query_id", row.get("sample_id", row.get("qid", ""))) or "")


def _dedupe(rows: Iterable[Mapping[str, Any]], *, run_id: str = "") -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    raw = [dict(r) for r in list(rows or []) if isinstance(r, Mapping)]
    if run_id and any(str(r.get("run_id", "")) for r in raw):
        raw = [r for r in raw if str(r.get("run_id", "")) == str(run_id)]
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        qid = _query_id(row)
        if qid:
            out[qid] = dict(row)
    return out, {
        "raw_rows": int(len(raw)),
        "unique_queries": int(len(out)),
        "duplicate_rows": int(max(0, len(raw) - len(out))),
    }


def _run_timestamp(run_dir: Path) -> str:
    summary = read_json(run_dir / "rag_summary.json")
    ts = str(summary.get("run_timestamp", "") or "")
    if ts:
        return ts
    rows = read_jsonl(run_dir / "phase8_chunk_query_trace.jsonl")
    timestamps = sorted({str(r.get("timestamp", r.get("run_timestamp", "")) or "") for r in rows if isinstance(r, Mapping)})
    return timestamps[-1] if timestamps else run_dir.name


def _candidate_run_dirs(root: Path, dataset: str, profile: str, variant: str) -> List[Path]:
    base = root / dataset / profile / variant
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _load_run(run_dir: Path) -> Dict[str, Any]:
    run_id = run_dir.name
    summary = read_json(run_dir / "rag_summary.json")
    maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    infos: Dict[str, Dict[str, int]] = {}
    for key, name in {
        "result": "rag_query_results.jsonl",
        "phase7": "phase7_query_trace.jsonl",
        "phase7_diag": "phase7_diagnostics.jsonl",
        "chunk_query": "phase8_chunk_query_trace.jsonl",
        "chunk_seed": "phase8_chunk_seed_trace.jsonl",
        "chunk_evidence": "phase8_chunk_evidence_trace.jsonl",
        "chunk_timing": "phase8_chunk_stage_timing.jsonl",
    }.items():
        maps[key], infos[name] = _dedupe(read_jsonl(run_dir / name), run_id=run_id)
    qids = sorted(set().union(*(set(m.keys()) for m in maps.values())))
    queries = []
    for qid in qids:
        queries.append(
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
        )
    expected = int(summary.get("num_queries", summary.get("count", 0)) or 0)
    return {
        "run_id": run_id,
        "timestamp": _run_timestamp(run_dir),
        "run_dir": str(run_dir),
        "expected_num_queries": expected or len(queries),
        "actual_num_queries": len(queries),
        "trace_infos": infos,
        "queries": queries,
    }


def _select_latest_complete(runs: List[Dict[str, Any]], expected_limit: int) -> Dict[str, Any]:
    complete = [
        r
        for r in runs
        if int(r.get("actual_num_queries", 0)) == int(expected_limit)
        and all(int(info.get("duplicate_rows", 0)) == 0 for info in dict(r.get("trace_infos", {}) or {}).values())
    ]
    pool = complete or runs
    if not pool:
        raise SystemExit("No matching Phase8 chunk-medoid run directories found.")
    return sorted(pool, key=lambda r: (str(r.get("timestamp", "")), str(r.get("run_id", ""))))[-1]


def _phase8_diag(query: Mapping[str, Any], section: str) -> Dict[str, Any]:
    p7 = query.get("phase7")
    if isinstance(p7, Mapping):
        diag = p7.get("phase8_chunk_diagnostics")
        if isinstance(diag, Mapping) and isinstance(diag.get(section), Mapping):
            return dict(diag.get(section) or {})
    return {}


def _seed_diag(query: Mapping[str, Any]) -> Dict[str, Any]:
    row = query.get("chunk_seed")
    if isinstance(row, Mapping) and isinstance(row.get("chunk_seed_selection"), Mapping):
        return dict(row.get("chunk_seed_selection") or {})
    return _phase8_diag(query, "chunk_seed_selection")


def _evidence_diag(query: Mapping[str, Any]) -> Dict[str, Any]:
    row = query.get("chunk_evidence")
    if isinstance(row, Mapping) and isinstance(row.get("chunk_medoid_evidence_proposal"), Mapping):
        return dict(row.get("chunk_medoid_evidence_proposal") or {})
    return _phase8_diag(query, "chunk_medoid_evidence_proposal")


def _metric(query: Mapping[str, Any], key: str) -> float:
    result = query.get("result")
    metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
    aliases = {
        "f1": "f1",
        "sf_recall": "supporting_fact_recall",
    }
    return safe_float(metrics.get(aliases.get(key, key), 0.0) if isinstance(metrics, Mapping) else 0.0, 0.0)


def _timing(query: Mapping[str, Any], key: str) -> float:
    row = query.get("chunk_timing")
    return safe_float(row.get(key, 0.0), 0.0) if isinstance(row, Mapping) else 0.0


def _question(query: Mapping[str, Any]) -> str:
    for key in ("chunk_query", "phase7", "phase7_diag", "result"):
        row = query.get(key)
        if isinstance(row, Mapping) and row.get("question"):
            return str(row.get("question"))
    return ""


def _label(row: Mapping[str, Any]) -> str:
    if safe_float(row.get("chunk_universe_gold_recall_eval_only"), 0.0) <= 0.0:
        return "CHUNK_UNIVERSE_MISS"
    if safe_float(row.get("seed_chunk_gold_recall_eval_only"), 0.0) <= 0.0:
        return "CHUNK_SEED_MISS"
    if safe_float(row.get("seed_carrier_atoms_gold_recall_eval_only"), 0.0) <= 0.0:
        return "SEED_CARRIER_ATOM_MAPPING_MISS"
    if safe_float(row.get("final_candidate_gold_recall_eval_only"), 0.0) <= 0.0:
        return "FINAL_CANDIDATE_EVIDENCE_MISS"
    if safe_float(row.get("selected_gold_recall_eval_only"), 0.0) <= 0.0:
        return "FINAL_SELECTION_FAILED"
    if safe_float(row.get("selected_gold_recall_eval_only"), 0.0) < 1.0:
        return "SELECTED_NOT_SUFFICIENT"
    if safe_float(row.get("f1"), 0.0) < 0.5:
        return "QA_PROMPT_FAILED"
    return "OTHER"


def _query_row(query: Mapping[str, Any]) -> Dict[str, Any]:
    cq = query.get("chunk_query") if isinstance(query.get("chunk_query"), Mapping) else {}
    seed = _seed_diag(query)
    ev = _evidence_diag(query)
    row = {
        "query_id": str(query.get("query_id", "")),
        "question": _question(query),
        "chunk_universe_size": safe_float(cq.get("chunk_universe_size", 0.0), 0.0),
        "chunk_universe_gold_recall_eval_only": safe_float(cq.get("chunk_universe_gold_recall_eval_only", 0.0), 0.0),
        "chunk_universe_gold_partial_eval_only": bool(cq.get("chunk_universe_gold_partial_eval_only", False)),
        "chunk_universe_gold_full_eval_only": bool(cq.get("chunk_universe_gold_full_eval_only", False)),
        "top_query_sim_gold_carrier_rank_eval_only": cq.get("top_query_sim_gold_carrier_rank_eval_only"),
        "seed_chunk_ids": list(seed.get("seed_chunk_ids", seed.get("best_seed_chunk_ids", [])) or []),
        "seed_chunk_titles": list(seed.get("seed_chunk_titles", seed.get("best_seed_titles", [])) or []),
        "seed_chunk_gold_recall_eval_only": safe_float(seed.get("seed_chunk_gold_recall_eval_only", 0.0), 0.0),
        "seed_chunk_gold_partial_eval_only": bool(seed.get("seed_chunk_gold_partial_eval_only", seed.get("seed_gold_hit_eval_only", False))),
        "seed_chunk_gold_full_eval_only": bool(seed.get("seed_chunk_gold_full_eval_only", False)),
        "best_seed_score": safe_float(seed.get("best_seed_score", cq.get("best_seed_score", 0.0)), 0.0),
        "seed_carrier_atoms_added_count": int(safe_float(ev.get("seed_carrier_atoms_added_count", ev.get("num_seed_sentence_atoms", 0)), 0.0)),
        "seed_carriers_without_sentence_atoms": int(safe_float(ev.get("seed_carriers_without_sentence_atoms", 0), 0.0)),
        "avg_sentence_atoms_per_seed_carrier": safe_float(ev.get("avg_sentence_atoms_per_seed_carrier", 0.0), 0.0),
        "seed_carrier_atoms_gold_recall_eval_only": safe_float(ev.get("seed_carrier_atoms_gold_recall_eval_only", 0.0), 0.0),
        "num_final_evidence_candidates": int(safe_float(ev.get("num_final_evidence_candidates", cq.get("num_final_evidence_candidates", 0)), 0.0)),
        "final_candidate_gold_recall_eval_only": safe_float(
            ev.get("final_candidate_gold_recall_eval_only", ev.get("candidate_gold_recall_eval_only", cq.get("normalized_candidate_gold_recall", 0.0))),
            0.0,
        ),
        "selected_chunk_medoid_source_rate": safe_float(cq.get("selected_chunk_medoid_source_rate", query.get("chunk_evidence", {}).get("selected_chunk_medoid_source_rate", 0.0) if isinstance(query.get("chunk_evidence"), Mapping) else 0.0), 0.0),
        "selected_gold_recall_eval_only": safe_float(cq.get("normalized_selected_gold_recall", 0.0), 0.0),
        "rendered_gold_recall_eval_only": safe_float(cq.get("normalized_rendered_gold_recall", 0.0), 0.0),
        "num_selected_atoms": int(safe_float(cq.get("avg_selected_atoms", 0), 0.0)),
        "f1": _metric(query, "f1"),
        "sf_recall": _metric(query, "sf_recall"),
    }
    for key in TIMING_KEYS:
        row[key] = _timing(query, key)
    row["failure_category"] = _label(row)
    return row


def _timing_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    retrieval = mean(row.get("retrieval_ms", 0.0) for row in rows)
    for key in TIMING_KEYS:
        s = stats(row.get(key, 0.0) for row in rows)
        if key != "retrieval_ms":
            s["share_of_retrieval"] = float(s["mean"] / float(retrieval or 1.0))
        out[key] = s
    return out


def _answer_lines(rows: List[Dict[str, Any]], timing: Dict[str, Dict[str, float]]) -> List[str]:
    c_universe_recall = mean(row["chunk_universe_gold_recall_eval_only"] for row in rows)
    seed_recall = mean(row["seed_chunk_gold_recall_eval_only"] for row in rows)
    seed_atom_recall = mean(row["seed_carrier_atoms_gold_recall_eval_only"] for row in rows)
    final_recall = mean(row["final_candidate_gold_recall_eval_only"] for row in rows)
    selected_recall = mean(row["selected_gold_recall_eval_only"] for row in rows)
    phase8_timing_keys = [
        "chunk_universe_ms",
        "chunk_sampling_ms",
        "chunk_kmedoids_ms",
        "chunk_seed_selection_ms",
        "chunk_medoid_proposal_ms",
        "local_graph_build_ms",
        "evidence_proposal_ms",
        "feature_construction_ms",
        "selection_ms",
    ]
    largest = max(phase8_timing_keys, key=lambda key: timing.get(key, {}).get("mean", 0.0)) if rows else ""
    if c_universe_recall <= 0.0:
        target = "chunk universe construction"
    elif seed_recall <= 0.0:
        target = "medoid seed selection"
    elif seed_atom_recall <= 0.0:
        target = "seed carrier atom mapping"
    elif final_recall <= 0.0:
        target = "final candidate construction"
    elif selected_recall <= 0.0:
        target = "selection"
    else:
        target = "QA / sufficiency after selection"
    return [
        "## Answers",
        "",
        f"1. Did C_q contain gold/equivalent evidence? {'yes' if c_universe_recall > 0 else 'no'} (mean recall={c_universe_recall:.4f}).",
        f"2. If yes, did k-medoids select it? {'yes' if seed_recall > 0 else 'no'} (mean seed recall={seed_recall:.4f}).",
        f"3. If seed selected it, were sentence atoms added? {'yes' if seed_atom_recall > 0 else 'no'} (mean seed-carrier atom recall={seed_atom_recall:.4f}).",
        f"4. If candidate contained it, did selection keep it? {'yes' if selected_recall > 0 else 'no'} (final candidate recall={final_recall:.4f}, selected recall={selected_recall:.4f}).",
        f"5. Largest timing bottleneck: `{largest}` (mean={timing.get(largest, {}).get('mean', 0.0):.2f} ms).",
        f"6. Is chunk_universe_ms still too high? {'yes' if timing.get('chunk_universe_ms', {}).get('mean', 0.0) > 5000.0 else 'no'} (mean={timing.get('chunk_universe_ms', {}).get('mean', 0.0):.2f} ms).",
        f"7. Next fix target: {target}.",
        "",
    ]


def _md(run: Mapping[str, Any], rows: List[Dict[str, Any]], timing: Dict[str, Dict[str, float]]) -> List[str]:
    counts = Counter(row["failure_category"] for row in rows)
    failure_rows = [
        {"category": cat, "count": int(counts.get(cat, 0)), "rate": float(counts.get(cat, 0) / float(max(1, len(rows))))}
        for cat in FAILURE_CATEGORIES
    ]
    timing_rows = []
    for key in TIMING_KEYS:
        s = timing.get(key, {})
        timing_rows.append(
            {
                "stage": key,
                "mean": s.get("mean", 0.0),
                "p50": s.get("p50", 0.0),
                "p95": s.get("p95", 0.0),
                "max": s.get("max", 0.0),
                "share_of_retrieval": s.get("share_of_retrieval", ""),
            }
        )
    query_keys = [
        "query_id",
        "chunk_universe_gold_recall_eval_only",
        "seed_chunk_gold_recall_eval_only",
        "seed_carrier_atoms_gold_recall_eval_only",
        "final_candidate_gold_recall_eval_only",
        "selected_gold_recall_eval_only",
        "failure_category",
        "chunk_universe_ms",
        "chunk_medoid_proposal_ms",
        "retrieval_ms",
    ]
    return [
        "# Phase8 Chunk-Medoid Minimal Probe5 Bottleneck Report",
        "",
        f"Run: `{run.get('run_id', '')}`",
        f"Dataset/profile/variant: `{run.get('dataset', '')}` / `{run.get('profile', '')}` / `{run.get('variant', '')}`",
        f"Queries: {run.get('actual_num_queries', 0)} / expected {run.get('expected_num_queries', 0)}",
        "",
        "## Query Funnel",
        "",
        md_table(rows, query_keys),
        "",
        "## Failure Categories",
        "",
        md_table(failure_rows, ["category", "count", "rate"]),
        "",
        "## Timing",
        "",
        md_table(timing_rows, ["stage", "mean", "p50", "p95", "max", "share_of_retrieval"]),
        "",
        *_answer_lines(rows, timing),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the latest 5-query Phase8 chunk-medoid bottleneck probe.")
    parser.add_argument("--root", default="outputs/phase8_chunk_medoid")
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--profile", default="balanced_384_8")
    parser.add_argument("--variant", default="chunk_pamae_k5")
    parser.add_argument("--expected-limit", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_dirs = _candidate_run_dirs(root, args.dataset, args.profile, args.variant)
    runs = [_load_run(d) for d in run_dirs]
    for run in runs:
        run["dataset"] = args.dataset
        run["profile"] = args.profile
        run["variant"] = args.variant
    selected = _select_latest_complete(runs, int(args.expected_limit))
    rows = [_query_row(q) for q in list(selected.get("queries", []) or [])]
    timing = _timing_summary(rows)
    payload = {
        "run": {k: v for k, v in selected.items() if k != "queries"},
        "queries": rows,
        "failure_counts": dict(Counter(row["failure_category"] for row in rows)),
        "timing": timing,
    }
    write_json(root / "minimal_probe5_bottleneck.json", payload)
    write_md(root / "minimal_probe5_bottleneck_report.md", _md(selected, rows, timing))
    print(root / "minimal_probe5_bottleneck_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
