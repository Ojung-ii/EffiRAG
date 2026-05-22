#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List


TIMING_KEYS = [
    "entity_universe_ms",
    "sampling_ms",
    "kmedoids_ms",
    "seed_selection_ms",
    "refinement_ms",
    "evidence_proposal_ms",
    "retrieval_ms",
    "generation_ms",
    "total_ms",
]

QUERY_KEYS = [
    "em",
    "f1",
    "sf_recall",
    "sf_precision",
    "sf_f1",
    "avg_context_tokens",
    "avg_selected_atoms",
    "phase1_partial_gold_hit",
    "phase1_full_gold_coverage",
    "phase1_gold_recall",
    "candidate_gold_full",
    "candidate_gold_recall",
    "candidate_oracle_F1",
    "chain_unit_oracle_feasible",
    "entity_universe_size",
    "num_samples",
    "sample_size",
    "k",
    "best_seed_score",
    "seed_mean_relevance",
    "seed_coverage",
    "seed_diversity",
    "num_refined_seeds",
    "num_changed_seeds",
    "seed_gold_hit_rate_eval_only",
    "seed_evidence_gold_hit_rate_eval_only",
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
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


def _query_id(row: Dict[str, Any]) -> str:
    return str(row.get("query_id") or row.get("qid") or row.get("sample_id") or "")


def _dedupe_by_query(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = _query_id(row)
        if qid and qid not in out:
            out[qid] = row
        elif not qid:
            out[str(len(out))] = row
    return list(out.values())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals]
    return float(sum(xs) / len(xs)) if xs else 0.0


def _percentile(vals: List[float], q: float) -> float:
    xs = sorted(float(v) for v in vals)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _run_dirs(root: Path) -> List[Path]:
    out = []
    for summary in root.glob("*/*/*/rag_summary.json"):
        out.append(summary.parent)
    return sorted(out)


def _parts(root: Path, run_dir: Path) -> Dict[str, str]:
    rel = run_dir.relative_to(root)
    parts = list(rel.parts)
    return {
        "profile": parts[0] if len(parts) > 0 else "",
        "variant": parts[1] if len(parts) > 1 else "",
        "dataset": parts[2] if len(parts) > 2 else "",
    }


def _summarize_run(root: Path, run_dir: Path) -> Dict[str, Any]:
    summary = _read_json(run_dir / "rag_summary.json")
    qrows = _dedupe_by_query(_read_jsonl(run_dir / "phase8_query_trace.jsonl"))
    p7rows = _dedupe_by_query(_read_jsonl(run_dir / "phase7_query_trace.jsonl"))
    p7_by_qid = {_query_id(row): row for row in p7rows if _query_id(row)}
    timings = _dedupe_by_query(_read_jsonl(run_dir / "phase8_stage_timing.jsonl"))
    parts = _parts(root, run_dir)
    row: Dict[str, Any] = {
        **parts,
        "run_dir": str(run_dir),
        "num_queries": int(len(qrows) or summary.get("num_queries", 0) or summary.get("count", 0) or 0),
    }
    for key in QUERY_KEYS:
        vals = [_safe_float(q.get(key), 0.0) for q in qrows if key in q]
        row[key] = _mean(vals)
    if p7rows:
        chain_vals = []
        full_vals = []
        partial_vals = []
        feasible_vals = []
        for p7 in p7rows:
            chain = dict(p7.get("candidate_chain_feasibility", {}) or {})
            recall = chain.get("candidate_gold_recall", p7.get("candidate_gold_recall", 0.0))
            chain_vals.append(_safe_float(recall, 0.0))
            full_vals.append(1.0 if bool(chain.get("candidate_gold_full", p7.get("candidate_gold_full", False))) else 0.0)
            partial_vals.append(1.0 if bool(chain.get("candidate_gold_partial", p7.get("candidate_gold_partial", False))) else 0.0)
            feasible_vals.append(1.0 if bool(chain.get("chain_unit_oracle_feasible", p7.get("chain_unit_oracle_feasible", False))) else 0.0)
        row["phase1_partial_gold_hit"] = _mean(partial_vals)
        row["phase1_full_gold_coverage"] = _mean(full_vals)
        row["phase1_gold_recall"] = _mean(chain_vals)
        row["candidate_gold_full"] = _mean(full_vals)
        row["candidate_gold_recall"] = _mean(chain_vals)
        row["chain_unit_oracle_feasible"] = _mean(feasible_vals)
    for key in TIMING_KEYS:
        raw_key = "total_retrieval_ms" if key == "retrieval_ms" else key
        vals = [_safe_float(t.get(raw_key), 0.0) for t in timings if raw_key in t]
        row[key] = _mean(vals)
        row[f"{key}_p50"] = float(median(vals)) if vals else 0.0
        row[f"{key}_p95"] = _percentile(vals, 0.95)
        row[f"{key}_max"] = max(vals) if vals else 0.0
    if not qrows:
        row["em"] = _safe_float(summary.get("exact_match", summary.get("em", 0.0)), 0.0)
        row["f1"] = _safe_float(summary.get("f1", 0.0), 0.0)
        row["sf_recall"] = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
        row["sf_precision"] = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
    return row


def _md_table(rows: List[Dict[str, Any]], keys: List[str]) -> str:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase8 PAMAE-inspired entity seeding runs.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = [_summarize_run(root, run_dir) for run_dir in _run_dirs(root)]
    rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("profile", "")), str(r.get("variant", ""))))

    out_json = root / "phase8_pamae_entity_seeding_summary.json"
    out_json.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    keys = [
        "dataset",
        "profile",
        "variant",
        "num_queries",
        "em",
        "f1",
        "sf_recall",
        "sf_precision",
        "phase1_full_gold_coverage",
        "candidate_gold_recall",
        "candidate_oracle_F1",
        "chain_unit_oracle_feasible",
        "retrieval_ms",
        "total_ms",
    ]
    md = ["# Phase8 PAMAE Entity Seeding Summary", "", _md_table(rows, keys), ""]
    timing_keys = ["dataset", "profile", "variant"] + TIMING_KEYS
    md.extend(["## Timing", "", _md_table(rows, timing_keys), ""])
    out_md = root / "phase8_pamae_entity_seeding_summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    report = [
        "# Phase8 PAMAE Entity Seeding Report",
        "",
        "## Questions",
        "",
        "1. Did PAMAE-inspired seeding improve Phase1 full-chain coverage?",
        "2. Did it improve candidate oracle feasibility?",
        "3. Did one-step refinement improve over raw seeding?",
        "4. Did final F1 improve?",
        "5. Did SF-P collapse?",
        "6. Did retrieval latency remain acceptable?",
        "7. Were medoid seeds query-relevant?",
        "8. Did medoid-linked evidence recover gold/equivalent evidence?",
        "9. Should PAMAE-inspired proposal replace Phase7 source-balanced proposal?",
        "10. If not, is the bottleneck U_q construction, medoid seeding, refinement, evidence mapping, or final selection?",
        "",
        "## Run Table",
        "",
        _md_table(rows, keys),
        "",
        "## Notes",
        "",
        "This report is generated from completed run artifacts. Interpret missing variants as not yet run.",
        "",
    ]
    out_report = root / "phase8_pamae_entity_seeding_report.md"
    out_report.write_text("\n".join(report), encoding="utf-8")
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
