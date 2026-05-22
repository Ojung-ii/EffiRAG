#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import (
    group_runs,
    iter_queries,
    load_runs,
    md_table,
    mean,
    missing_files_for,
    question_for,
    safe_float,
    sampling_diag,
    seed_best,
    seed_refinement,
    write_json,
    write_md,
)


def _case(query: Dict[str, Any], best: Dict[str, Any], ref: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "query_id": query.get("query_id", ""),
        "question": question_for(query),
        "reason": reason,
        "best_seed_names": best.get("best_seed_names", []),
        "best_seed_entity_ids": best.get("best_seed_entity_ids", []),
        "best_seed_score": safe_float(best.get("best_seed_score", 0.0), 0.0),
        "best_seed_relevance": safe_float(best.get("best_mean_relevance", 0.0), 0.0),
        "best_hub_penalty": safe_float(best.get("best_hub_penalty", 0.0), 0.0),
        "before_seed_gold_hit_eval_only": bool(ref.get("before_seed_gold_hit_eval_only", False)),
        "after_seed_gold_hit_eval_only": bool(ref.get("after_seed_gold_hit_eval_only", False)),
        "before_seed_evidence_gold_hit_eval_only": bool(ref.get("before_seed_evidence_gold_hit_eval_only", False)),
        "after_seed_evidence_gold_hit_eval_only": bool(ref.get("after_seed_evidence_gold_hit_eval_only", False)),
        "num_changed_seeds": safe_float(ref.get("num_changed_seeds", 0.0), 0.0),
    }


def _summarize_group(dataset: str, profile: str, variant: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries = list(iter_queries(runs))
    bests = [seed_best(q) for q in queries]
    refs = [seed_refinement(q) for q in queries]
    samplings = [sampling_diag(q) for q in queries]
    row: Dict[str, Any] = {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "num_queries": len(queries),
        "missing_files": {r["run_dir"]: missing_files_for(r) for r in runs},
        "k": mean(s.get("k", 0.0) for s in samplings),
        "sample_size": mean(s.get("sample_size", 0.0) for s in samplings),
        "num_samples": mean(s.get("num_samples", 0.0) for s in samplings),
        "best_seed_score_mean": mean(b.get("best_seed_score", 0.0) for b in bests),
        "best_seed_relevance_mean": mean(b.get("best_mean_relevance", 0.0) for b in bests),
        "best_seed_coverage_mean": mean(b.get("best_coverage", 0.0) for b in bests),
        "best_seed_diversity_mean": mean(b.get("best_diversity", 0.0) for b in bests),
        "best_seed_hub_penalty_mean": mean(b.get("best_hub_penalty", 0.0) for b in bests),
        "seed_gold_hit_rate_eval_only": mean(1.0 if b.get("seed_gold_hit_eval_only") else 0.0 for b in bests),
        "seed_evidence_gold_hit_rate_eval_only": mean(1.0 if b.get("seed_evidence_gold_hit_eval_only") else 0.0 for b in bests),
        "num_refined_seeds_mean": mean(r.get("num_refined_seeds", 0.0) for r in refs),
        "num_changed_seeds_mean": mean(r.get("num_changed_seeds", 0.0) for r in refs),
        "changed_seed_rate": mean(1.0 if safe_float(r.get("num_changed_seeds", 0.0)) > 0.0 else 0.0 for r in refs),
        "before_seed_gold_hit_rate_eval_only": mean(1.0 if r.get("before_seed_gold_hit_eval_only") else 0.0 for r in refs),
        "after_seed_gold_hit_rate_eval_only": mean(1.0 if r.get("after_seed_gold_hit_eval_only") else 0.0 for r in refs),
        "before_seed_evidence_gold_hit_rate_eval_only": mean(1.0 if r.get("before_seed_evidence_gold_hit_eval_only") else 0.0 for r in refs),
        "after_seed_evidence_gold_hit_rate_eval_only": mean(1.0 if r.get("after_seed_evidence_gold_hit_eval_only") else 0.0 for r in refs),
    }
    row["interpretation"] = _interpret(row)
    return row


def _interpret(row: Dict[str, Any]) -> str:
    if row.get("variant") == "source_balanced_128":
        return "Phase8 disabled baseline; seed quality is not applicable."
    if safe_float(row.get("seed_gold_hit_rate_eval_only", 0.0)) <= 0.01:
        return "Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing."
    if safe_float(row.get("seed_evidence_gold_hit_rate_eval_only", 0.0)) <= 0.01:
        return "Seeds exist but do not recover gold/equivalent evidence; seed-to-evidence mapping is suspicious."
    if safe_float(row.get("after_seed_evidence_gold_hit_rate_eval_only", 0.0)) < safe_float(row.get("before_seed_evidence_gold_hit_rate_eval_only", 0.0)):
        return "Refinement lowers evidence hit rate; refinement drift is present."
    return "Seed diagnostics do not show a primary failure."


def _representative_cases(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    all_queries = [q for run in runs for q in run.get("queries", [])]
    enriched = []
    for q in all_queries:
        best = seed_best(q)
        ref = seed_refinement(q)
        if not best and not ref:
            continue
        enriched.append((q, best, ref))
    low_rel = sorted(enriched, key=lambda t: safe_float(t[1].get("best_mean_relevance", 0.0), 0.0))[:10]
    high_hub = sorted(enriched, key=lambda t: safe_float(t[1].get("best_hub_penalty", 0.0), 0.0), reverse=True)[:10]
    drift = []
    improved = []
    for q, best, ref in enriched:
        changed = safe_float(ref.get("num_changed_seeds", 0.0), 0.0) > 0.0
        before = float(bool(ref.get("before_seed_gold_hit_eval_only"))) + float(bool(ref.get("before_seed_evidence_gold_hit_eval_only")))
        after = float(bool(ref.get("after_seed_gold_hit_eval_only"))) + float(bool(ref.get("after_seed_evidence_gold_hit_eval_only")))
        if changed and after < before:
            drift.append((q, best, ref))
        if changed and after > before:
            improved.append((q, best, ref))
    return {
        "low_seed_relevance": [_case(q, b, r, "low_seed_relevance") for q, b, r in low_rel],
        "high_hub_penalty": [_case(q, b, r, "high_hub_penalty") for q, b, r in high_hub],
        "refinement_decreased_hit": [_case(q, b, r, "refinement_decreased_hit") for q, b, r in drift[:10]],
        "refinement_improved_hit": [_case(q, b, r, "refinement_improved_hit") for q, b, r in improved[:10]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Phase8 medoid seed quality.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = load_runs(root)
    rows = []
    for (dataset, profile, variant), group in sorted(group_runs(runs).items()):
        rows.append(_summarize_group(dataset, profile, variant, group))

    cases = _representative_cases(runs)
    write_json(root / "phase8_seed_quality_debug.json", {"runs": rows, "representative_cases": cases})

    keys = [
        "dataset",
        "profile",
        "variant",
        "num_queries",
        "k",
        "sample_size",
        "num_samples",
        "best_seed_score_mean",
        "best_seed_relevance_mean",
        "best_seed_coverage_mean",
        "best_seed_diversity_mean",
        "best_seed_hub_penalty_mean",
        "seed_gold_hit_rate_eval_only",
        "seed_evidence_gold_hit_rate_eval_only",
        "num_refined_seeds_mean",
        "num_changed_seeds_mean",
        "changed_seed_rate",
        "before_seed_gold_hit_rate_eval_only",
        "after_seed_gold_hit_rate_eval_only",
        "before_seed_evidence_gold_hit_rate_eval_only",
        "after_seed_evidence_gold_hit_rate_eval_only",
        "interpretation",
    ]
    lines = [
        "# Phase8 Seed Quality Debug Report",
        "",
        md_table(rows, keys),
        "",
        "## Representative Cases",
        "",
    ]
    for name, items in cases.items():
        lines.extend([f"### {name}", ""])
        if items:
            lines.append(md_table(items, ["query_id", "reason", "best_seed_relevance", "best_hub_penalty", "num_changed_seeds", "question"]))
        else:
            lines.append("No cases found.")
        lines.append("")
    write_md(root / "phase8_seed_quality_debug_report.md", lines)
    print(root / "phase8_seed_quality_debug_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
