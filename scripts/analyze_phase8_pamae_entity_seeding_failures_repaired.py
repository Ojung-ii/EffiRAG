#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import (
    failure_label_repaired,
    group_runs,
    is_phase8_variant,
    is_source_balanced_variant,
    load_runs,
    md_table,
    normalized_query_diagnostics,
    question_for,
    write_json,
    write_md,
)


PHASE8_CATEGORIES = [
    "UQ_ENTITY_MISS",
    "SEED_SELECTION_MISS",
    "REFINEMENT_DRIFT",
    "SEED_TO_EVIDENCE_MISS",
    "CANDIDATE_CHAIN_INFEASIBLE",
    "FINAL_SELECTION_FAILED",
    "SELECTED_NOT_SUFFICIENT",
    "QA_PROMPT_FAILED",
    "OTHER",
]

SOURCE_BALANCED_CATEGORIES = [
    "PHASE1_NOT_FOUND",
    "FINAL_SELECTION_FAILED",
    "SELECTED_NOT_SUFFICIENT",
    "QA_PROMPT_FAILED_WITH_GOLD_CONTEXT",
    "QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT",
    "QA_PROMPT_FAILED",
    "OTHER",
    "N/A_PHASE8_STAGE",
]


def _row_for_query(dataset: str, profile: str, variant: str, query: Dict[str, Any]) -> Dict[str, Any]:
    label = failure_label_repaired(variant, query)
    nd = normalized_query_diagnostics(query)
    return {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "query_id": query.get("query_id", ""),
        "category": label,
        "question": question_for(query),
        "normalized_candidate_gold_recall": nd.get("normalized_candidate_gold_recall"),
        "normalized_selected_gold_recall": nd.get("normalized_selected_gold_recall"),
        "normalized_rendered_gold_recall": nd.get("normalized_rendered_gold_recall"),
        "candidate_oracle_F1": nd.get("candidate_oracle_F1"),
        "selected_context_F1": nd.get("selected_context_F1"),
        "chain_unit_oracle_feasible": nd.get("chain_unit_oracle_feasible"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repaired Phase8 failure attribution.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = load_runs(root)
    rows: List[Dict[str, Any]] = []
    summaries = []
    examples = defaultdict(list)
    for (dataset, profile, variant), group in sorted(group_runs(runs).items()):
        queries = [q for run in group for q in run.get("queries", [])]
        counter: Counter = Counter()
        allowed = PHASE8_CATEGORIES if is_phase8_variant(variant) else SOURCE_BALANCED_CATEGORIES
        for query in queries:
            row = _row_for_query(dataset, profile, variant, query)
            category = row["category"]
            if is_source_balanced_variant(variant) and category in PHASE8_CATEGORIES[:-1]:
                category = "N/A_PHASE8_STAGE"
                row["category"] = category
            counter[category] += 1
            rows.append(row)
            key = (dataset, profile, variant, category)
            if len(examples[key]) < 3:
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
        root / "phase8_pamae_failure_analysis_repaired.json",
        {"summary": summaries, "rows": rows, "examples": {"/".join(k): v for k, v in examples.items()}},
    )
    lines = [
        "# Phase8 PAMAE Failure Analysis Repaired",
        "",
        "This report is variant-aware: source-balanced baselines cannot receive Phase8-specific failure labels.",
        "",
        md_table(summaries, ["dataset", "profile", "variant", "category", "count", "rate", "num_queries"]),
        "",
        "## Checks",
        "",
        "- Source-balanced rows use `PHASE1_NOT_FOUND`, final-selection, selected-context, QA, or `OTHER` labels only.",
        "- Phase8-specific labels are reserved for pamae variants.",
        "- Candidate/selected/rendered recall values are normalized before attribution.",
        "",
    ]
    write_md(root / "phase8_pamae_failure_analysis_repaired.md", lines)
    print(root / "phase8_pamae_failure_analysis_repaired.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
