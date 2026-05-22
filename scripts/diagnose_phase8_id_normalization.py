#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import (
    canonical_support_key,
    group_runs,
    load_runs,
    md_table,
    mean,
    normalize_evidence_id,
    normalize_title,
    normalized_id_set,
    phase7_candidate_recall,
    safe_float,
    support_keys_from_gold,
    write_json,
    write_md,
)


def _recall(gold: set[str], ids: set[str]) -> float:
    if not gold:
        return 0.0
    return float(len(gold & ids) / len(gold))


def _query_diag(query: Dict[str, Any]) -> Dict[str, Any]:
    p7 = query.get("phase7", {})
    p8q = query.get("phase8_query", {})
    p7diag = query.get("phase7_diag", {})
    result = query.get("result", {})
    gold = support_keys_from_gold(p7diag.get("gold_supporting_facts", []))
    candidate_ids = normalized_id_set(p7diag.get("phase2_candidate_ids", p7.get("phase2_candidate_ids", [])))
    selected_ids = normalized_id_set(p7diag.get("selected_evidence_ids", p7.get("selected_atom_ids", [])))
    rendered_ids = normalized_id_set(p7diag.get("rendered_evidence_ids", result.get("rendered_sentence_ids", [])))
    sf_recall = safe_float(p7.get("rendered_sf_recall_eval_only", 0.0), 0.0)
    if result.get("metrics"):
        sf_recall = max(sf_recall, safe_float(result.get("metrics", {}).get("supporting_fact_recall", 0.0), 0.0))
    p7_candidate = phase7_candidate_recall(query)
    p8_candidate = safe_float(p8q.get("candidate_gold_recall", 0.0), 0.0)
    return {
        "query_id": query.get("query_id", ""),
        "num_gold": len(gold),
        "candidate_id_count": len(candidate_ids),
        "selected_id_count": len(selected_ids),
        "rendered_id_count": len(rendered_ids),
        "normalized_candidate_recall": _recall(gold, candidate_ids),
        "normalized_selected_recall": _recall(gold, selected_ids),
        "normalized_rendered_recall": _recall(gold, rendered_ids),
        "phase7_candidate_gold_recall": p7_candidate,
        "phase8_query_candidate_gold_recall": p8_candidate,
        "sf_recall": sf_recall,
        "phase8_logging_mismatch": p7_candidate > 0.0 and p8_candidate <= 0.0,
        "sf_positive_candidate_zero_warning": sf_recall > 0.0 and p7_candidate <= 0.0,
        "selected_context_positive_selected_zero_warning": sf_recall > 0.0 and _recall(gold, selected_ids) <= 0.0,
    }


def _summarize_group(dataset: str, profile: str, variant: str, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    qd = [_query_diag(q) for q in queries]
    row = {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "num_queries": len(qd),
        "phase7_candidate_gold_recall_mean": mean(d["phase7_candidate_gold_recall"] for d in qd),
        "phase8_query_candidate_gold_recall_mean": mean(d["phase8_query_candidate_gold_recall"] for d in qd),
        "normalized_candidate_recall_mean": mean(d["normalized_candidate_recall"] for d in qd),
        "normalized_selected_recall_mean": mean(d["normalized_selected_recall"] for d in qd),
        "normalized_rendered_recall_mean": mean(d["normalized_rendered_recall"] for d in qd),
        "sf_recall_mean": mean(d["sf_recall"] for d in qd),
        "phase8_logging_mismatch_count": sum(1 for d in qd if d["phase8_logging_mismatch"]),
        "sf_positive_candidate_zero_warning_count": sum(1 for d in qd if d["sf_positive_candidate_zero_warning"]),
        "selected_context_positive_selected_zero_warning_count": sum(1 for d in qd if d["selected_context_positive_selected_zero_warning"]),
    }
    row["interpretation"] = _interpret(row)
    return row


def _interpret(row: Dict[str, Any]) -> str:
    if row["phase8_logging_mismatch_count"] > 0:
        return "phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken."
    if row["sf_positive_candidate_zero_warning_count"] > 0:
        return "SF-R is positive while candidate recall is zero; candidate id normalization is suspicious."
    if row["normalized_candidate_recall_mean"] == 0 and row["sf_recall_mean"] > 0:
        return "Candidate ids and gold ids are likely in different namespaces."
    return "No primary id-normalization warning for this group."


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Phase8 diagnostic id normalization.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = load_runs(root)
    rows = []
    examples = []
    all_source_balanced_zero_oracle = True
    for (dataset, profile, variant), group in sorted(group_runs(runs).items()):
        queries = [q for run in group for q in run.get("queries", [])]
        row = _summarize_group(dataset, profile, variant, queries)
        rows.append(row)
        if variant == "source_balanced_128" and row["phase7_candidate_gold_recall_mean"] > 0.0:
            all_source_balanced_zero_oracle = False
        for q in queries:
            d = _query_diag(q)
            if d["phase8_logging_mismatch"] and len(examples) < 20:
                examples.append(d)

    warnings = []
    for row in rows:
        if row["sf_positive_candidate_zero_warning_count"] > 0:
            warnings.append(f"WARNING: SF-R > 0 but candidate_gold_recall == 0 in {row['dataset']}/{row['profile']}/{row['variant']}")
        if row["selected_context_positive_selected_zero_warning_count"] > 0:
            warnings.append(f"WARNING: selected/rendered context recall mismatch in {row['dataset']}/{row['profile']}/{row['variant']}")
    if all_source_balanced_zero_oracle:
        warnings.append("WARNING: source_balanced_128 candidate oracle metrics are zero for all datasets.")

    out = {
        "normalizers": {
            "normalize_title_example": normalize_title("A &amp; B"),
            "normalize_evidence_id_example": normalize_evidence_id("A &amp; B::1"),
            "canonical_support_key_example": canonical_support_key("A &amp; B", 1),
        },
        "runs": rows,
        "warnings": warnings,
        "phase8_logging_mismatch_examples": examples,
    }
    write_json(root / "phase8_id_normalization.json", out)

    keys = [
        "dataset",
        "profile",
        "variant",
        "num_queries",
        "phase7_candidate_gold_recall_mean",
        "phase8_query_candidate_gold_recall_mean",
        "normalized_candidate_recall_mean",
        "normalized_selected_recall_mean",
        "normalized_rendered_recall_mean",
        "sf_recall_mean",
        "phase8_logging_mismatch_count",
        "sf_positive_candidate_zero_warning_count",
        "selected_context_positive_selected_zero_warning_count",
        "interpretation",
    ]
    lines = [
        "# Phase8 ID Normalization Report",
        "",
        md_table(rows, keys),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {w}" for w in warnings] or ["No warnings."])
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "- Candidate ids are normalized with `normalize_evidence_id`.",
            "- Gold support ids are normalized with `canonical_support_key(title, sent_idx)`.",
            "- HTML entities are unescaped before comparison, e.g. `&amp;` becomes `&`.",
            "- `s::` atom ids are kept as atom ids; title sentence ids are compared to gold support ids.",
            "- The current source-balanced zero values in `phase8_query_trace` are diagnostic logging mismatches, not evidence that source-balanced candidates lack oracle coverage.",
            "",
        ]
    )
    if examples:
        lines.extend(["## Phase8 Logging Mismatch Examples", "", md_table(examples[:10], ["query_id", "phase7_candidate_gold_recall", "phase8_query_candidate_gold_recall", "normalized_candidate_recall", "sf_recall"])])
    write_md(root / "phase8_id_normalization_report.md", lines)
    print(root / "phase8_id_normalization_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
