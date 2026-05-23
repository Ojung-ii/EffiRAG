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
    iter_queries,
    load_runs,
    md_table,
    mean,
    missing_files_for,
    percentile,
    safe_float,
    write_json,
    write_md,
)


FIELDS = [
    "num_entities",
    "avg_query_relevance",
    "max_query_relevance",
    "avg_entity_degree",
    "num_high_degree_filtered",
    "num_gold_entities_in_universe_eval_only",
    "num_semantic_entities",
    "num_title_lookup_entities",
    "num_graph_flow_entities",
    "num_source_balanced_entities",
]


def _summarize_group(dataset: str, profile: str, variant: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries = list(iter_queries(runs))
    universes = [entity_universe(q) for q in queries]
    row: Dict[str, Any] = {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "num_queries": len(queries),
        "missing_files": {r["run_dir"]: missing_files_for(r) for r in runs},
    }
    for field in FIELDS:
        vals = [safe_float(u.get(field, 0.0), 0.0) for u in universes]
        suffix = "entity_universe_size" if field == "num_entities" else field
        row[f"{suffix}_mean"] = mean(vals)
        if field == "num_entities":
            row["entity_universe_size_p50"] = percentile(vals, 0.50)
            row["entity_universe_size_p95"] = percentile(vals, 0.95)
    hits = [1.0 if bool(u.get("gold_entity_hit_eval_only", False)) else 0.0 for u in universes]
    row["gold_entity_hit_rate_eval_only"] = mean(hits)
    row["interpretation"] = _interpret(row)
    return row


def _interpret(row: Dict[str, Any]) -> str:
    if row.get("variant") == "source_balanced_128":
        return "Phase8 disabled baseline; U_q is not expected."
    if safe_float(row.get("entity_universe_size_mean", 0.0)) <= 0.0:
        return "U_q was not built or was not logged."
    if safe_float(row.get("gold_entity_hit_rate_eval_only", 0.0)) <= 0.01:
        return "U_q is broad but eval-only gold entity hit is near zero; U_q construction or gold-entity diagnostic mapping is a primary suspect."
    if safe_float(row.get("num_source_balanced_entities_mean", 0.0)) <= 0.0:
        return "U_q exists but source-balanced contribution is absent; Phase8 may be replacing useful Phase7 candidates."
    return "U_q has nonzero eval-only gold hits; inspect seed selection next."


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Phase8 query-biased entity universe traces.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = load_runs(root)
    rows = []
    for (dataset, profile, variant), group in sorted(group_runs(runs).items()):
        rows.append(_summarize_group(dataset, profile, variant, group))

    out_json = root / "phase8_entity_universe_debug.json"
    write_json(out_json, {"runs": rows})

    keys = [
        "dataset",
        "profile",
        "variant",
        "num_queries",
        "entity_universe_size_mean",
        "entity_universe_size_p50",
        "entity_universe_size_p95",
        "avg_query_relevance_mean",
        "max_query_relevance_mean",
        "avg_entity_degree_mean",
        "num_high_degree_filtered_mean",
        "gold_entity_hit_rate_eval_only",
        "num_gold_entities_in_universe_eval_only_mean",
        "num_semantic_entities_mean",
        "num_title_lookup_entities_mean",
        "num_graph_flow_entities_mean",
        "num_source_balanced_entities_mean",
        "interpretation",
    ]
    lines = [
        "# Phase8 Entity Universe Debug Report",
        "",
        "This report reads existing traces only. Missing trace files are recorded in the JSON output.",
        "",
        md_table(rows, keys),
        "",
        "## Interpretation Rules",
        "",
        "- Near-zero `gold_entity_hit_rate_eval_only` marks U_q construction or gold-entity diagnostic mapping as suspicious.",
        "- Large `entity_universe_size_mean` with near-zero gold hits means the universe is broad but not landing on answer-relevant entities.",
        "- Zero `num_source_balanced_entities_mean` in pamae variants means the proposal is not benefiting from source-balanced candidates.",
        "",
    ]
    write_md(root / "phase8_entity_universe_debug_report.md", lines)
    print(root / "phase8_entity_universe_debug_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
