#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from effirag.eval_metrics import table_from_records, write_json, write_tsv
from effirag.utils import markdown_table, safe_div


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _fmt(v: Any, ndigits: int = 4) -> str:
    return f"{_safe_float(v, 0.0):.{ndigits}f}"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_run_records(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def _summary_metric(summary: Mapping[str, Any], key: str, fallback: float = 0.0) -> float:
    if key in summary:
        return _safe_float(summary.get(key), fallback)
    if key == "recall_at_1":
        return _safe_float(summary.get("supporting_fact_recall_at_1", fallback), fallback)
    if key == "recall_at_5":
        return _safe_float(summary.get("supporting_fact_recall_at_5", fallback), fallback)
    if key == "recall_at_10":
        return _safe_float(summary.get("supporting_fact_recall_at_10", fallback), fallback)
    if key == "recall_at_20":
        return _safe_float(summary.get("supporting_fact_recall_at_20", fallback), fallback)
    if key == "sf_precision":
        return _safe_float(summary.get("supporting_fact_precision", fallback), fallback)
    if key == "sf_recall":
        return _safe_float(summary.get("supporting_fact_recall", fallback), fallback)
    if key == "sf_f1":
        return _safe_float(summary.get("supporting_fact_f1", fallback), fallback)
    if key == "retrieval_ms":
        return _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", fallback)), fallback)
    if key == "generation_ms":
        return _safe_float(
            summary.get("generation_ms", summary.get("generation_latency_ms", fallback)),
            fallback,
        )
    if key == "total_ms":
        return _safe_float(summary.get("total_ms", summary.get("total_latency_ms", fallback)), fallback)
    return fallback


def _build_row(dataset: str, variant: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": str(dataset),
        "variant": str(variant),
        "R@1": _summary_metric(summary, "recall_at_1"),
        "R@5": _summary_metric(summary, "recall_at_5"),
        "R@10": _summary_metric(summary, "recall_at_10"),
        "R@20": _summary_metric(summary, "recall_at_20"),
        "Hit@1": _summary_metric(summary, "hit_at_1"),
        "Hit@5": _summary_metric(summary, "hit_at_5"),
        "Hit@10": _summary_metric(summary, "hit_at_10"),
        "Hit@20": _summary_metric(summary, "hit_at_20"),
        "MRR@10": _summary_metric(summary, "mrr_at_10"),
        "MRR@20": _summary_metric(summary, "mrr_at_20"),
        "nDCG@5": _summary_metric(summary, "ndcg_at_5"),
        "nDCG@10": _summary_metric(summary, "ndcg_at_10"),
        "nDCG@20": _summary_metric(summary, "ndcg_at_20"),
        "AtLeastOne@5": _summary_metric(summary, "atleastone_at_5"),
        "AtLeastOne@10": _summary_metric(summary, "atleastone_at_10"),
        "AtLeastOne@20": _summary_metric(summary, "atleastone_at_20"),
        "AllSupport@5": _summary_metric(summary, "allsupport_at_5"),
        "AllSupport@10": _summary_metric(summary, "allsupport_at_10"),
        "AllSupport@20": _summary_metric(summary, "allsupport_at_20"),
        "ContextPrecision": _summary_metric(summary, "context_precision"),
        "Faithfulness": _summary_metric(summary, "faithfulness"),
        "sf_P": _summary_metric(summary, "sf_precision"),
        "sf_R": _summary_metric(summary, "sf_recall"),
        "sf_F1": _summary_metric(summary, "sf_f1"),
        "EM": _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0),
        "F1": _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0),
        "retrieval_ms": _summary_metric(summary, "retrieval_ms"),
        "generation_ms": _summary_metric(summary, "generation_ms"),
        "total_ms": _summary_metric(summary, "total_ms"),
        "fallback_rate": _safe_float(summary.get("fallback_rate", 0.0), 0.0),
        "equivalent_evidence_coverage": _safe_float(summary.get("equivalent_evidence_coverage", 0.0), 0.0),
        "minimal_support_subset_coverage": _safe_float(summary.get("minimal_support_subset_coverage", 0.0), 0.0),
        "answer_bearing_chunk_present": _safe_float(summary.get("answer_bearing_chunk_present", 0.0), 0.0),
        "answer_present_but_generation_fail": _safe_float(summary.get("answer_present_but_generation_fail", 0.0), 0.0),
        "output_overlap_answer_bearing": _safe_float(summary.get("output_overlap_answer_bearing", 0.0), 0.0),
        "summary_path": str(summary.get("output_dir", "")),
    }
    row["R@5_per_100ms"] = float(safe_div(row["R@5"] * 100.0, row["retrieval_ms"])) if row["retrieval_ms"] > 0.0 else 0.0
    row["MRR@10_per_100ms"] = (
        float(safe_div(row["MRR@10"] * 100.0, row["retrieval_ms"])) if row["retrieval_ms"] > 0.0 else 0.0
    )
    row["ContextPrecision_per_100ms"] = (
        float(safe_div(row["ContextPrecision"] * 100.0, row["total_ms"])) if row["total_ms"] > 0.0 else 0.0
    )
    row["EM_per_100ms"] = float(safe_div(row["EM"] * 100.0, row["total_ms"])) if row["total_ms"] > 0.0 else 0.0
    row["F1_per_100ms"] = float(safe_div(row["F1"] * 100.0, row["total_ms"])) if row["total_ms"] > 0.0 else 0.0
    return row


def _delta_vs_baseline(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idx = {(r["dataset"], r["variant"]): r for r in rows}
    out = []
    for row in rows:
        base = idx.get((row["dataset"], "baseline_mid_reconfirm"))
        if base is None:
            continue
        out.append(
            {
                "dataset": row["dataset"],
                "variant": row["variant"],
                "ΔR@5": row["R@5"] - base["R@5"],
                "ΔEM": row["EM"] - base["EM"],
                "ΔF1": row["F1"] - base["F1"],
                "Δretrieval_ms": row["retrieval_ms"] - base["retrieval_ms"],
                "Δgeneration_ms": row["generation_ms"] - base["generation_ms"],
                "Δtotal_ms": row["total_ms"] - base["total_ms"],
            }
        )
    return out


def _records_to_tsv_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        for k, v in list(rec.items()):
            if isinstance(v, float):
                rec[k] = f"{v:.6f}"
        out.append(rec)
    return out


def _markdown_records(headers: List[str], rows: List[Dict[str, Any]], digits: int = 4) -> str:
    table_rows: List[List[str]] = []
    for row in rows:
        one: List[str] = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                one.append(f"{v:.{digits}f}")
            else:
                one.append(str(v))
        table_rows.append(one)
    return markdown_table(headers, table_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", required=True)
    parser.add_argument("--print-tables", default="true")
    args = parser.parse_args()

    round_root = Path(args.round_root).resolve()
    run_records_path = round_root / "run_records.tsv"
    out_json = round_root / "metric_round_metrics.json"
    out_md = round_root / "metric_round_metrics.md"
    out_delta_md = round_root / "metric_delta_table.md"
    out_eff_md = round_root / "metric_efficiency_table.md"
    out_diag_md = round_root / "metric_diagnostic_table.md"
    out_defs_md = round_root / "metric_definition_notes.md"
    out_tsv = round_root / "metric_round_metrics.tsv"

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for rr in _read_run_records(run_records_path):
        status = str(rr.get("status", "") or "")
        if status not in {"ok", "skipped"}:
            failures.append(rr)
            continue
        summary_path = str(rr.get("summary_path", "") or "").strip()
        if not summary_path:
            continue
        sp = Path(summary_path)
        if not sp.exists():
            failures.append(rr)
            continue
        summary = _load_json(sp)
        rows.append(_build_row(str(rr.get("dataset", "")), str(rr.get("variant", "")), summary))

    rows.sort(key=lambda x: (x.get("dataset", ""), x.get("variant", "")))
    delta_rows = _delta_vs_baseline(rows)

    retrieval_headers = [
        "dataset",
        "variant",
        "R@1",
        "R@5",
        "R@10",
        "R@20",
        "Hit@5",
        "Hit@10",
        "MRR@10",
        "nDCG@10",
        "AtLeastOne@10",
        "AllSupport@10",
    ]
    context_headers = [
        "dataset",
        "variant",
        "ContextPrecision",
        "Faithfulness",
        "sf_P",
        "sf_R",
        "sf_F1",
        "EM",
        "F1",
    ]
    efficiency_headers = [
        "dataset",
        "variant",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "R@5_per_100ms",
        "MRR@10_per_100ms",
        "ContextPrecision_per_100ms",
        "EM_per_100ms",
        "F1_per_100ms",
    ]
    diagnostic_headers = [
        "dataset",
        "variant",
        "equivalent_evidence_coverage",
        "minimal_support_subset_coverage",
        "answer_bearing_chunk_present",
        "answer_present_but_generation_fail",
        "output_overlap_answer_bearing",
    ]
    delta_headers = [
        "dataset",
        "variant",
        "ΔR@5",
        "ΔEM",
        "ΔF1",
        "Δretrieval_ms",
        "Δgeneration_ms",
        "Δtotal_ms",
    ]

    retrieval_md = _markdown_records(retrieval_headers, rows, digits=4)
    context_md = _markdown_records(context_headers, rows, digits=4)
    efficiency_md = _markdown_records(efficiency_headers, rows, digits=4)
    diagnostic_md = _markdown_records(diagnostic_headers, rows, digits=4)
    delta_md = _markdown_records(delta_headers, delta_rows, digits=4)

    out_delta_md.write_text(delta_md + "\n", encoding="utf-8")
    out_eff_md.write_text(efficiency_md + "\n", encoding="utf-8")
    out_diag_md.write_text(diagnostic_md + "\n", encoding="utf-8")

    metric_defs = """
# Metric Definition Notes

- Recall@K / Hit@K / MRR@K / nDCG@K:
  strict gold supporting-fact relevance (title + sentence index), chunk evidence via gold sentence containment.
- Context Precision:
  Ragas-style average precision over ranked retrieved contexts, adapted to supporting-fact relevance.
- Faithfulness:
  proxy/approximation; uses evidence-supported generator flag when present, otherwise lexical evidence overlap with rendered context.
- Supporting-fact P/R/F1:
  Hotpot-style evidence alignment metrics on retrieved/rendered units.
- AtLeastOne@AllSupport@K:
  multi-hop evidence sufficiency indicators (any-support vs all-support).
- Efficiency-normalized metrics:
  metric * 100 / latency(ms), used as efficiency support signals (not replacements for raw quality metrics).
- Diagnostics:
  equivalent/minimal/answer-bearing/ABGF/overlap are diagnostic-only and must be interpreted alongside strict retrieval metrics.
""".strip()
    out_defs_md.write_text(metric_defs + "\n", encoding="utf-8")

    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "records": rows,
        "delta_vs_baseline": delta_rows,
        "failures_count": len(failures),
    }
    write_json(str(out_json), bundle)

    if rows:
        headers = list(rows[0].keys())
        write_tsv(str(out_tsv), headers, _records_to_tsv_rows(rows))

    md_lines = [
        "# Retrieval Metric Round",
        "",
        "## Main Retrieval Metrics Table",
        "",
        retrieval_md,
        "",
        "## RAG / Context Metrics Table",
        "",
        context_md,
        "",
        "## Efficiency Table",
        "",
        efficiency_md,
        "",
        "## Diagnostic Table",
        "",
        diagnostic_md,
        "",
        "## Delta vs Baseline",
        "",
        delta_md,
        "",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    if str(args.print_tables).strip().lower() in {"1", "true", "yes", "y", "on"}:
        print("## Main Retrieval Metrics Table")
        print(retrieval_md)
        print("\n## RAG / Context Metrics Table")
        print(context_md)
        print("\n## Efficiency Table")
        print(efficiency_md)
        print("\n## Diagnostic Table")
        print(diagnostic_md)


if __name__ == "__main__":
    main()

