#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _parse_run_spec(spec: str):
    token = str(spec or "").strip()
    if not token or "=" not in token:
        raise ValueError(f"Invalid --run spec: {spec} (expected dataset:variant=/abs/path/to/rag_summary.json)")
    label, path = token.split("=", 1)
    label = label.strip()
    path = path.strip()
    dataset = ""
    variant = label
    if ":" in label:
        dataset, variant = label.split(":", 1)
        dataset = dataset.strip()
        variant = variant.strip()
    return dataset, variant, Path(path)


def _extract_metrics(payload: dict) -> dict:
    return {
        "dataset": str(payload.get("dataset", "")),
        "n_samples": int(_safe_float(payload.get("n_samples", 0), 0)),
        "supporting_fact_recall": _safe_float(payload.get("supporting_fact_recall", 0.0), 0.0),
        "recall_at_1": _safe_float(payload.get("supporting_fact_recall_at_1", 0.0), 0.0),
        "recall_at_5": _safe_float(payload.get("supporting_fact_recall_at_5", 0.0), 0.0),
        "recall_at_20": _safe_float(payload.get("supporting_fact_recall_at_20", 0.0), 0.0),
        "rendered_recall": _safe_float(
            payload.get("rendered_supporting_fact_recall", payload.get("supporting_fact_recall", 0.0)),
            0.0,
        ),
        "em": _safe_float(payload.get("em", 0.0), 0.0),
        "f1": _safe_float(payload.get("f1", 0.0), 0.0),
        "retrieval_ms": _safe_float(payload.get("retrieval_latency_ms", 0.0), 0.0),
        "generation_ms": _safe_float(payload.get("generation_ms", 0.0), 0.0),
        "total_ms": _safe_float(payload.get("total_latency_ms", 0.0), 0.0),
    }


def _schema_columns(schema: str):
    if schema == "f1":
        cols = [
            "dataset",
            "variant",
            "n_samples",
            "Recall@1",
            "Recall@5",
            "Recall@20",
            "rendered_recall",
            "EM",
            "F1",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
            "dEM",
            "dF1",
            "dRendered",
            "dGenerationMs",
            "dTotalMs",
            "summary_path",
        ]
        return cols

    cols = [
        "dataset",
        "variant",
        "n_samples",
        "Recall@1",
        "Recall@5",
        "Recall@20",
        "supporting_fact_recall",
        "rendered_recall",
        "EM",
        "F1",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "dRecall@20",
        "dRendered",
        "dEM",
        "dF1",
        "dRetrievalMs",
        "dTotalMs",
        "summary_path",
    ]
    return cols


def _format_row(row: dict, schema: str):
    if schema == "f1":
        return {
            "dataset": row["dataset"],
            "variant": row["variant"],
            "n_samples": row["n_samples"],
            "Recall@1": f"{row['recall_at_1']:.4f}",
            "Recall@5": f"{row['recall_at_5']:.4f}",
            "Recall@20": f"{row['recall_at_20']:.4f}",
            "rendered_recall": f"{row['rendered_recall']:.4f}",
            "EM": f"{row['em']:.4f}",
            "F1": f"{row['f1']:.4f}",
            "retrieval_ms": f"{row['retrieval_ms']:.2f}",
            "generation_ms": f"{row['generation_ms']:.2f}",
            "total_ms": f"{row['total_ms']:.2f}",
            "dEM": f"{row['delta_em']:+.4f}",
            "dF1": f"{row['delta_f1']:+.4f}",
            "dRendered": f"{row['delta_rendered_recall']:+.4f}",
            "dGenerationMs": f"{row['delta_generation_ms']:+.2f}",
            "dTotalMs": f"{row['delta_total_ms']:+.2f}",
            "summary_path": row["summary_path"],
        }

    return {
        "dataset": row["dataset"],
        "variant": row["variant"],
        "n_samples": row["n_samples"],
        "Recall@1": f"{row['recall_at_1']:.4f}",
        "Recall@5": f"{row['recall_at_5']:.4f}",
        "Recall@20": f"{row['recall_at_20']:.4f}",
        "supporting_fact_recall": f"{row['supporting_fact_recall']:.4f}",
        "rendered_recall": f"{row['rendered_recall']:.4f}",
        "EM": f"{row['em']:.4f}",
        "F1": f"{row['f1']:.4f}",
        "retrieval_ms": f"{row['retrieval_ms']:.2f}",
        "generation_ms": f"{row['generation_ms']:.2f}",
        "total_ms": f"{row['total_ms']:.2f}",
        "dRecall@20": f"{row['delta_recall_at_20']:+.4f}",
        "dRendered": f"{row['delta_rendered_recall']:+.4f}",
        "dEM": f"{row['delta_em']:+.4f}",
        "dF1": f"{row['delta_f1']:+.4f}",
        "dRetrievalMs": f"{row['delta_retrieval_ms']:+.2f}",
        "dTotalMs": f"{row['delta_total_ms']:+.2f}",
        "summary_path": row["summary_path"],
    }


def _markdown_table(columns: List[str], rows: List[dict]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def _analysis_text(rows: List[dict], schema: str, baseline_variant: str) -> str:
    by_dataset: Dict[str, List[dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    lines = ["# Experiment Interpretation", ""]
    lines.append(f"- schema: {schema}")
    lines.append(f"- baseline_variant: {baseline_variant}")
    lines.append("")

    best_f1: Optional[Tuple[float, dict]] = None
    best_cov: Optional[Tuple[float, dict]] = None

    for dataset, ds_rows in sorted(by_dataset.items()):
        lines.append(f"## {dataset}")
        baseline = next((x for x in ds_rows if x["variant"] == baseline_variant), None)
        if baseline is None:
            lines.append("- baseline row missing; cannot compute deltas reliably.")
            lines.append("")
            continue

        improved = [x for x in ds_rows if x["variant"] != baseline_variant]
        if not improved:
            lines.append("- no non-baseline variants.")
            lines.append("")
            continue

        top_f1 = max(improved, key=lambda x: x.get("delta_f1", -1e9))
        top_cov_row = max(improved, key=lambda x: x.get("delta_recall_at_20", -1e9))

        lines.append(
            f"- best ΔF1: `{top_f1['variant']}` ({top_f1.get('delta_f1', 0.0):+.4f}), "
            f"ΔEM={top_f1.get('delta_em', 0.0):+.4f}, Δtotal_ms={top_f1.get('delta_total_ms', 0.0):+.2f}"
        )
        lines.append(
            f"- best ΔRecall@20: `{top_cov_row['variant']}` ({top_cov_row.get('delta_recall_at_20', 0.0):+.4f}), "
            f"Δrendered={top_cov_row.get('delta_rendered_recall', 0.0):+.4f}, "
            f"Δretrieval_ms={top_cov_row.get('delta_retrieval_ms', 0.0):+.2f}"
        )
        lines.append("")

        if best_f1 is None or top_f1.get("delta_f1", -1e9) > best_f1[0]:
            best_f1 = (top_f1.get("delta_f1", -1e9), top_f1)
        if best_cov is None or top_cov_row.get("delta_recall_at_20", -1e9) > best_cov[0]:
            best_cov = (top_cov_row.get("delta_recall_at_20", -1e9), top_cov_row)

    lines.append("## Cross-Dataset Takeaway")
    if best_f1 is not None:
        row = best_f1[1]
        lines.append(
            f"- strongest F1 gain variant: `{row['variant']}` on `{row['dataset']}` (ΔF1={row.get('delta_f1', 0.0):+.4f})."
        )
    if best_cov is not None:
        row = best_cov[1]
        lines.append(
            f"- strongest coverage gain variant: `{row['variant']}` on `{row['dataset']}` "
            f"(ΔRecall@20={row.get('delta_recall_at_20', 0.0):+.4f})."
        )
    lines.append("- next operating-point candidate: choose variant with positive ΔRecall@20 and non-negative ΔF1 while keeping latency increase modest.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment runs (F1 boost / semantic ablation).")
    parser.add_argument("--schema", type=str, choices=["f1", "semantic"], required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--baseline-variant", type=str, default="baseline")
    parser.add_argument("--run", action="append", required=True, help="dataset:variant=/abs/path/to/rag_summary.json")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--analysis-md", type=str, default="")
    args = parser.parse_args()

    rows = []
    variant_order = {}

    for idx, spec in enumerate(args.run):
        dataset_label, variant_label, path = _parse_run_spec(spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = _extract_metrics(payload)
        dataset = dataset_label or str(metrics.get("dataset", "")) or "unknown"
        variant = variant_label or "variant"
        variant_order.setdefault(variant, idx)

        row = {
            "dataset": dataset,
            "variant": variant,
            **metrics,
            "summary_path": str(path.resolve()),
        }
        rows.append(row)

    baseline_by_dataset = {}
    for row in rows:
        if row["variant"] == args.baseline_variant:
            baseline_by_dataset[row["dataset"]] = row

    for row in rows:
        base = baseline_by_dataset.get(row["dataset"])
        if base is None:
            base = row

        row["delta_em"] = float(row["em"] - base["em"])
        row["delta_f1"] = float(row["f1"] - base["f1"])
        row["delta_rendered_recall"] = float(row["rendered_recall"] - base["rendered_recall"])
        row["delta_generation_ms"] = float(row["generation_ms"] - base["generation_ms"])
        row["delta_retrieval_ms"] = float(row["retrieval_ms"] - base["retrieval_ms"])
        row["delta_total_ms"] = float(row["total_ms"] - base["total_ms"])
        row["delta_recall_at_20"] = float(row["recall_at_20"] - base["recall_at_20"])

    rows.sort(key=lambda r: (r["dataset"], variant_order.get(r["variant"], 10**9), r["variant"]))

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": args.title,
        "schema": args.schema,
        "baseline_variant": args.baseline_variant,
        "n_runs": len(rows),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_rows = [_format_row(r, args.schema) for r in rows]
    cols = _schema_columns(args.schema)
    lines = [f"# {args.title}", "", _markdown_table(cols, md_rows), ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    if str(args.analysis_md or "").strip():
        analysis_path = Path(args.analysis_md)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(_analysis_text(rows, args.schema, args.baseline_variant), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if str(args.analysis_md or "").strip():
        print(f"Wrote {args.analysis_md}")


if __name__ == "__main__":
    main()
