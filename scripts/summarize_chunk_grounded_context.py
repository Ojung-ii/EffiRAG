#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _parse_run_spec(spec: str) -> Tuple[str, str, Path]:
    token = str(spec or "").strip()
    if not token or "=" not in token:
        raise ValueError(f"Invalid --run spec: {spec} (expected dataset:variant=/abs/path/to/rag_summary.json)")
    label, path = token.split("=", 1)
    label = label.strip()
    path = Path(path.strip())

    dataset = ""
    variant = label
    if ":" in label:
        dataset, variant = label.split(":", 1)
        dataset = dataset.strip()
        variant = variant.strip()
    return dataset, variant, path


def _extract_metrics(payload: dict) -> dict:
    finish_counts = payload.get("finish_reason_counts", {}) or {}
    top_finish_reason = ""
    if isinstance(finish_counts, dict) and finish_counts:
        top_finish_reason = max(finish_counts.items(), key=lambda kv: int(kv[1]))[0]

    retrieval_params = payload.get("retrieval_params", {}) or {}
    render_params = payload.get("render_params", {}) or {}

    node_count = 0
    edge_count = 0
    try:
        summary_path = Path(str(payload.get("__summary_path__", "") or ""))
        if summary_path:
            query_results_path = summary_path.parent / "rag_query_results.jsonl"
            if query_results_path.exists():
                with query_results_path.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                if first:
                    first_row = json.loads(first)
                    global_index = (
                        ((first_row.get("retrieval", {}) or {}).get("diagnostics", {}) or {}).get("global_index", {})
                        or {}
                    )
                    stats = (global_index.get("stats", {}) or {})
                    node_count = _safe_int(stats.get("num_nodes", 0), 0)
                    edge_count = _safe_int(stats.get("num_edges", 0), 0)
    except Exception:
        node_count = 0
        edge_count = 0

    return {
        "dataset": str(payload.get("dataset", "")),
        "n_samples": _safe_int(payload.get("n_samples", 0), 0),
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
        "prompt_tokens": _safe_float(payload.get("prompt_tokens_avg", 0.0), 0.0),
        "completion_tokens": _safe_float(payload.get("completion_tokens_avg", 0.0), 0.0),
        "chunk_excerpts_used_avg": _safe_float(payload.get("chunk_excerpts_used_avg", 0.0), 0.0),
        "chunk_excerpt_avg_len": _safe_float(payload.get("chunk_excerpt_avg_len", 0.0), 0.0),
        "evidence_package_count_avg": _safe_float(payload.get("evidence_package_count_avg", 0.0), 0.0),
        "truncated_corridors_avg": _safe_float(payload.get("truncated_corridors_avg", 0.0), 0.0),
        "truncated_sentences_avg": _safe_float(payload.get("truncated_sentences_avg", 0.0), 0.0),
        "fallback_count": _safe_int(payload.get("fallback_count", 0), 0),
        "top_finish_reason": str(top_finish_reason),
        "graph_mode": str(retrieval_params.get("graph_mode", "current_entity_graph")),
        "delivery_mode": str(render_params.get("delivery_mode", payload.get("render_mode_resolved", "sentence_compressed"))),
        "node_count": int(node_count),
        "edge_count": int(edge_count),
    }


def _markdown_table(columns: List[str], rows: List[dict]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize chunk-grounded context ablations.")
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--baseline-variant", type=str, default="cg_baseline")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--interpretation-md", type=str, required=True)
    parser.add_argument("--experiment-family", type=str, default="next_method_design")
    parser.add_argument("--question-being-answered", type=str, default="")
    parser.add_argument("--baseline-reference", type=str, default="")
    parser.add_argument("--frozen-config-reference", type=str, default="")
    parser.add_argument("--dataset-scope", type=str, default="")
    parser.add_argument("--changed-components", action="append", default=[])
    parser.add_argument("--git-branch", type=str, default="")
    parser.add_argument("--git-commit", type=str, default="")
    parser.add_argument("--git-tag", type=str, default="")
    args = parser.parse_args()

    rows = []
    for spec in args.run:
        ds_label, variant, path = _parse_run_spec(spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["__summary_path__"] = str(path.resolve())
        metrics = _extract_metrics(payload)
        dataset = ds_label or str(metrics.get("dataset", "")) or "unknown"

        row = {
            "dataset": dataset,
            "variant": variant,
            **metrics,
            "summary_path": str(path.resolve()),
        }
        rows.append(row)

    baseline_by_dataset: Dict[str, dict] = {}
    for row in rows:
        if row["variant"] == args.baseline_variant:
            baseline_by_dataset[row["dataset"]] = row

    for row in rows:
        base = baseline_by_dataset.get(row["dataset"])
        if base is None:
            row["delta_recall_at_20"] = 0.0
            row["delta_rendered_recall"] = 0.0
            row["delta_em"] = 0.0
            row["delta_f1"] = 0.0
            row["delta_retrieval_ms"] = 0.0
            row["delta_total_ms"] = 0.0
        else:
            row["delta_recall_at_20"] = row["recall_at_20"] - base["recall_at_20"]
            row["delta_rendered_recall"] = row["rendered_recall"] - base["rendered_recall"]
            row["delta_em"] = row["em"] - base["em"]
            row["delta_f1"] = row["f1"] - base["f1"]
            row["delta_retrieval_ms"] = row["retrieval_ms"] - base["retrieval_ms"]
            row["delta_total_ms"] = row["total_ms"] - base["total_ms"]

    rows.sort(key=lambda r: (r["dataset"], r["variant"]))

    payload = {
        "title": args.title,
        "baseline_variant": args.baseline_variant,
        "experiment_family": args.experiment_family,
        "question_being_answered": args.question_being_answered,
        "baseline_reference": args.baseline_reference,
        "frozen_config_reference": args.frozen_config_reference,
        "dataset_scope": args.dataset_scope,
        "changed_components": list(args.changed_components or []),
        "git_branch": args.git_branch,
        "git_commit": args.git_commit,
        "git_tag": args.git_tag,
        "rows": rows,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_interp = Path(args.interpretation_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_interp.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_columns = [
        "dataset",
        "variant",
        "graph_mode",
        "delivery_mode",
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
        "node_count",
        "edge_count",
        "prompt_tokens",
        "completion_tokens",
        "chunk_excerpts_used_avg",
        "chunk_excerpt_avg_len",
        "evidence_package_count_avg",
        "truncated_corridors_avg",
        "truncated_sentences_avg",
        "fallback_count",
        "top_finish_reason",
        "dRecall@20",
        "dRendered",
        "dEM",
        "dF1",
        "dRetrievalMs",
        "dTotalMs",
        "summary_path",
    ]

    md_rows = []
    for r in rows:
        md_rows.append(
            {
                "dataset": r["dataset"],
                "variant": r["variant"],
                "graph_mode": r["graph_mode"],
                "delivery_mode": r["delivery_mode"],
                "n_samples": r["n_samples"],
                "Recall@1": f"{r['recall_at_1']:.4f}",
                "Recall@5": f"{r['recall_at_5']:.4f}",
                "Recall@20": f"{r['recall_at_20']:.4f}",
                "supporting_fact_recall": f"{r['supporting_fact_recall']:.4f}",
                "rendered_recall": f"{r['rendered_recall']:.4f}",
                "EM": f"{r['em']:.4f}",
                "F1": f"{r['f1']:.4f}",
                "retrieval_ms": f"{r['retrieval_ms']:.2f}",
                "generation_ms": f"{r['generation_ms']:.2f}",
                "total_ms": f"{r['total_ms']:.2f}",
                "node_count": r["node_count"],
                "edge_count": r["edge_count"],
                "prompt_tokens": f"{r['prompt_tokens']:.2f}",
                "completion_tokens": f"{r['completion_tokens']:.2f}",
                "chunk_excerpts_used_avg": f"{r['chunk_excerpts_used_avg']:.2f}",
                "chunk_excerpt_avg_len": f"{r['chunk_excerpt_avg_len']:.2f}",
                "evidence_package_count_avg": f"{r['evidence_package_count_avg']:.2f}",
                "truncated_corridors_avg": f"{r['truncated_corridors_avg']:.2f}",
                "truncated_sentences_avg": f"{r['truncated_sentences_avg']:.2f}",
                "fallback_count": r["fallback_count"],
                "top_finish_reason": r["top_finish_reason"],
                "dRecall@20": f"{r['delta_recall_at_20']:+.4f}",
                "dRendered": f"{r['delta_rendered_recall']:+.4f}",
                "dEM": f"{r['delta_em']:+.4f}",
                "dF1": f"{r['delta_f1']:+.4f}",
                "dRetrievalMs": f"{r['delta_retrieval_ms']:+.2f}",
                "dTotalMs": f"{r['delta_total_ms']:+.2f}",
                "summary_path": r["summary_path"],
            }
        )

    md_lines = [f"# {args.title}", "", f"- baseline_variant: `{args.baseline_variant}`", ""]
    md_lines.append(_markdown_table(md_columns, md_rows))
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    interp_lines = [
        "# Chunk-Grounded Context Interpretation",
        "",
        f"- experiment_family: `{args.experiment_family}`",
        f"- question_being_answered: {args.question_being_answered}",
        f"- baseline_reference: {args.baseline_reference}",
        f"- frozen_config_reference: {args.frozen_config_reference}",
        f"- dataset_scope: {args.dataset_scope}",
        f"- git_branch: {args.git_branch}",
        f"- git_commit: {args.git_commit}",
        f"- git_tag: {args.git_tag}",
        "",
    ]

    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)

    for dataset, ds_rows in sorted(grouped.items()):
        interp_lines.append(f"## {dataset}")
        baseline = next((x for x in ds_rows if x["variant"] == args.baseline_variant), None)
        if baseline is None:
            interp_lines.append("- baseline row missing; skip dataset interpretation.")
            interp_lines.append("")
            continue
        non_base = [x for x in ds_rows if x["variant"] != args.baseline_variant]
        if not non_base:
            interp_lines.append("- no non-baseline variant rows.")
            interp_lines.append("")
            continue

        best_f1 = max(non_base, key=lambda x: x.get("delta_f1", -1e9))
        best_em = max(non_base, key=lambda x: x.get("delta_em", -1e9))
        best_rendered = max(non_base, key=lambda x: x.get("delta_rendered_recall", -1e9))

        interp_lines.append(
            f"- best ΔF1: `{best_f1['variant']}` ({best_f1['delta_f1']:+.4f}), "
            f"ΔEM={best_f1['delta_em']:+.4f}, Δtotal_ms={best_f1['delta_total_ms']:+.2f}"
        )
        interp_lines.append(
            f"- best ΔEM: `{best_em['variant']}` ({best_em['delta_em']:+.4f}), "
            f"ΔF1={best_em['delta_f1']:+.4f}, Δtotal_ms={best_em['delta_total_ms']:+.2f}"
        )
        interp_lines.append(
            f"- best Δrendered_recall: `{best_rendered['variant']}` ({best_rendered['delta_rendered_recall']:+.4f}), "
            f"ΔF1={best_rendered['delta_f1']:+.4f}, Δtotal_ms={best_rendered['delta_total_ms']:+.2f}"
        )
        interp_lines.append("")

    interp_lines.append("## Decision Hint")
    interp_lines.append("- Prefer a variant with non-negative Δrendered_recall and positive ΔF1 while keeping Δtotal_ms moderate.")
    interp_lines.append("- If ΔRecall@20 stays flat but ΔF1 improves, that indicates delivery/package quality gain over retrieval gain.")
    interp_lines.append("")
    out_interp.write_text("\n".join(interp_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
