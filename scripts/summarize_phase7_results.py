#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _percentiles(values: Iterable[float]) -> Dict[str, float]:
    seq = [float(v) for v in values]
    if not seq:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(seq, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _fmt_ms(x: float) -> str:
    return f"{float(x):.2f}"


def _fmt4(x: float) -> str:
    return f"{float(x):.4f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def _stage_timing_stats(stage_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(list)
    for row in stage_rows:
        grouped[str(row.get("stage", ""))].append(_safe_float(row.get("elapsed_ms", 0.0), 0.0))
    return {stage: _percentiles(vals) for stage, vals in grouped.items()}


def _evidence_survival(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    def _mean_key(key: str) -> float:
        vals = []
        for row in traces:
            v = row.get(key, None)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    return {
        "candidate_sf_recall_eval_only": _mean_key("candidate_sf_recall_eval_only"),
        "selected_sf_recall_eval_only": _mean_key("selected_sf_recall_eval_only"),
        "rendered_sf_recall_eval_only": _mean_key("rendered_sf_recall_eval_only"),
        "candidate_sf_precision_eval_only": _mean_key("candidate_sf_precision_eval_only"),
        "selected_sf_precision_eval_only": _mean_key("selected_sf_precision_eval_only"),
        "rendered_sf_precision_eval_only": _mean_key("rendered_sf_precision_eval_only"),
    }


def _build_run_summary(run_dir: Path) -> Dict[str, Any]:
    rag_summary = _read_json(run_dir / "rag_summary.json", {})
    traces = _read_jsonl(run_dir / "phase7_query_trace.jsonl")
    stage_rows = _read_jsonl(run_dir / "phase7_stage_timing.jsonl")
    manifest = _read_json(run_dir / "phase7_run_manifest.json", {})
    query_rows = _read_jsonl(run_dir / "rag_query_results.jsonl")

    timing_by_stage = _stage_timing_stats(stage_rows)
    survival = _evidence_survival(traces)

    slow_queries = sorted(
        traces,
        key=lambda r: _safe_float((r.get("stage_ms", {}) or {}).get("total_retrieval", 0.0), 0.0),
        reverse=True,
    )[:20]
    slow_graph_flow = sorted(
        traces,
        key=lambda r: _safe_float((r.get("stage_ms", {}) or {}).get("graph_flow", 0.0), 0.0),
        reverse=True,
    )[:20]
    large_graph = sorted(
        traces,
        key=lambda r: int(r.get("num_local_graph_nodes", 0) or 0),
        reverse=True,
    )[:20]
    collapse_cases = []
    for row in traces:
        c = row.get("candidate_sf_recall_eval_only")
        s = row.get("selected_sf_recall_eval_only")
        if isinstance(c, (int, float)) and isinstance(s, (int, float)):
            if float(c) - float(s) >= 0.2:
                collapse_cases.append(row)
    collapse_cases = sorted(
        collapse_cases,
        key=lambda r: (_safe_float(r.get("candidate_sf_recall_eval_only", 0.0), 0.0) - _safe_float(r.get("selected_sf_recall_eval_only", 0.0), 0.0)),
        reverse=True,
    )[:20]

    empty_candidate = [r for r in traces if int(r.get("num_candidate_atoms", 0) or 0) <= 0]
    empty_selected = [r for r in traces if int(r.get("num_selected_atoms", 0) or 0) <= 0]

    bottleneck_counter = defaultdict(int)
    for row in traces:
        for flag in list(row.get("bottleneck_flags", []) or []):
            bottleneck_counter[str(flag)] += 1

    summary = {
        "run_dir": str(run_dir.resolve()),
        "dataset": str(rag_summary.get("dataset", manifest.get("dataset", ""))),
        "method": str(rag_summary.get("method", "phase7_evidence_flow")),
        "n_samples": int(rag_summary.get("n_samples", len(query_rows))),
        "metrics": {
            "em": _safe_float(rag_summary.get("em", 0.0), 0.0),
            "f1": _safe_float(rag_summary.get("f1", 0.0), 0.0),
            "supporting_fact_recall": _safe_float(rag_summary.get("supporting_fact_recall", 0.0), 0.0),
            "supporting_fact_precision": _safe_float(rag_summary.get("supporting_fact_precision", 0.0), 0.0),
            "supporting_fact_f1": _safe_float(rag_summary.get("supporting_fact_f1", 0.0), 0.0),
            "recall_at_5": _safe_float(rag_summary.get("supporting_fact_recall_at_5", 0.0), 0.0),
            "avg_context_tokens": _safe_float(rag_summary.get("prompt_tokens_avg", 0.0), 0.0),
            "retrieval_ms": _safe_float(rag_summary.get("retrieval_latency_ms", 0.0), 0.0),
            "generation_ms": _safe_float(rag_summary.get("generation_ms", rag_summary.get("generation_latency_ms", 0.0)), 0.0),
            "total_ms": _safe_float(rag_summary.get("total_latency_ms", 0.0), 0.0),
        },
        "evidence_survival": survival,
        "stage_timing_stats": timing_by_stage,
        "bottleneck_counts": dict(bottleneck_counter),
        "top_slow_queries": [
            {
                "query_id": str(r.get("query_id", "")),
                "total_retrieval_ms": _safe_float((r.get("stage_ms", {}) or {}).get("total_retrieval", 0.0), 0.0),
            }
            for r in slow_queries
        ],
        "top_graph_flow_slow_queries": [
            {
                "query_id": str(r.get("query_id", "")),
                "graph_flow_ms": _safe_float((r.get("stage_ms", {}) or {}).get("graph_flow", 0.0), 0.0),
            }
            for r in slow_graph_flow
        ],
        "largest_local_graph_queries": [
            {
                "query_id": str(r.get("query_id", "")),
                "num_local_graph_nodes": int(r.get("num_local_graph_nodes", 0) or 0),
                "num_local_graph_edges": int(r.get("num_local_graph_edges", 0) or 0),
            }
            for r in large_graph
        ],
        "candidate_to_selected_collapse_cases": [
            {
                "query_id": str(r.get("query_id", "")),
                "candidate_sf_recall_eval_only": _safe_float(r.get("candidate_sf_recall_eval_only", 0.0), 0.0),
                "selected_sf_recall_eval_only": _safe_float(r.get("selected_sf_recall_eval_only", 0.0), 0.0),
            }
            for r in collapse_cases
        ],
        "empty_candidate_queries": [str(r.get("query_id", "")) for r in empty_candidate[:20]],
        "empty_selected_queries": [str(r.get("query_id", "")) for r in empty_selected[:20]],
    }
    return summary


def _write_run_reports(run_dir: Path, summary: Dict[str, Any]) -> None:
    stage_stats = summary.get("stage_timing_stats", {}) or {}
    ordered_stage = [
        "query_embedding",
        "anchor_extraction",
        "semantic_anchor_retrieval",
        "local_graph_build",
        "graph_flow",
        "candidate_truncation",
        "feature_extraction",
        "marginal_selection",
        "rendering",
        "total_retrieval",
        "generation",
        "total_query",
    ]
    stage_rows = []
    for st in ordered_stage:
        s = stage_stats.get(st, {}) or {}
        stage_rows.append(
            [
                st,
                _fmt_ms(s.get("mean", 0.0)),
                _fmt_ms(s.get("p50", 0.0)),
                _fmt_ms(s.get("p90", 0.0)),
                _fmt_ms(s.get("p95", 0.0)),
                _fmt_ms(s.get("p99", 0.0)),
            ]
        )

    surv = summary.get("evidence_survival", {}) or {}
    surv_rows = [
        ["candidate", _fmt4(surv.get("candidate_sf_recall_eval_only", 0.0)), _fmt4(surv.get("candidate_sf_precision_eval_only", 0.0))],
        ["selected", _fmt4(surv.get("selected_sf_recall_eval_only", 0.0)), _fmt4(surv.get("selected_sf_precision_eval_only", 0.0))],
        ["rendered", _fmt4(surv.get("rendered_sf_recall_eval_only", 0.0)), _fmt4(surv.get("rendered_sf_precision_eval_only", 0.0))],
    ]

    metrics = summary.get("metrics", {}) or {}
    md = []
    md.append("# PHASE7 Summary")
    md.append("")
    md.append(f"- dataset: {summary.get('dataset', '')}")
    md.append(f"- method: {summary.get('method', '')}")
    md.append(f"- n_samples: {summary.get('n_samples', 0)}")
    md.append(f"- EM: {_fmt4(metrics.get('em', 0.0))}")
    md.append(f"- F1: {_fmt4(metrics.get('f1', 0.0))}")
    md.append(f"- SF_recall: {_fmt4(metrics.get('supporting_fact_recall', 0.0))}")
    md.append(f"- retrieval_ms: {_fmt_ms(metrics.get('retrieval_ms', 0.0))}")
    md.append(f"- total_ms: {_fmt_ms(metrics.get('total_ms', 0.0))}")
    md.append("")
    md.append("## Stage Timing")
    md.append(_table(["stage", "mean_ms", "p50", "p90", "p95", "p99"], stage_rows))
    md.append("")
    md.append("## Evidence Survival (Eval-only)")
    md.append(_table(["stage", "sf_recall", "sf_precision"], surv_rows))
    md.append("")
    md.append("## Bottleneck Flags")
    bcounts = summary.get("bottleneck_counts", {}) or {}
    for k, v in sorted(bcounts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        md.append(f"- {k}: {v}")
    md.append("")
    (run_dir / "phase7_summary.md").write_text("\n".join(md), encoding="utf-8")
    (run_dir / "phase7_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    bottleneck_md = []
    bottleneck_md.append("# PHASE7 Bottleneck Summary")
    bottleneck_md.append("")
    bottleneck_md.append("## Top 20 slowest queries by total retrieval")
    for row in list(summary.get("top_slow_queries", []) or []):
        bottleneck_md.append(
            f"- {row.get('query_id', '')}: {_fmt_ms(row.get('total_retrieval_ms', 0.0))} ms"
        )
    bottleneck_md.append("")
    bottleneck_md.append("## Top 20 slowest graph-flow queries")
    for row in list(summary.get("top_graph_flow_slow_queries", []) or []):
        bottleneck_md.append(f"- {row.get('query_id', '')}: {_fmt_ms(row.get('graph_flow_ms', 0.0))} ms")
    bottleneck_md.append("")
    bottleneck_md.append("## Top 20 largest local graphs")
    for row in list(summary.get("largest_local_graph_queries", []) or []):
        bottleneck_md.append(
            f"- {row.get('query_id', '')}: nodes={row.get('num_local_graph_nodes', 0)}, edges={row.get('num_local_graph_edges', 0)}"
        )
    bottleneck_md.append("")
    bottleneck_md.append("## Candidate-to-selected collapse cases")
    for row in list(summary.get("candidate_to_selected_collapse_cases", []) or []):
        bottleneck_md.append(
            f"- {row.get('query_id', '')}: candidate={_fmt4(row.get('candidate_sf_recall_eval_only', 0.0))}, selected={_fmt4(row.get('selected_sf_recall_eval_only', 0.0))}"
        )
    bottleneck_md.append("")
    bottleneck_md.append("## Empty candidate/selected")
    bottleneck_md.append(f"- empty_candidate_queries: {len(summary.get('empty_candidate_queries', []) or [])}")
    bottleneck_md.append(f"- empty_selected_queries: {len(summary.get('empty_selected_queries', []) or [])}")
    bottleneck_md.append("")
    (run_dir / "phase7_bottleneck_summary.md").write_text("\n".join(bottleneck_md), encoding="utf-8")


def _discover_run_dirs(output_root: Path) -> List[Path]:
    found = []
    for p in output_root.rglob("rag_summary.json"):
        found.append(p.parent)
    found = sorted(set(found))
    return found


def _aggregate(output_root: Path, run_summaries: List[Dict[str, Any]]) -> None:
    rows = []
    for s in run_summaries:
        m = s.get("metrics", {}) or {}
        ctx = _safe_float(m.get("avg_context_tokens", 0.0), 0.0)
        f1 = _safe_float(m.get("f1", 0.0), 0.0)
        f1_per_1k = (f1 * 1000.0 / ctx) if ctx > 0.0 else 0.0
        rows.append(
            {
                "dataset": str(s.get("dataset", "")),
                "run_dir": str(s.get("run_dir", "")),
                "n_samples": int(s.get("n_samples", 0)),
                "em": _safe_float(m.get("em", 0.0), 0.0),
                "f1": f1,
                "supporting_fact_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
                "supporting_fact_precision": _safe_float(m.get("supporting_fact_precision", 0.0), 0.0),
                "avg_context_tokens": ctx,
                "f1_per_1k_context_tokens": f1_per_1k,
                "retrieval_ms": _safe_float(m.get("retrieval_ms", 0.0), 0.0),
                "total_ms": _safe_float(m.get("total_ms", 0.0), 0.0),
            }
        )
    payload = {
        "method": "phase7_evidence_flow",
        "output_root": str(output_root.resolve()),
        "runs": rows,
    }
    (output_root / "phase7_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# PHASE7 Evidence Flow Summary")
    md.append("")
    md.append(_table(
        [
            "dataset",
            "n",
            "EM",
            "F1",
            "SF_recall",
            "SF_precision",
            "avg_context_tokens",
            "F1_per_1k",
            "retrieval_ms",
            "total_ms",
        ],
        [
            [
                str(r.get("dataset", "")),
                str(int(r.get("n_samples", 0))),
                _fmt4(r.get("em", 0.0)),
                _fmt4(r.get("f1", 0.0)),
                _fmt4(r.get("supporting_fact_recall", 0.0)),
                _fmt4(r.get("supporting_fact_precision", 0.0)),
                _fmt_ms(r.get("avg_context_tokens", 0.0)),
                _fmt4(r.get("f1_per_1k_context_tokens", 0.0)),
                _fmt_ms(r.get("retrieval_ms", 0.0)),
                _fmt_ms(r.get("total_ms", 0.0)),
            ]
            for r in rows
        ],
    ))
    md.append("")
    (output_root / "phase7_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase7 evidence-flow runs.")
    parser.add_argument("--output-root", type=str, default="outputs/phase7_evidence_flow")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--run-dir", action="append", default=[])
    args = parser.parse_args()

    output_root = Path(args.root or args.output_root).resolve()
    run_dirs = [Path(p).resolve() for p in (args.run_dir or [])]
    if not run_dirs:
        run_dirs = _discover_run_dirs(output_root)
    if not run_dirs:
        print(f"No run directories found under: {output_root}")
        return 1

    summaries = []
    for run_dir in run_dirs:
        summary = _build_run_summary(run_dir)
        _write_run_reports(run_dir, summary)
        summaries.append(summary)
        print(f"[phase7-summary] wrote run summary: {run_dir}")

    _aggregate(output_root, summaries)
    print(f"[phase7-summary] wrote aggregate summary: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
