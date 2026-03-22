import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import RagConfig, apply_cli_overrides, dataclass_from_dict
from .efficiency import Timer, gpu_peak_mb, process_rss_mb, reset_gpu_peak
from .metrics import (
    DEFAULT_RECALL_KS,
    supporting_fact_ids,
    supporting_fact_precision,
    supporting_fact_recall,
    supporting_fact_recall_at_ks,
)
from .qa_metrics import exact_match_score, token_f1_score
from .registry import get_dataset_loader, get_generator, get_method, register_defaults
from .render import render_context
from .types import AnchorResult, RetrievalResult
from .utils import (
    append_jsonl,
    load_yaml,
    markdown_table,
    mean_or_zero,
    timestamp_for_filename,
    timestamp_iso_utc,
    write_json,
    write_jsonl,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _reconstruct_retrieval(payload: dict) -> RetrievalResult:
    anchor_results = []
    for item in payload.get("anchor_results", []) or []:
        anchor_results.append(
            AnchorResult(
                anchor=item.get("anchor", ""),
                scores=item.get("scores", {}) or {},
                top_candidates=item.get("top_candidates", []) or [],
                sample_index=int(item.get("sample_index", 0)),
                metadata=item.get("metadata", {}) or {},
            )
        )

    return RetrievalResult(
        sample_id=str(payload.get("sample_id", "")),
        method=str(payload.get("method", "")),
        anchors=[str(x) for x in (payload.get("anchors", []) or [])],
        seeds=[str(x) for x in (payload.get("seeds", []) or [])],
        selected_nodes=[str(x) for x in (payload.get("selected_nodes", []) or [])],
        selected_sentence_ids=[str(x) for x in (payload.get("selected_sentence_ids", []) or [])],
        selected_sentences=[str(x) for x in (payload.get("selected_sentences", []) or [])],
        corridors=payload.get("corridors", []) or [],
        anchor_results=anchor_results,
        diagnostics=payload.get("diagnostics", {}) or {},
        latency_ms=float(payload.get("latency_ms", 0.0)),
    )


def _load_precomputed_retrieval(path: str):
    by_sample_id = {}
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Precomputed retrieval file not found: {path}")

    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            retrieval_payload = row.get("retrieval", {}) or {}
            if not sample_id:
                continue
            by_sample_id[sample_id] = _reconstruct_retrieval(retrieval_payload)
    return by_sample_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end EffiRAG QA experiments.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", type=str, default=None, choices=["effirag", "naive_graphrag"])
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--samples-per-anchor", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--candidate-top-t", type=int, default=None)
    parser.add_argument("--seed-k", type=int, default=None)
    parser.add_argument("--pair-top-lp", type=int, default=None)
    parser.add_argument("--corridor-top-bc", type=int, default=None)
    parser.add_argument("--trim-on", type=str, default=None)
    parser.add_argument("--trim-rho", type=float, default=None)

    parser.add_argument("--run-qa", type=str, default=None)
    parser.add_argument("--generator", type=str, default=None, choices=["heuristic", "oracle", "hf"])
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-context-sentences", type=int, default=None)
    parser.add_argument("--render-mode", type=str, default=None, choices=["flat", "corridor", "corridor_aware_flat"])
    parser.add_argument("--max-corridors-in-context", type=int, default=None)
    parser.add_argument("--max-main-sentences-per-corridor", type=int, default=None)
    parser.add_argument("--max-support-per-corridor", type=int, default=None)
    parser.add_argument("--max-total-sentences", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma-main", type=float, default=None)
    parser.add_argument("--delta-support", type=float, default=None)
    parser.add_argument("--eta-connector", type=float, default=None)
    parser.add_argument("--zeta-query", type=float, default=None)
    parser.add_argument("--xi-locality", type=float, default=None)
    parser.add_argument("--lambda-redundancy", type=float, default=None)
    parser.add_argument("--top-corridors", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--reserve-top-corridor", type=str, default=None)
    parser.add_argument("--order-strategy", type=str, default=None, choices=["score", "retrieval", "corridor_rank"])
    parser.add_argument("--measure-gpu-peak", type=str, default=None)
    parser.add_argument("--measure-cpu-ram", type=str, default=None)

    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--ppr-alpha", type=float, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--edge-drop-prob", type=float, default=None)
    return parser


def execute_rag_experiment(cfg, show_progress: bool = True, precomputed_retrieval_path: str = None):
    register_defaults()

    loader = get_dataset_loader(cfg.dataset)
    method_fn = get_method(cfg.method)
    generator_fn = get_generator(cfg.generator)

    samples = loader(split=cfg.split, limit=cfg.limit, data_path=cfg.data_path)
    retrieval_cache = {}
    if precomputed_retrieval_path:
        retrieval_cache = _load_precomputed_retrieval(precomputed_retrieval_path)

    rows = []
    cfg_values = asdict(cfg)
    run_stamp = timestamp_for_filename()
    run_iso = timestamp_iso_utc()
    for sample in tqdm(
        samples,
        total=len(samples),
        desc=f"RAG[{cfg.method}/{cfg.generator}]",
        leave=False,
        disable=not show_progress,
    ):
        total_timer = Timer()
        total_timer.start()

        if cfg.measure_gpu_peak:
            reset_gpu_peak()

        cpu_peak = process_rss_mb() if cfg.measure_cpu_ram else 0.0

        retrieval = retrieval_cache.get(sample.qid)
        if retrieval is None:
            retrieval = method_fn(sample, cfg)
        render_mode = cfg.render_mode or ("corridor_aware_flat" if cfg.method == "effirag" else "flat")
        rendered = render_context(
            sample,
            retrieval,
            max_context_sentences=cfg.max_context_sentences,
            render_mode=render_mode,
            max_corridors_in_context=cfg.max_corridors_in_context,
            max_main_sentences_per_corridor=cfg.max_main_sentences_per_corridor,
            max_support_per_corridor=cfg.max_support_per_corridor,
            max_total_sentences=cfg.max_total_sentences,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma_main=cfg.gamma_main,
            delta_support=cfg.delta_support,
            eta_connector=cfg.eta_connector,
            zeta_query=cfg.zeta_query,
            xi_locality=cfg.xi_locality,
            lambda_redundancy=cfg.lambda_redundancy,
            top_corridors=cfg.top_corridors,
            max_sentences=cfg.max_sentences,
            reserve_top_corridor=cfg.reserve_top_corridor,
            order_strategy=cfg.order_strategy,
        )

        if cfg.measure_cpu_ram:
            cpu_peak = max(cpu_peak, process_rss_mb())

        generation = None
        em = 0.0
        f1 = 0.0
        if cfg.run_qa:
            generation = generator_fn(sample, rendered, cfg.model_name)
            em = exact_match_score(generation.prediction, sample.answer)
            f1 = token_f1_score(generation.prediction, sample.answer)

        if cfg.measure_cpu_ram:
            cpu_peak = max(cpu_peak, process_rss_mb())

        recall = supporting_fact_recall(sample, retrieval)
        precision = supporting_fact_precision(sample, retrieval)
        recall_at_k = supporting_fact_recall_at_ks(sample, retrieval, ks=DEFAULT_RECALL_KS)
        gold_ids = supporting_fact_ids(sample)
        rendered_ids = set(rendered.sentence_ids)
        rendered_recall = float(len(gold_ids.intersection(rendered_ids)) / len(gold_ids)) if gold_ids else 0.0
        rendered_precision = float(len(gold_ids.intersection(rendered_ids)) / len(rendered_ids)) if rendered_ids else 0.0

        efficiency = {
            "retrieval_latency_ms": retrieval.latency_ms,
            "generation_latency_ms": generation.latency_ms if generation else 0.0,
            "total_latency_ms": total_timer.elapsed_ms(),
        }
        if cfg.measure_gpu_peak:
            efficiency["gpu_peak_mb"] = gpu_peak_mb()
        if cfg.measure_cpu_ram:
            efficiency["cpu_ram_peak_mb"] = cpu_peak

        metrics = {
            "supporting_fact_recall": recall,
            "supporting_fact_precision": precision,
            "recall_at_k": recall_at_k,
            "rendered_supporting_fact_recall": rendered_recall,
            "rendered_supporting_fact_precision": rendered_precision,
            "em": em,
            "f1": f1,
        }

        rows.append(
            {
                "sample_id": sample.qid,
                "question": sample.question,
                "answer": sample.answer,
                "prediction": generation.prediction if generation else "",
                "method": retrieval.method,
                "generator": cfg.generator,
                "metrics": metrics,
                "efficiency": efficiency,
                "retrieval": asdict(retrieval),
                "retrieval_selected_sentence_ids": list(retrieval.selected_sentence_ids),
                "rendered": asdict(rendered),
                "rendered_sentence_ids": list(rendered.sentence_ids),
                "rendered_corridor_ids": list(rendered.rendered_corridor_ids),
                "rendering": {
                    "render_mode": rendered.render_mode,
                    "truncated_corridors": rendered.truncated_corridor_count,
                    "truncated_sentences": rendered.truncated_sentence_count,
                },
                "generation": asdict(generation) if generation else None,
            }
        )

    for row in rows:
        row["run_timestamp"] = run_stamp

    summary = {
        "n_samples": float(len(rows)),
        "dataset": cfg.dataset,
        "method": cfg.method,
        "generator": cfg.generator,
        "supporting_fact_recall": mean_or_zero([r["metrics"]["supporting_fact_recall"] for r in rows]),
        "supporting_fact_precision": mean_or_zero([r["metrics"]["supporting_fact_precision"] for r in rows]),
        "rendered_supporting_fact_recall": mean_or_zero(
            [r["metrics"].get("rendered_supporting_fact_recall", 0.0) for r in rows]
        ),
        "rendered_supporting_fact_precision": mean_or_zero(
            [r["metrics"].get("rendered_supporting_fact_precision", 0.0) for r in rows]
        ),
        "em": mean_or_zero([r["metrics"]["em"] for r in rows]),
        "f1": mean_or_zero([r["metrics"]["f1"] for r in rows]),
        "retrieval_latency_ms": mean_or_zero([r["efficiency"]["retrieval_latency_ms"] for r in rows]),
        "total_latency_ms": mean_or_zero([r["efficiency"]["total_latency_ms"] for r in rows]),
        "truncated_corridors_avg": mean_or_zero([r.get("rendering", {}).get("truncated_corridors", 0.0) for r in rows]),
        "truncated_sentences_avg": mean_or_zero([r.get("rendering", {}).get("truncated_sentences", 0.0) for r in rows]),
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
    }

    recall_at_k_summary = {}
    for k in DEFAULT_RECALL_KS:
        key = str(int(k))
        vals = [float((r["metrics"].get("recall_at_k", {}) or {}).get(key, 0.0)) for r in rows]
        agg = mean_or_zero(vals)
        recall_at_k_summary[key] = agg
        summary[f"supporting_fact_recall_at_{key}"] = agg
    summary["supporting_fact_recall_at_k"] = recall_at_k_summary

    if cfg.run_qa:
        summary["generation_latency_ms"] = mean_or_zero([r["efficiency"]["generation_latency_ms"] for r in rows])

    if cfg.measure_gpu_peak:
        summary["gpu_peak_mb"] = mean_or_zero([r["efficiency"].get("gpu_peak_mb", 0.0) for r in rows])
    if cfg.measure_cpu_ram:
        summary["cpu_ram_peak_mb"] = mean_or_zero([r["efficiency"].get("cpu_ram_peak_mb", 0.0) for r in rows])

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "rag_query_results.jsonl"
    summary_path = out_dir / "rag_summary.json"

    write_jsonl(query_path, rows)
    write_json(summary_path, summary)

    logs_dir = out_dir / "logs"
    cfg_log = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "task": "rag",
        "config": cfg_values,
    }
    result_log = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "task": "rag",
        "summary": summary,
    }
    write_json(logs_dir / ("config_%s.json" % run_stamp), cfg_log)
    write_json(logs_dir / ("result_%s.json" % run_stamp), result_log)
    append_jsonl(logs_dir / "config_history.jsonl", cfg_log)
    append_jsonl(logs_dir / "result_history.jsonl", result_log)

    return rows, summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base_config = load_yaml(args.config) if args.config else {}
    merged = apply_cli_overrides(base_config, args)
    cfg = dataclass_from_dict(RagConfig, merged)

    _, summary = execute_rag_experiment(cfg)

    print("RAG run complete")
    print(
        markdown_table(
            [
                "run_timestamp",
                "method",
                "generator",
                "samples",
                "sf_recall",
                "Recall@1",
                "Recall@5",
                "Recall@20",
                "rendered_sf_recall",
                "EM",
                "F1",
                "retrieval_ms",
                "total_ms",
                "trunc_corr_avg",
                "trunc_sent_avg",
            ],
            [
                [
                    summary["run_timestamp"],
                    summary["method"],
                    summary["generator"],
                    int(summary["n_samples"]),
                    "%.4f" % summary["supporting_fact_recall"],
                    "%.4f" % summary.get("supporting_fact_recall_at_1", 0.0),
                    "%.4f" % summary.get("supporting_fact_recall_at_5", 0.0),
                    "%.4f" % summary.get("supporting_fact_recall_at_20", 0.0),
                    "%.4f" % summary.get("rendered_supporting_fact_recall", 0.0),
                    "%.4f" % summary["em"],
                    "%.4f" % summary["f1"],
                    "%.2f" % summary["retrieval_latency_ms"],
                    "%.2f" % summary["total_latency_ms"],
                    "%.2f" % summary.get("truncated_corridors_avg", 0.0),
                    "%.2f" % summary.get("truncated_sentences_avg", 0.0),
                ]
            ],
        )
    )


if __name__ == "__main__":
    main()
