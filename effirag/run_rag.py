import argparse
from dataclasses import asdict
from pathlib import Path

from .config import RagConfig, apply_cli_overrides, dataclass_from_dict
from .efficiency import Timer, gpu_peak_mb, process_rss_mb, reset_gpu_peak
from .metrics import supporting_fact_precision, supporting_fact_recall
from .qa_metrics import exact_match_score, token_f1_score
from .registry import get_dataset_loader, get_generator, get_method, register_defaults
from .render import render_context
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
    parser.add_argument("--measure-gpu-peak", type=str, default=None)
    parser.add_argument("--measure-cpu-ram", type=str, default=None)

    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--ppr-alpha", type=float, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--edge-drop-prob", type=float, default=None)
    return parser


def execute_rag_experiment(cfg):
    register_defaults()

    loader = get_dataset_loader(cfg.dataset)
    method_fn = get_method(cfg.method)
    generator_fn = get_generator(cfg.generator)

    samples = loader(split=cfg.split, limit=cfg.limit, data_path=cfg.data_path)

    rows = []
    cfg_values = asdict(cfg)
    run_stamp = timestamp_for_filename()
    run_iso = timestamp_iso_utc()
    for sample in samples:
        total_timer = Timer()
        total_timer.start()

        if cfg.measure_gpu_peak:
            reset_gpu_peak()

        cpu_peak = process_rss_mb() if cfg.measure_cpu_ram else 0.0

        retrieval = method_fn(sample, cfg)
        rendered = render_context(sample, retrieval, cfg.max_context_sentences)

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
                "rendered": asdict(rendered),
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
        "em": mean_or_zero([r["metrics"]["em"] for r in rows]),
        "f1": mean_or_zero([r["metrics"]["f1"] for r in rows]),
        "retrieval_latency_ms": mean_or_zero([r["efficiency"]["retrieval_latency_ms"] for r in rows]),
        "total_latency_ms": mean_or_zero([r["efficiency"]["total_latency_ms"] for r in rows]),
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
    }

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
                "EM",
                "F1",
                "retrieval_ms",
                "total_ms",
            ],
            [
                [
                    summary["run_timestamp"],
                    summary["method"],
                    summary["generator"],
                    int(summary["n_samples"]),
                    "%.4f" % summary["supporting_fact_recall"],
                    "%.4f" % summary["em"],
                    "%.4f" % summary["f1"],
                    "%.2f" % summary["retrieval_latency_ms"],
                    "%.2f" % summary["total_latency_ms"],
                ]
            ],
        )
    )


if __name__ == "__main__":
    main()
