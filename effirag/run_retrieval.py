import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .config import RetrievalConfig, apply_cli_overrides, dataclass_from_dict
from .metrics import aggregate_retrieval_metrics, supporting_fact_precision, supporting_fact_recall
from .registry import get_dataset_loader, get_method, register_defaults
from .utils import (
    append_jsonl,
    load_yaml,
    markdown_table,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval-only EffiRAG experiments.")
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
    parser.add_argument("--generator", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-context-sentences", type=int, default=None)
    parser.add_argument("--measure-gpu-peak", type=str, default=None)
    parser.add_argument("--measure-cpu-ram", type=str, default=None)

    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--ppr-alpha", type=float, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--edge-drop-prob", type=float, default=None)
    return parser


def _worker(sample, cfg_values: dict):
    register_defaults()
    cfg = dataclass_from_dict(RetrievalConfig, cfg_values)
    method_fn = get_method(cfg.method)
    retrieval = method_fn(sample, cfg)

    recall = supporting_fact_recall(sample, retrieval)
    precision = supporting_fact_precision(sample, retrieval)

    return {
        "sample_id": sample.qid,
        "question": sample.question,
        "method": retrieval.method,
        "supporting_fact_recall": recall,
        "supporting_fact_precision": precision,
        "retrieval_latency_ms": retrieval.latency_ms,
        "selected_sentence_count": len(retrieval.selected_sentence_ids),
        "retrieval": asdict(retrieval),
    }


def execute_retrieval_experiment(cfg, show_progress: bool = True):
    register_defaults()

    loader = get_dataset_loader(cfg.dataset)
    method_fn = get_method(cfg.method)
    samples = loader(split=cfg.split, limit=cfg.limit, data_path=cfg.data_path)

    rows = []
    cfg_values = asdict(cfg)
    run_stamp = timestamp_for_filename()
    run_iso = timestamp_iso_utc()

    if cfg.num_workers > 1 and len(samples) > 1:
        with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
            futures = [executor.submit(_worker, sample, cfg_values) for sample in samples]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Retrieval[{cfg.method}]",
                leave=False,
                disable=not show_progress,
            ):
                rows.append(fut.result())
        rows.sort(key=lambda x: x["sample_id"])
    else:
        for sample in tqdm(
            samples,
            total=len(samples),
            desc=f"Retrieval[{cfg.method}]",
            leave=False,
            disable=not show_progress,
        ):
            retrieval = method_fn(sample, cfg)
            recall = supporting_fact_recall(sample, retrieval)
            precision = supporting_fact_precision(sample, retrieval)
            rows.append(
                {
                    "sample_id": sample.qid,
                    "question": sample.question,
                    "method": retrieval.method,
                    "supporting_fact_recall": recall,
                    "supporting_fact_precision": precision,
                    "retrieval_latency_ms": retrieval.latency_ms,
                    "selected_sentence_count": len(retrieval.selected_sentence_ids),
                    "retrieval": asdict(retrieval),
                }
            )

    for row in rows:
        row["run_timestamp"] = run_stamp

    summary = aggregate_retrieval_metrics(rows)
    summary["method"] = cfg.method
    summary["dataset"] = cfg.dataset
    summary["run_timestamp"] = run_stamp
    summary["run_timestamp_utc"] = run_iso

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "retrieval_query_results.jsonl"
    summary_path = out_dir / "retrieval_summary.json"
    write_jsonl(query_path, rows)
    write_json(summary_path, summary)

    logs_dir = out_dir / "logs"
    cfg_log = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "task": "retrieval",
        "config": cfg_values,
    }
    result_log = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "task": "retrieval",
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
    cfg = dataclass_from_dict(RetrievalConfig, merged)

    _, summary = execute_retrieval_experiment(cfg)

    print("Retrieval run complete")
    print(
        markdown_table(
            [
                "run_timestamp",
                "method",
                "samples",
                "supporting_fact_recall",
                "supporting_fact_precision",
                "retrieval_latency_ms",
            ],
            [
                [
                    summary["run_timestamp"],
                    summary["method"],
                    int(summary["n_samples"]),
                    "%.4f" % summary["supporting_fact_recall"],
                    "%.4f" % summary["supporting_fact_precision"],
                    "%.2f" % summary["retrieval_latency_ms"],
                ]
            ],
        )
    )


if __name__ == "__main__":
    main()
