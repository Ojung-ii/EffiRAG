import argparse
from dataclasses import replace
from pathlib import Path

from .config import RetrievalConfig, apply_cli_overrides, dataclass_from_dict
from .run_retrieval import execute_retrieval_experiment
from .utils import append_jsonl, load_yaml, markdown_table, timestamp_for_filename, timestamp_iso_utc, write_json

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EffiRAG retrieval ablations.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--global-corpus-path", type=str, default=None)
    parser.add_argument("--graph-cache-dir", type=str, default=None)
    parser.add_argument("--force-rebuild-graph-index", type=str, default=None)
    parser.add_argument("--prebuilt-igraph-path", type=str, default=None)
    parser.add_argument("--prebuilt-igraph-format", type=str, default=None)
    parser.add_argument("--prebuilt-entity-token-limit", type=int, default=None)
    parser.add_argument("--openie-mode", type=str, default=None, choices=["llm", "lexical"])
    parser.add_argument("--openie-model-name", type=str, default=None)
    parser.add_argument("--openie-text-max-chars", type=int, default=None)
    parser.add_argument("--openie-max-new-tokens", type=int, default=None)
    parser.add_argument("--openie-local-files-only", type=str, default=None)
    parser.add_argument("--openie-retry-attempts", type=int, default=None)
    parser.add_argument("--openie-retry-backoff-sec", type=float, default=None)
    parser.add_argument("--openie-error-sample-limit", type=int, default=None)
    parser.add_argument("--openie-api-base-url", type=str, default=None)
    parser.add_argument("--openie-api-key", type=str, default=None)
    parser.add_argument("--openie-api-timeout-sec", type=float, default=None)
    parser.add_argument("--openie-parallel-workers", type=int, default=None)
    parser.add_argument("--openie-log-every", type=int, default=None)
    parser.add_argument("--embedding-enabled", type=str, default=None)
    parser.add_argument("--embedding-model-name", type=str, default=None)
    parser.add_argument("--embedding-weight", type=float, default=None)
    parser.add_argument("--embedding-rerank-topn", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-max-length", type=int, default=None)
    parser.add_argument("--embedding-text-max-chars", type=int, default=None)
    parser.add_argument("--semantic-topn", type=int, default=None)
    parser.add_argument("--semantic-candidate-union", type=str, default=None)
    parser.add_argument("--semantic-scan-batch-size", type=int, default=None)
    parser.add_argument("--run-score-semantic-weight", type=float, default=None)
    parser.add_argument("--run-score-anchor-weight", type=float, default=None)
    parser.add_argument("--run-score-structure-weight", type=float, default=None)
    parser.add_argument("--run-score-bridge-weight", type=float, default=None)
    parser.add_argument("--run-score-redundancy-weight", type=float, default=None)
    parser.add_argument("--seed-score-semantic-weight", type=float, default=None)
    parser.add_argument("--seed-score-graph-weight", type=float, default=None)
    parser.add_argument("--seed-score-anchor-weight", type=float, default=None)

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
    parser.add_argument("--ppr-engine", type=str, default=None, choices=["auto", "power", "mc"])
    parser.add_argument("--ppr-power-max-iter", type=int, default=None)
    parser.add_argument("--ppr-power-tol", type=float, default=None)
    parser.add_argument("--ppr-min-score", type=float, default=None)
    parser.add_argument("--ppr-mc-walks", type=int, default=None)
    parser.add_argument("--ppr-mc-max-steps", type=int, default=None)
    parser.add_argument("--ppr-parallel-workers", type=int, default=None)
    parser.add_argument("--ppr-subgraph-enable", type=str, default=None)
    parser.add_argument("--ppr-subgraph-hops", type=int, default=None)
    parser.add_argument("--ppr-subgraph-max-nodes", type=int, default=None)
    parser.add_argument("--anchor-diag-topn", type=int, default=None)
    parser.add_argument("--anchor-diag-store-full-scores", type=str, default=None)
    return parser


def execute_ablation(cfg):
    run_stamp = timestamp_for_filename()
    run_iso = timestamp_iso_utc()
    cfg_values = cfg.__dict__.copy()

    variants = [
        (
            "effirag_trim",
            replace(
                cfg,
                method="effirag",
                trim_on=True,
                output_dir=str(Path(cfg.output_dir) / "effirag_trim"),
                timestamp_output=False,
            ),
        ),
        (
            "effirag_no_trim",
            replace(
                cfg,
                method="effirag",
                trim_on=False,
                output_dir=str(Path(cfg.output_dir) / "effirag_no_trim"),
                timestamp_output=False,
            ),
        ),
        (
            "naive_graphrag",
            replace(
                cfg,
                method="naive_graphrag",
                trim_on=False,
                output_dir=str(Path(cfg.output_dir) / "naive_graphrag"),
                timestamp_output=False,
            ),
        ),
    ]

    summaries = []
    for name, run_cfg in tqdm(
        variants,
        total=len(variants),
        desc="Ablation variants",
        unit="variant",
        leave=False,
    ):
        _, summary = execute_retrieval_experiment(run_cfg)
        summary["variant"] = name
        summary["ablation_run_timestamp"] = run_stamp
        summaries.append(summary)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "runs": summaries,
    }
    write_json(out_dir / "ablation_summary.json", payload)

    logs_dir = out_dir / "logs"
    cfg_log = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "task": "ablation",
        "config": cfg_values,
    }
    result_log = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "task": "ablation",
        "summary": payload,
    }
    write_json(logs_dir / ("config_%s.json" % run_stamp), cfg_log)
    write_json(logs_dir / ("result_%s.json" % run_stamp), result_log)
    append_jsonl(logs_dir / "config_history.jsonl", cfg_log)
    append_jsonl(logs_dir / "result_history.jsonl", result_log)
    return summaries


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base_config = load_yaml(args.config) if args.config else {}
    merged = apply_cli_overrides(base_config, args)
    cfg = dataclass_from_dict(RetrievalConfig, merged)

    summaries = execute_ablation(cfg)
    print("Ablation run complete")
    print(
        markdown_table(
            [
                "variant",
                "method",
                "samples",
                "sf_recall",
                "sf_precision",
                "retrieval_ms",
                "run_timestamp",
            ],
            [
                [
                    row["variant"],
                    row.get("method", ""),
                    int(row.get("n_samples", 0.0)),
                    "%.4f" % row.get("supporting_fact_recall", 0.0),
                    "%.4f" % row.get("supporting_fact_precision", 0.0),
                    "%.2f" % row.get("retrieval_latency_ms", 0.0),
                    row.get("ablation_run_timestamp", ""),
                ]
                for row in summaries
            ],
        )
    )


if __name__ == "__main__":
    main()
