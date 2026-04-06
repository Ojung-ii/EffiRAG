import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .config import RetrievalConfig, apply_cli_overrides, dataclass_from_dict
from .metrics import (
    DEFAULT_RECALL_KS,
    aggregate_retrieval_metrics,
    build_support_fact_debug_payload,
    supporting_fact_precision,
    supporting_fact_recall,
    supporting_fact_recall_at_ks,
)
from .registry import get_dataset_loader, get_method, register_defaults
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval-only EffiRAG experiments.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", type=str, default=None, choices=["effirag", "naive_graphrag"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--timestamp-output", type=str, default=None)
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
    parser.add_argument("--sentence-rerank-enabled", type=str, default=None)
    parser.add_argument("--top1-correction-enabled", type=str, default=None)
    parser.add_argument("--top1-correction-topk", type=int, default=None)
    parser.add_argument("--top1-correction-corridor-weight-base", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-anchor", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-support", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-bridge", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-semantic", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-redundancy", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-base", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-corridor", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-main", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-support", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-query", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-locality", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-redundancy", type=float, default=None)
    parser.add_argument("--embedding-model-name", type=str, default=None)
    parser.add_argument("--embedding-weight", type=float, default=None)
    parser.add_argument("--embedding-rerank-topn", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-max-length", type=int, default=None)
    parser.add_argument("--embedding-text-max-chars", type=int, default=None)
    parser.add_argument("--semantic-topn-entity", type=int, default=None)
    parser.add_argument("--semantic-topn-chunk", type=int, default=None)
    parser.add_argument("--graph-reserve-topn", type=int, default=None)
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
    parser.add_argument("--phase1-parallel-ppr", type=str, default=None)
    parser.add_argument("--phase1-run-shortlist-topk", type=int, default=None)
    parser.add_argument("--phase1-run-preshortlist-topm", type=int, default=None)
    parser.add_argument("--phase1-full-run-score-topk", type=int, default=None)
    parser.add_argument("--pair-shortlist-topb", type=int, default=None)
    parser.add_argument("--phase2-refine-mode", type=str, default=None)
    parser.add_argument("--phase2-bidirectional-full-ppr", type=str, default=None)
    parser.add_argument("--reuse-semantic-scores-in-final", type=str, default=None)
    parser.add_argument("--oracle-support-injection-enabled", type=str, default=None)
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
    parser.add_argument("--sf-debug-sample-limit", type=int, default=None)
    parser.add_argument("--sf-debug-output", type=str, default=None)
    return parser


def _worker(sample, cfg_values: dict):
    register_defaults()
    cfg = dataclass_from_dict(RetrievalConfig, cfg_values)
    method_fn = get_method(cfg.method)
    retrieval = method_fn(sample, cfg)

    recall = supporting_fact_recall(sample, retrieval)
    precision = supporting_fact_precision(sample, retrieval)
    recall_at_k = supporting_fact_recall_at_ks(sample, retrieval, ks=DEFAULT_RECALL_KS)

    return {
        "sample_id": sample.qid,
        "question": sample.question,
        "method": retrieval.method,
        "supporting_fact_recall": recall,
        "supporting_fact_precision": precision,
        "recall_at_k": recall_at_k,
        "retrieval_latency_ms": retrieval.latency_ms,
        "selected_sentence_count": len(retrieval.selected_sentence_ids),
        "retrieval": asdict(retrieval),
    }


def _extract_global_index_diag(row: dict) -> dict:
    retrieval = (row or {}).get("retrieval", {}) or {}
    diagnostics = retrieval.get("diagnostics", {}) or {}
    payload = diagnostics.get("global_index", {}) or {}
    if isinstance(payload, dict):
        return payload
    return {}


def execute_retrieval_experiment(cfg, show_progress: bool = True):
    register_defaults()

    loader = get_dataset_loader(cfg.dataset)
    method_fn = get_method(cfg.method)
    samples = loader(split=cfg.split, limit=cfg.limit, data_path=cfg.data_path)

    rows = []
    sf_debug_rows = []
    sf_debug_limit = max(0, int(getattr(cfg, "sf_debug_sample_limit", 0) or 0))
    debug_enabled = bool(sf_debug_limit > 0)
    cfg_values = asdict(cfg)
    run_stamp = timestamp_for_filename()
    run_iso = timestamp_iso_utc()
    effective_workers = int(cfg.num_workers)
    if debug_enabled and effective_workers > 1:
        # Keep debug rows aligned with sample objects and avoid worker-side payload bloat.
        effective_workers = 1

    if effective_workers > 1 and len(samples) > 1:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
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
            recall_at_k = supporting_fact_recall_at_ks(sample, retrieval, ks=DEFAULT_RECALL_KS)
            rows.append(
                {
                    "sample_id": sample.qid,
                    "question": sample.question,
                    "method": retrieval.method,
                    "supporting_fact_recall": recall,
                    "supporting_fact_precision": precision,
                    "recall_at_k": recall_at_k,
                    "retrieval_latency_ms": retrieval.latency_ms,
                    "selected_sentence_count": len(retrieval.selected_sentence_ids),
                    "retrieval": asdict(retrieval),
                }
            )
            if debug_enabled and len(sf_debug_rows) < sf_debug_limit:
                sf_debug_rows.append(
                    build_support_fact_debug_payload(
                        sample=sample,
                        retrieval=retrieval,
                        rendered=None,
                    )
                )

    for row in rows:
        row["run_timestamp"] = run_stamp

    summary = aggregate_retrieval_metrics(rows)
    summary["method"] = cfg.method
    summary["dataset"] = cfg.dataset
    summary["global_corpus_path"] = str(cfg.global_corpus_path or "")
    summary["graph_cache_dir"] = str(cfg.graph_cache_dir or "")
    summary["force_rebuild_graph_index"] = bool(cfg.force_rebuild_graph_index)
    summary["prebuilt_igraph_path"] = str(cfg.prebuilt_igraph_path or "")
    summary["prebuilt_igraph_format"] = str(cfg.prebuilt_igraph_format or "")
    summary["prebuilt_entity_token_limit"] = int(cfg.prebuilt_entity_token_limit)
    summary["openie_mode"] = str(cfg.openie_mode or "")
    summary["openie_model_name"] = str(cfg.openie_model_name or "")
    summary["openie_text_max_chars"] = int(cfg.openie_text_max_chars)
    summary["openie_max_new_tokens"] = int(cfg.openie_max_new_tokens)
    summary["openie_local_files_only"] = bool(cfg.openie_local_files_only)
    summary["openie_retry_attempts"] = int(cfg.openie_retry_attempts)
    summary["openie_retry_backoff_sec"] = float(cfg.openie_retry_backoff_sec)
    summary["openie_error_sample_limit"] = int(cfg.openie_error_sample_limit)
    summary["openie_api_base_url"] = str(cfg.openie_api_base_url or "")
    summary["openie_api_timeout_sec"] = float(cfg.openie_api_timeout_sec)
    summary["openie_parallel_workers"] = int(cfg.openie_parallel_workers)
    summary["openie_log_every"] = int(cfg.openie_log_every)
    summary["embedding_enabled"] = bool(cfg.embedding_enabled)
    summary["embedding_model_name"] = str(cfg.embedding_model_name or "")
    summary["embedding_weight"] = float(cfg.embedding_weight)
    summary["embedding_rerank_topn"] = int(cfg.embedding_rerank_topn)
    summary["embedding_batch_size"] = int(cfg.embedding_batch_size)
    summary["embedding_max_length"] = int(cfg.embedding_max_length)
    summary["embedding_text_max_chars"] = int(cfg.embedding_text_max_chars)
    summary["semantic_topn_entity"] = int(cfg.semantic_topn_entity)
    summary["semantic_topn_chunk"] = int(cfg.semantic_topn_chunk)
    summary["graph_reserve_topn"] = int(cfg.graph_reserve_topn)
    summary["semantic_topn"] = int(cfg.semantic_topn)
    summary["semantic_candidate_union"] = bool(cfg.semantic_candidate_union)
    summary["semantic_scan_batch_size"] = int(cfg.semantic_scan_batch_size)
    summary["run_score_semantic_weight"] = float(cfg.run_score_semantic_weight)
    summary["run_score_anchor_weight"] = float(cfg.run_score_anchor_weight)
    summary["run_score_structure_weight"] = float(cfg.run_score_structure_weight)
    summary["run_score_bridge_weight"] = float(cfg.run_score_bridge_weight)
    summary["run_score_redundancy_weight"] = float(cfg.run_score_redundancy_weight)
    summary["seed_score_semantic_weight"] = float(cfg.seed_score_semantic_weight)
    summary["seed_score_graph_weight"] = float(cfg.seed_score_graph_weight)
    summary["seed_score_anchor_weight"] = float(cfg.seed_score_anchor_weight)
    summary["ppr_engine"] = str(cfg.ppr_engine)
    summary["ppr_power_max_iter"] = int(cfg.ppr_power_max_iter)
    summary["ppr_power_tol"] = float(cfg.ppr_power_tol)
    summary["ppr_min_score"] = float(cfg.ppr_min_score)
    summary["ppr_mc_walks"] = int(cfg.ppr_mc_walks)
    summary["ppr_mc_max_steps"] = int(cfg.ppr_mc_max_steps)
    summary["ppr_parallel_workers"] = int(cfg.ppr_parallel_workers)
    summary["ppr_subgraph_enable"] = bool(cfg.ppr_subgraph_enable)
    summary["ppr_subgraph_hops"] = int(cfg.ppr_subgraph_hops)
    summary["ppr_subgraph_max_nodes"] = int(cfg.ppr_subgraph_max_nodes)
    summary["phase1_parallel_ppr"] = bool(cfg.phase1_parallel_ppr)
    summary["phase1_run_shortlist_topk"] = int(cfg.phase1_run_shortlist_topk)
    summary["pair_shortlist_topb"] = int(cfg.pair_shortlist_topb)
    summary["phase2_refine_mode"] = str(cfg.phase2_refine_mode)
    summary["phase2_bidirectional_full_ppr"] = bool(cfg.phase2_bidirectional_full_ppr)
    summary["reuse_semantic_scores_in_final"] = bool(cfg.reuse_semantic_scores_in_final)
    summary["anchor_diag_topn"] = int(cfg.anchor_diag_topn)
    summary["anchor_diag_store_full_scores"] = bool(cfg.anchor_diag_store_full_scores)
    summary["oracle_support_injection_enabled"] = bool(getattr(cfg, "oracle_support_injection_enabled", False))
    summary["oracle_mode"] = "oracle_upper_bound" if bool(getattr(cfg, "oracle_support_injection_enabled", False)) else "non_oracle"
    summary["run_timestamp"] = run_stamp
    summary["run_timestamp_utc"] = run_iso

    index_diags = []
    for row in rows:
        diag = _extract_global_index_diag(row)
        if diag:
            index_diags.append(diag)
    summary["index_timed_samples"] = int(len(index_diags))
    summary["index_total_ms"] = mean_or_zero([float(d.get("index_total_ms", 0.0)) for d in index_diags])
    summary["index_build_ms"] = mean_or_zero([float(d.get("index_build_ms", 0.0)) for d in index_diags])
    summary["index_load_graph_ms"] = mean_or_zero([float(d.get("index_load_graph_ms", 0.0)) for d in index_diags])
    summary["index_write_ms"] = mean_or_zero([float(d.get("index_write_ms", 0.0)) for d in index_diags])
    summary["index_cache_hit_rate"] = mean_or_zero(
        [1.0 if bool(d.get("cache_hit", False)) else 0.0 for d in index_diags]
    )

    if bool(getattr(cfg, "timestamp_output", True)):
        out_dir = Path(cfg.output_dir) / str(cfg.dataset) / run_stamp
    else:
        out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "retrieval_query_results.jsonl"
    summary_path = out_dir / "retrieval_summary.json"
    summary["output_dir"] = str(out_dir.resolve())
    write_jsonl(query_path, rows)
    if debug_enabled and sf_debug_rows:
        raw_debug_path = str(getattr(cfg, "sf_debug_output", "") or "").strip()
        debug_path = Path(raw_debug_path) if raw_debug_path else (out_dir / "supporting_fact_debug.jsonl")
        write_jsonl(debug_path, sf_debug_rows)
        summary["supporting_fact_debug_path"] = str(debug_path.resolve())
        summary["supporting_fact_debug_samples"] = int(len(sf_debug_rows))
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
    print(f"Output dir: {summary.get('output_dir', '')}")
    print(
        markdown_table(
            [
                "method",
                "samples",
                "supporting_fact_recall",
                "supporting_fact_precision",
                "recall@1",
                "recall@5",
                "recall@20",
                "retrieval_latency_ms",
                "index_total_ms",
                "run_timestamp",
            ],
            [
                [
                    summary["method"],
                    int(summary["n_samples"]),
                    "%.4f" % summary["supporting_fact_recall"],
                    "%.4f" % summary["supporting_fact_precision"],
                    "%.4f" % summary.get("supporting_fact_recall_at_1", 0.0),
                    "%.4f" % summary.get("supporting_fact_recall_at_5", 0.0),
                    "%.4f" % summary.get("supporting_fact_recall_at_20", 0.0),
                    "%.2f" % summary["retrieval_latency_ms"],
                    "%.2f" % summary.get("index_total_ms", 0.0),
                    summary["run_timestamp"],
                ]
            ],
        )
    )


if __name__ == "__main__":
    main()
