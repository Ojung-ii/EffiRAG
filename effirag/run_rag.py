import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _NoOpTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])

        def update(self, n=1):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def tqdm(iterable=None, **kwargs):
        return _NoOpTqdm(iterable=iterable, **kwargs)


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


def _is_generation_fallback(generation) -> bool:
    if generation is None:
        return False
    raw = str(getattr(generation, "raw_text", "") or "")
    return ("HF generation failed" in raw) or ("Fallback(heuristic)" in raw)


def _extract_global_index_diag_from_retrieval(retrieval_payload: dict) -> dict:
    diagnostics = (retrieval_payload or {}).get("diagnostics", {}) or {}
    payload = diagnostics.get("global_index", {}) or {}
    if isinstance(payload, dict):
        return payload
    return {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end EffiRAG QA experiments.")
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
    parser.add_argument("--pair-shortlist-topb", type=int, default=None)
    parser.add_argument("--phase2-refine-mode", type=str, default=None)
    parser.add_argument("--phase2-bidirectional-full-ppr", type=str, default=None)
    parser.add_argument("--reuse-semantic-scores-in-final", type=str, default=None)
    parser.add_argument("--trim-on", type=str, default=None)
    parser.add_argument("--trim-rho", type=float, default=None)

    parser.add_argument("--run-qa", type=str, default=None)
    parser.add_argument("--generator", type=str, default=None, choices=["heuristic", "oracle", "hf", "openai_compat", "vllm"])
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-timeout-sec", type=float, default=None)
    parser.add_argument("--llm-max-new-tokens", type=int, default=None)
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
    if bool(getattr(cfg, "timestamp_output", True)):
        out_dir = Path(cfg.output_dir) / str(cfg.dataset) / run_stamp
    else:
        out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "rag_query_results.jsonl"
    summary_path = out_dir / "rag_summary.json"
    # Stream per-sample outputs so progress is inspectable even before the run ends.
    query_path.write_text("", encoding="utf-8")
    render_mode_requested = str(cfg.render_mode or "").strip()
    resolved_render_mode = render_mode_requested or ("corridor_aware_flat" if cfg.method == "effirag" else "flat")
    retrieval_source_counts = {"precomputed": 0, "on_the_fly": 0}
    fallback_count = 0
    retrieval_bar = tqdm(
        total=len(samples),
        desc=f"Retrieval[{cfg.method}]",
        unit="sample",
        leave=False,
        disable=not show_progress,
    )
    generation_bar = tqdm(
        total=len(samples),
        desc=f"Generation[{cfg.generator}]",
        unit="sample",
        leave=False,
        disable=(not show_progress) or (not cfg.run_qa),
    )
    try:
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
            retrieval_source = "precomputed"
            if retrieval is None:
                retrieval = method_fn(sample, cfg)
                retrieval_source = "on_the_fly"
            retrieval_source_counts[retrieval_source] = retrieval_source_counts.get(retrieval_source, 0) + 1

            rendered = render_context(
                sample,
                retrieval,
                max_context_sentences=cfg.max_context_sentences,
                render_mode=resolved_render_mode,
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
            retrieval_bar.update(1)

            if cfg.measure_cpu_ram:
                cpu_peak = max(cpu_peak, process_rss_mb())

            generation = None
            em = 0.0
            f1 = 0.0
            if cfg.run_qa:
                generation = generator_fn(sample, rendered, cfg.model_name, cfg)
                em = exact_match_score(generation.prediction, sample.answer)
                f1 = token_f1_score(generation.prediction, sample.answer)
                if _is_generation_fallback(generation):
                    fallback_count += 1
                generation_bar.update(1)

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

            row = {
                "sample_id": sample.qid,
                "question": sample.question,
                "answer": sample.answer,
                "prediction": generation.prediction if generation else "",
                "qa_executed": bool(generation is not None),
                "generation_fallback": bool(_is_generation_fallback(generation)),
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
                    "render_mode_requested": render_mode_requested or "(auto)",
                    "truncated_corridors": rendered.truncated_corridor_count,
                    "truncated_sentences": rendered.truncated_sentence_count,
                },
                "retrieval_source": retrieval_source,
                "generation": asdict(generation) if generation else None,
                "run_timestamp": run_stamp,
            }
            rows.append(row)
            append_jsonl(query_path, row)
    finally:
        retrieval_bar.close()
        generation_bar.close()

    summary = {
        "n_samples": float(len(rows)),
        "dataset": cfg.dataset,
        "method": cfg.method,
        "generator": cfg.generator,
        "generator_display": str(cfg.model_name or cfg.generator),
        "run_qa": bool(cfg.run_qa),
        "qa_executed_samples": float(sum(1 for r in rows if r.get("qa_executed"))),
        "qa_skipped_samples": float(sum(1 for r in rows if not r.get("qa_executed"))),
        "render_mode_requested": render_mode_requested or "(auto)",
        "render_mode_resolved": resolved_render_mode,
        "precomputed_retrieval_used": bool(precomputed_retrieval_path),
        "retrieval_source_breakdown": retrieval_source_counts,
        "fallback_count": int(fallback_count),
        "fallback_rate": mean_or_zero(
            [1.0 if r.get("generation_fallback", False) else 0.0 for r in rows if r.get("qa_executed", False)]
        ),
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
        "retrieval_params": {
            "global_corpus_path": cfg.global_corpus_path,
            "graph_cache_dir": cfg.graph_cache_dir,
            "force_rebuild_graph_index": cfg.force_rebuild_graph_index,
            "prebuilt_igraph_path": cfg.prebuilt_igraph_path,
            "prebuilt_igraph_format": cfg.prebuilt_igraph_format,
            "prebuilt_entity_token_limit": cfg.prebuilt_entity_token_limit,
            "openie_mode": cfg.openie_mode,
            "openie_model_name": cfg.openie_model_name,
            "openie_text_max_chars": cfg.openie_text_max_chars,
            "openie_max_new_tokens": cfg.openie_max_new_tokens,
            "openie_local_files_only": cfg.openie_local_files_only,
            "openie_retry_attempts": cfg.openie_retry_attempts,
            "openie_retry_backoff_sec": cfg.openie_retry_backoff_sec,
            "openie_error_sample_limit": cfg.openie_error_sample_limit,
            "openie_api_base_url": cfg.openie_api_base_url,
            "openie_api_timeout_sec": cfg.openie_api_timeout_sec,
            "openie_parallel_workers": cfg.openie_parallel_workers,
            "openie_log_every": cfg.openie_log_every,
            "embedding_enabled": cfg.embedding_enabled,
            "embedding_model_name": cfg.embedding_model_name,
            "embedding_weight": cfg.embedding_weight,
            "embedding_rerank_topn": cfg.embedding_rerank_topn,
            "embedding_batch_size": cfg.embedding_batch_size,
            "embedding_max_length": cfg.embedding_max_length,
            "embedding_text_max_chars": cfg.embedding_text_max_chars,
            "semantic_topn_entity": cfg.semantic_topn_entity,
            "semantic_topn_chunk": cfg.semantic_topn_chunk,
            "graph_reserve_topn": cfg.graph_reserve_topn,
            "semantic_topn": cfg.semantic_topn,
            "semantic_candidate_union": cfg.semantic_candidate_union,
            "semantic_scan_batch_size": cfg.semantic_scan_batch_size,
            "run_score_semantic_weight": cfg.run_score_semantic_weight,
            "run_score_anchor_weight": cfg.run_score_anchor_weight,
            "run_score_structure_weight": cfg.run_score_structure_weight,
            "run_score_bridge_weight": cfg.run_score_bridge_weight,
            "run_score_redundancy_weight": cfg.run_score_redundancy_weight,
            "seed_score_semantic_weight": cfg.seed_score_semantic_weight,
            "seed_score_graph_weight": cfg.seed_score_graph_weight,
            "seed_score_anchor_weight": cfg.seed_score_anchor_weight,
            "max_anchors": cfg.max_anchors,
            "samples_per_anchor": cfg.samples_per_anchor,
            "candidate_top_t": cfg.candidate_top_t,
            "seed_k": cfg.seed_k,
            "pair_top_lp": cfg.pair_top_lp,
            "corridor_top_bc": cfg.corridor_top_bc,
            "phase1_parallel_ppr": cfg.phase1_parallel_ppr,
            "phase1_run_shortlist_topk": cfg.phase1_run_shortlist_topk,
            "pair_shortlist_topb": cfg.pair_shortlist_topb,
            "phase2_refine_mode": cfg.phase2_refine_mode,
            "phase2_bidirectional_full_ppr": cfg.phase2_bidirectional_full_ppr,
            "reuse_semantic_scores_in_final": cfg.reuse_semantic_scores_in_final,
            "trim_on": cfg.trim_on,
            "trim_rho": cfg.trim_rho,
            "ppr_alpha": cfg.ppr_alpha,
            "tau": cfg.tau,
            "edge_drop_prob": cfg.edge_drop_prob,
            "ppr_engine": cfg.ppr_engine,
            "ppr_power_max_iter": cfg.ppr_power_max_iter,
            "ppr_power_tol": cfg.ppr_power_tol,
            "ppr_min_score": cfg.ppr_min_score,
            "ppr_mc_walks": cfg.ppr_mc_walks,
            "ppr_mc_max_steps": cfg.ppr_mc_max_steps,
            "ppr_parallel_workers": cfg.ppr_parallel_workers,
            "ppr_subgraph_enable": cfg.ppr_subgraph_enable,
            "ppr_subgraph_hops": cfg.ppr_subgraph_hops,
            "ppr_subgraph_max_nodes": cfg.ppr_subgraph_max_nodes,
            "anchor_diag_topn": cfg.anchor_diag_topn,
            "anchor_diag_store_full_scores": cfg.anchor_diag_store_full_scores,
            "random_seed": cfg.random_seed,
        },
        "generation_params": {
            "model_name": cfg.model_name,
            "llm_base_url": cfg.llm_base_url,
            "llm_timeout_sec": cfg.llm_timeout_sec,
            "llm_max_new_tokens": cfg.llm_max_new_tokens,
        },
        "render_params": {
            "max_context_sentences": cfg.max_context_sentences,
            "max_corridors_in_context": cfg.max_corridors_in_context,
            "max_main_sentences_per_corridor": cfg.max_main_sentences_per_corridor,
            "max_support_per_corridor": cfg.max_support_per_corridor,
            "max_total_sentences": cfg.max_total_sentences,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "gamma_main": cfg.gamma_main,
            "delta_support": cfg.delta_support,
            "eta_connector": cfg.eta_connector,
            "zeta_query": cfg.zeta_query,
            "xi_locality": cfg.xi_locality,
            "lambda_redundancy": cfg.lambda_redundancy,
            "top_corridors": cfg.top_corridors,
            "max_sentences": cfg.max_sentences,
            "reserve_top_corridor": cfg.reserve_top_corridor,
            "order_strategy": cfg.order_strategy,
        },
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

    index_diags = []
    for row in rows:
        diag = _extract_global_index_diag_from_retrieval(row.get("retrieval", {}))
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
    summary["output_dir"] = str(out_dir.resolve())

    if cfg.measure_gpu_peak:
        summary["gpu_peak_mb"] = mean_or_zero([r["efficiency"].get("gpu_peak_mb", 0.0) for r in rows])
    if cfg.measure_cpu_ram:
        summary["cpu_ram_peak_mb"] = mean_or_zero([r["efficiency"].get("cpu_ram_peak_mb", 0.0) for r in rows])

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
    print(f"Output dir: {summary.get('output_dir', '')}")
    print(
        markdown_table(
            [
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
                "generation_ms",
                "total_ms",
                "index_total_ms",
                "trunc_corr_avg",
                "trunc_sent_avg",
                "run_timestamp",
            ],
            [
                [
                    summary["method"],
                    summary.get("generator_display", summary["generator"]),
                    int(summary["n_samples"]),
                    "%.4f" % summary["supporting_fact_recall"],
                    "%.4f" % summary.get("supporting_fact_recall_at_1", 0.0),
                    "%.4f" % summary.get("supporting_fact_recall_at_5", 0.0),
                    "%.4f" % summary.get("supporting_fact_recall_at_20", 0.0),
                    "%.4f" % summary.get("rendered_supporting_fact_recall", 0.0),
                    "%.4f" % summary["em"],
                    "%.4f" % summary["f1"],
                    "%.2f" % summary["retrieval_latency_ms"],
                    "%.2f" % summary.get("generation_latency_ms", 0.0),
                    "%.2f" % summary["total_latency_ms"],
                    "%.2f" % summary.get("index_total_ms", 0.0),
                    "%.2f" % summary.get("truncated_corridors_avg", 0.0),
                    "%.2f" % summary.get("truncated_sentences_avg", 0.0),
                    summary["run_timestamp"],
                ]
            ],
        )
    )


if __name__ == "__main__":
    main()
