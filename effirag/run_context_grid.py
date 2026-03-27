import argparse
import csv
import json
from itertools import product
from pathlib import Path

from .config import RagConfig, RetrievalConfig
from .run_rag import execute_rag_experiment
from .run_retrieval import execute_retrieval_experiment
from .utils import markdown_table, parse_bool, timestamp_for_filename, write_json

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _csv_to_list(raw: str, caster):
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        return []
    return [caster(x) for x in items]


def _bool_csv(raw: str):
    return _csv_to_list(raw, parse_bool)


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search for corridor-aware-flat context packing while keeping retrieval fixed."
    )
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--global-corpus-path", type=str, default="")
    parser.add_argument("--graph-cache-dir", type=str, default="outputs/index_cache")
    parser.add_argument("--force-rebuild-graph-index", type=str, default="false")
    parser.add_argument("--openie-mode", type=str, default="llm", choices=["llm", "lexical"])
    parser.add_argument("--openie-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--openie-text-max-chars", type=int, default=2200)
    parser.add_argument("--openie-max-new-tokens", type=int, default=256)
    parser.add_argument("--openie-api-base-url", type=str, default="")
    parser.add_argument("--openie-api-key", type=str, default="")
    parser.add_argument("--openie-api-timeout-sec", type=float, default=120.0)
    parser.add_argument("--openie-parallel-workers", type=int, default=4)
    parser.add_argument("--openie-log-every", type=int, default=200)
    parser.add_argument("--embedding-enabled", type=str, default="false")
    parser.add_argument("--embedding-model-name", type=str, default="NVIDIA/NV-Embed-v2")
    parser.add_argument("--embedding-weight", type=float, default=0.35)
    parser.add_argument("--embedding-rerank-topn", type=int, default=80)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--embedding-text-max-chars", type=int, default=600)
    parser.add_argument("--semantic-topn", type=int, default=50)
    parser.add_argument("--semantic-topn-entity", type=int, default=30)
    parser.add_argument("--semantic-topn-chunk", type=int, default=15)
    parser.add_argument("--graph-reserve-topn", type=int, default=15)
    parser.add_argument("--semantic-candidate-union", type=str, default="true")
    parser.add_argument("--semantic-scan-batch-size", type=int, default=8192)
    parser.add_argument("--run-score-semantic-weight", type=float, default=0.35)
    parser.add_argument("--run-score-anchor-weight", type=float, default=0.20)
    parser.add_argument("--run-score-structure-weight", type=float, default=0.25)
    parser.add_argument("--run-score-bridge-weight", type=float, default=0.15)
    parser.add_argument("--run-score-redundancy-weight", type=float, default=0.10)
    parser.add_argument("--seed-score-semantic-weight", type=float, default=0.35)
    parser.add_argument("--seed-score-graph-weight", type=float, default=0.45)
    parser.add_argument("--seed-score-anchor-weight", type=float, default=0.20)

    parser.add_argument("--method", type=str, default="effirag", choices=["effirag", "naive_graphrag"])
    parser.add_argument("--run-qa", type=str, default="true")
    parser.add_argument("--generator", type=str, default="hf", choices=["heuristic", "hf", "oracle", "openai_compat", "vllm"])
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-base-url", type=str, default="")
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--llm-timeout-sec", type=float, default=120.0)
    parser.add_argument("--llm-max-new-tokens", type=int, default=64)

    # Retrieval remains fixed.
    parser.add_argument("--max-anchors", type=int, default=5)
    parser.add_argument("--samples-per-anchor", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--candidate-top-t", type=int, default=20)
    parser.add_argument("--seed-k", type=int, default=4)
    parser.add_argument("--pair-top-lp", type=int, default=3)
    parser.add_argument("--corridor-top-bc", type=int, default=20)
    parser.add_argument("--phase1-parallel-ppr", type=str, default="true")
    parser.add_argument("--phase1-run-shortlist-topk", type=int, default=2)
    parser.add_argument("--pair-shortlist-topb", type=int, default=6)
    parser.add_argument("--phase2-refine-mode", type=str, default="bounded_local")
    parser.add_argument("--phase2-bidirectional-full-ppr", type=str, default="false")
    parser.add_argument("--reuse-semantic-scores-in-final", type=str, default="true")
    parser.add_argument("--trim-on", type=str, default="true")
    parser.add_argument("--trim-rho", type=float, default=0.6)
    parser.add_argument("--ppr-alpha", type=float, default=0.15)
    parser.add_argument("--tau", type=int, default=4)
    parser.add_argument("--edge-drop-prob", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)

    # Shared context budget.
    parser.add_argument("--max-context-sentences", type=int, default=10)
    parser.add_argument("--max-total-sentences", type=int, default=12)

    # Baseline corridor configuration.
    parser.add_argument("--corridor-k", type=int, default=3)
    parser.add_argument("--corridor-main", type=int, default=2)
    parser.add_argument("--corridor-support", type=int, default=1)

    # corridor-aware-flat grids.
    parser.add_argument("--alpha-grid", type=str, default="1.0")
    parser.add_argument("--beta-grid", type=str, default="0.25,0.35")
    parser.add_argument("--gamma-main-grid", type=str, default="0.30,0.45")
    parser.add_argument("--delta-support-grid", type=str, default="0.10")
    parser.add_argument("--eta-connector-grid", type=str, default="0.00,0.20")
    parser.add_argument("--zeta-query-grid", type=str, default="0.08")
    parser.add_argument("--xi-locality-grid", type=str, default="0.05")
    parser.add_argument("--lambda-redundancy-grid", type=str, default="0.25,0.35")
    parser.add_argument("--top-corridors-grid", type=str, default="2,3")
    parser.add_argument("--max-sentences-grid", type=str, default="10")
    parser.add_argument("--reserve-top-corridor-grid", type=str, default="false,true")
    parser.add_argument("--order-strategy-grid", type=str, default="score")
    parser.add_argument("--max-combinations", type=int, default=256)
    return parser


def _build_caf_profiles(args):
    axes = [
        ("alpha", _csv_to_list(args.alpha_grid, float)),
        ("beta", _csv_to_list(args.beta_grid, float)),
        ("gamma_main", _csv_to_list(args.gamma_main_grid, float)),
        ("delta_support", _csv_to_list(args.delta_support_grid, float)),
        ("eta_connector", _csv_to_list(args.eta_connector_grid, float)),
        ("zeta_query", _csv_to_list(args.zeta_query_grid, float)),
        ("xi_locality", _csv_to_list(args.xi_locality_grid, float)),
        ("lambda_redundancy", _csv_to_list(args.lambda_redundancy_grid, float)),
        ("top_corridors", _csv_to_list(args.top_corridors_grid, int)),
        ("max_sentences", _csv_to_list(args.max_sentences_grid, int)),
        ("reserve_top_corridor", _bool_csv(args.reserve_top_corridor_grid)),
        ("order_strategy", _csv_to_list(args.order_strategy_grid, str)),
    ]
    for name, values in axes:
        if not values:
            raise ValueError(f"Grid axis '{name}' is empty.")

    combinations = list(product(*[vals for _, vals in axes]))
    if len(combinations) > args.max_combinations:
        raise ValueError(
            f"Too many combinations: {len(combinations)} > {args.max_combinations}. "
            "Reduce grid size or increase --max-combinations."
        )

    profiles = []
    for i, combo in enumerate(combinations, start=1):
        payload = {name: combo[idx] for idx, (name, _) in enumerate(axes)}
        payload["profile"] = f"caf_{i:03d}"
        profiles.append(payload)
    return profiles


def _build_retrieval_cfg(args, output_dir: Path):
    return RetrievalConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.limit,
        method=args.method,
        output_dir=str(output_dir),
        timestamp_output=False,
        global_corpus_path=args.global_corpus_path,
        graph_cache_dir=args.graph_cache_dir,
        force_rebuild_graph_index=parse_bool(args.force_rebuild_graph_index),
        openie_mode=args.openie_mode,
        openie_model_name=args.openie_model_name,
        openie_text_max_chars=args.openie_text_max_chars,
        openie_max_new_tokens=args.openie_max_new_tokens,
        openie_api_base_url=args.openie_api_base_url,
        openie_api_key=args.openie_api_key,
        openie_api_timeout_sec=args.openie_api_timeout_sec,
        openie_parallel_workers=args.openie_parallel_workers,
        openie_log_every=args.openie_log_every,
        embedding_enabled=parse_bool(args.embedding_enabled),
        embedding_model_name=args.embedding_model_name,
        embedding_weight=args.embedding_weight,
        embedding_rerank_topn=args.embedding_rerank_topn,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        embedding_text_max_chars=args.embedding_text_max_chars,
        semantic_topn_entity=args.semantic_topn_entity,
        semantic_topn_chunk=args.semantic_topn_chunk,
        graph_reserve_topn=args.graph_reserve_topn,
        semantic_topn=args.semantic_topn,
        semantic_candidate_union=parse_bool(args.semantic_candidate_union),
        semantic_scan_batch_size=args.semantic_scan_batch_size,
        run_score_semantic_weight=args.run_score_semantic_weight,
        run_score_anchor_weight=args.run_score_anchor_weight,
        run_score_structure_weight=args.run_score_structure_weight,
        run_score_bridge_weight=args.run_score_bridge_weight,
        run_score_redundancy_weight=args.run_score_redundancy_weight,
        seed_score_semantic_weight=args.seed_score_semantic_weight,
        seed_score_graph_weight=args.seed_score_graph_weight,
        seed_score_anchor_weight=args.seed_score_anchor_weight,
        max_anchors=args.max_anchors,
        samples_per_anchor=args.samples_per_anchor,
        num_workers=args.num_workers,
        candidate_top_t=args.candidate_top_t,
        seed_k=args.seed_k,
        pair_top_lp=args.pair_top_lp,
        corridor_top_bc=args.corridor_top_bc,
        phase1_parallel_ppr=parse_bool(args.phase1_parallel_ppr),
        phase1_run_shortlist_topk=args.phase1_run_shortlist_topk,
        pair_shortlist_topb=args.pair_shortlist_topb,
        phase2_refine_mode=args.phase2_refine_mode,
        phase2_bidirectional_full_ppr=parse_bool(args.phase2_bidirectional_full_ppr),
        reuse_semantic_scores_in_final=parse_bool(args.reuse_semantic_scores_in_final),
        trim_on=parse_bool(args.trim_on),
        trim_rho=args.trim_rho,
        ppr_alpha=args.ppr_alpha,
        tau=args.tau,
        edge_drop_prob=args.edge_drop_prob,
        random_seed=args.random_seed,
    )


def _base_rag_kwargs(args, output_dir: Path):
    return dict(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.limit,
        method=args.method,
        output_dir=str(output_dir),
        timestamp_output=False,
        global_corpus_path=args.global_corpus_path,
        graph_cache_dir=args.graph_cache_dir,
        force_rebuild_graph_index=parse_bool(args.force_rebuild_graph_index),
        openie_mode=args.openie_mode,
        openie_model_name=args.openie_model_name,
        openie_text_max_chars=args.openie_text_max_chars,
        openie_max_new_tokens=args.openie_max_new_tokens,
        openie_api_base_url=args.openie_api_base_url,
        openie_api_key=args.openie_api_key,
        openie_api_timeout_sec=args.openie_api_timeout_sec,
        openie_parallel_workers=args.openie_parallel_workers,
        openie_log_every=args.openie_log_every,
        embedding_enabled=parse_bool(args.embedding_enabled),
        embedding_model_name=args.embedding_model_name,
        embedding_weight=args.embedding_weight,
        embedding_rerank_topn=args.embedding_rerank_topn,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        embedding_text_max_chars=args.embedding_text_max_chars,
        semantic_topn_entity=args.semantic_topn_entity,
        semantic_topn_chunk=args.semantic_topn_chunk,
        graph_reserve_topn=args.graph_reserve_topn,
        semantic_topn=args.semantic_topn,
        semantic_candidate_union=parse_bool(args.semantic_candidate_union),
        semantic_scan_batch_size=args.semantic_scan_batch_size,
        run_score_semantic_weight=args.run_score_semantic_weight,
        run_score_anchor_weight=args.run_score_anchor_weight,
        run_score_structure_weight=args.run_score_structure_weight,
        run_score_bridge_weight=args.run_score_bridge_weight,
        run_score_redundancy_weight=args.run_score_redundancy_weight,
        seed_score_semantic_weight=args.seed_score_semantic_weight,
        seed_score_graph_weight=args.seed_score_graph_weight,
        seed_score_anchor_weight=args.seed_score_anchor_weight,
        max_anchors=args.max_anchors,
        samples_per_anchor=args.samples_per_anchor,
        num_workers=args.num_workers,
        candidate_top_t=args.candidate_top_t,
        seed_k=args.seed_k,
        pair_top_lp=args.pair_top_lp,
        corridor_top_bc=args.corridor_top_bc,
        phase1_parallel_ppr=parse_bool(args.phase1_parallel_ppr),
        phase1_run_shortlist_topk=args.phase1_run_shortlist_topk,
        pair_shortlist_topb=args.pair_shortlist_topb,
        phase2_refine_mode=args.phase2_refine_mode,
        phase2_bidirectional_full_ppr=parse_bool(args.phase2_bidirectional_full_ppr),
        reuse_semantic_scores_in_final=parse_bool(args.reuse_semantic_scores_in_final),
        trim_on=parse_bool(args.trim_on),
        trim_rho=args.trim_rho,
        ppr_alpha=args.ppr_alpha,
        tau=args.tau,
        edge_drop_prob=args.edge_drop_prob,
        random_seed=args.random_seed,
        run_qa=parse_bool(args.run_qa),
        generator=args.generator,
        model_name=args.model_name if args.generator in {"hf", "openai_compat", "vllm"} else "",
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_timeout_sec=args.llm_timeout_sec,
        llm_max_new_tokens=args.llm_max_new_tokens,
        max_context_sentences=args.max_context_sentences,
        max_total_sentences=args.max_total_sentences,
        measure_gpu_peak=False,
        measure_cpu_ram=False,
    )


def _fallback_count(query_path: Path):
    if not query_path.exists():
        return 0
    count = 0
    with query_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "HF generation failed" in line or "Fallback(heuristic)" in line:
                count += 1
    return count


def _row_from_summary(label: str, mode: str, profile: str, summary: dict, summary_path: Path, query_path: Path, params: dict):
    row = {
        "label": label,
        "mode": mode,
        "profile": profile,
        "samples": int(summary.get("n_samples", 0)),
        "em": float(summary.get("em", 0.0)),
        "f1": float(summary.get("f1", 0.0)),
        "rendered_sf_recall": float(summary.get("rendered_supporting_fact_recall", 0.0)),
        "total_ms": float(summary.get("total_latency_ms", 0.0)),
        "trunc_sent_avg": float(summary.get("truncated_sentences_avg", 0.0)),
        "fallback_count": int(_fallback_count(query_path)),
        "summary_path": str(summary_path),
    }
    row.update(params)
    return row


def _rows_to_markdown(rows):
    headers = [
        "label",
        "mode",
        "profile",
        "samples",
        "em",
        "f1",
        "rendered_sf_recall",
        "total_ms",
        "trunc_sent_avg",
        "fallback_count",
        "alpha",
        "beta",
        "gamma_main",
        "delta_support",
        "eta_connector",
        "zeta_query",
        "xi_locality",
        "lambda_redundancy",
        "top_corridors",
        "max_sentences",
        "reserve_top_corridor",
        "order_strategy",
    ]
    body = [[_fmt(row.get(h, "")) for h in headers] for row in rows]
    return markdown_table(headers, body)


def _best_caf(caf_rows, flat_row, corridor_row):
    if not caf_rows:
        return None, []
    feasible = []
    for row in caf_rows:
        cond_em = row["em"] >= flat_row["em"]
        cond_f1 = (row["f1"] >= corridor_row["f1"]) or (row["f1"] > flat_row["f1"])
        cond_ms = row["total_ms"] < corridor_row["total_ms"]
        if cond_em and cond_f1 and cond_ms:
            feasible.append(row)

    if feasible:
        feasible.sort(key=lambda r: (r["f1"], r["em"], -r["total_ms"]), reverse=True)
        return feasible[0], feasible

    # Fallback best-effort candidate if no row satisfies all criteria.
    all_sorted = sorted(caf_rows, key=lambda r: (r["f1"], r["em"], -r["total_ms"]), reverse=True)
    return all_sorted[0], []


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    profiles = _build_caf_profiles(args)
    run_stamp = timestamp_for_filename()
    base_dir = Path(args.output_root) / f"context_grid_{run_stamp}"
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[context-grid] output root: {base_dir}")
    print(f"[context-grid] caf combinations: {len(profiles)}")

    # 1) Retrieval prepass once.
    retrieval_cfg = _build_retrieval_cfg(args, base_dir / "retrieval" / args.method)
    _, retrieval_summary = execute_retrieval_experiment(retrieval_cfg, show_progress=False)
    retrieval_query_path = Path(retrieval_cfg.output_dir) / "retrieval_query_results.jsonl"
    print(
        "[context-grid] retrieval prepass done: "
        f"sf_recall={retrieval_summary.get('supporting_fact_recall', 0.0):.4f}, "
        f"path={retrieval_query_path}"
    )

    rows = []

    # 2) Flat baseline.
    flat_dir = base_dir / "rag" / "flat"
    flat_cfg = RagConfig(
        **_base_rag_kwargs(args, flat_dir),
        render_mode="flat",
    )
    _, flat_summary = execute_rag_experiment(
        flat_cfg,
        show_progress=False,
        precomputed_retrieval_path=str(retrieval_query_path),
    )
    rows.append(
        _row_from_summary(
            label="baseline_flat",
            mode="flat",
            profile="flat",
            summary=flat_summary,
            summary_path=flat_dir / "rag_summary.json",
            query_path=flat_dir / "rag_query_results.jsonl",
            params={},
        )
    )

    # 3) Corridor baseline (fixed K=3, main=2, support=1 by default).
    corridor_dir = base_dir / "rag" / "corridor"
    corridor_cfg = RagConfig(
        **_base_rag_kwargs(args, corridor_dir),
        render_mode="corridor",
        max_corridors_in_context=args.corridor_k,
        max_main_sentences_per_corridor=args.corridor_main,
        max_support_per_corridor=args.corridor_support,
    )
    _, corridor_summary = execute_rag_experiment(
        corridor_cfg,
        show_progress=False,
        precomputed_retrieval_path=str(retrieval_query_path),
    )
    rows.append(
        _row_from_summary(
            label="baseline_corridor",
            mode="corridor",
            profile=f"k{args.corridor_k}_m{args.corridor_main}_s{args.corridor_support}",
            summary=corridor_summary,
            summary_path=corridor_dir / "rag_summary.json",
            query_path=corridor_dir / "rag_query_results.jsonl",
            params={
                "top_corridors": args.corridor_k,
                "max_sentences": args.max_context_sentences,
                "order_strategy": "corridor",
            },
        )
    )

    # 4) corridor_aware_flat grid.
    for profile in tqdm(profiles, total=len(profiles), desc="CAF grid", leave=False):
        out_dir = base_dir / "rag" / "corridor_aware_flat" / profile["profile"]
        cfg = RagConfig(
            **_base_rag_kwargs(args, out_dir),
            render_mode="corridor_aware_flat",
            alpha=profile["alpha"],
            beta=profile["beta"],
            gamma_main=profile["gamma_main"],
            delta_support=profile["delta_support"],
            eta_connector=profile["eta_connector"],
            zeta_query=profile["zeta_query"],
            xi_locality=profile["xi_locality"],
            lambda_redundancy=profile["lambda_redundancy"],
            top_corridors=profile["top_corridors"],
            max_sentences=profile["max_sentences"],
            reserve_top_corridor=profile["reserve_top_corridor"],
            order_strategy=profile["order_strategy"],
        )
        _, summary = execute_rag_experiment(
            cfg,
            show_progress=False,
            precomputed_retrieval_path=str(retrieval_query_path),
        )
        rows.append(
            _row_from_summary(
                label="grid_caf",
                mode="corridor_aware_flat",
                profile=profile["profile"],
                summary=summary,
                summary_path=out_dir / "rag_summary.json",
                query_path=out_dir / "rag_query_results.jsonl",
                params=profile,
            )
        )

    # Stable row ordering: baselines then CAF by f1 desc.
    baseline_rows = [r for r in rows if r["label"].startswith("baseline_")]
    caf_rows = [r for r in rows if r["label"] == "grid_caf"]
    caf_rows.sort(key=lambda r: (r["f1"], r["em"], -r["total_ms"]), reverse=True)
    ordered_rows = baseline_rows + caf_rows

    flat_row = [r for r in baseline_rows if r["mode"] == "flat"][0]
    corridor_row = [r for r in baseline_rows if r["mode"] == "corridor"][0]
    best, feasible = _best_caf(caf_rows, flat_row, corridor_row)

    md_table = _rows_to_markdown(ordered_rows)
    title = "**Context Grid Summary**"
    print("[context-grid] done")
    print(title)
    print(md_table)

    best_payload = {
        "best_profile": best,
        "success_criteria_met": bool(feasible and best in feasible),
        "criteria": {
            "em >= flat_em": flat_row["em"],
            "f1 >= corridor_f1 or f1 > flat_f1": {
                "corridor_f1": corridor_row["f1"],
                "flat_f1": flat_row["f1"],
            },
            "total_ms < corridor_total_ms": corridor_row["total_ms"],
        },
        "n_feasible": len(feasible),
    }
    print("[context-grid][best]")
    print(json.dumps(best_payload, ensure_ascii=False, indent=2))

    summary_md = logs_dir / "context_grid_summary.md"
    summary_json = logs_dir / "context_grid_summary.json"
    summary_csv = logs_dir / "context_grid_summary.csv"
    summary_md.write_text(title + "\n\n" + md_table + "\n", encoding="utf-8")
    write_json(summary_json, {"rows": ordered_rows, "best": best_payload})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in ordered_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(row)
    print(f"[context-grid] saved summary md: {summary_md}")
    print(f"[context-grid] saved summary json: {summary_json}")
    print(f"[context-grid] saved summary csv: {summary_csv}")


if __name__ == "__main__":
    main()
