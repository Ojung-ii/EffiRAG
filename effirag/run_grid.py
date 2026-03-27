import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EffiRAG parameter grid experiments.")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--retrieval-limit", type=int, default=100)
    parser.add_argument("--rag-limit", type=int, default=100)
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
    parser.add_argument("--embedding-model-name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-weight", type=float, default=0.35)
    parser.add_argument("--embedding-rerank-topn", type=int, default=80)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--embedding-text-max-chars", type=int, default=600)
    parser.add_argument("--semantic-topn", type=int, default=50)
    parser.add_argument("--semantic-candidate-union", type=str, default="true")
    parser.add_argument("--semantic-scan-batch-size", type=int, default=8192)
    parser.add_argument("--run-score-semantic-weight", type=float, default=0.35)
    parser.add_argument("--run-score-anchor-weight", type=float, default=0.20)
    parser.add_argument("--run-score-structure-weight", type=float, default=0.25)
    parser.add_argument("--run-score-bridge-weight", type=float, default=0.15)
    parser.add_argument("--run-score-redundancy-weight", type=float, default=0.05)
    parser.add_argument("--seed-score-semantic-weight", type=float, default=0.30)
    parser.add_argument("--seed-score-graph-weight", type=float, default=0.50)
    parser.add_argument("--seed-score-anchor-weight", type=float, default=0.20)

    parser.add_argument("--grid-method", type=str, default="effirag", choices=["effirag", "naive_graphrag"])
    parser.add_argument("--include-naive-baseline", type=str, default="true")
    parser.add_argument("--run-retrieval", type=str, default="true")
    parser.add_argument("--run-rag", type=str, default="true")
    parser.add_argument("--run-qa", type=str, default="true")
    parser.add_argument("--generators", type=str, default="heuristic,hf")
    parser.add_argument("--hf-model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--llm-base-url", type=str, default="")
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--llm-timeout-sec", type=float, default=120.0)
    parser.add_argument("--llm-max-new-tokens", type=int, default=64)

    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--rag-workers", type=int, default=1)
    parser.add_argument("--reuse-retrieval", type=str, default="true")
    parser.add_argument("--rag-topk-profiles", type=int, default=0)
    parser.add_argument("--rag-topk-metric", type=str, default="sf_recall", choices=["sf_recall", "sf_precision"])

    parser.add_argument("--samples-per-anchor-grid", type=str, default="8")
    parser.add_argument("--max-anchors-grid", type=str, default="4")
    parser.add_argument("--candidate-top-t-grid", type=str, default="20")
    parser.add_argument("--seed-k-grid", type=str, default="4")
    parser.add_argument("--pair-top-lp-grid", type=str, default="4")
    parser.add_argument("--corridor-top-bc-grid", type=str, default="20")
    parser.add_argument("--trim-rho-grid", type=str, default="0.75")
    parser.add_argument("--tau-grid", type=str, default="4")
    parser.add_argument("--max-context-sentences-grid", type=str, default="15")

    parser.add_argument("--trim-on", type=str, default="true")
    parser.add_argument("--ppr-alpha", type=float, default=0.15)
    parser.add_argument("--edge-drop-prob", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-combinations", type=int, default=256)
    return parser


def _profile_id(i: int) -> str:
    return f"p{i:03d}"


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _rows_to_markdown(rows):
    headers = [
        "task",
        "profile",
        "method",
        "generator",
        "samples",
        "sf_recall",
        "sf_precision",
        "em",
        "f1",
        "retrieval_ms",
        "total_ms",
        "max_anchors",
        "candidate_top_t",
        "seed_k",
        "pair_top_lp",
        "corridor_top_bc",
        "trim_rho",
        "tau",
        "max_context_sentences",
    ]
    body = []
    for row in rows:
        body.append([_fmt(row.get(k, "")) for k in headers])
    return markdown_table(headers, body)


def _build_grid(args):
    samples_per_anchor = _csv_to_list(args.samples_per_anchor_grid, int)
    max_anchors = _csv_to_list(args.max_anchors_grid, int)
    candidate_top_t = _csv_to_list(args.candidate_top_t_grid, int)
    seed_k = _csv_to_list(args.seed_k_grid, int)
    pair_top_lp = _csv_to_list(args.pair_top_lp_grid, int)
    corridor_top_bc = _csv_to_list(args.corridor_top_bc_grid, int)
    trim_rho = _csv_to_list(args.trim_rho_grid, float)
    tau = _csv_to_list(args.tau_grid, int)
    max_context_sentences = _csv_to_list(args.max_context_sentences_grid, int)

    axes = [
        ("samples_per_anchor", samples_per_anchor),
        ("max_anchors", max_anchors),
        ("candidate_top_t", candidate_top_t),
        ("seed_k", seed_k),
        ("pair_top_lp", pair_top_lp),
        ("corridor_top_bc", corridor_top_bc),
        ("trim_rho", trim_rho),
        ("tau", tau),
        ("max_context_sentences", max_context_sentences),
    ]
    for name, values in axes:
        if not values:
            raise ValueError(f"Grid axis '{name}' is empty.")

    combinations = list(product(*[vals for _, vals in axes]))
    if len(combinations) > args.max_combinations:
        raise ValueError(
            "Too many combinations: "
            f"{len(combinations)} > {args.max_combinations}. "
            "Reduce grid size or increase --max-combinations."
        )

    profiles = []
    for i, combo in enumerate(combinations, start=1):
        payload = {name: combo[idx] for idx, (name, _) in enumerate(axes)}
        payload["profile"] = _profile_id(i)
        profiles.append(payload)
    return profiles


def _retrieval_cfg(args, base_dir: Path, method: str, profile: dict) -> RetrievalConfig:
    return RetrievalConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.retrieval_limit,
        method=method,
        output_dir=str(base_dir / "retrieval" / method / profile["profile"]),
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
        max_anchors=profile["max_anchors"],
        samples_per_anchor=profile["samples_per_anchor"],
        num_workers=args.num_workers,
        candidate_top_t=profile["candidate_top_t"],
        seed_k=profile["seed_k"],
        pair_top_lp=profile["pair_top_lp"],
        corridor_top_bc=profile["corridor_top_bc"],
        trim_on=parse_bool(args.trim_on),
        trim_rho=profile["trim_rho"],
        ppr_alpha=args.ppr_alpha,
        tau=profile["tau"],
        edge_drop_prob=args.edge_drop_prob,
        random_seed=args.random_seed,
    )


def _rag_cfg(args, base_dir: Path, method: str, generator: str, profile: dict) -> RagConfig:
    model_name = args.hf_model if generator in {"hf", "openai_compat", "vllm"} else ""
    return RagConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        split=args.split,
        limit=args.rag_limit,
        method=method,
        output_dir=str(base_dir / "rag" / generator / method / profile["profile"]),
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
        max_anchors=profile["max_anchors"],
        samples_per_anchor=profile["samples_per_anchor"],
        num_workers=args.num_workers,
        candidate_top_t=profile["candidate_top_t"],
        seed_k=profile["seed_k"],
        pair_top_lp=profile["pair_top_lp"],
        corridor_top_bc=profile["corridor_top_bc"],
        trim_on=parse_bool(args.trim_on),
        trim_rho=profile["trim_rho"],
        ppr_alpha=args.ppr_alpha,
        tau=profile["tau"],
        edge_drop_prob=args.edge_drop_prob,
        random_seed=args.random_seed,
        run_qa=parse_bool(args.run_qa),
        generator=generator,
        model_name=model_name,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_timeout_sec=args.llm_timeout_sec,
        llm_max_new_tokens=args.llm_max_new_tokens,
        max_context_sentences=profile["max_context_sentences"],
        measure_gpu_peak=False,
        measure_cpu_ram=False,
    )


def _row_params(profile: dict):
    return {
        "max_anchors": profile["max_anchors"],
        "candidate_top_t": profile["candidate_top_t"],
        "seed_k": profile["seed_k"],
        "pair_top_lp": profile["pair_top_lp"],
        "corridor_top_bc": profile["corridor_top_bc"],
        "trim_rho": profile["trim_rho"],
        "tau": profile["tau"],
        "max_context_sentences": profile["max_context_sentences"],
    }


def _retrieval_row(profile: dict, method: str, summary: dict, summary_path: str):
    row = {
        "task": "retrieval",
        "profile": profile["profile"],
        "method": method,
        "generator": "-",
        "samples": int(summary.get("n_samples", 0)),
        "sf_recall": float(summary.get("supporting_fact_recall", 0.0)),
        "sf_precision": float(summary.get("supporting_fact_precision", 0.0)),
        "em": "",
        "f1": "",
        "retrieval_ms": float(summary.get("retrieval_latency_ms", 0.0)),
        "total_ms": "",
        "summary_path": summary_path,
    }
    row.update(_row_params(profile))
    return row


def _rag_row(profile: dict, method: str, generator: str, summary: dict, summary_path: str):
    row = {
        "task": "rag",
        "profile": profile["profile"],
        "method": method,
        "generator": generator,
        "samples": int(summary.get("n_samples", 0)),
        "sf_recall": float(summary.get("supporting_fact_recall", 0.0)),
        "sf_precision": float(summary.get("supporting_fact_precision", 0.0)),
        "em": float(summary.get("em", 0.0)),
        "f1": float(summary.get("f1", 0.0)),
        "retrieval_ms": float(summary.get("retrieval_latency_ms", 0.0)),
        "total_ms": float(summary.get("total_latency_ms", 0.0)),
        "summary_path": summary_path,
    }
    row.update(_row_params(profile))
    return row


def _naive_rag_profiles(selected_profiles):
    by_ctx = {}
    for p in selected_profiles:
        ctx = p["max_context_sentences"]
        key = f"naive_ctx{ctx}"
        if key in by_ctx:
            continue
        q = dict(p)
        q["profile"] = key
        q["max_context_sentences"] = ctx
        by_ctx[key] = q
    return list(by_ctx.values())


def _execute_rag_task(task: dict):
    cfg = RagConfig(**task["cfg"])
    _, summary = execute_rag_experiment(
        cfg,
        show_progress=False,
        precomputed_retrieval_path=task.get("precomputed_retrieval_path"),
    )
    return task["meta"], summary


def _run_rag_tasks(tasks, rag_workers: int, desc: str):
    results = []
    if not tasks:
        return results

    if rag_workers > 1:
        with ProcessPoolExecutor(max_workers=rag_workers) as executor:
            futures = [executor.submit(_execute_rag_task, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
                results.append(fut.result())
    else:
        for task in tqdm(tasks, total=len(tasks), desc=desc, leave=False):
            results.append(_execute_rag_task(task))
    return results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_retrieval = parse_bool(args.run_retrieval)
    run_rag = parse_bool(args.run_rag)
    reuse_retrieval = parse_bool(args.reuse_retrieval)
    include_naive = parse_bool(args.include_naive_baseline)
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    if not generators:
        raise ValueError("--generators cannot be empty.")
    for g in generators:
        if g not in {"heuristic", "hf", "oracle", "openai_compat", "vllm"}:
            raise ValueError(f"Unsupported generator: {g}")
    if args.rag_workers < 1:
        raise ValueError("--rag-workers must be >= 1.")
    if args.rag_topk_profiles < 0:
        raise ValueError("--rag-topk-profiles must be >= 0.")

    profiles = _build_grid(args)
    run_stamp = timestamp_for_filename()
    base_dir = Path(args.output_root) / f"grid_compare_{run_stamp}"
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    write_json(logs_dir / "grid_profiles.json", {"profiles": profiles})
    print(f"[grid] output root: {base_dir}")
    print(f"[grid] combinations: {len(profiles)}")
    print(f"[grid] rag_workers: {args.rag_workers}")
    print(f"[grid] reuse_retrieval: {reuse_retrieval}")

    rows = []
    grid_retrieval_rows = []
    grid_retrieval_artifacts = {}

    need_grid_retrieval_pass = run_retrieval or (run_rag and (reuse_retrieval or args.rag_topk_profiles > 0))
    if need_grid_retrieval_pass:
        if not run_retrieval:
            print("[grid] retrieval prepass enabled (for rag reuse/top-k selection)")
        for profile in tqdm(profiles, total=len(profiles), desc="Grid retrieval profiles", leave=False):
            cfg = _retrieval_cfg(args, base_dir, args.grid_method, profile)
            _, summary = execute_retrieval_experiment(cfg, show_progress=False)
            summary_path = str(Path(cfg.output_dir) / "retrieval_summary.json")
            query_path = str(Path(cfg.output_dir) / "retrieval_query_results.jsonl")
            row = _retrieval_row(profile, args.grid_method, summary, summary_path)
            grid_retrieval_rows.append(row)
            grid_retrieval_artifacts[profile["profile"]] = {
                "profile": profile,
                "summary": summary,
                "summary_path": summary_path,
                "query_path": query_path,
            }
            if run_retrieval:
                rows.append(row)

    rag_profiles = profiles
    if run_rag and args.rag_topk_profiles > 0:
        ranked = sorted(grid_retrieval_rows, key=lambda r: r[args.rag_topk_metric], reverse=True)
        topk = min(args.rag_topk_profiles, len(ranked))
        selected_ids = [row["profile"] for row in ranked[:topk]]
        selected = set(selected_ids)
        rag_profiles = [p for p in profiles if p["profile"] in selected]
        write_json(
            logs_dir / "rag_selected_profiles.json",
            {
                "selection_metric": args.rag_topk_metric,
                "requested_topk": args.rag_topk_profiles,
                "selected_topk": topk,
                "selected_profiles": selected_ids,
            },
        )
        print(
            f"[grid] rag profile prefilter: {len(rag_profiles)}/{len(profiles)} "
            f"(metric={args.rag_topk_metric}, topk={topk})"
        )

    if run_rag:
        rag_tasks = []
        for profile in rag_profiles:
            for generator in generators:
                cfg = _rag_cfg(args, base_dir, args.grid_method, generator, profile)
                precomputed_retrieval_path = None
                if reuse_retrieval:
                    artifact = grid_retrieval_artifacts.get(profile["profile"])
                    if artifact is None:
                        raise RuntimeError(
                            "Missing retrieval artifact for profile "
                            f"{profile['profile']} while reuse_retrieval=true."
                        )
                    precomputed_retrieval_path = artifact["query_path"]
                rag_tasks.append(
                    {
                        "cfg": cfg.__dict__,
                        "precomputed_retrieval_path": precomputed_retrieval_path,
                        "meta": {
                            "profile": profile,
                            "method": args.grid_method,
                            "generator": generator,
                            "summary_path": str(Path(cfg.output_dir) / "rag_summary.json"),
                        },
                    }
                )

        rag_results = _run_rag_tasks(
            rag_tasks,
            rag_workers=args.rag_workers,
            desc=f"Grid RAG tasks[{args.grid_method}]",
        )
        for meta, summary in rag_results:
            rows.append(
                _rag_row(
                    profile=meta["profile"],
                    method=meta["method"],
                    generator=meta["generator"],
                    summary=summary,
                    summary_path=meta["summary_path"],
                )
            )

    if include_naive:
        naive_profiles = _naive_rag_profiles(rag_profiles) if run_rag else []
        naive_retrieval_artifacts = {}

        if run_retrieval:
            ref = profiles[0]
            naive_retrieval_profile = dict(ref)
            naive_retrieval_profile["profile"] = "naive_retrieval"
            cfg = _retrieval_cfg(args, base_dir, "naive_graphrag", naive_retrieval_profile)
            _, summary = execute_retrieval_experiment(cfg, show_progress=False)
            naive_retrieval_artifacts[naive_retrieval_profile["profile"]] = {
                "profile": naive_retrieval_profile,
                "summary": summary,
                "summary_path": str(Path(cfg.output_dir) / "retrieval_summary.json"),
                "query_path": str(Path(cfg.output_dir) / "retrieval_query_results.jsonl"),
            }
            rows.append(
                _retrieval_row(
                    profile=naive_retrieval_profile,
                    method="naive_graphrag",
                    summary=summary,
                    summary_path=str(Path(cfg.output_dir) / "retrieval_summary.json"),
                )
            )

        if run_rag and reuse_retrieval:
            for profile in tqdm(naive_profiles, total=len(naive_profiles), desc="Naive retrieval prepass", leave=False):
                if profile["profile"] in naive_retrieval_artifacts:
                    continue
                cfg = _retrieval_cfg(args, base_dir, "naive_graphrag", profile)
                _, summary = execute_retrieval_experiment(cfg, show_progress=False)
                naive_retrieval_artifacts[profile["profile"]] = {
                    "profile": profile,
                    "summary": summary,
                    "summary_path": str(Path(cfg.output_dir) / "retrieval_summary.json"),
                    "query_path": str(Path(cfg.output_dir) / "retrieval_query_results.jsonl"),
                }

        if run_rag:
            naive_tasks = []
            for profile in naive_profiles:
                for generator in generators:
                    cfg = _rag_cfg(args, base_dir, "naive_graphrag", generator, profile)
                    precomputed_retrieval_path = None
                    if reuse_retrieval:
                        artifact = naive_retrieval_artifacts.get(profile["profile"])
                        if artifact is None:
                            raise RuntimeError(
                                "Missing naive retrieval artifact for profile "
                                f"{profile['profile']} while reuse_retrieval=true."
                            )
                        precomputed_retrieval_path = artifact["query_path"]
                    naive_tasks.append(
                        {
                            "cfg": cfg.__dict__,
                            "precomputed_retrieval_path": precomputed_retrieval_path,
                            "meta": {
                                "profile": profile,
                                "method": "naive_graphrag",
                                "generator": generator,
                                "summary_path": str(Path(cfg.output_dir) / "rag_summary.json"),
                            },
                        }
                    )

            naive_results = _run_rag_tasks(
                naive_tasks,
                rag_workers=args.rag_workers,
                desc="Grid RAG tasks[naive]",
            )
            for meta, summary in naive_results:
                rows.append(
                    _rag_row(
                        profile=meta["profile"],
                        method=meta["method"],
                        generator=meta["generator"],
                        summary=summary,
                        summary_path=meta["summary_path"],
                    )
                )

    rows.sort(key=lambda r: (r["task"], r["generator"], r["method"], r["profile"]))

    md_table = _rows_to_markdown(rows)
    title = "**Total Eval Summary**"
    summary_md = logs_dir / "grid_summary.md"
    summary_json = logs_dir / "grid_summary.json"
    summary_csv = logs_dir / "grid_summary.csv"

    summary_md.write_text(title + "\n\n" + md_table + "\n", encoding="utf-8")
    write_json(summary_json, {"rows": rows})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "profile",
                "method",
                "generator",
                "samples",
                "sf_recall",
                "sf_precision",
                "em",
                "f1",
                "retrieval_ms",
                "total_ms",
                "max_anchors",
                "candidate_top_t",
                "seed_k",
                "pair_top_lp",
                "corridor_top_bc",
                "trim_rho",
                "tau",
                "max_context_sentences",
                "summary_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("[grid] done")
    print("[grid][summary-table]")
    print(title)
    print(md_table)
    print(f"[grid] saved summary md: {summary_md}")
    print(f"[grid] saved summary json: {summary_json}")
    print(f"[grid] saved summary csv: {summary_csv}")


if __name__ == "__main__":
    main()
