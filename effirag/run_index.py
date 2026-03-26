import argparse
import json
from pathlib import Path

from .global_index import load_or_build_global_index
from .utils import parse_bool

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _build_parser():
    parser = argparse.ArgumentParser(description="Build/reuse global corpus KG index for EffiRAG.")
    parser.add_argument("--corpus-path", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="outputs/index_cache")
    parser.add_argument("--force-rebuild", type=str, default="false")
    parser.add_argument("--prebuilt-igraph-path", type=str, default="")
    parser.add_argument("--prebuilt-igraph-format", type=str, default="hipporag_pickle")
    parser.add_argument("--prebuilt-entity-token-limit", type=int, default=6)
    parser.add_argument("--openie-mode", type=str, default="llm", choices=["llm", "lexical"])
    parser.add_argument("--openie-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--openie-text-max-chars", type=int, default=2200)
    parser.add_argument("--openie-max-new-tokens", type=int, default=256)
    parser.add_argument("--openie-local-files-only", type=str, default="true")
    parser.add_argument("--openie-retry-attempts", type=int, default=3)
    parser.add_argument("--openie-retry-backoff-sec", type=float, default=0.2)
    parser.add_argument("--openie-error-sample-limit", type=int, default=20)
    parser.add_argument("--openie-api-base-url", type=str, default="")
    parser.add_argument("--openie-api-key", type=str, default="")
    parser.add_argument("--openie-api-timeout-sec", type=float, default=120.0)
    parser.add_argument("--openie-parallel-workers", type=int, default=4)
    parser.add_argument("--openie-log-every", type=int, default=200)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    force_rebuild = parse_bool(args.force_rebuild)
    openie_local_files_only = parse_bool(args.openie_local_files_only)
    corpus_path = str(args.corpus_path or "").strip()
    prebuilt_igraph_path = str(args.prebuilt_igraph_path or "").strip()
    if not corpus_path and not prebuilt_igraph_path:
        raise ValueError("Provide either --corpus-path or --prebuilt-igraph-path.")
    graph = None
    meta = {}
    summary_path = None
    payload = {}

    print(
        "[RunIndex] "
        f"mode={args.openie_mode} workers={int(args.openie_parallel_workers)} "
        f"log_every={int(args.openie_log_every)} timeout={float(args.openie_api_timeout_sec):.1f}s",
        flush=True,
    )
    if str(args.openie_api_base_url or "").strip():
        print(f"[RunIndex] openie_api_base_url={str(args.openie_api_base_url).strip()}", flush=True)

    for stage in tqdm(
        ["build_or_load_index", "write_summary"],
        total=2,
        desc="Run index",
        unit="stage",
        leave=True,
    ):
        if stage == "build_or_load_index":
            print("[RunIndex] stage=build_or_load_index start", flush=True)
            graph, meta = load_or_build_global_index(
                corpus_path=corpus_path,
                cache_dir=args.cache_dir,
                force_rebuild=force_rebuild,
                prebuilt_igraph_path=prebuilt_igraph_path,
                prebuilt_igraph_format=str(args.prebuilt_igraph_format or "hipporag_pickle"),
                prebuilt_entity_token_limit=int(args.prebuilt_entity_token_limit),
                openie_mode=args.openie_mode,
                openie_model_name=args.openie_model_name,
                openie_text_max_chars=args.openie_text_max_chars,
                openie_max_new_tokens=args.openie_max_new_tokens,
                openie_local_files_only=openie_local_files_only,
                openie_retry_attempts=args.openie_retry_attempts,
                openie_retry_backoff_sec=args.openie_retry_backoff_sec,
                openie_error_sample_limit=args.openie_error_sample_limit,
                openie_api_base_url=str(args.openie_api_base_url or ""),
                openie_api_key=str(args.openie_api_key or ""),
                openie_api_timeout_sec=float(args.openie_api_timeout_sec),
                openie_parallel_workers=int(args.openie_parallel_workers),
                openie_log_every=int(args.openie_log_every),
                show_progress=True,
            )
            print(
                "[RunIndex] stage=build_or_load_index done "
                f"(cache_hit={bool(meta.get('cache_hit', False))})",
                flush=True,
            )
        elif stage == "write_summary":
            print("[RunIndex] stage=write_summary start", flush=True)
            index_dir = Path(meta.get("index_dir", args.cache_dir))
            summary_path = index_dir / "index_summary.json"
            payload = {
                "corpus_path": str(Path(corpus_path).resolve()) if corpus_path else "",
                "prebuilt_igraph_path": str(Path(prebuilt_igraph_path).resolve()) if prebuilt_igraph_path else "",
                "prebuilt_igraph_format": str(args.prebuilt_igraph_format or "hipporag_pickle"),
                "prebuilt_entity_token_limit": int(args.prebuilt_entity_token_limit),
                "cache_dir": str(Path(args.cache_dir).resolve()),
                "force_rebuild": bool(force_rebuild),
                "openie_mode": str(args.openie_mode),
                "openie_model_name": str(args.openie_model_name),
                "openie_text_max_chars": int(args.openie_text_max_chars),
                "openie_max_new_tokens": int(args.openie_max_new_tokens),
                "openie_local_files_only": bool(openie_local_files_only),
                "openie_retry_attempts": int(args.openie_retry_attempts),
                "openie_retry_backoff_sec": float(args.openie_retry_backoff_sec),
                "openie_error_sample_limit": int(args.openie_error_sample_limit),
                "openie_api_base_url": str(args.openie_api_base_url or ""),
                "openie_api_timeout_sec": float(args.openie_api_timeout_sec),
                "openie_parallel_workers": int(args.openie_parallel_workers),
                "openie_log_every": int(args.openie_log_every),
                "cache_hit": bool(meta.get("cache_hit", False)),
                "memory_graph_nodes": int(graph.number_of_nodes()),
                "memory_graph_edges": int(graph.number_of_edges()),
                "meta": meta,
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[RunIndex] stage=write_summary done path={summary_path}", flush=True)

    print("Index ready")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
