import argparse
import json
from pathlib import Path

from .global_index import load_or_build_global_index
from .utils import parse_bool


def _build_parser():
    parser = argparse.ArgumentParser(description="Build/reuse global corpus KG index for EffiRAG.")
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="outputs/index_cache")
    parser.add_argument("--force-rebuild", type=str, default="false")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    force_rebuild = parse_bool(args.force_rebuild)
    graph, meta = load_or_build_global_index(
        corpus_path=args.corpus_path,
        cache_dir=args.cache_dir,
        force_rebuild=force_rebuild,
    )

    index_dir = Path(meta.get("index_dir", args.cache_dir))
    summary_path = index_dir / "index_summary.json"
    payload = {
        "corpus_path": str(Path(args.corpus_path).resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "force_rebuild": bool(force_rebuild),
        "cache_hit": bool(meta.get("cache_hit", False)),
        "memory_graph_nodes": int(graph.number_of_nodes()),
        "memory_graph_edges": int(graph.number_of_edges()),
        "meta": meta,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Index ready")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
