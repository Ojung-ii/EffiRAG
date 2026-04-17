import argparse
from pathlib import Path

from . import datasets as _datasets  # noqa: F401
from .registry import get_dataset_loader
from .utils import parse_bool, write_json

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _csv_to_list(raw: str):
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _find_source_path(source_root: str, dataset: str):
    if not source_root:
        return None
    root = Path(source_root)
    for ext in ("json", "jsonl"):
        path = root / f"{dataset}.{ext}"
        if path.exists():
            return str(path)
    return None


def _sample_to_row(sample):
    return {
        "id": sample.qid,
        "question": sample.question,
        "answer": sample.answer,
        "context": [[doc.title, list(doc.sentences)] for doc in sample.contexts],
        "supporting_facts": [[title, int(sent_idx)] for title, sent_idx in sample.supporting_facts],
        "metadata": dict(sample.metadata),
    }


def _is_demo_samples(samples):
    if not samples:
        return False
    marked = 0
    for sample in samples[: min(5, len(samples))]:
        if str(sample.metadata.get("source", "")).lower() == "demo":
            marked += 1
    return marked == min(5, len(samples))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build normalized QA datasets for EffiRAG.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,musique,2wikimultihopqa,popqa",
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", type=str, default="data/qa")
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Optional local directory containing <dataset>.json or <dataset>.jsonl files.",
    )
    parser.add_argument("--overwrite", type=str, default="false")
    parser.add_argument(
        "--allow-demo-fallback",
        type=str,
        default="false",
        help="Allow writing demo fallback rows when real data could not be loaded.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    datasets = _csv_to_list(args.datasets)
    if not datasets:
        raise ValueError("--datasets cannot be empty")

    overwrite = parse_bool(args.overwrite)
    allow_demo = parse_bool(args.allow_demo_fallback)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "split": args.split,
        "limit": args.limit,
        "datasets": [],
    }

    for dataset in tqdm(
        datasets,
        total=len(datasets),
        desc="Prepare datasets",
        unit="dataset",
        leave=False,
    ):
        loader = get_dataset_loader(dataset)
        source_path = _find_source_path(args.source_root, dataset)

        samples = loader(split=args.split, limit=args.limit, data_path=source_path)
        if _is_demo_samples(samples) and not allow_demo:
            raise RuntimeError(
                f"Dataset '{dataset}' resolved to demo fallback. "
                "Provide a valid --source-root or network access, or pass --allow-demo-fallback true."
            )

        rows = [
            _sample_to_row(sample)
            for sample in tqdm(
                samples,
                total=len(samples),
                desc=f"Normalize[{dataset}]",
                unit="sample",
                leave=False,
            )
        ]

        out_path = output_root / f"{dataset}.json"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {out_path}. Use --overwrite true to replace it.")
        write_json(out_path, rows)

        manifest["datasets"].append(
            {
                "dataset": dataset,
                "n_samples": len(rows),
                "source_path": source_path or "",
                "output_path": str(out_path),
            }
        )
        print(f"[prepare] dataset={dataset} samples={len(rows)} output={out_path}")

    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"[prepare] manifest={manifest_path}")


if __name__ == "__main__":
    main()
