#!/usr/bin/env python3
"""
Compare legacy SOTA replay summaries against strict unified summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


DATASET_ORDER = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]

QA_KEYS = ["f1", "em"]
EVIDENCE_KEYS = ["supporting_fact_recall", "supporting_fact_precision", "supporting_fact_f1"]
EFFICIENCY_KEYS = [
    "prompt_tokens_avg",
    "avg_context_tokens",
    "retrieval_latency_ms",
    "total_latency_ms",
]


def _tokens(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _find_summary(root: Path, dataset: str) -> Path | None:
    candidates = []
    dataset_root = root / dataset
    if dataset_root.exists():
        candidates.extend(dataset_root.rglob("rag_summary.json"))
    candidates.extend(root.glob(f"**/{dataset}/**/rag_summary.json"))
    candidates = sorted({path.resolve() for path in candidates})
    return candidates[-1] if candidates else None


def _load_summary(path: Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(mapping: Mapping[str, Any], key: str) -> float:
    try:
        return float(mapping.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _metric_rows(datasets: List[str], legacy: Mapping[str, Mapping[str, Any]], unified: Mapping[str, Mapping[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for key in keys:
            legacy_value = _safe_float(legacy.get(dataset, {}), key)
            unified_value = _safe_float(unified.get(dataset, {}), key)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": key,
                    "legacy": legacy_value,
                    "unified": unified_value,
                    "delta": unified_value - legacy_value,
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "metric", "legacy", "unified", "delta"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_md(delta_rows: List[Dict[str, Any]], legacy_paths: Mapping[str, str], unified_paths: Mapping[str, str]) -> str:
    lines = [
        "# Legacy SOTA Replay vs Strict Unified",
        "",
        "| dataset | metric | legacy | unified | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in delta_rows:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | "
            f"{row['legacy']:.6f} | {row['unified']:.6f} | {row['delta']:+.6f} |"
        )
    lines.extend(["", "## Sources", ""])
    for dataset in sorted(set(legacy_paths) | set(unified_paths)):
        lines.append(f"- {dataset} legacy: {legacy_paths.get(dataset, '')}")
        lines.append(f"- {dataset} unified: {unified_paths.get(dataset, '')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy SOTA replay and strict unified outputs.")
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--unified-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique", "popqa"])
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: outputs/phase6_strict_unified_comparison/<timestamp>",
    )
    args = parser.parse_args()

    requested = set(_tokens(args.datasets))
    datasets = [dataset for dataset in DATASET_ORDER if dataset in requested]
    if not datasets:
        raise ValueError(f"No valid dataset requested. got={sorted(requested)}")

    legacy_root = Path(args.legacy_root).resolve()
    unified_root = Path(args.unified_root).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "phase6_strict_unified_comparison" / stamp
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_summaries: Dict[str, Dict[str, Any]] = {}
    unified_summaries: Dict[str, Dict[str, Any]] = {}
    legacy_paths: Dict[str, str] = {}
    unified_paths: Dict[str, str] = {}

    for dataset in datasets:
        legacy_path = _find_summary(legacy_root, dataset)
        unified_path = _find_summary(unified_root, dataset)
        legacy_paths[dataset] = str(legacy_path or "")
        unified_paths[dataset] = str(unified_path or "")
        legacy_summaries[dataset] = _load_summary(legacy_path)
        unified_summaries[dataset] = _load_summary(unified_path)

    qa_rows = _metric_rows(datasets, legacy_summaries, unified_summaries, QA_KEYS)
    evidence_rows = _metric_rows(datasets, legacy_summaries, unified_summaries, EVIDENCE_KEYS)
    efficiency_rows = _metric_rows(datasets, legacy_summaries, unified_summaries, EFFICIENCY_KEYS)
    delta_rows = qa_rows + evidence_rows + efficiency_rows

    (output_dir / "delta_rows.json").write_text(
        json.dumps(delta_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "qa_table.csv", qa_rows)
    _write_csv(output_dir / "evidence_table.csv", evidence_rows)
    _write_csv(output_dir / "efficiency_table.csv", efficiency_rows)
    (output_dir / "summary.md").write_text(
        _summary_md(delta_rows, legacy_paths, unified_paths),
        encoding="utf-8",
    )

    print(str(output_dir))


if __name__ == "__main__":
    main()
