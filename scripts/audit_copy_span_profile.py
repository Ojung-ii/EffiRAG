#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Dict, List

from effirag.config import RagConfig
from effirag.profiles import (
    COPY_SPAN_CANONICAL_VARIANT,
    COPY_SPAN_DATASET_LOCKS,
    COPY_SPAN_LOCK_ROUND_ROOT,
    COPY_SPAN_PROFILE_NAME,
    DEPRECATED_OR_LEGACY_CONFIG_FIELDS,
    expected_copy_span_config,
)
from effirag.utils import load_yaml


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "popqa")
RAG_DEFAULTS = {}
for f in fields(RagConfig):
    if f.default is not MISSING:
        RAG_DEFAULTS[f.name] = f.default
    elif f.default_factory is not MISSING:  # pragma: no cover - not used currently
        RAG_DEFAULTS[f.name] = f.default_factory()


def _values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) is bool(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    return expected == actual


def _audit_dataset(config_root: Path, dataset: str) -> Dict[str, Any]:
    config_path = config_root / dataset / "rag.yaml"
    expected = expected_copy_span_config(dataset)
    checked_keys = sorted(expected.keys())

    result: Dict[str, Any] = {
        "dataset": dataset,
        "config_path": str(config_path),
        "checked_keys": checked_keys,
        "pass": False,
        "mismatched_keys": [],
        "deprecated_keys_present": [],
    }

    if not config_path.exists():
        result["mismatched_keys"] = [
            {
                "key": "__config_file__",
                "expected": "exists",
                "actual": "missing",
            }
        ]
        return result

    payload = load_yaml(config_path)
    mismatches: List[Dict[str, Any]] = []
    for key in checked_keys:
        expected_value = expected.get(key)
        actual_value = payload.get(key, RAG_DEFAULTS.get(key, "__missing__"))
        if not _values_equal(expected_value, actual_value):
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )

    deprecated_present = sorted(k for k in DEPRECATED_OR_LEGACY_CONFIG_FIELDS if k in payload)
    result["mismatched_keys"] = mismatches
    result["deprecated_keys_present"] = deprecated_present
    result["pass"] = len(mismatches) == 0
    return result


def _render_markdown(rows: List[Dict[str, Any]], config_root: Path) -> str:
    lines: List[str] = []
    lines.append("# Copy-Span Profile Audit")
    lines.append("")
    lines.append(f"- profile: `{COPY_SPAN_PROFILE_NAME}`")
    lines.append(f"- canonical_variant: `{COPY_SPAN_CANONICAL_VARIANT}`")
    lines.append(f"- lock_round_root: `{COPY_SPAN_LOCK_ROUND_ROOT}`")
    lines.append(f"- config_root_checked: `{config_root}`")
    lines.append("")
    for row in rows:
        lines.append(f"## {row['dataset']}")
        lines.append(f"- dataset: `{row['dataset']}`")
        lines.append(f"- checked_keys: `{', '.join(row['checked_keys'])}`")
        lines.append(f"- pass/fail: `{'PASS' if row['pass'] else 'FAIL'}`")
        if row["mismatched_keys"]:
            mismatch_items = []
            for item in row["mismatched_keys"]:
                mismatch_items.append(
                    f"{item['key']} (expected={item['expected']!r}, actual={item['actual']!r})"
                )
            lines.append(f"- mismatched_keys: `{'; '.join(mismatch_items)}`")
        else:
            lines.append("- mismatched_keys: `(none)`")
        if row["deprecated_keys_present"]:
            lines.append(f"- deprecated_keys_present: `{', '.join(row['deprecated_keys_present'])}`")
        else:
            lines.append("- deprecated_keys_present: `(none)`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen copy-span profile invariance.")
    parser.add_argument(
        "--config-root",
        required=True,
        help="Path to canonical copy-span config root (contains dataset directories).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where markdown/json audit reports will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_root = Path(args.config_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_audit_dataset(config_root=config_root, dataset=dataset) for dataset in DATASETS]
    has_failures = any(not row["pass"] for row in rows)
    has_deprecated = any(bool(row["deprecated_keys_present"]) for row in rows)

    report = {
        "profile": COPY_SPAN_PROFILE_NAME,
        "canonical_variant": COPY_SPAN_CANONICAL_VARIANT,
        "lock_round_root": COPY_SPAN_LOCK_ROUND_ROOT,
        "config_root_checked": str(config_root),
        "dataset_locks_defined": sorted(COPY_SPAN_DATASET_LOCKS.keys()),
        "results": rows,
        "has_failures": has_failures,
        "has_deprecated_keys": has_deprecated,
    }

    md_path = out_dir / "copy_span_profile_audit.md"
    json_path = out_dir / "copy_span_profile_audit.json"
    md_path.write_text(_render_markdown(rows=rows, config_root=config_root), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if has_deprecated:
        for row in rows:
            if row["deprecated_keys_present"]:
                print(
                    (
                        f"[WARN] {row['dataset']} contains deprecated/legacy keys: "
                        f"{', '.join(row['deprecated_keys_present'])}"
                    ),
                    file=sys.stderr,
                )

    print(f"[copy-span-audit] markdown: {md_path}")
    print(f"[copy-span-audit] json: {json_path}")
    if has_failures:
        print("[copy-span-audit] locked-key mismatches detected.", file=sys.stderr)
        return 1
    print("[copy-span-audit] all locked keys matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
