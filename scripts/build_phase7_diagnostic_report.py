#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _discover_summaries(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("diagnostic_summary.json")):
        row = _read_json(path, {})
        if not isinstance(row, dict):
            continue
        row = dict(row)
        row["_summary_path"] = str(path.resolve())
        if not row.get("dataset"):
            parts = path.parts
            if len(parts) >= 3:
                row["dataset"] = parts[-3]
        if not row.get("variant"):
            parts = path.parts
            if len(parts) >= 2:
                row["variant"] = parts[-2]
        rows.append(row)
    return rows


def _fmt4(value: Any) -> str:
    return f"{_safe_float(value, 0.0):.4f}"


def _fmt2(value: Any) -> str:
    return f"{_safe_float(value, 0.0):.2f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def _write_compare_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "variant",
        "n",
        "EM",
        "F1",
        "SF_recall",
        "SF_precision",
        "avg_context_tokens",
        "F1_per_1k",
        "phase1_gold_hit_rate",
        "phase2_gold_hit_rate",
        "selected_gold_hit_rate",
        "rendered_gold_hit_rate",
        "selected_to_rendered_match_rate",
        "retrieval_ms",
        "total_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": str(row.get("dataset", "") or ""),
                    "variant": str(row.get("variant", "") or ""),
                    "n": int(_safe_int(row.get("n", 0), 0)),
                    "EM": _fmt4(row.get("EM", 0.0)),
                    "F1": _fmt4(row.get("F1", 0.0)),
                    "SF_recall": _fmt4(row.get("SF_recall", 0.0)),
                    "SF_precision": _fmt4(row.get("SF_precision", 0.0)),
                    "avg_context_tokens": _fmt2(row.get("avg_context_tokens", 0.0)),
                    "F1_per_1k": _fmt4(row.get("F1_per_1k", 0.0)),
                    "phase1_gold_hit_rate": _fmt4(row.get("phase1_gold_hit_rate", 0.0)),
                    "phase2_gold_hit_rate": _fmt4(row.get("phase2_gold_hit_rate", 0.0)),
                    "selected_gold_hit_rate": _fmt4(row.get("selected_gold_hit_rate", 0.0)),
                    "rendered_gold_hit_rate": _fmt4(row.get("rendered_gold_hit_rate", 0.0)),
                    "selected_to_rendered_match_rate": _fmt4(row.get("selected_to_rendered_match_rate", 0.0)),
                    "retrieval_ms": _fmt2(row.get("avg_retrieval_ms", 0.0)),
                    "total_ms": _fmt2(row.get("avg_total_ms", 0.0)),
                }
            )


def _key(dataset: str, variant: str) -> Tuple[str, str]:
    return (str(dataset or "").strip().lower(), str(variant or "").strip().lower())


def _lookup(rows: List[Dict[str, Any]], dataset: str, variant: str) -> Dict[str, Any]:
    for row in rows:
        if _key(row.get("dataset", ""), row.get("variant", "")) == _key(dataset, variant):
            return row
    return {}


def _diagnosis_lines(rows: List[Dict[str, Any]], dataset: str) -> List[str]:
    full = _lookup(rows, dataset, "full")
    b2 = _lookup(rows, dataset, "phase2_budget_x2")
    p1 = _lookup(rows, dataset, "phase1_only")
    n2 = _lookup(rows, dataset, "no_phase2_refinement")
    legacy = _lookup(rows, dataset, "legacy_compatible_render")

    lines: List[str] = []
    if full:
        phase1_hit = _safe_float(full.get("phase1_gold_hit_rate", 0.0), 0.0)
        phase2_hit = _safe_float(full.get("phase2_gold_hit_rate", 0.0), 0.0)
        selected_hit = _safe_float(full.get("selected_gold_hit_rate", 0.0), 0.0)
        rendered_hit = _safe_float(full.get("rendered_gold_hit_rate", 0.0), 0.0)
        match_rate = _safe_float(full.get("selected_to_rendered_match_rate", 0.0), 0.0)
        lines.append(
            f"- {dataset}: phase1={phase1_hit:.4f}, phase2={phase2_hit:.4f}, selected={selected_hit:.4f}, rendered={rendered_hit:.4f}, sel->ren={match_rate:.4f}"
        )
        if phase1_hit < 0.35:
            lines.append("- Is Phase I failing?: YES (seed hit-rate 자체가 낮음)")
        else:
            lines.append("- Is Phase I failing?: NO/부분적 (seed hit-rate는 확보)")
        if phase2_hit > 0.0 and selected_hit + 1e-9 < phase2_hit - 0.05:
            lines.append("- Is Phase II failing?: YES (phase2 후보 대비 selected 단계에서 손실)")
        elif p1 and _safe_float(p1.get("F1", 0.0), 0.0) > _safe_float(full.get("F1", 0.0), 0.0) + 0.01:
            lines.append("- Is Phase II failing?: POSSIBLE (phase1_only가 full보다 높음)")
        elif n2 and _safe_float(n2.get("F1", 0.0), 0.0) > _safe_float(full.get("F1", 0.0), 0.0) + 0.01:
            lines.append("- Is Phase II failing?: POSSIBLE (no_phase2_refinement이 full보다 높음)")
        else:
            lines.append("- Is Phase II failing?: 불확실/약함")
        if rendered_hit + 1e-9 < selected_hit - 0.05 or match_rate < 0.85:
            lines.append("- Is rendering/id mapping failing?: YES (selected->rendered 손실 또는 불일치)")
        elif legacy and _safe_float(legacy.get("F1", 0.0), 0.0) > _safe_float(full.get("F1", 0.0), 0.0) + 0.01:
            lines.append("- Is rendering/id mapping failing?: POSSIBLE (legacy rendering에서 성능 회복)")
        else:
            lines.append("- Is rendering/id mapping failing?: 불확실/약함")
        if b2 and _safe_float(b2.get("F1", 0.0), 0.0) > _safe_float(full.get("F1", 0.0), 0.0) + 0.01:
            lines.append("- Is budget too aggressive?: YES (phase2_budget_x2에서 개선)")
        else:
            lines.append("- Is budget too aggressive?: 불확실/약함")
    else:
        lines.append(f"- {dataset}: full variant summary not found")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Phase7 diagnostic markdown report from per-run summaries.")
    ap.add_argument("--root", type=str, required=True, help="Root directory, e.g., outputs/phase7_diag")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = _discover_summaries(root)
    rows = sorted(rows, key=lambda r: (str(r.get("dataset", "")), str(r.get("variant", ""))))
    if not rows:
        raise FileNotFoundError(f"No diagnostic_summary.json found under: {root}")

    compare_csv_path = root / "variant_comparison.csv"
    _write_compare_csv(compare_csv_path, rows)

    variant_rows = []
    for row in rows:
        variant_rows.append(
            [
                str(row.get("dataset", "")),
                str(row.get("variant", "")),
                str(int(_safe_int(row.get("n", 0), 0))),
                _fmt4(row.get("EM", 0.0)),
                _fmt4(row.get("F1", 0.0)),
                _fmt4(row.get("SF_recall", 0.0)),
                _fmt4(row.get("SF_precision", 0.0)),
                _fmt2(row.get("avg_context_tokens", 0.0)),
                _fmt4(row.get("F1_per_1k", 0.0)),
                _fmt2(row.get("avg_retrieval_ms", 0.0)),
                _fmt2(row.get("avg_total_ms", 0.0)),
            ]
        )

    hit_rows = []
    for row in rows:
        hit_rows.append(
            [
                str(row.get("dataset", "")),
                str(row.get("variant", "")),
                _fmt4(row.get("phase1_gold_hit_rate", 0.0)),
                _fmt4(row.get("phase2_gold_hit_rate", 0.0)),
                _fmt4(row.get("selected_gold_hit_rate", 0.0)),
                _fmt4(row.get("rendered_gold_hit_rate", 0.0)),
                _fmt4(row.get("selected_to_rendered_match_rate", 0.0)),
            ]
        )

    lines: List[str] = []
    lines.append("# PHASE7_DIAGNOSTIC_REPORT")
    lines.append("")
    lines.append("## Baseline previous result")
    lines.append("- 2Wiki: EM 0.0700, F1 0.0762, SF_recall 0.2900, SF_precision 0.0862, avg tokens 486.85")
    lines.append("- HotpotQA: EM 0.2100, F1 0.2603, SF_recall 0.3903, SF_precision 0.1225, avg tokens 495.49")
    lines.append("")
    lines.append("## Variant comparison table")
    lines.append(
        _table(
            [
                "dataset",
                "variant",
                "n",
                "EM",
                "F1",
                "SF_recall",
                "SF_precision",
                "avg_context_tokens",
                "F1_per_1k",
                "retrieval_ms",
                "total_ms",
            ],
            variant_rows,
        )
    )
    lines.append("")
    lines.append("## Evidence flow hit-rate table")
    lines.append(
        _table(
            [
                "dataset",
                "variant",
                "phase1_gold_hit_rate",
                "phase2_gold_hit_rate",
                "selected_gold_hit_rate",
                "rendered_gold_hit_rate",
                "selected_to_rendered_match_rate",
            ],
            hit_rows,
        )
    )
    lines.append("")
    lines.append("## Diagnosis")
    lines.extend(_diagnosis_lines(rows, "hotpotqa"))
    lines.extend(_diagnosis_lines(rows, "2wikimultihopqa"))
    lines.append("")
    lines.append("## Recommendation")
    lines.append("- Fix unit mismatch first when selected/rendered node-type mismatch or id mismatch appears.")
    lines.append("- If phase2_budget_x2 consistently improves F1/rendered_hit, increase Phase II budget or add fallback.")
    lines.append("- If phase2 candidate hit is high but selected hit drops, adjust Phase II selection objective.")
    lines.append("- If selected hit is high but rendered hit drops, prioritize selected/rendered mapping and rendering contract fixes.")
    lines.append("- Do not add intent integration yet; stabilize Phase7 evidence flow first.")
    lines.append("")
    lines.append(f"- variant_comparison_csv: {compare_csv_path}")

    report_path = root / "PHASE7_DIAGNOSTIC_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[phase7-report] {report_path}")
    print(f"[phase7-report] {compare_csv_path}")


if __name__ == "__main__":
    main()

