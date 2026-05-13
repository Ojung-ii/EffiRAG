#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = _strip(line)
            if not text:
                continue
            rows.append(dict(json.loads(text) or {}))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_md(rows: Iterable[Mapping[str, Any]]) -> str:
    rows_list = list(rows)
    header = [
        "# LLM Judge Aggregation",
        "",
        "| dataset | true_system | true_variant | n | llm_sufficient_rate | llm_minimal_sufficient_rate | avg_score | answer_surface_present_rate | equivalent_evidence_rate | avg_minimal_evidence_count | empty_minimal_evidence_rate | judge_error_rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = []
    for r in rows_list:
        body.append(
            "| {dataset} | {true_system} | {true_variant} | {n} | {llm_sufficient_rate:.4f} | {llm_minimal_sufficient_rate:.4f} | {avg_score:.4f} | {answer_surface_present_rate:.4f} | {equivalent_evidence_rate:.4f} | {avg_minimal_evidence_count:.4f} | {empty_minimal_evidence_rate:.4f} | {judge_error_rate:.4f} |".format(
                **r
            )
        )
    return "\n".join(header + body + [""])


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate judged JSONL outputs using hidden mapping.")
    ap.add_argument("--round-dir", required=True, help="outputs/llm_judge_input_round/<timestamp>")
    ap.add_argument("--judge-output-dir", required=True, help="Directory containing judged JSONL files")
    ap.add_argument("--output-json", default="", help="Optional output json path")
    ap.add_argument("--output-md", default="", help="Optional output markdown path")
    args = ap.parse_args()

    round_dir = Path(args.round_dir).resolve()
    judge_out_dir = Path(args.judge_output_dir).resolve()
    mapping_path = round_dir / "hidden_mapping" / "record_mapping.jsonl"
    if not mapping_path.exists():
        raise FileNotFoundError(f"missing mapping file: {mapping_path}")

    mapping_rows = _read_jsonl(mapping_path)
    by_id = {str(row.get("judge_id")): dict(row) for row in mapping_rows}

    judged_rows: List[Dict[str, Any]] = []
    for fp in sorted(judge_out_dir.glob("*.jsonl")):
        judged_rows.extend(_read_jsonl(fp))

    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    missing_mapping = 0
    for row in judged_rows:
        judge_id = _strip(row.get("judge_id"))
        meta = by_id.get(judge_id)
        if meta is None:
            missing_mapping += 1
            continue
        ds = _strip(meta.get("dataset"))
        ts = _strip(meta.get("true_system"))
        tv = _strip(meta.get("true_variant"))
        merged = dict(row)
        merged["_dataset"] = ds
        merged["_true_system"] = ts
        merged["_true_variant"] = tv
        grouped[(ds, ts, tv)].append(merged)

    summary_rows: List[Dict[str, Any]] = []
    for (ds, ts, tv), rows in sorted(grouped.items()):
        n = len(rows)
        if n <= 0:
            continue
        scores = [_safe_float(r.get("score"), 0.0) for r in rows if _strip(r.get("score")) != ""]
        sufficient = [1.0 if _safe_float(r.get("score"), -1.0) >= 3.0 else 0.0 for r in rows]
        minimal = [1.0 if _safe_float(r.get("score"), -1.0) == 4.0 else 0.0 for r in rows]
        answer_surface = [1.0 if _as_bool(r.get("answer_surface_present")) else 0.0 for r in rows]
        equivalent = [1.0 if _as_bool(r.get("equivalent_evidence")) else 0.0 for r in rows]
        minimal_counts = [
            len(list(r.get("minimal_evidence_ids", []) or []))
            for r in rows
            if isinstance(r.get("minimal_evidence_ids"), list)
        ]
        empty_minimal = [
            1.0 if len(list(r.get("minimal_evidence_ids", []) or [])) == 0 else 0.0
            for r in rows
            if isinstance(r.get("minimal_evidence_ids"), list)
        ]
        error_flags = [1.0 if _strip(r.get("error")) else 0.0 for r in rows]

        def _mean(vals: List[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else 0.0

        summary_rows.append(
            {
                "dataset": ds,
                "true_system": ts,
                "true_variant": tv,
                "n": n,
                "llm_sufficient_rate": _mean(sufficient),
                "llm_minimal_sufficient_rate": _mean(minimal),
                "avg_score": _mean(scores),
                "answer_surface_present_rate": _mean(answer_surface),
                "equivalent_evidence_rate": _mean(equivalent),
                "avg_minimal_evidence_count": _mean([float(x) for x in minimal_counts]),
                "empty_minimal_evidence_rate": _mean(empty_minimal),
                "judge_error_rate": _mean(error_flags),
            }
        )

    out_json = Path(args.output_json).resolve() if _strip(args.output_json) else (round_dir / "aggregation" / "judge_aggregation_summary.json")
    out_md = Path(args.output_md).resolve() if _strip(args.output_md) else (round_dir / "aggregation" / "judge_aggregation_summary.md")

    payload = {
        "generated_at_utc": _utc_now_iso(),
        "round_dir": str(round_dir),
        "judge_output_dir": str(judge_out_dir),
        "missing_mapping_count": int(missing_mapping),
        "summary_rows": summary_rows,
    }
    _write_json(out_json, payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(summary_rows), encoding="utf-8")

    print(json.dumps({"rows": len(summary_rows), "missing_mapping_count": int(missing_mapping)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
