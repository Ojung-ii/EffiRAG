#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        header = [h.strip() for h in (f.readline().strip().split("\t")) if h.strip()]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or len(parts) == 0:
                continue
            row: Dict[str, str] = {}
            for idx, key in enumerate(header):
                row[key] = parts[idx] if idx < len(parts) else ""
            rows.append(row)
    return rows


def _r_at(summary: Dict[str, Any], k: int) -> float:
    direct = _safe_float(summary.get(f"supporting_fact_recall_at_{k}", 0.0), 0.0)
    if direct > 0.0:
        return direct
    bag = summary.get("supporting_fact_recall_at_k", {}) or {}
    if isinstance(bag, dict):
        if str(k) in bag:
            return _safe_float(bag.get(str(k), 0.0), 0.0)
        if k in bag:
            return _safe_float(bag.get(k, 0.0), 0.0)
    bag2 = summary.get("recall_at_k", {}) or {}
    if isinstance(bag2, dict):
        if str(k) in bag2:
            return _safe_float(bag2.get(str(k), 0.0), 0.0)
        if k in bag2:
            return _safe_float(bag2.get(k, 0.0), 0.0)
    return _safe_float(summary.get(f"Recall@{k}", 0.0), 0.0)


def _pick(summary: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    funnel = summary.get("stagewise_loss_funnel", {}) or {}
    for key in keys:
        if key in summary:
            return _safe_float(summary.get(key, default), default)
        if key in funnel:
            return _safe_float(funnel.get(key, default), default)
        skey = f"stagewise_{key}"
        if skey in summary:
            return _safe_float(summary.get(skey, default), default)
    return _safe_float(default, default)


def _extract_metrics(summary_path: Path) -> Dict[str, Any]:
    summary = _read_json(summary_path)
    r5 = _r_at(summary, 5)
    em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
    f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
    retrieval_ms = _safe_float(summary.get("retrieval_latency_ms", summary.get("retrieval_ms", 0.0)), 0.0)
    generation_ms = _safe_float(summary.get("generation_latency_ms", summary.get("generation_ms", 0.0)), 0.0)
    total_ms = _safe_float(summary.get("total_latency_ms", summary.get("total_ms", retrieval_ms + generation_ms)), retrieval_ms + generation_ms)

    fallback_rate = _pick(summary, "fallback_rate", default=0.0)
    strict_path_complete = _pick(summary, "strict_path_complete_rate", default=0.0)
    answer_bearing_path_hit = _pick(summary, "answer_bearing_path_hit", default=0.0)
    answer_present_generation_fail = _pick(summary, "answer_present_but_generation_fail", default=0.0)
    output_overlap = _pick(summary, "output_overlap_answer_bearing", "output_overlap_answer_bearing_rate", default=0.0)

    diagnostic_note = ""
    if r5 >= 0.20 and strict_path_complete <= 1.0e-9 and answer_bearing_path_hit <= 1.0e-9:
        diagnostic_note = "diagnostic-only / integrity check needed"

    return {
        "R@5": float(r5),
        "EM": float(em),
        "F1": float(f1),
        "retrieval_ms": float(retrieval_ms),
        "generation_ms": float(generation_ms),
        "total_ms": float(total_ms),
        "fallback_rate": float(fallback_rate),
        "strict_path_complete": float(strict_path_complete),
        "answer_bearing_path_hit": float(answer_bearing_path_hit),
        "answer_present_but_generation_fail": float(answer_present_generation_fail),
        "output_overlap_answer_bearing": float(output_overlap),
        "diagnostic_note": diagnostic_note,
    }


def _fmt_float(v: Any, nd: int = 4) -> str:
    return f"{_safe_float(v, 0.0):.{nd}f}"


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join([str(x) for x in row]) + " |")
    return "\n".join(lines)


def _variant_aggregate(variant: str, completed_rows: List[Dict[str, Any]], failed_datasets: List[str]) -> Dict[str, Any]:
    if not completed_rows:
        return {
            "variant": variant,
            "mean_EM": 0.0,
            "mean_F1": 0.0,
            "mean_retrieval_ms": 0.0,
            "mean_total_ms": 0.0,
            "completed_datasets": [],
            "failed_datasets": list(sorted(set(failed_datasets))),
        }
    return {
        "variant": variant,
        "mean_EM": float(mean([_safe_float(r.get("EM", 0.0), 0.0) for r in completed_rows])),
        "mean_F1": float(mean([_safe_float(r.get("F1", 0.0), 0.0) for r in completed_rows])),
        "mean_retrieval_ms": float(mean([_safe_float(r.get("retrieval_ms", 0.0), 0.0) for r in completed_rows])),
        "mean_total_ms": float(mean([_safe_float(r.get("total_ms", 0.0), 0.0) for r in completed_rows])),
        "completed_datasets": list(sorted({str(r.get("dataset", "")) for r in completed_rows if str(r.get("dataset", ""))})),
        "failed_datasets": list(sorted(set(failed_datasets))),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multi-dataset generalization run outputs.")
    p.add_argument("--round-root", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--run-records", default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    p.add_argument("--failures-json", default=None)
    p.add_argument("--fallback-warning-threshold", type=float, default=0.10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    round_root = Path(args.round_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else (round_root / "run_manifest.json")
    records_path = Path(args.run_records).resolve() if args.run_records else (round_root / "run_records.tsv")
    output_json = Path(args.output_json).resolve() if args.output_json else (round_root / "round_metrics.json")
    output_md = Path(args.output_md).resolve() if args.output_md else (round_root / "round_metrics.md")
    failures_json = Path(args.failures_json).resolve() if args.failures_json else (round_root / "failures.json")

    manifest = _read_json(manifest_path)
    run_records = _read_tsv(records_path)

    matrix: List[Dict[str, Any]] = list(manifest.get("matrix", []) or [])
    selected_datasets: List[str] = list((manifest.get("selection", {}) or {}).get("datasets", []) or [])
    selected_variants: List[str] = list((manifest.get("selection", {}) or {}).get("variants", []) or [])

    rec_idx: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in run_records:
        key = (str(row.get("stage", "")), str(row.get("variant", "")), str(row.get("dataset", "")))
        rec_idx[key] = row

    records_out: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for combo in matrix:
        stage = str(combo.get("stage", "s1"))
        variant = str(combo.get("variant", ""))
        dataset = str(combo.get("dataset", ""))
        profile = str(combo.get("profile", ""))
        key = (stage, variant, dataset)
        row = rec_idx.get(key, {})
        status = str(row.get("status", "missing") or "missing")
        summary_path_raw = str(row.get("summary_path", "") or "")
        summary_path = Path(summary_path_raw).resolve() if summary_path_raw else None

        entry: Dict[str, Any] = {
            "stage": stage,
            "variant": variant,
            "dataset": dataset,
            "profile": profile,
            "status": status,
            "summary_path": str(summary_path) if summary_path else "",
            "log_path": str(row.get("log_path", "") or ""),
        }

        if status in {"ok", "skipped"} and summary_path and summary_path.exists():
            try:
                entry.update(_extract_metrics(summary_path))
            except Exception as exc:
                entry["status"] = "failed"
                entry["error_message"] = f"summary_parse_failed: {exc}"
                failures.append(
                    {
                        "stage": stage,
                        "variant": variant,
                        "dataset": dataset,
                        "reason": "summary_parse_failed",
                        "detail": str(exc),
                        "summary_path": str(summary_path),
                    }
                )
            records_out.append(entry)
            continue

        if status in {"ok", "skipped"} and (not summary_path or not summary_path.exists()):
            failures.append(
                {
                    "stage": stage,
                    "variant": variant,
                    "dataset": dataset,
                    "reason": "missing_summary",
                    "detail": "summary file is missing",
                    "summary_path": summary_path_raw,
                }
            )
        elif status not in {"ok", "skipped"}:
            failures.append(
                {
                    "stage": stage,
                    "variant": variant,
                    "dataset": dataset,
                    "reason": "run_failed_or_missing",
                    "detail": str(row.get("error_message", "") or ""),
                    "summary_path": summary_path_raw,
                }
            )
        records_out.append(entry)

    dataset_order = {d: i for i, d in enumerate(selected_datasets)}
    variant_order = {v: i for i, v in enumerate(selected_variants)}

    metric_rows = [
        r
        for r in records_out
        if r.get("status") in {"ok", "skipped"} and "R@5" in r
    ]
    metric_rows.sort(key=lambda x: (dataset_order.get(str(x.get("dataset", "")), 999), variant_order.get(str(x.get("variant", "")), 999)))

    variant_summary: List[Dict[str, Any]] = []
    for variant in selected_variants:
        ok_rows = [r for r in metric_rows if str(r.get("variant", "")) == variant]
        failed_ds = [
            str(item.get("dataset", ""))
            for item in failures
            if str(item.get("variant", "")) == variant and str(item.get("dataset", ""))
        ]
        variant_summary.append(_variant_aggregate(variant, ok_rows, failed_ds))

    fallback_warnings = [
        {
            "dataset": str(r.get("dataset", "")),
            "variant": str(r.get("variant", "")),
            "fallback_rate": float(r.get("fallback_rate", 0.0)),
        }
        for r in metric_rows
        if _safe_float(r.get("fallback_rate", 0.0), 0.0) > float(args.fallback_warning_threshold)
    ]

    integrity_warnings = [
        {
            "dataset": str(r.get("dataset", "")),
            "variant": str(r.get("variant", "")),
            "note": str(r.get("diagnostic_note", "")),
        }
        for r in metric_rows
        if str(r.get("diagnostic_note", "")).strip()
    ]

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "methodology_note": "This round validates dataset generalization with a near-fixed methodology; retrieval core is unchanged.",
        "selection": manifest.get("selection", {}),
        "records": records_out,
        "variant_summary": variant_summary,
        "failures": failures,
        "warnings": {
            "fallback_threshold": float(args.fallback_warning_threshold),
            "fallback_rate": fallback_warnings,
            "diagnostic_integrity": integrity_warnings,
        },
        "files": {
            "manifest": str(manifest_path),
            "run_records": str(records_path),
            "round_metrics_json": str(output_json),
            "round_metrics_md": str(output_md),
            "failures_json": str(failures_json),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failures_json.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    headers_main = [
        "dataset",
        "variant",
        "R@5",
        "EM",
        "F1",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "fallback_rate",
        "strict_path_complete",
        "answer_bearing_path_hit",
        "answer_present_but_generation_fail",
        "output_overlap_answer_bearing",
        "diagnostic_note",
    ]
    rows_main = []
    for r in metric_rows:
        rows_main.append(
            [
                str(r.get("dataset", "")),
                str(r.get("variant", "")),
                _fmt_float(r.get("R@5", 0.0), 4),
                _fmt_float(r.get("EM", 0.0), 4),
                _fmt_float(r.get("F1", 0.0), 4),
                _fmt_float(r.get("retrieval_ms", 0.0), 2),
                _fmt_float(r.get("generation_ms", 0.0), 2),
                _fmt_float(r.get("total_ms", 0.0), 2),
                _fmt_float(r.get("fallback_rate", 0.0), 4),
                _fmt_float(r.get("strict_path_complete", 0.0), 4),
                _fmt_float(r.get("answer_bearing_path_hit", 0.0), 4),
                _fmt_float(r.get("answer_present_but_generation_fail", 0.0), 4),
                _fmt_float(r.get("output_overlap_answer_bearing", 0.0), 4),
                str(r.get("diagnostic_note", "")),
            ]
        )

    headers_variant = [
        "variant",
        "mean_EM",
        "mean_F1",
        "mean_retrieval_ms",
        "mean_total_ms",
        "completed_datasets",
        "failed_datasets",
    ]
    rows_variant = []
    for r in variant_summary:
        rows_variant.append(
            [
                str(r.get("variant", "")),
                _fmt_float(r.get("mean_EM", 0.0), 4),
                _fmt_float(r.get("mean_F1", 0.0), 4),
                _fmt_float(r.get("mean_retrieval_ms", 0.0), 2),
                _fmt_float(r.get("mean_total_ms", 0.0), 2),
                ",".join(r.get("completed_datasets", []) or []),
                ",".join(r.get("failed_datasets", []) or []),
            ]
        )

    md_lines: List[str] = []
    md_lines.append("# Multi-dataset Generalization Round")
    md_lines.append("")
    md_lines.append("- This round checks dataset generalization with a near-fixed EffiRAG methodology.")
    md_lines.append("- Retrieval core was not redesigned; the goal is reproducibility/scalability across datasets.")
    md_lines.append("- musique/popqa should be interpreted as benchmark extension diagnostics.")
    md_lines.append("")

    md_lines.append("## Main Results by Dataset")
    md_lines.append("")
    md_lines.append(_markdown_table(headers_main, rows_main))
    md_lines.append("")

    md_lines.append("## Variant Summary")
    md_lines.append("")
    md_lines.append(_markdown_table(headers_variant, rows_variant))
    md_lines.append("")

    md_lines.append("## Failure Summary")
    md_lines.append("")
    if failures:
        for item in failures:
            md_lines.append(
                f"- {item.get('stage','')} / {item.get('variant','')} / {item.get('dataset','')}: "
                f"{item.get('reason','')} ({item.get('detail','')})"
            )
    else:
        md_lines.append("- no failures")
    md_lines.append("")

    md_lines.append("## Fallback & Diagnostic Integrity")
    md_lines.append("")
    if fallback_warnings:
        md_lines.append("- fallback warnings:")
        for item in fallback_warnings:
            md_lines.append(
                f"  - {item.get('dataset','')} / {item.get('variant','')}: "
                f"fallback_rate={_fmt_float(item.get('fallback_rate', 0.0), 4)}"
            )
    else:
        md_lines.append("- fallback warnings: none")

    if integrity_warnings:
        md_lines.append("- strict diagnostic integrity warnings:")
        for item in integrity_warnings:
            md_lines.append(
                f"  - {item.get('dataset','')} / {item.get('variant','')}: {item.get('note','')}"
            )
    else:
        md_lines.append("- strict diagnostic integrity warnings: none")

    output_md.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    print(json.dumps({
        "round_metrics_json": str(output_json),
        "round_metrics_md": str(output_md),
        "failures_json": str(failures_json),
        "records": len(records_out),
        "failures": len(failures),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
