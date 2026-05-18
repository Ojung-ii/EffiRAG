#!/usr/bin/env python3
"""Summarize Phase-6T score-attachment sequential trace logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from validate_phase6s_artifacts import collect_phase6s_artifacts

TRACE_RE = re.compile(
    r"^\[SCORE_ATTACH_TRACE\]\s+qid=(?P<qid>\S+)\s+trace_id=(?P<trace_id>\S+)\s+step=(?P<step>\S+)\s+"
    r"dt_prev_ms=(?P<dt_prev>-?[0-9.]+)\s+dt_total_ms=(?P<dt_total>-?[0-9.]+)\s+extra=(?P<extra>.*)$"
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Sequence[float]) -> float:
    arr = [float(v) for v in values]
    if not arr:
        return 0.0
    return float(sum(arr) / float(len(arr)))


def _percentile(values: Sequence[float], q: float) -> float:
    arr = sorted([float(v) for v in values])
    if not arr:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    idx = (len(arr) - 1) * float(q)
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    frac = float(idx - lo)
    return float(arr[lo] * (1.0 - frac) + arr[hi] * frac)


def _fmt(value: Any, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def _load_trace(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_qid
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        m = TRACE_RE.match(text)
        if not m:
            continue
        qid = _safe_text(m.group("qid"))
        step = _safe_text(m.group("step"))
        dt_prev = _safe_float(m.group("dt_prev"), 0.0)
        dt_total = _safe_float(m.group("dt_total"), 0.0)
        extra_raw = _safe_text(m.group("extra") or "{}")
        try:
            extra = json.loads(extra_raw)
            if not isinstance(extra, Mapping):
                extra = {"raw": extra}
        except Exception:
            extra = {"raw": extra_raw}
        by_qid[qid].append(
            {
                "qid": qid,
                "step": step,
                "dt_prev_ms": dt_prev,
                "dt_total_ms": dt_total,
                "extra": dict(extra or {}),
            }
        )
    return by_qid


def _step_idx(rows: Sequence[Mapping[str, Any]], step: str) -> int:
    for i, row in enumerate(rows):
        if _safe_text(row.get("step")) == step:
            return i
    return -1


def _step_extra(rows: Sequence[Mapping[str, Any]], step: str) -> Mapping[str, Any]:
    idx = _step_idx(rows, step)
    if idx < 0:
        return {}
    raw = rows[idx].get("extra", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _step_total(rows: Sequence[Mapping[str, Any]], step: str) -> float:
    idx = _step_idx(rows, step)
    if idx < 0:
        return 0.0
    return _safe_float(rows[idx].get("dt_total_ms"), 0.0)


def _build_summary(by_qid: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    interval_values: Dict[str, List[float]] = defaultdict(list)
    interval_shares: Dict[str, List[float]] = defaultdict(list)
    per_query: List[Dict[str, Any]] = []

    for qid, raw_rows in by_qid.items():
        rows = list(raw_rows or [])
        if len(rows) < 2:
            continue
        score_total = _step_total(rows, "score_attachment_done")
        if score_total <= 0.0:
            score_total = _safe_float(rows[-1].get("dt_total_ms"), 0.0)

        largest_interval = ""
        largest_interval_ms = -1.0
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]
            from_step = _safe_text(prev.get("step"))
            to_step = _safe_text(curr.get("step"))
            dt = _safe_float(curr.get("dt_prev_ms"), 0.0)
            name = f"{from_step}->{to_step}"
            interval_values[name].append(dt)
            if score_total > 0.0:
                interval_shares[name].append(float(dt) / float(score_total))
            else:
                interval_shares[name].append(0.0)
            if dt > largest_interval_ms:
                largest_interval_ms = dt
                largest_interval = name

        input_extra = _step_extra(rows, "candidate_input_inspect_done")
        loop_extra = _step_extra(rows, "candidate_loop_done")
        done_extra = _step_extra(rows, "score_attachment_done")
        per_query.append(
            {
                "qid": qid,
                "score_attachment_total_ms": float(score_total),
                "largest_interval": largest_interval,
                "largest_interval_ms": float(max(0.0, largest_interval_ms)),
                "num_candidates": int(_safe_float(input_extra.get("num_candidates", done_extra.get("num_candidates", 0)), 0)),
                "num_unique_candidate_ids": int(_safe_float(input_extra.get("num_unique_candidate_ids", 0), 0)),
                "num_full_map_scans": int(_safe_float(loop_extra.get("num_full_map_scans", done_extra.get("num_full_map_scans", 0)), 0)),
                "num_candidates_processed": int(_safe_float(loop_extra.get("num_candidates_processed", 0), 0)),
                "num_semantic_lookups": int(_safe_float(loop_extra.get("num_semantic_lookups", 0), 0)),
                "num_graph_lookups": int(_safe_float(loop_extra.get("num_graph_lookups", 0), 0)),
                "num_missing_scores": int(_safe_float(loop_extra.get("num_missing_scores", 0), 0)),
            }
        )

    interval_rows: List[Dict[str, Any]] = []
    for name, vals in interval_values.items():
        shares = interval_shares.get(name, [])
        from_step, to_step = name.split("->", 1) if "->" in name else (name, "")
        interval_rows.append(
            {
                "from_step": from_step,
                "to_step": to_step,
                "avg_dt_prev_ms": _mean(vals),
                "p50_dt_prev_ms": _percentile(vals, 0.5),
                "p95_dt_prev_ms": _percentile(vals, 0.95),
                "max_dt_prev_ms": max(vals) if vals else 0.0,
                "share_of_score_attachment": _mean(shares) if shares else 0.0,
                "count": int(len(vals)),
            }
        )
    interval_rows.sort(key=lambda r: float(r.get("avg_dt_prev_ms", 0.0)), reverse=True)
    per_query.sort(key=lambda r: float(r.get("score_attachment_total_ms", 0.0)), reverse=True)

    return {
        "query_count": int(len(per_query)),
        "interval_count": int(len(interval_rows)),
        "interval_rows": interval_rows,
        "per_query": per_query,
    }


def _summary_row(out_root: Path) -> Mapping[str, Any]:
    report = collect_phase6s_artifacts(out_root)
    completed = list(report.get("completed_runs", []) or [])
    if not completed:
        return {}
    run = completed[0]
    summary = dict(run.get("summary", {}) or {})
    return {
        "run_name": _safe_text(run.get("run_name")),
        "dataset": _safe_text(run.get("dataset")),
        "profile": _safe_text(run.get("profile")),
        "EM": _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0),
        "F1": _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0),
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
        "prompt_tokens_avg": _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0),
        "supporting_fact_recall": summary.get("supporting_fact_recall"),
        "supporting_fact_precision": summary.get("supporting_fact_precision"),
        "summary_path": _safe_text(run.get("summary_path")),
    }


def _load_cprofile_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _to_markdown(
    trace_log: Path,
    summary: Mapping[str, Any],
    run_row: Mapping[str, Any],
    cprofile_text: str,
) -> str:
    interval_rows = list(summary.get("interval_rows", []) or [])
    per_query = list(summary.get("per_query", []) or [])

    lines: List[str] = []
    lines.append("# Phase-6T Score Attachment Trace Summary")
    lines.append("")
    lines.append(f"- trace_log: `{trace_log}`")
    lines.append("")

    lines.append("## Executed Run")
    lines.append("")
    lines.append("| run_name | dataset | profile | EM | F1 | retrieval_ms | total_ms | prompt_tokens_avg | summary_path |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    lines.append(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            _safe_text(run_row.get("run_name")),
            _safe_text(run_row.get("dataset")),
            _safe_text(run_row.get("profile")),
            _fmt(run_row.get("EM", 0.0), 4),
            _fmt(run_row.get("F1", 0.0), 4),
            _fmt(run_row.get("retrieval_ms", 0.0), 3),
            _fmt(run_row.get("total_ms", 0.0), 3),
            _fmt(run_row.get("prompt_tokens_avg", 0.0), 3),
            _safe_text(run_row.get("summary_path")),
        )
    )
    lines.append("")

    lines.append("## Score Attachment Interval Statistics")
    lines.append("")
    lines.append("| from_step | to_step | avg_dt_prev_ms | p50_dt_prev_ms | p95_dt_prev_ms | max_dt_prev_ms | share_of_score_attachment |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in interval_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("from_step")),
                _safe_text(row.get("to_step")),
                _fmt(row.get("avg_dt_prev_ms", 0.0), 3),
                _fmt(row.get("p50_dt_prev_ms", 0.0), 3),
                _fmt(row.get("p95_dt_prev_ms", 0.0), 3),
                _fmt(row.get("max_dt_prev_ms", 0.0), 3),
                _fmt(row.get("share_of_score_attachment", 0.0), 4),
            )
        )
    lines.append("")

    lines.append("## Per-Query Largest Interval")
    lines.append("")
    lines.append("| qid | score_attachment_total_ms | largest_interval | largest_interval_ms | num_candidates | num_full_map_scans |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for row in per_query:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("qid")),
                _fmt(row.get("score_attachment_total_ms", 0.0), 3),
                _safe_text(row.get("largest_interval")),
                _fmt(row.get("largest_interval_ms", 0.0), 3),
                int(_safe_float(row.get("num_candidates", 0), 0)),
                int(_safe_float(row.get("num_full_map_scans", 0), 0)),
            )
        )
    lines.append("")

    lines.append("## Count Diagnostics")
    lines.append("")
    lines.append("| qid | num_candidates_processed | num_semantic_lookups | num_graph_lookups | num_missing_scores |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in per_query:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("qid")),
                int(_safe_float(row.get("num_candidates_processed", 0), 0)),
                int(_safe_float(row.get("num_semantic_lookups", 0), 0)),
                int(_safe_float(row.get("num_graph_lookups", 0), 0)),
                int(_safe_float(row.get("num_missing_scores", 0), 0)),
            )
        )
    lines.append("")

    lines.append("## Full Map Scan Diagnostics")
    lines.append("")
    total_scans = int(sum(int(_safe_float(r.get("num_full_map_scans", 0), 0)) for r in per_query))
    lines.append(f"- total_num_full_map_scans: {total_scans}")
    lines.append("")

    if cprofile_text:
        lines.append("## cProfile Top Functions")
        lines.append("")
        lines.append("```text")
        lines.append(cprofile_text.strip())
        lines.append("```")
        lines.append("")

    lines.append("## Bottleneck Ranking")
    lines.append("")
    if interval_rows:
        top = interval_rows[0]
        lines.append(
            "- largest interval: `{} -> {}` (avg {} ms, share {})".format(
                _safe_text(top.get("from_step")),
                _safe_text(top.get("to_step")),
                _fmt(top.get("avg_dt_prev_ms", 0.0), 3),
                _fmt(top.get("share_of_score_attachment", 0.0), 4),
            )
        )
    else:
        lines.append("- no interval parsed")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append("- Pick the largest measured interval or top cProfile cumulative-time function as the single optimization target.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6T score-attachment trace log.")
    parser.add_argument("--trace-log", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    trace_log = Path(_safe_text(args.trace_log)).resolve()
    out_root = Path(_safe_text(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    by_qid = _load_trace(trace_log)
    summary = _build_summary(by_qid)
    run_row = _summary_row(out_root)

    cprofile_txt_path = out_root / "PHASE6T_SCORE_ATTACHMENT_CPROFILE_TOP_FUNCTIONS.txt"
    cprofile_text = _load_cprofile_text(cprofile_txt_path)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trace_log": str(trace_log),
        "summary": summary,
        "run_row": run_row,
        "cprofile_report_path": str(cprofile_txt_path) if cprofile_txt_path.exists() else "",
    }

    out_json = out_root / "score_attachment_trace_summary.json"
    out_md = out_root / "PHASE6T_SCORE_ATTACHMENT_TRACE_SUMMARY.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_markdown(trace_log, summary, run_row, cprofile_text) + "\n", encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()
