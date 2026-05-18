#!/usr/bin/env python3
"""Summarize candidate-union sequential trace logs for Phase-6T."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

TRACE_RE = re.compile(
    r"^\[CAND_UNION_TRACE\]\s+qid=(?P<qid>\S+)\s+step=(?P<step>\S+)\s+dt_prev_ms=(?P<dt_prev>-?[0-9.]+)\s+dt_total_ms=(?P<dt_total>-?[0-9.]+)\s+extra=(?P<extra>.*)$"
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _fmt(value: Any, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


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


def _mean(values: Sequence[float]) -> float:
    arr = [float(v) for v in values]
    if not arr:
        return 0.0
    return float(sum(arr) / float(len(arr)))


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
        qid = str(m.group("qid"))
        step = str(m.group("step"))
        dt_prev = _safe_float(m.group("dt_prev"), 0.0)
        dt_total = _safe_float(m.group("dt_total"), 0.0)
        extra_raw = str(m.group("extra") or "{}")
        try:
            extra = json.loads(extra_raw)
            if not isinstance(extra, dict):
                extra = {"raw": extra}
        except Exception:
            extra = {"raw": extra_raw}
        by_qid[qid].append(
            {
                "qid": qid,
                "step": step,
                "dt_prev_ms": float(dt_prev),
                "dt_total_ms": float(dt_total),
                "extra": extra,
            }
        )
    return by_qid


def _summarize(by_qid: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    interval_values: Dict[str, List[float]] = defaultdict(list)
    interval_shares: Dict[str, List[float]] = defaultdict(list)
    per_query: List[Dict[str, Any]] = []

    for qid, rows in by_qid.items():
        records = list(rows or [])
        if len(records) < 2:
            continue

        union_total = 0.0
        union_end_rows = [r for r in records if str(r.get("step", "")) == "candidate_union_done"]
        if union_end_rows:
            union_total = _safe_float((union_end_rows[-1].get("extra", {}) or {}).get("candidate_union_total_ms", 0.0), 0.0)
            if union_total <= 0.0:
                union_total = _safe_float(union_end_rows[-1].get("dt_total_ms", 0.0), 0.0)
        else:
            union_total = _safe_float(records[-1].get("dt_total_ms", 0.0), 0.0)

        largest_dt = -1.0
        largest_name = ""
        interval_sum = 0.0
        for i in range(1, len(records)):
            prev = records[i - 1]
            curr = records[i]
            from_step = str(prev.get("step", ""))
            to_step = str(curr.get("step", ""))
            dt = _safe_float(curr.get("dt_prev_ms", 0.0), 0.0)
            interval_sum += dt
            name = f"{from_step}->{to_step}"
            interval_values[name].append(dt)
            if union_total > 0.0:
                interval_shares[name].append(float(dt) / float(union_total))
            else:
                interval_shares[name].append(0.0)
            if dt > largest_dt:
                largest_dt = dt
                largest_name = name

        per_query.append(
            {
                "qid": str(qid),
                "candidate_union_total_ms": float(union_total),
                "largest_interval": str(largest_name),
                "largest_interval_ms": float(max(0.0, largest_dt)),
                "num_checkpoints": int(len(records)),
                "interval_sum_ms": float(interval_sum),
                "interval_gap_ms": float(union_total - interval_sum),
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
                "share_of_candidate_union": _mean(shares) if shares else 0.0,
                "count": int(len(vals)),
            }
        )
    interval_rows.sort(key=lambda x: float(x.get("avg_dt_prev_ms", 0.0)), reverse=True)
    per_query.sort(key=lambda x: float(x.get("candidate_union_total_ms", 0.0)), reverse=True)

    return {
        "query_count": int(len(per_query)),
        "interval_count": int(len(interval_rows)),
        "interval_rows": interval_rows,
        "per_query": per_query,
    }


def _to_markdown(summary: Mapping[str, Any], trace_log: Path) -> str:
    interval_rows = list(summary.get("interval_rows", []) or [])
    per_query = list(summary.get("per_query", []) or [])

    lines: List[str] = []
    lines.append("# Phase-6T Candidate-Union Trace Summary")
    lines.append("")
    lines.append(f"- trace_log: `{trace_log}`")
    lines.append(f"- query_count: {int(summary.get('query_count', 0))}")
    lines.append(f"- interval_count: {int(summary.get('interval_count', 0))}")
    lines.append("")

    lines.append("## Interval Statistics")
    lines.append("")
    lines.append("| from_step | to_step | avg_dt_prev_ms | p50_dt_prev_ms | p95_dt_prev_ms | max_dt_prev_ms | share_of_candidate_union |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in interval_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                str(row.get("from_step", "")),
                str(row.get("to_step", "")),
                _fmt(row.get("avg_dt_prev_ms", 0.0), 3),
                _fmt(row.get("p50_dt_prev_ms", 0.0), 3),
                _fmt(row.get("p95_dt_prev_ms", 0.0), 3),
                _fmt(row.get("max_dt_prev_ms", 0.0), 3),
                _fmt(row.get("share_of_candidate_union", 0.0), 4),
            )
        )
    lines.append("")

    lines.append("## Per-Query Largest Interval")
    lines.append("")
    lines.append("| qid | candidate_union_total_ms | largest_interval | largest_interval_ms | interval_sum_ms | interval_gap_ms |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for row in per_query:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                str(row.get("qid", "")),
                _fmt(row.get("candidate_union_total_ms", 0.0), 3),
                str(row.get("largest_interval", "")),
                _fmt(row.get("largest_interval_ms", 0.0), 3),
                _fmt(row.get("interval_sum_ms", 0.0), 3),
                _fmt(row.get("interval_gap_ms", 0.0), 3),
            )
        )
    lines.append("")

    lines.append("## Decision Hints")
    lines.append("")
    if interval_rows:
        top = interval_rows[0]
        lines.append(
            "- Largest internal interval: `{} -> {}` (avg {} ms, share {}).".format(
                str(top.get("from_step", "")),
                str(top.get("to_step", "")),
                _fmt(top.get("avg_dt_prev_ms", 0.0), 3),
                _fmt(top.get("share_of_candidate_union", 0.0), 4),
            )
        )
        lines.append(
            "- If this share is > 0.40, prioritize that interval as next optimization target."
        )
    else:
        lines.append("- No valid interval rows parsed from trace log.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6T candidate-union sequential trace log.")
    parser.add_argument("--trace-log", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    trace_log = Path(str(args.trace_log)).resolve()
    out_root = Path(str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    by_qid = _load_trace(trace_log)
    summary = _summarize(by_qid)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trace_log": str(trace_log),
        **summary,
    }

    out_json = out_root / "candidate_union_trace_summary.json"
    out_md = out_root / "PHASE6T_CANDIDATE_UNION_TRACE_SUMMARY.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_markdown(summary, trace_log) + "\n", encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()
