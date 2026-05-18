#!/usr/bin/env python3
"""Summarize Phase-6T aligned proposal/candidate-union trace."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

TRACE_RE = re.compile(
    r"^\[PROP_ALIGNED_TRACE\]\s+qid=(?P<qid>\S+)\s+trace_id=(?P<trace_id>\S+)\s+"
    r"scope=(?P<scope>\S+)\s+step=(?P<step>\S+)\s+t=(?P<t>-?[0-9.]+)\s+"
    r"dt_prev_ms=(?P<dt_prev>-?[0-9.]+)\s+dt_total_ms=(?P<dt_total>-?[0-9.]+)\s+extra=(?P<extra>.*)$"
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


def _fmt_opt(value: Any, nd: int = 3) -> str:
    if value is None:
        return "n/a"
    return _fmt(value, nd)


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


def _load_records(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        m = TRACE_RE.match(text)
        if not m:
            continue
        qid = str(m.group("qid"))
        trace_id = str(m.group("trace_id"))
        scope = str(m.group("scope"))
        step = str(m.group("step"))
        t = _safe_float(m.group("t"), 0.0)
        dt_prev = _safe_float(m.group("dt_prev"), 0.0)
        dt_total = _safe_float(m.group("dt_total"), 0.0)
        extra_raw = str(m.group("extra") or "{}")
        try:
            extra = json.loads(extra_raw)
            if not isinstance(extra, dict):
                extra = {"raw": extra}
        except Exception:
            extra = {"raw": extra_raw}
        out[trace_id].append(
            {
                "qid": qid,
                "trace_id": trace_id,
                "scope": scope,
                "step": step,
                "t": float(t),
                "dt_prev_ms": float(dt_prev),
                "dt_total_ms": float(dt_total),
                "extra": extra,
            }
        )
    return out


def _step_total(records: Sequence[Mapping[str, Any]], step: str) -> float | None:
    for rec in records:
        if str(rec.get("step", "")) == step:
            return _safe_float(rec.get("dt_total_ms", 0.0), 0.0)
    return None


def _total_proposal_ms(records: Sequence[Mapping[str, Any]]) -> float:
    for rec in reversed(list(records)):
        if str(rec.get("step", "")) == "proposal_end":
            val = _safe_float((rec.get("extra", {}) or {}).get("proposal_total_ms", 0.0), 0.0)
            if val > 0.0:
                return val
            return _safe_float(rec.get("dt_total_ms", 0.0), 0.0)
    return _safe_float(records[-1].get("dt_total_ms", 0.0), 0.0) if records else 0.0


def _summarize(by_trace: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    interval_values: Dict[str, List[float]] = defaultdict(list)
    interval_shares: Dict[str, List[float]] = defaultdict(list)
    per_query: List[Dict[str, Any]] = []

    for trace_id, rows in by_trace.items():
        records = list(rows or [])
        if not records:
            continue
        if str(trace_id).startswith("__runner__"):
            continue
        proposal_total = _total_proposal_ms(records)

        largest_name = ""
        largest_ms = -1.0
        for i in range(1, len(records)):
            prev = records[i - 1]
            curr = records[i]
            dt = _safe_float(curr.get("dt_prev_ms", 0.0), 0.0)
            from_step = str(prev.get("step", ""))
            to_step = str(curr.get("step", ""))
            scope_transition = f"{str(prev.get('scope', ''))}->{str(curr.get('scope', ''))}"
            key = f"{from_step}||{to_step}||{scope_transition}"
            interval_values[key].append(dt)
            if proposal_total > 0.0:
                interval_shares[key].append(float(dt) / float(proposal_total))
            else:
                interval_shares[key].append(0.0)
            if dt > largest_ms:
                largest_ms = dt
                largest_name = f"{from_step}->{to_step}"

        outer_start = _step_total(records, "candidate_union_outer_start")
        outer_done = _step_total(records, "candidate_union_outer_done")
        inner_start = _step_total(records, "candidate_union_inner_start")
        inner_done = _step_total(records, "candidate_union_inner_done")
        phase1_gen_start = _step_total(records, "phase1_run_generation_start")

        def _delta(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return float(b - a)

        per_query.append(
            {
                "qid": str(records[0].get("qid", trace_id)),
                "trace_id": str(trace_id),
                "proposal_total_ms": float(proposal_total),
                "outer_candidate_union_ms": _delta(outer_start, outer_done),
                "inner_candidate_union_ms": _delta(inner_start, inner_done),
                "outer_start_to_inner_start_ms": _delta(outer_start, inner_start),
                "inner_done_to_outer_done_ms": _delta(inner_done, outer_done),
                "outer_done_to_phase1_generation_start_ms": _delta(outer_done, phase1_gen_start),
                "largest_gap": largest_name,
                "largest_gap_ms": float(max(0.0, largest_ms)),
            }
        )

    interval_rows: List[Dict[str, Any]] = []
    for key, vals in interval_values.items():
        from_step, to_step, scope_transition = key.split("||", 2)
        shares = interval_shares.get(key, [])
        interval_rows.append(
            {
                "from_step": from_step,
                "to_step": to_step,
                "scope_transition": scope_transition,
                "avg_dt_prev_ms": _mean(vals),
                "p50_dt_prev_ms": _percentile(vals, 0.5),
                "p95_dt_prev_ms": _percentile(vals, 0.95),
                "max_dt_prev_ms": max(vals) if vals else 0.0,
                "share_of_total_proposal": _mean(shares) if shares else 0.0,
                "count": int(len(vals)),
            }
        )
    interval_rows.sort(key=lambda x: float(x.get("avg_dt_prev_ms", 0.0)), reverse=True)
    per_query.sort(key=lambda x: float(x.get("proposal_total_ms", 0.0)), reverse=True)

    def _avg_opt(key: str) -> float | None:
        vals = [r.get(key) for r in per_query if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return float(sum(float(v) for v in vals) / float(len(vals)))

    boundary_avg = {
        "outer_candidate_union_ms": _avg_opt("outer_candidate_union_ms"),
        "inner_candidate_union_ms": _avg_opt("inner_candidate_union_ms"),
        "outer_start_to_inner_start_ms": _avg_opt("outer_start_to_inner_start_ms"),
        "inner_done_to_outer_done_ms": _avg_opt("inner_done_to_outer_done_ms"),
        "outer_done_to_phase1_generation_start_ms": _avg_opt("outer_done_to_phase1_generation_start_ms"),
    }

    return {
        "query_count": int(len(per_query)),
        "interval_count": int(len(interval_rows)),
        "interval_rows": interval_rows,
        "per_query": per_query,
        "boundary_avg": boundary_avg,
    }


def _to_markdown(summary: Mapping[str, Any], trace_log: Path) -> str:
    interval_rows = list(summary.get("interval_rows", []) or [])
    per_query = list(summary.get("per_query", []) or [])
    boundary_avg = dict(summary.get("boundary_avg", {}) or {})

    lines: List[str] = []
    lines.append("# Phase-6T Aligned Proposal Trace Summary")
    lines.append("")
    lines.append(f"- trace_log: `{trace_log}`")
    lines.append(f"- query_count: {int(summary.get('query_count', 0))}")
    lines.append(f"- interval_count: {int(summary.get('interval_count', 0))}")
    lines.append("")

    lines.append("## Interval Statistics")
    lines.append("")
    lines.append("| from_step | to_step | scope_transition | avg_dt_prev_ms | p50_dt_prev_ms | p95_dt_prev_ms | max_dt_prev_ms | share_of_total_proposal |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in interval_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                str(row.get("from_step", "")),
                str(row.get("to_step", "")),
                str(row.get("scope_transition", "")),
                _fmt(row.get("avg_dt_prev_ms", 0.0), 3),
                _fmt(row.get("p50_dt_prev_ms", 0.0), 3),
                _fmt(row.get("p95_dt_prev_ms", 0.0), 3),
                _fmt(row.get("max_dt_prev_ms", 0.0), 3),
                _fmt(row.get("share_of_total_proposal", 0.0), 4),
            )
        )
    lines.append("")

    lines.append("## Candidate Union Boundary Table")
    lines.append("")
    lines.append("| qid | proposal_total_ms | outer_candidate_union_ms | inner_candidate_union_ms | outer_start_to_inner_start_ms | inner_done_to_outer_done_ms | outer_done_to_phase1_generation_start_ms | largest_gap | largest_gap_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in per_query:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                str(row.get("qid", "")),
                _fmt(row.get("proposal_total_ms", 0.0), 3),
                _fmt_opt(row.get("outer_candidate_union_ms"), 3),
                _fmt_opt(row.get("inner_candidate_union_ms"), 3),
                _fmt_opt(row.get("outer_start_to_inner_start_ms"), 3),
                _fmt_opt(row.get("inner_done_to_outer_done_ms"), 3),
                _fmt_opt(row.get("outer_done_to_phase1_generation_start_ms"), 3),
                str(row.get("largest_gap", "")),
                _fmt(row.get("largest_gap_ms", 0.0), 3),
            )
        )
    lines.append("")

    lines.append("## Aggregated Candidate Union Gaps")
    lines.append("")
    lines.append("| metric | avg_ms |")
    lines.append("|---|---:|")
    for key in [
        "outer_candidate_union_ms",
        "inner_candidate_union_ms",
        "outer_start_to_inner_start_ms",
        "inner_done_to_outer_done_ms",
        "outer_done_to_phase1_generation_start_ms",
    ]:
        lines.append(f"| {key} | {_fmt_opt(boundary_avg.get(key), 3)} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if boundary_avg.get("outer_start_to_inner_start_ms") not in (None, 0.0):
        lines.append(f"- Gap before inner start: avg {_fmt_opt(boundary_avg.get('outer_start_to_inner_start_ms'), 3)} ms")
    if boundary_avg.get("inner_candidate_union_ms") not in (None, 0.0):
        lines.append(f"- Gap inside inner union: avg {_fmt_opt(boundary_avg.get('inner_candidate_union_ms'), 3)} ms")
    if boundary_avg.get("inner_done_to_outer_done_ms") not in (None, 0.0):
        lines.append(f"- Gap after inner done before outer done: avg {_fmt_opt(boundary_avg.get('inner_done_to_outer_done_ms'), 3)} ms")
    if boundary_avg.get("outer_done_to_phase1_generation_start_ms") not in (None, 0.0):
        lines.append(f"- Gap after outer done before phase1 generation: avg {_fmt_opt(boundary_avg.get('outer_done_to_phase1_generation_start_ms'), 3)} ms")
    if interval_rows:
        top = interval_rows[0]
        lines.append(
            "- Largest measured interval overall: `{} -> {}` [{}] avg {} ms (share {}).".format(
                str(top.get("from_step", "")),
                str(top.get("to_step", "")),
                str(top.get("scope_transition", "")),
                _fmt(top.get("avg_dt_prev_ms", 0.0), 3),
                _fmt(top.get("share_of_total_proposal", 0.0), 4),
            )
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6T aligned proposal trace logs.")
    parser.add_argument("--trace-log", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    trace_log = Path(str(args.trace_log)).resolve()
    out_root = Path(str(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    by_trace = _load_records(trace_log)
    summary = _summarize(by_trace)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trace_log": str(trace_log),
        **summary,
    }

    out_json = out_root / "proposal_aligned_trace_summary.json"
    out_md = out_root / "PHASE6T_ALIGNED_PROPOSAL_TRACE_SUMMARY.md"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_markdown(summary, trace_log) + "\n", encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()
