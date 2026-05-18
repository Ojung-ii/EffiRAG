#!/usr/bin/env python3
"""Summarize Phase-6T candidate-union outer postprocess sequential traces."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts

TRACE_RE = re.compile(
    r"^\[CAND_UNION_POST_TRACE\]\s+qid=(?P<qid>\S+)\s+trace_id=(?P<trace_id>\S+)\s+mode=(?P<mode>\S+)\s+"
    r"step=(?P<step>\S+)\s+t=(?P<t>-?[0-9.]+)\s+dt_prev_ms=(?P<dt_prev>-?[0-9.]+)\s+"
    r"dt_total_ms=(?P<dt_total>-?[0-9.]+)\s+extra=(?P<extra>.*)$"
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


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


def _load_trace(path: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    by_mode: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return by_mode
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        m = TRACE_RE.match(text)
        if not m:
            continue
        qid = _safe_text(m.group("qid"))
        mode = _safe_text(m.group("mode"))
        step = _safe_text(m.group("step"))
        perf_t = _safe_float(m.group("t"), 0.0)
        dt_prev = _safe_float(m.group("dt_prev"), 0.0)
        dt_total = _safe_float(m.group("dt_total"), 0.0)
        extra_raw = _safe_text(m.group("extra") or "{}")
        try:
            extra = json.loads(extra_raw)
            if not isinstance(extra, Mapping):
                extra = {"raw": extra}
        except Exception:
            extra = {"raw": extra_raw}
        by_mode[mode][qid].append(
            {
                "qid": qid,
                "mode": mode,
                "step": step,
                "t": perf_t,
                "dt_prev_ms": dt_prev,
                "dt_total_ms": dt_total,
                "extra": dict(extra or {}),
            }
        )
    return by_mode


def _step_index(rows: Sequence[Mapping[str, Any]], step: str) -> int:
    for i, row in enumerate(rows):
        if _safe_text(row.get("step")) == step:
            return i
    return -1


def _step_total(rows: Sequence[Mapping[str, Any]], step: str) -> float:
    idx = _step_index(rows, step)
    if idx < 0:
        return 0.0
    return _safe_float(rows[idx].get("dt_total_ms"), 0.0)


def _step_extra(rows: Sequence[Mapping[str, Any]], step: str) -> Mapping[str, Any]:
    idx = _step_index(rows, step)
    if idx < 0:
        return {}
    extra = rows[idx].get("extra", {})
    return dict(extra) if isinstance(extra, Mapping) else {}


def _mode_summary(mode: str, by_qid: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    interval_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    interval_shares: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    boundary_rows: List[Dict[str, Any]] = []
    overlap_warnings: List[Dict[str, Any]] = []

    for qid, raw_rows in by_qid.items():
        rows = sorted(list(raw_rows or []), key=lambda r: float(r.get("t", 0.0)))
        if len(rows) < 2:
            continue

        outer_start_idx = _step_index(rows, "candidate_union_outer_start")
        inner_start_idx = _step_index(rows, "candidate_union_inner_start")
        inner_done_idx = _step_index(rows, "candidate_union_inner_done")
        outer_done_idx = _step_index(rows, "candidate_union_outer_done")

        outer_start = _step_total(rows, "candidate_union_outer_start")
        inner_start = _step_total(rows, "candidate_union_inner_start")
        inner_done = _step_total(rows, "candidate_union_inner_done")
        outer_done = _step_total(rows, "candidate_union_outer_done")

        outer_to_inner_start = float(inner_start - outer_start)
        inner_span = float(inner_done - inner_start)
        inner_to_outer_done = float(outer_done - inner_done)

        if outer_to_inner_start < 0.0 or inner_span < 0.0 or inner_to_outer_done < 0.0:
            overlap_warnings.append(
                {
                    "mode": mode,
                    "qid": qid,
                    "warning": "checkpoint_order_or_boundary_issue",
                    "outer_start_to_inner_start_ms": outer_to_inner_start,
                    "inner_start_to_inner_done_ms": inner_span,
                    "inner_done_to_outer_done_ms": inner_to_outer_done,
                }
            )

        inner_total_extra = _safe_float(_step_extra(rows, "candidate_union_inner_done").get("candidate_union_total_ms", 0.0), 0.0)
        outer_total_extra = _safe_float(_step_extra(rows, "candidate_union_outer_done").get("candidate_union_outer_total_ms", 0.0), 0.0)
        inner_total = inner_total_extra if inner_total_extra > 0.0 else max(0.0, inner_span)
        outer_total = outer_total_extra if outer_total_extra > 0.0 else max(0.0, outer_done - outer_start)

        post_start = inner_done_idx
        post_end = outer_done_idx
        largest_name = ""
        largest_dt = -1.0
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]
            from_step = _safe_text(prev.get("step"))
            to_step = _safe_text(curr.get("step"))
            dt = _safe_float(curr.get("dt_prev_ms"), 0.0)
            key = (from_step, to_step)
            interval_values[key].append(dt)

            if post_start >= 0 and post_end >= 0 and i - 1 >= post_start and i <= post_end:
                denom = max(1.0e-8, inner_to_outer_done)
                interval_shares[key].append(float(dt) / float(denom))
                if dt > largest_dt:
                    largest_dt = dt
                    largest_name = f"{from_step}->{to_step}"

        boundary_rows.append(
            {
                "qid": qid,
                "mode": mode,
                "outer_candidate_union_ms": float(outer_total),
                "inner_candidate_union_ms": float(inner_total),
                "outer_start_to_inner_start_ms": float(outer_to_inner_start),
                "inner_start_to_inner_done_ms": float(inner_span),
                "inner_done_to_outer_done_ms": float(inner_to_outer_done),
                "largest_postprocess_interval": largest_name,
                "largest_postprocess_interval_ms": float(max(0.0, largest_dt)),
            }
        )

    interval_rows: List[Dict[str, Any]] = []
    for (from_step, to_step), vals in interval_values.items():
        shares = interval_shares.get((from_step, to_step), [])
        interval_rows.append(
            {
                "mode": mode,
                "from_step": from_step,
                "to_step": to_step,
                "avg_dt_prev_ms": _mean(vals),
                "p50_dt_prev_ms": _percentile(vals, 0.5),
                "p95_dt_prev_ms": _percentile(vals, 0.95),
                "max_dt_prev_ms": max(vals) if vals else 0.0,
                "share_of_inner_done_to_outer_done": _mean(shares) if shares else 0.0,
                "count": int(len(vals)),
            }
        )
    interval_rows.sort(key=lambda r: float(r.get("avg_dt_prev_ms", 0.0)), reverse=True)
    boundary_rows.sort(key=lambda r: float(r.get("inner_done_to_outer_done_ms", 0.0)), reverse=True)

    return {
        "mode": mode,
        "query_count": int(len(boundary_rows)),
        "interval_rows": interval_rows,
        "boundary_rows": boundary_rows,
        "overlap_warnings": overlap_warnings,
    }


def _mode_from_run_name(name: str) -> str:
    text = _safe_text(name).lower()
    if "diag_off" in text:
        return "diag_off"
    return "diag_on"


def _summary_metrics_by_mode(out_root: Path) -> Dict[str, Dict[str, Any]]:
    report = collect_phase6s_artifacts(out_root)
    out: Dict[str, Dict[str, Any]] = {}
    for run in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(run.get("run_name"))
        mode = _mode_from_run_name(run_name)
        summary = dict(run.get("summary", {}) or {})
        out[mode] = {
            "run_name": run_name,
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
    return out


def _build_diag_compare(mode_summaries: Mapping[str, Mapping[str, Any]], mode_trace: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    on = dict(mode_summaries.get("diag_on", {}) or {})
    off = dict(mode_summaries.get("diag_off", {}) or {})

    def trace_avg(mode: str, key: str) -> float:
        rows = list((mode_trace.get(mode, {}) or {}).get("boundary_rows", []) or [])
        vals = [_safe_float(r.get(key, 0.0), 0.0) for r in rows]
        return _mean(vals)

    on_inner_outer = trace_avg("diag_on", "inner_done_to_outer_done_ms")
    off_inner_outer = trace_avg("diag_off", "inner_done_to_outer_done_ms")
    on_outer_total = trace_avg("diag_on", "outer_candidate_union_ms")
    off_outer_total = trace_avg("diag_off", "outer_candidate_union_ms")

    def _metric_pair(metric: str, on_val: Any, off_val: Any) -> Dict[str, Any]:
        if on_val is None or off_val is None:
            return {
                "metric": metric,
                "diag_on_avg": on_val,
                "diag_off_avg": off_val,
                "delta_ms": None,
                "delta_percent": None,
                "note": "metric unavailable due to diagnostics-off",
            }
        a = _safe_float(on_val, 0.0)
        b = _safe_float(off_val, 0.0)
        delta = float(b - a)
        pct = (delta / a) if abs(a) > 1.0e-8 else None
        return {
            "metric": metric,
            "diag_on_avg": a,
            "diag_off_avg": b,
            "delta_ms": delta,
            "delta_percent": pct,
            "note": "",
        }

    rows = [
        _metric_pair("retrieval_ms", on.get("retrieval_ms"), off.get("retrieval_ms")),
        _metric_pair("proposal_total_ms", on_outer_total, off_outer_total),
        _metric_pair("inner_done_to_outer_done_ms", on_inner_outer, off_inner_outer),
        _metric_pair("candidate_union_outer_ms", on_outer_total, off_outer_total),
        _metric_pair("F1", on.get("F1"), off.get("F1")),
        _metric_pair("prompt_tokens_avg", on.get("prompt_tokens_avg"), off.get("prompt_tokens_avg")),
        _metric_pair("supporting_fact_recall", on.get("supporting_fact_recall"), off.get("supporting_fact_recall")),
        _metric_pair("supporting_fact_precision", on.get("supporting_fact_precision"), off.get("supporting_fact_precision")),
    ]
    return rows


def _to_markdown(
    trace_log: Path,
    mode_trace: Mapping[str, Mapping[str, Any]],
    mode_summaries: Mapping[str, Mapping[str, Any]],
    diag_compare: Sequence[Mapping[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Phase-6T Candidate-Union Postprocess Trace Summary")
    lines.append("")
    lines.append(f"- trace_log: `{trace_log}`")
    lines.append("")

    lines.append("## Interval Statistics")
    lines.append("")
    lines.append("| mode | from_step | to_step | avg_dt_prev_ms | p50_dt_prev_ms | p95_dt_prev_ms | max_dt_prev_ms | share_of_inner_done_to_outer_done |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for mode in ["diag_on", "diag_off"]:
        for row in list((mode_trace.get(mode, {}) or {}).get("interval_rows", []) or []):
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    mode,
                    _safe_text(row.get("from_step")),
                    _safe_text(row.get("to_step")),
                    _fmt(row.get("avg_dt_prev_ms", 0.0), 3),
                    _fmt(row.get("p50_dt_prev_ms", 0.0), 3),
                    _fmt(row.get("p95_dt_prev_ms", 0.0), 3),
                    _fmt(row.get("max_dt_prev_ms", 0.0), 3),
                    _fmt(row.get("share_of_inner_done_to_outer_done", 0.0), 4),
                )
            )
    lines.append("")

    lines.append("## Boundary Table")
    lines.append("")
    lines.append("| qid | mode | outer_candidate_union_ms | inner_candidate_union_ms | inner_done_to_outer_done_ms | largest_postprocess_interval | largest_postprocess_interval_ms |")
    lines.append("|---|---|---:|---:|---:|---|---:|")
    for mode in ["diag_on", "diag_off"]:
        for row in list((mode_trace.get(mode, {}) or {}).get("boundary_rows", []) or []):
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    _safe_text(row.get("qid")),
                    mode,
                    _fmt(row.get("outer_candidate_union_ms", 0.0), 3),
                    _fmt(row.get("inner_candidate_union_ms", 0.0), 3),
                    _fmt(row.get("inner_done_to_outer_done_ms", 0.0), 3),
                    _safe_text(row.get("largest_postprocess_interval")),
                    _fmt(row.get("largest_postprocess_interval_ms", 0.0), 3),
                )
            )
    lines.append("")

    lines.append("## Diagnostics On/Off Comparison")
    lines.append("")
    lines.append("| metric | diag_on_avg | diag_off_avg | delta_ms | delta_percent | note |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in diag_compare:
        note = _safe_text(row.get("note"))
        delta_percent = row.get("delta_percent")
        delta_percent_text = _fmt(100.0 * float(delta_percent), 2) if delta_percent is not None else "n/a"
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("metric")),
                _fmt(row.get("diag_on_avg"), 3),
                _fmt(row.get("diag_off_avg"), 3),
                _fmt(row.get("delta_ms"), 3),
                delta_percent_text,
                note,
            )
        )
    lines.append("")

    lines.append("## Run Mapping")
    lines.append("")
    lines.append("| mode | run_name | dataset | profile | summary_path |")
    lines.append("|---|---|---|---|---|")
    for mode in ["diag_on", "diag_off"]:
        row = dict(mode_summaries.get(mode, {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                mode,
                _safe_text(row.get("run_name")),
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _safe_text(row.get("summary_path")),
            )
        )
    lines.append("")

    lines.append("## Overlap Warnings")
    lines.append("")
    warnings: List[Mapping[str, Any]] = []
    for mode in ["diag_on", "diag_off"]:
        warnings.extend(list((mode_trace.get(mode, {}) or {}).get("overlap_warnings", []) or []))
    if not warnings:
        lines.append("No checkpoint boundary warnings.")
    else:
        lines.append("| mode | qid | outer_start_to_inner_start_ms | inner_start_to_inner_done_ms | inner_done_to_outer_done_ms | warning |")
        lines.append("|---|---|---:|---:|---:|---|")
        for w in warnings:
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    _safe_text(w.get("mode")),
                    _safe_text(w.get("qid")),
                    _fmt(w.get("outer_start_to_inner_start_ms", 0.0), 3),
                    _fmt(w.get("inner_start_to_inner_done_ms", 0.0), 3),
                    _fmt(w.get("inner_done_to_outer_done_ms", 0.0), 3),
                    _safe_text(w.get("warning")),
                )
            )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6T candidate-union postprocess trace.")
    parser.add_argument("--trace-log", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    trace_log = Path(_safe_text(args.trace_log)).resolve()
    out_root = Path(_safe_text(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    by_mode_qid = _load_trace(trace_log)
    mode_trace = {
        mode: _mode_summary(mode, qids)
        for mode, qids in by_mode_qid.items()
    }
    for must_mode in ["diag_on", "diag_off"]:
        mode_trace.setdefault(must_mode, _mode_summary(must_mode, {}))

    mode_summaries = _summary_metrics_by_mode(out_root)
    diag_compare = _build_diag_compare(mode_summaries, mode_trace)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trace_log": str(trace_log),
        "mode_trace": mode_trace,
        "mode_summaries": mode_summaries,
        "diag_compare": diag_compare,
    }

    out_json = out_root / "candidate_union_postprocess_trace_summary.json"
    out_md = out_root / "PHASE6T_CANDIDATE_UNION_POSTPROCESS_TRACE_SUMMARY.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_to_markdown(trace_log, mode_trace, mode_summaries, diag_compare) + "\n", encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()
