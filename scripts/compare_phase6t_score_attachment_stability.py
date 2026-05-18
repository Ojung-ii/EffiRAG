#!/usr/bin/env python3
"""Compare Phase-6T score-attachment baseline vs optimized stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _safe_text(v: Any) -> str:
    return str(v or "").strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_list(v: Any) -> List[str]:
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for x in v:
            s = _safe_text(x)
            if s:
                out.append(s)
        return out
    return []


def _ordered_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        s = _safe_text(x)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


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


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_prediction(text: str) -> str:
    s = _safe_text(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set([_safe_text(x) for x in a if _safe_text(x)])
    sb = set([_safe_text(x) for x in b if _safe_text(x)])
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return float(len(sa & sb)) / float(len(union))


def _fmt(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _find_latest(root: Path, name: str) -> Path:
    files = sorted(root.rglob(name))
    if not files:
        raise FileNotFoundError(f"missing {name} under {root}")
    return files[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, Mapping):
        raise ValueError(f"Expected mapping JSON: {path}")
    return dict(obj)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if isinstance(obj, Mapping):
            rows.append(dict(obj))
    return rows


def _extract_chunk_from_sentence_id(sid: str) -> str:
    s = _safe_text(sid)
    if not s:
        return ""
    if s.startswith("chunk::") and "::" in s:
        left, right = s.rsplit("::", 1)
        if right.isdigit():
            return left
    if s.startswith("c::"):
        return s
    return ""


def _extract_row_id(row: Mapping[str, Any]) -> str:
    sid = _safe_text(row.get("sample_id"))
    if sid:
        return sid
    idx = row.get("sample_index")
    if idx is not None:
        return f"sample_index::{idx}"
    return ""


def _extract_rendered_text(row: Mapping[str, Any]) -> str:
    rendered = row.get("rendered")
    if isinstance(rendered, Mapping):
        return _safe_text(rendered.get("text"))
    return _safe_text(rendered)


def _extract_selected_sentence_ids(row: Mapping[str, Any]) -> List[str]:
    top = _safe_list(row.get("retrieval_selected_sentence_ids"))
    if top:
        return _ordered_unique(top)
    retrieval = row.get("retrieval")
    if isinstance(retrieval, Mapping):
        return _ordered_unique(_safe_list(retrieval.get("selected_sentence_ids")))
    return []


def _extract_rendered_sentence_ids(row: Mapping[str, Any]) -> List[str]:
    top = _safe_list(row.get("rendered_sentence_ids"))
    if top:
        return _ordered_unique(top)
    rendered = row.get("rendered")
    if isinstance(rendered, Mapping):
        return _ordered_unique(_safe_list(rendered.get("sentence_ids")))
    return []


def _extract_selected_chunk_ids(row: Mapping[str, Any], selected_sentence_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    retrieval = row.get("retrieval")
    if isinstance(retrieval, Mapping):
        nodes = _safe_list(retrieval.get("selected_nodes"))
        for n in nodes:
            if n.startswith("c::") or n.startswith("chunk::"):
                out.append(n)
    for sid in selected_sentence_ids:
        cid = _extract_chunk_from_sentence_id(sid)
        if cid:
            out.append(cid)
    return _ordered_unique(out)


def _extract_rendered_chunk_ids(rendered_sentence_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    for sid in rendered_sentence_ids:
        cid = _extract_chunk_from_sentence_id(sid)
        if cid:
            out.append(cid)
    return _ordered_unique(out)


def _extract_latency_breakdown(row: Mapping[str, Any]) -> Mapping[str, Any]:
    retrieval = row.get("retrieval")
    if isinstance(retrieval, Mapping):
        diag = retrieval.get("diagnostics")
        if isinstance(diag, Mapping):
            lb = diag.get("latency_breakdown_ms")
            if isinstance(lb, Mapping):
                return lb
    top = row.get("latency_breakdown_ms")
    if isinstance(top, Mapping):
        return top
    return {}


def _extract_query_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    qid = _extract_row_id(row)
    question = _safe_text(row.get("question"))
    prediction = _safe_text(row.get("prediction"))
    rendered_text = _extract_rendered_text(row)
    selected_sentence_ids = _extract_selected_sentence_ids(row)
    rendered_sentence_ids = _extract_rendered_sentence_ids(row)
    selected_chunk_ids = _extract_selected_chunk_ids(row, selected_sentence_ids)
    rendered_chunk_ids = _extract_rendered_chunk_ids(rendered_sentence_ids)

    prompt_source = f"{question}\n\n{rendered_text}"
    prompt_hash = _hash_text(prompt_source)
    context_hash = _hash_text(rendered_text)

    metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    lb = _extract_latency_breakdown(row)

    retrieval = row.get("retrieval") if isinstance(row.get("retrieval"), Mapping) else {}
    retrieval_ms = _safe_float(lb.get("retrieval_ms", retrieval.get("latency_ms", 0.0)), 0.0)
    total_ms = _safe_float(lb.get("total_ms", row.get("total_ms", 0.0)), 0.0)

    return {
        "sample_id": qid,
        "question": question,
        "prediction": prediction,
        "normalized_prediction": _normalize_prediction(prediction),
        "selected_sentence_ids": selected_sentence_ids,
        "rendered_sentence_ids": rendered_sentence_ids,
        "selected_chunk_ids": selected_chunk_ids,
        "rendered_chunk_ids": rendered_chunk_ids,
        "rendered_context": rendered_text,
        "prompt_hash": prompt_hash,
        "context_hash": context_hash,
        "retrieval_ms": retrieval_ms,
        "total_ms": total_ms,
        "score_attachment_ms": _safe_float(lb.get("score_attachment_ms", 0.0), 0.0),
        "proposal_total_ms": _safe_float(lb.get("proposal_total_ms", lb.get("proposal_time_ms", 0.0)), 0.0),
        "proposal_union_total_ms": _safe_float(lb.get("proposal_union_total_ms", 0.0), 0.0),
        "answer_bearing_chunk_present": _safe_float(metrics.get("answer_bearing_chunk_present", 0.0), 0.0),
        "answer_present_but_generation_fail": _safe_float(metrics.get("answer_present_but_generation_fail", 0.0), 0.0),
    }


def _load_run_artifacts(run_root: Path) -> Dict[str, Any]:
    summary_path = _find_latest(run_root, "rag_summary.json")
    jsonl_path = _find_latest(run_root, "rag_query_results.jsonl")
    summary = _load_json(summary_path)
    rows = _load_jsonl(jsonl_path)

    trace_log = run_root.parent.parent / "logs" / f"score_attachment_trace_{run_root.name}.log"
    if not trace_log.exists():
        trace_log = run_root.parent / "logs" / f"score_attachment_trace_{run_root.name}.log"
    if not trace_log.exists():
        trace_log = run_root / "logs" / "score_attachment_trace.log"

    qmap: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rec = _extract_query_record(row)
        sid = _safe_text(rec.get("sample_id"))
        if sid:
            qmap[sid] = rec

    return {
        "run_root": str(run_root.resolve()),
        "summary_path": str(summary_path.resolve()),
        "jsonl_path": str(jsonl_path.resolve()),
        "trace_log_path": str(trace_log.resolve()) if trace_log.exists() else "",
        "summary": summary,
        "query_map": qmap,
    }


def _summary_metric(summary: Mapping[str, Any], key: str, fallback_keys: Sequence[str] = ()) -> float:
    if key in summary:
        return _safe_float(summary.get(key), 0.0)
    for k in fallback_keys:
        if k in summary:
            return _safe_float(summary.get(k), 0.0)
    return 0.0


def _trace_path_candidate_avg(trace_log_path: str) -> float:
    if not trace_log_path:
        return 0.0
    path = Path(trace_log_path)
    if not path.exists():
        return 0.0
    # Parse SCORE_ATTACH_TRACE and collect interval from path_candidate_expansion_start -> path_candidate_expansion_done.
    pattern = re.compile(
        r"^\[SCORE_ATTACH_TRACE\]\s+qid=(?P<qid>\S+)\s+trace_id=(?P<trace_id>\S+)\s+step=(?P<step>\S+)\s+"
        r"dt_prev_ms=(?P<dt_prev>-?[0-9.]+)\s+dt_total_ms=(?P<dt_total>-?[0-9.]+)\s+extra=(?P<extra>.*)$"
    )
    prev_step_by_qid: Dict[str, str] = {}
    values: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        qid = _safe_text(m.group("qid"))
        step = _safe_text(m.group("step"))
        dt_prev = _safe_float(m.group("dt_prev"), 0.0)
        prev_step = prev_step_by_qid.get(qid, "")
        if prev_step == "path_candidate_expansion_start" and step == "path_candidate_expansion_done":
            values.append(dt_prev)
        prev_step_by_qid[qid] = step
    return mean(values) if values else 0.0


def _build_changed_examples(
    baseline_map: Mapping[str, Mapping[str, Any]],
    optimized_map: Mapping[str, Mapping[str, Any]],
    common_ids: Sequence[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for sid in common_ids:
        b = baseline_map[sid]
        o = optimized_map[sid]
        pred_changed = b.get("normalized_prediction") != o.get("normalized_prediction")
        selected_changed = b.get("selected_sentence_ids") != o.get("selected_sentence_ids")
        rendered_changed = b.get("rendered_sentence_ids") != o.get("rendered_sentence_ids")
        prompt_changed = b.get("prompt_hash") != o.get("prompt_hash")
        context_changed = b.get("context_hash") != o.get("context_hash")
        if not (pred_changed or selected_changed or rendered_changed or prompt_changed or context_changed):
            continue
        examples.append(
            {
                "sample_id": sid,
                "question": b.get("question", ""),
                "baseline_prediction": b.get("prediction", ""),
                "optimized_prediction": o.get("prediction", ""),
                "baseline_selected_sentence_ids": b.get("selected_sentence_ids", []),
                "optimized_selected_sentence_ids": o.get("selected_sentence_ids", []),
                "baseline_rendered_sentence_ids": b.get("rendered_sentence_ids", []),
                "optimized_rendered_sentence_ids": o.get("rendered_sentence_ids", []),
                "baseline_rendered_context": b.get("rendered_context", ""),
                "optimized_rendered_context": o.get("rendered_context", ""),
                "baseline_prompt_hash": b.get("prompt_hash", ""),
                "optimized_prompt_hash": o.get("prompt_hash", ""),
                "baseline_context_hash": b.get("context_hash", ""),
                "optimized_context_hash": o.get("context_hash", ""),
                "selected_jaccard": _jaccard(b.get("selected_sentence_ids", []), o.get("selected_sentence_ids", [])),
                "rendered_jaccard": _jaccard(b.get("rendered_sentence_ids", []), o.get("rendered_sentence_ids", [])),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _write_changed_examples(out_root: Path, examples: Sequence[Mapping[str, Any]]) -> Tuple[Path, Path]:
    out_json = out_root / "changed_query_examples.json"
    out_md = out_root / "changed_query_examples.md"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": int(len(examples)),
        "examples": list(examples),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# Changed Query Examples")
    lines.append("")
    lines.append(f"- total_examples: {len(examples)}")
    lines.append("")
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"## {idx}. sample_id={_safe_text(ex.get('sample_id'))}")
        lines.append("")
        lines.append(f"- question: {_safe_text(ex.get('question'))}")
        lines.append(f"- selected_jaccard: {_fmt(ex.get('selected_jaccard'), 4)}")
        lines.append(f"- rendered_jaccard: {_fmt(ex.get('rendered_jaccard'), 4)}")
        lines.append("")
        lines.append("baseline_prediction:")
        lines.append(f"`{_safe_text(ex.get('baseline_prediction'))}`")
        lines.append("")
        lines.append("optimized_prediction:")
        lines.append(f"`{_safe_text(ex.get('optimized_prediction'))}`")
        lines.append("")
        lines.append(f"- baseline_selected_sentence_ids: {_safe_text(ex.get('baseline_selected_sentence_ids'))}")
        lines.append(f"- optimized_selected_sentence_ids: {_safe_text(ex.get('optimized_selected_sentence_ids'))}")
        lines.append(f"- baseline_rendered_sentence_ids: {_safe_text(ex.get('baseline_rendered_sentence_ids'))}")
        lines.append(f"- optimized_rendered_sentence_ids: {_safe_text(ex.get('optimized_rendered_sentence_ids'))}")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md, out_json


def _decision_block(metrics: Mapping[str, float], drift: Mapping[str, float]) -> Tuple[str, str]:
    retrieval_gain = metrics.get("retrieval_ms_gain_ratio", 0.0)
    f1_drop = metrics.get("f1_drop_abs", 0.0)
    sf_recall_drop = metrics.get("sf_recall_drop_abs", 0.0)
    pred_rate = drift.get("prediction_changed_rate", 0.0)
    sel_rate = drift.get("selected_sentence_id_changed_rate", 0.0)
    rnd_rate = drift.get("rendered_sentence_id_changed_rate", 0.0)

    accepted = (
        retrieval_gain >= 0.50
        and f1_drop <= 0.01
        and sf_recall_drop <= 0.02
        and pred_rate <= 0.10
        and sel_rate <= 0.30
        and rnd_rate <= 0.30
    )
    needs_correction = (
        f1_drop > 0.01
        or sf_recall_drop > 0.02
        or pred_rate > 0.20
    )

    if accepted:
        return "accepted", "Latency gain and QA/evidence stability thresholds are satisfied."
    if needs_correction:
        return "needs_correction", "Behavior drift exceeds thresholds; inspect score tie-breaking and candidate-id normalization."
    return "candidate_with_caution", "Latency gain is strong but drift is moderate; keep as candidate and inspect changed examples."


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Phase-6T score-attachment baseline vs optimized stability.")
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--optimized-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    baseline_root = Path(_safe_text(args.baseline_root)).resolve()
    optimized_root = Path(_safe_text(args.optimized_root)).resolve()
    out_root = Path(_safe_text(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    baseline = _load_run_artifacts(baseline_root)
    optimized = _load_run_artifacts(optimized_root)

    b_summary = baseline["summary"]
    o_summary = optimized["summary"]
    b_map = baseline["query_map"]
    o_map = optimized["query_map"]

    b_ids = set(b_map.keys())
    o_ids = set(o_map.keys())
    common_ids = sorted(list(b_ids & o_ids))
    baseline_only_ids = sorted(list(b_ids - o_ids))
    optimized_only_ids = sorted(list(o_ids - b_ids))
    total_common = int(len(common_ids))

    selected_jaccards: List[float] = []
    rendered_jaccards: List[float] = []
    selected_chunk_jaccards: List[float] = []
    rendered_chunk_jaccards: List[float] = []

    prediction_changed = 0
    selected_changed = 0
    rendered_changed = 0
    prompt_hash_changed = 0
    context_hash_changed = 0

    for sid in common_ids:
        b = b_map[sid]
        o = o_map[sid]
        if b.get("normalized_prediction") != o.get("normalized_prediction"):
            prediction_changed += 1
        if b.get("selected_sentence_ids") != o.get("selected_sentence_ids"):
            selected_changed += 1
        if b.get("rendered_sentence_ids") != o.get("rendered_sentence_ids"):
            rendered_changed += 1
        if b.get("prompt_hash") != o.get("prompt_hash"):
            prompt_hash_changed += 1
        if b.get("context_hash") != o.get("context_hash"):
            context_hash_changed += 1

        selected_jaccards.append(_jaccard(b.get("selected_sentence_ids", []), o.get("selected_sentence_ids", [])))
        rendered_jaccards.append(_jaccard(b.get("rendered_sentence_ids", []), o.get("rendered_sentence_ids", [])))
        selected_chunk_jaccards.append(_jaccard(b.get("selected_chunk_ids", []), o.get("selected_chunk_ids", [])))
        rendered_chunk_jaccards.append(_jaccard(b.get("rendered_chunk_ids", []), o.get("rendered_chunk_ids", [])))

    b_retrieval_ms = _summary_metric(b_summary, "retrieval_ms")
    o_retrieval_ms = _summary_metric(o_summary, "retrieval_ms")
    b_total_ms = _summary_metric(b_summary, "total_ms")
    o_total_ms = _summary_metric(o_summary, "total_ms")
    b_em = _summary_metric(b_summary, "em", ("EM",))
    o_em = _summary_metric(o_summary, "em", ("EM",))
    b_f1 = _summary_metric(b_summary, "f1", ("F1",))
    o_f1 = _summary_metric(o_summary, "f1", ("F1",))
    b_prompt = _summary_metric(b_summary, "prompt_tokens_avg", ("avg_context_tokens",))
    o_prompt = _summary_metric(o_summary, "prompt_tokens_avg", ("avg_context_tokens",))
    b_sf_recall = _summary_metric(b_summary, "supporting_fact_recall", ("sf_recall",))
    o_sf_recall = _summary_metric(o_summary, "supporting_fact_recall", ("sf_recall",))
    b_sf_precision = _summary_metric(b_summary, "supporting_fact_precision", ("sf_precision",))
    o_sf_precision = _summary_metric(o_summary, "supporting_fact_precision", ("sf_precision",))
    b_abcp = _summary_metric(b_summary, "answer_bearing_chunk_present")
    o_abcp = _summary_metric(o_summary, "answer_bearing_chunk_present")
    b_abgf = _summary_metric(b_summary, "answer_present_but_generation_fail")
    o_abgf = _summary_metric(o_summary, "answer_present_but_generation_fail")

    b_rows = [b_map[sid] for sid in common_ids]
    o_rows = [o_map[sid] for sid in common_ids]

    b_retrieval_series = [float(r.get("retrieval_ms", 0.0)) for r in b_rows]
    o_retrieval_series = [float(r.get("retrieval_ms", 0.0)) for r in o_rows]
    b_score_attach_series = [float(r.get("score_attachment_ms", 0.0)) for r in b_rows]
    o_score_attach_series = [float(r.get("score_attachment_ms", 0.0)) for r in o_rows]
    b_proposal_series = [float(r.get("proposal_total_ms", 0.0)) for r in b_rows]
    o_proposal_series = [float(r.get("proposal_total_ms", 0.0)) for r in o_rows]
    b_union_series = [float(r.get("proposal_union_total_ms", 0.0)) for r in b_rows]
    o_union_series = [float(r.get("proposal_union_total_ms", 0.0)) for r in o_rows]

    b_path_expand_avg = _trace_path_candidate_avg(_safe_text(baseline.get("trace_log_path")))
    o_path_expand_avg = _trace_path_candidate_avg(_safe_text(optimized.get("trace_log_path")))

    drift = {
        "prediction_changed_count": int(prediction_changed),
        "selected_sentence_id_changed_count": int(selected_changed),
        "rendered_sentence_id_changed_count": int(rendered_changed),
        "prompt_hash_changed_count": int(prompt_hash_changed),
        "context_hash_changed_count": int(context_hash_changed),
        "total_compared": int(total_common),
        "prediction_changed_rate": (float(prediction_changed) / float(total_common)) if total_common else 0.0,
        "selected_sentence_id_changed_rate": (float(selected_changed) / float(total_common)) if total_common else 0.0,
        "rendered_sentence_id_changed_rate": (float(rendered_changed) / float(total_common)) if total_common else 0.0,
        "prompt_hash_changed_rate": (float(prompt_hash_changed) / float(total_common)) if total_common else 0.0,
        "context_hash_changed_rate": (float(context_hash_changed) / float(total_common)) if total_common else 0.0,
        "selected_jaccard_avg": mean(selected_jaccards) if selected_jaccards else 0.0,
        "rendered_jaccard_avg": mean(rendered_jaccards) if rendered_jaccards else 0.0,
        "selected_chunk_jaccard_avg": mean(selected_chunk_jaccards) if selected_chunk_jaccards else 0.0,
        "rendered_chunk_jaccard_avg": mean(rendered_chunk_jaccards) if rendered_chunk_jaccards else 0.0,
        "baseline_only_count": int(len(baseline_only_ids)),
        "optimized_only_count": int(len(optimized_only_ids)),
    }

    metrics = {
        "retrieval_ms_gain_ratio": ((b_retrieval_ms - o_retrieval_ms) / b_retrieval_ms) if b_retrieval_ms > 0 else 0.0,
        "f1_drop_abs": float(max(0.0, b_f1 - o_f1)),
        "sf_recall_drop_abs": float(max(0.0, b_sf_recall - o_sf_recall)),
    }
    decision_status, decision_reason = _decision_block(metrics, drift)

    examples = _build_changed_examples(b_map, o_map, common_ids, limit=10)
    changed_md, changed_json = _write_changed_examples(out_root, examples)

    summary_json_path = out_root / "phase6t_score_attachment_stability_summary.json"
    summary_md_path = out_root / "PHASE6T_SCORE_ATTACHMENT_STABILITY_SUMMARY.md"

    qa_rows = [
        ("EM", b_em, o_em),
        ("F1", b_f1, o_f1),
        ("prompt_tokens_avg", b_prompt, o_prompt),
        ("retrieval_ms", b_retrieval_ms, o_retrieval_ms),
        ("total_ms", b_total_ms, o_total_ms),
        ("supporting_fact_recall", b_sf_recall, o_sf_recall),
        ("supporting_fact_precision", b_sf_precision, o_sf_precision),
        ("answer_bearing_chunk_present", b_abcp, o_abcp),
        ("answer_present_but_generation_fail", b_abgf, o_abgf),
    ]

    latency_rows = [
        ("retrieval_ms_avg", mean(b_retrieval_series) if b_retrieval_series else 0.0, mean(o_retrieval_series) if o_retrieval_series else 0.0),
        ("retrieval_ms_p50", _percentile(b_retrieval_series, 0.5), _percentile(o_retrieval_series, 0.5)),
        ("retrieval_ms_p95", _percentile(b_retrieval_series, 0.95), _percentile(o_retrieval_series, 0.95)),
        ("score_attachment_ms_avg", mean(b_score_attach_series) if b_score_attach_series else 0.0, mean(o_score_attach_series) if o_score_attach_series else 0.0),
        ("score_attachment_ms_p95", _percentile(b_score_attach_series, 0.95), _percentile(o_score_attach_series, 0.95)),
        ("proposal_total_ms_avg", mean(b_proposal_series) if b_proposal_series else 0.0, mean(o_proposal_series) if o_proposal_series else 0.0),
        ("proposal_union_total_ms_avg", mean(b_union_series) if b_union_series else 0.0, mean(o_union_series) if o_union_series else 0.0),
        ("path_candidate_expansion_ms_avg", b_path_expand_avg, o_path_expand_avg),
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": {
            "run_root": baseline["run_root"],
            "summary_path": baseline["summary_path"],
            "jsonl_path": baseline["jsonl_path"],
            "trace_log_path": baseline.get("trace_log_path", ""),
        },
        "optimized": {
            "run_root": optimized["run_root"],
            "summary_path": optimized["summary_path"],
            "jsonl_path": optimized["jsonl_path"],
            "trace_log_path": optimized.get("trace_log_path", ""),
        },
        "qa_comparison": [
            {
                "metric": m,
                "baseline": b,
                "optimized": o,
                "delta": o - b,
                "delta_percent": ((o - b) / b * 100.0) if b else 0.0,
            }
            for m, b, o in qa_rows
        ],
        "latency_comparison": [
            {
                "metric": m,
                "baseline": b,
                "optimized": o,
                "delta": o - b,
                "delta_percent": ((o - b) / b * 100.0) if b else 0.0,
            }
            for m, b, o in latency_rows
        ],
        "drift": drift,
        "decision": {
            "status": decision_status,
            "reason": decision_reason,
            "rules": metrics,
        },
        "changed_examples": {
            "count": len(examples),
            "md_path": str(changed_md.resolve()),
            "json_path": str(changed_json.resolve()),
        },
        "unmatched": {
            "baseline_only_ids": baseline_only_ids,
            "optimized_only_ids": optimized_only_ids,
        },
    }

    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# Phase-6T Score Attachment Stability Summary")
    lines.append("")
    lines.append("## 1. Experiment Setup")
    lines.append("")
    lines.append(f"- baseline_root: `{baseline['run_root']}`")
    lines.append(f"- optimized_root: `{optimized['run_root']}`")
    lines.append(f"- baseline_summary: `{baseline['summary_path']}`")
    lines.append(f"- optimized_summary: `{optimized['summary_path']}`")
    lines.append(f"- compared_queries: {total_common}")
    lines.append(f"- baseline_only_queries: {len(baseline_only_ids)}")
    lines.append(f"- optimized_only_queries: {len(optimized_only_ids)}")
    lines.append("")

    lines.append("## 2. Aggregate QA Comparison")
    lines.append("")
    lines.append("| metric | baseline | optimized | delta | delta_% |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, b, o in qa_rows:
        d = o - b
        p = (d / b * 100.0) if b else 0.0
        lines.append(f"| {m} | {_fmt(b,4)} | {_fmt(o,4)} | {_fmt(d,4)} | {_fmt(p,2)} |")
    lines.append("")

    lines.append("## 3. Latency Comparison")
    lines.append("")
    lines.append("| metric | baseline | optimized | delta | delta_% |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, b, o in latency_rows:
        d = o - b
        p = (d / b * 100.0) if b else 0.0
        lines.append(f"| {m} | {_fmt(b,3)} | {_fmt(o,3)} | {_fmt(d,3)} | {_fmt(p,2)} |")
    lines.append("")

    lines.append("## 4. Query-Level Drift")
    lines.append("")
    lines.append("| metric | changed_count | total | rate |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| prediction_changed_rate | {prediction_changed} | {total_common} | {_fmt(drift['prediction_changed_rate'],4)} |")
    lines.append(f"| selected_sentence_id_changed_rate | {selected_changed} | {total_common} | {_fmt(drift['selected_sentence_id_changed_rate'],4)} |")
    lines.append(f"| rendered_sentence_id_changed_rate | {rendered_changed} | {total_common} | {_fmt(drift['rendered_sentence_id_changed_rate'],4)} |")
    lines.append(f"| prompt_hash_changed_rate | {prompt_hash_changed} | {total_common} | {_fmt(drift['prompt_hash_changed_rate'],4)} |")
    lines.append(f"| context_hash_changed_rate | {context_hash_changed} | {total_common} | {_fmt(drift['context_hash_changed_rate'],4)} |")
    lines.append("")

    lines.append("## 5. Evidence Stability")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| selected_jaccard_avg | {_fmt(drift['selected_jaccard_avg'],4)} |")
    lines.append(f"| rendered_jaccard_avg | {_fmt(drift['rendered_jaccard_avg'],4)} |")
    lines.append(f"| selected_chunk_jaccard_avg | {_fmt(drift['selected_chunk_jaccard_avg'],4)} |")
    lines.append(f"| rendered_chunk_jaccard_avg | {_fmt(drift['rendered_chunk_jaccard_avg'],4)} |")
    lines.append("")

    lines.append("## 6. Failure / Changed Query Examples")
    lines.append("")
    lines.append(f"- changed_examples_count: {len(examples)}")
    lines.append(f"- changed_examples_md: `{changed_md}`")
    lines.append(f"- changed_examples_json: `{changed_json}`")
    lines.append("")

    lines.append("## 7. Decision")
    lines.append("")
    lines.append(f"- status: **{decision_status}**")
    lines.append(f"- reason: {decision_reason}")
    lines.append(f"- retrieval_ms_gain_ratio: {_fmt(metrics['retrieval_ms_gain_ratio'],4)}")
    lines.append(f"- f1_drop_abs: {_fmt(metrics['f1_drop_abs'],4)}")
    lines.append(f"- sf_recall_drop_abs: {_fmt(metrics['sf_recall_drop_abs'],4)}")
    lines.append("")

    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(summary_md_path))


if __name__ == "__main__":
    main()
