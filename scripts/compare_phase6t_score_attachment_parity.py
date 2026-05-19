#!/usr/bin/env python3
"""Three-way parity/stability comparison for Phase-6T score attachment."""

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
        head, sep, tail = s.rpartition(":")
        if sep and tail.isdigit():
            return head
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


def _extract_candidate_score_map(row: Mapping[str, Any]) -> Dict[str, float]:
    retrieval = row.get("retrieval")
    diag = retrieval.get("diagnostics") if isinstance(retrieval, Mapping) else {}
    if not isinstance(diag, Mapping):
        diag = {}
    maps_to_try = [
        diag.get("sentence_scores"),
        diag.get("selected_sentence_score_map"),
        diag.get("candidate_score_map"),
        row.get("sentence_scores"),
    ]
    for cand in maps_to_try:
        if isinstance(cand, Mapping):
            out: Dict[str, float] = {}
            for k, v in cand.items():
                key = _safe_text(k)
                if not key:
                    continue
                out[key] = _safe_float(v, 0.0)
            if out:
                return out
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
        "f1": _safe_float(metrics.get("f1", 0.0), 0.0),
        "em": _safe_float(metrics.get("em", 0.0), 0.0),
        "supporting_fact_recall": _safe_float(metrics.get("supporting_fact_recall", 0.0), 0.0),
        "supporting_fact_precision": _safe_float(metrics.get("supporting_fact_precision", 0.0), 0.0),
        "prompt_tokens": _safe_float(metrics.get("prompt_tokens", row.get("prompt_tokens", 0.0)), 0.0),
        "candidate_scores": _extract_candidate_score_map(row),
    }


def _load_run_artifacts(run_root: Path) -> Dict[str, Any]:
    summary_path = _find_latest(run_root, "rag_summary.json")
    jsonl_path = _find_latest(run_root, "rag_query_results.jsonl")
    summary = _load_json(summary_path)
    rows = _load_jsonl(jsonl_path)

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


def _pair_drift_metrics(
    left_name: str,
    right_name: str,
    left_map: Mapping[str, Mapping[str, Any]],
    right_map: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    left_ids = set(left_map.keys())
    right_ids = set(right_map.keys())
    common_ids = sorted(list(left_ids & right_ids))
    total = len(common_ids)

    pred_changed = 0
    selected_changed = 0
    rendered_changed = 0
    prompt_changed = 0
    context_changed = 0

    selected_j: List[float] = []
    rendered_j: List[float] = []
    selected_chunk_j: List[float] = []
    rendered_chunk_j: List[float] = []

    for sid in common_ids:
        l = left_map[sid]
        r = right_map[sid]
        if l.get("normalized_prediction") != r.get("normalized_prediction"):
            pred_changed += 1
        if l.get("selected_sentence_ids") != r.get("selected_sentence_ids"):
            selected_changed += 1
        if l.get("rendered_sentence_ids") != r.get("rendered_sentence_ids"):
            rendered_changed += 1
        if l.get("prompt_hash") != r.get("prompt_hash"):
            prompt_changed += 1
        if l.get("context_hash") != r.get("context_hash"):
            context_changed += 1

        selected_j.append(_jaccard(l.get("selected_sentence_ids", []), r.get("selected_sentence_ids", [])))
        rendered_j.append(_jaccard(l.get("rendered_sentence_ids", []), r.get("rendered_sentence_ids", [])))
        selected_chunk_j.append(_jaccard(l.get("selected_chunk_ids", []), r.get("selected_chunk_ids", [])))
        rendered_chunk_j.append(_jaccard(l.get("rendered_chunk_ids", []), r.get("rendered_chunk_ids", [])))

    return {
        "pair": f"{left_name}_vs_{right_name}",
        "left": left_name,
        "right": right_name,
        "total_compared": int(total),
        "prediction_changed_count": int(pred_changed),
        "selected_sentence_id_changed_count": int(selected_changed),
        "rendered_sentence_id_changed_count": int(rendered_changed),
        "prompt_hash_changed_count": int(prompt_changed),
        "context_hash_changed_count": int(context_changed),
        "prediction_changed_rate": (float(pred_changed) / float(total)) if total else 0.0,
        "selected_sentence_id_changed_rate": (float(selected_changed) / float(total)) if total else 0.0,
        "rendered_sentence_id_changed_rate": (float(rendered_changed) / float(total)) if total else 0.0,
        "prompt_hash_changed_rate": (float(prompt_changed) / float(total)) if total else 0.0,
        "context_hash_changed_rate": (float(context_changed) / float(total)) if total else 0.0,
        "selected_jaccard_avg": mean(selected_j) if selected_j else 0.0,
        "rendered_jaccard_avg": mean(rendered_j) if rendered_j else 0.0,
        "selected_chunk_jaccard_avg": mean(selected_chunk_j) if selected_chunk_j else 0.0,
        "rendered_chunk_jaccard_avg": mean(rendered_chunk_j) if rendered_chunk_j else 0.0,
        "left_only_count": int(len(left_ids - right_ids)),
        "right_only_count": int(len(right_ids - left_ids)),
    }


def _normalize_candidate_id(v: str) -> str:
    return _safe_text(v).lower().replace(" ", "")


def _sorted_score_items(scores: Mapping[str, float], topk: int = 20) -> List[Tuple[str, float]]:
    items = [(str(k), float(v)) for k, v in scores.items() if _safe_text(k)]
    items.sort(key=lambda x: (-float(x[1]), str(x[0])))
    if topk > 0:
        return items[:topk]
    return items


def _score_parity_for_pair(
    pair_name: str,
    left_map: Mapping[str, Mapping[str, Any]],
    right_map: Mapping[str, Mapping[str, Any]],
    epsilon: float = 1.0e-12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    common_ids = sorted(list(set(left_map.keys()) & set(right_map.keys())))
    reason_counts: Dict[str, int] = {}

    for sid in common_ids:
        l_scores = dict(left_map[sid].get("candidate_scores", {}) or {})
        r_scores = dict(right_map[sid].get("candidate_scores", {}) or {})
        l_ids = set(l_scores.keys())
        r_ids = set(r_scores.keys())
        common = sorted(list(l_ids & r_ids))
        missing_in_right = sorted(list(l_ids - r_ids))
        missing_in_left = sorted(list(r_ids - l_ids))

        abs_diffs = [abs(float(l_scores[c]) - float(r_scores[c])) for c in common]
        max_abs = max(abs_diffs) if abs_diffs else 0.0
        mean_abs = mean(abs_diffs) if abs_diffs else 0.0
        num_changed = sum(1 for d in abs_diffs if d > epsilon)

        l_top = _sorted_score_items(l_scores, topk=20)
        r_top = _sorted_score_items(r_scores, topk=20)
        l_top_ids = [k for k, _ in l_top]
        r_top_ids = [k for k, _ in r_top]
        topk_jaccard = _jaccard(l_top_ids, r_top_ids)
        topk_order_changed = bool(l_top_ids != r_top_ids)

        reason = "unknown"
        if missing_in_left or missing_in_right:
            l_norm = {_normalize_candidate_id(k) for k in l_ids}
            r_norm = {_normalize_candidate_id(k) for k in r_ids}
            if len(l_norm & r_norm) > len(common):
                reason = "candidate_id_normalization_diff"
            else:
                reason = "missing_score"
        elif num_changed == 0 and topk_order_changed:
            reason = "tie_break_only"
        elif max_abs <= 1.0e-9:
            reason = "float_precision_diff" if num_changed > 0 else "tie_break_only"
        reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)

        out_rows.append(
            {
                "pair": pair_name,
                "sample_id": sid,
                "num_candidates_left": int(len(l_ids)),
                "num_candidates_right": int(len(r_ids)),
                "num_common_candidates": int(len(common)),
                "candidate_score_max_abs_diff": float(max_abs),
                "candidate_score_mean_abs_diff": float(mean_abs),
                "num_score_changed_candidates": int(num_changed),
                "num_score_missing_in_right": int(len(missing_in_right)),
                "num_score_missing_in_left": int(len(missing_in_left)),
                "topk_candidate_id_jaccard": float(topk_jaccard),
                "topk_score_order_changed": bool(topk_order_changed),
                "reason": reason,
            }
        )

    max_abs_all = [float(r["candidate_score_max_abs_diff"]) for r in out_rows]
    mean_abs_all = [float(r["candidate_score_mean_abs_diff"]) for r in out_rows]
    topk_j_all = [float(r["topk_candidate_id_jaccard"]) for r in out_rows]
    changed_counts = [int(r["num_score_changed_candidates"]) for r in out_rows]
    missing_r = [int(r["num_score_missing_in_right"]) for r in out_rows]
    missing_l = [int(r["num_score_missing_in_left"]) for r in out_rows]
    common_counts = [int(r["num_common_candidates"]) for r in out_rows]

    agg = {
        "pair": pair_name,
        "queries": int(len(out_rows)),
        "candidate_score_max_abs_diff_max": max(max_abs_all) if max_abs_all else 0.0,
        "candidate_score_max_abs_diff_avg": mean(max_abs_all) if max_abs_all else 0.0,
        "candidate_score_mean_abs_diff_avg": mean(mean_abs_all) if mean_abs_all else 0.0,
        "num_score_changed_candidates_avg": mean(changed_counts) if changed_counts else 0.0,
        "num_score_missing_in_right_avg": mean(missing_r) if missing_r else 0.0,
        "num_score_missing_in_left_avg": mean(missing_l) if missing_l else 0.0,
        "num_common_candidates_avg": mean(common_counts) if common_counts else 0.0,
        "topk_candidate_id_jaccard_avg": mean(topk_j_all) if topk_j_all else 0.0,
        "topk_score_order_changed_rate": (
            float(sum(1 for r in out_rows if bool(r.get("topk_score_order_changed")))) / float(len(out_rows))
            if out_rows
            else 0.0
        ),
        "reason_counts": reason_counts,
    }
    return out_rows, agg


def _mode_metrics(name: str, run: Mapping[str, Any]) -> Dict[str, Any]:
    summary = run["summary"]
    qmap = run["query_map"]
    rows = list(qmap.values())
    retrieval_series = [float(r.get("retrieval_ms", 0.0)) for r in rows]
    score_series = [float(r.get("score_attachment_ms", 0.0)) for r in rows]
    proposal_series = [float(r.get("proposal_total_ms", 0.0)) for r in rows]
    union_series = [float(r.get("proposal_union_total_ms", 0.0)) for r in rows]
    total_series = [float(r.get("total_ms", 0.0)) for r in rows]
    prompt_series = [float(r.get("prompt_tokens", 0.0)) for r in rows if float(r.get("prompt_tokens", 0.0)) > 0.0]

    return {
        "name": name,
        "run_root": run["run_root"],
        "summary_path": run["summary_path"],
        "jsonl_path": run["jsonl_path"],
        "query_count": len(rows),
        "em": _summary_metric(summary, "em", ("EM",)),
        "f1": _summary_metric(summary, "f1", ("F1",)),
        "supporting_fact_recall": _summary_metric(summary, "supporting_fact_recall", ("sf_recall",)),
        "supporting_fact_precision": _summary_metric(summary, "supporting_fact_precision", ("sf_precision",)),
        "answer_bearing_chunk_present": _summary_metric(summary, "answer_bearing_chunk_present"),
        "answer_present_but_generation_fail": _summary_metric(summary, "answer_present_but_generation_fail"),
        "prompt_tokens_avg": _summary_metric(summary, "prompt_tokens_avg", ("avg_context_tokens",)),
        "retrieval_ms": _summary_metric(summary, "retrieval_ms"),
        "score_attachment_ms": mean(score_series) if score_series else 0.0,
        "proposal_total_ms": _summary_metric(summary, "proposal_total_ms", ("proposal_time_ms",)),
        "total_ms": _summary_metric(summary, "total_ms"),
        "retrieval_ms_avg": mean(retrieval_series) if retrieval_series else 0.0,
        "retrieval_ms_p50": _percentile(retrieval_series, 0.5),
        "retrieval_ms_p95": _percentile(retrieval_series, 0.95),
        "score_attachment_ms_p95": _percentile(score_series, 0.95),
        "proposal_total_ms_avg": mean(proposal_series) if proposal_series else 0.0,
        "proposal_union_total_ms_avg": mean(union_series) if union_series else 0.0,
        "total_ms_avg": mean(total_series) if total_series else 0.0,
        "prompt_tokens_avg_from_rows": mean(prompt_series) if prompt_series else 0.0,
    }


def _build_changed_examples_three_way(
    baseline_map: Mapping[str, Mapping[str, Any]],
    optimized_map: Mapping[str, Mapping[str, Any]],
    stable_map: Mapping[str, Mapping[str, Any]],
    common_ids: Sequence[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sid in common_ids:
        b = baseline_map[sid]
        o = optimized_map[sid]
        s = stable_map[sid]
        changed = (
            b.get("normalized_prediction") != o.get("normalized_prediction")
            or b.get("normalized_prediction") != s.get("normalized_prediction")
            or b.get("selected_sentence_ids") != o.get("selected_sentence_ids")
            or b.get("selected_sentence_ids") != s.get("selected_sentence_ids")
            or b.get("rendered_sentence_ids") != o.get("rendered_sentence_ids")
            or b.get("rendered_sentence_ids") != s.get("rendered_sentence_ids")
        )
        if not changed:
            continue
        out.append(
            {
                "sample_id": sid,
                "question": b.get("question", ""),
                "baseline_prediction": b.get("prediction", ""),
                "optimized_prediction": o.get("prediction", ""),
                "stable_prediction": s.get("prediction", ""),
                "baseline_selected_sentence_ids": b.get("selected_sentence_ids", []),
                "optimized_selected_sentence_ids": o.get("selected_sentence_ids", []),
                "stable_selected_sentence_ids": s.get("selected_sentence_ids", []),
                "baseline_rendered_sentence_ids": b.get("rendered_sentence_ids", []),
                "optimized_rendered_sentence_ids": o.get("rendered_sentence_ids", []),
                "stable_rendered_sentence_ids": s.get("rendered_sentence_ids", []),
                "selected_jaccard_baseline_optimized": _jaccard(
                    b.get("selected_sentence_ids", []), o.get("selected_sentence_ids", [])
                ),
                "selected_jaccard_baseline_stable": _jaccard(
                    b.get("selected_sentence_ids", []), s.get("selected_sentence_ids", [])
                ),
                "rendered_jaccard_baseline_optimized": _jaccard(
                    b.get("rendered_sentence_ids", []), o.get("rendered_sentence_ids", [])
                ),
                "rendered_jaccard_baseline_stable": _jaccard(
                    b.get("rendered_sentence_ids", []), s.get("rendered_sentence_ids", [])
                ),
                "baseline_context": b.get("rendered_context", ""),
                "optimized_context": o.get("rendered_context", ""),
                "stable_context": s.get("rendered_context", ""),
                "candidate_score_diff_summary": {
                    "baseline_vs_optimized_common": len(
                        set((b.get("candidate_scores", {}) or {}).keys())
                        & set((o.get("candidate_scores", {}) or {}).keys())
                    ),
                    "baseline_vs_stable_common": len(
                        set((b.get("candidate_scores", {}) or {}).keys())
                        & set((s.get("candidate_scores", {}) or {}).keys())
                    ),
                },
            }
        )
        if len(out) >= limit:
            break
    return out


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
        lines.append(f"- selected_jaccard_baseline_optimized: {_fmt(ex.get('selected_jaccard_baseline_optimized'), 4)}")
        lines.append(f"- selected_jaccard_baseline_stable: {_fmt(ex.get('selected_jaccard_baseline_stable'), 4)}")
        lines.append(f"- rendered_jaccard_baseline_optimized: {_fmt(ex.get('rendered_jaccard_baseline_optimized'), 4)}")
        lines.append(f"- rendered_jaccard_baseline_stable: {_fmt(ex.get('rendered_jaccard_baseline_stable'), 4)}")
        lines.append("")
        lines.append(f"- baseline_prediction: `{_safe_text(ex.get('baseline_prediction'))}`")
        lines.append(f"- optimized_prediction: `{_safe_text(ex.get('optimized_prediction'))}`")
        lines.append(f"- stable_prediction: `{_safe_text(ex.get('stable_prediction'))}`")
        lines.append("")
        lines.append(f"- baseline_selected_sentence_ids: {_safe_text(ex.get('baseline_selected_sentence_ids'))}")
        lines.append(f"- optimized_selected_sentence_ids: {_safe_text(ex.get('optimized_selected_sentence_ids'))}")
        lines.append(f"- stable_selected_sentence_ids: {_safe_text(ex.get('stable_selected_sentence_ids'))}")
        lines.append("")
        lines.append(f"- baseline_rendered_sentence_ids: {_safe_text(ex.get('baseline_rendered_sentence_ids'))}")
        lines.append(f"- optimized_rendered_sentence_ids: {_safe_text(ex.get('optimized_rendered_sentence_ids'))}")
        lines.append(f"- stable_rendered_sentence_ids: {_safe_text(ex.get('stable_rendered_sentence_ids'))}")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md, out_json


def _decision(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("mode_metrics", {})
    drift = payload.get("pairwise_drift", {})
    parity = payload.get("score_parity", {})
    base = metrics.get("baseline", {})
    stable = metrics.get("optimized_stable_order", {})
    b_vs_s = drift.get("baseline_vs_optimized_stable_order", {})
    parity_b_vs_s = parity.get("baseline_vs_optimized_stable_order", {})

    b_ret = _safe_float(base.get("retrieval_ms"), 0.0)
    s_ret = _safe_float(stable.get("retrieval_ms"), 0.0)
    ret_gain = ((b_ret - s_ret) / b_ret) if b_ret > 0 else 0.0

    f1_drop = max(0.0, _safe_float(base.get("f1"), 0.0) - _safe_float(stable.get("f1"), 0.0))
    sf_recall_drop = max(
        0.0,
        _safe_float(base.get("supporting_fact_recall"), 0.0) - _safe_float(stable.get("supporting_fact_recall"), 0.0),
    )
    score_ms_gain = 0.0
    b_score = _safe_float(base.get("score_attachment_ms"), 0.0)
    s_score = _safe_float(stable.get("score_attachment_ms"), 0.0)
    if b_score > 0:
        score_ms_gain = (b_score - s_score) / b_score

    pred_rate = _safe_float(b_vs_s.get("prediction_changed_rate"), 0.0)
    sel_rate = _safe_float(b_vs_s.get("selected_sentence_id_changed_rate"), 0.0)
    rnd_rate = _safe_float(b_vs_s.get("rendered_sentence_id_changed_rate"), 0.0)
    sel_j = _safe_float(b_vs_s.get("selected_jaccard_avg"), 0.0)
    rnd_j = _safe_float(b_vs_s.get("rendered_jaccard_avg"), 0.0)
    score_max = _safe_float(parity_b_vs_s.get("candidate_score_max_abs_diff_max"), 0.0)

    accepted = (
        ret_gain >= 0.50
        and score_ms_gain >= 0.80
        and f1_drop <= 0.01
        and sf_recall_drop <= 0.02
        and pred_rate <= 0.10
        and sel_rate <= 0.30
        and rnd_rate <= 0.30
        and sel_j >= 0.90
        and rnd_j >= 0.90
        and score_max <= 1.0e-9
    )
    hold = (
        f1_drop > 0.01
        or sf_recall_drop > 0.02
        or pred_rate > 0.20
        or rnd_j < 0.85
    )
    if accepted:
        status = "accept_optimized_stable_order"
        reason = "Latency gain and parity/drift thresholds passed in stable-order mode."
    elif hold:
        status = "hold_or_reject"
        reason = "Behavior drift or score parity thresholds failed; keep optimization behind flags only."
    else:
        status = "candidate_with_caution"
        reason = "Latency gain is strong, but drift/parity requires further inspection before larger-scale runs."
    return {
        "status": status,
        "reason": reason,
        "metrics": {
            "retrieval_ms_gain_ratio_baseline_vs_stable": ret_gain,
            "score_attachment_ms_gain_ratio_baseline_vs_stable": score_ms_gain,
            "f1_drop_abs_baseline_vs_stable": f1_drop,
            "supporting_fact_recall_drop_abs_baseline_vs_stable": sf_recall_drop,
            "prediction_changed_rate_baseline_vs_stable": pred_rate,
            "selected_sentence_id_changed_rate_baseline_vs_stable": sel_rate,
            "rendered_sentence_id_changed_rate_baseline_vs_stable": rnd_rate,
            "selected_jaccard_avg_baseline_vs_stable": sel_j,
            "rendered_jaccard_avg_baseline_vs_stable": rnd_j,
            "candidate_score_max_abs_diff_max_baseline_vs_stable": score_max,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline/optimized/stable-order score-attachment outputs.")
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--optimized-root", required=True)
    ap.add_argument("--stable-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    baseline = _load_run_artifacts(Path(_safe_text(args.baseline_root)).resolve())
    optimized = _load_run_artifacts(Path(_safe_text(args.optimized_root)).resolve())
    stable = _load_run_artifacts(Path(_safe_text(args.stable_root)).resolve())

    mode_metrics = {
        "baseline": _mode_metrics("baseline", baseline),
        "optimized": _mode_metrics("optimized", optimized),
        "optimized_stable_order": _mode_metrics("optimized_stable_order", stable),
    }

    b_map = baseline["query_map"]
    o_map = optimized["query_map"]
    s_map = stable["query_map"]

    common_all = sorted(list(set(b_map.keys()) & set(o_map.keys()) & set(s_map.keys())))

    pairwise_drift = {
        "baseline_vs_optimized": _pair_drift_metrics("baseline", "optimized", b_map, o_map),
        "baseline_vs_optimized_stable_order": _pair_drift_metrics("baseline", "optimized_stable_order", b_map, s_map),
        "optimized_vs_optimized_stable_order": _pair_drift_metrics("optimized", "optimized_stable_order", o_map, s_map),
    }

    parity_rows: List[Dict[str, Any]] = []
    parity_summary: Dict[str, Any] = {}
    for pair_name, left_map, right_map in [
        ("baseline_vs_optimized", b_map, o_map),
        ("baseline_vs_optimized_stable_order", b_map, s_map),
        ("optimized_vs_optimized_stable_order", o_map, s_map),
    ]:
        rows, agg = _score_parity_for_pair(pair_name, left_map, right_map)
        parity_rows.extend(rows)
        parity_summary[pair_name] = agg

    per_query_path = out_root / "score_attachment_parity_per_query.jsonl"
    with per_query_path.open("w", encoding="utf-8") as f:
        for row in parity_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parity_summary_path = out_root / "score_attachment_parity_summary.json"
    parity_summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pairs": parity_summary,
        "per_query_path": str(per_query_path.resolve()),
    }
    parity_summary_path.write_text(json.dumps(parity_summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    changed_examples = _build_changed_examples_three_way(b_map, o_map, s_map, common_all, limit=10)
    changed_md, changed_json = _write_changed_examples(out_root, changed_examples)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode_metrics": mode_metrics,
        "pairwise_drift": pairwise_drift,
        "score_parity": parity_summary,
        "common_query_count_all_three": len(common_all),
        "paths": {
            "parity_per_query_jsonl": str(per_query_path.resolve()),
            "parity_summary_json": str(parity_summary_path.resolve()),
            "changed_examples_md": str(changed_md.resolve()),
            "changed_examples_json": str(changed_json.resolve()),
        },
    }
    payload["decision"] = _decision(payload)

    summary_json_path = out_root / "phase6t_score_attachment_parity_summary.json"
    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# Phase-6T Score Attachment Parity Summary")
    lines.append("")
    lines.append("## 1. Experiment Setup")
    lines.append("")
    lines.append(f"- baseline_root: `{baseline['run_root']}`")
    lines.append(f"- optimized_root: `{optimized['run_root']}`")
    lines.append(f"- stable_root: `{stable['run_root']}`")
    lines.append(f"- common_query_count_all_three: {len(common_all)}")
    lines.append("")

    lines.append("## 2. Aggregate QA Comparison")
    lines.append("")
    lines.append("| metric | baseline | optimized | optimized_stable_order |")
    lines.append("|---|---:|---:|---:|")
    for metric in [
        "em",
        "f1",
        "supporting_fact_recall",
        "supporting_fact_precision",
        "answer_bearing_chunk_present",
        "answer_present_but_generation_fail",
        "prompt_tokens_avg",
    ]:
        lines.append(
            f"| {metric} | {_fmt(mode_metrics['baseline'].get(metric),4)} | "
            f"{_fmt(mode_metrics['optimized'].get(metric),4)} | "
            f"{_fmt(mode_metrics['optimized_stable_order'].get(metric),4)} |"
        )
    lines.append("")

    lines.append("## 3. Latency Comparison")
    lines.append("")
    lines.append("| metric | baseline | optimized | optimized_stable_order |")
    lines.append("|---|---:|---:|---:|")
    for metric in [
        "retrieval_ms",
        "retrieval_ms_avg",
        "retrieval_ms_p50",
        "retrieval_ms_p95",
        "score_attachment_ms",
        "score_attachment_ms_p95",
        "proposal_total_ms",
        "proposal_total_ms_avg",
        "proposal_union_total_ms_avg",
        "total_ms",
        "total_ms_avg",
    ]:
        lines.append(
            f"| {metric} | {_fmt(mode_metrics['baseline'].get(metric),3)} | "
            f"{_fmt(mode_metrics['optimized'].get(metric),3)} | "
            f"{_fmt(mode_metrics['optimized_stable_order'].get(metric),3)} |"
        )
    lines.append("")

    lines.append("## 4. Score Parity Diagnostics")
    lines.append("")
    lines.append("| pair | candidate_score_max_abs_diff_max | candidate_score_max_abs_diff_avg | num_score_changed_candidates_avg | topk_jaccard_avg | topk_order_changed_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for pair in [
        "baseline_vs_optimized",
        "baseline_vs_optimized_stable_order",
        "optimized_vs_optimized_stable_order",
    ]:
        row = parity_summary.get(pair, {})
        lines.append(
            f"| {pair} | {_fmt(row.get('candidate_score_max_abs_diff_max'),10)} | "
            f"{_fmt(row.get('candidate_score_max_abs_diff_avg'),10)} | "
            f"{_fmt(row.get('num_score_changed_candidates_avg'),4)} | "
            f"{_fmt(row.get('topk_candidate_id_jaccard_avg'),4)} | "
            f"{_fmt(row.get('topk_score_order_changed_rate'),4)} |"
        )
    lines.append("")
    lines.append(f"- parity_per_query_jsonl: `{per_query_path}`")
    lines.append(f"- parity_summary_json: `{parity_summary_path}`")
    lines.append("")

    lines.append("## 5. Drift: Baseline vs Optimized")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for metric in [
        "prediction_changed_rate",
        "selected_sentence_id_changed_rate",
        "rendered_sentence_id_changed_rate",
        "prompt_hash_changed_rate",
        "context_hash_changed_rate",
        "selected_jaccard_avg",
        "rendered_jaccard_avg",
        "selected_chunk_jaccard_avg",
        "rendered_chunk_jaccard_avg",
    ]:
        lines.append(f"| {metric} | {_fmt(pairwise_drift['baseline_vs_optimized'].get(metric),4)} |")
    lines.append("")

    lines.append("## 6. Drift: Baseline vs Optimized Stable Order")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for metric in [
        "prediction_changed_rate",
        "selected_sentence_id_changed_rate",
        "rendered_sentence_id_changed_rate",
        "prompt_hash_changed_rate",
        "context_hash_changed_rate",
        "selected_jaccard_avg",
        "rendered_jaccard_avg",
        "selected_chunk_jaccard_avg",
        "rendered_chunk_jaccard_avg",
    ]:
        lines.append(f"| {metric} | {_fmt(pairwise_drift['baseline_vs_optimized_stable_order'].get(metric),4)} |")
    lines.append("")

    lines.append("## 7. Drift: Optimized vs Stable Order")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for metric in [
        "prediction_changed_rate",
        "selected_sentence_id_changed_rate",
        "rendered_sentence_id_changed_rate",
        "prompt_hash_changed_rate",
        "context_hash_changed_rate",
        "selected_jaccard_avg",
        "rendered_jaccard_avg",
        "selected_chunk_jaccard_avg",
        "rendered_chunk_jaccard_avg",
    ]:
        lines.append(f"| {metric} | {_fmt(pairwise_drift['optimized_vs_optimized_stable_order'].get(metric),4)} |")
    lines.append("")

    lines.append("## 8. Changed Query Examples")
    lines.append("")
    lines.append(f"- changed_examples_count: {len(changed_examples)}")
    lines.append(f"- changed_examples_md: `{changed_md}`")
    lines.append(f"- changed_examples_json: `{changed_json}`")
    lines.append("")

    lines.append("## 9. Decision")
    lines.append("")
    lines.append(f"- status: **{payload['decision']['status']}**")
    lines.append(f"- reason: {payload['decision']['reason']}")
    lines.append("")
    for k, v in payload["decision"]["metrics"].items():
        lines.append(f"- {k}: {_fmt(v, 6)}")
    lines.append("")

    summary_md_path = out_root / "PHASE6T_SCORE_ATTACHMENT_PARITY_SUMMARY.md"
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(summary_md_path))


if __name__ == "__main__":
    main()

