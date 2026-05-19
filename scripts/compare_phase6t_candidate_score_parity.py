#!/usr/bin/env python3
"""Phase-6T score-attachment parity repair comparison (baseline vs optimized)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

EPS = 1.0e-9
COMPONENTS = [
    "semantic_score",
    "graph_score",
    "bridge_score",
    "corridor_score",
    "redundancy_score",
    "answerability_score",
]
SOURCE_COMPONENTS = ["semantic", "graph", "bridge", "corridor", "redundancy"]
MISSING_FLAG_BY_COMPONENT = {
    "semantic_score": "semantic_missing",
    "graph_score": "graph_missing",
    "bridge_score": "bridge_missing",
    "corridor_score": "corridor_missing",
    "redundancy_score": "redundancy_missing",
    "answerability_score": "answerability_missing",
}
SOURCE_KEY_BY_COMPONENT = {
    "semantic_score": "semantic",
    "graph_score": "graph",
    "bridge_score": "bridge",
    "corridor_score": "corridor",
    "redundancy_score": "redundancy",
    "answerability_score": "answerability",
}


def _safe_text(v: Any) -> str:
    return str(v or "").strip()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return _safe_text(v).lower() in {"1", "true", "yes", "on", "y"}


def _safe_list(v: Any) -> List[str]:
    if isinstance(v, (list, tuple)):
        return [_safe_text(x) for x in v if _safe_text(x)]
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


def _sha1_text(text: str) -> str:
    s = _safe_text(text)
    if not s:
        return ""
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_prediction(text: str) -> str:
    s = _safe_text(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(_ordered_unique([_safe_text(x) for x in a]))
    sb = set(_ordered_unique([_safe_text(x) for x in b]))
    if not sa and not sb:
        return 1.0
    u = sa | sb
    if not u:
        return 1.0
    return float(len(sa & sb)) / float(len(u))


def _fmt(v: Any, nd: int = 6) -> str:
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
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    return {}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t:
            continue
        obj = json.loads(t)
        if isinstance(obj, Mapping):
            out.append(dict(obj))
    return out


def _extract_selected_sentence_ids(row: Mapping[str, Any]) -> List[str]:
    top = _safe_list(row.get("retrieval_selected_sentence_ids"))
    if top:
        return _ordered_unique(top)
    retrieval = row.get("retrieval")
    if isinstance(retrieval, Mapping):
        return _ordered_unique(_safe_list(retrieval.get("selected_sentence_ids")))
    rendered = row.get("rendered")
    if isinstance(rendered, Mapping):
        return _ordered_unique(_safe_list(rendered.get("retrieval_selected_sentence_ids")))
    return []


def _extract_rendered_sentence_ids(row: Mapping[str, Any]) -> List[str]:
    top = _safe_list(row.get("rendered_sentence_ids"))
    if top:
        return _ordered_unique(top)
    rendered = row.get("rendered")
    if isinstance(rendered, Mapping):
        return _ordered_unique(_safe_list(rendered.get("sentence_ids")))
    return []


def _extract_selected_chunk_ids(row: Mapping[str, Any]) -> List[str]:
    top = _safe_list(row.get("retrieval_selected_chunk_ids"))
    if top:
        return _ordered_unique(top)
    retrieval = row.get("retrieval")
    if isinstance(retrieval, Mapping):
        return _ordered_unique(_safe_list(retrieval.get("selected_chunk_ids")))
    return []


def _extract_rendered_chunk_ids(row: Mapping[str, Any]) -> List[str]:
    top = _safe_list(row.get("rendered_chunk_ids"))
    if top:
        return _ordered_unique(top)
    rendered = row.get("rendered")
    if isinstance(rendered, Mapping):
        return _ordered_unique(_safe_list(rendered.get("chunk_ids")))
    return []


def _extract_rendered_text(row: Mapping[str, Any]) -> str:
    rendered = row.get("rendered")
    if isinstance(rendered, Mapping):
        t = _safe_text(rendered.get("text"))
        if t:
            return t
        sents = _safe_list(rendered.get("sentences"))
        if sents:
            return "\n".join(sents)
    rendering = row.get("rendering")
    if isinstance(rendering, Mapping):
        t = _safe_text(rendering.get("context"))
        if t:
            return t
    return ""


def _extract_prompt_variant(row: Mapping[str, Any]) -> str:
    gdiag = row.get("generation_diagnostics")
    if isinstance(gdiag, Mapping):
        pv = _safe_text(gdiag.get("prompt_variant"))
        if pv:
            return pv
    rendering = row.get("rendering")
    if isinstance(rendering, Mapping):
        pv = _safe_text(rendering.get("prompt_variant"))
        if pv:
            return pv
    return ""


def _load_run_query_map(run_root: Path) -> Dict[str, Dict[str, Any]]:
    qpath = _find_latest(run_root, "rag_query_results.jsonl")
    rows = _load_jsonl(qpath)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = _safe_text(row.get("sample_id"))
        if not sid:
            idx = row.get("sample_index")
            if idx is not None:
                sid = f"sample_index::{idx}"
        if not sid:
            continue
        pred = _safe_text(row.get("prediction"))
        rendered_text = _extract_rendered_text(row)
        prompt_variant = _extract_prompt_variant(row)
        prompt_signature = "\n".join([
            _safe_text(row.get("question")),
            _safe_text(prompt_variant),
            _safe_text(rendered_text),
        ])
        out[sid] = {
            "sample_id": sid,
            "question": _safe_text(row.get("question")),
            "prediction": pred,
            "normalized_prediction": _normalize_prediction(pred),
            "selected_sentence_ids": _extract_selected_sentence_ids(row),
            "rendered_sentence_ids": _extract_rendered_sentence_ids(row),
            "selected_chunk_ids": _extract_selected_chunk_ids(row),
            "rendered_chunk_ids": _extract_rendered_chunk_ids(row),
            "rendered_text": rendered_text,
            "rendered_text_hash": _sha1_text(rendered_text),
            "prompt_variant": prompt_variant,
            # Prompt hash proxy (prompt text is not persisted in current artifact schema).
            "prompt_hash": _sha1_text(prompt_signature),
        }
    return out


def _load_rag_summary_metrics(run_root: Path) -> Dict[str, float]:
    path = _find_latest(run_root, "rag_summary.json")
    d = _load_json(path)
    keys = [
        "em",
        "f1",
        "supporting_fact_recall",
        "supporting_fact_precision",
        "answer_bearing_chunk_present",
        "answer_present_but_generation_fail",
        "avg_prompt_tokens",
        "retrieval_ms",
        "proposal_total_ms_avg",
        "proposal_union_total_ms_avg",
        "path_candidate_expansion_ms_avg",
        "total_ms",
        "score_attachment_ms_avg",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = _safe_float(d.get(k), 0.0)
    return out


def _candidate_key(row: Mapping[str, Any]) -> str:
    key = _safe_text(row.get("candidate_key_normalized"))
    if key:
        return key
    return _safe_text(row.get("candidate_id")).lower()


def _ensure_candidate_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["candidate_key_normalized"] = _candidate_key(out)
    out.setdefault("candidate_id", _safe_text(out.get("candidate_key_normalized")))
    out.setdefault("candidate_key_raw", _safe_text(out.get("candidate_id")))
    out.setdefault("candidate_type", "unknown")
    out.setdefault("source_stage", "unknown")
    out.setdefault("source_title", "")
    out.setdefault("chunk_id", "")
    out.setdefault("sentence_id", "")
    out.setdefault("text_hash", "")
    out.setdefault("included_before_score_attachment", True)
    out.setdefault("included_after_score_attachment", True)
    out.setdefault("selected", False)
    out.setdefault("rendered", False)
    out.setdefault("score_source", {})
    out.setdefault("missing_flags", {})
    out.setdefault("score_merge_priority", [])
    return out


def _extract_mode_rows(path: Path, mode_name: str) -> Dict[str, List[Dict[str, Any]]]:
    rows = _load_jsonl(path)
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = _safe_text(row.get("sample_id"))
        if not sid:
            continue
        mode = _safe_text(row.get("mode"))
        if mode and mode != mode_name:
            continue
        by_sample[sid].append(_ensure_candidate_fields(row))
    return by_sample


def _sorted_topk_keys(cands: Mapping[str, Mapping[str, Any]], k: int) -> List[str]:
    items = sorted(
        cands.items(),
        key=lambda kv: (_safe_float(kv[1].get("final_score"), 0.0), _safe_text(kv[1].get("candidate_id"))),
        reverse=True,
    )
    return [str(key) for key, _ in items[:k]]


def _component_missing(row: Mapping[str, Any], comp: str) -> bool:
    miss = dict(row.get("missing_flags", {}) or {})
    flag = MISSING_FLAG_BY_COMPONENT.get(comp, "")
    if flag and flag in miss:
        return _safe_bool(miss.get(flag))
    return row.get(comp) is None


def _component_source(row: Mapping[str, Any], comp: str) -> str:
    src = dict(row.get("score_source", {}) or {})
    key = SOURCE_KEY_BY_COMPONENT.get(comp, "")
    if key:
        return _safe_text(src.get(key))
    return ""


def _classify_reason(b: Mapping[str, Any] | None, o: Mapping[str, Any] | None) -> str:
    if b is None:
        return "candidate_missing_in_baseline"
    if o is None:
        return "candidate_missing_in_optimized"
    if _safe_text(b.get("candidate_id")) != _safe_text(o.get("candidate_id")):
        return "candidate_key_normalization_diff"

    for comp in COMPONENTS:
        if _component_missing(b, comp) != _component_missing(o, comp):
            return "missing_score_default_diff"

    max_comp = ""
    max_delta = 0.0
    for comp in COMPONENTS:
        d = abs(_safe_float(b.get(comp), 0.0) - _safe_float(o.get(comp), 0.0))
        if d > max_delta:
            max_comp = comp
            max_delta = d
    if max_delta > EPS:
        return {
            "semantic_score": "semantic_score_diff",
            "graph_score": "graph_score_diff",
            "bridge_score": "bridge_score_diff",
            "corridor_score": "corridor_score_diff",
            "redundancy_score": "redundancy_score_diff",
            "answerability_score": "answerability_score_diff",
        }.get(max_comp, "unknown")

    final_delta = abs(_safe_float(b.get("final_score"), 0.0) - _safe_float(o.get("final_score"), 0.0))
    if final_delta > EPS:
        return "score_merge_priority_diff"
    return "unknown"


def _classify_prediction_change(
    *,
    prediction_changed: bool,
    b_selected: List[str],
    o_selected: List[str],
    b_rendered: List[str],
    o_rendered: List[str],
    b_context_hash: str,
    o_context_hash: str,
    b_prompt_hash: str,
    o_prompt_hash: str,
) -> str:
    if not prediction_changed:
        return "unchanged"

    b_sel_set = set(b_selected)
    o_sel_set = set(o_selected)
    b_rnd_set = set(b_rendered)
    o_rnd_set = set(o_rendered)

    selected_set_equal = (b_sel_set == o_sel_set)
    rendered_set_equal = (b_rnd_set == o_rnd_set)
    selected_order_equal = (b_selected == o_selected)
    rendered_order_equal = (b_rendered == o_rendered)
    context_hash_equal = (_safe_text(b_context_hash) == _safe_text(o_context_hash))
    prompt_hash_equal = (_safe_text(b_prompt_hash) == _safe_text(o_prompt_hash))

    if selected_set_equal and rendered_set_equal and context_hash_equal and prompt_hash_equal:
        if selected_order_equal and rendered_order_equal:
            return "generation_nondeterminism_candidate"
        return "render_order_drift"

    if selected_set_equal and rendered_set_equal and context_hash_equal and (not prompt_hash_equal):
        return "prompt_metadata_drift"

    return "retrieval_drift"


def _candidate_output_impact(
    *,
    candidate_key: str,
    topk_keys: Sequence[str],
    row: Mapping[str, Any] | None,
) -> int:
    if candidate_key in set([_safe_text(x) for x in list(topk_keys or []) if _safe_text(x)]):
        return 1
    if isinstance(row, Mapping):
        if _safe_bool(row.get("selected")) or _safe_bool(row.get("rendered")):
            return 1
    return 0


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _build_candidate_diff_md(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Candidate Diff Examples", ""]
    if not rows:
        lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    lines.append("| sample_id | missing_side | candidate_key_normalized | source_stage | in_topk8_or_selected_or_rendered |")
    lines.append("|---|---|---|---|---:|")
    for r in rows[:60]:
        lines.append(
            f"| {r.get('sample_id','')} | {r.get('missing_side','')} | {r.get('candidate_key_normalized','')} | {r.get('source_stage','')} | {int(r.get('output_impact',0))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_score_diff_md(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Score Diff Outlier Examples", ""]
    if not rows:
        lines.append("- none")
        lines.append("")
        return "\n".join(lines)
    lines.append("| sample_id | candidate_key_normalized | component | abs_diff | baseline_source | optimized_source | in_topk8 | selected_b | selected_o | rendered_b | rendered_o |")
    lines.append("|---|---|---|---:|---|---|---:|---:|---:|---:|---:|")
    for r in rows[:80]:
        lines.append(
            "| {sid} | {key} | {comp} | {diff} | {bs} | {os} | {topk} | {sb} | {so} | {rb} | {ro} |".format(
                sid=r.get("sample_id", ""),
                key=r.get("candidate_key_normalized", ""),
                comp=r.get("component", ""),
                diff=_fmt(r.get("abs_diff", 0.0), 6),
                bs=r.get("baseline_source", ""),
                os=r.get("optimized_source", ""),
                topk=int(r.get("in_topk@8", 0)),
                sb=int(r.get("selected_by_baseline", 0)),
                so=int(r.get("selected_by_optimized", 0)),
                rb=int(r.get("rendered_by_baseline", 0)),
                ro=int(r.get("rendered_by_optimized", 0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_prediction_classification_md(rows: Sequence[Mapping[str, Any]], counts: Mapping[str, int], total: int) -> str:
    lines = ["# Prediction Change Classification", ""]
    lines.append("| class | count | rate |")
    lines.append("|---|---:|---:|")
    for cls in [
        "retrieval_drift",
        "generation_nondeterminism_candidate",
        "render_order_drift",
        "prompt_metadata_drift",
        "unchanged",
    ]:
        c = int(counts.get(cls, 0))
        r = (float(c) / float(total)) if total > 0 else 0.0
        lines.append(f"| {cls} | {c} | {_fmt(r, 6)} |")
    lines.append("")
    lines.append("## Changed Queries")
    if not rows:
        lines.append("- none")
    else:
        lines.append("| sample_id | class | prediction_changed | selected_sentence_jaccard | rendered_sentence_jaccard | context_hash_equal | prompt_hash_equal |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in rows[:50]:
            lines.append(
                f"| {r.get('sample_id','')} | {r.get('prediction_change_class','')} | {int(r.get('prediction_changed',0))} | {_fmt(r.get('selected_sentence_jaccard',0.0),4)} | {_fmt(r.get('rendered_sentence_jaccard',0.0),4)} | {int(r.get('context_hash_equal',0))} | {int(r.get('prompt_hash_equal',0))} |"
            )
    lines.append("")
    return "\n".join(lines)


def _build_summary_md(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# PHASE6T Score Attachment Parity Repair Summary")
    lines.append("")

    lines.append("## 1. Candidate Set Parity")
    csp = dict(summary.get("candidate_set_parity", {}) or {})
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for k in [
        "total_queries",
        "avg_num_baseline_candidates",
        "avg_num_optimized_candidates",
        "avg_num_common_candidates",
        "candidate_missing_in_baseline",
        "candidate_missing_in_optimized",
    ]:
        lines.append(f"| {k} | {_fmt(csp.get(k, 0.0), 6)} |")
    lines.append("")

    lines.append("## 2. Component Score Parity")
    comp = dict(summary.get("component_score_parity", {}) or {})
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for k in [
        "final_score_max_abs_diff",
        "final_score_mean_abs_diff",
        "semantic_score_max_abs_diff",
        "graph_score_max_abs_diff",
        "bridge_score_max_abs_diff",
        "corridor_score_max_abs_diff",
        "topk_jaccard@8",
        "selected_sentence_jaccard",
        "rendered_sentence_jaccard",
        "prediction_changed_rate",
        "retrieval_drift_prediction_changed_rate",
    ]:
        lines.append(f"| {k} | {_fmt(comp.get(k, 0.0), 6)} |")
    lines.append("")

    lines.append("## 3. Missing Score Default Audit")
    msd = dict(summary.get("missing_score_default_audit", {}) or {})
    lines.append(f"- missing_score_default_diff_count: {int(msd.get('missing_score_default_diff_count', 0))}")
    by_comp = dict(msd.get("missing_score_default_diff_by_component", {}) or {})
    if by_comp:
        lines.append("| component | count |")
        lines.append("|---|---:|")
        for k, v in sorted(by_comp.items(), key=lambda x: (-int(x[1]), str(x[0]))):
            lines.append(f"| {k} | {int(v)} |")
    lines.append("")

    lines.append("## 4. Score Source Priority Audit")
    sspa = dict(summary.get("score_source_priority_audit", {}) or {})
    lines.append("| component | source_priority_same | mismatched_count |")
    lines.append("|---|---|---:|")
    for c in SOURCE_COMPONENTS:
        d = dict(sspa.get(c, {}) or {})
        lines.append(f"| {c} | {str(bool(d.get('same', False))).lower()} | {int(d.get('mismatch_count', 0))} |")
    lines.append("")

    lines.append("## 5. Candidate Generation Parity")
    cgp = dict(summary.get("candidate_generation_parity", {}) or {})
    lines.append("- candidate_missing_in_optimized_by_source_stage:")
    for k, v in sorted(dict(cgp.get("candidate_missing_in_optimized_by_source_stage", {}) or {}).items(), key=lambda x: (-int(x[1]), str(x[0]))):
        lines.append(f"  - {k}: {int(v)}")
    lines.append("- candidate_missing_in_baseline_by_source_stage:")
    for k, v in sorted(dict(cgp.get("candidate_missing_in_baseline_by_source_stage", {}) or {}).items(), key=lambda x: (-int(x[1]), str(x[0]))):
        lines.append(f"  - {k}: {int(v)}")
    lines.append("")

    lines.append("## 6. Score Diff Outlier Analysis")
    for name in ["top_final_score_diff", "top_semantic_score_diff", "top_graph_score_diff"]:
        arr = list(summary.get("score_diff_outliers", {}).get(name, []) or [])
        lines.append(f"- {name}: {len(arr)} rows")
    lines.append("")

    lines.append("## 7. Prediction Change Classification")
    pcc = dict(summary.get("prediction_change_classification", {}) or {})
    for k in ["retrieval_drift", "generation_nondeterminism_candidate", "render_order_drift", "prompt_metadata_drift", "unchanged"]:
        lines.append(f"- {k}: {int(pcc.get(k, 0))}")
    lines.append("")

    lines.append("## 8. Latency Retention")
    lat = dict(summary.get("latency_retention", {}) or {})
    lines.append("| metric | baseline | optimized | delta | delta_% |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in ["retrieval_ms", "proposal_total_ms_avg", "total_ms", "f1", "supporting_fact_recall", "supporting_fact_precision"]:
        d = dict(lat.get(k, {}) or {})
        lines.append(f"| {k} | {_fmt(d.get('baseline',0.0),6)} | {_fmt(d.get('optimized',0.0),6)} | {_fmt(d.get('delta',0.0),6)} | {_fmt(d.get('delta_pct',0.0),2)} |")
    lines.append("")

    lines.append("## 9. Output Impact Assessment")
    oia = dict(summary.get("output_impact_assessment", {}) or {})
    lines.append(f"- missing_candidate_output_impact_count: {int(oia.get('missing_candidate_output_impact_count', 0))}")
    lines.append(f"- score_diff_output_impact_count: {int(oia.get('score_diff_output_impact_count', 0))}")
    lines.append(f"- tail_only_candidate_drift: {str(bool(oia.get('tail_only_candidate_drift', False))).lower()}")
    lines.append("")

    lines.append("## 10. Decision")
    lines.append(f"- {summary.get('decision', 'hold')}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare phase6t baseline/optimized candidate score parity")
    ap.add_argument("--baseline-candidates", required=True)
    ap.add_argument("--optimized-candidates", required=True)
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--optimized-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    score_root = out_root / "score_parity"
    score_root.mkdir(parents=True, exist_ok=True)

    baseline_by_sample = _extract_mode_rows(Path(args.baseline_candidates).resolve(), "baseline")
    optimized_by_sample = _extract_mode_rows(Path(args.optimized_candidates).resolve(), "optimized")

    baseline_rows_all = [r for sid in sorted(baseline_by_sample.keys()) for r in baseline_by_sample[sid]]
    optimized_rows_all = [r for sid in sorted(optimized_by_sample.keys()) for r in optimized_by_sample[sid]]

    _write_jsonl(score_root / "baseline_candidate_set.jsonl", baseline_rows_all)
    _write_jsonl(score_root / "optimized_candidate_set.jsonl", optimized_rows_all)
    _write_jsonl(score_root / "baseline_candidates.jsonl", baseline_rows_all)
    _write_jsonl(score_root / "optimized_candidates.jsonl", optimized_rows_all)

    b_query = _load_run_query_map(Path(args.baseline_root).resolve())
    o_query = _load_run_query_map(Path(args.optimized_root).resolve())
    b_summary = _load_rag_summary_metrics(Path(args.baseline_root).resolve())
    o_summary = _load_rag_summary_metrics(Path(args.optimized_root).resolve())

    sample_ids = sorted(set(baseline_by_sample.keys()) | set(optimized_by_sample.keys()))

    candidate_set_diff_rows: List[Dict[str, Any]] = []
    component_score_diff_rows: List[Dict[str, Any]] = []
    missing_default_rows: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []
    changed_query_rows: List[Dict[str, Any]] = []
    candidate_diff_examples: List[Dict[str, Any]] = []
    score_diff_examples: List[Dict[str, Any]] = []

    reason_counts: Counter[str] = Counter()
    missing_in_opt_stage: Counter[str] = Counter()
    missing_in_base_stage: Counter[str] = Counter()
    source_same_ctr = {c: Counter() for c in SOURCE_COMPONENTS}

    class_counts: Counter[str] = Counter()

    agg: Dict[str, List[float]] = defaultdict(list)

    for sid in sample_ids:
        b_rows = list(baseline_by_sample.get(sid, []))
        o_rows = list(optimized_by_sample.get(sid, []))
        b_map = {_candidate_key(r): r for r in b_rows if _candidate_key(r)}
        o_map = {_candidate_key(r): r for r in o_rows if _candidate_key(r)}

        b_top8 = _sorted_topk_keys(b_map, 8)
        o_top8 = _sorted_topk_keys(o_map, 8)
        b_top20 = _sorted_topk_keys(b_map, 20)
        o_top20 = _sorted_topk_keys(o_map, 20)

        common = sorted(set(b_map.keys()) & set(o_map.keys()))
        missing_in_baseline = sorted(set(o_map.keys()) - set(b_map.keys()))
        missing_in_optimized = sorted(set(b_map.keys()) - set(o_map.keys()))

        for key in missing_in_optimized:
            r = b_map.get(key, {})
            stage = _safe_text(r.get("source_stage")) or "unknown"
            missing_in_opt_stage[stage] += 1
            impact = _candidate_output_impact(candidate_key=key, topk_keys=b_top8, row=r)
            candidate_diff_examples.append(
                {
                    "sample_id": sid,
                    "missing_side": "optimized",
                    "candidate_key_normalized": key,
                    "source_stage": stage,
                    "output_impact": impact,
                }
            )
        for key in missing_in_baseline:
            r = o_map.get(key, {})
            stage = _safe_text(r.get("source_stage")) or "unknown"
            missing_in_base_stage[stage] += 1
            impact = _candidate_output_impact(candidate_key=key, topk_keys=o_top8, row=r)
            candidate_diff_examples.append(
                {
                    "sample_id": sid,
                    "missing_side": "baseline",
                    "candidate_key_normalized": key,
                    "source_stage": stage,
                    "output_impact": impact,
                }
            )

        candidate_set_diff_rows.append(
            {
                "sample_id": sid,
                "num_baseline_candidates": int(len(b_map)),
                "num_optimized_candidates": int(len(o_map)),
                "num_common_candidates": int(len(common)),
                "num_missing_in_baseline": int(len(missing_in_baseline)),
                "num_missing_in_optimized": int(len(missing_in_optimized)),
                "missing_candidate_examples": {
                    "missing_in_baseline": list(missing_in_baseline[:20]),
                    "missing_in_optimized": list(missing_in_optimized[:20]),
                },
                "missing_candidate_source_stage_breakdown": {
                    "missing_in_baseline": dict(Counter([_safe_text(o_map[k].get("source_stage")) or "unknown" for k in missing_in_baseline])),
                    "missing_in_optimized": dict(Counter([_safe_text(b_map[k].get("source_stage")) or "unknown" for k in missing_in_optimized])),
                },
            }
        )

        final_diffs: List[float] = []
        comp_max = {c: 0.0 for c in COMPONENTS}
        num_changed = Counter()

        for key in common:
            b = b_map[key]
            o = o_map[key]

            delta = {c: abs(_safe_float(b.get(c), 0.0) - _safe_float(o.get(c), 0.0)) for c in COMPONENTS}
            d_final = abs(_safe_float(b.get("final_score"), 0.0) - _safe_float(o.get("final_score"), 0.0))
            final_diffs.append(float(d_final))
            if d_final > EPS:
                num_changed["final_score"] += 1

            for comp in COMPONENTS:
                comp_max[comp] = max(comp_max[comp], float(delta[comp]))
                if delta[comp] > EPS:
                    num_changed[comp] += 1

            reason = _classify_reason(b, o)
            reason_counts[reason] += 1

            for comp in ["semantic_score", "graph_score", "bridge_score", "corridor_score", "redundancy_score"]:
                source_comp = SOURCE_KEY_BY_COMPONENT[comp]
                bs = _component_source(b, comp) or "none"
                os = _component_source(o, comp) or "none"
                if bs == os:
                    source_same_ctr[source_comp]["same"] += 1
                else:
                    source_same_ctr[source_comp]["diff"] += 1

            for comp in COMPONENTS:
                b_missing = _component_missing(b, comp)
                o_missing = _component_missing(o, comp)
                if b_missing == o_missing:
                    continue
                missing_default_rows.append(
                    {
                        "sample_id": sid,
                        "candidate_key_normalized": key,
                        "component": comp.replace("_score", ""),
                        "baseline_value": (None if b_missing else _safe_float(b.get(comp), 0.0)),
                        "optimized_value": (None if o_missing else _safe_float(o.get(comp), 0.0)),
                        "baseline_source": _component_source(b, comp) or "none",
                        "optimized_source": _component_source(o, comp) or "none",
                        "baseline_default_rule": "missing_component->None; final_score fallback: graph->semantic->0",
                        "optimized_default_rule": "missing_component->None; final_score fallback: graph->semantic->0",
                    }
                )

            in_top8 = int((key in b_top8) or (key in o_top8))
            selected_b = int(_safe_bool(b.get("selected")))
            selected_o = int(_safe_bool(o.get("selected")))
            rendered_b = int(_safe_bool(b.get("rendered")))
            rendered_o = int(_safe_bool(o.get("rendered")))

            row = {
                "sample_id": sid,
                "candidate_key_normalized": key,
                "source_stage": _safe_text(b.get("source_stage") or o.get("source_stage") or "unknown"),
                "baseline": {
                    "candidate_id": _safe_text(b.get("candidate_id")),
                    "source_title": _safe_text(b.get("source_title")),
                    "chunk_id": _safe_text(b.get("chunk_id")),
                    "sentence_id": _safe_text(b.get("sentence_id")),
                    "text_hash": _safe_text(b.get("text_hash")),
                    "final_score": _safe_float(b.get("final_score"), 0.0),
                    "semantic_score": (None if _component_missing(b, "semantic_score") else _safe_float(b.get("semantic_score"), 0.0)),
                    "graph_score": (None if _component_missing(b, "graph_score") else _safe_float(b.get("graph_score"), 0.0)),
                    "bridge_score": (None if _component_missing(b, "bridge_score") else _safe_float(b.get("bridge_score"), 0.0)),
                    "corridor_score": (None if _component_missing(b, "corridor_score") else _safe_float(b.get("corridor_score"), 0.0)),
                    "redundancy_score": (None if _component_missing(b, "redundancy_score") else _safe_float(b.get("redundancy_score"), 0.0)),
                    "answerability_score": (None if _component_missing(b, "answerability_score") else _safe_float(b.get("answerability_score"), 0.0)),
                    "score_source": dict(b.get("score_source", {}) or {}),
                    "missing_flags": dict(b.get("missing_flags", {}) or {}),
                    "rank_before_render": int(_safe_float(b.get("rank_before_render"), 0)),
                },
                "optimized": {
                    "candidate_id": _safe_text(o.get("candidate_id")),
                    "source_title": _safe_text(o.get("source_title")),
                    "chunk_id": _safe_text(o.get("chunk_id")),
                    "sentence_id": _safe_text(o.get("sentence_id")),
                    "text_hash": _safe_text(o.get("text_hash")),
                    "final_score": _safe_float(o.get("final_score"), 0.0),
                    "semantic_score": (None if _component_missing(o, "semantic_score") else _safe_float(o.get("semantic_score"), 0.0)),
                    "graph_score": (None if _component_missing(o, "graph_score") else _safe_float(o.get("graph_score"), 0.0)),
                    "bridge_score": (None if _component_missing(o, "bridge_score") else _safe_float(o.get("bridge_score"), 0.0)),
                    "corridor_score": (None if _component_missing(o, "corridor_score") else _safe_float(o.get("corridor_score"), 0.0)),
                    "redundancy_score": (None if _component_missing(o, "redundancy_score") else _safe_float(o.get("redundancy_score"), 0.0)),
                    "answerability_score": (None if _component_missing(o, "answerability_score") else _safe_float(o.get("answerability_score"), 0.0)),
                    "score_source": dict(o.get("score_source", {}) or {}),
                    "missing_flags": dict(o.get("missing_flags", {}) or {}),
                    "rank_before_render": int(_safe_float(o.get("rank_before_render"), 0)),
                },
                "delta": {
                    "final_score": float(d_final),
                    "semantic_score": float(delta["semantic_score"]),
                    "graph_score": float(delta["graph_score"]),
                    "bridge_score": float(delta["bridge_score"]),
                    "corridor_score": float(delta["corridor_score"]),
                    "redundancy_score": float(delta["redundancy_score"]),
                    "answerability_score": float(delta["answerability_score"]),
                },
                "reason": reason,
                "in_topk@8": int(in_top8),
                "selected_by_baseline": int(selected_b),
                "selected_by_optimized": int(selected_o),
                "rendered_by_baseline": int(rendered_b),
                "rendered_by_optimized": int(rendered_o),
            }
            component_score_diff_rows.append(row)

            if d_final > EPS:
                score_diff_examples.append(
                    {
                        "sample_id": sid,
                        "candidate_key_normalized": key,
                        "component": "final_score",
                        "abs_diff": float(d_final),
                        "baseline_source": _component_source(b, "graph_score") or "none",
                        "optimized_source": _component_source(o, "graph_score") or "none",
                        "source_stage": row["source_stage"],
                        "in_topk@8": int(in_top8),
                        "selected_by_baseline": int(selected_b),
                        "selected_by_optimized": int(selected_o),
                        "rendered_by_baseline": int(rendered_b),
                        "rendered_by_optimized": int(rendered_o),
                    }
                )
            for comp in ["semantic_score", "graph_score"]:
                d = float(delta[comp])
                if d > EPS:
                    score_diff_examples.append(
                        {
                            "sample_id": sid,
                            "candidate_key_normalized": key,
                            "component": comp,
                            "abs_diff": d,
                            "baseline_source": _component_source(b, comp) or "none",
                            "optimized_source": _component_source(o, comp) or "none",
                            "source_stage": row["source_stage"],
                            "in_topk@8": int(in_top8),
                            "selected_by_baseline": int(selected_b),
                            "selected_by_optimized": int(selected_o),
                            "rendered_by_baseline": int(rendered_b),
                            "rendered_by_optimized": int(rendered_o),
                        }
                    )

        bq = dict(b_query.get(sid, {}) or {})
        oq = dict(o_query.get(sid, {}) or {})

        b_sel = _ordered_unique(_safe_list(bq.get("selected_sentence_ids")))
        o_sel = _ordered_unique(_safe_list(oq.get("selected_sentence_ids")))
        b_rnd = _ordered_unique(_safe_list(bq.get("rendered_sentence_ids")))
        o_rnd = _ordered_unique(_safe_list(oq.get("rendered_sentence_ids")))
        b_sel_chunk = _ordered_unique(_safe_list(bq.get("selected_chunk_ids")))
        o_sel_chunk = _ordered_unique(_safe_list(oq.get("selected_chunk_ids")))
        b_rnd_chunk = _ordered_unique(_safe_list(bq.get("rendered_chunk_ids")))
        o_rnd_chunk = _ordered_unique(_safe_list(oq.get("rendered_chunk_ids")))

        b_pred = _safe_text(bq.get("prediction"))
        o_pred = _safe_text(oq.get("prediction"))
        pred_changed = int(_normalize_prediction(b_pred) != _normalize_prediction(o_pred))

        cls = _classify_prediction_change(
            prediction_changed=bool(pred_changed),
            b_selected=b_sel,
            o_selected=o_sel,
            b_rendered=b_rnd,
            o_rendered=o_rnd,
            b_context_hash=_safe_text(bq.get("rendered_text_hash")),
            o_context_hash=_safe_text(oq.get("rendered_text_hash")),
            b_prompt_hash=_safe_text(bq.get("prompt_hash")),
            o_prompt_hash=_safe_text(oq.get("prompt_hash")),
        )
        class_counts[cls] += 1

        qrow = {
            "sample_id": sid,
            "question": _safe_text(bq.get("question") or oq.get("question")),
            "num_baseline_candidates": int(len(b_map)),
            "num_optimized_candidates": int(len(o_map)),
            "num_common_candidates": int(len(common)),
            "num_missing_in_baseline": int(len(missing_in_baseline)),
            "num_missing_in_optimized": int(len(missing_in_optimized)),
            "final_score_max_abs_diff": (max(final_diffs) if final_diffs else 0.0),
            "final_score_mean_abs_diff": (mean(final_diffs) if final_diffs else 0.0),
            "semantic_score_max_abs_diff": float(comp_max["semantic_score"]),
            "graph_score_max_abs_diff": float(comp_max["graph_score"]),
            "bridge_score_max_abs_diff": float(comp_max["bridge_score"]),
            "corridor_score_max_abs_diff": float(comp_max["corridor_score"]),
            "redundancy_score_max_abs_diff": float(comp_max["redundancy_score"]),
            "answerability_score_max_abs_diff": float(comp_max["answerability_score"]),
            "topk_jaccard@8": _jaccard(b_top8, o_top8),
            "topk_jaccard@20": _jaccard(b_top20, o_top20),
            "topk_order_changed": int(b_top8 != o_top8),
            "selected_sentence_jaccard": _jaccard(b_sel, o_sel),
            "rendered_sentence_jaccard": _jaccard(b_rnd, o_rnd),
            "selected_chunk_jaccard": _jaccard(b_sel_chunk, o_sel_chunk),
            "rendered_chunk_jaccard": _jaccard(b_rnd_chunk, o_rnd_chunk),
            "prediction_changed": int(pred_changed),
            "prediction_change_class": cls,
            "context_hash_equal": int(_safe_text(bq.get("rendered_text_hash")) == _safe_text(oq.get("rendered_text_hash"))),
            "prompt_hash_equal": int(_safe_text(bq.get("prompt_hash")) == _safe_text(oq.get("prompt_hash"))),
            "baseline_prediction": b_pred,
            "optimized_prediction": o_pred,
            "baseline_selected_sentence_ids": b_sel,
            "optimized_selected_sentence_ids": o_sel,
            "baseline_rendered_sentence_ids": b_rnd,
            "optimized_rendered_sentence_ids": o_rnd,
            "baseline_rendered_text_hash": _safe_text(bq.get("rendered_text_hash")),
            "optimized_rendered_text_hash": _safe_text(oq.get("rendered_text_hash")),
            "baseline_prompt_hash": _safe_text(bq.get("prompt_hash")),
            "optimized_prompt_hash": _safe_text(oq.get("prompt_hash")),
            "baseline_prompt_variant": _safe_text(bq.get("prompt_variant")),
            "optimized_prompt_variant": _safe_text(oq.get("prompt_variant")),
            "baseline_rendered_context": _safe_text(bq.get("rendered_text")),
            "optimized_rendered_context": _safe_text(oq.get("rendered_text")),
            "missing_in_baseline_keys": list(missing_in_baseline[:20]),
            "missing_in_optimized_keys": list(missing_in_optimized[:20]),
        }
        per_query_rows.append(qrow)

        if int(pred_changed) == 1:
            changed_query_rows.append(dict(qrow))

        for k in [
            "num_baseline_candidates",
            "num_optimized_candidates",
            "num_common_candidates",
            "num_missing_in_baseline",
            "num_missing_in_optimized",
            "final_score_max_abs_diff",
            "final_score_mean_abs_diff",
            "semantic_score_max_abs_diff",
            "graph_score_max_abs_diff",
            "bridge_score_max_abs_diff",
            "corridor_score_max_abs_diff",
            "redundancy_score_max_abs_diff",
            "answerability_score_max_abs_diff",
            "topk_jaccard@8",
            "topk_jaccard@20",
            "topk_order_changed",
            "selected_sentence_jaccard",
            "rendered_sentence_jaccard",
            "selected_chunk_jaccard",
            "rendered_chunk_jaccard",
            "prediction_changed",
        ]:
            agg[k].append(float(qrow.get(k, 0.0)))

    candidate_missing_in_baseline = int(sum(r.get("num_missing_in_baseline", 0) for r in candidate_set_diff_rows))
    candidate_missing_in_optimized = int(sum(r.get("num_missing_in_optimized", 0) for r in candidate_set_diff_rows))

    missing_default_by_component = Counter([_safe_text(r.get("component")) or "unknown" for r in missing_default_rows])

    source_priority_audit = {}
    for c in SOURCE_COMPONENTS:
        d = source_same_ctr[c]
        diff = int(d.get("diff", 0))
        same = int(d.get("same", 0))
        source_priority_audit[c] = {
            "same": bool(diff == 0),
            "mismatch_count": int(diff),
            "total_compared": int(same + diff),
        }

    # output impact
    missing_candidate_output_impact_count = int(sum(int(r.get("output_impact", 0)) for r in candidate_diff_examples))
    score_diff_output_impact_count = int(
        sum(
            1
            for r in score_diff_examples
            if int(r.get("in_topk@8", 0))
            or int(r.get("selected_by_baseline", 0))
            or int(r.get("selected_by_optimized", 0))
            or int(r.get("rendered_by_baseline", 0))
            or int(r.get("rendered_by_optimized", 0))
        )
    )

    retrieval_drift_prediction_changed_rate = (
        float(class_counts.get("retrieval_drift", 0)) / float(len(sample_ids)) if sample_ids else 0.0
    )

    component_score_parity = {
        "final_score_max_abs_diff": max(agg.get("final_score_max_abs_diff", [0.0])) if agg.get("final_score_max_abs_diff") else 0.0,
        "final_score_mean_abs_diff": mean(agg.get("final_score_mean_abs_diff", [0.0])) if agg.get("final_score_mean_abs_diff") else 0.0,
        "semantic_score_max_abs_diff": max(agg.get("semantic_score_max_abs_diff", [0.0])) if agg.get("semantic_score_max_abs_diff") else 0.0,
        "graph_score_max_abs_diff": max(agg.get("graph_score_max_abs_diff", [0.0])) if agg.get("graph_score_max_abs_diff") else 0.0,
        "bridge_score_max_abs_diff": max(agg.get("bridge_score_max_abs_diff", [0.0])) if agg.get("bridge_score_max_abs_diff") else 0.0,
        "corridor_score_max_abs_diff": max(agg.get("corridor_score_max_abs_diff", [0.0])) if agg.get("corridor_score_max_abs_diff") else 0.0,
        "redundancy_score_max_abs_diff": max(agg.get("redundancy_score_max_abs_diff", [0.0])) if agg.get("redundancy_score_max_abs_diff") else 0.0,
        "answerability_score_max_abs_diff": max(agg.get("answerability_score_max_abs_diff", [0.0])) if agg.get("answerability_score_max_abs_diff") else 0.0,
        "topk_jaccard@8": mean(agg.get("topk_jaccard@8", [0.0])) if agg.get("topk_jaccard@8") else 0.0,
        "topk_jaccard@20": mean(agg.get("topk_jaccard@20", [0.0])) if agg.get("topk_jaccard@20") else 0.0,
        "topk_order_changed_rate": mean(agg.get("topk_order_changed", [0.0])) if agg.get("topk_order_changed") else 0.0,
        "selected_sentence_jaccard": mean(agg.get("selected_sentence_jaccard", [0.0])) if agg.get("selected_sentence_jaccard") else 0.0,
        "rendered_sentence_jaccard": mean(agg.get("rendered_sentence_jaccard", [0.0])) if agg.get("rendered_sentence_jaccard") else 0.0,
        "selected_chunk_jaccard": mean(agg.get("selected_chunk_jaccard", [0.0])) if agg.get("selected_chunk_jaccard") else 0.0,
        "rendered_chunk_jaccard": mean(agg.get("rendered_chunk_jaccard", [0.0])) if agg.get("rendered_chunk_jaccard") else 0.0,
        "prediction_changed_rate": mean(agg.get("prediction_changed", [0.0])) if agg.get("prediction_changed") else 0.0,
        "retrieval_drift_prediction_changed_rate": float(retrieval_drift_prediction_changed_rate),
    }

    candidate_set_parity = {
        "total_queries": int(len(sample_ids)),
        "avg_num_baseline_candidates": mean(agg.get("num_baseline_candidates", [0.0])) if agg.get("num_baseline_candidates") else 0.0,
        "avg_num_optimized_candidates": mean(agg.get("num_optimized_candidates", [0.0])) if agg.get("num_optimized_candidates") else 0.0,
        "avg_num_common_candidates": mean(agg.get("num_common_candidates", [0.0])) if agg.get("num_common_candidates") else 0.0,
        "candidate_missing_in_baseline": int(candidate_missing_in_baseline),
        "candidate_missing_in_optimized": int(candidate_missing_in_optimized),
    }

    missing_score_default_audit = {
        "missing_score_default_diff_count": int(len(missing_default_rows)),
        "missing_score_default_diff_by_component": {str(k): int(v) for k, v in sorted(missing_default_by_component.items(), key=lambda x: (-x[1], x[0]))},
    }

    candidate_generation_parity = {
        "candidate_missing_in_optimized_by_source_stage": {str(k): int(v) for k, v in sorted(missing_in_opt_stage.items(), key=lambda x: (-x[1], x[0]))},
        "candidate_missing_in_baseline_by_source_stage": {str(k): int(v) for k, v in sorted(missing_in_base_stage.items(), key=lambda x: (-x[1], x[0]))},
        "path_candidate_missing_count": int(missing_in_opt_stage.get("path", 0) + missing_in_base_stage.get("path", 0)),
        "bridge_candidate_missing_count": int(missing_in_opt_stage.get("bridge", 0) + missing_in_base_stage.get("bridge", 0)),
        "semantic_candidate_missing_count": int(missing_in_opt_stage.get("semantic", 0) + missing_in_base_stage.get("semantic", 0)),
        "graph_candidate_missing_count": int(missing_in_opt_stage.get("graph", 0) + missing_in_base_stage.get("graph", 0)),
    }

    latency_retention = {}
    for k in [
        "retrieval_ms",
        "proposal_total_ms_avg",
        "proposal_union_total_ms_avg",
        "path_candidate_expansion_ms_avg",
        "total_ms",
        "f1",
        "em",
        "supporting_fact_recall",
        "supporting_fact_precision",
        "answer_bearing_chunk_present",
        "answer_present_but_generation_fail",
        "avg_prompt_tokens",
    ]:
        b = _safe_float(b_summary.get(k), 0.0)
        o = _safe_float(o_summary.get(k), 0.0)
        delta = o - b
        delta_pct = (delta / b * 100.0) if abs(b) > 1.0e-12 else 0.0
        latency_retention[k] = {
            "baseline": float(b),
            "optimized": float(o),
            "delta": float(delta),
            "delta_pct": float(delta_pct),
        }

    top_final = sorted([r for r in score_diff_examples if _safe_text(r.get("component")) == "final_score"], key=lambda r: float(r.get("abs_diff", 0.0)), reverse=True)[:20]
    top_sem = sorted([r for r in score_diff_examples if _safe_text(r.get("component")) == "semantic_score"], key=lambda r: float(r.get("abs_diff", 0.0)), reverse=True)[:20]
    top_graph = sorted([r for r in score_diff_examples if _safe_text(r.get("component")) == "graph_score"], key=lambda r: float(r.get("abs_diff", 0.0)), reverse=True)[:20]

    output_impact_assessment = {
        "missing_candidate_output_impact_count": int(missing_candidate_output_impact_count),
        "score_diff_output_impact_count": int(score_diff_output_impact_count),
        "tail_only_candidate_drift": bool(
            (candidate_missing_in_baseline > 0 or candidate_missing_in_optimized > 0)
            and missing_candidate_output_impact_count == 0
        ),
    }

    # Decision
    retrieval_reduction_pct = -float(latency_retention.get("retrieval_ms", {}).get("delta_pct", 0.0))
    sf_recall_delta = abs(float(latency_retention.get("supporting_fact_recall", {}).get("delta", 0.0)))

    source_all_same = all(bool(dict(source_priority_audit.get(c, {})).get("same", False)) for c in SOURCE_COMPONENTS)
    strict_core_pass = (
        int(missing_score_default_audit["missing_score_default_diff_count"]) == 0
        and source_all_same
        and float(component_score_parity["topk_jaccard@8"]) >= 0.999999
        and float(component_score_parity["selected_sentence_jaccard"]) >= 0.98
        and float(component_score_parity["rendered_sentence_jaccard"]) >= 0.98
        and float(component_score_parity["retrieval_drift_prediction_changed_rate"]) <= 0.05
        and float(sf_recall_delta) <= 1.0e-12
        and float(retrieval_reduction_pct) >= 60.0
    )

    if strict_core_pass:
        decision = "accept"
    else:
        tail_warning_ok = (
            int(missing_score_default_audit["missing_score_default_diff_count"]) == 0
            and source_all_same
            and bool(output_impact_assessment["tail_only_candidate_drift"])
            and int(output_impact_assessment["score_diff_output_impact_count"]) == 0
            and float(retrieval_reduction_pct) >= 60.0
            and float(sf_recall_delta) <= 0.02
        )
        if tail_warning_ok:
            decision = "accept_with_tail_warning"
        else:
            # reject only when latency benefit lost or recall drops heavily.
            if float(retrieval_reduction_pct) < 20.0 or float(sf_recall_delta) > 0.05:
                decision = "reject"
            else:
                decision = "hold"

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "hotpotqa",
        "profile": "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
        "candidate_set_parity": candidate_set_parity,
        "component_score_parity": component_score_parity,
        "missing_score_default_audit": missing_score_default_audit,
        "score_source_priority_audit": source_priority_audit,
        "candidate_generation_parity": candidate_generation_parity,
        "score_diff_outliers": {
            "top_final_score_diff": top_final,
            "top_semantic_score_diff": top_sem,
            "top_graph_score_diff": top_graph,
        },
        "prediction_change_classification": {str(k): int(v) for k, v in sorted(class_counts.items(), key=lambda x: x[0])},
        "latency_retention": latency_retention,
        "output_impact_assessment": output_impact_assessment,
        "reason_counts": {str(k): int(v) for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))},
        "patch_applied": "Preserved baseline shortest-path semantics with cached cutoff reachability and deterministic candidate-loop iteration/tie handling in optimized path.",
        "decision": decision,
    }

    # Write JSONL diagnostics
    _write_jsonl(score_root / "candidate_set_diff.jsonl", candidate_set_diff_rows)
    _write_jsonl(score_root / "component_score_diff.jsonl", component_score_diff_rows)
    _write_jsonl(score_root / "candidate_score_diff.jsonl", per_query_rows)
    _write_jsonl(score_root / "missing_score_default_diff.jsonl", missing_default_rows)
    _write_jsonl(out_root / "score_attachment_parity_per_query.jsonl", per_query_rows)

    # Secondary outputs requested by user
    _write_json(score_root / "top_score_diff_examples.json", {"examples": score_diff_examples[:80]})
    _write_json(score_root / "top_missing_score_examples.json", {"examples": missing_default_rows[:80]})
    _write_json(out_root / "changed_query_examples.json", {"examples": changed_query_rows[:10]})

    candidate_diff_examples_sorted = sorted(candidate_diff_examples, key=lambda r: (int(r.get("output_impact", 0)), _safe_text(r.get("source_stage"))), reverse=True)
    score_diff_examples_sorted = sorted(score_diff_examples, key=lambda r: float(r.get("abs_diff", 0.0)), reverse=True)

    (out_root / "candidate_diff_examples.md").write_text(_build_candidate_diff_md(candidate_diff_examples_sorted), encoding="utf-8")
    (out_root / "score_diff_examples.md").write_text(_build_score_diff_md(score_diff_examples_sorted), encoding="utf-8")
    (out_root / "prediction_change_classification.md").write_text(
        _build_prediction_classification_md(changed_query_rows, class_counts, len(sample_ids)),
        encoding="utf-8",
    )

    # Keep legacy path for compatibility
    (score_root / "top_score_diff_examples.md").write_text(_build_score_diff_md(score_diff_examples_sorted), encoding="utf-8")
    (out_root / "top_score_diff_examples.md").write_text(_build_score_diff_md(score_diff_examples_sorted), encoding="utf-8")
    (out_root / "top_missing_score_examples.md").write_text(_build_score_diff_md([{**r, "component": r.get("component", "missing_default")} for r in missing_default_rows[:30]]), encoding="utf-8")

    _write_json(out_root / "phase6t_score_attachment_parity_repair_summary.json", summary)
    _write_json(out_root / "candidate_score_parity_summary.json", summary)
    _write_json(out_root / "phase6t_candidate_score_parity_summary.json", summary)

    summary_md = _build_summary_md(summary)
    (out_root / "PHASE6T_SCORE_ATTACHMENT_PARITY_REPAIR_SUMMARY.md").write_text(summary_md + "\n", encoding="utf-8")
    (out_root / "PHASE6T_CANDIDATE_SCORE_PARITY_SUMMARY.md").write_text(summary_md + "\n", encoding="utf-8")

    print(str(out_root / "PHASE6T_SCORE_ATTACHMENT_PARITY_REPAIR_SUMMARY.md"))


if __name__ == "__main__":
    main()
