#!/usr/bin/env python3
"""Summarize PHASE6X early-collapse recovery paired run."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts

from effirag.evidence_flow_audit import load_samples_for_dataset, stage_metrics
from summarize_phase6u_stagewise_evidence_audit import _build_stage_payloads


STAGE_KEYS = [
    ("S1_proposal", "proposal_candidates"),
    ("S2_phase1", "phase1_run_seed_shortlist"),
    ("S3_local_refinement", "local_corridor_candidates"),
    ("S4_sentence", "sentence_candidates_after_text_rerank"),
    ("S5_selected", "abr_selected_evidence"),
    ("S6_rendered", "rendered_compact_context"),
]


def _safe_text(v: Any) -> str:
    return str(v or "").strip()


def _safe_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _safe_int(v: Any, d: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(d)


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(float(x) for x in vals) / float(len(vals)))


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _truncate(text: Any, max_len: int = 220) -> str:
    s = _safe_text(text)
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 3)] + "..."


def _ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in values:
        s = _safe_text(x)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(_ordered_unique(a))
    sb = set(_ordered_unique(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa.intersection(sb)) / float(len(sa.union(sb)))
    )


def _normalize_for_match(text: str) -> str:
    return " ".join(str(text or "").strip().split()).lower()


def _normalize_sentence_support_id(sentence_id: str) -> str:
    sid = _safe_text(sentence_id)
    if not sid:
        return ""
    parts = sid.split("::")
    if len(parts) >= 3 and parts[0] in {"chunk", "sentence"}:
        return f"{parts[1]}::{parts[2]}"
    return sid


def _answer_aliases(sample: Any) -> List[str]:
    aliases: List[str] = []
    answer = _safe_text(getattr(sample, "answer", ""))
    if answer:
        aliases.append(answer)
    for key in ("answer_aliases", "aliases", "normalized_aliases"):
        for value in list(getattr(sample, key, []) or []):
            v = _safe_text(value)
            if v:
                aliases.append(v)
    seen = set()
    out: List[str] = []
    for alias in aliases:
        norm = _normalize_for_match(alias)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(alias)
    return out


def _answer_metrics(sample: Any, texts: Sequence[str]) -> Dict[str, float]:
    aliases = _answer_aliases(sample)
    norm_aliases = [_normalize_for_match(x) for x in aliases if _normalize_for_match(x)]
    sent_norm = [_normalize_for_match(x) for x in list(texts or [])]
    hit_indices: List[int] = []
    if norm_aliases:
        for i, s in enumerate(sent_norm):
            if any(alias in s for alias in norm_aliases):
                hit_indices.append(i)
    hit = 1.0 if hit_indices else 0.0
    rank = float(hit_indices[0]) if hit_indices else -1.0
    count = float(len(hit_indices))
    density = float(count / float(max(1, len(sent_norm)))) if sent_norm else 0.0
    return {
        "answer_hit": hit,
        "answer_rank": rank,
        "answer_count": count,
        "answer_density": density,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, Mapping):
            rows.append(dict(obj))
    return rows


def _sample_id_mismatch_count(
    baseline_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
) -> int:
    b_ids = {_safe_text(r.get("sample_id")) for r in list(baseline_rows or []) if _safe_text(r.get("sample_id"))}
    v_ids = {_safe_text(r.get("sample_id")) for r in list(variant_rows or []) if _safe_text(r.get("sample_id"))}
    return int(len(b_ids.symmetric_difference(v_ids)))


@dataclass
class RunData:
    run_name: str
    dataset: str
    profile: str
    role: str
    summary: Dict[str, Any]
    query_rows: List[Dict[str, Any]]


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
    f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
    recall5 = _safe_float(summary.get("recall_at_5", summary.get("Recall@5", 0.0)), 0.0)
    sf_p = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
    sf_r = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
    sf_f1 = (2.0 * sf_p * sf_r / (sf_p + sf_r)) if (sf_p + sf_r) > 0 else 0.0
    ctx = _safe_float(summary.get("avg_context_tokens", summary.get("prompt_tokens_avg", 0.0)), 0.0)
    f1k = (f1 * 1000.0 / ctx) if ctx > 0 else 0.0
    return {
        "Recall@5": recall5,
        "EM": em,
        "F1": f1,
        "avg_context_tokens": ctx,
        "F1_per_1k_context_tokens": f1k,
        "supporting_fact_precision": sf_p,
        "supporting_fact_recall": sf_r,
        "supporting_fact_f1": sf_f1,
        "answer_string_hit": _safe_float(summary.get("answer_surface_present", 0.0), 0.0),
        "answer_bearing_density": _safe_float(summary.get("answer_surface_token_density", 0.0), 0.0),
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
    }


def _query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    m = dict(row.get("metrics", {}) or {})
    e = dict(row.get("efficiency", {}) or {})
    g = dict(row.get("generation_diagnostics", {}) or {})
    return {
        "f1": _safe_float(m.get("f1", 0.0), 0.0),
        "em": _safe_float(m.get("em", 0.0), 0.0),
        "sf_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
        "sf_precision": _safe_float(m.get("supporting_fact_precision", 0.0), 0.0),
        "retrieval_ms": _safe_float(e.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(e.get("generation_latency_ms", 0.0), 0.0),
        "total_ms": _safe_float(e.get("total_latency_ms", 0.0), 0.0),
        "prompt_tokens": _safe_float(g.get("prompt_tokens", 0.0), 0.0),
        "context_tokens": _safe_float(g.get("prompt_tokens", 0.0), 0.0),
        "answer_string_hit": _safe_float(m.get("answer_surface_present", 0.0), 0.0),
        "answer_bearing_density": _safe_float(m.get("answer_surface_token_density", 0.0), 0.0),
    }


def _extract_selected_ids(row: Mapping[str, Any]) -> List[str]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    ids = list(row.get("retrieval_selected_sentence_ids", []) or retrieval.get("selected_sentence_ids", []) or [])
    return [_normalize_sentence_support_id(x) for x in _ordered_unique(ids)]


def _extract_rendered_ids(row: Mapping[str, Any]) -> List[str]:
    rendered = dict((row.get("rendered", {}) or {}))
    ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
    return [_normalize_sentence_support_id(x) for x in _ordered_unique(ids)]


def _extract_rendered_excerpt(row: Mapping[str, Any], max_len: int = 260) -> str:
    rendered = dict((row.get("rendered", {}) or {}))
    text = _safe_text(rendered.get("text", ""))
    if not text:
        sents = list(rendered.get("sentences", []) or [])
        text = " ".join(_safe_text(x) for x in sents if _safe_text(x))
    return _truncate(text, max_len=max_len)


def _extract_reserve_candidate_text(row: Mapping[str, Any], max_len: int = 260) -> str:
    diag = dict(((row.get("retrieval") or {}).get("diagnostics") or {}))
    texts = list(diag.get("phase1_reserve_unit_texts", []) or [])
    if texts:
        return _truncate(texts[0], max_len=max_len)
    return ""


def _load_runs(out_root: Path, datasets: Sequence[str], profiles: Sequence[str]) -> Tuple[Dict[str, Any], List[RunData]]:
    report = collect_phase6s_artifacts(out_root)
    scheduled = {_safe_text(r.get("run_name")): dict(r) for r in list(report.get("scheduled_runs", []) or [])}
    dataset_set = set(datasets)
    profile_set = set(profiles)
    runs: List[RunData] = []
    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        meta = scheduled.get(run_name, {})
        dataset = _safe_text(meta.get("dataset", item.get("dataset", "")))
        profile = _safe_text(meta.get("profile", item.get("profile", "")))
        role = _safe_text(meta.get("role", ""))
        if dataset not in dataset_set or profile not in profile_set:
            continue
        qpath = Path(_safe_text(item.get("query_path"))).resolve()
        qrows = _read_jsonl(qpath)
        runs.append(
            RunData(
                run_name=run_name,
                dataset=dataset,
                profile=profile,
                role=role,
                summary=dict(item.get("summary", {}) or {}),
                query_rows=qrows,
            )
        )
    return report, runs


def _stage_aggregate(run: RunData) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    sample_map = load_samples_for_dataset(run.dataset)
    stage_acc: Dict[str, Dict[str, List[float]]] = {}
    query_rows: List[Dict[str, Any]] = []
    for short_name, _ in STAGE_KEYS:
        stage_acc[short_name] = {
            "sf_recall": [],
            "sf_precision": [],
            "sf_f1": [],
            "answer_hit": [],
            "answer_density": [],
            "candidate_count": [],
        }

    for row in run.query_rows:
        sid = _safe_text(row.get("sample_id"))
        sample = sample_map.get(sid)
        if sample is None:
            continue
        graph_mode = _safe_text(
            ((row.get("retrieval") or {}).get("diagnostics") or {}).get("graph_mode", "current_entity_graph")
        )
        payloads = _build_stage_payloads(row)

        query_entry: Dict[str, Any] = {
            "dataset": run.dataset,
            "profile": run.profile,
            "sample_id": sid,
            "question": _safe_text(row.get("question", "")),
        }
        for short_name, stage_key in STAGE_KEYS:
            payload = dict(payloads.get(stage_key, {}) or {})
            ids = list(payload.get("ids", []) or [])
            texts = list(payload.get("texts", []) or [])
            metric = stage_metrics(sample, ids, texts, graph_mode)
            sf_r = _safe_float(metric.get("sf_R", 0.0), 0.0)
            sf_p = _safe_float(metric.get("sf_P", 0.0), 0.0)
            sf_f = _safe_float(metric.get("sf_F1", 0.0), 0.0)
            ans = _answer_metrics(sample, texts)
            stage_acc[short_name]["sf_recall"].append(sf_r)
            stage_acc[short_name]["sf_precision"].append(sf_p)
            stage_acc[short_name]["sf_f1"].append(sf_f)
            stage_acc[short_name]["answer_hit"].append(_safe_float(ans.get("answer_hit", 0.0), 0.0))
            stage_acc[short_name]["answer_density"].append(_safe_float(ans.get("answer_density", 0.0), 0.0))
            stage_acc[short_name]["candidate_count"].append(float(len(ids)))
            query_entry[f"{short_name}_SF_recall"] = sf_r
            query_entry[f"{short_name}_answer_hit"] = _safe_float(ans.get("answer_hit", 0.0), 0.0)
        query_rows.append(query_entry)

    stage_out: Dict[str, Dict[str, float]] = {}
    for short_name, vals in stage_acc.items():
        stage_out[short_name] = {
            "SF_recall": _mean(vals["sf_recall"]),
            "SF_precision": _mean(vals["sf_precision"]),
            "SF_f1": _mean(vals["sf_f1"]),
            "answer_hit": _mean(vals["answer_hit"]),
            "answer_density": _mean(vals["answer_density"]),
            "candidate_count": _mean(vals["candidate_count"]),
        }
    return stage_out, query_rows


def _transfer_deltas(stage_out: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    def s(name: str, key: str) -> float:
        return _safe_float(dict(stage_out.get(name, {}) or {}).get(key, 0.0), 0.0)

    return {
        "S1_to_S2_SF_delta": s("S2_phase1", "SF_recall") - s("S1_proposal", "SF_recall"),
        "S2_to_S3_SF_delta": s("S3_local_refinement", "SF_recall") - s("S2_phase1", "SF_recall"),
        "S3_to_S4_SF_delta": s("S4_sentence", "SF_recall") - s("S3_local_refinement", "SF_recall"),
        "S4_to_S5_SF_delta": s("S5_selected", "SF_recall") - s("S4_sentence", "SF_recall"),
        "S1_to_S2_answer_hit_delta": s("S2_phase1", "answer_hit") - s("S1_proposal", "answer_hit"),
        "S2_to_S3_answer_hit_delta": s("S3_local_refinement", "answer_hit") - s("S2_phase1", "answer_hit"),
        "S3_to_S4_answer_hit_delta": s("S4_sentence", "answer_hit") - s("S3_local_refinement", "answer_hit"),
        "S4_to_S5_answer_hit_delta": s("S5_selected", "answer_hit") - s("S4_sentence", "answer_hit"),
    }


def _build_paired_rows(
    dataset: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    bmap = {_safe_text(r.get("sample_id")): r for r in baseline_rows}
    vmap = {_safe_text(r.get("sample_id")): r for r in variant_rows}
    ordered = [_safe_text(r.get("sample_id")) for r in baseline_rows if _safe_text(r.get("sample_id"))]
    paired_rows: List[Dict[str, Any]] = []
    reserve_rows: List[Dict[str, Any]] = []

    for sid in ordered:
        b = bmap.get(sid)
        v = vmap.get(sid)
        if b is None or v is None:
            continue
        bm = _query_metrics(b)
        vm = _query_metrics(v)
        b_sel = _extract_selected_ids(b)
        v_sel = _extract_selected_ids(v)
        b_rnd = _extract_rendered_ids(b)
        v_rnd = _extract_rendered_ids(v)
        b_rnd_excerpt = _extract_rendered_excerpt(b)
        v_rnd_excerpt = _extract_rendered_excerpt(v)
        vdiag = dict(((v.get("retrieval") or {}).get("diagnostics") or {}))
        reserve_ids = [_normalize_sentence_support_id(x) for x in list(vdiag.get("phase1_reserve_unit_ids", []) or [])]
        reserve_text = _extract_reserve_candidate_text(v)
        rendered_hit = bool(set(reserve_ids).intersection(set(v_rnd))) if reserve_ids else False

        d_f1 = float(vm["f1"] - bm["f1"])
        d_sf_r = float(vm["sf_recall"] - bm["sf_recall"])
        if d_f1 > 0.05 or d_sf_r > 0.10:
            cls = "improved"
        elif d_f1 < -0.05 or d_sf_r < -0.10:
            cls = "regressed"
        else:
            cls = "neutral"

        paired_rows.append(
            {
                "dataset": dataset,
                "sample_id": sid,
                "question": _safe_text(v.get("question", b.get("question", ""))),
                "gold_answer": _safe_text(v.get("answer", b.get("answer", ""))),
                "baseline_prediction": _safe_text(b.get("prediction", "")),
                "variant_prediction": _safe_text(v.get("prediction", "")),
                "baseline_em": bm["em"],
                "variant_em": vm["em"],
                "baseline_f1": bm["f1"],
                "variant_f1": vm["f1"],
                "delta_f1": d_f1,
                "baseline_sf_recall": bm["sf_recall"],
                "variant_sf_recall": vm["sf_recall"],
                "delta_sf_recall": d_sf_r,
                "baseline_sf_precision": bm["sf_precision"],
                "variant_sf_precision": vm["sf_precision"],
                "baseline_answer_string_hit": bm["answer_string_hit"],
                "variant_answer_string_hit": vm["answer_string_hit"],
                "baseline_answer_bearing_density": bm["answer_bearing_density"],
                "variant_answer_bearing_density": vm["answer_bearing_density"],
                "baseline_context_tokens": bm["context_tokens"],
                "variant_context_tokens": vm["context_tokens"],
                "baseline_retrieval_ms": bm["retrieval_ms"],
                "variant_retrieval_ms": vm["retrieval_ms"],
                "baseline_total_ms": bm["total_ms"],
                "variant_total_ms": vm["total_ms"],
                "baseline_selected_sentence_ids": b_sel,
                "variant_selected_sentence_ids": v_sel,
                "baseline_rendered_sentence_ids": b_rnd,
                "variant_rendered_sentence_ids": v_rnd,
                "baseline_rendered_context_excerpt": b_rnd_excerpt,
                "variant_rendered_context_excerpt": v_rnd_excerpt,
                "selected_sentence_jaccard": _jaccard(b_sel, v_sel),
                "rendered_sentence_jaccard": _jaccard(b_rnd, v_rnd),
                "reserve_triggered": bool(vdiag.get("phase1_reserve_added", False)),
                "reserve_reason": _safe_text(vdiag.get("phase1_reserve_reason", "")),
                "reserve_ABR_rank": vdiag.get("phase1_reserve_ABR_rank", None),
                "reserve_candidate_text": reserve_text,
                "classification": cls,
            }
        )

        reserve_rows.append(
            {
                "dataset": dataset,
                "profile": "unified_acr_rcedr_v12_early_collapse_recovery",
                "sample_id": sid,
                "num_phase1_shortlist_before": _safe_int(vdiag.get("num_phase1_shortlist_before", 0), 0),
                "num_phase1_shortlist_after": _safe_int(vdiag.get("num_phase1_shortlist_after", 0), 0),
                "phase1_reserve_added": bool(vdiag.get("phase1_reserve_added", False)),
                "phase1_reserve_triggered": bool(vdiag.get("phase1_reserve_added", False)),
                "reserve_candidate_id": _safe_text(vdiag.get("phase1_reserve_candidate_id", (reserve_ids[0] if reserve_ids else ""))),
                "reserve_candidate_type": _safe_text(vdiag.get("phase1_reserve_candidate_type", "")) or "unknown",
                "reserve_source_run_id": _safe_int(vdiag.get("phase1_reserve_seed_or_run_id", -1), -1),
                "reserve_source_seed_id": _safe_text(vdiag.get("phase1_reserve_source_seed_id", "")),
                "reserve_reason": _safe_text(vdiag.get("phase1_reserve_reason", "")),
                "reserve_score": _safe_float(vdiag.get("phase1_reserve_score", 0.0), 0.0),
                "reserve_rank_before_reserve": _safe_int(vdiag.get("phase1_reserve_rank_before_reserve", -1), -1),
                "reserve_rank_after_reserve": _safe_int(vdiag.get("phase1_reserve_rank_after_reserve", -1), -1),
                "reserve_in_phase1_shortlist": bool(vdiag.get("phase1_reserve_in_phase1_shortlist", vdiag.get("phase1_reserve_added", False))),
                "reserve_reaches_phase2": bool(vdiag.get("phase1_reserve_reaches_phase2", False)),
                "reserve_has_sentence_candidates": bool(vdiag.get("phase1_reserve_has_sentence_candidates", False)),
                "reserve_enters_ABR_candidate_pool": bool(
                    vdiag.get(
                        "phase1_reserve_enters_ABR_candidate_pool",
                        vdiag.get("phase1_reserve_has_sentence_candidates", False),
                    )
                ),
                "reserve_ABR_score": vdiag.get("phase1_reserve_ABR_score", None),
                "reserve_ABR_rank": vdiag.get("phase1_reserve_ABR_rank", None),
                "reserve_selected_by_ABR": bool(vdiag.get("phase1_reserve_selected_by_ABR", False)),
                "reserve_rendered": bool(vdiag.get("phase1_reserve_rendered", rendered_hit) or rendered_hit),
                "reserve_forced_selected": bool(vdiag.get("phase1_reserve_forced_selected", False)),
                "reserve_forced_rendered": bool(vdiag.get("phase1_reserve_forced_rendered", False)),
                "selected_because_of_forced_pin": bool(vdiag.get("phase1_reserve_selected_because_of_forced_pin", False)),
                "phase1_reserve_seed_or_run_id": _safe_int(vdiag.get("phase1_reserve_seed_or_run_id", -1), -1),
                "phase1_reserve_reason": _safe_text(vdiag.get("phase1_reserve_reason", "")),
                "phase1_reserve_score": _safe_float(vdiag.get("phase1_reserve_score", 0.0), 0.0),
                "phase1_reserve_reaches_phase2": bool(vdiag.get("phase1_reserve_reaches_phase2", False)),
                "phase1_reserve_has_sentence_candidates": bool(vdiag.get("phase1_reserve_has_sentence_candidates", False)),
                "phase1_reserve_selected_by_ABR": bool(vdiag.get("phase1_reserve_selected_by_ABR", False)),
                "phase1_reserve_rendered": bool(vdiag.get("phase1_reserve_rendered", rendered_hit) or rendered_hit),
            }
        )
    return paired_rows, reserve_rows


def _markdown_examples(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        return "\n".join(lines) + "\n"
    for i, row in enumerate(rows, start=1):
        lines.append(f"## {i}. {_safe_text(row.get('dataset', ''))} / {_safe_text(row.get('sample_id', ''))}")
        lines.append(f"- question: {_safe_text(row.get('question', ''))}")
        lines.append(f"- gold_answer: {_safe_text(row.get('gold_answer', ''))}")
        lines.append(f"- baseline_prediction: {_safe_text(row.get('baseline_prediction', ''))}")
        lines.append(f"- variant_prediction: {_safe_text(row.get('variant_prediction', ''))}")
        lines.append(
            "- baseline_f1 / variant_f1: {b:.4f} / {v:.4f} (Δ {d:+.4f})".format(
                b=_safe_float(row.get("baseline_f1", 0.0), 0.0),
                v=_safe_float(row.get("variant_f1", 0.0), 0.0),
                d=_safe_float(row.get("delta_f1", 0.0), 0.0),
            )
        )
        lines.append(
            "- baseline_answer_string_hit / variant_answer_string_hit: {b:.4f} / {v:.4f}".format(
                b=_safe_float(row.get("baseline_answer_string_hit", 0.0), 0.0),
                v=_safe_float(row.get("variant_answer_string_hit", 0.0), 0.0),
            )
        )
        lines.append(f"- reserve_triggered: {bool(row.get('reserve_triggered', False))}")
        lines.append(f"- reserve_reason: {_safe_text(row.get('reserve_reason', ''))}")
        lines.append(f"- reserve_ABR_rank: {_safe_text(row.get('reserve_ABR_rank', ''))}")
        lines.append(f"- reserve_candidate_text: {_safe_text(row.get('reserve_candidate_text', ''))}")
        lines.append(f"- baseline_rendered_context_excerpt: {_safe_text(row.get('baseline_rendered_context_excerpt', ''))}")
        lines.append(f"- variant_rendered_context_excerpt: {_safe_text(row.get('variant_rendered_context_excerpt', ''))}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_trigger_subset_rows(paired_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_dataset: Dict[str, List[Mapping[str, Any]]] = {}
    for row in paired_rows:
        ds = _safe_text(row.get("dataset", ""))
        if not ds:
            continue
        by_dataset.setdefault(ds, []).append(row)

    def agg(rows: Sequence[Mapping[str, Any]], subset_name: str, dataset: str) -> Dict[str, Any]:
        vals = list(rows or [])
        return {
            "dataset": dataset,
            "subset": subset_name,
            "query_count": len(vals),
            "baseline_EM": _mean([_safe_float(r.get("baseline_em", 0.0), 0.0) for r in vals]),
            "variant_EM": _mean([_safe_float(r.get("variant_em", 0.0), 0.0) for r in vals]),
            "delta_EM": _mean([_safe_float(r.get("variant_em", 0.0), 0.0) - _safe_float(r.get("baseline_em", 0.0), 0.0) for r in vals]),
            "baseline_F1": _mean([_safe_float(r.get("baseline_f1", 0.0), 0.0) for r in vals]),
            "variant_F1": _mean([_safe_float(r.get("variant_f1", 0.0), 0.0) for r in vals]),
            "delta_F1": _mean([_safe_float(r.get("delta_f1", 0.0), 0.0) for r in vals]),
            "baseline_SF_recall": _mean([_safe_float(r.get("baseline_sf_recall", 0.0), 0.0) for r in vals]),
            "variant_SF_recall": _mean([_safe_float(r.get("variant_sf_recall", 0.0), 0.0) for r in vals]),
            "delta_SF_recall": _mean([_safe_float(r.get("delta_sf_recall", 0.0), 0.0) for r in vals]),
            "baseline_answer_string_hit": _mean([_safe_float(r.get("baseline_answer_string_hit", 0.0), 0.0) for r in vals]),
            "variant_answer_string_hit": _mean([_safe_float(r.get("variant_answer_string_hit", 0.0), 0.0) for r in vals]),
            "delta_answer_string_hit": _mean([_safe_float(r.get("variant_answer_string_hit", 0.0), 0.0) - _safe_float(r.get("baseline_answer_string_hit", 0.0), 0.0) for r in vals]),
            "baseline_answer_bearing_density": _mean([_safe_float(r.get("baseline_answer_bearing_density", 0.0), 0.0) for r in vals]),
            "variant_answer_bearing_density": _mean([_safe_float(r.get("variant_answer_bearing_density", 0.0), 0.0) for r in vals]),
            "delta_answer_bearing_density": _mean([_safe_float(r.get("variant_answer_bearing_density", 0.0), 0.0) - _safe_float(r.get("baseline_answer_bearing_density", 0.0), 0.0) for r in vals]),
            "baseline_avg_context_tokens": _mean([_safe_float(r.get("baseline_context_tokens", 0.0), 0.0) for r in vals]),
            "variant_avg_context_tokens": _mean([_safe_float(r.get("variant_context_tokens", 0.0), 0.0) for r in vals]),
            "delta_avg_context_tokens": _mean([_safe_float(r.get("variant_context_tokens", 0.0), 0.0) - _safe_float(r.get("baseline_context_tokens", 0.0), 0.0) for r in vals]),
            "baseline_retrieval_ms": _mean([_safe_float(r.get("baseline_retrieval_ms", 0.0), 0.0) for r in vals]),
            "variant_retrieval_ms": _mean([_safe_float(r.get("variant_retrieval_ms", 0.0), 0.0) for r in vals]),
            "delta_retrieval_ms": _mean([_safe_float(r.get("variant_retrieval_ms", 0.0), 0.0) - _safe_float(r.get("baseline_retrieval_ms", 0.0), 0.0) for r in vals]),
        }

    out: List[Dict[str, Any]] = []
    for dataset, rows in by_dataset.items():
        triggered = [r for r in rows if bool(r.get("reserve_triggered", False))]
        non_triggered = [r for r in rows if not bool(r.get("reserve_triggered", False))]
        out.append(agg(triggered, "reserve_triggered", dataset))
        out.append(agg(non_triggered, "non_triggered", dataset))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PHASE6X early-collapse recovery run.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--baseline-profile", default="unified_acr_rcedr_v12")
    ap.add_argument("--variant-profile", default="unified_acr_rcedr_v12_early_collapse_recovery")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--summary-title", default="PHASE6X Early-Collapse Recovery Summary")
    ap.add_argument("--summary-md-name", default="PHASE6X_EARLY_COLLAPSE_RECOVERY_SUMMARY.md")
    ap.add_argument("--summary-json-name", default="phase6x_early_collapse_recovery_summary.json")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    profiles = [args.baseline_profile, args.variant_profile]

    report, runs = _load_runs(out_root, datasets=datasets, profiles=profiles)

    by_dataset_profile: Dict[Tuple[str, str], RunData] = {}
    for r in runs:
        by_dataset_profile[(r.dataset, r.profile)] = r

    stage_csv_rows: List[Dict[str, Any]] = []
    paired_rows_all: List[Dict[str, Any]] = []
    reserve_rows_all: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    sample_id_mismatch_count = 0

    for dataset in datasets:
        b = by_dataset_profile.get((dataset, args.baseline_profile))
        v = by_dataset_profile.get((dataset, args.variant_profile))
        if b is None or v is None:
            continue
        sample_id_mismatch_count += _sample_id_mismatch_count(b.query_rows, v.query_rows)

        b_metrics = _summary_metrics(b.summary)
        v_metrics = _summary_metrics(v.summary)
        b_stage, _ = _stage_aggregate(b)
        v_stage, _ = _stage_aggregate(v)
        b_delta = _transfer_deltas(b_stage)
        v_delta = _transfer_deltas(v_stage)

        for prof, stage_out in ((args.baseline_profile, b_stage), (args.variant_profile, v_stage)):
            for short_name, _ in STAGE_KEYS:
                s = dict(stage_out.get(short_name, {}) or {})
                stage_csv_rows.append(
                    {
                        "dataset": dataset,
                        "profile": prof,
                        "stage": short_name,
                        "candidate_count": _safe_float(s.get("candidate_count", 0.0), 0.0),
                        "SF_recall": _safe_float(s.get("SF_recall", 0.0), 0.0),
                        "SF_precision": _safe_float(s.get("SF_precision", 0.0), 0.0),
                        "SF_f1": _safe_float(s.get("SF_f1", 0.0), 0.0),
                        "answer_hit": _safe_float(s.get("answer_hit", 0.0), 0.0),
                        "answer_bearing_density": _safe_float(s.get("answer_density", 0.0), 0.0),
                    }
                )

        paired_rows, reserve_rows = _build_paired_rows(dataset, b.query_rows, v.query_rows)
        paired_rows_all.extend(paired_rows)
        reserve_rows_all.extend(reserve_rows)

        reserve_trigger_rows = [r for r in reserve_rows if bool(r.get("phase1_reserve_added", False))]
        summary_rows.append(
            {
                "dataset": dataset,
                "baseline": b_metrics,
                "variant": v_metrics,
                "delta": {k: float(v_metrics.get(k, 0.0) - b_metrics.get(k, 0.0)) for k in b_metrics.keys()},
                "baseline_stage": b_stage,
                "variant_stage": v_stage,
                "baseline_stage_delta": b_delta,
                "variant_stage_delta": v_delta,
                "reserve_metrics": {
                    "reserve_trigger_rate": _mean([1.0 if bool(r.get("phase1_reserve_added", False)) else 0.0 for r in reserve_rows]),
                    "reserve_phase2_survival_rate": _mean([1.0 if bool(r.get("phase1_reserve_reaches_phase2", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_sentence_candidate_rate": _mean([1.0 if bool(r.get("phase1_reserve_has_sentence_candidates", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_ABR_candidate_rate": _mean([1.0 if bool(r.get("reserve_enters_ABR_candidate_pool", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_ABR_selected_rate": _mean([1.0 if bool(r.get("phase1_reserve_selected_by_ABR", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_rendered_rate": _mean([1.0 if bool(r.get("phase1_reserve_rendered", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_forced_selected_rate": _mean([1.0 if bool(r.get("reserve_forced_selected", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_forced_rendered_rate": _mean([1.0 if bool(r.get("reserve_forced_rendered", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                },
            }
        )

    improved = sorted(
        [r for r in paired_rows_all if _safe_float(r.get("delta_f1", 0.0), 0.0) > 0],
        key=lambda x: _safe_float(x.get("delta_f1", 0.0), 0.0),
        reverse=True,
    )[: max(1, int(args.top_k))]
    regressed = sorted(
        [r for r in paired_rows_all if _safe_float(r.get("delta_f1", 0.0), 0.0) < 0],
        key=lambda x: _safe_float(x.get("delta_f1", 0.0), 0.0),
    )[: max(1, int(args.top_k))]
    triggered_rows_all = [r for r in paired_rows_all if bool(r.get("reserve_triggered", False))]
    reserve_triggered_improved = sorted(
        [r for r in triggered_rows_all if _safe_float(r.get("delta_f1", 0.0), 0.0) > 0],
        key=lambda x: _safe_float(x.get("delta_f1", 0.0), 0.0),
        reverse=True,
    )[: max(1, int(args.top_k))]
    reserve_triggered_regressed = sorted(
        [r for r in triggered_rows_all if _safe_float(r.get("delta_f1", 0.0), 0.0) < 0],
        key=lambda x: _safe_float(x.get("delta_f1", 0.0), 0.0),
    )[: max(1, int(args.top_k))]
    subset_rows = _build_trigger_subset_rows(paired_rows_all)

    _write_jsonl(out_root / "paired_query_comparison.jsonl", paired_rows_all)
    _write_csv(
        out_root / "early_collapse_stagewise_metrics.csv",
        stage_csv_rows,
        [
            "dataset",
            "profile",
            "stage",
            "candidate_count",
            "SF_recall",
            "SF_precision",
            "SF_f1",
            "answer_hit",
            "answer_bearing_density",
        ],
    )
    _write_csv(
        out_root / "reserve_diagnostics.csv",
        reserve_rows_all,
        [
            "dataset",
            "sample_id",
            "num_phase1_shortlist_before",
            "num_phase1_shortlist_after",
            "phase1_reserve_added",
            "phase1_reserve_triggered",
            "reserve_candidate_id",
            "reserve_candidate_type",
            "reserve_source_run_id",
            "reserve_source_seed_id",
            "reserve_reason",
            "reserve_score",
            "reserve_rank_before_reserve",
            "reserve_rank_after_reserve",
            "reserve_in_phase1_shortlist",
            "reserve_reaches_phase2",
            "reserve_has_sentence_candidates",
            "reserve_enters_ABR_candidate_pool",
            "reserve_ABR_score",
            "reserve_ABR_rank",
            "reserve_selected_by_ABR",
            "reserve_rendered",
            "reserve_forced_selected",
            "reserve_forced_rendered",
            "selected_because_of_forced_pin",
            "phase1_reserve_seed_or_run_id",
            "phase1_reserve_reason",
            "phase1_reserve_score",
            "phase1_reserve_reaches_phase2",
            "phase1_reserve_has_sentence_candidates",
            "phase1_reserve_selected_by_ABR",
            "phase1_reserve_rendered",
        ],
    )
    _write_csv(
        out_root / "reserve_forcing_audit.csv",
        reserve_rows_all,
        [
            "dataset",
            "sample_id",
            "phase1_reserve_triggered",
            "reserve_candidate_id",
            "reserve_candidate_type",
            "reserve_source_run_id",
            "reserve_source_seed_id",
            "reserve_reason",
            "reserve_score",
            "reserve_rank_before_reserve",
            "reserve_rank_after_reserve",
            "reserve_in_phase1_shortlist",
            "reserve_reaches_phase2",
            "reserve_has_sentence_candidates",
            "reserve_enters_ABR_candidate_pool",
            "reserve_ABR_score",
            "reserve_ABR_rank",
            "reserve_selected_by_ABR",
            "reserve_rendered",
            "reserve_forced_selected",
            "reserve_forced_rendered",
            "selected_because_of_forced_pin",
        ],
    )
    _write_csv(
        out_root / "reserve_triggered_subset_analysis.csv",
        subset_rows,
        [
            "dataset",
            "subset",
            "query_count",
            "baseline_EM",
            "variant_EM",
            "delta_EM",
            "baseline_F1",
            "variant_F1",
            "delta_F1",
            "baseline_SF_recall",
            "variant_SF_recall",
            "delta_SF_recall",
            "baseline_answer_string_hit",
            "variant_answer_string_hit",
            "delta_answer_string_hit",
            "baseline_answer_bearing_density",
            "variant_answer_bearing_density",
            "delta_answer_bearing_density",
            "baseline_avg_context_tokens",
            "variant_avg_context_tokens",
            "delta_avg_context_tokens",
            "baseline_retrieval_ms",
            "variant_retrieval_ms",
            "delta_retrieval_ms",
        ],
    )

    (out_root / "improved_examples_top20.md").write_text(
        _markdown_examples(improved, "PHASE6X Improved Examples Top20"),
        encoding="utf-8",
    )
    (out_root / "regression_examples_top20.md").write_text(
        _markdown_examples(regressed, "PHASE6X Regression Examples Top20"),
        encoding="utf-8",
    )
    (out_root / "reserve_triggered_improved_examples_top20.md").write_text(
        _markdown_examples(reserve_triggered_improved, "PHASE6X Reserve Triggered Improved Examples Top20"),
        encoding="utf-8",
    )
    (out_root / "reserve_triggered_regression_examples_top20.md").write_text(
        _markdown_examples(reserve_triggered_regressed, "PHASE6X Reserve Triggered Regression Examples Top20"),
        encoding="utf-8",
    )

    decision = "hold"
    if summary_rows:
        avg_delta_f1 = _mean([_safe_float(r.get("delta", {}).get("F1", 0.0), 0.0) for r in summary_rows])
        max_drop = min([_safe_float(r.get("delta", {}).get("F1", 0.0), 0.0) for r in summary_rows])
        avg_token_delta = _mean([_safe_float(r.get("delta", {}).get("avg_context_tokens", 0.0), 0.0) for r in summary_rows])
        avg_ret_delta = _mean([_safe_float(r.get("delta", {}).get("retrieval_ms", 0.0), 0.0) for r in summary_rows])
        max_forced_selected = max(
            [_safe_float((r.get("reserve_metrics", {}) or {}).get("reserve_forced_selected_rate", 0.0), 0.0) for r in summary_rows]
            or [0.0]
        )
        max_forced_rendered = max(
            [_safe_float((r.get("reserve_metrics", {}) or {}).get("reserve_forced_rendered_rate", 0.0), 0.0) for r in summary_rows]
            or [0.0]
        )
        if max_forced_selected > 0.0 or max_forced_rendered > 0.0:
            decision = "invalid_until_explained"
        elif avg_delta_f1 >= 0.01 and max_drop >= -0.02 and avg_token_delta <= 10.0 and avg_ret_delta <= 20.0:
            decision = "strong_accept"
        elif max_drop <= -0.02:
            decision = "reject"
        else:
            decision = "hold"
    else:
        avg_delta_f1 = 0.0
        max_drop = 0.0
        avg_token_delta = 0.0
        avg_ret_delta = 0.0
        max_forced_selected = 0.0
        max_forced_rendered = 0.0

    trig_delta_f1 = _mean(
        [_safe_float(r.get("delta_F1", 0.0), 0.0) for r in subset_rows if _safe_text(r.get("subset", "")) == "reserve_triggered"]
    )
    nontrig_delta_f1 = _mean(
        [_safe_float(r.get("delta_F1", 0.0), 0.0) for r in subset_rows if _safe_text(r.get("subset", "")) == "non_triggered"]
    )
    trig_delta_answer_hit = _mean(
        [_safe_float(r.get("delta_answer_string_hit", 0.0), 0.0) for r in subset_rows if _safe_text(r.get("subset", "")) == "reserve_triggered"]
    )
    nontrig_delta_answer_hit = _mean(
        [_safe_float(r.get("delta_answer_string_hit", 0.0), 0.0) for r in subset_rows if _safe_text(r.get("subset", "")) == "non_triggered"]
    )

    counts = dict(report.get("counts", {}) or {})
    summary_json = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "artifact_validation": {
            "scheduled_runs": int(_safe_int(counts.get("scheduled_runs", 0), 0)),
            "completed_runs": int(_safe_int(counts.get("completed_run_names", len(list(report.get("completed_runs", []) or []))), 0)),
            "incomplete_attempts": int(_safe_int(counts.get("incomplete_attempts", 0), 0)),
            "extra_not_in_manifest": int(_safe_int(counts.get("extra_not_in_manifest", 0), 0)),
            "multi_attempt_run_names": int(_safe_int(counts.get("multi_attempt_run_names", 0), 0)),
            "parse_errors": int(_safe_int(counts.get("parse_errors", 0), 0)),
            "sample_id_mismatch": int(sample_id_mismatch_count),
        },
        "datasets": summary_rows,
        "reserve_triggered_subset_analysis": subset_rows,
        "aggregates": {
            "avg_delta_f1": avg_delta_f1,
            "max_dataset_drop_f1": max_drop,
            "avg_delta_context_tokens": avg_token_delta,
            "avg_delta_retrieval_ms": avg_ret_delta,
            "max_reserve_forced_selected_rate": max_forced_selected,
            "max_reserve_forced_rendered_rate": max_forced_rendered,
            "paired_query_count": int(len(paired_rows_all)),
            "triggered_subset_delta_F1": trig_delta_f1,
            "non_triggered_subset_delta_F1": nontrig_delta_f1,
            "triggered_subset_delta_answer_hit": trig_delta_answer_hit,
            "non_triggered_subset_delta_answer_hit": nontrig_delta_answer_hit,
        },
        "decision": decision,
    }
    summary_json_path = out_root / _safe_text(args.summary_json_name or "phase6x_early_collapse_recovery_summary.json")
    _write_json(summary_json_path, summary_json)
    # Backward-compatible alias for existing tooling.
    if summary_json_path.name != "phase6x_early_collapse_recovery_summary.json":
        _write_json(out_root / "phase6x_early_collapse_recovery_summary.json", summary_json)

    lines: List[str] = []
    lines.append(f"# {_safe_text(args.summary_title)}")
    lines.append("")
    lines.append("## 1. Goal")
    lines.append("Test one dataset-agnostic reserve candidate (`reserve=1`) to reduce Phase1/Phase2 early collapse without changing ABR or context budget.")
    lines.append("")
    lines.append("## 2. Artifact Validation")
    av = summary_json["artifact_validation"]
    lines.append(f"- scheduled_runs: {av['scheduled_runs']}")
    lines.append(f"- completed_runs: {av['completed_runs']}")
    lines.append(f"- incomplete_attempts: {av['incomplete_attempts']}")
    lines.append(f"- extra_not_in_manifest: {av['extra_not_in_manifest']}")
    lines.append(f"- multi_attempt_run_names: {av['multi_attempt_run_names']}")
    lines.append(f"- parse_errors: {av['parse_errors']}")
    lines.append(f"- sample_id_mismatch: {av['sample_id_mismatch']}")
    lines.append("")
    lines.append("## 3. Reserve Forcing Audit")
    lines.append("| dataset | reserve_trigger_rate | reserve_ABR_candidate_rate | reserve_ABR_selected_rate | reserve_rendered_rate | reserve_forced_selected_rate | reserve_forced_rendered_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        r = dict(row.get("reserve_metrics", {}) or {})
        lines.append(
            "| {dataset} | {t} | {cand} | {abr} | {rend} | {fs} | {fr} |".format(
                dataset=_safe_text(row.get("dataset", "")),
                t=_fmt(r.get("reserve_trigger_rate"), 4),
                cand=_fmt(r.get("reserve_ABR_candidate_rate"), 4),
                abr=_fmt(r.get("reserve_ABR_selected_rate"), 4),
                rend=_fmt(r.get("reserve_rendered_rate"), 4),
                fs=_fmt(r.get("reserve_forced_selected_rate"), 4),
                fr=_fmt(r.get("reserve_forced_rendered_rate"), 4),
            )
        )
    lines.append("")
    lines.append("## 4. Paired Main Results")
    lines.append("| dataset | profile | EM | F1 | Recall@5 | SF_recall | SF_precision | SF_f1 | avg_context_tokens | F1_per_1k_context_tokens | answer_string_hit | answer_bearing_density | retrieval_ms | generation_ms | total_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        dataset = _safe_text(row.get("dataset", ""))
        for prof_key in ("baseline", "variant"):
            prof_name = args.baseline_profile if prof_key == "baseline" else args.variant_profile
            m = dict(row.get(prof_key, {}) or {})
            lines.append(
                "| {d} | {p} | {em} | {f1} | {r5} | {sfr} | {sfp} | {sff} | {ctx} | {f1k} | {ah} | {ad} | {rm} | {gm} | {tm} |".format(
                    d=dataset,
                    p=prof_name,
                    em=_fmt(m.get("EM"), 4),
                    f1=_fmt(m.get("F1"), 4),
                    r5=_fmt(m.get("Recall@5"), 4),
                    sfr=_fmt(m.get("supporting_fact_recall"), 4),
                    sfp=_fmt(m.get("supporting_fact_precision"), 4),
                    sff=_fmt(m.get("supporting_fact_f1"), 4),
                    ctx=_fmt(m.get("avg_context_tokens"), 2),
                    f1k=_fmt(m.get("F1_per_1k_context_tokens"), 4),
                    ah=_fmt(m.get("answer_string_hit"), 4),
                    ad=_fmt(m.get("answer_bearing_density"), 4),
                    rm=_fmt(m.get("retrieval_ms"), 2),
                    gm=_fmt(m.get("generation_ms"), 2),
                    tm=_fmt(m.get("total_ms"), 2),
                )
            )
    lines.append("")
    lines.append("## 5. Delta vs Baseline")
    lines.append("| dataset | ΔEM | ΔF1 | ΔSF_recall | ΔSF_precision | Δavg_context_tokens | ΔF1_per_1k_context_tokens | Δanswer_string_hit | Δanswer_bearing_density | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        d = dict(row.get("delta", {}) or {})
        lines.append(
            "| {dataset} | {em} | {f1} | {sfr} | {sfp} | {ctx} | {f1k} | {ah} | {ad} | {rm} | {tm} |".format(
                dataset=_safe_text(row.get("dataset", "")),
                em=_fmt(d.get("EM"), 4),
                f1=_fmt(d.get("F1"), 4),
                sfr=_fmt(d.get("supporting_fact_recall"), 4),
                sfp=_fmt(d.get("supporting_fact_precision"), 4),
                ctx=_fmt(d.get("avg_context_tokens"), 2),
                f1k=_fmt(d.get("F1_per_1k_context_tokens"), 4),
                ah=_fmt(d.get("answer_string_hit"), 4),
                ad=_fmt(d.get("answer_bearing_density"), 4),
                rm=_fmt(d.get("retrieval_ms"), 2),
                tm=_fmt(d.get("total_ms"), 2),
            )
        )
    lines.append("")
    lines.append("## 6. Dataset-Level Interpretation")
    for row in summary_rows:
        d = dict(row.get("delta", {}) or {})
        lines.append(
            "- {dataset}: ΔF1={df1}, ΔSF_recall={dsf}, Δanswer_hit={dah}, Δctx={dctx}, Δretrieval_ms={drm}".format(
                dataset=_safe_text(row.get("dataset", "")),
                df1=_fmt(d.get("F1"), 4),
                dsf=_fmt(d.get("supporting_fact_recall"), 4),
                dah=_fmt(d.get("answer_string_hit"), 4),
                dctx=_fmt(d.get("avg_context_tokens"), 2),
                drm=_fmt(d.get("retrieval_ms"), 2),
            )
        )
    lines.append("")
    lines.append("## 7. Stagewise Evidence Transfer")
    for row in summary_rows:
        lines.append(f"- {row['dataset']} baseline deltas: {row['baseline_stage_delta']}")
        lines.append(f"- {row['dataset']} variant deltas: {row['variant_stage_delta']}")
    lines.append("")
    lines.append("## 8. Reserve Triggered Subset Analysis")
    lines.append("| dataset | subset | query_count | baseline_F1 | variant_F1 | delta_F1 | baseline_answer_hit | variant_answer_hit | delta_answer_hit |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in subset_rows:
        lines.append(
            "| {dataset} | {subset} | {n} | {bf1} | {vf1} | {df1} | {bah} | {vah} | {dah} |".format(
                dataset=_safe_text(row.get("dataset", "")),
                subset=_safe_text(row.get("subset", "")),
                n=_safe_int(row.get("query_count", 0), 0),
                bf1=_fmt(row.get("baseline_F1"), 4),
                vf1=_fmt(row.get("variant_F1"), 4),
                df1=_fmt(row.get("delta_F1"), 4),
                bah=_fmt(row.get("baseline_answer_string_hit"), 4),
                vah=_fmt(row.get("variant_answer_string_hit"), 4),
                dah=_fmt(row.get("delta_answer_string_hit"), 4),
            )
        )
    lines.append(f"- triggered_subset_delta_F1(avg): {_fmt(trig_delta_f1, 4)}")
    lines.append(f"- non_triggered_subset_delta_F1(avg): {_fmt(nontrig_delta_f1, 4)}")
    lines.append(f"- triggered_subset_delta_answer_hit(avg): {_fmt(trig_delta_answer_hit, 4)}")
    lines.append(f"- non_triggered_subset_delta_answer_hit(avg): {_fmt(nontrig_delta_answer_hit, 4)}")
    lines.append("")
    lines.append("## 9. Token/Latency Guardrail")
    lines.append("| dataset | reserve_trigger_rate | reserve_phase2_survival_rate | reserve_sentence_candidate_rate | reserve_ABR_candidate_rate | reserve_ABR_selected_rate | reserve_rendered_rate | reserve_forced_selected_rate | reserve_forced_rendered_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        r = dict(row.get("reserve_metrics", {}) or {})
        lines.append(
            "| {dataset} | {t} | {p2} | {sc} | {abr_cand} | {abr} | {rend} | {forced_sel} | {forced_rend} |".format(
                dataset=_safe_text(row.get("dataset", "")),
                t=_fmt(r.get("reserve_trigger_rate"), 4),
                p2=_fmt(r.get("reserve_phase2_survival_rate"), 4),
                sc=_fmt(r.get("reserve_sentence_candidate_rate"), 4),
                abr_cand=_fmt(r.get("reserve_ABR_candidate_rate"), 4),
                abr=_fmt(r.get("reserve_ABR_selected_rate"), 4),
                rend=_fmt(r.get("reserve_rendered_rate"), 4),
                forced_sel=_fmt(r.get("reserve_forced_selected_rate"), 4),
                forced_rend=_fmt(r.get("reserve_forced_rendered_rate"), 4),
            )
        )
    lines.append("")
    lines.append("- avg_delta_f1: {v}".format(v=_fmt(avg_delta_f1, 4)))
    lines.append("- max_dataset_drop_f1: {v}".format(v=_fmt(max_drop, 4)))
    lines.append("- avg_delta_context_tokens: {v}".format(v=_fmt(avg_token_delta, 2)))
    lines.append("- avg_delta_retrieval_ms: {v}".format(v=_fmt(avg_ret_delta, 2)))
    lines.append("- max_reserve_forced_selected_rate: {v}".format(v=_fmt(max_forced_selected, 4)))
    lines.append("- max_reserve_forced_rendered_rate: {v}".format(v=_fmt(max_forced_rendered, 4)))
    lines.append("")
    lines.append("## 10. Improved Examples")
    lines.append(f"- `improved_examples_top20.md` ({len(improved)} rows)")
    lines.append(f"- `reserve_triggered_improved_examples_top20.md` ({len(reserve_triggered_improved)} rows)")
    lines.append("")
    lines.append("## 11. Regression Examples")
    lines.append(f"- `regression_examples_top20.md` ({len(regressed)} rows)")
    lines.append(f"- `reserve_triggered_regression_examples_top20.md` ({len(reserve_triggered_regressed)} rows)")
    lines.append("")
    lines.append("## 12. Decision")
    lines.append(f"- {decision}")
    lines.append("")

    lines.append("## Appendix. Phase1 Reserve Diagnostics")
    lines.append("| dataset | reserve_trigger_rate | reserve_phase2_survival_rate | reserve_sentence_candidate_rate | reserve_ABR_candidate_rate | reserve_ABR_selected_rate | reserve_rendered_rate | reserve_forced_selected_rate | reserve_forced_rendered_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        r = dict(row.get("reserve_metrics", {}) or {})
        lines.append(
            "| {dataset} | {t} | {p2} | {sc} | {abr_cand} | {abr} | {rend} | {forced_sel} | {forced_rend} |".format(
                dataset=_safe_text(row.get("dataset", "")),
                t=_fmt(r.get("reserve_trigger_rate"), 4),
                p2=_fmt(r.get("reserve_phase2_survival_rate"), 4),
                sc=_fmt(r.get("reserve_sentence_candidate_rate"), 4),
                abr_cand=_fmt(r.get("reserve_ABR_candidate_rate"), 4),
                abr=_fmt(r.get("reserve_ABR_selected_rate"), 4),
                rend=_fmt(r.get("reserve_rendered_rate"), 4),
                forced_sel=_fmt(r.get("reserve_forced_selected_rate"), 4),
                forced_rend=_fmt(r.get("reserve_forced_rendered_rate"), 4),
            )
        )
    lines.append("")

    summary_md_path = out_root / _safe_text(args.summary_md_name or "PHASE6X_EARLY_COLLAPSE_RECOVERY_SUMMARY.md")
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    # Backward-compatible alias for existing tooling.
    if summary_md_path.name != "PHASE6X_EARLY_COLLAPSE_RECOVERY_SUMMARY.md":
        (out_root / "PHASE6X_EARLY_COLLAPSE_RECOVERY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
