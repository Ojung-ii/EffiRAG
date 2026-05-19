#!/usr/bin/env python3
"""Summarize PHASE6V semantic-sufficiency ABR paired run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts

from effirag.evidence_flow_audit import load_samples_for_dataset, stage_metrics
from summarize_phase6u_stagewise_evidence_audit import _build_stage_payloads

ABR_STAGE_KEYS = [
    "sentence_candidates_after_text_rerank",
    "evidence_atoms_before_abr_selector",
    "abr_selected_evidence",
    "rendered_compact_context",
]

STAGE_TO_METRIC_NAME = {
    "sentence_candidates_after_text_rerank": "sentence_candidate_SF_recall",
    "evidence_atoms_before_abr_selector": "evidence_atom_SF_recall",
    "abr_selected_evidence": "ABR_selected_SF_recall",
    "rendered_compact_context": "rendered_SF_recall",
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def _fmt(value: Any, nd: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def _ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        token = _safe_text(value)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(_ordered_unique(a))
    sb = set(_ordered_unique(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa.intersection(sb)) / float(len(sa.union(sb))))


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


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
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, Mapping):
            out.append(dict(obj))
    return out


def normalize_sentence_support_id(sentence_id: str) -> str:
    sid = _safe_text(sentence_id)
    if not sid:
        return ""
    parts = sid.split("::")
    if len(parts) >= 3 and parts[0] in {"chunk", "sentence"}:
        return f"{parts[1]}::{parts[2]}"
    return sid


def _extract_selected_ids(row: Mapping[str, Any]) -> List[str]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    ids = list(row.get("retrieval_selected_sentence_ids", []) or retrieval.get("selected_sentence_ids", []) or [])
    return [normalize_sentence_support_id(x) for x in _ordered_unique(ids)]


def _extract_rendered_ids(row: Mapping[str, Any]) -> List[str]:
    rendered = dict((row.get("rendered", {}) or {}))
    ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
    return [normalize_sentence_support_id(x) for x in _ordered_unique(ids)]


@dataclass
class RunData:
    run_name: str
    dataset: str
    profile: str
    role: str
    summary: Dict[str, Any]
    query_rows: List[Dict[str, Any]]


def _load_completed_runs(out_root: Path) -> List[RunData]:
    report = collect_phase6s_artifacts(out_root)
    scheduled = {_safe_text(r.get("run_name")): dict(r) for r in list(report.get("scheduled_runs", []) or [])}
    runs: List[RunData] = []
    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        meta = scheduled.get(run_name, {})
        dataset = _safe_text(meta.get("dataset", item.get("dataset", "")))
        profile = _safe_text(meta.get("profile", item.get("profile", "")))
        role = _safe_text(meta.get("role", ""))
        summary = dict(item.get("summary", {}) or {})
        query_path = Path(_safe_text(item.get("query_path"))).resolve()
        query_rows = _read_jsonl(query_path)
        runs.append(RunData(run_name=run_name, dataset=dataset, profile=profile, role=role, summary=summary, query_rows=query_rows))
    return runs


def _stagewise_recall_for_run(run: RunData) -> Dict[str, Optional[float]]:
    sample_map = load_samples_for_dataset(run.dataset)
    values: Dict[str, List[float]] = {k: [] for k in ABR_STAGE_KEYS}

    for row in run.query_rows:
        sample_id = _safe_text(row.get("sample_id"))
        sample = sample_map.get(sample_id)
        if sample is None:
            continue
        graph_mode = _safe_text(
            (((row.get("retrieval") or {}).get("diagnostics") or {}).get("graph_mode", "current_entity_graph"))
        )
        payloads = _build_stage_payloads(row)
        for stage in ABR_STAGE_KEYS:
            payload = dict(payloads.get(stage, {}) or {})
            ids = list(payload.get("ids", []) or [])
            texts = list(payload.get("texts", []) or [])
            metric = stage_metrics(sample, ids, texts, graph_mode)
            values[stage].append(_safe_float(metric.get("sf_R", 0.0), 0.0))

    out: Dict[str, Optional[float]] = {}
    for stage in ABR_STAGE_KEYS:
        metric_name = STAGE_TO_METRIC_NAME[stage]
        stage_vals = values.get(stage, [])
        out[metric_name] = (_mean(stage_vals) if stage_vals else None)
    return out


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
    f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
    recall5 = _safe_float(summary.get("recall_at_5", summary.get("Recall@5", 0.0)), 0.0)
    sf_precision = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
    sf_recall = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
    sf_f1 = (2.0 * sf_precision * sf_recall / (sf_precision + sf_recall)) if (sf_precision + sf_recall) > 0 else 0.0
    context_tokens = _safe_float(summary.get("avg_context_tokens", summary.get("prompt_tokens_avg", 0.0)), 0.0)
    f1_per_1k = (f1 * 1000.0 / context_tokens) if context_tokens > 0 else 0.0
    return {
        "Recall@5": recall5,
        "EM": em,
        "F1": f1,
        "avg_context_tokens": context_tokens,
        "F1_per_1k_context_tokens": f1_per_1k,
        "supporting_fact_precision": sf_precision,
        "supporting_fact_recall": sf_recall,
        "supporting_fact_f1": sf_f1,
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
        "answer_string_hit": _safe_float(summary.get("answer_surface_present", 0.0), 0.0),
        "answer_string_rank_avg": _safe_float(summary.get("answer_surface_position_avg", -1.0), -1.0),
        "answer_bearing_density": _safe_float(summary.get("answer_surface_token_density", 0.0), 0.0),
        "answer_bearing_sentence_count": _safe_float(summary.get("answer_bearing_chunk_present", 0.0), 0.0),
    }


def _semantic_diag_for_run(run: RunData) -> Dict[str, Optional[float]]:
    coverage_vals: List[float] = []
    q_vals: List[float] = []
    sel_vals: List[float] = []
    unsel_vals: List[float] = []
    prior_count_vals: List[float] = []
    enabled_vals: List[int] = []

    for row in run.query_rows:
        diag = dict((((row.get("retrieval") or {}).get("diagnostics") or {}).get("unified_acr_rcedr_diag", {}) or {}))
        if not diag:
            continue
        if "semantic_score_coverage_rate" in diag:
            coverage_vals.append(_safe_float(diag.get("semantic_score_coverage_rate"), 0.0))
        if "semantic_query_score_avg" in diag:
            q_vals.append(_safe_float(diag.get("semantic_query_score_avg"), 0.0))
        if "semantic_selected_score_avg" in diag:
            sel_vals.append(_safe_float(diag.get("semantic_selected_score_avg"), 0.0))
        if "semantic_unselected_score_avg" in diag:
            unsel_vals.append(_safe_float(diag.get("semantic_unselected_score_avg"), 0.0))
        if "semantic_prior_applied_count" in diag:
            prior_count_vals.append(_safe_float(diag.get("semantic_prior_applied_count"), 0.0))
        enabled_vals.append(1 if bool(diag.get("semantic_sufficiency_enabled", False)) else 0)

    return {
        "semantic_sufficiency_enabled": (bool(any(enabled_vals)) if enabled_vals else False),
        "semantic_score_coverage_rate": (_mean(coverage_vals) if coverage_vals else None),
        "semantic_query_score_avg": (_mean(q_vals) if q_vals else None),
        "semantic_selected_score_avg": (_mean(sel_vals) if sel_vals else None),
        "semantic_unselected_score_avg": (_mean(unsel_vals) if unsel_vals else None),
        "semantic_prior_applied_count_avg": (_mean(prior_count_vals) if prior_count_vals else None),
    }


def _extract_query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    metrics = dict(row.get("metrics", {}) or {})
    eff = dict(row.get("efficiency", {}) or {})
    return {
        "f1": _safe_float(metrics.get("f1", 0.0), 0.0),
        "em": _safe_float(metrics.get("em", 0.0), 0.0),
        "sf_recall": _safe_float(metrics.get("supporting_fact_recall", 0.0), 0.0),
        "sf_precision": _safe_float(metrics.get("supporting_fact_precision", 0.0), 0.0),
        "retrieval_ms": _safe_float(eff.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(eff.get("total_latency_ms", 0.0), 0.0),
        "prompt_tokens": _safe_float(dict(row.get("generation_diagnostics", {}) or {}).get("prompt_tokens", 0.0), 0.0),
        "answer_string_hit": _safe_float(metrics.get("answer_surface_present", 0.0), 0.0),
        "answer_string_rank": _safe_float(metrics.get("answer_surface_position", -1.0), -1.0),
        "answer_bearing_density": _safe_float(metrics.get("answer_surface_token_density", 0.0), 0.0),
    }


def _pair_rows(dataset: str, baseline_rows: Sequence[Mapping[str, Any]], variant_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bmap = {_safe_text(r.get("sample_id")): r for r in baseline_rows}
    vmap = {_safe_text(r.get("sample_id")): r for r in variant_rows}
    ordered_ids = [_safe_text(r.get("sample_id")) for r in baseline_rows if _safe_text(r.get("sample_id"))]
    paired: List[Dict[str, Any]] = []

    for sid in ordered_ids:
        b = bmap.get(sid)
        v = vmap.get(sid)
        if b is None or v is None:
            continue

        bm = _extract_query_metrics(b)
        vm = _extract_query_metrics(v)
        b_sel = _extract_selected_ids(b)
        v_sel = _extract_selected_ids(v)
        b_rnd = _extract_rendered_ids(b)
        v_rnd = _extract_rendered_ids(v)

        b_ctx = _safe_text(dict(b.get("rendered", {}) or {}).get("text", ""))
        v_ctx = _safe_text(dict(v.get("rendered", {}) or {}).get("text", ""))
        b_prompt_hash = _hash_text(_safe_text(b.get("question", "")) + "\n" + b_ctx)
        v_prompt_hash = _hash_text(_safe_text(v.get("question", "")) + "\n" + v_ctx)

        delta_f1 = float(vm["f1"] - bm["f1"])
        delta_sf = float(vm["sf_recall"] - bm["sf_recall"])
        if delta_f1 > 0.05 or delta_sf > 0.10:
            cls = "improved"
        elif delta_f1 < -0.05 or delta_sf < -0.10:
            cls = "regressed"
        elif (delta_f1 > 0 and delta_sf < 0) or (delta_f1 < 0 and delta_sf > 0):
            cls = "mixed"
        else:
            cls = "unchanged"

        paired.append(
            {
                "dataset": dataset,
                "sample_id": sid,
                "question": _safe_text(b.get("question", "")),
                "baseline_prediction": _safe_text(b.get("prediction", "")),
                "variant_prediction": _safe_text(v.get("prediction", "")),
                "baseline_f1": float(bm["f1"]),
                "variant_f1": float(vm["f1"]),
                "baseline_sf_recall": float(bm["sf_recall"]),
                "variant_sf_recall": float(vm["sf_recall"]),
                "baseline_sf_precision": float(bm["sf_precision"]),
                "variant_sf_precision": float(vm["sf_precision"]),
                "baseline_answer_string_hit": float(bm["answer_string_hit"]),
                "variant_answer_string_hit": float(vm["answer_string_hit"]),
                "baseline_answer_string_rank": float(bm["answer_string_rank"]),
                "variant_answer_string_rank": float(vm["answer_string_rank"]),
                "baseline_answer_bearing_density": float(bm["answer_bearing_density"]),
                "variant_answer_bearing_density": float(vm["answer_bearing_density"]),
                "baseline_retrieval_ms": float(bm["retrieval_ms"]),
                "variant_retrieval_ms": float(vm["retrieval_ms"]),
                "baseline_total_ms": float(bm["total_ms"]),
                "variant_total_ms": float(vm["total_ms"]),
                "baseline_prompt_tokens": float(bm["prompt_tokens"]),
                "variant_prompt_tokens": float(vm["prompt_tokens"]),
                "selected_sentence_jaccard": float(_jaccard(b_sel, v_sel)),
                "rendered_sentence_jaccard": float(_jaccard(b_rnd, v_rnd)),
                "prompt_hash_changed": bool(b_prompt_hash != v_prompt_hash),
                "context_hash_changed": bool(_hash_text(b_ctx) != _hash_text(v_ctx)),
                "delta_f1": float(delta_f1),
                "delta_sf_recall": float(delta_sf),
                "classification": str(cls),
            }
        )

    pairing = {
        "baseline_count": int(len(baseline_rows)),
        "variant_count": int(len(variant_rows)),
        "intersection_count": int(len(paired)),
        "identical_order": bool(
            [_safe_text(r.get("sample_id")) for r in baseline_rows] == [_safe_text(r.get("sample_id")) for r in variant_rows]
        ),
    }
    return paired, pairing


def _decision(delta_rows: Sequence[Mapping[str, Any]], rows_main: Sequence[Mapping[str, Any]]) -> str:
    if not delta_rows:
        return "hold"

    by_dataset = {str(r.get("dataset")): dict(r) for r in delta_rows}
    main_index = {(str(r.get("dataset")), str(r.get("role"))): dict(r) for r in rows_main}

    for ds, d in by_dataset.items():
        if _safe_float(d.get("delta_F1"), 0.0) < -0.02:
            return "reject"

    for ds in by_dataset:
        b = main_index.get((ds, "baseline"), {})
        v = main_index.get((ds, "variant"), {})
        b_ctx = _safe_float(b.get("avg_context_tokens"), 0.0)
        v_ctx = _safe_float(v.get("avg_context_tokens"), 0.0)
        b_ret = _safe_float(b.get("retrieval_ms"), 0.0)
        v_ret = _safe_float(v.get("retrieval_ms"), 0.0)
        if b_ctx > 0 and ((v_ctx - b_ctx) / b_ctx) > 0.20:
            return "reject"
        if b_ret > 0 and ((v_ret - b_ret) / b_ret) > 0.20:
            return "reject"

    avg_df1 = _mean([_safe_float(d.get("delta_F1"), 0.0) for d in delta_rows])
    avg_drec = _mean([_safe_float(d.get("delta_SF_recall"), 0.0) for d in delta_rows])
    avg_df1k = _mean([_safe_float(d.get("delta_F1_per_1k_context_tokens"), 0.0) for d in delta_rows])
    avg_dhit = _mean([_safe_float(d.get("delta_answer_string_hit"), 0.0) for d in delta_rows])

    all_non_regressive = all(_safe_float(d.get("delta_F1"), 0.0) >= 0.0 for d in delta_rows)
    if all_non_regressive and avg_df1 >= 0.02 and avg_df1k >= 0.0 and avg_dhit >= 0.0:
        token_ok = True
        latency_ok = True
        for ds in by_dataset:
            b = main_index.get((ds, "baseline"), {})
            v = main_index.get((ds, "variant"), {})
            b_ctx = _safe_float(b.get("avg_context_tokens"), 0.0)
            v_ctx = _safe_float(v.get("avg_context_tokens"), 0.0)
            b_ret = _safe_float(b.get("retrieval_ms"), 0.0)
            v_ret = _safe_float(v.get("retrieval_ms"), 0.0)
            if b_ctx > 0 and ((v_ctx - b_ctx) / b_ctx) > 0.10:
                token_ok = False
            if b_ret > 0 and ((v_ret - b_ret) / b_ret) > 0.15:
                latency_ok = False
        if token_ok and latency_ok:
            return "strong_accept"

    if avg_df1 >= -0.005 and (avg_drec >= -0.01 or avg_df1 > 0.0):
        return "hold"
    return "reject"


def _write_examples(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    lines: List[str] = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for idx, row in enumerate(rows, start=1):
        lines.append(f"## {idx}. {_safe_text(row.get('dataset'))} / {_safe_text(row.get('sample_id'))}")
        lines.append(f"- question: {_safe_text(row.get('question'))}")
        lines.append(f"- baseline_prediction: {_safe_text(row.get('baseline_prediction'))}")
        lines.append(f"- variant_prediction: {_safe_text(row.get('variant_prediction'))}")
        lines.append(f"- delta_f1: {_fmt(row.get('delta_f1'), 4)}")
        lines.append(f"- delta_sf_recall: {_fmt(row.get('delta_sf_recall'), 4)}")
        lines.append(f"- baseline_answer_string_hit / variant_answer_string_hit: {_fmt(row.get('baseline_answer_string_hit'), 4)} / {_fmt(row.get('variant_answer_string_hit'), 4)}")
        lines.append(f"- baseline_answer_bearing_density / variant_answer_bearing_density: {_fmt(row.get('baseline_answer_bearing_density'), 4)} / {_fmt(row.get('variant_answer_bearing_density'), 4)}")
        lines.append(f"- selected_sentence_jaccard: {_fmt(row.get('selected_sentence_jaccard'), 4)}")
        lines.append(f"- rendered_sentence_jaccard: {_fmt(row.get('rendered_sentence_jaccard'), 4)}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(
    *,
    out_root: Path,
    datasets: Sequence[str],
    baseline_profile: str,
    variant_profile: str,
    top_k: int,
) -> Dict[str, Any]:
    artifact = collect_phase6s_artifacts(out_root)
    runs = _load_completed_runs(out_root)
    run_index = {(r.dataset, r.profile): r for r in runs}

    rows_main: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    paired_all: List[Dict[str, Any]] = []
    pairing_info: Dict[str, Dict[str, Any]] = {}

    for dataset in datasets:
        base = run_index.get((dataset, baseline_profile))
        variant = run_index.get((dataset, variant_profile))
        if base is None or variant is None:
            continue

        base_stage = _stagewise_recall_for_run(base)
        var_stage = _stagewise_recall_for_run(variant)
        base_sem = _semantic_diag_for_run(base)
        var_sem = _semantic_diag_for_run(variant)

        bm = _summary_metrics(base.summary)
        vm = _summary_metrics(variant.summary)

        base_row = {
            "dataset": dataset,
            "profile": baseline_profile,
            "role": "baseline",
            **bm,
            **base_stage,
            **base_sem,
        }
        var_row = {
            "dataset": dataset,
            "profile": variant_profile,
            "role": "variant",
            **vm,
            **var_stage,
            **var_sem,
        }
        rows_main.extend([base_row, var_row])

        delta_rows.append(
            {
                "dataset": dataset,
                "variant_profile": variant_profile,
                "delta_EM": float(vm["EM"] - bm["EM"]),
                "delta_F1": float(vm["F1"] - bm["F1"]),
                "delta_F1_per_1k_context_tokens": float(vm["F1_per_1k_context_tokens"] - bm["F1_per_1k_context_tokens"]),
                "delta_SF_recall": float(vm["supporting_fact_recall"] - bm["supporting_fact_recall"]),
                "delta_SF_precision": float(vm["supporting_fact_precision"] - bm["supporting_fact_precision"]),
                "delta_avg_context_tokens": float(vm["avg_context_tokens"] - bm["avg_context_tokens"]),
                "delta_retrieval_ms": float(vm["retrieval_ms"] - bm["retrieval_ms"]),
                "delta_total_ms": float(vm["total_ms"] - bm["total_ms"]),
                "delta_answer_string_hit": float(vm["answer_string_hit"] - bm["answer_string_hit"]),
                "delta_answer_string_rank_avg": float(vm["answer_string_rank_avg"] - bm["answer_string_rank_avg"]),
                "delta_answer_bearing_density": float(vm["answer_bearing_density"] - bm["answer_bearing_density"]),
            }
        )

        paired, pairing = _pair_rows(dataset, base.query_rows, variant.query_rows)
        paired_all.extend(paired)
        pairing_info[dataset] = dict(pairing)

    # Required outputs.
    _write_jsonl(out_root / "paired_query_comparison.jsonl", paired_all)

    semantic_csv_rows: List[Dict[str, Any]] = []
    for row in rows_main:
        semantic_csv_rows.append(
            {
                "dataset": row.get("dataset"),
                "profile": row.get("profile"),
                "role": row.get("role"),
                "semantic_sufficiency_enabled": row.get("semantic_sufficiency_enabled"),
                "semantic_score_coverage_rate": row.get("semantic_score_coverage_rate"),
                "semantic_query_score_avg": row.get("semantic_query_score_avg"),
                "semantic_selected_score_avg": row.get("semantic_selected_score_avg"),
                "semantic_unselected_score_avg": row.get("semantic_unselected_score_avg"),
                "semantic_prior_applied_count_avg": row.get("semantic_prior_applied_count_avg"),
                "sentence_candidate_SF_recall": row.get("sentence_candidate_SF_recall"),
                "evidence_atom_SF_recall": row.get("evidence_atom_SF_recall"),
                "ABR_selected_SF_recall": row.get("ABR_selected_SF_recall"),
                "rendered_SF_recall": row.get("rendered_SF_recall"),
            }
        )
    _write_csv(
        out_root / "semantic_sufficiency_diagnostics.csv",
        semantic_csv_rows,
        [
            "dataset",
            "profile",
            "role",
            "semantic_sufficiency_enabled",
            "semantic_score_coverage_rate",
            "semantic_query_score_avg",
            "semantic_selected_score_avg",
            "semantic_unselected_score_avg",
            "semantic_prior_applied_count_avg",
            "sentence_candidate_SF_recall",
            "evidence_atom_SF_recall",
            "ABR_selected_SF_recall",
            "rendered_SF_recall",
        ],
    )

    improved_examples = sorted(
        [r for r in paired_all if _safe_text(r.get("classification")) == "improved"],
        key=lambda r: (_safe_float(r.get("delta_f1"), 0.0), _safe_float(r.get("delta_sf_recall"), 0.0)),
        reverse=True,
    )[: max(1, int(top_k))]
    regression_examples = sorted(
        [r for r in paired_all if _safe_text(r.get("classification")) == "regressed"],
        key=lambda r: (_safe_float(r.get("delta_f1"), 0.0), _safe_float(r.get("delta_sf_recall"), 0.0)),
    )[: max(1, int(top_k))]

    _write_examples(
        out_root / "answer_sufficiency_examples_top20.md",
        "Answer Sufficiency Improved Examples (Top 20)",
        improved_examples,
    )
    _write_examples(
        out_root / "regression_examples_top20.md",
        "Regression Examples (Top 20)",
        regression_examples,
    )

    decision = _decision(delta_rows, rows_main)

    summary = {
        "phase": "phase6v_semantic_sufficiency_abr",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "datasets": list(datasets),
        "baseline_profile": baseline_profile,
        "variant_profile": variant_profile,
        "artifact_counts": dict((artifact.get("counts", {}) or {})),
        "rows_main": rows_main,
        "delta_rows": delta_rows,
        "pairing_info": pairing_info,
        "paired_count": int(len(paired_all)),
        "decision": decision,
    }
    _write_json(out_root / "phase6v_semantic_sufficiency_abr_summary.json", summary)

    # Markdown summary.
    counts = dict((artifact.get("counts", {}) or {}))
    lines: List[str] = []
    lines.append("# PHASE6V Semantic-Sufficiency ABR Summary")
    lines.append("")
    lines.append("## 1. Goal")
    lines.append("Paired n=100 diagnostic for semantic-sufficiency prior inside unified ACR-RCEDR selector.")
    lines.append("")
    lines.append("## 2. Artifact Validation")
    lines.append(
        "- scheduled_runs={scheduled_runs}, completed_run_names={completed_run_names}, incomplete_attempts={incomplete_attempts}, extra_not_in_manifest={extra_not_in_manifest}, multi_attempt_run_names={multi_attempt_run_names}, parse_errors={parse_errors}".format(
            scheduled_runs=counts.get("scheduled_runs", "n/a"),
            completed_run_names=counts.get("completed_run_names", "n/a"),
            incomplete_attempts=counts.get("incomplete_attempts", "n/a"),
            extra_not_in_manifest=counts.get("extra_not_in_manifest", "n/a"),
            multi_attempt_run_names=counts.get("multi_attempt_run_names", "n/a"),
            parse_errors=counts.get("parse_errors", "n/a"),
        )
    )
    lines.append("")

    lines.append("## 3. Paired Main Results")
    lines.append("| dataset | profile | role | Recall@5 | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | supporting_fact_precision | supporting_fact_recall | supporting_fact_f1 | retrieval_ms | generation_ms | total_ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows_main, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {role} | {r5} | {em} | {f1} | {ctx} | {f1k} | {sp} | {sr} | {sf1} | {rm} | {gm} | {tm} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("profile")),
                role=_safe_text(row.get("role")),
                r5=_fmt(row.get("Recall@5"), 4),
                em=_fmt(row.get("EM"), 4),
                f1=_fmt(row.get("F1"), 4),
                ctx=_fmt(row.get("avg_context_tokens"), 3),
                f1k=_fmt(row.get("F1_per_1k_context_tokens"), 4),
                sp=_fmt(row.get("supporting_fact_precision"), 4),
                sr=_fmt(row.get("supporting_fact_recall"), 4),
                sf1=_fmt(row.get("supporting_fact_f1"), 4),
                rm=_fmt(row.get("retrieval_ms"), 2),
                gm=_fmt(row.get("generation_ms"), 2),
                tm=_fmt(row.get("total_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 4. Delta vs Baseline")
    lines.append("| dataset | variant_profile | ΔEM | ΔF1 | ΔF1_per_1k_context_tokens | ΔSF_recall | ΔSF_precision | Δavg_context_tokens | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(delta_rows, key=lambda r: _safe_text(r.get("dataset"))):
        lines.append(
            "| {dataset} | {profile} | {dem} | {df1} | {df1k} | {dsr} | {dsp} | {dctx} | {drm} | {dtm} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("variant_profile")),
                dem=_fmt(row.get("delta_EM"), 4),
                df1=_fmt(row.get("delta_F1"), 4),
                df1k=_fmt(row.get("delta_F1_per_1k_context_tokens"), 4),
                dsr=_fmt(row.get("delta_SF_recall"), 4),
                dsp=_fmt(row.get("delta_SF_precision"), 4),
                dctx=_fmt(row.get("delta_avg_context_tokens"), 3),
                drm=_fmt(row.get("delta_retrieval_ms"), 2),
                dtm=_fmt(row.get("delta_total_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 5. Semantic Score Coverage")
    lines.append("| dataset | profile | role | semantic_sufficiency_enabled | semantic_score_coverage_rate | semantic_query_score_avg | semantic_selected_score_avg | semantic_unselected_score_avg | semantic_prior_applied_count_avg |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(rows_main, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {role} | {enabled} | {cov} | {q} | {s} | {u} | {p} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("profile")),
                role=_safe_text(row.get("role")),
                enabled=str(bool(row.get("semantic_sufficiency_enabled", False))).lower(),
                cov=_fmt(row.get("semantic_score_coverage_rate"), 4),
                q=_fmt(row.get("semantic_query_score_avg"), 4),
                s=_fmt(row.get("semantic_selected_score_avg"), 4),
                u=_fmt(row.get("semantic_unselected_score_avg"), 4),
                p=_fmt(row.get("semantic_prior_applied_count_avg"), 4),
            )
        )
    lines.append("")

    lines.append("## 6. Answer-Sufficiency Metrics")
    lines.append("| dataset | profile | role | answer_string_hit | answer_string_rank_avg | answer_bearing_density | answer_bearing_sentence_count |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in sorted(rows_main, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {role} | {hit} | {rank} | {density} | {bearing} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("profile")),
                role=_safe_text(row.get("role")),
                hit=_fmt(row.get("answer_string_hit"), 4),
                rank=_fmt(row.get("answer_string_rank_avg"), 4),
                density=_fmt(row.get("answer_bearing_density"), 4),
                bearing=_fmt(row.get("answer_bearing_sentence_count"), 4),
            )
        )
    lines.append("")

    lines.append("## 7. ABR-stage Diagnostics")
    lines.append("| dataset | profile | role | sentence_candidate_SF_recall | evidence_atom_SF_recall | ABR_selected_SF_recall | rendered_SF_recall |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in sorted(rows_main, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {role} | {s1} | {s2} | {s3} | {s4} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("profile")),
                role=_safe_text(row.get("role")),
                s1=_fmt(row.get("sentence_candidate_SF_recall"), 4),
                s2=_fmt(row.get("evidence_atom_SF_recall"), 4),
                s3=_fmt(row.get("ABR_selected_SF_recall"), 4),
                s4=_fmt(row.get("rendered_SF_recall"), 4),
            )
        )
    lines.append("")

    lines.append("## 8. Improved Examples")
    lines.append("See `answer_sufficiency_examples_top20.md`.")
    lines.append("")
    lines.append("## 9. Regression Examples")
    lines.append("See `regression_examples_top20.md`.")
    lines.append("")
    lines.append("## 10. Decision")
    lines.append(f"- decision: `{decision}`")

    (out_root / "PHASE6V_SEMANTIC_SUFFICIENCY_ABR_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PHASE6V semantic-sufficiency ABR run")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--baseline-profile", default="unified_acr_rcedr_v12")
    ap.add_argument("--variant-profile", default="unified_acr_rcedr_v12_semantic_sufficiency")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    datasets = [_safe_text(x) for x in str(args.datasets).split(",") if _safe_text(x)]

    summary = build_summary(
        out_root=out_root,
        datasets=datasets,
        baseline_profile=_safe_text(args.baseline_profile),
        variant_profile=_safe_text(args.variant_profile),
        top_k=max(1, int(args.top_k)),
    )

    print(out_root / "PHASE6V_SEMANTIC_SUFFICIENCY_ABR_SUMMARY.md")
    print(out_root / "phase6v_semantic_sufficiency_abr_summary.json")
    print(f"paired_count={summary.get('paired_count', 0)}")


if __name__ == "__main__":
    main()
