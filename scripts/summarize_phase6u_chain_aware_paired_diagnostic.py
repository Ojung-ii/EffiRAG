#!/usr/bin/env python3
"""Summarize PHASE6U paired baseline vs chain-aware diagnostic (smoke50)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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

LAT_KEYS = [
    "retrieval_ms",
    "proposal_total_ms",
    "phase1_run_scoring_ms",
    "phase2_refine_ms",
    "sentence_rerank_ms",
    "unified_acr_rcedr_ms",
    "render_ms",
    "total_ms",
    "chain_aware_extra_ms",
    "role_estimation_ms",
    "complementarity_eval_ms",
    "role_aware_redundancy_ms",
    "chain_gain_eval_calls",
    "redundancy_eval_calls",
    "objective_eval_calls",
    "num_atoms",
    "num_selected_atoms",
    "selected_tokens",
]


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


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(float(v) for v in vals) / float(len(vals)))


def _p95(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    arr = sorted(float(v) for v in vals)
    idx = min(len(arr) - 1, max(0, int(round(0.95 * (len(arr) - 1)))))
    return float(arr[idx])


def _fmt(value: Any, nd: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        s = _safe_text(v)
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
    return float(len(sa & sb) / len(sa | sb))


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
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def normalize_sentence_support_id(sentence_id: str) -> str:
    """Map sentence/chunk unit id to support-fact comparable key: title::sent_idx."""
    sid = _safe_text(sentence_id)
    if not sid:
        return ""
    parts = sid.split("::")
    if len(parts) >= 3 and parts[0] in {"chunk", "sentence"}:
        return f"{parts[1]}::{parts[2]}"
    return sid


def pair_sample_ids(baseline_ids: Sequence[str], variant_ids: Sequence[str]) -> Dict[str, Any]:
    b = [str(x) for x in baseline_ids]
    v = [str(x) for x in variant_ids]
    return {
        "baseline_count": len(b),
        "variant_count": len(v),
        "identical_order": b == v,
        "baseline_only": [x for x in b if x not in set(v)],
        "variant_only": [x for x in v if x not in set(b)],
        "intersection_count": len(set(b) & set(v)),
    }


def extract_latency_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    lb = dict((row.get("latency_breakdown_ms", {}) or {}))
    eff = dict((row.get("efficiency", {}) or {}))
    retrieval = dict((row.get("retrieval", {}) or {}))
    diag = dict((retrieval.get("diagnostics", {}) or {}))
    unified = dict((diag.get("unified_acr_rcedr_diag", {}) or {}))

    out: Dict[str, Any] = {
        "retrieval_ms": _safe_float(eff.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "proposal_total_ms": _safe_float(lb.get("proposal_total_ms", lb.get("proposal_time_ms", 0.0)), 0.0),
        "phase1_run_scoring_ms": _safe_float(lb.get("phase1_run_scoring_ms", 0.0), 0.0),
        "phase2_refine_ms": _safe_float(lb.get("phase2_refine_ms", lb.get("phase2_refinement_time_ms", 0.0)), 0.0),
        "sentence_rerank_ms": _safe_float(lb.get("sentence_rerank_ms", 0.0), 0.0),
        "unified_acr_rcedr_ms": _safe_float(lb.get("unified_acr_rcedr_ms", 0.0), 0.0),
        "render_ms": _safe_float(lb.get("render_ms", eff.get("render_ms", 0.0)), 0.0),
        "total_ms": _safe_float(eff.get("total_latency_ms", 0.0), 0.0),
        "chain_aware_extra_ms": _safe_float(lb.get("unified_acr_rcedr_ms", 0.0), 0.0)
        if bool(unified.get("chain_aware_enabled", False))
        else 0.0,
        "role_estimation_ms": None,
        "complementarity_eval_ms": None,
        "role_aware_redundancy_ms": None,
        "chain_gain_eval_calls": None,
        "redundancy_eval_calls": None,
        "objective_eval_calls": _safe_int(unified.get("objective_eval_calls", 0), 0),
        "num_atoms": _safe_int(unified.get("num_atoms", 0), 0),
        "num_selected_atoms": _safe_int(unified.get("num_selected_atoms", 0), 0),
        "selected_tokens": _safe_int(unified.get("selected_tokens", 0), 0),
    }
    return out


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


@dataclass
class RunData:
    run_name: str
    dataset: str
    profile: str
    role: str
    summary_path: Path
    query_path: Path
    summary: Dict[str, Any]
    query_rows: List[Dict[str, Any]]


def _load_completed_runs(out_root: Path) -> List[RunData]:
    report = collect_phase6s_artifacts(out_root)
    scheduled = {_safe_text(r.get("run_name")): dict(r) for r in list(report.get("scheduled_runs", []) or [])}
    out: List[RunData] = []
    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        meta = scheduled.get(run_name, {})
        dataset = _safe_text(meta.get("dataset", item.get("dataset", "")))
        profile = _safe_text(meta.get("profile", item.get("profile", "")))
        role = _safe_text(meta.get("role", ""))
        summary_path = Path(_safe_text(item.get("summary_path"))).resolve()
        query_path = Path(_safe_text(item.get("query_path"))).resolve()
        summary = dict(item.get("summary", {}) or {})
        qrows = _read_jsonl(query_path)
        out.append(
            RunData(
                run_name=run_name,
                dataset=dataset,
                profile=profile,
                role=role,
                summary_path=summary_path,
                query_path=query_path,
                summary=summary,
                query_rows=qrows,
            )
        )
    return out


def _stagewise_recall_for_run(run: RunData) -> Dict[str, Any]:
    sample_map = load_samples_for_dataset(run.dataset)
    stage_vals: Dict[str, List[float]] = {k: [] for k in ABR_STAGE_KEYS}
    available_rows = 0

    for qrow in run.query_rows:
        sample_id = _safe_text(qrow.get("sample_id"))
        sample = sample_map.get(sample_id)
        if sample is None:
            continue
        available_rows += 1
        graph_mode = _safe_text(
            (((qrow.get("retrieval") or {}).get("diagnostics") or {}).get("graph_mode", "current_entity_graph"))
        )
        payloads = _build_stage_payloads(qrow)
        for stage_key in ABR_STAGE_KEYS:
            p = dict(payloads.get(stage_key, {}) or {})
            ids = list(p.get("ids", []) or [])
            texts = list(p.get("texts", []) or [])
            m = stage_metrics(sample, ids, texts, graph_mode)
            stage_vals[stage_key].append(_safe_float(m.get("sf_R", 0.0), 0.0))

    result: Dict[str, Any] = {
        "available": bool(available_rows > 0),
        "available_rows": int(available_rows),
        "by_stage": {},
    }
    for stage_key in ABR_STAGE_KEYS:
        metric_name = STAGE_TO_METRIC_NAME[stage_key]
        vals = stage_vals.get(stage_key, [])
        if vals:
            result["by_stage"][metric_name] = {
                "available": True,
                "value": _mean(vals),
            }
        else:
            result["by_stage"][metric_name] = {
                "available": False,
                "value": None,
            }
    return result


def _normalize_query_row(run: RunData, row: Mapping[str, Any], stagewise_q: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    diag = dict((retrieval.get("diagnostics", {}) or {}))
    unified = dict((diag.get("unified_acr_rcedr_diag", {}) or {}))
    metrics = dict((row.get("metrics", {}) or {}))
    gen_diag = dict((row.get("generation_diagnostics", {}) or {}))
    rendered = dict((row.get("rendered", {}) or {}))

    selected_ids = list(row.get("retrieval_selected_sentence_ids", []) or retrieval.get("selected_sentence_ids", []) or [])
    rendered_ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])

    context_text = _safe_text(rendered.get("text", ""))
    question = _safe_text(row.get("question", ""))
    prompt_variant = _safe_text(gen_diag.get("prompt_variant", rendered.get("metadata", {}).get("prompt_variant", "")))

    lat = extract_latency_fields(row)

    score_trace = list(unified.get("score_trace", []) or [])
    chain_gain_activated = [
        _safe_text(item.get("sentence_id", ""))
        for item in score_trace
        if _safe_float(item.get("chain_gain", 0.0), 0.0) > 0.0 and _safe_text(item.get("sentence_id", ""))
    ]

    out = {
        "sample_id": _safe_text(row.get("sample_id", "")),
        "question": question,
        "prediction": _safe_text(row.get("prediction", "")),
        "em": _safe_float(metrics.get("em", 0.0), 0.0),
        "f1": _safe_float(metrics.get("f1", 0.0), 0.0),
        "supporting_fact_recall": _safe_float(metrics.get("supporting_fact_recall", 0.0), 0.0),
        "supporting_fact_precision": _safe_float(metrics.get("supporting_fact_precision", 0.0), 0.0),
        "prompt_tokens": _safe_float(gen_diag.get("prompt_tokens", 0.0), 0.0),
        "selected_sentence_ids": _ordered_unique(selected_ids),
        "rendered_sentence_ids": _ordered_unique(rendered_ids),
        "context_text": context_text,
        "context_hash": _hash_text(context_text),
        "prompt_hash": _hash_text(question + "\n" + context_text + "\n" + prompt_variant),
        "chain_gain_total": _safe_float(unified.get("chain_gain_total_avg", 0.0), 0.0),
        "role_aware_redundancy_applied_count": _safe_float(unified.get("role_aware_redundancy_applied_avg", 0.0), 0.0),
        "chain_gain_activated_sentences": _ordered_unique(chain_gain_activated),
        "latency": lat,
        "stage_sf": dict(stagewise_q or {}),
    }
    return out


def _build_latency_rows(dataset: str, profile: str, role: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, List[float]] = {k: [] for k in LAT_KEYS}
    for r in rows:
        lat = dict(r.get("latency", {}) or {})
        for k in LAT_KEYS:
            v = lat.get(k)
            if v is None:
                continue
            out[k].append(_safe_float(v, 0.0))

    row: Dict[str, Any] = {
        "dataset": dataset,
        "profile": profile,
        "role": role,
    }
    for k in LAT_KEYS:
        vals = out.get(k, [])
        row[f"{k}_avg"] = _mean(vals) if vals else None
        row[f"{k}_p95"] = _p95(vals) if vals else None
    return [row]


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    sf_r = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
    sf_p = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
    sf_f1 = (2.0 * sf_r * sf_p / (sf_r + sf_p)) if (sf_r + sf_p) > 0 else 0.0
    prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
    f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
    return {
        "EM": _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0),
        "F1": f1,
        "Recall@5": _safe_float(summary.get("recall_at_5", summary.get("Recall@5", 0.0)), 0.0),
        "avg_context_tokens": prompt_tokens,
        "F1_per_1k_context_tokens": (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0,
        "supporting_fact_precision": sf_p,
        "supporting_fact_recall": sf_r,
        "supporting_fact_f1": sf_f1,
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
    }


def _classify_pair(base: Mapping[str, Any], var: Mapping[str, Any]) -> str:
    b_f1 = _safe_float(base.get("f1", 0.0), 0.0)
    v_f1 = _safe_float(var.get("f1", 0.0), 0.0)
    b_r = _safe_float(base.get("supporting_fact_recall", 0.0), 0.0)
    v_r = _safe_float(var.get("supporting_fact_recall", 0.0), 0.0)

    if (v_f1 > b_f1 + 0.05) or (v_r > b_r + 0.10):
        return "improved"
    if (v_f1 < b_f1 - 0.05) or (v_r < b_r - 0.10):
        return "regressed"
    if ((v_f1 > b_f1 and v_r < b_r) or (v_f1 < b_f1 and v_r > b_r)):
        return "mixed"
    return "unchanged"


def _pair_rows(dataset: str, baseline_rows: Sequence[Mapping[str, Any]], variant_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bmap = {str(r.get("sample_id")): r for r in baseline_rows}
    vmap = {str(r.get("sample_id")): r for r in variant_rows}
    b_ids = [str(r.get("sample_id")) for r in baseline_rows]
    v_ids = [str(r.get("sample_id")) for r in variant_rows]
    pairing = pair_sample_ids(b_ids, v_ids)
    inter = [sid for sid in b_ids if sid in vmap]

    out: List[Dict[str, Any]] = []
    for sid in inter:
        b = bmap[sid]
        v = vmap[sid]
        selected_j = _jaccard(b.get("selected_sentence_ids", []), v.get("selected_sentence_ids", []))
        rendered_j = _jaccard(b.get("rendered_sentence_ids", []), v.get("rendered_sentence_ids", []))
        cls = _classify_pair(b, v)

        new_sel = sorted(set(v.get("selected_sentence_ids", [])) - set(b.get("selected_sentence_ids", [])))
        dropped_sel = sorted(set(b.get("selected_sentence_ids", [])) - set(v.get("selected_sentence_ids", [])))

        out.append(
            {
                "dataset": dataset,
                "sample_id": sid,
                "question": _safe_text(b.get("question", v.get("question", ""))),
                "baseline_prediction": _safe_text(b.get("prediction", "")),
                "chain_prediction": _safe_text(v.get("prediction", "")),
                "baseline_em": _safe_float(b.get("em", 0.0), 0.0),
                "chain_em": _safe_float(v.get("em", 0.0), 0.0),
                "baseline_f1": _safe_float(b.get("f1", 0.0), 0.0),
                "chain_f1": _safe_float(v.get("f1", 0.0), 0.0),
                "baseline_sf_recall": _safe_float(b.get("supporting_fact_recall", 0.0), 0.0),
                "chain_sf_recall": _safe_float(v.get("supporting_fact_recall", 0.0), 0.0),
                "baseline_sf_precision": _safe_float(b.get("supporting_fact_precision", 0.0), 0.0),
                "chain_sf_precision": _safe_float(v.get("supporting_fact_precision", 0.0), 0.0),
                "baseline_context_tokens": _safe_float(b.get("prompt_tokens", 0.0), 0.0),
                "chain_context_tokens": _safe_float(v.get("prompt_tokens", 0.0), 0.0),
                "baseline_retrieval_ms": _safe_float((b.get("latency", {}) or {}).get("retrieval_ms", 0.0), 0.0),
                "chain_retrieval_ms": _safe_float((v.get("latency", {}) or {}).get("retrieval_ms", 0.0), 0.0),
                "baseline_selected_sentence_ids": list(b.get("selected_sentence_ids", []) or []),
                "chain_selected_sentence_ids": list(v.get("selected_sentence_ids", []) or []),
                "baseline_rendered_sentence_ids": list(b.get("rendered_sentence_ids", []) or []),
                "chain_rendered_sentence_ids": list(v.get("rendered_sentence_ids", []) or []),
                "selected_sentence_jaccard": selected_j,
                "rendered_sentence_jaccard": rendered_j,
                "chain_gain_total": _safe_float(v.get("chain_gain_total", 0.0), 0.0),
                "role_aware_redundancy_applied_count": _safe_float(v.get("role_aware_redundancy_applied_count", 0.0), 0.0),
                "classification": cls,
                "newly_selected_sentences_by_chain_aware": new_sel,
                "dropped_baseline_sentences": dropped_sel,
                "chain_gain_activated_sentences": list(v.get("chain_gain_activated_sentences", []) or []),
                "possible_noise_indicator": bool(
                    cls == "regressed"
                    and len(new_sel) > 0
                    and _safe_float(v.get("supporting_fact_recall", 0.0), 0.0)
                    < _safe_float(b.get("supporting_fact_recall", 0.0), 0.0)
                ),
                "context_hash_changed": _safe_text(b.get("context_hash", "")) != _safe_text(v.get("context_hash", "")),
                "prompt_hash_changed": _safe_text(b.get("prompt_hash", "")) != _safe_text(v.get("prompt_hash", "")),
            }
        )

    return out, pairing


def _write_examples(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    lines: List[str] = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for i, r in enumerate(rows, start=1):
        lines.append(f"## {i}. sample_id={_safe_text(r.get('sample_id'))}")
        lines.append(f"- question: {_safe_text(r.get('question'))}")
        lines.append(f"- baseline_answer: {_safe_text(r.get('baseline_prediction'))}")
        lines.append(f"- chain_aware_answer: {_safe_text(r.get('chain_prediction'))}")
        lines.append(f"- baseline_f1 / chain_f1: {_fmt(r.get('baseline_f1'))} / {_fmt(r.get('chain_f1'))}")
        lines.append(
            f"- baseline_SF_recall / chain_SF_recall: {_fmt(r.get('baseline_sf_recall'))} / {_fmt(r.get('chain_sf_recall'))}"
        )
        lines.append(f"- newly_selected_sentences_by_chain_aware: {len(list(r.get('newly_selected_sentences_by_chain_aware', []) or []))}")
        lines.append(f"- dropped_baseline_sentences: {len(list(r.get('dropped_baseline_sentences', []) or []))}")
        lines.append(f"- chain_gain_activated_sentences: {len(list(r.get('chain_gain_activated_sentences', []) or []))}")
        lines.append(f"- possible_noise_indicator: {bool(r.get('possible_noise_indicator', False))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_main_summary(
    out_root: Path,
    artifact_report: Mapping[str, Any],
    rows_main: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    latency_rows: Sequence[Mapping[str, Any]],
    chain_activation_rows: Sequence[Mapping[str, Any]],
    diagnostics_validity: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    decision: str,
) -> Tuple[str, Dict[str, Any]]:
    counts = dict(artifact_report.get("counts", {}) or {})

    lines: List[str] = []
    lines.append("# PHASE6U Chain-aware ABR Paired Diagnostic Summary")
    lines.append("")
    lines.append("## 1. Goal")
    lines.append("Paired smoke50 baseline vs chain-aware on identical datasets/sample-size; diagnostic-only run.")
    lines.append("")
    lines.append("## 2. Artifact Validation")
    lines.append("")
    lines.append(
        f"- scheduled_runs={int(counts.get('scheduled_runs', 0))}, completed_run_names={int(counts.get('completed_run_names', 0))}, "
        f"incomplete_attempts={int(counts.get('incomplete_attempts', 0))}, extra_not_in_manifest={int(counts.get('extra_not_in_manifest', 0))}, "
        f"multi_attempt_run_names={int(counts.get('multi_attempt_run_names', 0))}, parse_errors={int(counts.get('parse_errors', 0))}"
    )
    lines.append("")

    lines.append("## 3. Paired Main Results")
    lines.append("")
    lines.append("| dataset | profile | role | Recall@5 | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | supporting_fact_precision | supporting_fact_recall | supporting_fact_f1 | retrieval_ms | generation_ms | total_ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_main:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(r.get("dataset")),
                _safe_text(r.get("profile")),
                _safe_text(r.get("role")),
                _fmt(r.get("Recall@5"), 4),
                _fmt(r.get("EM"), 4),
                _fmt(r.get("F1"), 4),
                _fmt(r.get("avg_context_tokens"), 3),
                _fmt(r.get("F1_per_1k_context_tokens"), 4),
                _fmt(r.get("supporting_fact_precision"), 4),
                _fmt(r.get("supporting_fact_recall"), 4),
                _fmt(r.get("supporting_fact_f1"), 4),
                _fmt(r.get("retrieval_ms"), 2),
                _fmt(r.get("generation_ms"), 2),
                _fmt(r.get("total_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 4. Delta by Dataset (chain-aware - baseline)")
    lines.append("")
    lines.append("| dataset | ΔEM | ΔF1 | ΔSF_recall | ΔSF_precision | Δavg_context_tokens | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in delta_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(d.get("dataset")),
                _fmt(d.get("dEM"), 4),
                _fmt(d.get("dF1"), 4),
                _fmt(d.get("dSF_recall"), 4),
                _fmt(d.get("dSF_precision"), 4),
                _fmt(d.get("davg_context_tokens"), 3),
                _fmt(d.get("dretrieval_ms"), 2),
                _fmt(d.get("dtotal_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 5. ABR-stage Diagnostics Validity")
    lines.append("")
    lines.append("| dataset | profile | role | sentence_candidate_SF_recall | evidence_atom_SF_recall | ABR_selected_SF_recall | rendered_SF_recall | final_supporting_fact_recall | mismatch_flag |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in diagnostics_validity:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(r.get("dataset")),
                _safe_text(r.get("profile")),
                _safe_text(r.get("role")),
                _fmt(r.get("sentence_candidate_SF_recall"), 4),
                _fmt(r.get("evidence_atom_SF_recall"), 4),
                _fmt(r.get("ABR_selected_SF_recall"), 4),
                _fmt(r.get("rendered_SF_recall"), 4),
                _fmt(r.get("final_supporting_fact_recall"), 4),
                _safe_text(r.get("stagewise_final_sf_mismatch")),
            )
        )
    lines.append("")

    lines.append("## 6. Latency Decomposition")
    lines.append("")
    lines.append("| dataset | profile | role | retrieval_ms_avg | proposal_total_ms_avg | phase1_run_scoring_ms_avg | phase2_refine_ms_avg | sentence_rerank_ms_avg | unified_acr_rcedr_ms_avg | render_ms_avg | total_ms_avg | objective_eval_calls_avg | num_atoms_avg | num_selected_atoms_avg |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in latency_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(r.get("dataset")),
                _safe_text(r.get("profile")),
                _safe_text(r.get("role")),
                _fmt(r.get("retrieval_ms_avg"), 2),
                _fmt(r.get("proposal_total_ms_avg"), 2),
                _fmt(r.get("phase1_run_scoring_ms_avg"), 2),
                _fmt(r.get("phase2_refine_ms_avg"), 2),
                _fmt(r.get("sentence_rerank_ms_avg"), 2),
                _fmt(r.get("unified_acr_rcedr_ms_avg"), 2),
                _fmt(r.get("render_ms_avg"), 2),
                _fmt(r.get("total_ms_avg"), 2),
                _fmt(r.get("objective_eval_calls_avg"), 2),
                _fmt(r.get("num_atoms_avg"), 2),
                _fmt(r.get("num_selected_atoms_avg"), 2),
            )
        )
    lines.append("")

    lines.append("## 7. Chain-aware Activation Analysis")
    lines.append("")
    lines.append("| dataset | chain_gain_total_avg | chain_gain_activated_count_avg | chain_gain_rejected_count_avg | role_aware_redundancy_applied_avg |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in chain_activation_rows:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                _safe_text(r.get("dataset")),
                _fmt(r.get("chain_gain_total_avg"), 4),
                _fmt(r.get("chain_gain_activated_count_avg"), 4),
                _fmt(r.get("chain_gain_rejected_count_avg"), 4),
                _fmt(r.get("role_aware_redundancy_applied_avg"), 4),
            )
        )
    lines.append("")

    lines.append("## 8. HotpotQA Improvement Examples")
    lines.append("See `hotpotqa_improvement_examples_top20.md`.")
    lines.append("")
    lines.append("## 9. 2Wiki Regression Examples")
    lines.append("See `2wiki_regression_examples_top20.md`.")
    lines.append("")

    # paired outcome counts
    by_cls: Dict[str, int] = defaultdict(int)
    for r in paired_rows:
        by_cls[_safe_text(r.get("classification"))] += 1

    lines.append("## 10. Interpretation")
    lines.append(f"- improved: {int(by_cls.get('improved', 0))}, regressed: {int(by_cls.get('regressed', 0))}, mixed: {int(by_cls.get('mixed', 0))}, unchanged: {int(by_cls.get('unchanged', 0))}")
    lines.append("- If ABR stage values are `n/a`, diagnostics were unavailable and not coerced to zero.")
    lines.append("")

    lines.append("## 11. Decision")
    lines.append(f"- decision: `{decision}`")
    lines.append("- Note: diagnostic-only smoke50; not an n=1000 acceptance run.")
    lines.append("")

    payload = {
        "phase": "phase6u_chain_aware_paired_diagnostic",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "artifact_counts": dict(counts),
        "rows_main": list(rows_main),
        "delta_rows": list(delta_rows),
        "latency_rows": list(latency_rows),
        "chain_activation_rows": list(chain_activation_rows),
        "diagnostics_validity": list(diagnostics_validity),
        "paired_counts": {k: int(v) for k, v in by_cls.items()},
        "decision": decision,
    }

    return "\n".join(lines) + "\n", payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PHASE6U paired chain-aware diagnostic")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--baseline-profile", default="unified_acr_rcedr_v12")
    ap.add_argument("--variant-profile", default="unified_acr_rcedr_v12_chain_aware")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    datasets = [_safe_text(x) for x in args.datasets.split(",") if _safe_text(x)]

    artifact_report = collect_phase6s_artifacts(out_root)
    run_list = _load_completed_runs(out_root)

    # Index runs by dataset/profile/role.
    by_key: Dict[Tuple[str, str], RunData] = {}
    for run in run_list:
        if run.dataset not in datasets:
            continue
        key = (run.dataset, run.profile)
        by_key[key] = run

    rows_main: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []
    chain_activation_rows: List[Dict[str, Any]] = []
    diagnostics_validity_rows: List[Dict[str, Any]] = []
    paired_rows_all: List[Dict[str, Any]] = []

    sampled_ids_root = out_root / "sampled_query_ids"
    sampled_ids_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        b = by_key.get((dataset, args.baseline_profile))
        v = by_key.get((dataset, args.variant_profile))
        if b is None or v is None:
            continue

        b_stage = _stagewise_recall_for_run(b)
        v_stage = _stagewise_recall_for_run(v)

        # Per-query normalized rows
        b_norm = [_normalize_query_row(b, qr) for qr in b.query_rows]
        v_norm = [_normalize_query_row(v, qr) for qr in v.query_rows]

        # sample id pairing evidence
        b_ids = [str(r.get("sample_id")) for r in b_norm]
        v_ids = [str(r.get("sample_id")) for r in v_norm]
        pairing = pair_sample_ids(b_ids, v_ids)
        _write_json(sampled_ids_root / f"{dataset}.json", {
            "dataset": dataset,
            "baseline_run": b.run_name,
            "variant_run": v.run_name,
            "baseline_profile": b.profile,
            "variant_profile": v.profile,
            "baseline_ids": b_ids,
            "variant_ids": v_ids,
            "pairing": pairing,
        })

        pairs, _ = _pair_rows(dataset, b_norm, v_norm)
        paired_rows_all.extend(pairs)

        b_metrics = _summary_metrics(b.summary)
        v_metrics = _summary_metrics(v.summary)

        rows_main.append({"dataset": dataset, "profile": b.profile, "role": "baseline", **b_metrics})
        rows_main.append({"dataset": dataset, "profile": v.profile, "role": "variant", **v_metrics})

        delta_rows.append(
            {
                "dataset": dataset,
                "dEM": v_metrics["EM"] - b_metrics["EM"],
                "dF1": v_metrics["F1"] - b_metrics["F1"],
                "dSF_recall": v_metrics["supporting_fact_recall"] - b_metrics["supporting_fact_recall"],
                "dSF_precision": v_metrics["supporting_fact_precision"] - b_metrics["supporting_fact_precision"],
                "davg_context_tokens": v_metrics["avg_context_tokens"] - b_metrics["avg_context_tokens"],
                "dretrieval_ms": v_metrics["retrieval_ms"] - b_metrics["retrieval_ms"],
                "dtotal_ms": v_metrics["total_ms"] - b_metrics["total_ms"],
            }
        )

        latency_rows.extend(_build_latency_rows(dataset, b.profile, "baseline", b_norm))
        latency_rows.extend(_build_latency_rows(dataset, v.profile, "variant", v_norm))

        # Diagnostics validity table
        for run, stage, role, m in [(b, b_stage, "baseline", b_metrics), (v, v_stage, "variant", v_metrics)]:
            svals = dict(stage.get("by_stage", {}) or {})

            def _pick(name: str) -> Optional[float]:
                item = dict(svals.get(name, {}) or {})
                if not bool(item.get("available", False)):
                    return None
                return _safe_float(item.get("value", 0.0), 0.0)

            rendered_stage = _pick("rendered_SF_recall")
            mismatch = False
            if rendered_stage is not None:
                mismatch = abs(rendered_stage - _safe_float(m.get("supporting_fact_recall", 0.0), 0.0)) > 0.15

            diagnostics_validity_rows.append(
                {
                    "dataset": dataset,
                    "profile": run.profile,
                    "role": role,
                    "sentence_candidate_SF_recall": _pick("sentence_candidate_SF_recall"),
                    "evidence_atom_SF_recall": _pick("evidence_atom_SF_recall"),
                    "ABR_selected_SF_recall": _pick("ABR_selected_SF_recall"),
                    "rendered_SF_recall": rendered_stage,
                    "final_supporting_fact_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
                    "stagewise_final_sf_mismatch": bool(mismatch),
                    "diagnostic_available": bool(stage.get("available", False)),
                }
            )

        # chain activation stats (variant only)
        chain_totals = []
        chain_acts = []
        chain_rejects = []
        relax_vals = []
        for r in v_norm:
            chain_totals.append(_safe_float(r.get("chain_gain_total", 0.0), 0.0))
            acts = len(list(r.get("chain_gain_activated_sentences", []) or []))
            chain_acts.append(float(acts))
            lat = dict(r.get("latency", {}) or {})
            nsel = _safe_int(lat.get("num_selected_atoms", 0), 0)
            chain_rejects.append(float(max(0, nsel - acts)))
            relax_vals.append(_safe_float(r.get("role_aware_redundancy_applied_count", 0.0), 0.0))

        chain_activation_rows.append(
            {
                "dataset": dataset,
                "profile": v.profile,
                "chain_gain_total_avg": _mean(chain_totals),
                "chain_gain_activated_count_avg": _mean(chain_acts),
                "chain_gain_rejected_count_avg": _mean(chain_rejects),
                "role_aware_redundancy_applied_avg": _mean(relax_vals),
                "selected_role_counts_avg": None,
                "candidate_role_counts_avg": None,
            }
        )

    # sort outputs
    rows_main.sort(key=lambda x: (x.get("dataset", ""), x.get("role", "")))
    delta_rows.sort(key=lambda x: x.get("dataset", ""))
    latency_rows.sort(key=lambda x: (x.get("dataset", ""), x.get("role", "")))
    chain_activation_rows.sort(key=lambda x: x.get("dataset", ""))
    diagnostics_validity_rows.sort(key=lambda x: (x.get("dataset", ""), x.get("role", "")))
    paired_rows_all.sort(key=lambda x: (x.get("dataset", ""), x.get("sample_id", "")))

    # write paired query comparison
    _write_jsonl(out_root / "paired_query_comparison.jsonl", paired_rows_all)

    # write latency decomposition csv
    if latency_rows:
        fieldnames = ["dataset", "profile", "role"] + [f"{k}_{sfx}" for k in LAT_KEYS for sfx in ("avg", "p95")]
        _write_csv(out_root / "latency_decomposition.csv", latency_rows, fieldnames)

    # chain activation csv
    if chain_activation_rows:
        _write_csv(
            out_root / "chain_activation_stats.csv",
            chain_activation_rows,
            [
                "dataset",
                "profile",
                "chain_gain_total_avg",
                "chain_gain_activated_count_avg",
                "chain_gain_rejected_count_avg",
                "role_aware_redundancy_applied_avg",
                "selected_role_counts_avg",
                "candidate_role_counts_avg",
            ],
        )

    # diagnostics validity report
    diag_md = ["# Diagnostics Validity Report", ""]
    for r in diagnostics_validity_rows:
        diag_md.append(
            "- dataset={} role={} profile={}: sentence={} atoms={} abr={} rendered={} final_sf={} mismatch={}".format(
                _safe_text(r.get("dataset")),
                _safe_text(r.get("role")),
                _safe_text(r.get("profile")),
                _fmt(r.get("sentence_candidate_SF_recall"), 4),
                _fmt(r.get("evidence_atom_SF_recall"), 4),
                _fmt(r.get("ABR_selected_SF_recall"), 4),
                _fmt(r.get("rendered_SF_recall"), 4),
                _fmt(r.get("final_supporting_fact_recall"), 4),
                _safe_text(r.get("stagewise_final_sf_mismatch")),
            )
        )
    (out_root / "diagnostics_validity_report.md").write_text("\n".join(diag_md) + "\n", encoding="utf-8")

    # example files
    two_wiki_reg = [r for r in paired_rows_all if _safe_text(r.get("dataset")) == "2wikimultihopqa" and _safe_text(r.get("classification")) == "regressed"]
    two_wiki_reg = sorted(two_wiki_reg, key=lambda x: (_safe_float(x.get("chain_f1", 0.0), 0.0) - _safe_float(x.get("baseline_f1", 0.0), 0.0)))[: args.top_k]
    _write_examples(out_root / "2wiki_regression_examples_top20.md", "2Wiki Regression Examples", two_wiki_reg)

    hotpot_imp = [r for r in paired_rows_all if _safe_text(r.get("dataset")) == "hotpotqa" and _safe_text(r.get("classification")) == "improved"]
    hotpot_imp = sorted(hotpot_imp, key=lambda x: (_safe_float(x.get("chain_f1", 0.0), 0.0) - _safe_float(x.get("baseline_f1", 0.0), 0.0)), reverse=True)[: args.top_k]
    _write_examples(out_root / "hotpotqa_improvement_examples_top20.md", "HotpotQA Improvement Examples", hotpot_imp)

    # decision
    diag_invalid = any(
        (r.get("rendered_SF_recall") is None) or bool(r.get("stagewise_final_sf_mismatch", False))
        for r in diagnostics_validity_rows
    )
    by_ds_delta = {d["dataset"]: d for d in delta_rows}
    hot = by_ds_delta.get("hotpotqa")
    two = by_ds_delta.get("2wikimultihopqa")
    if diag_invalid:
        decision = "reject"
    elif hot and two and _safe_float(hot.get("dF1", 0.0), 0.0) >= 0 and _safe_float(two.get("dF1", 0.0), 0.0) >= 0:
        decision = "diagnostic_accept"
    elif hot and two and (_safe_float(hot.get("dF1", 0.0), 0.0) > 0 and _safe_float(two.get("dF1", 0.0), 0.0) < 0):
        decision = "hold"
    else:
        decision = "reject"

    md_text, summary_payload = _build_main_summary(
        out_root=out_root,
        artifact_report=artifact_report,
        rows_main=rows_main,
        delta_rows=delta_rows,
        latency_rows=latency_rows,
        chain_activation_rows=chain_activation_rows,
        diagnostics_validity=diagnostics_validity_rows,
        paired_rows=paired_rows_all,
        decision=decision,
    )

    (out_root / "PHASE6U_CHAIN_AWARE_PAIRED_DIAGNOSTIC_SUMMARY.md").write_text(md_text, encoding="utf-8")
    _write_json(out_root / "phase6u_chain_aware_paired_diagnostic_summary.json", summary_payload)

    print(out_root / "PHASE6U_CHAIN_AWARE_PAIRED_DIAGNOSTIC_SUMMARY.md")
    print(out_root / "phase6u_chain_aware_paired_diagnostic_summary.json")
    print(out_root / "paired_query_comparison.jsonl")
    print(out_root / "latency_decomposition.csv")
    print(out_root / "chain_activation_stats.csv")
    print(out_root / "diagnostics_validity_report.md")


if __name__ == "__main__":
    main()
