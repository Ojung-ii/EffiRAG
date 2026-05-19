#!/usr/bin/env python3
"""Summarize PHASE6U-3 ABR objective redesign paired diagnostic."""

from __future__ import annotations

import argparse
import csv
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
        for row in rows:
            w.writerow(dict(row))


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


def _supporting_fact_f1(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def normalize_sentence_support_id(sentence_id: str) -> str:
    sid = _safe_text(sentence_id)
    if not sid:
        return ""
    parts = sid.split("::")
    if len(parts) >= 3 and parts[0] in {"chunk", "sentence"}:
        return f"{parts[1]}::{parts[2]}"
    return sid


def _extract_ids(row: Mapping[str, Any], *, selected: bool) -> List[str]:
    if selected:
        ids = list((((row.get("retrieval") or {}).get("selected_sentence_ids")) or []))
        return [normalize_sentence_support_id(x) for x in ids]
    ids = list(row.get("rendered_sentence_ids") or [])
    return [normalize_sentence_support_id(x) for x in ids]


def _extract_text(row: Mapping[str, Any]) -> str:
    rendered = dict(row.get("rendered", {}) or {})
    return _safe_text(rendered.get("text", ""))


def _extract_query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    metrics = dict(row.get("metrics", {}) or {})
    eff = dict(row.get("efficiency", {}) or {})
    return {
        "em": _safe_float(metrics.get("em", 0.0), 0.0),
        "f1": _safe_float(metrics.get("f1", 0.0), 0.0),
        "sf_recall": _safe_float(metrics.get("supporting_fact_recall", 0.0), 0.0),
        "sf_precision": _safe_float(metrics.get("supporting_fact_precision", 0.0), 0.0),
        "retrieval_ms": _safe_float(eff.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(eff.get("total_latency_ms", 0.0), 0.0),
    }


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
    runs: List[RunData] = []
    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        meta = scheduled.get(run_name, {})
        dataset = _safe_text(meta.get("dataset", item.get("dataset", "")))
        profile = _safe_text(meta.get("profile", item.get("profile", "")))
        role = _safe_text(meta.get("role", ""))
        summary_path = Path(_safe_text(item.get("summary_path"))).resolve()
        query_path = Path(_safe_text(item.get("query_path"))).resolve()
        summary = dict(item.get("summary", {}) or {})
        query_rows = _read_jsonl(query_path)
        runs.append(
            RunData(
                run_name=run_name,
                dataset=dataset,
                profile=profile,
                role=role,
                summary_path=summary_path,
                query_path=query_path,
                summary=summary,
                query_rows=query_rows,
            )
        )
    return runs


def _stagewise_recall_for_run(run: RunData) -> Dict[str, Any]:
    sample_map = load_samples_for_dataset(run.dataset)
    values: Dict[str, List[float]] = {k: [] for k in ABR_STAGE_KEYS}
    available_rows = 0

    for row in run.query_rows:
        sample_id = _safe_text(row.get("sample_id"))
        sample = sample_map.get(sample_id)
        if sample is None:
            continue
        available_rows += 1
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

    out: Dict[str, Any] = {
        "available": bool(available_rows > 0),
        "available_rows": int(available_rows),
    }
    for stage in ABR_STAGE_KEYS:
        metric_name = STAGE_TO_METRIC_NAME[stage]
        out[metric_name] = (_mean(values[stage]) if values[stage] else None)
    return out


def _run_metrics(run: RunData) -> Dict[str, Any]:
    s = dict(run.summary or {})
    em = _safe_float(s.get("em", 0.0), 0.0)
    f1 = _safe_float(s.get("f1", 0.0), 0.0)
    recall5 = _safe_float(s.get("recall_at_5", s.get("supporting_fact_recall_at_5", 0.0)), 0.0)
    sf_precision = _safe_float(s.get("supporting_fact_precision", 0.0), 0.0)
    sf_recall = _safe_float(s.get("supporting_fact_recall", 0.0), 0.0)
    sf_f1 = _safe_float(s.get("supporting_fact_f1", _supporting_fact_f1(sf_precision, sf_recall)), 0.0)
    avg_context_tokens = _safe_float(s.get("avg_context_tokens", s.get("prompt_tokens_avg", 0.0)), 0.0)
    f1_per_1k = float(f1 / (avg_context_tokens / 1000.0)) if avg_context_tokens > 0 else 0.0
    retrieval_ms = _safe_float(s.get("retrieval_latency_ms", 0.0), 0.0)
    generation_ms = _safe_float(s.get("generation_latency_ms", s.get("generation_ms", 0.0)), 0.0)
    total_ms = _safe_float(s.get("total_latency_ms", 0.0), 0.0)

    st = _stagewise_recall_for_run(run)

    unified_diag_rows = []
    for row in run.query_rows:
        diag = dict((((row.get("retrieval") or {}).get("diagnostics") or {}).get("unified_acr_rcedr_diag", {}) or {}))
        if diag:
            unified_diag_rows.append(diag)

    def _avg_diag(key: str) -> Optional[float]:
        vals: List[float] = []
        for d in unified_diag_rows:
            if key in d and d.get(key) is not None:
                vals.append(_safe_float(d.get(key), 0.0))
        if not vals:
            return None
        return _mean(vals)

    return {
        "dataset": run.dataset,
        "profile": run.profile,
        "role": run.role,
        "run_name": run.run_name,
        "Recall@5": recall5,
        "EM": em,
        "F1": f1,
        "avg_context_tokens": avg_context_tokens,
        "F1_per_1k_context_tokens": f1_per_1k,
        "supporting_fact_precision": sf_precision,
        "supporting_fact_recall": sf_recall,
        "supporting_fact_f1": sf_f1,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
        "sentence_candidate_SF_recall": st.get("sentence_candidate_SF_recall"),
        "evidence_atom_SF_recall": st.get("evidence_atom_SF_recall"),
        "ABR_selected_SF_recall": st.get("ABR_selected_SF_recall"),
        "rendered_SF_recall": st.get("rendered_SF_recall"),
        "selected_atoms_avg": _avg_diag("num_selected_atoms"),
        "selected_tokens_avg": _avg_diag("selected_tokens"),
        "objective_eval_calls_avg": _avg_diag("objective_eval_calls"),
        "unified_acr_rcedr_ms": _avg_diag("selection_ms"),
        "role_coverage_gain_total_avg": _avg_diag("role_coverage_gain_total_avg"),
        "role_balance_activated_avg": _avg_diag("role_balance_activated_count"),
        "role_balance_rejected_avg": _avg_diag("role_balance_rejected_count"),
        "redundancy_relaxed_count_avg": _avg_diag("redundancy_relaxed_count"),
        "redundancy_before_avg": _avg_diag("redundancy_before_avg"),
        "redundancy_after_avg": _avg_diag("redundancy_after_avg"),
        "redundancy_recalibration_applied_avg": _avg_diag("redundancy_recalibration_applied_avg"),
    }


def _pair_query_rows(
    *,
    dataset: str,
    baseline: RunData,
    variant: RunData,
    variant_label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_map = {_safe_text(r.get("sample_id")): r for r in baseline.query_rows}
    var_map = {_safe_text(r.get("sample_id")): r for r in variant.query_rows}
    ordered_ids = [_safe_text(r.get("sample_id")) for r in baseline.query_rows if _safe_text(r.get("sample_id"))]
    rows: List[Dict[str, Any]] = []

    improved = 0
    regressed = 0
    mixed = 0
    unchanged = 0

    for sid in ordered_ids:
        b = base_map.get(sid)
        v = var_map.get(sid)
        if b is None or v is None:
            continue

        bm = _extract_query_metrics(b)
        vm = _extract_query_metrics(v)
        b_sel = _extract_ids(b, selected=True)
        v_sel = _extract_ids(v, selected=True)
        b_rnd = _extract_ids(b, selected=False)
        v_rnd = _extract_ids(v, selected=False)

        selected_j = _jaccard(b_sel, v_sel)
        rendered_j = _jaccard(b_rnd, v_rnd)
        delta_f1 = float(vm["f1"] - bm["f1"])
        delta_sf = float(vm["sf_recall"] - bm["sf_recall"])

        if delta_f1 > 0.05 or delta_sf > 0.10:
            outcome = "improved"
            improved += 1
        elif delta_f1 < -0.05 or delta_sf < -0.10:
            outcome = "regressed"
            regressed += 1
        elif abs(delta_f1) <= 1e-12 and abs(delta_sf) <= 1e-12:
            outcome = "unchanged"
            unchanged += 1
        else:
            outcome = "mixed"
            mixed += 1

        unified_diag = dict((((v.get("retrieval") or {}).get("diagnostics") or {}).get("unified_acr_rcedr_diag", {}) or {}))
        rows.append(
            {
                "dataset": dataset,
                "variant_profile": variant.profile,
                "variant_label": variant_label,
                "sample_id": sid,
                "question": _safe_text(b.get("question", "")),
                "baseline_prediction": _safe_text(b.get("prediction", "")),
                "variant_prediction": _safe_text(v.get("prediction", "")),
                "baseline_em": bm["em"],
                "variant_em": vm["em"],
                "baseline_f1": bm["f1"],
                "variant_f1": vm["f1"],
                "delta_f1": delta_f1,
                "baseline_sf_recall": bm["sf_recall"],
                "variant_sf_recall": vm["sf_recall"],
                "delta_sf_recall": delta_sf,
                "baseline_sf_precision": bm["sf_precision"],
                "variant_sf_precision": vm["sf_precision"],
                "baseline_retrieval_ms": bm["retrieval_ms"],
                "variant_retrieval_ms": vm["retrieval_ms"],
                "baseline_total_ms": bm["total_ms"],
                "variant_total_ms": vm["total_ms"],
                "baseline_selected_sentence_ids": b_sel,
                "variant_selected_sentence_ids": v_sel,
                "baseline_rendered_sentence_ids": b_rnd,
                "variant_rendered_sentence_ids": v_rnd,
                "selected_sentence_jaccard": selected_j,
                "rendered_sentence_jaccard": rendered_j,
                "prediction_changed": int(_safe_text(b.get("prediction")) != _safe_text(v.get("prediction"))),
                "rendered_text_changed": int(_extract_text(b) != _extract_text(v)),
                "outcome": outcome,
                "role_coverage_gain_total": _safe_float(unified_diag.get("role_coverage_gain_total_avg", 0.0), 0.0),
                "redundancy_recalibration_applied": _safe_float(
                    unified_diag.get("redundancy_recalibration_applied_avg", 0.0), 0.0
                ),
            }
        )

    stats = {
        "dataset": dataset,
        "variant_profile": variant.profile,
        "improved": int(improved),
        "regressed": int(regressed),
        "mixed": int(mixed),
        "unchanged": int(unchanged),
        "n": int(improved + regressed + mixed + unchanged),
    }
    return rows, stats


def _variant_decision(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "reject"

    by_dataset: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "delta_f1": 0.0,
        "delta_sf_recall": 0.0,
        "delta_tokens": 0.0,
        "delta_retrieval": 0.0,
        "count": 0.0,
    })

    for row in rows:
        ds = _safe_text(row.get("dataset"))
        by_dataset[ds]["delta_f1"] += _safe_float(row.get("delta_f1", 0.0), 0.0)
        by_dataset[ds]["delta_sf_recall"] += _safe_float(row.get("delta_sf_recall", 0.0), 0.0)
        by_dataset[ds]["count"] += 1.0

    # Conservative aggregate decision using query deltas and dataset guards.
    avg_delta_f1 = _mean([_safe_float(r.get("delta_f1", 0.0), 0.0) for r in rows])
    avg_delta_sf = _mean([_safe_float(r.get("delta_sf_recall", 0.0), 0.0) for r in rows])

    # Hard reject: notable quality drop.
    for ds in by_dataset:
        c = max(1.0, by_dataset[ds]["count"])
        ds_f1 = by_dataset[ds]["delta_f1"] / c
        if ds_f1 < -0.02:
            return "reject"

    if avg_delta_f1 >= 0.02 and avg_delta_sf >= 0.04:
        return "strong_accept"
    if avg_delta_f1 >= -0.005 and avg_delta_sf >= 0.0:
        return "hold"
    return "reject"


def build_summary(
    *,
    out_root: Path,
    datasets: Sequence[str],
    baseline_profile: str,
    variant_profiles: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    runs = _load_completed_runs(out_root)
    run_index: Dict[Tuple[str, str], RunData] = {}
    for run in runs:
        key = (_safe_text(run.dataset), _safe_text(run.profile))
        run_index[key] = run

    result_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    paired_stats: List[Dict[str, Any]] = []

    for dataset in datasets:
        bkey = (dataset, baseline_profile)
        if bkey not in run_index:
            continue
        base_run = run_index[bkey]
        base_metrics = _run_metrics(base_run)
        result_rows.append(base_metrics)

        for vprof in variant_profiles:
            vkey = (dataset, vprof)
            if vkey not in run_index:
                continue
            var_run = run_index[vkey]
            var_metrics = _run_metrics(var_run)
            result_rows.append(var_metrics)
            qrows, qstats = _pair_query_rows(
                dataset=dataset,
                baseline=base_run,
                variant=var_run,
                variant_label=vprof,
            )
            paired_rows.extend(qrows)
            paired_stats.append(qstats)

    # Aggregate deltas vs baseline per dataset/profile.
    deltas: List[Dict[str, Any]] = []
    for dataset in datasets:
        b = next((r for r in result_rows if r["dataset"] == dataset and r["profile"] == baseline_profile), None)
        if b is None:
            continue
        for vprof in variant_profiles:
            v = next((r for r in result_rows if r["dataset"] == dataset and r["profile"] == vprof), None)
            if v is None:
                continue
            deltas.append(
                {
                    "dataset": dataset,
                    "variant_profile": vprof,
                    "delta_EM": float(v["EM"] - b["EM"]),
                    "delta_F1": float(v["F1"] - b["F1"]),
                    "delta_SF_recall": float(v["supporting_fact_recall"] - b["supporting_fact_recall"]),
                    "delta_SF_precision": float(v["supporting_fact_precision"] - b["supporting_fact_precision"]),
                    "delta_avg_context_tokens": float(v["avg_context_tokens"] - b["avg_context_tokens"]),
                    "delta_retrieval_ms": float(v["retrieval_ms"] - b["retrieval_ms"]),
                    "delta_total_ms": float(v["total_ms"] - b["total_ms"]),
                }
            )

    # Variant decisions.
    variant_decisions: Dict[str, str] = {}
    for vprof in variant_profiles:
        rows = [r for r in paired_rows if _safe_text(r.get("variant_profile")) == vprof]
        variant_decisions[vprof] = _variant_decision(rows)

    # Build examples.
    role_rows = sorted(
        [r for r in paired_rows if _safe_text(r.get("variant_profile")) == "unified_acr_rcedr_v12_role_balanced" and _safe_text(r.get("outcome")) == "improved"],
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_sf_recall", 0.0), 0.0)),
        reverse=True,
    )[:top_k]
    red_rows = sorted(
        [r for r in paired_rows if _safe_text(r.get("variant_profile")) == "unified_acr_rcedr_v12_redundancy_recalibrated" and _safe_text(r.get("outcome")) == "improved"],
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_sf_recall", 0.0), 0.0)),
        reverse=True,
    )[:top_k]
    reg_rows = sorted(
        [r for r in paired_rows if _safe_text(r.get("outcome")) == "regressed"],
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_sf_recall", 0.0), 0.0)),
    )[:top_k]

    summary = {
        "phase": "phase6u_abr_objective_redesign",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root.resolve()),
        "datasets": list(datasets),
        "baseline_profile": baseline_profile,
        "variant_profiles": list(variant_profiles),
        "result_rows": result_rows,
        "deltas": deltas,
        "paired_stats": paired_stats,
        "variant_decisions": variant_decisions,
    }

    # Output files.
    _write_json(out_root / "phase6u_abr_objective_redesign_summary.json", summary)
    _write_jsonl(out_root / "paired_query_comparison.jsonl", paired_rows)

    diag_rows: List[Dict[str, Any]] = []
    for row in result_rows:
        if row.get("profile") == baseline_profile:
            continue
        diag_rows.append(
            {
                "dataset": row.get("dataset"),
                "profile": row.get("profile"),
                "role_coverage_gain_total_avg": row.get("role_coverage_gain_total_avg"),
                "role_balance_activated_avg": row.get("role_balance_activated_avg"),
                "role_balance_rejected_avg": row.get("role_balance_rejected_avg"),
                "redundancy_relaxed_count_avg": row.get("redundancy_relaxed_count_avg"),
                "redundancy_before_avg": row.get("redundancy_before_avg"),
                "redundancy_after_avg": row.get("redundancy_after_avg"),
                "redundancy_recalibration_applied_avg": row.get("redundancy_recalibration_applied_avg"),
                "sentence_candidate_SF_recall": row.get("sentence_candidate_SF_recall"),
                "ABR_selected_SF_recall": row.get("ABR_selected_SF_recall"),
                "rendered_SF_recall": row.get("rendered_SF_recall"),
            }
        )
    _write_csv(
        out_root / "abr_variant_diagnostics.csv",
        diag_rows,
        [
            "dataset",
            "profile",
            "role_coverage_gain_total_avg",
            "role_balance_activated_avg",
            "role_balance_rejected_avg",
            "redundancy_relaxed_count_avg",
            "redundancy_before_avg",
            "redundancy_after_avg",
            "redundancy_recalibration_applied_avg",
            "sentence_candidate_SF_recall",
            "ABR_selected_SF_recall",
            "rendered_SF_recall",
        ],
    )

    def _write_examples(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
        lines: List[str] = [f"# {title}", ""]
        if not rows:
            lines.append("No examples.")
        for i, row in enumerate(rows, 1):
            lines.append(f"## {i}. {row.get('dataset')} / {row.get('sample_id')}")
            lines.append(f"- question: {_safe_text(row.get('question'))}")
            lines.append(f"- baseline_prediction: {_safe_text(row.get('baseline_prediction'))}")
            lines.append(f"- variant_prediction: {_safe_text(row.get('variant_prediction'))}")
            lines.append(f"- delta_f1: {_fmt(row.get('delta_f1'))}")
            lines.append(f"- delta_sf_recall: {_fmt(row.get('delta_sf_recall'))}")
            lines.append(f"- selected_sentence_jaccard: {_fmt(row.get('selected_sentence_jaccard'))}")
            lines.append(f"- rendered_sentence_jaccard: {_fmt(row.get('rendered_sentence_jaccard'))}")
            lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_examples(
        out_root / "role_balanced_examples_top20.md",
        role_rows,
        "Role-balanced Variant Improved Examples (Top 20)",
    )
    _write_examples(
        out_root / "redundancy_recalibrated_examples_top20.md",
        red_rows,
        "Redundancy-recalibrated Variant Improved Examples (Top 20)",
    )
    _write_examples(
        out_root / "regression_examples_top20.md",
        reg_rows,
        "Regression Examples (Top 20)",
    )

    # Markdown summary.
    lines: List[str] = []
    lines.append("# PHASE6U ABR Objective Redesign Summary")
    lines.append("")
    lines.append("## 1. Goal")
    lines.append("Paired n=100 diagnostic for ABR objective variants without changing candidate generation or budgets.")
    lines.append("")
    lines.append("## 2. Artifact Validation")
    validation_path = out_root / "phase6u_artifact_validation.json"
    if validation_path.exists():
        try:
            vobj = json.loads(validation_path.read_text(encoding="utf-8"))
        except Exception:
            vobj = {}
        counts = dict(vobj.get("counts", {}) or {})
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
    else:
        lines.append("- validation file missing")
    lines.append("")

    lines.append("## 3. Paired Main Results")
    lines.append("| dataset | profile | role | Recall@5 | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | supporting_fact_precision | supporting_fact_recall | supporting_fact_f1 | retrieval_ms | generation_ms | total_ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(result_rows, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")))):
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
    lines.append("| dataset | variant_profile | ΔEM | ΔF1 | ΔSF_recall | ΔSF_precision | Δavg_context_tokens | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(deltas, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("variant_profile")))):
        lines.append(
            "| {dataset} | {profile} | {dem} | {df1} | {dsr} | {dsp} | {dctx} | {drm} | {dtm} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("variant_profile")),
                dem=_fmt(row.get("delta_EM"), 4),
                df1=_fmt(row.get("delta_F1"), 4),
                dsr=_fmt(row.get("delta_SF_recall"), 4),
                dsp=_fmt(row.get("delta_SF_precision"), 4),
                dctx=_fmt(row.get("delta_avg_context_tokens"), 3),
                drm=_fmt(row.get("delta_retrieval_ms"), 2),
                dtm=_fmt(row.get("delta_total_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 5. ABR-stage Diagnostics")
    lines.append("| dataset | profile | sentence_candidate_SF_recall | evidence_atom_SF_recall | ABR_selected_SF_recall | rendered_SF_recall | selected_atoms_avg | selected_tokens_avg | objective_eval_calls_avg | unified_acr_rcedr_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(result_rows, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {s1} | {s2} | {s3} | {s4} | {a} | {t} | {o} | {u} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("profile")),
                s1=_fmt(row.get("sentence_candidate_SF_recall"), 4),
                s2=_fmt(row.get("evidence_atom_SF_recall"), 4),
                s3=_fmt(row.get("ABR_selected_SF_recall"), 4),
                s4=_fmt(row.get("rendered_SF_recall"), 4),
                a=_fmt(row.get("selected_atoms_avg"), 2),
                t=_fmt(row.get("selected_tokens_avg"), 2),
                o=_fmt(row.get("objective_eval_calls_avg"), 2),
                u=_fmt(row.get("unified_acr_rcedr_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 6. Role-balanced Variant Analysis")
    lines.append(f"- decision: {variant_decisions.get('unified_acr_rcedr_v12_role_balanced', 'n/a')}")
    lines.append("- examples: role_balanced_examples_top20.md")
    lines.append("")

    lines.append("## 7. Redundancy-recalibrated Variant Analysis")
    lines.append(f"- decision: {variant_decisions.get('unified_acr_rcedr_v12_redundancy_recalibrated', 'n/a')}")
    lines.append("- examples: redundancy_recalibrated_examples_top20.md")
    lines.append("")

    lines.append("## 8. Token/Latency Guardrail")
    lines.append("See Delta vs Baseline table for token/latency changes.")
    lines.append("")

    lines.append("## 9. Regression Examples")
    lines.append("See regression_examples_top20.md")
    lines.append("")

    lines.append("## 10. Decision")
    rb = variant_decisions.get("unified_acr_rcedr_v12_role_balanced", "reject")
    rr = variant_decisions.get("unified_acr_rcedr_v12_redundancy_recalibrated", "reject")
    if rb == "strong_accept" and rr != "strong_accept":
        overall = "role_balanced"
    elif rr == "strong_accept" and rb != "strong_accept":
        overall = "redundancy_recalibrated"
    elif rb == "strong_accept" and rr == "strong_accept":
        overall = "both_strong_accept"
    elif rb == "hold" or rr == "hold":
        overall = "hold"
    else:
        overall = "reject"
    lines.append(f"- overall_decision: `{overall}`")

    (out_root / "PHASE6U_ABR_OBJECTIVE_REDESIGN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary["overall_decision"] = overall
    _write_json(out_root / "phase6u_abr_objective_redesign_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PHASE6U ABR objective redesign run.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--baseline-profile", default="unified_acr_rcedr_v12")
    ap.add_argument(
        "--variant-profiles",
        default="unified_acr_rcedr_v12_role_balanced,unified_acr_rcedr_v12_redundancy_recalibrated",
    )
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    variants = [x.strip() for x in str(args.variant_profiles).split(",") if x.strip()]

    build_summary(
        out_root=out_root,
        datasets=datasets,
        baseline_profile=str(args.baseline_profile).strip(),
        variant_profiles=variants,
        top_k=max(1, int(args.top_k)),
    )

    print(out_root / "PHASE6U_ABR_OBJECTIVE_REDESIGN_SUMMARY.md")
    print(out_root / "phase6u_abr_objective_redesign_summary.json")


if __name__ == "__main__":
    main()
