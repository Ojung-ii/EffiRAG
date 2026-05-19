#!/usr/bin/env python3
"""Summarize PHASE6V legacy-contract component ablation runs."""

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


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


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
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
        "answer_string_hit": _safe_float(summary.get("answer_surface_present", 0.0), 0.0),
        "answer_string_rank_avg": _safe_float(summary.get("answer_surface_position_avg", -1.0), -1.0),
        "answer_bearing_density": _safe_float(summary.get("answer_surface_token_density", 0.0), 0.0),
        "answer_context_early_hit": _safe_float(summary.get("answer_surface_early_hit", 0.0), 0.0),
    }


def _query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    m = dict(row.get("metrics", {}) or {})
    e = dict(row.get("efficiency", {}) or {})
    gdiag = dict(row.get("generation_diagnostics", {}) or {})
    return {
        "f1": _safe_float(m.get("f1", 0.0), 0.0),
        "em": _safe_float(m.get("em", 0.0), 0.0),
        "sf_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
        "sf_precision": _safe_float(m.get("supporting_fact_precision", 0.0), 0.0),
        "retrieval_ms": _safe_float(e.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(e.get("generation_latency_ms", 0.0), 0.0),
        "total_ms": _safe_float(e.get("total_latency_ms", 0.0), 0.0),
        "prompt_tokens": _safe_float(gdiag.get("prompt_tokens", 0.0), 0.0),
        "answer_string_hit": _safe_float(m.get("answer_surface_present", 0.0), 0.0),
        "answer_string_rank": _safe_float(m.get("answer_surface_position", -1.0), -1.0),
        "answer_bearing_density": _safe_float(m.get("answer_surface_token_density", 0.0), 0.0),
        "answer_context_early_hit": _safe_float(m.get("answer_surface_early_hit", 0.0), 0.0),
    }


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
        qpath = Path(_safe_text(item.get("query_path"))).resolve()
        qrows = _read_jsonl(qpath)
        runs.append(RunData(run_name, dataset, profile, role, summary, qrows))
    return runs


def _contract_diag_for_run(run: RunData) -> Dict[str, Any]:
    reorder_applied: List[float] = []
    reorder_changed: List[float] = []
    minimal_applied: List[float] = []
    num_extra: List[float] = []
    extra_tokens: List[float] = []
    budget_skip: List[float] = []
    token_delta: List[float] = []
    skip_reason_counts: Dict[str, int] = {}

    for row in run.query_rows:
        rendered = dict(row.get("rendered", {}) or {})
        meta = dict(rendered.get("metadata", {}) or {})
        reorder_applied.append(1.0 if bool(meta.get("top_slice_reorder_applied", False)) else 0.0)
        reorder_changed.append(1.0 if bool(meta.get("top_slice_reorder_changed_order", False)) else 0.0)
        minimal_applied.append(1.0 if bool(meta.get("minimal_package_applied", False)) else 0.0)

        n_extra = _safe_float(meta.get("num_extra_package_sentences", 0.0), 0.0)
        n_tok = _safe_float(meta.get("extra_sentence_token_count", 0.0), 0.0)
        n_skip = _safe_float(meta.get("budget_skip_count", 0.0), 0.0)
        before = _safe_float(meta.get("rendered_token_count_before_contract", 0.0), 0.0)
        after = _safe_float(meta.get("rendered_token_count_after_contract", 0.0), 0.0)
        num_extra.append(n_extra)
        extra_tokens.append(n_tok)
        budget_skip.append(n_skip)
        if before > 0 or after > 0:
            token_delta.append(after - before)

        skip_map = dict(meta.get("minimal_package_skip_reason_counts", {}) or {})
        for reason, cnt in skip_map.items():
            key = _safe_text(reason) or "unknown"
            skip_reason_counts[key] = int(skip_reason_counts.get(key, 0) + _safe_int(cnt, 0))

        if not skip_map:
            maybe_reason = _safe_text(meta.get("dominant_skip_reason", "")) or _safe_text(
                meta.get("minimal_package_dominant_skip_reason", "")
            )
            if maybe_reason:
                skip_reason_counts[maybe_reason] = int(skip_reason_counts.get(maybe_reason, 0) + 1)

    dominant_reason = ""
    if skip_reason_counts:
        dominant_reason = sorted(skip_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    return {
        "top_slice_reorder_rate": (_mean(reorder_applied) if reorder_applied else None),
        "top_slice_reorder_changed_order_rate": (_mean(reorder_changed) if reorder_changed else None),
        "minimal_package_apply_rate": (_mean(minimal_applied) if minimal_applied else None),
        "avg_extra_sentences": (_mean(num_extra) if num_extra else None),
        "avg_extra_tokens": (_mean(extra_tokens) if extra_tokens else None),
        "budget_skip_rate": (_mean([1.0 if x > 0 else 0.0 for x in budget_skip]) if budget_skip else None),
        "avg_context_token_delta": (_mean(token_delta) if token_delta else None),
        "dominant_skip_reason": dominant_reason or "n/a",
        "skip_reason_counts": dict(skip_reason_counts),
    }


def _pair_baseline_variant(
    dataset: str,
    baseline: RunData,
    variant: RunData,
    variant_label: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bmap = {_safe_text(r.get("sample_id")): r for r in baseline.query_rows}
    vmap = {_safe_text(r.get("sample_id")): r for r in variant.query_rows}
    ordered = [_safe_text(r.get("sample_id")) for r in baseline.query_rows if _safe_text(r.get("sample_id"))]
    paired: List[Dict[str, Any]] = []
    improved: List[Dict[str, Any]] = []
    regressed: List[Dict[str, Any]] = []

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

        b_ctx = _safe_text(dict(b.get("rendered", {}) or {}).get("text", ""))
        v_ctx = _safe_text(dict(v.get("rendered", {}) or {}).get("text", ""))

        delta_f1 = float(vm["f1"] - bm["f1"])
        delta_sf = float(vm["sf_recall"] - bm["sf_recall"])
        delta_ans = float(vm["answer_string_hit"] - bm["answer_string_hit"])
        delta_density = float(vm["answer_bearing_density"] - bm["answer_bearing_density"])

        row = {
            "dataset": dataset,
            "sample_id": sid,
            "variant_profile": variant_label,
            "question": _safe_text(b.get("question", "")),
            "baseline_prediction": _safe_text(b.get("prediction", "")),
            "variant_prediction": _safe_text(v.get("prediction", "")),
            "baseline_f1": bm["f1"],
            "variant_f1": vm["f1"],
            "delta_f1": delta_f1,
            "baseline_sf_recall": bm["sf_recall"],
            "variant_sf_recall": vm["sf_recall"],
            "delta_sf_recall": delta_sf,
            "baseline_sf_precision": bm["sf_precision"],
            "variant_sf_precision": vm["sf_precision"],
            "baseline_prompt_tokens": bm["prompt_tokens"],
            "variant_prompt_tokens": vm["prompt_tokens"],
            "baseline_retrieval_ms": bm["retrieval_ms"],
            "variant_retrieval_ms": vm["retrieval_ms"],
            "baseline_total_ms": bm["total_ms"],
            "variant_total_ms": vm["total_ms"],
            "baseline_answer_string_hit": bm["answer_string_hit"],
            "variant_answer_string_hit": vm["answer_string_hit"],
            "delta_answer_string_hit": delta_ans,
            "baseline_answer_string_rank": bm["answer_string_rank"],
            "variant_answer_string_rank": vm["answer_string_rank"],
            "baseline_answer_bearing_density": bm["answer_bearing_density"],
            "variant_answer_bearing_density": vm["answer_bearing_density"],
            "delta_answer_bearing_density": delta_density,
            "baseline_answer_context_early_hit": bm["answer_context_early_hit"],
            "variant_answer_context_early_hit": vm["answer_context_early_hit"],
            "selected_sentence_jaccard": _jaccard(b_sel, v_sel),
            "rendered_sentence_jaccard": _jaccard(b_rnd, v_rnd),
            "prompt_hash_changed": bool(_hash_text(_safe_text(b.get("question", "")) + "\n" + b_ctx) != _hash_text(_safe_text(v.get("question", "")) + "\n" + v_ctx)),
            "context_hash_changed": bool(_hash_text(b_ctx) != _hash_text(v_ctx)),
            "baseline_selected_sentence_ids": list(b_sel),
            "variant_selected_sentence_ids": list(v_sel),
            "baseline_rendered_sentence_ids": list(b_rnd),
            "variant_rendered_sentence_ids": list(v_rnd),
        }
        paired.append(row)

        if delta_f1 >= 0.05 or delta_ans > 0.0:
            improved.append(row)
        if delta_f1 <= -0.05:
            regressed.append(row)

    improved = sorted(improved, key=lambda r: (_safe_float(r.get("delta_f1")), _safe_float(r.get("delta_sf_recall"))), reverse=True)
    regressed = sorted(regressed, key=lambda r: (_safe_float(r.get("delta_f1")), _safe_float(r.get("delta_sf_recall"))))
    return paired, improved, regressed


def _write_examples_md(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    lines: List[str] = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for i, row in enumerate(rows, start=1):
        lines.append(f"## {i}. {_safe_text(row.get('dataset'))} / {_safe_text(row.get('sample_id'))}")
        lines.append(f"- variant_profile: {_safe_text(row.get('variant_profile'))}")
        lines.append(f"- question: {_safe_text(row.get('question'))}")
        lines.append(
            f"- delta_f1: {_fmt(row.get('delta_f1'), 4)}, delta_sf_recall: {_fmt(row.get('delta_sf_recall'), 4)}, delta_answer_string_hit: {_fmt(row.get('delta_answer_string_hit'), 4)}"
        )
        lines.append(f"- baseline_prediction: {_safe_text(row.get('baseline_prediction'))}")
        lines.append(f"- variant_prediction: {_safe_text(row.get('variant_prediction'))}")
        lines.append(
            f"- selected_jaccard/rendered_jaccard: {_fmt(row.get('selected_sentence_jaccard'), 4)} / {_fmt(row.get('rendered_sentence_jaccard'), 4)}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_skip_reason_md(path: Path, skip_rows: Sequence[Mapping[str, Any]]) -> None:
    lines: List[str] = ["# Minimal Package Skip Reasons", ""]
    if not skip_rows:
        lines.append("No minimal-package diagnostics available.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for row in skip_rows:
        lines.append(f"## {_safe_text(row.get('dataset'))} / {_safe_text(row.get('profile'))}")
        lines.append(f"- dominant_skip_reason: {_safe_text(row.get('dominant_skip_reason')) or 'n/a'}")
        lines.append(f"- minimal_package_apply_rate: {_fmt(row.get('minimal_package_apply_rate'), 4)}")
        lines.append("- skip_reason_counts:")
        counts = dict(row.get("skip_reason_counts", {}) or {})
        if not counts:
            lines.append("  - (empty)")
        else:
            for reason, cnt in sorted(counts.items(), key=lambda kv: (-_safe_int(kv[1], 0), _safe_text(kv[0]))):
                lines.append(f"  - {reason}: {cnt}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision(rows_main: Sequence[Mapping[str, Any]], delta_rows: Sequence[Mapping[str, Any]]) -> str:
    if not delta_rows:
        return "hold"
    by_profile: Dict[str, List[Mapping[str, Any]]] = {}
    for row in delta_rows:
        by_profile.setdefault(_safe_text(row.get("variant_profile")), []).append(row)

    baseline_index = {
        (_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): r
        for r in rows_main
    }

    def _profile_reject(profile_rows: Sequence[Mapping[str, Any]]) -> bool:
        for d in profile_rows:
            if _safe_float(d.get("delta_F1"), 0.0) < -0.02:
                return True
            if _safe_float(d.get("delta_avg_context_tokens_pct"), 0.0) > 20.0:
                return True
            if _safe_float(d.get("delta_retrieval_ms_pct"), 0.0) > 20.0:
                return True
        return False

    for profile_rows in by_profile.values():
        if _profile_reject(profile_rows):
            return "reject"

    # Strong accept for a non-baseline component if all dataset deltas are non-regressive.
    for profile, rows in by_profile.items():
        if not rows:
            continue
        if all(_safe_float(r.get("delta_F1"), 0.0) >= -0.01 for r in rows):
            avg_df1 = _mean([_safe_float(r.get("delta_F1"), 0.0) for r in rows])
            avg_df1k = _mean([_safe_float(r.get("delta_F1_per_1k_context_tokens"), 0.0) for r in rows])
            if avg_df1 >= 0.0 and avg_df1k >= 0.0 and profile != "unified_acr_rcedr_v12_semantic_sufficiency":
                return "accept"
    return "hold"


def build_summary(
    *,
    out_root: Path,
    datasets: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    artifact = collect_phase6s_artifacts(out_root)
    runs = _load_completed_runs(out_root)
    run_index = {(r.dataset, r.profile): r for r in runs}

    baseline_profile = "unified_acr_rcedr_v12"
    variant_profiles = [
        "unified_acr_rcedr_v12_semantic_sufficiency",
        "unified_acr_rcedr_v12_top_slice_only",
        "unified_acr_rcedr_v12_minimal_package_only",
        "unified_acr_rcedr_v12_contract_only",
    ]

    rows_main: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    paired_all: List[Dict[str, Any]] = []
    top_slice_examples: List[Dict[str, Any]] = []
    minimal_examples: List[Dict[str, Any]] = []
    regression_examples: List[Dict[str, Any]] = []

    for dataset in datasets:
        base = run_index.get((dataset, baseline_profile))
        if base is None:
            continue
        bm = _summary_metrics(base.summary)
        bdiag = _contract_diag_for_run(base)
        rows_main.append(
            {
                "dataset": dataset,
                "profile": baseline_profile,
                "role": "baseline",
                **bm,
                **bdiag,
            }
        )

        for profile in variant_profiles:
            run = run_index.get((dataset, profile))
            if run is None:
                continue
            vm = _summary_metrics(run.summary)
            vdiag = _contract_diag_for_run(run)
            rows_main.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "role": "variant",
                    **vm,
                    **vdiag,
                }
            )

            base_ctx = bm["avg_context_tokens"]
            base_ret = bm["retrieval_ms"]
            delta_rows.append(
                {
                    "dataset": dataset,
                    "variant_profile": profile,
                    "delta_EM": float(vm["EM"] - bm["EM"]),
                    "delta_F1": float(vm["F1"] - bm["F1"]),
                    "delta_F1_per_1k_context_tokens": float(vm["F1_per_1k_context_tokens"] - bm["F1_per_1k_context_tokens"]),
                    "delta_SF_precision": float(vm["supporting_fact_precision"] - bm["supporting_fact_precision"]),
                    "delta_SF_recall": float(vm["supporting_fact_recall"] - bm["supporting_fact_recall"]),
                    "delta_avg_context_tokens": float(vm["avg_context_tokens"] - bm["avg_context_tokens"]),
                    "delta_retrieval_ms": float(vm["retrieval_ms"] - bm["retrieval_ms"]),
                    "delta_total_ms": float(vm["total_ms"] - bm["total_ms"]),
                    "delta_answer_string_hit": float(vm["answer_string_hit"] - bm["answer_string_hit"]),
                    "delta_answer_string_rank_avg": float(vm["answer_string_rank_avg"] - bm["answer_string_rank_avg"]),
                    "delta_answer_bearing_density": float(vm["answer_bearing_density"] - bm["answer_bearing_density"]),
                    "delta_avg_context_tokens_pct": float(((vm["avg_context_tokens"] - base_ctx) / base_ctx) * 100.0) if base_ctx > 0 else 0.0,
                    "delta_retrieval_ms_pct": float(((vm["retrieval_ms"] - base_ret) / base_ret) * 100.0) if base_ret > 0 else 0.0,
                }
            )

            paired, improved, regressed = _pair_baseline_variant(dataset, base, run, profile)
            paired_all.extend(paired)
            regression_examples.extend(regressed[: max(1, top_k)])

            if profile.endswith("top_slice_only"):
                top_slice_examples.extend(improved[: max(1, top_k)])
            if profile.endswith("minimal_package_only") or profile.endswith("contract_only"):
                minimal_examples.extend(improved[: max(1, top_k)])

    # Artifacts.
    _write_jsonl(out_root / "paired_query_comparison.jsonl", paired_all)

    csv_rows: List[Dict[str, Any]] = []
    for row in rows_main:
        csv_rows.append(
            {
                "dataset": row.get("dataset"),
                "profile": row.get("profile"),
                "role": row.get("role"),
                "top_slice_reorder_rate": row.get("top_slice_reorder_rate"),
                "top_slice_reorder_changed_order_rate": row.get("top_slice_reorder_changed_order_rate"),
                "minimal_package_apply_rate": row.get("minimal_package_apply_rate"),
                "avg_extra_sentences": row.get("avg_extra_sentences"),
                "avg_extra_tokens": row.get("avg_extra_tokens"),
                "budget_skip_rate": row.get("budget_skip_rate"),
                "dominant_skip_reason": row.get("dominant_skip_reason"),
                "avg_context_token_delta": row.get("avg_context_token_delta"),
            }
        )
    _write_csv(
        out_root / "contract_component_diagnostics.csv",
        csv_rows,
        [
            "dataset",
            "profile",
            "role",
            "top_slice_reorder_rate",
            "top_slice_reorder_changed_order_rate",
            "minimal_package_apply_rate",
            "avg_extra_sentences",
            "avg_extra_tokens",
            "budget_skip_rate",
            "dominant_skip_reason",
            "avg_context_token_delta",
        ],
    )

    top_slice_examples = sorted(
        top_slice_examples,
        key=lambda r: (_safe_float(r.get("delta_f1")), _safe_float(r.get("delta_answer_string_hit"))),
        reverse=True,
    )[: max(1, top_k)]
    minimal_examples = sorted(
        minimal_examples,
        key=lambda r: (_safe_float(r.get("delta_f1")), _safe_float(r.get("delta_answer_string_hit"))),
        reverse=True,
    )[: max(1, top_k)]
    regression_examples = sorted(
        regression_examples,
        key=lambda r: (_safe_float(r.get("delta_f1")), _safe_float(r.get("delta_sf_recall"))),
    )[: max(1, top_k)]

    _write_examples_md(out_root / "top_slice_examples_top20.md", "Top-slice Examples (Top 20)", top_slice_examples)
    _write_examples_md(out_root / "minimal_package_examples_top20.md", "Minimal Package Examples (Top 20)", minimal_examples)
    _write_examples_md(out_root / "regression_examples_top20.md", "Regression Examples (Top 20)", regression_examples)

    skip_rows = [
        r
        for r in rows_main
        if _safe_text(r.get("profile")).endswith("minimal_package_only")
        or _safe_text(r.get("profile")).endswith("contract_only")
    ]
    _write_skip_reason_md(out_root / "minimal_package_skip_reasons.md", skip_rows)

    decision = _decision(rows_main, delta_rows)

    summary = {
        "phase": "phase6v_legacy_contract_component_ablation",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "datasets": list(datasets),
        "rows_main": rows_main,
        "delta_rows": delta_rows,
        "paired_count": int(len(paired_all)),
        "artifact_counts": dict((artifact.get("counts", {}) or {})),
        "decision": decision,
    }
    _write_json(out_root / "phase6v_legacy_contract_component_ablation_summary.json", summary)

    # Markdown summary.
    counts = dict((artifact.get("counts", {}) or {}))
    lines: List[str] = []
    lines.append("# PHASE6V Legacy Contract Component Ablation Summary")
    lines.append("")
    lines.append("## 1. Goal")
    lines.append("Component-wise ablation of top-slice reorder and minimal-package rendering under unified v12.")
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
                r5=_fmt(row.get("Recall@5")),
                em=_fmt(row.get("EM")),
                f1=_fmt(row.get("F1")),
                ctx=_fmt(row.get("avg_context_tokens"), 2),
                f1k=_fmt(row.get("F1_per_1k_context_tokens")),
                sp=_fmt(row.get("supporting_fact_precision")),
                sr=_fmt(row.get("supporting_fact_recall")),
                sf1=_fmt(row.get("supporting_fact_f1")),
                rm=_fmt(row.get("retrieval_ms"), 2),
                gm=_fmt(row.get("generation_ms"), 2),
                tm=_fmt(row.get("total_ms"), 2),
            )
        )
    lines.append("")
    lines.append("## 4. Delta vs Baseline")
    lines.append("| dataset | variant_profile | ΔEM | ΔF1 | ΔF1_per_1k | ΔSF_precision | ΔSF_recall | Δavg_context_tokens | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(delta_rows, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("variant_profile")))):
        lines.append(
            "| {dataset} | {profile} | {dem} | {df1} | {df1k} | {dsp} | {dsr} | {dctx} | {drm} | {dtm} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("variant_profile")),
                dem=_fmt(row.get("delta_EM")),
                df1=_fmt(row.get("delta_F1")),
                df1k=_fmt(row.get("delta_F1_per_1k_context_tokens")),
                dsp=_fmt(row.get("delta_SF_precision")),
                dsr=_fmt(row.get("delta_SF_recall")),
                dctx=_fmt(row.get("delta_avg_context_tokens"), 2),
                drm=_fmt(row.get("delta_retrieval_ms"), 2),
                dtm=_fmt(row.get("delta_total_ms"), 2),
            )
        )
    lines.append("")
    lines.append("## 5. Top-slice Reorder Analysis")
    lines.append("| dataset | profile | top_slice_reorder_rate | top_slice_reorder_changed_order_rate |")
    lines.append("|---|---|---:|---:|")
    for row in rows_main:
        profile = _safe_text(row.get("profile"))
        if not profile.endswith("top_slice_only") and not profile.endswith("contract_only"):
            continue
        lines.append(
            "| {dataset} | {profile} | {applied} | {changed} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=profile,
                applied=_fmt(row.get("top_slice_reorder_rate")),
                changed=_fmt(row.get("top_slice_reorder_changed_order_rate")),
            )
        )
    lines.append("")
    lines.append("## 6. Minimal Package Rendering Analysis")
    lines.append("| dataset | profile | minimal_package_apply_rate | avg_extra_sentences | avg_extra_tokens | budget_skip_rate | dominant_skip_reason | avg_context_token_delta |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|")
    for row in rows_main:
        profile = _safe_text(row.get("profile"))
        if not profile.endswith("minimal_package_only") and not profile.endswith("contract_only"):
            continue
        lines.append(
            "| {dataset} | {profile} | {apply} | {extra} | {tok} | {skip} | {reason} | {delta} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=profile,
                apply=_fmt(row.get("minimal_package_apply_rate")),
                extra=_fmt(row.get("avg_extra_sentences")),
                tok=_fmt(row.get("avg_extra_tokens"), 2),
                skip=_fmt(row.get("budget_skip_rate")),
                reason=_safe_text(row.get("dominant_skip_reason")) or "n/a",
                delta=_fmt(row.get("avg_context_token_delta"), 2),
            )
        )
    lines.append("")
    lines.append("## 7. Answer-Sufficiency Metrics")
    lines.append("| dataset | profile | answer_string_hit | answer_string_rank_avg | answer_bearing_density | answer_context_early_hit |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in sorted(rows_main, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {hit} | {rank} | {density} | {early} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("profile")),
                hit=_fmt(row.get("answer_string_hit")),
                rank=_fmt(row.get("answer_string_rank_avg"), 2),
                density=_fmt(row.get("answer_bearing_density")),
                early=_fmt(row.get("answer_context_early_hit")),
            )
        )
    lines.append("")
    lines.append("## 8. Improved Examples")
    lines.append(f"- top_slice examples: `{out_root / 'top_slice_examples_top20.md'}`")
    lines.append(f"- minimal_package examples: `{out_root / 'minimal_package_examples_top20.md'}`")
    lines.append("")
    lines.append("## 9. Regression Examples")
    lines.append(f"- regression examples: `{out_root / 'regression_examples_top20.md'}`")
    lines.append("")
    lines.append("## 10. Decision")
    lines.append(f"- {decision}")
    lines.append("")
    (out_root / "PHASE6V_LEGACY_CONTRACT_COMPONENT_ABLATION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--datasets", type=str, default="hotpotqa,2wikimultihopqa")
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root).resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    build_summary(out_root=out_root, datasets=datasets, top_k=max(1, int(args.top_k)))
    print(out_root / "PHASE6V_LEGACY_CONTRACT_COMPONENT_ABLATION_SUMMARY.md")
    print(out_root / "phase6v_legacy_contract_component_ablation_summary.json")


if __name__ == "__main__":
    main()

