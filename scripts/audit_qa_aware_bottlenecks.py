#!/usr/bin/env python3
"""Phase-6O QA-aware bottleneck audit."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa"]
DEFAULT_PROFILES = [
    "legacy_sota",
    "unified_large",
    "unified_gl_rcedr_v1",
    "unified_gl_rcedr_v1_sentence_contract",
    "unified_gl_rcedr_v1_support_span_contract",
    "unified_gl_rcedr_v1_support_span_contract_no_cap",
    "unified_gl_rcedr_v1_adaptive_support_span",
    "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
    "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty",
    "unified_gl_rcedr_v1_adaptive_support_span_span40",
]
REFERENCE_PROFILES = {"legacy_sota", "unified_large", "unified_gl_rcedr_v1"}
PRIMARY_BASELINE_PROFILE_DEFAULT = "unified_large"
TEACHER_REFERENCE_PROFILE_DEFAULT = "legacy_sota"
EXTERNAL_BASELINE_PROFILE_DEFAULT = "unified_gl_rcedr_v1"


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


def _tokens(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return dict(obj or {}) if isinstance(obj, Mapping) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row or {}))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row or {}))


def _find_retrieval_aggregate_csv(root: Path) -> Path:
    candidates = [
        root / "aggregate_by_profile_dataset.csv",
        root / "audit" / "aggregate_by_profile_dataset.csv",
        root / "champion_audit" / "aggregate_by_profile_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find aggregate_by_profile_dataset.csv under {root}. "
        "Expected root/, root/audit/, or root/champion_audit/."
    )


def _collect_qa_summary_rows(input_root: Path, datasets: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_manifest_paths = sorted(input_root.rglob("run_manifest.json"))
    for manifest_path in run_manifest_paths:
        manifest = _read_json(manifest_path)
        profile = _safe_text(manifest.get("profile"))
        profile_name = _safe_text(manifest.get("profile_name"))
        if not profile and profile_name:
            profile = "legacy_sota"
        if not profile:
            profile = _safe_text(manifest_path.parent.name)

        records = list(manifest.get("records", []) or [])
        for record in records:
            if not isinstance(record, Mapping):
                continue
            dataset = _safe_text(record.get("dataset"))
            if dataset not in datasets:
                continue
            summary_path = _safe_text(record.get("summary_path"))
            if not summary_path:
                continue
            summary = _read_json(Path(summary_path))
            if not summary:
                continue

            f1 = _safe_float(summary.get("f1", 0.0), 0.0)
            prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
            f1_per_1k_prompt = float((f1 * 1000.0) / prompt_tokens) if prompt_tokens > 0 else 0.0

            rows.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "EM": _safe_float(summary.get("em", 0.0), 0.0),
                    "F1": f1,
                    "prompt_tokens_avg": prompt_tokens,
                    "completion_tokens_avg": _safe_float(summary.get("completion_tokens_avg", 0.0), 0.0),
                    "retrieval_ms": _safe_float(
                        summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)),
                        0.0,
                    ),
                    "generation_ms": _safe_float(
                        summary.get("generation_ms", summary.get("render_ms", 0.0)),
                        0.0,
                    ),
                    "total_ms": _safe_float(
                        summary.get("total_ms", summary.get("total_latency_ms", 0.0)),
                        0.0,
                    ),
                    "sf_F1_per_1k_prompt_tokens": f1_per_1k_prompt,
                    "qa_executed_samples": _safe_int(summary.get("qa_executed_samples", 0), 0),
                    "summary_path": summary_path,
                }
            )
    return rows


def _aggregate_mean(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        vals = [_safe_float(r.get(key, 0.0), 0.0) for r in rows]
        out[key] = float(sum(vals) / max(1, len(vals)))
    return out


def _summarize_qa_rows(
    *,
    qa_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    profiles: Sequence[str],
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in qa_rows:
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))
        if dataset not in datasets or profile not in profiles:
            continue
        buckets.setdefault((dataset, profile), []).append(row)

    metric_keys = [
        "EM",
        "F1",
        "prompt_tokens_avg",
        "completion_tokens_avg",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "sf_F1_per_1k_prompt_tokens",
    ]
    out: List[Dict[str, Any]] = []
    for dataset in datasets:
        for profile in profiles:
            group = list(buckets.get((dataset, profile), []) or [])
            if not group:
                out.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "qa_num_runs": 0,
                        "qa_num_queries": 0,
                        "qa_available": False,
                        "EM": 0.0,
                        "F1": 0.0,
                        "prompt_tokens_avg": 0.0,
                        "completion_tokens_avg": 0.0,
                        "retrieval_ms": 0.0,
                        "generation_ms": 0.0,
                        "total_ms": 0.0,
                        "sf_F1_per_1k_prompt_tokens": 0.0,
                    }
                )
                continue
            means = _aggregate_mean(group, metric_keys)
            out.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "qa_num_runs": len(group),
                    "qa_num_queries": int(sum(_safe_int(r.get("qa_executed_samples", 0), 0) for r in group)),
                    "qa_available": True,
                    **means,
                }
            )
    return out


def _prepare_retrieval_index(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))
        if not dataset or not profile:
            continue
        out[(dataset, profile)] = dict(row)
    return out


def _merge_qa_and_retrieval(
    *,
    qa_rows: Sequence[Mapping[str, Any]],
    retrieval_index: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in qa_rows:
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))
        retrieval = dict(retrieval_index.get((dataset, profile), {}) or {})
        retrieval_available = bool(retrieval)
        rendered_tokens = _safe_float(retrieval.get("rendered_tokens", retrieval.get("rendered_tokens_avg", 0.0)), 0.0)
        f1 = _safe_float(row.get("F1", 0.0), 0.0)

        merged = {
            "dataset": dataset,
            "profile": profile,
            "qa_num_runs": _safe_int(row.get("qa_num_runs", 0), 0),
            "qa_num_queries": _safe_int(row.get("qa_num_queries", 0), 0),
            "qa_available": bool(row.get("qa_available", False)),
            "retrieval_available": retrieval_available,
            "EM": _safe_float(row.get("EM", 0.0), 0.0),
            "F1": f1,
            "prompt_tokens_avg": _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0),
            "completion_tokens_avg": _safe_float(row.get("completion_tokens_avg", 0.0), 0.0),
            "retrieval_ms": _safe_float(row.get("retrieval_ms", 0.0), 0.0),
            "generation_ms": _safe_float(row.get("generation_ms", 0.0), 0.0),
            "total_ms": _safe_float(row.get("total_ms", 0.0), 0.0),
            "sf_F1_per_1k_prompt_tokens": _safe_float(row.get("sf_F1_per_1k_prompt_tokens", 0.0), 0.0),
            "rendered_sf_R": _safe_float(retrieval.get("rendered_sf_R", 0.0), 0.0),
            "rendered_sf_P": _safe_float(retrieval.get("rendered_sf_P", 0.0), 0.0),
            "rendered_sf_F1": _safe_float(retrieval.get("rendered_sf_F1", 0.0), 0.0),
            "rendered_sf_F1_per_1k_tokens": _safe_float(retrieval.get("rendered_sf_F1_per_1k_tokens", 0.0), 0.0),
            "rendered_tokens": rendered_tokens,
            "avg_item_tokens": _safe_float(
                retrieval.get(
                    "avg_item_tokens",
                    retrieval.get("avg_tokens_per_item", rendered_tokens),
                ),
                0.0,
            ),
            "chunk_like_rate": _safe_float(retrieval.get("chunk_like_rate", 0.0), 0.0),
            "render_drop_rate": _safe_float(retrieval.get("render_drop_rate", retrieval.get("selector_drop_rate", 0.0)), 0.0),
            "generation_ready_rate": _safe_float(retrieval.get("generation_ready_rate", 0.0), 0.0),
        }
        merged["F1_per_1k_prompt_tokens"] = (
            (f1 * 1000.0) / merged["prompt_tokens_avg"] if merged["prompt_tokens_avg"] > 0 else 0.0
        )
        merged["F1_per_1k_rendered_tokens"] = (
            (f1 * 1000.0) / merged["rendered_tokens"] if merged["rendered_tokens"] > 0 else 0.0
        )
        out.append(merged)
    out.sort(key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("profile"))))
    return out


def _annotate_baseline_deltas(
    *,
    rows: Sequence[Dict[str, Any]],
    datasets: Sequence[str],
    primary_baseline_profile: str,
    teacher_reference_profile: str,
    external_baseline_profile: str,
) -> None:
    index = {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): r for r in rows}
    for dataset in datasets:
        primary = dict(index.get((dataset, primary_baseline_profile), {}) or {})
        teacher = dict(index.get((dataset, teacher_reference_profile), {}) or {})
        external = dict(index.get((dataset, external_baseline_profile), {}) or {})

        for row in rows:
            if _safe_text(row.get("dataset")) != dataset:
                continue
            f1 = _safe_float(row.get("F1", 0.0), 0.0)
            prompt = _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0)
            rendered = _safe_float(row.get("rendered_tokens", 0.0), 0.0)

            p_f1 = _safe_float(primary.get("F1", 0.0), 0.0)
            p_prompt = _safe_float(primary.get("prompt_tokens_avg", 0.0), 0.0)
            p_rendered = _safe_float(primary.get("rendered_tokens", 0.0), 0.0)
            row["delta_F1_vs_primary_baseline"] = f1 - p_f1
            row["delta_prompt_tokens_vs_primary_baseline"] = prompt - p_prompt
            row["delta_rendered_tokens_vs_primary_baseline"] = rendered - p_rendered

            t_f1 = _safe_float(teacher.get("F1", 0.0), 0.0)
            t_prompt = _safe_float(teacher.get("prompt_tokens_avg", 0.0), 0.0)
            t_rendered = _safe_float(teacher.get("rendered_tokens", 0.0), 0.0)
            row["delta_F1_vs_teacher_reference"] = f1 - t_f1
            row["delta_prompt_tokens_vs_teacher_reference"] = prompt - t_prompt
            row["delta_rendered_tokens_vs_teacher_reference"] = rendered - t_rendered

            e_f1 = _safe_float(external.get("F1", 0.0), 0.0)
            e_prompt = _safe_float(external.get("prompt_tokens_avg", 0.0), 0.0)
            e_rendered = _safe_float(external.get("rendered_tokens", 0.0), 0.0)
            row["delta_F1_vs_external_baseline"] = f1 - e_f1
            row["delta_prompt_tokens_vs_external_baseline"] = prompt - e_prompt
            row["delta_rendered_tokens_vs_external_baseline"] = rendered - e_rendered

            row["primary_baseline_profile"] = primary_baseline_profile
            row["teacher_reference_profile"] = teacher_reference_profile
            row["external_baseline_profile"] = external_baseline_profile


def _build_candidate_compression_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    primary_baseline_profile: str,
    teacher_reference_profile: str,
    external_baseline_profile: str,
) -> List[Dict[str, Any]]:
    _ = frontier_rows  # frontier CSV is still exported; summary computes target-aware non-domination directly.

    def _non_dominated_status(
        *,
        dataset: str,
        target_profile: str,
        maximize: Sequence[str],
        minimize: Sequence[str],
    ) -> bool:
        pool = [
            r
            for r in rows
            if _safe_text(r.get("dataset")) == dataset
            and _safe_text(r.get("profile")) != teacher_reference_profile
            and bool(r.get("qa_available", False))
            and bool(r.get("retrieval_available", False))
        ]
        target = None
        for row in pool:
            if _safe_text(row.get("profile")) == target_profile:
                target = row
                break
        if target is None:
            return False
        for other in pool:
            if _safe_text(other.get("profile")) == target_profile:
                continue
            if _dominates(other, target, maximize=maximize, minimize=minimize):
                return False
        return True

    profiles = sorted({_safe_text(r.get("profile")) for r in rows if _safe_text(r.get("profile"))})
    excluded = {primary_baseline_profile, teacher_reference_profile, external_baseline_profile}
    summary: List[Dict[str, Any]] = []
    for profile in profiles:
        if profile in excluded:
            continue
        ds_rows = [r for r in rows if _safe_text(r.get("profile")) == profile and _safe_text(r.get("dataset")) in datasets]
        if not ds_rows:
            continue
        primary_gate_pass = 0
        prompt_pareto = 0
        rendered_pareto = 0
        f1_deltas: List[float] = []
        prompt_reductions: List[float] = []
        rendered_reductions: List[float] = []
        for row in ds_rows:
            dataset = _safe_text(row.get("dataset"))
            f1_delta = _safe_float(row.get("delta_F1_vs_primary_baseline", 0.0), 0.0)
            prompt_delta = _safe_float(row.get("delta_prompt_tokens_vs_primary_baseline", 0.0), 0.0)
            rendered_delta = _safe_float(row.get("delta_rendered_tokens_vs_primary_baseline", 0.0), 0.0)
            qa_ok = bool(row.get("qa_available", False))
            retrieval_ok = bool(row.get("retrieval_available", False))
            if qa_ok and retrieval_ok and f1_delta >= 0.0 and prompt_delta <= 0.0 and rendered_delta <= 0.0:
                primary_gate_pass += 1
            if _non_dominated_status(
                dataset=dataset,
                target_profile=profile,
                maximize=["F1"],
                minimize=["prompt_tokens_avg"],
            ):
                prompt_pareto += 1
            if _non_dominated_status(
                dataset=dataset,
                target_profile=profile,
                maximize=["F1"],
                minimize=["rendered_tokens"],
            ):
                rendered_pareto += 1
            f1_deltas.append(f1_delta)
            prompt_reductions.append(-prompt_delta)
            rendered_reductions.append(-rendered_delta)

        dataset_count = len(ds_rows)
        summary.append(
            {
                "profile": profile,
                "dataset_count": dataset_count,
                "primary_gate_pass_count": primary_gate_pass,
                "primary_gate_all": primary_gate_pass == dataset_count and dataset_count > 0,
                "qa_prompt_pareto_count": prompt_pareto,
                "qa_rendered_pareto_count": rendered_pareto,
                "qa_prompt_pareto_all": prompt_pareto == dataset_count and dataset_count > 0,
                "qa_rendered_pareto_all": rendered_pareto == dataset_count and dataset_count > 0,
                "mean_delta_F1_vs_primary_baseline": float(sum(f1_deltas) / max(1, len(f1_deltas))),
                "mean_prompt_token_reduction_vs_primary_baseline": float(
                    sum(prompt_reductions) / max(1, len(prompt_reductions))
                ),
                "mean_rendered_token_reduction_vs_primary_baseline": float(
                    sum(rendered_reductions) / max(1, len(rendered_reductions))
                ),
            }
        )

    summary.sort(
        key=lambda x: (
            -_safe_int(x.get("primary_gate_pass_count", 0), 0),
            -_safe_int(x.get("qa_prompt_pareto_count", 0), 0),
            -_safe_int(x.get("qa_rendered_pareto_count", 0), 0),
            -_safe_float(x.get("mean_delta_F1_vs_primary_baseline", 0.0), 0.0),
            -_safe_float(x.get("mean_prompt_token_reduction_vs_primary_baseline", 0.0), 0.0),
            -_safe_float(x.get("mean_rendered_token_reduction_vs_primary_baseline", 0.0), 0.0),
            _safe_text(x.get("profile")),
        )
    )
    return summary


def _dominates(a: Mapping[str, Any], b: Mapping[str, Any], maximize: Sequence[str], minimize: Sequence[str]) -> bool:
    ge_all = True
    better_any = False
    for key in maximize:
        av = _safe_float(a.get(key, 0.0), 0.0)
        bv = _safe_float(b.get(key, 0.0), 0.0)
        if av < bv:
            ge_all = False
            break
        if av > bv:
            better_any = True
    if not ge_all:
        return False
    for key in minimize:
        av = _safe_float(a.get(key, 0.0), 0.0)
        bv = _safe_float(b.get(key, 0.0), 0.0)
        if av > bv:
            ge_all = False
            break
        if av < bv:
            better_any = True
    return ge_all and better_any


def _metric_available(row: Mapping[str, Any], metric: str) -> bool:
    qa_required = {
        "EM",
        "F1",
        "prompt_tokens_avg",
        "completion_tokens_avg",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "F1_per_1k_prompt_tokens",
        "sf_F1_per_1k_prompt_tokens",
    }
    retrieval_required = {
        "rendered_sf_R",
        "rendered_sf_P",
        "rendered_sf_F1",
        "rendered_sf_F1_per_1k_tokens",
        "rendered_tokens",
        "avg_item_tokens",
        "chunk_like_rate",
        "render_drop_rate",
        "generation_ready_rate",
        "F1_per_1k_rendered_tokens",
    }
    if metric in qa_required and not bool(row.get("qa_available", False)):
        return False
    if metric in retrieval_required and not bool(row.get("retrieval_available", False)):
        return False
    value = row.get(metric, None)
    if value is None:
        return False
    if metric in {"prompt_tokens_avg", "rendered_tokens"} and _safe_float(value, 0.0) <= 0.0:
        return False
    return True


def compute_frontier_status_rows(
    *,
    merged_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    profiles: Sequence[str],
    reference_profiles: Sequence[str] = ("legacy_sota", "unified_large", "unified_gl_rcedr_v1"),
) -> List[Dict[str, Any]]:
    frontier_specs = [
        ("qa_f1_vs_prompt_tokens", ["F1"], ["prompt_tokens_avg"], ["F1", "prompt_tokens_avg"]),
        ("qa_f1_vs_rendered_tokens", ["F1"], ["rendered_tokens"], ["F1", "rendered_tokens"]),
        ("qa_f1_vs_f1_per_1k_prompt_tokens", ["F1", "F1_per_1k_prompt_tokens"], [], ["F1", "F1_per_1k_prompt_tokens"]),
        ("qa_f1_vs_rendered_sf_f1_per_1k_tokens", ["F1", "rendered_sf_F1_per_1k_tokens"], [], ["F1", "rendered_sf_F1_per_1k_tokens"]),
        ("rendered_sf_R_vs_rendered_tokens", ["rendered_sf_R"], ["rendered_tokens"], ["rendered_sf_R", "rendered_tokens"]),
        ("rendered_sf_f1_per_1k_tokens_vs_qa_f1", ["rendered_sf_F1_per_1k_tokens", "F1"], [], ["rendered_sf_F1_per_1k_tokens", "F1"]),
    ]

    ref_set = set(reference_profiles)
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        ds_rows = [dict(r) for r in merged_rows if _safe_text(r.get("dataset")) == dataset]
        for frontier_name, maximize, minimize, required in frontier_specs:
            complete_rows = [
                r
                for r in ds_rows
                if all(_metric_available(r, metric) for metric in required)
            ]
            for row in ds_rows:
                profile = _safe_text(row.get("profile"))
                row_complete = all(_metric_available(row, metric) for metric in required)
                status = "incomplete_metrics"
                note = "missing required metrics"

                if row_complete:
                    if profile in ref_set:
                        status = "reference_only"
                        note = "reference profile"
                    else:
                        dominator = ""
                        for other in complete_rows:
                            if _safe_text(other.get("profile")) == profile:
                                continue
                            if _dominates(other, row, maximize=maximize, minimize=minimize):
                                dominator = _safe_text(other.get("profile"))
                                break
                        if dominator:
                            status = "dominated"
                            note = f"dominated by {dominator}"
                        else:
                            status = "pareto"
                            note = "non-dominated among complete metrics"

                rows.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "frontier_name": frontier_name,
                        "maximize": ",".join(maximize),
                        "minimize": ",".join(minimize),
                        "pareto_status": status,
                        "note": note,
                        "F1": _safe_float(row.get("F1", 0.0), 0.0),
                        "prompt_tokens_avg": _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0),
                        "rendered_tokens": _safe_float(row.get("rendered_tokens", 0.0), 0.0),
                        "F1_per_1k_prompt_tokens": _safe_float(row.get("F1_per_1k_prompt_tokens", 0.0), 0.0),
                        "rendered_sf_R": _safe_float(row.get("rendered_sf_R", 0.0), 0.0),
                        "rendered_sf_F1_per_1k_tokens": _safe_float(row.get("rendered_sf_F1_per_1k_tokens", 0.0), 0.0),
                    }
                )
    rows.sort(key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("frontier_name")), _safe_text(x.get("profile"))))
    return rows


def _dataset_medians(rows: Sequence[Mapping[str, Any]], dataset: str) -> Dict[str, float]:
    ds_rows = [r for r in rows if _safe_text(r.get("dataset")) == dataset]
    medians: Dict[str, float] = {}
    for key in ("rendered_sf_R", "rendered_sf_F1_per_1k_tokens", "F1"):
        vals = [_safe_float(r.get(key, 0.0), 0.0) for r in ds_rows if _safe_float(r.get(key, 0.0), 0.0) > 0]
        medians[key] = float(statistics.median(vals)) if vals else 0.0
    return medians


def classify_retrieval_qa_mismatch(
    row: Mapping[str, Any],
    *,
    medians: Mapping[str, float],
) -> Tuple[str, str, str]:
    if not bool(row.get("qa_available", False)) or not bool(row.get("retrieval_available", False)):
        return "unknown", "unknown", "incomplete_metrics"

    rs_r = _safe_float(row.get("rendered_sf_R", 0.0), 0.0)
    rs_density = _safe_float(row.get("rendered_sf_F1_per_1k_tokens", 0.0), 0.0)
    qa_f1 = _safe_float(row.get("F1", 0.0), 0.0)

    r_thr = max(0.35, _safe_float(medians.get("rendered_sf_R", 0.0), 0.0))
    d_thr = max(0.6, _safe_float(medians.get("rendered_sf_F1_per_1k_tokens", 0.0), 0.0))
    q_thr = max(0.20, _safe_float(medians.get("F1", 0.0), 0.0))

    retrieval_strong = (rs_r >= r_thr) and (rs_density >= d_thr)
    qa_strong = qa_f1 >= q_thr

    retrieval_strength = "strong" if retrieval_strong else "weak"
    qa_strength = "strong" if qa_strong else "weak"

    if retrieval_strong and not qa_strong:
        mismatch = "retrieval_strong_qa_weak"
    elif (not retrieval_strong) and qa_strong:
        mismatch = "retrieval_weak_qa_strong"
    elif retrieval_strong and qa_strong:
        mismatch = "aligned_strong"
    else:
        mismatch = "aligned_weak"
    return retrieval_strength, qa_strength, mismatch


def classify_dominant_bottleneck(row: Mapping[str, Any]) -> Tuple[str, str]:
    qa_f1 = _safe_float(row.get("F1", 0.0), 0.0)
    r = _safe_float(row.get("rendered_sf_R", 0.0), 0.0)
    p = _safe_float(row.get("rendered_sf_P", 0.0), 0.0)
    tokens = _safe_float(row.get("rendered_tokens", 0.0), 0.0)
    density = _safe_float(row.get("rendered_sf_F1_per_1k_tokens", 0.0), 0.0)
    prompt = _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0)

    qa_low = qa_f1 < 0.35
    if tokens <= 170 and density >= 1.2 and r < 0.45 and qa_low:
        return (
            "over_compression_bottleneck",
            f"tokens={tokens:.1f}, density={density:.3f}, rendered_sf_R={r:.3f}, F1={qa_f1:.3f}",
        )
    if r < 0.45 and qa_low:
        return (
            "evidence_recall_bottleneck",
            f"rendered_sf_R={r:.3f}, F1={qa_f1:.3f}",
        )
    if r >= 0.45 and p <= 0.12 and tokens >= 260 and density <= 0.9:
        return (
            "evidence_density_bottleneck",
            f"rendered_sf_R={r:.3f}, rendered_sf_P={p:.3f}, rendered_tokens={tokens:.1f}, density={density:.3f}",
        )
    if r >= 0.55 and p >= 0.16 and prompt <= 750 and qa_low:
        return (
            "generation_bottleneck",
            f"rendered_sf_R={r:.3f}, rendered_sf_P={p:.3f}, prompt_tokens={prompt:.1f}, F1={qa_f1:.3f}",
        )
    if r >= 0.45 and prompt <= 900 and qa_low:
        return (
            "prompt_usability_bottleneck",
            f"retrieval appears usable (R={r:.3f}) but QA F1 remains low ({qa_f1:.3f})",
        )
    return (
        "balanced_or_unclear",
        f"R={r:.3f}, P={p:.3f}, tokens={tokens:.1f}, density={density:.3f}, F1={qa_f1:.3f}",
    )


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def _fmt(value: Any, nd: int = 4) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "0.0000"


def _build_phase6o_markdown(
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
    joined_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    complexity_scan_rows: Sequence[Mapping[str, Any]],
    complexity_profile_rows: Sequence[Mapping[str, Any]],
    primary_baseline_profile: str,
    teacher_reference_profile: str,
    external_baseline_profile: str,
    candidate_summary_rows: Sequence[Mapping[str, Any]],
    decision_lines: Sequence[str],
) -> str:
    joined_index = {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): dict(r) for r in joined_rows}
    frontier_index = {
        (_safe_text(r.get("dataset")), _safe_text(r.get("profile")), _safe_text(r.get("frontier_name"))): dict(r)
        for r in frontier_rows
    }
    tax_index = {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): dict(r) for r in taxonomy_rows}

    high_scan = [r for r in complexity_scan_rows if _safe_text(r.get("risk_level")) == "high"]
    high_complexity_profiles = [
        r for r in complexity_profile_rows if "high complexity" in _safe_text(r.get("risk_note")).lower()
    ]

    lines: List[str] = []
    lines.append("# Phase-6O QA-aware Bottleneck Audit")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("1. `legacy_sota` is treated as teacher/reference, not a direct optimization target.")
    lines.append(
        f"2. Primary target is QA-token Pareto improvement versus `{primary_baseline_profile}` and external baseline `{external_baseline_profile}`."
    )
    lines.append(
        "3. Evaluate whether GL-RCEDR/adaptive-support-span variants are Pareto-superior on QA F1, prompt tokens, rendered tokens, and evidence density."
    )
    lines.append("4. Audit risk that method appears dataset-specific or profile-engineered.")
    lines.append("5. Provide evidence to compress main method candidates to 1-2 profiles.")
    lines.append("")

    lines.append("## 2. Method Complexity and Heuristic Risk")
    lines.append("")
    lines.append("| item | finding | risk | recommendation |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| dataset-specific branch scan | {len(complexity_scan_rows)} matches, high-risk={len(high_scan)} | "
        f"{'high' if len(high_scan) > 0 else 'low'} | keep dataset-independent method logic; keep `{teacher_reference_profile}` as reference only |"
    )
    lines.append(
        f"| unified profile complexity | profiles={len(complexity_profile_rows)}, high-complexity={len(high_complexity_profiles)} | "
        f"{'medium' if len(high_complexity_profiles) > 0 else 'low'} | restrict main narrative to 1-2 primary candidates + appendix ablations |"
    )
    lines.append(
        "| reviewer framing | many similarly named variants can look like heuristic search | medium | emphasize query/evidence-adaptive signals with fixed global rules |"
    )
    lines.append("")

    lines.append("## 2.1 Candidate Compression Summary")
    lines.append("")
    lines.append(
        f"| profile | primary_gate_pass_count | qa_prompt_pareto_count | qa_rendered_pareto_count | mean_delta_F1_vs_{primary_baseline_profile} | mean_prompt_token_reduction_vs_{primary_baseline_profile} | mean_rendered_token_reduction_vs_{primary_baseline_profile} |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in candidate_summary_rows:
        profile = _safe_text(row.get("profile"))
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                profile,
                _safe_int(row.get("primary_gate_pass_count", 0), 0),
                _safe_int(row.get("qa_prompt_pareto_count", 0), 0),
                _safe_int(row.get("qa_rendered_pareto_count", 0), 0),
                _fmt(row.get("mean_delta_F1_vs_primary_baseline", 0.0)),
                _fmt(row.get("mean_prompt_token_reduction_vs_primary_baseline", 0.0), 1),
                _fmt(row.get("mean_rendered_token_reduction_vs_primary_baseline", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 3. Joined QA + Retrieval Metrics")
    lines.append("")
    lines.append("| dataset | profile | F1 | prompt_tokens | rendered_tokens | rendered_sf_R | rendered_sf_P | sf_F1/token | F1/token |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = joined_index.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("prompt_tokens_avg", 0.0), 1),
                    _fmt(row.get("rendered_tokens", 0.0), 1),
                    _fmt(row.get("rendered_sf_R", 0.0)),
                    _fmt(row.get("rendered_sf_P", 0.0)),
                    _fmt(row.get("rendered_sf_F1_per_1k_tokens", 0.0)),
                    _fmt(row.get("F1_per_1k_rendered_tokens", 0.0)),
                )
            )
    lines.append("")

    lines.append("## 4. Pareto Frontier")
    lines.append("")
    lines.append(
        f"Note: global frontier includes reference profiles (teacher `{teacher_reference_profile}` and baselines). Main decision uses Section 2.1 primary-target gate."
    )
    lines.append("")
    lines.append("| dataset | profile | F1 | prompt_tokens | rendered_tokens | pareto_status | note |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    target_frontier = "qa_f1_vs_prompt_tokens"
    for dataset in datasets:
        for profile in profiles:
            row = joined_index.get((dataset, profile), {})
            fr = frontier_index.get((dataset, profile, target_frontier), {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("prompt_tokens_avg", 0.0), 1),
                    _fmt(row.get("rendered_tokens", 0.0), 1),
                    _safe_text(fr.get("pareto_status", "incomplete_metrics")),
                    _safe_text(fr.get("note", "")),
                )
            )
    lines.append("")

    lines.append("## 5. Retrieval-QA Mismatch")
    lines.append("")
    lines.append("| dataset | profile | retrieval_strength | QA_strength | mismatch_type |")
    lines.append("|---|---|---|---|---|")
    for dataset in datasets:
        for profile in profiles:
            row = joined_index.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _safe_text(row.get("retrieval_strength", "unknown")),
                    _safe_text(row.get("qa_strength", "unknown")),
                    _safe_text(row.get("retrieval_QA_mismatch_type", "incomplete_metrics")),
                )
            )
    lines.append("")

    lines.append("## 6. Bottleneck Taxonomy")
    lines.append("")
    lines.append("| dataset | profile | dominant_bottleneck | evidence |")
    lines.append("|---|---|---|---|")
    for dataset in datasets:
        for profile in profiles:
            row = tax_index.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _safe_text(row.get("dominant_bottleneck", "unknown")),
                    _safe_text(row.get("evidence", "")),
                )
            )
    lines.append("")

    lines.append("## 7. Decision")
    lines.append("")
    lines.append("Should we:")
    lines.append("1. continue adaptive support span refinement,")
    lines.append("2. switch to prompt/evidence ordering,")
    lines.append("3. return to local selection/global seed selection,")
    lines.append("4. simplify method and stop adding profiles?")
    lines.append("")
    for line in decision_lines:
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## 8. Recommendation")
    lines.append("")
    lines.append("- Use QA-aware Pareto status plus mismatch taxonomy as the primary decision signal.")
    lines.append(
        f"- Optimize for Pareto superiority against `{primary_baseline_profile}`; use `{teacher_reference_profile}` only as teacher/reference context."
    )
    lines.append("- Treat retrieval-only density gains as supportive, not decisive, when QA diverges.")
    lines.append("")
    return "\n".join(lines)


def _decision_recommendation(
    *,
    datasets: Sequence[str],
    joined_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
    candidate_summary_rows: Sequence[Mapping[str, Any]],
    primary_baseline_profile: str,
    teacher_reference_profile: str,
    external_baseline_profile: str,
    high_risk_dataset_branch_count: int,
) -> List[str]:
    lines: List[str] = []
    best_rows = list(candidate_summary_rows[:2])
    if best_rows:
        shortlist = ", ".join(_safe_text(r.get("profile")) for r in best_rows if _safe_text(r.get("profile")))
    else:
        shortlist = ""

    if best_rows and bool(best_rows[0].get("primary_gate_all", False)):
        lines.append(
            f"Primary target met: at least one non-reference profile achieves QA-token Pareto gains vs `{primary_baseline_profile}` across datasets."
        )
        lines.append(
            f"Compress main method candidates to 1-2 profiles: {shortlist}."
        )
    else:
        lines.append(
            f"No candidate consistently satisfies QA+token superiority over `{primary_baseline_profile}` across datasets."
        )
        lines.append("Prioritize bottleneck-focused refinement before adding new profile variants.")

    mismatch_count = sum(
        1 for row in joined_rows if _safe_text(row.get("retrieval_QA_mismatch_type")) == "retrieval_strong_qa_weak"
    )
    if mismatch_count >= max(2, len(datasets)):
        lines.append("Observed retrieval-QA mismatch indicates prompt/evidence ordering or answer instruction bottleneck.")

    if high_risk_dataset_branch_count > 0:
        lines.append("Dataset-conditional branch risk detected; remove or isolate to keep reviewer-safe framing.")
    else:
        lines.append("No high-risk dataset-specific branch in audited scope; method remains dataset-agnostic.")

    lines.append(
        f"`{teacher_reference_profile}` remains teacher/reference only, while `{external_baseline_profile}` is used for contextual comparison."
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-6O QA-aware bottleneck audit")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--qa-root", required=True, help="Path to QA runs root (contains */run_manifest.json)")
    parser.add_argument("--retrieval-audit-root", required=True, help="Path to retrieval audit root")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument(
        "--primary-baseline-profile",
        default=PRIMARY_BASELINE_PROFILE_DEFAULT,
        help="Primary optimization baseline profile (default: unified_large).",
    )
    parser.add_argument(
        "--teacher-reference-profile",
        default=TEACHER_REFERENCE_PROFILE_DEFAULT,
        help="Teacher/reference profile (default: legacy_sota).",
    )
    parser.add_argument(
        "--external-baseline-profile",
        default=EXTERNAL_BASELINE_PROFILE_DEFAULT,
        help="External comparison baseline profile (default: unified_gl_rcedr_v1).",
    )
    parser.add_argument("--method-complexity-dir", required=True, help="Path from audit_method_complexity.py outputs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    datasets = _tokens(args.datasets)
    profiles = _tokens(args.profiles)
    qa_root = Path(args.qa_root).resolve()
    retrieval_root = Path(args.retrieval_audit_root).resolve()
    method_complexity_dir = Path(args.method_complexity_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    primary_baseline_profile = _safe_text(args.primary_baseline_profile) or PRIMARY_BASELINE_PROFILE_DEFAULT
    teacher_reference_profile = _safe_text(args.teacher_reference_profile) or TEACHER_REFERENCE_PROFILE_DEFAULT
    external_baseline_profile = _safe_text(args.external_baseline_profile) or EXTERNAL_BASELINE_PROFILE_DEFAULT
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_raw_rows = _collect_qa_summary_rows(qa_root, datasets)
    qa_rows = _summarize_qa_rows(qa_rows=qa_raw_rows, datasets=datasets, profiles=profiles)
    retrieval_csv = _find_retrieval_aggregate_csv(retrieval_root)
    retrieval_rows = _read_csv(retrieval_csv)
    retrieval_index = _prepare_retrieval_index(retrieval_rows)
    merged_rows = _merge_qa_and_retrieval(qa_rows=qa_rows, retrieval_index=retrieval_index)
    _annotate_baseline_deltas(
        rows=merged_rows,
        datasets=datasets,
        primary_baseline_profile=primary_baseline_profile,
        teacher_reference_profile=teacher_reference_profile,
        external_baseline_profile=external_baseline_profile,
    )

    # Add mismatch + taxonomy annotations.
    medians_by_dataset = {dataset: _dataset_medians(merged_rows, dataset) for dataset in datasets}
    taxonomy_rows: List[Dict[str, Any]] = []
    for row in merged_rows:
        dataset = _safe_text(row.get("dataset"))
        retrieval_strength, qa_strength, mismatch = classify_retrieval_qa_mismatch(
            row, medians=medians_by_dataset.get(dataset, {})
        )
        bottleneck, evidence = classify_dominant_bottleneck(row)
        row["retrieval_strength"] = retrieval_strength
        row["qa_strength"] = qa_strength
        row["retrieval_QA_mismatch_type"] = mismatch
        row["dominant_bottleneck"] = bottleneck
        row["bottleneck_evidence"] = evidence
        taxonomy_rows.append(
            {
                "dataset": dataset,
                "profile": _safe_text(row.get("profile")),
                "dominant_bottleneck": bottleneck,
                "evidence": evidence,
                "retrieval_QA_mismatch_type": mismatch,
            }
        )

    frontier_rows = compute_frontier_status_rows(
        merged_rows=merged_rows,
        datasets=datasets,
        profiles=profiles,
        reference_profiles=tuple(REFERENCE_PROFILES),
    )
    candidate_summary_rows = _build_candidate_compression_summary(
        rows=merged_rows,
        frontier_rows=frontier_rows,
        datasets=datasets,
        primary_baseline_profile=primary_baseline_profile,
        teacher_reference_profile=teacher_reference_profile,
        external_baseline_profile=external_baseline_profile,
    )
    frontier_lookup = {
        (_safe_text(r.get("dataset")), _safe_text(r.get("profile")), _safe_text(r.get("frontier_name"))): _safe_text(
            r.get("pareto_status", "incomplete_metrics")
        )
        for r in frontier_rows
    }
    for row in merged_rows:
        row["QA_token_pareto_status"] = frontier_lookup.get(
            (_safe_text(row.get("dataset")), _safe_text(row.get("profile")), "qa_f1_vs_prompt_tokens"),
            "incomplete_metrics",
        )

    # Bring method-complexity outputs into phase root.
    dataset_branch_scan_src = method_complexity_dir / "dataset_branch_scan.csv"
    profile_complexity_src = method_complexity_dir / "profile_complexity.csv"
    dataset_branch_scan_rows = _read_csv(dataset_branch_scan_src) if dataset_branch_scan_src.exists() else []
    profile_complexity_rows = _read_csv(profile_complexity_src) if profile_complexity_src.exists() else []
    _copy_if_exists(dataset_branch_scan_src, output_dir / "dataset_branch_scan.csv")
    _copy_if_exists(profile_complexity_src, output_dir / "profile_complexity.csv")
    _copy_if_exists(method_complexity_dir / "METHOD_COMPLEXITY_AUDIT.md", output_dir / "METHOD_COMPLEXITY_AUDIT.md")

    decision_lines = _decision_recommendation(
        datasets=datasets,
        joined_rows=merged_rows,
        frontier_rows=frontier_rows,
        candidate_summary_rows=candidate_summary_rows,
        primary_baseline_profile=primary_baseline_profile,
        teacher_reference_profile=teacher_reference_profile,
        external_baseline_profile=external_baseline_profile,
        high_risk_dataset_branch_count=sum(
            1 for row in dataset_branch_scan_rows if _safe_text(row.get("risk_level", "")) == "high"
        ),
    )

    joined_csv = output_dir / "qa_retrieval_joined_metrics.csv"
    frontier_csv = output_dir / "qa_pareto_frontier.csv"
    taxonomy_csv = output_dir / "bottleneck_taxonomy.csv"
    candidate_summary_csv = output_dir / "candidate_compression_summary.csv"
    report_md = output_dir / "PHASE6O_QA_AWARE_BOTTLENECK_AUDIT.md"

    _write_csv(
        joined_csv,
        merged_rows,
        [
            "dataset",
            "profile",
            "qa_num_runs",
            "qa_num_queries",
            "qa_available",
            "retrieval_available",
            "EM",
            "F1",
            "prompt_tokens_avg",
            "completion_tokens_avg",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
            "sf_F1_per_1k_prompt_tokens",
            "rendered_sf_R",
            "rendered_sf_P",
            "rendered_sf_F1",
            "rendered_sf_F1_per_1k_tokens",
            "rendered_tokens",
            "avg_item_tokens",
            "chunk_like_rate",
            "render_drop_rate",
            "generation_ready_rate",
            "F1_per_1k_prompt_tokens",
            "F1_per_1k_rendered_tokens",
            "QA_token_pareto_status",
            "retrieval_strength",
            "qa_strength",
            "retrieval_QA_mismatch_type",
            "dominant_bottleneck",
            "bottleneck_evidence",
            "primary_baseline_profile",
            "teacher_reference_profile",
            "external_baseline_profile",
            "delta_F1_vs_primary_baseline",
            "delta_prompt_tokens_vs_primary_baseline",
            "delta_rendered_tokens_vs_primary_baseline",
            "delta_F1_vs_teacher_reference",
            "delta_prompt_tokens_vs_teacher_reference",
            "delta_rendered_tokens_vs_teacher_reference",
            "delta_F1_vs_external_baseline",
            "delta_prompt_tokens_vs_external_baseline",
            "delta_rendered_tokens_vs_external_baseline",
        ],
    )
    _write_csv(
        frontier_csv,
        frontier_rows,
        [
            "dataset",
            "profile",
            "frontier_name",
            "maximize",
            "minimize",
            "pareto_status",
            "note",
            "F1",
            "prompt_tokens_avg",
            "rendered_tokens",
            "F1_per_1k_prompt_tokens",
            "rendered_sf_R",
            "rendered_sf_F1_per_1k_tokens",
        ],
    )
    _write_csv(
        taxonomy_csv,
        taxonomy_rows,
        ["dataset", "profile", "dominant_bottleneck", "evidence", "retrieval_QA_mismatch_type"],
    )
    _write_csv(
        candidate_summary_csv,
        candidate_summary_rows,
        [
            "profile",
            "dataset_count",
            "primary_gate_pass_count",
            "primary_gate_all",
            "qa_prompt_pareto_count",
            "qa_rendered_pareto_count",
            "qa_prompt_pareto_all",
            "qa_rendered_pareto_all",
            "mean_delta_F1_vs_primary_baseline",
            "mean_prompt_token_reduction_vs_primary_baseline",
            "mean_rendered_token_reduction_vs_primary_baseline",
        ],
    )

    report_md.write_text(
        _build_phase6o_markdown(
            datasets=datasets,
            profiles=profiles,
            joined_rows=merged_rows,
            frontier_rows=frontier_rows,
            taxonomy_rows=taxonomy_rows,
            complexity_scan_rows=dataset_branch_scan_rows,
            complexity_profile_rows=profile_complexity_rows,
            primary_baseline_profile=primary_baseline_profile,
            teacher_reference_profile=teacher_reference_profile,
            external_baseline_profile=external_baseline_profile,
            candidate_summary_rows=candidate_summary_rows,
            decision_lines=decision_lines,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "datasets": list(datasets),
        "profiles": list(profiles),
        "qa_root": str(qa_root),
        "retrieval_audit_root": str(retrieval_root),
        "retrieval_aggregate_csv": str(retrieval_csv),
        "method_complexity_dir": str(method_complexity_dir),
        "output_dir": str(output_dir),
        "primary_baseline_profile": primary_baseline_profile,
        "teacher_reference_profile": teacher_reference_profile,
        "external_baseline_profile": external_baseline_profile,
        "num_qa_rows_raw": len(qa_raw_rows),
        "num_joined_rows": len(merged_rows),
        "num_frontier_rows": len(frontier_rows),
        "num_taxonomy_rows": len(taxonomy_rows),
        "num_candidate_summary_rows": len(candidate_summary_rows),
    }
    (output_dir / "phase6o_qa_aware_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
