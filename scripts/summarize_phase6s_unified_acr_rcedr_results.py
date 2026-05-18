#!/usr/bin/env python3
"""Summarize Phase-6S outputs with strict manifest/attempt accounting."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]
DEFAULT_PROFILES = [
    "legacy_sota",
    "unified_large",
    "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
    "unified_acr_v11",
    "unified_acr_v11_beam3",
    "unified_acr_rcedr_v12",
    "unified_acr_rcedr_v12_answerability_only",
    "unified_acr_rcedr_v12_no_bridge",
    "unified_acr_rcedr_v12_no_redundancy",
    "unified_acr_rcedr_v12_no_sentence_rerank",
    "unified_acr_rcedr_v12_beam3",
    "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
]

CORE_EXPECTED_MATRIX = {
    "hotpotqa": [
        "legacy_sota",
        "unified_large",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
        "unified_acr_v11",
        "unified_acr_v11_beam3",
        "unified_acr_rcedr_v12",
        "unified_acr_rcedr_v12_answerability_only",
        "unified_acr_rcedr_v12_no_bridge",
        "unified_acr_rcedr_v12_no_redundancy",
        "unified_acr_rcedr_v12_no_sentence_rerank",
        "unified_acr_rcedr_v12_beam3",
        "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
    ],
    "2wikimultihopqa": [
        "legacy_sota",
        "unified_large",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
        "unified_acr_v11",
        "unified_acr_v11_beam3",
        "unified_acr_rcedr_v12",
        "unified_acr_rcedr_v12_answerability_only",
        "unified_acr_rcedr_v12_no_bridge",
        "unified_acr_rcedr_v12_no_redundancy",
        "unified_acr_rcedr_v12_no_sentence_rerank",
        "unified_acr_rcedr_v12_beam3",
        "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
    ],
    "popqa": [
        "unified_large",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
        "unified_acr_rcedr_v12",
        "unified_acr_rcedr_v12_beam3",
        "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
    ],
    "musique": [
        "unified_large",
        "unified_acr_rcedr_v12",
        "unified_acr_rcedr_v12_beam3",
        "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
    ],
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


def _tokens(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _fmt(value: Any, nd: int = 4) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = _safe_text(line)
                if not row:
                    continue
                try:
                    payload = json.loads(row)
                except Exception:
                    continue
                if isinstance(payload, Mapping):
                    out.append(dict(payload))
    except Exception:
        return []
    return out


def _load_methodology_audit(out_root: Path) -> Dict[str, Any]:
    path = out_root / "phase6s_methodology_audit.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _extract_stage_means(query_jsonl_path: Path) -> Dict[str, float]:
    sentence_rerank_ms: List[float] = []
    top1_correction_ms: List[float] = []
    gl_rcedr_ms: List[float] = []
    answerability_selection_ms: List[float] = []
    unified_acr_rcedr_ms: List[float] = []
    retrieval_latency_ms: List[float] = []

    for row in _iter_jsonl(query_jsonl_path):
        retrieval = dict(row.get("retrieval", {}) or {})
        diagnostics = dict(retrieval.get("diagnostics", {}) or {})
        stage_ms = dict(diagnostics.get("latency_breakdown_ms", {}) or {})
        gl_diag = dict(diagnostics.get("gl_rcedr_diag", {}) or {})
        acr_diag = dict(diagnostics.get("answerability_selection_diag", {}) or {})
        unified_diag = dict(diagnostics.get("unified_acr_rcedr_diag", {}) or {})

        sentence_rerank_ms.append(_safe_float(stage_ms.get("sentence_rerank_ms", 0.0), 0.0))
        top1_correction_ms.append(_safe_float(stage_ms.get("top1_correction_ms", 0.0), 0.0))
        retrieval_latency_ms.append(_safe_float(retrieval.get("latency_ms", 0.0), 0.0))
        gl_rcedr_ms.append(_safe_float(gl_diag.get("selection_ms", gl_diag.get("selection_time_ms", 0.0)), 0.0))
        answerability_selection_ms.append(
            _safe_float(acr_diag.get("selection_ms", acr_diag.get("selection_time_ms", 0.0)), 0.0)
        )
        unified_acr_rcedr_ms.append(
            _safe_float(unified_diag.get("selection_ms", unified_diag.get("selection_time_ms", 0.0)), 0.0)
        )

    return {
        "sentence_rerank_ms": _mean(sentence_rerank_ms),
        "top1_correction_ms": _mean(top1_correction_ms),
        "gl_rcedr_ms": _mean(gl_rcedr_ms),
        "answerability_selection_ms": _mean(answerability_selection_ms),
        "unified_acr_rcedr_ms": _mean(unified_acr_rcedr_ms),
        "retrieval_pipeline_latency_ms": _mean(retrieval_latency_ms),
    }


def _profile_role(profile: str) -> str:
    mapping = {
        "legacy_sota": "heuristic/reference",
        "unified_large": "primary baseline",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge": "compact GL reference",
        "unified_acr_v11": "cascade ACR main",
        "unified_acr_v11_beam3": "cascade ACR beam3",
        "unified_acr_rcedr_v12": "unified selector main",
        "unified_acr_rcedr_v12_answerability_only": "ablation",
        "unified_acr_rcedr_v12_no_bridge": "ablation",
        "unified_acr_rcedr_v12_no_redundancy": "ablation",
        "unified_acr_rcedr_v12_no_sentence_rerank": "ablation",
        "unified_acr_rcedr_v12_beam3": "ablation",
        "unified_acr_rcedr_v12_beam3_no_sentence_rerank": "ablation",
    }
    return mapping.get(profile, "candidate")


def _build_completed_rows(report: Mapping[str, Any], datasets: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    allowed = set(datasets)
    for run in list(report.get("completed_runs", []) or []):
        dataset = _safe_text(run.get("dataset"))
        profile = _safe_text(run.get("profile"))
        run_name = _safe_text(run.get("run_name"))
        if dataset not in allowed:
            continue
        summary = dict(run.get("summary", {}) or {})
        summary_path = Path(_safe_text(run.get("summary_path")))
        query_path = Path(_safe_text(run.get("query_path")))
        stage_means = _extract_stage_means(query_path)

        em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
        f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
        prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
        completion_tokens = _safe_float(summary.get("completion_tokens_avg", 0.0), 0.0)
        retrieval_ms = _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0)
        generation_ms = _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0)
        total_ms = _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0)
        qa_queries = _safe_int(summary.get("qa_executed_samples", 0), 0)

        rows.append(
            {
                "run_name": run_name,
                "dataset": dataset,
                "profile": profile,
                "EM": em,
                "F1": f1,
                "avg_context_tokens": prompt_tokens,
                "prompt_tokens_avg": prompt_tokens,
                "completion_tokens_avg": completion_tokens,
                "F1_per_1k_context_tokens": (float(f1 * 1000.0) / prompt_tokens) if prompt_tokens > 0 else 0.0,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "qa_num_queries": qa_queries,
                "qa_available": bool(qa_queries > 0),
                "summary_path": str(summary_path),
                "query_path": str(query_path),
                "attempt_count": _safe_int(run.get("attempt_count", 0), 0),
                "complete_attempt_count": _safe_int(run.get("complete_attempt_count", 0), 0),
                "incomplete_attempt_count": _safe_int(run.get("incomplete_attempt_count", 0), 0),
                **stage_means,
            }
        )

    rows.sort(key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("profile")), _safe_text(x.get("run_name"))))
    return rows


def _rows_for_table(
    completed_rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in completed_rows]
    if datasets is not None:
        wanted = set(datasets)
        rows = [row for row in rows if _safe_text(row.get("dataset")) in wanted]
    if profiles is not None:
        wanted = set(profiles)
        rows = [row for row in rows if _safe_text(row.get("profile")) in wanted]
    rows.sort(key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("profile")), _safe_text(x.get("run_name"))))
    return rows


def _main_comparison_rows(completed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = _rows_for_table(
        completed_rows,
        datasets=["hotpotqa", "2wikimultihopqa"],
        profiles=[
            "unified_large",
            "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
            "unified_acr_v11",
            "unified_acr_v11_beam3",
            "unified_acr_rcedr_v12",
            "unified_acr_rcedr_v12_answerability_only",
            "unified_acr_rcedr_v12_no_bridge",
            "unified_acr_rcedr_v12_no_redundancy",
            "unified_acr_rcedr_v12_no_sentence_rerank",
            "unified_acr_rcedr_v12_beam3",
            "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
        ],
    )
    baseline = {
        _safe_text(r.get("dataset")): dict(r)
        for r in rows
        if _safe_text(r.get("profile")) == "unified_large"
    }
    out: List[Dict[str, Any]] = []
    for row in rows:
        dataset = _safe_text(row.get("dataset"))
        base = baseline.get(dataset)
        if base is None:
            delta_f1 = None
            delta_tok = None
            delta_f1k = None
        else:
            delta_f1 = _safe_float(row.get("F1", 0.0), 0.0) - _safe_float(base.get("F1", 0.0), 0.0)
            delta_tok = _safe_float(row.get("avg_context_tokens", 0.0), 0.0) - _safe_float(
                base.get("avg_context_tokens", 0.0), 0.0
            )
            delta_f1k = _safe_float(row.get("F1_per_1k_context_tokens", 0.0), 0.0) - _safe_float(
                base.get("F1_per_1k_context_tokens", 0.0), 0.0
            )
        out.append(
            {
                **dict(row),
                "delta_F1_vs_unified_large": delta_f1,
                "delta_tokens_vs_unified_large": delta_tok,
                "delta_F1_per_1k_vs_unified_large": delta_f1k,
            }
        )
    return out


def _completeness_rows(
    *,
    report: Mapping[str, Any],
    completed_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
) -> List[Dict[str, Any]]:
    status_rows = list(report.get("statuses", []) or [])

    status_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in status_rows:
        pair = (_safe_text(row.get("dataset")), _safe_text(row.get("profile")))
        status_by_pair.setdefault(pair, []).append(dict(row))

    completed_pairs = {
        (_safe_text(row.get("dataset")), _safe_text(row.get("profile")))
        for row in completed_rows
    }
    scheduled_pairs = {
        (_safe_text(row.get("dataset")), _safe_text(row.get("profile")))
        for row in list(report.get("scheduled_runs", []) or [])
        if _safe_text(row.get("dataset")) and _safe_text(row.get("profile"))
    }

    def _priority(status: str) -> int:
        order = {
            "parse_error": 0,
            "scheduled_but_incomplete": 1,
            "scheduled_but_missing": 2,
            "completed": 3,
        }
        return order.get(status, 9)

    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for profile in list(CORE_EXPECTED_MATRIX.get(dataset, []) or []):
            pair = (dataset, profile)
            if pair in completed_pairs:
                rows.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "status": "completed",
                        "reason": "latest_complete_attempt",
                    }
                )
                continue
            if pair in scheduled_pairs:
                candidates = sorted(
                    [
                        r
                        for r in status_by_pair.get(pair, [])
                        if _safe_text(r.get("status")) in {"parse_error", "scheduled_but_incomplete", "scheduled_but_missing"}
                    ],
                    key=lambda x: _priority(_safe_text(x.get("status"))),
                )
                chosen = candidates[0] if candidates else {"status": "scheduled_but_missing", "reason": "scheduled_missing_status_row"}
                rows.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "status": _safe_text(chosen.get("status")),
                        "reason": _safe_text(chosen.get("reason")),
                    }
                )
            else:
                rows.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "status": "not_scheduled",
                        "reason": "not in run manifest",
                    }
                )

    rows.sort(key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("profile"))))
    return rows


def _artifact_warning_rows(report: Mapping[str, Any], completed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(r) for r in list(report.get("artifact_warnings", []) or [])]

    for warning in list(report.get("schedule_warnings", []) or []):
        rows.append(
            {
                "run_name": "-",
                "dataset": "-",
                "profile": "-",
                "status": "schedule_warning",
                "attempt_count": 0,
                "complete_attempt_count": 0,
                "incomplete_attempt_count": 0,
                "path": _safe_text(warning),
            }
        )
    for err in list(report.get("manifest_errors", []) or []):
        rows.append(
            {
                "run_name": "-",
                "dataset": "-",
                "profile": "-",
                "status": "manifest_error",
                "attempt_count": 0,
                "complete_attempt_count": 0,
                "incomplete_attempt_count": 0,
                "path": _safe_text(err),
            }
        )

    rows.sort(key=lambda x: (_safe_text(x.get("status")), _safe_text(x.get("run_name"))))
    return rows


def _result_warning_rows(completed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in completed_rows:
        run_name = _safe_text(row.get("run_name"))
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))

        if not bool(row.get("qa_available", False)):
            rows.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "warning": "qa_available_false",
                    "value": _safe_int(row.get("qa_num_queries", 0), 0),
                    "path": _safe_text(row.get("summary_path")),
                }
            )
        if _safe_float(row.get("avg_context_tokens", 0.0), 0.0) <= 0.0:
            rows.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "warning": "empty_context",
                    "value": _safe_float(row.get("avg_context_tokens", 0.0), 0.0),
                    "path": _safe_text(row.get("summary_path")),
                }
            )
        if _safe_float(row.get("retrieval_ms", 0.0), 0.0) <= 0.0:
            rows.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "warning": "zero_retrieval_ms",
                    "value": _safe_float(row.get("retrieval_ms", 0.0), 0.0),
                    "path": _safe_text(row.get("summary_path")),
                }
            )
        if _safe_float(row.get("generation_ms", 0.0), 0.0) <= 0.0:
            rows.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "warning": "zero_generation_ms",
                    "value": _safe_float(row.get("generation_ms", 0.0), 0.0),
                    "path": _safe_text(row.get("summary_path")),
                }
            )
        if _safe_float(row.get("total_ms", 0.0), 0.0) <= 0.0:
            rows.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "warning": "zero_total_ms",
                    "value": _safe_float(row.get("total_ms", 0.0), 0.0),
                    "path": _safe_text(row.get("summary_path")),
                }
            )

    rows.sort(key=lambda x: (_safe_text(x.get("warning")), _safe_text(x.get("run_name"))))
    return rows


def _markdown(
    *,
    report: Mapping[str, Any],
    completed_rows: Sequence[Mapping[str, Any]],
    completeness_rows: Sequence[Mapping[str, Any]],
    artifact_warning_rows: Sequence[Mapping[str, Any]],
    result_warning_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    profiles: Sequence[str],
    methodology_audit: Mapping[str, Any],
) -> str:
    executed_rows = _rows_for_table(completed_rows, datasets=datasets)
    main_rows = _main_comparison_rows(completed_rows)
    ablation_rows = _rows_for_table(
        completed_rows,
        datasets=["hotpotqa", "2wikimultihopqa", "popqa"],
        profiles=[
            "unified_acr_rcedr_v12",
            "unified_acr_rcedr_v12_answerability_only",
            "unified_acr_rcedr_v12_no_bridge",
            "unified_acr_rcedr_v12_no_redundancy",
            "unified_acr_rcedr_v12_no_sentence_rerank",
            "unified_acr_rcedr_v12_beam3",
            "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
        ],
    )
    popqa_rows = _rows_for_table(completed_rows, datasets=["popqa"])
    musique_rows = _rows_for_table(completed_rows, datasets=["musique"])
    counts = dict(report.get("counts", {}) or {})

    lines: List[str] = []
    lines.append("# Phase-6S Unified ACR-RCEDR Summary")
    lines.append("")
    lines.append("## 1. Executed QA Results")
    lines.append("")
    lines.append(
        "| dataset | profile | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | retrieval_ms | generation_ms | total_ms | run_name |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in executed_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("EM", 0.0)),
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_context_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_context_tokens", 0.0)),
                _fmt(row.get("retrieval_ms", 0.0), 1),
                _fmt(row.get("generation_ms", 0.0), 1),
                _fmt(row.get("total_ms", 0.0), 1),
                _safe_text(row.get("run_name")),
            )
        )
    lines.append("")

    lines.append("## 2. Main HotpotQA/2Wiki Comparison")
    lines.append("")
    lines.append(
        "| dataset | profile | F1 | avg_context_tokens | F1_per_1k_context_tokens | delta_F1_vs_unified_large | delta_tokens_vs_unified_large | delta_F1_per_1k_vs_unified_large |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in main_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_context_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_context_tokens", 0.0)),
                _fmt(row.get("delta_F1_vs_unified_large", "n/a")),
                _fmt(row.get("delta_tokens_vs_unified_large", "n/a"), 1),
                _fmt(row.get("delta_F1_per_1k_vs_unified_large", "n/a")),
            )
        )
    lines.append("")

    lines.append("## 3. Ablation Results")
    lines.append("")
    lines.append("| dataset | profile | F1 | avg_context_tokens | F1_per_1k_context_tokens | retrieval_ms |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in ablation_rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_context_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_context_tokens", 0.0)),
                _fmt(row.get("retrieval_ms", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 4. Robustness / Stress-Test")
    lines.append("")
    lines.append("### 4.1 PopQA")
    lines.append("")
    lines.append("| profile | F1 | avg_context_tokens | F1_per_1k_context_tokens | retrieval_ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in popqa_rows:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("profile")),
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_context_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_context_tokens", 0.0)),
                _fmt(row.get("retrieval_ms", 0.0), 1),
            )
        )
    lines.append("")
    lines.append("### 4.2 MuSiQue")
    lines.append("")
    lines.append("| profile | F1 | avg_context_tokens | retrieval_ms |")
    lines.append("|---|---:|---:|---:|")
    for row in musique_rows:
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("profile")),
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_context_tokens", 0.0), 1),
                _fmt(row.get("retrieval_ms", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 5. Completeness Check")
    lines.append("")
    lines.append("| dataset | profile | status | reason |")
    lines.append("|---|---|---|---|")
    for row in completeness_rows:
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _safe_text(row.get("status")),
                _safe_text(row.get("reason")),
            )
        )
    lines.append("")

    lines.append("## 6. Artifact Consistency")
    lines.append("")
    lines.append("| item | count |")
    lines.append("|---|---:|")
    lines.append(f"| scheduled_runs | {int(counts.get('scheduled_runs', 0))} |")
    lines.append(f"| run_names_with_attempts | {int(counts.get('run_names_with_attempts', 0))} |")
    lines.append(f"| completed_run_names | {int(counts.get('completed_run_names', 0))} |")
    lines.append(f"| incomplete_attempts | {int(counts.get('incomplete_attempts', 0))} |")
    lines.append(f"| extra_not_in_manifest | {int(counts.get('extra_not_in_manifest', 0))} |")
    lines.append(f"| multi_attempt_run_names | {int(counts.get('multi_attempt_run_names', 0))} |")
    lines.append(f"| parse_errors | {int(counts.get('parse_errors', 0))} |")
    lines.append("")

    lines.append("## 7. Artifact Warnings")
    lines.append("")
    if artifact_warning_rows:
        lines.append("| run_name | status | attempt_count | complete_attempt_count | incomplete_attempt_count | path |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in artifact_warning_rows:
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    _safe_text(row.get("run_name")),
                    _safe_text(row.get("status")),
                    _safe_int(row.get("attempt_count", 0), 0),
                    _safe_int(row.get("complete_attempt_count", 0), 0),
                    _safe_int(row.get("incomplete_attempt_count", 0), 0),
                    _safe_text(row.get("path")),
                )
            )
    else:
        lines.append("No artifact warnings.")
    lines.append("")

    lines.append("## 8. Result Warnings")
    lines.append("")
    if result_warning_rows:
        lines.append("| run_name | dataset | profile | warning | value | path |")
        lines.append("|---|---|---|---|---:|---|")
        for row in result_warning_rows:
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    _safe_text(row.get("run_name")),
                    _safe_text(row.get("dataset")),
                    _safe_text(row.get("profile")),
                    _safe_text(row.get("warning")),
                    _fmt(row.get("value", 0.0), 4),
                    _safe_text(row.get("path")),
                )
            )
    else:
        lines.append("No result warnings.")
    lines.append("")

    lines.append("## 9. Compared Profiles")
    lines.append("")
    lines.append("| profile | role |")
    lines.append("|---|---|")
    for profile in profiles:
        lines.append(f"| {profile} | {_profile_role(profile)} |")
    lines.append("")

    lines.append("## 10. Decision Checklist")
    lines.append("")
    lines.append("| dataset | method | F1 | avg_context_tokens | F1_per_1k_context_tokens | retrieval_ms | note |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    key_methods = [
        "unified_large",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
        "unified_acr_v11",
        "unified_acr_v11_beam3",
        "unified_acr_rcedr_v12",
        "unified_acr_rcedr_v12_beam3",
        "unified_acr_rcedr_v12_beam3_no_sentence_rerank",
    ]
    key_datasets = ["hotpotqa", "2wikimultihopqa", "popqa", "musique"]
    row_map = {
        (_safe_text(row.get("dataset")), _safe_text(row.get("profile"))): row
        for row in executed_rows
    }
    for dataset in key_datasets:
        for method in key_methods:
            row = row_map.get((dataset, method))
            if row is None:
                continue
            note = "completed"
            if _safe_float(row.get("avg_context_tokens", 0.0), 0.0) <= 0.0:
                note = "zero_context_warning"
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    method,
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("avg_context_tokens", 0.0), 1),
                    _fmt(row.get("F1_per_1k_context_tokens", 0.0)),
                    _fmt(row.get("retrieval_ms", 0.0), 1),
                    note,
                )
            )
    lines.append("")

    lines.append("## 11. Methodology Audit Result")
    lines.append("")
    if methodology_audit:
        lines.append("| item | value |")
        lines.append("|---|---|")
        lines.append(f"| status | {_safe_text(methodology_audit.get('status'))} |")
        lines.append(f"| main_profile | {_safe_text(methodology_audit.get('main_profile'))} |")
        lines.append(
            f"| dataset_specific_main_profile | {bool(methodology_audit.get('dataset_specific_main_profile', False))} |"
        )
        lines.append(f"| unified_selector_active | {bool(methodology_audit.get('unified_selector_active', False))} |")
        lines.append(f"| old_cascade_active | {bool(methodology_audit.get('old_cascade_active', False))} |")
        lines.append(
            f"| hard_token_budget_enabled | {bool(methodology_audit.get('hard_token_budget_enabled', False))} |"
        )
        lines.append(
            f"| repeated_cost_penalty_warning | {bool(methodology_audit.get('repeated_cost_penalty_warning', False))} |"
        )
        lines.append(
            f"| new_score_terms_detected | {bool(methodology_audit.get('new_score_terms_detected', False))} |"
        )
        lines.append(
            f"| ablation_profiles_used_as_main | {bool(methodology_audit.get('ablation_profiles_used_as_main', False))} |"
        )
        warnings = list(methodology_audit.get("warnings", []) or [])
        errors = list(methodology_audit.get("errors", []) or [])
        lines.append("")
        lines.append(f"- warnings: {len(warnings)}")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append(f"- errors: {len(errors)}")
        for e in errors:
            lines.append(f"  - {e}")
    else:
        lines.append("Methodology audit file not found: `phase6s_methodology_audit.json`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6S outputs with manifest/attempt consistency accounting.")
    parser.add_argument("--out-root", default="", help="Phase-6S output root (preferred)")
    parser.add_argument("--input-root", default="", help="qa_runs root; used if --out-root is omitted")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    out_root_text = _safe_text(args.out_root)
    input_root_text = _safe_text(args.input_root)

    if out_root_text:
        out_root = Path(out_root_text).resolve()
    else:
        if not input_root_text:
            raise SystemExit("Either --out-root or --input-root must be provided.")
        out_root = Path(input_root_text).resolve().parent

    output_md = Path(_safe_text(args.output_md) or str(out_root / "PHASE6S_UNIFIED_ACR_RCEDR_SUMMARY.md")).resolve()
    output_json = Path(_safe_text(args.output_json) or str(out_root / "phase6s_unified_acr_rcedr_summary.json")).resolve()

    datasets = _tokens(args.datasets)
    profiles = _tokens(args.profiles)

    report = collect_phase6s_artifacts(out_root)
    methodology_audit = _load_methodology_audit(out_root)
    completed_rows = _build_completed_rows(report, datasets)
    completeness_rows = _completeness_rows(report=report, completed_rows=completed_rows, datasets=datasets)
    artifact_warning_rows = _artifact_warning_rows(report, completed_rows)
    result_warning_rows = _result_warning_rows(completed_rows)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    markdown = _markdown(
        report=report,
        completed_rows=completed_rows,
        completeness_rows=completeness_rows,
        artifact_warning_rows=artifact_warning_rows,
        result_warning_rows=result_warning_rows,
        datasets=datasets,
        profiles=profiles,
        methodology_audit=methodology_audit,
    )
    output_md.write_text(markdown + "\n", encoding="utf-8")

    status_counts: Dict[str, int] = {}
    for row in completeness_rows:
        status = _safe_text(row.get("status"))
        status_counts[status] = int(status_counts.get(status, 0) + 1)

    payload: MutableMapping[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "output_md": str(output_md),
        "output_json": str(output_json),
        "schedule_source": _safe_text(report.get("schedule_source")),
        "root_manifest_exists": bool(report.get("root_manifest_exists", False)),
        "schedule_warnings": list(report.get("schedule_warnings", []) or []),
        "manifest_errors": list(report.get("manifest_errors", []) or []),
        "duplicate_run_names": list(report.get("duplicate_run_names", []) or []),
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "completed_row_count": int(len(completed_rows)),
        "artifact_warning_row_count": int(len(artifact_warning_rows)),
        "result_warning_row_count": int(len(result_warning_rows)),
        "completeness_status_counts": status_counts,
        "methodology_audit": methodology_audit,
        "completed_rows": completed_rows,
        "completeness_rows": completeness_rows,
        "artifact_warnings": artifact_warning_rows,
        "result_warnings": result_warning_rows,
        "report": report,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(str(output_md))


if __name__ == "__main__":
    main()
