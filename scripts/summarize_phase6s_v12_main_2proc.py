#!/usr/bin/env python3
"""Summarize Phase-6S v12 main 2-process outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from validate_phase6s_artifacts import collect_phase6s_artifacts


TARGET_PROFILE = "unified_acr_rcedr_v12"
TARGET_DATASETS = ["hotpotqa", "2wikimultihopqa"]


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


def _fmt(value: Any, nd: int = 4) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, Mapping):
                    rows.append(dict(obj))
    except Exception:
        return []
    return rows


def _load_methodology_audit(out_root: Path) -> Dict[str, Any]:
    p = out_root / "phase6s_methodology_audit.json"
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _extract_latency_profile(query_path: Path) -> Dict[str, Any]:
    proposal_ms: List[float] = []
    graph_ms: List[float] = []
    corridor_ms: List[float] = []
    atomization_ms: List[float] = []
    answerability_gain_ms: List[float] = []
    bridge_gain_ms: List[float] = []
    redundancy_ms: List[float] = []
    marginal_selection_ms: List[float] = []
    rendering_ms: List[float] = []

    candidate_atoms: List[float] = []
    selected_atoms: List[float] = []
    candidate_corridors: List[float] = []
    objective_eval_calls: List[float] = []

    n = 0
    for row in _iter_jsonl(query_path):
        n += 1
        retrieval = dict(row.get("retrieval", {}) or {})
        diag = dict(retrieval.get("diagnostics", {}) or {})
        latency = dict(diag.get("latency_breakdown_ms", {}) or {})
        unified_diag = dict(diag.get("unified_acr_rcedr_diag", {}) or {})
        shared_diag = dict(diag.get("shared_budget_diagnostics", {}) or {})

        proposal_ms.append(_safe_float(latency.get("proposal_time_ms", 0.0), 0.0))
        graph_ms.append(_safe_float(latency.get("proposal_subgraph_build_ms", 0.0), 0.0))
        corridor_ms.append(_safe_float(latency.get("phase2_pair_shortlist_ms", 0.0), 0.0))
        atomization_ms.append(_safe_float(unified_diag.get("atomization_ms", 0.0), 0.0))
        answerability_gain_ms.append(_safe_float(unified_diag.get("answerability_gain_ms", 0.0), 0.0))
        bridge_gain_ms.append(_safe_float(unified_diag.get("bridge_gain_ms", 0.0), 0.0))
        redundancy_ms.append(_safe_float(unified_diag.get("redundancy_scoring_ms", 0.0), 0.0))

        sel_ms = _safe_float(unified_diag.get("selection_ms", 0.0), 0.0)
        if sel_ms <= 0.0:
            sel_ms = _safe_float(latency.get("unified_acr_rcedr_ms", 0.0), 0.0)
        marginal_selection_ms.append(sel_ms)
        rendering_ms.append(
            _safe_float(
                latency.get(
                    "render_ms",
                    latency.get("final_render_time_ms", 0.0),
                ),
                0.0,
            )
        )

        candidate_atoms.append(_safe_float(unified_diag.get("num_atoms", 0.0), 0.0))
        selected_atoms.append(_safe_float(unified_diag.get("num_selected_atoms", 0.0), 0.0))
        objective_eval_calls.append(_safe_float(unified_diag.get("objective_eval_calls", 0.0), 0.0))

        corridor_count = _safe_float(
            shared_diag.get(
                "candidate_corridor_count",
                shared_diag.get("corridor_count", 0.0),
            ),
            0.0,
        )
        candidate_corridors.append(corridor_count)

    return {
        "num_queries": n,
        "timing_ms_avg": {
            "candidate_proposal": _mean(proposal_ms),
            "local_graph_construction": _mean(graph_ms),
            "corridor_feature_extraction": _mean(corridor_ms),
            "evidence_atomization": _mean(atomization_ms),
            "answerability_gain_scoring": _mean(answerability_gain_ms),
            "bridge_gain_scoring": _mean(bridge_gain_ms),
            "redundancy_scoring": _mean(redundancy_ms),
            "marginal_selection": _mean(marginal_selection_ms),
            "rendering": _mean(rendering_ms),
        },
        "counts_avg": {
            "candidate_atoms": _mean(candidate_atoms),
            "selected_atoms": _mean(selected_atoms),
            "candidate_corridors": _mean(candidate_corridors),
            "objective_eval_calls": _mean(objective_eval_calls),
        },
    }


def _build_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    scheduled = list(report.get("scheduled_runs", []) or [])
    completed = {
        _safe_text(r.get("run_name")): r
        for r in list(report.get("completed_runs", []) or [])
        if _safe_text(r.get("run_name"))
    }

    rows: List[Dict[str, Any]] = []
    for run in scheduled:
        run_name = _safe_text(run.get("run_name"))
        profile = _safe_text(run.get("profile"))
        dataset = _safe_text(run.get("dataset"))
        if profile != TARGET_PROFILE or dataset not in TARGET_DATASETS:
            continue
        done = completed.get(run_name)
        if done is None:
            continue
        summary = dict(done.get("summary", {}) or {})
        query_path = Path(_safe_text(done.get("query_path")))
        latency_profile = _extract_latency_profile(query_path)

        em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
        f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
        context_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
        completion_tokens = _safe_float(summary.get("completion_tokens_avg", 0.0), 0.0)
        retrieval_ms = _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0)
        generation_ms = _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0)
        total_ms = _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0)
        qa_n = _safe_int(summary.get("qa_executed_samples", 0), 0)

        rows.append(
            {
                "run_name": run_name,
                "dataset": dataset,
                "profile": profile,
                "EM": em,
                "F1": f1,
                "avg_context_tokens": context_tokens,
                "completion_tokens_avg": completion_tokens,
                "F1_per_1k_context_tokens": (f1 * 1000.0 / context_tokens) if context_tokens > 0 else 0.0,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "qa_num_queries": qa_n,
                "summary_path": _safe_text(done.get("summary_path")),
                "query_path": _safe_text(done.get("query_path")),
                "latency_profile": latency_profile,
            }
        )

    rows.sort(key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("run_name"))))
    return rows


def _artifact_warning_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(report.get("artifact_warnings", []) or []):
        run_name = _safe_text(row.get("run_name"))
        if not run_name:
            continue
        out.append(dict(row))
    return out


def _result_warning_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    for row in rows:
        if _safe_float(row.get("avg_context_tokens", 0.0), 0.0) <= 0:
            warnings.append(
                {
                    "run_name": _safe_text(row.get("run_name")),
                    "dataset": _safe_text(row.get("dataset")),
                    "profile": _safe_text(row.get("profile")),
                    "warning": "empty_context",
                    "value": _safe_float(row.get("avg_context_tokens", 0.0), 0.0),
                    "path": _safe_text(row.get("summary_path")),
                }
            )
        if _safe_float(row.get("retrieval_ms", 0.0), 0.0) <= 0:
            warnings.append(
                {
                    "run_name": _safe_text(row.get("run_name")),
                    "dataset": _safe_text(row.get("dataset")),
                    "profile": _safe_text(row.get("profile")),
                    "warning": "nonpositive_retrieval_ms",
                    "value": _safe_float(row.get("retrieval_ms", 0.0), 0.0),
                    "path": _safe_text(row.get("summary_path")),
                }
            )
    return warnings


def _markdown(
    *,
    rows: List[Dict[str, Any]],
    report: Mapping[str, Any],
    artifact_warnings: List[Dict[str, Any]],
    result_warnings: List[Dict[str, Any]],
    methodology_audit: Mapping[str, Any],
) -> str:
    counts = dict(report.get("counts", {}) or {})
    lines: List[str] = []
    lines.append("# Phase-6S V12 Main 2-Process Summary")
    lines.append("")
    lines.append(
        "This report covers only the fixed main profile `unified_acr_rcedr_v12` on the two primary datasets "
        "(HotpotQA, 2WikiMultihopQA) with 2 processes (GPU0/GPU1)."
    )
    lines.append("")

    lines.append("## 1. Executed QA Results")
    lines.append("")
    lines.append(
        "| run_name | dataset | profile | EM | F1 | avg_context_tokens | completion_tokens_avg | "
        "F1_per_1k_context_tokens | retrieval_ms | generation_ms | total_ms |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("EM", 0.0)),
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_context_tokens", 0.0), 1),
                _fmt(row.get("completion_tokens_avg", 0.0), 1),
                _fmt(row.get("F1_per_1k_context_tokens", 0.0)),
                _fmt(row.get("retrieval_ms", 0.0), 1),
                _fmt(row.get("generation_ms", 0.0), 1),
                _fmt(row.get("total_ms", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 2. HotpotQA / 2Wiki Main V12 Results")
    lines.append("")
    lines.append("| dataset | F1 | avg_context_tokens | F1_per_1k_context_tokens | retrieval_ms | total_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset in TARGET_DATASETS:
        row = next((r for r in rows if _safe_text(r.get("dataset")) == dataset), None)
        if row is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                dataset,
                _fmt(row.get("F1", 0.0)),
                _fmt(row.get("avg_context_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_context_tokens", 0.0)),
                _fmt(row.get("retrieval_ms", 0.0), 1),
                _fmt(row.get("total_ms", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 3. Context-token Efficiency")
    lines.append("")
    lines.append("| dataset | avg_context_tokens | F1_per_1k_context_tokens |")
    lines.append("|---|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _fmt(row.get("avg_context_tokens", 0.0), 1),
                _fmt(row.get("F1_per_1k_context_tokens", 0.0)),
            )
        )
    lines.append("")

    lines.append("## 4. Latency Summary")
    lines.append("")
    lines.append(
        "| dataset | retrieval_ms | candidate_proposal_ms | local_graph_ms | corridor_feature_ms | "
        "marginal_selection_ms | rendering_ms | candidate_atoms | selected_atoms | objective_eval_calls |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        latency = dict(row.get("latency_profile", {}) or {})
        t = dict(latency.get("timing_ms_avg", {}) or {})
        c = dict(latency.get("counts_avg", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _fmt(row.get("retrieval_ms", 0.0), 1),
                _fmt(t.get("candidate_proposal", 0.0), 1),
                _fmt(t.get("local_graph_construction", 0.0), 1),
                _fmt(t.get("corridor_feature_extraction", 0.0), 1),
                _fmt(t.get("marginal_selection", 0.0), 1),
                _fmt(t.get("rendering", 0.0), 1),
                _fmt(c.get("candidate_atoms", 0.0), 2),
                _fmt(c.get("selected_atoms", 0.0), 2),
                _fmt(c.get("objective_eval_calls", 0.0), 2),
            )
        )
    lines.append("")

    lines.append("## 5. Artifact Consistency")
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

    lines.append("## 6. Result Warnings")
    lines.append("")
    if result_warnings:
        lines.append("| run_name | dataset | warning | value | path |")
        lines.append("|---|---|---|---:|---|")
        for row in result_warnings:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    _safe_text(row.get("run_name")),
                    _safe_text(row.get("dataset")),
                    _safe_text(row.get("warning")),
                    _fmt(row.get("value", 0.0), 4),
                    _safe_text(row.get("path")),
                )
            )
    else:
        lines.append("No result warnings.")
    lines.append("")

    lines.append("## 7. Next Decision Checklist")
    lines.append("")
    lines.append("- Compare both n=1000 rows against prior n=100 trend for `unified_acr_rcedr_v12`.")
    lines.append("- If quality holds and retrieval remains high, proceed to latency profiling/optimization phase (Phase-6T).")
    lines.append("- Keep objective fixed; do not switch profiles per dataset.")
    lines.append("")

    if methodology_audit:
        lines.append("### Methodology Audit Snapshot")
        lines.append("")
        lines.append("| item | value |")
        lines.append("|---|---|")
        lines.append(f"| status | {_safe_text(methodology_audit.get('status'))} |")
        lines.append(f"| main_profile | {_safe_text(methodology_audit.get('main_profile'))} |")
        lines.append(f"| unified_selector_active | {bool(methodology_audit.get('unified_selector_active', False))} |")
        lines.append(f"| old_cascade_active | {bool(methodology_audit.get('old_cascade_active', False))} |")
        lines.append("")

    if artifact_warnings:
        lines.append("### Artifact Warnings (for diagnostics)")
        lines.append("")
        lines.append("| run_name | status | attempt_count | complete_attempt_count | incomplete_attempt_count |")
        lines.append("|---|---|---:|---:|---:|")
        for row in artifact_warnings:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    _safe_text(row.get("run_name")),
                    _safe_text(row.get("status")),
                    _safe_int(row.get("attempt_count", 0), 0),
                    _safe_int(row.get("complete_attempt_count", 0), 0),
                    _safe_int(row.get("incomplete_attempt_count", 0), 0),
                )
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6S v12 main 2-process runs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--profile-json", default="")
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    output_md = Path(_safe_text(args.output_md) or str(out_root / "PHASE6S_V12_MAIN_2PROC_SUMMARY.md")).resolve()
    output_json = Path(_safe_text(args.output_json) or str(out_root / "phase6s_v12_main_2proc_summary.json")).resolve()
    profile_json = Path(_safe_text(args.profile_json) or str(out_root / "retrieval_profile_summary.json")).resolve()

    report = collect_phase6s_artifacts(out_root)
    methodology_audit = _load_methodology_audit(out_root)
    rows = _build_rows(report)
    artifact_warnings = _artifact_warning_rows(report)
    result_warnings = _result_warning_rows(rows)

    latency_profiles = [
        {
            "dataset": _safe_text(row.get("dataset")),
            "profile": _safe_text(row.get("profile")),
            "run_name": _safe_text(row.get("run_name")),
            "n": _safe_int(row.get("qa_num_queries", 0), 0),
            **dict(row.get("latency_profile", {}) or {}),
        }
        for row in rows
    ]

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    profile_json.parent.mkdir(parents=True, exist_ok=True)

    output_md.write_text(
        _markdown(
            rows=rows,
            report=report,
            artifact_warnings=artifact_warnings,
            result_warnings=result_warnings,
            methodology_audit=methodology_audit,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "rows": rows,
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "artifact_warnings": artifact_warnings,
        "result_warnings": result_warnings,
        "methodology_audit": methodology_audit,
        "retrieval_profile_summary_path": str(profile_json),
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    profile_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": "phase6s_v12_main_2proc",
        "runs": latency_profiles,
    }
    profile_json.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(str(output_md))


if __name__ == "__main__":
    main()
