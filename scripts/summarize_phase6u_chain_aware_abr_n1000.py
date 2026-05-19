#!/usr/bin/env python3
"""Summarize Phase-6U chain-aware ABR n1000 runs against reused baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from validate_phase6s_artifacts import collect_phase6s_artifacts

TARGET_PROFILE = "unified_acr_rcedr_v12_chain_aware"
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


def _mean(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals]
    if not arr:
        return 0.0
    return float(sum(arr) / len(arr))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
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
    except Exception:
        return []
    return rows


def _load_baseline_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"baseline metrics file missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("baseline metrics JSON must be a mapping")
    out: Dict[str, Dict[str, float]] = {}
    for dataset in TARGET_DATASETS:
        row = dict(raw.get(dataset, {}) or {})
        out[dataset] = {
            "EM": _safe_float(row.get("EM", 0.0)),
            "F1": _safe_float(row.get("F1", 0.0)),
            "avg_context_tokens": _safe_float(row.get("avg_context_tokens", 0.0)),
            "supporting_fact_recall": _safe_float(row.get("supporting_fact_recall", 0.0)),
            "supporting_fact_precision": _safe_float(row.get("supporting_fact_precision", 0.0)),
            "retrieval_ms": _safe_float(row.get("retrieval_ms", 0.0)),
        }
    return out


def _extract_query_level_diag(query_path: Path) -> Dict[str, float]:
    selected_atoms: List[float] = []
    selected_tokens: List[float] = []
    chain_gain_total: List[float] = []
    role_relax_total: List[float] = []

    sentence_after_rerank_sf_recall: List[float] = []
    evidence_atoms_sf_recall: List[float] = []
    abr_selected_sf_recall: List[float] = []
    rendered_sf_recall: List[float] = []

    for row in _iter_jsonl(query_path):
        retrieval = dict(row.get("retrieval", {}) or {})
        diag = dict(retrieval.get("diagnostics", {}) or {})
        unified = dict(diag.get("unified_acr_rcedr_diag", {}) or {})

        selected_atoms.append(_safe_float(unified.get("num_selected_atoms", 0.0), 0.0))
        selected_tokens.append(_safe_float(unified.get("selected_tokens", 0.0), 0.0))
        chain_gain_total.append(_safe_float(unified.get("chain_gain_total_avg", unified.get("chain_gain_total", 0.0)), 0.0))
        role_relax_total.append(
            _safe_float(
                unified.get("role_aware_redundancy_applied_avg", unified.get("role_aware_redundancy_applied_total", 0.0)),
                0.0,
            )
        )

        stage_rows = list(diag.get("stagewise_supporting_fact_metrics", []) or [])
        for s in stage_rows:
            if not isinstance(s, Mapping):
                continue
            stage = _safe_text(s.get("stage"))
            recall = _safe_float(s.get("supporting_fact_recall", 0.0), 0.0)
            if stage == "sentence_candidates_after_text_rerank":
                sentence_after_rerank_sf_recall.append(recall)
            elif stage == "evidence_atoms_before_abr":
                evidence_atoms_sf_recall.append(recall)
            elif stage == "abr_selected_evidence":
                abr_selected_sf_recall.append(recall)
            elif stage == "rendered_compact_context":
                rendered_sf_recall.append(recall)

    return {
        "selected_atoms_avg": _mean(selected_atoms),
        "selected_tokens_avg": _mean(selected_tokens),
        "chain_gain_total_avg": _mean(chain_gain_total),
        "role_aware_redundancy_applied_avg": _mean(role_relax_total),
        "sentence_candidate_SF_recall": _mean(sentence_after_rerank_sf_recall),
        "evidence_atom_SF_recall": _mean(evidence_atoms_sf_recall),
        "ABR_selected_SF_recall": _mean(abr_selected_sf_recall),
        "rendered_SF_recall": _mean(rendered_sf_recall),
    }


def _build_rows(report: Mapping[str, Any], baseline: Mapping[str, Mapping[str, float]]) -> List[Dict[str, Any]]:
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
        qdiag = _extract_query_level_diag(query_path)

        em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
        f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
        sf_r = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
        sf_p = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
        sf_f1 = _safe_float(summary.get("supporting_fact_f1", (2 * sf_r * sf_p / (sf_r + sf_p)) if (sf_r + sf_p) > 0 else 0.0), 0.0)
        prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
        recall5 = _safe_float(
            summary.get(
                "recall_at_5",
                summary.get("recall5", summary.get("retrieval_recall_at_5", summary.get("Recall@5", 0.0))),
            ),
            0.0,
        )
        retrieval_ms = _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0)
        generation_ms = _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0)
        total_ms = _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0)

        f1_per_1k = (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0
        base = dict(baseline.get(dataset, {}) or {})
        base_f1_per_1k = (base.get("F1", 0.0) * 1000.0 / base.get("avg_context_tokens", 0.0)) if base.get("avg_context_tokens", 0.0) > 0 else 0.0
        base_sf_f1 = (
            2.0 * base.get("supporting_fact_recall", 0.0) * base.get("supporting_fact_precision", 0.0)
            / (base.get("supporting_fact_recall", 0.0) + base.get("supporting_fact_precision", 0.0))
            if (base.get("supporting_fact_recall", 0.0) + base.get("supporting_fact_precision", 0.0)) > 0
            else 0.0
        )

        rows.append(
            {
                "run_name": run_name,
                "dataset": dataset,
                "profile": profile,
                "EM": em,
                "F1": f1,
                "Recall@5": recall5,
                "avg_context_tokens": prompt_tokens,
                "F1_per_1k_context_tokens": f1_per_1k,
                "supporting_fact_precision": sf_p,
                "supporting_fact_recall": sf_r,
                "supporting_fact_f1": sf_f1,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "selected_atoms_avg": qdiag.get("selected_atoms_avg", 0.0),
                "selected_tokens_avg": qdiag.get("selected_tokens_avg", 0.0),
                "chain_gain_total_avg": qdiag.get("chain_gain_total_avg", 0.0),
                "role_aware_redundancy_applied_avg": qdiag.get("role_aware_redundancy_applied_avg", 0.0),
                "sentence_candidate_SF_recall": qdiag.get("sentence_candidate_SF_recall", 0.0),
                "evidence_atom_SF_recall": qdiag.get("evidence_atom_SF_recall", 0.0),
                "ABR_selected_SF_recall": qdiag.get("ABR_selected_SF_recall", 0.0),
                "rendered_SF_recall": qdiag.get("rendered_SF_recall", 0.0),
                "baseline": {
                    "EM": base.get("EM", 0.0),
                    "F1": base.get("F1", 0.0),
                    "avg_context_tokens": base.get("avg_context_tokens", 0.0),
                    "supporting_fact_precision": base.get("supporting_fact_precision", 0.0),
                    "supporting_fact_recall": base.get("supporting_fact_recall", 0.0),
                    "supporting_fact_f1": base_sf_f1,
                    "retrieval_ms": base.get("retrieval_ms", 0.0),
                    "F1_per_1k_context_tokens": base_f1_per_1k,
                },
            }
        )

    rows.sort(key=lambda r: (r.get("dataset", ""), r.get("run_name", "")))
    for row in rows:
        base = dict(row.get("baseline", {}) or {})
        row["delta"] = {
            "EM": row["EM"] - _safe_float(base.get("EM", 0.0), 0.0),
            "F1": row["F1"] - _safe_float(base.get("F1", 0.0), 0.0),
            "avg_context_tokens": row["avg_context_tokens"] - _safe_float(base.get("avg_context_tokens", 0.0), 0.0),
            "F1_per_1k_context_tokens": row["F1_per_1k_context_tokens"] - _safe_float(base.get("F1_per_1k_context_tokens", 0.0), 0.0),
            "supporting_fact_precision": row["supporting_fact_precision"] - _safe_float(base.get("supporting_fact_precision", 0.0), 0.0),
            "supporting_fact_recall": row["supporting_fact_recall"] - _safe_float(base.get("supporting_fact_recall", 0.0), 0.0),
            "supporting_fact_f1": row["supporting_fact_f1"] - _safe_float(base.get("supporting_fact_f1", 0.0), 0.0),
            "retrieval_ms": row["retrieval_ms"] - _safe_float(base.get("retrieval_ms", 0.0), 0.0),
            "total_ms": row["total_ms"] - (_safe_float(base.get("retrieval_ms", 0.0), 0.0) + row["generation_ms"]),
        }
    return rows


def _decision(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"decision": "reject", "reason": "no_completed_rows"}

    f1_deltas = [_safe_float(r.get("delta", {}).get("F1", 0.0), 0.0) for r in rows]
    sf_deltas = [_safe_float(r.get("delta", {}).get("supporting_fact_recall", 0.0), 0.0) for r in rows]
    tok_deltas_pct = []
    ret_deltas_pct = []
    for r in rows:
        base = dict(r.get("baseline", {}) or {})
        bt = _safe_float(base.get("avg_context_tokens", 0.0), 0.0)
        br = _safe_float(base.get("retrieval_ms", 0.0), 0.0)
        tok_deltas_pct.append((_safe_float(r.get("delta", {}).get("avg_context_tokens", 0.0), 0.0) / bt) if bt > 0 else 0.0)
        ret_deltas_pct.append((_safe_float(r.get("delta", {}).get("retrieval_ms", 0.0), 0.0) / br) if br > 0 else 0.0)

    strong = all(
        _safe_float(r.get("delta", {}).get("F1", 0.0), 0.0) >= 0.03
        and _safe_float(r.get("delta", {}).get("supporting_fact_recall", 0.0), 0.0) >= 0.06
        for r in rows
    ) and max(tok_deltas_pct) <= 0.15 and max(ret_deltas_pct) <= 0.20

    if strong:
        return {"decision": "strong_success", "reason": "both_datasets_clear_improvement_with_bounded_cost"}

    acceptable = _mean(f1_deltas) > 0 and _mean(sf_deltas) > 0 and max(tok_deltas_pct) <= 0.20 and max(ret_deltas_pct) <= 0.30
    if acceptable:
        return {"decision": "acceptable_success", "reason": "average_improvement_with_acceptable_token_latency_cost"}

    reject = any(_safe_float(r.get("delta", {}).get("F1", 0.0), 0.0) < -0.02 for r in rows) or max(tok_deltas_pct) > 0.25 or max(ret_deltas_pct) > 0.30
    if reject:
        return {"decision": "reject", "reason": "quality_regression_or_cost_explosion"}

    return {"decision": "hold", "reason": "mixed_signal_needs_follow_up"}


def _render_markdown(out_root: Path, rows: Sequence[Mapping[str, Any]], decision: Mapping[str, Any], baseline_path: Path) -> str:
    lines: List[str] = []
    lines.append("# PHASE6U Chain-aware ABR N1000 Summary")
    lines.append("")
    lines.append("Baseline metrics are reused from existing unified_acr_rcedr_v12 n=1000 result; baseline was not rerun.")
    lines.append(f"- baseline_metrics_source: `{baseline_path}`")
    lines.append(f"- out_root: `{out_root}`")
    lines.append("")

    lines.append("## Executed QA Results")
    lines.append("")
    lines.append("| dataset | profile | EM | F1 | Recall@5 | avg_context_tokens | F1_per_1k_context_tokens | supporting_fact_precision | supporting_fact_recall | supporting_fact_f1 | retrieval_ms | generation_ms | total_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(r.get("dataset")),
                _safe_text(r.get("profile")),
                _fmt(r.get("EM"), 4),
                _fmt(r.get("F1"), 4),
                _fmt(r.get("Recall@5"), 4),
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

    lines.append("## Delta vs Reused Baseline")
    lines.append("")
    lines.append("| dataset | ΔEM | ΔF1 | Δavg_context_tokens | ΔF1_per_1k_context_tokens | ΔSF_precision | ΔSF_recall | ΔSF_f1 | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        d = dict(r.get("delta", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(r.get("dataset")),
                _fmt(d.get("EM"), 4),
                _fmt(d.get("F1"), 4),
                _fmt(d.get("avg_context_tokens"), 3),
                _fmt(d.get("F1_per_1k_context_tokens"), 4),
                _fmt(d.get("supporting_fact_precision"), 4),
                _fmt(d.get("supporting_fact_recall"), 4),
                _fmt(d.get("supporting_fact_f1"), 4),
                _fmt(d.get("retrieval_ms"), 2),
                _fmt(d.get("total_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## ABR-stage Diagnostics")
    lines.append("")
    lines.append("| dataset | selected_atoms_avg | selected_tokens_avg | chain_gain_total_avg | role_aware_redundancy_applied_avg | sentence_candidate_SF_recall | evidence_atom_SF_recall | ABR_selected_SF_recall | rendered_SF_recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(r.get("dataset")),
                _fmt(r.get("selected_atoms_avg"), 3),
                _fmt(r.get("selected_tokens_avg"), 3),
                _fmt(r.get("chain_gain_total_avg"), 4),
                _fmt(r.get("role_aware_redundancy_applied_avg"), 4),
                _fmt(r.get("sentence_candidate_SF_recall"), 4),
                _fmt(r.get("evidence_atom_SF_recall"), 4),
                _fmt(r.get("ABR_selected_SF_recall"), 4),
                _fmt(r.get("rendered_SF_recall"), 4),
            )
        )
    lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(f"- decision: `{_safe_text(decision.get('decision'))}`")
    lines.append(f"- reason: {_safe_text(decision.get('reason'))}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Phase-6U chain-aware ABR n1000 run")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--baseline-metrics", default="configs/SOTA_config/phase6s_v12_n1000_baseline_metrics.json")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    baseline_path = Path(args.baseline_metrics).resolve()

    report = collect_phase6s_artifacts(out_root)
    baseline = _load_baseline_metrics(baseline_path)
    rows = _build_rows(report, baseline)
    decision = _decision(rows)

    summary = {
        "phase": "phase6u_chain_aware_abr_n1000",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "profile": TARGET_PROFILE,
        "datasets": TARGET_DATASETS,
        "optimized_path_enabled": True,
        "phase6t_optimized": True,
        "baseline_reused": True,
        "baseline_metrics_source": str(baseline_path),
        "rows": rows,
        "decision": decision,
    }

    md = _render_markdown(out_root, rows, decision, baseline_path)
    md_path = out_root / "PHASE6U_CHAIN_AWARE_ABR_N1000_SUMMARY.md"
    json_path = out_root / "phase6u_chain_aware_abr_n1000_summary.json"
    md_path.write_text(md + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(str(md_path))
    print(str(json_path))


if __name__ == "__main__":
    main()
