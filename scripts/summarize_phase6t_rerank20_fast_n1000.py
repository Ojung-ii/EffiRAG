#!/usr/bin/env python3
"""Summarize Phase-6T rerank20 vs fast n1000 runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from validate_phase6s_artifacts import collect_phase6s_artifacts


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa"]
DEFAULT_PROFILES = [
    "unified_acr_rcedr_v12_sota_contract_rerank20",
    "unified_acr_rcedr_v12_sota_contract_fast",
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


def _fmt(value: Any, nd: int = 4) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def _tokenize(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _first_num(summary: Mapping[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        if key in summary:
            return _safe_float(summary.get(key), default)
    return float(default)


def _build_rows(report: Mapping[str, Any], *, datasets: Sequence[str], profiles: Sequence[str]) -> List[Dict[str, Any]]:
    allowed_datasets = set(datasets)
    allowed_profiles = set(profiles)
    rows: List[Dict[str, Any]] = []
    for run in list(report.get("completed_runs", []) or []):
        dataset = _safe_text(run.get("dataset"))
        profile = _safe_text(run.get("profile"))
        if dataset not in allowed_datasets or profile not in allowed_profiles:
            continue

        summary = dict(run.get("summary", {}) or {})
        f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
        prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
        rows.append(
            {
                "run_name": _safe_text(run.get("run_name")),
                "dataset": dataset,
                "profile": profile,
                "EM": _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0),
                "F1": f1,
                "prompt_tokens_avg": prompt_tokens,
                "completion_tokens_avg": _safe_float(summary.get("completion_tokens_avg", 0.0), 0.0),
                "F1_per_1k_prompt": (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0,
                "retrieval_ms": _first_num(summary, ["retrieval_ms", "retrieval_latency_ms"], 0.0),
                "total_ms": _first_num(summary, ["total_ms", "total_latency_ms"], 0.0),
                "supporting_fact_recall": _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0),
                "supporting_fact_precision": _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0),
                "bridge_noise_ratio": _first_num(summary, ["bridge_noise_ratio", "stagewise_bridge_noise_ratio"], 0.0),
                "answer_bearing_chunk_present": _safe_float(summary.get("answer_bearing_chunk_present", 0.0), 0.0),
                "answer_present_but_generation_fail": _first_num(
                    summary,
                    ["answer_present_but_generation_fail", "stagewise_answer_present_but_generation_fail"],
                    0.0,
                ),
                "qa_num_queries": _safe_int(summary.get("qa_executed_samples", 0), 0),
                "summary_path": _safe_text(run.get("summary_path")),
            }
        )
    rows.sort(key=lambda x: (_safe_text(x["dataset"]), _safe_text(x["profile"]), _safe_text(x["run_name"])))
    return rows


def _row(rows: Sequence[Mapping[str, Any]], dataset: str, profile: str) -> Dict[str, Any] | None:
    for row in rows:
        if _safe_text(row.get("dataset")) == dataset and _safe_text(row.get("profile")) == profile:
            return dict(row)
    return None


def _aggregate(rows: Sequence[Mapping[str, Any]], profile: str) -> Dict[str, float]:
    subset = [r for r in rows if _safe_text(r.get("profile")) == profile]
    if not subset:
        return {}
    n = float(len(subset))
    return {
        "avg_F1": sum(_safe_float(r.get("F1"), 0.0) for r in subset) / n,
        "avg_F1_per_1k_prompt": sum(_safe_float(r.get("F1_per_1k_prompt"), 0.0) for r in subset) / n,
        "avg_retrieval_ms": sum(_safe_float(r.get("retrieval_ms"), 0.0) for r in subset) / n,
    }


def _choose_winner(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rerank = "unified_acr_rcedr_v12_sota_contract_rerank20"
    fast = "unified_acr_rcedr_v12_sota_contract_fast"

    hotpot_r = _row(rows, "hotpotqa", rerank)
    hotpot_f = _row(rows, "hotpotqa", fast)
    wiki_r = _row(rows, "2wikimultihopqa", rerank)
    wiki_f = _row(rows, "2wikimultihopqa", fast)

    dataset_pref: Dict[str, str] = {}
    if hotpot_r and hotpot_f:
        dataset_pref["hotpotqa"] = rerank if _safe_float(hotpot_r["F1"]) >= _safe_float(hotpot_f["F1"]) else fast
    if wiki_r and wiki_f:
        dataset_pref["2wikimultihopqa"] = rerank if _safe_float(wiki_r["F1"]) >= _safe_float(wiki_f["F1"]) else fast

    agg_r = _aggregate(rows, rerank)
    agg_f = _aggregate(rows, fast)

    final_choice = "insufficient_data"
    reason = "missing_comparison_rows"
    if agg_r and agg_f:
        if dataset_pref.get("hotpotqa") == rerank and dataset_pref.get("2wikimultihopqa") == rerank:
            final_choice = rerank
            reason = "wins_both_datasets"
        elif dataset_pref.get("hotpotqa") == fast and dataset_pref.get("2wikimultihopqa") == fast:
            final_choice = fast
            reason = "wins_both_datasets"
        else:
            avg_f1_gap = _safe_float(agg_r.get("avg_F1"), 0.0) - _safe_float(agg_f.get("avg_F1"), 0.0)
            avg_ms_gap = _safe_float(agg_r.get("avg_retrieval_ms"), 0.0) - _safe_float(agg_f.get("avg_retrieval_ms"), 0.0)
            if abs(avg_f1_gap) <= 0.02 and avg_ms_gap > 0:
                final_choice = fast
                reason = "avg_F1_within_0.02_and_fast_lower_retrieval_ms"
            else:
                final_choice = rerank
                reason = "avg_F1_prefers_rerank20_or_fast_not_substantially_faster"

    return {
        "dataset_preference": dataset_pref,
        "aggregate_rerank20": agg_r,
        "aggregate_fast": agg_f,
        "final_choice": final_choice,
        "final_reason": reason,
    }


def _mk_md(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]) -> str:
    counts = dict(report.get("counts", {}) or {})
    lines: List[str] = []
    lines.append("# Phase-6T Rerank20/Fast n1000 Summary")
    lines.append("")

    lines.append("## 1. Executed QA Results")
    lines.append("")
    lines.append(
        "| dataset | profile | EM | F1 | prompt_tokens_avg | F1_per_1k_prompt | retrieval_ms | total_ms | "
        "supporting_fact_recall | supporting_fact_precision | bridge_noise_ratio | "
        "answer_bearing_chunk_present | answer_present_but_generation_fail |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("EM"), 4),
                _fmt(row.get("F1"), 4),
                _fmt(row.get("prompt_tokens_avg"), 1),
                _fmt(row.get("F1_per_1k_prompt"), 4),
                _fmt(row.get("retrieval_ms"), 1),
                _fmt(row.get("total_ms"), 1),
                _fmt(row.get("supporting_fact_recall"), 4),
                _fmt(row.get("supporting_fact_precision"), 4),
                _fmt(row.get("bridge_noise_ratio"), 4),
                _fmt(row.get("answer_bearing_chunk_present"), 4),
                _fmt(row.get("answer_present_but_generation_fail"), 4),
            )
        )
    lines.append("")

    lines.append("## 2. Rerank20 vs Fast Comparison")
    lines.append("")
    lines.append("| dataset | rerank20_F1 | fast_F1 | delta_F1(rerank20-fast) | rerank20_retrieval_ms | fast_retrieval_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset in DEFAULT_DATASETS:
        r = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_rerank20")
        f = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_fast")
        if r is None or f is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                dataset,
                _fmt(r.get("F1"), 4),
                _fmt(f.get("F1"), 4),
                _fmt(_safe_float(r.get("F1"), 0.0) - _safe_float(f.get("F1"), 0.0), 4),
                _fmt(r.get("retrieval_ms"), 1),
                _fmt(f.get("retrieval_ms"), 1),
            )
        )
    lines.append("")

    lines.append("## 3. HotpotQA Decision")
    lines.append("")
    lines.append(f"- preferred_profile: `{_safe_text(decision.get('dataset_preference', {}).get('hotpotqa', 'n/a'))}`")
    lines.append("")

    lines.append("## 4. 2Wiki Decision")
    lines.append("")
    lines.append(
        f"- preferred_profile: `{_safe_text(decision.get('dataset_preference', {}).get('2wikimultihopqa', 'n/a'))}`"
    )
    lines.append("")

    lines.append("## 5. Cross-dataset Main Candidate Decision")
    lines.append("")
    agg_r = dict(decision.get("aggregate_rerank20", {}) or {})
    agg_f = dict(decision.get("aggregate_fast", {}) or {})
    lines.append("| profile | avg_F1 | avg_F1_per_1k_prompt | avg_retrieval_ms |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        "| unified_acr_rcedr_v12_sota_contract_rerank20 | {} | {} | {} |".format(
            _fmt(agg_r.get("avg_F1"), 4),
            _fmt(agg_r.get("avg_F1_per_1k_prompt"), 4),
            _fmt(agg_r.get("avg_retrieval_ms"), 1),
        )
    )
    lines.append(
        "| unified_acr_rcedr_v12_sota_contract_fast | {} | {} | {} |".format(
            _fmt(agg_f.get("avg_F1"), 4),
            _fmt(agg_f.get("avg_F1_per_1k_prompt"), 4),
            _fmt(agg_f.get("avg_retrieval_ms"), 1),
        )
    )
    lines.append("")
    lines.append(f"- final_choice: `{_safe_text(decision.get('final_choice'))}`")
    lines.append(f"- reason: `{_safe_text(decision.get('final_reason'))}`")
    lines.append("")

    lines.append("## 6. Latency Summary")
    lines.append("")
    lines.append("| dataset | profile | retrieval_ms | total_ms |")
    lines.append("|---|---|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("retrieval_ms"), 1),
                _fmt(row.get("total_ms"), 1),
            )
        )
    lines.append("")

    lines.append("## 7. Evidence Quality Metrics")
    lines.append("")
    lines.append(
        "| dataset | profile | supporting_fact_recall | supporting_fact_precision | bridge_noise_ratio | "
        "answer_bearing_chunk_present | answer_present_but_generation_fail |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("supporting_fact_recall"), 4),
                _fmt(row.get("supporting_fact_precision"), 4),
                _fmt(row.get("bridge_noise_ratio"), 4),
                _fmt(row.get("answer_bearing_chunk_present"), 4),
                _fmt(row.get("answer_present_but_generation_fail"), 4),
            )
        )
    lines.append("")

    lines.append("## 8. Artifact Consistency")
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

    lines.append("## 9. Final Recommendation")
    lines.append("")
    lines.append(f"- recommended_candidate: `{_safe_text(decision.get('final_choice'))}`")
    lines.append(f"- rationale: `{_safe_text(decision.get('final_reason'))}`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6T rerank20 vs fast n1000 runs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    output_md = Path(
        _safe_text(args.output_md) or str(out_root / "PHASE6T_RERANK20_FAST_N1000_SUMMARY.md")
    ).resolve()
    output_json = Path(
        _safe_text(args.output_json) or str(out_root / "phase6t_rerank20_fast_n1000_summary.json")
    ).resolve()

    datasets = _tokenize(args.datasets) or list(DEFAULT_DATASETS)
    profiles = _tokenize(args.profiles) or list(DEFAULT_PROFILES)

    report = collect_phase6s_artifacts(out_root)
    rows = _build_rows(report, datasets=datasets, profiles=profiles)
    decision = _choose_winner(rows)
    markdown = _mk_md(report, rows, decision)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown + "\n", encoding="utf-8")

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "output_md": str(output_md),
        "output_json": str(output_json),
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "rows": rows,
        "decision": decision,
        "report": report,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(output_md))


if __name__ == "__main__":
    main()
