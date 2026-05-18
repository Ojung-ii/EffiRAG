#!/usr/bin/env python3
"""Summarize Phase-6T latency triage runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from validate_phase6s_artifacts import collect_phase6s_artifacts


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa"]
DEFAULT_PROFILES = [
    "unified_acr_rcedr_v12",
    "unified_acr_rcedr_v12_sota_contract_unified",
    "unified_acr_rcedr_v12_sota_contract_rerank20",
    "unified_acr_rcedr_v12_sota_contract_search3x3",
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
            value = _safe_float(summary.get(key), default)
            return value
    return float(default)


def _build_rows(
    report: Mapping[str, Any],
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_datasets = set(datasets)
    allowed_profiles = set(profiles)
    rows: List[Dict[str, Any]] = []
    for run in list(report.get("completed_runs", []) or []):
        dataset = _safe_text(run.get("dataset"))
        profile = _safe_text(run.get("profile"))
        if dataset not in allowed_datasets or profile not in allowed_profiles:
            continue
        summary = dict(run.get("summary", {}) or {})
        em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
        f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
        prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)

        row = {
            "run_name": _safe_text(run.get("run_name")),
            "dataset": dataset,
            "profile": profile,
            "EM": em,
            "F1": f1,
            "prompt_tokens_avg": prompt_tokens,
            "completion_tokens_avg": _safe_float(summary.get("completion_tokens_avg", 0.0), 0.0),
            "F1_per_1k_prompt": (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0,
            "retrieval_ms": _first_num(summary, ["retrieval_ms", "retrieval_latency_ms"], 0.0),
            "generation_ms": _first_num(summary, ["generation_ms", "generation_latency_ms"], 0.0),
            "total_ms": _first_num(summary, ["total_ms", "total_latency_ms"], 0.0),
            "supporting_fact_recall": _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0),
            "supporting_fact_precision": _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0),
            "bridge_noise_ratio": _first_num(
                summary,
                ["bridge_noise_ratio", "stagewise_bridge_noise_ratio"],
                0.0,
            ),
            "answer_bearing_chunk_present": _safe_float(summary.get("answer_bearing_chunk_present", 0.0), 0.0),
            "answer_present_but_generation_fail": _first_num(
                summary,
                ["answer_present_but_generation_fail", "stagewise_answer_present_but_generation_fail"],
                0.0,
            ),
            "sentence_rerank_ms": _first_num(
                summary,
                ["sentence_rerank_ms", "embedding_rerank_ms", "stagewise_sentence_rerank_ms"],
                0.0,
            ),
            "proposal_time_ms": _first_num(
                summary,
                ["proposal_time_ms", "proposal_ms", "stagewise_proposal_time_ms"],
                0.0,
            ),
            "unified_acr_rcedr_ms": _first_num(
                summary,
                ["unified_acr_rcedr_ms", "marginal_selection_ms", "selector_ms"],
                0.0,
            ),
            "ppr_time_ms": _first_num(summary, ["ppr_time_ms", "ppr_ms"], 0.0),
            "qa_num_queries": _safe_int(summary.get("qa_executed_samples", 0), 0),
            "summary_path": _safe_text(run.get("summary_path")),
        }
        rows.append(row)

    rows.sort(key=lambda x: (_safe_text(x["dataset"]), _safe_text(x["profile"]), _safe_text(x["run_name"])))
    return rows


def _row(rows: Sequence[Mapping[str, Any]], dataset: str, profile: str) -> Dict[str, Any] | None:
    for row in rows:
        if _safe_text(row.get("dataset")) == dataset and _safe_text(row.get("profile")) == profile:
            return dict(row)
    return None


def _mk_table_rows(rows: Sequence[Mapping[str, Any]], dataset: str) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if _safe_text(row.get("dataset")) == dataset]


def _latency_deltas(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for dataset in DEFAULT_DATASETS:
        base = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_unified")
        current = _row(rows, dataset, "unified_acr_rcedr_v12")
        if base is None or current is None:
            continue
        for profile in [
            "unified_acr_rcedr_v12_sota_contract_unified",
            "unified_acr_rcedr_v12_sota_contract_rerank20",
            "unified_acr_rcedr_v12_sota_contract_search3x3",
            "unified_acr_rcedr_v12_sota_contract_fast",
        ]:
            row = _row(rows, dataset, profile)
            if row is None:
                continue
            out.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "delta_F1_vs_current": _safe_float(row.get("F1"), 0.0) - _safe_float(current.get("F1"), 0.0),
                    "delta_prompt_tokens_vs_current": _safe_float(row.get("prompt_tokens_avg"), 0.0)
                    - _safe_float(current.get("prompt_tokens_avg"), 0.0),
                    "delta_retrieval_ms_vs_contract_unified": _safe_float(row.get("retrieval_ms"), 0.0)
                    - _safe_float(base.get("retrieval_ms"), 0.0),
                    "delta_F1_vs_contract_unified": _safe_float(row.get("F1"), 0.0)
                    - _safe_float(base.get("F1"), 0.0),
                }
            )
    return out


def _mk_md(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], deltas: Sequence[Mapping[str, Any]]) -> str:
    counts = dict(report.get("counts", {}) or {})
    lines: List[str] = []
    lines.append("# Phase-6T Latency Triage Summary")
    lines.append("")

    lines.append("## 1. Executed QA Results")
    lines.append("")
    lines.append(
        "| dataset | profile | EM | F1 | prompt_tokens_avg | F1_per_1k_prompt | retrieval_ms | "
        "sentence_rerank_ms | proposal_time_ms | unified_acr_rcedr_ms | total_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("EM"), 4),
                _fmt(row.get("F1"), 4),
                _fmt(row.get("prompt_tokens_avg"), 1),
                _fmt(row.get("F1_per_1k_prompt"), 4),
                _fmt(row.get("retrieval_ms"), 1),
                _fmt(row.get("sentence_rerank_ms"), 1),
                _fmt(row.get("proposal_time_ms"), 1),
                _fmt(row.get("unified_acr_rcedr_ms"), 1),
                _fmt(row.get("total_ms"), 1),
            )
        )
    lines.append("")

    lines.append("## 2. Latency Triage Comparison")
    lines.append("")
    lines.append(
        "| dataset | profile | delta_F1_vs_current | delta_prompt_tokens_vs_current | "
        "delta_retrieval_ms_vs_contract_unified | delta_F1_vs_contract_unified |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in deltas:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("delta_F1_vs_current"), 4),
                _fmt(row.get("delta_prompt_tokens_vs_current"), 1),
                _fmt(row.get("delta_retrieval_ms_vs_contract_unified"), 1),
                _fmt(row.get("delta_F1_vs_contract_unified"), 4),
            )
        )
    lines.append("")

    lines.append("## 3. SOTA Contract Alignment Effect")
    lines.append("")
    lines.append("| dataset | current_v12_F1 | contract_unified_F1 | current_retrieval_ms | contract_unified_retrieval_ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset in DEFAULT_DATASETS:
        current = _row(rows, dataset, "unified_acr_rcedr_v12")
        contract = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_unified")
        if current is None or contract is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                dataset,
                _fmt(current.get("F1"), 4),
                _fmt(contract.get("F1"), 4),
                _fmt(current.get("retrieval_ms"), 1),
                _fmt(contract.get("retrieval_ms"), 1),
            )
        )
    lines.append("")

    lines.append("## 4. Rerank Top20 Effect")
    lines.append("")
    lines.append("| dataset | contract_unified_F1 | rerank20_F1 | contract_unified_retrieval_ms | rerank20_retrieval_ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset in DEFAULT_DATASETS:
        contract = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_unified")
        rerank20 = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_rerank20")
        if contract is None or rerank20 is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                dataset,
                _fmt(contract.get("F1"), 4),
                _fmt(rerank20.get("F1"), 4),
                _fmt(contract.get("retrieval_ms"), 1),
                _fmt(rerank20.get("retrieval_ms"), 1),
            )
        )
    lines.append("")

    lines.append("## 5. Search 3x3 Effect")
    lines.append("")
    lines.append("| dataset | contract_unified_F1 | search3x3_F1 | contract_unified_retrieval_ms | search3x3_retrieval_ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset in DEFAULT_DATASETS:
        contract = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_unified")
        search33 = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_search3x3")
        if contract is None or search33 is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                dataset,
                _fmt(contract.get("F1"), 4),
                _fmt(search33.get("F1"), 4),
                _fmt(contract.get("retrieval_ms"), 1),
                _fmt(search33.get("retrieval_ms"), 1),
            )
        )
    lines.append("")

    lines.append("## 6. Fast Combined Effect")
    lines.append("")
    lines.append("| dataset | contract_unified_F1 | fast_F1 | contract_unified_retrieval_ms | fast_retrieval_ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset in DEFAULT_DATASETS:
        contract = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_unified")
        fast = _row(rows, dataset, "unified_acr_rcedr_v12_sota_contract_fast")
        if contract is None or fast is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                dataset,
                _fmt(contract.get("F1"), 4),
                _fmt(fast.get("F1"), 4),
                _fmt(contract.get("retrieval_ms"), 1),
                _fmt(fast.get("retrieval_ms"), 1),
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

    lines.append("## 8. Bridge Noise Analysis")
    lines.append("")
    for dataset in DEFAULT_DATASETS:
        dataset_rows = _mk_table_rows(rows, dataset)
        if not dataset_rows:
            continue
        dataset_rows = sorted(dataset_rows, key=lambda r: _safe_float(r.get("bridge_noise_ratio"), 0.0), reverse=True)
        top = dataset_rows[0]
        lines.append(
            "- {}: highest bridge noise profile={} ({})".format(
                dataset,
                _safe_text(top.get("profile")),
                _fmt(top.get("bridge_noise_ratio"), 4),
            )
        )
    lines.append("")

    lines.append("## 9. Artifact Consistency")
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

    lines.append("## 10. Decision Checklist")
    lines.append("")
    lines.append("- `sota_contract_unified` improves F1 vs current v12?")
    lines.append("- `rerank20` meaningfully reduces retrieval_ms with acceptable F1 drop?")
    lines.append("- `search3x3` meaningfully reduces retrieval_ms with acceptable F1 drop?")
    lines.append("- `fast` gives best retrieval_ms/F1 trade-off and lower bridge noise?")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6T latency triage runs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    output_md = Path(_safe_text(args.output_md) or str(out_root / "PHASE6T_LATENCY_TRIAGE_SUMMARY.md")).resolve()
    output_json = Path(_safe_text(args.output_json) or str(out_root / "phase6t_latency_triage_summary.json")).resolve()

    datasets = _tokenize(args.datasets) or list(DEFAULT_DATASETS)
    profiles = _tokenize(args.profiles) or list(DEFAULT_PROFILES)

    report = collect_phase6s_artifacts(out_root)
    rows = _build_rows(report, datasets=datasets, profiles=profiles)
    deltas = _latency_deltas(rows)
    markdown = _mk_md(report, rows, deltas)

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
        "latency_deltas": deltas,
        "report": report,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(output_md))


if __name__ == "__main__":
    main()
