#!/usr/bin/env python3
"""Summarize Phase-6T latency isolation runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from validate_phase6s_artifacts import collect_phase6s_artifacts


DEFAULT_PROFILES = [
    "unified_acr_rcedr_v12_sota_contract_fast",
    "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank",
    "unified_acr_rcedr_v12_sota_contract_fast_rerank5",
    "unified_acr_rcedr_v12_sota_contract_fast_proposal_ultralight",
]

STAGE_KEYS = [
    "semantic_scan_ms",
    "embedding_rerank_ms",
    "sentence_rerank_ms",
    "proposal_time_ms",
    "ppr_time_ms",
    "local_graph_construction_ms",
    "corridor_extraction_ms",
    "bridge_feature_ms",
    "answerability_feature_ms",
    "redundancy_scoring_ms",
    "unified_acr_rcedr_ms",
    "render_ms",
]

COUNT_KEYS = [
    "semantic_candidate_count",
    "embedding_rerank_candidate_count",
    "sentence_rerank_candidate_count",
    "num_anchors",
    "samples_per_anchor",
    "ppr_call_count",
    "local_graph_nodes_avg",
    "local_graph_edges_avg",
    "corridor_candidate_count",
    "candidate_atom_count",
    "selected_atom_count",
    "objective_eval_calls",
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


def _percentile(values: Sequence[float], p: float) -> float:
    seq = sorted([float(v) for v in values if v is not None])
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    idx = (len(seq) - 1) * float(p)
    lo = int(idx)
    hi = min(lo + 1, len(seq) - 1)
    frac = float(idx - lo)
    return float(seq[lo] * (1.0 - frac) + seq[hi] * frac)


def _mean(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


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


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _build_completed_rows(
    report: Mapping[str, Any],
    *,
    dataset_filter: str,
    profile_filter: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_profiles = set(profile_filter)
    rows: List[Dict[str, Any]] = []
    for run in list(report.get("completed_runs", []) or []):
        dataset = _safe_text(run.get("dataset"))
        profile = _safe_text(run.get("profile"))
        if dataset_filter and dataset != dataset_filter:
            continue
        if allowed_profiles and profile not in allowed_profiles:
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
                "generation_ms": _first_num(summary, ["generation_ms", "generation_latency_ms"], 0.0),
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
                "query_path": _safe_text(run.get("query_path")),
            }
        )
    rows.sort(key=lambda x: (_safe_text(x["dataset"]), _safe_text(x["profile"]), _safe_text(x["run_name"])))
    return rows


def _build_profile_timing_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        query_path = Path(_safe_text(row.get("query_path")))
        query_rows = _load_jsonl(query_path)
        timing_values = {key: [] for key in STAGE_KEYS}
        count_values = {key: [] for key in COUNT_KEYS}

        for qrow in query_rows:
            stage = dict((qrow.get("latency_breakdown_ms", {}) or {}))
            diag = dict(((qrow.get("retrieval", {}) or {}).get("diagnostics", {}) or {}))
            for key in STAGE_KEYS:
                timing_values[key].append(_safe_float(stage.get(key, 0.0), 0.0))
            for key in COUNT_KEYS:
                count_values[key].append(_safe_float(diag.get(key, 0.0), 0.0))

        timing_avg = {key: _mean(values) for key, values in timing_values.items()}
        timing_p95 = {key: _percentile(values, 0.95) for key, values in timing_values.items()}
        counts_avg = {key: _mean(values) for key, values in count_values.items()}
        counts_p95 = {key: _percentile(values, 0.95) for key, values in count_values.items()}

        out.append(
            {
                "run_name": _safe_text(row.get("run_name")),
                "dataset": _safe_text(row.get("dataset")),
                "profile": _safe_text(row.get("profile")),
                "num_queries": int(len(query_rows)),
                "timing_ms_avg": timing_avg,
                "timing_ms_p95": timing_p95,
                "counts_avg": counts_avg,
                "counts_p95": counts_p95,
            }
        )

    out.sort(key=lambda x: (_safe_text(x["dataset"]), _safe_text(x["profile"]), _safe_text(x["run_name"])))
    return out


def _timing_row(timing_rows: Sequence[Mapping[str, Any]], run_name: str) -> Dict[str, Any] | None:
    for row in timing_rows:
        if _safe_text(row.get("run_name")) == run_name:
            return dict(row)
    return None


def _mk_markdown(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], timing_rows: Sequence[Mapping[str, Any]]) -> str:
    counts = dict(report.get("counts", {}) or {})
    lines: List[str] = []
    lines.append("# Phase-6T Latency Isolation Summary")
    lines.append("")

    lines.append("## 1. Executed QA Results")
    lines.append("")
    lines.append(
        "| dataset | profile | EM | F1 | prompt_tokens_avg | retrieval_ms | total_ms | F1_per_1k_prompt | "
        "supporting_fact_recall | supporting_fact_precision | bridge_noise_ratio |"
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
                _fmt(row.get("retrieval_ms"), 1),
                _fmt(row.get("total_ms"), 1),
                _fmt(row.get("F1_per_1k_prompt"), 4),
                _fmt(row.get("supporting_fact_recall"), 4),
                _fmt(row.get("supporting_fact_precision"), 4),
                _fmt(row.get("bridge_noise_ratio"), 4),
            )
        )
    lines.append("")

    lines.append("## 2. Latency Breakdown")
    lines.append("")
    lines.append(
        "| run_name | profile | semantic_scan_ms | embedding_rerank_ms | proposal_time_ms | ppr_time_ms | "
        "local_graph_construction_ms | corridor_extraction_ms | unified_acr_rcedr_ms | render_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        trow = _timing_row(timing_rows, _safe_text(row.get("run_name"))) or {}
        tavg = dict(trow.get("timing_ms_avg", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _safe_text(row.get("profile")),
                _fmt(tavg.get("semantic_scan_ms", 0.0), 1),
                _fmt(tavg.get("embedding_rerank_ms", 0.0), 1),
                _fmt(tavg.get("proposal_time_ms", 0.0), 1),
                _fmt(tavg.get("ppr_time_ms", 0.0), 1),
                _fmt(tavg.get("local_graph_construction_ms", 0.0), 1),
                _fmt(tavg.get("corridor_extraction_ms", 0.0), 1),
                _fmt(tavg.get("unified_acr_rcedr_ms", 0.0), 1),
                _fmt(tavg.get("render_ms", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 3. Rerank Ablation")
    lines.append("")
    lines.append("- Compare `fast` vs `fast_no_sentence_rerank` vs `fast_rerank5` on retrieval_ms and F1.")
    lines.append("")

    lines.append("## 4. Proposal Ablation")
    lines.append("")
    lines.append("- Compare `fast` vs `fast_proposal_ultralight` for proposal/PPR/local-graph latency drop and F1 impact.")
    lines.append("")

    lines.append("## 5. Evidence Quality Metrics")
    lines.append("")
    lines.append(
        "| run_name | profile | supporting_fact_recall | supporting_fact_precision | "
        "answer_bearing_chunk_present | answer_present_but_generation_fail |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _safe_text(row.get("profile")),
                _fmt(row.get("supporting_fact_recall"), 4),
                _fmt(row.get("supporting_fact_precision"), 4),
                _fmt(row.get("answer_bearing_chunk_present"), 4),
                _fmt(row.get("answer_present_but_generation_fail"), 4),
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

    lines.append("## 7. Decision Checklist")
    lines.append("")
    lines.append("- Does `fast_no_sentence_rerank` substantially reduce retrieval_ms vs `fast` without F1 collapse?")
    lines.append("- Does `fast_rerank5` preserve F1 with clear rerank latency reduction vs `fast`?")
    lines.append("- Does `fast_proposal_ultralight` reduce proposal/PPR-local graph cost enough to justify F1 trade-off?")
    lines.append("- Which stage dominates (`embedding_rerank_ms` vs `proposal_time_ms`/`ppr_time_ms`)?")
    lines.append("")

    return "\n".join(lines)


def _mk_profile_markdown(timing_rows: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Phase-6T Latency Profile Summary")
    lines.append("")
    lines.append("| run_name | dataset | profile | num_queries | semantic_scan_ms_avg | embedding_rerank_ms_avg | proposal_time_ms_avg | ppr_time_ms_avg | unified_acr_rcedr_ms_avg |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in timing_rows:
        tavg = dict(row.get("timing_ms_avg", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                int(row.get("num_queries", 0) or 0),
                _fmt(tavg.get("semantic_scan_ms", 0.0), 1),
                _fmt(tavg.get("embedding_rerank_ms", 0.0), 1),
                _fmt(tavg.get("proposal_time_ms", 0.0), 1),
                _fmt(tavg.get("ppr_time_ms", 0.0), 1),
                _fmt(tavg.get("unified_acr_rcedr_ms", 0.0), 1),
            )
        )
    lines.append("")
    return "\\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6T latency isolation runs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-profile-json", default="")
    parser.add_argument("--output-profile-md", default="")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    output_md = Path(_safe_text(args.output_md) or str(out_root / "PHASE6T_LATENCY_ISOLATION_SUMMARY.md")).resolve()
    output_json = Path(_safe_text(args.output_json) or str(out_root / "phase6t_latency_isolation_summary.json")).resolve()
    output_profile_json = Path(
        _safe_text(args.output_profile_json) or str(out_root / "retrieval_profile_summary.json")
    ).resolve()
    output_profile_md = Path(
        _safe_text(args.output_profile_md) or str(out_root / "PHASE6T_LATENCY_PROFILE_SUMMARY.md")
    ).resolve()

    dataset = _safe_text(args.dataset)
    profiles = _tokenize(args.profiles) or list(DEFAULT_PROFILES)

    report = collect_phase6s_artifacts(out_root)
    rows = _build_completed_rows(report, dataset_filter=dataset, profile_filter=profiles)
    timing_rows = _build_profile_timing_rows(rows)
    markdown = _mk_markdown(report, rows, timing_rows)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_profile_json.parent.mkdir(parents=True, exist_ok=True)
    output_profile_md.parent.mkdir(parents=True, exist_ok=True)

    output_md.write_text(markdown + "\n", encoding="utf-8")
    output_profile_json.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "out_root": str(out_root),
                "dataset": dataset,
                "profiles": profiles,
                "runs": timing_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    output_profile_md.write_text(_mk_profile_markdown(timing_rows) + "\\n", encoding="utf-8")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "dataset": dataset,
        "profiles": profiles,
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "rows": rows,
        "timing_rows": timing_rows,
        "report": report,
        "output_md": str(output_md),
        "output_profile_json": str(output_profile_json),
        "output_profile_md": str(output_profile_md),
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(str(output_md))


if __name__ == "__main__":
    main()
