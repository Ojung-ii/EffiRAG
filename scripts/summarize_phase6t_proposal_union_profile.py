#!/usr/bin/env python3
"""Summarize Phase-6T proposal-union deep profiling runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from validate_phase6s_artifacts import collect_phase6s_artifacts

DEFAULT_PROFILE = "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank"

PROPOSAL_UNION_STAGE_KEYS = [
    "proposal_union_total_ms",
    "raw_semantic_entity_fetch_ms",
    "raw_semantic_chunk_fetch_ms",
    "graph_reserve_fetch_ms",
    "entity_to_chunk_expand_ms",
    "chunk_candidate_lookup_ms",
    "candidate_materialization_ms",
    "chunk_text_lookup_ms",
    "title_lookup_ms",
    "metadata_lookup_ms",
    "token_count_lookup_ms",
    "score_merge_ms",
    "score_normalization_ms",
    "dedup_ms",
    "sort_topk_ms",
    "early_pruning_ms",
    "candidate_object_build_ms",
    "candidate_validation_ms",
    "candidate_filter_ms",
    "cache_lookup_ms",
    "cache_miss_io_ms",
    "python_loop_overhead_ms",
]

PROPOSAL_UNION_COUNT_KEYS = [
    "num_raw_semantic_entities",
    "num_raw_semantic_chunks",
    "num_graph_reserve_candidates",
    "num_entity_to_chunk_expansions",
    "num_chunk_candidates_before_dedup",
    "num_chunk_candidates_after_dedup",
    "num_chunk_candidates_after_topk",
    "num_candidate_objects_built",
    "num_text_lookups",
    "num_title_lookups",
    "num_metadata_lookups",
    "num_token_count_lookups",
    "num_score_entries_merged",
    "num_candidates_sorted",
    "num_candidates_filtered",
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


def _mean(values: Sequence[float]) -> float:
    arr = [float(v) for v in values]
    if not arr:
        return 0.0
    return float(sum(arr) / float(len(arr)))


def _percentile(values: Sequence[float], q: float) -> float:
    arr = sorted([float(v) for v in values])
    if not arr:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    idx = (len(arr) - 1) * float(q)
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    frac = float(idx - lo)
    return float(arr[lo] * (1.0 - frac) + arr[hi] * frac)


def _fmt(value: Any, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


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


def _canonical_proposal_union_ms(stage: Mapping[str, Any]) -> float:
    for key in ("proposal_union_total_ms", "proposal_union_ms", "semantic_candidate_union_ms"):
        value = _safe_float(stage.get(key, 0.0), 0.0)
        if value > 0.0:
            return value
    return 0.0


def _build_completed_rows(report: Mapping[str, Any], dataset: str, profile: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in list(report.get("completed_runs", []) or []):
        ds = _safe_text(run.get("dataset"))
        pf = _safe_text(run.get("profile"))
        if dataset and ds != dataset:
            continue
        if profile and pf != profile:
            continue
        summary = dict(run.get("summary", {}) or {})
        f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
        prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
        rows.append(
            {
                "run_name": _safe_text(run.get("run_name")),
                "dataset": ds,
                "profile": pf,
                "EM": _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0),
                "F1": f1,
                "prompt_tokens_avg": prompt_tokens,
                "F1_per_1k_prompt": (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0,
                "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
                "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
                "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
                "supporting_fact_recall": _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0),
                "supporting_fact_precision": _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0),
                "bridge_noise_ratio": _safe_float(summary.get("bridge_noise_ratio", 0.0), 0.0),
                "answer_bearing_chunk_present": _safe_float(summary.get("answer_bearing_chunk_present", 0.0), 0.0),
                "answer_present_but_generation_fail": _safe_float(summary.get("answer_present_but_generation_fail", 0.0), 0.0),
                "summary_path": _safe_text(run.get("summary_path")),
                "query_path": _safe_text(run.get("query_path")),
            }
        )
    rows.sort(key=lambda x: (_safe_text(x["dataset"]), _safe_text(x["run_name"])))
    return rows


def _aggregate_metrics(values: Sequence[float]) -> Dict[str, float]:
    arr = [float(v) for v in values]
    if not arr:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "avg": _mean(arr),
        "p50": _percentile(arr, 0.50),
        "p95": _percentile(arr, 0.95),
        "max": max(arr),
    }


def _build_per_query_records(query_rows: Sequence[Mapping[str, Any]], dataset: str, profile: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in query_rows:
        stage = dict((row.get("latency_breakdown_ms", {}) or {}))
        diag = dict((row.get("retrieval", {}) or {}).get("diagnostics", {}) or {})

        proposal_total = _safe_float(stage.get("proposal_total_ms", stage.get("proposal_time_ms", 0.0)), 0.0)
        proposal_union = _canonical_proposal_union_ms(stage)

        item: Dict[str, Any] = {
            "sample_id": _safe_text(row.get("sample_id")),
            "dataset": dataset,
            "profile": profile,
            "retrieval_ms": _safe_float(stage.get("retrieval_ms", 0.0), 0.0),
            "total_ms": _safe_float(stage.get("total_ms", 0.0), 0.0),
            "proposal_total_ms": proposal_total,
            "proposal_union_total_ms": proposal_union,
            "semantic_candidate_union_ms": proposal_union,
            "proposal_union_ms": proposal_union,
        }
        if item["retrieval_ms"] <= 0.0:
            eff = dict((row.get("efficiency", {}) or {}))
            item["retrieval_ms"] = _safe_float(eff.get("retrieval_latency_ms", eff.get("retrieval_ms", 0.0)), 0.0)
        if item["total_ms"] <= 0.0:
            eff = dict((row.get("efficiency", {}) or {}))
            item["total_ms"] = _safe_float(eff.get("total_latency_ms", eff.get("total_ms", 0.0)), 0.0)

        for key in PROPOSAL_UNION_STAGE_KEYS:
            if key == "proposal_union_total_ms":
                continue
            item[key] = _safe_float(stage.get(key, 0.0), 0.0)

        unique_substages = [
            "raw_semantic_entity_fetch_ms",
            "raw_semantic_chunk_fetch_ms",
            "graph_reserve_fetch_ms",
            "entity_to_chunk_expand_ms",
            "chunk_candidate_lookup_ms",
            "candidate_materialization_ms",
            "chunk_text_lookup_ms",
            "title_lookup_ms",
            "metadata_lookup_ms",
            "token_count_lookup_ms",
            "score_merge_ms",
            "score_normalization_ms",
            "dedup_ms",
            "sort_topk_ms",
            "early_pruning_ms",
            "candidate_object_build_ms",
            "candidate_validation_ms",
            "candidate_filter_ms",
            "cache_lookup_ms",
            "cache_miss_io_ms",
            "python_loop_overhead_ms",
        ]
        substage_total = 0.0
        for key in unique_substages:
            substage_total += _safe_float(item.get(key, 0.0), 0.0)
        item["proposal_substage_total_ms"] = float(substage_total)
        item["unattributed_proposal_ms"] = max(0.0, float(proposal_total) - float(substage_total))
        item["overlap_note"] = (
            "nested_or_overlapping_timers_detected" if (proposal_total > 0.0 and substage_total > proposal_total) else ""
        )

        for key in PROPOSAL_UNION_COUNT_KEYS:
            item[key] = _safe_float(diag.get(key, 0.0), 0.0)

        out.append(item)
    return out


def _build_run_profile(row: Mapping[str, Any]) -> Dict[str, Any]:
    query_path = Path(_safe_text(row.get("query_path")))
    query_rows = _load_jsonl(query_path)
    per_query = _build_per_query_records(
        query_rows,
        dataset=_safe_text(row.get("dataset")),
        profile=_safe_text(row.get("profile")),
    )

    stage_keys = [
        "retrieval_ms",
        "total_ms",
        "proposal_total_ms",
        "proposal_union_total_ms",
        "proposal_substage_total_ms",
        "unattributed_proposal_ms",
    ] + PROPOSAL_UNION_STAGE_KEYS

    stage_metrics: Dict[str, Dict[str, float]] = {}
    for key in stage_keys:
        stage_metrics[key] = _aggregate_metrics([_safe_float(r.get(key, 0.0), 0.0) for r in per_query])

    count_metrics: Dict[str, Dict[str, float]] = {}
    for key in PROPOSAL_UNION_COUNT_KEYS:
        count_metrics[key] = _aggregate_metrics([_safe_float(r.get(key, 0.0), 0.0) for r in per_query])

    retrieval_avg = _safe_float((stage_metrics.get("retrieval_ms", {}) or {}).get("avg", 0.0), 0.0)
    proposal_avg = _safe_float((stage_metrics.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)

    ranking: List[Dict[str, Any]] = []
    for key in PROPOSAL_UNION_STAGE_KEYS:
        avg_ms = _safe_float((stage_metrics.get(key, {}) or {}).get("avg", 0.0), 0.0)
        p95_ms = _safe_float((stage_metrics.get(key, {}) or {}).get("p95", 0.0), 0.0)
        ranking.append(
            {
                "stage": key,
                "avg_ms": avg_ms,
                "p95_ms": p95_ms,
                "share_of_retrieval": (avg_ms / retrieval_avg) if retrieval_avg > 0 else 0.0,
                "share_of_proposal": (avg_ms / proposal_avg) if proposal_avg > 0 else 0.0,
            }
        )
    ranking.sort(key=lambda x: float(x.get("avg_ms", 0.0)), reverse=True)

    return {
        "run_name": _safe_text(row.get("run_name")),
        "dataset": _safe_text(row.get("dataset")),
        "profile": _safe_text(row.get("profile")),
        "num_queries": int(len(per_query)),
        "stage_metrics": stage_metrics,
        "count_metrics": count_metrics,
        "bottleneck_ranking": ranking,
        "per_query": per_query,
    }


def _make_markdown(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run_profiles: Sequence[Mapping[str, Any]],
    overall_ranking: Sequence[Mapping[str, Any]],
) -> str:
    counts = dict(report.get("counts", {}) or {})
    by_run = {str(r.get("run_name")): dict(r) for r in run_profiles}

    lines: List[str] = []
    lines.append("# Phase-6T Proposal-Union Deep Profile")
    lines.append("")

    lines.append("## 1. Executed Runs")
    lines.append("")
    lines.append("| run_name | dataset | profile | EM | F1 | prompt_tokens_avg | retrieval_ms | total_ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _safe_text(row.get("dataset")),
                _safe_text(row.get("profile")),
                _fmt(row.get("EM"), 4),
                _fmt(row.get("F1"), 4),
                _fmt(row.get("prompt_tokens_avg"), 1),
                _fmt(row.get("retrieval_ms"), 1),
                _fmt(row.get("total_ms"), 1),
            )
        )
    lines.append("")

    lines.append("## 2. Top-Level Latency")
    lines.append("")
    lines.append("| run_name | retrieval_ms_avg | proposal_total_ms_avg | proposal_union_total_ms_avg | proposal_union_share_of_proposal | proposal_union_share_of_retrieval |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        run_name = _safe_text(row.get("run_name"))
        rp = by_run.get(run_name, {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        retrieval = _safe_float((sm.get("retrieval_ms", {}) or {}).get("avg", 0.0), 0.0)
        proposal = _safe_float((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        union = _safe_float((sm.get("proposal_union_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                run_name,
                _fmt(retrieval, 1),
                _fmt(proposal, 1),
                _fmt(union, 1),
                _fmt((union / proposal) if proposal > 0 else 0.0, 4),
                _fmt((union / retrieval) if retrieval > 0 else 0.0, 4),
            )
        )
    lines.append("")

    lines.append("## 3. Proposal Union Breakdown")
    lines.append("")
    lines.append("| stage | avg_ms | p95_ms | share_of_retrieval | share_of_proposal |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in overall_ranking:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                _safe_text(item.get("stage")),
                _fmt(item.get("avg_ms"), 2),
                _fmt(item.get("p95_ms"), 2),
                _fmt(item.get("share_of_retrieval"), 4),
                _fmt(item.get("share_of_proposal"), 4),
            )
        )
    lines.append("")

    lines.append("## 4. Candidate Count Diagnostics")
    lines.append("")
    lines.append("| run_name | raw_entities_avg | raw_chunks_avg | after_dedup_avg | after_topk_avg | candidate_objects_built_avg |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        cm = dict(rp.get("count_metrics", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt((cm.get("num_raw_semantic_entities", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_raw_semantic_chunks", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_chunk_candidates_after_dedup", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_chunk_candidates_after_topk", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_candidate_objects_built", {}) or {}).get("avg", 0.0), 2),
            )
        )
    lines.append("")

    lines.append("## 5. Unattributed Time")
    lines.append("")
    lines.append("| run_name | proposal_substage_total_ms | proposal_total_ms | unattributed_proposal_ms | unattributed_share_of_proposal | overlap_note |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        subtotal = _safe_float((sm.get("proposal_substage_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        proposal = _safe_float((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        unattributed = _safe_float((sm.get("unattributed_proposal_ms", {}) or {}).get("avg", 0.0), 0.0)
        overlap_note = "nested_or_overlap_detected" if subtotal > proposal and proposal > 0 else ""
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt(subtotal, 1),
                _fmt(proposal, 1),
                _fmt(unattributed, 1),
                _fmt((unattributed / proposal) if proposal > 0 else 0.0, 4),
                overlap_note,
            )
        )
    lines.append("")

    lines.append("## 6. Bottleneck Ranking")
    lines.append("")
    if overall_ranking:
        top = dict(overall_ranking[0])
        lines.append(
            "- Dominant proposal-union stage: `{}` (avg {} ms, proposal share {}).".format(
                _safe_text(top.get("stage")),
                _fmt(top.get("avg_ms"), 2),
                _fmt(top.get("share_of_proposal"), 4),
            )
        )
    else:
        lines.append("- No completed rows found.")
    lines.append("- Alias handling: `proposal_union_total_ms` is canonical; `proposal_union_ms` and `semantic_candidate_union_ms` are treated as aliases and not double-summed.")
    lines.append("")

    lines.append("## 7. Recommendation")
    lines.append("")
    lines.append("- If one sub-stage exceeds 40% of proposal-union share, optimize that single stage first.")
    lines.append("- If unattributed share exceeds 20%, refine timers before algorithmic changes.")
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

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize proposal-union profiling runs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    dataset = _safe_text(args.dataset)
    profile = _safe_text(args.profile) or DEFAULT_PROFILE

    output_md = Path(_safe_text(args.output_md) or str(out_root / "PHASE6T_PROPOSAL_UNION_PROFILE.md")).resolve()
    output_json = Path(_safe_text(args.output_json) or str(out_root / "proposal_union_profile_summary.json")).resolve()

    report = collect_phase6s_artifacts(out_root)
    rows = _build_completed_rows(report, dataset=dataset, profile=profile)

    run_profiles: List[Dict[str, Any]] = []
    for row in rows:
        rp = _build_run_profile(row)
        run_profiles.append(rp)

        run_name = _safe_text(row.get("run_name"))
        run_root = out_root / "qa_runs" / run_name
        _write_json(
            run_root / "proposal_union_profile_summary.json",
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "out_root": str(out_root),
                "run_name": run_name,
                "dataset": _safe_text(row.get("dataset")),
                "profile": _safe_text(row.get("profile")),
                "summary_path": _safe_text(row.get("summary_path")),
                "query_path": _safe_text(row.get("query_path")),
                "run_profile": {k: v for k, v in rp.items() if k != "per_query"},
            },
        )
        _write_jsonl(run_root / "proposal_union_profile_per_query.jsonl", rp.get("per_query", []))

    overall_map: Dict[str, Dict[str, List[float]]] = {
        key: {"avg_ms": [], "p95_ms": [], "share_of_retrieval": [], "share_of_proposal": []}
        for key in PROPOSAL_UNION_STAGE_KEYS
    }
    for rp in run_profiles:
        sm = dict(rp.get("stage_metrics", {}) or {})
        retrieval_avg = _safe_float((sm.get("retrieval_ms", {}) or {}).get("avg", 0.0), 0.0)
        proposal_avg = _safe_float((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        for key in PROPOSAL_UNION_STAGE_KEYS:
            avg_ms = _safe_float((sm.get(key, {}) or {}).get("avg", 0.0), 0.0)
            p95_ms = _safe_float((sm.get(key, {}) or {}).get("p95", 0.0), 0.0)
            overall_map[key]["avg_ms"].append(avg_ms)
            overall_map[key]["p95_ms"].append(p95_ms)
            overall_map[key]["share_of_retrieval"].append((avg_ms / retrieval_avg) if retrieval_avg > 0 else 0.0)
            overall_map[key]["share_of_proposal"].append((avg_ms / proposal_avg) if proposal_avg > 0 else 0.0)

    overall_ranking: List[Dict[str, Any]] = []
    for key in PROPOSAL_UNION_STAGE_KEYS:
        bucket = overall_map[key]
        overall_ranking.append(
            {
                "stage": key,
                "avg_ms": _mean(bucket["avg_ms"]),
                "p95_ms": _mean(bucket["p95_ms"]),
                "share_of_retrieval": _mean(bucket["share_of_retrieval"]),
                "share_of_proposal": _mean(bucket["share_of_proposal"]),
            }
        )
    overall_ranking.sort(key=lambda x: float(x.get("avg_ms", 0.0)), reverse=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "dataset": dataset,
        "profile": profile,
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "rows": rows,
        "run_profiles": [{k: v for k, v in rp.items() if k != "per_query"} for rp in run_profiles],
        "overall_stage_ranking": overall_ranking,
    }

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_make_markdown(report, rows, run_profiles, overall_ranking) + "\n", encoding="utf-8")
    _write_json(output_json, payload)

    print(str(output_md))


if __name__ == "__main__":
    main()
