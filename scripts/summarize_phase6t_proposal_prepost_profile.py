#!/usr/bin/env python3
"""Summarize Phase-6T proposal pre/union/post bottleneck profiling runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from validate_phase6s_artifacts import collect_phase6s_artifacts

DEFAULT_PROFILE = "unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank"

PRE_STAGE_KEYS = [
    "query_preprocess_ms",
    "query_entity_extraction_ms",
    "query_embedding_prepare_ms",
    "anchor_extraction_ms",
    "anchor_candidate_lookup_ms",
    "graph_handle_prepare_ms",
    "semantic_score_map_prepare_ms",
    "semantic_score_load_or_reuse_ms",
    "pre_union_misc_ms",
]

UNION_STAGE_KEYS = [
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
    "candidate_filter_ms",
    "candidate_object_build_ms",
    "cache_lookup_ms",
    "cache_miss_io_ms",
]

POST_STAGE_KEYS = [
    "phase1_run_generation_ms",
    "phase1_run_scoring_ms",
    "phase1_run_shortlist_ms",
    "seed_candidate_build_ms",
    "seed_selection_ms",
    "anchor_seed_pair_build_ms",
    "pair_construction_ms",
    "pair_scoring_ms",
    "pair_shortlist_ms",
    "bounded_local_refine_setup_ms",
    "local_subgraph_build_ms",
    "ppr_total_ms",
    "corridor_candidate_generation_ms",
    "corridor_feature_extraction_ms",
    "corridor_scoring_ms",
    "corridor_shortlist_ms",
    "final_candidate_packaging_ms",
    "stagewise_diagnostics_ms",
    "post_union_misc_ms",
]

COUNT_KEYS = [
    "num_query_entities",
    "num_anchor_candidates",
    "num_selected_anchors",
    "semantic_score_map_size",
    "num_generated_runs",
    "num_run_candidates",
    "num_shortlisted_runs",
    "num_seed_candidates",
    "num_selected_seeds",
    "num_anchor_seed_pairs",
    "num_pair_candidates",
    "num_corridor_candidates",
    "num_selected_corridors",
    "num_ppr_calls",
    "local_graph_nodes_avg",
    "local_graph_edges_avg",
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


def _canonical_union_ms(stage: Mapping[str, Any]) -> float:
    for key in ("proposal_union_total_ms", "proposal_union_ms", "semantic_candidate_union_ms"):
        value = _safe_float(stage.get(key, 0.0), 0.0)
        if value > 0.0:
            return value
    return 0.0


def _aggregate(values: Sequence[float]) -> Dict[str, float]:
    arr = [float(v) for v in values]
    if not arr:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "avg": _mean(arr),
        "p50": _percentile(arr, 0.5),
        "p95": _percentile(arr, 0.95),
        "max": max(arr),
    }


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
                "summary_path": _safe_text(run.get("summary_path")),
                "query_path": _safe_text(run.get("query_path")),
            }
        )
    rows.sort(key=lambda x: (_safe_text(x["dataset"]), _safe_text(x["run_name"])))
    return rows


def _build_per_query_records(query_rows: Sequence[Mapping[str, Any]], dataset: str, profile: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in query_rows:
        stage = dict((row.get("latency_breakdown_ms", {}) or {}))
        diag = dict((row.get("retrieval", {}) or {}).get("diagnostics", {}) or {})
        eff = dict((row.get("efficiency", {}) or {}))

        retrieval_ms = _safe_float(stage.get("retrieval_ms", 0.0), 0.0)
        if retrieval_ms <= 0.0:
            retrieval_ms = _safe_float(eff.get("retrieval_latency_ms", eff.get("retrieval_ms", 0.0)), 0.0)

        proposal_total = _safe_float(stage.get("proposal_total_ms", stage.get("proposal_time_ms", 0.0)), 0.0)
        proposal_pre = _safe_float(stage.get("proposal_pre_union_ms", 0.0), 0.0)
        proposal_union = _canonical_union_ms(stage)
        proposal_post = _safe_float(stage.get("proposal_post_union_ms", 0.0), 0.0)
        proposal_known = _safe_float(
            stage.get("proposal_known_total_ms", proposal_pre + proposal_union + proposal_post),
            proposal_pre + proposal_union + proposal_post,
        )
        proposal_unattributed = _safe_float(
            stage.get("unattributed_proposal_ms", proposal_total - proposal_known),
            proposal_total - proposal_known,
        )
        proposal_unattributed_share = (
            (proposal_unattributed / proposal_total) if proposal_total > 0.0 else 0.0
        )

        item: Dict[str, Any] = {
            "sample_id": _safe_text(row.get("sample_id")),
            "dataset": dataset,
            "profile": profile,
            "retrieval_ms": retrieval_ms,
            "proposal_total_ms": proposal_total,
            "proposal_pre_union_ms": proposal_pre,
            "proposal_union_total_ms": proposal_union,
            "proposal_post_union_ms": proposal_post,
            "proposal_known_total_ms": proposal_known,
            "proposal_unattributed_ms": proposal_unattributed,
            "proposal_unattributed_share": proposal_unattributed_share,
            "proposal_timer_overlap_ms": _safe_float(stage.get("proposal_timer_overlap_ms", 0.0), 0.0),
            "overlap_note": _safe_text(diag.get("proposal_timer_overlap_note")),
        }

        for key in PRE_STAGE_KEYS + UNION_STAGE_KEYS + POST_STAGE_KEYS:
            item[key] = _safe_float(stage.get(key, 0.0), 0.0)

        for key in COUNT_KEYS:
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
        "proposal_total_ms",
        "proposal_pre_union_ms",
        "proposal_union_total_ms",
        "proposal_post_union_ms",
        "proposal_known_total_ms",
        "proposal_unattributed_ms",
        "proposal_unattributed_share",
        "proposal_timer_overlap_ms",
    ] + PRE_STAGE_KEYS + UNION_STAGE_KEYS + POST_STAGE_KEYS

    stage_metrics: Dict[str, Dict[str, float]] = {}
    for key in stage_keys:
        stage_metrics[key] = _aggregate([_safe_float(r.get(key, 0.0), 0.0) for r in per_query])

    count_metrics: Dict[str, Dict[str, float]] = {}
    for key in COUNT_KEYS:
        count_metrics[key] = _aggregate([_safe_float(r.get(key, 0.0), 0.0) for r in per_query])

    retrieval_avg = _safe_float((stage_metrics.get("retrieval_ms", {}) or {}).get("avg", 0.0), 0.0)
    proposal_avg = _safe_float((stage_metrics.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)

    ranking_stages = PRE_STAGE_KEYS + UNION_STAGE_KEYS + POST_STAGE_KEYS
    ranking: List[Dict[str, Any]] = []
    for key in ranking_stages:
        avg_ms = _safe_float((stage_metrics.get(key, {}) or {}).get("avg", 0.0), 0.0)
        p95_ms = _safe_float((stage_metrics.get(key, {}) or {}).get("p95", 0.0), 0.0)
        ranking.append(
            {
                "stage": key,
                "avg_ms": avg_ms,
                "p95_ms": p95_ms,
                "share_of_retrieval": (avg_ms / retrieval_avg) if retrieval_avg > 0.0 else 0.0,
                "share_of_proposal": (avg_ms / proposal_avg) if proposal_avg > 0.0 else 0.0,
            }
        )
    ranking.sort(key=lambda x: float(x.get("avg_ms", 0.0)), reverse=True)

    pre_avg = _safe_float((stage_metrics.get("proposal_pre_union_ms", {}) or {}).get("avg", 0.0), 0.0)
    union_avg = _safe_float((stage_metrics.get("proposal_union_total_ms", {}) or {}).get("avg", 0.0), 0.0)
    post_avg = _safe_float((stage_metrics.get("proposal_post_union_ms", {}) or {}).get("avg", 0.0), 0.0)
    dominant_block = max(
        [("pre_union", pre_avg), ("union", union_avg), ("post_union", post_avg)],
        key=lambda x: x[1],
    )[0]

    post_stage_ranking = sorted(
        [
            {
                "stage": k,
                "avg_ms": _safe_float((stage_metrics.get(k, {}) or {}).get("avg", 0.0), 0.0),
                "p95_ms": _safe_float((stage_metrics.get(k, {}) or {}).get("p95", 0.0), 0.0),
            }
            for k in POST_STAGE_KEYS
        ],
        key=lambda x: float(x.get("avg_ms", 0.0)),
        reverse=True,
    )

    overlaps = [r for r in per_query if _safe_float(r.get("proposal_unattributed_ms", 0.0), 0.0) < 0.0]

    return {
        "run_name": _safe_text(row.get("run_name")),
        "dataset": _safe_text(row.get("dataset")),
        "profile": _safe_text(row.get("profile")),
        "num_queries": int(len(per_query)),
        "stage_metrics": stage_metrics,
        "count_metrics": count_metrics,
        "bottleneck_ranking": ranking,
        "dominant_block": dominant_block,
        "largest_post_union_stage": post_stage_ranking[0] if post_stage_ranking else {"stage": "", "avg_ms": 0.0, "p95_ms": 0.0},
        "overlap_detected": bool(len(overlaps) > 0),
        "overlap_count": int(len(overlaps)),
        "per_query": per_query,
    }


def _stage_rank_table(run_profiles: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for key in keys:
        avgs = []
        p50s = []
        p95s = []
        mxs = []
        shares_r = []
        shares_p = []
        for rp in run_profiles:
            sm = dict(rp.get("stage_metrics", {}) or {})
            stage = dict(sm.get(key, {}) or {})
            retrieval_avg = _safe_float((sm.get("retrieval_ms", {}) or {}).get("avg", 0.0), 0.0)
            proposal_avg = _safe_float((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)
            avg_ms = _safe_float(stage.get("avg", 0.0), 0.0)
            avgs.append(avg_ms)
            p50s.append(_safe_float(stage.get("p50", 0.0), 0.0))
            p95s.append(_safe_float(stage.get("p95", 0.0), 0.0))
            mxs.append(_safe_float(stage.get("max", 0.0), 0.0))
            shares_r.append((avg_ms / retrieval_avg) if retrieval_avg > 0 else 0.0)
            shares_p.append((avg_ms / proposal_avg) if proposal_avg > 0 else 0.0)
        table.append(
            {
                "stage": key,
                "avg_ms": _mean(avgs),
                "p50_ms": _mean(p50s),
                "p95_ms": _mean(p95s),
                "max_ms": _mean(mxs),
                "share_of_retrieval": _mean(shares_r),
                "share_of_proposal": _mean(shares_p),
            }
        )
    table.sort(key=lambda x: float(x.get("avg_ms", 0.0)), reverse=True)
    return table


def _make_markdown(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run_profiles: Sequence[Mapping[str, Any]],
    pre_table: Sequence[Mapping[str, Any]],
    union_table: Sequence[Mapping[str, Any]],
    post_table: Sequence[Mapping[str, Any]],
    all_table: Sequence[Mapping[str, Any]],
) -> str:
    counts = dict(report.get("counts", {}) or {})
    by_run = {str(r.get("run_name")): dict(r) for r in run_profiles}

    lines: List[str] = []
    lines.append("# Phase-6T Proposal Pre/Post Profile")
    lines.append("")

    lines.append("## 1. Executed Run")
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

    lines.append("## 2. Top-Level Retrieval Latency")
    lines.append("")
    lines.append("| run_name | proposal_total_ms_avg | retrieval_ms_avg | proposal_share_of_retrieval |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        proposal = _safe_float((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        retrieval = _safe_float((sm.get("retrieval_ms", {}) or {}).get("avg", 0.0), 0.0)
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt(proposal, 1),
                _fmt(retrieval, 1),
                _fmt((proposal / retrieval) if retrieval > 0 else 0.0, 4),
            )
        )
    lines.append("")

    lines.append("## 3. Proposal Pre/Union/Post Breakdown")
    lines.append("")
    lines.append("| run_name | proposal_pre_union_ms | proposal_union_total_ms | proposal_post_union_ms | proposal_known_total_ms | proposal_total_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt((sm.get("proposal_pre_union_ms", {}) or {}).get("avg", 0.0), 2),
                _fmt((sm.get("proposal_union_total_ms", {}) or {}).get("avg", 0.0), 2),
                _fmt((sm.get("proposal_post_union_ms", {}) or {}).get("avg", 0.0), 2),
                _fmt((sm.get("proposal_known_total_ms", {}) or {}).get("avg", 0.0), 2),
                _fmt((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 2),
            )
        )
    lines.append("")

    def _append_stage_table(title: str, table: Sequence[Mapping[str, Any]]) -> None:
        lines.append(title)
        lines.append("")
        lines.append("| stage | avg_ms | p50_ms | p95_ms | max_ms | share_of_retrieval | share_of_proposal |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for item in table:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    _safe_text(item.get("stage")),
                    _fmt(item.get("avg_ms"), 2),
                    _fmt(item.get("p50_ms"), 2),
                    _fmt(item.get("p95_ms"), 2),
                    _fmt(item.get("max_ms"), 2),
                    _fmt(item.get("share_of_retrieval"), 4),
                    _fmt(item.get("share_of_proposal"), 4),
                )
            )
        lines.append("")

    _append_stage_table("## 4. Proposal Pre-Union Breakdown", pre_table)
    _append_stage_table("## 5. Proposal Union Breakdown", union_table)
    _append_stage_table("## 6. Proposal Post-Union Breakdown", post_table)

    lines.append("## 7. Count Diagnostics")
    lines.append("")
    lines.append("| run_name | num_query_entities_avg | num_anchor_candidates_avg | semantic_score_map_size_avg | num_run_candidates_avg | num_shortlisted_runs_avg | num_corridor_candidates_avg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        cm = dict(rp.get("count_metrics", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt((cm.get("num_query_entities", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_anchor_candidates", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("semantic_score_map_size", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_run_candidates", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_shortlisted_runs", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_corridor_candidates", {}) or {}).get("avg", 0.0), 2),
            )
        )
    lines.append("")

    lines.append("## 8. Unattributed Time")
    lines.append("")
    lines.append("| run_name | proposal_unattributed_ms_avg | proposal_unattributed_share_avg | overlap_note |")
    lines.append("|---|---:|---:|---|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        overlap = bool(rp.get("overlap_detected", False))
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt((sm.get("proposal_unattributed_ms", {}) or {}).get("avg", 0.0), 2),
                _fmt((sm.get("proposal_unattributed_share", {}) or {}).get("avg", 0.0), 4),
                "nested_or_overlapping_timers_detected" if overlap else "",
            )
        )
    lines.append("")

    lines.append("## 9. Bottleneck Ranking")
    lines.append("")
    lines.append("| stage | avg_ms | p95_ms | share_of_retrieval | share_of_proposal |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in all_table[:20]:
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

    lines.append("## 10. Recommendation")
    lines.append("")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        dominant = _safe_text(rp.get("dominant_block"))
        largest_post = dict(rp.get("largest_post_union_stage", {}) or {})
        lines.append(
            "- `{}`: dominant block is `{}`; largest post-union stage is `{}` ({:.2f} ms avg).".format(
                _safe_text(row.get("run_name")),
                dominant,
                _safe_text(largest_post.get("stage")),
                _safe_float(largest_post.get("avg_ms", 0.0), 0.0),
            )
        )
    lines.append("")

    lines.append("## 11. Artifact Consistency")
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
    parser = argparse.ArgumentParser(description="Summarize proposal pre/post profile runs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    dataset = _safe_text(args.dataset)
    profile = _safe_text(args.profile) or DEFAULT_PROFILE

    output_md = Path(_safe_text(args.output_md) or str(out_root / "PHASE6T_PROPOSAL_PRE_POST_PROFILE.md")).resolve()
    output_json = Path(_safe_text(args.output_json) or str(out_root / "proposal_pre_post_profile_summary.json")).resolve()

    report = collect_phase6s_artifacts(out_root)
    rows = _build_completed_rows(report, dataset=dataset, profile=profile)

    run_profiles: List[Dict[str, Any]] = []
    for row in rows:
        rp = _build_run_profile(row)
        run_profiles.append(rp)

        run_name = _safe_text(row.get("run_name"))
        run_root = out_root / "qa_runs" / run_name
        _write_json(
            run_root / "proposal_pre_post_profile_summary.json",
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
        _write_jsonl(run_root / "proposal_pre_post_profile_per_query.jsonl", rp.get("per_query", []))

    pre_table = _stage_rank_table(run_profiles, PRE_STAGE_KEYS)
    union_table = _stage_rank_table(run_profiles, UNION_STAGE_KEYS)
    post_table = _stage_rank_table(run_profiles, POST_STAGE_KEYS)
    all_table = _stage_rank_table(run_profiles, PRE_STAGE_KEYS + UNION_STAGE_KEYS + POST_STAGE_KEYS)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "dataset": dataset,
        "profile": profile,
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "rows": rows,
        "run_profiles": [{k: v for k, v in rp.items() if k != "per_query"} for rp in run_profiles],
        "pre_union_ranking": pre_table,
        "union_ranking": union_table,
        "post_union_ranking": post_table,
        "overall_ranking": all_table,
    }

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        _make_markdown(
            report=report,
            rows=rows,
            run_profiles=run_profiles,
            pre_table=pre_table,
            union_table=union_table,
            post_table=post_table,
            all_table=all_table,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(output_json, payload)

    print(str(output_md))


if __name__ == "__main__":
    main()
