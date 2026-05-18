#!/usr/bin/env python3
"""Summarize Phase-6T proposal bottleneck profiling runs."""

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
]

PROPOSAL_STAGE_KEYS = [
    "proposal_total_ms",
    "proposal_time_ms",
    "semantic_entity_scan_ms",
    "semantic_chunk_scan_ms",
    "semantic_score_reuse_ms",
    "semantic_candidate_union_ms",
    "embedding_query_encode_ms",
    "embedding_candidate_rerank_ms",
    "embedding_endpoint_wait_ms",
    "embedding_cache_lookup_ms",
    "bridge_candidate_expansion_ms",
    "bridge_candidate_filter_ms",
    "proposal_union_ms",
    "proposal_subgraph_build_ms",
    "local_subgraph_build_ms",
    "score_normalization_ms",
    "unattributed_proposal_ms",
]

BREAKDOWN_STAGE_KEYS = [
    "retrieval_ms",
    "total_ms",
    "proposal_total_ms",
    "proposal_time_ms",
    "query_preprocess_ms",
    "query_entity_extraction_ms",
    "anchor_candidate_lookup_ms",
    "semantic_scan_ms",
    "semantic_entity_scan_ms",
    "semantic_chunk_scan_ms",
    "semantic_score_reuse_ms",
    "semantic_candidate_union_ms",
    "embedding_query_encode_ms",
    "embedding_candidate_encode_ms",
    "embedding_candidate_rerank_ms",
    "embedding_endpoint_wait_ms",
    "embedding_cache_lookup_ms",
    "bridge_candidate_expansion_ms",
    "bridge_candidate_filter_ms",
    "proposal_union_ms",
    "proposal_subgraph_build_ms",
    "local_subgraph_build_ms",
    "ppr_total_ms",
    "ppr_avg_ms",
    "ppr_p95_ms",
    "phase1_run_generation_ms",
    "phase1_run_scoring_ms",
    "phase1_run_shortlist_ms",
    "seed_selection_ms",
    "pair_construction_ms",
    "pair_scoring_ms",
    "pair_shortlist_ms",
    "corridor_candidate_generation_ms",
    "corridor_feature_extraction_ms",
    "corridor_scoring_ms",
    "corridor_shortlist_ms",
    "local_graph_construction_ms",
    "corridor_extraction_ms",
    "bridge_feature_ms",
    "answerability_feature_ms",
    "redundancy_scoring_ms",
    "unified_acr_rcedr_ms",
    "render_ms",
    "misc_python_overhead_ms",
    "proposal_substage_total_ms",
    "unattributed_proposal_ms",
]

COUNT_KEYS = [
    "num_query_entities",
    "num_anchor_candidates",
    "num_selected_anchors",
    "samples_per_anchor",
    "num_generated_runs",
    "num_semantic_entity_candidates",
    "num_semantic_chunk_candidates",
    "num_union_candidates",
    "num_bridge_candidates_before_filter",
    "num_bridge_candidates_after_filter",
    "num_ppr_calls",
    "num_ppr_sources",
    "num_ppr_target_nodes",
    "num_seed_candidates",
    "num_selected_seeds",
    "num_anchor_seed_pairs",
    "num_pair_candidates",
    "num_corridor_candidates",
    "num_selected_corridors",
    "num_candidate_atoms",
    "num_selected_atoms",
    "objective_eval_calls",
    "local_subgraph_node_count_avg",
    "local_subgraph_edge_count_avg",
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


def _tokenize(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


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
        retrieval_ms = _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0)
        total_ms = _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0)
        rows.append(
            {
                "run_name": _safe_text(run.get("run_name")),
                "dataset": dataset,
                "profile": profile,
                "EM": _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0),
                "F1": f1,
                "prompt_tokens_avg": prompt_tokens,
                "F1_per_1k_prompt": (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0,
                "retrieval_ms": retrieval_ms,
                "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
                "total_ms": total_ms,
                "supporting_fact_recall": _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0),
                "supporting_fact_precision": _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0),
                "bridge_noise_ratio": _safe_float(summary.get("bridge_noise_ratio", 0.0), 0.0),
                "answer_bearing_chunk_present": _safe_float(summary.get("answer_bearing_chunk_present", 0.0), 0.0),
                "answer_present_but_generation_fail": _safe_float(summary.get("answer_present_but_generation_fail", 0.0), 0.0),
                "summary_path": _safe_text(run.get("summary_path")),
                "query_path": _safe_text(run.get("query_path")),
            }
        )
    rows.sort(key=lambda x: (_safe_text(x["dataset"]), _safe_text(x["profile"]), _safe_text(x["run_name"])))
    return rows


def _build_per_query_records(query_rows: Sequence[Mapping[str, Any]], dataset: str, profile: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in query_rows:
        stage = dict((row.get("latency_breakdown_ms", {}) or {}))
        diag = dict(((row.get("retrieval", {}) or {}).get("diagnostics", {}) or {}))
        item: Dict[str, Any] = {
            "sample_id": _safe_text(row.get("sample_id")),
            "dataset": dataset,
            "profile": profile,
        }
        for key in BREAKDOWN_STAGE_KEYS:
            item[key] = _safe_float(stage.get(key, 0.0), 0.0)
        if item.get("retrieval_ms", 0.0) <= 0.0:
            eff = dict((row.get("efficiency", {}) or {}))
            item["retrieval_ms"] = _safe_float(
                eff.get("retrieval_latency_ms", eff.get("retrieval_ms", 0.0)),
                0.0,
            )
        if item.get("total_ms", 0.0) <= 0.0:
            eff = dict((row.get("efficiency", {}) or {}))
            item["total_ms"] = _safe_float(
                eff.get("total_latency_ms", eff.get("total_ms", 0.0)),
                0.0,
            )
        if item["proposal_total_ms"] <= 0.0:
            item["proposal_total_ms"] = _safe_float(stage.get("proposal_time_ms", 0.0), 0.0)
        for key in COUNT_KEYS:
            item[key] = _safe_float(diag.get(key, 0.0), 0.0)
        out.append(item)
    return out


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


def _build_run_profile(row: Mapping[str, Any]) -> Dict[str, Any]:
    query_path = Path(_safe_text(row.get("query_path")))
    query_rows = _load_jsonl(query_path)
    per_query = _build_per_query_records(
        query_rows,
        dataset=_safe_text(row.get("dataset")),
        profile=_safe_text(row.get("profile")),
    )

    stage_metrics: Dict[str, Dict[str, float]] = {}
    for key in BREAKDOWN_STAGE_KEYS:
        stage_metrics[key] = _aggregate_metrics([_safe_float(r.get(key, 0.0), 0.0) for r in per_query])

    count_metrics: Dict[str, Dict[str, float]] = {}
    for key in COUNT_KEYS:
        count_metrics[key] = _aggregate_metrics([_safe_float(r.get(key, 0.0), 0.0) for r in per_query])

    retrieval_avg = _safe_float(stage_metrics.get("retrieval_ms", {}).get("avg", 0.0), 0.0)
    proposal_avg = _safe_float(stage_metrics.get("proposal_total_ms", {}).get("avg", 0.0), 0.0)
    bottleneck_rows: List[Dict[str, Any]] = []
    for key in PROPOSAL_STAGE_KEYS:
        avg_ms = _safe_float(stage_metrics.get(key, {}).get("avg", 0.0), 0.0)
        p95_ms = _safe_float(stage_metrics.get(key, {}).get("p95", 0.0), 0.0)
        bottleneck_rows.append(
            {
                "stage": key,
                "avg_ms": avg_ms,
                "p95_ms": p95_ms,
                "share_of_retrieval": (avg_ms / retrieval_avg) if retrieval_avg > 0 else 0.0,
                "share_of_proposal": (avg_ms / proposal_avg) if proposal_avg > 0 else 0.0,
            }
        )
    bottleneck_rows.sort(key=lambda x: float(x.get("avg_ms", 0.0)), reverse=True)

    return {
        "run_name": _safe_text(row.get("run_name")),
        "dataset": _safe_text(row.get("dataset")),
        "profile": _safe_text(row.get("profile")),
        "num_queries": int(len(per_query)),
        "stage_metrics": stage_metrics,
        "count_metrics": count_metrics,
        "bottleneck_ranking": bottleneck_rows,
        "per_query": per_query,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _make_markdown(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run_profiles: Sequence[Mapping[str, Any]],
    overall_stage_ranking: Sequence[Mapping[str, Any]],
) -> str:
    counts = dict(report.get("counts", {}) or {})
    by_run = {str(r.get("run_name")): dict(r) for r in run_profiles}

    lines: List[str] = []
    lines.append("# Phase-6T Proposal Bottleneck Profile")
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
    lines.append("| run_name | retrieval_ms_avg | proposal_total_ms_avg | proposal_share_of_retrieval | unattributed_proposal_ms_avg |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        run_name = _safe_text(row.get("run_name"))
        rp = by_run.get(run_name, {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        retrieval = _safe_float((sm.get("retrieval_ms", {}) or {}).get("avg", row.get("retrieval_ms", 0.0)), 0.0)
        proposal = _safe_float((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        unattributed = _safe_float((sm.get("unattributed_proposal_ms", {}) or {}).get("avg", 0.0), 0.0)
        share = (proposal / retrieval) if retrieval > 0 else 0.0
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                run_name,
                _fmt(retrieval, 1),
                _fmt(proposal, 1),
                _fmt(share, 4),
                _fmt(unattributed, 1),
            )
        )
    lines.append("")

    lines.append("## 3. Proposal Breakdown")
    lines.append("")
    lines.append("| stage | avg_ms | p95_ms | share_of_retrieval | share_of_proposal |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in overall_stage_ranking:
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

    lines.append("## 4. Embedding/Rerank Breakdown")
    lines.append("")
    lines.append("| run_name | embedding_query_encode_ms | embedding_candidate_rerank_ms | embedding_endpoint_wait_ms | embedding_cache_lookup_ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt((sm.get("embedding_query_encode_ms", {}) or {}).get("avg", 0.0), 1),
                _fmt((sm.get("embedding_candidate_rerank_ms", {}) or {}).get("avg", 0.0), 1),
                _fmt((sm.get("embedding_endpoint_wait_ms", {}) or {}).get("avg", 0.0), 1),
                _fmt((sm.get("embedding_cache_lookup_ms", {}) or {}).get("avg", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 5. PPR/Graph Breakdown")
    lines.append("")
    lines.append("| run_name | ppr_total_ms | ppr_avg_ms | ppr_p95_ms | proposal_subgraph_build_ms | local_subgraph_build_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt((sm.get("ppr_total_ms", {}) or {}).get("avg", 0.0), 1),
                _fmt((sm.get("ppr_avg_ms", {}) or {}).get("avg", 0.0), 1),
                _fmt((sm.get("ppr_p95_ms", {}) or {}).get("avg", 0.0), 1),
                _fmt((sm.get("proposal_subgraph_build_ms", {}) or {}).get("avg", 0.0), 1),
                _fmt((sm.get("local_subgraph_build_ms", {}) or {}).get("avg", 0.0), 1),
            )
        )
    lines.append("")

    lines.append("## 6. Candidate Count Diagnostics")
    lines.append("")
    lines.append("| run_name | num_anchor_candidates_avg | num_union_candidates_avg | num_ppr_calls_avg | num_pair_candidates_avg | num_corridor_candidates_avg | objective_eval_calls_avg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        cm = dict(rp.get("count_metrics", {}) or {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt((cm.get("num_anchor_candidates", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_union_candidates", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_ppr_calls", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_pair_candidates", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("num_corridor_candidates", {}) or {}).get("avg", 0.0), 2),
                _fmt((cm.get("objective_eval_calls", {}) or {}).get("avg", 0.0), 2),
            )
        )
    lines.append("")

    lines.append("## 7. Unattributed Time")
    lines.append("")
    lines.append("| run_name | proposal_substage_total_ms | unattributed_proposal_ms | unattributed_share_of_proposal |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        rp = by_run.get(_safe_text(row.get("run_name")), {})
        sm = dict(rp.get("stage_metrics", {}) or {})
        subtotal = _safe_float((sm.get("proposal_substage_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        unattributed = _safe_float((sm.get("unattributed_proposal_ms", {}) or {}).get("avg", 0.0), 0.0)
        proposal = _safe_float((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        share = (unattributed / proposal) if proposal > 0 else 0.0
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("run_name")),
                _fmt(subtotal, 1),
                _fmt(unattributed, 1),
                _fmt(share, 4),
            )
        )
    lines.append("")

    lines.append("## 8. Bottleneck Ranking")
    lines.append("")
    lines.append("- Largest sub-stage inside `proposal_total_ms` appears at the top of Section 3.")
    lines.append("- Compare `fast` vs `fast_no_sentence_rerank` in Sections 2 and 4 to verify rerank contribution.")
    lines.append("")

    lines.append("## 9. Recommendation")
    lines.append("")
    lines.append("- If a single stage exceeds 40% of proposal share, optimize that stage first.")
    lines.append("- If `unattributed_proposal_ms` stays above 20%, add deeper timers before algorithmic changes.")
    lines.append("- If no-sentence-rerank preserves F1 but proposal share remains dominant, prioritize proposal internals over rerank tuning.")
    lines.append("")

    lines.append("## 10. Artifact Consistency")
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
    parser = argparse.ArgumentParser(description="Summarize proposal bottleneck profiling runs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-profile-json", default="")
    args = parser.parse_args()

    out_root = Path(_safe_text(args.out_root)).resolve()
    dataset = _safe_text(args.dataset)
    profiles = _tokenize(args.profiles) or list(DEFAULT_PROFILES)

    output_md = Path(_safe_text(args.output_md) or str(out_root / "PHASE6T_PROPOSAL_BOTTLENECK_PROFILE.md")).resolve()
    output_json = Path(_safe_text(args.output_json) or str(out_root / "phase6t_proposal_bottleneck_profile_summary.json")).resolve()
    output_profile_json = Path(
        _safe_text(args.output_profile_json) or str(out_root / "retrieval_profile_summary.json")
    ).resolve()

    report = collect_phase6s_artifacts(out_root)
    rows = _build_completed_rows(report, dataset_filter=dataset, profile_filter=profiles)

    run_profiles: List[Dict[str, Any]] = []
    for row in rows:
        rp = _build_run_profile(row)
        run_profiles.append(rp)

        run_name = _safe_text(row.get("run_name"))
        run_root = out_root / "qa_runs" / run_name
        _write_json(run_root / "retrieval_profile_summary.json", {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "out_root": str(out_root),
            "run_name": run_name,
            "dataset": _safe_text(row.get("dataset")),
            "profile": _safe_text(row.get("profile")),
            "summary_path": _safe_text(row.get("summary_path")),
            "query_path": _safe_text(row.get("query_path")),
            "run_profile": {k: v for k, v in rp.items() if k != "per_query"},
        })
        _write_jsonl(run_root / "retrieval_profile_per_query.jsonl", rp.get("per_query", []))

    overall_stage_map: Dict[str, Dict[str, List[float]]] = {
        key: {"avg_ms": [], "p95_ms": [], "share_of_retrieval": [], "share_of_proposal": []}
        for key in PROPOSAL_STAGE_KEYS
    }
    for rp in run_profiles:
        sm = dict(rp.get("stage_metrics", {}) or {})
        retrieval_avg = _safe_float((sm.get("retrieval_ms", {}) or {}).get("avg", 0.0), 0.0)
        proposal_avg = _safe_float((sm.get("proposal_total_ms", {}) or {}).get("avg", 0.0), 0.0)
        for key in PROPOSAL_STAGE_KEYS:
            avg_ms = _safe_float((sm.get(key, {}) or {}).get("avg", 0.0), 0.0)
            p95_ms = _safe_float((sm.get(key, {}) or {}).get("p95", 0.0), 0.0)
            overall_stage_map[key]["avg_ms"].append(avg_ms)
            overall_stage_map[key]["p95_ms"].append(p95_ms)
            overall_stage_map[key]["share_of_retrieval"].append((avg_ms / retrieval_avg) if retrieval_avg > 0 else 0.0)
            overall_stage_map[key]["share_of_proposal"].append((avg_ms / proposal_avg) if proposal_avg > 0 else 0.0)

    overall_stage_ranking: List[Dict[str, Any]] = []
    for key in PROPOSAL_STAGE_KEYS:
        bucket = overall_stage_map[key]
        overall_stage_ranking.append(
            {
                "stage": key,
                "avg_ms": _mean(bucket["avg_ms"]),
                "p95_ms": _mean(bucket["p95_ms"]),
                "share_of_retrieval": _mean(bucket["share_of_retrieval"]),
                "share_of_proposal": _mean(bucket["share_of_proposal"]),
            }
        )
    overall_stage_ranking.sort(key=lambda x: float(x.get("avg_ms", 0.0)), reverse=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "dataset": dataset,
        "profiles": profiles,
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "rows": rows,
        "run_profiles": [{k: v for k, v in rp.items() if k != "per_query"} for rp in run_profiles],
        "overall_stage_ranking": overall_stage_ranking,
    }

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_profile_json.parent.mkdir(parents=True, exist_ok=True)

    output_md.write_text(_make_markdown(report, rows, run_profiles, overall_stage_ranking) + "\n", encoding="utf-8")
    _write_json(output_json, payload)
    _write_json(output_profile_json, payload)

    print(str(output_md))


if __name__ == "__main__":
    main()
