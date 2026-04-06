import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .config import RagConfig, apply_cli_overrides, dataclass_from_dict
from .eval.evaluator import QAEvaluator, extract_gold_answers
from .efficiency import Timer, gpu_peak_mb, process_rss_mb, reset_gpu_peak
from .metrics import (
    DEFAULT_RECALL_KS,
    build_support_fact_debug_payload,
    supporting_fact_match_details,
    supporting_fact_precision,
    supporting_fact_recall,
    supporting_fact_recall_at_ks,
)
from .registry import get_dataset_loader, get_generator, get_method, register_defaults
from .render import render_context
from .types import AnchorResult, RetrievalResult
from .utils import (
    append_jsonl,
    load_yaml,
    markdown_table,
    mean_or_zero,
    timestamp_for_filename,
    timestamp_iso_utc,
    write_json,
    write_jsonl,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _NoOpTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])

        def update(self, n=1):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def tqdm(iterable=None, **kwargs):
        return _NoOpTqdm(iterable=iterable, **kwargs)


def _reconstruct_retrieval(payload: dict) -> RetrievalResult:
    anchor_results = []
    for item in payload.get("anchor_results", []) or []:
        anchor_results.append(
            AnchorResult(
                anchor=item.get("anchor", ""),
                scores=item.get("scores", {}) or {},
                top_candidates=item.get("top_candidates", []) or [],
                sample_index=int(item.get("sample_index", 0)),
                metadata=item.get("metadata", {}) or {},
            )
        )

    return RetrievalResult(
        sample_id=str(payload.get("sample_id", "")),
        method=str(payload.get("method", "")),
        anchors=[str(x) for x in (payload.get("anchors", []) or [])],
        seeds=[str(x) for x in (payload.get("seeds", []) or [])],
        selected_nodes=[str(x) for x in (payload.get("selected_nodes", []) or [])],
        selected_sentence_ids=[str(x) for x in (payload.get("selected_sentence_ids", []) or [])],
        selected_sentences=[str(x) for x in (payload.get("selected_sentences", []) or [])],
        corridors=payload.get("corridors", []) or [],
        anchor_results=anchor_results,
        diagnostics=payload.get("diagnostics", {}) or {},
        latency_ms=float(payload.get("latency_ms", 0.0)),
    )


def _load_precomputed_retrieval(path: str):
    by_sample_id = {}
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Precomputed retrieval file not found: {path}")

    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            retrieval_payload = row.get("retrieval", {}) or {}
            if not sample_id:
                continue
            by_sample_id[sample_id] = _reconstruct_retrieval(retrieval_payload)
    return by_sample_id


def _is_generation_fallback(generation) -> bool:
    if generation is None:
        return False
    raw = str(getattr(generation, "raw_text", "") or "")
    return ("HF generation failed" in raw) or ("Fallback(heuristic)" in raw)


def _extract_global_index_diag_from_retrieval(retrieval_payload: dict) -> dict:
    diagnostics = (retrieval_payload or {}).get("diagnostics", {}) or {}
    payload = diagnostics.get("global_index", {}) or {}
    if isinstance(payload, dict):
        return payload
    return {}


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_query_indices(raw: str):
    text = str(raw or "").strip()
    if not text:
        return []
    out = []
    seen = set()
    for token in text.split(","):
        tok = str(token or "").strip()
        if not tok:
            continue
        try:
            idx = int(tok)
        except Exception:
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _select_profile_samples(samples, profile_limit: int = 0, profile_query_indices=None):
    indexed = list(enumerate(samples))
    indices = list(profile_query_indices or [])
    if indices:
        by_idx = {idx: sample for idx, sample in indexed}
        selected = [(idx, by_idx[idx]) for idx in indices if idx in by_idx]
    else:
        selected = indexed
    lim = int(profile_limit or 0)
    if lim > 0:
        selected = selected[:lim]
    return selected


def _percentile(values, p: float) -> float:
    seq = sorted([_safe_float(v, 0.0) for v in (values or [])])
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    pp = max(0.0, min(1.0, float(p)))
    pos = pp * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(seq[lo])
    frac = pos - float(lo)
    return float(seq[lo] * (1.0 - frac) + seq[hi] * frac)


def _describe(values):
    seq = [_safe_float(v, 0.0) for v in (values or [])]
    if not seq:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(seq)),
        "mean": float(sum(seq) / float(len(seq))),
        "median": float(_percentile(seq, 0.5)),
        "p90": float(_percentile(seq, 0.9)),
        "max": float(max(seq)),
    }


def _pearson(xs, ys):
    xv = [_safe_float(x, 0.0) for x in (xs or [])]
    yv = [_safe_float(y, 0.0) for y in (ys or [])]
    n = min(len(xv), len(yv))
    if n < 2:
        return None
    xv = xv[:n]
    yv = yv[:n]
    mx = sum(xv) / float(n)
    my = sum(yv) / float(n)
    num = 0.0
    vx = 0.0
    vy = 0.0
    for x, y in zip(xv, yv):
        dx = float(x) - float(mx)
        dy = float(y) - float(my)
        num += dx * dy
        vx += dx * dx
        vy += dy * dy
    den = math.sqrt(vx * vy)
    if den <= 0.0:
        return None
    return float(num / den)


PROFILE_STAGE_KEYS = [
    "query_embed_ms",
    "semantic_lookup_entity_ms",
    "semantic_lookup_chunk_ms",
    "proposal_union_ms",
    "proposal_subgraph_build_ms",
    "phase1_ppr_ms",
    "phase1_run_scoring_ms",
    "phase2_pair_shortlist_ms",
    "phase2_refine_ms",
    "top1_correction_ms",
    "sentence_rerank_ms",
    "render_ms",
    "generation_ms",
    "retrieval_total_ms",
    "total_ms",
]

ACTIONABLE_STAGE_KEYS = [
    "query_embed_ms",
    "semantic_lookup_entity_ms",
    "semantic_lookup_chunk_ms",
    "proposal_union_ms",
    "proposal_subgraph_build_ms",
    "phase1_ppr_ms",
    "phase1_run_scoring_ms",
    "phase2_pair_shortlist_ms",
    "phase2_refine_ms",
    "top1_correction_ms",
    "sentence_rerank_ms",
    "render_ms",
    "generation_ms",
]


def _build_profile_record(row: dict, retrieval_only: bool = False):
    retrieval = (row.get("retrieval", {}) or {})
    diag = (retrieval.get("diagnostics", {}) or {})
    stage = (diag.get("latency_breakdown_ms", {}) or {})
    efficiency = (row.get("efficiency", {}) or {})
    rendering = (row.get("rendering", {}) or {})
    gen_diag = (row.get("generation_diagnostics", {}) or {})

    retrieval_total_ms = _safe_float(efficiency.get("retrieval_latency_ms", 0.0), 0.0)
    total_ms = _safe_float(efficiency.get("total_latency_ms", retrieval_total_ms), retrieval_total_ms)
    generation_ms = 0.0 if bool(retrieval_only) else _safe_float(efficiency.get("generation_ms", 0.0), 0.0)
    prompt_tokens = 0 if bool(retrieval_only) else _safe_int(gen_diag.get("prompt_tokens", 0), 0)
    completion_tokens = 0 if bool(retrieval_only) else _safe_int(gen_diag.get("completion_tokens", 0), 0)
    finish_reason = "retrieval_only" if bool(retrieval_only) else str(gen_diag.get("finish_reason", "") or "")

    rendered_sentence_count = _safe_int(gen_diag.get("rendered_sentence_count", len(row.get("rendered_sentence_ids", []) or [])), 0)
    truncated_sentence_count = _safe_int(
        gen_diag.get("truncated_sentence_count", rendering.get("truncated_sentences", 0)),
        0,
    )
    truncated_corridor_count = _safe_int(
        gen_diag.get("truncated_corridor_count", rendering.get("truncated_corridors", 0)),
        0,
    )

    return {
        "query_index": _safe_int(row.get("sample_index", -1), -1),
        "query_id": str(row.get("sample_id", "")),
        "question_preview": str(row.get("question", "") or "")[:160],
        "graph_mode": str(diag.get("graph_mode", "current_entity_graph") or "current_entity_graph"),
        "chunk_scoring_mode": str(diag.get("chunk_scoring_mode", "entity_aggregate") or "entity_aggregate"),
        "delivery_mode": str((row.get("rendering", {}) or {}).get("delivery_mode", "sentence_compressed") or "sentence_compressed"),
        "ppr_graph_nodes": _safe_int(diag.get("ppr_graph_nodes", 0), 0),
        "ppr_graph_edges": _safe_int(diag.get("ppr_graph_edges", 0), 0),
        "anchor_count": _safe_int(len(retrieval.get("anchors", []) or []), 0),
        "proposal_entity_count": _safe_int(diag.get("proposal_entity_count", 0), 0),
        "proposal_chunk_count": _safe_int(diag.get("proposal_chunk_count", 0), 0),
        "graph_reserve_count": _safe_int(diag.get("graph_reserve_count", 0), 0),
        "union_candidate_count": _safe_int(diag.get("union_candidate_count", 0), 0),
        "proposal_subgraph_nodes": _safe_int(diag.get("proposal_subgraph_nodes", 0), 0),
        "proposal_subgraph_edges": _safe_int(diag.get("proposal_subgraph_edges", 0), 0),
        "phase1_run_count": _safe_int(diag.get("phase1_run_count", 0), 0),
        "selected_run_count": _safe_int(diag.get("selected_run_count", len(diag.get("shortlisted_run_ids", []) or [])), 0),
        "seed_count": _safe_int(diag.get("num_seeds", len(retrieval.get("seeds", []) or [])), 0),
        "phase2_refined_pair_count": _safe_int(diag.get("phase2_refined_pair_count", 0), 0),
        "corridor_count_before_trim": _safe_int(
            diag.get("corridor_count_before_trim", len(retrieval.get("corridors", []) or [])),
            0,
        ),
        "corridor_count_after_trim": _safe_int(
            diag.get("corridor_count_after_trim", len(retrieval.get("corridors", []) or [])),
            0,
        ),
        "rendered_sentence_count": int(rendered_sentence_count),
        "truncated_sentence_count": int(truncated_sentence_count),
        "truncated_corridor_count": int(truncated_corridor_count),
        "query_embedding_recomputed": bool(diag.get("query_embedding_recomputed", False)),
        "query_embedding_cache_hit": bool(diag.get("query_embedding_cache_hit", False)),
        "semantic_entity_lookup_mode": str(diag.get("semantic_entity_lookup_mode", "")),
        "semantic_chunk_lookup_mode": str(diag.get("semantic_chunk_lookup_mode", "")),
        "candidate_similarity_recomputed_count": _safe_int(diag.get("candidate_similarity_recomputed_count", 0), 0),
        "semantic_scores_reused_in_final": bool(diag.get("semantic_scores_reused_in_final", False)),
        "sentence_rerank_semantic_calls": _safe_int(diag.get("sentence_rerank_semantic_calls", 0), 0),
        "entity_chunk_graph_applied": bool(((diag.get("entity_chunk_graph", {}) or {}).get("applied", False))),
        "entity_chunk_graph_selected_chunks": _safe_int(len(((diag.get("entity_chunk_graph", {}) or {}).get("selected_chunks", [])) or []), 0),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "finish_reason": finish_reason,
        "query_embed_ms": _safe_float(stage.get("query_embed_ms", 0.0), 0.0),
        "semantic_lookup_entity_ms": _safe_float(stage.get("semantic_lookup_entity_ms", 0.0), 0.0),
        "semantic_lookup_chunk_ms": _safe_float(stage.get("semantic_lookup_chunk_ms", 0.0), 0.0),
        "proposal_union_ms": _safe_float(stage.get("proposal_union_ms", 0.0), 0.0),
        "proposal_subgraph_build_ms": _safe_float(stage.get("proposal_subgraph_build_ms", 0.0), 0.0),
        "phase1_ppr_ms": _safe_float(stage.get("phase1_ppr_ms", 0.0), 0.0),
        "phase1_run_scoring_ms": _safe_float(stage.get("phase1_run_scoring_ms", 0.0), 0.0),
        "phase2_pair_shortlist_ms": _safe_float(stage.get("phase2_pair_shortlist_ms", 0.0), 0.0),
        "phase2_refine_ms": _safe_float(stage.get("phase2_refine_ms", 0.0), 0.0),
        "top1_correction_ms": _safe_float(stage.get("top1_correction_ms", 0.0), 0.0),
        "top1_corridor_applied": bool(((diag.get("top1_correction", {}) or {}).get("corridor", {}) or {}).get("applied", False)),
        "top1_sentence_applied": bool(((diag.get("top1_correction", {}) or {}).get("sentence", {}) or {}).get("applied", False)),
        "sentence_rerank_ms": _safe_float(stage.get("sentence_rerank_ms", 0.0), 0.0),
        "render_ms": _safe_float(stage.get("render_ms", _safe_float(efficiency.get("render_ms", 0.0), 0.0)), 0.0),
        "generation_ms": float(generation_ms),
        "retrieval_total_ms": float(retrieval_total_ms),
        "total_ms": float(total_ms),
    }


def _recommend_priority(stage_means: dict, profile_rows):
    phase1_ppr = _safe_float(stage_means.get("phase1_ppr_ms", 0.0), 0.0)
    phase1_run = _safe_float(stage_means.get("phase1_run_scoring_ms", 0.0), 0.0)
    phase1_total = phase1_ppr + phase1_run

    phase2_short = _safe_float(stage_means.get("phase2_pair_shortlist_ms", 0.0), 0.0)
    phase2_refine = _safe_float(stage_means.get("phase2_refine_ms", 0.0), 0.0)
    phase2_total = phase2_short + phase2_refine

    query_embed = _safe_float(stage_means.get("query_embed_ms", 0.0), 0.0)
    sem_ent = _safe_float(stage_means.get("semantic_lookup_entity_ms", 0.0), 0.0)
    sem_chk = _safe_float(stage_means.get("semantic_lookup_chunk_ms", 0.0), 0.0)
    sem_total = query_embed + sem_ent + sem_chk

    proposal_union = _safe_float(stage_means.get("proposal_union_ms", 0.0), 0.0)
    proposal_build = _safe_float(stage_means.get("proposal_subgraph_build_ms", 0.0), 0.0)
    proposal_total = proposal_union + proposal_build

    render_total = _safe_float(stage_means.get("sentence_rerank_ms", 0.0), 0.0) + _safe_float(stage_means.get("render_ms", 0.0), 0.0)
    top_stage = max(stage_means.items(), key=lambda x: x[1])[0] if stage_means else "unknown"

    proposal_nodes_mean = _describe([r.get("proposal_subgraph_nodes", 0) for r in profile_rows]).get("mean", 0.0)
    retrieval_mean = _describe([r.get("retrieval_total_ms", 0.0) for r in profile_rows]).get("mean", 0.0)
    proposal_share = (proposal_build / retrieval_mean) if retrieval_mean > 0.0 else 0.0

    category_scores = {
        "phase1": phase1_total,
        "phase2": phase2_total,
        "semantic": sem_total,
        "proposal": proposal_total,
        "render": render_total,
    }
    top_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else "phase1"

    if top_category == "phase1":
        return {
            "case": "A",
            "primary_stage": top_stage,
            "why": "Phase 1 retrieval stack (PPR + run scoring) dominates retrieval wall-clock.",
            "next_actions": [
                "Further shrink reduced subgraph size before Phase 1 runs.",
                "Tighten proposal/gating to reduce anchor-run workload.",
                "Revisit anchor/run budget (max_anchors, samples_per_anchor).",
            ],
        }
    if top_category == "phase2":
        return {
            "case": "B",
            "primary_stage": top_stage,
            "why": "Phase 2 local refinement dominates retrieval wall-clock.",
            "next_actions": [
                "Reduce pair shortlist budget.",
                "Narrow local BFS/diffusion radius and cache reusable local overlap/path signals.",
                "Keep Phase 2 strictly assembly-oriented, not re-retrieval.",
            ],
        }
    if top_category == "semantic":
        return {
            "case": "C",
            "primary_stage": top_stage,
            "why": "Semantic lookup cost dominates retrieval wall-clock.",
            "next_actions": [
                "Optimize top-N lookup path (ANN/cache) and avoid repeated cosine scans.",
                "Reduce semantic candidate counts where safe.",
                "Enforce semantic score reuse through downstream stages.",
            ],
        }
    if top_category == "proposal" or (proposal_share >= 0.15 and proposal_nodes_mean >= 5000):
        return {
            "case": "D",
            "primary_stage": top_stage,
            "why": "Proposal construction/merge and reduced-subgraph build dominate retrieval wall-clock.",
            "next_actions": [
                "Retune semantic top-N and graph reserve budgets.",
                "Delay or narrow anchor-level proposal merge.",
                "Use support mapping for more direct local graph assembly.",
            ],
        }
    if top_category == "render":
        return {
            "case": "E",
            "primary_stage": top_stage,
            "why": "Render/rerank is a meaningful latency contributor.",
            "next_actions": [
                "Simplify sentence rerank pipeline.",
                "Reduce render-time corridor/sentence ordering overhead.",
            ],
        }
    return {
        "case": "A",
        "primary_stage": top_stage,
        "why": "Top stage by mean latency is treated as current bottleneck.",
        "next_actions": ["Prioritize the top-ranked stage first and re-profile after one targeted change."],
    }


def _build_profile_summary(profile_rows, retrieval_only: bool = False):
    rows = list(profile_rows or [])
    stage_stats = {}
    stage_means = {}
    for key in PROFILE_STAGE_KEYS:
        desc = _describe([row.get(key, 0.0) for row in rows])
        stage_stats[key] = desc
        stage_means[key] = float(desc.get("mean", 0.0))

    actionable_stage_means = {k: float(stage_means.get(k, 0.0)) for k in ACTIONABLE_STAGE_KEYS}
    ranked = sorted(actionable_stage_means.items(), key=lambda x: x[1], reverse=True)
    bottleneck_ranking = [{"stage": stage, "mean_ms": float(val)} for stage, val in ranked]
    primary_stage = ranked[0][0] if ranked else ""

    ordered = sorted(rows, key=lambda x: _safe_float(x.get("retrieval_total_ms", 0.0), 0.0))
    n = len(ordered)
    group_k = max(1, int(math.ceil(float(n) * 0.3))) if n > 0 else 0
    fast_rows = ordered[:group_k] if group_k > 0 else []
    slow_rows = ordered[-group_k:] if group_k > 0 else []

    major_keys = [
        "proposal_subgraph_build_ms",
        "phase1_ppr_ms",
        "phase2_refine_ms",
        "semantic_lookup_entity_ms",
        "semantic_lookup_chunk_ms",
        "sentence_rerank_ms",
        "render_ms",
    ]
    fast_stage_mean = {k: float(_describe([r.get(k, 0.0) for r in fast_rows]).get("mean", 0.0)) for k in major_keys}
    slow_stage_mean = {k: float(_describe([r.get(k, 0.0) for r in slow_rows]).get("mean", 0.0)) for k in major_keys}

    corr_proposal_phase1 = _pearson(
        [r.get("proposal_subgraph_nodes", 0) for r in rows],
        [r.get("phase1_ppr_ms", 0.0) for r in rows],
    )
    corr_pairs_phase2 = _pearson(
        [r.get("phase2_refined_pair_count", 0) for r in rows],
        [r.get("phase2_refine_ms", 0.0) for r in rows],
    )

    semantic_entity_mean = float(stage_stats.get("semantic_lookup_entity_ms", {}).get("mean", 0.0))
    semantic_chunk_mean = float(stage_stats.get("semantic_lookup_chunk_ms", {}).get("mean", 0.0))
    semantic_lookup_total_mean = semantic_entity_mean + semantic_chunk_mean
    retrieval_total_mean = float(stage_stats.get("retrieval_total_ms", {}).get("mean", 0.0))
    semantic_lookup_share = (semantic_lookup_total_mean / retrieval_total_mean) if retrieval_total_mean > 0.0 else 0.0

    representative = {"fast": [], "middle": [], "slow": [], "selected_query_indices": []}
    if len(ordered) >= 10:
        fast_sel = ordered[:3]
        slow_sel = ordered[-4:]
        used = {int(r.get("query_index", -1)) for r in (fast_sel + slow_sel)}
        mids = [r for r in ordered if int(r.get("query_index", -1)) not in used]
        if len(mids) >= 3:
            center = len(mids) // 2
            left = max(0, center - 1)
            right = min(len(mids), left + 3)
            mid_sel = mids[left:right]
            if len(mid_sel) < 3:
                mid_sel = mids[:3]
        else:
            mid_sel = mids[:3]
        representative["fast"] = [int(r.get("query_index", -1)) for r in fast_sel]
        representative["middle"] = [int(r.get("query_index", -1)) for r in mid_sel]
        representative["slow"] = [int(r.get("query_index", -1)) for r in slow_sel]
        representative["selected_query_indices"] = representative["fast"] + representative["middle"] + representative["slow"]

    recommendation = _recommend_priority(actionable_stage_means, rows)

    return {
        "query_count": int(len(rows)),
        "retrieval_only": bool(retrieval_only),
        "stage_stats_ms": stage_stats,
        "stage_bottleneck_ranking": bottleneck_ranking,
        "primary_bottleneck_stage": primary_stage,
        "fast_vs_slow": {
            "group_size": int(group_k),
            "fast_query_indices": [int(r.get("query_index", -1)) for r in fast_rows],
            "slow_query_indices": [int(r.get("query_index", -1)) for r in slow_rows],
            "fast_retrieval_total_ms_mean": float(_describe([r.get("retrieval_total_ms", 0.0) for r in fast_rows]).get("mean", 0.0)),
            "slow_retrieval_total_ms_mean": float(_describe([r.get("retrieval_total_ms", 0.0) for r in slow_rows]).get("mean", 0.0)),
            "fast_stage_mean_ms": fast_stage_mean,
            "slow_stage_mean_ms": slow_stage_mean,
        },
        "correlations": {
            "proposal_subgraph_nodes_vs_phase1_ppr_ms": corr_proposal_phase1,
            "phase2_refined_pair_count_vs_phase2_refine_ms": corr_pairs_phase2,
        },
        "semantic_lookup_breakdown": {
            "entity_mean_ms": semantic_entity_mean,
            "chunk_mean_ms": semantic_chunk_mean,
            "total_mean_ms": semantic_lookup_total_mean,
            "share_of_retrieval_total": float(semantic_lookup_share),
        },
        "representative_top10": representative,
        "next_priority_recommendation": recommendation,
    }


def _profile_summary_markdown(profile_summary: dict):
    stage_stats = profile_summary.get("stage_stats_ms", {}) or {}
    stage_rows = []
    for stage in PROFILE_STAGE_KEYS:
        s = stage_stats.get(stage, {}) or {}
        stage_rows.append(
            [
                stage,
                f"{_safe_float(s.get('mean', 0.0), 0.0):.3f}",
                f"{_safe_float(s.get('median', 0.0), 0.0):.3f}",
                f"{_safe_float(s.get('p90', 0.0), 0.0):.3f}",
                f"{_safe_float(s.get('max', 0.0), 0.0):.3f}",
            ]
        )

    ranking = profile_summary.get("stage_bottleneck_ranking", []) or []
    rank_rows = [[str(i + 1), str(item.get("stage", "")), f"{_safe_float(item.get('mean_ms', 0.0), 0.0):.3f}"] for i, item in enumerate(ranking)]

    fast_slow = profile_summary.get("fast_vs_slow", {}) or {}
    corr = profile_summary.get("correlations", {}) or {}
    semantic_break = profile_summary.get("semantic_lookup_breakdown", {}) or {}
    rec = profile_summary.get("next_priority_recommendation", {}) or {}
    rep = profile_summary.get("representative_top10", {}) or {}

    lines = []
    lines.append(f"# Profiling Summary")
    lines.append("")
    lines.append(f"- query_count: {int(profile_summary.get('query_count', 0))}")
    lines.append(f"- retrieval_only: {bool(profile_summary.get('retrieval_only', False))}")
    lines.append(f"- primary_bottleneck_stage: {str(profile_summary.get('primary_bottleneck_stage', ''))}")
    lines.append("")
    lines.append("## Stage Stats (ms)")
    lines.append(
        markdown_table(
            ["stage", "mean", "median", "p90", "max"],
            stage_rows,
        )
    )
    lines.append("")
    lines.append("## Bottleneck Ranking")
    lines.append(markdown_table(["rank", "stage", "mean_ms"], rank_rows[: min(10, len(rank_rows))]))
    lines.append("")
    lines.append("## Fast vs Slow")
    lines.append(
        markdown_table(
            ["group", "size", "retrieval_total_ms_mean", "query_indices"],
            [
                [
                    "fast",
                    str(int(fast_slow.get("group_size", 0))),
                    f"{_safe_float(fast_slow.get('fast_retrieval_total_ms_mean', 0.0), 0.0):.3f}",
                    ",".join(str(x) for x in (fast_slow.get("fast_query_indices", []) or [])),
                ],
                [
                    "slow",
                    str(int(fast_slow.get("group_size", 0))),
                    f"{_safe_float(fast_slow.get('slow_retrieval_total_ms_mean', 0.0), 0.0):.3f}",
                    ",".join(str(x) for x in (fast_slow.get("slow_query_indices", []) or [])),
                ],
            ],
        )
    )
    lines.append("")
    lines.append("## Correlations")
    lines.append(f"- proposal_subgraph_nodes vs phase1_ppr_ms: {corr.get('proposal_subgraph_nodes_vs_phase1_ppr_ms')}")
    lines.append(f"- phase2_refined_pair_count vs phase2_refine_ms: {corr.get('phase2_refined_pair_count_vs_phase2_refine_ms')}")
    lines.append("")
    lines.append("## Semantic Lookup Breakdown")
    lines.append(
        markdown_table(
            ["entity_mean_ms", "chunk_mean_ms", "total_mean_ms", "share_of_retrieval_total"],
            [[
                f"{_safe_float(semantic_break.get('entity_mean_ms', 0.0), 0.0):.3f}",
                f"{_safe_float(semantic_break.get('chunk_mean_ms', 0.0), 0.0):.3f}",
                f"{_safe_float(semantic_break.get('total_mean_ms', 0.0), 0.0):.3f}",
                f"{_safe_float(semantic_break.get('share_of_retrieval_total', 0.0), 0.0):.4f}",
            ]],
        )
    )
    lines.append("")
    lines.append("## Representative Top10")
    lines.append(f"- fast(3): {','.join(str(x) for x in (rep.get('fast', []) or []))}")
    lines.append(f"- middle(3): {','.join(str(x) for x in (rep.get('middle', []) or []))}")
    lines.append(f"- slow(4): {','.join(str(x) for x in (rep.get('slow', []) or []))}")
    lines.append(f"- selected_query_indices: {','.join(str(x) for x in (rep.get('selected_query_indices', []) or []))}")
    lines.append("")
    lines.append("## Next Priority Recommendation")
    lines.append(f"- case: {str(rec.get('case', ''))}")
    lines.append(f"- primary_stage: {str(rec.get('primary_stage', ''))}")
    lines.append(f"- why: {str(rec.get('why', ''))}")
    for action in (rec.get("next_actions", []) or []):
        lines.append(f"- action: {str(action)}")
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end EffiRAG QA experiments.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", type=str, default=None, choices=["effirag", "naive_graphrag"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--timestamp-output", type=str, default=None)
    parser.add_argument("--global-corpus-path", type=str, default=None)
    parser.add_argument("--graph-cache-dir", type=str, default=None)
    parser.add_argument("--force-rebuild-graph-index", type=str, default=None)
    parser.add_argument("--prebuilt-igraph-path", type=str, default=None)
    parser.add_argument("--prebuilt-igraph-format", type=str, default=None)
    parser.add_argument("--prebuilt-entity-token-limit", type=int, default=None)
    parser.add_argument("--openie-mode", type=str, default=None, choices=["llm", "lexical"])
    parser.add_argument("--openie-model-name", type=str, default=None)
    parser.add_argument("--openie-text-max-chars", type=int, default=None)
    parser.add_argument("--openie-max-new-tokens", type=int, default=None)
    parser.add_argument("--openie-local-files-only", type=str, default=None)
    parser.add_argument("--openie-retry-attempts", type=int, default=None)
    parser.add_argument("--openie-retry-backoff-sec", type=float, default=None)
    parser.add_argument("--openie-error-sample-limit", type=int, default=None)
    parser.add_argument("--openie-api-base-url", type=str, default=None)
    parser.add_argument("--openie-api-key", type=str, default=None)
    parser.add_argument("--openie-api-timeout-sec", type=float, default=None)
    parser.add_argument("--openie-parallel-workers", type=int, default=None)
    parser.add_argument("--openie-log-every", type=int, default=None)
    parser.add_argument("--embedding-enabled", type=str, default=None)
    parser.add_argument("--sentence-rerank-enabled", type=str, default=None)
    parser.add_argument("--top1-correction-enabled", type=str, default=None)
    parser.add_argument("--top1-correction-topk", type=int, default=None)
    parser.add_argument("--top1-correction-corridor-weight-base", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-anchor", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-support", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-bridge", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-semantic", type=float, default=None)
    parser.add_argument("--top1-correction-corridor-weight-redundancy", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-base", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-corridor", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-main", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-support", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-query", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-locality", type=float, default=None)
    parser.add_argument("--top1-correction-sentence-weight-redundancy", type=float, default=None)
    parser.add_argument("--embedding-model-name", type=str, default=None)
    parser.add_argument("--embedding-weight", type=float, default=None)
    parser.add_argument("--embedding-rerank-topn", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-max-length", type=int, default=None)
    parser.add_argument("--embedding-text-max-chars", type=int, default=None)
    parser.add_argument("--candidate-embedding-fallback-enabled", type=str, default=None)
    parser.add_argument("--query-embedding-max-length", type=int, default=None)
    parser.add_argument("--query-embedding-text-max-chars", type=int, default=None)
    parser.add_argument("--query-embedding-cache-enabled", type=str, default=None)
    parser.add_argument("--query-embedding-cache-dir", type=str, default=None)
    parser.add_argument("--semantic-topn-entity", type=int, default=None)
    parser.add_argument("--semantic-topn-chunk", type=int, default=None)
    parser.add_argument("--graph-reserve-topn", type=int, default=None)
    parser.add_argument("--semantic-topn", type=int, default=None)
    parser.add_argument("--index-chunk-unit", type=str, default=None, choices=["auto", "sentence", "passage"])
    parser.add_argument("--graph-mode", type=str, default=None, choices=["current_entity_graph", "entity_chunk_graph"])
    parser.add_argument("--chunk-scoring-mode", type=str, default=None, choices=["entity_aggregate", "direct_chunk"])
    parser.add_argument("--entity-chunk-transition-weight", type=float, default=None)
    parser.add_argument("--chunk-node-enabled-in-diffusion", type=str, default=None)
    parser.add_argument("--chunk-score-topk", type=int, default=None)
    parser.add_argument("--chunk-package-enabled", type=str, default=None)
    parser.add_argument("--entity-lookup-use-two-tier", type=str, default=None)
    parser.add_argument("--entity-lookup-tier1-topk", type=int, default=None)
    parser.add_argument("--entity-lookup-alias-token-limit", type=int, default=None)
    parser.add_argument("--entity-lookup-global-fallback-topn", type=int, default=None)
    parser.add_argument("--semantic-candidate-union", type=str, default=None)
    parser.add_argument("--semantic-chunk-lookup-strategy", type=str, default=None, choices=["adaptive", "full", "cache_only", "two_tier"])
    parser.add_argument("--semantic-chunk-cache-min-ratio", type=float, default=None)
    parser.add_argument("--chunk-lookup-anchor-cache-topn", type=int, default=None)
    parser.add_argument("--chunk-lookup-tier1-topk", type=int, default=None)
    parser.add_argument("--chunk-lookup-dense-topk", type=int, default=None)
    parser.add_argument("--chunk-lookup-global-fallback-topn", type=int, default=None)
    parser.add_argument("--semantic-chunk-support-expand-per-entity", type=int, default=None)
    parser.add_argument("--semantic-chunk-support-expand-total", type=int, default=None)
    parser.add_argument("--semantic-scan-batch-size", type=int, default=None)
    parser.add_argument("--run-score-semantic-weight", type=float, default=None)
    parser.add_argument("--run-score-anchor-weight", type=float, default=None)
    parser.add_argument("--run-score-structure-weight", type=float, default=None)
    parser.add_argument("--run-score-bridge-weight", type=float, default=None)
    parser.add_argument("--run-score-redundancy-weight", type=float, default=None)
    parser.add_argument("--canonical-variant-name", type=str, default=None)
    parser.add_argument("--run-light-rerank-enabled", type=str, default=None)
    parser.add_argument("--run-light-rerank-topk", type=int, default=None)
    parser.add_argument("--run-light-rerank-weight-base", type=float, default=None)
    parser.add_argument("--run-light-rerank-weight-bridge-completeness", type=float, default=None)
    parser.add_argument("--run-light-rerank-weight-anchor-coverage", type=float, default=None)
    parser.add_argument("--run-light-rerank-weight-grounding", type=float, default=None)
    parser.add_argument("--run-score-pair-coverage-weight", type=float, default=None)
    parser.add_argument("--run-score-bridge-completeness-weight", type=float, default=None)
    parser.add_argument("--run-score-entity-chunk-grounding-weight", type=float, default=None)
    parser.add_argument("--run-score-anchor-dispersion-penalty", type=float, default=None)
    parser.add_argument("--seed-score-semantic-weight", type=float, default=None)
    parser.add_argument("--seed-score-graph-weight", type=float, default=None)
    parser.add_argument("--seed-score-anchor-weight", type=float, default=None)
    parser.add_argument("--seed-score-bridge-weight", type=float, default=None)
    parser.add_argument("--seed-score-chunk-grounding-weight", type=float, default=None)
    parser.add_argument("--corridor-score-chunk-support-weight", type=float, default=None)
    parser.add_argument("--corridor-score-answer-alignment-weight", type=float, default=None)

    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--samples-per-anchor", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--candidate-top-t", type=int, default=None)
    parser.add_argument("--seed-k", type=int, default=None)
    parser.add_argument("--pair-top-lp", type=int, default=None)
    parser.add_argument("--corridor-top-bc", type=int, default=None)
    parser.add_argument("--phase1-parallel-ppr", type=str, default=None)
    parser.add_argument("--phase1-run-shortlist-topk", type=int, default=None)
    parser.add_argument("--phase1-run-preshortlist-topm", type=int, default=None)
    parser.add_argument("--phase1-full-run-score-topk", type=int, default=None)
    parser.add_argument("--run-score-sparse-topk", type=int, default=None)
    parser.add_argument("--run-score-surrogate-topk", type=int, default=None)
    parser.add_argument("--proposal-lazy-union-topk", type=int, default=None)
    parser.add_argument("--proposal-anchor-local-topn", type=int, default=None)
    parser.add_argument("--proposal-shared-high-conf-topn", type=int, default=None)
    parser.add_argument("--proposal-global-fallback-topn", type=int, default=None)
    parser.add_argument("--proposal-union-experiment-mode", type=str, default=None, choices=["off", "mild", "medium", "aggressive"])
    parser.add_argument("--proposal-adaptive-budget", type=str, default=None)
    parser.add_argument("--proposal-chunk-rank-cap", type=int, default=None)
    parser.add_argument("--proposal-reserve-hops", type=int, default=None)
    parser.add_argument("--proposal-reserve-graph-mix-topn", type=int, default=None)
    parser.add_argument("--proposal-reserve-bfs-fallback", type=str, default=None)
    parser.add_argument("--proposal-anchor-distance-bonus", type=str, default=None)
    parser.add_argument("--proposal-sparse-subgraph-build", type=str, default=None)
    parser.add_argument("--proposal-subgraph-anchor-neighbor-cap", type=int, default=None)
    parser.add_argument("--proposal-subgraph-support-cap", type=int, default=None)
    parser.add_argument("--proposal-subgraph-connector-cap", type=int, default=None)
    parser.add_argument("--pair-shortlist-topb", type=int, default=None)
    parser.add_argument("--phase2-refine-mode", type=str, default=None)
    parser.add_argument("--phase2-bidirectional-full-ppr", type=str, default=None)
    parser.add_argument("--reuse-semantic-scores-in-final", type=str, default=None)
    parser.add_argument("--final-top-slice-reorder-enabled", type=str, default=None)
    parser.add_argument("--final-top-slice-reorder-topk", type=int, default=None)
    parser.add_argument("--answer-support-pinning-enabled", type=str, default=None)
    parser.add_argument("--answer-support-pinning-min", type=int, default=None)
    parser.add_argument("--oracle-support-injection-enabled", type=str, default=None)
    parser.add_argument("--trim-on", type=str, default=None)
    parser.add_argument("--trim-rho", type=float, default=None)

    parser.add_argument("--run-qa", type=str, default=None)
    parser.add_argument("--evaluator-mode", type=str, default=None, choices=["legacy", "hipporag2_parity"])
    parser.add_argument("--generator", type=str, default=None, choices=["heuristic", "oracle", "hf", "openai_compat", "vllm"])
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-timeout-sec", type=float, default=None)
    parser.add_argument("--llm-max-new-tokens", type=int, default=None)
    parser.add_argument("--max-context-sentences", type=int, default=None)
    parser.add_argument("--render-mode", type=str, default=None, choices=["flat", "corridor", "corridor_aware_flat"])
    parser.add_argument("--max-corridors-in-context", type=int, default=None)
    parser.add_argument("--max-main-sentences-per-corridor", type=int, default=None)
    parser.add_argument("--max-support-per-corridor", type=int, default=None)
    parser.add_argument("--max-total-sentences", type=int, default=None)
    parser.add_argument("--delivery-mode", type=str, default=None, choices=["sentence_compressed", "chunk_package_basic", "chunk_grounded_bridge", "chunk_package_grounded_support"])
    parser.add_argument("--max-chunk-packages", type=int, default=None)
    parser.add_argument("--max-excerpt-sentences-per-package", type=int, default=None)
    parser.add_argument("--chunk-grounding-enabled", type=str, default=None)
    parser.add_argument("--chunk-grounding-mode", type=str, default=None, choices=["sentence_backfill", "corridor_lift", "package_score"])
    parser.add_argument("--chunk-excerpt-max-per-corridor", type=int, default=None)
    parser.add_argument("--chunk-excerpt-window-sentences-before", type=int, default=None)
    parser.add_argument("--chunk-excerpt-window-sentences-after", type=int, default=None)
    parser.add_argument("--chunk-excerpt-max-total-sentences", type=int, default=None)
    parser.add_argument("--chunk-excerpt-dedup-enabled", type=str, default=None)
    parser.add_argument("--chunk-grounding-top-corridor-chunks", type=int, default=None)
    parser.add_argument("--chunk-grounding-top-k-packages", type=int, default=None)
    parser.add_argument("--package-score-answer-weight", type=float, default=None)
    parser.add_argument("--package-score-bridge-weight", type=float, default=None)
    parser.add_argument("--package-score-support-weight", type=float, default=None)
    parser.add_argument("--package-score-chunk-grounding-weight", type=float, default=None)
    parser.add_argument("--package-score-redundancy-weight", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma-main", type=float, default=None)
    parser.add_argument("--delta-support", type=float, default=None)
    parser.add_argument("--eta-connector", type=float, default=None)
    parser.add_argument("--zeta-query", type=float, default=None)
    parser.add_argument("--xi-locality", type=float, default=None)
    parser.add_argument("--lambda-redundancy", type=float, default=None)
    parser.add_argument("--top-corridors", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--reserve-top-corridor", type=str, default=None)
    parser.add_argument("--order-strategy", type=str, default=None, choices=["score", "retrieval", "corridor_rank"])
    parser.add_argument("--measure-gpu-peak", type=str, default=None)
    parser.add_argument("--measure-cpu-ram", type=str, default=None)
    parser.add_argument("--profile-stages", type=str, default=None)
    parser.add_argument("--profile-output", type=str, default=None)
    parser.add_argument("--profile-limit", type=int, default=None)
    parser.add_argument("--profile-query-indices", type=str, default=None)
    parser.add_argument("--retrieval-only", type=str, default=None)
    parser.add_argument("--sf-debug-sample-limit", type=int, default=None)
    parser.add_argument("--sf-debug-output", type=str, default=None)

    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--ppr-alpha", type=float, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--edge-drop-prob", type=float, default=None)
    parser.add_argument("--ppr-engine", type=str, default=None, choices=["auto", "power", "mc"])
    parser.add_argument("--ppr-power-max-iter", type=int, default=None)
    parser.add_argument("--ppr-power-tol", type=float, default=None)
    parser.add_argument("--ppr-min-score", type=float, default=None)
    parser.add_argument("--ppr-mc-walks", type=int, default=None)
    parser.add_argument("--ppr-mc-max-steps", type=int, default=None)
    parser.add_argument("--ppr-parallel-workers", type=int, default=None)
    parser.add_argument("--ppr-subgraph-enable", type=str, default=None)
    parser.add_argument("--ppr-subgraph-hops", type=int, default=None)
    parser.add_argument("--ppr-subgraph-max-nodes", type=int, default=None)
    parser.add_argument("--anchor-diag-topn", type=int, default=None)
    parser.add_argument("--anchor-diag-store-full-scores", type=str, default=None)
    return parser


def execute_rag_experiment(cfg, show_progress: bool = True, precomputed_retrieval_path: str = None):
    register_defaults()

    loader = get_dataset_loader(cfg.dataset)
    method_fn = get_method(cfg.method)
    generator_fn = get_generator(cfg.generator)

    samples_all = loader(split=cfg.split, limit=cfg.limit, data_path=cfg.data_path)
    profile_indices = _parse_query_indices(getattr(cfg, "profile_query_indices", ""))
    profile_limit = int(getattr(cfg, "profile_limit", 0) or 0)
    indexed_samples = _select_profile_samples(
        samples=samples_all,
        profile_limit=profile_limit,
        profile_query_indices=profile_indices,
    )
    run_qa_enabled = bool(getattr(cfg, "run_qa", True)) and (not bool(getattr(cfg, "retrieval_only", False)))
    qa_evaluator = QAEvaluator(mode=str(getattr(cfg, "evaluator_mode", "legacy")))
    retrieval_cache = {}
    if precomputed_retrieval_path:
        retrieval_cache = _load_precomputed_retrieval(precomputed_retrieval_path)

    rows = []
    cfg_values = asdict(cfg)
    run_stamp = timestamp_for_filename()
    run_iso = timestamp_iso_utc()
    if bool(getattr(cfg, "timestamp_output", True)):
        out_dir = Path(cfg.output_dir) / str(cfg.dataset) / run_stamp
    else:
        out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "rag_query_results.jsonl"
    summary_path = out_dir / "rag_summary.json"
    sf_debug_rows = []
    sf_debug_limit = max(0, int(getattr(cfg, "sf_debug_sample_limit", 0) or 0))
    sf_debug_enabled = bool(sf_debug_limit > 0)
    # Stream per-sample outputs so progress is inspectable even before the run ends.
    query_path.write_text("", encoding="utf-8")
    render_mode_requested = str(cfg.render_mode or "").strip()
    resolved_render_mode = render_mode_requested or ("corridor_aware_flat" if cfg.method == "effirag" else "flat")
    retrieval_source_counts = {"precomputed": 0, "on_the_fly": 0}
    fallback_count = 0
    profile_rows = []
    retrieval_bar = tqdm(
        total=len(indexed_samples),
        desc=f"Retrieval[{cfg.method}]",
        unit="sample",
        leave=False,
        disable=not show_progress,
    )
    generation_bar = tqdm(
        total=len(indexed_samples),
        desc=f"Generation[{cfg.generator}]",
        unit="sample",
        leave=False,
        disable=(not show_progress) or (not run_qa_enabled),
    )
    try:
        for sample_index, sample in tqdm(
            indexed_samples,
            total=len(indexed_samples),
            desc=f"RAG[{cfg.method}/{cfg.generator}]",
            leave=False,
            disable=not show_progress,
        ):
            total_timer = Timer()
            total_timer.start()

            if cfg.measure_gpu_peak:
                reset_gpu_peak()

            cpu_peak = process_rss_mb() if cfg.measure_cpu_ram else 0.0

            retrieval = retrieval_cache.get(sample.qid)
            retrieval_source = "precomputed"
            if retrieval is None:
                retrieval = method_fn(sample, cfg)
                retrieval_source = "on_the_fly"
            retrieval_source_counts[retrieval_source] = retrieval_source_counts.get(retrieval_source, 0) + 1

            render_ms = 0.0
            delivery_mode = str(getattr(cfg, "delivery_mode", "sentence_compressed") or "sentence_compressed").strip().lower()
            eff_chunk_grounding_enabled = bool(cfg.chunk_grounding_enabled)
            eff_chunk_grounding_mode = str(cfg.chunk_grounding_mode or "sentence_backfill")
            retrieval_graph_mode = str(((retrieval.diagnostics or {}).get("graph_mode", getattr(cfg, "graph_mode", "current_entity_graph")) or "current_entity_graph")).strip().lower()
            if delivery_mode != "sentence_compressed" and retrieval_graph_mode != "entity_chunk_graph":
                eff_chunk_grounding_enabled = True
                if delivery_mode == "chunk_package_basic":
                    eff_chunk_grounding_mode = "package_score"
                elif delivery_mode == "chunk_grounded_bridge":
                    eff_chunk_grounding_mode = "corridor_lift"
                elif delivery_mode == "chunk_package_grounded_support":
                    eff_chunk_grounding_mode = "package_score"
            elif retrieval_graph_mode == "entity_chunk_graph":
                eff_chunk_grounding_enabled = False

            eff_chunk_top_packages = int(cfg.chunk_grounding_top_k_packages)
            if int(getattr(cfg, "max_chunk_packages", 0) or 0) > 0:
                eff_chunk_top_packages = int(getattr(cfg, "max_chunk_packages", eff_chunk_top_packages))
            eff_chunk_excerpt_total = int(cfg.chunk_excerpt_max_total_sentences)
            max_excerpt_per_pkg = int(getattr(cfg, "max_excerpt_sentences_per_package", 0) or 0)
            if max_excerpt_per_pkg > 0 and eff_chunk_top_packages > 0:
                eff_chunk_excerpt_total = max(eff_chunk_excerpt_total, max_excerpt_per_pkg * eff_chunk_top_packages)

            eff_package_support_weight = float(cfg.package_score_support_weight)
            eff_package_grounding_weight = float(cfg.package_score_chunk_grounding_weight)
            if delivery_mode == "chunk_package_grounded_support":
                eff_package_support_weight = max(eff_package_support_weight, 0.22)
                eff_package_grounding_weight = max(eff_package_grounding_weight, 0.20)

            render_start = time.perf_counter()
            try:
                rendered = render_context(
                    sample,
                    retrieval,
                    max_context_sentences=cfg.max_context_sentences,
                    render_mode=resolved_render_mode,
                    max_corridors_in_context=cfg.max_corridors_in_context,
                    max_main_sentences_per_corridor=cfg.max_main_sentences_per_corridor,
                    max_support_per_corridor=cfg.max_support_per_corridor,
                    max_total_sentences=cfg.max_total_sentences,
                    chunk_grounding_enabled=eff_chunk_grounding_enabled,
                    chunk_grounding_mode=eff_chunk_grounding_mode,
                    chunk_excerpt_max_per_corridor=cfg.chunk_excerpt_max_per_corridor,
                    chunk_excerpt_window_sentences_before=cfg.chunk_excerpt_window_sentences_before,
                    chunk_excerpt_window_sentences_after=cfg.chunk_excerpt_window_sentences_after,
                    chunk_excerpt_max_total_sentences=eff_chunk_excerpt_total,
                    chunk_excerpt_dedup_enabled=cfg.chunk_excerpt_dedup_enabled,
                    chunk_grounding_top_corridor_chunks=cfg.chunk_grounding_top_corridor_chunks,
                    chunk_grounding_top_k_packages=eff_chunk_top_packages,
                    package_score_answer_weight=cfg.package_score_answer_weight,
                    package_score_bridge_weight=cfg.package_score_bridge_weight,
                    package_score_support_weight=eff_package_support_weight,
                    package_score_chunk_grounding_weight=eff_package_grounding_weight,
                    package_score_redundancy_weight=cfg.package_score_redundancy_weight,
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    gamma_main=cfg.gamma_main,
                    delta_support=cfg.delta_support,
                    eta_connector=cfg.eta_connector,
                    zeta_query=cfg.zeta_query,
                    xi_locality=cfg.xi_locality,
                    lambda_redundancy=cfg.lambda_redundancy,
                    top_corridors=cfg.top_corridors,
                    max_sentences=cfg.max_sentences,
                    reserve_top_corridor=cfg.reserve_top_corridor,
                    order_strategy=cfg.order_strategy,
                )
            finally:
                render_ms = float((time.perf_counter() - render_start) * 1000.0)
            retrieval_bar.update(1)

            if cfg.measure_cpu_ram:
                cpu_peak = max(cpu_peak, process_rss_mb())

            generation = None
            em = 0.0
            f1 = 0.0
            generation_call_ms = 0.0
            gold_answers = extract_gold_answers(sample) or [str(sample.answer or "")]
            if run_qa_enabled:
                generation_start = time.perf_counter()
                try:
                    generation = generator_fn(sample, rendered, cfg.model_name, cfg)
                finally:
                    generation_call_ms = float((time.perf_counter() - generation_start) * 1000.0)
                eval_result = qa_evaluator.evaluate(generation.prediction, sample)
                em = float(eval_result.em)
                f1 = float(eval_result.f1)
                if _is_generation_fallback(generation):
                    fallback_count += 1
                generation_bar.update(1)

            if cfg.measure_cpu_ram:
                cpu_peak = max(cpu_peak, process_rss_mb())

            generation_meta = (generation.metadata if generation is not None else {}) or {}
            prompt_tokens = _safe_int(generation_meta.get("prompt_tokens", 0) or 0, default=0)
            completion_tokens = _safe_int(generation_meta.get("completion_tokens", 0) or 0, default=0)
            finish_reason = str(generation_meta.get("finish_reason", "") or "")
            rendered_sentence_count = int(len(rendered.sentence_ids))
            truncated_sentence_count = int(rendered.truncated_sentence_count)
            truncated_corridor_count = int(rendered.truncated_corridor_count)
            rendered_meta = (rendered.metadata or {}) if rendered is not None else {}
            chunk_excerpts_used = _safe_int(rendered_meta.get("chunk_excerpts_used", 0) or 0, default=0)
            chunk_excerpt_sentence_count = _safe_int(
                rendered_meta.get("chunk_excerpt_sentence_count", 0) or 0, default=0
            )
            evidence_package_count = _safe_int(rendered_meta.get("evidence_package_count", 0) or 0, default=0)
            chunk_excerpt_truncated_count = _safe_int(
                rendered_meta.get("chunk_excerpt_truncated_count", 0) or 0, default=0
            )
            chunk_excerpt_avg_len = 0.0
            if chunk_excerpts_used > 0:
                chunk_excerpt_avg_len = float(chunk_excerpt_sentence_count) / float(chunk_excerpts_used)

            recall = supporting_fact_recall(sample, retrieval)
            precision = supporting_fact_precision(sample, retrieval)
            recall_at_k = supporting_fact_recall_at_ks(sample, retrieval, ks=DEFAULT_RECALL_KS)
            rendered_match = supporting_fact_match_details(
                sample=sample,
                unit_ids=list(rendered.sentence_ids or []),
                unit_texts=list(rendered.sentences or []),
                graph_mode=retrieval_graph_mode,
            )
            rendered_gold_total = int(rendered_match.get("gold_total", 0))
            rendered_pred_total = int(rendered_match.get("predicted_unit_total", 0))
            rendered_recall = float(rendered_match.get("matched_gold_total", 0)) / float(rendered_gold_total) if rendered_gold_total > 0 else 0.0
            rendered_precision = float(rendered_match.get("matched_unit_total", 0)) / float(rendered_pred_total) if rendered_pred_total > 0 else 0.0

            efficiency = {
                "retrieval_latency_ms": retrieval.latency_ms,
                "generation_latency_ms": generation.latency_ms if generation else 0.0,
                "generation_ms": generation_call_ms if generation else 0.0,
                "render_ms": render_ms,
                "total_latency_ms": total_timer.elapsed_ms(),
            }
            if cfg.measure_gpu_peak:
                efficiency["gpu_peak_mb"] = gpu_peak_mb()
            if cfg.measure_cpu_ram:
                efficiency["cpu_ram_peak_mb"] = cpu_peak

            metrics = {
                "supporting_fact_recall": recall,
                "supporting_fact_precision": precision,
                "recall_at_k": recall_at_k,
                "rendered_supporting_fact_recall": rendered_recall,
                "rendered_supporting_fact_precision": rendered_precision,
                "em": em,
                "f1": f1,
            }
            latency_breakdown = dict(
                ((retrieval.diagnostics or {}).get("latency_breakdown_ms", {}) or {})
            )
            latency_breakdown["generation_ms"] = float(generation_call_ms if generation else 0.0)
            latency_breakdown["render_context_ms"] = float(render_ms)

            row = {
                "sample_index": int(sample_index),
                "sample_id": sample.qid,
                "question": sample.question,
                "answer": sample.answer,
                "gold_answers": list(gold_answers),
                "prediction": generation.prediction if generation else "",
                "qa_executed": bool(generation is not None),
                "generation_fallback": bool(_is_generation_fallback(generation)),
                "evaluator_mode": str(qa_evaluator.mode),
                "method": retrieval.method,
                "generator": cfg.generator,
                "metrics": metrics,
                "efficiency": efficiency,
                "latency_breakdown_ms": latency_breakdown,
                "retrieval": asdict(retrieval),
                "retrieval_selected_sentence_ids": list(retrieval.selected_sentence_ids),
                "rendered": asdict(rendered),
                "rendered_sentence_ids": list(rendered.sentence_ids),
                "rendered_corridor_ids": list(rendered.rendered_corridor_ids),
                "rendering": {
                    "render_mode": rendered.render_mode,
                    "render_mode_requested": render_mode_requested or "(auto)",
                    "truncated_corridors": rendered.truncated_corridor_count,
                    "truncated_sentences": rendered.truncated_sentence_count,
                    "chunk_grounding_enabled": bool(rendered_meta.get("chunk_grounding_enabled", False)),
                    "chunk_grounding_mode": str(rendered_meta.get("chunk_grounding_mode", "") or ""),
                    "chunk_excerpts_used": int(chunk_excerpts_used),
                    "chunk_excerpt_sentence_count": int(chunk_excerpt_sentence_count),
                    "chunk_excerpt_avg_len": float(chunk_excerpt_avg_len),
                    "evidence_package_count": int(evidence_package_count),
                    "chunk_excerpt_truncated_count": int(chunk_excerpt_truncated_count),
                    "oracle_support_injection_applied": bool(rendered_meta.get("oracle_support_injection_applied", False)),
                    "oracle_support_injected": int(_safe_int(rendered_meta.get("oracle_support_injected", 0), 0)),
                },
                "generation_diagnostics": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "rendered_sentence_count": rendered_sentence_count,
                    "truncated_sentence_count": truncated_sentence_count,
                    "truncated_corridor_count": truncated_corridor_count,
                    "chunk_excerpts_used": int(chunk_excerpts_used),
                    "chunk_excerpt_sentence_count": int(chunk_excerpt_sentence_count),
                    "chunk_excerpt_avg_len": float(chunk_excerpt_avg_len),
                    "evidence_package_count": int(evidence_package_count),
                    "chunk_excerpt_truncated_count": int(chunk_excerpt_truncated_count),
                },
                "retrieval_source": retrieval_source,
                "oracle_support_injection_enabled": bool(getattr(cfg, "oracle_support_injection_enabled", False)),
                "generation": asdict(generation) if generation else None,
                "run_timestamp": run_stamp,
            }
            rows.append(row)
            append_jsonl(query_path, row)
            if sf_debug_enabled and len(sf_debug_rows) < sf_debug_limit:
                sf_debug_rows.append(
                    build_support_fact_debug_payload(
                        sample=sample,
                        retrieval=retrieval,
                        rendered=rendered,
                    )
                )
            if bool(getattr(cfg, "profile_stages", False)):
                profile_rows.append(_build_profile_record(row=row, retrieval_only=bool(getattr(cfg, "retrieval_only", False))))
    finally:
        retrieval_bar.close()
        generation_bar.close()

    summary = {
        "n_samples": float(len(rows)),
        "dataset": cfg.dataset,
        "method": cfg.method,
        "generator": cfg.generator,
        "generator_display": str(cfg.model_name or cfg.generator),
        "evaluator_mode": str(qa_evaluator.mode),
        "run_qa": bool(run_qa_enabled),
        "run_qa_requested": bool(getattr(cfg, "run_qa", True)),
        "retrieval_only": bool(getattr(cfg, "retrieval_only", False)),
        "oracle_support_injection_enabled": bool(getattr(cfg, "oracle_support_injection_enabled", False)),
        "oracle_mode": "oracle_upper_bound" if bool(getattr(cfg, "oracle_support_injection_enabled", False)) else "non_oracle",
        "qa_executed_samples": float(sum(1 for r in rows if r.get("qa_executed"))),
        "qa_skipped_samples": float(sum(1 for r in rows if not r.get("qa_executed"))),
        "render_mode_requested": render_mode_requested or "(auto)",
        "render_mode_resolved": resolved_render_mode,
        "precomputed_retrieval_used": bool(precomputed_retrieval_path),
        "retrieval_source_breakdown": retrieval_source_counts,
        "fallback_count": int(fallback_count),
        "fallback_rate": mean_or_zero(
            [1.0 if r.get("generation_fallback", False) else 0.0 for r in rows if r.get("qa_executed", False)]
        ),
        "oracle_support_injection_applied_rate": mean_or_zero(
            [
                1.0
                if bool((r.get("rendering", {}) or {}).get("oracle_support_injection_applied", False))
                else 0.0
                for r in rows
            ]
        ),
        "supporting_fact_recall": mean_or_zero([r["metrics"]["supporting_fact_recall"] for r in rows]),
        "supporting_fact_precision": mean_or_zero([r["metrics"]["supporting_fact_precision"] for r in rows]),
        "rendered_supporting_fact_recall": mean_or_zero(
            [r["metrics"].get("rendered_supporting_fact_recall", 0.0) for r in rows]
        ),
        "rendered_supporting_fact_precision": mean_or_zero(
            [r["metrics"].get("rendered_supporting_fact_precision", 0.0) for r in rows]
        ),
        "em": mean_or_zero([r["metrics"]["em"] for r in rows]),
        "f1": mean_or_zero([r["metrics"]["f1"] for r in rows]),
        "retrieval_latency_ms": mean_or_zero([r["efficiency"]["retrieval_latency_ms"] for r in rows]),
        "render_ms": mean_or_zero([r["efficiency"].get("render_ms", 0.0) for r in rows]),
        "total_latency_ms": mean_or_zero([r["efficiency"]["total_latency_ms"] for r in rows]),
        "prompt_tokens_avg": mean_or_zero(
            [float((r.get("generation_diagnostics", {}) or {}).get("prompt_tokens", 0.0)) for r in rows]
        ),
        "completion_tokens_avg": mean_or_zero(
            [float((r.get("generation_diagnostics", {}) or {}).get("completion_tokens", 0.0)) for r in rows]
        ),
        "rendered_sentence_count_avg": mean_or_zero(
            [float((r.get("generation_diagnostics", {}) or {}).get("rendered_sentence_count", 0.0)) for r in rows]
        ),
        "truncated_sentence_count_avg": mean_or_zero(
            [float((r.get("generation_diagnostics", {}) or {}).get("truncated_sentence_count", 0.0)) for r in rows]
        ),
        "truncated_corridor_count_avg": mean_or_zero(
            [float((r.get("generation_diagnostics", {}) or {}).get("truncated_corridor_count", 0.0)) for r in rows]
        ),
        "truncated_corridors_avg": mean_or_zero([r.get("rendering", {}).get("truncated_corridors", 0.0) for r in rows]),
        "truncated_sentences_avg": mean_or_zero([r.get("rendering", {}).get("truncated_sentences", 0.0) for r in rows]),
        "chunk_excerpts_used_avg": mean_or_zero(
            [float((r.get("rendering", {}) or {}).get("chunk_excerpts_used", 0.0)) for r in rows]
        ),
        "chunk_excerpt_sentence_count_avg": mean_or_zero(
            [float((r.get("rendering", {}) or {}).get("chunk_excerpt_sentence_count", 0.0)) for r in rows]
        ),
        "chunk_excerpt_avg_len": mean_or_zero(
            [float((r.get("rendering", {}) or {}).get("chunk_excerpt_avg_len", 0.0)) for r in rows]
        ),
        "evidence_package_count_avg": mean_or_zero(
            [float((r.get("rendering", {}) or {}).get("evidence_package_count", 0.0)) for r in rows]
        ),
        "chunk_excerpt_truncated_count_avg": mean_or_zero(
            [float((r.get("rendering", {}) or {}).get("chunk_excerpt_truncated_count", 0.0)) for r in rows]
        ),
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "retrieval_params": {
            "global_corpus_path": cfg.global_corpus_path,
            "graph_cache_dir": cfg.graph_cache_dir,
            "force_rebuild_graph_index": cfg.force_rebuild_graph_index,
            "prebuilt_igraph_path": cfg.prebuilt_igraph_path,
            "prebuilt_igraph_format": cfg.prebuilt_igraph_format,
            "prebuilt_entity_token_limit": cfg.prebuilt_entity_token_limit,
            "openie_mode": cfg.openie_mode,
            "openie_model_name": cfg.openie_model_name,
            "openie_text_max_chars": cfg.openie_text_max_chars,
            "openie_max_new_tokens": cfg.openie_max_new_tokens,
            "openie_local_files_only": cfg.openie_local_files_only,
            "openie_retry_attempts": cfg.openie_retry_attempts,
            "openie_retry_backoff_sec": cfg.openie_retry_backoff_sec,
            "openie_error_sample_limit": cfg.openie_error_sample_limit,
            "openie_api_base_url": cfg.openie_api_base_url,
            "openie_api_timeout_sec": cfg.openie_api_timeout_sec,
            "openie_parallel_workers": cfg.openie_parallel_workers,
            "openie_log_every": cfg.openie_log_every,
            "embedding_enabled": cfg.embedding_enabled,
            "embedding_model_name": cfg.embedding_model_name,
            "embedding_weight": cfg.embedding_weight,
            "embedding_rerank_topn": cfg.embedding_rerank_topn,
            "embedding_batch_size": cfg.embedding_batch_size,
            "embedding_max_length": cfg.embedding_max_length,
            "embedding_text_max_chars": cfg.embedding_text_max_chars,
            "semantic_topn_entity": cfg.semantic_topn_entity,
            "semantic_topn_chunk": cfg.semantic_topn_chunk,
            "graph_reserve_topn": cfg.graph_reserve_topn,
            "semantic_topn": cfg.semantic_topn,
            "index_chunk_unit": cfg.index_chunk_unit,
            "graph_mode": cfg.graph_mode,
            "chunk_scoring_mode": cfg.chunk_scoring_mode,
            "entity_chunk_transition_weight": cfg.entity_chunk_transition_weight,
            "chunk_node_enabled_in_diffusion": cfg.chunk_node_enabled_in_diffusion,
            "chunk_score_topk": cfg.chunk_score_topk,
            "chunk_package_enabled": cfg.chunk_package_enabled,
            "semantic_candidate_union": cfg.semantic_candidate_union,
            "semantic_scan_batch_size": cfg.semantic_scan_batch_size,
            "semantic_chunk_lookup_strategy": cfg.semantic_chunk_lookup_strategy,
            "proposal_union_experiment_mode": str(cfg.proposal_union_experiment_mode),
            "run_score_semantic_weight": cfg.run_score_semantic_weight,
            "run_score_anchor_weight": cfg.run_score_anchor_weight,
            "run_score_structure_weight": cfg.run_score_structure_weight,
            "run_score_bridge_weight": cfg.run_score_bridge_weight,
            "run_score_redundancy_weight": cfg.run_score_redundancy_weight,
            "canonical_variant_name": cfg.canonical_variant_name,
            "run_light_rerank_enabled": cfg.run_light_rerank_enabled,
            "run_light_rerank_topk": cfg.run_light_rerank_topk,
            "run_light_rerank_weight_base": cfg.run_light_rerank_weight_base,
            "run_light_rerank_weight_bridge_completeness": cfg.run_light_rerank_weight_bridge_completeness,
            "run_light_rerank_weight_anchor_coverage": cfg.run_light_rerank_weight_anchor_coverage,
            "run_light_rerank_weight_grounding": cfg.run_light_rerank_weight_grounding,
            "run_score_pair_coverage_weight": cfg.run_score_pair_coverage_weight,
            "run_score_bridge_completeness_weight": cfg.run_score_bridge_completeness_weight,
            "run_score_entity_chunk_grounding_weight": cfg.run_score_entity_chunk_grounding_weight,
            "run_score_anchor_dispersion_penalty": cfg.run_score_anchor_dispersion_penalty,
            "seed_score_semantic_weight": cfg.seed_score_semantic_weight,
            "seed_score_graph_weight": cfg.seed_score_graph_weight,
            "seed_score_anchor_weight": cfg.seed_score_anchor_weight,
            "seed_score_bridge_weight": cfg.seed_score_bridge_weight,
            "seed_score_chunk_grounding_weight": cfg.seed_score_chunk_grounding_weight,
            "corridor_score_chunk_support_weight": cfg.corridor_score_chunk_support_weight,
            "corridor_score_answer_alignment_weight": cfg.corridor_score_answer_alignment_weight,
            "top1_correction_enabled": cfg.top1_correction_enabled,
            "top1_correction_topk": cfg.top1_correction_topk,
            "top1_correction_corridor_weight_base": cfg.top1_correction_corridor_weight_base,
            "top1_correction_corridor_weight_anchor": cfg.top1_correction_corridor_weight_anchor,
            "top1_correction_corridor_weight_support": cfg.top1_correction_corridor_weight_support,
            "top1_correction_corridor_weight_bridge": cfg.top1_correction_corridor_weight_bridge,
            "top1_correction_corridor_weight_semantic": cfg.top1_correction_corridor_weight_semantic,
            "top1_correction_corridor_weight_redundancy": cfg.top1_correction_corridor_weight_redundancy,
            "top1_correction_sentence_weight_base": cfg.top1_correction_sentence_weight_base,
            "top1_correction_sentence_weight_corridor": cfg.top1_correction_sentence_weight_corridor,
            "top1_correction_sentence_weight_main": cfg.top1_correction_sentence_weight_main,
            "top1_correction_sentence_weight_support": cfg.top1_correction_sentence_weight_support,
            "top1_correction_sentence_weight_query": cfg.top1_correction_sentence_weight_query,
            "top1_correction_sentence_weight_locality": cfg.top1_correction_sentence_weight_locality,
            "top1_correction_sentence_weight_redundancy": cfg.top1_correction_sentence_weight_redundancy,
            "max_anchors": cfg.max_anchors,
            "samples_per_anchor": cfg.samples_per_anchor,
            "num_workers": cfg.num_workers,
            "candidate_top_t": cfg.candidate_top_t,
            "seed_k": cfg.seed_k,
            "pair_top_lp": cfg.pair_top_lp,
            "corridor_top_bc": cfg.corridor_top_bc,
            "phase1_parallel_ppr": cfg.phase1_parallel_ppr,
            "phase1_run_shortlist_topk": cfg.phase1_run_shortlist_topk,
            "phase1_run_preshortlist_topm": cfg.phase1_run_preshortlist_topm,
            "phase1_full_run_score_topk": cfg.phase1_full_run_score_topk,
            "pair_shortlist_topb": cfg.pair_shortlist_topb,
            "phase2_refine_mode": cfg.phase2_refine_mode,
            "phase2_bidirectional_full_ppr": cfg.phase2_bidirectional_full_ppr,
            "reuse_semantic_scores_in_final": cfg.reuse_semantic_scores_in_final,
            "final_top_slice_reorder_enabled": cfg.final_top_slice_reorder_enabled,
            "final_top_slice_reorder_topk": cfg.final_top_slice_reorder_topk,
            "answer_support_pinning_enabled": cfg.answer_support_pinning_enabled,
            "answer_support_pinning_min": cfg.answer_support_pinning_min,
            "oracle_support_injection_enabled": cfg.oracle_support_injection_enabled,
            "trim_on": cfg.trim_on,
            "trim_rho": cfg.trim_rho,
            "ppr_alpha": cfg.ppr_alpha,
            "tau": cfg.tau,
            "edge_drop_prob": cfg.edge_drop_prob,
            "ppr_engine": cfg.ppr_engine,
            "ppr_power_max_iter": cfg.ppr_power_max_iter,
            "ppr_power_tol": cfg.ppr_power_tol,
            "ppr_min_score": cfg.ppr_min_score,
            "ppr_mc_walks": cfg.ppr_mc_walks,
            "ppr_mc_max_steps": cfg.ppr_mc_max_steps,
            "ppr_parallel_workers": cfg.ppr_parallel_workers,
            "ppr_subgraph_enable": cfg.ppr_subgraph_enable,
            "ppr_subgraph_hops": cfg.ppr_subgraph_hops,
            "ppr_subgraph_max_nodes": cfg.ppr_subgraph_max_nodes,
            "anchor_diag_topn": cfg.anchor_diag_topn,
            "anchor_diag_store_full_scores": cfg.anchor_diag_store_full_scores,
            "random_seed": cfg.random_seed,
        },
        "generation_params": {
            "model_name": cfg.model_name,
            "llm_base_url": cfg.llm_base_url,
            "llm_timeout_sec": cfg.llm_timeout_sec,
            "llm_max_new_tokens": cfg.llm_max_new_tokens,
        },
        "evaluation_params": {
            "evaluator_mode": str(qa_evaluator.mode),
        },
        "render_params": {
            "max_context_sentences": cfg.max_context_sentences,
            "max_corridors_in_context": cfg.max_corridors_in_context,
            "max_main_sentences_per_corridor": cfg.max_main_sentences_per_corridor,
            "max_support_per_corridor": cfg.max_support_per_corridor,
            "max_total_sentences": cfg.max_total_sentences,
            "delivery_mode": cfg.delivery_mode,
            "max_chunk_packages": cfg.max_chunk_packages,
            "max_excerpt_sentences_per_package": cfg.max_excerpt_sentences_per_package,
            "chunk_grounding_enabled": cfg.chunk_grounding_enabled,
            "chunk_grounding_mode": cfg.chunk_grounding_mode,
            "chunk_excerpt_max_per_corridor": cfg.chunk_excerpt_max_per_corridor,
            "chunk_excerpt_window_sentences_before": cfg.chunk_excerpt_window_sentences_before,
            "chunk_excerpt_window_sentences_after": cfg.chunk_excerpt_window_sentences_after,
            "chunk_excerpt_max_total_sentences": cfg.chunk_excerpt_max_total_sentences,
            "chunk_excerpt_dedup_enabled": cfg.chunk_excerpt_dedup_enabled,
            "chunk_grounding_top_corridor_chunks": cfg.chunk_grounding_top_corridor_chunks,
            "chunk_grounding_top_k_packages": cfg.chunk_grounding_top_k_packages,
            "package_score_answer_weight": cfg.package_score_answer_weight,
            "package_score_bridge_weight": cfg.package_score_bridge_weight,
            "package_score_support_weight": cfg.package_score_support_weight,
            "package_score_chunk_grounding_weight": cfg.package_score_chunk_grounding_weight,
            "package_score_redundancy_weight": cfg.package_score_redundancy_weight,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "gamma_main": cfg.gamma_main,
            "delta_support": cfg.delta_support,
            "eta_connector": cfg.eta_connector,
            "zeta_query": cfg.zeta_query,
            "xi_locality": cfg.xi_locality,
            "lambda_redundancy": cfg.lambda_redundancy,
            "top_corridors": cfg.top_corridors,
            "max_sentences": cfg.max_sentences,
            "reserve_top_corridor": cfg.reserve_top_corridor,
            "order_strategy": cfg.order_strategy,
        },
        "profile_config": {
            "profile_stages": bool(getattr(cfg, "profile_stages", False)),
            "profile_output": str(getattr(cfg, "profile_output", "") or ""),
            "profile_limit": int(getattr(cfg, "profile_limit", 0) or 0),
            "profile_query_indices": str(getattr(cfg, "profile_query_indices", "") or ""),
            "retrieval_only": bool(getattr(cfg, "retrieval_only", False)),
            "selected_query_count": int(len(indexed_samples)),
        },
    }

    recall_at_k_summary = {}
    for k in DEFAULT_RECALL_KS:
        key = str(int(k))
        vals = [float((r["metrics"].get("recall_at_k", {}) or {}).get(key, 0.0)) for r in rows]
        agg = mean_or_zero(vals)
        recall_at_k_summary[key] = agg
        summary[f"supporting_fact_recall_at_{key}"] = agg
    summary["supporting_fact_recall_at_k"] = recall_at_k_summary

    if run_qa_enabled:
        summary["generation_latency_ms"] = mean_or_zero([r["efficiency"]["generation_latency_ms"] for r in rows])
        summary["generation_ms"] = mean_or_zero([r["efficiency"].get("generation_ms", 0.0) for r in rows])
        finish_reason_counts = {}
        for row in rows:
            reason = str((row.get("generation_diagnostics", {}) or {}).get("finish_reason", "") or "")
            if not reason:
                reason = "unknown"
            finish_reason_counts[reason] = int(finish_reason_counts.get(reason, 0) + 1)
        summary["finish_reason_counts"] = finish_reason_counts

    index_diags = []
    for row in rows:
        diag = _extract_global_index_diag_from_retrieval(row.get("retrieval", {}))
        if diag:
            index_diags.append(diag)
    summary["index_timed_samples"] = int(len(index_diags))
    summary["index_total_ms"] = mean_or_zero([float(d.get("index_total_ms", 0.0)) for d in index_diags])
    summary["index_build_ms"] = mean_or_zero([float(d.get("index_build_ms", 0.0)) for d in index_diags])
    summary["index_load_graph_ms"] = mean_or_zero([float(d.get("index_load_graph_ms", 0.0)) for d in index_diags])
    summary["index_write_ms"] = mean_or_zero([float(d.get("index_write_ms", 0.0)) for d in index_diags])
    summary["index_cache_hit_rate"] = mean_or_zero(
        [1.0 if bool(d.get("cache_hit", False)) else 0.0 for d in index_diags]
    )
    summary["output_dir"] = str(out_dir.resolve())

    if cfg.measure_gpu_peak:
        summary["gpu_peak_mb"] = mean_or_zero([r["efficiency"].get("gpu_peak_mb", 0.0) for r in rows])
    if cfg.measure_cpu_ram:
        summary["cpu_ram_peak_mb"] = mean_or_zero([r["efficiency"].get("cpu_ram_peak_mb", 0.0) for r in rows])

    if sf_debug_enabled and sf_debug_rows:
        raw_debug_path = str(getattr(cfg, "sf_debug_output", "") or "").strip()
        debug_path = Path(raw_debug_path) if raw_debug_path else (out_dir / "supporting_fact_debug.jsonl")
        write_jsonl(debug_path, sf_debug_rows)
        summary["supporting_fact_debug_path"] = str(debug_path.resolve())
        summary["supporting_fact_debug_samples"] = int(len(sf_debug_rows))

    write_json(summary_path, summary)

    if bool(getattr(cfg, "profile_stages", False)):
        raw_profile_path = str(getattr(cfg, "profile_output", "") or "").strip()
        if not raw_profile_path:
            raw_profile_path = str((out_dir / f"{cfg.dataset}_{cfg.method}_query_profile.jsonl").resolve())
        profile_path = Path(raw_profile_path)
        if profile_path.suffix.lower() != ".jsonl":
            profile_path = profile_path.with_suffix(".jsonl")
        profile_summary_path = profile_path.with_name(f"{profile_path.stem}_summary.json")
        profile_md_path = profile_path.with_name(f"{profile_path.stem}_summary.md")

        write_jsonl(profile_path, profile_rows)
        profile_summary = _build_profile_summary(
            profile_rows=profile_rows,
            retrieval_only=bool(getattr(cfg, "retrieval_only", False)),
        )
        profile_summary["dataset"] = str(cfg.dataset)
        profile_summary["method"] = str(cfg.method)
        profile_summary["generator"] = str(cfg.generator)
        profile_summary["run_timestamp"] = run_stamp
        profile_summary["run_timestamp_utc"] = run_iso
        profile_summary["profile_output_path"] = str(profile_path.resolve())
        profile_summary["profile_summary_path"] = str(profile_summary_path.resolve())
        profile_summary["profile_summary_md_path"] = str(profile_md_path.resolve())
        profile_summary["profile_query_indices"] = [int(r.get("query_index", -1)) for r in profile_rows]
        write_json(profile_summary_path, profile_summary)
        profile_md_path.parent.mkdir(parents=True, exist_ok=True)
        profile_md_path.write_text(_profile_summary_markdown(profile_summary), encoding="utf-8")

        summary["profile_output_path"] = str(profile_path.resolve())
        summary["profile_summary_path"] = str(profile_summary_path.resolve())
        summary["profile_summary_md_path"] = str(profile_md_path.resolve())
        summary["profile_primary_bottleneck_stage"] = str(profile_summary.get("primary_bottleneck_stage", ""))
        summary["profile_next_priority_recommendation"] = profile_summary.get("next_priority_recommendation", {})

    logs_dir = out_dir / "logs"
    cfg_log = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "task": "rag",
        "config": cfg_values,
    }
    result_log = {
        "run_timestamp": run_stamp,
        "run_timestamp_utc": run_iso,
        "task": "rag",
        "summary": summary,
    }
    write_json(logs_dir / ("config_%s.json" % run_stamp), cfg_log)
    write_json(logs_dir / ("result_%s.json" % run_stamp), result_log)
    append_jsonl(logs_dir / "config_history.jsonl", cfg_log)
    append_jsonl(logs_dir / "result_history.jsonl", result_log)

    return rows, summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base_config = load_yaml(args.config) if args.config else {}
    merged = apply_cli_overrides(base_config, args)
    cfg = dataclass_from_dict(RagConfig, merged)

    _, summary = execute_rag_experiment(cfg)

    print("RAG run complete")
    print(f"Output dir: {summary.get('output_dir', '')}")
    print(
        markdown_table(
            [
                "method",
                "generator",
                "samples",
                "sf_recall",
                "Recall@1",
                "Recall@5",
                "Recall@20",
                "rendered_sf_recall",
                "EM",
                "F1",
                "retrieval_ms",
                "generation_ms",
                "total_ms",
                "index_total_ms",
                "trunc_corr_avg",
                "trunc_sent_avg",
                "run_timestamp",
            ],
            [
                [
                    summary["method"],
                    summary.get("generator_display", summary["generator"]),
                    int(summary["n_samples"]),
                    "%.4f" % summary["supporting_fact_recall"],
                    "%.4f" % summary.get("supporting_fact_recall_at_1", 0.0),
                    "%.4f" % summary.get("supporting_fact_recall_at_5", 0.0),
                    "%.4f" % summary.get("supporting_fact_recall_at_20", 0.0),
                    "%.4f" % summary.get("rendered_supporting_fact_recall", 0.0),
                    "%.4f" % summary["em"],
                    "%.4f" % summary["f1"],
                    "%.2f" % summary["retrieval_latency_ms"],
                    "%.2f" % summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)),
                    "%.2f" % summary["total_latency_ms"],
                    "%.2f" % summary.get("index_total_ms", 0.0),
                    "%.2f" % summary.get("truncated_corridors_avg", 0.0),
                    "%.2f" % summary.get("truncated_sentences_avg", 0.0),
                    summary["run_timestamp"],
                ]
            ],
        )
    )


if __name__ == "__main__":
    main()
