from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


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


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metric(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = dict(row.get("metrics", {}) or {})
    if key in metrics:
        return _safe_float(metrics.get(key), default)
    return _safe_float(row.get(key), default)


def _diagnostics(row: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = dict(row.get("retrieval", {}) or {})
    return dict(retrieval.get("diagnostics", {}) or {})


def _phase8_chunk_diag(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(_diagnostics(row).get("phase8_chunk_diagnostics", {}) or {})


def _first_dict(diag: Dict[str, Any], key: str) -> Dict[str, Any]:
    val = diag.get(key, {})
    return dict(val if isinstance(val, dict) else {})


def _stage_ms(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(_diagnostics(row).get("stage_ms", {}) or {})


def _latency_ms(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(_diagnostics(row).get("latency_breakdown_ms", {}) or {})


def _source_distribution(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        for tag in list(item.get("source_tags", []) or []):
            key = str(tag or "")
            if not key:
                continue
            out[key] = int(out.get(key, 0)) + 1
    return out


def append_phase8_chunk_runtime_trace(
    output_dir: str,
    row: Dict[str, Any],
    generation_ms: float,
    total_query_ms: float,
) -> None:
    out_dir = Path(str(output_dir or "")).resolve()
    diagnostics = _diagnostics(row)
    enabled = bool(diagnostics.get("phase8_chunk_medoid_enabled", False))
    phase8 = _phase8_chunk_diag(row)
    stage = _stage_ms(row)
    latency = _latency_ms(row)
    qid = str(row.get("sample_id", row.get("qid", "")) or "")
    question = str(row.get("question", "") or "")

    universe_diag = _first_dict(phase8, "chunk_universe")
    sampling_diag = _first_dict(phase8, "chunk_medoid_sampling")
    seed_diag = _first_dict(phase8, "chunk_seed_selection")
    refine_diag = _first_dict(phase8, "chunk_bridge_refinement")
    evidence_diag = _first_dict(phase8, "chunk_medoid_evidence_proposal")
    candidate_chain = dict(diagnostics.get("candidate_chain_feasibility", {}) or {})
    selected_atoms = list(diagnostics.get("phase7_selected_atoms", []) or [])
    source_dist = _source_distribution(selected_atoms)
    selected_chunk_count = int(source_dist.get("phase8_chunk_medoid", 0))
    selected_count = max(1, len(selected_atoms))

    query_trace = {
        "query_id": qid,
        "question": question,
        "phase8_chunk_medoid_enabled": bool(enabled),
        "chunk_universe_size": int(universe_diag.get("num_chunks", 0) or 0),
        "num_samples": int(sampling_diag.get("num_samples", 0) or 0),
        "sample_size": int(sampling_diag.get("sample_size", 0) or 0),
        "k": int(sampling_diag.get("k", 0) or 0),
        "best_seed_score": _safe_float(seed_diag.get("best_seed_score", sampling_diag.get("best_seed_score", 0.0)), 0.0),
        "chunk_seed_relevance": _safe_float(seed_diag.get("best_seed_relevance", sampling_diag.get("best_seed_relevance", 0.0)), 0.0),
        "chunk_seed_coverage": _safe_float(seed_diag.get("best_seed_coverage", sampling_diag.get("best_seed_coverage", 0.0)), 0.0),
        "chunk_seed_diversity": _safe_float(seed_diag.get("best_seed_diversity", sampling_diag.get("best_seed_diversity", 0.0)), 0.0),
        "chunk_seed_token_cost": _safe_float(seed_diag.get("best_seed_token_cost", sampling_diag.get("best_seed_token_cost", 0.0)), 0.0),
        "num_changed_chunk_seeds": int(refine_diag.get("num_changed_seeds", 0) or 0),
        "num_bridge_entities": _safe_float(refine_diag.get("avg_bridge_entities_per_seed", 0.0), 0.0),
        "num_final_evidence_candidates": int(evidence_diag.get("num_final_evidence_candidates", 0) or 0),
        "normalized_candidate_gold_recall": _safe_float(
            diagnostics.get("normalized_candidate_gold_recall", diagnostics.get("candidate_gold_recall", candidate_chain.get("candidate_gold_recall", 0.0))),
            0.0,
        ),
        "normalized_selected_gold_recall": _safe_float(diagnostics.get("normalized_selected_gold_recall", diagnostics.get("selected_sf_recall_eval_only", 0.0)), 0.0),
        "normalized_rendered_gold_recall": _safe_float(diagnostics.get("normalized_rendered_gold_recall", diagnostics.get("rendered_sf_recall_eval_only", 0.0)), 0.0),
        "candidate_oracle_F1": _safe_float(evidence_diag.get("candidate_oracle_F1_eval_only", 0.0), 0.0),
        "chain_unit_oracle_feasible": _safe_float(
            diagnostics.get("chain_unit_oracle_feasible", candidate_chain.get("chain_unit_oracle_feasible", 0.0)),
            0.0,
        ),
        "chunk_seed_gold_hit_rate_eval_only": 1.0 if bool(seed_diag.get("seed_gold_hit_eval_only", False)) else 0.0,
        "chunk_candidate_gold_recall_eval_only": _safe_float(evidence_diag.get("candidate_gold_recall_eval_only", 0.0), 0.0),
        "selected_chunk_medoid_source_rate": float(selected_chunk_count) / float(selected_count),
        "em": _metric(row, "exact_match", _metric(row, "em", 0.0)),
        "f1": _metric(row, "f1", 0.0),
        "sf_recall": _metric(row, "supporting_fact_recall", 0.0),
        "sf_precision": _metric(row, "supporting_fact_precision", 0.0),
        "sf_f1": _metric(row, "supporting_fact_f1", 0.0),
        "avg_context_tokens": _safe_float(dict(row.get("generation_diagnostics", {}) or {}).get("prompt_tokens", 0.0), 0.0),
        "avg_selected_atoms": _safe_float(diagnostics.get("num_selected_atoms", 0.0), 0.0),
    }
    _append_jsonl(out_dir / "phase8_chunk_query_trace.jsonl", query_trace)

    timing = {
        "query_id": qid,
        "chunk_universe_ms": _safe_float(universe_diag.get("chunk_universe_ms", 0.0), 0.0),
        "chunk_sampling_ms": _safe_float(sampling_diag.get("sampling_ms", 0.0), 0.0),
        "chunk_kmedoids_ms": _safe_float(sampling_diag.get("kmedoids_ms", 0.0), 0.0),
        "chunk_seed_selection_ms": _safe_float(seed_diag.get("chunk_seed_selection_ms", 0.0), 0.0),
        "bridge_refinement_ms": _safe_float(refine_diag.get("bridge_refinement_ms", 0.0), 0.0),
        "evidence_proposal_ms": _safe_float(evidence_diag.get("evidence_proposal_ms", 0.0), 0.0),
        "feature_construction_ms": _safe_float(stage.get("feature_extraction", latency.get("feature_extraction_ms", 0.0)), 0.0),
        "selection_ms": _safe_float(stage.get("marginal_selection", latency.get("marginal_selection_ms", 0.0)), 0.0),
        "rendering_ms": _safe_float(stage.get("rendering", latency.get("render_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(generation_ms, 0.0),
        "retrieval_ms": _safe_float(stage.get("total_retrieval", latency.get("retrieval_total_ms", 0.0)), 0.0),
        "total_ms": _safe_float(total_query_ms, 0.0),
    }
    _append_jsonl(out_dir / "phase8_chunk_stage_timing.jsonl", timing)

    _append_jsonl(
        out_dir / "phase8_chunk_seed_trace.jsonl",
        {
            "query_id": qid,
            "phase8_chunk_medoid_enabled": bool(enabled),
            "chunk_seed_set_candidates": list(phase8.get("chunk_seed_set_candidates", []) or []),
            "chunk_seed_selection": seed_diag,
            "chunk_bridge_refinement": refine_diag,
        },
    )
    _append_jsonl(
        out_dir / "phase8_chunk_evidence_trace.jsonl",
        {
            "query_id": qid,
            "phase8_chunk_medoid_enabled": bool(enabled),
            "chunk_medoid_evidence_proposal": evidence_diag,
            "evidence_source_tags": dict(phase8.get("evidence_source_tags", {}) or {}),
            "selected_chunk_medoid_source_rate": float(selected_chunk_count) / float(selected_count),
            "selected_source_tag_distribution": dict(source_dist),
        },
    )
