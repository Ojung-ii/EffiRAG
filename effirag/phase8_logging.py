from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def _phase8_diag(row: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = dict(row.get("retrieval", {}) or {})
    diagnostics = dict(retrieval.get("diagnostics", {}) or {})
    return dict(diagnostics.get("phase8_diagnostics", {}) or {})


def _stage_ms(row: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = dict(row.get("retrieval", {}) or {})
    diagnostics = dict(retrieval.get("diagnostics", {}) or {})
    return dict(diagnostics.get("stage_ms", {}) or {})


def _latency_ms(row: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = dict(row.get("retrieval", {}) or {})
    diagnostics = dict(retrieval.get("diagnostics", {}) or {})
    return dict(diagnostics.get("latency_breakdown_ms", {}) or {})


def _first_dict(diag: Dict[str, Any], key: str) -> Dict[str, Any]:
    val = diag.get(key, {})
    return dict(val if isinstance(val, dict) else {})


def append_phase8_runtime_trace(
    output_dir: str,
    row: Dict[str, Any],
    generation_ms: float,
    total_query_ms: float,
) -> None:
    out_dir = Path(str(output_dir or "")).resolve()
    retrieval = dict(row.get("retrieval", {}) or {})
    diagnostics = dict(retrieval.get("diagnostics", {}) or {})
    enabled = bool(diagnostics.get("phase8_pamae_enabled", False))
    phase8 = _phase8_diag(row)
    stage = _stage_ms(row)
    latency = _latency_ms(row)
    qid = str(row.get("sample_id", row.get("qid", "")) or "")
    question = str(row.get("question", "") or "")

    entity_diag = _first_dict(phase8, "entity_universe")
    sampling_diag = _first_dict(phase8, "pamae_sampling")
    best_diag = _first_dict(phase8, "best_seed_selection")
    refine_diag = _first_dict(phase8, "refinement")
    evidence_diag = _first_dict(phase8, "evidence_proposal")
    seed_rows = list(phase8.get("seed_set_candidates", []) or [])
    evidence_tags = dict(phase8.get("evidence_source_tags", {}) or {})
    candidate_chain = dict(diagnostics.get("candidate_chain_feasibility", {}) or {})

    query_trace = {
        "query_id": qid,
        "question": question,
        "phase8_pamae_enabled": bool(enabled),
        "entity_universe_size": int(entity_diag.get("num_entities", 0) or 0),
        "num_samples": int(sampling_diag.get("num_samples", 0) or 0),
        "sample_size": int(sampling_diag.get("sample_size", 0) or 0),
        "k": int(sampling_diag.get("k", 0) or 0),
        "best_seed_score": _safe_float(best_diag.get("best_seed_score", 0.0), 0.0),
        "seed_mean_relevance": _safe_float(best_diag.get("best_mean_relevance", 0.0), 0.0),
        "seed_coverage": _safe_float(best_diag.get("best_coverage", 0.0), 0.0),
        "seed_diversity": _safe_float(best_diag.get("best_diversity", 0.0), 0.0),
        "num_refined_seeds": int(refine_diag.get("num_refined_seeds", 0) or 0),
        "num_changed_seeds": int(refine_diag.get("num_changed_seeds", 0) or 0),
        "phase1_partial_gold_hit": bool(
            diagnostics.get("candidate_gold_partial", candidate_chain.get("candidate_gold_partial", False))
        ),
        "phase1_full_gold_coverage": bool(
            diagnostics.get("candidate_gold_full", candidate_chain.get("candidate_gold_full", False))
        ),
        "phase1_gold_recall": _safe_float(
            diagnostics.get("candidate_gold_recall", candidate_chain.get("candidate_gold_recall", 0.0)),
            0.0,
        ),
        "candidate_gold_full": bool(
            diagnostics.get("candidate_gold_full", candidate_chain.get("candidate_gold_full", False))
        ),
        "candidate_gold_recall": _safe_float(
            diagnostics.get("candidate_gold_recall", candidate_chain.get("candidate_gold_recall", 0.0)),
            0.0,
        ),
        "candidate_oracle_F1": _safe_float(evidence_diag.get("candidate_oracle_F1_eval_only", 0.0), 0.0),
        "chain_unit_oracle_feasible": _safe_float(
            diagnostics.get(
                "chain_unit_oracle_feasible",
                candidate_chain.get("chain_unit_oracle_feasible", evidence_diag.get("chain_unit_oracle_feasible_eval_only", 0.0)),
            ),
            0.0,
        ),
        "seed_gold_hit_rate_eval_only": 1.0 if bool(best_diag.get("seed_gold_hit_eval_only", False)) else 0.0,
        "seed_evidence_gold_hit_rate_eval_only": 1.0
        if bool(best_diag.get("seed_evidence_gold_hit_eval_only", False))
        else 0.0,
        "em": _metric(row, "exact_match", _metric(row, "em", 0.0)),
        "f1": _metric(row, "f1", 0.0),
        "sf_recall": _metric(row, "supporting_fact_recall", 0.0),
        "sf_precision": _metric(row, "supporting_fact_precision", 0.0),
        "sf_f1": _metric(row, "supporting_fact_f1", 0.0),
        "avg_context_tokens": _safe_float(dict(row.get("generation_diagnostics", {}) or {}).get("prompt_tokens", 0.0), 0.0),
        "avg_selected_atoms": _safe_float(diagnostics.get("num_selected_atoms", 0.0), 0.0),
    }
    _append_jsonl(out_dir / "phase8_query_trace.jsonl", query_trace)

    timing = {
        "query_id": qid,
        "entity_universe_ms": _safe_float(entity_diag.get("universe_build_ms", 0.0), 0.0),
        "sampling_ms": _safe_float(sampling_diag.get("sampling_ms", 0.0), 0.0),
        "kmedoids_ms": _safe_float(sampling_diag.get("kmedoids_ms", 0.0), 0.0),
        "seed_selection_ms": _safe_float(best_diag.get("seed_selection_ms", 0.0), 0.0),
        "refinement_ms": _safe_float(refine_diag.get("refinement_ms", 0.0), 0.0),
        "evidence_proposal_ms": _safe_float(evidence_diag.get("evidence_proposal_ms", 0.0), 0.0),
        "feature_construction_ms": _safe_float(stage.get("feature_extraction", latency.get("feature_extraction_ms", 0.0)), 0.0),
        "selection_ms": _safe_float(stage.get("marginal_selection", latency.get("marginal_selection_ms", 0.0)), 0.0),
        "rendering_ms": _safe_float(stage.get("rendering", latency.get("render_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(generation_ms, 0.0),
        "total_retrieval_ms": _safe_float(stage.get("total_retrieval", latency.get("retrieval_total_ms", 0.0)), 0.0),
        "total_ms": _safe_float(total_query_ms, 0.0),
    }
    _append_jsonl(out_dir / "phase8_stage_timing.jsonl", timing)

    seed_trace = {
        "query_id": qid,
        "phase8_pamae_enabled": bool(enabled),
        "seed_set_candidates": seed_rows,
        "best_seed_selection": best_diag,
        "refinement": refine_diag,
    }
    _append_jsonl(out_dir / "phase8_seed_trace.jsonl", seed_trace)

    evidence_trace = {
        "query_id": qid,
        "phase8_pamae_enabled": bool(enabled),
        "evidence_proposal": evidence_diag,
        "evidence_source_tags": evidence_tags,
    }
    _append_jsonl(out_dir / "phase8_evidence_trace.jsonl", evidence_trace)
