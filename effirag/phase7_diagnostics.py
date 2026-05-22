from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .metrics import supporting_fact_match_details
from .utils import content_tokens


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


def _norm_id_list(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in list(values or []):
        sid = str(value or "").strip()
        if sid:
            out.append(sid)
    return out


def _norm_text_list(values: Iterable[Any]) -> List[str]:
    return [str(value or "") for value in list(values or [])]


def _unit_type_from_id(unit_id: str) -> str:
    sid = str(unit_id or "").strip()
    if not sid:
        return "unknown"
    if sid.startswith("chunk::"):
        return "chunk"
    if "::" in sid:
        return "sentence"
    return "unknown"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _match_hit(sample: Any, unit_ids: List[str], unit_texts: List[str], graph_mode: str) -> Dict[str, Any]:
    if sample is None:
        return {"gold_hit": False, "matched_gold_total": 0, "gold_total": 0}
    match = supporting_fact_match_details(
        sample=sample,
        unit_ids=list(unit_ids or []),
        unit_texts=list(unit_texts or []),
        graph_mode=str(graph_mode or "current_entity_graph"),
    )
    matched_gold_total = int(match.get("matched_gold_total", 0) or 0)
    return {
        "gold_hit": bool(matched_gold_total > 0),
        "matched_gold_total": int(matched_gold_total),
        "gold_total": int(match.get("gold_total", 0) or 0),
        "match": match,
    }


def _id_type_map(ids: List[str], node_types: List[Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for idx, sid in enumerate(list(ids or [])):
        if not sid:
            continue
        if idx < len(node_types):
            ntype = str(node_types[idx] or "").strip().lower()
            if ntype:
                out[sid] = ntype
    return out


def _id_text_map(ids: List[str], texts: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for idx, sid in enumerate(list(ids or [])):
        if not sid:
            continue
        txt = ""
        if idx < len(texts):
            txt = str(texts[idx] or "")
        if sid not in out:
            out[sid] = txt
    return out


def _rendered_unit_types(rendered_ids: List[str], selected_type_map: Mapping[str, str]) -> List[str]:
    out: List[str] = []
    for sid in list(rendered_ids or []):
        out.append(str(selected_type_map.get(sid, _unit_type_from_id(sid))))
    return out


def build_phase7_diagnostic_record(
    *,
    sample: Any,
    row: Mapping[str, Any],
    dump_text: bool,
    dump_context: bool,
    dump_scores: bool,
) -> Dict[str, Any]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    retrieval_diag = dict((retrieval.get("diagnostics", {}) or {}))
    rendering = dict((row.get("rendered", {}) or {}))
    efficiency = dict((row.get("efficiency", {}) or {}))
    metrics = dict((row.get("metrics", {}) or {}))
    gen_diag = dict((row.get("generation_diagnostics", {}) or {}))
    cfg_snapshot = dict((row.get("config_snapshot", {}) or {}))

    qid = str(row.get("sample_id", getattr(sample, "qid", "")) or "")
    question = str(row.get("question", getattr(sample, "question", "")) or "")
    prediction = str(row.get("prediction", "") or "")
    gold_answer = str(row.get("answer", getattr(sample, "answer", "")) or "")
    dataset = str(row.get("dataset", cfg_snapshot.get("dataset", "")) or "")
    prompt_variant = str(gen_diag.get("prompt_variant", cfg_snapshot.get("prompt_variant", "default")) or "default")

    graph_mode = str(retrieval_diag.get("graph_mode", cfg_snapshot.get("graph_mode", "current_entity_graph")) or "current_entity_graph")
    index_chunk_unit = str(cfg_snapshot.get("index_chunk_unit", retrieval_diag.get("index_chunk_unit", "unknown")) or "unknown")

    phase1_seed_ids = _norm_id_list(retrieval_diag.get("phase1_seed_ids", []) or retrieval.get("seeds", []))
    phase1_seed_node_types = _norm_text_list(retrieval_diag.get("phase1_seed_node_types", []))
    phase1_seed_scores_raw = list(retrieval_diag.get("phase1_seed_scores", []) or [])
    phase1_seed_texts = _norm_text_list(retrieval_diag.get("phase1_seed_texts", []))

    phase2_candidate_ids = _norm_id_list(
        retrieval_diag.get("phase2_candidate_ids", []) or retrieval.get("candidate_sentence_ids", [])
    )
    phase2_candidate_node_types = _norm_text_list(retrieval_diag.get("phase2_candidate_node_types", []))
    phase2_candidate_scores_raw = list(retrieval_diag.get("phase2_candidate_scores", []) or [])
    phase2_candidate_texts = _norm_text_list(
        retrieval_diag.get("phase2_candidate_texts", []) or retrieval.get("candidate_sentences", [])
    )

    selected_ids = _norm_id_list(
        retrieval_diag.get("selected_evidence_ids", []) or retrieval.get("selected_sentence_ids", [])
    )
    selected_node_types = _norm_text_list(retrieval_diag.get("selected_evidence_node_types", []))
    selected_scores_raw = list(retrieval_diag.get("selected_evidence_scores", []) or [])
    selected_texts = _norm_text_list(
        retrieval_diag.get("selected_evidence_texts", []) or retrieval.get("selected_sentences", [])
    )

    rendered_ids = _norm_id_list(rendering.get("sentence_ids", []) or row.get("rendered_sentence_ids", []))
    rendered_texts = _norm_text_list(rendering.get("sentences", []))
    rendered_context = str(rendering.get("text", "") or "")
    context_tokens = int(len(content_tokens(rendered_context)))

    selected_type_map = _id_type_map(selected_ids, selected_node_types)
    rendered_node_types = _rendered_unit_types(rendered_ids, selected_type_map)

    selected_text_map = _id_text_map(selected_ids, selected_texts)
    rendered_text_map = _id_text_map(rendered_ids, rendered_texts)

    if not phase1_seed_node_types:
        phase1_seed_node_types = [_unit_type_from_id(x) for x in phase1_seed_ids]
    if not phase2_candidate_node_types:
        phase2_candidate_node_types = [_unit_type_from_id(x) for x in phase2_candidate_ids]
    if not selected_node_types:
        selected_node_types = [_unit_type_from_id(x) for x in selected_ids]

    if not phase1_seed_texts:
        phase1_seed_texts = ["" for _ in phase1_seed_ids]
    if not phase2_candidate_texts:
        phase2_candidate_texts = [phase2_candidate_texts[idx] if idx < len(phase2_candidate_texts) else "" for idx in range(len(phase2_candidate_ids))]
    if not selected_texts:
        selected_texts = [selected_text_map.get(sid, "") for sid in selected_ids]
    if not rendered_texts:
        rendered_texts = [rendered_text_map.get(sid, "") for sid in rendered_ids]

    if len(phase1_seed_scores_raw) < len(phase1_seed_ids):
        phase1_seed_scores_raw = list(phase1_seed_scores_raw) + [None] * (len(phase1_seed_ids) - len(phase1_seed_scores_raw))
    if len(phase2_candidate_scores_raw) < len(phase2_candidate_ids):
        phase2_candidate_scores_raw = list(phase2_candidate_scores_raw) + [None] * (len(phase2_candidate_ids) - len(phase2_candidate_scores_raw))
    if len(selected_scores_raw) < len(selected_ids):
        selected_scores_raw = list(selected_scores_raw) + [None] * (len(selected_ids) - len(selected_scores_raw))

    phase1_match = _match_hit(sample, phase1_seed_ids, phase1_seed_texts, graph_mode=graph_mode)
    phase2_match = _match_hit(sample, phase2_candidate_ids, phase2_candidate_texts, graph_mode=graph_mode)
    selected_match = _match_hit(sample, selected_ids, selected_texts, graph_mode=graph_mode)
    rendered_match = _match_hit(sample, rendered_ids, rendered_texts, graph_mode=graph_mode)

    selected_to_rendered_match = bool(selected_ids == rendered_ids)
    missing_render_text = bool(any((not str(rendered_text_map.get(sid, "") or "").strip()) for sid in rendered_ids))
    empty_context = bool(not str(rendered_context or "").strip())

    mismatch_reasons: List[str] = []
    for sid in rendered_ids:
        if sid not in set(selected_ids):
            mismatch_reasons.append("rendered_evidence_id_not_in_selected")
            break
    if any(not str(selected_text_map.get(sid, "") or "").strip() for sid in selected_ids):
        mismatch_reasons.append("selected_evidence_text_missing")
    sel_type_map = _id_type_map(selected_ids, selected_node_types)
    ren_type_map = _id_type_map(rendered_ids, rendered_node_types)
    for sid in set(selected_ids).intersection(set(rendered_ids)):
        st = str(sel_type_map.get(sid, _unit_type_from_id(sid)))
        rt = str(ren_type_map.get(sid, _unit_type_from_id(sid)))
        if st and rt and st != rt:
            mismatch_reasons.append("selected_rendered_node_type_mismatch")
            break
    if (
        str(graph_mode) == "entity_chunk_graph"
        and all(str(x) == "chunk" for x in list(selected_node_types or []))
        and "sentence_or_chunk_title_fallback" in str(retrieval_diag.get("unit_consistency", {}).get("gold_matching_unit", ""))
    ):
        mismatch_reasons.append("gold_matching_unit_sentence_vs_chunk_selection")
    if not selected_to_rendered_match:
        mismatch_reasons.append("selected_to_rendered_order_or_count_mismatch")

    gold_supporting_facts: List[Dict[str, Any]] = []
    for title, sent_idx in list(getattr(sample, "supporting_facts", []) or []):
        t = str(title or "").strip()
        idx = _safe_int(sent_idx, 0)
        unit_id = f"{t}::{idx}" if t else ""
        gold_supporting_facts.append(
            {
                "title": t,
                "sent_idx": int(idx),
                "unit_id": unit_id,
            }
        )

    rendered_context_out: Optional[str] = str(rendered_context) if bool(dump_context) else None
    phase1_texts_out = list(phase1_seed_texts) if bool(dump_text) else None
    phase2_texts_out = list(phase2_candidate_texts) if bool(dump_text) else None
    selected_texts_out = list(selected_texts) if bool(dump_text) else None
    rendered_texts_out = list(rendered_texts) if bool(dump_text) else None

    phase1_scores_out = [None if s is None else _safe_float(s, 0.0) for s in phase1_seed_scores_raw] if bool(dump_scores) else None
    phase2_scores_out = [None if s is None else _safe_float(s, 0.0) for s in phase2_candidate_scores_raw] if bool(dump_scores) else None
    selected_scores_out = [None if s is None else _safe_float(s, 0.0) for s in selected_scores_raw] if bool(dump_scores) else None

    context_hash = _sha256_text(rendered_context)
    prompt_hash = _sha256_text(f"{question}\n\n{rendered_context}\n\nvariant={prompt_variant}")

    corridor_diag = dict(retrieval_diag.get("phase7_corridor_diagnostics", {}) or {})
    corridor_anchor_nodes = list(retrieval_diag.get("corridor_anchor_nodes", []) or [])
    corridor_seed_nodes = list(retrieval_diag.get("corridor_seed_nodes", []) or [])
    selected_feature_breakdown = list(retrieval_diag.get("selected_evidence_feature_breakdown", []) or [])
    corridor_gold_diag_eval_only = dict(retrieval_diag.get("corridor_gold_diagnostics_eval_only", {}) or {})

    return {
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "gold_answer": gold_answer,
        "prediction": prediction,
        "gold_supporting_facts": gold_supporting_facts,
        "graph_mode": graph_mode,
        "index_chunk_unit": index_chunk_unit,
        "phase7_variant": str(retrieval_diag.get("phase7_variant", cfg_snapshot.get("phase7_variant", "full")) or "full"),
        "phase1_seed_ids": phase1_seed_ids,
        "phase1_seed_node_types": phase1_seed_node_types,
        "phase1_seed_scores": phase1_scores_out,
        "phase1_seed_texts": phase1_texts_out,
        "phase2_candidate_ids": phase2_candidate_ids,
        "phase2_candidate_node_types": phase2_candidate_node_types,
        "phase2_candidate_scores": phase2_scores_out,
        "phase2_candidate_texts": phase2_texts_out,
        "selected_evidence_ids": selected_ids,
        "selected_evidence_node_types": selected_node_types,
        "selected_evidence_scores": selected_scores_out,
        "selected_evidence_texts": selected_texts_out,
        "rendered_evidence_ids": rendered_ids,
        "rendered_evidence_node_types": rendered_node_types,
        "rendered_evidence_texts": rendered_texts_out,
        "rendered_context": rendered_context_out,
        "context_tokens": int(context_tokens),
        "gold_hit_phase1": bool(phase1_match["gold_hit"]),
        "gold_hit_phase2": bool(phase2_match["gold_hit"]),
        "gold_hit_selected": bool(selected_match["gold_hit"]),
        "gold_hit_rendered": bool(rendered_match["gold_hit"]),
        "selected_to_rendered_match": bool(selected_to_rendered_match),
        "missing_render_text": bool(missing_render_text),
        "empty_context": bool(empty_context),
        "unit_mismatch_reasons": list(dict.fromkeys(mismatch_reasons)),
        "phase7_corridor_diagnostics": corridor_diag,
        "corridor_anchor_nodes": corridor_anchor_nodes,
        "corridor_seed_nodes": corridor_seed_nodes,
        "selected_evidence_feature_breakdown": selected_feature_breakdown,
        "corridor_gold_diagnostics_eval_only": corridor_gold_diag_eval_only,
        "corridor_gold_hit": bool(
            int(corridor_gold_diag_eval_only.get("num_gold_candidates_on_corridor", 0) or 0) > 0
        ),
        "context_hash": context_hash,
        "prompt_hash": prompt_hash,
        "retrieval_ms": _safe_float(efficiency.get("retrieval_latency_ms", 0.0), 0.0),
        "generation_ms": _safe_float(efficiency.get("generation_ms", 0.0), 0.0),
        "total_ms": _safe_float(efficiency.get("total_latency_ms", 0.0), 0.0),
        "query_em": _safe_float(metrics.get("em", 0.0), 0.0),
        "query_f1": _safe_float(metrics.get("f1", 0.0), 0.0),
    }
