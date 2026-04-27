from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .eval.metrics_hipporag2_parity import normalize_answer_parity
from .metrics import supporting_fact_match_details
from .utils import content_tokens, markdown_table, mean_or_zero, safe_div

# Retrieval ranking metrics (research-standard IR metrics)
RECALL_KS = (1, 5, 10, 20)
HIT_KS = (1, 5, 10, 20)
MRR_KS = (10, 20)
NDCG_KS = (5, 10, 20)
ATLEASTONE_KS = (5, 10, 20)
ALLSUPPORT_KS = (5, 10, 20)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip01(value: Any) -> float:
    x = _safe_float(value, 0.0)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_text(text: Any) -> str:
    return str(normalize_answer_parity(str(text or "")) or "").strip()


def _token_set(text: Any) -> set:
    norm = _normalize_text(text)
    if not norm:
        return set()
    return set(tok for tok in norm.split() if tok)


def _token_overlap_ratio(prediction: str, evidence_text: str) -> float:
    # Diagnostic-only lexical overlap proxy between prediction and answer-bearing evidence.
    p = _token_set(prediction)
    if not p:
        return 0.0
    e = _token_set(evidence_text)
    if not e:
        return 0.0
    return float(safe_div(len(p & e), len(p)))


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    return []


def _collect_answer_surface_aliases(sample: Any) -> List[str]:
    out: List[str] = []
    seen = set()

    answer = str(getattr(sample, "answer", "") or "").strip()
    if answer:
        seen.add(answer.lower())
        out.append(answer)

    metadata = dict(getattr(sample, "metadata", {}) or {})
    for key in ("possible_answers", "answers", "answer_aliases", "o_aliases", "aliases"):
        for val in _as_text_list(metadata.get(key)):
            low = val.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(val)
    return out


def _answer_surface_stats(rendered_text: str, aliases: Sequence[str]) -> Dict[str, Any]:
    norm_text = _normalize_text(rendered_text)
    tokens = [tok for tok in norm_text.split() if tok]
    if not tokens:
        return {
            "present": False,
            "token_count": 0,
            "position": -1.0,
        }

    matched_positions = set()
    for alias in list(aliases or []):
        norm_alias = _normalize_text(alias)
        if not norm_alias:
            continue
        alias_tokens = [tok for tok in norm_alias.split() if tok]
        if not alias_tokens:
            continue
        n = len(alias_tokens)
        if n > len(tokens):
            continue
        for idx in range(0, len(tokens) - n + 1):
            if tokens[idx : idx + n] == alias_tokens:
                for pos in range(idx, idx + n):
                    matched_positions.add(int(pos))

    if not matched_positions:
        return {
            "present": False,
            "token_count": 0,
            "position": -1.0,
        }

    first_pos = min(matched_positions)
    return {
        "present": True,
        "token_count": int(len(matched_positions)),
        "position": float(safe_div(first_pos, max(1, len(tokens)))),
    }


def _dcg_binary(relevance: Sequence[float], k: int) -> float:
    if k <= 0:
        return 0.0
    score = 0.0
    for idx, rel in enumerate(relevance[:k], start=1):
        if rel <= 0.0:
            continue
        score += float(rel) / math.log2(idx + 1.0)
    return float(score)


def _ndcg_binary(relevance: Sequence[float], k: int) -> float:
    if k <= 0:
        return 0.0
    dcg = _dcg_binary(relevance, k)
    num_rel = int(sum(1 for x in relevance if _safe_float(x, 0.0) > 0.0))
    if num_rel <= 0:
        return 0.0
    ideal = [1.0] * min(num_rel, int(k))
    idcg = _dcg_binary(ideal, int(k))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def _mrr_at_k(relevance: Sequence[float], k: int) -> float:
    if k <= 0:
        return 0.0
    for idx, rel in enumerate(relevance[:k], start=1):
        if _safe_float(rel, 0.0) > 0.0:
            return float(1.0 / float(idx))
    return 0.0


def _average_precision(relevance: Sequence[float]) -> float:
    # Ragas-style Context Precision adapts AP over ranked contexts.
    rel_total = int(sum(1 for x in relevance if _safe_float(x, 0.0) > 0.0))
    if rel_total <= 0:
        return 0.0
    rel_seen = 0
    ap_sum = 0.0
    for idx, rel in enumerate(relevance, start=1):
        if _safe_float(rel, 0.0) <= 0.0:
            continue
        rel_seen += 1
        ap_sum += float(rel_seen) / float(idx)
    return float(ap_sum / float(rel_total))


def _query_faithfulness_proxy(
    prediction: str,
    rendered_sentences: Sequence[str],
    rendered_supporting_recall: float,
    generation_diagnostics: Optional[Mapping[str, Any]],
    em: float,
    f1: float,
) -> float:
    # Faithfulness (proxy/approximation):
    # 1) If generator already emits evidence-supported tag, trust it.
    # 2) Else use lexical support overlap with rendered context, gated by evidence presence.
    gd = dict(generation_diagnostics or {})
    if "evidence_supported_answer" in gd:
        return 1.0 if bool(gd.get("evidence_supported_answer", False)) else 0.0

    pred_norm = _normalize_text(prediction)
    if not pred_norm:
        return 0.0
    rendered_text = " ".join(str(s or "") for s in list(rendered_sentences or []))
    overlap = _token_overlap_ratio(pred_norm, rendered_text)
    supported = bool((pred_norm in _normalize_text(rendered_text)) or (overlap >= 0.35))
    evidence_present = bool(rendered_supporting_recall > 0.0)
    if supported and (evidence_present or em > 0.0 or f1 > 0.0):
        return 1.0
    return 0.0


def _collect_rank_data(match_details: Mapping[str, Any]) -> Dict[str, Any]:
    predicted_units = list(match_details.get("predicted_units", []) or [])
    gold_total = _safe_int(match_details.get("gold_total", 0), 0)
    relevance = []
    matched_by_prefix: Dict[int, set] = {}
    running = set()
    for idx, unit in enumerate(predicted_units, start=1):
        matched = list(unit.get("matched_gold_sentence_ids", []) or [])
        relevance.append(1.0 if matched else 0.0)
        for gid in matched:
            running.add(str(gid))
        matched_by_prefix[idx] = set(running)
    return {
        "predicted_units": predicted_units,
        "gold_total": int(gold_total),
        "relevance": relevance,
        "matched_by_prefix": matched_by_prefix,
    }


def _rank_metric_bundle(match_details: Mapping[str, Any]) -> Dict[str, float]:
    rank = _collect_rank_data(match_details)
    relevance = list(rank["relevance"])
    gold_total = int(rank["gold_total"])
    matched_by_prefix = dict(rank["matched_by_prefix"])

    out: Dict[str, float] = {}
    for k in RECALL_KS:
        matched = len(matched_by_prefix.get(int(k), set()))
        out[f"recall_at_{k}"] = float(safe_div(matched, gold_total))
    for k in HIT_KS:
        out[f"hit_at_{k}"] = 1.0 if any(_safe_float(x, 0.0) > 0.0 for x in relevance[: int(k)]) else 0.0
    for k in MRR_KS:
        out[f"mrr_at_{k}"] = _mrr_at_k(relevance, int(k))
    for k in NDCG_KS:
        out[f"ndcg_at_{k}"] = _ndcg_binary(relevance, int(k))
    for k in ATLEASTONE_KS:
        out[f"atleastone_at_{k}"] = 1.0 if len(matched_by_prefix.get(int(k), set())) > 0 else 0.0
    for k in ALLSUPPORT_KS:
        matched = len(matched_by_prefix.get(int(k), set()))
        out[f"allsupport_at_{k}"] = 1.0 if gold_total > 0 and matched >= gold_total else 0.0
    out["context_precision"] = _average_precision(relevance)
    return out


def compute_query_eval_metrics(
    sample: Any,
    retrieval: Any,
    rendered: Any,
    prediction: str,
    em: float,
    f1: float,
    qa_executed: bool,
    generation_diagnostics: Optional[Mapping[str, Any]] = None,
    retrieval_match: Optional[Mapping[str, Any]] = None,
    rendered_match: Optional[Mapping[str, Any]] = None,
    supporting_fact_recall: Optional[float] = None,
    supporting_fact_precision: Optional[float] = None,
    rendered_supporting_fact_recall: Optional[float] = None,
    rendered_supporting_fact_precision: Optional[float] = None,
) -> Dict[str, float]:
    # Input relevance unit: strict supporting fact (title + sentence index).
    # Retrieval relevance mapping:
    # - sentence units: exact title/sent_idx match.
    # - chunk units: gold sentence containment in chunk text.
    graph_mode = str(((getattr(retrieval, "diagnostics", {}) or {}).get("graph_mode", "current_entity_graph")))
    if retrieval_match is None:
        retrieval_match = supporting_fact_match_details(
            sample=sample,
            unit_ids=list(getattr(retrieval, "selected_sentence_ids", []) or []),
            unit_texts=list(getattr(retrieval, "selected_sentences", []) or []),
            graph_mode=graph_mode,
        )
    if rendered_match is None:
        rendered_match = supporting_fact_match_details(
            sample=sample,
            unit_ids=list(getattr(rendered, "sentence_ids", []) or []),
            unit_texts=list(getattr(rendered, "sentences", []) or []),
            graph_mode=graph_mode,
        )

    gold_total = _safe_int(retrieval_match.get("gold_total", 0), 0)
    matched_gold_total = _safe_int(retrieval_match.get("matched_gold_total", 0), 0)
    pred_unit_total = _safe_int(retrieval_match.get("predicted_unit_total", 0), 0)
    rend_gold_total = _safe_int(rendered_match.get("gold_total", 0), 0)
    rend_matched_total = _safe_int(rendered_match.get("matched_gold_total", 0), 0)
    rend_pred_total = _safe_int(rendered_match.get("predicted_unit_total", 0), 0)

    sf_recall = (
        _safe_float(supporting_fact_recall, -1.0)
        if supporting_fact_recall is not None
        else float(safe_div(matched_gold_total, gold_total))
    )
    if sf_recall < 0.0:
        sf_recall = float(safe_div(matched_gold_total, gold_total))
    sf_precision = (
        _safe_float(supporting_fact_precision, -1.0)
        if supporting_fact_precision is not None
        else float(safe_div(matched_gold_total, pred_unit_total))
    )
    if sf_precision < 0.0:
        sf_precision = float(safe_div(matched_gold_total, pred_unit_total))

    rendered_recall = (
        _safe_float(rendered_supporting_fact_recall, -1.0)
        if rendered_supporting_fact_recall is not None
        else float(safe_div(rend_matched_total, rend_gold_total))
    )
    if rendered_recall < 0.0:
        rendered_recall = float(safe_div(rend_matched_total, rend_gold_total))
    rendered_precision = (
        _safe_float(rendered_supporting_fact_precision, -1.0)
        if rendered_supporting_fact_precision is not None
        else float(safe_div(rend_matched_total, rend_pred_total))
    )
    if rendered_precision < 0.0:
        rendered_precision = float(safe_div(rend_matched_total, rend_pred_total))

    rank_bundle = _rank_metric_bundle(retrieval_match)
    metrics: Dict[str, float] = dict(rank_bundle)

    metrics["supporting_fact_precision"] = float(sf_precision)
    metrics["supporting_fact_recall"] = float(sf_recall)
    metrics["supporting_fact_f1"] = float(
        safe_div(2.0 * sf_precision * sf_recall, (sf_precision + sf_recall))
    )
    metrics["em"] = _clip01(em)
    metrics["f1"] = _clip01(f1)
    metrics["rendered_supporting_fact_precision"] = float(rendered_precision)
    metrics["rendered_supporting_fact_recall"] = float(rendered_recall)
    metrics["rendered_supporting_fact_f1"] = float(
        safe_div(2.0 * rendered_precision * rendered_recall, (rendered_precision + rendered_recall))
    )

    diagnostics = dict(getattr(retrieval, "diagnostics", {}) or {})
    gdiag = dict(generation_diagnostics or {})
    answer_bearing_texts = []
    for unit in list(retrieval_match.get("predicted_units", []) or []):
        if not list(unit.get("matched_gold_sentence_ids", []) or []):
            continue
        txt = str(unit.get("text", "") or "").strip()
        if txt:
            answer_bearing_texts.append(txt)
    if not answer_bearing_texts:
        answer_bearing_texts = [str(s or "") for s in list(getattr(rendered, "sentences", []) or [])]
    overlap = _token_overlap_ratio(prediction, " ".join(answer_bearing_texts))
    rendered_text = str(getattr(rendered, "text", "") or "")
    if not rendered_text:
        rendered_text = " ".join([str(s or "") for s in list(getattr(rendered, "sentences", []) or [])]).strip()
    rendered_token_count = max(1, len(content_tokens(rendered_text)))
    answer_aliases = _collect_answer_surface_aliases(sample)
    surface = _answer_surface_stats(rendered_text, answer_aliases)
    answer_surface_present = bool(surface.get("present", False))
    answer_surface_token_count = int(surface.get("token_count", 0) or 0)
    answer_surface_position = _safe_float(surface.get("position", -1.0), -1.0)

    answer_bearing_chunk_present = False
    if "answer_bearing_chunk_present" in diagnostics:
        answer_bearing_chunk_present = bool(diagnostics.get("answer_bearing_chunk_present"))
    else:
        answer_bearing_chunk_present = bool(
            _safe_float(diagnostics.get("raw_focus_answer_bearing_chunk_count", 0.0), 0.0) > 0.0
            or _safe_float(diagnostics.get("answer_bearing_path_hit", 0.0), 0.0) > 0.0
            or matched_gold_total > 0
        )

    minimal_cov = diagnostics.get("minimal_support_subset_coverage", None)
    if minimal_cov is None:
        target = 1 if gold_total <= 1 else min(2, gold_total)
        minimal_cov = float(safe_div(matched_gold_total, target))

    equivalent_cov = diagnostics.get("equivalent_evidence_coverage", None)
    if equivalent_cov is None:
        equivalent_cov = max(float(sf_recall), float(rendered_recall))

    f1_low = bool(_safe_float(f1, 0.0) <= 0.01)
    support_present = bool(matched_gold_total > 0)
    qa_util_applied = bool(gdiag.get("qa_utilization_applied", False))
    qa_util_variant = str(gdiag.get("qa_utilization_variant", "") or "").strip()
    if (not qa_util_applied) and qa_util_variant:
        qa_util_applied = True
    qa_util_changed = bool(gdiag.get("qa_utilization_changed", False))
    if "answer_present_but_generation_fail" in diagnostics:
        abgf = _clip01(diagnostics.get("answer_present_but_generation_fail", 0.0))
    else:
        abgf = 1.0 if (bool(answer_bearing_chunk_present) and bool(qa_executed) and f1_low) else 0.0

    abgf_support_present_generation_fail = 1.0 if (support_present and bool(qa_executed) and f1_low) else 0.0
    abgf_answer_surface_present_generation_fail = 1.0 if (answer_surface_present and bool(qa_executed) and f1_low) else 0.0
    support_present_answer_surface_absent = 1.0 if (support_present and (not answer_surface_present)) else 0.0
    answer_surface_present_but_low_overlap = 1.0 if (answer_surface_present and float(overlap) < 0.35) else 0.0

    metrics["equivalent_evidence_coverage"] = _clip01(equivalent_cov)
    metrics["minimal_support_subset_coverage"] = _clip01(minimal_cov)
    metrics["answer_bearing_chunk_present"] = 1.0 if bool(answer_bearing_chunk_present) else 0.0
    metrics["answer_present_but_generation_fail"] = float(abgf)
    metrics["abgf_support_present_generation_fail"] = float(abgf_support_present_generation_fail)
    metrics["abgf_answer_surface_present_generation_fail"] = float(abgf_answer_surface_present_generation_fail)
    metrics["support_present_answer_surface_absent"] = float(support_present_answer_surface_absent)
    metrics["answer_surface_present_but_low_overlap"] = float(answer_surface_present_but_low_overlap)
    metrics["answer_surface_present"] = 1.0 if answer_surface_present else 0.0
    metrics["answer_surface_position"] = float(answer_surface_position if answer_surface_position >= 0.0 else -1.0)
    metrics["answer_surface_position_avg"] = float(answer_surface_position if answer_surface_position >= 0.0 else 0.0)
    metrics["answer_surface_token_density"] = float(
        safe_div(float(answer_surface_token_count), float(rendered_token_count))
    )
    metrics["qa_utilization_activation"] = 1.0 if qa_util_applied else 0.0
    metrics["qa_utilization_changed"] = 1.0 if qa_util_changed else 0.0
    metrics["matched_gold_total"] = float(matched_gold_total)
    metrics["support_present"] = 1.0 if support_present else 0.0
    metrics["output_overlap_answer_bearing"] = _clip01(overlap)
    metrics["faithfulness"] = _query_faithfulness_proxy(
        prediction=str(prediction or ""),
        rendered_sentences=list(getattr(rendered, "sentences", []) or []),
        rendered_supporting_recall=float(rendered_recall),
        generation_diagnostics=generation_diagnostics,
        em=_safe_float(em, 0.0),
        f1=_safe_float(f1, 0.0),
    )
    return metrics


def aggregate_run_eval_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    seq = list(rows or [])
    if not seq:
        return {}

    def _m(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        return _safe_float(((row.get("metrics", {}) or {}).get(key, default)), default)

    def _mean_nonneg(vals: Sequence[float]) -> float:
        seq_vals = [float(v) for v in list(vals or []) if _safe_float(v, -1.0) >= 0.0]
        if not seq_vals:
            return 0.0
        return float(sum(seq_vals) / len(seq_vals))

    out: Dict[str, float] = {
        "n_samples": float(len(seq)),
        "recall_at_1": mean_or_zero([_m(r, "recall_at_1") for r in seq]),
        "recall_at_5": mean_or_zero([_m(r, "recall_at_5") for r in seq]),
        "recall_at_10": mean_or_zero([_m(r, "recall_at_10") for r in seq]),
        "recall_at_20": mean_or_zero([_m(r, "recall_at_20") for r in seq]),
        "hit_at_1": mean_or_zero([_m(r, "hit_at_1") for r in seq]),
        "hit_at_5": mean_or_zero([_m(r, "hit_at_5") for r in seq]),
        "hit_at_10": mean_or_zero([_m(r, "hit_at_10") for r in seq]),
        "hit_at_20": mean_or_zero([_m(r, "hit_at_20") for r in seq]),
        "mrr_at_10": mean_or_zero([_m(r, "mrr_at_10") for r in seq]),
        "mrr_at_20": mean_or_zero([_m(r, "mrr_at_20") for r in seq]),
        "ndcg_at_5": mean_or_zero([_m(r, "ndcg_at_5") for r in seq]),
        "ndcg_at_10": mean_or_zero([_m(r, "ndcg_at_10") for r in seq]),
        "ndcg_at_20": mean_or_zero([_m(r, "ndcg_at_20") for r in seq]),
        "atleastone_at_5": mean_or_zero([_m(r, "atleastone_at_5") for r in seq]),
        "atleastone_at_10": mean_or_zero([_m(r, "atleastone_at_10") for r in seq]),
        "atleastone_at_20": mean_or_zero([_m(r, "atleastone_at_20") for r in seq]),
        "allsupport_at_5": mean_or_zero([_m(r, "allsupport_at_5") for r in seq]),
        "allsupport_at_10": mean_or_zero([_m(r, "allsupport_at_10") for r in seq]),
        "allsupport_at_20": mean_or_zero([_m(r, "allsupport_at_20") for r in seq]),
        "context_precision": mean_or_zero([_m(r, "context_precision") for r in seq]),
        "faithfulness": mean_or_zero([_m(r, "faithfulness") for r in seq]),
        "supporting_fact_precision": mean_or_zero([_m(r, "supporting_fact_precision") for r in seq]),
        "supporting_fact_recall": mean_or_zero([_m(r, "supporting_fact_recall") for r in seq]),
        "supporting_fact_f1": mean_or_zero([_m(r, "supporting_fact_f1") for r in seq]),
        "rendered_supporting_fact_precision": mean_or_zero([_m(r, "rendered_supporting_fact_precision") for r in seq]),
        "rendered_supporting_fact_recall": mean_or_zero([_m(r, "rendered_supporting_fact_recall") for r in seq]),
        "rendered_supporting_fact_f1": mean_or_zero([_m(r, "rendered_supporting_fact_f1") for r in seq]),
        "equivalent_evidence_coverage": mean_or_zero([_m(r, "equivalent_evidence_coverage") for r in seq]),
        "minimal_support_subset_coverage": mean_or_zero([_m(r, "minimal_support_subset_coverage") for r in seq]),
        "answer_bearing_chunk_present": mean_or_zero([_m(r, "answer_bearing_chunk_present") for r in seq]),
        "answer_present_but_generation_fail": mean_or_zero([_m(r, "answer_present_but_generation_fail") for r in seq]),
        "abgf_support_present_generation_fail": mean_or_zero([_m(r, "abgf_support_present_generation_fail") for r in seq]),
        "abgf_answer_surface_present_generation_fail": mean_or_zero(
            [_m(r, "abgf_answer_surface_present_generation_fail") for r in seq]
        ),
        "support_present_answer_surface_absent": mean_or_zero([_m(r, "support_present_answer_surface_absent") for r in seq]),
        "answer_surface_present_but_low_overlap": mean_or_zero(
            [_m(r, "answer_surface_present_but_low_overlap") for r in seq]
        ),
        "answer_surface_present": mean_or_zero([_m(r, "answer_surface_present") for r in seq]),
        "answer_surface_position_avg": _mean_nonneg([_m(r, "answer_surface_position", -1.0) for r in seq]),
        "answer_surface_token_density": mean_or_zero([_m(r, "answer_surface_token_density") for r in seq]),
        "qa_utilization_activation_rate": mean_or_zero([_m(r, "qa_utilization_activation") for r in seq]),
        "qa_utilization_changed_rate": mean_or_zero([_m(r, "qa_utilization_changed") for r in seq]),
        "output_overlap_answer_bearing": mean_or_zero([_m(r, "output_overlap_answer_bearing") for r in seq]),
        "em": mean_or_zero([_m(r, "em") for r in seq]),
        "f1": mean_or_zero([_m(r, "f1") for r in seq]),
        "fallback_rate": mean_or_zero([1.0 if bool(r.get("generation_fallback", False)) else 0.0 for r in seq]),
        "retrieval_ms": mean_or_zero(
            [_safe_float((r.get("efficiency", {}) or {}).get("retrieval_latency_ms", 0.0), 0.0) for r in seq]
        ),
        "generation_ms": mean_or_zero(
            [
                _safe_float(
                    (r.get("efficiency", {}) or {}).get(
                        "generation_ms",
                        (r.get("efficiency", {}) or {}).get("generation_latency_ms", 0.0),
                    ),
                    0.0,
                )
                for r in seq
            ]
        ),
        "total_ms": mean_or_zero(
            [_safe_float((r.get("efficiency", {}) or {}).get("total_latency_ms", 0.0), 0.0) for r in seq]
        ),
    }

    out["r5_per_100ms"] = float(safe_div(out["recall_at_5"] * 100.0, out["retrieval_ms"])) if out["retrieval_ms"] > 0.0 else 0.0
    out["mrr_at_10_per_100ms"] = (
        float(safe_div(out["mrr_at_10"] * 100.0, out["retrieval_ms"])) if out["retrieval_ms"] > 0.0 else 0.0
    )
    out["context_precision_per_100ms"] = (
        float(safe_div(out["context_precision"] * 100.0, out["total_ms"])) if out["total_ms"] > 0.0 else 0.0
    )
    out["em_per_100ms"] = float(safe_div(out["em"] * 100.0, out["total_ms"])) if out["total_ms"] > 0.0 else 0.0
    out["f1_per_100ms"] = float(safe_div(out["f1"] * 100.0, out["total_ms"])) if out["total_ms"] > 0.0 else 0.0
    return out


def table_from_records(headers: Sequence[str], records: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for record in list(records or []):
        rows.append([record.get(h, "") for h in headers])
    return markdown_table(list(headers), rows)


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(payload or {}), ensure_ascii=False, indent=2), encoding="utf-8")


def write_tsv(path: str, headers: Sequence[str], records: Sequence[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(str(h) for h in headers)]
    for record in list(records or []):
        lines.append("\t".join(str(record.get(h, "")) for h in headers))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    display_name: str
    group: str
    description: str = ""
    higher_is_better: bool = True
    fmt: str = ".4f"
    enabled: bool = True
    requires: Tuple[str, ...] = field(default_factory=tuple)
    is_diagnostic: bool = False
    is_proxy: bool = False
    source_level: str = "context"


class MetricRegistry:
    def __init__(self, specs: Optional[Iterable[MetricSpec]] = None) -> None:
        self._specs: Dict[str, MetricSpec] = {}
        self._order: List[str] = []
        for spec in list(specs or []):
            self.register(spec)

    def register(self, spec: MetricSpec) -> None:
        if spec.name in self._specs:
            self._order = [n for n in self._order if n != spec.name]
        self._specs[spec.name] = spec
        self._order.append(spec.name)

    def get(self, name: str) -> Optional[MetricSpec]:
        return self._specs.get(str(name))

    def all_specs(self) -> List[MetricSpec]:
        return [self._specs[n] for n in self._order if n in self._specs]

    def enabled_specs(self) -> List[MetricSpec]:
        return [spec for spec in self.all_specs() if bool(spec.enabled)]

    def enabled_names(self) -> List[str]:
        return [spec.name for spec in self.enabled_specs()]

    def groups(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for spec in self.enabled_specs():
            if spec.group in seen:
                continue
            seen.add(spec.group)
            out.append(spec.group)
        return out

    def enabled_by_group(self, group: str) -> List[MetricSpec]:
        g = str(group or "")
        return [spec for spec in self.enabled_specs() if str(spec.group) == g]


def _coerce_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return float(x)


def _lookup_metric(payload: Mapping[str, Any], aliases: Sequence[str], default: float = 0.0) -> float:
    src = dict(payload or {})
    for key in list(aliases or []):
        if key in src:
            vv = _coerce_float_or_none(src.get(key))
            if vv is not None:
                return float(vv)
    return float(default)


def _percentile(values: Sequence[float], q: float) -> float:
    seq = sorted([_safe_float(v, 0.0) for v in list(values or [])])
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    qq = min(max(_safe_float(q, 0.0), 0.0), 1.0)
    pos = qq * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(seq[lo])
    frac = pos - float(lo)
    return float(seq[lo] * (1.0 - frac) + seq[hi] * frac)


def build_context_efficiency_metric_registry() -> MetricRegistry:
    specs = [
        MetricSpec(
            name="recall_at_1",
            display_name="R@1",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="recall_at_5",
            display_name="R@5",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@5.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="recall_at_10",
            display_name="R@10",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@10.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="recall_at_20",
            display_name="R@20",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@20.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="hit_at_5",
            display_name="Hit@5",
            group="Retrieval Ranking Metrics",
            description="At least one supporting-fact hit in top-5.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="hit_at_10",
            display_name="Hit@10",
            group="Retrieval Ranking Metrics",
            description="At least one supporting-fact hit in top-10.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="mrr_at_10",
            display_name="MRR@10",
            group="Retrieval Ranking Metrics",
            description="Mean reciprocal rank at top-10.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="ndcg_at_10",
            display_name="nDCG@10",
            group="Retrieval Ranking Metrics",
            description="Binary nDCG at top-10.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_precision",
            display_name="sf_P",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact precision.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_recall",
            display_name="sf_R",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact recall.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_f1",
            display_name="sf_F1",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact F1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="rendered_supporting_fact_precision",
            display_name="Rendered_sf_P",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact precision after rendering.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="rendered_supporting_fact_recall",
            display_name="Rendered_sf_R",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact recall after rendering.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="rendered_supporting_fact_f1",
            display_name="Rendered_sf_F1",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact F1 after rendering.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="context_precision",
            display_name="ContextPrecision",
            group="Multi-hop Evidence Metrics",
            description="Ragas-style context precision over ranked contexts.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="avg_context_tokens",
            display_name="AvgContextTok",
            group="Context Efficiency Metrics",
            description="Average tokens in final context passed to generator.",
            higher_is_better=False,
            fmt=".1f",
            source_level="context",
        ),
        MetricSpec(
            name="median_context_tokens",
            display_name="MedianContextTok",
            group="Context Efficiency Metrics",
            description="Median final context tokens.",
            higher_is_better=False,
            fmt=".1f",
            source_level="context",
        ),
        MetricSpec(
            name="p90_context_tokens",
            display_name="P90ContextTok",
            group="Context Efficiency Metrics",
            description="P90 final context tokens.",
            higher_is_better=False,
            fmt=".1f",
            source_level="context",
        ),
        MetricSpec(
            name="max_context_tokens",
            display_name="MaxContextTok",
            group="Context Efficiency Metrics",
            description="Max final context tokens.",
            higher_is_better=False,
            fmt=".1f",
            source_level="context",
        ),
        MetricSpec(
            name="raw_candidate_tokens",
            display_name="RawCandidateTok",
            group="Context Efficiency Metrics",
            description="Average raw candidate tokens before rendering.",
            higher_is_better=False,
            fmt=".1f",
            source_level="context",
        ),
        MetricSpec(
            name="rendered_context_tokens",
            display_name="RenderedContextTok",
            group="Context Efficiency Metrics",
            description="Average rendered context tokens.",
            higher_is_better=False,
            fmt=".1f",
            source_level="context",
        ),
        MetricSpec(
            name="compression_ratio",
            display_name="CompressionRatio",
            group="Context Efficiency Metrics",
            description="rendered_context_tokens / raw_candidate_tokens.",
            higher_is_better=False,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="context_reduction_rate",
            display_name="ContextReductionRate",
            group="Context Efficiency Metrics",
            description="1 - compression_ratio.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="supporting_fact_token_density",
            display_name="SFDensity",
            group="Context Efficiency Metrics",
            description="Matched supporting-fact tokens / rendered tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="answer_bearing_token_density",
            display_name="AnswerDensity",
            group="Context Efficiency Metrics",
            description="Answer-bearing tokens / rendered tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="non_support_token_ratio",
            display_name="NonSupportRatio",
            group="Context Efficiency Metrics",
            description="1 - supporting_fact_token_density.",
            higher_is_better=False,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="non_answer_bearing_token_ratio",
            display_name="NonAnswerRatio",
            group="Context Efficiency Metrics",
            description="1 - answer_bearing_token_density.",
            higher_is_better=False,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="tokens_per_support_hit",
            display_name="TokPerSupportHit",
            group="Token-normalized Utility Metrics",
            description="Rendered tokens per supporting-fact hit.",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="tokens_per_answer_bearing_hit",
            display_name="TokPerAnswerHit",
            group="Token-normalized Utility Metrics",
            description="Rendered tokens per answer-bearing hit.",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="sf_F1_per_1k_context_tokens",
            display_name="sf_F1/1KTok",
            group="Token-normalized Utility Metrics",
            description="Supporting-fact F1 normalized by average context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="efficiency",
            is_proxy=True,
        ),
        MetricSpec(
            name="ContextPrecision_per_1k_context_tokens",
            display_name="ContextPrecision/1KTok",
            group="Token-normalized Utility Metrics",
            description="Context precision normalized by average context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="efficiency",
            is_proxy=True,
        ),
        MetricSpec(
            name="EM_per_1k_context_tokens",
            display_name="EM/1KTok",
            group="Token-normalized Utility Metrics",
            description="EM normalized by average context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="efficiency",
            is_proxy=True,
        ),
        MetricSpec(
            name="F1_per_1k_context_tokens",
            display_name="F1/1KTok",
            group="Token-normalized Utility Metrics",
            description="F1 normalized by average context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="efficiency",
            is_proxy=True,
        ),
        MetricSpec(
            name="em",
            display_name="EM",
            group="QA Metrics",
            description="Exact match.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
        ),
        MetricSpec(
            name="f1",
            display_name="F1",
            group="QA Metrics",
            description="Token-level F1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
        ),
        MetricSpec(
            name="retrieval_ms",
            display_name="Retrieval(ms)",
            group="Latency Metrics",
            description="Average retrieval latency in milliseconds.",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="generation_ms",
            display_name="Generation(ms)",
            group="Latency Metrics",
            description="Average generation latency in milliseconds.",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="total_ms",
            display_name="Total(ms)",
            group="Latency Metrics",
            description="Average total latency in milliseconds.",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="context_tokens_per_ms",
            display_name="ContextTok/ms",
            group="Latency Metrics",
            description="Average context tokens divided by total latency.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
            is_proxy=True,
            is_diagnostic=True,
        ),
        MetricSpec(
            name="F1_per_1k_tokens_per_100ms",
            display_name="F1/1KTok/100ms",
            group="Diagnostic Metrics",
            description="Diagnostic compound efficiency metric.",
            higher_is_better=True,
            fmt=".6f",
            source_level="diagnostic",
            is_proxy=True,
            is_diagnostic=True,
        ),
        MetricSpec(
            name="support_text_match_rate",
            display_name="SupportTextMatchRate",
            group="Diagnostic Metrics",
            description="Share of gold support texts matched in rendered context.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="answer_bearing_match_rate",
            display_name="AnswerBearingMatchRate",
            group="Diagnostic Metrics",
            description="Share of queries with at least one answer-bearing context hit.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="missing_rendered_context_rate",
            display_name="MissingRenderedRate",
            group="Diagnostic Metrics",
            description="Share of queries missing rendered context text.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="raw_candidate_token_coverage",
            display_name="RawTokenCoverage",
            group="Diagnostic Metrics",
            description="Share of queries with raw candidate token counts available.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
    ]
    return MetricRegistry(specs)


def build_abgf_improvement_metric_registry() -> MetricRegistry:
    specs = [
        MetricSpec(
            name="recall_at_1",
            display_name="R@1",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="recall_at_5",
            display_name="R@5",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@5.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="recall_at_10",
            display_name="R@10",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@10.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_precision",
            display_name="sf_P",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact precision.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_recall",
            display_name="sf_R",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact recall.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_f1",
            display_name="sf_F1",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact F1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="avg_context_tokens",
            display_name="AvgContextTok",
            group="Context Compactness Guard Table",
            description="Average rendered context tokens.",
            higher_is_better=False,
            fmt=".1f",
            source_level="context",
        ),
        MetricSpec(
            name="sf_token_density",
            display_name="sf_token_density",
            group="Context Compactness Guard Table",
            description="Matched supporting-fact tokens / rendered context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="answer_token_density",
            display_name="answer_token_density",
            group="Context Compactness Guard Table",
            description="Answer-surface cue tokens / rendered context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="F1_per_1k_context_tokens",
            display_name="F1/1KTok",
            group="Context Compactness Guard Table",
            description="F1 normalized by average context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="efficiency",
            is_proxy=True,
        ),
        MetricSpec(
            name="answer_bearing_chunk_present",
            display_name="answer_bearing_chunk_present",
            group="ABGF Breakdown Table",
            description="Rate of answer-bearing chunk present.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="output_overlap_answer_bearing",
            display_name="output_overlap_answer_bearing",
            group="ABGF Breakdown Table",
            description="Prediction overlap with answer-bearing evidence.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="answer_present_but_generation_fail",
            display_name="ABGF",
            group="ABGF Breakdown Table",
            description="Legacy ABGF metric.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="abgf_support_present_generation_fail",
            display_name="abgf_support_present_generation_fail",
            group="ABGF Breakdown Table",
            description="support present + qa executed + low F1.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="abgf_answer_surface_present_generation_fail",
            display_name="abgf_answer_surface_present_generation_fail",
            group="ABGF Breakdown Table",
            description="answer surface present + qa executed + low F1.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="support_present_answer_surface_absent",
            display_name="support_present_answer_surface_absent",
            group="ABGF Breakdown Table",
            description="support present but answer surface absent in rendered context.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="answer_surface_present_but_low_overlap",
            display_name="answer_surface_present_but_low_overlap",
            group="ABGF Breakdown Table",
            description="answer surface present with low output overlap.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="qa_utilization_activation_rate",
            display_name="qa_utilization_activation_rate",
            group="QA Utilization Activation Table",
            description="Share of queries where QA utilization/postprocess was applied.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
        ),
        MetricSpec(
            name="qa_utilization_changed_rate",
            display_name="qa_utilization_changed_rate",
            group="QA Utilization Activation Table",
            description="Share of queries where QA utilization changed prediction.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
        ),
        MetricSpec(
            name="normalization_activation_rate",
            display_name="normalization_activation_rate",
            group="QA Utilization Activation Table",
            description="Share with normalization variant activated.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="normalization_changed_rate",
            display_name="normalization_changed_rate",
            group="QA Utilization Activation Table",
            description="Share with normalization variant changing prediction.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="normalization_helped_count",
            display_name="normalization_helped_count",
            group="QA Utilization Activation Table",
            description="Count of normalization queries where final F1 improved vs initial.",
            higher_is_better=True,
            fmt=".0f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="normalization_hurt_count",
            display_name="normalization_hurt_count",
            group="QA Utilization Activation Table",
            description="Count of normalization queries where final F1 dropped vs initial.",
            higher_is_better=False,
            fmt=".0f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="extraction_activation_rate",
            display_name="extraction_activation_rate",
            group="QA Utilization Activation Table",
            description="Share with type-aware extraction activated.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="extraction_changed_rate",
            display_name="extraction_changed_rate",
            group="QA Utilization Activation Table",
            description="Share with type-aware extraction changing prediction.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="extraction_helped_count",
            display_name="extraction_helped_count",
            group="QA Utilization Activation Table",
            description="Count of type-aware extraction queries where final F1 improved.",
            higher_is_better=True,
            fmt=".0f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="extraction_hurt_count",
            display_name="extraction_hurt_count",
            group="QA Utilization Activation Table",
            description="Count of type-aware extraction queries where final F1 dropped.",
            higher_is_better=False,
            fmt=".0f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="verification_activation_rate",
            display_name="verification_activation_rate",
            group="QA Utilization Activation Table",
            description="Share with evidence-supported verification activated.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="verification_changed_rate",
            display_name="verification_changed_rate",
            group="QA Utilization Activation Table",
            description="Share with evidence-supported verification changing prediction.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="verification_helped_count",
            display_name="verification_helped_count",
            group="QA Utilization Activation Table",
            description="Count of verification queries where final F1 improved.",
            higher_is_better=True,
            fmt=".0f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="verification_hurt_count",
            display_name="verification_hurt_count",
            group="QA Utilization Activation Table",
            description="Count of verification queries where final F1 dropped.",
            higher_is_better=False,
            fmt=".0f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="highlight_activation_rate",
            display_name="highlight_activation_rate",
            group="QA Utilization Activation Table",
            description="Share of queries where answer-cue highlight marker was applied.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="em",
            display_name="EM",
            group="Main QA Table",
            description="Exact match.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
        ),
        MetricSpec(
            name="f1",
            display_name="F1",
            group="Main QA Table",
            description="Token-level F1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
        ),
        MetricSpec(
            name="retrieval_ms",
            display_name="retrieval_ms",
            group="Latency Metrics",
            description="Average retrieval latency (ms).",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="generation_ms",
            display_name="generation_ms",
            group="Latency Metrics",
            description="Average generation latency (ms).",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="total_ms",
            display_name="total_ms",
            group="Latency Metrics",
            description="Average end-to-end latency (ms).",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="fallback_rate",
            display_name="fallback_rate",
            group="Latency Metrics",
            description="Fallback generation rate.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="answer_surface_position_avg",
            display_name="answer_surface_position_avg",
            group="Diagnostic Metrics",
            description="Average normalized position of first answer surface cue in rendered context.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="answer_surface_token_density",
            display_name="answer_surface_token_density",
            group="Diagnostic Metrics",
            description="Token density of answer surface cues in rendered context.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
    ]
    return MetricRegistry(specs)


def build_context_interface_abgf_metric_registry() -> MetricRegistry:
    specs = [
        MetricSpec(
            name="recall_at_1",
            display_name="R@1",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="recall_at_5",
            display_name="R@5",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@5.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="recall_at_10",
            display_name="R@10",
            group="Retrieval Ranking Metrics",
            description="Strict supporting-fact Recall@10.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_precision",
            display_name="sf_P",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact precision.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_recall",
            display_name="sf_R",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact recall.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="supporting_fact_f1",
            display_name="sf_F1",
            group="Multi-hop Evidence Metrics",
            description="Supporting-fact F1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="retrieval",
        ),
        MetricSpec(
            name="rendered_sf_recall",
            display_name="rendered_sf_recall",
            group="Multi-hop Evidence Metrics",
            description="Rendered-context supporting-fact recall.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="em",
            display_name="EM",
            group="Main QA Table",
            description="Exact match.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
        ),
        MetricSpec(
            name="f1",
            display_name="F1",
            group="Main QA Table",
            description="Token-level F1.",
            higher_is_better=True,
            fmt=".4f",
            source_level="generation",
        ),
        MetricSpec(
            name="answer_present_but_generation_fail",
            display_name="ABGF",
            group="Main QA Table",
            description="Answer present but generation failed.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="output_overlap_answer_bearing",
            display_name="output_overlap",
            group="Main QA Table",
            description="Prediction overlap with answer-bearing evidence.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="answer_bearing_chunk_present",
            display_name="answer_bearing_chunk_present",
            group="ABGF Breakdown Table",
            description="Rate of answer-bearing chunk present.",
            higher_is_better=True,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="abgf_support_present_generation_fail",
            display_name="abgf_support_present_generation_fail",
            group="ABGF Breakdown Table",
            description="support present + qa executed + low F1.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="abgf_answer_surface_present_generation_fail",
            display_name="abgf_answer_surface_present_generation_fail",
            group="ABGF Breakdown Table",
            description="answer surface present + qa executed + low F1.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="support_present_answer_surface_absent",
            display_name="support_present_answer_surface_absent",
            group="ABGF Breakdown Table",
            description="support present but answer surface absent in rendered context.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="answer_surface_present_but_low_overlap",
            display_name="answer_surface_present_but_low_overlap",
            group="ABGF Breakdown Table",
            description="answer surface present with low output overlap.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="avg_context_tokens",
            display_name="avg_context_tokens",
            group="Context Compactness Guard Table",
            description="Average rendered context tokens.",
            higher_is_better=False,
            fmt=".1f",
            source_level="context",
        ),
        MetricSpec(
            name="sf_token_density",
            display_name="sf_token_density",
            group="Context Compactness Guard Table",
            description="Matched supporting-fact tokens / rendered context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="answer_token_density",
            display_name="answer_token_density",
            group="Context Compactness Guard Table",
            description="Answer-surface cue tokens / rendered context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
        ),
        MetricSpec(
            name="F1_per_1k_context_tokens",
            display_name="F1_per_1k_context_tokens",
            group="Context Compactness Guard Table",
            description="F1 normalized by average context tokens.",
            higher_is_better=True,
            fmt=".4f",
            source_level="efficiency",
            is_proxy=True,
        ),
        MetricSpec(
            name="avg_context_tokens_delta",
            display_name="avg_context_tokens_delta",
            group="Context Compactness Guard Table",
            description="Difference in average context tokens vs champion_reconfirm.",
            higher_is_better=False,
            fmt=".2f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="render_variant",
            display_name="render_variant",
            group="Render/Prompt Variant Table",
            description="Render-side variant name.",
            higher_is_better=True,
            fmt="",
            source_level="context",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="prompt_variant",
            display_name="prompt_variant",
            group="Render/Prompt Variant Table",
            description="Prompt-side variant name.",
            higher_is_better=True,
            fmt="",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="corridor_group_count",
            display_name="corridor_group_count",
            group="Render/Prompt Variant Table",
            description="Average number of corridor groups in rendered context.",
            higher_is_better=True,
            fmt=".2f",
            source_level="context",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="path_block_count",
            display_name="path_block_count",
            group="Render/Prompt Variant Table",
            description="Average number of path blocks in rendered context.",
            higher_is_better=True,
            fmt=".2f",
            source_level="context",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="role_tag_activation_rate",
            display_name="role_tag_activation_rate",
            group="Render/Prompt Variant Table",
            description="Share of queries where role tagging activated.",
            higher_is_better=True,
            fmt=".4f",
            source_level="context",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="role_unknown_rate",
            display_name="role_unknown_rate",
            group="Render/Prompt Variant Table",
            description="Average unknown role-tag rate.",
            higher_is_better=False,
            fmt=".4f",
            source_level="context",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="retrieval_ms",
            display_name="retrieval_ms",
            group="Latency Metrics",
            description="Average retrieval latency (ms).",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="generation_ms",
            display_name="generation_ms",
            group="Latency Metrics",
            description="Average generation latency (ms).",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="total_ms",
            display_name="total_ms",
            group="Latency Metrics",
            description="Average end-to-end latency (ms).",
            higher_is_better=False,
            fmt=".2f",
            source_level="efficiency",
        ),
        MetricSpec(
            name="fallback_rate",
            display_name="fallback_rate",
            group="Latency Metrics",
            description="Fallback generation rate.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
        ),
        MetricSpec(
            name="answer_surface_position_avg",
            display_name="answer_surface_position_avg",
            group="Diagnostic Metrics",
            description="Average normalized answer-surface position in rendered context.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="answer_surface_position_delta",
            display_name="answer_surface_position_delta",
            group="Diagnostic Metrics",
            description="Delta of answer-surface position avg vs champion_reconfirm.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="qa_utilization_activation_rate",
            display_name="qa_utilization_activation_rate",
            group="Diagnostic Metrics",
            description="Share of queries where QA postprocess activated.",
            higher_is_better=False,
            fmt=".4f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="qa_utilization_changed_rate",
            display_name="qa_utilization_changed_rate",
            group="Diagnostic Metrics",
            description="Share of queries where QA postprocess changed prediction.",
            higher_is_better=False,
            fmt=".4f",
            source_level="generation",
            is_diagnostic=True,
        ),
        MetricSpec(
            name="no_prediction_rewrite_violation_rate",
            display_name="no_prediction_rewrite_violation_rate",
            group="Diagnostic Metrics",
            description="Rate of queries violating no-rewrite constraint.",
            higher_is_better=False,
            fmt=".4f",
            source_level="diagnostic",
            is_diagnostic=True,
        ),
    ]
    return MetricRegistry(specs)


def aggregate_context_efficiency_metrics(
    query_metrics: Sequence[Mapping[str, Any]],
    base_metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rows = list(query_metrics or [])
    base = dict(base_metrics or {})
    n = len(rows)
    if n <= 0:
        return {}

    rendered_tokens = [_safe_float(r.get("rendered_context_tokens", 0.0), 0.0) for r in rows]
    raw_tokens_opt = [_coerce_float_or_none(r.get("raw_candidate_tokens")) for r in rows]
    raw_tokens = [float(v) for v in raw_tokens_opt if v is not None and v > 0.0]

    total_rendered_tokens = float(sum(rendered_tokens))
    total_raw_tokens = float(sum(raw_tokens))

    support_token_total = float(sum(_safe_float(r.get("matched_support_tokens", 0.0), 0.0) for r in rows))
    answer_token_total = float(sum(_safe_float(r.get("answer_bearing_tokens", 0.0), 0.0) for r in rows))
    support_hit_total = float(sum(_safe_float(r.get("support_hits", 0.0), 0.0) for r in rows))
    answer_hit_total = float(sum(_safe_float(r.get("answer_bearing_hits", 0.0), 0.0) for r in rows))
    support_match_total = float(sum(_safe_float(r.get("support_match_count", 0.0), 0.0) for r in rows))
    support_gold_total = float(sum(_safe_float(r.get("support_total_count", 0.0), 0.0) for r in rows))
    answer_match_queries = float(
        sum(1 for r in rows if _safe_float(r.get("answer_bearing_hits", 0.0), 0.0) > 0.0)
    )
    missing_rendered = float(sum(1 for r in rows if bool(r.get("missing_rendered_context", False))))
    raw_coverage_count = float(sum(1 for v in raw_tokens_opt if v is not None and v > 0.0))

    retrieval_ms = _lookup_metric(base, ("retrieval_ms", "retrieval_latency_ms"), default=0.0)
    generation_ms = _lookup_metric(base, ("generation_ms", "generation_latency_ms"), default=0.0)
    total_ms = _lookup_metric(base, ("total_ms", "total_latency_ms"), default=0.0)

    out: Dict[str, Any] = {
        "n_samples": float(n),
        "avg_context_tokens": float(mean_or_zero(rendered_tokens)),
        "median_context_tokens": float(_percentile(rendered_tokens, 0.5)),
        "p90_context_tokens": float(_percentile(rendered_tokens, 0.9)),
        "max_context_tokens": float(max(rendered_tokens) if rendered_tokens else 0.0),
        "raw_candidate_tokens": float(mean_or_zero(raw_tokens)) if raw_tokens else None,
        "rendered_context_tokens": float(mean_or_zero(rendered_tokens)),
        "compression_ratio": float(safe_div(total_rendered_tokens, total_raw_tokens)) if total_raw_tokens > 0.0 else None,
        "context_reduction_rate": None,
        "supporting_fact_token_density": float(safe_div(support_token_total, total_rendered_tokens)),
        "answer_bearing_token_density": float(safe_div(answer_token_total, total_rendered_tokens)),
        "tokens_per_support_hit": float(safe_div(total_rendered_tokens, support_hit_total)) if support_hit_total > 0.0 else 0.0,
        "tokens_per_answer_bearing_hit": float(safe_div(total_rendered_tokens, answer_hit_total))
        if answer_hit_total > 0.0
        else 0.0,
        "support_text_match_rate": float(safe_div(support_match_total, support_gold_total)),
        "answer_bearing_match_rate": float(safe_div(answer_match_queries, float(n))),
        "missing_rendered_context_rate": float(safe_div(missing_rendered, float(n))),
        "raw_candidate_token_coverage": float(safe_div(raw_coverage_count, float(n))),
        "retrieval_ms": float(retrieval_ms),
        "generation_ms": float(generation_ms),
        "total_ms": float(total_ms if total_ms > 0.0 else (retrieval_ms + generation_ms)),
    }
    if out["compression_ratio"] is not None:
        out["context_reduction_rate"] = float(1.0 - float(out["compression_ratio"]))
    out["non_support_token_ratio"] = float(max(0.0, 1.0 - out["supporting_fact_token_density"]))
    out["non_answer_bearing_token_ratio"] = float(max(0.0, 1.0 - out["answer_bearing_token_density"]))

    out["recall_at_1"] = _lookup_metric(base, ("recall_at_1", "R@1", "supporting_fact_recall_at_1"), default=0.0)
    out["recall_at_5"] = _lookup_metric(
        base,
        ("recall_at_5", "R@5", "supporting_fact_recall_at_5"),
        default=_lookup_metric(base, ("Recall@5",), default=0.0),
    )
    out["recall_at_10"] = _lookup_metric(base, ("recall_at_10", "R@10", "supporting_fact_recall_at_10"), default=0.0)
    out["recall_at_20"] = _lookup_metric(base, ("recall_at_20", "R@20", "supporting_fact_recall_at_20"), default=0.0)
    out["hit_at_5"] = _lookup_metric(base, ("hit_at_5", "Hit@5"), default=0.0)
    out["hit_at_10"] = _lookup_metric(base, ("hit_at_10", "Hit@10"), default=0.0)
    out["mrr_at_10"] = _lookup_metric(base, ("mrr_at_10", "MRR@10"), default=0.0)
    out["ndcg_at_10"] = _lookup_metric(base, ("ndcg_at_10", "nDCG@10"), default=0.0)
    out["supporting_fact_precision"] = _lookup_metric(
        base,
        ("supporting_fact_precision", "sf_P", "sf_precision"),
        default=0.0,
    )
    out["supporting_fact_recall"] = _lookup_metric(
        base,
        ("supporting_fact_recall", "sf_R", "sf_recall"),
        default=0.0,
    )
    out["supporting_fact_f1"] = _lookup_metric(
        base,
        ("supporting_fact_f1", "sf_F1", "sf_f1"),
        default=0.0,
    )
    out["rendered_supporting_fact_precision"] = _lookup_metric(
        base,
        ("rendered_supporting_fact_precision", "rendered_sf_P"),
        default=0.0,
    )
    out["rendered_supporting_fact_recall"] = _lookup_metric(
        base,
        ("rendered_supporting_fact_recall", "rendered_sf_R"),
        default=0.0,
    )
    out["rendered_supporting_fact_f1"] = _lookup_metric(
        base,
        ("rendered_supporting_fact_f1", "rendered_sf_F1"),
        default=0.0,
    )
    out["context_precision"] = _lookup_metric(base, ("context_precision", "ContextPrecision"), default=0.0)
    out["em"] = _lookup_metric(base, ("em", "EM"), default=0.0)
    out["f1"] = _lookup_metric(base, ("f1", "F1"), default=0.0)

    avg_ctx_k = float(safe_div(out["avg_context_tokens"], 1000.0))
    out["sf_F1_per_1k_context_tokens"] = float(safe_div(out["supporting_fact_f1"], avg_ctx_k)) if avg_ctx_k > 0.0 else 0.0
    out["ContextPrecision_per_1k_context_tokens"] = (
        float(safe_div(out["context_precision"], avg_ctx_k)) if avg_ctx_k > 0.0 else 0.0
    )
    out["EM_per_1k_context_tokens"] = float(safe_div(out["em"], avg_ctx_k)) if avg_ctx_k > 0.0 else 0.0
    out["F1_per_1k_context_tokens"] = float(safe_div(out["f1"], avg_ctx_k)) if avg_ctx_k > 0.0 else 0.0
    out["context_tokens_per_ms"] = float(safe_div(out["avg_context_tokens"], out["total_ms"])) if out["total_ms"] > 0.0 else 0.0

    if avg_ctx_k > 0.0 and out["total_ms"] > 0.0:
        out["F1_per_1k_tokens_per_100ms"] = float(
            safe_div(out["f1"], avg_ctx_k) / (float(out["total_ms"]) / 100.0)
        )
    else:
        out["F1_per_1k_tokens_per_100ms"] = 0.0
    out["sf_token_density"] = float(out.get("supporting_fact_token_density", 0.0))
    out["answer_token_density"] = float(out.get("answer_bearing_token_density", 0.0))
    return out


def validate_context_efficiency_metrics(metrics: Mapping[str, Any]) -> List[str]:
    m = dict(metrics or {})
    issues: List[str] = []

    avg_t = _safe_float(m.get("avg_context_tokens", 0.0), 0.0)
    med_t = _safe_float(m.get("median_context_tokens", 0.0), 0.0)
    p90_t = _safe_float(m.get("p90_context_tokens", 0.0), 0.0)
    max_t = _safe_float(m.get("max_context_tokens", 0.0), 0.0)
    sf_d = _safe_float(m.get("supporting_fact_token_density", 0.0), 0.0)
    ans_d = _safe_float(m.get("answer_bearing_token_density", 0.0), 0.0)
    comp = _safe_float(m.get("compression_ratio", 0.0), 0.0)
    raw_cov = _safe_float(m.get("raw_candidate_token_coverage", 0.0), 0.0)

    if avg_t <= 0.0:
        issues.append("avg_context_tokens must be > 0")
    if med_t <= 0.0:
        issues.append("median_context_tokens must be > 0")
    if p90_t < med_t:
        issues.append("p90_context_tokens must be >= median_context_tokens")
    if max_t < p90_t:
        issues.append("max_context_tokens must be >= p90_context_tokens")
    if not (0.0 <= sf_d <= 1.0):
        issues.append("supporting_fact_token_density must be in [0,1]")
    if not (0.0 <= ans_d <= 1.0):
        issues.append("answer_bearing_token_density must be in [0,1]")
    if raw_cov <= 0.0 and comp > 0.0:
        issues.append("compression_ratio must be computed only when raw_candidate_tokens are available")
    return issues


def metric_definition_notes(
    registry: MetricRegistry,
    tokenizer_mode: str,
    tokenizer_name: str,
    optional_notes: Optional[Sequence[str]] = None,
) -> str:
    lines = [
        "# Metric Definition Notes",
        "",
        "## Tokenization",
        f"- tokenizer_mode: `{tokenizer_mode}`",
        f"- tokenizer_name: `{tokenizer_name}`",
        "- EffiRAG and HippoRAG2 are scored under the same tokenization mode for comparability.",
        "",
        "## Registered Metrics",
    ]
    for group in registry.groups():
        lines.append("")
        lines.append(f"### {group}")
        for spec in registry.enabled_by_group(group):
            traits: List[str] = []
            if spec.is_diagnostic:
                traits.append("diagnostic")
            if spec.is_proxy:
                traits.append("proxy")
            traits_text = f" ({', '.join(traits)})" if traits else ""
            lines.append(f"- `{spec.name}`{traits_text}: {spec.description}")

    extras = [str(x).strip() for x in list(optional_notes or []) if str(x).strip()]
    if extras:
        lines.append("")
        lines.append("## Additional Notes")
        for item in extras:
            lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"
