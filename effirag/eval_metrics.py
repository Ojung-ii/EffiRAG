from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .eval.metrics_hipporag2_parity import normalize_answer_parity
from .metrics import supporting_fact_match_details
from .utils import markdown_table, mean_or_zero, safe_div

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

    if "answer_present_but_generation_fail" in diagnostics:
        abgf = _clip01(diagnostics.get("answer_present_but_generation_fail", 0.0))
    else:
        abgf = 1.0 if (bool(answer_bearing_chunk_present) and bool(qa_executed) and _safe_float(f1, 0.0) <= 0.01) else 0.0

    metrics["equivalent_evidence_coverage"] = _clip01(equivalent_cov)
    metrics["minimal_support_subset_coverage"] = _clip01(minimal_cov)
    metrics["answer_bearing_chunk_present"] = 1.0 if bool(answer_bearing_chunk_present) else 0.0
    metrics["answer_present_but_generation_fail"] = float(abgf)
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
        "output_overlap_answer_bearing": mean_or_zero([_m(r, "output_overlap_answer_bearing") for r in seq]),
        "em": mean_or_zero([_m(r, "em") for r in seq]),
        "f1": mean_or_zero([_m(r, "f1") for r in seq]),
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
