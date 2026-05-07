from typing import Any, Dict, List, Optional, Tuple

from .types import RetrievalResult, Sample
from .utils import mean_or_zero, safe_div

DEFAULT_RECALL_KS = (1, 2, 5, 10, 20, 30, 50, 100, 150, 200)


def _normalize_graph_mode(mode):
    mode = str(mode or "current_entity_graph").strip().lower()
    if mode not in {"current_entity_graph", "entity_chunk_graph"}:
        mode = "current_entity_graph"
    return mode


def _retrieval_graph_mode(retrieval):
    if retrieval is None:
        return "current_entity_graph"
    diagnostics = getattr(retrieval, "diagnostics", {}) or {}
    return _normalize_graph_mode(diagnostics.get("graph_mode", "current_entity_graph"))


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _ordered_unique(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _sample_sentence_lookup(sample):
    lookup = {}
    for doc in list(getattr(sample, "contexts", []) or []):
        title = str(getattr(doc, "title", "") or "")
        for sent_idx, sent in enumerate(list(getattr(doc, "sentences", []) or [])):
            lookup[f"{title}::{sent_idx}"] = str(sent or "")
    return lookup


def _normalize_text_for_match(text):
    return " ".join(str(text or "").strip().lower().split())


def _parse_sentence_unit_id(unit_id):
    sid = str(unit_id or "").strip()
    if (not sid) or sid.startswith("chunk::"):
        return None
    if "::" not in sid:
        return None
    title, sent_idx_raw = sid.rsplit("::", 1)
    title = str(title or "").strip()
    if not title:
        return None
    try:
        sent_idx = int(sent_idx_raw)
    except Exception:
        return None
    return title, int(sent_idx)


def _parse_chunk_unit_id(unit_id):
    sid = str(unit_id or "").strip()
    if not sid.startswith("chunk::"):
        return None
    payload = sid[len("chunk::") :]
    if "::" not in payload:
        return None
    title, chunk_idx_raw = payload.rsplit("::", 1)
    title = str(title or "").strip()
    if not title:
        return None
    try:
        chunk_idx = int(chunk_idx_raw)
    except Exception:
        chunk_idx = 0
    return title, int(chunk_idx)


def _gold_support_items(sample):
    sentence_lookup = _sample_sentence_lookup(sample)
    out = []
    seen = set()
    for title, sent_idx in list(getattr(sample, "supporting_facts", []) or []):
        t = str(title or "").strip()
        if not t:
            continue
        idx = _safe_int(sent_idx, 0)
        sid = f"{t}::{idx}"
        if sid in seen:
            continue
        seen.add(sid)
        out.append(
            {
                "sentence_id": sid,
                "title": t,
                "sent_idx": int(idx),
                "sentence_text": str(sentence_lookup.get(sid, "") or ""),
            }
        )
    return out


def _predicted_units(unit_ids, unit_texts=None):
    source_ids = [str(sid or "").strip() for sid in list(unit_ids or [])]
    source_ids = [sid for sid in source_ids if sid]
    source_texts = [str(txt or "") for txt in list(unit_texts or [])]

    text_by_id = {}
    for idx, sid in enumerate(source_ids):
        if idx < len(source_texts):
            txt = str(source_texts[idx] or "").strip()
            if txt and sid not in text_by_id:
                text_by_id[sid] = txt

    ordered_ids = _ordered_unique(source_ids)
    units = []
    for rank, sid in enumerate(ordered_ids, start=1):
        sent_info = _parse_sentence_unit_id(sid)
        chunk_info = _parse_chunk_unit_id(sid)
        if sent_info is not None:
            unit_type = "sentence"
            title = sent_info[0]
            sentence_idx = int(sent_info[1])
            chunk_idx = None
        elif chunk_info is not None:
            unit_type = "chunk"
            title = chunk_info[0]
            sentence_idx = None
            chunk_idx = int(chunk_info[1])
        else:
            unit_type = "unknown"
            title = ""
            sentence_idx = None
            chunk_idx = None
        units.append(
            {
                "rank": int(rank),
                "unit_id": sid,
                "unit_type": unit_type,
                "title": title,
                "sentence_idx": sentence_idx,
                "chunk_idx": chunk_idx,
                "text": str(text_by_id.get(sid, "") or ""),
            }
        )
    return units


def supporting_fact_match_details(
    sample,
    unit_ids,
    unit_texts=None,
    graph_mode: Optional[str] = None,
    allow_title_fallback_when_no_text: Optional[bool] = None,
):
    mode = _normalize_graph_mode(graph_mode or "current_entity_graph")
    if allow_title_fallback_when_no_text is None:
        allow_title_fallback_when_no_text = bool(mode == "entity_chunk_graph")

    gold_items = _gold_support_items(sample)
    gold_by_sid = {item["sentence_id"]: item for item in gold_items}
    gold_by_title = {}
    for item in gold_items:
        gold_by_title.setdefault(str(item["title"]), []).append(item)

    sentence_lookup = _sample_sentence_lookup(sample)
    predicted_units = _predicted_units(unit_ids=unit_ids, unit_texts=unit_texts)

    matched_gold = set()
    matched_unit = set()
    for unit in predicted_units:
        unit_id = str(unit.get("unit_id", "") or "")
        unit_type = str(unit.get("unit_type", "unknown") or "unknown")
        title = str(unit.get("title", "") or "")
        sentence_idx = unit.get("sentence_idx")
        unit_text = str(unit.get("text", "") or "")
        if (not unit_text) and unit_type == "sentence" and title and sentence_idx is not None:
            unit_text = str(sentence_lookup.get(f"{title}::{int(sentence_idx)}", "") or "")
            unit["text"] = unit_text
        unit_text_norm = _normalize_text_for_match(unit_text)

        reasons = []
        matched_gold_ids = []
        same_title_gold = list(gold_by_title.get(title, []) or [])

        if unit_type == "sentence":
            if title and sentence_idx is not None:
                exact_sid = f"{title}::{int(sentence_idx)}"
                if exact_sid in gold_by_sid:
                    matched_gold_ids.append(exact_sid)
                    reasons.append("exact_sentence_id")
                elif same_title_gold:
                    reasons.append("same_title_sentence_index_mismatch")
                else:
                    reasons.append("sentence_title_not_in_gold")
            elif unit_id in gold_by_sid:
                matched_gold_ids.append(unit_id)
                reasons.append("exact_sentence_id_raw")
            else:
                reasons.append("invalid_sentence_id_format")
        elif unit_type == "chunk":
            if not same_title_gold:
                reasons.append("chunk_title_not_in_gold")
            else:
                for item in same_title_gold:
                    gold_text_norm = _normalize_text_for_match(item.get("sentence_text", ""))
                    if unit_text_norm and gold_text_norm and (gold_text_norm in unit_text_norm):
                        matched_gold_ids.append(str(item["sentence_id"]))
                if matched_gold_ids:
                    reasons.append("chunk_contains_gold_sentence")
                else:
                    if unit_text_norm:
                        reasons.append("chunk_text_no_gold_sentence_match")
                    else:
                        reasons.append("chunk_text_missing")
                        if allow_title_fallback_when_no_text:
                            matched_gold_ids.extend([str(item["sentence_id"]) for item in same_title_gold])
                            reasons.append("chunk_title_fallback_no_text")
        else:
            if unit_id in gold_by_sid:
                matched_gold_ids.append(unit_id)
                reasons.append("exact_sentence_id_raw")
            else:
                reasons.append("unrecognized_unit_id_format")

        matched_gold_ids = sorted(set(matched_gold_ids))
        unit["matched_gold_sentence_ids"] = list(matched_gold_ids)
        unit["match_reasons"] = list(reasons)
        unit["matched"] = bool(matched_gold_ids)
        if matched_gold_ids:
            matched_unit.add(unit_id)
            matched_gold.update(matched_gold_ids)

    missed_gold_ids = [item["sentence_id"] for item in gold_items if item["sentence_id"] not in matched_gold]
    miss_reason_map = {}
    predicted_by_title = {}
    for unit in predicted_units:
        predicted_by_title.setdefault(str(unit.get("title", "") or ""), []).append(unit)

    for gold_sid in missed_gold_ids:
        item = gold_by_sid.get(gold_sid, {}) or {}
        title = str(item.get("title", "") or "")
        same_title_units = list(predicted_by_title.get(title, []) or [])
        if not same_title_units:
            reason = "no_predicted_unit_with_same_title"
        elif any(str(u.get("unit_type", "")) == "sentence" for u in same_title_units):
            reason = "same_title_sentence_present_but_gold_sentence_missing"
        elif any("chunk_text_missing" in list(u.get("match_reasons", []) or []) for u in same_title_units):
            reason = "same_title_chunk_text_missing"
        elif any(str(u.get("unit_type", "")) == "chunk" for u in same_title_units):
            reason = "same_title_chunk_without_gold_sentence_containment"
        else:
            reason = "unmatched"
        miss_reason_map[gold_sid] = reason

    return {
        "graph_mode": mode,
        "gold_items": gold_items,
        "gold_sentence_ids": [item["sentence_id"] for item in gold_items],
        "gold_total": int(len(gold_items)),
        "predicted_units": predicted_units,
        "predicted_unit_total": int(len(predicted_units)),
        "matched_gold_sentence_ids": sorted(matched_gold),
        "matched_gold_total": int(len(matched_gold)),
        "matched_unit_ids": sorted(matched_unit),
        "matched_unit_total": int(len(matched_unit)),
        "missed_gold_sentence_ids": list(missed_gold_ids),
        "gold_miss_reasons": miss_reason_map,
    }


def _match_from_retrieval(sample, retrieval, k: Optional[int] = None):
    ids = [str(sid or "") for sid in list(getattr(retrieval, "selected_sentence_ids", []) or []) if str(sid or "")]
    texts = [str(txt or "") for txt in list(getattr(retrieval, "selected_sentences", []) or [])]
    if k is not None and int(k) > 0:
        ids = ids[: int(k)]
        texts = texts[: int(k)]
    return supporting_fact_match_details(
        sample=sample,
        unit_ids=ids,
        unit_texts=texts,
        graph_mode=_retrieval_graph_mode(retrieval),
    )


def supporting_fact_ids(sample, retrieval=None):
    _ = retrieval
    return {item["sentence_id"] for item in _gold_support_items(sample)}


def supporting_fact_recall(sample, retrieval):
    match = _match_from_retrieval(sample=sample, retrieval=retrieval, k=None)
    return safe_div(float(match.get("matched_gold_total", 0.0)), float(match.get("gold_total", 0.0)))


def supporting_fact_precision(sample, retrieval):
    match = _match_from_retrieval(sample=sample, retrieval=retrieval, k=None)
    return safe_div(float(match.get("matched_unit_total", 0.0)), float(match.get("predicted_unit_total", 0.0)))


def supporting_fact_recall_at_k(sample, retrieval, k: int):
    if k <= 0:
        return 0.0
    match = _match_from_retrieval(sample=sample, retrieval=retrieval, k=int(k))
    return safe_div(float(match.get("matched_gold_total", 0.0)), float(match.get("gold_total", 0.0)))


def supporting_fact_recall_at_ks(sample, retrieval, ks=DEFAULT_RECALL_KS):
    return {str(int(k)): supporting_fact_recall_at_k(sample, retrieval, int(k)) for k in ks}


def build_support_fact_debug_payload(sample, retrieval, rendered=None):
    graph_mode = _retrieval_graph_mode(retrieval)
    retrieval_match = _match_from_retrieval(sample=sample, retrieval=retrieval, k=None)

    rendered_match = None
    if rendered is not None:
        rendered_match = supporting_fact_match_details(
            sample=sample,
            unit_ids=list(getattr(rendered, "sentence_ids", []) or []),
            unit_texts=list(getattr(rendered, "sentences", []) or []),
            graph_mode=graph_mode,
        )

    payload = {
        "sample_id": str(getattr(sample, "qid", "") or ""),
        "question": str(getattr(sample, "question", "") or ""),
        "graph_mode": graph_mode,
        "gold_supporting_facts": list(retrieval_match.get("gold_items", []) or []),
        "retrieval": {
            "predicted_units": list(retrieval_match.get("predicted_units", []) or []),
            "retrieved_final_chunks": [
                item
                for item in list(retrieval_match.get("predicted_units", []) or [])
                if str(item.get("unit_type", "")) == "chunk"
            ],
            "matched_gold_sentence_ids": list(retrieval_match.get("matched_gold_sentence_ids", []) or []),
            "missed_gold_sentence_ids": list(retrieval_match.get("missed_gold_sentence_ids", []) or []),
            "gold_miss_reasons": dict(retrieval_match.get("gold_miss_reasons", {}) or {}),
        },
    }
    if rendered_match is not None:
        payload["rendered"] = {
            "predicted_units": list(rendered_match.get("predicted_units", []) or []),
            "rendered_chunks": [
                item
                for item in list(rendered_match.get("predicted_units", []) or [])
                if str(item.get("unit_type", "")) == "chunk"
            ],
            "matched_gold_sentence_ids": list(rendered_match.get("matched_gold_sentence_ids", []) or []),
            "missed_gold_sentence_ids": list(rendered_match.get("missed_gold_sentence_ids", []) or []),
            "gold_miss_reasons": dict(rendered_match.get("gold_miss_reasons", {}) or {}),
        }
    return payload


def aggregate_retrieval_metrics(rows, recall_ks=DEFAULT_RECALL_KS):
    recall = [float(r.get("supporting_fact_recall", 0.0)) for r in rows]
    precision = [float(r.get("supporting_fact_precision", 0.0)) for r in rows]
    latency = [float(r.get("retrieval_latency_ms", 0.0)) for r in rows]

    summary = {
        "n_samples": float(len(rows)),
        "supporting_fact_recall": mean_or_zero(recall),
        "supporting_fact_precision": mean_or_zero(precision),
        "retrieval_latency_ms": mean_or_zero(latency),
    }

    recall_at_k_summary = {}
    for k in recall_ks:
        key = str(int(k))
        vals = []
        for row in rows:
            per_row = row.get("recall_at_k", {}) or {}
            vals.append(float(per_row.get(key, 0.0)))
        agg = mean_or_zero(vals)
        recall_at_k_summary[key] = agg
        summary[f"recall_at_{key}"] = agg
        summary[f"R@{key}"] = agg
        summary[f"supporting_fact_recall_at_{key}"] = agg

    summary["supporting_fact_recall_at_k"] = recall_at_k_summary
    return summary
