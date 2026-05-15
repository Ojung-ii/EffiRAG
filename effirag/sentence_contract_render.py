from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .types import RenderedContext
from .utils import content_tokens, safe_div


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
WORD_RE = re.compile(r"\S+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ROLE_PREFIX_RE = re.compile(r"^\[[A-Za-z]{1,16}\]\s*")
SOURCE_PREFIX_RE = re.compile(r"^[^:\n]{1,120}::\d+\s*[-:|]\s*")
META_PREFIX_RE = re.compile(
    r"^(?:title|source|chunk|doc|document|path|entity|entities|metadata)\s*[:=]\s*",
    flags=re.IGNORECASE,
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _token_count(text: str) -> int:
    return int(len(TOKEN_RE.findall(str(text or ""))))


def _query_tokens(question: str) -> set[str]:
    return {str(tok) for tok in content_tokens(str(question or ""))}


def _sentence_lookup_from_sample(sample: Any) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    contexts = list(getattr(sample, "contexts", []) or [])
    for doc in contexts:
        title = str(getattr(doc, "title", "") or "").strip()
        sentences = list(getattr(doc, "sentences", []) or [])
        for sent_idx, sentence in enumerate(sentences):
            sid = f"{title}::{int(sent_idx)}"
            text = _normalize_space(str(sentence or ""))
            if sid and text:
                lookup[sid] = text
    return lookup


def _selected_text_lookup(sample: Any, retrieval_result: Any, rendered: RenderedContext) -> Dict[str, str]:
    lookup: Dict[str, str] = {}

    selected_ids = list(getattr(retrieval_result, "selected_sentence_ids", []) or [])
    selected_texts = list(getattr(retrieval_result, "selected_sentences", []) or [])
    for sid, text in zip(selected_ids, selected_texts):
        sid = str(sid or "").strip()
        body = _normalize_space(str(text or ""))
        if sid and body:
            lookup[sid] = body

    diagnostics = dict((getattr(retrieval_result, "diagnostics", {}) or {}))
    selected_map = diagnostics.get("selected_text_map", {}) or {}
    if isinstance(selected_map, Mapping):
        for sid, text in selected_map.items():
            sid = str(sid or "").strip()
            body = _normalize_space(str(text or ""))
            if sid and body and sid not in lookup:
                lookup[sid] = body

    for sid, text in zip(list(rendered.sentence_ids or []), list(rendered.sentences or [])):
        sid = str(sid or "").strip()
        body = _normalize_space(str(text or ""))
        if sid and body and sid not in lookup:
            lookup[sid] = body

    sample_lookup = _sentence_lookup_from_sample(sample)
    for sid, text in sample_lookup.items():
        if sid not in lookup:
            lookup[sid] = text
    return lookup


def _split_sentences(text: str) -> List[str]:
    raw = _normalize_space(text)
    if not raw:
        return []
    parts: List[str] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        sub = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(line) if piece.strip()]
        if sub:
            parts.extend(sub)
        else:
            parts.append(line)
    return parts or [raw]


def _sentence_score(sentence: str, query_toks: set[str]) -> float:
    sent_toks = {str(tok) for tok in content_tokens(sentence)}
    if not query_toks:
        return float(-_token_count(sentence))
    overlap = float(len(sent_toks.intersection(query_toks)))
    density = float(safe_div(overlap, float(max(1, len(sent_toks)))))
    shortness_bonus = float(1.0 / float(max(1, _token_count(sentence))))
    return float((2.0 * overlap) + density + shortness_bonus)


def _choose_sentence(text: str, question: str) -> str:
    candidates = _split_sentences(text)
    if not candidates:
        return _normalize_space(text)
    if len(candidates) == 1:
        return _normalize_space(candidates[0])
    query_toks = _query_tokens(question)
    scored = []
    for idx, candidate in enumerate(candidates):
        scored.append((float(_sentence_score(candidate, query_toks)), int(idx), candidate))
    scored.sort(key=lambda item: (item[0], -_token_count(item[2])), reverse=True)
    return _normalize_space(scored[0][2])


def _window_score(window_text: str, query_toks: set[str]) -> float:
    if not query_toks:
        return 0.0
    toks = {str(tok) for tok in content_tokens(window_text)}
    if not toks:
        return 0.0
    overlap = float(len(toks.intersection(query_toks)))
    return float(overlap + safe_div(overlap, float(max(1, len(toks)))))


def _extract_minimal_span(text: str, question: str, cap_tokens: int) -> str:
    raw = _normalize_space(text)
    cap = max(1, int(cap_tokens))
    if _token_count(raw) <= cap:
        return raw
    words = WORD_RE.findall(raw)
    if len(words) <= cap:
        return _normalize_space(" ".join(words))

    query_toks = _query_tokens(question)
    if not query_toks:
        return _normalize_space(" ".join(words[:cap]))

    best_start = 0
    best_score = -1.0
    max_start = max(0, len(words) - cap)
    for start in range(0, max_start + 1):
        window_text = " ".join(words[start : start + cap])
        score = _window_score(window_text, query_toks)
        if score > best_score:
            best_score = score
            best_start = start
    return _normalize_space(" ".join(words[best_start : best_start + cap]))


def _truncate_to_cap(text: str, cap_tokens: int) -> str:
    raw = _normalize_space(text)
    cap = max(1, int(cap_tokens))
    if _token_count(raw) <= cap:
        return raw
    words = WORD_RE.findall(raw)
    if not words:
        return raw
    return _normalize_space(" ".join(words[:cap]))


def _strip_known_metadata_prefixes(text: str) -> Tuple[str, int]:
    body = _normalize_space(text)
    removed = 0
    changed = True
    while changed:
        changed = False
        for pattern in (ROLE_PREFIX_RE, SOURCE_PREFIX_RE, META_PREFIX_RE):
            match = pattern.match(body)
            if not match:
                continue
            prefix = str(match.group(0) or "")
            removed += _token_count(prefix)
            body = _normalize_space(body[match.end() :])
            changed = True
    if " | " in body:
        left, right = body.split(" | ", 1)
        if _token_count(left) <= 6 and not re.search(r"[.!?]", left):
            removed += _token_count(left)
            body = _normalize_space(right)
    return body, int(removed)


def _estimate_metadata_tokens(text: str) -> int:
    body = str(text or "")
    total = 0
    role_match = ROLE_PREFIX_RE.match(body)
    if role_match:
        total += _token_count(role_match.group(0))
    source_match = SOURCE_PREFIX_RE.match(body)
    if source_match:
        total += _token_count(source_match.group(0))
    meta_match = META_PREFIX_RE.match(body)
    if meta_match:
        total += _token_count(meta_match.group(0))
    return int(total)


def _is_chunk_like(text: str, unit_id: str) -> bool:
    sid = str(unit_id or "").strip().lower()
    tok = _token_count(text)
    boundary_count = len(re.findall(r"[.!?]", str(text or "")))
    if sid.startswith("chunk::"):
        return True
    if tok > 45:
        return True
    if boundary_count >= 2:
        return True
    return False


def _contract_single_item(
    *,
    text: str,
    unit_id: str,
    question: str,
    max_item_tokens: int | None,
    metadata_pruning: bool,
    minimal_span_fallback: bool,
) -> Dict[str, Any]:
    raw = _normalize_space(text)
    metadata_removed_tokens = 0
    if metadata_pruning:
        raw, metadata_removed_tokens = _strip_known_metadata_prefixes(raw)

    chosen = _choose_sentence(raw, question)
    truncated = False
    minimal_span_used = False
    fallback_truncation_used = False

    cap = None if max_item_tokens is None else int(max(1, int(max_item_tokens)))
    if cap is not None and _token_count(chosen) > cap:
        if bool(minimal_span_fallback):
            minimal = _extract_minimal_span(chosen, question, cap)
            if minimal:
                chosen = _normalize_space(minimal)
                minimal_span_used = True
        if _token_count(chosen) > cap:
            chosen = _truncate_to_cap(chosen, cap)
            fallback_truncation_used = True
        truncated = True

    chunk_like = _is_chunk_like(chosen, unit_id)
    return {
        "text": _normalize_space(chosen),
        "token_count": int(_token_count(chosen)),
        "chunk_like": bool(chunk_like),
        "sentence_level": bool(not chunk_like),
        "truncated": bool(truncated),
        "minimal_span_used": bool(minimal_span_used),
        "fallback_truncation_used": bool(fallback_truncation_used),
        "metadata_removed_tokens": int(metadata_removed_tokens),
        "metadata_tokens": int(_estimate_metadata_tokens(chosen)),
    }


def apply_sentence_level_evidence_contract(
    *,
    sample: Any,
    retrieval_result: Any,
    rendered: RenderedContext,
    sentence_contract_render_enabled: bool = False,
    sentence_contract_max_item_tokens: int | None = None,
    sentence_contract_metadata_pruning: bool = True,
    sentence_contract_chunk_expansion_allowed: bool = False,
    sentence_contract_preserve_selected_items: bool = True,
    sentence_contract_minimal_span_fallback: bool = True,
    sentence_contract_log_diagnostics: bool = True,
) -> RenderedContext:
    if not bool(sentence_contract_render_enabled):
        return rendered

    meta = dict((getattr(rendered, "metadata", {}) or {}))
    selected_ids = _ordered_unique(list(getattr(retrieval_result, "selected_sentence_ids", []) or []))
    rendered_ids = _ordered_unique(list(getattr(rendered, "sentence_ids", []) or []))
    text_lookup = _selected_text_lookup(sample, retrieval_result, rendered)
    sample_lookup = _sentence_lookup_from_sample(sample)

    source_ids: List[str] = []
    if _safe_bool(sentence_contract_preserve_selected_items, True) and selected_ids:
        source_ids = list(selected_ids)
    else:
        source_ids = list(rendered_ids or selected_ids)

    if (not _safe_bool(sentence_contract_chunk_expansion_allowed, False)) and selected_ids:
        source_ids = list(selected_ids)

    out_ids: List[str] = []
    out_texts: List[str] = []

    truncated_count = 0
    minimal_span_count = 0
    fallback_truncation_count = 0
    chunk_like_count = 0
    sentence_level_count = 0
    metadata_tokens = 0
    metadata_removed_tokens = 0
    evidence_tokens = 0
    item_token_counts: List[int] = []

    for sid in source_ids:
        unit_id = str(sid or "").strip()
        if not unit_id:
            continue
        raw_text = text_lookup.get(unit_id, "")
        if not raw_text:
            raw_text = sample_lookup.get(unit_id, "")
        if not raw_text:
            raw_text = unit_id

        item = _contract_single_item(
            text=raw_text,
            unit_id=unit_id,
            question=str(getattr(sample, "question", "") or ""),
            max_item_tokens=sentence_contract_max_item_tokens,
            metadata_pruning=bool(sentence_contract_metadata_pruning),
            minimal_span_fallback=bool(sentence_contract_minimal_span_fallback),
        )
        text = str(item.get("text", "") or "").strip()
        if not text:
            text = str(raw_text or "").strip()
        if not text:
            continue

        out_ids.append(unit_id)
        out_texts.append(text)
        item_token_counts.append(int(item.get("token_count", _token_count(text))))
        evidence_tokens += int(item.get("token_count", _token_count(text)))
        metadata_tokens += int(item.get("metadata_tokens", 0))
        metadata_removed_tokens += int(item.get("metadata_removed_tokens", 0))
        if bool(item.get("chunk_like", False)):
            chunk_like_count += 1
        if bool(item.get("sentence_level", False)):
            sentence_level_count += 1
        if bool(item.get("truncated", False)):
            truncated_count += 1
        if bool(item.get("minimal_span_used", False)):
            minimal_span_count += 1
        if bool(item.get("fallback_truncation_used", False)):
            fallback_truncation_count += 1

    if _safe_bool(sentence_contract_preserve_selected_items, True):
        seen = set(out_ids)
        for sid in selected_ids:
            if sid in seen:
                continue
            fallback_text = _normalize_space(text_lookup.get(sid, sample_lookup.get(sid, sid)))
            out_ids.append(sid)
            out_texts.append(fallback_text)
            tcount = _token_count(fallback_text)
            item_token_counts.append(int(tcount))
            evidence_tokens += int(tcount)
            sentence_level_count += 1
            seen.add(sid)

    rendered.text = "\n".join(out_texts)
    rendered.sentences = list(out_texts)
    rendered.sentence_ids = list(out_ids)
    rendered.truncated = bool(rendered.truncated or truncated_count > 0)
    rendered.truncated_sentence_count = int(_safe_int(rendered.truncated_sentence_count, 0) + truncated_count)
    rendered.retrieval_selected_sentence_ids = list(selected_ids or list(rendered.retrieval_selected_sentence_ids or []))

    selected_set = set(selected_ids)
    rendered_set = set(out_ids)
    preserved_count = int(len(selected_set.intersection(rendered_set)))
    selected_count = int(len(selected_set))
    rendered_count = int(len(out_ids))
    preservation_rate = float(safe_div(float(preserved_count), float(max(1, selected_count))))
    render_drop_rate = float(max(0.0, 1.0 - preservation_rate))

    avg_item_tokens = float(safe_div(float(sum(item_token_counts)), float(max(1, rendered_count))))
    max_item_tokens = int(max(item_token_counts) if item_token_counts else 0)
    chunk_like_rate = float(safe_div(float(chunk_like_count), float(max(1, rendered_count))))
    sentence_level_rate = float(safe_div(float(sentence_level_count), float(max(1, rendered_count))))
    truncated_item_rate = float(safe_div(float(truncated_count), float(max(1, rendered_count))))
    minimal_span_fallback_rate = float(safe_div(float(minimal_span_count), float(max(1, rendered_count))))
    separator_tokens = int(max(0, rendered_count - 1))

    meta.update(
        {
            "sentence_contract_render_enabled": True,
            "sentence_contract_applied": True,
            "sentence_contract_max_item_tokens": (
                None if sentence_contract_max_item_tokens is None else int(sentence_contract_max_item_tokens)
            ),
            "sentence_contract_metadata_pruning": bool(sentence_contract_metadata_pruning),
            "sentence_contract_chunk_expansion_allowed": bool(sentence_contract_chunk_expansion_allowed),
            "sentence_contract_preserve_selected_items": bool(sentence_contract_preserve_selected_items),
            "sentence_contract_minimal_span_fallback": bool(sentence_contract_minimal_span_fallback),
            "sentence_contract_log_diagnostics": bool(sentence_contract_log_diagnostics),
            "selected_unit_type": "sentence",
            "rendered_item_count": int(rendered_count),
            "selected_item_count": int(selected_count),
            "selected_to_rendered_preserved_count": int(preserved_count),
            "selected_to_rendered_preservation_rate": float(preservation_rate),
            "render_drop_rate": float(render_drop_rate),
            "avg_item_tokens": float(avg_item_tokens),
            "max_item_tokens": int(max_item_tokens),
            "chunk_like_rate": float(chunk_like_rate),
            "sentence_level_rate": float(sentence_level_rate),
            "metadata_tokens": int(metadata_tokens),
            "metadata_removed_tokens": int(metadata_removed_tokens),
            "evidence_text_tokens": int(evidence_tokens),
            "separator_tokens": int(separator_tokens),
            "separator_line_count": int(separator_tokens),
            "truncated_item_count": int(truncated_count),
            "truncated_item_rate": float(truncated_item_rate),
            "minimal_span_fallback_count": int(minimal_span_count),
            "minimal_span_fallback_rate": float(minimal_span_fallback_rate),
            "fallback_truncation_count": int(fallback_truncation_count),
            "fallback_truncation_rate": float(
                safe_div(float(fallback_truncation_count), float(max(1, rendered_count)))
            ),
            "sentence_contract_item_token_counts": list(item_token_counts),
        }
    )
    render_diag = dict(meta.get("render_diagnostics", {}) or {})
    if bool(sentence_contract_log_diagnostics):
        render_diag.update(
            {
                "sentence_contract_render_enabled": True,
                "rendered_item_count": int(rendered_count),
                "selected_item_count": int(selected_count),
                "selected_to_rendered_preservation_rate": float(preservation_rate),
                "render_drop_rate": float(render_drop_rate),
                "avg_item_tokens": float(avg_item_tokens),
                "max_item_tokens": int(max_item_tokens),
                "chunk_like_rate": float(chunk_like_rate),
                "sentence_level_rate": float(sentence_level_rate),
                "metadata_tokens": int(metadata_tokens),
                "evidence_text_tokens": int(evidence_tokens),
                "separator_tokens": int(separator_tokens),
                "truncated_item_rate": float(truncated_item_rate),
                "minimal_span_fallback_rate": float(minimal_span_fallback_rate),
            }
        )
    meta["render_diagnostics"] = render_diag
    rendered.metadata = meta
    return rendered
