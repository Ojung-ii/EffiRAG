from __future__ import annotations

from collections import Counter
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


def _anchor_tokens(retrieval_result: Any) -> set[str]:
    tokens: set[str] = set()
    for anchor in list(getattr(retrieval_result, "anchors", []) or []):
        tokens.update(str(tok) for tok in content_tokens(str(anchor or "")))
    for anchor_result in list(getattr(retrieval_result, "anchor_results", []) or []):
        anchor = ""
        if isinstance(anchor_result, Mapping):
            anchor = str(anchor_result.get("anchor", "") or "")
        else:
            anchor = str(getattr(anchor_result, "anchor", "") or "")
        tokens.update(str(tok) for tok in content_tokens(anchor))
    return tokens


def _bridge_tokens_from_selected(
    *,
    selected_ids: List[str],
    text_lookup: Mapping[str, str],
    query_tokens: set[str],
    anchor_tokens: set[str],
) -> set[str]:
    token_counts: Counter[str] = Counter()
    for sid in selected_ids:
        text = str(text_lookup.get(str(sid or "").strip(), "") or "")
        if not text:
            continue
        sent_tokens = {
            str(tok)
            for tok in content_tokens(text)
            if str(tok) and len(str(tok)) >= 3 and str(tok) not in query_tokens and str(tok) not in anchor_tokens
        }
        for tok in sent_tokens:
            token_counts[tok] += 1
    return {tok for tok, count in token_counts.items() if int(count) >= 2}


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


def _parse_sentence_id(sentence_id: str) -> Tuple[str, int | None]:
    sid = str(sentence_id or "").strip()
    if not sid:
        return "", None
    parts = sid.split("::")
    if len(parts) < 2:
        return sid, None
    maybe_idx = parts[-1]
    if not str(maybe_idx).isdigit():
        return sid, None
    title = "::".join(parts[:-1])
    if title.startswith("chunk::"):
        title = title[len("chunk::") :]
    return title, int(maybe_idx)


def _support_span_score(
    *,
    sentence: str,
    query_tokens: set[str],
    anchor_tokens: set[str],
    bridge_tokens: set[str],
    use_query_signal: bool,
    use_anchor_signal: bool,
    use_bridge_signal: bool,
    use_view_stability_signal: bool,
    length_penalty_enabled: bool,
    rank_index: int,
    total_items: int,
) -> Tuple[float, Dict[str, float]]:
    sent_tokens = {str(tok) for tok in content_tokens(str(sentence or ""))}
    query_hit = float(len(sent_tokens.intersection(query_tokens))) if use_query_signal else 0.0
    anchor_hit = float(len(sent_tokens.intersection(anchor_tokens))) if use_anchor_signal else 0.0
    bridge_hit = float(len(sent_tokens.intersection(bridge_tokens))) if use_bridge_signal else 0.0
    view_signal = 0.0
    if use_view_stability_signal:
        view_signal = float(safe_div(float(max(0, total_items - rank_index)), float(max(1, total_items))))
    length_penalty = 0.0
    if length_penalty_enabled:
        length_penalty = float(0.12 * float(max(0, _token_count(sentence) - 12)))
    score = float((1.8 * query_hit) + (1.3 * anchor_hit) + (1.5 * bridge_hit) + (0.5 * view_signal) - length_penalty)
    return score, {
        "query_hit": float(query_hit),
        "anchor_hit": float(anchor_hit),
        "bridge_hit": float(bridge_hit),
        "view_signal": float(view_signal),
        "length_penalty": float(length_penalty),
    }


def _pick_support_span(
    *,
    text: str,
    unit_id: str,
    sample_lookup: Mapping[str, str],
    query_tokens: set[str],
    anchor_tokens: set[str],
    bridge_tokens: set[str],
    max_item_tokens: int | None,
    use_query_signal: bool,
    use_anchor_signal: bool,
    use_bridge_signal: bool,
    use_view_stability_signal: bool,
    length_penalty_enabled: bool,
    adjacent_sentence_enabled: bool,
    adjacent_sentence_max_count: int,
    rank_index: int,
    total_items: int,
) -> Tuple[str, Dict[str, float]]:
    sentence_candidates = _split_sentences(text)
    if not sentence_candidates:
        sentence_candidates = [_normalize_space(text)]

    scored: List[Tuple[float, int, str, Dict[str, float]]] = []
    for idx, sentence in enumerate(sentence_candidates):
        score, feats = _support_span_score(
            sentence=sentence,
            query_tokens=query_tokens,
            anchor_tokens=anchor_tokens,
            bridge_tokens=bridge_tokens,
            use_query_signal=use_query_signal,
            use_anchor_signal=use_anchor_signal,
            use_bridge_signal=use_bridge_signal,
            use_view_stability_signal=use_view_stability_signal,
            length_penalty_enabled=length_penalty_enabled,
            rank_index=rank_index,
            total_items=total_items,
        )
        scored.append((float(score), int(idx), sentence, feats))
    scored.sort(key=lambda item: (item[0], -_token_count(item[2])), reverse=True)

    best_score, best_idx, best_sentence, best_feats = scored[0]
    selected_sentences = [str(best_sentence)]
    adjacent_used = 0

    cap = None if max_item_tokens is None else int(max(1, int(max_item_tokens)))
    remaining_adj = int(max(0, adjacent_sentence_max_count))

    if bool(adjacent_sentence_enabled) and remaining_adj > 0:
        local_neighbors: List[str] = []
        if int(best_idx) - 1 >= 0:
            local_neighbors.append(sentence_candidates[int(best_idx) - 1])
        if int(best_idx) + 1 < len(sentence_candidates):
            local_neighbors.append(sentence_candidates[int(best_idx) + 1])

        # If the selected sentence is short, also inspect document-adjacent sentences by id.
        title, sent_idx = _parse_sentence_id(unit_id)
        if sent_idx is not None and title:
            for neighbor_idx in (sent_idx - 1, sent_idx + 1):
                if neighbor_idx < 0:
                    continue
                sid = f"{title}::{int(neighbor_idx)}"
                if sid in sample_lookup:
                    local_neighbors.append(str(sample_lookup[sid] or ""))

        neighbor_scored: List[Tuple[float, str, Dict[str, float]]] = []
        for neighbor in _ordered_unique(local_neighbors):
            n_score, n_feats = _support_span_score(
                sentence=neighbor,
                query_tokens=query_tokens,
                anchor_tokens=anchor_tokens,
                bridge_tokens=bridge_tokens,
                use_query_signal=use_query_signal,
                use_anchor_signal=use_anchor_signal,
                use_bridge_signal=use_bridge_signal,
                use_view_stability_signal=use_view_stability_signal,
                length_penalty_enabled=length_penalty_enabled,
                rank_index=rank_index,
                total_items=total_items,
            )
            neighbor_scored.append((float(n_score), neighbor, n_feats))

        neighbor_scored.sort(key=lambda item: (item[0], -_token_count(item[1])), reverse=True)
        for n_score, neighbor, n_feats in neighbor_scored:
            if remaining_adj <= 0:
                break
            if float(n_score) <= 0.0:
                continue
            candidate_text = _normalize_space(" ".join(selected_sentences + [neighbor]))
            if cap is not None and _token_count(candidate_text) > cap:
                continue
            # Only expand when the extra sentence improves signal coverage.
            signal_before = float(best_feats.get("query_hit", 0.0) + best_feats.get("anchor_hit", 0.0) + best_feats.get("bridge_hit", 0.0))
            signal_after = float(
                max(best_feats.get("query_hit", 0.0), n_feats.get("query_hit", 0.0))
                + max(best_feats.get("anchor_hit", 0.0), n_feats.get("anchor_hit", 0.0))
                + max(best_feats.get("bridge_hit", 0.0), n_feats.get("bridge_hit", 0.0))
            )
            short_core = _token_count(selected_sentences[0]) <= 8
            if signal_after > signal_before or short_core:
                selected_sentences.append(neighbor)
                adjacent_used += 1
                remaining_adj -= 1
                best_score += float(max(0.0, 0.5 * n_score))
                best_feats = {
                    "query_hit": float(max(best_feats.get("query_hit", 0.0), n_feats.get("query_hit", 0.0))),
                    "anchor_hit": float(max(best_feats.get("anchor_hit", 0.0), n_feats.get("anchor_hit", 0.0))),
                    "bridge_hit": float(max(best_feats.get("bridge_hit", 0.0), n_feats.get("bridge_hit", 0.0))),
                    "view_signal": float(best_feats.get("view_signal", 0.0)),
                    "length_penalty": float(best_feats.get("length_penalty", 0.0) + n_feats.get("length_penalty", 0.0)),
                }

    return _normalize_space(" ".join(selected_sentences)), {
        "support_span_score": float(best_score),
        "query_hit": float(best_feats.get("query_hit", 0.0)),
        "anchor_hit": float(best_feats.get("anchor_hit", 0.0)),
        "bridge_hit": float(best_feats.get("bridge_hit", 0.0)),
        "adjacent_used": float(adjacent_used),
    }


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
    sample_lookup: Mapping[str, str],
    query_tokens: set[str],
    anchor_tokens: set[str],
    bridge_tokens: set[str],
    item_rank: int,
    total_items: int,
    max_item_tokens: int | None,
    metadata_pruning: bool,
    minimal_span_fallback: bool,
    support_span_contract_enabled: bool,
    support_span_use_query_entity_signal: bool,
    support_span_use_anchor_entity_signal: bool,
    support_span_use_bridge_signal: bool,
    support_span_use_view_stability_signal: bool,
    support_span_length_penalty_enabled: bool,
    support_span_adjacent_sentence_enabled: bool,
    support_span_adjacent_sentence_max_count: int,
) -> Dict[str, Any]:
    raw = _normalize_space(text)
    metadata_removed_tokens = 0
    if metadata_pruning:
        raw, metadata_removed_tokens = _strip_known_metadata_prefixes(raw)

    support_diag: Dict[str, float] = {
        "support_span_score": 0.0,
        "query_hit": 0.0,
        "anchor_hit": 0.0,
        "bridge_hit": 0.0,
        "adjacent_used": 0.0,
    }
    if bool(support_span_contract_enabled):
        chosen, support_diag = _pick_support_span(
            text=raw,
            unit_id=unit_id,
            sample_lookup=sample_lookup,
            query_tokens=query_tokens,
            anchor_tokens=anchor_tokens,
            bridge_tokens=bridge_tokens,
            max_item_tokens=max_item_tokens,
            use_query_signal=bool(support_span_use_query_entity_signal),
            use_anchor_signal=bool(support_span_use_anchor_entity_signal),
            use_bridge_signal=bool(support_span_use_bridge_signal),
            use_view_stability_signal=bool(support_span_use_view_stability_signal),
            length_penalty_enabled=bool(support_span_length_penalty_enabled),
            adjacent_sentence_enabled=bool(support_span_adjacent_sentence_enabled),
            adjacent_sentence_max_count=int(max(0, support_span_adjacent_sentence_max_count)),
            rank_index=int(max(0, item_rank)),
            total_items=int(max(1, total_items)),
        )
    else:
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
        "support_span_score": float(support_diag.get("support_span_score", 0.0)),
        "query_hit": float(support_diag.get("query_hit", 0.0)),
        "anchor_hit": float(support_diag.get("anchor_hit", 0.0)),
        "bridge_hit": float(support_diag.get("bridge_hit", 0.0)),
        "adjacent_used": float(support_diag.get("adjacent_used", 0.0)),
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
    support_span_contract_enabled: bool = False,
    support_span_max_item_tokens: int | None = 48,
    support_span_metadata_pruning: bool = True,
    support_span_use_query_entity_signal: bool = True,
    support_span_use_anchor_entity_signal: bool = True,
    support_span_use_bridge_signal: bool = True,
    support_span_use_view_stability_signal: bool = True,
    support_span_length_penalty_enabled: bool = True,
    support_span_adjacent_sentence_enabled: bool = True,
    support_span_adjacent_sentence_max_count: int = 1,
    support_span_chunk_expansion_allowed: bool = False,
    support_span_preserve_selected_items: bool = True,
    support_span_log_diagnostics: bool = True,
) -> RenderedContext:
    render_contract_enabled = bool(sentence_contract_render_enabled) or bool(support_span_contract_enabled)
    if not render_contract_enabled:
        return rendered

    meta = dict((getattr(rendered, "metadata", {}) or {}))
    selected_ids = _ordered_unique(list(getattr(retrieval_result, "selected_sentence_ids", []) or []))
    rendered_ids = _ordered_unique(list(getattr(rendered, "sentence_ids", []) or []))
    text_lookup = _selected_text_lookup(sample, retrieval_result, rendered)
    sample_lookup = _sentence_lookup_from_sample(sample)

    query_tokens = _query_tokens(str(getattr(sample, "question", "") or ""))
    anchor_tokens = _anchor_tokens(retrieval_result)
    bridge_tokens = _bridge_tokens_from_selected(
        selected_ids=selected_ids,
        text_lookup=text_lookup,
        query_tokens=query_tokens,
        anchor_tokens=anchor_tokens,
    )

    source_ids: List[str] = []
    preserve_selected_items = _safe_bool(
        support_span_preserve_selected_items if support_span_contract_enabled else sentence_contract_preserve_selected_items,
        True,
    )
    chunk_expansion_allowed = _safe_bool(
        support_span_chunk_expansion_allowed if support_span_contract_enabled else sentence_contract_chunk_expansion_allowed,
        False,
    )

    if preserve_selected_items and selected_ids:
        source_ids = list(selected_ids)
    else:
        source_ids = list(rendered_ids or selected_ids)

    if (not chunk_expansion_allowed) and selected_ids:
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
    support_span_score_sum = 0.0
    query_hit_item_count = 0
    anchor_hit_item_count = 0
    bridge_hit_item_count = 0
    adjacent_used_item_count = 0
    support_span_item_count = 0

    total_items = int(max(1, len(source_ids)))
    for item_rank, sid in enumerate(source_ids):
        unit_id = str(sid or "").strip()
        if not unit_id:
            continue
        raw_text = text_lookup.get(unit_id, "")
        if not raw_text:
            raw_text = sample_lookup.get(unit_id, "")
        if not raw_text:
            raw_text = unit_id

        max_item_tokens = support_span_max_item_tokens if support_span_contract_enabled else sentence_contract_max_item_tokens
        metadata_pruning = support_span_metadata_pruning if support_span_contract_enabled else sentence_contract_metadata_pruning
        item = _contract_single_item(
            text=raw_text,
            unit_id=unit_id,
            question=str(getattr(sample, "question", "") or ""),
            sample_lookup=sample_lookup,
            query_tokens=query_tokens,
            anchor_tokens=anchor_tokens,
            bridge_tokens=bridge_tokens,
            item_rank=int(item_rank),
            total_items=int(total_items),
            max_item_tokens=max_item_tokens,
            metadata_pruning=bool(metadata_pruning),
            minimal_span_fallback=bool(sentence_contract_minimal_span_fallback),
            support_span_contract_enabled=bool(support_span_contract_enabled),
            support_span_use_query_entity_signal=bool(support_span_use_query_entity_signal),
            support_span_use_anchor_entity_signal=bool(support_span_use_anchor_entity_signal),
            support_span_use_bridge_signal=bool(support_span_use_bridge_signal),
            support_span_use_view_stability_signal=bool(support_span_use_view_stability_signal),
            support_span_length_penalty_enabled=bool(support_span_length_penalty_enabled),
            support_span_adjacent_sentence_enabled=bool(support_span_adjacent_sentence_enabled),
            support_span_adjacent_sentence_max_count=int(max(0, support_span_adjacent_sentence_max_count)),
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
        support_span_score_sum += float(item.get("support_span_score", 0.0) or 0.0)
        if float(item.get("query_hit", 0.0) or 0.0) > 0.0:
            query_hit_item_count += 1
        if float(item.get("anchor_hit", 0.0) or 0.0) > 0.0:
            anchor_hit_item_count += 1
        if float(item.get("bridge_hit", 0.0) or 0.0) > 0.0:
            bridge_hit_item_count += 1
        if float(item.get("adjacent_used", 0.0) or 0.0) > 0.0:
            adjacent_used_item_count += 1
        support_span_item_count += 1

    if preserve_selected_items:
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
    support_span_score_avg = float(safe_div(float(support_span_score_sum), float(max(1, support_span_item_count))))
    query_entity_hit_rate = float(safe_div(float(query_hit_item_count), float(max(1, support_span_item_count))))
    anchor_entity_hit_rate = float(safe_div(float(anchor_hit_item_count), float(max(1, support_span_item_count))))
    bridge_entity_hit_rate = float(safe_div(float(bridge_hit_item_count), float(max(1, support_span_item_count))))
    adjacent_sentence_used_rate = float(
        safe_div(float(adjacent_used_item_count), float(max(1, support_span_item_count)))
    )

    meta.update(
        {
            "sentence_contract_render_enabled": bool(sentence_contract_render_enabled),
            "support_span_contract_enabled": bool(support_span_contract_enabled),
            "sentence_contract_applied": True,
            "sentence_contract_max_item_tokens": (
                None if sentence_contract_max_item_tokens is None else int(sentence_contract_max_item_tokens)
            ),
            "support_span_max_item_tokens": (
                None if support_span_max_item_tokens is None else int(support_span_max_item_tokens)
            ),
            "sentence_contract_metadata_pruning": bool(sentence_contract_metadata_pruning),
            "support_span_metadata_pruning": bool(support_span_metadata_pruning),
            "sentence_contract_chunk_expansion_allowed": bool(sentence_contract_chunk_expansion_allowed),
            "support_span_chunk_expansion_allowed": bool(support_span_chunk_expansion_allowed),
            "sentence_contract_preserve_selected_items": bool(sentence_contract_preserve_selected_items),
            "support_span_preserve_selected_items": bool(support_span_preserve_selected_items),
            "sentence_contract_minimal_span_fallback": bool(sentence_contract_minimal_span_fallback),
            "sentence_contract_log_diagnostics": bool(sentence_contract_log_diagnostics),
            "support_span_use_query_entity_signal": bool(support_span_use_query_entity_signal),
            "support_span_use_anchor_entity_signal": bool(support_span_use_anchor_entity_signal),
            "support_span_use_bridge_signal": bool(support_span_use_bridge_signal),
            "support_span_use_view_stability_signal": bool(support_span_use_view_stability_signal),
            "support_span_length_penalty_enabled": bool(support_span_length_penalty_enabled),
            "support_span_adjacent_sentence_enabled": bool(support_span_adjacent_sentence_enabled),
            "support_span_adjacent_sentence_max_count": int(max(0, support_span_adjacent_sentence_max_count)),
            "support_span_log_diagnostics": bool(support_span_log_diagnostics),
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
            "support_span_score_avg": float(support_span_score_avg),
            "query_entity_hit_rate": float(query_entity_hit_rate),
            "anchor_entity_hit_rate": float(anchor_entity_hit_rate),
            "bridge_entity_hit_rate": float(bridge_entity_hit_rate),
            "adjacent_sentence_used_rate": float(adjacent_sentence_used_rate),
            "sentence_contract_item_token_counts": list(item_token_counts),
        }
    )
    render_diag = dict(meta.get("render_diagnostics", {}) or {})
    if bool(sentence_contract_log_diagnostics) or bool(support_span_log_diagnostics):
        render_diag.update(
            {
                "sentence_contract_render_enabled": True,
                "support_span_contract_enabled": bool(support_span_contract_enabled),
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
                "support_span_score_avg": float(support_span_score_avg),
                "query_entity_hit_rate": float(query_entity_hit_rate),
                "anchor_entity_hit_rate": float(anchor_entity_hit_rate),
                "bridge_entity_hit_rate": float(bridge_entity_hit_rate),
                "adjacent_sentence_used_rate": float(adjacent_sentence_used_rate),
            }
        )
    meta["render_diagnostics"] = render_diag
    rendered.metadata = meta
    return rendered
