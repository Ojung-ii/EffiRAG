from __future__ import annotations

from typing import Any, Mapping

from .utils import content_tokens


def _ordered_unique(values):
    seen = set()
    out = []
    for value in values or []:
        key = str(value or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _dedup_key(text: str) -> str:
    return _normalize_text(text).lower()


def _jaccard(left, right) -> float:
    lset = set(left or [])
    rset = set(right or [])
    union = lset.union(rset)
    if not union:
        return 1.0
    return float(len(lset.intersection(rset)) / len(union))


def _unit_title(unit_id: str) -> str:
    sid = str(unit_id or "")
    if sid.startswith("chunk::"):
        parts = sid.split("::")
        if len(parts) >= 3:
            return str(parts[1])
    if "::" in sid:
        return str(sid.rsplit("::", 1)[0])
    return sid


def _prompt_instruction_lines(prompt_variant: str | None) -> list[str]:
    variant = str(prompt_variant or "default").strip().lower()
    if variant in {"light_separator_copy_span_instruction", "copy_span_instruction"}:
        return [
            "You are a QA assistant.",
            "Use only the provided context.",
            "When the answer is stated in the context, copy the shortest exact answer span.",
            "Return only the final answer span.",
            "Do not output reasoning, explanations, or <think> tags.",
            "If the question is yes/no, output exactly yes or no.",
        ]
    if variant in {"light_separator_bridge_instruction", "bridge_instruction"}:
        return [
            "You are a QA assistant.",
            "Use only the provided context.",
            "Use source titles and evidence lines to resolve entity links.",
            "When the answer is stated in the context, copy the shortest exact answer span.",
            "Return only the final answer span.",
            "Do not output reasoning, explanations, or <think> tags.",
            "If the question is yes/no, output exactly yes or no.",
        ]
    return [
        "You are a QA assistant.",
        "Use only the provided context.",
        "Return only the final answer span.",
        "Do not output reasoning, explanations, or <think> tags.",
        "If the question is yes/no, output exactly yes or no.",
    ]


def estimate_actual_prompt_tokens(
    *,
    question_text: str,
    context_text: str,
    prompt_variant: str | None = "default",
) -> int:
    """Estimate the final QA prompt token count from the actual rendered string.

    The project uses multiple generation backends, so this helper intentionally
    keeps a backend-agnostic estimate. The character-based floor prevents the
    compact selector from under-counting dense punctuation/title wrappers.
    """

    parts = list(_prompt_instruction_lines(prompt_variant))
    parts.append(f"Question: {question_text}")
    parts.append(f"Context:\n{context_text}")
    parts.append("Final answer:")
    prompt = "\n".join(parts)
    lexical = int(len(content_tokens(prompt or "")))
    char_floor = int((len(prompt or "") + 3) // 4)
    return int(max(lexical, char_floor))


def render_selected_only_context(
    *,
    selected_ids: list[str],
    candidate_text_map: Mapping[str, str],
    question_text: str,
    prompt_variant: str | None = "default",
    min_render_topn: int = 6,
    max_render_topn: int = 24,
    max_prompt_tokens: int = 650,
    selector_prompt_tokens: int = 0,
    render_include_source_titles: str = "minimal",
    render_include_metadata: str = "minimal",
    render_deduplicate_selected_text: bool = True,
    render_enforce_actual_prompt_budget: bool = True,
) -> dict[str, Any]:
    """Render only selector-selected evidence with minimal global formatting."""

    selector_ids = _ordered_unique(selected_ids)
    max_n = max(1, int(max_render_topn))
    min_n = max(0, int(min_render_topn))
    max_prompt = max(1, int(max_prompt_tokens))
    include_titles = str(render_include_source_titles or "minimal").strip().lower()
    include_metadata = str(render_include_metadata or "minimal").strip().lower()
    selector_prompt = max(0, int(selector_prompt_tokens or 0))

    title_aliases: dict[str, str] = {}
    lines: list[str] = []
    rendered_ids: list[str] = []
    rendered_sentences: list[str] = []
    seen_text = set()
    skipped_duplicates = 0
    skipped_budget = 0
    early_stop_reason = ""

    def _title_label(sid: str) -> str:
        title = _unit_title(sid)
        if include_titles in {"none", "false", "off"}:
            return ""
        if include_titles == "minimal":
            if title not in title_aliases:
                title_aliases[title] = f"D{len(title_aliases) + 1}"
            return title_aliases[title]
        return title

    def _line_for(next_idx: int, sid: str, text: str) -> str:
        title = _title_label(sid)
        if title:
            return f"[{next_idx}|{title}] {text}"
        if include_metadata == "full":
            return f"[{next_idx}|{sid}] {text}"
        return f"[{next_idx}] {text}"

    for sid in selector_ids:
        if len(rendered_ids) >= max_n:
            early_stop_reason = early_stop_reason or "max_render_topn"
            break
        text = _normalize_text(str(candidate_text_map.get(sid, "") or ""))
        if not text:
            continue
        dkey = _dedup_key(text)
        if bool(render_deduplicate_selected_text) and dkey in seen_text:
            skipped_duplicates += 1
            continue

        line = _line_for(len(rendered_ids) + 1, sid, text)
        tentative_text = "\n".join(lines + [line])
        estimated = estimate_actual_prompt_tokens(
            question_text=question_text,
            context_text=tentative_text,
            prompt_variant=prompt_variant,
        )
        if (
            bool(render_enforce_actual_prompt_budget)
            and estimated > max_prompt
            and len(rendered_ids) >= min_n
        ):
            skipped_budget += 1
            early_stop_reason = "max_prompt_tokens"
            break

        lines.append(line)
        rendered_ids.append(sid)
        rendered_sentences.append(text)
        seen_text.add(dkey)

    rendered_text = "\n".join(lines)
    actual_estimate = estimate_actual_prompt_tokens(
        question_text=question_text,
        context_text=rendered_text,
        prompt_variant=prompt_variant,
    )
    selector_set = set(selector_ids)
    rendered_set = set(rendered_ids)
    extra_ids = sorted(rendered_set.difference(selector_set))
    missing_ids = [sid for sid in selector_ids if sid not in rendered_set]
    diagnostics = {
        "selector_selected_ids": list(selector_ids),
        "rendered_sentence_ids": list(rendered_ids),
        "final_prompt_sentence_ids": list(rendered_ids),
        "selector_prompt_tokens": int(selector_prompt),
        "estimated_actual_prompt_tokens": int(actual_estimate),
        "actual_prompt_tokens": 0,
        "extra_prompt_tokens_after_selector": int(max(0, actual_estimate - selector_prompt)),
        "selected_to_rendered_jaccard": float(_jaccard(selector_set, rendered_set)),
        "selected_to_prompt_jaccard": float(_jaccard(selector_set, rendered_set)),
        "extra_sentences_after_selector": int(len(extra_ids)),
        "num_extra_sentences_after_selector": int(len(extra_ids)),
        "missing_selected_sentence_count": int(len(missing_ids)),
        "skipped_duplicate_count": int(skipped_duplicates),
        "skipped_budget_count": int(skipped_budget),
        "num_selected": int(len(selector_ids)),
        "num_rendered": int(len(rendered_ids)),
        "early_stop_reason": str(early_stop_reason),
        "render_include_source_titles": str(include_titles),
        "render_include_metadata": str(include_metadata),
        "render_deduplicate_selected_text": bool(render_deduplicate_selected_text),
        "render_enforce_actual_prompt_budget": bool(render_enforce_actual_prompt_budget),
        "max_prompt_tokens": int(max_prompt),
        "min_render_topn": int(min_n),
        "max_render_topn": int(max_n),
    }
    return {
        "text": rendered_text,
        "sentences": list(rendered_sentences),
        "sentence_ids": list(rendered_ids),
        "diagnostics": diagnostics,
    }


_ANAPHORA_TERMS = {
    "he",
    "she",
    "it",
    "they",
    "them",
    "their",
    "his",
    "her",
    "its",
    "this",
    "that",
    "these",
    "those",
    "former",
    "latter",
}


def _has_named_entity_like_token(text: str) -> bool:
    for tok in str(text or "").split():
        t = str(tok).strip(".,;:!?()[]{}\"'")
        if not t:
            continue
        if any(ch.isupper() for ch in t[1:]):
            return True
        if t[:1].isupper() and len(t) > 1 and t.lower() not in {"the", "a", "an"}:
            return True
    return False


def _needs_context_expansion(text: str, *, bridge_score: float, path_score: float) -> bool:
    norm = _normalize_text(text)
    tokens = [x for x in norm.lower().split() if x]
    short_clause = len(tokens) <= 7
    pronoun_heavy = any(tok in _ANAPHORA_TERMS for tok in tokens)
    lacks_entity = not _has_named_entity_like_token(norm)
    bridge_like = float(bridge_score) >= 0.35 or float(path_score) >= 0.50
    return bool(short_clause or pronoun_heavy or lacks_entity or bridge_like)


def render_selected_centered_contextual_context(
    *,
    selected_ids: list[str],
    candidate_ids: list[str],
    candidate_text_map: Mapping[str, str],
    feature_map: Mapping[str, Mapping[str, Any]],
    score_map: Mapping[str, float],
    question_text: str,
    prompt_variant: str | None = "default",
    min_render_topn: int = 8,
    max_render_topn: int = 24,
    max_prompt_tokens: int = 850,
    selector_prompt_tokens: int = 0,
    render_include_source_titles: str = "minimal",
    render_include_metadata: str = "minimal",
    render_deduplicate_selected_text: bool = True,
    render_deduplicate_context_text: bool = True,
    render_enforce_actual_prompt_budget: bool = True,
    render_contextual_expansion_enabled: bool = True,
    render_conditional_neighbor_sentences: bool = True,
    render_bridge_context_enabled: bool = True,
    render_path_context_enabled: bool = True,
    max_neighbors_per_selected: int = 1,
    max_context_sentences_per_selected: int = 1,
    max_bridge_context_sentences: int = 2,
    max_path_context_sentences: int = 2,
) -> dict[str, Any]:
    """Render selected-core evidence plus minimal contextual expansions."""

    selector_ids = _ordered_unique(selected_ids)
    ordered_candidates = _ordered_unique(candidate_ids)
    rank_map = {sid: idx for idx, sid in enumerate(ordered_candidates)}
    max_n = max(1, int(max_render_topn))
    min_n = max(0, int(min_render_topn))
    max_prompt = max(1, int(max_prompt_tokens))
    include_titles = str(render_include_source_titles or "minimal").strip().lower()
    include_metadata = str(render_include_metadata or "minimal").strip().lower()
    selector_prompt = max(0, int(selector_prompt_tokens or 0))

    core_entries = []
    core_set = set(selector_ids)
    for sid in selector_ids:
        text = _normalize_text(str(candidate_text_map.get(sid, "") or ""))
        if not text:
            continue
        feat = dict(feature_map.get(sid, {}) or {})
        bridge_score = float(0.0)
        if bool(feat.get("is_connector_adjacent", False)):
            bridge_score += 0.65
        if bool(feat.get("is_support_candidate", False)):
            bridge_score += 0.25
        if bool(feat.get("is_main_candidate", False)):
            bridge_score += 0.10
        best_rank = int(feat.get("best_corridor_rank", 10**9) or 10**9)
        path_score = float(0.0)
        if best_rank < 10**9:
            path_score = max(0.0, 1.0 - float(best_rank - 1) / float(max(1, len(ordered_candidates))))
        core_entries.append(
            {
                "sid": sid,
                "text": text,
                "tag": "core",
                "score": float(score_map.get(sid, 0.0) or 0.0),
                "bridge_score": float(min(1.0, bridge_score)),
                "path_score": float(path_score),
                "rank": int(rank_map.get(sid, 10**9)),
            }
        )

    contextual_entries = []
    if bool(render_contextual_expansion_enabled):
        for core in core_entries:
            sid = str(core["sid"])
            text = str(core["text"])
            if not _needs_context_expansion(
                text,
                bridge_score=float(core.get("bridge_score", 0.0)),
                path_score=float(core.get("path_score", 0.0)),
            ):
                continue
            if not bool(render_conditional_neighbor_sentences):
                continue

            same_title_pool = []
            core_title = _unit_title(sid)
            sid_rank = int(rank_map.get(sid, 10**9))
            for cand in ordered_candidates:
                if cand in core_set:
                    continue
                if _unit_title(cand) != core_title:
                    continue
                ctext = _normalize_text(str(candidate_text_map.get(cand, "") or ""))
                if not ctext:
                    continue
                same_title_pool.append(
                    (
                        abs(int(rank_map.get(cand, 10**9)) - sid_rank),
                        int(rank_map.get(cand, 10**9)),
                        cand,
                        ctext,
                    )
                )
            same_title_pool.sort(key=lambda x: (x[0], x[1]))
            max_ctx = max(0, int(max_context_sentences_per_selected))
            max_neighbor = max(0, int(max_neighbors_per_selected))
            cap = min(max_ctx, max_neighbor) if max_ctx > 0 else max_neighbor
            for _, _, cand, ctext in same_title_pool[:cap]:
                contextual_entries.append(
                    {
                        "sid": cand,
                        "text": ctext,
                        "tag": "ctx",
                        "score": float(score_map.get(cand, 0.0) or 0.0),
                        "bridge_score": 0.0,
                        "path_score": 0.0,
                        "rank": int(rank_map.get(cand, 10**9)),
                    }
                )

    bridge_entries = []
    if bool(render_bridge_context_enabled):
        bridge_pool = []
        for cand in ordered_candidates:
            if cand in core_set:
                continue
            feat = dict(feature_map.get(cand, {}) or {})
            if not bool(feat.get("is_connector_adjacent", False)):
                continue
            text = _normalize_text(str(candidate_text_map.get(cand, "") or ""))
            if not text:
                continue
            bridge_pool.append(
                (
                    float(score_map.get(cand, 0.0) or 0.0),
                    -int(rank_map.get(cand, 10**9)),
                    cand,
                    text,
                )
            )
        bridge_pool.sort(reverse=True)
        for _, _, cand, text in bridge_pool[: max(0, int(max_bridge_context_sentences))]:
            bridge_entries.append(
                {
                    "sid": cand,
                    "text": text,
                    "tag": "bridge",
                    "score": float(score_map.get(cand, 0.0) or 0.0),
                    "bridge_score": 1.0,
                    "path_score": 0.0,
                    "rank": int(rank_map.get(cand, 10**9)),
                }
            )

    path_entries = []
    if bool(render_path_context_enabled):
        path_pool = []
        for cand in ordered_candidates:
            if cand in core_set:
                continue
            feat = dict(feature_map.get(cand, {}) or {})
            best_rank = int(feat.get("best_corridor_rank", 10**9) or 10**9)
            if best_rank >= 10**9:
                continue
            path_score = max(0.0, 1.0 - float(best_rank - 1) / float(max(1, len(ordered_candidates))))
            if path_score < 0.45:
                continue
            text = _normalize_text(str(candidate_text_map.get(cand, "") or ""))
            if not text:
                continue
            path_pool.append(
                (
                    float(path_score),
                    float(score_map.get(cand, 0.0) or 0.0),
                    -int(rank_map.get(cand, 10**9)),
                    cand,
                    text,
                )
            )
        path_pool.sort(reverse=True)
        for pscore, _, _, cand, text in path_pool[: max(0, int(max_path_context_sentences))]:
            path_entries.append(
                {
                    "sid": cand,
                    "text": text,
                    "tag": "path",
                    "score": float(score_map.get(cand, 0.0) or 0.0),
                    "bridge_score": 0.0,
                    "path_score": float(pscore),
                    "rank": int(rank_map.get(cand, 10**9)),
                }
            )

    combined = list(core_entries) + list(contextual_entries) + list(bridge_entries) + list(path_entries)
    combined.sort(
        key=lambda row: (
            0 if row.get("tag") == "core" else 1,
            int(row.get("rank", 10**9)),
            -float(row.get("score", 0.0)),
        )
    )

    title_aliases: dict[str, str] = {}
    rendered_entries = []
    seen_text = set()
    dedup_removed = 0
    budget_pruned = 0
    early_stop_reason = ""

    def _title_label(sid: str) -> str:
        title = _unit_title(sid)
        if include_titles in {"none", "false", "off"}:
            return ""
        if include_titles == "minimal":
            if title not in title_aliases:
                title_aliases[title] = f"D{len(title_aliases) + 1}"
            return title_aliases[title]
        return title

    def _line_for(next_idx: int, entry: Mapping[str, Any]) -> str:
        sid = str(entry.get("sid", "") or "")
        tag = str(entry.get("tag", "core") or "core")
        text = str(entry.get("text", "") or "")
        tag_suffix = ""
        if tag in {"ctx", "bridge", "path"}:
            tag_suffix = f"|{tag}"
        title = _title_label(sid)
        if title:
            return f"[{next_idx}{tag_suffix}|{title}] {text}"
        if include_metadata == "full":
            return f"[{next_idx}{tag_suffix}|{sid}] {text}"
        return f"[{next_idx}{tag_suffix}] {text}"

    for entry in combined:
        if len(rendered_entries) >= max_n:
            early_stop_reason = early_stop_reason or "max_render_topn"
            break
        sid = str(entry.get("sid", "") or "")
        text = _normalize_text(str(entry.get("text", "") or ""))
        if not sid or not text:
            continue

        should_dedup = bool(render_deduplicate_selected_text) if entry.get("tag") == "core" else bool(
            render_deduplicate_context_text
        )
        dkey = _dedup_key(text)
        if should_dedup and dkey in seen_text:
            dedup_removed += 1
            continue

        tentative_entries = rendered_entries + [dict(entry, sid=sid, text=text)]
        tentative_lines = [_line_for(idx + 1, row) for idx, row in enumerate(tentative_entries)]
        estimated = estimate_actual_prompt_tokens(
            question_text=question_text,
            context_text="\n".join(tentative_lines),
            prompt_variant=prompt_variant,
        )
        if bool(render_enforce_actual_prompt_budget) and estimated > max_prompt and len(rendered_entries) >= min_n:
            budget_pruned += 1
            early_stop_reason = "max_prompt_tokens"
            if entry.get("tag") == "core":
                # Core evidence is prioritized; keep it and prune later context.
                break
            continue

        rendered_entries.append(dict(entry, sid=sid, text=text))
        seen_text.add(dkey)

    # If we exceeded max_n with mostly non-core entries, enforce core priority.
    rendered_entries.sort(
        key=lambda row: (
            0 if row.get("tag") == "core" else 1,
            int(row.get("rank", 10**9)),
        )
    )
    rendered_entries = rendered_entries[:max_n]

    rendered_lines = [_line_for(idx + 1, row) for idx, row in enumerate(rendered_entries)]
    rendered_text = "\n".join(rendered_lines)
    rendered_ids = [str(row.get("sid", "") or "") for row in rendered_entries]
    rendered_sentences = [str(row.get("text", "") or "") for row in rendered_entries]
    actual_estimate = estimate_actual_prompt_tokens(
        question_text=question_text,
        context_text=rendered_text,
        prompt_variant=prompt_variant,
    )

    core_ids = [str(row.get("sid", "") or "") for row in core_entries]
    core_set = set(core_ids)
    rendered_set = set(rendered_ids)
    contextual_count = int(sum(1 for row in rendered_entries if row.get("tag") == "ctx"))
    bridge_count = int(sum(1 for row in rendered_entries if row.get("tag") == "bridge"))
    path_count = int(sum(1 for row in rendered_entries if row.get("tag") == "path"))
    diagnostics = {
        "selector_selected_ids": list(core_ids),
        "rendered_sentence_ids": list(rendered_ids),
        "final_prompt_sentence_ids": list(rendered_ids),
        "selector_prompt_tokens": int(selector_prompt),
        "estimated_actual_prompt_tokens": int(actual_estimate),
        "actual_prompt_tokens": 0,
        "extra_prompt_tokens_after_selector": int(max(0, actual_estimate - selector_prompt)),
        "selected_to_rendered_jaccard": float(_jaccard(core_set, rendered_set)),
        "selected_to_prompt_jaccard": float(_jaccard(core_set, rendered_set)),
        "extra_sentences_after_selector": int(max(0, len(rendered_ids) - len(core_ids))),
        "num_extra_sentences_after_selector": int(max(0, len(rendered_ids) - len(core_ids))),
        "num_candidates": int(len(ordered_candidates)),
        "num_selected_core": int(len(core_ids)),
        "num_context_sentences": int(contextual_count),
        "num_bridge_context_sentences": int(bridge_count),
        "num_path_context_sentences": int(path_count),
        "context_expansion_rate": float(
            0.0 if len(core_ids) <= 0 else float(contextual_count) / float(max(1, len(core_ids)))
        ),
        "bridge_context_rate": float(
            0.0 if len(rendered_ids) <= 0 else float(bridge_count) / float(max(1, len(rendered_ids)))
        ),
        "dedup_removed_count": int(dedup_removed),
        "budget_pruned_count": int(budget_pruned),
        "budget_pruned_rate": float(
            0.0 if len(rendered_ids) <= 0 else float(budget_pruned) / float(max(1, len(rendered_ids)))
        ),
        "num_selected": int(len(core_ids)),
        "num_rendered": int(len(rendered_ids)),
        "early_stop_reason": str(early_stop_reason),
        "render_include_source_titles": str(include_titles),
        "render_include_metadata": str(include_metadata),
        "render_deduplicate_selected_text": bool(render_deduplicate_selected_text),
        "render_deduplicate_context_text": bool(render_deduplicate_context_text),
        "render_enforce_actual_prompt_budget": bool(render_enforce_actual_prompt_budget),
        "max_prompt_tokens": int(max_prompt),
        "min_render_topn": int(min_n),
        "max_render_topn": int(max_n),
    }
    return {
        "text": rendered_text,
        "sentences": list(rendered_sentences),
        "sentence_ids": list(rendered_ids),
        "diagnostics": diagnostics,
    }
