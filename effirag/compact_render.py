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
