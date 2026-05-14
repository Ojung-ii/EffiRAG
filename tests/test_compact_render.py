from effirag.compact_render import estimate_actual_prompt_tokens, render_selected_only_context


def test_selected_only_render_excludes_non_selected_ids():
    candidate_text_map = {
        "Doc A::0": "Alpha bridge evidence.",
        "Doc B::0": "Beta answer evidence.",
        "Doc C::0": "Gamma unused evidence.",
    }

    rendered = render_selected_only_context(
        selected_ids=["Doc A::0", "Doc B::0"],
        candidate_text_map=candidate_text_map,
        question_text="What links alpha and beta?",
        prompt_variant="light_separator_copy_span_instruction",
        min_render_topn=1,
        max_render_topn=4,
        max_prompt_tokens=200,
    )

    assert rendered["sentence_ids"] == ["Doc A::0", "Doc B::0"]
    assert "Gamma unused evidence" not in rendered["text"]
    assert "Doc C::0" not in rendered["text"]
    assert rendered["diagnostics"]["extra_sentences_after_selector"] == 0


def test_selected_only_render_deduplicates_repeated_text():
    rendered = render_selected_only_context(
        selected_ids=["Doc A::0", "Doc A::1", "Doc B::0"],
        candidate_text_map={
            "Doc A::0": "Repeated evidence text.",
            "Doc A::1": "Repeated evidence text.",
            "Doc B::0": "Fresh coverage evidence.",
        },
        question_text="Which evidence is fresh?",
        min_render_topn=1,
        max_render_topn=4,
        max_prompt_tokens=200,
        render_deduplicate_selected_text=True,
    )

    assert rendered["sentence_ids"] == ["Doc A::0", "Doc B::0"]
    assert rendered["text"].count("Repeated evidence text.") == 1
    assert rendered["diagnostics"]["skipped_duplicate_count"] == 1


def test_selected_only_render_enforces_actual_prompt_budget_after_minimum():
    long_a = " ".join(["alpha"] * 80)
    long_b = " ".join(["beta"] * 80)
    long_c = " ".join(["gamma"] * 80)
    rendered = render_selected_only_context(
        selected_ids=["Doc A::0", "Doc B::0", "Doc C::0"],
        candidate_text_map={
            "Doc A::0": long_a,
            "Doc B::0": long_b,
            "Doc C::0": long_c,
        },
        question_text="What is alpha?",
        min_render_topn=1,
        max_render_topn=3,
        max_prompt_tokens=120,
        render_enforce_actual_prompt_budget=True,
    )

    assert len(rendered["sentence_ids"]) >= 1
    assert len(rendered["sentence_ids"]) < 3
    assert rendered["diagnostics"]["early_stop_reason"] == "max_prompt_tokens"


def test_actual_prompt_estimate_uses_full_prompt_wrapper():
    context = "[1|D1] Alpha evidence."
    context_only_tokens = len(context.split())
    estimated = estimate_actual_prompt_tokens(
        question_text="What is alpha?",
        context_text=context,
        prompt_variant="light_separator_copy_span_instruction",
    )

    assert estimated > context_only_tokens
