from effirag.dynamic_compact_selector import select_dynamic_compact_evidence


def _candidate(cid, text, *, score=0.5, bridge=0.0, path=0.0, rank=1, source="doc"):
    return {
        "candidate_id": cid,
        "text": text,
        "semantic_score": score,
        "bridge_score": bridge,
        "path_score": path,
        "rank": rank,
        "source_doc_id": source,
        "entity_ids": [source],
        "token_count": max(1, len(text.split())),
    }


def test_dynamic_selector_is_deterministic_and_respects_bounds():
    candidates = [
        _candidate("a", "alpha bridge clue one", score=0.8, bridge=0.7, rank=1, source="A"),
        _candidate("b", "beta answer clue two", score=0.7, rank=2, source="B"),
        _candidate("c", "gamma extra clue three", score=0.6, rank=3, source="C"),
        _candidate("d", "delta extra clue four", score=0.5, rank=4, source="D"),
    ]

    first = select_dynamic_compact_evidence(
        candidates,
        question_text="alpha beta bridge",
        min_render_topn=2,
        max_render_topn=3,
        max_prompt_tokens=80,
    )
    second = select_dynamic_compact_evidence(
        candidates,
        question_text="alpha beta bridge",
        min_render_topn=2,
        max_render_topn=3,
        max_prompt_tokens=80,
    )

    assert first.selected_ids == second.selected_ids
    assert 2 <= len(first.selected_ids) <= 3
    assert first.diagnostics["prompt_tokens"] <= 80


def test_dynamic_selector_prefers_coverage_over_duplicate_text():
    candidates = [
        _candidate("dup1", "alpha alpha shared duplicate evidence", score=0.9, rank=1, source="A"),
        _candidate("dup2", "alpha alpha shared duplicate evidence", score=0.88, rank=2, source="A"),
        _candidate("new", "beta gamma fresh coverage evidence", score=0.7, rank=3, source="B"),
    ]

    result = select_dynamic_compact_evidence(
        candidates,
        question_text="alpha beta gamma",
        min_render_topn=2,
        max_render_topn=2,
        redundancy_penalty_enabled=True,
        max_prompt_tokens=80,
    )

    assert "new" in result.selected_ids
    assert not {"dup1", "dup2"}.issubset(set(result.selected_ids))
    assert result.diagnostics["redundancy_penalty_sum"] >= 0.0


def test_dynamic_selector_preserves_bridge_candidate_globally():
    candidates = [
        _candidate("plain1", "plain answer sentence", score=0.8, rank=1, source="A"),
        _candidate("bridge", "connector bridge sentence", score=0.4, bridge=0.9, path=0.8, rank=2, source="B"),
        _candidate("plain2", "another answer sentence", score=0.7, rank=3, source="C"),
    ]

    result = select_dynamic_compact_evidence(
        candidates,
        question_text="connector answer",
        min_render_topn=2,
        max_render_topn=2,
        bridge_preserve_enabled=True,
        bridge_score_threshold=0.35,
        max_prompt_tokens=80,
    )

    assert "bridge" in result.selected_ids
    assert result.diagnostics["bridge_preserved"] is True


def test_dynamic_selector_keeps_minimum_even_under_tight_budget():
    candidates = [
        _candidate("a", " ".join(["alpha"] * 40), score=0.9, rank=1, source="A"),
        _candidate("b", " ".join(["beta"] * 40), score=0.8, rank=2, source="B"),
        _candidate("c", " ".join(["gamma"] * 40), score=0.7, rank=3, source="C"),
    ]

    result = select_dynamic_compact_evidence(
        candidates,
        question_text="alpha beta gamma",
        min_render_topn=2,
        max_render_topn=3,
        max_prompt_tokens=50,
    )

    assert len(result.selected_ids) >= 2
    assert len(result.selected_ids) <= 3
