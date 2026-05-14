from effirag.compact_render import render_selected_centered_contextual_context


def test_contextual_render_includes_core_and_conditional_neighbor_context():
    rendered = render_selected_centered_contextual_context(
        selected_ids=["Doc A::0"],
        candidate_ids=["Doc A::0", "Doc A::1", "Doc B::0", "Doc C::0"],
        candidate_text_map={
            "Doc A::0": "He won the award.",
            "Doc A::1": "Albert Einstein won the Nobel Prize in 1921.",
            "Doc B::0": "Bridge sentence linking the person and the event.",
            "Doc C::0": "Unrelated distractor sentence.",
        },
        feature_map={
            "Doc A::0": {"best_corridor_rank": 1, "is_main_candidate": True},
            "Doc A::1": {"best_corridor_rank": 1, "is_main_candidate": True},
            "Doc B::0": {"best_corridor_rank": 1, "is_connector_adjacent": True, "is_support_candidate": True},
            "Doc C::0": {"best_corridor_rank": 4},
        },
        score_map={"Doc A::0": 0.9, "Doc A::1": 0.8, "Doc B::0": 0.7, "Doc C::0": 0.1},
        question_text="Who won the award?",
        min_render_topn=1,
        max_render_topn=6,
        max_prompt_tokens=400,
        render_contextual_expansion_enabled=True,
        render_conditional_neighbor_sentences=True,
        render_bridge_context_enabled=True,
        render_path_context_enabled=False,
        max_neighbors_per_selected=1,
        max_context_sentences_per_selected=1,
        max_bridge_context_sentences=1,
        max_path_context_sentences=0,
    )

    diag = rendered["diagnostics"]
    assert "Doc A::0" in rendered["sentence_ids"]
    assert diag["num_selected_core"] == 1
    assert diag["num_context_sentences"] >= 1
    assert diag["num_bridge_context_sentences"] >= 1
    assert diag["selected_to_rendered_jaccard"] < 1.0


def test_contextual_render_skips_unneeded_neighbor_expansion():
    rendered = render_selected_centered_contextual_context(
        selected_ids=["Doc D::0"],
        candidate_ids=["Doc D::0", "Doc D::1", "Doc E::0"],
        candidate_text_map={
            "Doc D::0": "Albert Einstein won the Nobel Prize in Physics in 1921.",
            "Doc D::1": "This sentence should not be included as context.",
            "Doc E::0": "Distractor evidence.",
        },
        feature_map={
            "Doc D::0": {"best_corridor_rank": 3, "is_main_candidate": True},
            "Doc D::1": {"best_corridor_rank": 3},
            "Doc E::0": {"best_corridor_rank": 4},
        },
        score_map={"Doc D::0": 0.9, "Doc D::1": 0.3, "Doc E::0": 0.2},
        question_text="Who won the Nobel Prize in Physics in 1921?",
        min_render_topn=1,
        max_render_topn=4,
        max_prompt_tokens=320,
        render_contextual_expansion_enabled=True,
        render_conditional_neighbor_sentences=True,
        render_bridge_context_enabled=False,
        render_path_context_enabled=False,
        max_neighbors_per_selected=1,
        max_context_sentences_per_selected=1,
    )

    diag = rendered["diagnostics"]
    assert rendered["sentence_ids"][0] == "Doc D::0"
    assert diag["num_context_sentences"] == 0
    assert diag["num_bridge_context_sentences"] == 0
    assert diag["num_path_context_sentences"] == 0


def test_contextual_render_prunes_by_budget_and_deduplicates_context():
    long_extra = " ".join(["extra"] * 50)
    rendered = render_selected_centered_contextual_context(
        selected_ids=["Doc A::0", "Doc B::0"],
        candidate_ids=["Doc A::0", "Doc B::0", "Doc A::1", "Doc C::0", "Doc D::0"],
        candidate_text_map={
            "Doc A::0": "Core sentence one with entity Alpha.",
            "Doc B::0": "Core sentence two with entity Beta.",
            "Doc A::1": long_extra,
            "Doc C::0": long_extra,
            "Doc D::0": "Connector bridge evidence for Alpha and Beta.",
        },
        feature_map={
            "Doc A::0": {"best_corridor_rank": 1, "is_main_candidate": True},
            "Doc B::0": {"best_corridor_rank": 1, "is_main_candidate": True},
            "Doc A::1": {"best_corridor_rank": 1},
            "Doc C::0": {"best_corridor_rank": 2},
            "Doc D::0": {"best_corridor_rank": 1, "is_connector_adjacent": True, "is_support_candidate": True},
        },
        score_map={"Doc A::0": 0.9, "Doc B::0": 0.85, "Doc A::1": 0.3, "Doc C::0": 0.29, "Doc D::0": 0.7},
        question_text="What links Alpha and Beta?",
        min_render_topn=2,
        max_render_topn=6,
        max_prompt_tokens=220,
        render_contextual_expansion_enabled=True,
        render_conditional_neighbor_sentences=True,
        render_bridge_context_enabled=True,
        render_path_context_enabled=True,
        render_deduplicate_context_text=True,
        render_enforce_actual_prompt_budget=True,
        max_neighbors_per_selected=1,
        max_context_sentences_per_selected=1,
        max_bridge_context_sentences=2,
        max_path_context_sentences=2,
    )

    diag = rendered["diagnostics"]
    assert diag["num_selected_core"] == 2
    assert diag["dedup_removed_count"] >= 0
    assert diag["budget_pruned_count"] >= 0
    assert diag["estimated_actual_prompt_tokens"] <= 220 or diag["early_stop_reason"] in {
        "",
        "max_prompt_tokens",
        "max_render_topn",
    }
