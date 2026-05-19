from effirag.compact_render import render_selected_centered_contextual_context
from effirag.render import _apply_top_slice_reorder, render_context
from effirag.types import ContextDocument, RetrievalResult, Sample


def _sample_and_retrieval():
    docs = [
        ContextDocument("Doc A", ["He won the award.", "Albert Einstein won the Nobel Prize in 1921."]),
        ContextDocument("Doc B", ["Bridge sentence linking the person and the event.", "Answer-bearing clue."]),
        ContextDocument("Doc C", ["Distractor sentence."]),
    ]
    sample = Sample(
        qid="q_contract",
        question="Who won the Nobel Prize in 1921?",
        answer="Albert Einstein",
        contexts=docs,
    )
    retrieval = RetrievalResult(
        sample_id="q_contract",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc A::0", "Doc B::1", "Doc B::0", "Doc C::0"],
        selected_sentences=[
            "He won the award.",
            "Answer-bearing clue.",
            "Bridge sentence linking the person and the event.",
            "Distractor sentence.",
        ],
        corridors=[
            {
                "corridor_id": "c1",
                "corridor_score": 1.0,
                "main_path_sentence_ids": ["Doc A::0", "Doc B::1"],
                "support_sentence_ids": ["Doc B::0"],
                "connector_adjacent_sentence_ids": ["Doc B::0"],
                "sentence_score_map": {
                    "Doc A::0": 0.9,
                    "Doc B::1": 0.8,
                    "Doc B::0": 0.7,
                    "Doc C::0": 0.1,
                },
            }
        ],
        diagnostics={},
    )
    return sample, retrieval


def test_baseline_unchanged_when_legacy_contract_disabled():
    sample, retrieval = _sample_and_retrieval()
    common = dict(
        max_context_sentences=8,
        render_mode="corridor_aware_flat",
        order_strategy="score",
        selector_aware_render_enabled=True,
        render_selected_only=True,
        render_include_source_titles="minimal",
        render_include_metadata="minimal",
        render_enforce_actual_prompt_budget=True,
        max_prompt_tokens=256,
    )
    base = render_context(sample, retrieval, **common)
    contract_off = render_context(
        sample,
        retrieval,
        legacy_contract_top_slice_reorder_enabled=False,
        legacy_contract_minimal_package_enabled=False,
        **common,
    )
    assert base.sentence_ids == contract_off.sentence_ids
    assert base.sentences == contract_off.sentences
    assert base.text == contract_off.text


def test_top_slice_reorder_changes_order_only_not_selected_set():
    selected = ["Doc A::0", "Doc B::0", "Doc C::0"]
    features = {
        "Doc A::0": {"query_overlap_score": 0.0, "is_support_candidate": False, "is_main_candidate": False},
        "Doc B::0": {"query_overlap_score": 1.0, "is_support_candidate": True, "is_main_candidate": True},
        "Doc C::0": {"query_overlap_score": 0.0, "is_support_candidate": False, "is_main_candidate": False},
    }
    score_at_pick = {"Doc A::0": 0.8, "Doc B::0": 0.4, "Doc C::0": 0.2}
    base_scores = dict(score_at_pick)
    retrieval_rank = {"Doc A::0": 0, "Doc B::0": 1, "Doc C::0": 2}
    reordered, diag = _apply_top_slice_reorder(
        selected_ids=selected,
        features=features,
        score_at_pick=score_at_pick,
        base_scores=base_scores,
        retrieval_rank=retrieval_rank,
        topk=2,
    )
    assert set(reordered) == set(selected)
    assert reordered != selected
    assert diag["applied"] is True


def test_minimal_package_rendering_is_bounded_and_budget_preserving():
    rendered = render_selected_centered_contextual_context(
        selected_ids=["Doc A::0"],
        candidate_ids=["Doc A::0", "Doc A::1", "Doc B::0"],
        candidate_text_map={
            "Doc A::0": "He won the award.",
            "Doc A::1": "Albert Einstein won the Nobel Prize in 1921.",
            "Doc B::0": "Bridge clue.",
        },
        feature_map={
            "Doc A::0": {"best_corridor_rank": 1, "is_main_candidate": True},
            "Doc A::1": {"best_corridor_rank": 1, "query_overlap_score": 1.0, "semantic_query_score": 0.7},
            "Doc B::0": {"best_corridor_rank": 1, "is_connector_adjacent": True},
        },
        score_map={"Doc A::0": 0.9, "Doc A::1": 0.8, "Doc B::0": 0.7},
        question_text="Who won the award?",
        min_render_topn=1,
        max_render_topn=4,
        max_prompt_tokens=80,
        render_contextual_expansion_enabled=True,
        render_conditional_neighbor_sentences=True,
        render_bridge_context_enabled=False,
        render_path_context_enabled=False,
        render_enforce_actual_prompt_budget=True,
        max_neighbors_per_selected=1,
        max_context_sentences_per_selected=1,
        max_total_extra_sentences=2,
    )
    diag = rendered["diagnostics"]
    assert int(diag["num_context_sentences"]) <= 1
    assert int(diag["num_extra_sentences_after_selector"]) <= 2
    assert int(diag["extra_sentence_token_count"]) >= 0
    assert len(list(diag["extra_sentence_ids"])) <= 2


def test_contract_skips_safely_when_sentence_metadata_missing():
    rendered = render_selected_centered_contextual_context(
        selected_ids=["chunk::DocZ::0"],
        candidate_ids=["chunk::DocZ::0", "chunk::DocZ::1"],
        candidate_text_map={
            "chunk::DocZ::0": "Compact chunk evidence.",
            "chunk::DocZ::1": "Neighbor chunk evidence.",
        },
        feature_map={"chunk::DocZ::0": {"best_corridor_rank": 1}, "chunk::DocZ::1": {"best_corridor_rank": 1}},
        score_map={"chunk::DocZ::0": 0.8, "chunk::DocZ::1": 0.7},
        question_text="What is the evidence?",
        min_render_topn=1,
        max_render_topn=3,
        max_prompt_tokens=120,
        render_contextual_expansion_enabled=True,
        render_conditional_neighbor_sentences=True,
        render_bridge_context_enabled=False,
        render_path_context_enabled=False,
        max_neighbors_per_selected=1,
        max_context_sentences_per_selected=1,
        max_total_extra_sentences=1,
    )
    diag = rendered["diagnostics"]
    assert isinstance(diag.get("extra_sentence_ids", []), list)
    assert int(diag.get("num_selected_core", 0)) == 1
