from effirag.render import render_corridor_aware_flat_context
from effirag.types import ContextDocument, RetrievalResult, Sample


def _sample_and_retrieval():
    docs = [
        ContextDocument("Doc A", ["Alpha bridge clue.", "Alpha duplicate clue."]),
        ContextDocument("Doc B", ["Beta answer clue.", "Beta supporting clue."]),
        ContextDocument("Doc C", ["Unused distractor clue."]),
    ]
    sample = Sample(
        qid="q1",
        question="What connects alpha and beta?",
        answer="beta",
        contexts=docs,
    )
    ids = ["Doc A::0", "Doc B::0", "Doc B::1", "Doc C::0"]
    texts = [
        "Alpha bridge clue.",
        "Beta answer clue.",
        "Beta supporting clue.",
        "Unused distractor clue.",
    ]
    retrieval = RetrievalResult(
        sample_id="q1",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=ids,
        selected_sentences=texts,
        corridors=[
            {
                "corridor_id": "c1",
                "corridor_score": 1.0,
                "main_path_sentence_ids": ["Doc A::0", "Doc B::0"],
                "support_sentence_ids": ["Doc B::1"],
                "connector_adjacent_sentence_ids": ["Doc A::0"],
                "sentence_score_map": {"Doc A::0": 1.0, "Doc B::0": 0.9, "Doc B::1": 0.8},
            }
        ],
        diagnostics={},
    )
    return sample, retrieval


def test_selector_aware_render_keeps_prompt_ids_aligned_with_selected_ids():
    sample, retrieval = _sample_and_retrieval()

    rendered = render_corridor_aware_flat_context(
        sample,
        retrieval,
        max_context_sentences=24,
        alpha=1.0,
        beta=0.35,
        gamma_main=0.45,
        delta_support=0.20,
        eta_connector=0.20,
        zeta_query=0.12,
        xi_locality=0.08,
        lambda_redundancy=0.25,
        top_corridors=0,
        max_sentences=24,
        reserve_top_corridor=False,
        order_strategy="score",
        dynamic_compact_selection_enabled=True,
        max_render_topn=4,
        min_render_topn=2,
        max_prompt_tokens=220,
        prompt_variant="light_separator_copy_span_instruction",
        selector_aware_render_enabled=True,
        render_selected_only=True,
        render_include_neighbor_sentences=False,
        render_include_corridor_headers=False,
        render_include_source_titles="minimal",
        render_include_metadata="minimal",
        render_deduplicate_selected_text=True,
        render_enforce_actual_prompt_budget=True,
    )

    diag = rendered.metadata["render_diagnostics"]
    assert rendered.metadata["selector_aware_render_enabled"] is True
    assert diag["selected_to_rendered_jaccard"] == 1.0
    assert diag["selected_to_prompt_jaccard"] == 1.0
    assert diag["extra_sentences_after_selector"] == 0
    assert diag["rendered_sentence_ids"] == rendered.sentence_ids
    assert diag["final_prompt_sentence_ids"] == rendered.sentence_ids
    assert set(rendered.sentence_ids).issubset(set(diag["selector_selected_ids"]))
    assert "Unused distractor clue." not in rendered.text or "Doc C::0" in diag["selector_selected_ids"]
