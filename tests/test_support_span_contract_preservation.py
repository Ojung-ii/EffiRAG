from effirag.sentence_contract_render import apply_sentence_level_evidence_contract
from effirag.types import ContextDocument, RenderedContext, RetrievalResult, Sample


def _base_kwargs() -> dict:
    return {
        "sentence_contract_render_enabled": True,
        "sentence_contract_max_item_tokens": None,
        "sentence_contract_metadata_pruning": True,
        "sentence_contract_chunk_expansion_allowed": False,
        "sentence_contract_preserve_selected_items": True,
        "sentence_contract_minimal_span_fallback": True,
        "sentence_contract_log_diagnostics": True,
        "support_span_contract_enabled": True,
        "support_span_max_item_tokens": 48,
        "support_span_metadata_pruning": True,
        "support_span_use_query_entity_signal": True,
        "support_span_use_anchor_entity_signal": True,
        "support_span_use_bridge_signal": True,
        "support_span_use_view_stability_signal": True,
        "support_span_length_penalty_enabled": True,
        "support_span_adjacent_sentence_enabled": False,
        "support_span_adjacent_sentence_max_count": 0,
        "support_span_chunk_expansion_allowed": False,
        "support_span_preserve_selected_items": True,
        "support_span_log_diagnostics": True,
    }


def test_support_span_selected_to_rendered_mapping_is_preserved():
    sample = Sample(
        qid="q-preserve-1",
        question="Who founded the lab and what does it study?",
        answer="unused",
        contexts=[
            ContextDocument(
                title="OpenAI",
                sentences=[
                    "OpenAI was founded by Sam Altman and collaborators.",
                    "OpenAI studies AI safety.",
                ],
            )
        ],
        supporting_facts=[("OpenAI", 0), ("OpenAI", 1)],
    )
    retrieval = RetrievalResult(
        sample_id="q-preserve-1",
        method="unit-test",
        anchors=["OpenAI"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["OpenAI::0", "OpenAI::1"],
        selected_sentences=[
            "OpenAI was founded by Sam Altman and collaborators.",
            "OpenAI studies AI safety with additional long details that may be compressed.",
        ],
    )
    rendered = RenderedContext(
        sample_id="q-preserve-1",
        method="unit-test",
        text="OpenAI was founded by Sam Altman and collaborators.",
        sentences=["OpenAI was founded by Sam Altman and collaborators."],
        sentence_ids=["OpenAI::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["OpenAI::0", "OpenAI::1"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        **_base_kwargs(),
    )
    assert out.sentence_ids == ["OpenAI::0", "OpenAI::1"]
    assert float(out.metadata.get("selected_to_rendered_preservation_rate", 0.0)) == 1.0
    assert float(out.metadata.get("render_drop_rate", 1.0)) == 0.0


def test_rendered_text_is_subset_of_selected_text_without_chunk_expansion():
    sample = Sample(
        qid="q-preserve-2",
        question="Which phrase mentions bridge linkage?",
        answer="unused",
        contexts=[
            ContextDocument(
                title="DocSubset",
                sentences=[
                    "BridgeAlpha links source one to source two.",
                    "Additional context that should not be expanded when disabled.",
                ],
            )
        ],
        supporting_facts=[("DocSubset", 0)],
    )
    selected_text = (
        "Noise sentence with no overlap. "
        "BridgeAlpha links source one to source two."
    )
    retrieval = RetrievalResult(
        sample_id="q-preserve-2",
        method="unit-test",
        anchors=["BridgeAlpha"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocSubset::0"],
        selected_sentences=[selected_text],
    )
    rendered = RenderedContext(
        sample_id="q-preserve-2",
        method="unit-test",
        text=selected_text,
        sentences=[selected_text],
        sentence_ids=["DocSubset::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocSubset::0"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        **_base_kwargs(),
    )
    rendered_text = str(out.sentences[0])
    assert rendered_text in selected_text
    assert rendered_text != selected_text
    assert out.metadata.get("support_span_chunk_expansion_allowed") is False
