from effirag.sentence_contract_render import apply_sentence_level_evidence_contract
from effirag.types import ContextDocument, RenderedContext, RetrievalResult, Sample


def test_selected_to_rendered_preservation_and_id_mapping():
    sample = Sample(
        qid="q-preserve",
        question="Who founded the lab and what is the focus?",
        answer="Sam Altman; AI safety",
        contexts=[
            ContextDocument(
                title="OpenAI",
                sentences=[
                    "OpenAI was founded by Sam Altman and collaborators.",
                    "OpenAI focuses on AI safety research.",
                ],
            )
        ],
        supporting_facts=[("OpenAI", 0), ("OpenAI", 1)],
    )

    retrieval = RetrievalResult(
        sample_id="q-preserve",
        method="unit-test",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["OpenAI::0", "OpenAI::1"],
        selected_sentences=[
            "OpenAI was founded by Sam Altman and collaborators.",
            "OpenAI focuses on AI safety research with additional details that may be shortened.",
        ],
    )

    rendered = RenderedContext(
        sample_id="q-preserve",
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
        sentence_contract_render_enabled=True,
        sentence_contract_max_item_tokens=12,
        sentence_contract_metadata_pruning=True,
        sentence_contract_chunk_expansion_allowed=False,
        sentence_contract_preserve_selected_items=True,
        sentence_contract_minimal_span_fallback=True,
        sentence_contract_log_diagnostics=True,
    )

    assert out.sentence_ids == ["OpenAI::0", "OpenAI::1"]
    assert out.metadata["selected_to_rendered_preservation_rate"] == 1.0
    assert out.metadata["render_drop_rate"] == 0.0
    assert int(out.metadata["rendered_item_count"]) == 2
    assert int(out.metadata["selected_item_count"]) == 2
    assert all(len(str(text).strip()) > 0 for text in out.sentences)
