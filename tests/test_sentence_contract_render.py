import re

from effirag.sentence_contract_render import apply_sentence_level_evidence_contract
from effirag.types import ContextDocument, RenderedContext, RetrievalResult, Sample


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", str(text or "")))


def _build_sample() -> Sample:
    return Sample(
        qid="q1",
        question="Who founded OpenAI?",
        answer="Sam Altman",
        contexts=[
            ContextDocument(
                title="OpenAI",
                sentences=[
                    "OpenAI was founded in 2015 by a group including Sam Altman and Elon Musk.",
                    "The company focuses on artificial intelligence safety research.",
                ],
            )
        ],
        supporting_facts=[("OpenAI", 0)],
    )


def test_sentence_contract_enforces_sentence_level_and_item_cap():
    sample = _build_sample()
    long_chunk = (
        "[SUPPORT] OpenAI::0 - OpenAI was founded in 2015 by a group including Sam Altman and Elon Musk. "
        "The company focuses on artificial intelligence safety research and large language models."
    )
    retrieval = RetrievalResult(
        sample_id="q1",
        method="unit-test",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["chunk::OpenAI::0", "OpenAI::1"],
        selected_sentences=[long_chunk, sample.contexts[0].sentences[1]],
    )
    rendered = RenderedContext(
        sample_id="q1",
        method="unit-test",
        text=long_chunk,
        sentences=[long_chunk],
        sentence_ids=["chunk::OpenAI::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["chunk::OpenAI::0", "OpenAI::1"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        sentence_contract_render_enabled=True,
        sentence_contract_max_item_tokens=8,
        sentence_contract_metadata_pruning=True,
        sentence_contract_chunk_expansion_allowed=False,
        sentence_contract_preserve_selected_items=True,
        sentence_contract_minimal_span_fallback=True,
        sentence_contract_log_diagnostics=True,
    )

    assert out.sentence_ids == ["chunk::OpenAI::0", "OpenAI::1"]
    assert max(_token_count(sent) for sent in out.sentences) <= 8
    assert out.metadata["sentence_contract_render_enabled"] is True
    assert out.metadata["selected_to_rendered_preservation_rate"] == 1.0
    assert out.metadata["render_drop_rate"] == 0.0
    assert out.metadata["truncated_item_rate"] > 0.0


def test_sentence_contract_sets_fallback_truncation_flag_when_needed():
    sample = _build_sample()
    very_long_single_line = (
        "OpenAI foundation history mentions Sam Altman Elon Musk Greg Brockman Ilya Sutskever "
        "Wojciech Zaremba John Schulman and many additional details without punctuation markers"
    )
    retrieval = RetrievalResult(
        sample_id="q1",
        method="unit-test",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["chunk::OpenAI::fallback"],
        selected_sentences=[very_long_single_line],
    )
    rendered = RenderedContext(
        sample_id="q1",
        method="unit-test",
        text=very_long_single_line,
        sentences=[very_long_single_line],
        sentence_ids=["chunk::OpenAI::fallback"],
        truncated=False,
        retrieval_selected_sentence_ids=["chunk::OpenAI::fallback"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        sentence_contract_render_enabled=True,
        sentence_contract_max_item_tokens=10,
        sentence_contract_metadata_pruning=True,
        sentence_contract_chunk_expansion_allowed=False,
        sentence_contract_preserve_selected_items=True,
        sentence_contract_minimal_span_fallback=False,
        sentence_contract_log_diagnostics=True,
    )

    assert _token_count(out.sentences[0]) <= 10
    assert int(out.metadata.get("fallback_truncation_count", 0)) == 1
    assert float(out.metadata.get("truncated_item_rate", 0.0)) > 0.0
