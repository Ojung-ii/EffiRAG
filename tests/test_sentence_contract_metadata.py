from effirag.sentence_contract_render import apply_sentence_level_evidence_contract
from effirag.types import ContextDocument, RenderedContext, RetrievalResult, Sample


def _sample() -> Sample:
    return Sample(
        qid="q-meta",
        question="What does the project study?",
        answer="AI safety",
        contexts=[
            ContextDocument(
                title="DocA",
                sentences=[
                    "The project studies AI safety and model alignment.",
                ],
            )
        ],
        supporting_facts=[("DocA", 0)],
    )


def _build_retrieval_with_prefixed_text(text: str) -> RetrievalResult:
    return RetrievalResult(
        sample_id="q-meta",
        method="unit-test",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocA::0"],
        selected_sentences=[text],
    )


def _build_rendered(text: str) -> RenderedContext:
    return RenderedContext(
        sample_id="q-meta",
        method="unit-test",
        text=text,
        sentences=[text],
        sentence_ids=["DocA::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocA::0"],
        metadata={},
    )


def test_metadata_pruning_removes_prefixes_and_preserves_evidence_text():
    sample = _sample()
    prefixed = "[SUPPORT] DocA::0 - The project studies AI safety and model alignment."
    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=_build_retrieval_with_prefixed_text(prefixed),
        rendered=_build_rendered(prefixed),
        sentence_contract_render_enabled=True,
        sentence_contract_max_item_tokens=32,
        sentence_contract_metadata_pruning=True,
        sentence_contract_chunk_expansion_allowed=False,
        sentence_contract_preserve_selected_items=True,
        sentence_contract_minimal_span_fallback=True,
        sentence_contract_log_diagnostics=True,
    )
    text = out.sentences[0]
    assert "[SUPPORT]" not in text
    assert "DocA::0 -" not in text
    assert "AI safety" in text


def test_metadata_on_profile_keeps_metadata():
    sample = _sample()
    prefixed = "[SUPPORT] DocA::0 - The project studies AI safety and model alignment."
    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=_build_retrieval_with_prefixed_text(prefixed),
        rendered=_build_rendered(prefixed),
        sentence_contract_render_enabled=True,
        sentence_contract_max_item_tokens=64,
        sentence_contract_metadata_pruning=False,
        sentence_contract_chunk_expansion_allowed=False,
        sentence_contract_preserve_selected_items=True,
        sentence_contract_minimal_span_fallback=True,
        sentence_contract_log_diagnostics=True,
    )
    text = out.sentences[0]
    assert "[SUPPORT]" in text
    assert "DocA::0 -" in text
