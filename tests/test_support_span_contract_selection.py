import re

from effirag.sentence_contract_render import apply_sentence_level_evidence_contract
from effirag.types import ContextDocument, RenderedContext, RetrievalResult, Sample


def _tok(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", str(text or "")))


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
        "support_span_adjacent_sentence_enabled": True,
        "support_span_adjacent_sentence_max_count": 1,
        "support_span_chunk_expansion_allowed": False,
        "support_span_preserve_selected_items": True,
        "support_span_log_diagnostics": True,
    }


def test_support_like_sentence_is_selected_and_preserved():
    sample = Sample(
        qid="q-sel-1",
        question="Which lab studies alignment?",
        answer="OpenAI",
        contexts=[
            ContextDocument(
                title="DocA",
                sentences=[
                    "Generic background sentence without support.",
                    "OpenAI studies alignment and safety methods.",
                ],
            )
        ],
        supporting_facts=[("DocA", 1)],
    )
    raw = "Generic background sentence without support. OpenAI studies alignment and safety methods."
    retrieval = RetrievalResult(
        sample_id="q-sel-1",
        method="unit-test",
        anchors=["OpenAI"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocA::1"],
        selected_sentences=[raw],
    )
    rendered = RenderedContext(
        sample_id="q-sel-1",
        method="unit-test",
        text=raw,
        sentences=[raw],
        sentence_ids=["DocA::1"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocA::1"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        **_base_kwargs(),
    )
    assert out.sentence_ids == ["DocA::1"]
    assert "OpenAI studies alignment" in out.sentences[0]


def test_support_span_respects_item_cap():
    sample = Sample(
        qid="q-sel-2",
        question="What shows alignment evidence?",
        answer="unused",
        contexts=[ContextDocument(title="DocB", sentences=["Alignment evidence is detailed and lengthy for token cap testing."])],
        supporting_facts=[("DocB", 0)],
    )
    raw = "Alignment evidence is detailed and lengthy for token cap testing with extra words for truncation."
    retrieval = RetrievalResult(
        sample_id="q-sel-2",
        method="unit-test",
        anchors=["alignment"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocB::0"],
        selected_sentences=[raw],
    )
    rendered = RenderedContext(
        sample_id="q-sel-2",
        method="unit-test",
        text=raw,
        sentences=[raw],
        sentence_ids=["DocB::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocB::0"],
        metadata={},
    )

    kwargs = _base_kwargs()
    kwargs["support_span_max_item_tokens"] = 6
    out = apply_sentence_level_evidence_contract(sample=sample, retrieval_result=retrieval, rendered=rendered, **kwargs)
    assert _tok(out.sentences[0]) <= 6


def test_support_span_no_cap_ablation_disables_hard_cap():
    sample = Sample(
        qid="q-sel-3",
        question="What alignment statement is present?",
        answer="unused",
        contexts=[ContextDocument(title="DocC", sentences=["Alignment " + "token " * 30])],
        supporting_facts=[("DocC", 0)],
    )
    raw = "Alignment " + "token " * 30
    retrieval = RetrievalResult(
        sample_id="q-sel-3",
        method="unit-test",
        anchors=["alignment"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocC::0"],
        selected_sentences=[raw],
    )
    rendered = RenderedContext(
        sample_id="q-sel-3",
        method="unit-test",
        text=raw,
        sentences=[raw],
        sentence_ids=["DocC::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocC::0"],
        metadata={},
    )

    kwargs = _base_kwargs()
    kwargs["support_span_max_item_tokens"] = None
    out = apply_sentence_level_evidence_contract(sample=sample, retrieval_result=retrieval, rendered=rendered, **kwargs)
    assert _tok(out.sentences[0]) > 20
    assert out.metadata.get("support_span_max_item_tokens") is None


def test_adjacent_sentence_rule_triggers_only_when_needed():
    sample = Sample(
        qid="q-sel-4",
        question="Which short anchor is connected by bridgebeta?",
        answer="unused",
        contexts=[
            ContextDocument(
                title="DocD",
                sentences=[
                    "AnchorX.",
                    "BridgeBeta connects AnchorX to the next clue.",
                    "Irrelevant trailing note.",
                ],
            )
        ],
        supporting_facts=[("DocD", 0), ("DocD", 1)],
    )
    raw = "AnchorX. BridgeBeta connects AnchorX to the next clue."
    retrieval = RetrievalResult(
        sample_id="q-sel-4",
        method="unit-test",
        anchors=["AnchorX"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocD::0", "DocD::1"],
        selected_sentences=[raw, "BridgeBeta connects AnchorX to the next clue."],
    )
    rendered = RenderedContext(
        sample_id="q-sel-4",
        method="unit-test",
        text=raw,
        sentences=[raw],
        sentence_ids=["DocD::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocD::0", "DocD::1"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        **_base_kwargs(),
    )
    assert float(out.metadata.get("adjacent_sentence_used_rate", 0.0)) > 0.0

    sample2 = Sample(
        qid="q-sel-5",
        question="What bridgebeta clue mentions anchorx?",
        answer="unused",
        contexts=[ContextDocument(title="DocE", sentences=["BridgeBeta explains AnchorX with sufficient detail."])],
        supporting_facts=[("DocE", 0)],
    )
    raw2 = "BridgeBeta explains AnchorX with sufficient detail."
    retrieval2 = RetrievalResult(
        sample_id="q-sel-5",
        method="unit-test",
        anchors=["AnchorX"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocE::0"],
        selected_sentences=[raw2],
    )
    rendered2 = RenderedContext(
        sample_id="q-sel-5",
        method="unit-test",
        text=raw2,
        sentences=[raw2],
        sentence_ids=["DocE::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocE::0"],
        metadata={},
    )
    out2 = apply_sentence_level_evidence_contract(
        sample=sample2,
        retrieval_result=retrieval2,
        rendered=rendered2,
        **_base_kwargs(),
    )
    assert float(out2.metadata.get("adjacent_sentence_used_rate", 0.0)) == 0.0


def test_selected_evidence_items_are_not_dropped():
    sample = Sample(
        qid="q-sel-6",
        question="What two facts are selected?",
        answer="unused",
        contexts=[
            ContextDocument(
                title="DocF",
                sentences=[
                    "Fact one is relevant.",
                    "Fact two is also relevant.",
                ],
            )
        ],
        supporting_facts=[("DocF", 0), ("DocF", 1)],
    )
    retrieval = RetrievalResult(
        sample_id="q-sel-6",
        method="unit-test",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocF::0", "DocF::1"],
        selected_sentences=["Fact one is relevant.", "Fact two is also relevant."],
    )
    rendered = RenderedContext(
        sample_id="q-sel-6",
        method="unit-test",
        text="Fact one is relevant.",
        sentences=["Fact one is relevant."],
        sentence_ids=["DocF::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocF::0", "DocF::1"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        **_base_kwargs(),
    )
    assert out.sentence_ids == ["DocF::0", "DocF::1"]
    assert float(out.metadata.get("render_drop_rate", 1.0)) == 0.0
