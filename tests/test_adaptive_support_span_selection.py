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
        "support_span_contract_enabled": False,
        "adaptive_support_span_enabled": True,
        "adaptive_support_span_hard_cap_enabled": False,
        "adaptive_support_span_max_item_tokens": 40,
        "adaptive_support_span_soft_length_penalty_enabled": True,
        "adaptive_support_span_use_query_entity_signal": True,
        "adaptive_support_span_use_anchor_entity_signal": True,
        "adaptive_support_span_use_bridge_signal": True,
        "adaptive_support_span_bridge_mode": "conditional",
        "adaptive_support_span_metadata_pruning": True,
        "adaptive_support_span_preserve_selected_items": True,
        "adaptive_support_span_log_diagnostics": True,
    }


def test_adaptive_support_span_selects_highest_scoring_contiguous_span_without_hard_cap():
    sample = Sample(
        qid="q-adapt-sel-1",
        question="Which organization works on alignment safety?",
        answer="unused",
        contexts=[
            ContextDocument(
                title="DocA",
                sentences=[
                    "Generic context with little relevance.",
                    "OpenAI is an organization.",
                    "It focuses on alignment safety research.",
                ],
            )
        ],
        supporting_facts=[("DocA", 1), ("DocA", 2)],
    )
    raw = (
        "Generic context with little relevance. "
        "OpenAI is an organization. "
        "It focuses on alignment safety research."
    )
    retrieval = RetrievalResult(
        sample_id="q-adapt-sel-1",
        method="unit-test",
        anchors=["openai"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocA::1"],
        selected_sentences=[raw],
    )
    rendered = RenderedContext(
        sample_id="q-adapt-sel-1",
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
    assert "OpenAI is an organization" in out.sentences[0]
    assert "alignment safety research" in out.sentences[0]
    assert out.metadata.get("adaptive_support_span_enabled") is True


def test_adaptive_support_span_span40_ablation_applies_hard_cap():
    sample = Sample(
        qid="q-adapt-sel-2",
        question="What alignment statement appears?",
        answer="unused",
        contexts=[ContextDocument(title="DocB", sentences=["alignment " + ("token " * 60)])],
        supporting_facts=[("DocB", 0)],
    )
    raw = "alignment " + ("token " * 60)
    retrieval = RetrievalResult(
        sample_id="q-adapt-sel-2",
        method="unit-test",
        anchors=["alignment"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocB::0"],
        selected_sentences=[raw],
    )
    rendered = RenderedContext(
        sample_id="q-adapt-sel-2",
        method="unit-test",
        text=raw,
        sentences=[raw],
        sentence_ids=["DocB::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocB::0"],
        metadata={},
    )

    kwargs = _base_kwargs()
    kwargs["adaptive_support_span_hard_cap_enabled"] = True
    kwargs["adaptive_support_span_max_item_tokens"] = 40
    out = apply_sentence_level_evidence_contract(sample=sample, retrieval_result=retrieval, rendered=rendered, **kwargs)
    assert _tok(out.sentences[0]) <= 40
    assert out.metadata.get("adaptive_support_span_hard_cap_enabled") is True


def test_adaptive_support_span_preserves_selected_item_mapping():
    sample = Sample(
        qid="q-adapt-sel-3",
        question="What are the two selected facts?",
        answer="unused",
        contexts=[
            ContextDocument(
                title="DocC",
                sentences=["Fact one is relevant.", "Fact two is also relevant."],
            )
        ],
        supporting_facts=[("DocC", 0), ("DocC", 1)],
    )
    retrieval = RetrievalResult(
        sample_id="q-adapt-sel-3",
        method="unit-test",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocC::0", "DocC::1"],
        selected_sentences=["Fact one is relevant.", "Fact two is also relevant."],
    )
    rendered = RenderedContext(
        sample_id="q-adapt-sel-3",
        method="unit-test",
        text="Fact one is relevant.",
        sentences=["Fact one is relevant."],
        sentence_ids=["DocC::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocC::0", "DocC::1"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        **_base_kwargs(),
    )
    assert out.sentence_ids == ["DocC::0", "DocC::1"]
    assert float(out.metadata.get("render_drop_rate", 1.0)) == 0.0


def test_adaptive_selection_does_not_depend_on_answer_or_gold_support():
    contexts = [ContextDocument(title="DocD", sentences=["OpenAI studies alignment and safety."])]
    retrieval = RetrievalResult(
        sample_id="q-adapt-sel-4",
        method="unit-test",
        anchors=["OpenAI"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocD::0"],
        selected_sentences=["OpenAI studies alignment and safety."],
    )
    rendered = RenderedContext(
        sample_id="q-adapt-sel-4",
        method="unit-test",
        text="OpenAI studies alignment and safety.",
        sentences=["OpenAI studies alignment and safety."],
        sentence_ids=["DocD::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocD::0"],
        metadata={},
    )

    sample_a = Sample(
        qid="q-adapt-sel-4",
        question="Who studies alignment?",
        answer="OpenAI",
        contexts=contexts,
        supporting_facts=[("DocD", 0)],
    )
    sample_b = Sample(
        qid="q-adapt-sel-4",
        question="Who studies alignment?",
        answer="DifferentAnswer",
        contexts=contexts,
        supporting_facts=[],
    )

    out_a = apply_sentence_level_evidence_contract(
        sample=sample_a,
        retrieval_result=retrieval,
        rendered=rendered,
        **_base_kwargs(),
    )
    out_b = apply_sentence_level_evidence_contract(
        sample=sample_b,
        retrieval_result=retrieval,
        rendered=rendered,
        **_base_kwargs(),
    )
    assert out_a.sentences == out_b.sentences
