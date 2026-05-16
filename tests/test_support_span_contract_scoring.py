import re
import inspect

import effirag.sentence_contract_render as sentence_contract_render
from effirag.sentence_contract_render import apply_sentence_level_evidence_contract
from effirag.types import ContextDocument, RenderedContext, RetrievalResult, Sample


def _tok(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", str(text or "")))


def _sample(question: str) -> Sample:
    return Sample(
        qid="q-span-score",
        question=question,
        answer="unused-answer",
        contexts=[
            ContextDocument(
                title="DocA",
                sentences=[
                    "OpenAI studies alignment and safety.",
                    "Background sentence without relevant entities.",
                ],
            )
        ],
        supporting_facts=[("DocA", 0)],
    )


def test_support_span_prefers_query_entity_sentence():
    sample = _sample("Which organization studies alignment?")
    raw = (
        "Background sentence without overlap. "
        "OpenAI studies alignment and safety."
    )
    retrieval = RetrievalResult(
        sample_id="q-span-score",
        method="unit-test",
        anchors=["alignment", "openai"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocA::0"],
        selected_sentences=[raw],
    )
    rendered = RenderedContext(
        sample_id="q-span-score",
        method="unit-test",
        text=raw,
        sentences=[raw],
        sentence_ids=["DocA::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocA::0"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        sentence_contract_render_enabled=True,
        sentence_contract_max_item_tokens=48,
        sentence_contract_metadata_pruning=True,
        sentence_contract_chunk_expansion_allowed=False,
        sentence_contract_preserve_selected_items=True,
        sentence_contract_minimal_span_fallback=True,
        sentence_contract_log_diagnostics=True,
        support_span_contract_enabled=True,
        support_span_max_item_tokens=48,
        support_span_metadata_pruning=True,
        support_span_use_query_entity_signal=True,
        support_span_use_anchor_entity_signal=True,
        support_span_use_bridge_signal=True,
        support_span_use_view_stability_signal=True,
        support_span_length_penalty_enabled=True,
        support_span_adjacent_sentence_enabled=False,
        support_span_adjacent_sentence_max_count=0,
        support_span_chunk_expansion_allowed=False,
        support_span_preserve_selected_items=True,
        support_span_log_diagnostics=True,
    )

    assert "OpenAI studies alignment" in out.sentences[0]
    assert float(out.metadata.get("query_entity_hit_rate", 0.0)) > 0.0


def test_support_span_bridge_signal_helps_bridge_sentence_selection():
    sample = _sample("What connects the evidence chain?")
    bridge_text = "ConnectorAlpha links the two sources."
    target_text = "Irrelevant generic note. ConnectorAlpha appears as the bridge term."

    retrieval = RetrievalResult(
        sample_id="q-span-score",
        method="unit-test",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocA::0", "DocA::1"],
        selected_sentences=[bridge_text, target_text],
    )
    rendered = RenderedContext(
        sample_id="q-span-score",
        method="unit-test",
        text=target_text,
        sentences=[target_text],
        sentence_ids=["DocA::0", "DocA::1"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocA::0", "DocA::1"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        sentence_contract_render_enabled=True,
        sentence_contract_max_item_tokens=48,
        sentence_contract_metadata_pruning=True,
        sentence_contract_chunk_expansion_allowed=False,
        sentence_contract_preserve_selected_items=True,
        sentence_contract_minimal_span_fallback=True,
        sentence_contract_log_diagnostics=True,
        support_span_contract_enabled=True,
        support_span_max_item_tokens=48,
        support_span_metadata_pruning=True,
        support_span_use_query_entity_signal=False,
        support_span_use_anchor_entity_signal=False,
        support_span_use_bridge_signal=True,
        support_span_use_view_stability_signal=True,
        support_span_length_penalty_enabled=True,
        support_span_adjacent_sentence_enabled=False,
        support_span_adjacent_sentence_max_count=0,
        support_span_chunk_expansion_allowed=False,
        support_span_preserve_selected_items=True,
        support_span_log_diagnostics=True,
    )

    assert "ConnectorAlpha" in out.sentences[1]
    assert float(out.metadata.get("bridge_entity_hit_rate", 0.0)) > 0.0


def test_support_span_length_penalty_prefers_shorter_sentence_when_signal_ties():
    sample = _sample("alignment evidence")
    long = "alignment " + "verylongtoken " * 30
    short = "alignment concise proof"
    raw = f"{long}. {short}."

    retrieval = RetrievalResult(
        sample_id="q-span-score",
        method="unit-test",
        anchors=["alignment"],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["DocA::0"],
        selected_sentences=[raw],
    )
    rendered = RenderedContext(
        sample_id="q-span-score",
        method="unit-test",
        text=raw,
        sentences=[raw],
        sentence_ids=["DocA::0"],
        truncated=False,
        retrieval_selected_sentence_ids=["DocA::0"],
        metadata={},
    )

    out = apply_sentence_level_evidence_contract(
        sample=sample,
        retrieval_result=retrieval,
        rendered=rendered,
        sentence_contract_render_enabled=True,
        sentence_contract_max_item_tokens=48,
        sentence_contract_metadata_pruning=True,
        sentence_contract_chunk_expansion_allowed=False,
        sentence_contract_preserve_selected_items=True,
        sentence_contract_minimal_span_fallback=True,
        sentence_contract_log_diagnostics=True,
        support_span_contract_enabled=True,
        support_span_max_item_tokens=48,
        support_span_metadata_pruning=True,
        support_span_use_query_entity_signal=True,
        support_span_use_anchor_entity_signal=True,
        support_span_use_bridge_signal=False,
        support_span_use_view_stability_signal=True,
        support_span_length_penalty_enabled=True,
        support_span_adjacent_sentence_enabled=False,
        support_span_adjacent_sentence_max_count=0,
        support_span_chunk_expansion_allowed=False,
        support_span_preserve_selected_items=True,
        support_span_log_diagnostics=True,
    )

    assert "concise proof" in out.sentences[0]
    assert _tok(out.sentences[0]) <= 48


def test_support_span_scoring_does_not_reference_answer_gold_or_dataset_names():
    score_src = inspect.getsource(sentence_contract_render._support_span_score)
    pick_src = inspect.getsource(sentence_contract_render._pick_support_span)
    combined = (score_src + "\n" + pick_src).lower()

    banned_terms = [
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "popqa",
        "supporting_facts",
        "gold_support",
    ]
    for term in banned_terms:
        assert term not in combined

    score_params = set(inspect.signature(sentence_contract_render._support_span_score).parameters)
    pick_params = set(inspect.signature(sentence_contract_render._pick_support_span).parameters)
    assert "answer" not in score_params
    assert "answer" not in pick_params
