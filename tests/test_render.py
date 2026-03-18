from effirag.render import render_context
from effirag.types import ContextDocument, RetrievalResult, Sample


def test_render_respects_max_context_sentences():
    sample = Sample(
        qid="r1",
        question="q",
        answer="a",
        contexts=[ContextDocument(title="Doc", sentences=["s0", "s1", "s2"])],
        supporting_facts=[],
    )

    retrieval = RetrievalResult(
        sample_id="r1",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc::0", "Doc::1", "Doc::2"],
        selected_sentences=["s0", "s1", "s2"],
    )

    rendered = render_context(sample, retrieval, max_context_sentences=2)
    assert len(rendered.sentences) == 2
    assert rendered.truncated is True
    assert "Doc::0" in rendered.text
