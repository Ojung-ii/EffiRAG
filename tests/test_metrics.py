from effirag.metrics import supporting_fact_recall
from effirag.qa_metrics import exact_match_score, token_f1_score
from effirag.types import ContextDocument, RetrievalResult, Sample


def test_qa_metrics():
    assert exact_match_score("Paris", "paris") == 1.0
    assert token_f1_score("Paris France", "Paris") > 0.0


def test_supporting_fact_recall_metric():
    sample = Sample(
        qid="m1",
        question="q",
        answer="a",
        contexts=[ContextDocument(title="Doc", sentences=["s0"])],
        supporting_facts=[("Doc", 0)],
    )
    retrieval = RetrievalResult(
        sample_id="m1",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc::0"],
        selected_sentences=["s0"],
    )

    assert supporting_fact_recall(sample, retrieval) == 1.0
