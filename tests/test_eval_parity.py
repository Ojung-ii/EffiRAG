from __future__ import annotations

from effirag.eval.evaluator import QAEvaluator
from effirag.eval.metrics_hipporag2_parity import (
    qa_exact_match_parity,
    qa_f1_parity,
)
from effirag.metrics import supporting_fact_recall_at_k
from effirag.types import RetrievalResult, Sample, ContextDocument


def test_single_answer_exact_and_f1():
    assert qa_exact_match_parity("Paris", ["paris"]) == 1.0
    assert qa_f1_parity("Paris France", ["Paris"]) > 0.0


def test_multi_answer_em_order_independent():
    pred = "usa"
    gold_a = ["United States", "USA"]
    gold_b = list(reversed(gold_a))
    assert qa_exact_match_parity(pred, gold_a) == qa_exact_match_parity(pred, gold_b) == 1.0


def test_partial_match_f1():
    score = qa_f1_parity("Paris France", ["Paris"])
    assert abs(score - (2.0 / 3.0)) < 1.0e-8


def test_empty_and_no_overlap_cases():
    assert qa_f1_parity("", [""]) == 0.0
    assert qa_f1_parity("tokyo", ["paris"]) == 0.0
    assert qa_exact_match_parity("", [""]) == 1.0


def test_recall_at_k_compatibility():
    sample = Sample(
        qid="qid-1",
        question="q",
        answer="a",
        contexts=[ContextDocument(title="Doc", sentences=["s0", "s1", "s2"])],
        supporting_facts=[("Doc", 0), ("Doc", 2)],
    )
    retrieval = RetrievalResult(
        sample_id="qid-1",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc::2", "Doc::9", "Doc::0"],
        selected_sentences=["s2", "bad", "s0"],
    )
    assert supporting_fact_recall_at_k(sample, retrieval, 1) == 0.5
    assert supporting_fact_recall_at_k(sample, retrieval, 3) == 1.0


def test_batch_evaluation_parity_mode():
    evaluator = QAEvaluator(mode="hipporag2_parity")
    preds = ["paris", "usa", "seoul"]
    gold_batch = [["Paris"], ["United States", "USA"], ["Korea"]]
    out = evaluator.evaluate_batch(preds, gold_batch)
    assert out["count"] == 3.0
    assert out["ExactMatch"] == (1.0 + 1.0 + 0.0) / 3.0
    assert out["F1"] >= out["ExactMatch"]

