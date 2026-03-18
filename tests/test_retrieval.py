from effirag.baselines import run_naive_graphrag
from effirag.config import RetrievalConfig
from effirag.metrics import supporting_fact_recall
from effirag.retrieval import _personalized_pagerank, run_effirag
from effirag.types import ContextDocument, Sample


def _sample():
    return Sample(
        qid="t1",
        question="What city is the capital of France?",
        answer="Paris",
        contexts=[
            ContextDocument(
                title="Paris",
                sentences=[
                    "Paris is the capital of France.",
                    "Paris is known as the City of Light.",
                ],
            ),
            ContextDocument(
                title="France",
                sentences=["France is in Europe."],
            ),
        ],
        supporting_facts=[("Paris", 0)],
    )


def test_effirag_and_baseline_schema_compatible():
    sample = _sample()
    cfg = RetrievalConfig(samples_per_anchor=3, max_anchors=3, seed_k=2, candidate_top_t=8)

    eff = run_effirag(sample, cfg)
    base = run_naive_graphrag(sample, cfg)

    assert set(eff.__dict__.keys()) == set(base.__dict__.keys())
    assert eff.method == "effirag"
    assert base.method == "naive_graphrag"
    assert isinstance(eff.selected_sentence_ids, list)
    assert isinstance(base.selected_sentence_ids, list)


def test_supporting_fact_recall_in_range():
    sample = _sample()
    cfg = RetrievalConfig(samples_per_anchor=2)
    result = run_effirag(sample, cfg)
    recall = supporting_fact_recall(sample, result)
    assert 0.0 <= recall <= 1.0


def test_personalized_pagerank_spreads_mass_to_neighbors():
    import networkx as nx

    g = nx.Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")

    scores = _personalized_pagerank(g, source="a", alpha=0.15)
    assert scores["a"] > 0.0
    assert scores["b"] > 0.0
    assert scores["c"] > 0.0
