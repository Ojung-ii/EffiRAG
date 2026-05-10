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


def test_ppr_graph_struct_cache_no_leak_between_graph_objects():
    import networkx as nx
    import effirag.retrieval as retrieval_mod

    retrieval_mod._PPR_GRAPH_STRUCT_MEMO.clear()
    retrieval_mod._PPR_GRAPH_STRUCT_MEMO_FALLBACK.clear()

    g1 = nx.Graph()
    g1.add_edge("x", "y")
    s1 = _personalized_pagerank(g1, source="x", alpha=0.15)
    assert "x" in s1 and "y" in s1

    g2 = nx.Graph()
    g2.add_edge("a", "b")
    g2.add_edge("b", "c")
    s2 = _personalized_pagerank(g2, source="a", alpha=0.15)
    assert "a" in s2 and "b" in s2
    assert "x" not in s2 and "y" not in s2


def test_get_graph_struct_rejects_stale_fallback_entry():
    import weakref
    import networkx as nx
    import effirag.retrieval as retrieval_mod

    retrieval_mod._PPR_GRAPH_STRUCT_MEMO.clear()
    retrieval_mod._PPR_GRAPH_STRUCT_MEMO_FALLBACK.clear()

    g1 = nx.Graph()
    g1.add_edge("x", "y")
    _ = retrieval_mod._get_graph_struct(g1)

    g2 = nx.Graph()
    g2.add_edge("a", "b")
    g2.add_edge("b", "c")
    retrieval_mod._PPR_GRAPH_STRUCT_MEMO_FALLBACK[id(g2)] = (
        weakref.ref(g1),
        {"nodes": ["x", "y"], "node_to_idx": {"x": 0, "y": 1}, "neighbors": [[1], [0]], "degrees": [1, 1], "dangling": [], "n": 2},
    )

    struct = retrieval_mod._get_graph_struct(g2)
    assert set(struct["nodes"]) == {"a", "b", "c"}
