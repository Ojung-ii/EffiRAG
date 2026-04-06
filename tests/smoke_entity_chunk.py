from effirag.global_index import build_corpus_graph
from effirag.anchors import select_lexical_anchors
from effirag.render import render_context
from effirag.types import ContextDocument, RetrievalResult, Sample


def main():
    rows = [
        {
            "idx": 0,
            "title": "DocA",
            "text": "Alpha connects beta. Beta meets gamma. Gamma reaches delta. Delta returns alpha.",
        }
    ]
    g, stats = build_corpus_graph(
        rows,
        graph_mode="entity_chunk_graph",
        index_chunk_unit="passage",
        openie_mode="lexical",
        passage_chunking_strategy="sentence_window",
        passage_chunk_size_sentences=2,
        passage_chunk_stride_sentences=1,
        passage_chunk_min_sentences=2,
        show_progress=False,
    )
    chunk_nodes = [n for n in g.nodes if g.nodes[n].get("node_type") == "chunk"]
    assert len(chunk_nodes) >= 2, f"expected >=2 chunks, got {len(chunk_nodes)}"

    sample = Sample(
        qid="q1",
        question="How does alpha connect to gamma?",
        answer="",
        contexts=[ContextDocument(title="DocA", sentences=[
            "Alpha connects beta.", "Beta meets gamma.", "Gamma reaches delta.", "Delta returns alpha."
        ])],
    )
    anchors = select_lexical_anchors(sample, g, 3)
    assert anchors, "expected lexical anchors"

    retrieval = RetrievalResult(
        sample_id="q1",
        method="effirag",
        anchors=anchors,
        seeds=[anchors[0]],
        selected_nodes=list(chunk_nodes[:2]),
        selected_sentence_ids=[g.nodes[n]["chunk_id"] for n in chunk_nodes[:2]],
        selected_sentences=[g.nodes[n]["text"] for n in chunk_nodes[:2]],
        corridors=[
            {
                "corridor_id": "c01",
                "corridor_score": 1.0,
                "anchors": anchors[:2],
                "main_path_unit_ids": [g.nodes[chunk_nodes[0]]["chunk_id"]],
                "support_unit_ids": [g.nodes[chunk_nodes[1]]["chunk_id"]],
                "connector_adjacent_unit_ids": [],
                "unit_score_map": {
                    g.nodes[chunk_nodes[0]]["chunk_id"]: 1.0,
                    g.nodes[chunk_nodes[1]]["chunk_id"]: 0.8,
                },
            }
        ],
        diagnostics={
            "graph_mode": "entity_chunk_graph",
            "selected_unit_type": "chunk",
            "selected_text_map": {
                g.nodes[n]["chunk_id"]: g.nodes[n]["text"] for n in chunk_nodes[:2]
            },
            "text_unit_feature_table": {
                g.nodes[chunk_nodes[0]]["chunk_id"]: {
                    "corridor_ids": ["c01"],
                    "best_corridor_rank": 1,
                    "best_corridor_score": 1.0,
                    "is_main_candidate": True,
                    "is_support_candidate": False,
                    "is_connector_adjacent": False,
                    "query_overlap_score": 2.0,
                    "locality_score": 1.0,
                    "base_retrieval_score": 1.0,
                },
                g.nodes[chunk_nodes[1]]["chunk_id"]: {
                    "corridor_ids": ["c01"],
                    "best_corridor_rank": 1,
                    "best_corridor_score": 0.8,
                    "is_main_candidate": False,
                    "is_support_candidate": True,
                    "is_connector_adjacent": False,
                    "query_overlap_score": 1.0,
                    "locality_score": 0.5,
                    "base_retrieval_score": 0.8,
                },
            },
        },
        latency_ms=0.0,
    )
    rendered = render_context(
        sample,
        retrieval,
        max_context_sentences=2,
        render_mode="corridor_aware_flat",
        max_sentences=2,
        top_corridors=1,
    )
    assert rendered.sentences, "expected rendered chunk text"
    assert rendered.metadata.get("selected_unit_type") == "chunk"
    print({"chunk_nodes": len(chunk_nodes), "anchors": anchors, "rendered": rendered.sentences})


if __name__ == "__main__":
    main()
