from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from effirag.global_index import build_corpus_graph
from effirag.phase7_evidence_flow import (
    _candidate_atoms_from_local_graph,
    run_phase7_evidence_flow,
)
from effirag.types import ContextDocument, Sample


def _tiny_sample() -> Sample:
    return Sample(
        qid="q1",
        question="Who wrote the book?",
        answer="Alice",
        contexts=[ContextDocument(title="Doc", sentences=["Alice wrote the book."])],
        supporting_facts=[],
    )


def _tiny_cfg(tmp_path):
    return SimpleNamespace(
        method="phase7_evidence_flow",
        retrieval_method="phase7_evidence_flow",
        dataset="hotpotqa",
        limit=1,
        _config_path="tests/synthetic_phase7.yaml",
        _resolved_output_dir=str(tmp_path / "phase7_test_out"),
        output_dir=str(tmp_path / "phase7_test_out"),
        max_anchors=5,
        phase7_enabled=True,
        phase7_atom_unit="sentence",
        phase7_require_sentence_layer=True,
        phase7_build_sentence_layer=True,
        phase7_build_carrier_layer=True,
        phase7_link_sentence_to_carrier=True,
        phase7_link_sentence_to_entities=True,
        phase7_carrier_unit="sentence_window",
        phase7_carrier_chunk_size_sentences=3,
        phase7_carrier_chunk_stride_sentences=1,
        phase7_allow_empty_candidates_for_debug=False,
        phase7_semantic_anchor_top_t=16,
        phase7_candidate_top_m=16,
        phase7_max_selected_atoms=4,
        phase7_max_context_tokens=256,
        phase7_min_positive_gain=-1.0,
    )


def _build_tiny_phase7_graph():
    graph = nx.Graph()
    graph.add_node("e::alice", node_type="entity")
    graph.add_node("c::0:0", node_type="chunk", title="Doc", chunk_idx=0, text="Alice wrote the book.")
    graph.add_node(
        "s::0:0",
        node_type="sentence",
        sentence_id="Doc::0",
        title="Doc",
        sent_idx=0,
        text="Alice wrote the book.",
    )
    graph.add_edge("c::0:0", "s::0:0", edge_type="chunk_contains_sentence")
    graph.add_edge("s::0:0", "e::alice", edge_type="mentions")
    graph.add_edge("c::0:0", "e::alice", edge_type="entity_chunk_support")
    return graph


def test_phase7_force_sentence_layer_builds_sentence_and_carrier_links():
    corpus_rows = [
        {
            "idx": 0,
            "title": "Doc",
            "text": "S1. S2. S3. S4. S5.",
        }
    ]
    graph, stats = build_corpus_graph(
        corpus_rows=corpus_rows,
        graph_mode="entity_chunk_graph",
        index_chunk_unit="passage",
        passage_chunking_strategy="sentence_window",
        passage_chunk_size_sentences=3,
        passage_chunk_stride_sentences=1,
        force_sentence_layer=True,
        openie_mode="lexical",
        show_progress=False,
    )
    sentence_nodes = [n for n in graph.nodes if str(graph.nodes[n].get("node_type", "")) == "sentence"]
    carrier_nodes = [n for n in graph.nodes if str(graph.nodes[n].get("node_type", "")) in {"chunk", "passage", "document"}]
    sentence_carrier_edges = 0
    for u, v in graph.edges:
        tu = str(graph.nodes[u].get("node_type", ""))
        tv = str(graph.nodes[v].get("node_type", ""))
        if {"sentence", "chunk"} == {tu, tv}:
            sentence_carrier_edges += 1
    assert stats.get("build_sentence_layer") is True
    assert len(sentence_nodes) > 0
    assert len(carrier_nodes) > 0
    assert sentence_carrier_edges > 0


def test_phase7_fail_fast_on_passage_only_index(tmp_path, monkeypatch):
    graph = nx.Graph()
    graph.add_node("e::alice", node_type="entity")
    graph.add_node("c::0:0", node_type="chunk", title="Doc", chunk_idx=0, text="Alice wrote the book.")
    graph.add_edge("c::0:0", "e::alice", edge_type="entity_chunk_support")
    state = {
        "graph": graph,
        "meta": {"build_config": {"graph_mode": "entity_chunk_graph"}},
        "semantic_state": {},
        "sentence_nodes": [],
        "chunk_nodes": ["c::0:0"],
        "sentence_id_to_node": {},
    }
    monkeypatch.setattr("effirag.phase7_evidence_flow._build_index_state", lambda cfg, p7: state)
    monkeypatch.setattr("effirag.phase7_evidence_flow._question_embedding", lambda question, cfg: np.asarray([1.0], dtype=np.float32))
    monkeypatch.setattr("effirag.phase7_evidence_flow.select_lexical_anchors", lambda sample, graph, max_anchors, cfg: ["e::alice"])
    monkeypatch.setattr("effirag.phase7_evidence_flow._semantic_top_chunks", lambda question_vec, state, top_t: (["c::0:0"], {"c::0:0": 1.0}))
    monkeypatch.setattr("effirag.phase7_evidence_flow._semantic_top_entities", lambda question_vec, state, topn=24: (["e::alice"], {"e::alice": 1.0}))

    sample = _tiny_sample()
    cfg = _tiny_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="requires sentence nodes"):
        run_phase7_evidence_flow(sample, cfg)


def test_phase7_candidate_generation_returns_sentence_atoms_only():
    graph = _build_tiny_phase7_graph()
    atoms, _ = _candidate_atoms_from_local_graph(
        question="Who wrote the book?",
        local_graph=graph,
        flow_scores={"s::0:0": 0.9},
        semantic_chunk_scores={"c::0:0": 0.8},
        anchor_nodes=["e::alice"],
        candidate_top_m=8,
    )
    assert len(atoms) > 0
    for atom in atoms:
        assert graph.nodes[atom.atom_id]["node_type"] == "sentence"


def test_phase7_active_path_has_no_passage_fallback_in_final_selection(tmp_path, monkeypatch):
    graph = _build_tiny_phase7_graph()
    state = {
        "graph": graph,
        "meta": {"build_config": {"graph_mode": "entity_chunk_graph"}},
        "semantic_state": {},
        "sentence_nodes": ["s::0:0"],
        "chunk_nodes": ["c::0:0"],
        "sentence_id_to_node": {"Doc::0": "s::0:0"},
    }
    monkeypatch.setattr("effirag.phase7_evidence_flow._build_index_state", lambda cfg, p7: state)
    monkeypatch.setattr("effirag.phase7_evidence_flow._question_embedding", lambda question, cfg: np.asarray([1.0], dtype=np.float32))
    monkeypatch.setattr("effirag.phase7_evidence_flow.select_lexical_anchors", lambda sample, graph, max_anchors, cfg: ["e::alice"])
    monkeypatch.setattr("effirag.phase7_evidence_flow._semantic_top_chunks", lambda question_vec, state, top_t: (["c::0:0"], {"c::0:0": 1.0}))
    monkeypatch.setattr("effirag.phase7_evidence_flow._semantic_top_entities", lambda question_vec, state, topn=24: (["e::alice"], {"e::alice": 1.0}))

    sample = _tiny_sample()
    cfg = _tiny_cfg(tmp_path)
    result = run_phase7_evidence_flow(sample, cfg)
    assert len(result.selected_nodes) > 0
    for node in result.selected_nodes:
        assert graph.nodes[node]["node_type"] == "sentence"
