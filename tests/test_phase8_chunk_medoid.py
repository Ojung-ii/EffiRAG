import networkx as nx
import numpy as np

from effirag.phase8_chunk_medoid import (
    build_chunk_medoid_evidence_candidates,
    build_query_chunk_universe,
    refine_chunk_medoids_via_bridge_entities,
    sample_chunk_medoid_seed_sets,
    select_best_chunk_medoid_seed_set,
)


def _toy_state():
    graph = nx.Graph()
    graph.add_node("c::a", node_type="chunk", title="A", text="Alpha carrier evidence", chunk_idx=0)
    graph.add_node("c::b", node_type="chunk", title="B", text="Beta carrier evidence", chunk_idx=1)
    graph.add_node("s::a::0", node_type="sentence", title="A", sent_idx=0, sentence_id="A::0", text="Alpha supports the answer.")
    graph.add_node("s::a::1", node_type="sentence", title="A", sent_idx=1, sentence_id="A::1", text="Alpha has bridge context.")
    graph.add_node("s::b::0", node_type="sentence", title="B", sent_idx=0, sentence_id="B::0", text="Beta supports another hop.")
    graph.add_node("e::Alpha", node_type="entity", name="Alpha", title="A")
    graph.add_node("e::Beta", node_type="entity", name="Beta", title="B")
    graph.add_edges_from(
        [
            ("c::a", "s::a::0"),
            ("c::a", "s::a::1"),
            ("c::b", "s::b::0"),
            ("s::a::0", "e::Alpha"),
            ("s::a::1", "e::Beta"),
            ("s::b::0", "e::Beta"),
            ("e::Alpha", "c::a"),
            ("e::Beta", "c::b"),
        ]
    )
    semantic = {
        "chunk_ids": ["c::a", "c::b"],
        "chunk_id_to_idx": {"c::a": 0, "c::b": 1},
        "chunk_embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    return {
        "graph": graph,
        "semantic_state": semantic,
    }


def test_chunk_universe_uses_carrier_candidates_without_rebuild():
    state = _toy_state()
    cfg = {
        "phase8_chunk_universe_top_n": 10,
        "phase8_chunk_universe_min_n": 1,
        "phase8_chunk_universe_unit": "carrier",
    }
    universe = build_query_chunk_universe(
        "alpha evidence",
        state,
        {"query_embedding": np.asarray([1.0, 0.0], dtype=np.float32)},
        cfg,
    )
    ids = [candidate.chunk_id for candidate in universe]
    assert "c::a" in ids
    assert universe[0].query_relevance >= universe[-1].query_relevance


def test_chunk_medoid_seeding_and_refinement_stay_chunk_typed():
    state = _toy_state()
    cfg = {
        "phase8_chunk_universe_top_n": 10,
        "phase8_chunk_universe_min_n": 2,
        "phase8_chunk_medoid_k": 1,
        "phase8_chunk_medoid_sample_size_per_k": 2,
        "phase8_chunk_medoid_num_samples": 2,
        "phase8_chunk_medoid_random_seed": 7,
        "phase8_chunk_bridge_refine_enabled": True,
        "phase8_chunk_bridge_max_entities_per_seed": 4,
        "phase8_chunk_bridge_entity_degree_cap": 10,
        "phase8_chunk_bridge_max_refine_candidates_per_seed": 8,
    }
    qvec = np.asarray([1.0, 0.0], dtype=np.float32)
    universe = build_query_chunk_universe("alpha evidence", state, {"query_embedding": qvec}, cfg)
    seed_sets = sample_chunk_medoid_seed_sets(universe, qvec, cfg)
    best = select_best_chunk_medoid_seed_set(seed_sets, universe, qvec, cfg)
    refined = refine_chunk_medoids_via_bridge_entities(best, universe, state, qvec, cfg)
    assert refined
    assert all(seed.chunk_id.startswith("c::") for seed in refined)


def test_chunk_medoid_evidence_candidates_are_sentence_atoms():
    state = _toy_state()
    cfg = {
        "phase8_chunk_seed_total_candidate_cap": 10,
        "phase8_chunk_seed_top_sentences_per_carrier": 2,
        "phase8_chunk_seed_top_entities_per_seed": 4,
        "phase8_chunk_seed_top_atoms_per_entity": 2,
        "phase8_chunk_seed_top_carriers_per_entity": 1,
    }
    qvec = np.asarray([1.0, 0.0], dtype=np.float32)
    universe = build_query_chunk_universe("alpha evidence", state, {"query_embedding": qvec}, cfg)
    seed_sets = sample_chunk_medoid_seed_sets(universe, qvec, {"phase8_chunk_medoid_k": 1})
    best = select_best_chunk_medoid_seed_set(seed_sets, universe, qvec, cfg)
    evidence = build_chunk_medoid_evidence_candidates(best, state, cfg)
    assert evidence
    assert all(eid.startswith("s::") for eid in evidence)
