import networkx as nx
import numpy as np

from effirag.phase8_pamae_entity_seeding import (
    build_medoid_evidence_candidates,
    build_query_entity_universe,
    refine_medoid_seeds,
    sample_medoid_seed_sets,
    select_best_seed_set,
)


class Cfg:
    phase8_entity_universe_top_n = 10
    phase8_entity_universe_min_n = 3
    phase8_entity_degree_cap = 20
    phase8_pamae_k = 2
    phase8_pamae_sample_size_per_k = 2
    phase8_pamae_num_samples = 2
    phase8_pamae_random_seed = 7
    phase8_refine_enabled = True
    phase8_refine_hops = 2
    phase8_refine_max_candidates_per_seed = 8
    phase8_refine_degree_cap = 20
    phase8_seed_top_atoms_per_entity = 2
    phase8_seed_top_carriers_per_entity = 1
    phase8_seed_path_max_hops = 3
    phase8_seed_path_max_pairs = 4
    phase8_seed_total_candidate_cap = 12


def _toy_state():
    g = nx.Graph()
    for eid, name in [
        ("e::alpha", "Alpha"),
        ("e::beta", "Beta"),
        ("e::gamma", "Gamma"),
        ("e::delta", "Delta"),
    ]:
        g.add_node(eid, node_type="entity", name=name)
    for sid, title, idx, text in [
        ("s::alpha::0", "Alpha", 0, "Alpha connects to beta."),
        ("s::alpha::1", "Alpha", 1, "Alpha has local support."),
        ("s::beta::0", "Beta", 0, "Beta bridges gamma."),
        ("s::gamma::0", "Gamma", 0, "Gamma evidence appears here."),
    ]:
        g.add_node(sid, node_type="sentence", title=title, sent_idx=idx, sentence_id=f"{title}::{idx}", text=text)
    g.add_node("c::alpha", node_type="chunk", title="Alpha", chunk_idx=0)
    g.add_node("c::beta", node_type="chunk", title="Beta", chunk_idx=1)
    g.add_edges_from(
        [
            ("e::alpha", "s::alpha::0"),
            ("e::alpha", "s::alpha::1"),
            ("e::beta", "s::alpha::0"),
            ("e::beta", "s::beta::0"),
            ("e::gamma", "s::beta::0"),
            ("e::gamma", "s::gamma::0"),
            ("s::alpha::0", "c::alpha"),
            ("s::alpha::1", "c::alpha"),
            ("s::beta::0", "c::beta"),
            ("s::gamma::0", "c::beta"),
            ("e::alpha", "e::beta"),
            ("e::beta", "e::gamma"),
            ("e::gamma", "e::delta"),
        ]
    )
    entity_ids = ["e::alpha", "e::beta", "e::gamma", "e::delta"]
    entity_embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    semantic = {
        "entity_ids": entity_ids,
        "entity_embeddings": entity_embeddings,
        "entity_id_to_idx": {eid: i for i, eid in enumerate(entity_ids)},
        "chunk_ids": ["c::alpha", "c::beta"],
        "chunk_embeddings": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "chunk_id_to_idx": {"c::alpha": 0, "c::beta": 1},
    }
    return {"graph": g, "semantic_state": semantic}


def test_phase8_universe_sampling_refinement_and_evidence():
    state = _toy_state()
    qvec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = {
        "query_embedding": qvec,
        "flow_scores": {"e::alpha": 0.4, "e::beta": 0.3, "e::gamma": 0.1},
        "semantic_entities": ["e::alpha", "e::beta"],
        "semantic_chunks": ["c::alpha"],
        "anchor_nodes": ["e::alpha"],
    }
    universe = build_query_entity_universe("alpha beta bridge", state, embeddings, Cfg())
    assert len(universe) >= 3
    assert universe[0].entity_id in {"e::alpha", "e::beta"}

    seed_sets = sample_medoid_seed_sets(universe, qvec, Cfg())
    assert len(seed_sets) == 2
    assert all(len(seed_set) == 2 for seed_set in seed_sets)

    best = select_best_seed_set(seed_sets, universe, qvec, Cfg())
    assert len(best) == 2

    refined = refine_medoid_seeds(best, universe, state, qvec, ["e::alpha"], Cfg())
    assert len(refined) == 2

    evidence = build_medoid_evidence_candidates(refined, ["e::alpha"], state, Cfg())
    assert evidence
    assert all(state["graph"].nodes[eid]["node_type"] == "sentence" for eid in evidence)
