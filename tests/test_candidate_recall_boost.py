from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from effirag.candidate_recall_boost import apply_candidate_recall_boost


@dataclass
class DummyCfg:
    path_candidate_expansion_enabled: bool = True
    anchor_expansion_enabled: bool = True
    candidate_diversity_enabled: bool = True
    entity_diversity_enabled: bool = True
    source_diversity_enabled: bool = True
    candidate_dedup_enabled: bool = True
    max_bridge_candidates: int = 4
    max_path_candidates: int = 4
    max_anchor_expansion_hops: int = 1
    max_expanded_candidates: int = 6
    proposal_anchor_local_topn: int = 8
    tau: int = 4
    dataset: str = "hotpotqa"


def _build_graph():
    g = nx.Graph()
    g.add_node("e::a", node_type="entity", token="A")
    g.add_node("e::b", node_type="entity", token="B")
    g.add_node("e::x", node_type="entity", token="X")
    g.add_node("s::1", node_type="sentence", text="A founded X.", title="DocA")
    g.add_node("s::2", node_type="sentence", text="X later merged with B.", title="DocBridge")
    g.add_node("s::3", node_type="sentence", text="B was acquired.", title="DocB")
    g.add_node("s::4", node_type="sentence", text="B was acquired.", title="DocB2")
    g.add_node("s::5", node_type="sentence", text="A met C.", title="SharedDoc")
    g.add_node("s::6", node_type="sentence", text="A met D.", title="SharedDoc")

    g.add_edge("e::a", "s::1")
    g.add_edge("s::1", "e::x")
    g.add_edge("e::x", "s::2")
    g.add_edge("s::2", "e::b")
    g.add_edge("e::b", "s::3")
    g.add_edge("e::b", "s::4")
    g.add_edge("e::a", "s::5")
    g.add_edge("e::a", "s::6")
    return g


def _base_inputs():
    proposal_by_anchor = {
        "e::a": ["e::a", "s::1", "s::5", "s::6"],
        "e::b": ["e::b", "s::3", "s::4"],
    }
    proposal_nodes = {"e::a", "e::b", "s::1", "s::3", "s::4", "s::5", "s::6"}
    proposal_scores = {
        "e::a": 0.6,
        "e::b": 0.59,
        "s::1": 0.78,
        "s::3": 0.76,
        "s::4": 0.75,
        "s::5": 0.80,
        "s::6": 0.79,
        "s::2": 0.72,
        "e::x": 0.68,
    }
    semantic_scores = dict(proposal_scores)
    anchors = ["e::a", "e::b"]
    return anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores


def test_candidate_recall_boost_is_deterministic():
    g = _build_graph()
    cfg = DummyCfg()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()

    out_a = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=cfg,
    )
    out_b = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=cfg,
    )

    assert out_a[1] == out_b[1]
    assert out_a[4]["selected_boost_nodes"] == out_b[4]["selected_boost_nodes"]


def test_path_expansion_adds_bridge_candidate():
    g = _build_graph()
    cfg = DummyCfg()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()

    _, updated_nodes, _, _, diag = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=cfg,
    )

    assert diag["enabled"] is True
    assert "s::2" in updated_nodes


def test_anchor_expansion_respects_max_hops():
    g = _build_graph()
    cfg = DummyCfg(path_candidate_expansion_enabled=False, max_anchor_expansion_hops=1)
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()

    _, updated_nodes, _, _, _ = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=cfg,
    )

    # e::x is 2 hops from e::a and 2 hops from e::b, so hop=1 expansion must exclude it.
    assert "e::x" not in updated_nodes


def test_dedup_and_source_diversity_are_applied():
    g = _build_graph()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()

    # Dedup should remove repeated sentence text even with diversity off.
    dedup_cfg = DummyCfg(
        max_expanded_candidates=8,
        candidate_diversity_enabled=False,
        entity_diversity_enabled=False,
        source_diversity_enabled=False,
    )
    _, _, _, _, dedup_diag = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=dedup_cfg,
    )
    assert int(dedup_diag.get("dedup_skipped", 0)) >= 1

    # Source diversity should avoid over-concentrating on one source key.
    diversity_cfg = DummyCfg(max_expanded_candidates=4)
    _, _, _, _, diversity_diag = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=diversity_cfg,
    )
    selected = list(diversity_diag.get("selected_boost_nodes", []) or [])
    shared_doc_count = 0
    for node in selected:
        if str((g.nodes[node] or {}).get("title", "") or "") == "SharedDoc":
            shared_doc_count += 1
    assert shared_doc_count <= 1


def test_candidate_recall_boost_is_dataset_independent():
    g = _build_graph()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()
    cfg_hotpot = DummyCfg(dataset="hotpotqa")
    cfg_2wiki = DummyCfg(dataset="2wikimultihopqa")

    out_hotpot = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=cfg_hotpot,
    )
    out_2wiki = apply_candidate_recall_boost(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_nodes=proposal_nodes,
        proposal_scores=proposal_scores,
        semantic_scores=semantic_scores,
        cfg=cfg_2wiki,
    )

    assert out_hotpot[1] == out_2wiki[1]
    assert out_hotpot[4]["selected_boost_nodes"] == out_2wiki[4]["selected_boost_nodes"]
