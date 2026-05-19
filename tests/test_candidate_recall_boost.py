from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

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


def _build_tie_graph():
    g = nx.Graph()
    g.add_node("e::anchor", node_type="entity", token="anchor")
    g.add_node("e::target", node_type="entity", token="target")
    g.add_node("s::x", node_type="sentence", text="X bridge", title="DocX")
    g.add_node("s::y", node_type="sentence", text="Y bridge", title="DocY")
    g.add_edge("e::anchor", "s::x")
    g.add_edge("s::x", "e::target")
    g.add_edge("e::anchor", "s::y")
    g.add_edge("s::y", "e::target")
    return g


def _tie_inputs():
    anchors = ["e::anchor"]
    proposal_by_anchor = {"e::anchor": ["e::anchor", "e::target"]}
    proposal_nodes = {"e::anchor", "e::target"}
    proposal_scores = {
        "e::anchor": 0.1,
        "e::target": 0.9,
        "s::x": 0.05,
        "s::y": 0.05,
    }
    semantic_scores = dict(proposal_scores)
    return anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores


def _clone_inputs(
    anchors: Sequence[str],
    proposal_by_anchor: MutableMapping[str, List[str]],
    proposal_nodes: Iterable[str],
    proposal_scores: MutableMapping[str, float],
    semantic_scores: MutableMapping[str, float],
) -> Tuple[List[str], Dict[str, List[str]], set[str], Dict[str, float], Dict[str, float]]:
    return (
        list(anchors),
        {str(k): list(v or []) for k, v in dict(proposal_by_anchor).items()},
        {str(x) for x in list(proposal_nodes or [])},
        {str(k): float(v) for k, v in dict(proposal_scores).items()},
        {str(k): float(v) for k, v in dict(semantic_scores).items()},
    )


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


def test_score_attachment_optimized_path_matches_baseline_scores(monkeypatch):
    g = _build_graph()
    cfg = DummyCfg()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()

    a_anchors, a_pba, a_nodes, a_ps, a_ss = _clone_inputs(
        anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores
    )
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "0")
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "0")
    baseline = apply_candidate_recall_boost(
        g=g,
        anchors=a_anchors,
        proposal_by_anchor=a_pba,
        proposal_nodes=a_nodes,
        proposal_scores=a_ps,
        semantic_scores=a_ss,
        cfg=cfg,
    )

    b_anchors, b_pba, b_nodes, b_ps, b_ss = _clone_inputs(
        anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores
    )
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "1")
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "0")
    optimized = apply_candidate_recall_boost(
        g=g,
        anchors=b_anchors,
        proposal_by_anchor=b_pba,
        proposal_nodes=b_nodes,
        proposal_scores=b_ps,
        semantic_scores=b_ss,
        cfg=cfg,
    )

    common = sorted(set(baseline[2].keys()) & set(optimized[2].keys()))
    assert common
    max_abs_diff = max(abs(float(baseline[2][k]) - float(optimized[2][k])) for k in common)
    assert max_abs_diff <= 1.0e-9


def test_score_attachment_stable_order_is_deterministic(monkeypatch):
    g = _build_graph()
    cfg = DummyCfg()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "1")
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "1")

    a_anchors, a_pba, a_nodes, a_ps, a_ss = _clone_inputs(
        anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores
    )
    out_a = apply_candidate_recall_boost(
        g=g,
        anchors=a_anchors,
        proposal_by_anchor=a_pba,
        proposal_nodes=a_nodes,
        proposal_scores=a_ps,
        semantic_scores=a_ss,
        cfg=cfg,
    )

    rev_anchors = list(reversed(list(anchors)))
    rev_pba = {k: list(v) for k, v in proposal_by_anchor.items()}
    rev_nodes = list(reversed(list(proposal_nodes)))
    b_anchors, b_pba, b_nodes, b_ps, b_ss = _clone_inputs(
        rev_anchors, rev_pba, rev_nodes, proposal_scores, semantic_scores
    )
    out_b = apply_candidate_recall_boost(
        g=g,
        anchors=b_anchors,
        proposal_by_anchor=b_pba,
        proposal_nodes=b_nodes,
        proposal_scores=b_ps,
        semantic_scores=b_ss,
        cfg=cfg,
    )

    assert out_a[0] == out_b[0]
    assert out_a[1] == out_b[1]
    assert out_a[4].get("selected_boost_nodes") == out_b[4].get("selected_boost_nodes")


def test_stable_order_does_not_change_score_values(monkeypatch):
    g = _build_graph()
    cfg = DummyCfg()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "1")
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "0")
    out_unstable = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "1")
    out_stable = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )

    common_proposal = sorted(set(out_unstable[2].keys()) & set(out_stable[2].keys()))
    common_semantic = sorted(set(out_unstable[3].keys()) & set(out_stable[3].keys()))
    assert common_proposal and common_semantic
    max_prop_diff = max(abs(float(out_unstable[2][k]) - float(out_stable[2][k])) for k in common_proposal)
    max_sem_diff = max(abs(float(out_unstable[3][k]) - float(out_stable[3][k])) for k in common_semantic)
    assert max_prop_diff <= 1.0e-9
    assert max_sem_diff <= 1.0e-9


def test_missing_score_handling_is_identical_between_baseline_and_optimized(monkeypatch):
    g = _build_graph()
    cfg = DummyCfg()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()
    for key in ["s::1", "s::3", "e::a"]:
        proposal_scores.pop(key, None)
        semantic_scores.pop(key, None)

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "0")
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "0")
    out_base = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "1")
    out_opt = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )

    target_keys = sorted(set(out_base[2].keys()) & set(out_opt[2].keys()))
    assert target_keys
    max_abs_diff = max(abs(float(out_base[2][k]) - float(out_opt[2][k])) for k in target_keys)
    assert max_abs_diff <= 1.0e-9


def test_candidate_key_normalization_and_score_defaults_match_between_modes(monkeypatch):
    g = _build_graph()
    cfg = DummyCfg()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()
    for key in ["s::1", "s::3", "e::a"]:
        proposal_scores.pop(key, None)
        semantic_scores.pop(key, None)

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "0")
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "0")
    out_base = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )
    base_rows = list((out_base[4] or {}).get("candidate_score_table", []) or [])

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "1")
    out_opt = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )
    opt_rows = list((out_opt[4] or {}).get("candidate_score_table", []) or [])

    assert base_rows and opt_rows
    base_map = {str(r.get("candidate_key_normalized", "")): dict(r) for r in base_rows}
    opt_map = {str(r.get("candidate_key_normalized", "")): dict(r) for r in opt_rows}
    assert set(base_map.keys()) == set(opt_map.keys())
    for key in sorted(base_map.keys()):
        b = base_map[key]
        o = opt_map[key]
        assert str(b.get("candidate_key_normalized", "")) == str(o.get("candidate_key_normalized", ""))
        assert dict(b.get("missing_flags", {}) or {}) == dict(o.get("missing_flags", {}) or {})
        assert abs(float(b.get("final_score", 0.0)) - float(o.get("final_score", 0.0))) <= 1.0e-9


def test_optimized_score_attachment_does_not_change_candidate_set(monkeypatch):
    g = _build_graph()
    cfg = DummyCfg()
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _base_inputs()

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "0")
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "0")
    out_base = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "1")
    out_opt = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )

    assert set(out_base[1]) == set(out_opt[1])


def test_optimized_path_expansion_matches_baseline_on_tie_paths(monkeypatch):
    g = _build_tie_graph()
    cfg = DummyCfg(
        path_candidate_expansion_enabled=True,
        anchor_expansion_enabled=False,
        candidate_diversity_enabled=False,
        entity_diversity_enabled=False,
        source_diversity_enabled=False,
        candidate_dedup_enabled=False,
        max_path_candidates=8,
        max_bridge_candidates=8,
        max_expanded_candidates=8,
        tau=4,
    )
    anchors, proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores = _tie_inputs()

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "0")
    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "0")
    out_base = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )

    monkeypatch.setenv("PHASE6T_SCORE_ATTACHMENT_OPT", "1")
    out_opt = apply_candidate_recall_boost(
        g=g,
        anchors=list(anchors),
        proposal_by_anchor={k: list(v) for k, v in proposal_by_anchor.items()},
        proposal_nodes=set(proposal_nodes),
        proposal_scores=dict(proposal_scores),
        semantic_scores=dict(semantic_scores),
        cfg=cfg,
    )

    assert set(out_base[1]) == set(out_opt[1])

    base_rows = list((out_base[4] or {}).get("candidate_score_table", []) or [])
    opt_rows = list((out_opt[4] or {}).get("candidate_score_table", []) or [])
    base_map = {str(r.get("candidate_key_normalized", "")): dict(r) for r in base_rows}
    opt_map = {str(r.get("candidate_key_normalized", "")): dict(r) for r in opt_rows}
    assert set(base_map.keys()) == set(opt_map.keys())

    for key in sorted(base_map.keys()):
        b = base_map[key]
        o = opt_map[key]
        assert abs(float(b.get("final_score", 0.0)) - float(o.get("final_score", 0.0))) <= 1.0e-9
        assert dict(b.get("missing_flags", {}) or {}) == dict(o.get("missing_flags", {}) or {})
        assert dict(b.get("score_source", {}) or {}) == dict(o.get("score_source", {}) or {})
