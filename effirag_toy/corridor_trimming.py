import networkx as nx
from typing import Dict, List, Set, Tuple

from diffusion import personalized_pagerank


def pair_support(anchor_scores: Dict[int, float], seed_scores: Dict[int, float], node: int) -> float:
    return (anchor_scores.get(node, 0.0) * seed_scores.get(node, 0.0)) ** 0.5


def build_corridor(
    G: nx.Graph,
    anchors: List[int],
    seeds: Set[int],
    best_anchor_scores: List[Dict[int, float]],
    alpha: float,
    Lp: int,
    Bc: int,
):
    seed_score_map = {z: personalized_pagerank(G, z, alpha=alpha) for z in seeds}

    pair_stats = []
    for i, a in enumerate(anchors):
        a_scores = best_anchor_scores[i]
        for z in seeds:
            z_scores = seed_score_map[z]
            support_sum = 0.0
            for v in G.nodes():
                support_sum += pair_support(a_scores, z_scores, v)
            pair_stats.append(((a, z), support_sum))

    pair_stats.sort(key=lambda x: x[1], reverse=True)
    retained_pairs = [p for p, _ in pair_stats[:Lp]]

    corridor_nodes = set()
    for a, z in retained_pairs:
        a_scores = best_anchor_scores[anchors.index(a)]
        z_scores = seed_score_map[z]

        node_scores = []
        for v in G.nodes():
            s = pair_support(a_scores, z_scores, v)
            node_scores.append((v, s))

        node_scores.sort(key=lambda x: x[1], reverse=True)
        keep = [v for v, _ in node_scores[:Bc]]
        corridor_nodes.update(keep)
        corridor_nodes.add(a)
        corridor_nodes.add(z)

    H = G.subgraph(corridor_nodes).copy()
    return H, retained_pairs


def can_remove_node(H: nx.Graph, node: int, anchors: Set[int], min_seed_keep: int, seeds: Set[int]) -> bool:
    if node in anchors:
        return False

    trial = H.copy()
    trial.remove_node(node)

    # anchor coverage: all anchors must remain in same connected component if possible
    if not anchors.issubset(trial.nodes()):
        return False

    comps = list(nx.connected_components(trial))
    ok = any(anchors.issubset(comp) for comp in comps)
    if not ok:
        return False

    kept_seed_count = len(seeds.intersection(set(trial.nodes())))
    if kept_seed_count < min_seed_keep:
        return False

    return True


def greedy_trim(H: nx.Graph, anchors: List[int], seeds: Set[int], rho: float):
    anchors_set = set(anchors)
    min_seed_keep = max(1, int(round(rho * len(seeds))))

    changed = True
    while changed:
        changed = False

        leaf_like = []
        for v in H.nodes():
            deg = H.degree(v)
            if deg <= 1 and v not in anchors_set:
                leaf_like.append(v)

        for v in leaf_like:
            if v not in H:
                continue
            if can_remove_node(H, v, anchors_set, min_seed_keep, seeds):
                H.remove_node(v)
                changed = True

        comps = list(nx.connected_components(H))
        for comp in comps:
            if len(anchors_set.intersection(comp)) == 0:
                H.remove_nodes_from(comp)
                changed = True

    return H