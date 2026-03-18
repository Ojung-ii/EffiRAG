import random
import networkx as nx
from typing import Dict, List, Set, Tuple

from diffusion import personalized_pagerank, topk_nodes


def bfs_distance_with_cutoff(G: nx.Graph, src: int, cutoff: int) -> Dict[int, int]:
    return dict(nx.single_source_shortest_path_length(G, src, cutoff=cutoff))


def min_set_distance(G: nx.Graph, u: int, seeds: Set[int], tau: int) -> int:
    best = tau
    for z in seeds:
        try:
            d = nx.shortest_path_length(G, u, z)
            best = min(best, min(d, tau))
        except nx.NetworkXNoPath:
            continue
    return best


def stochastic_perturb_graph(G: nx.Graph, rng: random.Random, drop_prob: float = 0.1) -> nx.Graph:
    H = G.copy()
    for e in list(H.edges()):
        if rng.random() < drop_prob:
            H.remove_edge(*e)
    if H.number_of_edges() == 0:
        return G.copy()
    return H


def greedy_seed_set(G: nx.Graph, candidates: List[int], weights: Dict[int, float], K: int, tau: int) -> Set[int]:
    selected = set()
    remaining = set(candidates)

    while len(selected) < K and remaining:
        best_z = None
        best_obj = float("inf")

        for z in remaining:
            trial = selected | {z}
            obj = 0.0
            for u in candidates:
                obj += weights.get(u, 0.0) * min_set_distance(G, u, trial, tau)
            if obj < best_obj:
                best_obj = obj
                best_z = z

        selected.add(best_z)
        remaining.remove(best_z)

    return selected


def stable_seed_selection(
    G: nx.Graph,
    anchors: List[int],
    M: int,
    T: int,
    K: int,
    tau: int,
    alpha: float,
    base_seed: int = 42,
):
    rng = random.Random(base_seed)
    run_results = []

    for b in range(M):
        H = stochastic_perturb_graph(G, rng, drop_prob=0.1)

        anchor_scores = []
        for a in anchors:
            pr = personalized_pagerank(H, a, alpha=alpha)
            anchor_scores.append(pr)

        agg = {}
        for sd in anchor_scores:
            for n, s in sd.items():
                agg[n] = agg.get(n, 0.0) + s

        candidates = topk_nodes(agg, T)
        weights = {n: agg[n] for n in candidates}
        seeds = greedy_seed_set(G, candidates, weights, K, tau)

        run_results.append({
            "run_id": b,
            "anchor_scores": anchor_scores,
            "agg_scores": agg,
            "candidates": candidates,
            "seeds": seeds,
        })

    surrogate_universe = set(anchors)
    for rr in run_results:
        surrogate_universe.update(rr["candidates"])

    surrogate_weights = {}
    for u in surrogate_universe:
        vals = [rr["agg_scores"].get(u, 0.0) for rr in run_results]
        surrogate_weights[u] = sum(vals) / len(vals)

    best_loss = float("inf")
    best_result = None
    for rr in run_results:
        seeds = rr["seeds"]
        loss = 0.0
        for u in surrogate_universe:
            loss += surrogate_weights[u] * min_set_distance(G, u, seeds, tau)
        rr["global_loss"] = loss
        if loss < best_loss:
            best_loss = loss
            best_result = rr

    return best_result