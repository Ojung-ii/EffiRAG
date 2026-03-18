import networkx as nx
from typing import List, Set

from diffusion import personalized_pagerank, topk_nodes
from types import RetrievalResult


def canonical_edge(u: int, v: int):
    return (u, v) if u < v else (v, u)


def ppr_union_baseline(G: nx.Graph, anchors: List[int], alpha: float, topk: int) -> RetrievalResult:
    scores = {}
    for a in anchors:
        pr = personalized_pagerank(G, a, alpha=alpha)
        for n, s in pr.items():
            scores[n] = scores.get(n, 0.0) + s

    nodes = set(topk_nodes(scores, topk))
    nodes.update(anchors)
    H = G.subgraph(nodes).copy()

    return RetrievalResult(
        method="ppr_union",
        selected_nodes=set(H.nodes()),
        selected_edges={canonical_edge(u, v) for u, v in H.edges()},
        diagnostics={"topk": topk},
    )


def shortest_path_baseline(G: nx.Graph, anchors: List[int]) -> RetrievalResult:
    nodes = set(anchors)
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            try:
                path = nx.shortest_path(G, anchors[i], anchors[j])
                nodes.update(path)
            except nx.NetworkXNoPath:
                pass

    H = G.subgraph(nodes).copy()
    return RetrievalResult(
        method="shortest_path",
        selected_nodes=set(H.nodes()),
        selected_edges={canonical_edge(u, v) for u, v in H.edges()},
        diagnostics={},
    )


def khop_baseline(G: nx.Graph, anchors: List[int], k: int = 2) -> RetrievalResult:
    nodes = set(anchors)
    for a in anchors:
        lengths = nx.single_source_shortest_path_length(G, a, cutoff=k)
        nodes.update(lengths.keys())

    H = G.subgraph(nodes).copy()
    return RetrievalResult(
        method="khop",
        selected_nodes=set(H.nodes()),
        selected_edges={canonical_edge(u, v) for u, v in H.edges()},
        diagnostics={"k": k},
    )