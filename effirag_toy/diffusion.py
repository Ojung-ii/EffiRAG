import networkx as nx
from typing import Dict, Iterable


def personalized_pagerank(
    G: nx.Graph,
    source: int,
    alpha: float = 0.15,
    max_iter: int = 100
) -> Dict[int, float]:
    personalization = {n: 0.0 for n in G.nodes()}
    personalization[source] = 1.0
    return nx.pagerank(
        G,
        alpha=1 - alpha,
        personalization=personalization,
        max_iter=max_iter
    )


def aggregate_anchor_scores(score_dicts: Iterable[Dict[int, float]], mode: str = "sum") -> Dict[int, float]:
    agg = {}
    for sd in score_dicts:
        for n, s in sd.items():
            agg[n] = agg.get(n, 0.0) + s

    if mode == "sum":
        return agg
    elif mode == "mean":
        k = len(list(score_dicts))
        return {n: v / max(k, 1) for n, v in agg.items()}
    else:
        raise ValueError(f"Unknown mode: {mode}")


def topk_nodes(scores: Dict[int, float], k: int):
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]