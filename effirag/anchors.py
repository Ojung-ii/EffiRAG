import networkx as nx

from .types import Sample
from .utils import content_tokens


def select_lexical_anchors(sample, graph, max_anchors):
    q_tokens = content_tokens(sample.question)
    scored = []

    for tok in q_tokens:
        node = f"e::{tok}"
        if node not in graph:
            continue

        freq = 0
        for nbr in graph.neighbors(node):
            if graph.nodes[nbr].get("node_type") == "sentence":
                freq += 1

        score = 1.0 / (1.0 + freq)
        scored.append((node, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    anchors = [n for n, _ in scored[:max_anchors]]

    if anchors:
        return anchors

    fallback = []
    for tok in q_tokens:
        node = f"e::{tok}"
        if node in graph:
            fallback.append(node)
    return fallback[:max_anchors]
