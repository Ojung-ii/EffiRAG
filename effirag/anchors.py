import networkx as nx

from .types import Sample
from .utils import content_tokens


_TEXT_LIKE_TYPES = {"sentence", "chunk", "passage", "document"}


def _anchor_support_frequency(graph, node):
    seen = set()
    for nbr in graph.neighbors(node):
        ntype = str(graph.nodes[nbr].get("node_type", "") or "").strip().lower()
        if ntype in _TEXT_LIKE_TYPES:
            doc_idx = graph.nodes[nbr].get("doc_idx", nbr)
            seen.add((ntype, doc_idx, nbr if ntype == "sentence" else doc_idx))
    return len(seen)


def select_lexical_anchors(sample, graph, max_anchors, cfg=None):
    q_tokens = content_tokens(sample.question)
    scored = []

    for tok in q_tokens:
        node = f"e::{tok}"
        if node not in graph:
            continue

        support_freq = _anchor_support_frequency(graph, node)
        degree = int(graph.degree(node))
        rarity = 1.0 / (1.0 + float(support_freq))
        degree_penalty = 1.0 / (1.0 + 0.15 * max(0, degree - 1))
        score = 0.75 * rarity + 0.25 * degree_penalty
        scored.append((node, score, support_freq, degree))

    scored.sort(key=lambda x: (x[1], -x[2], -x[3]), reverse=True)
    anchors = [n for n, _, _, _ in scored[:max_anchors]]

    if anchors:
        return anchors

    fallback = []
    for tok in q_tokens:
        node = f"e::{tok}"
        if node in graph:
            fallback.append(node)
    return fallback[:max_anchors]
