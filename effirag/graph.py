from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from .types import Sample
from .utils import content_tokens


@dataclass
class GraphArtifacts:
    graph: nx.Graph
    sentence_nodes: List[str]
    sentence_id_by_node: Dict[str, str]


def sentence_id(title, sent_idx):
    return f"{title}::{sent_idx}"


def build_document_entity_graph(sample):
    g = nx.Graph()
    sentence_nodes = []
    sentence_id_by_node = {}

    for doc_idx, doc in enumerate(sample.contexts):
        prev_node = None
        for sent_idx, sent in enumerate(doc.sentences):
            node = f"s::{doc_idx}:{sent_idx}"
            sid = sentence_id(doc.title, sent_idx)

            g.add_node(
                node,
                node_type="sentence",
                title=doc.title,
                sent_idx=sent_idx,
                sentence_id=sid,
                text=sent,
            )
            sentence_nodes.append(node)
            sentence_id_by_node[node] = sid

            if prev_node is not None:
                g.add_edge(prev_node, node, edge_type="adjacent_sentence")
            prev_node = node

            for tok in content_tokens(sent):
                entity_node = f"e::{tok}"
                if entity_node not in g:
                    g.add_node(entity_node, node_type="entity", token=tok)
                g.add_edge(node, entity_node, edge_type="mentions")

    return GraphArtifacts(graph=g, sentence_nodes=sentence_nodes, sentence_id_by_node=sentence_id_by_node)
