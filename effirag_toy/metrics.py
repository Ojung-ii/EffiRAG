from typing import Dict, Set, Tuple
from types import RetrievalResult
from types import SimpleNamespace  # optional
from typing import Any


def compute_metrics(instance, result: RetrievalResult) -> Dict[str, float]:
    gold_nodes = instance.gold_nodes
    gold_edges = instance.gold_edges
    connector_nodes = instance.connector_nodes

    pred_nodes = result.selected_nodes
    pred_edges = result.selected_edges

    node_recall = len(pred_nodes & gold_nodes) / max(len(gold_nodes), 1)
    edge_recall = len(pred_edges & gold_edges) / max(len(gold_edges), 1)
    connector_recall = len(pred_nodes & connector_nodes) / max(len(connector_nodes), 1)

    node_precision = len(pred_nodes & gold_nodes) / max(len(pred_nodes), 1)
    noise_ratio = len(pred_nodes - gold_nodes) / max(len(pred_nodes), 1)

    anchors = set(instance.query.anchors)
    anchor_connected = 1.0 if anchors.issubset(pred_nodes) else 0.0

    return {
        "node_recall": node_recall,
        "edge_recall": edge_recall,
        "connector_recall": connector_recall,
        "node_precision": node_precision,
        "noise_ratio": noise_ratio,
        "subgraph_size": float(len(pred_nodes)),
        "anchor_covered": anchor_connected,
    }