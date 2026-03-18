from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict


@dataclass
class ToyQuery:
    qid: str
    anchors: List[int]
    answer_nodes: Set[int] = field(default_factory=set)


@dataclass
class ToyGraphInstance:
    graph_edges: List[Tuple[int, int]]
    num_nodes: int
    query: ToyQuery
    gold_nodes: Set[int]
    gold_edges: Set[Tuple[int, int]]
    connector_nodes: Set[int]
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    method: str
    selected_nodes: Set[int]
    selected_edges: Set[Tuple[int, int]]
    diagnostics: Dict = field(default_factory=dict)


@dataclass
class ToyConfig:
    num_graphs: int = 50
    base_seed: int = 42

    path_len: int = 5
    branch_len: int = 2
    num_anchor_noise: int = 8
    num_connector_noise: int = 4
    extra_random_edges: int = 10

    ppr_alpha: float = 0.15
    ppr_steps: int = 20

    M: int = 8
    T: int = 20
    K: int = 4
    tau: int = 4
    Lp: int = 4
    Bc: int = 8
    rho: float = 0.75

    k_top_baseline: int = 12
    khop_k: int = 2