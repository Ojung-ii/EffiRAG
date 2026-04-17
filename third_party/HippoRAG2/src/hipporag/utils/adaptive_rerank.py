# from __future__ import annotations
# import math
# import numpy as np
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional, Any

# # ===============[ Utils ]===============
# def _safe_softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
#     x = np.asarray(x, dtype=np.float64) / max(tau, 1e-8)
#     x = x - np.max(x)
#     ex = np.exp(x)
#     s = ex.sum()
#     return ex / max(s, 1e-12)

# def _zscore(x: np.ndarray) -> np.ndarray:
#     x = np.asarray(x, dtype=np.float64)
#     mu, sd = x.mean(), x.std()
#     if sd < 1e-12: return np.zeros_like(x)
#     return (x - mu) / sd

# def _quantile_rank(x: np.ndarray) -> np.ndarray:
#     order = np.argsort(x)
#     ranks = np.empty_like(order, dtype=np.float64)
#     ranks[order] = np.arange(1, len(x) + 1)
#     return (ranks - 0.5) / len(x)

# def _entropy_from_probs(p: np.ndarray) -> float:
#     p = np.asarray(p, dtype=np.float64)
#     p = p / max(p.sum(), 1e-12)
#     p = np.clip(p, 1e-12, 1.0)
#     return float(-(p * np.log(p)).sum())

# # ===============[ SubgraphConfig proxy ]===============
# @dataclass
# class SubgraphConfig:
#     subgraph_mode: str = "ppr"
#     max_depth: int = 2
#     per_hop_topk: int = 12
#     allow_topk: int = 150
#     per_node_top_r: int = 6
#     flow_quantile: float = 0.85
#     rank_metric: str = "ppr"
#     keep_bridges: bool = True
#     keep_articulation: bool = True
#     keep_top_edge_betw: int = 0

# # ===============[ Query features ]===============
# @dataclass
# class QueryFeatures:
#     H_sem: float          # semantic entropy (불확실성)
#     flow_q85: float       # 상위 15% PPR 누적비
#     G_betw: float         # betweenness gini 근사
#     tree_depth: int       # 추정 depth
#     tail_mass: float      # Top-10 밖 의미 질량

# def _estimate_graph_hierarchy_metrics(last_ppr_scores: Optional[Dict[Any, float]]
# ) -> Tuple[float, float, int]:
#     if not last_ppr_scores:
#         return 0.5, 0.45, 2  # flow_q85, G_betw, depth
#     scores = np.asarray(sorted(last_ppr_scores.values(), reverse=True), dtype=np.float64)
#     if scores.sum() <= 0:
#         return 0.5, 0.45, 2
#     cum = np.cumsum(scores) / scores.sum()
#     idx = int(max(1, math.floor(0.15 * len(scores))) - 1)
#     flow_q85 = float(cum[idx])
#     G_betw = float(np.clip(0.4 + 0.6 * flow_q85, 0.0, 1.0))  # 간단 근사
#     depth = 3 if flow_q85 > 0.62 else 2
#     return flow_q85, G_betw, depth

# def _build_features_from_semantic(scores_sem: np.ndarray,
#                                   last_ppr_scores: Optional[Dict[Any, float]]) -> QueryFeatures:
#     p_sem = _safe_softmax(np.asarray(scores_sem, dtype=np.float64), tau=1.0)
#     H_sem = _entropy_from_probs(p_sem)
#     tail_mass = float(p_sem[10:].sum()) if len(p_sem) > 10 else 0.0
#     flow_q85, G_betw, depth = _estimate_graph_hierarchy_metrics(last_ppr_scores)
#     return QueryFeatures(H_sem=H_sem, flow_q85=flow_q85, G_betw=G_betw,
#                          tree_depth=depth, tail_mass=tail_mass)

# # ===============[ Adaptive policy ]===============
# @dataclass
# class AdaptivePolicy:
#     alpha: float
#     pin_k: int
#     gate_ceiling: float
#     sim_mode: str                 # "node_level" | "tangent" | "lorentz"
#     attn_topk: Optional[int]
#     attn_lambda: Optional[float]
#     attn_residual: Optional[float]
#     lorentz_temp: Optional[float]
#     depth_B: int
#     allow_topk_B: int
#     per_hop_topk_B: int
#     per_node_top_r_B: int
#     flow_q_B: float

# def _decide_policy(feat: QueryFeatures) -> AdaptivePolicy:
#     # 계층성 판단 & 의미 불확실성
#     hier = (feat.G_betw > 0.55) or (feat.flow_q85 > 0.65) or (feat.tree_depth >= 3)
#     very_uncertain = (feat.H_sem > 1.6)
#     mid_uncertain  = (feat.H_sem > 1.2)

#     # alpha: 계층↑ → 구조 가중↑, 의미 확실→ 구조 가중↓
#     alpha = 0.70
#     if feat.G_betw > 0.55: alpha += 0.06
#     if feat.H_sem < 1.2:   alpha -= 0.04
#     alpha = float(np.clip(alpha, 0.68, 0.80))

#     # pin_k: 의미 불확실할수록 증가
#     pin_k = 2 if very_uncertain else (1 if mid_uncertain else 0)

#     # gate_ceiling: 계층↑ → 더 낮게 (구조 과적용 억제)
#     gate_ceiling = 0.88 if hier else 0.95

#     # Stage-B subgraph 세부
#     if hier:
#         depth_B = 3
#         allow_topk_B = 150
#         per_hop_topk_B = 10
#         per_node_top_r_B = 5
#         flow_q_B = 0.88
#         if very_uncertain:
#             sim_mode = "lorentz"
#             attn_topk, attn_lambda, attn_residual, lorentz_temp = 10, 7.5, 0.12, 1.2
#         else:
#             sim_mode = "tangent"
#             attn_topk, attn_lambda, attn_residual, lorentz_temp = 8, 7.5, 0.12, None
#     else:
#         # 비계층: node_level 유지
#         depth_B = 2
#         allow_topk_B = 150
#         per_hop_topk_B = 12
#         per_node_top_r_B = 6
#         flow_q_B = 0.85
#         sim_mode = "node_level"
#         attn_topk, attn_lambda, attn_residual, lorentz_temp = None, None, None, None

#     return AdaptivePolicy(alpha=alpha, pin_k=pin_k, gate_ceiling=gate_ceiling,
#                           sim_mode=sim_mode, attn_topk=attn_topk, attn_lambda=attn_lambda,
#                           attn_residual=attn_residual, lorentz_temp=lorentz_temp,
#                           depth_B=depth_B, allow_topk_B=allow_topk_B,
#                           per_hop_topk_B=per_hop_topk_B, per_node_top_r_B=per_node_top_r_B,
#                           flow_q_B=flow_q_B)

# # ===============[ SoftRank calibration ]===============
# def _calibrated_fusion(scores_sem: np.ndarray,
#                        scores_struct: np.ndarray,
#                        alpha: float,
#                        tau_sem: float = 0.8,
#                        tau_struct: float = 0.8) -> np.ndarray:
#     z_sem     = _zscore(scores_sem)
#     z_struct  = _zscore(scores_struct)
#     p_sem     = _safe_softmax(z_sem, tau=tau_sem)
#     p_struct  = _safe_softmax(z_struct, tau=tau_struct)
#     fused = (1.0 - alpha) * p_sem + alpha * p_struct
#     return fused

# # ===============[ TreeRep bonus(선택) ]===============
# def _apply_treerep_bonus(scores: np.ndarray,
#                          is_bridge: Optional[np.ndarray],
#                          is_articulation: Optional[np.ndarray],
#                          gamma: float = 0.03) -> np.ndarray:
#     if is_bridge is None and is_articulation is None:
#         return scores
#     bonus = np.zeros_like(scores, dtype=np.float64)
#     if is_bridge is not None:       bonus += gamma * (is_bridge.astype(np.float64))
#     if is_articulation is not None: bonus += gamma * (is_articulation.astype(np.float64))
#     return np.clip(scores + bonus, 0.0, 1.0)

# # ===============[ 2-Stage Adaptive Rerank ]===============
# def adaptive_rerank(
#     query_graph,
#     passage_topk_ids,
#     passage_topk_scores,
#     passage_node_idxs,
#     last_ppr_scores,
#     cfg,
#     embedder,
#     scorer,
#     passage_embeddings,
#     entity_embeddings,
#     reranker=None,
#     **rerank_kwargs,   # ⬅️ 예상치 못한 키워드들도 안전하게 수용
# ):
#     """
#     Wrapper that forwards arbitrary reranking kwargs to `reranker.rerank_with_structure`.
#     Accepts any of:
#       sim_cac_mode, alpha, struct_k, gating_enabled, gate_floor, gate_ceiling,
#       attn_topk, attn_lambda, attn_residual, lorentz_temp, k_rrf, pin_k,
#       treerep_backend, ltr_config, enable_hgnn_fusion, ...
#     """

#     # 1) None 값은 제거해서 깨끗하게 전달
#     clean_kwargs: Dict[str, Any] = {k: v for k, v in rerank_kwargs.items() if v is not None}

#     # 2) 안전한 기본값 세팅 (누락시)
#     clean_kwargs.setdefault("sim_cac_mode", "tangent")
#     clean_kwargs.setdefault("alpha", 0.78)
#     clean_kwargs.setdefault("struct_k", 30)
#     clean_kwargs.setdefault("gating_enabled", True)
#     clean_kwargs.setdefault("gate_floor", 0.60)
#     clean_kwargs.setdefault("gate_ceiling", 0.90)
#     clean_kwargs.setdefault("k_rrf", 16)
#     clean_kwargs.setdefault("pin_k", 0)
#     clean_kwargs.setdefault("treerep_backend", "bfs")
#     # ltr_config/LTR_DISABLED 스코프에 맞게 유지
#     try:
#         clean_kwargs.setdefault("ltr_config", LTR_DISABLED)
#     except NameError:
#         pass
#     clean_kwargs.setdefault("enable_hgnn_fusion", False)

#     # 3) 실제 리랭킹 호출 (reranker는 인자로 받거나, self.reranker를 사용)
#     if reranker is None:
#         # 클래스 메서드라면 self가 있을 가능성이 큼
#         # self.reranker가 있다면 활용
#         try:
#             _reranker = self.reranker  # type: ignore[name-defined]
#         except Exception as e:
#             raise RuntimeError("`reranker` 인스턴스를 전달하거나 self.reranker가 필요합니다.") from e
#     else:
#         _reranker = reranker

#     reranked_doc_ids, reranked_scores = _reranker.rerank_with_structure(
#         query_graph=query_graph,
#         passage_topk_ids=passage_topk_ids,
#         passage_topk_scores=passage_topk_scores,
#         passage_node_idxs=passage_node_idxs,
#         last_ppr_scores=last_ppr_scores,
#         cfg=cfg,
#         embedder=embedder,
#         scorer=scorer,
#         passage_embeddings=passage_embeddings,
#         entity_embeddings=entity_embeddings,
#         **clean_kwargs,
#     )

#     # 4) 디버깅 정보
#     dbg = {
#         "cfg": {
#             "max_depth": getattr(cfg, "max_depth", None),
#             "per_hop_topk": getattr(cfg, "per_hop_topk", None),
#             "allow_topk": getattr(cfg, "allow_topk", None),
#             "per_node_top_r": getattr(cfg, "per_node_top_r", None),
#             "flow_quantile": getattr(cfg, "flow_quantile", None),
#             "rank_metric": getattr(cfg, "rank_metric", None),
#         },
#         "rerank_kwargs": clean_kwargs,
#     }

#     return reranked_doc_ids, reranked_scores, dbg


from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import igraph as ig
import networkx as nx
from collections import deque, defaultdict, Counter
import random
import logging
from scipy.stats import spearmanr
import torch
import geoopt
from geoopt.manifolds.lorentz import Lorentz
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# *****************************************************************************
# ====[ 통합된 TreeCache 및 BFS 유틸리티 ]====
# *****************************************************************************

# =====[ Global Cache and TreeCache Definition ]=====
_TRE_CACHE: Dict[int, "TreeCache"] = {}


class TreeCache:
    def __init__(self, T: nx.Graph, root: Optional[int] = None, max_k: int = 16):
        if T.number_of_nodes() == 0:
            raise ValueError("Empty tree")
        self.T = T

        if root is None or root not in T:
            try:
                centers = nx.center(T)
                root = centers[0] if centers else list(T.nodes())[0]
            except Exception:
                root = list(T.nodes())[0]
        self.root = root
        self.max_k = max_k

        self.parent: Dict[int, Optional[int]] = {}
        self.level: Dict[int, int] = {}
        self.dist_to_root: Dict[int, float] = {}
        self.nodes = list(T.nodes())
        self.index_of = {v: i for i, v in enumerate(self.nodes)}
        self._dfs_root_fill_all()

        n = len(self.nodes)
        self.up: List[List[int]] = [[-1] * n for _ in range(max_k + 1)]
        for v in self.nodes:
            vi = self.index_of[v]
            p = self.parent.get(v, None)
            self.up[0][vi] = -1 if p is None else self.index_of.get(p, -1)
        for k in range(1, max_k + 1):
            for vi in range(n):
                mid = self.up[k - 1][vi]
                self.up[k][vi] = -1 if mid == -1 else self.up[k - 1][mid]

        self.diameter = self._tree_diameter()
        self.max_dist = max(self.dist_to_root.values()) if self.dist_to_root else 1e-9

    def _dfs_from(self, r: int):
        self.parent.setdefault(r, None)
        self.level.setdefault(r, 0)
        self.dist_to_root.setdefault(r, 0.0)
        st, vis = [r], {r}
        while st:
            u = st.pop()
            for v in self.T.neighbors(u):
                if v in vis:
                    continue
                vis.add(v)
                self.parent[v] = u
                self.level[v] = self.level[u] + 1
                w = float(self.T[u][v].get("weight", 1.0))
                self.dist_to_root[v] = self.dist_to_root[u] + w
                st.append(v)
        return vis

    def _dfs_root_fill_all(self):
        visited = self._dfs_from(self.root)
        if len(visited) != self.T.number_of_nodes():
            for u in self.T.nodes():
                if u not in visited:
                    sub_vis = self._dfs_from(u)
                    visited |= sub_vis
        self.max_dist = max(self.dist_to_root.values()) if self.dist_to_root else 1e-9

    def _tree_diameter(self) -> float:
        if self.T.number_of_nodes() <= 1:
            return 0.0
        try:
            comp_nodes = next(c for c in nx.connected_components(self.T) if self.root in c)
            H = self.T.subgraph(comp_nodes).copy()
        except Exception:
            H = self.T

        adj: Dict[int, List[Tuple[int, float]]] = {u: [] for u in H.nodes()}
        for u, v, data in H.edges(data=True):
            w = float(data.get("weight", 1.0))
            adj[u].append((v, w))
            adj[v].append((u, w))

        def farthest_from(src: int) -> Tuple[int, float]:
            stack: List[Tuple[int, int, float]] = [(src, -1, 0.0)]
            far_node, far_dist = src, 0.0
            while stack:
                u, p, d = stack.pop()
                if d > far_dist:
                    far_node, far_dist = u, d
                for v, w in adj[u]:
                    if v == p:
                        continue
                    stack.append((v, u, d + w))
            return far_node, far_dist

        a, _ = farthest_from(self.root)
        _, diam = farthest_from(a)
        return float(diam)

    def _lift(self, idx: int, steps: int) -> int:
        """안전한 바이너리 리프팅 (idx가 -1이면 그대로 -1 반환)."""
        if idx == -1 or steps <= 0:
            return idx
        bit = 0
        while steps and idx != -1:
            if steps & 1:
                idx = self.up[bit][idx]
            steps >>= 1
            bit += 1
        return idx

    def lca(self, u: int, v: int) -> int:
        """안전한 LCA: 음수 du 처리 및 -1 인덱스 가드."""
        if u not in self.index_of or v not in self.index_of:
            return self.root
        if u == v:
            return u
        if (u not in self.level) or (v not in self.level):
            return self.root

        ui = self.index_of[u]
        vi = self.index_of[v]
        du = self.level[u] - self.level[v]

        if du > 0:
            ui = self._lift(ui, du)
        elif du < 0:
            vi = self._lift(vi, -du)

        if ui == -1 or vi == -1:
            return self.root
        if ui == vi:
            return self.nodes[ui]

        for k in range(self.max_k, -1, -1):
            u2 = self.up[k][ui] if ui != -1 else -1
            v2 = self.up[k][vi] if vi != -1 else -1
            if u2 != v2:
                ui, vi = u2, v2
                if ui == -1 or vi == -1:
                    return self.root

        p = self.up[0][ui] if ui != -1 else -1
        return self.nodes[p] if p != -1 else self.root

    def dist(self, u: int, v: int) -> float:
        if u not in self.dist_to_root or v not in self.dist_to_root:
            a = self.root
            du = self.dist_to_root.get(u, self.dist_to_root.get(a, 0.0))
            dv = self.dist_to_root.get(v, self.dist_to_root.get(a, 0.0))
            return abs(du - dv)
        a = self.lca(u, v)
        return self.dist_to_root[u] + self.dist_to_root[v] - 2.0 * self.dist_to_root[a]

    def radial(self, v: int) -> float:
        """루트로부터의 정규화 반경 [0, 1]."""
        if not self.dist_to_root:
            return 0.0
        d = self.dist_to_root.get(v, None)
        if d is None:
            return 1.0 if (self.max_dist and self.max_dist > 0) else 0.0
        if not self.max_dist or self.max_dist <= 1e-9:
            return 0.0
        r = float(d) / float(self.max_dist)
        return 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)


# ======================================================================================
# StructFeatureExtractor (종속성 만족용)
# ======================================================================================
class StructFeatureExtractor:
    def __init__(self, cache: TreeCache):
        self.C = cache
    pass


# ======================================================================================
# _build_bfs_tree_cache_from_igraph (HyperbolicEmbedder에서 호출되는 유틸리티)
# ======================================================================================
def _build_bfs_tree_cache_from_igraph(
    sg: ig.Graph,
    *,
    backend: str = "bfs",
    ppr_scores: Optional[np.ndarray] = None,
    treerep_ppr_beta: Optional[float] = None,
) -> Optional[TreeCache]:
    if sg.vcount() == 0:
        return None

    cache_key = id(sg)
    if cache_key in _TRE_CACHE:
        return _TRE_CACHE[cache_key]

    gu = sg.as_undirected() if sg.is_directed() else sg
    gids: List[int] = [
        int(x) for x in (gu.vs["global_id"] if "global_id" in gu.vs.attributes() else range(gu.vcount()))
    ]
    has_w = "weight" in gu.es.attribute_names()
    idx2gid = {i: g for i, g in enumerate(gids)}

    if "root_id" in gu.attributes():
        raw_root = gu["root_id"]
        try:
            raw_root = int(raw_root)
        except Exception:
            pass
        if isinstance(raw_root, int) and raw_root in gids:
            root_gid = raw_root
        elif isinstance(raw_root, int) and raw_root in idx2gid:
            root_gid = idx2gid[raw_root]
        else:
            root_gid = gids[0]
    else:
        root_gid = gids[0]

    Tfull = nx.Graph()
    Tfull.add_nodes_from(gids)
    for e in gu.es:
        u, v = idx2gid[int(e.source)], idx2gid[int(e.target)]
        w = float(e["weight"]) if has_w else 1.0
        Tfull.add_edge(u, v, weight=w)

    if Tfull.number_of_edges() == 0:
        T = nx.Graph()
        T.add_nodes_from(gids)
        T.graph["root_id"] = int(root_gid)
        cache = TreeCache(T, root=root_gid, max_k=16)
        _TRE_CACHE[cache_key] = cache
        return cache

    Tbfs = nx.bfs_tree(Tfull, root_gid)
    T = nx.Graph()
    T.graph["root_id"] = int(root_gid)
    T.add_nodes_from(Tbfs.nodes())
    for u, v in Tbfs.edges():
        T.add_edge(u, v, weight=Tfull[u][v].get("weight", 1.0))

    cache = TreeCache(T, root=root_gid, max_k=16)
    _TRE_CACHE[cache_key] = cache
    return cache


# ======================================================================================
# HGNN Layer PLACEHOLDERS (HyperbolicEmbedder에서 사용됨)
# ======================================================================================

class HGCNLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x_lorentz, adj_mat):
        return x_lorentz  # Mock return


class HGATLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x_lorentz, adj_mat):
        return x_lorentz  # Mock return


# =============================================================================
# ====[ Utils & Features ]====
# =============================================================================
def _safe_softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) / max(tau, 1e-8)
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    return ex / max(s, 1e-12)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu, sd = x.mean(), x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def _quantile_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    return (ranks - 0.5) / len(x)


def _entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


# =============================================================================
# ====[ SubgraphConfig & QueryFeatures ]====
# =============================================================================
@dataclass
class SubgraphConfig:
    subgraph_mode: str = "ppr"
    max_depth: int = 2
    per_hop_topk: int = 12
    allow_topk: int = 150
    per_node_top_r: int = 6
    flow_quantile: float = 0.85
    rank_metric: str = "ppr"
    keep_bridges: bool = True
    keep_articulation: bool = True
    keep_top_edge_betw: int = 0


@dataclass
class QueryFeatures:
    H_sem: float          # semantic entropy (불확실성)
    flow_q85: float       # 상위 15% PPR 누적비
    G_betw: float         # betweenness gini 근사
    tree_depth: int       # 추정 depth
    tail_mass: float      # Top-10 밖 의미 질량


def _estimate_graph_hierarchy_metrics(
    last_ppr_scores: Optional[Dict[Any, float]]
) -> Tuple[float, float, int]:
    if not last_ppr_scores:
        return 0.5, 0.45, 2
    scores = np.asarray(sorted(last_ppr_scores.values(), reverse=True), dtype=np.float64)
    if scores.sum() <= 0:
        return 0.5, 0.45, 2
    cum = np.cumsum(scores) / scores.sum()
    idx = int(max(1, math.floor(0.15 * len(scores))) - 1)
    flow_q85 = float(cum[idx])
    G_betw = float(np.clip(0.4 + 0.6 * flow_q85, 0.0, 1.0))
    depth = 3 if flow_q85 > 0.62 else 2
    return flow_q85, G_betw, depth


def _build_features_from_semantic(
    scores_sem: np.ndarray, last_ppr_scores: Optional[Dict[Any, float]]
) -> QueryFeatures:
    p_sem = _safe_softmax(np.asarray(scores_sem, dtype=np.float64), tau=1.0)
    H_sem = _entropy_from_probs(p_sem)
    tail_mass = float(p_sem[10:].sum()) if len(p_sem) > 10 else 0.0
    flow_q85, G_betw, depth = _estimate_graph_hierarchy_metrics(last_ppr_scores)
    return QueryFeatures(
        H_sem=H_sem, flow_q85=flow_q85, G_betw=G_betw, tree_depth=depth, tail_mass=tail_mass
    )


# =============================================================================
# ====[ Adaptive policy ]====
# =============================================================================
@dataclass
class AdaptivePolicy:
    alpha: float
    pin_k: int
    gate_ceiling: float
    sim_mode: str
    attn_topk: Optional[int]
    attn_lambda: Optional[float]
    attn_residual: Optional[float]
    lorentz_temp: Optional[float]
    depth_B: int
    allow_topk_B: int
    per_hop_topk_B: int
    per_node_top_r_B: int
    flow_q_B: float


def _decide_policy(feat: QueryFeatures) -> AdaptivePolicy:
    # 계층성 판단 & 의미 불확실성
    hier = (feat.G_betw > 0.55) or (feat.flow_q85 > 0.65) or (feat.tree_depth >= 3)
    very_uncertain = feat.H_sem > 1.6
    mid_uncertain = feat.H_sem > 1.2

    # alpha: 계층↑ → 구조 가중↑, 의미 확실→ 구조 가중↓
    alpha = 0.70
    if feat.G_betw > 0.55:
        alpha += 0.06
    if feat.H_sem < 1.2:
        alpha -= 0.04
    alpha = float(np.clip(alpha, 0.68, 0.80))

    # pin_k: 의미 불확실할수록 증가
    pin_k = 2 if very_uncertain else (1 if mid_uncertain else 0)

    # gate_ceiling: 계층↑ → 더 낮게 (구조 과적용 억제)
    gate_ceiling = 0.88 if hier else 0.95

    # Stage-B subgraph 세부
    if hier:
        depth_B = 3
        allow_topk_B = 150
        per_hop_topk_B = 10
        per_node_top_r_B = 5
        flow_q_B = 0.88
        if very_uncertain:
            sim_mode = "lorentz"
            attn_topk, attn_lambda, attn_residual, lorentz_temp = 10, 7.5, 0.12, 1.2
        else:
            sim_mode = "tangent"
            attn_topk, attn_lambda, attn_residual, lorentz_temp = 8, 7.5, 0.12, None
    else:
        # 비계층: node_level 유지
        depth_B = 2
        allow_topk_B = 150
        per_hop_topk_B = 12
        per_node_top_r_B = 6
        flow_q_B = 0.85
        sim_mode = "node_level"
        attn_topk, attn_lambda, attn_residual, lorentz_temp = None, None, None, None

    return AdaptivePolicy(
        alpha=alpha,
        pin_k=pin_k,
        gate_ceiling=gate_ceiling,
        sim_mode=sim_mode,
        attn_topk=attn_topk,
        attn_lambda=attn_lambda,
        attn_residual=attn_residual,
        lorentz_temp=lorentz_temp,
        depth_B=depth_B,
        allow_topk_B=allow_topk_B,
        per_hop_topk_B=per_hop_topk_B,
        per_node_top_r_B=per_node_top_r_B,
        flow_q_B=flow_q_B,
    )


# =============================================================================
# ====[ SoftRank calibration ]====
# =============================================================================
def _calibrated_fusion(
    scores_sem: np.ndarray,
    scores_struct: np.ndarray,
    alpha: float,
    tau_sem: float = 0.8,
    tau_struct: float = 0.8,
) -> np.ndarray:
    z_sem = _zscore(scores_sem)
    z_struct = _zscore(scores_struct)
    p_sem = _safe_softmax(z_sem, tau=tau_sem)
    p_struct = _safe_softmax(z_struct, tau=tau_struct)
    fused = (1.0 - alpha) * p_sem + alpha * p_struct
    return fused


# =============================================================================
# ====[ TreeRep bonus(선택) ]====
# =============================================================================
def _apply_treerep_bonus(
    scores: np.ndarray,
    is_bridge: Optional[np.ndarray],
    is_articulation: Optional[np.ndarray],
    gamma: float = 0.03,
) -> np.ndarray:
    if is_bridge is None and is_articulation is None:
        return scores
    bonus = np.zeros_like(scores, dtype=np.float64)
    if is_bridge is not None:
        bonus += gamma * (is_bridge.astype(np.float64))
    if is_articulation is not None:
        bonus += gamma * (is_articulation.astype(np.float64))
    return np.clip(scores + bonus, 0.0, 1.0)


# =============================================================================
# ====[ 2-Stage Adaptive Rerank ]====
# =============================================================================
def adaptive_rerank(
    query_graph: Any,
    passage_topk_ids: List[Any],
    passage_topk_scores: np.ndarray,
    passage_node_idxs: np.ndarray,
    last_ppr_scores: Optional[Dict[Any, float]],
    cfg: SubgraphConfig,
    embedder: Any,
    scorer: Any,
    passage_embeddings: np.ndarray,
    entity_embeddings: np.ndarray,
    reranker: Optional[Any] = None,
    adaptive_alpha: bool = True,  # <--- 이 부분이 누락되었을 가능성이 높습니다.

    **rerank_kwargs: Any,
):

    # 1) None 값은 제거해서 깨끗하게 전달
    clean_kwargs: Dict[str, Any] = {k: v for k, v in rerank_kwargs.items() if v is not None}

    # 2) 안전한 기본값 세팅
    clean_kwargs.setdefault("sim_cac_mode", "lorentz")
    clean_kwargs.setdefault("alpha", 0.78)
    clean_kwargs.setdefault("struct_k", 30)
    clean_kwargs.setdefault("gating_enabled", True)
    clean_kwargs.setdefault("gate_floor", 0.60)
    clean_kwargs.setdefault("gate_ceiling", 0.90)
    clean_kwargs.setdefault("k_rrf", 16)
    clean_kwargs.setdefault("pin_k", 0)
    clean_kwargs.setdefault("treerep_backend", "bfs")
    try:
        clean_kwargs.setdefault("ltr_config", LTR_DISABLED) # noqa: F821
    except NameError:
        pass
    clean_kwargs.setdefault("enable_hgnn_fusion", False)
    
    # ★★★ Step 1: HGNN Type 및 Simple Struct Feats 인자 추출 및 제거 (BUG FIX) ★★★
    # pop()을 사용하여 clean_kwargs에서 인자를 추출 및 제거합니다.
    hgnn_type = clean_kwargs.pop("hgnn_type", "hgcn")
    use_struct_feats = clean_kwargs.pop("use_simple_struct_feats", True) 


    # 3) 실제 리랭킹 호출
    if reranker is None:
        raise RuntimeError("`reranker` 인스턴스를 반드시 전달해야 합니다.")
    
    # ★★★ Step 2: embedder 인스턴스의 속성에 직접 설정 (Embedder로 플래그 전달) ★★★
    # HyperbolicEmbedder는 이 속성을 graph_to_tensor() 등 내부에서 사용해야 합니다.
    try:
        embedder.use_simple_struct_feats = use_struct_feats
        embedder.hgnn_type_used = hgnn_type
    except Exception as e:
         logger.warning(f"Could not set embedder properties: {e}") 

    # ★★★ Step 3: HyperReranker.rerank_with_structure 호출 ★★★
    # clean_kwargs에는 이제 use_simple_struct_feats가 없습니다.
    reranked_doc_ids, reranked_scores,rho, p_value = reranker.rerank_with_structure(
        query_graph=query_graph,
        passage_topk_ids=passage_topk_ids,
        passage_topk_scores=passage_topk_scores,
        passage_node_idxs=passage_node_idxs,
        last_ppr_scores=last_ppr_scores,
        cfg=cfg,
        embedder=embedder,
        scorer=scorer,
        passage_embeddings=passage_embeddings,
        entity_embeddings=entity_embeddings,
        adaptive_alpha=adaptive_alpha,
        **clean_kwargs, # <--- 안전한 인자만 전달
    )

    # 4) 디버깅 정보
    dbg = {
        "cfg": {
            "max_depth": getattr(cfg, "max_depth", None),
            "per_hop_topk": getattr(cfg, "per_hop_topk", None),
            "allow_topk": getattr(cfg, "allow_topk", None),
            "per_node_top_r": getattr(cfg, "per_node_top_r", None),
            "flow_quantile": getattr(cfg, "flow_quantile", None),
            "rank_metric": getattr(cfg, "rank_metric", None),
        },
        "rerank_kwargs": clean_kwargs,
        "hgnn_type_used": hgnn_type,
        "use_struct_feats": use_struct_feats,
        'rho_spearman': float(rho),
        'p_value_spearman': float(p_value) # ★★★ P-value 추가 저장 ★★★   
        }

    return reranked_doc_ids, reranked_scores, dbg