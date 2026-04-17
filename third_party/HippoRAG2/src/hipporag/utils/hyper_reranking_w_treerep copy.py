from __future__ import annotations
import igraph as ig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Set, Callable, Literal, Dict, Any, Mapping
from collections import deque, defaultdict, Counter
import logging, os, json, math, random, time
import pandas as pd
from scipy.stats import spearmanr
import networkx as nx
import geoopt
from geoopt.manifolds.lorentz import Lorentz

logger = logging.getLogger(__name__)

 
# [파일 상단, TreeCache 클래스 정의 이전이나 이후 빈 공간에 추가]

# ======================================================================================
# Hyperbolic Layer Implementations (HGCN & HGAT) — Stable Frozen Variants
# ======================================================================================

# ---------- small utils ----------
def _xavier_(lin: nn.Linear, bias_zero: bool = True):
    nn.init.xavier_uniform_(lin.weight, gain=1.0)
    if lin.bias is not None and bias_zero:
        nn.init.zeros_(lin.bias)

@torch.no_grad()
def _tangent_clip(x: torch.Tensor, clip: float = 5.0):
    if clip is not None and clip > 0:
        return torch.clamp(x, -clip, clip)
    return x

def _gcn_norm(adj: torch.Tensor) -> torch.Tensor:
    # \tilde{A} = A + I, \tilde{D} = deg(\tilde{A}), \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
    N = adj.size(0)
    A_tilde = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
    D_tilde = A_tilde.sum(dim=1).clamp(min=1e-12)
    D_inv_sqrt = torch.diag_embed(D_tilde.pow(-0.5))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

# ---------- segment ops (torch_scatter 없이도 안전하게) ----------
def _segment_max(values: torch.Tensor, index: torch.Tensor, num_segments: int):
    # O(E) 루프지만 서브그래프가 작을 때(struct_k<=30 등) 충분히 빠름
    out = torch.full((num_segments,), float("-inf"), device=values.device)
    for i in range(values.numel()):
        j = index[i].item()
        v = values[i]
        if v > out[j]:
            out[j] = v
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out

def _segment_sum(values: torch.Tensor, index: torch.Tensor, num_segments: int, eps: float = 1e-9):
    out = torch.zeros((num_segments,), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out.clamp_min(eps)

# ======================================================================================
# A. HGCN Layer (Hyperbolic Graph Convolutional Network) — stable frozen
# ======================================================================================
# class HGCNLayer(nn.Module):
#     def __init__(self, in_features, out_features, manifold, bias: bool = False, tangent_clip: float = 5.0):
#         super().__init__()
#         self.manifold = manifold
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
#         # 1) init 후 2) freeze
#         _xavier_(self.linear, bias_zero=True)
#         self.tangent_clip = tangent_clip
#         self.eval()
#         for p in self.parameters():
#             p.requires_grad = False

#     def forward(self, x_lorentz: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
#         N = x_lorentz.size(0)
#         # Lorentz -> tangent
#         x_tangent = self.manifold.logmap0(x_lorentz)
#         x_tangent = _tangent_clip(x_tangent, self.tangent_clip)
#         h = self.linear(x_tangent)

#         # \hat{A}
#         A_hat = _gcn_norm(adj_mat)

#         # aggregate
#         h_agg = A_hat @ h
#         y = self.manifold.expmap0(h_agg)
#         return self.manifold.projx(y)

class HGCNLayer(nn.Module):
    """
    통합 이전 버전과 1:1 동작 동일:
      - reset_parameters() 후 bias=0
      - logmap0 -> Linear -> A_hat@h -> expmap0 -> projx
      - freeze/eval 없음, clip/residual 없음
    """
    def __init__(self, in_features, out_features, manifold, bias=True):
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # === 초기화: 이전 코드와 동일 ===
        self.linear.reset_parameters()
        if self.linear.bias is not None:
            with torch.no_grad():
                self.linear.bias.zero_()  # bias=0

    def forward(self, x_lorentz: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        # dtype/device 정확히 맞추기
        adj_mat = adj_mat.to(dtype=x_lorentz.dtype, device=x_lorentz.device)

        # 1) 탄젠트 맵
        x_tangent = self.manifold.logmap0(x_lorentz)  # (클립/스케일 없음)
        h = self.linear(x_tangent)

        # 2) 정규화 인접행렬: A_hat = D^{-1/2} (A + I) D^{-1/2}
        N = adj_mat.size(0)
        I = torch.eye(N, dtype=adj_mat.dtype, device=adj_mat.device)
        A_tilde = adj_mat + I

        D = A_tilde.sum(dim=1).clamp(min=1e-12)
        D_inv_sqrt = torch.diag_embed(D.pow(-0.5))
        A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        # 3) 집계 후 하이퍼볼릭으로 복귀
        h_aggregated = A_hat @ h
        y = self.manifold.expmap0(h_aggregated)
        return self.manifold.projx(y)
# ======================================================================================
# B. HGAT Layer (Hyperbolic Graph Attention Network) — stable frozen
# ======================================================================================
class HGATLayer(nn.Module):
    """
    Tangent-space attention with optional structural edge bias; then expmap to Lorentz.
    Frozen (non-trainable) but carefully initialized & stabilized.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold,
        n_heads: int = 1,
        concat: bool = False,          # 추천: False(평균) → 스케일 안정
        beta_struct: float = 0.3,      # 구조 바이어스 가중치(너무 크면 불안정)
        residual: float = 0.10,        # tangent residual
        attn_temp: float = 1.0,        # softmax temperature (>=1.0 권장)
        add_undirected: bool = False,
        tangent_clip: float = 5.0,
        freeze_params: bool = True,
        eps: float = 1e-9,
    ):
        super().__init__()
        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"
        self.manifold = manifold
        self.n_heads = n_heads
        self.concat = concat
        self.beta_struct = beta_struct
        self.residual = residual
        self.attn_temp = max(attn_temp, 1e-6)
        self.add_undirected = add_undirected
        self.tangent_clip = tangent_clip
        self.eps = eps

        self.out_per_head = out_features // n_heads
        self.W_v = nn.Linear(in_features, self.out_per_head * n_heads, bias=False)
        self.a = nn.Linear(2 * self.out_per_head, 1, bias=False)
        self.W_o = nn.Linear(self.out_per_head * n_heads, out_features, bias=False) if (concat and n_heads > 1) else None

        # 안정적 초기화
        _xavier_(self.W_v, bias_zero=True)
        _xavier_(self.a, bias_zero=True)
        if self.W_o is not None:
            _xavier_(self.W_o, bias_zero=True)

        if freeze_params:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    # ----- edge utils -----
    def _to_edge_index(self, adj_or_edge: torch.Tensor, N: int) -> torch.Tensor:
        x = adj_or_edge
        if x.dim() == 2 and x.size(0) == x.size(1):                       # dense (N,N)
            ei = torch.nonzero(x > 0, as_tuple=False).t().contiguous()
        elif x.dim() == 2 and x.size(0) == 2:                              # [2,E]
            ei = x
        elif x.dim() == 2 and x.size(1) == 2:                              # [E,2]
            ei = x.t().contiguous()
        elif x.dim() == 2 and x.size(0) > 2:
            ei = x[:2, :]
        else:
            raise ValueError(f"Unsupported edge format: {x.shape}")
        ei = ei.to(dtype=torch.long, device=x.device)
        if ei.numel() > 0:
            ei.clamp_(min=0, max=max(N - 1, 0))
        return ei

    @torch.no_grad()
    def _ensure_self_loop(self, edge_index: torch.Tensor, N: int) -> torch.Tensor:
        dev = edge_index.device
        self_edges = torch.arange(N, device=dev, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        cat = torch.cat([edge_index, self_edges], dim=1)
        return torch.unique(cat.t().contiguous(), dim=0).t().contiguous()

    def _maybe_make_undirected(self, edge_index: torch.Tensor) -> torch.Tensor:
        if not self.add_undirected:
            return edge_index
        ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return torch.unique(ei.t().contiguous(), dim=0).t().contiguous()

    def _fetch_struct_edge_feats(self, struct_edge_feats, edge_index: torch.Tensor, N: int):
        if struct_edge_feats is None:
            return None
        if struct_edge_feats.dim() == 2 and struct_edge_feats.size(0) == struct_edge_feats.size(1):  # dense [N,N]
            s = struct_edge_feats[edge_index[0], edge_index[1]]
        else:
            s = struct_edge_feats.squeeze(-1) if (struct_edge_feats.dim() == 2 and struct_edge_feats.size(1) == 1) else struct_edge_feats
            if s.numel() != edge_index.size(1):
                return None
        s = s.to(edge_index.device, dtype=torch.float32)
        # min-max to [0,1]
        s_min = torch.minimum(s.min(), torch.tensor(0.0, device=s.device))
        s_max = torch.maximum(s.max(), torch.tensor(1.0, device=s.device))
        s = (s - s_min) / (s_max - s_min + self.eps)
        return s

    # ----- forward -----
    def forward(
        self,
        x_lorentz: torch.Tensor,                 # [N, D_L]
        adj_or_edge: torch.Tensor,               # (N,N) or [2,E] or [E,2]
        struct_edge_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = x_lorentz.size(0)

        # 0) edges
        edge_index = self._to_edge_index(adj_or_edge, N)
        edge_index = self._ensure_self_loop(edge_index, N)
        edge_index = self._maybe_make_undirected(edge_index)

        src, tgt = edge_index[0], edge_index[1]   # messages j->i

        # 1) Lorentz -> tangent (clip) -> value proj -> heads
        x_tan = self.manifold.logmap0(x_lorentz)
        x_tan = _tangent_clip(x_tan, self.tangent_clip)
        h_all = self.W_v(x_tan).view(N, self.n_heads, self.out_per_head)

        head_outs = []
        for head in range(self.n_heads):
            h = h_all[:, head, :]                # [N, Dh]
            h_src = h[src]                       # [E, Dh]
            h_tgt = h[tgt]                       # [E, Dh]

            logits = self.a(torch.cat([h_src, h_tgt], dim=-1)).squeeze(-1)  # [E]
            s_bias = self._fetch_struct_edge_feats(struct_edge_feats, edge_index, N)
            if s_bias is not None and self.beta_struct != 0.0:
                logits = logits + self.beta_struct * s_bias

            # stable segment softmax over incoming edges of each target i
            m = _segment_max(logits, tgt, N)                    # [N]
            exp_e = torch.exp((logits - m[tgt]) / self.attn_temp)
            denom = _segment_sum(exp_e, tgt, N, eps=self.eps)   # [N]
            alpha = exp_e / denom[tgt]                          # [E]

            # aggregate to targets
            out = torch.zeros(N, self.out_per_head, device=h.device, dtype=h.dtype)
            out.index_add_(0, tgt, alpha.unsqueeze(-1) * h_src)

            # tangent residual
            out = (1.0 - self.residual) * out + self.residual * h
            head_outs.append(out)

        if self.concat and self.n_heads > 1:
            H = torch.cat(head_outs, dim=-1)                    # [N, H*Dh]
            if self.W_o is not None:
                H = self.W_o(H)                                 # [N, out_features]
        else:
            H = torch.stack(head_outs, dim=1).mean(dim=1)       # [N, Dh] or [N, out_features]

        # 2) tangent -> Lorentz
        y = self.manifold.expmap0(H)
        return self.manifold.projx(y)


# ======================================================================================


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
        self.up = [[-1]*n for _ in range(max_k+1)]
        for v in self.nodes:
            vi = self.index_of[v]
            p = self.parent.get(v, None)
            self.up[0][vi] = -1 if p is None else self.index_of.get(p, -1)
        for k in range(1, max_k+1):
            for vi in range(n):
                mid = self.up[k-1][vi]
                self.up[k][vi] = -1 if mid == -1 else self.up[k-1][mid]

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

        adj = {u: [] for u in H.nodes()}
        for u, v, data in H.edges(data=True):
            w = float(data.get("weight", 1.0))
            adj[u].append((v, w))
            adj[v].append((u, w))

        def farthest_from(src: int) -> tuple[int, float]:
            seen = set([src])
            stack = [(src, -1, 0.0)]
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
    
    def lca(self, u, v):
        if u not in self.index_of or v not in self.index_of:
            return self.root
        if u == v:
            return u
        if (u not in self.level) or (v not in self.level):
            return self.root

        ui = self.index_of[u]; vi = self.index_of[v]
        du = self.level[u] - self.level[v]
        bit = 0
        while du:
            if du & 1:
                ui = self.up[bit][ui] if ui != -1 else -1
            du >>= 1; bit += 1
        if ui == vi:
            return self.nodes[ui]
        for k in range(self.max_k, -1, -1):
            u2 = self.up[k][ui]; v2 = self.up[k][vi]
            if u2 != v2:
                ui, vi = u2, v2
                if ui == -1 or vi == -1:
                    return self.root
        p = self.up[0][ui]
        return self.nodes[p] if p != -1 else self.root

    def dist(self, u, v) -> float:
        if u not in self.dist_to_root or v not in self.dist_to_root:
            a = self.root
            du = self.dist_to_root.get(u, self.dist_to_root.get(a, 0.0))
            dv = self.dist_to_root.get(v, self.dist_to_root.get(a, 0.0))
            return abs(du - dv)
        a = self.lca(u, v)
        return self.dist_to_root[u] + self.dist_to_root[v] - 2.0 * self.dist_to_root[a]

    def radial(self, v) -> float:
        """루트로부터의 정규화 반경 [0,1]."""
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
# StructFeatureExtractor
# ======================================================================================
class StructFeatureExtractor:
    """QA-지향 구조피처(그룹 단위)."""
    def __init__(self, cache: TreeCache):
        self.C = cache

    def radial_gap(self, node, rq: float, tau: float=0.5) -> float:
        gap = abs(self.C.radial(node)-rq)
        return max(0.0, 1.0 - (gap/(tau+1e-9)))

    def geodesic_coherence(self, nodes: List[int], max_pairs: int=200) -> float:
        if len(nodes)<=1: return 1.0
        pairs=[]
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                pairs.append((nodes[i],nodes[j]))
        if len(pairs)>max_pairs:
            pairs=random.sample(pairs, max_pairs)
        s=0.0
        for u,v in pairs:
            s+=self.C.dist(u,v) 
        mean_d = s/max(1,len(pairs))
        if self.C.diameter<=1e-9: return 1.0
        return max(0.0, 1.0 - (mean_d/self.C.diameter))

    def steiner_coverage(self, nodes: List[int]) -> float:
        if len(nodes)<=1: return 1.0
        es=set()
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                u,v=nodes[i],nodes[j]; a=self.C.lca(u,v)
                cur=u
                while cur!=a:
                    p=self.C.parent.get(cur, None)
                    if p is None: break
                    es.add(tuple(sorted((cur,p)))); cur=p
                cur=v
                while cur!=a:
                    p=self.C.parent.get(cur, None)
                    if p is None: break
                    es.add(tuple(sorted((cur,p)))); cur=p
        steiner_edges=len(es)
        avg_depth=sum(self.C.level.get(n,0) for n in nodes)/max(1,len(nodes))
        denom=max(1.0, (len(nodes)-1)+0.5*avg_depth)
        return max(0.0, 1.0 - (steiner_edges/denom))

    def gromov_anchor_summary(self, anchor: int, nodes: List[int]) -> Dict[str,float]:
        if not nodes: return {"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0}
        z = max(self.C.max_dist, 1e-9)
        vals = []
        if anchor not in self.C.T: anchor = self.C.root
        ar = self.C.dist_to_root.get(anchor, 0.0)
        for v in nodes:
            gp = (ar + self.C.dist_to_root.get(v, ar) - self.C.dist(anchor, v)) * 0.5 
            vals.append(max(0.0, gp / z))
        return {"gp_mean":float(np.mean(vals)), "gp_min":min(vals), "gp_max":max(vals)}

    def branch_focus(self, nodes: List[int]) -> float:
        child_deg=[]
        for v in nodes:
            deg=0
            for u in self.C.T.neighbors(v):
                if self.C.parent.get(u,None)==v: deg+=1
            child_deg.append(deg)
        if not child_deg: return 1.0
        c=Counter(child_deg); total=sum(c.values())
        probs=[cnt/total for cnt in c.values()]
        H = -sum(p*math.log(p+1e-12) for p in probs)
        Hmax = math.log(max(1,max(child_deg)+1))
        return 1.0 if Hmax<=1e-9 else max(0.0, 1.0 - H/Hmax)

    def features_for_group(self, nodes: List[int], rq: float, tau: float=0.5, anchor: Optional[int]=None) -> Dict[str,float]:
        if not nodes:
            return {"radial_gap_mean":0.0,"geo_coherence":1.0,"steiner_coverage":1.0,"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0,"branch_focus":1.0}
        nodes = [v for v in nodes if v in self.C.T]
        if not nodes:
            return {"radial_gap_mean":0.0,"geo_coherence":1.0,"steiner_coverage":1.0,"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0,"branch_focus":1.0}
        rads=[self.C.radial(v) for v in nodes]
        radial_gap_mean=float(np.mean([max(0.0,1.0-abs(r- rq)/ (tau+1e-9)) for r in rads]))
        geo=self.geodesic_coherence(nodes)
        cov=self.steiner_coverage(nodes)
        if anchor is None: anchor=min(nodes, key=lambda x:self.C.dist_to_root.get(x,0.0))
        gp=self.gromov_anchor_summary(anchor, nodes)
        bf=self.branch_focus(nodes)
        return { 
            "radial_gap_mean":radial_gap_mean, 
            "geo_coherence":geo, 
            "steiner_coverage":cov, 
            "gp_mean":gp["gp_mean"], 
            "gp_min":gp["gp_min"], 
            "gp_max":gp["gp_max"], 
            "branch_focus":bf, 
        }

# ======================================================================================
# _build_bfs_tree_cache_from_igraph
# ======================================================================================
def _build_bfs_tree_cache_from_igraph( 
    sg: ig.Graph,
    *,
    backend: Literal["treerep", "bfs"] = "bfs", 
    ppr_scores: Optional[np.ndarray] = None,
    treerep_ppr_beta: Optional[float] = None,
) -> Optional[TreeCache]:
    """무방향 서브그래프(ig) → BFS 기반 트리 → TreeCache. (O(V^3) Treerep 로직 제거)"""
    if sg.vcount() == 0:
        return None

    cache_key = id(sg)
    if cache_key in _TRE_CACHE:
        return _TRE_CACHE[cache_key]

    gu = sg.as_undirected() if sg.is_directed() else sg
    gids = [int(x) for x in (gu.vs["global_id"] if "global_id" in gu.vs.attributes() else range(gu.vcount()))]
    has_w = 'weight' in gu.es.attribute_names()
    idx2gid = {i: g for i, g in enumerate(gids)}
    if "root_id" in gu.attributes():
        root_loc = int(gu["root_id"])
        root_gid = idx2gid.get(root_loc, gids[0])
    else:
        root_gid = gids[0]

    Tfull = nx.Graph()
    Tfull.add_nodes_from(gids)
    for e in gu.es:
        u, v = idx2gid[int(e.source)], idx2gid[int(e.target)]
        w = float(e["weight"]) if has_w else 1.0
        Tfull.add_edge(u, v, weight=w)

    if Tfull.number_of_edges() == 0:
        T = nx.Graph(); T.add_nodes_from(gids)
        T.graph["root_id"] = int(root_gid)
        cache = TreeCache(T, root=root_gid, max_k=16)
        _TRE_CACHE[cache_key] = cache
        return cache

    Tbfs = nx.bfs_tree(Tfull, root_gid)
    T = nx.Graph(); T.graph["root_id"] = int(root_gid)
    T.add_nodes_from(Tbfs.nodes())
    for u, v in Tbfs.edges():
        T.add_edge(u, v, weight=Tfull[u][v].get("weight", 1.0))
    cache = TreeCache(T, root=root_gid, max_k=16)
    _TRE_CACHE[cache_key] = cache
    return cache

# ======================================================================================
# MetricsLogger & compute_struct_metrics
# ======================================================================================
class MetricsLogger:
    def __init__(self, save_dir: str, dataset_name: str, fname: str = "struct_metrics.jsonl"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, f"{dataset_name}_{fname}")
        
    def write(self, row: Mapping[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _safe_deg_hist(g: ig.Graph, exclude_virtual=False) -> dict:
    if g.vcount() == 0: return {}
    mask = [True] * g.vcount()
    if exclude_virtual and "is_virtual" in g.vs.attributes():
        mask = [not bool(v) for v in g.vs["is_virtual"]]
    vids = [i for i, ok in enumerate(mask) if ok]
    if not vids: return {}
    degs = g.degree(vids, mode="all")
    hist = defaultdict(int)
    for d in degs: hist[int(d)] += 1
    return {int(k): int(v) for k, v in sorted(hist.items(), key=lambda x: x[0])}

def _gini(arr: List[float]) -> float:
    x = np.asarray(arr, dtype=float)
    if x.size == 0: return 0.0
    x = np.clip(x, 0, None)
    if np.all(x == 0): return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)

def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0: return 0.0
    return float(-np.sum(p * np.log(p)))

def compute_struct_metrics(
    g: ig.Graph, 
    *, 
    graph_role: str, 
    dataset_name: str, 
    query_id: str, 
    root_global_id: Optional[int] = None, 
    ppr_scores: Optional[np.ndarray] = None, 
    exclude_virtual: bool = False
) -> dict:
    if exclude_virtual and "is_virtual" in g.vs.attributes():
        keep_vids = [i for i, flag in enumerate(g.vs["is_virtual"]) if not bool(flag)]
        g2 = g.induced_subgraph(keep_vids) if keep_vids else ig.Graph(directed=g.is_directed())
    else:
        g2 = g
        
    m = {
        "dataset": dataset_name, "query_id": str(query_id), "graph_role": graph_role, 
        "v": int(g2.vcount()), "e": int(g2.ecount()), "directed": bool(g.is_directed()),
    }
    
    if g2.vcount() > 0:
        comps = g2.components(mode="WEAK")
        comp_sizes = [len(c) for c in comps]
        m["comp_cnt"] = int(len(comp_sizes))
        m["giant_ratio"] = float((max(comp_sizes) / g2.vcount()) if comp_sizes else 0.0)
    else:
        m["comp_cnt"] = 0; m["giant_ratio"] = 0.0

    hist = _safe_deg_hist(g2, exclude_virtual=False)
    m["deg_hist"] = hist
    if hist:
        deg_list = []
        for k, v in hist.items(): deg_list.extend([k] * v)
        deg_arr = np.asarray(deg_list, dtype=float)
        m["deg_mean"] = float(np.mean(deg_arr))
        m["deg_max"] = float(np.max(deg_arr))
        m["deg_p95"] = float(np.percentile(deg_arr, 95))
        m["deg_gini"] = _gini(deg_arr.tolist())
        total_deg_sq = float(np.sum(deg_arr ** 2))
        total_deg = float(np.sum(deg_arr))
        m["deg_hhi"] = float(total_deg_sq / (total_deg**2 + 1e-9))
        m["tail_frac_ge3"] = float(np.mean(deg_arr >= 3.0))
    else:
        m.update(dict(deg_mean=0.0, deg_max=0.0, deg_p95=0.0, deg_gini=0.0, deg_hhi=0.0, tail_frac_ge3=0.0))
        
    try: m["star_ratio"] = 0.0
    except Exception: m["star_ratio"] = 0.0
        
    try:
        if g2.vcount() > 1:
            gu = g2.as_undirected() if g2.is_directed() else g2
            comps = gu.components()
            giant = max(comps, key=len) if len(comps) else []
            if len(giant) > 1:
                sub = gu.induced_subgraph(giant)
                dist = np.array(sub.shortest_paths())
                dist = dist[np.isfinite(dist)]
                if dist.size:
                    m["path_len_mean"] = float(np.mean(dist))
                    m["path_len_p95"] = float(np.percentile(dist, 95))
                else: m["path_len_mean"] = 0.0; m["path_len_p95"] = 0.0
            else: m["path_len_mean"] = 0.0; m["path_len_p95"] = 0.0
        else: m["path_len_mean"] = 0.0; m["path_len_p95"] = 0.0
    except Exception: m["path_len_mean"] = 0.0; m["path_len_p95"] = 0.0
        
    try:
        bet = np.asarray(g2.betweenness(), dtype=float) if g2.vcount() else np.array([])
        if bet.size:
            m["betw_p90"] = float(np.percentile(bet, 90))
            m["betw_frac_gt_p90"] = float(np.mean(bet > m["betw_p90"]))
            m["betw_gini"] = _gini(bet.tolist())
        else: m["betw_p90"] = 0.0; m["betw_frac_gt_p90"] = 0.0; m["betw_gini"] = 0.0
    except Exception: m["betw_p90"] = 0.0; m["betw_frac_gt_p90"] = 0.0; m["betw_gini"] = 0.0
        
    try:
        clu = g2.transitivity_avglocal_undirected()
        m["clustering_avg"] = float(0.0 if (clu is None or math.isnan(clu)) else clu)
    except Exception: m["clustering_avg"] = 0.0
        
    if ppr_scores is not None and "global_id" in g2.vs.attributes():
        gids = np.asarray(g2.vs["global_id"])
        pv = np.nan_to_num(np.asarray(ppr_scores, dtype=float), nan=0.0)
        sel = pv[gids] if gids.size else np.array([])
        if sel.size and np.sum(sel) > 0:
            p = sel / np.sum(sel)
            m["ppr_entropy"] = _entropy(p)
        else: m["ppr_entropy"] = 0.0
    else: m["ppr_entropy"] = 0.0
        
    m["long_tail_hint"] = {"deg_gini": m["deg_gini"], "tail_frac_ge3": m["tail_frac_ge3"]}
    return m

def _to_1d_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().flatten()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        return x
    arr = np.asarray(x, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# ======================================================================================
# QueryGraphBuilder, SubgraphBuilder, PPRPruner
# ======================================================================================

class QueryGraphBuilder:
    def __init__(self, root_node: str = "question", directed: bool = True, save_dir: str = f"./outputs/query_graph_analysis", dataset_name: str = "default"):
        self.root_node = root_node
        self.directed = directed
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_logger = MetricsLogger(save_dir=self.save_dir, dataset_name=self.dataset_name)
        self.log_file = os.path.join(self.save_dir, f"{self.dataset_name}_query_graph_degree_distribution.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("query_id,dataset_name,num_nodes,num_edges,degree_distribution\n")
                
    def build(self, triples: list[tuple], query_id: str) -> ig.Graph:
        def norm(x): return str(x).strip()
        
        if not triples:
            g = ig.Graph(directed=self.directed)
            g.add_vertex(name=self.root_node, is_virtual=True)
            g.vs["global_id"] = [-1]
            g["root_id"] = g.vs.find(name=self.root_node).index
            with open(self.log_file, 'a') as f: f.write(f"{query_id},{self.dataset_name},0,0,[]\n")
            return g
            
        valid_triples = []
        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, p, o = map(norm, t)
                if s and p and o and s != o: valid_triples.append((s, p, o))
                
        g = ig.Graph(directed=self.directed)
        if not valid_triples:
            g.add_vertex(name=self.root_node, is_virtual=True)
            g.vs["global_id"] = [-1]
            g["root_id"] = g.vs.find(name=self.root_node).index
            with open(self.log_file, 'a') as f: f.write(f"{query_id},{self.dataset_name},1,0,[]\n")
            return g
            
        node_names = set()
        for s, _, o in valid_triples: node_names.add(s); node_names.add(o)
        node_names.add(self.root_node)
        names_sorted = sorted(node_names)
        
        g.add_vertices(len(names_sorted))
        g.vs["name"] = names_sorted
        g.vs["is_virtual"] = [False] * g.vcount()
        g.vs.find(name=self.root_node)["is_virtual"] = True
        g["root_id"] = g.vs.find(name=self.root_node).index
        name2id = {n: i for i, n in enumerate(g.vs["name"])}
        g.vs["global_id"] = [-1] * g.vcount()
        
        edge_map = defaultdict(list)
        for s, p, o in valid_triples:
            u, v = name2id[s], name2id[o]
            edge_map.setdefault((u, v), []).append(p)
            
        if edge_map:
            uv = list(edge_map.keys())
            g.add_edges(uv)
            rel_lists = [edge_map[k] for k in uv]
            g.es["relations"] = [list(dict.fromkeys(rels)) for rels in rel_lists]
            g.es["relation"] = [rels[0] for rels in g.es["relations"]]
        else:
            g.es["relations"] = []; g.es["relation"] = []
            
        root_id = name2id[self.root_node]
        comps = g.components(mode="WEAK")
        for comp in comps:
            if root_id in comp: continue
            degs = g.degree(comp, mode="all")
            hub_local = int(np.argmax(degs))
            hub_vid = comp[hub_local]
            if g.get_eid(root_id, hub_vid, directed=False, error=False) == -1:
                g.add_edge(root_id, hub_vid)
                g.es[-1]["relation"] = "root"
                g.es[-1]["relations"] = ["root"]

        non_virtual_nodes = [v for v in g.vs if not v["is_virtual"]]
        degrees = g.degree(non_virtual_nodes, mode="all") if non_virtual_nodes else []
        degree_counts = defaultdict(int)
        for d in degrees: degree_counts[d] += 1
        sorted_degrees = sorted(degree_counts.items(), key=lambda item: item[0])
        with open(self.log_file, 'a') as f:
            f.write(f"{query_id},{self.dataset_name},{g.vcount()},{g.ecount()},{sorted_degrees}\n")
            
        try:
            row = compute_struct_metrics(
                g, graph_role="query", dataset_name=self.dataset_name, query_id=str(query_id), 
                root_global_id=int(g["root_id"]) if "root_id" in g.attributes() else None, ppr_scores=None
            )
            self.metrics_logger.write(row)
        except Exception as e:
            logger.warning(f"[QueryGraphBuilder] metrics logging failed: {e}")
            
        g.vs["global_id"] = list(range(g.vcount()))
        g["root_id"] = int(g.vs.find(name=self.root_node).index)
        return g

@dataclass
class SubgraphConfig:
    max_depth: int = 2
    weight_threshold: float = -1e-9
    allow_topk: int = 200
    flow_quantile: float = 0.75
    per_node_top_r: int = 7
    ensure_connectivity: bool = True
    starter_beam: int = 8
    per_hop_topk: int = 10
    rank_metric: Literal["ppr","weight","deg"] = "ppr"
    subgraph_mode: Literal["bfs","ppr"] = "ppr"
    treat_undirected_for_structure: bool = True
    keep_bridges: bool = True
    keep_articulation: bool = True
    keep_top_edge_betw: int = 0 

class SubgraphBuilder:
    def __init__(self, g: ig.Graph, cfg: SubgraphConfig, node_scores: Optional[np.ndarray] = None):
        self.g=g; self.cfg=cfg
        self.node_scores=None if node_scores is None else np.nan_to_num(np.asarray(node_scores,float), nan=0.0)
        self.weighted = ('weight' in g.es.attribute_names())
        self.outstr = None
        if self.weighted:
            self.outstr=np.zeros(g.vcount(), dtype=float)
            for e in g.es: self.outstr[e.source]+=float(e['weight'])

    def _rank_key(self, u:int, v:int, w:float)->float:
        if self.cfg.rank_metric=="ppr" and self.node_scores is not None: return float(self.node_scores[v])
        if self.cfg.rank_metric=="weight" and self.weighted: return float(w)
        if self.cfg.rank_metric=="deg": return float(self.g.degree(v))
        return float(w) if self.weighted else 1.0

    def _collect_candidates_by_bfs(self, root:int, allow_ids:Optional[Set[int]])->List[Tuple[int,int,float]]:
        visited={root}; q=deque([(root,0)]); edges=[]
        mode="ALL"; K=max(1,int(self.cfg.per_hop_topk)) if self.cfg.per_hop_topk else None
        while q:
            u,depth=q.popleft()
            if depth>=self.cfg.max_depth: continue
            neighs=[]
            for eid in self.g.incident(u, mode=mode):
                e=self.g.es[eid]
                v = e.target if e.source==u else e.source
                w=float(e['weight']) if self.weighted else 1.0
                if w<self.cfg.weight_threshold: continue
                if allow_ids is not None and v not in allow_ids: continue
                neighs.append((u,v,w))
            if K is not None and len(neighs)>K:
                neighs.sort(key=lambda t:self._rank_key(*t), reverse=True)
                neighs=neighs[:K]
            for uu,vv,ww in neighs:
                edges.append((uu,vv,ww))
                if vv not in visited:
                    visited.add(vv); q.append((vv, depth+1))
        if not edges and self.cfg.starter_beam>0:
            cand=[]
            for eid in self.g.incident(root, mode=mode):
                e=self.g.es[eid]; v=e.target if e.source==root else e.source
                w=float(e['weight']) if self.weighted else 1.0
                if allow_ids is None or v in allow_ids: cand.append((root,v,w))
            if K is not None and len(cand)>self.cfg.starter_beam:
                cand.sort(key=lambda t:self._rank_key(*t), reverse=True)
                cand=cand[:self.cfg.starter_beam]
            edges=cand
        return edges

    def _collect_candidates_by_ppr(self, root:int, allow_ids:Optional[Set[int]])->List[Tuple[int,int,float]]:
        if self.node_scores is None: return []
        ppr_sorted_vids=np.argsort(self.node_scores)[::-1]
        selected=set()
        for vid in ppr_sorted_vids:
            if allow_ids is not None and vid not in allow_ids: continue
            selected.add(vid)
            if len(selected)>=self.cfg.allow_topk: break
        if 0<=root<self.g.vcount(): selected.add(root)
        edges=[]; seen=set()
        for u in selected:
            for v in self.g.neighbors(u, mode="ALL"):
                if v in selected:
                    key=tuple(sorted((u,v)))
                    if key in seen: continue
                    seen.add(key)
                    eids=self.g.get_eids([(u,v)], directed=False)
                    if eids:
                        w=float(self.g.es[eids[0]].attributes().get('weight',1.0))
                        edges.append((u,v,w))
        return edges

    def collect_candidates(self, root:int, allow_ids:Optional[Set[int]])->List[Tuple[int,int,float]]:
        if self.cfg.subgraph_mode=="bfs": return self._collect_candidates_by_bfs(root, allow_ids)
        if self.cfg.subgraph_mode=="ppr": return self._collect_candidates_by_ppr(root, allow_ids)
        raise ValueError("Unknown subgraph_mode")

    def _preserve_connectivity_layer(self, base_edges: List[Tuple[int,int,float]])->List[Tuple[int,int,float]]:
        if not base_edges: return base_edges
        nodes = sorted({u for u,_,_ in base_edges} | {v for _,v,_ in base_edges})
        G = nx.Graph()
        for n in nodes: G.add_node(n)
        for u,v,w in base_edges: G.add_edge(u,v,weight=float(w))
        extra_edges=set()

        if self.cfg.keep_bridges:
            try:
                for (u,v) in nx.bridges(G):
                    extra_edges.add(tuple(sorted((u,v))))
            except Exception:
                pass

        if self.cfg.keep_articulation:
            try:
                arts=list(nx.articulation_points(G))
                for a in arts:
                    for nb in G.neighbors(a):
                        extra_edges.add(tuple(sorted((a,nb))))
            except Exception:
                pass

        K = max(0, int(self.cfg.keep_top_edge_betw))
        if K>0:
            try:
                k_sample = min(64, max(2, int(np.sqrt(G.number_of_nodes()))))
                ebc = nx.edge_betweenness_centrality(G, k=k_sample, seed=42)
            except Exception:
                ebc = nx.edge_betweenness_centrality(G)
            top = sorted(ebc.items(), key=lambda x:x[1], reverse=True)[:K]
            for (u,v),_ in top:
                extra_edges.add(tuple(sorted((u,v))))

        base_set = {tuple(sorted((u,v))) for u,v,_ in base_edges}
        merged=[]
        base_map = {tuple(sorted((u,v))): float(w) for u,v,w in base_edges}
        for key in base_set | extra_edges:
            w = base_map.get(key, 1.0)
            merged.append((key[0], key[1], w))
        return merged

    def build_subgraph(self, edges: List[Tuple[int,int,float]], root: Optional[int]=None)->ig.Graph:
        edges = self._preserve_connectivity_layer(edges)
        sg = ig.Graph(directed=False)
        if not edges:
            sg.add_vertex(name=("root" if root is None else self.g.vs[root]['name']))
            sg.vs['global_id']=[root if root is not None else -1]
            return sg
        nodes=sorted({u for u,_,_ in edges}|{v for _,v,_ in edges})
        idx={n:i for i,n in enumerate(nodes)}
        sg.add_vertices(len(nodes))
        sg.vs['name']=[self.g.vs[n]['name'] for n in nodes]
        sg.vs['global_id']=nodes
        if root is not None: sg["root_id"]=int(root) if root in nodes else -1
        sg.add_edges([(idx[u],idx[v]) for u,v,_ in edges])
        sg.es['weight']=[w for *_,w in edges]
        import numpy as _np
        sg.simplify(multiple=True, loops=True, combine_edges=_np.sum)
        return sg

class PPRPruner:
    def __init__(self, g: ig.Graph, pi: Optional[np.ndarray], cfg: SubgraphConfig):
        self.g = g
        self.pi = np.asarray(pi, dtype=float) if pi is not None else None
        self.cfg = cfg
        self.weighted = ('weight' in g.es.attribute_names())
        self.outdeg = np.array(g.outdegree() if g.is_directed() else g.degree(), dtype=float)
        
        if self.weighted:
            self.outstr = np.zeros(g.vcount(), dtype=float)
            for e in g.es: self.outstr[e.source] += float(e['weight'])
        else:
            self.outstr = None

    def _P(self, u: int, eid: int) -> float:
        if self.weighted:
            if self.g.es[eid].source != u: return 0.0
            w = float(self.g.es[eid]['weight'])
            denom = self.outstr[u]
            return (w/denom) if denom > 0 else 0.0
        d = self.outdeg[u]
        return (1.0/d) if d > 0 else 0.0

    def _flow(self, u: int, v: int) -> float:
        if self.pi is None: return 0.0
        eid = self.g.get_eid(u, v, directed=True, error=False)
        if eid == -1: return 0.0
        return float(self.pi[u]) * self._P(u, eid)

    def prune(self, root: int, candidate_edges: List[Tuple[int,int,float]]) -> List[Tuple[int,int,float]]:
        if not candidate_edges or self.pi is None: return candidate_edges
        flows = np.array([self._flow(u, v) for u, v, _ in candidate_edges], dtype=float)
        flows = np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0)
        kept_idx = np.arange(len(candidate_edges))

        if self.cfg.flow_quantile is not None:
            finite = np.isfinite(flows)
            q = float(np.quantile(flows[finite], self.cfg.flow_quantile)) if finite.any() else 0.0
            cut = max(self.cfg.weight_threshold, q)
            mask = (flows >= cut)
            kept_idx = kept_idx[mask]
            if kept_idx.size == 0:
                med = float(np.quantile(flows[finite], 0.5)) if finite.any() else 0.0
                cut = max(self.cfg.weight_threshold, med)
                kept_idx = np.where(flows >= cut)[0]
            if kept_idx.size == 0:
                N = min(10, len(candidate_edges))
                kept_idx = np.argsort(-flows)[:N]

        kept = [candidate_edges[i] for i in kept_idx]
        r = self.cfg.per_node_top_r
        if r is None or r <= 0: return kept
        
        by_src: dict[int, List[Tuple[int,float]]] = {}
        for i in kept_idx:
            u, v, _ = candidate_edges[i]
            by_src.setdefault(u, []).append((i, flows[i]))
            
        final_idx: List[int] = []
        for u, items in by_src.items():
            items.sort(key=lambda x: x[1], reverse=True)
            r_eff = min(r, len(items))
            th = items[r_eff - 1][1]
            final_idx.extend([i for (i, f) in items if f >= th])

        if self.cfg.allow_topk is not None and len(final_idx) > self.cfg.allow_topk:
            order = np.argsort(-flows[final_idx])
            final_idx = [final_idx[i] for i in order[:self.cfg.allow_topk]]
            
        return [candidate_edges[i] for i in final_idx]

# # ======================================================================================
# # Hyperbolic Smoothing Layer (HSL) — 비학습형
# # ======================================================================================
# class HGCNLayer(torch.nn.Module):
#     def __init__(self, in_features, out_features, manifold: Lorentz, bias=True):
#         super().__init__()
#         self.manifold = manifold
#         self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.linear.reset_parameters()
#         if self.linear.bias is not None:
#             self.linear.bias.data.zero_() 

#     def forward(self, x_lorentz, adj_mat):
#         # 1. Tangent → Lorentz (안전)
#         x_tangent = self.manifold.logmap0(x_lorentz)  # (안정적 재투영을 위해 유지)
#         h = self.linear(x_tangent)

#         # 2. 정규화 인접 행렬 (self-loop 포함)
#         D_diag = adj_mat.sum(dim=1).clamp(min=1e-12)
#         D_inv_sqrt = torch.diag_embed(D_diag.pow(-0.5))
#         A_hat = D_inv_sqrt @ (adj_mat + torch.eye(adj_mat.size(0), device=adj_mat.device)) @ D_inv_sqrt
        
#         h_aggregated = A_hat @ h
#         y = self.manifold.expmap0(h_aggregated)
#         return self.manifold.projx(y)

# ======================================================================================
# HyperbolicEmbedder (구조 보조피처 + Fréchet mean pooling)
# ======================================================================================
class HyperbolicEmbedder:
    
    def __init__(self, hr_instance, embed_dim=64, curvature=1.0, device="cpu",
                 use_simple_struct_feats: bool=False, struct_feat_scale: float=0.2,
                 hgnn_type: str = "hgcn"): # ★★★ HGNN Type 인자 추가 ★★★
        
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.embed_dim = embed_dim
        self.device = device
        self.hr = hr_instance
        
        # ★★★ Simple Struct Feats 관련 속성 ★★★
        self.use_simple_struct_feats = use_simple_struct_feats
        self.struct_feat_scale = float(struct_feat_scale)
        self.hgnn_type = hgnn_type
        
        # ★★★ HGNN Layer 동적 초기화 ★★★
        input_dim = self.embed_dim + (2 if use_simple_struct_feats else 0)

        if hgnn_type.lower() == "hgat":
            self.hgnn_fusion = HGATLayer(
                in_features=input_dim, out_features=self.embed_dim, 
                manifold=self.manifold,  n_heads=1
            ).to(device)
            logger.info(f"Initialized HGATLayer (Input Dim: {input_dim}).")
        elif hgnn_type.lower() == "hgcn":
            self.hgnn_fusion = HGCNLayer(
                in_features=input_dim, out_features=self.embed_dim, 
                manifold=self.manifold, bias=True
            ).to(device)
            logger.info(f"Initialized HGCNLayer (Input Dim: {input_dim}).")
        else:
            raise ValueError(f"Unknown HGNN type: {hgnn_type}. Use 'hgcn' or 'hgat'.")


        self._projW: Optional[torch.Tensor] = None
        torch.manual_seed(1234)

    # ----- 안정 유틸 -----
    def _safe_projx(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return self.manifold.projx(y)

    def _frechet_mean(self, X: torch.Tensor, iters: int = 10, lr: float = 1.0) -> torch.Tensor:
        """
        Lorentz 공간에서의 근사 Fréchet mean. 간단한 급강하 업데이트.
        X: [N, D]
        """
        if X.dim() == 1:
            return X
        m = X[0:1]  # 초기값
        for _ in range(max(1, iters)):
            # 로그맵 평균 방향
            V = self.manifold.logmap(m, X)  # [1,N,D] or [N,D] depending impl; geoopt에서는 logmap(x,y)
            if V.dim() == 3:
                V = V.squeeze(0)
            grad = V.mean(dim=0, keepdim=True)
            m = self.manifold.expmap(m, lr * grad)
            m = self._safe_projx(m)
        return m.squeeze(0)

    # ----- 트리 캐시 -----
    def _get_bfs_tree(self, g: ig.Graph, ppr_scores: Optional[np.ndarray]=None, treerep_ppr_beta: Optional[float]=None):
        n=g.vcount()
        cache = _build_bfs_tree_cache_from_igraph(
            g.as_undirected() if g.is_directed() else g,
            backend="bfs", ppr_scores=ppr_scores, treerep_ppr_beta=treerep_ppr_beta
        )
        T = cache.T if cache is not None else nx.Graph()
        D = np.zeros((n, n), dtype=np.float32)
        return D, T, cache

    def _make_simple_struct_feats(self, g: ig.Graph, cache: Optional[TreeCache]) -> np.ndarray:
        """
        ★ 간단한 구조 보조피처: [depth_norm, radial] 2D
        - depth_norm = level / (max_level+1e-9)
        - radial = cache.radial(node_global_id)
        """
        N = g.vcount()
        feats = np.zeros((N, 2), dtype=np.float32)
        if cache is None or N == 0:
            return feats

        # igraph의 vertex global_id 가 TreeCache 노드 id와 일치한다고 가정 (상위에서 세팅됨)
        gids = g.vs["global_id"] if "global_id" in g.vs.attributes() else list(range(N))
        # max depth
        max_level = 0
        for v in cache.T.nodes():
            lv = cache.level.get(v, 0)
            if lv > max_level:
                max_level = lv
        denom = float(max(1, max_level))

        for i, gid in enumerate(gids):
            lv = cache.level.get(gid, 0)
            depth_norm = float(lv) / denom
            radial = float(cache.radial(gid))
            feats[i, 0] = depth_norm
            feats[i, 1] = radial
        # 스케일 적용
        return (self.struct_feat_scale * feats).astype(np.float32)

    def _project_and_stabilize(self, X: np.ndarray)->torch.Tensor:
        if X.size==0:
            return torch.zeros((1,self.embed_dim), dtype=torch.float32, device=self.device)
        Z=torch.tensor(X, dtype=torch.float32, device=self.device)
        n=torch.linalg.norm(Z, dim=1, keepdim=True)+1e-6
        Z=Z*torch.clamp(3.0/n, max=1.0)
        d_in=Z.size(1); d_out=self.embed_dim
        
        if d_in != d_out:
            if self._projW is None or self._projW.size()!=torch.Size([d_in, d_out]):
                W=torch.empty(d_in, d_out, device=self.device)
                torch.nn.init.orthogonal_(W)
                self._projW=W
            Z=Z@self._projW
        return Z

    # ----- 그래프 → 텐서 -----
    def graph_to_tensor(self, g: ig.Graph, ppr_scores: Optional[np.ndarray]=None,
                        passage_embeddings: Optional[np.ndarray]=None,
                        entity_embeddings: Optional[np.ndarray]=None,
                        use_semantic_features: bool=True,
                        use_treerep_metric: bool=False, 
                        treerep_ppr_beta: Optional[float]=None,
                        enable_hgnn_fusion: bool=False) -> torch.Tensor: 
        if g.vcount() == 0: 
            return torch.zeros(1,self.embed_dim, device=self.device)
        
        X = None 
        num_nodes = g.vcount()

        # --- Semantic Features ---
        if use_semantic_features and passage_embeddings is not None and passage_embeddings.size > 0:
            D_embed = passage_embeddings.shape[1]
            X_sem = np.zeros((num_nodes, D_embed), dtype=np.float32)
            query_node_name = "question"
            for i, node_name in enumerate(g.vs["name"]):
                if node_name == query_node_name and entity_embeddings is not None and entity_embeddings.size > 0:
                    X_sem[i] = entity_embeddings[0]
                elif node_name.startswith("chunk-") or node_name.startswith("entity-"):
                    try:
                        passage_id = int(node_name.split('-')[-1])
                        idx_in_array = passage_id % passage_embeddings.shape[0]
                        X_sem[i] = passage_embeddings[idx_in_array]
                    except ValueError:
                        pass
            if np.sum(np.abs(X_sem)) > 1e-6:
                X = X_sem
            else:
                X = np.zeros((num_nodes, D_embed), dtype=np.float32)

        if X is None or X.size == 0:
            return torch.zeros(1, self.embed_dim, device=self.device)

        # --- ★ 간단 구조 보조피처 concat ---
        D, T, cache = self._get_bfs_tree(g, ppr_scores, treerep_ppr_beta)
        if self.use_simple_struct_feats:
            S2 = self._make_simple_struct_feats(g, cache)  # [N,2]
            # concat → 투영 이전에 결합
            X = np.concatenate([X, S2], axis=1)

        # 1) 선형 투영 + 안정화
        Zlin = self._project_and_stabilize(X)  # [N, D]

        # 2) (옵션) 비학습형 Hyperbolic Smoothing
        if enable_hgnn_fusion and g.ecount() > 0 and Zlin.size(0) > 1 and Zlin.size(1) == self.embed_dim:
            x_L = self._safe_projx(self.manifold.expmap0(Zlin))
            edges = g.get_edgelist()
            if len(edges) == 0:
                return Zlin
            edge_indices = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
            num_nodes = g.vcount()
            adj_mat = torch.zeros((num_nodes, num_nodes), device=self.device)
            adj_mat[edge_indices[0], edge_indices[1]] = 1.0
            adj_mat[edge_indices[1], edge_indices[0]] = 1.0
            fused_L = self.hgnn_fusion(x_L, adj_mat)
            # fused_L = self.hgnn_fusion(x_L, adj_mat, struct_edge_feats=edge_bias_dense_or_edgewise)

            # Tangent으로 되돌려 사용
            return self.manifold.logmap0(fused_L)
        return Zlin

    # ----- 임베딩 & 풀링 -----
    def embed_lorentz(self, g: ig.Graph, ppr_scores=None, passage_embeddings=None,
                      entity_embeddings=None, use_semantic_features=True,
                      use_treerep_metric=False, treerep_ppr_beta=None,
                      enable_hgnn_fusion: bool=False, pool: Literal["fmean","max"]="fmean")->torch.Tensor:
        if g.vcount()==0: 
            return None
        t_nodes = self.graph_to_tensor(
            g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
            entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
            use_treerep_metric=use_treerep_metric, treerep_ppr_beta=treerep_ppr_beta,
            enable_hgnn_fusion=enable_hgnn_fusion
        )
        with torch.no_grad():
            xL = self._safe_projx(self.manifold.expmap0(t_nodes))  # [N,D]
            if xL.dim()==2 and xL.size(0)>1:
                if pool == "fmean":
                    pooled = self._frechet_mean(xL, iters=8, lr=0.8)
                    pooled = self._safe_projx(pooled)
                else:
                    pooled, _ = torch.max(xL, dim=0, keepdim=False)
                    pooled = self._safe_projx(pooled)
            else:
                pooled = xL.squeeze(0)
            # 빈 벡터 방지
            if torch.linalg.norm(self.manifold.logmap0(pooled.unsqueeze(0))).item()<1e-8:
                noise = 1e-3*torch.randn_like(pooled)
                pooled = self._safe_projx(self.manifold.expmap0(self.manifold.logmap0(pooled.unsqueeze(0))+noise).squeeze(0))
            return pooled

    def embed_nodes_lorentz(self, g: ig.Graph, enable_hgnn_fusion: bool=False, **kw)->torch.Tensor:
        t_fused = self.graph_to_tensor(g, enable_hgnn_fusion=enable_hgnn_fusion, **kw)
        with torch.no_grad():
            nmap=self._safe_projx(self.manifold.expmap0(t_fused))
            return nmap

    def logmap0_safe(self, y: torch.Tensor)->torch.Tensor:
        t=self.manifold.logmap0(y)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def shared_standardize(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        X = torch.stack(tensors, dim=0)  # [B, D]
        mu = X.mean(dim=0, keepdim=True)
        sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        Y = [(t - mu.squeeze(0)) / sd.squeeze(0) for t in tensors]
        return Y

    def embed_and_logmap0(self, g: ig.Graph, enable_hgnn_fusion: bool=False, **kw) -> torch.Tensor:
        t_nodes = self.graph_to_tensor(g, enable_hgnn_fusion=enable_hgnn_fusion, **kw)
        with torch.no_grad():
            if t_nodes.dim() == 2 and t_nodes.size(0) > 1:
                x_lorentz = self._safe_projx(self.manifold.expmap0(t_nodes))
                # Fréchet mean 기반 풀링 후 tangent로
                pooled = self._frechet_mean(x_lorentz, iters=8, lr=0.8)
                pooled = self._safe_projx(pooled.unsqueeze(0)).squeeze(0)
                return self.logmap0_safe(pooled.unsqueeze(0)).squeeze(0)
            elif t_nodes.dim() == 1 or (t_nodes.dim() == 2 and t_nodes.size(0) == 1):
                return t_nodes.squeeze(0)
            return torch.zeros(self.embed_dim, device=self.device)

    def attention_pool_tangent(self, tq: torch.Tensor, td_nodes: torch.Tensor, lam: float=6.0, topk: Optional[int]=12, residual: float=0.2) -> torch.Tensor:
        sims = F.cosine_similarity(td_nodes, tq.unsqueeze(0), dim=-1)  # [N]
        w = torch.sigmoid(lam * sims)
        if topk is not None and topk < w.numel():
            k = max(1, int(topk))
            idx = torch.topk(w, k=k, dim=0).indices
            mask = torch.zeros_like(w); mask[idx] = 1.0
            w = w * mask
        w = residual + (1.0 - residual) * w
        s = w.sum()
        if s <= 1e-6:
            return td_nodes.mean(dim=0)
        w = w / s
        return (td_nodes * w.unsqueeze(-1)).sum(dim=0)

# ======================================================================================
# StructralSimilarity, LTRConfig, HyperReranker
# ======================================================================================
class StructralSimilarity:
    def __init__(self, mode="cosine"):
        assert mode in ["cosine", "l2", "dot"], "Invalid similarity mode"
        self.mode = mode

    def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(scores, dtype=float)
        if x.size == 0: return x
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        if hi - lo < 1e-8: return np.full_like(x, 0.5)
        y = (x - lo) / (hi - lo + 1e-8)
        y = np.clip(y, 0.0, 1.0)
        return y

    def compute(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.mode == "cosine": return F.cosine_similarity(q, d, dim=-1)
        if self.mode == "dot": return torch.sum(q * d, dim=-1)
        return -torch.norm(q - d, dim=-1)

    def compute_listwise(self, query_list: List[torch.Tensor], doc_list: List[torch.Tensor]) -> torch.Tensor:
        assert len(query_list) == len(doc_list)
        sims = []
        for q, d in zip(query_list, doc_list):
            sim = self.compute(q.unsqueeze(0), d.unsqueeze(0)).squeeze()
            sims.append(sim)
        return torch.stack(sims)

    def robust_normalize_struct(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0: return x
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(x, 10), np.percentile(x, 90)
        if hi - lo < 1e-8: return np.full_like(x, 0.5)
        x = np.clip(x, lo, hi)
        y = (x - lo) / (hi - lo + 1e-9)
        return y

@dataclass
class LTRConfig:
    enabled: bool = False
    weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.5, 0.5])
    bias: float = 0.0
    blend: float = 0.5

class HyperReranker:
    
    class IDMapper:
        @staticmethod
        def docids_to_vids(doc_ids: Iterable[int], passage_node_idxs: List[int]) -> List[int]:
            return [int(passage_node_idxs[d]) for d in doc_ids]

    def __init__(self, g: ig.Graph, save_dir: str = "./outputs/subgraph_analysis", dataset_name: str = "default"):
        self.g = g
        self.metrics_logger = MetricsLogger(save_dir=save_dir, dataset_name=dataset_name)
        self.cfg = None
        self.dataset_name = dataset_name
        self.query_id = "unknown"

    @staticmethod
    def make_allow_ids(g: ig.Graph, seeds_vids: List[int], ppr_scores: Optional[np.ndarray], allow_topk: int) -> Optional[Set[int]]:
        if ppr_scores is None or not allow_topk or allow_topk <= 0: return None
        ppr = np.nan_to_num(np.asarray(ppr_scores, float), nan=0.0, posinf=0.0, neginf=0.0)
        topN = np.argsort(-ppr)[:allow_topk].tolist()
        allow = set(topN)
        for s in seeds_vids:
            for nb in g.neighbors(s, mode="all"): allow.add(nb)
        return allow

    def build_pruned_subgraphs_for_topk_passages(
        self, topk_doc_ids: List[int], topk_scores: List[float], passage_node_idxs: List[int], 
        last_ppr_scores: Optional[np.ndarray], cfg: SubgraphConfig, head_k: int = 30,
    ) -> Tuple[List[ig.Graph], List[int], List[int]]:
        if not topk_doc_ids: return [], [], []
        self.cfg = cfg
        
        sem = np.asarray(topk_scores, float)
        head_idx = np.argsort(-sem)[: min(head_k, len(topk_doc_ids))]
        head_doc_ids = [topk_doc_ids[i] for i in head_idx]
        
        head_vids = self.IDMapper.docids_to_vids(head_doc_ids, passage_node_idxs) 
        
        allow_ids = self.make_allow_ids(self.g, head_vids, last_ppr_scores, cfg.allow_topk)
        
        builder = SubgraphBuilder(self.g, cfg, node_scores=last_ppr_scores)
        pruner = PPRPruner(self.g, last_ppr_scores, cfg)
        subgraphs: List[ig.Graph] = []
        
        for rid in head_vids:
            cands = builder.collect_candidates(root=rid, allow_ids=allow_ids)
            pruned = pruner.prune(root=rid, candidate_edges=cands) if cfg.subgraph_mode == "ppr" else cands
            sg = builder.build_subgraph(pruned, root=rid)
            try:
                row = compute_struct_metrics(
                    sg, graph_role="subgraph", 
                    dataset_name=self.dataset_name, 
                    query_id=str(self.query_id), 
                    root_global_id=rid, ppr_scores=last_ppr_scores , exclude_virtual=False
                )
                row.update({ 
                    "root_doc_vid": int(rid), 
                    "root_doc_id": int(head_doc_ids[len(subgraphs)]) if len(subgraphs) < len(head_doc_ids) else None, 
                    "subgraph_idx": int(len(subgraphs)), 
                    "collect_mode": self.cfg.subgraph_mode
                })
                self.metrics_logger.write(row)
            except Exception as e:
                logger.warning(f"[SubgraphMetrics] logging failed: {e}")
            subgraphs.append(sg)
        return subgraphs, head_doc_ids, head_vids

    def _struct_scores_tangent(
        self, query_graph: ig.Graph, subgraphs: List[ig.Graph], embedder: "HyperbolicEmbedder", scorer: "StructralSimilarity", 
        ppr_scores: Optional[np.ndarray] = None, passage_embeddings: Optional[np.ndarray] = None, entity_embeddings: Optional[np.ndarray] = None, 
        use_semantic_features: bool = True, use_treerep_metric: bool = False,
        attn_pooling: bool = True, attn_lambda: float = 6.0, attn_topk: Optional[int] = 12, attn_residual: float = 0.2,
        enable_hgnn_fusion: bool=False 
    ) -> np.ndarray:
        with torch.no_grad():
            tq = embedder.embed_and_logmap0(
                query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
                enable_hgnn_fusion=enable_hgnn_fusion 
            )
            tdocs = []
            for g in subgraphs:
                d_nodes_L = embedder.embed_nodes_lorentz(
                    g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                    entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
                    enable_hgnn_fusion=enable_hgnn_fusion 
                )
                if d_nodes_L.numel() == 0:
                    tdocs.append(torch.zeros_like(tq)); continue
                td_nodes = embedder.logmap0_safe(d_nodes_L) # [N, D] Tangent Vector
                if attn_pooling:
                    pooled = embedder.attention_pool_tangent(
                        tq, td_nodes, lam=attn_lambda, topk=attn_topk, residual=attn_residual
                    )
                else:
                    pooled, _ = torch.max(td_nodes, dim=0, keepdim=False) 
                tdocs.append(pooled) # [D] Vector
            
            tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
            
            q_list = [tq_std for _ in range(len(tdocs_std))]
            struct_head = scorer.compute_listwise(q_list, tdocs_std)
            return _to_1d_numpy(struct_head)

    def _struct_scores_lorentz(
        self, query_graph: ig.Graph, subgraphs: List[ig.Graph], embedder: "HyperbolicEmbedder", temp: float = 2.0, 
        ppr_scores: Optional[np.ndarray] = None, passage_embeddings: Optional[np.ndarray] = None, entity_embeddings: Optional[np.ndarray] = None, 
        use_semantic_features: bool = True, use_treerep_metric: bool = False,
        enable_hgnn_fusion: bool=False 
    ) -> np.ndarray:
        with torch.no_grad():
            qL = embedder.embed_lorentz(
                query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
                enable_hgnn_fusion=enable_hgnn_fusion, pool="fmean"
            )
            dL = [
                embedder.embed_lorentz(
                    g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                    entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
                    enable_hgnn_fusion=enable_hgnn_fusion, pool="fmean"
                ) for g in subgraphs
            ]
            qL = qL if qL is not None else torch.zeros(embedder.embed_dim, device=embedder.device)
            dL_tensors = []
            for x in dL:
                if x is None: dL_tensors.append(torch.zeros(embedder.embed_dim, device=embedder.device))
                else: dL_tensors.append(x)
            dists2 = torch.stack([
                embedder.manifold.dist2(qL.unsqueeze(0), x.unsqueeze(0)).squeeze() for x in dL_tensors
            ])
            d2 = _to_1d_numpy(dists2)
            sim = -d2
            if sim.size:
                norm_sim = StructralSimilarity().robust_normalize_struct(sim)
                sim = 1.0 / (1.0 + np.exp(-((norm_sim - 0.5) / (1.0 / temp))))
            return sim

    def _struct_scores_struct_feat(
        self, query_graph: ig.Graph, subgraphs: List[ig.Graph], query_radial_anchor: Optional[float] = None,
        radial_tau: float = 0.5, ppr_scores: Optional[np.ndarray] = None, treerep_backend: Literal["treerep","bfs"] = "bfs",
        treerep_ppr_beta: Optional[float] = None,
    ) -> np.ndarray:
        logger.warning("O(V^2) cost in _struct_scores_struct_feat: Disabling in favor of Tangent/LTR.")
        return np.zeros(len(subgraphs), dtype=float)

    def _pick_landmarks_diverse(self, cache: TreeCache, L:int=10)->List[int]:
        nodes=list(cache.T.nodes())
        if not nodes: return []
        S=[cache.root]
        def score(v,S):
            base=min(cache.dist(v,s) for s in S)
            deg=cache.T.degree(v)
            return base - 0.1*deg
        while len(S)<min(L,len(nodes)):
            cand=max(nodes, key=lambda v: score(v,S))
            if cand in S:
                alt=random.choice([x for x in nodes if x not in S])
                S.append(alt)
            else:
                S.append(cand)
        return S

    def _struct_scores_gromov(self, query_graph: ig.Graph, subgraphs: List[ig.Graph], landmark_k:int=10,
                              ppr_scores: Optional[np.ndarray]=None,
                              treerep_backend: Literal["treerep","bfs"]="bfs",
                              treerep_ppr_beta: Optional[float]=None)->np.ndarray:
        logger.warning("O(V^2) cost in _struct_scores_gromov: Disabling in favor of Tangent/LTR.")
        return np.zeros(len(subgraphs), dtype=float)

    def _compute_gate(self, sem_head: np.ndarray, struct_head: np.ndarray,
                        gating_enabled: bool, gate_floor: float, gate_ceiling: float) -> Tuple[float, np.ndarray]:
            rho = 0.0
            p_value = 1.0
            
            def is_const(x): return x.size == 0 or np.allclose(x, x[0])
            
            if gating_enabled and not (is_const(sem_head) or is_const(struct_head)):
                try:
                    # ★★★ 스피어만 상관계수 및 P-value 계산 (기존 로직 유지) ★★★
                    r_result = spearmanr(sem_head, struct_head)
                    r = float(r_result.correlation)
                    p_value = float(r_result.pvalue)
                    if np.isnan(r): r = 0.0
                    if np.isnan(p_value): p_value = 1.0
                    rho = r
                except Exception:
                    rho = 0.0
                    p_value = 1.0

                # =========================================================================
                # ★★★ [수정된 게이팅 로직]: rho -> 보완성(Complementarity) 기반 ★★★
                # =========================================================================
                
                rho_clipped = np.clip(rho, -1.0, 1.0)
                
                # --- 파라미터 ---
                # BASE_GATE_MIN: 구조적 신호가 중복/충돌 시 가지는 최소 가중치 (0.70)
                BASE_GATE_MIN = 0.70  
                # MAX_RANGE: 게이트의 최대 변동 범위 (1.00 - 0.70 = 0.30)
                MAX_RANGE = 0.30  
                # MIN_GATE_CONFLICT: 음의 상관(충돌) 시 적용할 절대 최소 가중치 (0.50)
                MIN_GATE_CONFLICT = 0.50 

                if rho_clipped >= 0.0:
                    # 1. 양의 상관관계 또는 무상관 (rho >= 0)
                    #    보완성 = 1.0 - rho. rho=0일 때 1.0 (최대 보완), rho=1.0일 때 0.0 (최대 중복)
                    complementarity = 1.0 - rho_clipped
                    gate = BASE_GATE_MIN + (complementarity * MAX_RANGE)
                    
                else: # rho_clipped < 0.0
                    # 2. 음의 상관관계 (충돌): 강력하게 억제하여 안정성 우선
                    #    충돌 강도 = abs(rho). rho=-1.0일 때 1.0 (최대 충돌)
                    conflict_strength = abs(rho_clipped)
                    
                    # BASE_GATE_MIN에서 시작하여 충돌이 강해질수록 0.50까지 감소
                    gate = BASE_GATE_MIN - (conflict_strength * (BASE_GATE_MIN - MIN_GATE_CONFLICT))
                    gate = max(MIN_GATE_CONFLICT, gate) # 절대 최소값 0.50 보장
                
                # (선택적) 기존의 tanh 및 floor/ceiling 클리핑을 제거하고,
                # 계산된 gate 값을 최종적으로 floor/ceiling 범위에 맞춰 조정 (안전성 유지)
                gate = float(np.clip(gate, gate_floor, gate_ceiling))
                
                # =========================================================================

            else:
                # Gating 비활성화 또는 데이터가 상수일 경우, 기존 로직 유지
                gate = float(np.clip(0.6, gate_floor, gate_ceiling)) # 기존 tanh()의 기본값 근사
                
            return gate, np.array([rho, p_value], dtype=float)

    # def _compute_gate(self, sem_head: np.ndarray, struct_head: np.ndarray,
    #                   gating_enabled: bool, gate_floor: float, gate_ceiling: float)->Tuple[float, np.ndarray]:
    #     rho = 0.0
    #     p_value = 1.0 # ★★★ P-value 초기값 설정 (P-value는 1.0이 가장 안전한 기본값) ★★★
    #     def is_const(x): return x.size==0 or np.allclose(x, x[0])
    #     if gating_enabled and not (is_const(sem_head) or is_const(struct_head)):
    #         # try:
    #         #     r = float(spearmanr(sem_head, struct_head).correlation)
    #         #     if np.isnan(r): r=0.0
    #         #     rho=r
    #         # except Exception: rho=0.0            
    #         try:
    #             # ★★★ P-value 함께 계산 ★★★
    #             r_result = spearmanr(sem_head, struct_head)
    #             r = float(r_result.correlation)
    #             p_value = float(r_result.pvalue)
    #             if np.isnan(r): r=0.0
    #             if np.isnan(p_value): p_value=1.0
    #             rho=r
    #         except Exception: rho=0.0; p_value=1.0

                    
    #     med=0.0; mad=1.0
    #     try:
    #         arr=np.array([rho], dtype=float)
    #         med=np.median(arr); mad=np.median(np.abs(arr-med))+1e-6
    #         z=(rho-med)/mad
    #     except Exception:
    #         z=0.0
    #     gate = 0.6 + 0.1*np.tanh(z)
    #     gate=float(np.clip(gate, gate_floor, gate_ceiling))
    #     return gate, np.array([rho, p_value], dtype=float)

    # ----- 메인 결합 -----
    def rerank_with_structure(self, query_graph: Optional[ig.Graph],
                              passage_topk_ids: List[int], passage_topk_scores: List[float],
                              passage_node_idxs: List[int], last_ppr_scores: Optional[np.ndarray],
                              cfg: SubgraphConfig, embedder: HyperbolicEmbedder, scorer: StructralSimilarity,
                              passage_embeddings: np.ndarray, entity_embeddings: np.ndarray,
                              use_semantic_features: bool=True, use_treerep_metric=False,
                              gating_enabled: bool=True, ensemble_ratio: float=0.6,
                              radial_tau: float=0.5, landmark_k: int=10, alpha: float=0.8,
                              struct_k: int=30, device: str="cpu", save_cb: Optional[Callable]=None,
                              save_ctx: Optional[dict]=None, sim_cac_mode: Literal["tangent","lorentz","node_level","gromov","struct_feat","ensemble"]="ensemble",
                              lorentz_temp: float=2.0, treerep_backend: Literal["treerep","bfs"]="bfs",
                              treerep_ppr_beta: Optional[float]=None, pin_k:int=0, k_rrf:int=20,
                              attn_pooling: bool=True, attn_lambda: float=6.0, attn_topk: int=12, attn_residual: float=0.2,
                              gate_floor: float=0.60, gate_ceiling: float=0.95,
                              ltr_config: Optional[Any]=None,
                              enable_hgnn_fusion: bool=False,
                              adaptive_alpha: bool=True,
                              ) -> Tuple[List[int], List[float]]:

        passage_topk_ids=list(np.asarray(passage_topk_ids).tolist())
        passage_topk_scores=list(np.asarray(passage_topk_scores, dtype=float).tolist())
        if query_graph is None or len(passage_topk_ids)==0:
            return passage_topk_ids, passage_topk_scores

        subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
            passage_topk_ids, passage_topk_scores, passage_node_idxs, last_ppr_scores, cfg, head_k=struct_k
        )
        if not subgraphs: return passage_topk_ids, passage_topk_scores

        # 구조 점수
        if sim_cac_mode in ["tangent", "node_level"]:
            s = self._struct_scores_tangent(
                query_graph, subgraphs, embedder, scorer, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
                attn_pooling=(sim_cac_mode!="node_level"),
                enable_hgnn_fusion=enable_hgnn_fusion 
            )
        elif sim_cac_mode=="lorentz":
            s = self._struct_scores_lorentz(
                query_graph, subgraphs, embedder, temp=lorentz_temp, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
                enable_hgnn_fusion=enable_hgnn_fusion
            )
        elif sim_cac_mode in ["gromov", "struct_feat", "ensemble"]:
            s = np.zeros(len(subgraphs), dtype=float)
            logger.warning(f"Sim mode {sim_cac_mode} returned zero scores (O(V^2) risk).")
        else:
            raise ValueError("Unknown mode")

        if s.size==0: return passage_topk_ids, passage_topk_scores

        # 헤드 정렬/결합
        sem=np.asarray(passage_topk_scores,float); head_idx=np.argsort(-sem)[:len(subgraphs)]
        sem_head=sem[head_idx]
        s_norm=StructralSimilarity().robust_normalize_struct(np.asarray(s,float))

        # RRF
        def ranks(x): return (np.argsort(np.argsort(-x))+1).astype(np.int32)
        rank_sem=ranks(sem_head); rank_struct=ranks(s_norm)
        rrf_sem=1.0/(k_rrf+rank_sem); rrf_struct=1.0/(k_rrf+rank_struct)

        # 게이트
        # gate, _rho = self._compute_gate(sem_head, s_norm, gating_enabled, gate_floor, gate_ceiling)
        gate, rho_p_array = self._compute_gate(sem_head, s_norm, gating_enabled, gate_floor, gate_ceiling)
        rho = rho_p_array[0] if rho_p_array.size > 0 else 0.0
        p_value = rho_p_array[1] if rho_p_array.size > 1 else 1.0

        if adaptive_alpha:
            alpha_eff = float(np.clip(alpha * gate, 0.0, 1.0))
        else:
            alpha_eff = alpha
        
        # LTR(옵션)
        if ltr_config and ltr_config.enabled:
            final_head = alpha_eff*rrf_sem + (1.0-alpha_eff)*rrf_struct 
        else:
            final_head = alpha_eff*rrf_sem + (1.0-alpha_eff)*gate*rrf_struct


        # 핀 고정
        pin_k=max(0,int(pin_k))
        pin_local_idx=np.argsort(-sem_head)[:pin_k]
        pinned_ids=[passage_topk_ids[head_idx[i]] for i in pin_local_idx]
        head_pairs=list(zip([passage_topk_ids[i] for i in head_idx], final_head.tolist()))
        pinned_pairs=[(passage_topk_ids[head_idx[i]], float(final_head[i])) for i in pin_local_idx]
        others=[(did,sc) for (did,sc) in head_pairs if did not in pinned_ids]
        others_sorted=sorted(others, key=lambda x:x[1], reverse=True)
        head_sorted=pinned_pairs+[p for p in others_sorted if p[0] not in set(pinned_ids)]

        orig=dict(zip(passage_topk_ids, passage_topk_scores))
        final_scores={doc:score for doc,score in head_sorted}
        merged_ids=[]; merged_scores=[]
        for did,sc in head_sorted: merged_ids.append(did); merged_scores.append(sc)
        remain=[did for did in passage_topk_ids if did not in final_scores]
        remain_pairs=sorted([(did,orig[did]) for did in remain], key=lambda x:x[1], reverse=True)
        for did,sc in remain_pairs: merged_ids.append(did); merged_scores.append(sc)
        return merged_ids, merged_scores, rho, p_value




# ========================================================================================
# from __future__ import annotations
# import igraph as ig
# import numpy as np
# import torch
# import torch.nn.functional as F
# from dataclasses import dataclass, field
# from typing import Iterable, List, Optional, Tuple, Set, Callable, Literal, Dict, Any, Mapping
# from collections import deque, defaultdict, Counter
# import logging, os, json, math, random
# import pandas as pd
# from scipy.stats import spearmanr
# import networkx as nx
# import geoopt
# from geoopt.manifolds.lorentz import Lorentz

# logger = logging.getLogger(__name__)

# # ======================================================================================
# # Global Cache
# # ======================================================================================
# _TRE_CACHE: Dict[int, "TreeCache"] = {}

# # ======================================================================================
# # TreeCache
# # ======================================================================================
# class TreeCache:
#     def __init__(self, T: nx.Graph, root: Optional[int] = None, max_k: int = 16):
#         if T.number_of_nodes() == 0:
#             raise ValueError("Empty tree")
#         self.T = T
#         if root is None or root not in T:
#             try:
#                 centers = nx.center(T)
#                 root = centers[0] if centers else list(T.nodes())[0]
#             except Exception:
#                 root = list(T.nodes())[0]
#         self.root = root
#         self.max_k = max_k

#         self.parent: Dict[int, Optional[int]] = {}
#         self.level: Dict[int, int] = {}
#         self.dist_to_root: Dict[int, float] = {}
#         self.nodes = list(T.nodes())
#         self.index_of = {v: i for i, v in enumerate(self.nodes)}
#         self._dfs_root_fill_all()

#         n = len(self.nodes)
#         self.up = [[-1]*n for _ in range(max_k+1)]
#         for v in self.nodes:
#             vi = self.index_of[v]
#             p = self.parent.get(v, None)
#             self.up[0][vi] = -1 if p is None else self.index_of.get(p, -1)
#         for k in range(1, max_k+1):
#             for vi in range(n):
#                 mid = self.up[k-1][vi]
#                 self.up[k][vi] = -1 if mid == -1 else self.up[k-1][mid]

#         self.diameter = self._tree_diameter()
#         self.max_dist = max(self.dist_to_root.values()) if self.dist_to_root else 1e-9

#     def _dfs_from(self, r: int):
#         self.parent.setdefault(r, None)
#         self.level.setdefault(r, 0)
#         self.dist_to_root.setdefault(r, 0.0)
#         st, vis = [r], {r}
#         while st:
#             u = st.pop()
#             for v in self.T.neighbors(u):
#                 if v in vis:
#                     continue
#                 vis.add(v)
#                 self.parent[v] = u
#                 self.level[v] = self.level[u] + 1
#                 w = float(self.T[u][v].get("weight", 1.0))
#                 self.dist_to_root[v] = self.dist_to_root[u] + w
#                 st.append(v)
#         return vis

#     def _dfs_root_fill_all(self):
#         visited = self._dfs_from(self.root)
#         if len(visited) != self.T.number_of_nodes():
#             for u in self.T.nodes():
#                 if u not in visited:
#                     sub_vis = self._dfs_from(u)
#                     visited |= sub_vis
#         self.max_dist = max(self.dist_to_root.values()) if self.dist_to_root else 1e-9

#     def _tree_diameter(self) -> float:
#         if self.T.number_of_nodes() <= 1:
#             return 0.0
#         try:
#             comp_nodes = next(c for c in nx.connected_components(self.T) if self.root in c)
#             H = self.T.subgraph(comp_nodes).copy()
#         except Exception:
#             H = self.T

#         adj = {u: [] for u in H.nodes()}
#         for u, v, data in H.edges(data=True):
#             w = float(data.get("weight", 1.0))
#             adj[u].append((v, w))
#             adj[v].append((u, w))

#         def farthest_from(src: int) -> tuple[int, float]:
#             seen = set([src])
#             stack = [(src, -1, 0.0)]
#             far_node, far_dist = src, 0.0
#             while stack:
#                 u, p, d = stack.pop()
#                 if d > far_dist:
#                     far_node, far_dist = u, d
#                 for v, w in adj[u]:
#                     if v == p:
#                         continue
#                     stack.append((v, u, d + w))
#             return far_node, far_dist

#         a, _ = farthest_from(self.root)
#         _, diam = farthest_from(a)
#         return float(diam)
    
#     def lca(self, u, v):
#         if u not in self.index_of or v not in self.index_of:
#             return self.root
#         if u == v:
#             return u
#         if (u not in self.level) or (v not in self.level):
#             return self.root

#         ui = self.index_of[u]; vi = self.index_of[v]
#         du = self.level[u] - self.level[v]
#         bit = 0
#         while du:
#             if du & 1:
#                 ui = self.up[bit][ui] if ui != -1 else -1
#             du >>= 1; bit += 1
#         if ui == vi:
#             return self.nodes[ui]
#         for k in range(self.max_k, -1, -1):
#             u2 = self.up[k][ui]; v2 = self.up[k][vi]
#             if u2 != v2:
#                 ui, vi = u2, v2
#                 if ui == -1 or vi == -1:
#                     return self.root
#         p = self.up[0][ui]
#         return self.nodes[p] if p != -1 else self.root

#     def dist(self, u, v) -> float:
#         if u not in self.dist_to_root or v not in self.dist_to_root:
#             a = self.root
#             du = self.dist_to_root.get(u, self.dist_to_root.get(a, 0.0))
#             dv = self.dist_to_root.get(v, self.dist_to_root.get(a, 0.0))
#             return abs(du - dv)
#         a = self.lca(u, v)
#         return self.dist_to_root[u] + self.dist_to_root[v] - 2.0 * self.dist_to_root[a]

#     def radial(self, v) -> float:
#         if not self.dist_to_root:
#             return 0.0
#         d = self.dist_to_root.get(v, None)
#         if d is None:
#             return 1.0 if (self.max_dist and self.max_dist > 0) else 0.0
#         if not self.max_dist or self.max_dist <= 1e-9:
#             return 0.0
#         r = float(d) / float(self.max_dist)
#         return 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)

# # ======================================================================================
# # StructFeatureExtractor (현재 사용 비중 낮음. 안전하게 유지)
# # ======================================================================================
# class StructFeatureExtractor:
#     def __init__(self, cache: TreeCache):
#         self.C = cache

#     def radial_gap(self, node, rq: float, tau: float=0.5) -> float:
#         gap = abs(self.C.radial(node)-rq)
#         return max(0.0, 1.0 - (gap/(tau+1e-9)))

#     def geodesic_coherence(self, nodes: List[int], max_pairs: int=200) -> float:
#         if len(nodes)<=1: return 1.0
#         pairs=[]
#         for i in range(len(nodes)):
#             for j in range(i+1,len(nodes)):
#                 pairs.append((nodes[i],nodes[j]))
#         if len(pairs)>max_pairs:
#             pairs=random.sample(pairs, max_pairs)
#         s=0.0
#         for u,v in pairs:
#             s+=self.C.dist(u,v) 
#         mean_d = s/max(1,len(pairs))
#         if self.C.diameter<=1e-9: return 1.0
#         return max(0.0, 1.0 - (mean_d/self.C.diameter))

#     def steiner_coverage(self, nodes: List[int]) -> float:
#         if len(nodes)<=1: return 1.0
#         es=set()
#         for i in range(len(nodes)):
#             for j in range(i+1,len(nodes)):
#                 u,v=nodes[i],nodes[j]; a=self.C.lca(u,v)
#                 cur=u
#                 while cur!=a:
#                     p=self.C.parent.get(cur, None)
#                     if p is None: break
#                     es.add(tuple(sorted((cur,p)))); cur=p
#                 cur=v
#                 while cur!=a:
#                     p=self.C.parent.get(cur, None)
#                     if p is None: break
#                     es.add(tuple(sorted((cur,p)))); cur=p
#         steiner_edges=len(es)
#         avg_depth=sum(self.C.level.get(n,0) for n in nodes)/max(1,len(nodes))
#         denom=max(1.0, (len(nodes)-1)+0.5*avg_depth)
#         return max(0.0, 1.0 - (steiner_edges/denom))

#     def gromov_anchor_summary(self, anchor: int, nodes: List[int]) -> Dict[str,float]:
#         if not nodes: return {"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0}
#         z = max(self.C.max_dist, 1e-9)
#         vals = []
#         if anchor not in self.C.T: anchor = self.C.root
#         ar = self.C.dist_to_root.get(anchor, 0.0)
#         for v in nodes:
#             gp = (ar + self.C.dist_to_root.get(v, ar) - self.C.dist(anchor, v)) * 0.5 
#             vals.append(max(0.0, gp / z))
#         return {"gp_mean":float(np.mean(vals)), "gp_min":min(vals), "gp_max":max(vals)}

#     def branch_focus(self, nodes: List[int]) -> float:
#         child_deg=[]
#         for v in nodes:
#             deg=0
#             for u in self.C.T.neighbors(v):
#                 if self.C.parent.get(u,None)==v: deg+=1
#             child_deg.append(deg)
#         if not child_deg: return 1.0
#         c=Counter(child_deg); total=sum(c.values())
#         probs=[cnt/total for cnt in c.values()]
#         H = -sum(p*math.log(p+1e-12) for p in probs)
#         Hmax = math.log(max(1,max(child_deg)+1))
#         return 1.0 if Hmax<=1e-9 else max(0.0, 1.0 - H/Hmax)

#     def features_for_group(self, nodes: List[int], rq: float, tau: float=0.5, anchor: Optional[int]=None) -> Dict[str,float]:
#         if not nodes:
#             return {"radial_gap_mean":0.0,"geo_coherence":1.0,"steiner_coverage":1.0,"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0,"branch_focus":1.0}
#         nodes = [v for v in nodes if v in self.C.T]
#         if not nodes:
#             return {"radial_gap_mean":0.0,"geo_coherence":1.0,"steiner_coverage":1.0,"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0,"branch_focus":1.0}
#         rads=[self.radial_gap(v,rq,tau) for v in nodes]
#         radial_gap_mean=float(np.mean(rads))
#         geo=self.geodesic_coherence(nodes)
#         cov=self.steiner_coverage(nodes)
#         if anchor is None: anchor=min(nodes, key=lambda x:self.C.dist_to_root.get(x,0.0))
#         gp=self.gromov_anchor_summary(anchor, nodes)
#         bf=self.branch_focus(nodes)
#         return { 
#             "radial_gap_mean":radial_gap_mean, 
#             "geo_coherence":geo, 
#             "steiner_coverage":cov, 
#             "gp_mean":gp["gp_mean"], 
#             "gp_min":gp["gp_min"], 
#             "gp_max":gp["gp_max"], 
#             "branch_focus":bf, 
#         }

# # ======================================================================================
# # BFS Tree Cache from igraph (exclude virtual)
# # ======================================================================================
# def _build_bfs_tree_cache_from_igraph( 
#     sg: ig.Graph,
#     *,
#     backend: Literal["treerep", "bfs"] = "bfs", 
#     ppr_scores: Optional[np.ndarray] = None,
#     treerep_ppr_beta: Optional[float] = None,
# ) -> Optional[TreeCache]:
#     if sg.vcount() == 0:
#         return None

#     cache_key = id(sg)
#     if cache_key in _TRE_CACHE:
#         return _TRE_CACHE[cache_key]

#     gu = sg.as_undirected() if sg.is_directed() else sg

#     # ★ 가상 노드 제거
#     keep_vids = list(range(gu.vcount()))
#     if "is_virtual" in gu.vs.attributes():
#         keep_vids = [i for i, v in enumerate(gu.vs["is_virtual"]) if not bool(v)]
#     gu = gu.induced_subgraph(keep_vids) if keep_vids else ig.Graph()

#     gids = [int(x) for x in (gu.vs["global_id"] if "global_id" in gu.vs.attributes() else range(gu.vcount()))]
#     has_w = 'weight' in gu.es.attribute_names()
#     idx2gid = {i: g for i, g in enumerate(gids)}

#     Tfull = nx.Graph()
#     Tfull.add_nodes_from(gids)
#     for e in gu.es:
#         u, v = idx2gid[int(e.source)], idx2gid[int(e.target)]
#         w = float(e["weight"]) if has_w else 1.0
#         Tfull.add_edge(u, v, weight=w)

#     if Tfull.number_of_edges() == 0:
#         T = nx.Graph(); T.add_nodes_from(gids)
#         root_gid = int(gids[0]) if gids else -1
#         T.graph["root_id"] = root_gid
#         cache = TreeCache(T, root=root_gid, max_k=16)
#         _TRE_CACHE[cache_key] = cache
#         return cache

#     # ★ 실노드 루트 선택(중심 선호)
#     try:
#         centers = nx.center(Tfull) if Tfull.number_of_nodes() > 0 else []
#         root_gid = int(centers[0]) if centers else int(gids[0])
#     except Exception:
#         root_gid = int(gids[0]) if gids else -1

#     Tbfs = nx.bfs_tree(Tfull, root_gid)
#     T = nx.Graph(); T.graph["root_id"] = int(root_gid)
#     T.add_nodes_from(Tbfs.nodes())
#     for u, v in Tbfs.edges():
#         T.add_edge(u, v, weight=Tfull[u][v].get("weight", 1.0))
#     cache = TreeCache(T, root=root_gid, max_k=16)
#     _TRE_CACHE[cache_key] = cache
#     return cache

# # ======================================================================================
# # Metrics & Logger
# # ======================================================================================
# class MetricsLogger:
#     def __init__(self, save_dir: str, dataset_name: str, fname: str = "struct_metrics.jsonl"):
#         os.makedirs(save_dir, exist_ok=True)
#         self.path = os.path.join(save_dir, f"{dataset_name}_{fname}")
        
#     def write(self, row: Mapping[str, Any]):
#         with open(self.path, "a", encoding="utf-8") as f:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")

# def _safe_deg_hist(g: ig.Graph, exclude_virtual=False) -> dict:
#     if g.vcount() == 0: return {}
#     mask = [True] * g.vcount()
#     if exclude_virtual and "is_virtual" in g.vs.attributes():
#         mask = [not bool(v) for v in g.vs["is_virtual"]]
#     vids = [i for i, ok in enumerate(mask) if ok]
#     if not vids: return {}
#     degs = g.degree(vids, mode="all")
#     hist = defaultdict(int)
#     for d in degs: hist[int(d)] += 1
#     return {int(k): int(v) for k, v in sorted(hist.items(), key=lambda x: x[0])}

# def _gini(arr: List[float]) -> float:
#     x = np.asarray(arr, dtype=float)
#     if x.size == 0: return 0.0
#     x = np.clip(x, 0, None)
#     if np.all(x == 0): return 0.0
#     x = np.sort(x)
#     n = x.size
#     cumx = np.cumsum(x)
#     return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)

# def _entropy(p: np.ndarray) -> float:
#     p = np.asarray(p, dtype=float)
#     p = p[p > 0]
#     if p.size == 0: return 0.0
#     return float(-np.sum(p * np.log(p)))

# def compute_struct_metrics(
#     g: ig.Graph, 
#     *, 
#     graph_role: str, 
#     dataset_name: str, 
#     query_id: str, 
#     root_global_id: Optional[int] = None, 
#     ppr_scores: Optional[np.ndarray] = None, 
#     exclude_virtual: bool = False
# ) -> dict:
#     if exclude_virtual and "is_virtual" in g.vs.attributes():
#         keep_vids = [i for i, flag in enumerate(g.vs["is_virtual"]) if not bool(flag)]
#         g2 = g.induced_subgraph(keep_vids) if keep_vids else ig.Graph(directed=g.is_directed())
#     else:
#         g2 = g
        
#     m = {
#         "dataset": dataset_name, "query_id": str(query_id), "graph_role": graph_role, 
#         "v": int(g2.vcount()), "e": int(g2.ecount()), "directed": bool(g.is_directed()),
#     }
    
#     if g2.vcount() > 0:
#         comps = g2.components(mode="WEAK")
#         comp_sizes = [len(c) for c in comps]
#         m["comp_cnt"] = int(len(comp_sizes))
#         m["giant_ratio"] = float((max(comp_sizes) / g2.vcount()) if comp_sizes else 0.0)
#     else:
#         m["comp_cnt"] = 0; m["giant_ratio"] = 0.0

#     hist = _safe_deg_hist(g2, exclude_virtual=False)
#     m["deg_hist"] = hist
#     if hist:
#         deg_list = []
#         for k, v in hist.items(): deg_list.extend([k] * v)
#         deg_arr = np.asarray(deg_list, dtype=float)
#         m["deg_mean"] = float(np.mean(deg_arr))
#         m["deg_max"] = float(np.max(deg_arr))
#         m["deg_p95"] = float(np.percentile(deg_arr, 95))
#         m["deg_gini"] = _gini(deg_arr.tolist())
#         total_deg_sq = float(np.sum(deg_arr ** 2))
#         total_deg = float(np.sum(deg_arr))
#         m["deg_hhi"] = float(total_deg_sq / (total_deg**2 + 1e-9))
#         m["tail_frac_ge3"] = float(np.mean(deg_arr >= 3.0))
#     else:
#         m.update(dict(deg_mean=0.0, deg_max=0.0, deg_p95=0.0, deg_gini=0.0, deg_hhi=0.0, tail_frac_ge3=0.0))
        
#     try: m["star_ratio"] = 0.0
#     except Exception: m["star_ratio"] = 0.0
        
#     try:
#         if g2.vcount() > 1:
#             gu = g2.as_undirected() if g2.is_directed() else g2
#             comps = gu.components()
#             giant = max(comps, key=len) if len(comps) else []
#             if len(giant) > 1:
#                 sub = gu.induced_subgraph(giant)
#                 dist = np.array(sub.shortest_paths())
#                 dist = dist[np.isfinite(dist)]
#                 if dist.size:
#                     m["path_len_mean"] = float(np.mean(dist))
#                     m["path_len_p95"] = float(np.percentile(dist, 95))
#                 else: m["path_len_mean"] = 0.0; m["path_len_p95"] = 0.0
#             else: m["path_len_mean"] = 0.0; m["path_len_p95"] = 0.0
#         else: m["path_len_mean"] = 0.0; m["path_len_p95"] = 0.0
#     except Exception: m["path_len_mean"] = 0.0; m["path_len_p95"] = 0.0
        
#     try:
#         bet = np.asarray(g2.betweenness(), dtype=float) if g2.vcount() else np.array([])
#         if bet.size:
#             m["betw_p90"] = float(np.percentile(bet, 90))
#             m["betw_frac_gt_p90"] = float(np.mean(bet > m["betw_p90"]))
#             m["betw_gini"] = _gini(bet.tolist())
#         else: m["betw_p90"] = 0.0; m["betw_frac_gt_p90"] = 0.0; m["betw_gini"] = 0.0
#     except Exception: m["betw_p90"] = 0.0; m["betw_frac_gt_p90"] = 0.0; m["betw_gini"] = 0.0
        
#     try:
#         clu = g2.transitivity_avglocal_undirected()
#         m["clustering_avg"] = float(0.0 if (clu is None or math.isnan(clu)) else clu)
#     except Exception: m["clustering_avg"] = 0.0
        
#     if ppr_scores is not None and "global_id" in g2.vs.attributes():
#         gids = np.asarray(g2.vs["global_id"])
#         pv = np.nan_to_num(np.asarray(ppr_scores, dtype=float), nan=0.0)
#         sel = pv[gids] if gids.size else np.array([])
#         if sel.size and np.sum(sel) > 0:
#             p = sel / np.sum(sel)
#             m["ppr_entropy"] = _entropy(p)
#         else: m["ppr_entropy"] = 0.0
#     else: m["ppr_entropy"] = 0.0
        
#     m["long_tail_hint"] = {"deg_gini": m["deg_gini"], "tail_frac_ge3": m["tail_frac_ge3"]}
#     return m

# def _to_1d_numpy(x) -> np.ndarray:
#     if isinstance(x, torch.Tensor):
#         x = x.detach().flatten()
#         x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
#         return x
#     arr = np.asarray(x, dtype=float).reshape(-1)
#     return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# # ======================================================================================
# # QueryGraphBuilder
# # ======================================================================================
# class QueryGraphBuilder:
#     def __init__(self, root_node: str = "question", directed: bool = True, save_dir: str = f"./outputs/query_graph_analysis", dataset_name: str = "default"):
#         self.root_node = root_node
#         self.directed = directed
#         self.save_dir = save_dir
#         self.dataset_name = dataset_name
#         os.makedirs(self.save_dir, exist_ok=True)
#         self.metrics_logger = MetricsLogger(save_dir=self.save_dir, dataset_name=self.dataset_name)
#         self.log_file = os.path.join(self.save_dir, f"{self.dataset_name}_query_graph_degree_distribution.csv")
#         if not os.path.exists(self.log_file):
#             with open(self.log_file, "w", encoding="utf-8") as f:
#                 f.write("query_id,dataset_name,num_nodes,num_edges,degree_distribution\n")
                
#     def build(self, triples: list[tuple], query_id: str) -> ig.Graph:
#         def norm(x): return str(x).strip()
        
#         if not triples:
#             g = ig.Graph(directed=self.directed)
#             g.add_vertex(name=self.root_node, is_virtual=True)
#             g.vs["global_id"] = [-1]
#             g["root_id"] = g.vs.find(name=self.root_node).index
#             with open(self.log_file, 'a') as f: f.write(f"{query_id},{self.dataset_name},0,0,[]\n")
#             return g
            
#         valid_triples = []
#         for t in triples:
#             if isinstance(t, (list, tuple)) and len(t) == 3:
#                 s, p, o = map(norm, t)
#                 if s and p and o and s != o: valid_triples.append((s, p, o))
                
#         g = ig.Graph(directed=self.directed)
#         if not valid_triples:
#             g.add_vertex(name=self.root_node, is_virtual=True)
#             g.vs["global_id"] = [-1]
#             g["root_id"] = g.vs.find(name=self.root_node).index
#             with open(self.log_file, 'a') as f: f.write(f"{query_id},{self.dataset_name},1,0,[]\n")
#             return g
            
#         node_names = set()
#         for s, _, o in valid_triples: node_names.add(s); node_names.add(o)
#         node_names.add(self.root_node)
#         names_sorted = sorted(node_names)
        
#         g.add_vertices(len(names_sorted))
#         g.vs["name"] = names_sorted
#         g.vs["is_virtual"] = [False] * g.vcount()
#         g.vs.find(name=self.root_node)["is_virtual"] = True
#         g["root_id"] = g.vs.find(name=self.root_node).index
#         name2id = {n: i for i, n in enumerate(g.vs["name"])}
#         g.vs["global_id"] = [-1] * g.vcount()
        
#         edge_map = defaultdict(list)
#         for s, p, o in valid_triples:
#             u, v = name2id[s], name2id[o]
#             edge_map.setdefault((u, v), []).append(p)
            
#         if edge_map:
#             uv = list(edge_map.keys())
#             g.add_edges(uv)
#             rel_lists = [edge_map[k] for k in uv]
#             g.es["relations"] = [list(dict.fromkeys(rels)) for rels in rel_lists]
#             g.es["relation"] = [rels[0] for rels in g.es["relations"]]
#         else:
#             g.es["relations"] = []; g.es["relation"] = []
            
#         root_id = name2id[self.root_node]
#         comps = g.components(mode="WEAK")
#         for comp in comps:
#             if root_id in comp: continue
#             degs = g.degree(comp, mode="all")
#             hub_local = int(np.argmax(degs))
#             hub_vid = comp[hub_local]
#             if g.get_eid(root_id, hub_vid, directed=False, error=False) == -1:
#                 g.add_edge(root_id, hub_vid)
#                 g.es[-1]["relation"] = "root"
#                 g.es[-1]["relations"] = ["root"]

#         non_virtual_nodes = [v for v in g.vs if not v["is_virtual"]]
#         degrees = g.degree(non_virtual_nodes, mode="all") if non_virtual_nodes else []
#         degree_counts = defaultdict(int)
#         for d in degrees: degree_counts[d] += 1
#         sorted_degrees = sorted(degree_counts.items(), key=lambda item: item[0])
#         with open(self.log_file, 'a') as f:
#             f.write(f"{query_id},{self.dataset_name},{g.vcount()},{g.ecount()},{sorted_degrees}\n")
            
#         try:
#             # ★ 가상 노드 제외하여 로깅
#             row = compute_struct_metrics(
#                 g, graph_role="query", dataset_name=self.dataset_name, query_id=str(query_id), 
#                 root_global_id=int(g["root_id"]) if "root_id" in g.attributes() else None, 
#                 ppr_scores=None, exclude_virtual=True
#             )
#             self.metrics_logger.write(row)
#         except Exception as e:
#             logger.warning(f"[QueryGraphBuilder] metrics logging failed: {e}")
            
#         g.vs["global_id"] = list(range(g.vcount()))
#         g["root_id"] = int(g.vs.find(name=self.root_node).index)
#         return g

# # ======================================================================================
# # Subgraph Builder & PPR Pruner
# # ======================================================================================
# @dataclass
# class SubgraphConfig:
#     max_depth: int = 2
#     weight_threshold: float = -1e-9
#     allow_topk: int = 200
#     flow_quantile: float = 0.75
#     per_node_top_r: int = 7
#     ensure_connectivity: bool = True
#     starter_beam: int = 8
#     per_hop_topk: int = 10
#     rank_metric: Literal["ppr","weight","deg"] = "ppr"
#     subgraph_mode: Literal["bfs","ppr"] = "ppr"
#     treat_undirected_for_structure: bool = True
#     keep_bridges: bool = True
#     keep_articulation: bool = True
#     keep_top_edge_betw: int = 0 

# class SubgraphBuilder:
#     def __init__(self, g: ig.Graph, cfg: SubgraphConfig, node_scores: Optional[np.ndarray] = None):
#         self.g=g; self.cfg=cfg
#         self.node_scores=None if node_scores is None else np.nan_to_num(np.asarray(node_scores,float), nan=0.0)
#         self.weighted = ('weight' in g.es.attribute_names())
#         self.outstr = None
#         if self.weighted:
#             self.outstr=np.zeros(g.vcount(), dtype=float)
#             for e in g.es: self.outstr[e.source]+=float(e['weight'])

#     def _rank_key(self, u:int, v:int, w:float)->float:
#         if self.cfg.rank_metric=="ppr" and self.node_scores is not None: return float(self.node_scores[v])
#         if self.cfg.rank_metric=="weight" and self.weighted: return float(w)
#         if self.cfg.rank_metric=="deg": return float(self.g.degree(v))
#         return float(w) if self.weighted else 1.0

#     def _collect_candidates_by_bfs(self, root:int, allow_ids:Optional[Set[int]])->List[Tuple[int,int,float]]:
#         visited={root}; q=deque([(root,0)]); edges=[]
#         mode="ALL"; K=max(1,int(self.cfg.per_hop_topk)) if self.cfg.per_hop_topk else None
#         while q:
#             u,depth=q.popleft()
#             if depth>=self.cfg.max_depth: continue
#             neighs=[]
#             for eid in self.g.incident(u, mode=mode):
#                 e=self.g.es[eid]
#                 v = e.target if e.source==u else e.source
#                 w=float(e['weight']) if self.weighted else 1.0
#                 if w<self.cfg.weight_threshold: continue
#                 if allow_ids is not None and v not in allow_ids: continue
#                 neighs.append((u,v,w))
#             if K is not None and len(neighs)>K:
#                 neighs.sort(key=lambda t:self._rank_key(*t), reverse=True)
#                 neighs=neighs[:K]
#             for uu,vv,ww in neighs:
#                 edges.append((uu,vv,ww))
#                 if vv not in visited:
#                     visited.add(vv); q.append((vv, depth+1))
#         if not edges and self.cfg.starter_beam>0:
#             cand=[]
#             for eid in self.g.incident(root, mode=mode):
#                 e=self.g.es[eid]; v=e.target if e.source==root else e.source
#                 w=float(e['weight']) if self.weighted else 1.0
#                 if allow_ids is None or v in allow_ids: cand.append((root,v,w))
#             if K is not None and len(cand)>self.cfg.starter_beam:
#                 cand.sort(key=lambda t:self._rank_key(*t), reverse=True)
#                 cand=cand[:self.cfg.starter_beam]
#             edges=cand
#         return edges

#     def _collect_candidates_by_ppr(self, root:int, allow_ids:Optional[Set[int]])->List[Tuple[int,int,float]]:
#         if self.node_scores is None: return []
#         ppr_sorted_vids=np.argsort(self.node_scores)[::-1]
#         selected=set()
#         for vid in ppr_sorted_vids:
#             if allow_ids is not None and vid not in allow_ids: continue
#             selected.add(vid)
#             if len(selected)>=self.cfg.allow_topk: break
#         if 0<=root<self.g.vcount(): selected.add(root)
#         edges=[]; seen=set()
#         for u in selected:
#             for v in self.g.neighbors(u, mode="ALL"):
#                 if v in selected:
#                     key=tuple(sorted((u,v)))
#                     if key in seen: continue
#                     seen.add(key)
#                     eids=self.g.get_eids([(u,v)], directed=False)
#                     if eids:
#                         w=float(self.g.es[eids[0]].attributes().get('weight',1.0))
#                         edges.append((u,v,w))
#         return edges

#     def collect_candidates(self, root:int, allow_ids:Optional[Set[int]])->List[Tuple[int,int,float]]:
#         if self.cfg.subgraph_mode=="bfs": return self._collect_candidates_by_bfs(root, allow_ids)
#         if self.cfg.subgraph_mode=="ppr": return self._collect_candidates_by_ppr(root, allow_ids)
#         raise ValueError("Unknown subgraph_mode")

#     def _preserve_connectivity_layer(self, base_edges: List[Tuple[int,int,float]])->List[Tuple[int,int,float]]:
#         if not base_edges: return base_edges
#         nodes = sorted({u for u,_,_ in base_edges} | {v for _,v,_ in base_edges})
#         G = nx.Graph()
#         for n in nodes: G.add_node(n)
#         for u,v,w in base_edges: G.add_edge(u,v,weight=float(w))
#         extra_edges=set()

#         if self.cfg.keep_bridges:
#             try:
#                 for (u,v) in nx.bridges(G):
#                     extra_edges.add(tuple(sorted((u,v))))
#             except Exception:
#                 pass

#         if self.cfg.keep_articulation:
#             try:
#                 arts=list(nx.articulation_points(G))
#                 for a in arts:
#                     for nb in G.neighbors(a):
#                         extra_edges.add(tuple(sorted((a,nb))))
#             except Exception:
#                 pass

#         K = max(0, int(self.cfg.keep_top_edge_betw))
#         if K>0:
#             try:
#                 k_sample = min(64, max(2, int(np.sqrt(G.number_of_nodes()))))
#                 ebc = nx.edge_betweenness_centrality(G, k=k_sample, seed=42)
#             except Exception:
#                 ebc = nx.edge_betweenness_centrality(G)
#             top = sorted(ebc.items(), key=lambda x:x[1], reverse=True)[:K]
#             for (u,v),_ in top:
#                 extra_edges.add(tuple(sorted((u,v))))

#         base_set = {tuple(sorted((u,v))) for u,v,_ in base_edges}
#         merged=[]
#         base_map = {tuple(sorted((u,v))): float(w) for u,v,w in base_edges}
#         for key in base_set | extra_edges:
#             w = base_map.get(key, 1.0)
#             merged.append((key[0], key[1], w))
#         return merged

#     def build_subgraph(self, edges: List[Tuple[int,int,float]], root: Optional[int]=None)->ig.Graph:
#         edges = self._preserve_connectivity_layer(edges)
#         sg = ig.Graph(directed=False)
#         if not edges:
#             sg.add_vertex(name=("root" if root is None else self.g.vs[root]['name']))
#             sg.vs['global_id']=[root if root is not None else -1]
#             return sg
#         nodes=sorted({u for u,_,_ in edges}|{v for _,v,_ in edges})
#         idx={n:i for i,n in enumerate(nodes)}
#         sg.add_vertices(len(nodes))
#         sg.vs['name']=[self.g.vs[n]['name'] for n in nodes]
#         sg.vs['global_id']=nodes
#         # is_virtual 전달(가능한 경우) — 기본 False
#         if "is_virtual" in self.g.vs.attributes():
#             sg.vs['is_virtual'] = [bool(self.g.vs[n]['is_virtual']) for n in nodes]
#         else:
#             sg.vs['is_virtual'] = [False]*len(nodes)
#         if root is not None: sg["root_id"]=int(root) if root in nodes else -1
#         sg.add_edges([(idx[u],idx[v]) for u,v,_ in edges])
#         sg.es['weight']=[w for *_,w in edges]
#         import numpy as _np
#         sg.simplify(multiple=True, loops=True, combine_edges=_np.sum)
#         return sg

# class PPRPruner:
#     def __init__(self, g: ig.Graph, pi: Optional[np.ndarray], cfg: SubgraphConfig):
#         self.g = g
#         self.pi = np.asarray(pi, dtype=float) if pi is not None else None
#         self.cfg = cfg
#         self.weighted = ('weight' in g.es.attribute_names())
#         self.outdeg = np.array(g.outdegree() if g.is_directed() else g.degree(), dtype=float)
        
#         if self.weighted:
#             self.outstr = np.zeros(g.vcount(), dtype=float)
#             for e in g.es: self.outstr[e.source] += float(e['weight'])
#         else:
#             self.outstr = None

#     def _P(self, u: int, eid: int) -> float:
#         if self.weighted:
#             if self.g.es[eid].source != u: return 0.0
#             w = float(self.g.es[eid]['weight'])
#             denom = self.outstr[u]
#             return (w/denom) if denom > 0 else 0.0
#         d = self.outdeg[u]
#         return (1.0/d) if d > 0 else 0.0

#     def _flow(self, u: int, v: int) -> float:
#         if self.pi is None: return 0.0
#         eid = self.g.get_eid(u, v, directed=True, error=False)
#         if eid == -1: return 0.0
#         return float(self.pi[u]) * self._P(u, eid)

#     def prune(self, root: int, candidate_edges: List[Tuple[int,int,float]]) -> List[Tuple[int,int,float]]:
#         if not candidate_edges or self.pi is None: return candidate_edges
#         flows = np.array([self._flow(u, v) for u, v, _ in candidate_edges], dtype=float)
#         flows = np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0)
#         kept_idx = np.arange(len(candidate_edges))

#         if self.cfg.flow_quantile is not None:
#             finite = np.isfinite(flows)
#             q = float(np.quantile(flows[finite], self.cfg.flow_quantile)) if finite.any() else 0.0
#             cut = max(self.cfg.weight_threshold, q)
#             mask = (flows >= cut)
#             kept_idx = kept_idx[mask]
#             if kept_idx.size == 0:
#                 med = float(np.quantile(flows[finite], 0.5)) if finite.any() else 0.0
#                 cut = max(self.cfg.weight_threshold, med)
#                 kept_idx = np.where(flows >= cut)[0]
#             if kept_idx.size == 0:
#                 N = min(10, len(candidate_edges))
#                 kept_idx = np.argsort(-flows)[:N]

#         kept = [candidate_edges[i] for i in kept_idx]
#         r = self.cfg.per_node_top_r
#         if r is None or r <= 0: return kept
        
#         by_src: dict[int, List[Tuple[int,float]]] = {}
#         for i in kept_idx:
#             u, v, _ = candidate_edges[i]
#             by_src.setdefault(u, []).append((i, flows[i]))
            
#         final_idx: List[int] = []
#         for u, items in by_src.items():
#             items.sort(key=lambda x: x[1], reverse=True)
#             r_eff = min(r, len(items))
#             th = items[r_eff - 1][1]
#             final_idx.extend([i for (i, f) in items if f >= th])

#         if self.cfg.allow_topk is not None and len(final_idx) > self.cfg.allow_topk:
#             order = np.argsort(-flows[final_idx])
#             final_idx = [final_idx[i] for i in order[:self.cfg.allow_topk]]
            
#         return [candidate_edges[i] for i in final_idx]

# # ======================================================================================
# # HGCN Layer (non-train; fusion용)
# # ======================================================================================
# class HGCNLayer(torch.nn.Module):
#     def __init__(self, in_features, out_features, manifold: Lorentz, bias=True):
#         super().__init__()
#         self.manifold = manifold
#         self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.linear.reset_parameters()
#         if self.linear.bias is not None:
#             self.linear.bias.data.zero_() 

#     def forward(self, x_lorentz, adj_mat):
#         x_tangent = self.manifold.logmap0(x_lorentz) 
#         h = self.linear(x_tangent) 
#         D_diag = adj_mat.sum(dim=1).clamp(min=1e-12)
#         D_inv_sqrt = torch.diag_embed(D_diag.pow(-0.5))
#         A_hat = D_inv_sqrt @ (adj_mat + torch.eye(adj_mat.size(0), device=adj_mat.device)) @ D_inv_sqrt
#         h_aggregated = A_hat @ h
#         y = self.manifold.expmap0(h_aggregated)
#         return self.manifold.projx(y) 

# # ======================================================================================
# # HyperbolicEmbedder (question_embedding & node2pid 반영)
# # ======================================================================================
# class HyperbolicEmbedder:
#     def __init__(self, hr_instance, embed_dim=64, curvature=1.0, device="cpu"):
#         self.manifold = geoopt.manifolds.Lorentz(k=curvature)
#         self.embed_dim = embed_dim
#         self.device = device
#         self.hr = hr_instance
#         self.feature_extractors = {} 
#         torch.manual_seed(1234)
        
#         self.hgnn_fusion = HGCNLayer(
#             in_features=self.embed_dim, 
#             out_features=self.embed_dim, 
#             manifold=self.manifold,
#             bias=True
#         ).to(device)

#         self._projW: Optional[torch.Tensor] = None

#     def _project_and_stabilize(self, X: np.ndarray)->torch.Tensor:
#         if X.size==0:
#             return torch.zeros((1,self.embed_dim), dtype=torch.float32, device=self.device)
#         Z=torch.tensor(X, dtype=torch.float32, device=self.device)
#         n=torch.linalg.norm(Z, dim=1, keepdim=True)+1e-6
#         Z=Z*torch.clamp(3.0/n, max=1.0)
#         d_in=Z.size(1); d_out=self.embed_dim
#         if d_in != d_out:
#             if self._projW is None or self._projW.size()!=torch.Size([d_in, d_out]):
#                 W=torch.empty(d_in, d_out, device=self.device)
#                 torch.nn.init.orthogonal_(W)
#                 self._projW=W
#             Z=Z@self._projW
#         return Z

#     # ★ question_embedding & node2pid 추가
#     def graph_to_tensor(self, g: ig.Graph, ppr_scores: Optional[np.ndarray]=None,
#                         passage_embeddings: Optional[np.ndarray]=None,
#                         entity_embeddings: Optional[np.ndarray]=None,
#                         use_semantic_features: bool=True,
#                         use_treerep_metric: bool=False, 
#                         treerep_ppr_beta: Optional[float]=None,
#                         enable_hgnn_fusion: bool=False,
#                         question_embedding: Optional[np.ndarray]=None,
#                         node2pid: Optional[Dict[str, int]]=None) -> torch.Tensor:
#         if g.vcount() == 0: 
#             return torch.zeros(1,self.embed_dim, device=self.device)
        
#         X = None 
#         num_nodes = g.vcount()

#         if use_semantic_features and passage_embeddings is not None and passage_embeddings.size > 0:
#             D_embed = passage_embeddings.shape[1]
#             X_sem = np.zeros((num_nodes, D_embed), dtype=np.float32)

#             names = g.vs["name"]
#             is_virtual = g.vs["is_virtual"] if "is_virtual" in g.vs.attributes() else [False]*num_nodes

#             for i, node_name in enumerate(names):
#                 # ★ 가상 노드 스킵
#                 if is_virtual[i]:
#                     continue
#                 if node_name == "question":
#                     if question_embedding is not None:
#                         X_sem[i] = question_embedding
#                     continue
#                 if node2pid is not None and node_name in node2pid:
#                     pid = node2pid[node_name]
#                     if 0 <= pid < passage_embeddings.shape[0]:
#                         X_sem[i] = passage_embeddings[pid]
#                     continue
#                 if node_name.startswith("chunk-") or node_name.startswith("entity-"):
#                     try:
#                         passage_id = int(node_name.split('-')[-1])
#                         idx_in_array = passage_id % passage_embeddings.shape[0]
#                         X_sem[i] = passage_embeddings[idx_in_array]
#                     except ValueError:
#                         pass

#             X = X_sem if np.sum(X_sem) > 1e-6 else np.zeros((num_nodes, D_embed), dtype=np.float32)

#         if X is None or X.size == 0:
#             return torch.zeros(1, self.embed_dim, device=self.device)

#         Zlin=self._project_and_stabilize(X) 
        
#         if enable_hgnn_fusion and g.ecount() > 0 and Zlin.size(0) > 1 and Zlin.size(1) == self.embed_dim:
#             x_lorentz = self.manifold.expmap0(Zlin)
#             x_lorentz = self.manifold.projx(x_lorentz)
#             edges = g.get_edgelist()
#             edge_indices = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
#             num_nodes = g.vcount()
#             adj_mat = torch.zeros((num_nodes, num_nodes), device=self.device)
#             adj_mat[edge_indices[0], edge_indices[1]] = 1.0
#             adj_mat[edge_indices[1], edge_indices[0]] = 1.0
#             fused_lorentz = self.hgnn_fusion(x_lorentz, adj_mat)
#             return self.manifold.logmap0(fused_lorentz)
            
#         return Zlin

#     def embed_lorentz(self, g: ig.Graph, ppr_scores=None, passage_embeddings=None,
#                       entity_embeddings=None, use_semantic_features=True,
#                       use_treerep_metric=False, treerep_ppr_beta=None,
#                       enable_hgnn_fusion: bool=False, **kw)->torch.Tensor:
#         if g.vcount()==0: return None
#         t_nodes = self.graph_to_tensor(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
#                                        entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
#                                        use_treerep_metric=use_treerep_metric, treerep_ppr_beta=treerep_ppr_beta,
#                                        enable_hgnn_fusion=enable_hgnn_fusion, **kw) 
#         with torch.no_grad():
#             x_lorentz = self.manifold.expmap0(t_nodes)
#             x_lorentz = torch.nan_to_num(x_lorentz, nan=0.0, posinf=0.0, neginf=0.0)
#             pooled, _ = torch.max(x_lorentz, dim=0, keepdim=False)
#             pooled = self.manifold.projx(pooled)
#             if torch.linalg.norm(pooled).item()<1e-8:
#                 pooled = pooled + 1e-3*torch.randn_like(pooled)
#                 pooled = self.manifold.projx(pooled)
#             return pooled

#     def embed_nodes_lorentz(self, g: ig.Graph, enable_hgnn_fusion: bool=False, **kw)->torch.Tensor:
#         t_fused = self.graph_to_tensor(g, enable_hgnn_fusion=enable_hgnn_fusion, **kw)
#         with torch.no_grad():
#             nmap=self.manifold.expmap0(t_fused)
#             return torch.nan_to_num(nmap, nan=0.0, posinf=0.0, neginf=0.0)

#     def logmap0_safe(self, y: torch.Tensor)->torch.Tensor:
#         t=self.manifold.logmap0(y)
#         return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

#     def shared_standardize(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
#         X = torch.stack(tensors, dim=0)  # [B, D]
#         mu = X.mean(dim=0, keepdim=True)
#         sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
#         Y = [(t - mu.squeeze(0)) / sd.squeeze(0) for t in tensors]
#         return Y

#     def embed_and_logmap0(self, g: ig.Graph, enable_hgnn_fusion: bool=False, **kw) -> torch.Tensor:
#         t_nodes = self.graph_to_tensor(g, enable_hgnn_fusion=enable_hgnn_fusion, **kw)
        
#         with torch.no_grad():
#             if t_nodes.dim() == 2 and t_nodes.size(0) > 1:
#                 x_lorentz = self.manifold.expmap0(t_nodes)
#                 pooled, _ = torch.max(x_lorentz, dim=0, keepdim=False)
#                 pooled = self.manifold.projx(pooled)
#                 return self.logmap0_safe(pooled.unsqueeze(0)).squeeze(0)
#             elif t_nodes.dim() == 1 or (t_nodes.dim() == 2 and t_nodes.size(0) == 1):
#                 return t_nodes.squeeze(0)
#             return torch.zeros(self.embed_dim, device=self.device)

#     def attention_pool_tangent(self, tq: torch.Tensor, td_nodes: torch.Tensor, lam: float=6.0, topk: Optional[int]=12, residual: float=0.2) -> torch.Tensor:
#         sims = F.cosine_similarity(td_nodes, tq.unsqueeze(0), dim=-1)
#         w = torch.sigmoid(lam * sims)
#         if topk is not None and topk < w.numel():
#             k = max(1, int(topk))
#             idx = torch.topk(w, k=k, dim=0).indices
#             mask = torch.zeros_like(w); mask[idx] = 1.0
#             w = w * mask
#         w = residual + (1.0 - residual) * w
#         s = w.sum()
#         if s <= 1e-6:
#             return td_nodes.mean(dim=0)
#         w = w / s
#         return (td_nodes * w.unsqueeze(-1)).sum(dim=0)

# # ======================================================================================
# # Structural Similarity & LTRConfig
# # ======================================================================================
# class StructralSimilarity:
#     def __init__(self, mode="cosine"):
#         assert mode in ["cosine", "l2", "dot"], "Invalid similarity mode"
#         self.mode = mode

#     def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
#         x = np.asarray(scores, dtype=float)
#         if x.size == 0: return x
#         lo, hi = np.percentile(x, 5), np.percentile(x, 95)
#         if hi - lo < 1e-8: return np.full_like(x, 0.5)
#         y = (x - lo) / (hi - lo + 1e-8)
#         y = np.clip(y, 0.0, 1.0)
#         return y

#     def compute(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
#         if self.mode == "cosine": return F.cosine_similarity(q, d, dim=-1)
#         if self.mode == "dot": return torch.sum(q * d, dim=-1)
#         return -torch.norm(q - d, dim=-1)

#     def compute_listwise(self, query_list: List[torch.Tensor], doc_list: List[torch.Tensor]) -> torch.Tensor:
#         assert len(query_list) == len(doc_list)
#         sims = []
#         for q, d in zip(query_list, doc_list):
#             sim = self.compute(q.unsqueeze(0), d.unsqueeze(0)).squeeze()
#             sims.append(sim)
#         return torch.stack(sims)

#     def robust_normalize_struct(self, x: np.ndarray) -> np.ndarray:
#         if x.size == 0: return x
#         x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#         lo, hi = np.percentile(x, 10), np.percentile(x, 90)
#         if hi - lo < 1e-8: return np.full_like(x, 0.5)
#         x = np.clip(x, lo, hi)
#         y = (x - lo) / (hi - lo + 1e-9)
#         return y

# @dataclass
# class LTRConfig:
#     enabled: bool = False
#     weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.5, 0.5])
#     bias: float = 0.0
#     blend: float = 0.5

# # ======================================================================================
# # HyperReranker
# # ======================================================================================
# class HyperReranker:
    
#     class IDMapper:
#         @staticmethod
#         def docids_to_vids(doc_ids: Iterable[int], passage_node_idxs: List[int]) -> List[int]:
#             return [int(passage_node_idxs[d]) for d in doc_ids]

#     def __init__(self, g: ig.Graph, save_dir: str = "./outputs/subgraph_analysis", dataset_name: str = "default"):
#         self.g = g
#         self.metrics_logger = MetricsLogger(save_dir=save_dir, dataset_name=dataset_name)
#         self.cfg = None
#         self.dataset_name = dataset_name
#         self.query_id = "unknown"

#     @staticmethod
#     def make_allow_ids(g: ig.Graph, seeds_vids: List[int], ppr_scores: Optional[np.ndarray], allow_topk: int) -> Optional[Set[int]]:
#         if ppr_scores is None or not allow_topk or allow_topk <= 0: return None
#         ppr = np.nan_to_num(np.asarray(ppr_scores, float), nan=0.0, posinf=0.0, neginf=0.0)
#         topN = np.argsort(-ppr)[:allow_topk].tolist()
#         allow = set(topN)
#         for s in seeds_vids:
#             for nb in g.neighbors(s, mode="all"): allow.add(nb)
#         return allow

#     def build_pruned_subgraphs_for_topk_passages(
#         self, topk_doc_ids: List[int], topk_scores: List[float], passage_node_idxs: List[int], 
#         last_ppr_scores: Optional[np.ndarray], cfg: SubgraphConfig, head_k: int = 30,
#     ) -> Tuple[List[ig.Graph], List[int], List[int]]:
#         if not topk_doc_ids: return [], [], []
#         self.cfg = cfg
        
#         sem = np.asarray(topk_scores, float)
#         head_idx = np.argsort(-sem)[: min(head_k, len(topk_doc_ids))]
#         head_doc_ids = [topk_doc_ids[i] for i in head_idx]
        
#         head_vids = self.IDMapper.docids_to_vids(head_doc_ids, passage_node_idxs) 
#         allow_ids = self.make_allow_ids(self.g, head_vids, last_ppr_scores, cfg.allow_topk)
        
#         builder = SubgraphBuilder(self.g, cfg, node_scores=last_ppr_scores)
#         pruner = PPRPruner(self.g, last_ppr_scores, cfg)
#         subgraphs: List[ig.Graph] = []
        
#         for rid in head_vids:
#             cands = builder.collect_candidates(root=rid, allow_ids=allow_ids)
#             pruned = pruner.prune(root=rid, candidate_edges=cands) if cfg.subgraph_mode == "ppr" else cands
#             sg = builder.build_subgraph(pruned, root=rid)
#             try:
#                 # ★ 가상 노드 제외하여 로깅
#                 row = compute_struct_metrics(
#                     sg, graph_role="subgraph", 
#                     dataset_name=self.dataset_name, 
#                     query_id=str(self.query_id), 
#                     root_global_id=rid, ppr_scores=last_ppr_scores , exclude_virtual=True
#                 )
#                 row.update({ 
#                     "root_doc_vid": int(rid), 
#                     "root_doc_id": int(head_doc_ids[len(subgraphs)]) if len(subgraphs) < len(head_doc_ids) else None, 
#                     "subgraph_idx": int(len(subgraphs)), 
#                     "collect_mode": self.cfg.subgraph_mode
#                 })
#                 self.metrics_logger.write(row)
#             except Exception as e:
#                 logger.warning(f"[SubgraphMetrics] logging failed: {e}")
#             subgraphs.append(sg)
#         return subgraphs, head_doc_ids, head_vids

#     def _struct_scores_tangent(
#         self, query_graph: ig.Graph, subgraphs: List[ig.Graph], embedder: "HyperbolicEmbedder", scorer: "StructralSimilarity", 
#         ppr_scores: Optional[np.ndarray] = None, passage_embeddings: Optional[np.ndarray] = None, entity_embeddings: Optional[np.ndarray] = None, 
#         use_semantic_features: bool = True, use_treerep_metric: bool = False,
#         attn_pooling: bool = True, attn_lambda: float = 6.0, attn_topk: Optional[int] = 12, attn_residual: float = 0.2,
#         enable_hgnn_fusion: bool=False,
#         **kw  # question_embedding, node2pid 등 전달
#     ) -> np.ndarray:
#         with torch.no_grad():
#             tq = embedder.embed_and_logmap0(
#                 query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
#                 entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
#                 enable_hgnn_fusion=enable_hgnn_fusion, **kw
#             )
#             tdocs = []
#             for g in subgraphs:
#                 d_nodes_L = embedder.embed_nodes_lorentz(
#                     g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
#                     entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
#                     enable_hgnn_fusion=enable_hgnn_fusion, **kw
#                 )
#                 # ★ 가상노드 마스크
#                 if "is_virtual" in g.vs.attributes() and d_nodes_L.size(0) == g.vcount():
#                     mask = torch.tensor([not v for v in g.vs["is_virtual"]], device=embedder.device)
#                     if mask.any():
#                         d_nodes_L = d_nodes_L[mask]
#                 if d_nodes_L.numel() == 0:
#                     tdocs.append(torch.zeros_like(tq)); continue
#                 td_nodes = embedder.logmap0_safe(d_nodes_L) # [N, D] Tangent Vector
#                 if attn_pooling:
#                     pooled = embedder.attention_pool_tangent(
#                         tq, td_nodes, lam=attn_lambda, topk=attn_topk, residual=attn_residual
#                     )
#                 else:
#                     pooled, _ = torch.max(td_nodes, dim=0, keepdim=False) 
#                 tdocs.append(pooled) 
            
#             tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
#             q_list = [tq_std for _ in range(len(tdocs_std))]
#             struct_head = scorer.compute_listwise(q_list, tdocs_std)
#             return _to_1d_numpy(struct_head)

#     def _struct_scores_lorentz(
#         self, query_graph: ig.Graph, subgraphs: List[ig.Graph], embedder: "HyperbolicEmbedder", temp: float = 2.0, 
#         ppr_scores: Optional[np.ndarray] = None, passage_embeddings: Optional[np.ndarray] = None, entity_embeddings: Optional[np.ndarray] = None, 
#         use_semantic_features: bool = True, use_treerep_metric: bool = False,
#         enable_hgnn_fusion: bool=False,
#         **kw
#     ) -> np.ndarray:
#         with torch.no_grad():
#             qL = embedder.embed_lorentz(
#                 query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
#                 entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
#                 enable_hgnn_fusion=enable_hgnn_fusion, **kw
#             )
#             dL = [
#                 embedder.embed_lorentz(
#                     g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
#                     entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
#                     enable_hgnn_fusion=enable_hgnn_fusion, **kw
#                 ) for g in subgraphs
#             ]
#             qL = qL if qL is not None else torch.zeros(embedder.embed_dim, device=embedder.device)
#             dL_tensors = []
#             for x in dL:
#                 if x is None: dL_tensors.append(torch.zeros(embedder.embed_dim, device=embedder.device))
#                 else: dL_tensors.append(x)
#             dists2 = torch.stack([
#                 embedder.manifold.dist2(qL.unsqueeze(0), x.unsqueeze(0)).squeeze() for x in dL_tensors
#             ])
#             d2 = _to_1d_numpy(dists2)
#             sim = -d2
#             if sim.size:
#                 norm_sim = StructralSimilarity().robust_normalize_struct(sim)
#                 sim = 1.0 / (1.0 + np.exp(-((norm_sim - 0.5) / (1.0 / temp))))
#             return sim

#     def _struct_scores_struct_feat(
#         self, query_graph: ig.Graph, subgraphs: List[ig.Graph], query_radial_anchor: Optional[float] = None,
#         radial_tau: float = 0.5, ppr_scores: Optional[np.ndarray] = None, treerep_backend: Literal["treerep","bfs"] = "bfs",
#         treerep_ppr_beta: Optional[float] = None,
#     ) -> np.ndarray:
#         logger.warning("O(V^2) cost in _struct_scores_struct_feat: Disabling in favor of Tangent/LTR.")
#         return np.zeros(len(subgraphs), dtype=float)

#     def _pick_landmarks_diverse(self, cache: TreeCache, L:int=10)->List[int]:
#         nodes=list(cache.T.nodes())
#         if not nodes: return []
#         S=[cache.root]
#         def score(v,S):
#             base=min(cache.dist(v,s) for s in S)
#             deg=cache.T.degree(v)
#             return base - 0.1*deg
#         while len(S)<min(L,len(nodes)):
#             cand=max(nodes, key=lambda v: score(v,S))
#             if cand in S:
#                 alt=random.choice([x for x in nodes if x not in S])
#                 S.append(alt)
#             else:
#                 S.append(cand)
#         return S

#     def _struct_scores_gromov(self, query_graph: ig.Graph, subgraphs: List[ig.Graph], landmark_k:int=10,
#                               ppr_scores: Optional[np.ndarray]=None,
#                               treerep_backend: Literal["treerep","bfs"]="bfs",
#                               treerep_ppr_beta: Optional[float]=None)->np.ndarray:
#         logger.warning("O(V^2) cost in _struct_scores_gromov: Disabling in favor of Tangent/LTR.")
#         return np.zeros(len(subgraphs), dtype=float)

#     def _compute_gate(self, sem_head: np.ndarray, struct_head: np.ndarray,
#                       gating_enabled: bool, gate_floor: float, gate_ceiling: float)->Tuple[float, np.ndarray]:
#         rho=0.0
#         def is_const(x): return x.size==0 or np.allclose(x, x[0])
#         if gating_enabled and not (is_const(sem_head) or is_const(struct_head)):
#             try:
#                 r = float(spearmanr(sem_head, struct_head).correlation)
#                 if np.isnan(r): r=0.0
#                 rho=r
#             except Exception: rho=0.0
#         med=0.0; mad=1.0
#         try:
#             arr=np.array([rho], dtype=float)
#             med=np.median(arr); mad=np.median(np.abs(arr-med))+1e-6
#             z=(rho-med)/mad
#         except Exception:
#             z=0.0
#         gate = 0.6 + 0.1*np.tanh(z)
#         gate=float(np.clip(gate, gate_floor, gate_ceiling))
#         return gate, np.array([rho], dtype=float)

#     def rerank_with_structure(self, query_graph: Optional[ig.Graph],
#                               passage_topk_ids: List[int], passage_topk_scores: List[float],
#                               passage_node_idxs: List[int], last_ppr_scores: Optional[np.ndarray],
#                               cfg: SubgraphConfig, embedder: HyperbolicEmbedder, scorer: StructralSimilarity,
#                               passage_embeddings: np.ndarray, entity_embeddings: np.ndarray,
#                               use_semantic_features: bool=True, use_treerep_metric=False,
#                               gating_enabled: bool=True, ensemble_ratio: float=0.6,
#                               radial_tau: float=0.5, landmark_k: int=10, alpha: float=0.8,
#                               struct_k: int=30, device: str="cpu", save_cb: Optional[Callable]=None,
#                               save_ctx: Optional[dict]=None, sim_cac_mode: Literal["tangent","lorentz","node_level","gromov","struct_feat","ensemble"]="ensemble",
#                               lorentz_temp: float=2.0, treerep_backend: Literal["treerep","bfs"]="bfs",
#                               treerep_ppr_beta: Optional[float]=None, pin_k:int=0, k_rrf:int=20,
#                               attn_pooling: bool=True, attn_lambda: float=6.0, attn_topk: int=12, attn_residual: float=0.2,
#                               gate_floor: float=0.60, gate_ceiling: float=0.95,
#                               ltr_config: Optional[Any]=None,
#                               enable_hgnn_fusion: bool=False,
#                               **kw  # question_embedding, node2pid 등 받을 그릇
#                               ) -> Tuple[List[int], List[float]]:

#         passage_topk_ids=list(np.asarray(passage_topk_ids).tolist())
#         passage_topk_scores=list(np.asarray(passage_topk_scores, dtype=float).tolist())
#         if query_graph is None or len(passage_topk_ids)==0:
#             return passage_topk_ids, passage_topk_scores

#         subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
#             passage_topk_ids, passage_topk_scores, passage_node_idxs, last_ppr_scores, cfg, head_k=struct_k
#         )
#         if not subgraphs: return passage_topk_ids, passage_topk_scores

#         if sim_cac_mode in ["tangent", "node_level"]:
#             s = self._struct_scores_tangent(
#                 query_graph, subgraphs, embedder, scorer, ppr_scores=last_ppr_scores,
#                 passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
#                 attn_pooling=(sim_cac_mode!="node_level"),
#                 enable_hgnn_fusion=enable_hgnn_fusion, **kw
#             )
#         elif sim_cac_mode=="lorentz":
#             s = self._struct_scores_lorentz(
#                 query_graph, subgraphs, embedder, temp=lorentz_temp, ppr_scores=last_ppr_scores,
#                 passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
#                 enable_hgnn_fusion=enable_hgnn_fusion, **kw
#             )
#         elif sim_cac_mode in ["gromov", "struct_feat", "ensemble"]:
#             s = np.zeros(len(subgraphs), dtype=float)
#             logger.warning(f"Sim mode {sim_cac_mode} returned zero scores (O(V^2) risk).")
#         else:
#             raise ValueError("Unknown mode")

#         if s.size==0: return passage_topk_ids, passage_topk_scores

#         sem=np.asarray(passage_topk_scores,float); head_idx=np.argsort(-sem)[:len(subgraphs)]
#         sem_head=sem[head_idx]
#         s_norm=StructralSimilarity().robust_normalize_struct(np.asarray(s,float))

#         def ranks(x): return (np.argsort(np.argsort(-x))+1).astype(np.int32)
#         rank_sem=ranks(sem_head); rank_struct=ranks(s_norm)
#         rrf_sem=1.0/(k_rrf+rank_sem); rrf_struct=1.0/(k_rrf+rank_struct)

#         gate, _rho = self._compute_gate(sem_head, s_norm, gating_enabled, gate_floor, gate_ceiling)

#         if ltr_config and ltr_config.enabled:
#             final_head = alpha*rrf_sem + (1.0-alpha)*rrf_struct 
#         else:
#             final_head = alpha*rrf_sem + (1.0-alpha)*gate*rrf_struct

#         pin_k=max(0,int(pin_k))
#         pin_local_idx=np.argsort(-sem_head)[:pin_k]
#         pinned_ids=[passage_topk_ids[head_idx[i]] for i in pin_local_idx]
#         head_pairs=list(zip([passage_topk_ids[i] for i in head_idx], final_head.tolist()))
#         pinned_pairs=[(passage_topk_ids[head_idx[i]], float(final_head[i])) for i in pin_local_idx]
#         others=[(did,sc) for (did,sc) in head_pairs if did not in pinned_ids]
#         others_sorted=sorted(others, key=lambda x:x[1], reverse=True)
#         head_sorted=pinned_pairs+[p for p in others_sorted if p[0] not in set(pinned_ids)]

#         orig=dict(zip(passage_topk_ids, passage_topk_scores))
#         final_scores={doc:score for doc,score in head_sorted}
#         merged_ids=[]; merged_scores=[]
#         for did,sc in head_sorted: merged_ids.append(did); merged_scores.append(sc)
#         remain=[did for did in passage_topk_ids if did not in final_scores]
#         remain_pairs=sorted([(did,orig[did]) for did in remain], key=lambda x:x[1], reverse=True)
#         for did,sc in remain_pairs: merged_ids.append(did); merged_scores.append(sc)
#         return merged_ids, merged_scores
