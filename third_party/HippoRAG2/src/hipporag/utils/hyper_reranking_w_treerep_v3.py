# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import json
import logging
from dataclasses import dataclass
from typing import (
    Iterable, List, Optional, Tuple, Set, Callable, Dict, Any, Mapping, Literal
)
from collections import deque, defaultdict

import numpy as np
import igraph as ig
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import networkx as nx
import geoopt

# TreeRep
from .TreeRep import TreeRep

logger = logging.getLogger(__name__)

# =============================================================================
# TreeCache / StructFeatureExtractor
# =============================================================================

class TreeCache:
    """무방향 트리 캐시: LCA/거리/반지름(depth) 계산용 (서브그래프 단위)."""
    def __init__(self, T: nx.Graph, root: Optional[int] = None, max_k: int = 16):
        if T.number_of_nodes() == 0:
            raise ValueError("Empty tree")
        self.T = T
        self.root = root if root is not None else T.graph.get("root_id", list(T.nodes)[0])
        self.max_k = max_k
        self.parent, self.level, self.dist_to_root = {}, {}, {}
        self._dfs_root()
        self.nodes = list(T.nodes)
        self.index_of = {v: i for i, v in enumerate(self.nodes)}
        n = len(self.nodes)
        self.up = [[-1] * n for _ in range(max_k + 1)]
        for v in self.nodes:
            vi = self.index_of[v]
            p = self.parent.get(v, None)
            self.up[0][vi] = -1 if p is None else self.index_of[p]
        for k in range(1, max_k + 1):
            for vi in range(n):
                mid = self.up[k - 1][vi]
                self.up[k][vi] = -1 if mid == -1 else self.up[k - 1][mid]
        self.diameter = self._tree_diameter()
        self.max_dist = max(self.dist_to_root.values()) if self.dist_to_root else 1e-9

    def _dfs_root(self):
        r = self.root
        self.parent[r], self.level[r], self.dist_to_root[r] = None, 0, 0.0
        st, vis = [r], {r}
        while st:
            u = st.pop()
            for v in self.T.neighbors(u):
                if v in vis: continue
                vis.add(v)
                self.parent[v] = u
                self.level[v] = self.level[u] + 1
                w = float(self.T[u][v].get("weight", 1.0))
                self.dist_to_root[v] = self.dist_to_root[u] + w
                st.append(v)

    def _tree_diameter(self) -> float:
        def far(src):
            dist = {src: 0.0}; st=[src]; vis={src}
            while st:
                u=st.pop()
                for v in self.T.neighbors(u):
                    if v in vis: continue
                    vis.add(v)
                    w=float(self.T[u][v].get("weight",1.0))
                    dist[v]=dist[u]+w; st.append(v)
            fv=max(dist.items(), key=lambda x:x[1])[0]
            return fv, dist[fv]
        a,_=far(self.root); _,diam=far(a)
        return float(diam)

    def lca(self, u, v):
        if u==v: return u
        ui=self.index_of[u]; vi=self.index_of[v]
        du=self.level[u]-self.level[v]
        if du<0: ui,vi,du=vi,ui,-du
        bit=0
        while du:
            if du&1: ui=self.up[bit][ui] if ui!=-1 else -1
            du >>=1; bit+=1
        if ui==vi: return self.nodes[ui]
        for k in range(self.max_k,-1,-1):
            u2=self.up[k][ui]; v2=self.up[k][vi]
            if u2!=v2: ui,vi=u2,v2
        p=self.up[0][ui]
        return self.nodes[p] if p!=-1 else self.root

    def dist(self, u, v) -> float:
        a=self.lca(u,v)
        return self.dist_to_root[u]+self.dist_to_root[v]-2.0*self.dist_to_root[a]

    def radial(self, v) -> float:
        # 트리에 없는 노드는 바깥(1.0)로 간주
        if v not in self.dist_to_root:
            return 1.0
        if self.max_dist <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, self.dist_to_root[v] / self.max_dist))


class StructFeatureExtractor:
    """QA-지향 구조 피처(그룹 단위)."""
    def __init__(self, cache: TreeCache):
        self.C = cache

    def radial_gap(self, node, rq: float, tau: float=0.5) -> float:
        gap = abs(self.C.radial(node)-rq)
        return max(0.0, 1.0 - (gap/(tau+1e-9)))

    def geodesic_coherence(self, nodes: List[int], max_pairs: int=200) -> float:
        if len(nodes)<=1: return 1.0
        import random
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
                    p=self.C.parent[cur]; es.add(tuple(sorted((cur,p)))); cur=p
                cur=v
                while cur!=a:
                    p=self.C.parent[cur]; es.add(tuple(sorted((cur,p)))); cur=p
        steiner_edges=len(es)
        avg_depth=sum(self.C.level[n] for n in nodes)/len(nodes)
        denom=max(1.0, (len(nodes)-1)+0.5*avg_depth)
        return max(0.0, 1.0 - (steiner_edges/denom))

    def gromov_anchor_summary(self, anchor: int, nodes: List[int]) -> Dict[str,float]:
        if not nodes:
            return {"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0}
        z = max(self.C.max_dist, 1e-9)
        vals = []
        if anchor not in self.C.T:
            anchor = self.C.root
        ar = self.C.dist_to_root.get(anchor, 0.0)
        for v in nodes:
            gp = (ar + self.C.dist_to_root.get(v, ar) - self.C.dist(anchor, v)) * 0.5
            vals.append(max(0.0, gp / z))
        return {"gp_mean":float(np.mean(vals)), "gp_min":min(vals), "gp_max":max(vals)}

    def branch_focus(self, nodes: List[int]) -> float:
        from collections import Counter
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
        rads=[self.radial_gap(v,rq,tau) for v in nodes]
        radial_gap_mean=float(np.mean(rads))
        geo=self.geodesic_coherence(nodes)
        cov=self.steiner_coverage(nodes)
        if anchor is None:
            anchor=min(nodes, key=lambda x:self.C.dist_to_root[x])
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

# =============================================================================
# TreeRep cache (igraph → nx tree)
# =============================================================================

def _build_treerep_cache_from_igraph(
    sg: ig.Graph,
    backend: Literal["treerep","bfs"] = "treerep",
    ppr_scores: Optional[np.ndarray] = None
) -> Optional[TreeCache]:
    """무방향 서브그래프(ig) → (treerep|bfs) 트리 → TreeCache."""
    if sg.vcount() == 0:
        return None

    gu = sg.as_undirected() if sg.is_directed() else sg
    gids = [int(x) for x in (gu.vs["global_id"] if "global_id" in gu.vs.attributes() else range(gu.vcount()))]
    has_w = 'weight' in gu.es.attribute_names()
    idx2gid = {i: g for i, g in enumerate(gids)}

    # root gid
    if "root_id" in gu.attributes():
        root_loc = int(gu["root_id"])
        root_gid = idx2gid.get(root_loc, gids[0])
    else:
        root_gid = gids[0]

    # (A) TreeRep
    if backend == "treerep" and gu.ecount() > 0:
        try:
            eps = 1e-6
            w_raw = np.asarray(gu.es["weight"], dtype=float) if has_w else np.ones(gu.ecount(), dtype=float)
            w_dist = 1.0 / (eps + np.maximum(w_raw, 0.0))

            if ppr_scores is not None and "global_id" in gu.vs.attributes():
                pv = np.nan_to_num(np.asarray(ppr_scores, float), nan=0.0)
                pv = np.clip(pv, 0.0, None)
                with np.errstate(divide="ignore"):
                    node_pen = -np.log(pv + eps)
                beta = float(os.getenv("TREEREP_PPR_BETA", "0.00"))
                add_cost = []
                for e in gu.es:
                    u, v = e.source, e.target
                    add_cost.append(beta * 0.5 * (node_pen[gids[u]] + node_pen[gids[v]]))
                w_dist = w_dist + np.asarray(add_cost, float)

            D = np.array(gu.shortest_paths(weights=w_dist), dtype=np.float32)
            if np.isinf(D).any():
                finite = D[np.isfinite(D)]
                cap = float(np.max(finite)) if finite.size else 0.0
                D[~np.isfinite(D)] = cap + 1.0

            treerep = TreeRep(d=D)
            treerep.learn_tree()
            Tloc = treerep.G

            mapping = {loc: int(gids[loc]) for loc in range(min(len(gids), Tloc.number_of_nodes()))}
            T = nx.relabel_nodes(Tloc, mapping, copy=True)
            T.graph["root_id"] = int(root_gid)
            return TreeCache(T, root=root_gid, max_k=16)
        except Exception as e:
            logger.warning(f"[treerep-backend] fallback to BFS due to: {e}")

    # (B) Fallback BFS
    Tfull = nx.Graph()
    [Tfull.add_node(g) for g in gids]
    for e in gu.es:
        u, v = idx2gid[int(e.source)], idx2gid[int(e.target)]
        w = float(e["weight"]) if has_w else 1.0
        Tfull.add_edge(u, v, weight=w)

    if Tfull.number_of_edges() == 0:
        T = nx.Graph()
        [T.add_node(g) for g in gids]
        T.graph["root_id"] = int(root_gid)
        return TreeCache(T, root=root_gid, max_k=16)

    Tbfs = nx.bfs_tree(Tfull, root_gid)
    T = nx.Graph()
    T.graph["root_id"] = int(root_gid)
    [T.add_node(u) for u in Tbfs.nodes()]
    for u, v in Tbfs.edges():
        T.add_edge(u, v, weight=Tfull[u][v].get("weight", 1.0))
    return TreeCache(T, root=root_gid, max_k=16)

# =============================================================================
# Struct normalization helper
# =============================================================================

class StructralSimilarity:
    def __init__(self, mode="cosine"):
        assert mode in ["cosine", "l2", "dot"], "Invalid similarity mode"
        self.mode = mode

    def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(scores, dtype=float)
        if x.size == 0:
            return x
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        if hi - lo < 1e-8:
            return np.full_like(x, 0.5)
        y = (x - lo) / (hi - lo + 1e-8)
        y = np.clip(y, 0.0, 1.0)
        return y

    def compute(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.mode == "cosine":
            return F.cosine_similarity(q, d, dim=-1)
        if self.mode == "dot":
            return torch.sum(q * d, dim=-1)
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
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        if hi - lo < 1e-8:
            return np.full_like(x, 0.5)
        x = np.clip(x, lo, hi)
        y = (x - lo) / (hi - lo + 1e-9)
        return y


def _normalize_struct_v4(x: np.ndarray, mode: str = "robust"):
    """
    mode ∈ {"robust","zscore","softmax"}
    - robust: 퍼센타일 기반 윈저라이즈 → minmax
    - zscore: Z-score 후 sigmoid → [0,1]
    - softmax: 온도=1.0 softmax
    """
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x

    if mode == "robust":
        return StructralSimilarity().robust_normalize_struct(x)

    if mode == "zscore":
        mu, sd = float(np.mean(x)), float(np.std(x))
        if sd < 1e-9:
            z = np.zeros_like(x)
        else:
            z = (x - mu) / sd
        return 1.0 / (1.0 + np.exp(-z))

    if mode == "softmax":
        mx = np.max(x)
        ex = np.exp(x - mx)
        sm = ex / (np.sum(ex) + 1e-9)
        return sm

    return StructralSimilarity().robust_normalize_struct(x)

# =============================================================================
# Metrics logging / utilities
# =============================================================================

class MetricsLogger:
    def __init__(self, save_dir: str, dataset_name: str, fname: str = "struct_metrics.jsonl"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, f"{dataset_name}_{fname}")

    def write(self, row: Mapping[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_deg_hist(g: ig.Graph, exclude_virtual=False) -> dict:
    if g.vcount() == 0:
        return {}
    mask = [True] * g.vcount()
    if exclude_virtual and "is_virtual" in g.vs.attributes():
        mask = [not bool(v) for v in g.vs["is_virtual"]]
    vids = [i for i, ok in enumerate(mask) if ok]
    if not vids:
        return {}
    degs = g.degree(vids, mode="all")
    hist = defaultdict(int)
    for d in degs:
        hist[int(d)] += 1
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
        "dataset": dataset_name,
        "query_id": str(query_id),
        "graph_role": graph_role,
        "v": int(g2.vcount()),
        "e": int(g2.ecount()),
        "directed": bool(g.is_directed()),
    }

    if g2.vcount() > 0:
        comps = g2.components(mode="WEAK")
        comp_sizes = [len(c) for c in comps]
        m["comp_cnt"] = int(len(comp_sizes))
        m["giant_ratio"] = float((max(comp_sizes) / g2.vcount()) if comp_sizes else 0.0)
    else:
        m["comp_cnt"] = 0
        m["giant_ratio"] = 0.0

    hist = _safe_deg_hist(g2, exclude_virtual=False)
    m["deg_hist"] = hist
    if hist:
        deg_list = []
        for k, v in hist.items():
            deg_list.extend([k] * v)
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

    try:
        m["star_ratio"] = 0.0
    except Exception:
        m["star_ratio"] = 0.0

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
                else:
                    m["path_len_mean"] = 0.0
                    m["path_len_p95"] = 0.0
            else:
                m["path_len_mean"] = 0.0
                m["path_len_p95"] = 0.0
        else:
            m["path_len_mean"] = 0.0
            m["path_len_p95"] = 0.0
    except Exception:
        m["path_len_mean"] = 0.0
        m["path_len_p95"] = 0.0

    try:
        bet = np.asarray(g2.betweenness(), dtype=float) if g2.vcount() else np.array([])
        if bet.size:
            m["betw_p90"] = float(np.percentile(bet, 90))
            m["betw_frac_gt_p90"] = float(np.mean(bet > m["betw_p90"]))
            m["betw_gini"] = _gini(bet.tolist())
        else:
            m["betw_p90"] = 0.0
            m["betw_frac_gt_p90"] = 0.0
            m["betw_gini"] = 0.0
    except Exception:
        m["betw_p90"] = 0.0
        m["betw_frac_gt_p90"] = 0.0
        m["betw_gini"] = 0.0

    try:
        clu = g2.transitivity_avglocal_undirected()
        m["clustering_avg"] = float(0.0 if (clu is None or math.isnan(clu)) else clu)
    except Exception:
        m["clustering_avg"] = 0.0

    if ppr_scores is not None and "global_id" in g2.vs.attributes():
        gids = np.asarray(g2.vs["global_id"])
        pv = np.nan_to_num(np.asarray(ppr_scores, dtype=float), nan=0.0)
        sel = pv[gids] if gids.size else np.array([])
        if sel.size and np.sum(sel) > 0:
            p = sel / np.sum(sel)
            m["ppr_entropy"] = _entropy(p)
        else:
            m["ppr_entropy"] = 0.0
    else:
        m["ppr_entropy"] = 0.0

    m["long_tail_hint"] = {"deg_gini": m["deg_gini"], "tail_frac_ge3": m["tail_frac_ge3"]}
    return m


def _to_1d_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().flatten()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        return x
    arr = np.asarray(x, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# QueryGraphBuilder
# =============================================================================

class QueryGraphBuilder:
    def __init__(self, root_node: str = "question", directed: bool = True,
                 save_dir: str = f"./outputs/query_graph_analysis", dataset_name: str = "default"):
        self.root_node = root_node
        self.directed = directed
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_logger = MetricsLogger(save_dir=self.save_dir, dataset_name=self.dataset_name)
        self.log_file = os.path.join(
            self.save_dir, f"{self.dataset_name}_query_graph_degree_distribution.csv"
        )
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("query_id,dataset_name,num_nodes,num_edges,degree_distribution\n")

    def build(self, triples: List[Tuple], query_id: str) -> ig.Graph:
        def norm(x): return str(x).strip()

        if not triples:
            g = ig.Graph(directed=self.directed)
            g.add_vertex(name=self.root_node, is_virtual=True)
            g.vs["global_id"] = [-1]
            g["root_id"] = g.vs.find(name=self.root_node).index
            with open(self.log_file, 'a') as f:
                f.write(f"{query_id},{self.dataset_name},0,0,[]\n")
            return g

        valid_triples = []
        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, p, o = map(norm, t)
                if s and p and o and s != o:
                    valid_triples.append((s, p, o))

        g = ig.Graph(directed=self.directed)
        if not valid_triples:
            g.add_vertex(name=self.root_node, is_virtual=True)
            g.vs["global_id"] = [-1]
            g["root_id"] = g.vs.find(name=self.root_node).index
            with open(self.log_file, 'a') as f:
                f.write(f"{query_id},{self.dataset_name},1,0,[]\n")
            return g

        node_names = set()
        for s, _, o in valid_triples:
            node_names.add(s); node_names.add(o)
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
            if root_id in comp:
                continue
            degs = g.degree(comp, mode="all")
            hub_local = int(np.argmax(degs))
            hub_vid = comp[hub_local]
            if g.get_eid(root_id, hub_vid, directed=False, error=False) == -1:
                g.add_edge(root_id, hub_vid)
                g.es[-1]["relation"] = "root"
                g.es[-1]["relations"] = ["root"]

        non_virtual_nodes = [v for v in g.vs if not v["is_virtual"]]
        if non_virtual_nodes:
            degrees = g.degree(non_virtual_nodes, mode="all")
            degree_counts = defaultdict(int)
            for d in degrees:
                degree_counts[d] += 1
            sorted_degrees = sorted(degree_counts.items(), key=lambda item: item[0])
        else:
            sorted_degrees = []

        with open(self.log_file, 'a') as f:
            f.write(f"{query_id},{self.dataset_name},{g.vcount()},{g.ecount()},{sorted_degrees}\n")
        logger.debug("[QueryGraphBuilder] directed=%s nodes=%d, edges=%d", self.directed, g.vcount(), g.ecount())

        try:
            row = compute_struct_metrics(
                g,
                graph_role="query",
                dataset_name=self.dataset_name,
                query_id=str(query_id),
                root_global_id=int(g["root_id"]) if "root_id" in g.attributes() else None,
                ppr_scores=None
            )
            self.metrics_logger.write(row)
        except Exception as e:
            logger.warning(f"[QueryGraphBuilder] metrics logging failed: {e}")

        g.vs["global_id"] = list(range(g.vcount()))
        g["root_id"] = int(g.vs.find(name=self.root_node).index)

        return g

# =============================================================================
# Subgraph builder / pruner
# =============================================================================

class IDMapper:
    @staticmethod
    def docids_to_vids(doc_ids: Iterable[int], passage_node_idxs: List[int]) -> List[int]:
        return [int(passage_node_idxs[d]) for d in doc_ids]


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
    # subgraph_mode: Literal["bfs", "ppr"] = "ppr"
    subgraph_mode: Literal["bfs", "ppr", "ppr_rooted"] = "ppr_rooted"  # C2 기본을 이것으로
    # 구조용 후보/거리/TreeRep는 무방향으로 처리
    treat_undirected_for_structure: bool = True


class SubgraphBuilder:
    def __init__(self, g: ig.Graph, cfg: SubgraphConfig, node_scores: Optional[np.ndarray] = None):
        self.g = g
        self.cfg = cfg
        self.node_scores = None if node_scores is None else np.nan_to_num(np.asarray(node_scores, float), nan=0.0)
        self.weighted = ('weight' in g.es.attribute_names())
        if self.weighted:
            self.outstr = np.zeros(g.vcount(), dtype=float)
            for e in g.es:
                self.outstr[e.source] += float(e['weight'])
        else:
            self.outstr = None

    def _rank_key(self, u: int, v: int, w: float) -> float:
        if self.cfg.rank_metric == "ppr" and self.node_scores is not None:
            return float(self.node_scores[v])
        if self.cfg.rank_metric == "weight" and self.weighted:
            return float(w)
        if self.cfg.rank_metric == "deg":
            return float(self.g.degree(v))
        return float(w) if self.weighted else 1.0

    def _collect_candidates_by_bfs(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
        visited = {root}
        q = deque([(root, 0)])
        edges: List[Tuple[int,int,float]] = []
        mode = "ALL"
        K = max(1, int(self.cfg.per_hop_topk)) if self.cfg.per_hop_topk else None

        while q:
            u, depth = q.popleft()
            if depth >= self.cfg.max_depth:
                continue

            neighs = []
            for eid in self.g.incident(u, mode=mode):
                e = self.g.es[eid]
                v = e.target if e.source == u else e.source
                w = float(e['weight']) if self.weighted else 1.0
                if w < self.cfg.weight_threshold:
                    continue
                if allow_ids is not None and v not in allow_ids:
                    continue
                neighs.append((u, v, w))

            if K is not None and len(neighs) > K:
                neighs.sort(key=lambda t: self._rank_key(*t), reverse=True)
                neighs = neighs[:K]

            for (uu, vv, ww) in neighs:
                edges.append((uu, vv, ww))
                if vv not in visited:
                    visited.add(vv)
                    q.append((vv, depth + 1))

        if not edges and self.cfg.starter_beam > 0:
            nbr_eids = self.g.incident(root, mode=mode)
            cand = []
            for eid in nbr_eids:
                e = self.g.es[eid]
                v = e.target if e.source == root else e.source
                w = float(e['weight']) if self.weighted else 1.0
                if allow_ids is None or v in allow_ids:
                    cand.append((root, v, w))
            if K is not None and len(cand) > self.cfg.starter_beam:
                cand.sort(key=lambda t: self._rank_key(*t), reverse=True)
                cand = cand[:self.cfg.starter_beam]
            edges = cand
        return edges

    def _collect_candidates_by_ppr(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
        if self.node_scores is None:
            return []
        ppr_sorted_vids = np.argsort(self.node_scores)[::-1]
        selected_vids_set = set()
        for vid in ppr_sorted_vids:
            if allow_ids is not None and vid not in allow_ids:
                continue
            selected_vids_set.add(vid)
            if len(selected_vids_set) >= self.cfg.allow_topk:
                break
        if (0 <= root < self.g.vcount()) and (root not in selected_vids_set):
            selected_vids_set.add(root)

        edges: List[Tuple[int,int,float]] = []
        visited_edges: Set[Tuple[int, int]] = set()

        for u in selected_vids_set:
            for v in self.g.neighbors(u, mode="ALL"):
                if v in selected_vids_set:
                    edge_tuple = tuple(sorted((u, v)))
                    if edge_tuple not in visited_edges:
                        visited_edges.add(edge_tuple)
                        eids = self.g.get_eids([(u, v)], directed=False)
                        if eids:
                            w = float(self.g.es[eids[0]].attributes().get('weight', 1.0))
                            edges.append((u, v, w))
        return edges

    # def collect_candidates(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
    #     if self.cfg.subgraph_mode == "bfs":
    #         return self._collect_candidates_by_bfs(root, allow_ids)
    #     elif self.cfg.subgraph_mode == "ppr":
    #         return self._collect_candidates_by_ppr(root, allow_ids)
    #     else:
    #         raise ValueError(f"Unknown subgraph_mode: {self.cfg.subgraph_mode}")
    def collect_candidates(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
        if self.cfg.subgraph_mode == "bfs":
            return self._collect_candidates_by_bfs(root, allow_ids)
        elif self.cfg.subgraph_mode == "ppr":
            return self._collect_candidates_by_ppr(root, allow_ids)
        elif self.cfg.subgraph_mode == "ppr_rooted":
            # 1) PPR 상위 allow set 구성 (root는 반드시 포함)
            if self.node_scores is None:
                return []
            ppr_sorted = np.argsort(self.node_scores)[::-1]
            topN = set(ppr_sorted[: self.cfg.allow_topk].tolist())
            if 0 <= root < self.g.vcount():
                topN.add(root)
            # 2) BFS 확장(기존 함수 재사용) — 단, allow_ids = topN
            return self._collect_candidates_by_bfs(root, allow_ids=topN)
        else:
            raise ValueError(f"Unknown subgraph_mode: {self.cfg.subgraph_mode}")

    def build_subgraph(self, edges: List[Tuple[int,int,float]], root: Optional[int] = None) -> ig.Graph:
        sg = ig.Graph(directed=False)
        if not edges:
            sg.add_vertex(name=(self.g.vs[root]['name'] if root is not None else "root"))
            sg.vs['global_id'] = [root if root is not None else -1]
            return sg

        nodes = sorted({u for u,_,_ in edges} | {v for _,v,_ in edges})
        idx = {n:i for i,n in enumerate(nodes)}
        sg.add_vertices(len(nodes))
        sg.vs['name'] = [self.g.vs[n]['name'] for n in nodes]
        sg.vs['global_id'] = nodes
        if root is not None:
            sg["root_id"] = int(root) if root in idx else -1

        sg.add_edges([(idx[u], idx[v]) for u,v,_ in edges])
        sg.es['weight'] = [w for *_,w in edges]
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
            for e in g.es:
                self.outstr[e.source] += float(e['weight'])
        else:
            self.outstr = None

    def _P(self, u: int, eid: int) -> float:
        # if self.weighted:
        #     if self.g.es[eid].source != u:
        #         return 0.0
        #     w = float(self.g.es[eid]['weight'])
        #     denom = self.outstr[u]
        #     return (w/denom) if denom > 0 else 0.0
        # d = self.g.degree(u) if not self.g.is_directed() else self.g.outdegree(u)
        # return (1.0/d) if d > 0 else 0.0

        if self.weighted:
            if self.g.is_directed() and self.g.es[eid].source != u:
                return 0.0
            w = float(self.g.es[eid]['weight'])
            denom = self.outstr[u]
            return (w/denom) if denom > 0 else 0.0
        d = self.g.outdegree(u) if self.g.is_directed() else self.g.degree(u)
        return (1.0/d) if d > 0 else 0.0

    
    def _flow(self, u: int, v: int) -> float:
        if self.pi is None:
            return 0.0
        # eid = self.g.get_eid(u, v, directed=True, error=False)
        eid = self.g.get_eid(u, v, directed=self.g.is_directed(), error=False)

        if eid == -1:
            return 0.0
        return float(self.pi[u]) * self._P(u, eid)

    def prune(self, root: int, candidate_edges: List[Tuple[int,int,float]]) -> List[Tuple[int,int,float]]:
        if not candidate_edges or self.pi is None:
            return candidate_edges

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
        if r is None or r <= 0:
            return kept

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

# =============================================================================
# Hyperbolic embedder
# =============================================================================

class HyperbolicEmbedder:
    def __init__(self, hr_instance, embed_dim=32, curvature=1.0, device="cpu"):
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.embed_dim = embed_dim
        self.device = device
        self.hr = hr_instance
        self.feature_extractors = {
            'logdeg': self._get_logdeg,
            'leaf': self._get_leaf,
            # 필요시 확장: 'depth', 'eig_centrality','clustering','pr_undirected','katz'
        }

    def _get_logdeg(self, g):
        return np.log1p(np.asarray(g.degree(mode="all"), dtype=np.float32)).reshape(-1, 1)

    def _get_leaf(self, g):
        deg = np.asarray(g.degree(mode="all"), dtype=np.float32)
        return (deg <= 1).astype(np.float32).reshape(-1, 1)

    def _get_treerep_tree(self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, nx.Graph]:
        n = g.vcount()
        if n < 2:
            return np.zeros((n, n), dtype=np.float32), nx.Graph()

        gu: ig.Graph = g.as_undirected() if g.is_directed() else g

        eps = 1e-6
        if 'weight' in gu.es.attribute_names():
            w_raw = np.asarray(gu.es['weight'], dtype=float)
        else:
            w_raw = np.ones(gu.ecount(), dtype=float)
        w_dist = 1.0 / (eps + np.maximum(w_raw, 0.0))

        if ppr_scores is not None and "global_id" in gu.vs.attributes():
            gids = np.asarray(gu.vs["global_id"], dtype=int)
            pv = np.nan_to_num(np.asarray(ppr_scores, dtype=float), nan=0.0)
            pv = np.clip(pv, 0.0, None)
            with np.errstate(divide='ignore'):
                pen_nodes = -np.log(pv + eps)
            beta = float(os.getenv("TREEREP_PPR_BETA", "0.00"))
            add_cost = []
            for e in gu.es:
                u, v = e.source, e.target
                add_cost.append(beta * 0.5 * (pen_nodes[gids[u]] + pen_nodes[gids[v]]))
            w_dist = w_dist + np.asarray(add_cost, dtype=float)

        try:
            distances = np.array(gu.shortest_paths(weights=w_dist), dtype=np.float32)
        except Exception:
            distances = np.array(gu.shortest_paths(), dtype=np.float32)

        if np.isinf(distances).any():
            finite = distances[np.isfinite(distances)]
            cap = float(np.max(finite)) if finite.size else 0.0
            distances[~np.isfinite(distances)] = cap + 1.0

        try:
            treerep_instance = TreeRep(d=distances)
            treerep_instance.learn_tree()
            learned_tree = treerep_instance.G
        except Exception as e:
            logger.warning(f"TreeRep learning failed: {e}. Returning empty tree.")
            learned_tree = nx.Graph()

        try:
            loc2gid = list(np.asarray(g.vs["global_id"], dtype=int))
            upto = min(len(loc2gid), learned_tree.number_of_nodes())
            mapping = {loc: int(loc2gid[loc]) for loc in range(upto)}
            learned_tree = nx.relabel_nodes(learned_tree, mapping, copy=True)
        except Exception as e:
            logger.warning(f"[TreeRep relabel to global_id] failed: {e}")

        return distances, learned_tree

    def _treerep_features_from_tree(self, learned_tree: nx.Graph, g: ig.Graph) -> np.ndarray:
        n = g.vcount()
        if n == 0:
            return np.zeros((0, 7), dtype=np.float32)

        gids = np.asarray(g.vs["global_id"], dtype=int)
        gid2loc = {int(x): i for i, x in enumerate(gids)}

        try:
            root_gid = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
            if root_gid not in gid2loc:
                root_gid = int(gids[0])
        except Exception:
            root_gid = int(gids[0])

        if learned_tree.number_of_nodes() == 0:
            feats = np.zeros((n, 7), dtype=np.float32)
            if root_gid in gid2loc:
                feats[gid2loc[root_gid], 3] = 1.0
            return feats

        depth = np.zeros(n, dtype=np.float32)
        try:
            dist = dict(nx.single_source_dijkstra_path_length(learned_tree, root_gid, weight='weight'))
        except Exception:
            dist = {}
        for gid, i in gid2loc.items():
            depth[i] = float(dist.get(gid, 0.0))
        max_depth = float(depth.max())
        depth_norm = (depth / (max_depth + 1e-6)).astype(np.float32) if max_depth > 0 else depth

        deg_arr = np.zeros(n, dtype=np.float32)
        for gid, i in gid2loc.items():
            if gid in learned_tree:
                deg_arr[i] = float(learned_tree.degree(gid))
        leaf = (deg_arr <= 1.0).astype(np.float32)

        parent_w = np.zeros(n, dtype=np.float32)
        for gid, i in gid2loc.items():
            if gid in learned_tree:
                nbrs = list(learned_tree.neighbors(gid))
                if nbrs:
                    parent_w[i] = min(learned_tree[gid][v].get('weight', 1.0) for v in nbrs)
        max_pw = float(parent_w.max())
        parent_w_norm = (parent_w / (max_pw + 1e-6)).astype(np.float32) if max_pw > 0 else parent_w

        is_root = np.zeros(n, dtype=np.float32)
        if root_gid in gid2loc:
            is_root[gid2loc[root_gid]] = 1.0

        subtree_size = np.zeros(n, dtype=np.float32)
        try:
            Tdir = nx.bfs_tree(learned_tree, root_gid)
            post = list(reversed(list(nx.topological_sort(Tdir))))
            size_map = {u: 1 for u in Tdir.nodes()}
            for u in post:
                for v in Tdir.successors(u):
                    size_map[u] += size_map[v]
            for gid, i in gid2loc.items():
                subtree_size[i] = float(size_map.get(gid, 1))
        except Exception:
            for gid, i in gid2loc.items():
                subtree_size[i] = 1.0
        subtree_size_norm = (subtree_size / max(1.0, learned_tree.number_of_nodes())).astype(np.float32)

        max_deg = float(deg_arr.max()) if deg_arr.size else 0.0
        branching_factor_norm = (deg_arr / (max_deg + 1e-6)).astype(np.float32) if max_deg > 0 else deg_arr

        centroid_flag = np.zeros(n, dtype=np.float32)
        try:
            centers = nx.center(learned_tree, e=None, usebounds=False)
            if len(centers) == 0:
                centers = nx.center(learned_tree, weight='weight')
            for c in centers:
                if c in gid2loc:
                    centroid_flag[gid2loc[c]] = 1.0
        except Exception:
            pass

        feats = np.stack([
            depth_norm,                 # 0
            leaf,                       # 1
            # parent_w_norm,            # 2 (비활성; 필요시 채널 확장)
            is_root,                    # 3
            # subtree_size_norm,        # 4
            # branching_factor_norm,    # 5
            # centroid_flag,            # 6
        ], axis=1).astype(np.float32)

        return feats

    def graph_to_tensor(
        self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None,
        passage_embeddings: Optional[np.ndarray] = None,
        entity_embeddings: Optional[np.ndarray] = None,
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False
    ):
        if g.vcount() == 0:
            return torch.zeros(1, self.embed_dim, device=self.device)

        feature_list = []

        if use_treerep_metric:
            _, learned_tree = self._get_treerep_tree(g, ppr_scores=ppr_scores)
            feats_treerep = self._treerep_features_from_tree(learned_tree, g)
            feature_list.append(feats_treerep)
            for extractor_name, extractor_func in self.feature_extractors.items():
                feats_general = extractor_func(g)
                feature_list.append(feats_general)
        else:
            for extractor in self.feature_extractors.values():
                feature_list.append(extractor(g))

        if use_semantic_features and g.vcount() > 0:
            semantic_features = []
            global_ids = g.vs["global_id"]
            for v_id in global_ids:
                node_key = self.hr.graph.vs[v_id]["name"]
                embed = np.zeros(self.hr.global_config.embedding_dim, dtype=np.float32)
                if node_key.startswith("chunk-") and passage_embeddings is not None:
                    row = self.hr.chunk_embedding_store.get_row(node_key)
                    if row and 'embedding' in row:
                        embed = row['embedding']
                elif node_key.startswith("entity-") and entity_embeddings is not None:
                    row = self.hr.entity_embedding_store.get_row(node_key)
                    if row and 'embedding' in row:
                        embed = row['embedding']
                semantic_features.append(embed)
            if semantic_features:
                semantic_features = np.vstack(semantic_features)
                feature_list.append(semantic_features)

        if not feature_list:
            X = np.zeros((g.vcount(), self.embed_dim), dtype=np.float32)
        else:
            X = np.hstack(feature_list)

        num_features = X.shape[1]

        if num_features > 0:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True)
            if np.all(sd < 1e-6):
                Z = np.zeros_like(X, dtype=np.float32)
            else:
                sd = np.maximum(sd, 1e-6)
                Z = np.clip((X - mu) / sd, -3.0, 3.0)
        else:
            Z = np.zeros((g.vcount(), self.embed_dim), dtype=np.float32)

        return torch.tensor(Z, dtype=torch.float32, device=self.device)

    def embed_lorentz(
        self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None,
        passage_embeddings: Optional[np.ndarray] = None,
        entity_embeddings: Optional[np.ndarray] = None,
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False
    ) -> torch.Tensor:
        if g.vcount() == 0:
            return None

        x = self.graph_to_tensor(
            g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
            entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
            use_treerep_metric=use_treerep_metric
        )
        with torch.no_grad():
            if g.vcount() > 1:
                max_norm = 5.0
                n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
                x = x * torch.clamp(max_norm / n, max=1.0)
            x_lorentz = self.manifold.expmap0(x)
            x_lorentz = torch.nan_to_num(x_lorentz, nan=0.0, posinf=0.0, neginf=0.0)
            if g.vcount() > 1:
                pooled, _ = torch.max(x_lorentz, dim=0, keepdim=False)  # Max Pooling
            else:
                pooled = x_lorentz.mean(dim=0)
            pooled = self.manifold.projx(pooled)
            if torch.linalg.norm(pooled).item() < 1e-8:
                pooled = pooled + 1e-3 * torch.randn_like(pooled)
                pooled = self.manifold.projx(pooled)
            return pooled

    def logmap0_safe(self, lorentz_vec: torch.Tensor) -> torch.Tensor:
        t = self.manifold.logmap0(lorentz_vec)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def embed_and_logmap0(
        self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None,
        passage_embeddings: Optional[np.ndarray] = None,
        entity_embeddings: Optional[np.ndarray] = None,
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False
    ) -> torch.Tensor:
        y = self.embed_lorentz(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
                               entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                               use_treerep_metric=use_treerep_metric)
        if y is None:
            return torch.zeros(self.embed_dim, device=self.device)
        if torch.linalg.norm(y).item() < 1e-8:
            y = y + 1e-3 * torch.randn_like(y)
            y = self.manifold.projx(y)
        t = self.logmap0_safe(y)
        if torch.linalg.norm(t).item() < 1e-8:
            t[0] = 1e-3
        return t

    def embed_nodes_lorentz(
        self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None,
        passage_embeddings: Optional[np.ndarray] = None,
        entity_embeddings: Optional[np.ndarray] = None,
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False
    ) -> torch.Tensor:
        if g.vcount() == 0:
            return torch.empty(0, self.embed_dim, device=self.device)
        x = self.graph_to_tensor(
            g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
            entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
            use_treerep_metric=use_treerep_metric
        )
        with torch.no_grad():
            max_norm = 5.0
            n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
            x = x * torch.clamp(max_norm / n, max=1.0)
        return self.manifold.expmap0(x)

    def shared_standardize(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if not tensors:
            return tensors
        all_data = torch.cat([t.view(-1) for t in tensors], dim=0)
        mean = torch.mean(all_data)
        std = torch.std(all_data)
        if std.item() == 0:
            std = torch.tensor(1.0, device=all_data.device)
        standardized = []
        for t in tensors:
            z = (t - mean) / std
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            standardized.append(z)
        return standardized

# =============================================================================
# HyperReranker
# =============================================================================

class HyperReranker:
    def __init__(self, g: ig.Graph, save_dir: str = "./outputs/subgraph_analysis", dataset_name: str = "default"):
        self.g = g
        self.metrics_logger = MetricsLogger(save_dir=save_dir, dataset_name=dataset_name)
        self.cfg = None

    @staticmethod
    def make_allow_ids(g: ig.Graph, seeds_vids: List[int],
                       ppr_scores: Optional[np.ndarray], allow_topk: int) -> Optional[Set[int]]:
        if ppr_scores is None or not allow_topk or allow_topk <= 0:
            return None
        ppr = np.nan_to_num(np.asarray(ppr_scores, float), nan=0.0, posinf=0.0, neginf=0.0)
        topN = np.argsort(-ppr)[:allow_topk].tolist()
        allow = set(topN)
        for s in seeds_vids:
            for nb in g.neighbors(s, mode="all"):
                allow.add(nb)
        return allow

    def build_pruned_subgraphs_for_topk_passages(
        self,
        topk_doc_ids: List[int],
        topk_scores: List[float],
        passage_node_idxs: List[int],
        last_ppr_scores: Optional[np.ndarray],
        cfg: SubgraphConfig,
        head_k: int = 30,
    ) -> Tuple[List[ig.Graph], List[int], List[int]]:
        if not topk_doc_ids:
            return [], [], []
        self.cfg = cfg

        sem = np.asarray(topk_scores, float)
        head_idx = np.argsort(-sem)[: min(head_k, len(topk_doc_ids))]
        head_doc_ids = [topk_doc_ids[i] for i in head_idx]
        head_vids = IDMapper.docids_to_vids(head_doc_ids, passage_node_idxs)

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
                    sg,
                    graph_role="subgraph",
                    dataset_name=getattr(self, "dataset_name", "default") if hasattr(self, "dataset_name") else "default",
                    query_id=str(getattr(self, "query_id", "unknown")) if hasattr(self, "query_id") else "unknown",
                    root_global_id=rid,
                    ppr_scores=last_ppr_scores,
                    exclude_virtual=False
                )
                row.update({
                    "root_doc_vid": int(rid),
                    "root_doc_id": int(head_doc_ids[len(subgraphs)]) if len(subgraphs) < len(head_doc_ids) else None,
                    "subgraph_idx": int(len(subgraphs)),
                    "collect_mode": self.cfg.subgraph_mode if hasattr(self, "cfg") else "unknown"
                })
                self.metrics_logger.write(row)
            except Exception as e:
                logger.warning(f"[SubgraphMetrics] logging failed: {e}")

            subgraphs.append(sg)
            logger.debug(f"[Subgraph] mode={cfg.subgraph_mode} root={rid} cands={len(cands)} pruned={len(pruned)} v={sg.vcount()} e={sg.ecount()}")
        return subgraphs, head_doc_ids, head_vids

    # ---------- Structural scoring backends ----------

    def _struct_scores_tangent(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        embedder: HyperbolicEmbedder,
        scorer: StructralSimilarity,
        ppr_scores: Optional[np.ndarray] = None,
        passage_embeddings: Optional[np.ndarray] = None,
        entity_embeddings: Optional[np.ndarray] = None,
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            tq = embedder.embed_and_logmap0(
                query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
                entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                use_treerep_metric=use_treerep_metric
            )
            tdocs = [
                embedder.embed_and_logmap0(
                    g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
                    entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                    use_treerep_metric=use_treerep_metric
                ) for g in subgraphs
            ]
        try:
            if tdocs:
                T = torch.stack(tdocs)
                C = F.cosine_similarity(T.unsqueeze(1), T.unsqueeze(0), dim=-1)
                mask = ~torch.eye(C.size(0), dtype=bool, device=C.device)
                vals = C[mask]
                logger.debug(
                    "[EMB-DIAG][raw] pairwise_cos_mean=%.4f p95=%.4f",
                    vals.mean().item(),
                    vals.kthvalue(max(1, int(0.95 * len(vals))))[0].item()
                )
                cos_raw = scorer.compute_listwise([tq for _ in tdocs], tdocs)
                q05, q50, q95 = torch.quantile(cos_raw, torch.tensor([0.05, 0.5, 0.95], device=cos_raw.device))
                logger.debug(
                    "[EMB-DIAG][raw] qvdoc_cos p05=%.4f p50=%.4f p95=%.4f",
                    q05.item(), q50.item(), q95.item()
                )
        except Exception as e:
            logger.debug("[EMB-DIAG] raw diag skipped: %s", e)

        tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
        q_list = [tq_std for _ in range(len(tdocs_std))]
        struct_head = scorer.compute_listwise(q_list, tdocs_std)
        return _to_1d_numpy(struct_head)

    def _struct_scores_lorentz(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        embedder: HyperbolicEmbedder,
        temp: float = 2.0,
        ppr_scores: Optional[np.ndarray] = None,
        passage_embeddings: Optional[np.ndarray] = None,
        entity_embeddings: Optional[np.ndarray] = None,
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            qL = embedder.embed_lorentz(
                query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
                entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                use_treerep_metric=use_treerep_metric
            )
            dL = [
                embedder.embed_lorentz(
                    g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
                    entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                    use_treerep_metric=use_treerep_metric
                ) for g in subgraphs
            ]

            qL = qL if qL is not None else torch.zeros(embedder.embed_dim, device=embedder.device)
            dL_tensors = []
            for x in dL:
                if x is None:
                    dL_tensors.append(torch.zeros(embedder.embed_dim, device=embedder.device))
                else:
                    dL_tensors.append(x)

            dists2 = torch.stack([
                embedder.manifold.dist2(qL.unsqueeze(0), x.unsqueeze(0)).squeeze()
                for x in dL_tensors
            ])
        d2 = _to_1d_numpy(dists2)
        sim = -d2
        if sim.size:
            norm_sim = StructralSimilarity().normalize(sim)
            sim = 1.0 / (1.0 + np.exp(-((norm_sim - 0.5) / (1.0 / temp))))
        return sim

    def _struct_scores_node_level(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        embedder: HyperbolicEmbedder,
        ppr_scores: Optional[np.ndarray] = None,
        passage_embeddings: Optional[np.ndarray] = None,
        entity_embeddings: Optional[np.ndarray] = None,
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            q_node_embeddings = embedder.embed_nodes_lorentz(
                query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
                entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                use_treerep_metric=use_treerep_metric
            )
            if q_node_embeddings.numel() == 0:
                return np.zeros(len(subgraphs))

            # Top-k pooling: 각 쿼리 노드가 문서 노드 중 최고 매칭과 맺도록
            struct_scores = []
            for doc_g in subgraphs:
                d_node_embeddings = embedder.embed_nodes_lorentz(
                    doc_g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
                    entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                    use_treerep_metric=use_treerep_metric
                )
                if d_node_embeddings.numel() == 0:
                    struct_scores.append(0.0)
                    continue
                tq = embedder.logmap0_safe(q_node_embeddings)
                td = embedder.logmap0_safe(d_node_embeddings)
                similarity_matrix = F.cosine_similarity(tq.unsqueeze(1), td.unsqueeze(0), dim=-1)
                max_sim_per_query_node = torch.max(similarity_matrix, dim=1)[0]
                score = torch.mean(max_sim_per_query_node)
                struct_scores.append(score.item())
        return np.array(struct_scores)

    def _struct_scores_gromov(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        landmark_k: int = 10,
        ppr_scores: Optional[np.ndarray] = None,
        landmark_weighting: str = "none",   # "none"|"degree"|"betweenness"
    ) -> np.ndarray:

        def _pick_landmarks(cache: TreeCache, L: int) -> List[int]:
            nodes = list(cache.T.nodes())
            if not nodes:
                return []
            lm = [cache.root]

            def farthest_from(S):
                best, bestd = cache.root, -1.0
                for v in nodes:
                    d = min(cache.dist(v, s) for s in S)
                    if d > bestd:
                        best, bestd = v, d
                return best

            while len(lm) < min(L, len(nodes)):
                lm.append(farthest_from(lm))
            return lm

        def _distmat(cache: TreeCache, nodes: List[int]) -> np.ndarray:
            L = len(nodes)
            M = np.zeros((L, L), dtype=np.float32)
            for i in range(L):
                for j in range(i + 1, L):
                    d = cache.dist(nodes[i], nodes[j])
                    M[i, j] = M[j, i] = d
            m = float(np.max(M)) if M.size else 0.0
            return (M / m) if m > 0 else M

        def _landmark_weights(G: nx.Graph, nodes: List[int], mode: str) -> np.ndarray:
            if not nodes:
                return np.zeros((0,), dtype=np.float32)
            if mode == "degree":
                w = np.array([G.degree(n) for n in nodes], dtype=np.float32)
            elif mode == "betweenness":
                try:
                    bt = nx.betweenness_centrality(G, normalized=True, weight="weight")
                    w = np.array([bt.get(n, 0.0) for n in nodes], dtype=np.float32)
                except Exception:
                    w = np.ones((len(nodes),), dtype=np.float32)
            else:  # "none"
                w = np.ones((len(nodes),), dtype=np.float32)
            s = float(np.sum(w))
            return (w / (s + 1e-9)).astype(np.float32)

        qcache = _build_treerep_cache_from_igraph(
            query_graph.as_undirected() if query_graph.is_directed() else query_graph,
            backend="treerep",
            ppr_scores=ppr_scores,
        )
        if qcache is None:
            return np.zeros(len(subgraphs))
        q_lm = _pick_landmarks(qcache, landmark_k)
        Q = _distmat(qcache, q_lm)
        qw = _landmark_weights(qcache.T, q_lm, landmark_weighting)

        sims = []
        for sg in subgraphs:
            dcache = _build_treerep_cache_from_igraph(
                sg, backend="treerep", ppr_scores=ppr_scores
            )
            if dcache is None:
                sims.append(0.0)
                continue

            d_lm = _pick_landmarks(dcache, min(landmark_k, len(dcache.T.nodes())))
            L = min(len(q_lm), len(d_lm))
            if L <= 1:
                sims.append(0.0)
                continue

            QL = Q[:L, :L]
            DL = _distmat(dcache, d_lm)[:L, :L]
            dw = _landmark_weights(dcache.T, d_lm, landmark_weighting)[:L]

            W = np.outer(qw[:L], dw)
            W = W / (np.sum(W) + 1e-9)
            score = 1.0 - float(np.sum(W * np.abs(QL - DL)))
            sims.append(max(0.0, min(1.0, score)))
        return np.asarray(sims, dtype=float)

    def _struct_scores_struct_feat(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        query_radial_anchor: Optional[float] = None,
        radial_tau: float = 0.5,
        ppr_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        rq = 0.5
        if query_radial_anchor is not None:
            rq = float(query_radial_anchor)
        else:
            qcache = _build_treerep_cache_from_igraph(
                query_graph.as_undirected() if query_graph.is_directed() else query_graph,
                backend="treerep",
                ppr_scores=ppr_scores,
            )
            if qcache is not None:
                try:
                    qnodes = [
                        int(x)
                        for x in (
                            query_graph.vs["global_id"]
                            if "global_id" in query_graph.vs.attributes()
                            else list(range(query_graph.vcount()))
                        )
                    ]
                    rads = [qcache.radial(g) for g in qnodes]
                    if rads:
                        rq = float(np.median(rads))
                except Exception:
                    pass

        scores = []
        for sg in subgraphs:
            cache = _build_treerep_cache_from_igraph(
                sg, backend="treerep", ppr_scores=ppr_scores
            )
            if cache is None:
                scores.append(0.0)
                continue

            ext = StructFeatureExtractor(cache)
            cand = [
                int(x)
                for x in (
                    sg.vs["global_id"]
                    if "global_id" in sg.vs.attributes()
                    else list(range(sg.vcount()))
                )
            ]
            feats = ext.features_for_group(cand, rq, tau=radial_tau, anchor=None)
            w = {
                "radial_gap_mean": 0.35,
                "geo_coherence": 0.30,
                "steiner_coverage": 0.25,
                "gp_mean": 0.05,
                "branch_focus": 0.05,
            }
            s = 0.0
            for k, a in w.items():
                s += a * float(feats.get(k, 0.0))
            scores.append(max(0.0, min(1.0, s)))
        return np.asarray(scores, dtype=float)

    # ---------- Fusion / Rerank ----------

    def rerank_with_structure(
        self,
        query_graph: Optional[ig.Graph],
        passage_topk_ids: List[int],
        passage_topk_scores: List[float],
        passage_node_idxs: List[int],
        last_ppr_scores: Optional[np.ndarray],
        cfg: SubgraphConfig,
        embedder: HyperbolicEmbedder,
        scorer: StructralSimilarity,
        passage_embeddings: np.ndarray,
        entity_embeddings: np.ndarray,
        # 공통/기존
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False,
        gating_enabled: bool = False,
        ensemble_ratio: float = 0.4,
        radial_tau: float = 0.5,
        landmark_k: int = 10,
        alpha: float = 0.8,
        struct_k: int = 30,
        k_rrf: int = 20,
        pin_k: int = 1,
        device: str = "cpu",
        save_cb: Optional[Callable[[ig.Graph, str, dict], None]] = None,
        save_ctx: Optional[dict] = None,
        sim_cac_mode: Literal["tangent","lorentz","node_level","gromov","struct_feat","ensemble"] = "tangent",
        lorentz_temp: float = 2.0,
        struct_norm: Literal["robust","zscore","softmax"] = "robust",
        sem_struct_mode: Literal["linear","gated_add"] = "linear",
        struct_nonlin: Literal["none","tanh"] = "none",
        landmark_weighting: Literal["none","degree","betweenness"] = "none",
        node_ratio: float = 0.00,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        rho_low: float = -0.20,
        rho_high: float = 0.30,
        # [B-옵션] 안정화/핀/Top-M 보정
        stabilize_struct: bool = False,
        enable_soft_pin: bool = False,
        soft_pin_margin_lo: float = 0.05,
        soft_pin_margin_hi: float = 0.10,
        topm_refine: bool = False,
        topm_M: int = 20,
        topm_refine_weight: float = 0.15,
        topm_adaptive: bool = False,
    ) -> Tuple[List[int], List[float]]:

        def _clip(a, lo, hi): 
            return max(lo, min(hi, a))

        passage_topk_ids = list(np.asarray(passage_topk_ids).tolist())
        passage_topk_scores = list(np.asarray(passage_topk_scores, dtype=float).tolist())

        if query_graph is None or len(passage_topk_ids) == 0:
            return passage_topk_ids, passage_topk_scores

        # 1) subgraphs
        subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
            passage_topk_ids, passage_topk_scores, passage_node_idxs,
            last_ppr_scores, cfg, head_k=struct_k
        )
        if not subgraphs:
            return passage_topk_ids, passage_topk_scores

        # 2) head sem
        sem = np.asarray(passage_topk_scores, float)
        head_idx = np.argsort(-sem)[:len(subgraphs)]
        sem_head = sem[head_idx]

        # 3) struct scores
        if sim_cac_mode == "tangent":
            struct_head = self._struct_scores_tangent(
                query_graph, subgraphs, embedder, scorer, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
            )
        elif sim_cac_mode == "lorentz":
            struct_head = self._struct_scores_lorentz(
                query_graph, subgraphs, embedder, temp=lorentz_temp, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
            )
        elif sim_cac_mode == "node_level":
            struct_head = self._struct_scores_node_level(
                query_graph, subgraphs, embedder, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
            )
        elif sim_cac_mode == "gromov":
            struct_head = self._struct_scores_gromov(
                query_graph, subgraphs, landmark_k=landmark_k,
                ppr_scores=last_ppr_scores, landmark_weighting=landmark_weighting
            )
        elif sim_cac_mode == "struct_feat":
            struct_head = self._struct_scores_struct_feat(
                query_graph, subgraphs, query_radial_anchor=None, radial_tau=radial_tau, ppr_scores=last_ppr_scores
            )
        elif sim_cac_mode == "ensemble":
            s1 = self._struct_scores_struct_feat(
                query_graph, subgraphs, None, radial_tau, ppr_scores=last_ppr_scores
            )
            s2 = self._struct_scores_gromov(
                query_graph, subgraphs, landmark_k=landmark_k,
                ppr_scores=last_ppr_scores, landmark_weighting=landmark_weighting
            )
            s3 = self._struct_scores_node_level(
                query_graph, subgraphs, embedder, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
            )
            node_ratio_clamped = float(np.clip(node_ratio, 0.0, 1.0))
            struct_pie = 1.0 - node_ratio_clamped
            w1 = struct_pie * float(ensemble_ratio)
            w2 = struct_pie * float(1.0 - ensemble_ratio)
            w3 = node_ratio_clamped
            s1, s2, s3 = np.asarray(s1, float), np.asarray(s2, float), np.asarray(s3, float)
            assert s1.shape == s2.shape == s3.shape, f"shape mismatch: {s1.shape}, {s2.shape}, {s3.shape}"
            struct_head = w1 * s1 + w2 * s2 + w3 * s3
        else:
            raise ValueError(f"Unknown mode: {sim_cac_mode}")

        if np.asarray(struct_head).size == 0:
            return passage_topk_ids, passage_topk_scores

        # 4) normalize → stabilize
        s = _normalize_struct_v4(np.asarray(struct_head, float), mode=struct_norm)

        if stabilize_struct:
            var = float(np.var(s))
            tau = 0.20 if var < 0.01 else (0.30 if var < 0.02 else 0.40)
            s = np.tanh((s - 0.5) / tau) * 0.5 + 0.5

            winsor_p = 0.05
            tanh_w   = 0.20
            lo, hi = np.percentile(s, 100*winsor_p), np.percentile(s, 100*(1-winsor_p))
            if hi > lo:
                s = np.clip(s, lo, hi)
                s = (s - s.min()) / (s.max() - s.min() + 1e-12)
                s = np.tanh((s - 0.5) / tanh_w) * 0.5 + 0.5
            if struct_nonlin == "tanh":
                s = np.tanh((s - 0.5) / 0.15) * 0.5 + 0.5

        # 5) RRF
        def _ranks(x: np.ndarray) -> np.ndarray:
            return (np.argsort(np.argsort(-x)) + 1).astype(np.int32)
        rank_sem    = _ranks(sem_head)
        rank_struct = _ranks(s)
        rrf_sem     = 1.0 / (k_rrf + rank_sem)
        rrf_struct  = 1.0 / (k_rrf + rank_struct)

        # 6) gating + dynamic alpha
        rho = 0.0
        def _is_const_vec(x: np.ndarray) -> bool:
            return x.size == 0 or np.allclose(x, x[0])
        if gating_enabled and not (_is_const_vec(sem_head) or _is_const_vec(s)):
            try:
                rho = float(spearmanr(sem_head, s).correlation)
                if np.isnan(rho): rho = 0.0
            except Exception:
                rho = 0.0
        gate = (np.clip(0.6 + 0.4 * max(0.0, rho), 0.6, 1.0)
                if gating_enabled else 1.0)
        if gating_enabled and rho <= -0.2:
            rrf_struct = 1.0 - rrf_struct

        alpha_eff = float(alpha)
        if (alpha_min is not None) and (alpha_max is not None) and (rho_high > rho_low):
            t = np.clip((rho - float(rho_low)) / (float(rho_high) - float(rho_low)), 0.0, 1.0)
            t = float(3*t*t - 2*t*t*t)  # smoothstep
            alpha_eff = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * t

        if sem_struct_mode == "linear":
            final_head = alpha_eff * rrf_sem + (1.0 - alpha_eff) * gate * rrf_struct
        elif sem_struct_mode == "gated_add":
            gate_w = 1.0 / (1.0 + np.exp(-(rho - 0.10) / 0.15))
            g_eff = gate * gate_w
            final_head = alpha_eff * rrf_sem + (1.0 - alpha_eff) * g_eff * rrf_struct
        else:
            final_head = alpha_eff * rrf_sem + (1.0 - alpha_eff) * gate * rrf_struct

        # 7) 보호핀(soft pin 포함)
        head_pairs = list(zip([passage_topk_ids[i] for i in head_idx], final_head.tolist()))

        pin_k_eff = pin_k
        if enable_soft_pin:
            sem_sorted = np.sort(sem_head)[::-1]
            if sem_sorted.size >= 2:
                margin = float(sem_sorted[0] - sem_sorted[1]) / (abs(sem_sorted[0]) + 1e-6)
                if margin >= soft_pin_margin_hi:
                    pin_k_eff = max(pin_k, 2)
                elif margin >= soft_pin_margin_lo:
                    pin_k_eff = max(pin_k, 1)
                else:
                    pin_k_eff = 0
        pin_local_idx = np.argsort(-sem_head)[:pin_k_eff]
        pinned_ids    = [passage_topk_ids[head_idx[i]] for i in pin_local_idx]
        pinned_pairs  = [(passage_topk_ids[head_idx[i]], float(final_head[i])) for i in pin_local_idx]

        others        = [(did, sc) for (did, sc) in head_pairs if did not in pinned_ids]
        others_sorted = sorted(others, key=lambda x: x[1], reverse=True)
        head_pairs_sorted = pinned_pairs + [p for p in others_sorted if p[0] not in set(pinned_ids)]

        # 8) Top-M node-level 정밀 보정
        if topm_refine and len(head_pairs_sorted) > 0:
            M_fix = int(topm_M)
            W_fix = float(topm_refine_weight)
            M_eff, W_eff = M_fix, W_fix
            if topm_adaptive:
                var_s = float(np.var(s))
                M_eff = _clip(int(16 + 80 * var_s), 12, 28)
                W_eff = _clip(0.10 + 0.60 * var_s, 0.10, 0.22)

            M = min(M_eff, len(head_pairs_sorted))
            if M > 0:
                topM_ids = [doc_id for doc_id, _ in head_pairs_sorted[:M]]
                topM_graphs, topM_idx = [], []
                head_id_list = [passage_topk_ids[i] for i in head_idx]
                for i, (doc_id, _) in enumerate(head_pairs_sorted[:M]):
                    try:
                        loc = head_id_list.index(doc_id)
                        topM_graphs.append(subgraphs[loc]); topM_idx.append(i)
                    except ValueError:
                        continue

                if len(topM_graphs) > 0:
                    s_refine = self._struct_scores_node_level(
                        query_graph, topM_graphs, embedder,
                        ppr_scores=last_ppr_scores,
                        passage_embeddings=passage_embeddings,
                        entity_embeddings=entity_embeddings,
                        use_semantic_features=use_semantic_features,
                        use_treerep_metric=use_treerep_metric
                    )
                    s_refine = _normalize_struct_v4(s_refine, mode=struct_norm)
                    w = float(_clip(W_eff, 0.0, 0.5))
                    for j, i_head in enumerate(topM_idx):
                        head_pairs_sorted[i_head] = (
                            head_pairs_sorted[i_head][0],
                            float((1.0 - w) * head_pairs_sorted[i_head][1] + w * s_refine[j])
                        )
                    head_pairs_sorted[:M] = sorted(head_pairs_sorted[:M], key=lambda x: x[1], reverse=True)

        # 9) 나머지 merge
        orig_scores  = dict(zip(passage_topk_ids, passage_topk_scores))
        final_scores = {doc_id: score for doc_id, score in head_pairs_sorted}
        merged_ids, merged_scores = [], []
        for doc_id, score in head_pairs_sorted:
            merged_ids.append(doc_id); merged_scores.append(score)
        remaining_ids = [did for did in passage_topk_ids if did not in final_scores]
        remaining_pairs = sorted(
            [(did, orig_scores[did]) for did in remaining_ids],
            key=lambda x: x[1], reverse=True
        )
        for doc_id, score in remaining_pairs:
            merged_ids.append(doc_id); merged_scores.append(score)

        try:
            logger.info(f"[FUSION] k_rrf={k_rrf} gate={float(gating_enabled)} rho={rho:.3f} alpha_eff={alpha_eff:.3f} pin_k_eff={pin_k_eff} pinned_ids={pinned_ids}")
        except Exception:
            pass

        return merged_ids, merged_scores
