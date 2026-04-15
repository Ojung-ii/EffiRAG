from __future__ import annotations
import igraph as ig
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Set, Callable, Literal, Dict, Any, Mapping
from collections import deque, defaultdict, Counter
import logging, os, json, math, random, time
import pandas as pd
from scipy.stats import spearmanr
import networkx as nx
import geoopt
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

# =====[ TreeRep 더미 제거 ]=====
# -------------------------------------------------------------------------------------
# 전역 캐시(선택): 서브그래프 → TreeCache
# -------------------------------------------------------------------------------------
_TRE_CACHE: Dict[int, "TreeCache"] = {}

# ======================================================================================
# TreeCache (변경 없음)
# ======================================================================================
class TreeCache:
    def __init__(self, T: nx.Graph, root: Optional[int] = None, max_k: int = 16):
        if T.number_of_nodes() == 0:
            raise ValueError("Empty tree")
        self.T = T
        # 1) 루트 보정: 트리에 없으면 임의 노드로 교체(가능하면 중앙)
        if root is None or root not in T:
            try:
                centers = nx.center(T)
                root = centers[0] if centers else list(T.nodes())[0]
            except Exception:
                root = list(T.nodes())[0]
        self.root = root
        self.max_k = max_k

        # 메타
        self.parent: Dict[int, Optional[int]] = {}
        self.level: Dict[int, int] = {}
        self.dist_to_root: Dict[int, float] = {}
        self.nodes = list(T.nodes())
        self.index_of = {v: i for i, v in enumerate(self.nodes)}
        self._dfs_root_fill_all()   # 전체 컴포넌트 안전 채움

        # LCA 테이블
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
        """
        트리의 가중 지름(직경) 추정: 두 번 스윕 O(V).
        """
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
        # LCA 로직 유지
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
        # dist 로직 유지
        if u not in self.dist_to_root or v not in self.dist_to_root:
            a = self.root
            du = self.dist_to_root.get(u, self.dist_to_root.get(a, 0.0))
            dv = self.dist_to_root.get(v, self.dist_to_root.get(a, 0.0))
            return abs(du - dv)
        a = self.lca(u, v)
        return self.dist_to_root[u] + self.dist_to_root[v] - 2.0 * self.dist_to_root[a]

    def radial(self, v) -> float:
        # radial 로직 유지
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
# StructFeatureExtractor (O(V^2) 로직 사용하지 않음)
# ======================================================================================
class StructFeatureExtractor:
    """QA-지향 구조피처(그룹 단위)."""
    def __init__(self, cache: TreeCache):
        self.C = cache

    def radial_gap(self, node, rq: float, tau: float=0.5) -> float:
        gap = abs(self.C.radial(node)-rq)
        return max(0.0, 1.0 - (gap/(tau+1e-9)))

    def geodesic_coherence(self, nodes: List[int], max_pairs: int=200) -> float:
        # O(V^2) 연산: RRF 모드에서 사용하지 않음
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
        # O(V^2) 연산: RRF 모드에서 사용하지 않음
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
        # Gromov 유사도 관련 피처 추출 로직 유지 (O(V log V)로 비교적 안전)
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
        # Branch Focus 로직 유지
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
        # 그룹 피처 추출 로직 유지
        if not nodes:
            return {"radial_gap_mean":0.0,"geo_coherence":1.0,"steiner_coverage":1.0,"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0,"branch_focus":1.0}
        nodes = [v for v in nodes if v in self.C.T]
        if not nodes:
            return {"radial_gap_mean":0.0,"geo_coherence":1.0,"steiner_coverage":1.0,"gp_mean":0.0,"gp_min":0.0,"gp_max":0.0,"branch_focus":1.0}
        rads=[self.radial_gap(v,rq,tau) for v in nodes]
        radial_gap_mean=float(np.mean(rads))
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
# _build_bfs_tree_cache_from_igraph (함수명 변경 및 O(V^3) Treerep 로직 제거)
# ======================================================================================
def _build_bfs_tree_cache_from_igraph( # 함수명 변경
    sg: ig.Graph,
    *,
    backend: Literal["treerep", "bfs"] = "bfs", # 기본값을 "bfs"로 변경
    ppr_scores: Optional[np.ndarray] = None,
    treerep_ppr_beta: Optional[float] = None,
) -> Optional[TreeCache]:
    """무방향 서브그래프(ig) → BFS 기반 트리 → TreeCache. (O(V^3) Treerep 로직 제거)"""
    if sg.vcount() == 0:
        return None

    # 캐시 히트
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

    # ★ Treerep 로직 전체 제거 (O(V^3) 회피)
    
    # ---- BFS 경로 (Treerep 대체를 위한 경량화된 기본 트리 구조)
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
# MetricsLogger & compute_struct_metrics (변경 없음)
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
# QueryGraphBuilder (변경 없음)
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

# ======================================================================================
# SubgraphBuilder & PPRPruner (변경 없음)
# ======================================================================================
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
    subgraph_mode: Literal["bfs","ppr"] = "ppr"
    treat_undirected_for_structure: bool = True
    # [기본] 보존층 세팅: edge-betweenness는 끔(병목 방지)
    keep_bridges: bool = True
    keep_articulation: bool = True
    keep_top_edge_betw: int = 0  # ★ 기본 0 (중요)

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

    # ---------- 보존층(브릿지/아티큘레이션/edge-betw) 병합 ----------
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

# ======================================================================================
# HyperbolicEmbedder (피처 강화 및 Treerep 로직 수정)
# ======================================================================================
class HyperbolicEmbedder:
    def __init__(self, hr_instance, embed_dim=64, curvature=1.0, device="cpu"):
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.embed_dim = embed_dim
        self.device = device
        self.hr = hr_instance
        # ★ 1) 피처 추가: 국소/전역 비계층적 피처 추가
        self.feature_extractors = {
            'logdeg': self._get_logdeg,
            'leaf': self._get_leaf,
            'clustering': self._get_clustering_coeff, # Clustering Coeff 추가
            'betweenness': self._get_betweenness_centrality, # Betweenness Centrality 추가
        }
        torch.manual_seed(1234)
        self._projW: Optional[torch.Tensor] = None

    def _get_logdeg(self, g):
        return np.log1p(np.asarray(g.degree(mode="all"), dtype=np.float32)).reshape(-1,1)
    def _get_leaf(self, g):
        deg=np.asarray(g.degree(mode="all"), dtype=np.float32)
        return (deg<=1).astype(np.float32).reshape(-1,1)
    
    # ★ 2) 추가 피처 정의: Clustering Coeff (O(V) 또는 O(E)로 효율적)
    def _get_clustering_coeff(self, g: igraph.Graph):
        try:
            cc = g.transitivity_local_undirected()
            cc = np.nan_to_num(np.asarray(cc, dtype=np.float32), nan=0.0)
            return cc.reshape(-1, 1)
        except Exception:
            return np.zeros((g.vcount(), 1), dtype=np.float32)
            
    # ★ 3) 추가 피처 정의: Betweenness Centrality (Scaling 튜닝)
    def _get_betweenness_centrality(self, g: igraph.Graph):
        try:
            bet_raw = g.betweenness(weights='weight') if 'weight' in g.es.attribute_names() else g.betweenness()
            bet_raw = np.nan_to_num(np.asarray(bet_raw, dtype=np.float32), nan=0.0)
            
            # ★ F1 정체 해결을 위한 공격적인 Log Scaling: log1p를 두 번 적용하여 분산 증폭 시도
            bet = np.log1p(bet_raw)
            bet = np.log1p(bet) # 공격적인 scaling: 변별력 극대화
            
            return bet.reshape(-1, 1)
        except Exception:
            return np.zeros((g.vcount(), 1), dtype=np.float32)

    # (수정) 함수명 변경 및 BFS 백엔드 고정
    def _get_bfs_tree(self, g: ig.Graph, ppr_scores: Optional[np.ndarray]=None, treerep_ppr_beta: Optional[float]=None):
        n=g.vcount()
        cache = _build_bfs_tree_cache_from_igraph(
            g.as_undirected() if g.is_directed() else g,
            backend="bfs", ppr_scores=ppr_scores, treerep_ppr_beta=treerep_ppr_beta
        )
        T = cache.T if cache is not None else nx.Graph()
        D = np.zeros((n, n), dtype=np.float32)
        return D, T

    def _treerep_features_from_tree(self, T: nx.Graph, g: ig.Graph)->np.ndarray:
        # Treerep 피처 추출 로직은 유지 (BFS 기반 계층성 피처로 활용)
        n=g.vcount()
        if n==0: return np.zeros((0, 12), dtype=np.float32)
        gids=np.asarray(g.vs["global_id"], dtype=int); gid2loc={int(x):i for i,x in enumerate(gids)}
        root_gid = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
        if T.number_of_nodes()==0:
            feats=np.zeros((n,12), dtype=np.float32)
            if root_gid in gid2loc: feats[gid2loc[root_gid],3]=1.0
            return feats

        # O(V log V) 또는 O(V) 수준의 피처 계산 로직 유지

        depth=np.zeros(n, dtype=np.float32)
        try:
            dist=nx.single_source_dijkstra_path_length(T, root_gid, weight='weight')
        except Exception:
            dist={}
        for gid,i in gid2loc.items(): depth[i]=float(dist.get(gid,0.0))
        maxd=float(depth.max()); depth_norm= (depth/(maxd+1e-6)).astype(np.float32) if maxd>0 else depth

        deg_arr=np.zeros(n, dtype=np.float32)
        for gid,i in gid2loc.items():
            if gid in T: deg_arr[i]=float(T.degree(gid))
        leaf=(deg_arr<=1.0).astype(np.float32)

        pw=np.zeros(n, dtype=np.float32)
        for gid,i in gid2loc.items():
            if gid in T:
                nbr=list(T.neighbors(gid))
                if nbr: pw[i]=min(T[gid][v].get('weight',1.0) for v in nbr)
        maxpw=float(pw.max()); pw_norm=(pw/(maxpw+1e-6)).astype(np.float32) if maxpw>0 else pw

        is_root=np.zeros(n, dtype=np.float32)
        if root_gid in gid2loc: is_root[gid2loc[root_gid]]=1.0

        subtree_size=np.ones(n, dtype=np.float32)
        balance=np.zeros(n, dtype=np.float32)
        try:
            Tdir=nx.bfs_tree(T, root_gid)
            post=list(reversed(list(nx.topological_sort(Tdir))))
            size_map={u:1 for u in Tdir.nodes()}
            for u in post:
                children=list(Tdir.successors(u))
                for v in children: size_map[u]+=size_map[v]
            for gid,i in gid2loc.items():
                subtree_size[i]=float(size_map.get(gid,1))
                ch=list(Tdir.successors(gid)) if gid in Tdir else []
                if len(ch)>=2:
                    arr=np.array([size_map.get(c,1) for c in ch], dtype=np.float32)
                    v=arr.var()
                    balance[i]=1.0/(1.0+v)
                elif len(ch)==1:
                    balance[i]=0.5
                else:
                    balance[i]=1.0
        except Exception:
            pass
        subtree_size_norm=(subtree_size/max(1.0, T.number_of_nodes())).astype(np.float32)

        max_deg=float(deg_arr.max()) if deg_arr.size else 0.0
        branch_norm=(deg_arr/(max_deg+1e-6)).astype(np.float32) if max_deg>0 else deg_arr

        center_flag=np.zeros(n, dtype=np.float32)
        try:
            centers=nx.center(T, usebounds=False)
            if not centers:
                centers=nx.center(T, weight='weight')
            for c in centers:
                if c in gid2loc: center_flag[gid2loc[c]]=1.0
        except Exception: pass

        art_flag=np.zeros(n, dtype=np.float32)
        try:
            for a in nx.articulation_points(T):
                if a in gid2loc: art_flag[gid2loc[a]]=1.0
        except Exception: pass

        chain_flag=((deg_arr==2.0) & (art_flag==0.0)).astype(np.float32)

        hs_order=np.zeros(n, dtype=np.float32)
        try:
            Tdir=nx.bfs_tree(T, root_gid)
            order_map={}
            post=list(reversed(list(nx.topological_sort(Tdir))))
            for u in post:
                ch=list(Tdir.successors(u))
                if not ch: order_map[u]=1
                else:
                    vals=sorted([order_map.get(v,1) for v in ch], reverse=True)
                    if len(vals)>=2 and vals[0]==vals[1]: order_map[u]=vals[0]+1
                    else: order_map[u]=vals[0]
            for gid,i in gid2loc.items(): hs_order[i]=float(order_map.get(gid,1))
            hs_order = (hs_order / (hs_order.max()+1e-6)).astype(np.float32)
        except Exception:
            pass

        feats=np.stack([
            depth_norm,        # 0
            leaf,              # 1
            pw_norm,           # 2
            is_root,           # 3
            subtree_size_norm, # 4
            branch_norm,       # 5
            center_flag,       # 6
            art_flag,          # 7
            chain_flag,        # 8
            balance,           # 9
            hs_order,          # 10
            deg_arr.reshape(-1,1).squeeze(-1)/(max_deg+1e-6) if max_deg>0 else deg_arr # 11
        ], axis=1).astype(np.float32)
        return feats

    def _project_and_stabilize(self, X: np.ndarray)->torch.Tensor:
        if X.size==0:
            return torch.zeros((1,self.embed_dim), dtype=torch.float32, device=self.device)
        Z=torch.tensor(X, dtype=torch.float32, device=self.device)
        n=torch.linalg.norm(Z, dim=1, keepdim=True)+1e-6
        Z=Z*torch.clamp(3.0/n, max=1.0)
        d_in=Z.size(1); d_out=self.embed_dim
        if self._projW is None or self._projW.size()!=torch.Size([d_in, d_out]):
            W=torch.empty(d_in, d_out, device=self.device)
            torch.nn.init.orthogonal_(W)
            self._projW=W
        Z=Z@self._projW
        return Z

    def graph_to_tensor(self, g: ig.Graph, ppr_scores: Optional[np.ndarray]=None,
                        passage_embeddings: Optional[np.ndarray]=None,
                        entity_embeddings: Optional[np.ndarray]=None,
                        use_semantic_features: bool=True,
                        use_treerep_metric: bool=False,
                        treerep_ppr_beta: Optional[float]=None)->torch.Tensor:
        if g.vcount()==0: return torch.zeros(1,self.embed_dim, device=self.device)
        parts=[]; weights=[]

        if use_treerep_metric:
            # ★ _get_treerep_tree 대신 _get_bfs_tree 사용
            _, T = self._get_bfs_tree(g, ppr_scores=ppr_scores, treerep_ppr_beta=treerep_ppr_beta)
            tfeat=self._treerep_features_from_tree(T,g)
            if tfeat.size>0:
                mu=tfeat.mean(0,keepdims=True); sd=tfeat.std(0,keepdims=True); sd=np.maximum(sd, 1e-6)
                tfeat=((tfeat-mu)/sd).astype(np.float32)
                parts.append(tfeat); weights.append(1.0)

        g2=[]
        # ★ 강화된 feature_extractors 사용: O(V) 국소 피처 포함
        for _,fn in self.feature_extractors.items():
            g2.append(fn(g))
        if g2:
            g2=np.hstack(g2)
            mu=g2.mean(0,keepdims=True); sd=g2.std(0,keepdims=True); sd=np.maximum(sd,1e-6)
            g2=((g2-mu)/sd).astype(np.float32)
            parts.append(g2); weights.append(1.0)
        
        # ... (생략: Semantic Features 로직 유지)
        if use_semantic_features and g.vcount()>0:
            sem=[]
            if "global_id" in g.vs.attributes() and "name" in g.vs.attributes():
                for key in g.vs["name"]:
                    emb=None
                    if key.startswith("chunk-") and hasattr(self.hr, "chunk_embedding_store"):
                        row=self.hr.chunk_embedding_store.get_row(key)
                        if row and 'embedding' in row: emb=row['embedding']
                    elif key.startswith("entity-") and hasattr(self.hr, "entity_embedding_store"):
                        row=self.hr.entity_embedding_store.get_row(key)
                        if row and 'embedding' in row: emb=row['embedding']
                    if emb is None and passage_embeddings is not None:
                        emb=np.zeros((passage_embeddings.shape[1],), dtype=np.float32)
                    if emb is None: emb=np.zeros((64,), dtype=np.float32)
                    sem.append(emb)
            if sem:
                sem=np.vstack(sem)
                mu=sem.mean(0,keepdims=True); sd=sem.std(0,keepdims=True); sd=np.maximum(sd,1e-6)
                sem=((sem-mu)/sd).astype(np.float32)
                parts.append(sem); weights.append(0.5)

        if not parts:
            X=np.zeros((g.vcount(), 8), dtype=np.float32)
        else:
            X = np.hstack([w*p for (p,w) in zip(parts,weights)])

        Zlin=self._project_and_stabilize(X)
        return Zlin

    def embed_lorentz(self, g: ig.Graph, ppr_scores=None, passage_embeddings=None,
                      entity_embeddings=None, use_semantic_features=True,
                      use_treerep_metric=False, treerep_ppr_beta: Optional[float]=None)->torch.Tensor:
    # ... (생략: 로직 유지)
        if g.vcount()==0: return None
        x=self.graph_to_tensor(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
                               entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                               use_treerep_metric=use_treerep_metric, treerep_ppr_beta=treerep_ppr_beta)
        with torch.no_grad():
            x= self.manifold.expmap0(x)
            x= torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            pooled, _ = torch.max(x, dim=0, keepdim=False)
            pooled = self.manifold.projx(pooled)
            if torch.linalg.norm(pooled).item()<1e-8:
                pooled = pooled + 1e-3*torch.randn_like(pooled)
                pooled = self.manifold.projx(pooled)
            return pooled

    def embed_nodes_lorentz(self, g: ig.Graph, **kw)->torch.Tensor:
        x=self.graph_to_tensor(g, **kw)
        with torch.no_grad():
            nmap=self.manifold.expmap0(x)
            return torch.nan_to_num(nmap, nan=0.0, posinf=0.0, neginf=0.0)

    def logmap0_safe(self, y: torch.Tensor)->torch.Tensor:
        t=self.manifold.logmap0(y)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def shared_standardize(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        X = torch.stack(tensors, dim=0)  # [B, D]
        mu = X.mean(dim=0, keepdim=True)
        sd = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        Y = [(t - mu.squeeze(0)) / sd.squeeze(0) for t in tensors]
        return Y

    def embed_and_logmap0(self, g: ig.Graph, **kw) -> torch.Tensor:
        L = self.embed_lorentz(g, **kw)
        if L is None:  
            return torch.zeros(self.embed_dim, device=self.device)
        return self.logmap0_safe(L.unsqueeze(0)).squeeze(0)

    def attention_pool_tangent(self, tq: torch.Tensor, td_nodes: torch.Tensor, lam: float=6.0, topk: Optional[int]=12, residual: float=0.2) -> torch.Tensor:
        # cosine 기반 score → sigmoid → top-k → 가중합
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
# StructralSimilarity (변경 없음)
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

# ======================================================================================
# LTRConfig (변경 없음)
# ======================================================================================
@dataclass
class LTRConfig:
    enabled: bool = False
    weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.5, 0.5, 0.5])
    bias: float = 0.0
    blend: float = 0.5

# ======================================================================================
# HyperReranker (O(V^2) 모드 비활성화 및 함수명 변경 반영)
# ======================================================================================
class HyperReranker:
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

    # --- 구조 유사도 계산 함수들 ---
    def _struct_scores_tangent(
        self, query_graph: ig.Graph, subgraphs: List[ig.Graph], embedder: "HyperbolicEmbedder", scorer: "StructralSimilarity", 
        ppr_scores: Optional[np.ndarray] = None, passage_embeddings: Optional[np.ndarray] = None, entity_embeddings: Optional[np.ndarray] = None, 
        use_semantic_features: bool = True, use_treerep_metric: bool = False,
        attn_pooling: bool = True, attn_lambda: float = 6.0, attn_topk: Optional[int] = 12, attn_residual: float = 0.2,
    ) -> np.ndarray:
        # Tangent 모드는 O(V) 피처만 사용하므로 유지
        with torch.no_grad():
            tq = embedder.embed_and_logmap0(
                query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
            )
            tdocs = []
            for g in subgraphs:
                d_nodes_L = embedder.embed_nodes_lorentz(
                    g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                    entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
                )
                if d_nodes_L.numel() == 0:
                    tdocs.append(torch.zeros_like(tq)); continue
                td_nodes = embedder.logmap0_safe(d_nodes_L)
                if attn_pooling:
                    pooled = embedder.attention_pool_tangent(
                        tq, td_nodes, lam=attn_lambda, topk=attn_topk, residual=attn_residual
                    )
                else:
                    pooled, _ = torch.max(td_nodes, dim=0)
                tdocs.append(pooled)
            # ★ 오타 수정 완료
            tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
            q_list = [tq_std for _ in range(len(tdocs_std))]
            struct_head = scorer.compute_listwise(q_list, tdocs_std)
            return _to_1d_numpy(struct_head)

    def _struct_scores_lorentz(
        self, query_graph: ig.Graph, subgraphs: List[ig.Graph], embedder: "HyperbolicEmbedder", temp: float = 2.0, 
        ppr_scores: Optional[np.ndarray] = None, passage_embeddings: Optional[np.ndarray] = None, entity_embeddings: Optional[np.ndarray] = None, 
        use_semantic_features: bool = True, use_treerep_metric: bool = False
    ) -> np.ndarray:
        # Lorentz 모드는 O(V) 피처만 사용하므로 유지
        with torch.no_grad():
            qL = embedder.embed_lorentz(
                query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
            )
            dL = [
                embedder.embed_lorentz(
                    g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                    entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
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
        self,
        query_graph: igraph.Graph,
        subgraphs: List[ig.Graph],
        query_radial_anchor: Optional[float] = None,
        radial_tau: float = 0.5,
        ppr_scores: Optional[np.ndarray] = None,
        treerep_backend: Literal["treerep","bfs"] = "bfs",
        treerep_ppr_beta: Optional[float] = None,
    ) -> np.ndarray:
        # O(V^2) 연산 포함. RRF 점수로는 0 처리
        logger.warning("O(V^2) cost in _struct_scores_struct_feat: Disabling in favor of Tangent/LTR.")
        return np.zeros(len(subgraphs), dtype=float)

    def _pick_landmarks_diverse(self, cache: TreeCache, L:int=10)->List[int]:
        # ... (생략)
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
        # O(V^2) 연산 포함. RRF 점수로는 0 처리
        logger.warning("O(V^2) cost in _struct_scores_gromov: Disabling in favor of Tangent/LTR.")
        return np.zeros(len(subgraphs), dtype=float)

    def _compute_gate(self, sem_head: np.ndarray, struct_head: np.ndarray,
                      gating_enabled: bool, gate_floor: float, gate_ceiling: float)->Tuple[float, np.ndarray]:
        rho=0.0
        def is_const(x): return x.size==0 or np.allclose(x, x[0])
        if gating_enabled and not (is_const(sem_head) or is_const(struct_head)):
            try:
                r = float(spearmanr(sem_head, struct_head).correlation)
                if np.isnan(r): r=0.0
                rho=r
            except Exception: rho=0.0
        med=0.0; mad=1.0
        try:
            arr=np.array([rho], dtype=float)
            med=np.median(arr); mad=np.median(np.abs(arr-med))+1e-6
            z=(rho-med)/mad
        except Exception:
            z=0.0
        gate = 0.6 + 0.1*np.tanh(z)
        gate=float(np.clip(gate, gate_floor, gate_ceiling))
        return gate, np.array([rho], dtype=float)

    # ----- 메인 결합 -----
    def rerank_with_structure(self, query_graph: Optional[ig.Graph],
                              passage_topk_ids: List[int], passage_topk_scores: List[float],
                              passage_node_idxs: List[int], last_ppr_scores: Optional[np.ndarray],
                              cfg: SubgraphConfig, embedder: HyperbolicEmbedder, scorer: StructralSimilarity,
                              passage_embeddings: np.ndarray, entity_embeddings: np.ndarray,
                              use_semantic_features: bool=True, use_treerep_metric: bool=False,
                              gating_enabled: bool=True, ensemble_ratio: float=0.6,
                              radial_tau: float=0.5, landmark_k: int=10, alpha: float=0.8,
                              struct_k: int=30, device: str="cpu", save_cb: Optional[Callable]=None,
                              save_ctx: Optional[dict]=None, sim_cac_mode: Literal["tangent","lorentz","node_level","gromov","struct_feat","ensemble"]="ensemble",
                              lorentz_temp: float=2.0, treerep_backend: Literal["treerep","bfs"]="bfs",
                              treerep_ppr_beta: Optional[float]=None, pin_k:int=0, k_rrf:int=20,
                              attn_pooling: bool=True, attn_lambda: float=6.0, attn_topk: int=12, attn_residual: float=0.2,
                              gate_floor: float=0.60, gate_ceiling: float=0.95,
                              ltr_config: Optional[Any]=None
                              )->Tuple[List[int], List[float]]:

        passage_topk_ids=list(np.asarray(passage_topk_ids).tolist())
        passage_topk_scores=list(np.asarray(passage_topk_scores, dtype=float).tolist())
        if query_graph is None or len(passage_topk_ids)==0:
            return passage_topk_ids, passage_topk_scores

        subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
            passage_topk_ids, passage_topk_scores, passage_node_idxs, last_ppr_scores, cfg, head_k=struct_k
        )
        if not subgraphs: return passage_topk_ids, passage_topk_scores

        # 구조 점수 (O(V^2) 모드 제거)
        if sim_cac_mode in ["tangent", "node_level"]:
            s = self._struct_scores_tangent(
                query_graph, subgraphs, embedder, scorer, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric,
                attn_pooling=(sim_cac_mode!="node_level")
            )
        elif sim_cac_mode=="lorentz":
            s = self._struct_scores_lorentz(
                query_graph, subgraphs, embedder, temp=lorentz_temp, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
            )
        elif sim_cac_mode in ["gromov", "struct_feat", "ensemble"]:
            # O(V^2) 모드는 RRF 점수로는 0 처리하여 Baseline에 영향을 미치지 않도록 함 (안정성 확보)
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
        gate, _rho = self._compute_gate(sem_head, s_norm, gating_enabled, gate_floor, gate_ceiling)

        # ★ LTR 모드일 경우 LTR 로직 추가 (현재는 RRF와 게이팅 로직만 유지)
        if ltr_config and ltr_config.enabled:
            # LTR Logic: 최종 목표 (학습이 필요)
            final_head = alpha*rrf_sem + (1.0-alpha)*rrf_struct 
        else:
             final_head = alpha*rrf_sem + (1.0-alpha)*gate*rrf_struct


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
        return merged_ids, merged_scores