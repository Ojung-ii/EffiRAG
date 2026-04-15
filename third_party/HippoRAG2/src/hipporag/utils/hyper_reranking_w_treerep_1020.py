from __future__ import annotations 
import igraph as ig
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Set, Callable, Literal, Dict, Any
from collections import deque, defaultdict
import logging
import os
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from typing import Literal

# TreeRep
from .TreeRep import TreeRep
import networkx as nx
import geoopt

logger = logging.getLogger(__name__)

import json, math
from typing import Mapping

# =====================[ NEW: TreeCache & StructFeatureExtractor ]=====================
import math
import numpy as np
import networkx as nx
from typing import List, Dict, Optional

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
        # tree에서 두 번의 BFS로 정확히 구함(가중)
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
        a,_=far(self.root); b,diam=far(a)
        return float(diam)

    def lca(self, u, v):
        if u==v: return u
        ui=self.index_of[u]; vi=self.index_of[v]
        # 깊은 쪽 올리기
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
        # NEW: 트리에 없는 노드는 바깥(1.0)로 간주해 보수적으로 페널티 처리
        if v not in self.dist_to_root:
            return 1.0
        if self.max_dist <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, self.dist_to_root[v] / self.max_dist))

class StructFeatureExtractor:
    """QA-지향 구조피처(그룹 단위)."""
    def __init__(self, cache: TreeCache):
        self.C = cache

    def radial_gap(self, node, rq: float, tau: float=0.5) -> float:
        gap = abs(self.C.radial(node)-rq)
        return max(0.0, 1.0 - (gap/(tau+1e-9)))

    def geodesic_coherence(self, nodes: List[int], max_pairs: int=200) -> float:
        # 평균 쌍거리 ↓ → 응집 ↑  (샘플링으로 O(1) 근사)
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
        # 트리에서 모든 쌍 경로의 union edge 수를 근사 스테이너 크기로 사용
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
        # NEW: anchor가 없으면 루트로 대체
        if anchor not in self.C.T:
            anchor = self.C.root
        ar = self.C.dist_to_root.get(anchor, 0.0)  # NEW: 기본값 가드
        for v in nodes:
            # dist/lca는 TreeCache 안에서 처리되므로 그대로 호출
            gp = (ar + self.C.dist_to_root.get(v, ar) - self.C.dist(anchor, v)) * 0.5
            vals.append(max(0.0, gp / z))
        return {"gp_mean":float(np.mean(vals)), "gp_min":min(vals), "gp_max":max(vals)}

    def branch_focus(self, nodes: List[int]) -> float:
        # 자식 수 분포 엔트로피↓ → 집중↑
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
        # NEW: 트리에 존재하는 노드만 사용
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
# =====================[ /NEW ]=====================
# =====================[ REPLACE THIS WHOLE FUNCTION ]=====================
from typing import Literal

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

    # root gid 결정
    if "root_id" in gu.attributes():
        root_loc = int(gu["root_id"])
        root_gid = idx2gid.get(root_loc, gids[0])
    else:
        root_gid = gids[0]

    # --- (A) TreeRep 백엔드: 간선가중→거리 + PPR penalty → 거리행렬 → TreeRep ---
    if backend == "treerep" and gu.ecount() > 0:
        try:
            eps = 1e-6
            # 간선 거리
            if has_w:
                w_raw = np.asarray(gu.es["weight"], dtype=float)
            else:
                w_raw = np.ones(gu.ecount(), dtype=float)
            w_dist = 1.0 / (eps + np.maximum(w_raw, 0.0))

            # PPR penalty (옵션)
            if ppr_scores is not None and "global_id" in gu.vs.attributes():
                pv = np.nan_to_num(np.asarray(ppr_scores, float), nan=0.0)
                pv = np.clip(pv, 0.0, None)
                with np.errstate(divide="ignore"):
                    node_pen = -np.log(pv + eps)  # 낮은 ppr일수록 cost↑
                beta = float(os.getenv("TREEREP_PPR_BETA", "0.25"))
                add_cost = []
                for e in gu.es:
                    u, v = e.source, e.target
                    add_cost.append(beta * 0.5 * (node_pen[gids[u]] + node_pen[gids[v]]))
                w_dist = w_dist + np.asarray(add_cost, float)

            # 최단거리 행렬
            D = np.array(gu.shortest_paths(weights=w_dist), dtype=np.float32)
            if np.isinf(D).any():
                finite = D[np.isfinite(D)]
                cap = float(np.max(finite)) if finite.size else 0.0
                D[~np.isfinite(D)] = cap + 1.0

            # TreeRep로 트리 학습
            treerep = TreeRep(d=D)
            treerep.learn_tree()
            Tloc = treerep.G  # 노드: 0..n-1

            # 로컬 idx → global_id 리레이블
            mapping = {loc: int(gids[loc]) for loc in range(min(len(gids), Tloc.number_of_nodes()))}
            T = nx.relabel_nodes(Tloc, mapping, copy=True)
            T.graph["root_id"] = int(root_gid)
            return TreeCache(T, root=root_gid, max_k=16)
        except Exception as e:
            logger.warning(f"[treerep-backend] fallback to BFS due to: {e}")

    # --- (B) Fallback: BFS 트리 ---
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


        # [v4] --- add struct normalization helper ---
def _normalize_struct_v4(x: np.ndarray, mode: str = "robust"):
    """
    mode ∈ {"robust","zscore","softmax"}
    - robust: 기존 robust_normalize_struct
    - zscore: Z-score 후 [0,1]으로 클립
    - softmax: 온도 1.0 소프트맥스
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
        # z-score를 시그모이드로 안정스케일 → [0,1]
        return 1.0 / (1.0 + np.exp(-z))

    if mode == "softmax":
        # numerically stable softmax
        mx = np.max(x)
        ex = np.exp(x - mx)
        sm = ex / (np.sum(ex) + 1e-9)
        return sm  # 이미 [0,1] 합=1

    # fallback
    return StructralSimilarity().robust_normalize_struct(x)
# =====================[ /REPLACE ]=====================



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

    def build(self, triples: list[tuple], query_id: str) -> ig.Graph:
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
    subgraph_mode: Literal["bfs", "ppr"] = "ppr"
    # [A] 구조용 후보/거리/TreeRep는 무방향으로 처리해 일관성 유지
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
        # [A] 구조 후보 수집은 무방향으로: 항상 ALL 사용
        mode = "ALL"
        K = max(1, int(self.cfg.per_hop_topk)) if self.cfg.per_hop_topk else None

        while q:
            u, depth = q.popleft()
            if depth >= self.cfg.max_depth:
                continue

            neighs = []
            for eid in self.g.incident(u, mode=mode):
                e = self.g.es[eid]
                # [A] 무방향 처리: source/target 구분 없이 반대편 노드 선택
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
        
        # root 보장
        if (0 <= root < self.g.vcount()) and (root not in selected_vids_set):
            selected_vids_set.add(root)

        edges: List[Tuple[int,int,float]] = []
        visited_edges: Set[Tuple[int, int]] = set()

        for u in selected_vids_set:
            # [A] 구조용은 무방향 이웃으로 연결
            for v in self.g.neighbors(u, mode="ALL"):
                if v in selected_vids_set:
                    edge_tuple = tuple(sorted((u, v)))
                    if edge_tuple not in visited_edges:
                        visited_edges.add(edge_tuple)
                        # [A] 항상 directed=False로 간선 조회
                        eids = self.g.get_eids([(u, v)], directed=False)
                        if eids:
                            w = float(self.g.es[eids[0]].attributes().get('weight', 1.0))
                            edges.append((u, v, w))
        return edges

    def collect_candidates(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
        if self.cfg.subgraph_mode == "bfs":
            return self._collect_candidates_by_bfs(root, allow_ids)
        elif self.cfg.subgraph_mode == "ppr":
            return self._collect_candidates_by_ppr(root, allow_ids)
        else:
            raise ValueError(f"Unknown subgraph_mode: {self.cfg.subgraph_mode}")

    def build_subgraph(self, edges: List[Tuple[int,int,float]], root: Optional[int] = None) -> ig.Graph:
        # [A] 구조 서브그래프는 무방향으로 생성해 단순화(원그래프가 유향이어도 구조 특성은 무방향)
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
        if self.weighted:
            if self.g.es[eid].source != u:
                return 0.0
            w = float(self.g.es[eid]['weight'])
            denom = self.outstr[u]
            return (w/denom) if denom > 0 else 0.0
        d = self.outdeg[u]
        return (1.0/d) if d > 0 else 0.0

    def _flow(self, u: int, v: int) -> float:
        if self.pi is None:
            return 0.0
        eid = self.g.get_eid(u, v, directed=True, error=False)
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


class HyperbolicEmbedder:
    def __init__(self, hr_instance, embed_dim=32, curvature=1.0, device="cpu"):
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.embed_dim = embed_dim
        self.device = device
        self.hr = hr_instance
        self.feature_extractors = {
            'logdeg': self._get_logdeg,
            # 'depth': self._get_depth,
            'leaf': self._get_leaf,
            # 'eig_centrality': self._get_eig_centrality,
            # 'clustering': self._get_clustering,
            # 'pr_undirected': self._get_pr_undirected,
            # 'katz': self._get_katz,  # 필요시 활성화
        }
    
    def _get_logdeg(self, g):
        return np.log1p(np.asarray(g.degree(mode="all"), dtype=np.float32)).reshape(-1, 1)

    def _get_depth(self, g):
        root_local = 0
        try:
            gids = np.asarray(g.vs["global_id"])
            rid = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
            loc = np.where(gids == rid)[0]
            if len(loc): root_local = int(loc[0])
        except Exception:
            pass
        g_ud = g.as_undirected() if g.is_directed() else g
        d = np.asarray(g_ud.shortest_paths(source=root_local)[0], dtype=np.float32)
        finite = np.isfinite(d); maxd = float(np.max(d[finite])) if finite.any() else 0.0
        d[~finite] = maxd + 1.0
        return (d / (maxd + 1e-6)).reshape(-1, 1)
        
    def _get_leaf(self, g):
        deg = np.asarray(g.degree(mode="all"), dtype=np.float32)
        return (deg <= 1).astype(np.float32).reshape(-1, 1)

    def _get_eig_centrality(self, g):
        try:
            eig = np.array(g.eigenvector_centrality()).astype(np.float32)
            return (eig / (eig.max() + 1e-6)).reshape(-1, 1)
        except Exception:
            return np.zeros((g.vcount(), 1), dtype=np.float32)

    def _get_clustering(self, g):
        try:
            clu = np.array(g.transitivity_local_undirected()).astype(np.float32)
            return np.nan_to_num(clu, nan=0.0).reshape(-1, 1)
        except Exception:
            return np.zeros((g.vcount(), 1), dtype=np.float32)
    def _get_pr_undirected(self, g: ig.Graph):
        if g.vcount() == 0:
            return np.zeros((0,1), dtype=np.float32)
        gu = g.as_undirected() if g.is_directed() else g
        try:
            pr = np.array(gu.pagerank(weights='weight' if 'weight' in gu.es.attributes() else None), dtype=np.float32)
        except Exception:
            pr = np.ones(gu.vcount(), dtype=np.float32) / max(1, gu.vcount())
        pr = pr / (pr.max() + 1e-6)
        return pr.reshape(-1,1)

    def _get_katz(self, g: ig.Graph, alpha: float = 0.01, beta: float = 1.0):
        # igraph Katz가 없다면 networkx 변환 또는 간단 행렬식 근사 사용
        try:
            import networkx as nx
            G = nx.DiGraph() if g.is_directed() else nx.Graph()
            G.add_nodes_from(range(g.vcount()))
            for e in g.es:
                G.add_edge(e.source, e.target, weight=float(e['weight']) if 'weight' in g.es.attributes() else 1.0)
            kc = nx.katz_centrality_numpy(G, alpha=alpha, beta=beta, weight='weight')
            arr = np.array([kc[i] for i in range(g.vcount())], dtype=np.float32)
            arr = arr / (arr.max() + 1e-6)
            return arr.reshape(-1,1)
        except Exception:
            return np.zeros((g.vcount(),1), dtype=np.float32)
    # ---------------------------------------------------------------------
    # [A] + [B]: TreeRep 입력 거리/후처리 일관 무방향 + global_id로 리레이블
    # ---------------------------------------------------------------------
    def _get_treerep_tree(self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, nx.Graph]:
        """TreeRep 학습용 거리행렬(무방향)과 learned_tree(nx.Graph, 노드=global_id)를 만든다."""
        n = g.vcount()
        if n < 2:
            return np.zeros((n, n), dtype=np.float32), nx.Graph()

        # 무방향 그래프
        gu: ig.Graph = g.as_undirected() if g.is_directed() else g

        # (A) 간선 가중치 → 거리 변환: 1/(eps + w). weight 없으면 w=1.0 가정
        eps = 1e-6
        if 'weight' in gu.es.attribute_names():
            w_raw = np.asarray(gu.es['weight'], dtype=float)
        else:
            w_raw = np.ones(gu.ecount(), dtype=float)
        w_dist = 1.0 / (eps + np.maximum(w_raw, 0.0))

        # (B) PPR penalty(옵션): β * ( -log ppr[u] - log ppr[v] ) / 2
        if ppr_scores is not None and "global_id" in gu.vs.attributes():
            gids = np.asarray(gu.vs["global_id"], dtype=int)
            pv = np.nan_to_num(np.asarray(ppr_scores, dtype=float), nan=0.0)
            pv = np.clip(pv, 0.0, None)
            with np.errstate(divide='ignore'):
                pen_nodes = -np.log(pv + eps)
            beta = 0.25  # 제안값
            add_cost = []
            for e in gu.es:
                u, v = e.source, e.target
                add_cost.append(beta * 0.5 * (pen_nodes[gids[u]] + pen_nodes[gids[v]]))
            w_dist = w_dist + np.asarray(add_cost, dtype=float)

        # (C) 최단거리 행렬
        try:
            distances = np.array(gu.shortest_paths(weights=w_dist), dtype=np.float32)
        except Exception:
            # fallback: 무가중 최단거리
            distances = np.array(gu.shortest_paths(), dtype=np.float32)

        # 무한대 보정
        if np.isinf(distances).any():
            finite = distances[np.isfinite(distances)]
            cap = float(np.max(finite)) if finite.size else 0.0
            distances[~np.isfinite(distances)] = cap + 1.0

        # TreeRep 학습
        try:
            treerep_instance = TreeRep(d=distances)
            treerep_instance.learn_tree()
            learned_tree = treerep_instance.G  # 노드: 0..n-1
        except Exception as e:
            logger.warning(f"TreeRep learning failed: {e}. Returning empty tree.")
            learned_tree = nx.Graph()

        # (D) 노드 라벨을 global_id로 리레이블
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

        # root(gid)
        try:
            root_gid = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
            if root_gid not in gid2loc:
                root_gid = int(gids[0])
        except Exception:
            root_gid = int(gids[0])

        # ----- 공통: learned_tree가 비었을 때 제로벡터 -----
        if learned_tree.number_of_nodes() == 0:
            feats = np.zeros((n, 7), dtype=np.float32)
            # is_root만 세움
            if root_gid in gid2loc:
                feats[gid2loc[root_gid], 3] = 1.0  # is_root 위치(아래에서 3번째 컬럼)
            return feats

        # 1) depth_norm (가중 최단거리)
        depth = np.zeros(n, dtype=np.float32)
        try:
            dist = dict(nx.single_source_dijkstra_path_length(learned_tree, root_gid, weight='weight'))
        except Exception:
            dist = {}
        for gid, i in gid2loc.items():
            depth[i] = float(dist.get(gid, 0.0))
        max_depth = float(depth.max())
        depth_norm = (depth / (max_depth + 1e-6)).astype(np.float32) if max_depth > 0 else depth

        # 2) leaf (차수<=1)
        deg_arr = np.zeros(n, dtype=np.float32)
        for gid, i in gid2loc.items():
            if gid in learned_tree:
                deg_arr[i] = float(learned_tree.degree(gid))
        leaf = (deg_arr <= 1.0).astype(np.float32)

        # 3) parent_w_norm (인접 간선 최소가중치)
        parent_w = np.zeros(n, dtype=np.float32)
        for gid, i in gid2loc.items():
            if gid in learned_tree:
                nbrs = list(learned_tree.neighbors(gid))
                if nbrs:
                    parent_w[i] = min(learned_tree[gid][v].get('weight', 1.0) for v in nbrs)
        max_pw = float(parent_w.max())
        parent_w_norm = (parent_w / (max_pw + 1e-6)).astype(np.float32) if max_pw > 0 else parent_w

        # 4) is_root (one-hot)
        is_root = np.zeros(n, dtype=np.float32)
        if root_gid in gid2loc:
            is_root[gid2loc[root_gid]] = 1.0

        # 5) subtree_size_norm (루팅된 서브트리 크기 / N)
        subtree_size = np.zeros(n, dtype=np.float32)
        # learned_tree는 (가중)트리. 루트 기준으로 부모-자식 방향성 정해 DFS.
        # parent 맵 구성
        try:
            bfs_order = list(nx.bfs_tree(learned_tree, root_gid))
            parent = {}
            for u in bfs_order:
                for v in learned_tree.neighbors(u):
                    # BFS트리의 간선만 부모-자식으로 채택
                    # (nx.bfs_tree가 이미 방향 트리를 준다)
                    pass
            # bfs_tree로 한 번 더 깔끔하게
            Tdir = nx.bfs_tree(learned_tree, root_gid)
            # bottom-up로 서브트리 크기 계산
            post = list(reversed(list(nx.topological_sort(Tdir))))
            size_map = {u: 1 for u in Tdir.nodes()}
            for u in post:
                for v in Tdir.successors(u):
                    size_map[u] += size_map[v]
            for gid, i in gid2loc.items():
                subtree_size[i] = float(size_map.get(gid, 1))
        except Exception:
            # fallback: degree 기반 근사(노이즈 최소)
            for gid, i in gid2loc.items():
                subtree_size[i] = 1.0
        subtree_size_norm = (subtree_size / max(1.0, learned_tree.number_of_nodes())).astype(np.float32)

        # 6) branching_factor_norm (deg / max_deg)
        max_deg = float(deg_arr.max()) if deg_arr.size else 0.0
        branching_factor_norm = (deg_arr / (max_deg + 1e-6)).astype(np.float32) if max_deg > 0 else deg_arr

        # 7) centroid_flag (tree center)
        centroid_flag = np.zeros(n, dtype=np.float32)
        try:
            centers = nx.center(learned_tree, e=None, usebounds=False)
            # 가중 그래프면 weight='weight'로 재시도
            if len(centers) == 0:
                centers = nx.center(learned_tree, weight='weight')
            for c in centers:
                if c in gid2loc:
                    centroid_flag[gid2loc[c]] = 1.0
        except Exception:
            pass

        # [소형 트리 가드] v<5면 depth/parent_w 변동폭이 과도하게 작을 수 있어 안정적으로만 사용
        if learned_tree.number_of_nodes() < 5:
            depth_norm = depth_norm  # 그대로 두되, 이미 표준화/클리핑 단계에서 안정화됨
            parent_w_norm = parent_w_norm

        feats = np.stack([
            depth_norm,                 # 0
            leaf,                       # 1
            # parent_w_norm,              # 2
            is_root,                    # 3
            # subtree_size_norm,          # 4 (신규)
            # branching_factor_norm,      # 5 (신규)/
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
            # [A]+[B] 적용된 TreeRep 트리/피처
            # distances, learned_tree = self._get_treerep_tree(g) 
            distances, learned_tree = self._get_treerep_tree(g, ppr_scores=ppr_scores)
            feats_treerep = self._treerep_features_from_tree(learned_tree, g) 
            feature_list.append(feats_treerep)
            
            # 일반 구조 피처(원본 그래프 기반)
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

        # 표준화(안전가드 포함)
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
        
        x = self.graph_to_tensor(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
                                 entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
                                 use_treerep_metric=use_treerep_metric)
        
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
        y = self.embed_lorentz(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric)
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
        self, g: igraph.Graph, ppr_scores: Optional[np.ndarray] = None,
        passage_embeddings: Optional[np.ndarray] = None,
        entity_embeddings: Optional[np.ndarray] = None,
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False
    ) -> torch.Tensor:
        if g.vcount() == 0:
            return torch.empty(0, self.embed_dim, device=self.device)
        x = self.graph_to_tensor(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric)
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
        # 퍼센타일 기반 윈저라이즈 → min-max
        if x.size == 0: return x
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(x, 10), np.percentile(x, 90)
        if hi - lo < 1e-8:
            return np.full_like(x, 0.5)
        x = np.clip(x, lo, hi)
        y = (x - lo) / (hi - lo + 1e-9)
        return y


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
            # [A] 후보는 무방향으로 모았지만, PPR 기반 flow-prune은 기존대로(유향 전이) 적용
            pruned = pruner.prune(root=rid, candidate_edges=cands) if cfg.subgraph_mode == "ppr" else cands
            
            sg = builder.build_subgraph(pruned, root=rid)
            try:
                row = compute_struct_metrics(
                    sg,
                    graph_role="subgraph",
                    dataset_name=getattr(self, "dataset_name", "default") if hasattr(self, "dataset_name") else "default",
                    query_id=str(getattr(self, "query_id", "unknown")) if hasattr(self, "query_id") else "unknown",
                    root_global_id=rid,
                    ppr_scores=last_ppr_scores ,
                    exclude_virtual= False
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
    
    def _struct_scores_tangent(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        embedder: "HyperbolicEmbedder",
        scorer: "StructralSimilarity",
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
            if not tdocs:
                return np.array([])
            T = torch.stack(tdocs)
            C = F.cosine_similarity(T.unsqueeze(1), T.unsqueeze(0), dim=-1)
            mask = ~torch.eye(C.size(0), dtype=bool, device=C.device)
            vals = C[mask]
            logger.debug(
                "[EMB-DIAG][raw] tdocs_norm_mean=%.4f tdocs_std=%.4f | pairwise_cos_mean=%.4f p95=%.4f",
                T.norm(dim=1).mean().item(),
                T.std().item(),
                vals.mean().item(),
                vals.kthvalue(max(1, int(0.95 * len(vals))))[0].item()
            )
            cos_raw = scorer.compute_listwise([tq for _ in tdocs], tdocs)
            q05, q50, q95 = torch.quantile(cos_raw, torch.tensor([0.05, 0.5, 0.95], device=cos_raw.device))
            logger.debug(
                "[EMB-DIAG][raw] qvdoc_cos min=%.4f max=%.4f | p05=%.4f p50=%.4f p95=%.4f",
                cos_raw.min().item(), cos_raw.max().item(), q05.item(), q50.item(), q95.item()
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
        embedder: "HyperbolicEmbedder",
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
        embedder: "HyperbolicEmbedder",
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
        
    # =====================[ FIXED: Gromov 형상 유사도 ]=====================
    def _struct_scores_gromov(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        landmark_k: int = 10,
        ppr_scores: Optional[np.ndarray] = None,
        landmark_weighting: str = "none",   # [v4] "none"|"degree"|"betweenness"
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

        # --- query cache ---
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

            # [v4] landmark 쌍별 가중 평균 |QL-DL|
            # weight_ij = qw[i]*dw[j] (outer product)
            W = np.outer(qw[:L], dw)
            W = W / (np.sum(W) + 1e-9)
            score = 1.0 - float(np.sum(W * np.abs(QL - DL)))
            sims.append(max(0.0, min(1.0, score)))
        return np.asarray(sims, dtype=float)


    # =====================[ FIXED: QA-지향 구조점수 ]=====================
    def _struct_scores_struct_feat(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        query_radial_anchor: Optional[float] = None,
        radial_tau: float = 0.5,
        ppr_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        # 쿼리 rq 추정(중앙값 반지름)
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
            # --- doc: TreeRep  ←←← 중요: backend & ppr_scores 넘기기
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
    # =====================[ /FIXED ]=====================


    # def rerank_with_structure(
    #     self,
    #     query_graph: Optional[ig.Graph],
    #     passage_topk_ids: List[int],
    #     passage_topk_scores: List[float],
    #     passage_node_idxs: List[int],
    #     last_ppr_scores: Optional[np.ndarray],
    #     cfg: SubgraphConfig,
    #     embedder: HyperbolicEmbedder,
    #     scorer: StructralSimilarity,
    #     passage_embeddings: np.ndarray,
    #     entity_embeddings: np.ndarray,
    #     use_semantic_features: bool = True,
    #     use_treerep_metric: bool = False,
    #     gating_enabled: bool = True,
    #     ensemble_ratio: float = 0.4,
    #     radial_tau: float = 0.5,
    #     landmark_k: int = 10,
    #     alpha: float = 0.8,
    #     struct_k: int = 30,
    #     k_rrf: int = 20,           # ← NEW: 하드코딩 제거
    #     pin_k: int = 1,   
    #     device: str = "cpu",
    #     save_cb: Optional[Callable[[ig.Graph, str, dict], None]] = None,
    #     save_ctx: Optional[dict] = None,
    #     sim_cac_mode: Literal["tangent", "lorentz", "node_level"] = "tangent",
    #     lorentz_temp: float = 2.0,
    #     # [v4] NEW:
    #     struct_norm: Literal["robust","zscore","softmax"] = "robust",
    #     sem_struct_mode: Literal["linear","gated_add"] = "linear",
    #     struct_nonlin: Literal["none","tanh"] = "none",
    #     landmark_weighting: Literal["none","degree","betweenness"] = "none",
    #     node_ratio: float = 0.20,   # node_level 분량(0~1). 0이면 기존과 동일 동작
    #     alpha_min: Optional[float] = None,
    #     alpha_max: Optional[float] = None,
    #     rho_low: float = -0.20,
    #     rho_high: float = 0.30,
    # ) -> Tuple[List[int], List[float]]:


    #     passage_topk_ids = list(np.asarray(passage_topk_ids).tolist())
    #     passage_topk_scores = list(np.asarray(passage_topk_scores, dtype=float).tolist())

    #     if query_graph is None or len(passage_topk_ids) == 0:
    #         return passage_topk_ids, passage_topk_scores

    #     subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
    #         passage_topk_ids, passage_topk_scores, passage_node_idxs,
    #         last_ppr_scores, cfg, head_k=struct_k
    #     )
        
    #     if not subgraphs:
    #         return passage_topk_ids, passage_topk_scores

    #     sem = np.asarray(passage_topk_scores, float)
    #     head_idx = np.argsort(-sem)[:len(subgraphs)]
    #     sem_head = sem[head_idx]
        
    #     if sim_cac_mode == "tangent":
    #         struct_head = self._struct_scores_tangent(
    #             query_graph, subgraphs, embedder, scorer, ppr_scores=last_ppr_scores,
    #             passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
    #             use_semantic_features=use_semantic_features,
    #             use_treerep_metric=use_treerep_metric
    #         )
            
    #     elif sim_cac_mode == "lorentz":
    #         struct_head = self._struct_scores_lorentz(
    #             query_graph, subgraphs, embedder, temp=lorentz_temp, ppr_scores=last_ppr_scores,
    #             passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
    #             use_semantic_features=use_semantic_features,
    #             use_treerep_metric=use_treerep_metric
    #         )
    #     elif sim_cac_mode == "node_level":
    #         struct_head = self._struct_scores_node_level(
    #             query_graph, subgraphs, embedder, ppr_scores=last_ppr_scores,
    #             passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
    #             use_semantic_features=use_semantic_features,
    #             use_treerep_metric=use_treerep_metric
    #         )
    #     elif sim_cac_mode == "gromov":
    #         struct_head = self._struct_scores_gromov(
    #             query_graph, subgraphs, landmark_k=landmark_k,
    #             ppr_scores=last_ppr_scores,
    #             landmark_weighting=landmark_weighting,  # [v4]
    #         )

    #     elif sim_cac_mode == "struct_feat":
    #         struct_head = self._struct_scores_struct_feat(
    #             query_graph, subgraphs, query_radial_anchor=None, radial_tau=radial_tau, ppr_scores=last_ppr_scores
    #         )
    #     # elif sim_cac_mode == "ensemble":
    #     #     # s1: struct_feat (radial)
    #     #     s1 = self._struct_scores_struct_feat(
    #     #         query_graph, subgraphs, None, radial_tau, ppr_scores=last_ppr_scores
    #     #     )
    #     #     # s2: gromov (landmark) — landmark_k 동적화(선택)
    #     #     sizes = [max(1, sg.vcount()) for sg in subgraphs]
    #     #     med = float(np.median(sizes))
    #     #     L_eff = int(np.clip(np.sqrt(med) + 6, 8, 20))
    #     #     s2 = self._struct_scores_gromov(
    #     #         query_graph, subgraphs, landmark_k=L_eff,  # ← 동적 L
    #     #         ppr_scores=last_ppr_scores,
    #     #         landmark_weighting=landmark_weighting,
    #     #     )
    #     #     # s3: node-level
    #     #     s3 = self._struct_scores_node_level(
    #     #         query_graph, subgraphs, embedder, ppr_scores=last_ppr_scores,
    #     #         passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
    #     #         use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric
    #     #     )

    #     #     # ---------- pass0: provisional mix → s0 ----------
    #     #     node_ratio_clamped = float(np.clip(node_ratio, 0.0, 1.0))
    #     #     struct_pie0 = 1.0 - node_ratio_clamped
    #     #     w1_0 = struct_pie0 * float(ensemble_ratio)
    #     #     w2_0 = struct_pie0 * float(1.0 - ensemble_ratio)
    #     #     w3_0 = node_ratio_clamped
    #     #     s1 = np.asarray(s1, float); s2 = np.asarray(s2, float); s3 = np.asarray(s3, float)
    #     #     assert s1.shape == s2.shape == s3.shape
    #     #     struct_head0 = w1_0 * s1 + w2_0 * s2 + w3_0 * s3
    #     #     s0 = _normalize_struct_v4(np.asarray(struct_head0, float), mode=struct_norm)

    #     #     # 상위 M 불일치율 + rho0 (Spearman)
    #     #     M = max(5, int(0.3 * len(subgraphs)))
    #     #     top_sem_idx = set(np.argsort(-sem_head)[:M].tolist())
    #     #     top_str_idx0 = set(np.argsort(-s0)[:M].tolist())
    #     #     disagree = 1.0 - (len(top_sem_idx & top_str_idx0) / max(1, len(top_sem_idx)))
    #     #     try:
    #     #         rho0 = float(spearmanr(sem_head, s0).correlation)
    #     #         if np.isnan(rho0): rho0 = 0.0
    #     #     except Exception:
    #     #         rho0 = 0.0

    #     #     # ---------- pass1: adaptive node_ratio_eff → final struct_head ----------
    #     #     node_ratio_eff = float(np.clip(0.10 * (1.0 - max(0.0, rho0)) * (0.5 + disagree), 0.0, 0.12))
    #     #     struct_pie = 1.0 - node_ratio_eff
    #     #     w1 = struct_pie * float(ensemble_ratio)          # struct_feat
    #     #     w2 = struct_pie * float(1.0 - ensemble_ratio)    # gromov
    #     #     w3 = node_ratio_eff                              # node_level
    #     #     struct_head = w1 * s1 + w2 * s2 + w3 * s3

    #     elif sim_cac_mode == "ensemble":
    #         # s1: struct_feat (radial)
    #         s1 = self._struct_scores_struct_feat(
    #             query_graph, subgraphs, None, radial_tau, ppr_scores=last_ppr_scores
    #         )
            
    #         # sizes = [max(1, sg.vcount()) for sg in subgraphs]
    #         # med = float(np.median(sizes))
    #         # L_eff = int(np.clip(np.sqrt(med) + 6, 8, 20))  # 경험칙: sqrt + 6
    #         # s2 = self._struct_scores_gromov(
    #         #     query_graph, subgraphs, landmark_k=L_eff, ppr_scores=last_ppr_scores,
    #         #     landmark_weighting=landmark_weighting
    #         # )
            
    #         # s2: gromov (landmark)
    #         s2 = self._struct_scores_gromov(
    #             query_graph, subgraphs, landmark_k=landmark_k,
    #             ppr_scores=last_ppr_scores,
    #             landmark_weighting=landmark_weighting,   # v4 반영
    #         )

    #         # (구) 2-way 가중합을 '구조 파이'로 보고,
    #         # node_ratio 만큼 node_level(s3)에 할당하는 3-way로 확장
    #         node_ratio_clamped = float(np.clip(node_ratio, 0.0, 1.0))
    #         struct_pie = 1.0 - node_ratio_clamped

    #         # s3: node-level 구조 유사도
    #         s3 = self._struct_scores_node_level(
    #             query_graph, subgraphs, embedder, ppr_scores=last_ppr_scores,
    #             passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
    #             use_semantic_features=use_semantic_features,
    #             use_treerep_metric=use_treerep_metric
    #         )
    
    #         # 파이 내부 비율: ensemble_ratio로 s1↔s2 분배
    #         w1 = struct_pie * float(ensemble_ratio)          # struct_feat
    #         w2 = struct_pie * float(1.0 - ensemble_ratio)    # gromov
    #         w3 = node_ratio_clamped                          # node_level

    #         # 모든 벡터 길이 정합(안전 가드)
    #         def _as_np(x): return np.asarray(x, dtype=float)
    #         s1, s2, s3 = _as_np(s1), _as_np(s2), _as_np(s3)
    #         assert s1.shape == s2.shape == s3.shape, f"shape mismatch: {s1.shape}, {s2.shape}, {s3.shape}"

    #         struct_head = w1 * s1 + w2 * s2 + w3 * s3

  
    #     else:
    #         raise ValueError(f"Unknown mode: {sim_cac_mode}")


    #     if struct_head.size == 0:
    #         return passage_topk_ids, passage_topk_scores

    #     sem = np.asarray(passage_topk_scores, float)
    #     head_idx = np.argsort(-sem)[:len(subgraphs)]


    #     # 0) 헤드 선택
    #     sem = np.asarray(passage_topk_scores, float)
    #     head_idx = np.argsort(-sem)[:len(subgraphs)]
    #     sem_head = sem[head_idx]
        
    #     # 1) 구조점수 정규화 (v4 옵션)
    #     s = _normalize_struct_v4(np.asarray(struct_head, float), mode=struct_norm)

    #     var = float(np.var(s))
    #     # 분산이 작으면 온도↓(스케일↑)로 차등 강조, 크면 온도↑로 과도한 펌핑 억제
    #     tau = 0.20 if var < 0.01 else (0.30 if var < 0.02 else 0.40)
    #     s = np.tanh((s - 0.5) / tau) * 0.5 + 0.5
        
        
    #     # ==== [NEW] 구조 신호 안정화: winsorize + mild tanh ====
    #     winsor_p = 0.05    # 하위/상위 5% 잘라 고정
    #     tanh_w   = 0.20    # 약한 포화 폭
    #     lo, hi = np.percentile(s, 100*winsor_p), np.percentile(s, 100*(1-winsor_p))
    #     if hi > lo:
    #         s = np.clip(s, lo, hi)
    #         s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    #         s = np.tanh((s - 0.5) / tanh_w) * 0.5 + 0.5
        
    #     # 2) 비선형 스케일 (선택)
    #     if struct_nonlin == "tanh":
    #         # 중앙 0.5 기준으로 부드러운 포화: 지나친 펌핑 억제
    #         s = np.tanh((s - 0.5) / 0.15) * 0.5 + 0.5

    #     # 3) RRF로 순위 스케일 정규화 (동일)
    #     def _ranks(x: np.ndarray) -> np.ndarray:
    #         return (np.argsort(np.argsort(-x)) + 1).astype(np.int32)

    #     rank_sem    = _ranks(sem_head)
    #     rank_struct = _ranks(s)
    #     rrf_sem     = 1.0 / (k_rrf + rank_sem)
    #     rrf_struct  = 1.0 / (k_rrf + rank_struct)

    #     # 4) 상관 기반 게이트 계산(기존 로직 유지)
    #     rho = 0.0
    #     def _is_const_vec(x: np.ndarray) -> bool:
    #         return x.size == 0 or np.allclose(x, x[0])

    #     if gating_enabled and not (_is_const_vec(sem_head) or _is_const_vec(s)):
    #         try:
    #             rho = float(spearmanr(sem_head, s).correlation)
    #             if np.isnan(rho):
    #                 rho = 0.0
    #         except Exception:
    #             rho = 0.0
    #     gate = (np.clip(0.6 + 0.4 * max(0.0, rho), 0.6, 1.0)
    #             if gating_enabled else 1.0)
    #     if gating_enabled and rho <= -0.2:
    #         rrf_struct = 1.0 - rrf_struct

    #     # [PLUS_6] 동적 alpha 계산: rho를 [rho_low, rho_high] → [alpha_min, alpha_max]로 선형 매핑
    #     alpha_eff = float(alpha)
    #     if (alpha_min is not None) and (alpha_max is not None) and (rho_high > rho_low):
    #         t = np.clip((rho - float(rho_low)) / (float(rho_high) - float(rho_low)), 0.0, 1.0)
    #         # smoothstep: 3t^2 - 2t^3 (급격한 중간 상승)
    #         t = float(3*t*t - 2*t*t*t)
    #         alpha_eff = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * t
    #     # else: alpha_eff는 고정 alpha 유지

        
    #     # 5) 결합 방식 (v4 옵션)
    #     if sem_struct_mode == "linear":
    #         final_head = alpha_eff * rrf_sem + (1.0 - alpha_eff) * gate * rrf_struct
    #     elif sem_struct_mode == "gated_add":
    #         gate_w = 1.0 / (1.0 + np.exp(-(rho - 0.10) / 0.15))
    #         g_eff = gate * gate_w
    #         final_head = alpha_eff * rrf_sem + (1.0 - alpha_eff) * g_eff * rrf_struct
    #     else:
    #         final_head = alpha_eff * rrf_sem + (1.0 - alpha_eff) * gate * rrf_struct

    #     # head_pairs = list(zip([passage_topk_ids[i] for i in head_idx], final_head.tolist()))

    #     # # --- adaptive soft pinning ---
    #     # sem_sorted = np.sort(sem_head)[::-1]
    #     # pin_k_eff = pin_k  # 기본값(전달 인자)
    #     # if sem_sorted.size >= 2:
    #     #     margin = float(sem_sorted[0] - sem_sorted[1]) / (abs(sem_sorted[0]) + 1e-6)
    #     #     if margin >= 0.10:
    #     #         pin_k_eff = max(pin_k, 2)   # 강한 1위 → 2개 보호
    #     #     elif margin >= 0.05:
    #     #         pin_k_eff = max(pin_k, 1)   # 약한 1위 → 1개 보호
    #     #     else:
    #     #         pin_k_eff = 0               # 비슷하면 보호 해제

    #     # pin_local_idx = np.argsort(-sem_head)[:pin_k_eff]
    #     # pinned_ids    = [passage_topk_ids[head_idx[i]] for i in pin_local_idx]
    #     # pinned_pairs  = [(passage_topk_ids[head_idx[i]], float(final_head[i])) for i in pin_local_idx]

    #     # others        = [(did, sc) for (did, sc) in head_pairs if did not in pinned_ids]
    #     # others_sorted = sorted(others, key=lambda x: x[1], reverse=True)
    #     # head_pairs_sorted = pinned_pairs + [p for p in others_sorted if p[0] not in set(pinned_ids)]
        
    #     # # 보호핀: semantic 상위 N개는 맨 위 고정
    #     pin_local_idx = np.argsort(-sem_head)[:pin_k]                 # head 내 로컬 인덱스
    #     pinned_ids    = [passage_topk_ids[head_idx[i]] for i in pin_local_idx]
    #     pinned_pairs  = [(passage_topk_ids[head_idx[i]], float(final_head[i])) for i in pin_local_idx]

    #     #보호핀 제외 나머지 정렬
    #     others        = [(did, sc) for (did, sc) in head_pairs if did not in pinned_ids]
    #     others_sorted = sorted(others, key=lambda x: x[1], reverse=True)

    #     # 보호핀을 맨 위에 두고 나머지 이어 붙이기 (중복 제거)
    #     head_pairs_sorted = pinned_pairs + [p for p in others_sorted if p[0] not in set(pinned_ids)]

    #     # 1st-stage 결과(현재 final_head)로 상위 M 후보만 선별
    #     M = min(20, len(head_pairs_sorted))
    #     topM_ids = [doc_id for doc_id, _ in head_pairs_sorted[:M]]
    #     topM_graphs = []
    #     topM_idx = []
    #     for i, (doc_id, _) in enumerate(head_pairs_sorted[:M]):
    #         # head_idx로 subgraphs 내 위치 찾기
    #         try:
    #             loc = [passage_topk_ids[idx] for idx in head_idx].index(doc_id)
    #             topM_graphs.append(subgraphs[loc]); topM_idx.append(i)
    #         except ValueError:
    #             continue

    #     # node-level 정밀 점수
    #     s_refine = self._struct_scores_node_level(
    #         query_graph, topM_graphs, embedder,
    #         ppr_scores=last_ppr_scores,
    #         passage_embeddings=passage_embeddings,
    #         entity_embeddings=entity_embeddings,
    #         use_semantic_features=use_semantic_features,
    #         use_treerep_metric=use_treerep_metric
    #     )
    #     # 정규화 후 소규모 가중 추가(예: 10~15%)
    #     s_refine = _normalize_struct_v4(s_refine, mode=struct_norm)
    #     for j, i_head in enumerate(topM_idx):
    #         head_pairs_sorted[i_head] = (
    #             head_pairs_sorted[i_head][0],
    #             float(0.85 * head_pairs_sorted[i_head][1] + 0.15 * s_refine[j])
    #         )

    #     # topM 내부 재정렬
    #     head_pairs_sorted[:M] = sorted(head_pairs_sorted[:M], key=lambda x: x[1], reverse=True)


    #     # --- 아래는 기존 merge 로직 재사용 ---
    #     orig_scores  = dict(zip(passage_topk_ids, passage_topk_scores))
    #     final_scores = {doc_id: score for doc_id, score in head_pairs_sorted}
    #     merged_ids, merged_scores = [], []
    #     for doc_id, score in head_pairs_sorted:
    #         merged_ids.append(doc_id); merged_scores.append(score)
    #     remaining_ids = [did for did in passage_topk_ids if did not in final_scores]
    #     remaining_pairs = sorted(
    #         [(did, orig_scores[did]) for did in remaining_ids],
    #         key=lambda x: x[1], reverse=True
    #     )
    #     for doc_id, score in remaining_pairs:
    #         merged_ids.append(doc_id); merged_scores.append(score)

    #     # --- 진단 로그(옵션) ---
    #     try:
    #         logger.info(f"[FUSION] k_rrf={k_rrf} gate={gate:.3f} rho={rho:.3f} alpha_eff={alpha_eff:.3f} pinned_ids={pinned_ids}")
    #     except Exception:
    #         pass

    #     return merged_ids, merged_scores

        # # =====================[ 기존 코드 ]=====================
        # # ① semantic [0,1]
        # sem_head = sem[head_idx]
        # sem_norm_head = (sem_head - sem_head.min()) / (sem_head.max() - sem_head.min() + 1e-9)

        # # # ② structural [0,1]
        # # # s = np.asarray(struct_head, float)
        # # # s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        # # # s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        # # s = np.asarray(struct_head, float)
        # # s = StructralSimilarity().robust_normalize_struct(s)

        # # # ③ 결합 (alpha는 의미 가중)
        # # final_head = alpha * sem_norm_head + (1 - alpha) * s

        # # ② structural [0,1] 정규화(강건)
        # s = np.asarray(struct_head, float)
        # s = StructralSimilarity().robust_normalize_struct(s)

        # # ----[ NEW: 동적 게이팅으로 구조 가중 조절 ]----
        # # s = 구조 점수 정규화 결과
        # try:
        #     from scipy.stats import spearmanr
        #     rho = float(spearmanr(sem_norm_head, s).correlation)
        # except Exception:
        #     rho = 0.0

        # if gating_enabled:
        #     if rho < 0.0:     gate = 0.4
        #     elif rho < 0.2:   gate = 0.7
        #     else:             gate = 1.0
        # else:
        #     gate = 1.0

        # final_head = alpha * sem_norm_head + (1 - alpha) * gate * s


        # head_pairs = sorted(
        #     zip([passage_topk_ids[i] for i in head_idx], final_head),
        #     key=lambda x: x[1], reverse=True
        # )

        # diag = {}
        # diag["struct_raw_min"] = float(np.min(struct_head)) if struct_head.size > 0 else 0.0
        # diag["struct_raw_max"] = float(np.max(struct_head)) if struct_head.size > 0 else 0.0
        # diag["struct_norm_min"] = float(np.min(s)) if s.size > 0 else 0.0
        # diag["struct_norm_max"] = float(np.max(s)) if s.size > 0 else 0.0
        # diag["sem_norm_min"] = float(np.min(sem_norm_head)) if sem_norm_head.size > 0 else 0.0
        # diag["sem_norm_max"] = float(np.max(sem_norm_head)) if sem_norm_head.size > 0 else 0.0

        # diag["final_min"] = float(np.min(final_head)) if final_head.size > 0 else 0.0
        # diag["final_max"] = float(np.max(final_head)) if final_head.size > 0 else 0.0
        
        # try:
        #     sem_head = sem[head_idx]
        #     diag['sem_struct_spearmanr'] = float(spearmanr(sem_head, struct_head).correlation)
        #     diag['sem_struct_kendalltau'] = float(kendalltau(sem_head, struct_head).correlation)
        # except Exception:
        #     diag['sem_struct_spearmanr'] = np.nan
        #     diag['sem_struct_kendalltau'] = np.nan

        # diag_msg = f"[RERANK] head_k={struct_k} alpha={alpha} | {diag}"
        # if len(diag_msg) > 1000: diag_msg = diag_msg[:1000] + "..."
        # logger.info(diag_msg)
        
        # try:
        #     head_diag_logger = MetricsLogger(save_dir="./outputs/diag", dataset_name="diag")
        #     head_diag_logger.write({
        #         "dataset": getattr(self, "dataset_name", "default"),
        #         "query_id": getattr(self, "query_id", "unknown"),
        #         "graph_role": "head_diag",
        #         "struct_head_min": diag["struct_raw_min"],
        #         "struct_head_max": diag["struct_raw_max"],
        #         "sem_struct_spearmanr": diag["sem_struct_spearmanr"],
        #         "sem_struct_kendalltau": diag["sem_struct_kendalltau"],
        #         "alpha": float(alpha),
        #         "mode": str(sim_cac_mode),
        #         "struct_k": int(struct_k),
        #     })
        # except Exception:
        #     pass

        # orig_scores = dict(zip(passage_topk_ids, passage_topk_scores))
        # final_scores = {doc_id: score for doc_id, score in head_pairs}
        
        # merged_ids, merged_scores = [], []
        
        # for doc_id, score in head_pairs:
        #     merged_ids.append(doc_id)
        #     merged_scores.append(score)
            
        # remaining_ids = [did for did in passage_topk_ids if did not in final_scores]
        # remaining_pairs = sorted(
        #     [(did, orig_scores[did]) for did in remaining_ids],
        #     key=lambda x: x[1], reverse=True
        # )
        
        # for doc_id, score in remaining_pairs:
        #     merged_ids.append(doc_id)
        #     merged_scores.append(score)

        # return merged_ids, merged_scores
        
        
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
        # -------- 공통/기존 ----------
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False,
        gating_enabled: bool = True,
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
        sim_cac_mode: Literal["tangent", "lorentz", "node_level", "gromov", "struct_feat", "ensemble"] = "tangent",
        lorentz_temp: float = 2.0,
        struct_norm: Literal["robust","zscore","softmax"] = "robust",
        sem_struct_mode: Literal["linear","gated_add"] = "linear",
        struct_nonlin: Literal["none","tanh"] = "none",
        landmark_weighting: Literal["none","degree","betweenness"] = "none",
        node_ratio: float = 0.20,       # 1차 앙상블에 node-level 주입 비율
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        rho_low: float = -0.20,
        rho_high: float = 0.30,
        # --------- [B1/B2/B3 패치 옵션] ----------
        stabilize_struct: bool = True,      # False면 winsorize/tanh 완전 OFF (베이스 재현)
        enable_soft_pin: bool = False,      # True면 Top-1 margin에 따라 pin_k를 동적으로 조정
        soft_pin_margin_lo: float = 0.05,   # margin ∈ [lo, hi]에서 pin 완화/강화
        soft_pin_margin_hi: float = 0.10,
        topm_refine: bool = False,          # True면 Top-M node-level 정밀 재평가
        topm_M: int = 20,                   # 고정 Top-M
        topm_refine_weight: float = 0.15,   # Top-M 재평가 가중(합산 비율)
        topm_adaptive: bool = False,        # True면 s의 분산으로 M/weight 자동 조정 (B2)
    ) -> Tuple[List[int], List[float]]:

        def _clip(a, lo, hi): 
            return max(lo, min(hi, a))

        passage_topk_ids = list(np.asarray(passage_topk_ids).tolist())
        passage_topk_scores = list(np.asarray(passage_topk_scores, dtype=float).tolist())

        if query_graph is None or len(passage_topk_ids) == 0:
            return passage_topk_ids, passage_topk_scores

        # 1) 서브그래프 생성
        subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
            passage_topk_ids, passage_topk_scores, passage_node_idxs,
            last_ppr_scores, cfg, head_k=struct_k
        )
        if not subgraphs:
            return passage_topk_ids, passage_topk_scores

        # 2) 헤드 시멘틱 스코어 추출
        sem = np.asarray(passage_topk_scores, float)
        head_idx = np.argsort(-sem)[:len(subgraphs)]
        sem_head = sem[head_idx]

        # 3) 구조 점수 산출
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
            # s1: struct_feat
            s1 = self._struct_scores_struct_feat(
                query_graph, subgraphs, None, radial_tau, ppr_scores=last_ppr_scores
            )
            # s2: gromov
            s2 = self._struct_scores_gromov(
                query_graph, subgraphs, landmark_k=landmark_k,
                ppr_scores=last_ppr_scores, landmark_weighting=landmark_weighting
            )
            # s3: node-level
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

        # 4) 구조점수 정규화 → (필요시) 안정화
        s = _normalize_struct_v4(np.asarray(struct_head, float), mode=struct_norm)

        # ---- [B2] 분산 적응형 tanh (기존 코드 보강) ----
        # stabilize_struct가 False면 모든 안정화/비선형 처리 스킵 (베이스 재현)
        if stabilize_struct:
            var = float(np.var(s))
            # 분산에 따른 약한 tanh: 낮은 분산에서 분해능↑, 높은 분산에서 펌핑 억제
            tau = 0.20 if var < 0.01 else (0.30 if var < 0.02 else 0.40)
            s = np.tanh((s - 0.5) / tau) * 0.5 + 0.5

            # winsorize + mild tanh (기존 안정화)
            winsor_p = 0.05
            tanh_w   = 0.20
            lo, hi = np.percentile(s, 100*winsor_p), np.percentile(s, 100*(1-winsor_p))
            if hi > lo:
                s = np.clip(s, lo, hi)
                s = (s - s.min()) / (s.max() - s.min() + 1e-12)
                s = np.tanh((s - 0.5) / tanh_w) * 0.5 + 0.5
            if struct_nonlin == "tanh":
                s = np.tanh((s - 0.5) / 0.15) * 0.5 + 0.5
        # else: 안정화 전부 OFF

        # 5) RRF 결합 준비
        def _ranks(x: np.ndarray) -> np.ndarray:
            return (np.argsort(np.argsort(-x)) + 1).astype(np.int32)
        rank_sem    = _ranks(sem_head)
        rank_struct = _ranks(s)
        rrf_sem     = 1.0 / (k_rrf + rank_sem)
        rrf_struct  = 1.0 / (k_rrf + rank_struct)

        # 6) 상관 기반 게이트 + 동적 alpha
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
            # smoothstep으로 완만하게
            t = float(3*t*t - 2*t*t*t)
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
        # head_pairs를 먼저 만든다(이전 NameError 방지)
        head_pairs = list(zip([passage_topk_ids[i] for i in head_idx], final_head.tolist()))

        pin_k_eff = pin_k
        if enable_soft_pin:
            sem_sorted = np.sort(sem_head)[::-1]
            if sem_sorted.size >= 2:
                margin = float(sem_sorted[0] - sem_sorted[1]) / (abs(sem_sorted[0]) + 1e-6)
                if margin >= soft_pin_margin_hi:
                    pin_k_eff = max(pin_k, 2)  # 1등이 너무 강하면 2개 보호
                elif margin >= soft_pin_margin_lo:
                    pin_k_eff = max(pin_k, 1)  # 약한 과점이면 1개 보호 유지
                else:
                    pin_k_eff = 0              # 과점 아님 → 보호 해제
        pin_local_idx = np.argsort(-sem_head)[:pin_k_eff]
        pinned_ids    = [passage_topk_ids[head_idx[i]] for i in pin_local_idx]
        pinned_pairs  = [(passage_topk_ids[head_idx[i]], float(final_head[i])) for i in pin_local_idx]

        others        = [(did, sc) for (did, sc) in head_pairs if did not in pinned_ids]
        others_sorted = sorted(others, key=lambda x: x[1], reverse=True)
        head_pairs_sorted = pinned_pairs + [p for p in others_sorted if p[0] not in set(pinned_ids)]

        # 8) Top-M node-level 정밀 보정 (B2: adaptive M/weight 가능)
        if topm_refine and len(head_pairs_sorted) > 0:
            M_fix = int(topm_M)
            W_fix = float(topm_refine_weight)
            M_eff, W_eff = M_fix, W_fix
            if topm_adaptive:
                # s 분산 기반 적응: M = clip(int(16 + 80*var), 12, 28), w = clip(0.10 + 0.60*var, 0.10, 0.22)
                var_s = float(np.var(s))
                M_eff = _clip(int(16 + 80 * var_s), 12, 28)
                W_eff = _clip(0.10 + 0.60 * var_s, 0.10, 0.22)

            M = min(M_eff, len(head_pairs_sorted))
            if M > 0:
                topM_ids = [doc_id for doc_id, _ in head_pairs_sorted[:M]]
                topM_graphs, topM_idx = [], []
                # head_idx → subgraphs 위치 매핑
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
                    w = float(_clip(W_eff, 0.0, 0.5))  # 과도 방지
                    for j, i_head in enumerate(topM_idx):
                        head_pairs_sorted[i_head] = (
                            head_pairs_sorted[i_head][0],
                            float((1.0 - w) * head_pairs_sorted[i_head][1] + w * s_refine[j])
                        )
                    head_pairs_sorted[:M] = sorted(head_pairs_sorted[:M], key=lambda x: x[1], reverse=True)

        # 9) 남은 것 합치기
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
            logger.info(f"[FUSION] k_rrf={k_rrf} gate={gate:.3f} rho={rho:.3f} alpha_eff={alpha_eff:.3f} pin_k_eff={pin_k_eff} pinned_ids={pinned_ids}")
        except Exception:
            pass

        return merged_ids, merged_scores
