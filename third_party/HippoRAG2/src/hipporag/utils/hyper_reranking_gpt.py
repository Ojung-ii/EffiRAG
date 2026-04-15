# utils/hyper_reranking.py
from __future__ import annotations
import igraph as ig
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Set, Callable, Literal
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ------------ Query triples to Graph-------------------------------------
# --- helper: torch/tensor-like → 안전한 1D numpy ---
def _to_1d_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().flatten()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        return x
    arr = np.asarray(x, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

class QueryGraphBuilder:
    def __init__(self, root_node: str = "question", directed: bool = True):
        self.root_node = root_node
        self.directed = directed

    def build(self, triples: list[tuple]) -> ig.Graph:
        def norm(x): return str(x).strip()

        # 빈 입력 처리
        if not triples:
            g = ig.Graph(directed=self.directed)
            g.add_vertex(name=self.root_node, is_virtual=True)
            g["root_id"] = g.vs.find(name=self.root_node).index
            return g

        # 유효 (s,p,o)만 추출
        valid_triples = []
        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, p, o = map(norm, t)
                if s and p and o and s != o:
                    valid_triples.append((s, p, o))

        g = ig.Graph(directed=self.directed)
        if not valid_triples:
            g.add_vertex(name=self.root_node, is_virtual=True)
            g["root_id"] = g.vs.find(name=self.root_node).index
            return g

        # 노드 준비 (+ question 루트 항상 추가)
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

        # (s→o) 방향 유지하여 간선 추가, 관계는 리스트로 누적
        edge_map = {}  # key=(u,v) 그대로(유향), 값=list of predicates
        for s, p, o in valid_triples:
            u, v = name2id[s], name2id[o]
            edge_map.setdefault((u, v), []).append(p)

        if edge_map:
            uv = list(edge_map.keys())
            g.add_edges(uv)
            rel_lists = [edge_map[k] for k in uv]
            g.es["relations"] = [list(dict.fromkeys(rels)) for rels in rel_lists]  # 중복 제거
            g.es["relation"] = [rels[0] for rels in g.es["relations"]]
        else:
            g.es["relations"] = []; g.es["relation"] = []

        # 컴포넌트 허브와 루트 question **항상** 연결
        root_id = name2id[self.root_node]
        comps = g.components(mode="WEAK")
        for comp in comps:
            if root_id in comp:
                continue
            degs = g.degree(comp, mode="all")
            hub_local = int(np.argmax(degs))
            hub_vid = comp[hub_local]
            if g.get_eid(root_id, hub_vid, directed=False, error=False) == -1:
                # 루트→허브(유향)로 연결(방향 의미 없으면 directed=False 옵션으로 그래프 생성)
                g.add_edge(root_id, hub_vid)
                g.es[-1]["relation"] = "root"
                g.es[-1]["relations"] = ["root"]

        logger.debug("[QueryGraphBuilder] directed=%s nodes=%d, edges=%d",
                     self.directed, g.vcount(), g.ecount())
        return g

# --------------------------- ID Mapper ---------------------------

class IDMapper:
    @staticmethod
    def docids_to_vids(doc_ids: Iterable[int], passage_node_idxs: List[int]) -> List[int]:
        """PASSAGE 배열 인덱스 -> 그래프 전역 정점 ID(vid)"""
        return [int(passage_node_idxs[d]) for d in doc_ids]

# --------------------------- Config ------------------------------

@dataclass
class SubgraphConfig:
    max_depth: int = 2
    weight_threshold: float = -1e-9
    allow_topk: int = 200
    flow_quantile: float = 0.65
    per_node_top_r: int = 7
    ensure_connectivity: bool = True
    starter_beam: int = 8
    # --- 추가 ---
    per_hop_topk: int = 10                     # 각 홉에서 노드별 확장 상한
    rank_metric: Literal["ppr","weight","deg"] = "ppr"  # 확장 우선순위 기준

# ------------------------ Subgraph Builder -----------------------
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
        # 폴백
        return float(w) if self.weighted else 1.0

    def collect_candidates(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
        """
        유향은 OUT만, 무향은 ALL.
        각 홉마다 노드별로 이웃을 점수순 상위 per_hop_topk만 확장(확장 단계 필터링).
        """
        visited = {root}
        q = deque([(root, 0)])
        edges: List[Tuple[int,int,float]] = []
        mode = "OUT" if self.g.is_directed() else "ALL"
        K = max(1, int(self.cfg.per_hop_topk)) if self.cfg.per_hop_topk else None

        while q:
            u, depth = q.popleft()
            if depth >= self.cfg.max_depth:
                continue

            # 후보 이웃 수집
            neighs = []
            for eid in self.g.incident(u, mode=mode):
                e = self.g.es[eid]
                if self.g.is_directed() and e.source != u:
                    continue  # OUT-only
                v = e.target if e.source == u else e.source
                w = float(e['weight']) if self.weighted else 1.0
                if w < self.cfg.weight_threshold:
                    continue
                if allow_ids is not None and v not in allow_ids:
                    continue
                neighs.append((u, v, w))

            # per-hop top-k 확장
            if K is not None and len(neighs) > K:
                neighs.sort(key=lambda t: self._rank_key(*t), reverse=True)
                neighs = neighs[:K]

            # 수락
            for (uu, vv, ww) in neighs:
                edges.append((uu, vv, ww))
                if vv not in visited:
                    visited.add(vv)
                    q.append((vv, depth + 1))

        # depth-1 비면 starter_beam 보장(기존 로직 유지)
        if not edges and self.cfg.starter_beam > 0:
            nbr_eids = self.g.incident(root, mode=mode)
            cand = []
            for eid in nbr_eids:
                e = self.g.es[eid]
                if self.g.is_directed() and e.source != root:
                    continue
                v = e.target if e.source == root else e.source
                w = float(e['weight']) if self.weighted else 1.0
                if allow_ids is None or v in allow_ids:
                    cand.append((root, v, w))
            # starter_beam도 같은 기준으로 상위 선별
            if K is not None and len(cand) > self.cfg.starter_beam:
                cand.sort(key=lambda t: self._rank_key(*t), reverse=True)
                cand = cand[:self.cfg.starter_beam]
            edges = cand

        return edges

    def build_subgraph(self, edges: List[Tuple[int,int,float]], root: Optional[int] = None) -> ig.Graph:
        """후보 엣지로 서브그래프 생성. 중복 간선은 가중치 합산으로 병합."""
        sg = ig.Graph(directed=self.g.is_directed())
        if not edges:
            sg.add_vertex(name=(root if root is not None else "root"))
            sg.vs['global_id'] = [root if root is not None else -1]
            return sg

        nodes = sorted({u for u,_,_ in edges} | {v for _,v,_ in edges})
        idx = {n:i for i,n in enumerate(nodes)}
        sg.add_vertices(len(nodes))
        sg.vs['name'] = nodes
        sg.vs['global_id'] = nodes
        if root is not None:
            sg["root_id"] = int(root)
        sg.add_edges([(idx[u], idx[v]) for u,v,_ in edges])
        sg.es['weight'] = [w for *_,w in edges]

        # 중복 간선 병합: weight 합산
        import numpy as _np
        sg.simplify(multiple=True, loops=True, combine_edges=_np.sum)
        return sg


# --------------------------- PPR Pruner --------------------------

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
            return 0.0  # 역방향/무향 폴백 금지
        return float(self.pi[u]) * self._P(u, eid)

    def prune(self, root: int, candidate_edges: List[Tuple[int,int,float]]) -> List[Tuple[int,int,float]]:
        """유량 분위수 컷(>=) → 전멸 방지 → 노드별 top‑r(타이 포함)"""
        if not candidate_edges or self.pi is None:
            return candidate_edges

        flows = np.array([self._flow(u, v) for u, v, _ in candidate_edges], dtype=float)
        flows = np.nan_to_num(flows, nan=0.0, posinf=0.0, neginf=0.0)

        kept_idx = np.arange(len(candidate_edges))
        if self.cfg.flow_quantile is not None:
            finite = np.isfinite(flows)
            q = float(np.quantile(flows[finite], self.cfg.flow_quantile)) if finite.any() else 0.0
            cut = max(self.cfg.weight_threshold, q)
            mask = (flows >= cut)  # >= 동률 포함
            kept_idx = kept_idx[mask]

            # 전멸 방지: 중앙값 재시도 → 그래도 없으면 상위 N 강제
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
            final_idx.extend([i for (i, f) in items if f >= th])  # 타이 포함

        # (선택) 전체 간선 수도 캡핑하고 싶으면 allow_topk 재활용
        if self.cfg.allow_topk is not None and len(final_idx) > self.cfg.allow_topk:
            order = np.argsort(-flows[final_idx])
            final_idx = [final_idx[i] for i in order[:self.cfg.allow_topk]]

        return [candidate_edges[i] for i in final_idx]

# ------------------------ Hyperbolic Embedder --------------------

class HyperbolicEmbedder:
    def __init__(self, embed_dim=32, curvature=1.0, device="cpu"):
        import geoopt
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.embed_dim = embed_dim
        self.device = device
    def graph_to_tensor(self, g: ig.Graph):
        if g.vcount() == 0:
            feats = np.zeros((1, self.embed_dim), dtype=np.float32)
            return torch.tensor(feats, device=self.device)

        # 1) 기본 차수(방향 반영)
        indeg  = np.asarray(g.indegree()  if g.is_directed() else g.degree(), dtype=np.float32)
        outdeg = np.asarray(g.outdegree() if g.is_directed() else g.degree(), dtype=np.float32)
        indeg  = np.log1p(indeg)
        outdeg = np.log1p(outdeg)

        # 2) 루트 기준 깊이/홉 (undirected로 거리 계산이 더 안정적)
        root_local = 0
        try:
            if "root_id" in g.attributes() and "global_id" in g.vs.attributes():
                gids = np.asarray(g.vs["global_id"])
                rid  = int(g["root_id"])
                idx  = np.where(gids == rid)[0]
                if len(idx): root_local = int(idx[0])
        except: pass
        gud = g.as_undirected() if g.is_directed() else g
        d = np.asarray(gud.shortest_paths(source=root_local)[0], dtype=np.float32)
        finite = np.isfinite(d)
        maxd = float(np.max(d[finite])) if finite.any() else 0.0
        d[~finite] = maxd + 1.0
        depth = d / (maxd + 1e-6)
        hop1 = (d == 1).astype(np.float32)
        hop2 = (d == 2).astype(np.float32)

        # 3) PageRank(방향 반영)
        try:
            pr = np.asarray(g.pagerank(directed=g.is_directed()), dtype=np.float32)
        except:
            pr = np.zeros(g.vcount(), dtype=np.float32)

        # 4) 리프 플래그(무향 기준)

        gud_deg = np.asarray(gud.degree(), dtype=np.int32)
        leaf = (gud_deg == 1).astype(np.float32)

        # 5) 피처 스택 (여기까지만으로도 7D)
        F = np.stack([indeg, outdeg, depth.astype(np.float32), hop1, hop2, pr, leaf], axis=1)

        # 6) 각 feature-dim z-score (그래프 내부 표준화)
        mean = F.mean(axis=0, keepdims=True)
        std  = F.std(axis=0, keepdims=True)
        std  = np.maximum(std, 1e-3)
        Fz   = (F - mean) / std
        # 과도한 포화 방지
        Fz   = np.clip(Fz, -4.0, 4.0)

        # 7) bias(1) 추가해 극단적 제로벡터 방지
        bias = np.ones((Fz.shape[0], 1), dtype=np.float32)
        Fz   = np.concatenate([Fz, bias], axis=1)  # (N, 8)

        # 8) 패딩
        if Fz.shape[1] < self.embed_dim:
            pad = np.zeros((Fz.shape[0], self.embed_dim - Fz.shape[1]), dtype=np.float32)
            Fz  = np.concatenate([Fz, pad], axis=1)
        else:
            Fz = Fz[:, :self.embed_dim]

        return torch.tensor(Fz, dtype=torch.float32, device=self.device)

#     def graph_to_tensor(self, g: ig.Graph):
#         if g.vcount() == 0:
#             feats = np.zeros((1, self.embed_dim), dtype=np.float32)
#             return torch.tensor(feats, device=self.device)

# # ------------------- DEGREE만 사용 시작 ------------------
#         degrees = np.array(g.degree()).reshape(-1, 1)
#         # clustering = np.array(g.transitivity_local_undirected(vertices=None)).reshape(-1, 1)
#         # clustering[np.isnan(clustering)] = 0
#         # pagerank = np.array(g.pagerank()).reshape(-1, 1)
#         # for arr in (clustering, pagerank):
#         #     arr[~np.isfinite(arr)] = 0.0

#         feats = np.concatenate([degrees], axis = 1) #, clustering, pagerank], axis=1) #


#         mean, std = feats.mean(axis=0), feats.std(axis=0)
#         std = np.maximum(std, 1e-3)
#         z = (feats - mean) / std
#         z = np.clip(z, -3.0, 3.0) * 3.0
#         pad = np.zeros((z.shape[0], self.embed_dim - z.shape[1]), dtype=z.dtype)
#         feats = np.concatenate([z, pad], axis=1)
# ------------- DEGREE만 사용 끝 ------------------



    def embed_lorentz(self, g: ig.Graph) -> torch.Tensor:
        if g.vcount() == 0:
            return None
        if g.vcount() == 1 and g.ecount() == 0:
            x = torch.zeros((1, self.embed_dim), device=self.device)
            x[0, 0] = 1e-2
        else:
            x = self.graph_to_tensor(g)

        with torch.no_grad():
            max_norm = 5.0
            n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
            x = x * torch.clamp(max_norm / n, max=1.0)

        x_L = self.manifold.expmap0(x)

        try:
            root_local = ...
            gud = g.as_undirected() if g.is_directed() else g
            d = np.asarray(gud.shortest_paths(source=root_local)[0], dtype=np.float32)
            finite = np.isfinite(d); maxd = float(np.max(d[finite])) if finite.any() else 0.0
            d[~finite] = maxd + 1.0
            depth = torch.tensor(d/(maxd+1e-6), dtype=torch.float32, device=self.device)
            w = torch.exp(- depth / 2.5)  # τ=2.5 정도
            w = (w / (w.mean() + 1e-6)).clamp(0.5, 2.0).unsqueeze(1)  # 안정화
            pooled = (w * x_L).sum(dim=0) / w.sum()
        except:
            pooled = x_L.mean(dim=0)
        # x_lorentz = self.manifold.expmap0(x)
        # x_lorentz = torch.nan_to_num(x_lorentz, nan=0.0, posinf=0.0, neginf=0.0)
        # pooled = x_lorentz.mean(dim=0)
        # pooled = self.manifold.projx(pooled) ## ----> 평균을 Lorentz 다양체로 투영
        # if torch.linalg.norm(pooled).item() < 1e-8:
        #     pooled = pooled + 1e-3 * torch.randn_like(pooled)
        #     pooled = self.manifold.projx(pooled)

        return pooled


    def logmap0_safe(self, lorentz_vec: torch.Tensor) -> torch.Tensor:
        t = self.manifold.logmap0(lorentz_vec)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def embed_and_logmap0(self, g: ig.Graph) -> torch.Tensor:
        y = self.embed_lorentz(g)
        if y is None:
            return torch.zeros(self.embed_dim, device=self.device)
        # ★ 로그맵 전에도 보호
        if torch.linalg.norm(y).item() < 1e-8:
            y = y + 1e-3 * torch.randn_like(y)
            y = self.manifold.projx(y)
        t = self.logmap0_safe(y)
        # ★ 로그맵 후에도 보호
        if torch.linalg.norm(t).item() < 1e-8:
            t[0] = 1e-3
        return t


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

# ---------------------- Structural Similarity --------------------

class StructralSimilarity:
    def __init__(self, mode="cosine"):
        assert mode in ["cosine", "l2", "dot"], "Invalid similarity mode"
        self.mode = mode

    # def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
    #     scores = np.asarray(scores, dtype=float)
    #     mn, mx = scores.min(initial=0.0), scores.max(initial=0.0)
    #     if mx - mn < 1e-8:
    #         return np.zeros_like(scores, dtype=float)
    #     return (scores - mn) / (mx - mn + 1e-8)
    def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(scores, dtype=float)
        if x.size == 0:
            return x
        # ✅ 강건 정규화: 5~95 분위수 구간만 [0,1]로 스케일링
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        if hi - lo < 1e-8:
            return np.full_like(x, 0.5)  # 전부 같은 값이면 중립
        y = (x - lo) / (hi - lo + 1e-8)
        y = np.clip(y, 0.0, 1.0)        # 구간 밖은 클리핑
        return y
    
    def compute(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.mode == "cosine":
            return F.cosine_similarity(q, d, dim=-1)
        if self.mode == "dot":
            return torch.sum(q * d, dim=-1)
        return -torch.norm(q - d, dim=-1)  # l2

    def compute_listwise(self, query_list: List[torch.Tensor], doc_list: List[torch.Tensor]) -> torch.Tensor:
        assert len(query_list) == len(doc_list)
        sims = []
        for q, d in zip(query_list, doc_list):
            sim = self.compute(q.unsqueeze(0), d.unsqueeze(0)).squeeze()
            sims.append(sim)
        return torch.stack(sims)

# ------------------------- Orchestrator --------------------------

class HyperReranker:
    """Top‑K passage를 루트로 PPR 기반 프루닝 서브그래프 생성 → 구조 점수 융합"""
    def __init__(self, g: ig.Graph):
        self.g = g

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
        topk_doc_ids: List[int],            # PASSAGE 배열 인덱스
        topk_scores: List[float],
        passage_node_idxs: List[int],
        last_ppr_scores: Optional[np.ndarray],
        cfg: SubgraphConfig,
        head_k: int = 30,
    ) -> Tuple[List[ig.Graph], List[int], List[int]]:
        if not topk_doc_ids:
            return [], [], []
        sem = np.asarray(topk_scores, float)
        head_idx = np.argsort(-sem)[: min(head_k, len(topk_doc_ids))]
        head_doc_ids = [topk_doc_ids[i] for i in head_idx]
        head_vids    = IDMapper.docids_to_vids(head_doc_ids, passage_node_idxs)

        allow_ids = self.make_allow_ids(self.g, head_vids, last_ppr_scores, cfg.allow_topk)
        builder = SubgraphBuilder(self.g, cfg, node_scores=last_ppr_scores)  # ★ 추가: PPR로 per-hop 랭킹
        pruner  = PPRPruner(self.g, last_ppr_scores, cfg)

        subgraphs: List[ig.Graph] = []
        for rid in head_vids:
            cands  = builder.collect_candidates(root=rid, allow_ids=allow_ids)
            pruned = pruner.prune(root=rid, candidate_edges=cands)
            sg     = builder.build_subgraph(pruned, root=rid)
            subgraphs.append(sg)
            logger.debug(f"[Subgraph] root={rid} cands={len(cands)} pruned={len(pruned)} "
                         f"v={sg.vcount()} e={sg.ecount()}")
        return subgraphs, head_doc_ids, head_vids
    
    # ---------------- 구조점수 계산 함수: 탄젠트/로런츠 ----------------
#     def _struct_scores_tangent(
#         self,
#         query_graph: ig.Graph,
#         subgraphs: List[ig.Graph],
#         embedder: "HyperbolicEmbedder",
#         scorer: "StructralSimilarity",
#     ) -> np.ndarray:
#         """하이퍼볼릭→탄젠트(logmap0) 투영 + 공통 표준화 → listwise 유사도 (큰값=유사)"""
#         with torch.no_grad():
#             tq    = embedder.embed_and_logmap0(query_graph)
#             tdocs = [embedder.embed_and_logmap0(g) for g in subgraphs]

#         # tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)        
#         # q_list = [tq_std for _ in range(len(tdocs_std))]
#         # struct_head = scorer.compute_listwise(q_list, tdocs_std)
#         # return _to_1d_numpy(struct_head)  # 1D numpy, NaN/Inf 제거
#         # ✅ 벡터별 L2 정규화 제거
#         # tq = F.normalize(tq, dim=0, eps=1e-6)
#         # tdocs = [F.normalize(t, dim=0, eps=1e-6) for t in tdocs]

#         # ✅ 공통 표준화 사용 (배치 단일 평균/표준편차)
# #         tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
# #         q_list = [tq_std for _ in range(len(tdocs_std))]

# #         struct_head = scorer.compute_listwise(q_list, tdocs_std)

# #                 # _struct_scores_tangent 끝나기 전에 예: 
# #         logger.debug(
# #             "[EMB-NORM] tq=%.3e, tdocs_min=%.3e, tdocs_max=%.3e",
# #             torch.linalg.norm(tq).item(),
# #             min(torch.linalg.norm(t).item() for t in tdocs),
# #             max(torch.linalg.norm(t).item() for t in tdocs),
# # )

# #         return _to_1d_numpy(struct_head)
# # ------------------ 임시 ------------
#     # (A) 세트 표준화 유지 (공간 통일)
#         tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
#         q_list = [tq_std for _ in range(len(tdocs_std))]

#         # (B) 코사인 → 각도(sim in [0,1]) 변환
#         cos = scorer.compute_listwise(q_list, tdocs_std)        # [-1, 1]
#         cos = torch.clamp(cos, -0.999, 0.999)                   # 수치 안전
#         ang = torch.arccos(cos)                                 # [0, π]
#         sim = 1.0 - (ang / np.pi)                               # [0,1], 중앙부 민감 ↑, 극단부 민감 ↓

#         # (C) 소온도 squash (선택, 과민하면 켜기)
#         # sim = torch.sigmoid( (sim - 0.5) / 0.15 )              # temp=0.15~0.3

#         return _to_1d_numpy(sim)

    def _struct_scores_tangent(self, query_graph, subgraphs, embedder, scorer) -> np.ndarray:
        """하이퍼볼릭→탄젠트(logmap0) 투영 + 공통 표준화 → listwise 유사도"""
        # with torch.no_grad():
        #     tq    = embedder.embed_and_logmap0(query_graph)
        #     tdocs = [embedder.embed_and_logmap0(g) for g in subgraphs]

        # # ---------- [진단 1] 문서 임베딩 분산/상관 (정규화 전 RAW) ----------
        # try:
        #     T = torch.stack(tdocs)  # [H, D]
        #     C = F.cosine_similarity(T.unsqueeze(1), T.unsqueeze(0), dim=-1)  # [H,H]
        #     mask = ~torch.eye(C.size(0), dtype=bool, device=C.device)
        #     vals = C[mask]
        #     logger.debug(
        #         "[EMB-DIAG][raw] tdocs_norm_mean=%.4f tdocs_std=%.4f | pairwise_cos_mean=%.4f p95=%.4f",
        #         T.norm(dim=1).mean().item(),
        #         T.std().item(),
        #         vals.mean().item(),
        #         vals.kthvalue(max(1, int(0.95 * len(vals))))[0].item()
        #     )
        #     # 쿼리 vs 문서 RAW cos
        #     cos_raw = scorer.compute_listwise([tq for _ in tdocs], tdocs)  # tensor [H]
        #     q05, q50, q95 = torch.quantile(cos_raw, torch.tensor([0.05, 0.5, 0.95], device=cos_raw.device))
        #     logger.debug(
        #         "[EMB-DIAG][raw] qvdoc_cos min=%.4f max=%.4f | p05=%.4f p50=%.4f p95=%.4f",
        #         cos_raw.min().item(), cos_raw.max().item(), q05.item(), q50.item(), q95.item()
        #     )
        # except Exception as e:
        #     logger.debug("[EMB-DIAG] raw diag skipped: %s", e)

        # # ---- (여기서 니가 쓰는 정규화/표준화 진행) ----
        # tq = F.normalize(tq, dim=0, eps=1e-6)
        # tdocs = [F.normalize(t, dim=0, eps=1e-6) for t in tdocs]

        # # ---------- [진단 2] 정규화 이후 확인 (선택) ----------
        # try:
        #     Tn = torch.stack(tdocs)
        #     Cn = F.cosine_similarity(Tn.unsqueeze(1), Tn.unsqueeze(0), dim=-1)
        #     maskn = ~torch.eye(Cn.size(0), dtype=bool, device=Cn.device)
        #     valsn = Cn[maskn]
        #     cos_n = scorer.compute_listwise([tq for _ in tdocs], tdocs)
        #     q05n, q50n, q95n = torch.quantile(cos_n, torch.tensor([0.05, 0.5, 0.95], device=cos_n.device))
        #     logger.debug(
        #         "[EMB-DIAG][norm] pairwise_cos_mean=%.4f p95=%.4f | qvdoc min=%.4f max=%.4f p05=%.4f p50=%.4f p95=%.4f",
        #         valsn.mean().item(), valsn.kthvalue(max(1, int(0.95 * len(valsn))))[0].item(),
        #         cos_n.min().item(), cos_n.max().item(), q05n.item(), q50n.item(), q95n.item()
        #     )
        # except Exception as e:
        #     logger.debug("[EMB-DIAG] norm diag skipped: %s", e)

        # q_list = [tq for _ in range(len(tdocs))]
        # struct_head = scorer.compute_listwise(q_list, tdocs)
        # return _to_1d_numpy(struct_head)

        with torch.no_grad():
            tq  = embedder.embed_and_logmap0(query_graph)
            tds = [embedder.embed_and_logmap0(g) for g in subgraphs]

        # 1) 세트 표준화 (차원별)
        stack = torch.stack([tq] + tds, dim=0)                  # [(1+K), d]
        mu    = stack.mean(dim=0, keepdim=True)
        sigma = stack.std(dim=0, keepdim=True).clamp_min(1e-6)
        stack = (stack - mu) / sigma                            # 공통 스케일

        # 2) 코사인 계산 (q: [d], docs: [K,d])
        q  = F.normalize(stack[0], dim=0, eps=1e-6)
        Ds = F.normalize(stack[1:], dim=1, eps=1e-6)
        q_list = [q for _ in range(Ds.size(0))]
        struct_head = scorer.compute_listwise(q_list, list(Ds))
        # ---------------------프라이머 추가 ----------
        def _overlap_prior(qg: ig.Graph, sg: ig.Graph) -> float:
            qv = set(map(str.lower, (qg.vs["name"] if "name" in qg.vs.attributes() else [])))
            sv = set(map(str.lower, (sg.vs["name"] if "name" in sg.vs.attributes() else [])))
            node_j = (len(qv & sv) / max(1, len(qv | sv))) if (qv and sv) else 0.0

            qe = set(map(str.lower, (qg.es["relation"] if "relation" in qg.es.attributes() else [])))
            se = set(map(str.lower, (sg.es["relation"] if "relation" in sg.es.attributes() else [])))
            edge_j = (len(qe & se) / max(1, len(qe | se))) if (qe and se) else 0.0

            return 0.7 * node_j + 0.3 * edge_j

        # _struct_scores_tangent() 또는 _struct_scores_lorentz() 안에서:
        priors = np.array([_overlap_prior(query_graph, sg) for sg in subgraphs], dtype=float)
        # 너무 0으로 죽지 않게 바닥값
        priors = np.clip(priors, 0.2, 1.0)

        # 정규화 전에 소프트 게이트
        struct_head = struct_head * priors  
        # ------------------------------프라이머 추가 끝 ----------

        return _to_1d_numpy(struct_head)
    ############################################################################
    def _struct_scores_lorentz(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        embedder: "HyperbolicEmbedder",
        temp: float = 2.0,
    ) -> np.ndarray:
        """로런츠 거리(dist2, 작을수록 가깝다) → 0~1 유사도(큰값=유사)로 스쿼시"""
        with torch.no_grad():
            qL = embedder.embed_lorentz(query_graph)            # [d]
            dL = [embedder.embed_lorentz(g) for g in subgraphs] # list[[d]]
            dists2 = torch.stack([
                embedder.manifold.dist2(qL.unsqueeze(0), x.unsqueeze(0)).squeeze()
                for x in dL
            ])
        d2 = _to_1d_numpy(dists2)        # 작을수록 유사
        sim = -d2                        # 큰값=유사
        if sim.size:
            sim = (sim - sim.min()) / (sim.ptp() + 1e-8)                # [0,1]
            sim = 1.0 / (1.0 + np.exp(-(sim - 0.5) / (1.0 / temp)))     # 부드럽게
        return sim



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
        alpha: float = 0.5,
        struct_k: int = 30,
        device: str = "cpu",
        save_cb: Optional[Callable[[ig.Graph, str, dict], None]] = None,
        save_ctx: Optional[dict] = None,
        sim_cac_mode: Literal["tangent", "lorentz"] = "tangent",
        lorentz_temp: float = 2.0,
    ) -> Tuple[List[int], List[float]]:

        passage_topk_ids = list(np.asarray(passage_topk_ids).tolist())
        passage_topk_scores = list(np.asarray(passage_topk_scores, dtype=float).tolist())

        # DPR fast-path
        if query_graph is None or len(passage_topk_ids) == 0:
            return passage_topk_ids, passage_topk_scores

        # 1) PPR 기반 프루닝 서브그래프
        subgraphs, head_doc_ids, head_vids  = self.build_pruned_subgraphs_for_topk_passages(
            passage_topk_ids, passage_topk_scores, passage_node_idxs,
            last_ppr_scores, cfg, head_k=struct_k
        )
        if not subgraphs:
            return passage_topk_ids, passage_topk_scores

        # 2) 구조점수 계산
        if sim_cac_mode == "tangent":
            struct_head = self._struct_scores_tangent(query_graph, subgraphs, embedder, scorer)
        elif sim_cac_mode == "lorentz":
            struct_head = self._struct_scores_lorentz(query_graph, subgraphs, embedder, temp=lorentz_temp)
        else:
            raise ValueError(f"Unknown mode: {sim_cac_mode}")

        if struct_head.size == 0:
            return passage_topk_ids, passage_topk_scores

        # 3) 점수 융합
        sem = np.asarray(passage_topk_scores, float)
        head_idx = np.argsort(-sem)[: len(subgraphs)]
# -------------기존 가중합
        # sem_norm_head = scorer.normalize(sem[head_idx])
        # str_norm_head = scorer.normalize(struct_head)

        # final_head = alpha * sem_norm_head + (1 - alpha) * str_norm_head

        # head_pairs = sorted(
        #     zip([passage_topk_ids[i] for i in head_idx], final_head),
        #     key=lambda x: x[1], reverse=True
        # )
# ------------기존 가중합 끝 -------------
# ---------- 타이브레이커 버전 -------------
        sem_head = sem[head_idx]
        sem_norm_head = scorer.normalize(sem_head)

        # 1) top-1 보호 (semantic margin 크면 고정)
        margin = 0.10  # 0.08~0.15 권장
        keep_top1 = (len(sem_norm_head) > 1) and ((sem_norm_head[0] - sem_norm_head[1]) >= margin)

        # 2) 의미 점수가 충분히 비슷한 구간만 재정렬
        eps = 0.12  # 0.08~0.15 탐색. eps 안쪽만 구조로 재정렬
        groups = []  # 그룹 단위로만 구조 재정렬
        used = np.zeros_like(sem_norm_head, dtype=bool)

        for i in range(len(sem_norm_head)):
            if used[i]: 
                continue
            close = np.where(np.abs(sem_norm_head - sem_norm_head[i]) <= eps)[0]
            close = [j for j in close if not used[j]]
            for j in close: used[j] = True
            groups.append(close)

        # 3) 각 그룹 내부에서만 구조 점수로 정렬 (그룹 간 순서는 semantic 유지)
        str_norm_head = scorer.normalize(struct_head)  # 0~1
        beta = 0.15  # 구조 비중은 작게 시작
        final = sem_norm_head.copy()

        # 그룹별 재정렬
        new_order = []
        offset = 0
        for g in groups:
            # 그룹 밖 순서는 그대로
            if len(g) == 1:
                new_order.append(g[0])
                continue
            # 그룹 내부: 구조 점수로만 정렬(=타이브레이커). 필요시 sem+beta*struct로 더 미세 보정
            g_sorted = sorted(g, key=lambda j: str_norm_head[j], reverse=True)
            new_order.extend(g_sorted)

        # top-1 보호
        if keep_top1:
            # new_order에서 top1의 위치 찾아 맨 앞으로 스왑
            pos = new_order.index(0)
            new_order[0], new_order[pos] = new_order[pos], new_order[0]

        # 최종 head 정렬된 결과 적용
        head_ids_sorted = [passage_topk_ids[head_idx[j]] for j in new_order]
        head_scores_out = [sem_head[j] + beta * (str_norm_head[j] - 0.5) for j in new_order]  # 가벼운 보정

        tail_ids   = [doc_id for j, doc_id in enumerate(passage_topk_ids) if j not in head_idx]
        reranked_doc_ids = head_ids_sorted + tail_ids
        reranked_scores  = head_scores_out + list(sem[[j for j in range(len(sem)) if j not in head_idx]])

        # ---------------------- 진단/출력 블록 ----------------------
        from scipy.stats import spearmanr, kendalltau
        diag = {}

        # 기본 통계
        diag["struct_raw_min"]  = float(np.min(struct_head)) if len(struct_head) else None
        diag["struct_raw_max"]  = float(np.max(struct_head)) if len(struct_head) else None

        str_norm_head = scorer.normalize(struct_head)
        diag["struct_norm_min"] = float(np.min(str_norm_head)) if len(str_norm_head) else None
        diag["struct_norm_max"] = float(np.max(str_norm_head)) if len(str_norm_head) else None
        diag["struct_is_constant"] = bool(len(struct_head) and np.allclose(struct_head, struct_head[0]))
        diag["struct_norm_is_all_zero"] = bool(len(str_norm_head) and np.allclose(str_norm_head, 0.0))

        # 순위 변화 측정: "타이브레이커 적용 전/후"
        head_ids_before  = [passage_topk_ids[i] for i in head_idx]   # semantic 기준 head
        head_ids_after   = head_ids_sorted                           # 타이브레이커 적용 후 head
        diag["head_rank_changed"] = (head_ids_before != head_ids_after)

        if len(head_ids_before) >= 2:
            rank_before = {d:i for i,d in enumerate(head_ids_before)}
            rank_after  = {d:i for i,d in enumerate(head_ids_after)}
            order_b = [rank_before[d] for d in head_ids_before]
            order_a = [rank_after[d]  for d in head_ids_before]
            diag["spearman_rho"], _ = spearmanr(order_b, order_a)
            diag["kendall_tau"], _  = kendalltau(order_b, order_a)
        else:
            diag["spearman_rho"] = None
            diag["kendall_tau"]  = None

        # 로그용 기여도(규모 감지): 타이브레이커 방식이므로 단순 norm 보고
        diag["sem_head_l2"] = float(np.linalg.norm(sem_norm_head)) if len(sem_norm_head) else None
        diag["struct_head_l2"] = float(np.linalg.norm(str_norm_head)) if len(str_norm_head) else None
        diag["beta_used"] = float(beta)
        diag["eps_used"] = float(eps)
        diag["keep_top1"] = bool(keep_top1)

        logger.info(
            "[STRUCT-DIAG/TIEBREAKER][%s] raw=[%.6f, %.6f] norm=[%.6f, %.6f] "
            "const=%s allzero=%s rank_changed=%s rho=%s tau=%s | "
            "||sem||=%.4f ||struct||=%.4f beta=%.3f eps=%.3f keep_top1=%s",
            sim_cac_mode,
            diag["struct_raw_min"] or 0, diag["struct_raw_max"] or 0,
            diag["struct_norm_min"] or 0, diag["struct_norm_max"] or 0,
            diag["struct_is_constant"], diag["struct_norm_is_all_zero"],
            diag["head_rank_changed"], str(diag["spearman_rho"]), str(diag["kendall_tau"]),
            diag["sem_head_l2"] or 0, diag["struct_head_l2"] or 0,
            diag["beta_used"], diag["eps_used"], diag["keep_top1"],
        )

        if save_cb is not None:
            import json, os
            outdir = (save_ctx or {}).get("outdir", ".")
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "struct_influence_diag.json"), "w", encoding="utf-8") as f:
                json.dump(diag, f, ensure_ascii=False, indent=2)
        # ------------------------------------------------------------

        return reranked_doc_ids, reranked_scores        
        # # ---------------------- 진단/출력 블록 ----------------------
        # from scipy.stats import spearmanr, kendalltau
        # diag = {}

        # diag["struct_raw_min"]  = float(np.min(struct_head)) if len(struct_head) else None
        # diag["struct_raw_max"]  = float(np.max(struct_head)) if len(struct_head) else None
        # diag["struct_norm_min"] = float(np.min(str_norm_head)) if len(struct_head) else None
        # diag["struct_norm_max"] = float(np.max(str_norm_head)) if len(struct_head) else None
        # diag["struct_is_constant"] = bool(len(struct_head) and np.allclose(struct_head, struct_head[0]))
        # diag["struct_norm_is_all_zero"] = bool(len(struct_head) and np.allclose(str_norm_head, 0.0))

        # head_ids_before  = [passage_topk_ids[i] for i in head_idx]
        # head_ids_after   = [doc for doc, _ in head_pairs]
        # diag["head_rank_changed"] = (head_ids_before != head_ids_after)

        # if len(head_ids_before) >= 2:
        #     rank_before = {d:i for i,d in enumerate(head_ids_before)}
        #     rank_after  = {d:i for i,d in enumerate(head_ids_after)}
        #     order_b = [rank_before[d] for d in head_ids_before]
        #     order_a = [rank_after[d]  for d in head_ids_before]
        #     diag["spearman_rho"], _ = spearmanr(order_b, order_a)
        #     diag["kendall_tau"], _  = kendalltau(order_b, order_a)
        # else:
        #     diag["spearman_rho"] = None
        #     diag["kendall_tau"]  = None

        # sem_contrib = alpha * sem_norm_head
        # str_contrib = (1.0 - alpha) * str_norm_head
        # diag["sem_contrib_l2"] = float(np.linalg.norm(sem_contrib)) if len(sem_contrib) else None
        # diag["str_contrib_l2"] = float(np.linalg.norm(str_contrib)) if len(str_contrib) else None
        # diag["str_to_sem_contrib_ratio"] = (
        #     float(diag["str_contrib_l2"] / (diag["sem_contrib_l2"] + 1e-12))
        #     if (diag["sem_contrib_l2"] is not None and diag["sem_contrib_l2"] > 0) else None
        # )

        # logger.info(
        #     "[STRUCT-DIAG][%s] raw=[%.6f, %.6f] norm=[%.6f, %.6f] const=%s allzero=%s "
        #     "rank_changed=%s rho=%s tau=%s | ||contrib|| sem=%.4f struct=%.4f ratio=%.4f",
        #     sim_cac_mode,
        #     diag["struct_raw_min"] or 0, diag["struct_raw_max"] or 0,
        #     diag["struct_norm_min"] or 0, diag["struct_norm_max"] or 0,
        #     diag["struct_is_constant"], diag["struct_norm_is_all_zero"],
        #     diag["head_rank_changed"], str(diag["spearman_rho"]), str(diag["kendall_tau"]),
        #     diag["sem_contrib_l2"] or 0, diag["str_contrib_l2"] or 0,
        #     diag["str_to_sem_contrib_ratio"] or 0,
        # )

        # if save_cb is not None:
        #     import json, os
        #     outdir = (save_ctx or {}).get("outdir", ".")
        #     os.makedirs(outdir, exist_ok=True)
        #     with open(os.path.join(outdir, "struct_influence_diag.json"), "w", encoding="utf-8") as f:
        #         json.dump(diag, f, ensure_ascii=False, indent=2)
        # # ------------------------------------------------------------

        # # tail 유지 + 결합
        # tail_ids   = [doc_id for j, doc_id in enumerate(passage_topk_ids) if j not in head_idx]
        # reranked_doc_ids = [doc for doc, _ in head_pairs] + tail_ids
        # reranked_scores  = [sc  for _, sc in head_pairs] + list(sem[[j for j in range(len(sem)) if j not in head_idx]])

        # return reranked_doc_ids, reranked_scores
    