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
    def __init__(self, root_node: str = "question"):
        self.root_node = root_node

    def build(self, triples: list[tuple]) -> ig.Graph:
        """
        (subject, predicate, object) 트리플 → igraph.Graph
        - predicate는 간선 속성으로 저장 (relations: list, relation: 대표 1개)
        - 컴포넌트가 2개 이상일 때만 각 성분의 허브를 root_node와 1개 간선으로 연결
        - 트리플이 1개이거나 컴포넌트가 1개면 root_node는 '존재'만 하고 연결하지 않음
        - root_node에는 is_virtual=True
        """
        def norm(x): return str(x).strip()

        # 0) 입력 검사
        if not triples:
            g = ig.Graph()
            g.add_vertex(name=self.root_node)
            g.vs.find(name=self.root_node)["is_virtual"] = True
            return g

        # 1) 유효 (s,p,o)만 추출 (self-loop 제거)
        valid_triples = []
        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, p, o = map(norm, t)
                if s and p and o and s != o:
                    valid_triples.append((s, p, o))

        if not valid_triples:
            g = ig.Graph()
            g.add_vertex(name=self.root_node)
            g.vs.find(name=self.root_node)["is_virtual"] = True
            return g

        # 2) 정점 집합 (subject/object만 정점, predicate는 간선 속성)
        node_names = set()
        for s, _, o in valid_triples:
            node_names.add(s)
            node_names.add(o)
        node_names.add(self.root_node)  # root는 항상 존재

        # 3) 그래프/정점 추가 (이름을 확실히 설정)
        g = ig.Graph()
        names_sorted = sorted(node_names)
        g.add_vertices(len(names_sorted))
        g.vs["name"] = names_sorted
        g.vs["is_virtual"] = [False] * g.vcount()
        g.vs.find(name=self.root_node)["is_virtual"] = True

        name2id = {n: i for i, n in enumerate(g.vs["name"])}

        # 4) 중복 간선 병합: (u,v) -> [predicates]  (무향 키)
        edge_map = {}
        for s, p, o in valid_triples:
            u, v = name2id[s], name2id[o]
            key = (u, v) if u <= v else (v, u)
            edge_map.setdefault(key, []).append(p)

        if edge_map:
            keys = list(edge_map.keys())
            g.add_edges(keys)
            rel_lists = [edge_map[k] for k in keys]
            g.es["relations"] = rel_lists
            g.es["relation"]  = [rels[0] for rels in rel_lists]
        else:
            g.es["relations"] = []
            g.es["relation"]  = []

        # 5) 컴포넌트 계산 (무향 성분)
        comps = g.components(mode="WEAK")
        num_comps = len(comps)

        # 6) 연결 정책
        should_connect = (num_comps > 1) and (len(valid_triples) > 1)
        if should_connect:
            root_id = name2id[self.root_node]
            for comp in comps:
                # root가 포함된 성분은 스킵
                if root_id in comp:
                    continue
                # 허브: 총 차수 최대
                degs = g.degree(comp, mode="all")
                rep_global = comp[int(np.argmax(degs))]
                # 중복 연결 방지
                if g.get_eid(root_id, rep_global, directed=False, error=False) == -1:
                    g.add_edge(root_id, rep_global)
                    g.es[-1]["relation"]  = "root"
                    g.es[-1]["relations"] = ["root"]

        logger.debug("[QueryGraphBuilder] nodes=%d, edges=%d", g.vcount(), g.ecount())
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
    starter_beam: int = 8    # depth-1 최소 보장

# ------------------------ Subgraph Builder -----------------------

class SubgraphBuilder:
    def __init__(self, g: ig.Graph, cfg: SubgraphConfig):
        self.g = g
        self.cfg = cfg
        self.weighted = ('weight' in g.es.attribute_names())
        if self.weighted:
            self.outstr = np.zeros(g.vcount(), dtype=float)
            for e in g.es:
                self.outstr[e.source] += float(e['weight'])
        else:
            self.outstr = None

    def collect_candidates(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
        """
        유향: OUT만, 무향: ALL. 후보 엣지 (u,v,w) 수집. depth-1 비면 starter_beam 보장.
        """
        visited = {root}
        q = deque([(root, 0)])
        edges: List[Tuple[int,int,float]] = []
        mode = "OUT" if self.g.is_directed() else "ALL"

        while q:
            u, depth = q.popleft()
            if depth >= self.cfg.max_depth:
                continue
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
                edges.append((u, v, w))
                if v not in visited:
                    visited.add(v)
                    q.append((v, depth + 1))

        # depth-1에서 비면 root의 이웃을 weight 상위로 starter_beam 보장
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
            cand.sort(key=lambda x: x[2], reverse=True)
            edges = cand[:self.cfg.starter_beam]

        return edges

    def build_subgraph(self, edges: List[Tuple[int,int,float]], root: Optional[int] = None) -> ig.Graph:
        """후보 엣지로 서브그래프 생성. 비면 루트만 보존. 전역 vid를 global_id로 저장."""
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
        sg.add_edges([(idx[u], idx[v]) for u,v,_ in edges])
        sg.es['weight'] = [w for *_,w in edges]

        sg.simplify(multiple=True, loops=True, combine_edges=None)
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

# ------------------- DEGREE만 사용 시작 ------------------
        # degrees = np.array(g.degree()).reshape(-1, 1)
        # clustering = np.array(g.transitivity_local_undirected(vertices=None)).reshape(-1, 1)
        # clustering[np.isnan(clustering)] = 0
        # pagerank = np.array(g.pagerank()).reshape(-1, 1)
        # for arr in (clustering, pagerank):
        #     arr[~np.isfinite(arr)] = 0.0

        # feats = np.concatenate([degrees], axis = 1) #, clustering, pagerank], axis=1) #


        # mean, std = feats.mean(axis=0), feats.std(axis=0)
        # std = np.maximum(std, 1e-3)
        # z = (feats - mean) / std
        # z = np.clip(z, -3.0, 3.0) * 3.0
        # pad = np.zeros((z.shape[0], self.embed_dim - z.shape[1]), dtype=z.dtype)
        # feats = np.concatenate([z, pad], axis=1)
# ------------- DEGREE만 사용 끝 ------------------


        # deg = np.asarray(g.degree(), dtype=np.float32)
        # logdeg = np.log1p(deg)
        # # 루트 local 인덱스 (없으면 0)
        # root_local = 0
        # try:
        #     gids = np.array(g.vs["global_id"])
        #     rid = int(g["root_id"]) if "root_id" in g.attributes() else gids[0]
        #     idx = np.where(gids == rid)[0]
        #     if len(idx): root_local = int(idx[0])
        # except: pass

        # # depth & hop
        # d = np.asarray(g.shortest_paths(source=root_local)[0], dtype=np.float32)
        # finite = np.isfinite(d); maxd = float(np.max(d[finite])) if finite.any() else 0.0
        # d[~finite] = maxd + 1.0
        # depth_norm = d / (maxd + 1e-6)
        # hop1 = (d == 1).astype(np.float32)
        # hop2 = (d == 2).astype(np.float32)

        # # 로컬 RW (무향 근사)
        # n = g.vcount()
        # A = np.zeros((n,n), dtype=np.float32)
        # for e in g.es:
        #     u,v = e.source, e.target
        #     A[u,v] = 1.0; 
        #     if not g.is_directed(): A[v,u] = 1.0
        # deg_safe = deg.copy(); deg_safe[deg_safe==0] = 1.0
        # A_norm = A / deg_safe[:,None]
        # e_root = np.zeros(n, dtype=np.float32); e_root[root_local]=1.0
        # rw1 = A_norm @ e_root
        # rw2 = A_norm @ rw1
        # rw1 = np.log1p(rw1); rw2 = np.log1p(rw2)

        # leaf = (deg == 1).astype(np.float32)

        # # 최종 노드 피처 (7D)
        # feats = np.stack([logdeg, depth_norm, hop1, hop2, rw1, rw2, leaf], axis=1)
        
        deg = np.asarray(g.degree(), dtype=np.float32).reshape(-1, 1)
        x = np.log1p(deg)  # 안정화

        # 방법 A: max-정규화(그래프-내이지만 중심化가 아님)
        x /= (np.max(x, axis=0, keepdims=True) + 1e-6)

        # 방법 B(대안): 노드별 L2 정규화
        # x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)

        pad = np.zeros((x.shape[0], self.embed_dim - x.shape[1]), dtype=np.float32)
        feats = np.concatenate([x, pad], axis=1)
        return torch.tensor(feats, dtype=torch.float32, device=self.device)

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

        x_lorentz = self.manifold.expmap0(x)
        x_lorentz = torch.nan_to_num(x_lorentz, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = x_lorentz.mean(dim=0)
        return pooled

    # # ------------------ 개선된 풀링: 허브+깊이 가중 평균 -------------------
    # def embed_lorentz(self, g: ig.Graph) -> torch.Tensor: 
    #     if g.vcount() == 0:
    #         return None
    #     # (1) 기존 텐서 그대로 사용: degree 기반
    #     if g.vcount() == 1 and g.ecount() == 0:
    #         x = torch.zeros((1, self.embed_dim), device=self.device)
    #         x[0, 0] = 1e-2
    #     else:
    #         x = self.graph_to_tensor(g)  # ← graph_to_tensor는 건드리지 않음

    #     # 안정 클립
    #     with torch.no_grad():
    #         max_norm = 5.0
    #         n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
    #         x = x * torch.clamp(max_norm / n, max=1.0)

    #     # (2) 하이퍼볼릭으로 사상
    #     x_L = self.manifold.expmap0(x)
    #     x_L = torch.nan_to_num(x_L, nan=0.0, posinf=0.0, neginf=0.0)

    #     # (3) 루트 local 인덱스 추정
    #     root_local = 0
    #     try:
    #         rid = g["root_id"]  # 있으면 최고
    #         gids = np.array(g.vs["global_id"])
    #         idx = np.where(gids == int(rid))[0]
    #         if len(idx): root_local = int(idx[0])
    #     except Exception:
    #         # 폴백: 허브(최대 차수) 노드를 루트로 간주
    #         degs = np.asarray(g.degree(), dtype=np.float32)
    #         if degs.size:
    #             root_local = int(np.argmax(degs))

    #     # # (4) depth_norm (루트 기준 최단거리 정규화)
    #     # d = np.asarray(g.shortest_paths(source=root_local)[0], dtype=np.float32)
    #     # finite = np.isfinite(d)
    #     # maxd = float(np.max(d[finite])) if finite.any() else 0.0
    #     # d[~finite] = maxd + 1.0
    #     # depth_norm = torch.tensor(d / (maxd + 1e-6), dtype=torch.float32, device=self.device)

    #     # # (5) 허브 보정 + 깊이 가중치
    #     # deg = torch.tensor(g.degree(), dtype=torch.float32, device=self.device)
    #     # max_deg = torch.clamp(deg.max(), min=1.0)
    #     # logdeg_norm = torch.log1p(deg) / torch.log1p(max_deg)
    #     # tau = 1.5
    #     # w = torch.exp(- depth_norm / tau) * (1.0 + logdeg_norm)  # [N]
    #     # w = (w + 1e-6).unsqueeze(1)                              # [N,1]

    #     # # (6) 가중 하이퍼볼릭 평균(간단한 가중 합)
    #     # pooled = (w * x_L).sum(dim=0) / w.sum()
    # #------------------------------------------------
    #     # embed_lorentz 내 가중만 교체   ------------> 깊이 가중만
    #     # # 1) depth (undirected로 계산)
    #     # g_ud = g.as_undirected() if g.is_directed() else g
    #     # d = np.asarray(g_ud.shortest_paths(source=root_local)[0], dtype=np.float32)
    #     # finite = np.isfinite(d); maxd = float(np.max(d[finite])) if finite.any() else 0.0
    #     # d[~finite] = maxd + 1.0
    #     # depth_norm = torch.tensor(d / (maxd + 1e-6), dtype=torch.float32, device=self.device)

    #     # # 2) degree 요인 제거, depth만 사용 + 완만한 τ
    #     # tau = 2.5  # 2.5~4.0 탐색
    #     # w = torch.exp(- depth_norm / tau)   # [N]

    #     # # 3) 안정화: 정규화+클램프
    #     # w = w / (w.mean() + 1e-6)
    #     # w = torch.clamp(w, 0.5, 2.0).unsqueeze(1)

    #     # # 4) (로런츠 임베딩 x_L는 기존과 동일)
    #     # pooled = (w * x_L).sum(dim=0) / w.sum()

    #     return pooled





    def logmap0_safe(self, lorentz_vec: torch.Tensor) -> torch.Tensor:
        t = self.manifold.logmap0(lorentz_vec)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def embed_and_logmap0(self, g: ig.Graph) -> torch.Tensor:
        y = self.embed_lorentz(g)
        if y is None:
            # 빈 그래프면 0 텐서 반환
            return torch.zeros(self.embed_dim, device=self.device)
        return self.logmap0_safe(y)

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

    def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        mn, mx = scores.min(initial=0.0), scores.max(initial=0.0)
        if mx - mn < 1e-8:
            return np.zeros_like(scores, dtype=float)
        return (scores - mn) / (mx - mn + 1e-8)

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
        builder = SubgraphBuilder(self.g, cfg)
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
    def _struct_scores_tangent(
        self,
        query_graph: ig.Graph,
        subgraphs: List[ig.Graph],
        embedder: "HyperbolicEmbedder",
        scorer: "StructralSimilarity",
    ) -> np.ndarray:
        """하이퍼볼릭→탄젠트(logmap0) 투영 + 공통 표준화 → listwise 유사도 (큰값=유사)"""
        with torch.no_grad():
            tq    = embedder.embed_and_logmap0(query_graph)
            tdocs = [embedder.embed_and_logmap0(g) for g in subgraphs]

        tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
        q_list = [tq_std for _ in range(len(tdocs_std))]

        struct_head = scorer.compute_listwise(q_list, tdocs_std)
        return _to_1d_numpy(struct_head)  # 1D numpy, NaN/Inf 제거

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
        sem_norm_head = scorer.normalize(sem[head_idx])
        str_norm_head = scorer.normalize(struct_head)

        final_head = alpha * sem_norm_head + (1 - alpha) * str_norm_head

        head_pairs = sorted(
            zip([passage_topk_ids[i] for i in head_idx], final_head),
            key=lambda x: x[1], reverse=True
        )

        # ---------------------- 진단/출력 블록 ----------------------
        from scipy.stats import spearmanr, kendalltau
        diag = {}

        diag["struct_raw_min"]  = float(np.min(struct_head)) if len(struct_head) else None
        diag["struct_raw_max"]  = float(np.max(struct_head)) if len(struct_head) else None
        diag["struct_norm_min"] = float(np.min(str_norm_head)) if len(struct_head) else None
        diag["struct_norm_max"] = float(np.max(str_norm_head)) if len(struct_head) else None
        diag["struct_is_constant"] = bool(len(struct_head) and np.allclose(struct_head, struct_head[0]))
        diag["struct_norm_is_all_zero"] = bool(len(struct_head) and np.allclose(str_norm_head, 0.0))

        head_ids_before  = [passage_topk_ids[i] for i in head_idx]
        head_ids_after   = [doc for doc, _ in head_pairs]
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

        sem_contrib = alpha * sem_norm_head
        str_contrib = (1.0 - alpha) * str_norm_head
        diag["sem_contrib_l2"] = float(np.linalg.norm(sem_contrib)) if len(sem_contrib) else None
        diag["str_contrib_l2"] = float(np.linalg.norm(str_contrib)) if len(str_contrib) else None
        diag["str_to_sem_contrib_ratio"] = (
            float(diag["str_contrib_l2"] / (diag["sem_contrib_l2"] + 1e-12))
            if (diag["sem_contrib_l2"] is not None and diag["sem_contrib_l2"] > 0) else None
        )

        logger.info(
            "[STRUCT-DIAG][%s] raw=[%.6f, %.6f] norm=[%.6f, %.6f] const=%s allzero=%s "
            "rank_changed=%s rho=%s tau=%s | ||contrib|| sem=%.4f struct=%.4f ratio=%.4f",
            sim_cac_mode,
            diag["struct_raw_min"] or 0, diag["struct_raw_max"] or 0,
            diag["struct_norm_min"] or 0, diag["struct_norm_max"] or 0,
            diag["struct_is_constant"], diag["struct_norm_is_all_zero"],
            diag["head_rank_changed"], str(diag["spearman_rho"]), str(diag["kendall_tau"]),
            diag["sem_contrib_l2"] or 0, diag["str_contrib_l2"] or 0,
            diag["str_to_sem_contrib_ratio"] or 0,
        )

        if save_cb is not None:
            import json, os
            outdir = (save_ctx or {}).get("outdir", ".")
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "struct_influence_diag.json"), "w", encoding="utf-8") as f:
                json.dump(diag, f, ensure_ascii=False, indent=2)
        # ------------------------------------------------------------

        # tail 유지 + 결합
        tail_ids   = [doc_id for j, doc_id in enumerate(passage_topk_ids) if j not in head_idx]
        reranked_doc_ids = [doc for doc, _ in head_pairs] + tail_ids
        reranked_scores  = [sc  for _, sc in head_pairs] + list(sem[[j for j in range(len(sem)) if j not in head_idx]])

        return reranked_doc_ids, reranked_scores
