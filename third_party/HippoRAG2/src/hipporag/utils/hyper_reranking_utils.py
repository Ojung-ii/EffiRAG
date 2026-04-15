import igraph as ig
import numpy as np
import geoopt
import torch
import torch.nn.functional as F
from collections import deque


# class TreeStructureSubgraph:  
#     def __init__(self, graph, start_idx=None, max_depth=2, weight_threshold=0.5):
#         """
#         Initialize the tree-based subgraph extractor.

#         Args:
#             graph (igraph.Graph): The input full graph.
#             start_idx (int, optional): Starting node index for BFS (used if extract_tree is called without a root_id).
#             max_depth (int): Maximum BFS traversal depth from the root node.
#             weight_threshold (float): Minimum edge weight to include in the subgraph.
#         """
#         self.graph = graph
#         self.start_idx = start_idx
#         self.max_depth = max_depth
#         self.weight_threshold = weight_threshold

#     def total_subgraph(self, topk_root_ids):
#         """
#         Extracts a list of tree-structured subgraphs from top-k root nodes.

#         Args:
#             topk_root_ids (List[int]): List of node IDs to use as roots.

#         Returns:
#             List[igraph.Graph]: A list of directed subgraphs rooted at given nodes.
#         """
#         return [self.extract_tree(root_id) for root_id in topk_root_ids]

#     def extract_tree(self, root_id=None):
#         """
#         Extracts a single tree-structured subgraph rooted at a given node.

#         Args:
#             root_id (int, optional): Node ID to start BFS from. Defaults to self.start_idx.

#         Returns:
#             igraph.Graph: Extracted subgraph as a directed igraph graph.
#         """
#         if root_id is None:
#             root_id = self.start_idx
#         edges = self._extract_weighted_bfs_tree(root_id)
#         if not edges:   # Debugging: Check if edges are empty
#             print(f"[TreeStructureSubgraph] No edges found for root {root_id}. Returning empty graph.")
#         return self._build_subgraph(edges)

#     def _extract_weighted_bfs_tree(self, start_idx):
#         """
#         Perform weighted BFS traversal to collect edges satisfying depth and weight constraints.

#         Args:
#             start_idx (int): The starting node index.

#         Returns:
#             List[Tuple[int, int, float]]: A list of (source, target, weight) edges.
#         """
#         visited = set()
#         edges = []
#         queue = deque([(start_idx, 0)])
#         visited.add(start_idx)

#         while queue:
#             current, depth = queue.popleft()
#             if depth >= self.max_depth:
#                 continue
#             for eid in self.graph.incident(current):
#                 edge = self.graph.es[eid]
#                 neighbor = edge.target if edge.source == current else edge.source
#                 weight = edge['weight']
#                 if weight >= self.weight_threshold and neighbor not in visited:
#                     visited.add(neighbor)
#                     edges.append((current, neighbor, weight))
#                     queue.append((neighbor, depth + 1))
#         return edges

#     def _build_subgraph(self, edges):
#         """
#         Construct a directed igraph graph from a list of edges.

#         Args:
#             edges (List[Tuple[int, int, float]]): List of (source, target, weight) edges.

#         Returns:
#             igraph.Graph: Constructed subgraph.
#         """
#         subg = ig.Graph(directed=True)
#         node_set = set()
#         for s, t, _ in edges:
#             node_set.add(s)
#             node_set.add(t)

#         if len(node_set) == 0:
#             subg.add_vertices(1)
#             subg.vs["name"] = [0]
#             subg.es["weight"] = []
#             return subg
        
#         node_list = list(node_set)
#         node_map = {n: i for i, n in enumerate(node_list)}
#         subg.add_vertices(len(node_list))
#         subg.vs["name"] = node_list
#         subg.add_edges([(node_map[s], node_map[t]) for s, t, _ in edges])
#         subg.es["weight"] = [w for _, _, w in edges]
#         # print(f"[TreeStructureSubgraph] Subgraph built with {len(node_list)} nodes and {len(edges)} edges.") # Debugging
#         return subg
import igraph as ig
from collections import deque
import numpy as np
from dataclasses import dataclass


class QueryGraphBuilder:
    def __init__(self, root_node="question"):
        self.root_node = root_node

    def build(self, triples: list[tuple]) -> ig.Graph:
        """
        Converts a list of (subject, predicate, object) triples into an igraph.Graph.
        - predicate는 간선 속성으로 저장
        - 컴포넌트가 2개 이상일 때만 각 컴포넌트 대표 허브를 root_node와 연결
        - 트리플이 1개이거나 컴포넌트가 1개면 root_node는 '존재'만 하고 연결하지 않음
        - root_node에는 is_virtual=True 설정(후처리 시 마스킹 용이)
        """
        def norm(x):
            return str(x).strip()

        # 0) 입력 검사
        if not triples:
            g = ig.Graph()
            g.add_vertex(root_node)
            g.vs.find(name=root_node)["is_virtual"] = True
            # logger.debug("[TriplesToQueryGraph] Created graph with only virtual root (no edges).")
            return g

        # 1) 유효 (s,p,o)만 추출 (self-loop 제거)
        valid_triples = []
        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, p, o = map(norm, t)
                if s and p and o and s != o:
                    valid_triples.append((s, p, o))

        if not valid_triples:
            # logger.warning("[TriplesToQueryGraph] No valid triples after filtering.")
            g = ig.Graph()
            g.add_vertex(root_node)
            g.vs.find(name=root_node)["is_virtual"] = True
            return g

        # 2) 정점 집합 (subject/object만 정점, predicate는 간선 속성)
        node_names = set()
        for s, _, o in valid_triples:
            node_names.add(s)
            node_names.add(o)

        # root는 항상 '존재'하게
        node_names.add(root_node)

        # 3) 그래프/정점 추가
        g = ig.Graph()
        g.add_vertices(sorted(node_names))   # g.vs['name'] 생성
        # is_virtual 기본 False, root만 True
        g.vs["is_virtual"] = [False] * g.vcount()
        g.vs.find(name=root_node)["is_virtual"] = True

        name2id = {n: i for i, n in enumerate(g.vs["name"])}

        # 4) 중복 간선 병합: (u,v) -> [predicates]
        #    무향으로 가정(원 코드와 동일한 기본 ig.Graph())
        edge_map = {}
        for s, p, o in valid_triples:
            u, v = name2id[s], name2id[o]
            key = tuple(sorted((u, v)))  # 무향 키
            edge_map.setdefault(key, []).append(p)

        if edge_map:
            keys = list(edge_map.keys())
            g.add_edges(keys)
            rels = [edge_map[k] for k in keys]
            g.es["relations"] = rels               # 리스트(여러 predicate 누적)
            g.es["relation"]  = [r[0] for r in rels]  # 대표 1개(첫 번째)
        else:
            g.es["relations"] = []
            g.es["relation"]  = []

        # 5) 컴포넌트 계산
        comps = g.components(mode="WEAK")  # 무향 성분
        num_comps = len(comps)

        # 6) 연결 정책:
        # - 컴포넌트가 2개 이상일 때만 각 성분의 '허브'를 root와 1개 간선으로 연결
        # - 트리플이 1개인 경우에도 연결하지 않음(구조 왜곡 방지)
        should_connect = (num_comps > 1) and (len(valid_triples) > 1)

        if should_connect:
            root_id = name2id[root_node]
            for comp in comps:
                comp_names = [g.vs[idx]["name"] for idx in comp]
                if root_node in comp_names:
                    continue  # 이미 root가 포함된 성분이면 skip

                # 허브 선택: 총 차수(degree) 최대
                degs = g.degree(comp, mode="all")
                rep_local = int(np.argmax(degs))
                rep_global = comp[rep_local]

                # 중복 연결 방지
                if g.get_eid(root_id, rep_global, directed=False, error=False) == -1:
                    g.add_edge(root_id, rep_global)
                    # 새로 추가된 간선 속성 부여
                    g.es[-1]["relation"]  = "root"
                    g.es[-1]["relations"] = ["root"]

        # 7) 디버그 로그
        logger.debug("[TriplesToQueryGraph] Final query graph summary:")
        logger.debug(f"- Nodes ({g.vcount()}): {[v['name'] for v in g.vs]}")
        logger.debug(f"- is_virtual: {g.vs['is_virtual']}")
        logger.debug(f"- Edges ({g.ecount()}): {g.get_edgelist()}")
        logger.debug(f"- Relations (repr): {[e for e in g.es['relation']]}")
        return g






# ----------------------------------------------------------
@dataclass
class SubgraphConfig:
    max_depth: int = 3
    weight_threshold: float = 0.0
    allow_topk: int = 200
    flow_quantile: float = 0.50
    per_node_top_r: int = 3
    ensure_connectivity: bool = True

class SubgraphBuilder:
    def __init__(self, g: ig.Graph, cfg: SubgraphConfig):
        self.g = g
        self.cfg = cfg
        self.weighted = ('weight' in g.es.attribute_names())
        self.outdeg = np.array(g.outdegree() if g.is_directed() else g.degree(), dtype=float)
        if self.weighted:
            self.outstr = np.zeros(g.vcount(), dtype=float)
            for e in g.es:
                self.outstr[e.source] += float(e['weight'])
        else:
            self.outstr = None

    def collect_candidates(self, root: int, allow_ids: set[int] | None):
        """BFS로 후보 엣지 수집만 수행: [(u,v,w)]"""

        visited = set([root])
        q = deque([(root, 0)])
        edges = []

        while q:
            u, depth = q.popleft()
            if depth >= self.cfg.max_depth: 
                continue
            mode = "OUT" if self.g.is_directed() else "ALL"
            for eid in self.g.incident(u, mode=mode):
                e = self.g.es[eid]
                a, b = e.source, e.target
                v = b if a == u else a
                w = float(e['weight']) if self.weighted else 1.0
                if self.g.is_directed() and e.source != u:
                     continue
                if w < self.cfg.weight_threshold: 
                    continue
                if allow_ids is not None and v not in allow_ids:
                    continue
                edges.append((u, v, w))  # 후보만 모음
                if v not in visited:
                    visited.add(v)
                    q.append((v, depth + 1))
        return edges

    def build_subgraph(self, edges: list[tuple[int,int,float]]) -> ig.Graph:
        """모은 엣지로 서브그래프 생성"""
        if not edges:
            g = ig.Graph(directed=True); g.add_vertices(1); g.vs['name']=[0]; return g
        nodes = sorted({u for u,_,_ in edges} | {v for _,v,_ in edges})
        idx = {n:i for i,n in enumerate(nodes)}
        sg = ig.Graph(directed=True)
        sg.add_vertices(len(nodes)); sg.vs['name'] = nodes
        sg.add_edges([(idx[u], idx[v]) for u,v,_ in edges])
        sg.es['weight'] = [w for *_,w in edges]
        return sg


    
    import os

    def save_graph(g: ig.Graph, filename: str, out_dir: str = "graph_debug"):
        """
        Save igraph.Graph to GraphML file for visualization.
        """
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        try:
            g.write_graphml(path)
            print(f"[SAVE] {filename} saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save {filename}: {e}")

class PPRPruner:
    def __init__(self, g: ig.Graph, pi: np.ndarray, cfg: SubgraphConfig):
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
            w = float(self.g.es[eid]['weight'])
            denom = self.outstr[u]
            return (w/denom) if denom>0 else 0.0
        else:
            d = self.outdeg[u]
            return (1.0/d) if d>0 else 0.0

    def _flow(self, u: int, v: int) -> float:
        if self.pi is None: return 0.0
        # eid 찾기(방향 우선, 없으면 무향)
        eid = self.g.get_eid(u, v, directed=True, error=False)
        if eid == -1:
            eid = self.g.get_eid(v, u, directed=False, error=False)
        if eid == -1: return 0.0
        return float(self.pi[u]) * self._P(u, eid)

    def prune(self, root: int, candidate_edges: list[tuple[int,int,float]]):
        """후처리: 유량 분위수 컷 → 노드별 Top-r"""
        if not candidate_edges or self.pi is None:
            return candidate_edges

        # 1) 전역 분위수 컷
        flows = np.array([self._flow(u,v) for u,v,_ in candidate_edges], dtype=float)
        if self.cfg.flow_quantile is not None:
            tau = float(np.quantile(flows, self.cfg.flow_quantile))
            kept = [(e,f) for e,f in zip(candidate_edges, flows) if f >= tau]
        else:
            kept = list(zip(candidate_edges, flows))

        if not kept: 
            return []

        # 2) 노드별 Top-r
        r = self.cfg.per_node_top_r
        if r is None or r <= 0: 
            return [e for e,_ in kept]

        by_src = {}
        for (u,v,w), f in kept:
            by_src.setdefault(u, []).append(((u,v,w), f))
        pruned = []
        for u, lst in by_src.items():
            lst.sort(key=lambda x: x[1], reverse=True)
            pruned.extend([e for e,_ in lst[:r]])
        return pruned


class HyperbolicEmbedder:
    def __init__(self, embed_dim=32, curvature=1.0, device="cuda"):
        """
        Initialize the Lorentz space embedder.

        Args:
            embed_dim (int): Embedding dimension in tangent space (final vector will be (D+1) in Lorentz space).
            curvature (float): Curvature parameter for the Lorentz manifold.
            device (str): Device to perform computation on ("cpu" or "cuda").
        """
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.embed_dim = embed_dim
        self.device = device

    def graph_to_tensor(self, g: ig.Graph):
        """
        Convert a graph into input tensor features for embedding.

        Args:
            g (igraph.Graph): Input graph.

        Returns:
            torch.Tensor: Node features of shape (num_nodes, embed_dim), padded with zeros except degree.
        """
        if g.vcount() == 0:
            facts = np.zeros((1,self.embed_dim), dtype=np.float32)
            return torch.tensor(facts, device = self.device)

        degrees = np.array(g.degree()).reshape(-1, 1)
        # pad = np.zeros((degrees.shape[0], self.embed_dim - 1))
        # feats = np.concatenate([degrees, pad], axis=1)


        # 제안: degree + clustering coefficient
        clustering = np.array(g.transitivity_local_undirected(vertices=None)).reshape(-1, 1)
        clustering[np.isnan(clustering)] = 0
        pagerank = np.array(g.pagerank()).reshape(-1, 1)
        # eigen = np.array(g.eigenvector_centrality()).reshape(-1, 1)

        for arr in (clustering, pagerank): #, eigen):
            arr[~np.isfinite(arr)] = 0.0

        # concatenate
        feats = np.concatenate([degrees, clustering, pagerank], axis=1) # , eigen], axis=1)

        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std = np.maximum(std, 1e-3)         # 바닥 크게
        z = (feats - mean) / std
        z = np.clip(z, -3.0, 3.0)          # 클리핑
        scale = 3.0                      # <<< 실험: 3.0 ~ 5.0
        z *= scale          
        pad = np.zeros((z.shape[0], self.embed_dim - z.shape[1]))
        feats = np.concatenate([z, pad], axis=1)

        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    def embed_lorentz(self, g: ig.Graph) -> torch.Tensor:
        """
        Embed a graph into Lorentz hyperbolic space.

        Args:
            g (igraph.Graph): Input graph.

        Returns:
            torch.Tensor: Graph embedding in Lorentz space of shape (D+1,)
        """
        if g.vcount() == 0:
        # None으로 돌려서 상위 로직에서 구조 스킵
            return None
        if g.vcount() == 1 and g.ecount() == 0:
            # 작은 epsilon 임베딩 (완전 0은 피하기)
            x = torch.zeros((1, self.embed_dim), device=self.device)
            x[0,0] = 1e-2
        else:
            x = self.graph_to_tensor(g)


        # print("[DEBUG] input degrees:", x[:, 0].cpu().numpy())
        # --- ADD: tangent norm clamp ---
        with torch.no_grad():
            max_norm = 5.0
            n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
            scale = torch.clamp(max_norm / n, max=1.0)
            x = x * scale

        x_lorentz = self.manifold.expmap0(x)
        if not torch.isfinite(x_lorentz).all():
            print("[HyperbolicEmbedder] non-finite after expmap -> nan_to_num(0).")
            x_lorentz = torch.nan_to_num(x_lorentz, nan=0.0, posinf=0.0, neginf=0.0)

        # 디버그: norm(mean)
        try:
            ln = self.manifold.norm(x_lorentz).mean()
            print("[DEBUG] lorentz norm mean:", float(torch.nan_to_num(ln, nan=0.0)))
        except Exception:
            pass

        # 간단 풀링(평균). 필요시 로렌츠 평균으로 교체 가능.
        pooled = x_lorentz.mean(dim=0)
        return pooled

        # === T0 베이스라인을 위한 헬퍼들 ===
    def logmap0_safe(self, lorentz_vec: torch.Tensor) -> torch.Tensor:
        t = self.manifold.logmap0(lorentz_vec)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    def embed_and_logmap0(self, g: ig.Graph) -> torch.Tensor:
        """
        그래프 -> 하이퍼볼릭(원점) -> T0 탄젠트로 바로 투영.
        """
        y = self.embed_lorentz(g)
        t0 = self.logmap0_safe(y)
        return t0

    # def logmap0_batch(self, lorentz_list: list[torch.Tensor]) -> list[torch.Tensor]:
    #     """
    #     Project multiple Lorentz embeddings to tangent space at origin.

    #     Args:
    #         lorentz_list (List[torch.Tensor]): List of (D+1,) Lorentz vectors.

    #     Returns:
    #         List[torch.Tensor]: List of (D,) tangent space vectors.
    #     """
    #     return [self.manifold.logmap0(x) for x in lorentz_list]

    # def transport_query_to_all(self, query_lorentz: torch.Tensor, subgraph_lorentz_list: list[torch.Tensor]) -> list[torch.Tensor]:
    #     """
    #     Parallel transport query vector to each subgraph’s tangent space.

    #     Args:
    #         query_lorentz (torch.Tensor): Query embedding in Lorentz space (D+1,)
    #         subgraph_lorentz_list (List[torch.Tensor]): List of (D+1,) subgraph embeddings.

    #     Returns:
    #         List[torch.Tensor]: List of (D,) transported query vectors in each subgraph's tangent space.
    #     """
    #     q_logmap0 = self.manifold.logmap0(query_lorentz)
    #     q_logmap0 = torch.nan_to_num(q_logmap0, nan=0.0, posinf=0.0, neginf=0.0)

    #     print("[DEBUG] query_logmap0 norm:", q_logmap0.norm().item())
    #     result = []
    #     for i, d in enumerate(subgraph_lorentz_list):
    #         d_vec = self.manifold.transp0(d, q_logmap0)
    #         d_vec = torch.nan_to_num(d_vec, nan=0.0, posinf=0.0, neginf=0.0)

    #         print(f"[DEBUG] doc[{i}] transp0 norm:", d_vec.norm().item())
    #         result.append(d_vec)
    #     return result


    def shared_standardize(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        여러 텐서를 하나로 묶어서 μ,σ를 구하고, 각 텐서를 동일 기준으로 z-score 표준화.
        NaN/Inf 방어 포함.
        """
        if not tensors:
            return tensors
        
        # 텐서 -> 하나로 연결
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
        """
        Initialize the structural similarity function.

        Args:
            mode (str): Similarity metric, one of {"cosine", "l2", "dot"}.
        """
        assert mode in ["cosine", "l2", "dot"], "Invalid similarity mode"
        self.mode = mode

    def normalize(self, scores: list[float]) -> np.ndarray:
        scores = np.array(scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    def compute(self, query_tangent: torch.Tensor, subgraph_tangent: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two vectors in tangent space.

        Args:
            query_tangent (torch.Tensor): Query vector in tangent space (D,)
            subgraph_tangent (torch.Tensor): Document/subgraph vector in same tangent space (D,)

        Returns:
            torch.Tensor: Scalar similarity score.
        """
        if self.mode == "cosine":
            return F.cosine_similarity(query_tangent, subgraph_tangent, dim=-1)
        elif self.mode == "dot":
            return torch.sum(query_tangent * subgraph_tangent, dim=-1)
        elif self.mode == "l2":
            return -torch.norm(query_tangent - subgraph_tangent, dim=-1)

    def compute_listwise(self, query_list: list[torch.Tensor], doc_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute pairwise similarities for a list of query and document vectors.

        Args:
            query_list (List[torch.Tensor]): List of query vectors (D,)
            doc_list (List[torch.Tensor]): List of document vectors (D,)

        Returns:
            torch.Tensor: Similarity scores of shape (len(query_list),)
        """
        assert len(query_list) == len(doc_list), "Mismatch in list length"
        sims = []
        for i, (q, d) in enumerate(zip(query_list, doc_list)):
            print(f"[DEBUG] q norm: {q.norm().item()}, d norm: {d.norm().item()}")

            sim = self.compute(q.unsqueeze(0), d.unsqueeze(0)).squeeze()
            sims.append(sim)
            # print(f"[StructralSimilarity] sim[{i}]: {sim.item():.4f}")

        return torch.stack(sims)


