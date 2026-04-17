# from __future__ import annotations
# import igraph as ig
# import numpy as np
# import torch
# import torch.nn.functional as F
# from dataclasses import dataclass, field
# from typing import Iterable, List, Optional, Tuple, Set, Callable, Literal, Dict, Any
# from collections import deque, defaultdict
# import logging
# import os
# import pandas as pd
# from scipy.stats import spearmanr, kendalltau

# # TreeRep
# from .TreeRep import TreeRep
# import networkx as nx
# from sklearn.manifold import MDS
# import geoopt

# logger = logging.getLogger(__name__)

# import json, math
# from typing import Mapping

# class MetricsLogger:
#     def __init__(self, save_dir: str, dataset_name: str, fname: str = "struct_metrics.jsonl"):
#         os.makedirs(save_dir, exist_ok=True)
#         self.path = os.path.join(save_dir, f"{dataset_name}_{fname}")

#     def write(self, row: Mapping[str, Any]):
#         with open(self.path, "a", encoding="utf-8") as f:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")

# def _safe_deg_hist(g: ig.Graph, exclude_virtual=False) -> dict:
#     if g.vcount() == 0:
#         return {}
#     mask = [True] * g.vcount()
#     if exclude_virtual and "is_virtual" in g.vs.attributes():
#         mask = [not bool(v) for v in g.vs["is_virtual"]]
#     vids = [i for i, ok in enumerate(mask) if ok]
#     if not vids:
#         return {}
#     degs = g.degree(vids, mode="all")
#     hist = defaultdict(int)
#     for d in degs:
#         hist[int(d)] += 1
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
#         "dataset": dataset_name,
#         "query_id": str(query_id),
#         "graph_role": graph_role,
#         "v": int(g2.vcount()),
#         "e": int(g2.ecount()),
#         "directed": bool(g.is_directed()),
#     }

#     if g2.vcount() > 0:
#         comps = g2.components(mode="WEAK")
#         comp_sizes = [len(c) for c in comps]
#         m["comp_cnt"] = int(len(comp_sizes))
#         m["giant_ratio"] = float((max(comp_sizes) / g2.vcount()) if comp_sizes else 0.0)
#     else:
#         m["comp_cnt"] = 0
#         m["giant_ratio"] = 0.0

#     hist = _safe_deg_hist(g2, exclude_virtual=False)
#     m["deg_hist"] = hist
#     if hist:
#         deg_list = []
#         for k, v in hist.items():
#             deg_list.extend([k] * v)
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

#     try:
#         m["star_ratio"] = 0.0
#     except Exception:
#         m["star_ratio"] = 0.0

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
#                 else:
#                     m["path_len_mean"] = 0.0
#                     m["path_len_p95"] = 0.0
#             else:
#                 m["path_len_mean"] = 0.0
#                 m["path_len_p95"] = 0.0
#         else:
#             m["path_len_mean"] = 0.0
#             m["path_len_p95"] = 0.0
#     except Exception:
#         m["path_len_mean"] = 0.0
#         m["path_len_p95"] = 0.0

#     try:
#         bet = np.asarray(g2.betweenness(), dtype=float) if g2.vcount() else np.array([])
#         if bet.size:
#             m["betw_p90"] = float(np.percentile(bet, 90))
#             m["betw_frac_gt_p90"] = float(np.mean(bet > m["betw_p90"]))
#             m["betw_gini"] = _gini(bet.tolist())
#         else:
#             m["betw_p90"] = 0.0
#             m["betw_frac_gt_p90"] = 0.0
#             m["betw_gini"] = 0.0
#     except Exception:
#         m["betw_p90"] = 0.0
#         m["betw_frac_gt_p90"] = 0.0
#         m["betw_gini"] = 0.0

#     try:
#         clu = g2.transitivity_avglocal_undirected()
#         m["clustering_avg"] = float(0.0 if (clu is None or math.isnan(clu)) else clu)
#     except Exception:
#         m["clustering_avg"] = 0.0

#     if ppr_scores is not None and "global_id" in g2.vs.attributes():
#         gids = np.asarray(g2.vs["global_id"])
#         pv = np.nan_to_num(np.asarray(ppr_scores, dtype=float), nan=0.0)
#         sel = pv[gids] if gids.size else np.array([])
#         if sel.size and np.sum(sel) > 0:
#             p = sel / np.sum(sel)
#             m["ppr_entropy"] = _entropy(p)
#         else:
#             m["ppr_entropy"] = 0.0
#     else:
#         m["ppr_entropy"] = 0.0

#     m["long_tail_hint"] = {"deg_gini": m["deg_gini"], "tail_frac_ge3": m["tail_frac_ge3"]}
#     return m

# def _to_1d_numpy(x) -> np.ndarray:
#     if isinstance(x, torch.Tensor):
#         x = x.detach().flatten()
#         x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
#         return x
#     arr = np.asarray(x, dtype=float).reshape(-1)
#     return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# class QueryGraphBuilder:
#     def __init__(self, root_node: str = "question", directed: bool = True,
#                  save_dir: str = f"./outputs/query_graph_analysis", dataset_name: str = "default"):
#         self.root_node = root_node
#         self.directed = directed
#         self.save_dir = save_dir
#         self.dataset_name = dataset_name
#         os.makedirs(self.save_dir, exist_ok=True)
#         self.metrics_logger = MetricsLogger(save_dir=self.save_dir, dataset_name=self.dataset_name)
#         self.log_file = os.path.join(
#             self.save_dir, f"{self.dataset_name}_query_graph_degree_distribution.csv"
#         )
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
#             with open(self.log_file, 'a') as f:
#                 f.write(f"{query_id},{self.dataset_name},0,0,[]\n")
#             return g

#         valid_triples = []
#         for t in triples:
#             if isinstance(t, (list, tuple)) and len(t) == 3:
#                 s, p, o = map(norm, t)
#                 if s and p and o and s != o:
#                     valid_triples.append((s, p, o))

#         g = ig.Graph(directed=self.directed)
#         if not valid_triples:
#             g.add_vertex(name=self.root_node, is_virtual=True)
#             g.vs["global_id"] = [-1]
#             g["root_id"] = g.vs.find(name=self.root_node).index
#             with open(self.log_file, 'a') as f:
#                 f.write(f"{query_id},{self.dataset_name},1,0,[]\n")
#             return g

#         node_names = set()
#         for s, _, o in valid_triples:
#             node_names.add(s); node_names.add(o)
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
#             if root_id in comp:
#                 continue
#             degs = g.degree(comp, mode="all")
#             hub_local = int(np.argmax(degs))
#             hub_vid = comp[hub_local]
#             if g.get_eid(root_id, hub_vid, directed=False, error=False) == -1:
#                 g.add_edge(root_id, hub_vid)
#                 g.es[-1]["relation"] = "root"
#                 g.es[-1]["relations"] = ["root"]

#         non_virtual_nodes = [v for v in g.vs if not v["is_virtual"]]
#         if non_virtual_nodes:
#             degrees = g.degree(non_virtual_nodes, mode="all")
#             degree_counts = defaultdict(int)
#             for d in degrees:
#                 degree_counts[d] += 1
#             sorted_degrees = sorted(degree_counts.items(), key=lambda item: item[0])
#         else:
#             sorted_degrees = []

#         with open(self.log_file, 'a') as f:
#             f.write(f"{query_id},{self.dataset_name},{g.vcount()},{g.ecount()},{sorted_degrees}\n")
#         logger.debug("[QueryGraphBuilder] directed=%s nodes=%d, edges=%d", self.directed, g.vcount(), g.ecount())
        
#         try:
#             row = compute_struct_metrics(
#                 g,
#                 graph_role="query",
#                 dataset_name=self.dataset_name,
#                 query_id=str(query_id),
#                 root_global_id=int(g["root_id"]) if "root_id" in g.attributes() else None,
#                 ppr_scores=None
#             )
#             self.metrics_logger.write(row)

            
#         except Exception as e:
#             logger.warning(f"[QueryGraphBuilder] metrics logging failed: {e}")

#         g.vs["global_id"] = list(range(g.vcount()))
#         g["root_id"] = int(g.vs.find(name=self.root_node).index)

#         return g

# class IDMapper:
#     @staticmethod
#     def docids_to_vids(doc_ids: Iterable[int], passage_node_idxs: List[int]) -> List[int]:
#         return [int(passage_node_idxs[d]) for d in doc_ids]

# @dataclass
# class SubgraphConfig:
#     max_depth: int = 2
#     weight_threshold: float = -1e-9
#     allow_topk: int = 200
#     flow_quantile: float = 0.95
#     per_node_top_r: int = 7
#     ensure_connectivity: bool = True
#     starter_beam: int = 8
#     per_hop_topk: int = 10
#     rank_metric: Literal["ppr","weight","deg"] = "ppr"
#     subgraph_mode: Literal["bfs", "ppr"] = "ppr"

# class SubgraphBuilder:
#     def __init__(self, g: ig.Graph, cfg: SubgraphConfig, node_scores: Optional[np.ndarray] = None):
#         self.g = g
#         self.cfg = cfg
#         self.node_scores = None if node_scores is None else np.nan_to_num(np.asarray(node_scores, float), nan=0.0)
#         self.weighted = ('weight' in g.es.attribute_names())
#         if self.weighted:
#             self.outstr = np.zeros(g.vcount(), dtype=float)
#             for e in g.es:
#                 self.outstr[e.source] += float(e['weight'])
#         else:
#             self.outstr = None

#     def _rank_key(self, u: int, v: int, w: float) -> float:
#         if self.cfg.rank_metric == "ppr" and self.node_scores is not None:
#             return float(self.node_scores[v])
#         if self.cfg.rank_metric == "weight" and self.weighted:
#             return float(w)
#         if self.cfg.rank_metric == "deg":
#             return float(self.g.degree(v))
#         return float(w) if self.weighted else 1.0

#     def _collect_candidates_by_bfs(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
#         visited = {root}
#         q = deque([(root, 0)])
#         edges: List[Tuple[int,int,float]] = []
#         mode = "OUT" if self.g.is_directed() else "ALL"
#         K = max(1, int(self.cfg.per_hop_topk)) if self.cfg.per_hop_topk else None

#         while q:
#             u, depth = q.popleft()
#             if depth >= self.cfg.max_depth:
#                 continue

#             neighs = []
#             for eid in self.g.incident(u, mode=mode):
#                 e = self.g.es[eid]
#                 if self.g.is_directed() and e.source != u:
#                     continue
#                 v = e.target if e.source == u else e.source
#                 w = float(e['weight']) if self.weighted else 1.0
#                 if w < self.cfg.weight_threshold:
#                     continue
#                 if allow_ids is not None and v not in allow_ids:
#                     continue
#                 neighs.append((u, v, w))

#             if K is not None and len(neighs) > K:
#                 neighs.sort(key=lambda t: self._rank_key(*t), reverse=True)
#                 neighs = neighs[:K]

#             for (uu, vv, ww) in neighs:
#                 edges.append((uu, vv, ww))
#                 if vv not in visited:
#                     visited.add(vv)
#                     q.append((vv, depth + 1))

#         if not edges and self.cfg.starter_beam > 0:
#             nbr_eids = self.g.incident(root, mode=mode)
#             cand = []
#             for eid in nbr_eids:
#                 e = self.g.es[eid]
#                 if self.g.is_directed() and e.source != root:
#                     continue
#                 v = e.target if e.source == root else e.source
#                 w = float(e['weight']) if self.weighted else 1.0
#                 if allow_ids is None or v in allow_ids:
#                     cand.append((root, v, w))
#             if K is not None and len(cand) > self.cfg.starter_beam:
#                 cand.sort(key=lambda t: self._rank_key(*t), reverse=True)
#                 cand = cand[:self.cfg.starter_beam]
#             edges = cand
#         return edges

#     def _collect_candidates_by_ppr(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
#         if self.node_scores is None:
#             return []
        
#         ppr_sorted_vids = np.argsort(self.node_scores)[::-1]
        
#         selected_vids_set = set()
#         for vid in ppr_sorted_vids:
#             if allow_ids is not None and vid not in allow_ids:
#                 continue
#             selected_vids_set.add(vid)
#             if len(selected_vids_set) >= self.cfg.allow_topk:
#                 break
        
#         # if root in self.g.vs["name"] and root not in selected_vids_set:
#         #     selected_vids_set.add(self.g.vs.find(name=root).index)

#         # root는 이미 vid(정점 인덱스)이므로, 단순 포함 검사면 충분
#         if (0 <= root < self.g.vcount()) and (root not in selected_vids_set):
#             selected_vids_set.add(root)

#         edges: List[Tuple[int,int,float]] = []
#         visited_edges: Set[Tuple[int, int]] = set()

#         for u in selected_vids_set:
#             for v in self.g.neighbors(u, mode="ALL"):
#                 if v in selected_vids_set:
#                     edge_tuple = tuple(sorted((u, v)))
#                     if edge_tuple not in visited_edges:
#                         visited_edges.add(edge_tuple)
#                         eids = self.g.get_eids([(u, v)], directed=False)
#                         if eids:
#                             w = float(self.g.es[eids[0]].attributes().get('weight', 1.0))
#                             edges.append((u, v, w))
#         return edges

#     def collect_candidates(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
#         if self.cfg.subgraph_mode == "bfs":
#             return self._collect_candidates_by_bfs(root, allow_ids)
#         elif self.cfg.subgraph_mode == "ppr":
#             return self._collect_candidates_by_ppr(root, allow_ids)
#         else:
#             raise ValueError(f"Unknown subgraph_mode: {self.cfg.subgraph_mode}")

#     def build_subgraph(self, edges: List[Tuple[int,int,float]], root: Optional[int] = None) -> ig.Graph:
#         sg = ig.Graph(directed=self.g.is_directed())
#         if not edges:
#             sg.add_vertex(name=(self.g.vs[root]['name'] if root is not None else "root"))
#             sg.vs['global_id'] = [root if root is not None else -1]
#             return sg

#         nodes = sorted({u for u,_,_ in edges} | {v for _,v,_ in edges})
#         idx = {n:i for i,n in enumerate(nodes)}
#         sg.add_vertices(len(nodes))
#         sg.vs['name'] = [self.g.vs[n]['name'] for n in nodes]
#         sg.vs['global_id'] = nodes
#         if root is not None:
#             if root in idx:
#                 sg["root_id"] = int(root)
#             else:
#                 sg["root_id"] = -1
        
#         sg.add_edges([(idx[u], idx[v]) for u,v,_ in edges])
#         sg.es['weight'] = [w for *_,w in edges]
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
#             for e in g.es:
#                 self.outstr[e.source] += float(e['weight'])
#         else:
#             self.outstr = None

#     def _P(self, u: int, eid: int) -> float:
#         if self.weighted:
#             if self.g.es[eid].source != u:
#                 return 0.0
#             w = float(self.g.es[eid]['weight'])
#             denom = self.outstr[u]
#             return (w/denom) if denom > 0 else 0.0
#         d = self.outdeg[u]
#         return (1.0/d) if d > 0 else 0.0

#     def _flow(self, u: int, v: int) -> float:
#         if self.pi is None:
#             return 0.0
#         eid = self.g.get_eid(u, v, directed=True, error=False)
#         if eid == -1:
#             return 0.0
#         return float(self.pi[u]) * self._P(u, eid)

#     def prune(self, root: int, candidate_edges: List[Tuple[int,int,float]]) -> List[Tuple[int,int,float]]:
#         if not candidate_edges or self.pi is None:
#             return candidate_edges

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
#         if r is None or r <= 0:
#             return kept

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

# class HyperbolicEmbedder:
#     def __init__(self, hr_instance, embed_dim=32, curvature=1.0, device="cpu"):
#         self.manifold = geoopt.manifolds.Lorentz(k=curvature)
#         self.embed_dim = embed_dim
#         self.device = device
#         self.hr = hr_instance
#         self.feature_extractors = {
#             # 'logdeg': self._get_logdeg,
#             # 'depth': self._get_depth,
#             # 'leaf': self._get_leaf,
#             # 'eig_centrality': self._get_eig_centrality,
#             # 'clustering': self._get_clustering,
#         }
    
#     def _get_logdeg(self, g):
#         return np.log1p(np.asarray(g.degree(mode="all"), dtype=np.float32)).reshape(-1, 1)

#     def _get_depth(self, g):
#         root_local = 0
#         try:
#             gids = np.asarray(g.vs["global_id"])
#             rid = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
#             loc = np.where(gids == rid)[0]
#             if len(loc): root_local = int(loc[0])
#         except Exception:
#             pass
#         g_ud = g.as_undirected() if g.is_directed() else g
#         d = np.asarray(g_ud.shortest_paths(source=root_local)[0], dtype=np.float32)
#         finite = np.isfinite(d); maxd = float(np.max(d[finite])) if finite.any() else 0.0
#         d[~finite] = maxd + 1.0
#         return (d / (maxd + 1e-6)).reshape(-1, 1)
        
#     def _get_leaf(self, g):
#         deg = np.asarray(g.degree(mode="all"), dtype=np.float32)
#         return (deg <= 1).astype(np.float32).reshape(-1, 1)

#     def _get_eig_centrality(self, g):
#         try:
#             eig = np.array(g.eigenvector_centrality()).astype(np.float32)
#             return (eig / (eig.max() + 1e-6)).reshape(-1, 1)
#         except Exception:
#             return np.zeros((g.vcount(), 1), dtype=np.float32)

#     def _get_clustering(self, g):
#         try:
#             clu = np.array(g.transitivity_local_undirected()).astype(np.float32)
#             return np.nan_to_num(clu, nan=0.0).reshape(-1, 1)
#         except Exception:
#             return np.zeros((g.vcount(), 1), dtype=np.float32)

#     def _get_treerep_metric(self, g: ig.Graph) -> Tuple[np.ndarray, nx.Graph]:
#         if g.vcount() < 2:
#             return np.zeros((g.vcount(), g.vcount()), dtype=np.float32), nx.Graph()

#         distances = np.array(g.shortest_paths(weights='weight' if 'weight' in g.es.attributes() else None))
#         if np.isinf(distances).any():
#             max_finite = np.max(distances[np.isfinite(distances)]) if np.isfinite(distances).any() else 0.0
#             distances[np.isinf(distances)] = max_finite + 1.0

#         treerep_instance = TreeRep(d=distances)
#         treerep_instance.learn_tree()
#         learned_tree = treerep_instance.G

#         num_original_nodes = g.vcount()
#         treerep_metric = np.zeros((num_original_nodes, num_original_nodes), dtype=np.float32)

#         if learned_tree.number_of_nodes() > 1:
#             try:
#                 tree_dist_dict = dict(nx.all_pairs_dijkstra_path_length(learned_tree, weight='weight'))
#                 for i in range(num_original_nodes):
#                     for j in range(num_original_nodes):
#                         if i in tree_dist_dict and j in tree_dist_dict[i]:
#                             treerep_metric[i, j] = tree_dist_dict[i][j]
#                         else:
#                             treerep_metric[i, j] = distances[i, j]
#             except Exception as e:
#                 logger.warning(f"Error computing tree distances: {e}. Falling back to original distances.")
#                 treerep_metric = distances
#         else:
#             treerep_metric = distances

#         # 대칭/유한/비음수/대각0 보정(안전)
#         finite = np.isfinite(treerep_metric)
#         if finite.any():
#             mx = treerep_metric[finite].max()
#             treerep_metric = np.where(finite, treerep_metric, mx)
#         treerep_metric = 0.5 * (treerep_metric + treerep_metric.T)
#         np.fill_diagonal(treerep_metric, 0.0)
#         treerep_metric = np.maximum(treerep_metric, 0.0)

#         return treerep_metric, learned_tree
    
#     def _treerep_features_from_tree(self, learned_tree: nx.Graph, g: ig.Graph) -> np.ndarray:
#         n = g.vcount()
#         if n == 0:
#             return np.zeros((0, 3), dtype=np.float32)
#         gids = np.asarray(g.vs["global_id"])
#         # root 후보: g["root_id"]가 전역 vid이면 그걸 우선 사용
#         try:
#             root = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
#         except Exception:
#             root = int(gids[0]) if len(gids) else 0

#         # 깊이(= root까지의 최단거리)
#         depth = np.zeros(n, dtype=np.float32)
#         if learned_tree.number_of_nodes() > 0 and root in learned_tree:
#             dist = dict(nx.single_source_dijkstra_path_length(learned_tree, root, weight='weight'))
#             for i in range(n):
#                 depth[i] = float(dist.get(int(gids[i]), 0.0))
#             if depth.max() > 0:
#                 depth /= (depth.max() + 1e-6)

#         # 리프 플래그(자식 수 1 이하)
#         deg = np.array([learned_tree.degree(int(gids[i])) if int(gids[i]) in learned_tree else 0 for i in range(n)], dtype=np.float32)
#         leaf = (deg <= 1).astype(np.float32)

#         # 부모 간선 가중치 근사(가까운 이웃 간선 최솟값)
#         parent_w = np.zeros(n, dtype=np.float32)
#         for i in range(n):
#             u = int(gids[i])
#             if u in learned_tree:
#                 nbrs = list(learned_tree.neighbors(u))
#                 if nbrs:
#                     parent_w[i] = min(learned_tree[u][v].get('weight', 1.0) for v in nbrs)
#         if parent_w.max() > 0:
#             parent_w /= (parent_w.max() + 1e-6)

#         return np.stack([depth, leaf, parent_w], axis=1).astype(np.float32)  # (N,3)


#     def graph_to_tensor(
#         self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ):
#         if g.vcount() == 0:
#             return torch.zeros(1, self.embed_dim, device=self.device)

#         feature_list = []

#         if use_treerep_metric:
#             treerep_metric, learned_tree = self._get_treerep_metric(g)
#             feats = self._treerep_features_from_tree(learned_tree, g)  # (N, F)  # ★ MDS 제거
#             feature_list.append(feats)
#         else:
#             for extractor in self.feature_extractors.values():
#                 feature_list.append(extractor(g))

#         if use_semantic_features and g.vcount() > 0:
#             semantic_features = []
#             global_ids = g.vs["global_id"]
#             for v_id in global_ids:
#                 node_key = self.hr.graph.vs[v_id]["name"]
#                 embed = np.zeros(self.hr.global_config.embedding_dim, dtype=np.float32)
#                 if node_key.startswith("chunk-") and passage_embeddings is not None:
#                     row = self.hr.chunk_embedding_store.get_row(node_key)
#                     if row and 'embedding' in row:
#                         embed = row['embedding']
#                 elif node_key.startswith("entity-") and entity_embeddings is not None:
#                     row = self.hr.entity_embedding_store.get_row(node_key)
#                     if row and 'embedding' in row:
#                         embed = row['embedding']
#                 semantic_features.append(embed)
            
#             if semantic_features:
#                 semantic_features = np.vstack(semantic_features)
#                 feature_list.append(semantic_features)
        
#         if not feature_list:
#             X = np.zeros((g.vcount(), self.embed_dim), dtype=np.float32)
#         else:
#             X = np.hstack(feature_list)
        
#         num_features = X.shape[1]
        
#         if num_features < self.embed_dim:
#             X = np.pad(X, ((0, 0), (0, self.embed_dim - num_features)), 'constant')
#         elif num_features > self.embed_dim:
#             X = X[:, :self.embed_dim]

#         if num_features > 0:
#             mu = X.mean(axis=0, keepdims=True)
#             sd = X.std(axis=0, keepdims=True)
#             sd = np.maximum(sd, 1e-3)
#             Z = np.clip((X - mu) / sd, -3.0, 3.0)
#         else:
#             Z = np.zeros((g.vcount(), self.embed_dim), dtype=np.float32)
        
#         return torch.tensor(Z, dtype=torch.float32, device=self.device)

#     def embed_lorentz(
#         self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> torch.Tensor:
#         if g.vcount() == 0:
#             return None
        
#         x = self.graph_to_tensor(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, 
#                                  entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
#                                  use_treerep_metric=use_treerep_metric)
        
#         with torch.no_grad():
#             if g.vcount() > 1:
#                 max_norm = 5.0
#                 n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
#                 x = x * torch.clamp(max_norm / n, max=1.0)
            
#             x_lorentz = self.manifold.expmap0(x)
#             x_lorentz = torch.nan_to_num(x_lorentz, nan=0.0, posinf=0.0, neginf=0.0)

#             if g.vcount() > 1:
#                 weights = np.array(g.betweenness()).astype(np.float32)
#                 weights = weights / (weights.sum() + 1e-6)
#                 weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
#                 pooled = (x_lorentz * weights).sum(dim=0)
#             else:
#                 pooled = x_lorentz.mean(dim=0)

#             pooled = self.manifold.projx(pooled)
#             if torch.linalg.norm(pooled).item() < 1e-8:
#                 pooled = pooled + 1e-3 * torch.randn_like(pooled)
#                 pooled = self.manifold.projx(pooled)
                
#             return pooled

#     def logmap0_safe(self, lorentz_vec: torch.Tensor) -> torch.Tensor:
#         t = self.manifold.logmap0(lorentz_vec)
#         return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

#     def embed_and_logmap0(
#         self, g: ig.Graph, ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> torch.Tensor:
#         y = self.embed_lorentz(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric)
#         if y is None:
#             return torch.zeros(self.embed_dim, device=self.device)
#         if torch.linalg.norm(y).item() < 1e-8:
#             y = y + 1e-3 * torch.randn_like(y)
#             y = self.manifold.projx(y)
#         t = self.logmap0_safe(y)
#         if torch.linalg.norm(t).item() < 1e-8:
#             t[0] = 1e-3
#         return t

#     def embed_nodes_lorentz(
#         self, g: igraph.Graph, ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> torch.Tensor:
#         if g.vcount() == 0:
#             return torch.empty(0, self.embed_dim, device=self.device)
#         x = self.graph_to_tensor(g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features, use_treerep_metric=use_treerep_metric)
#         with torch.no_grad():
#             max_norm = 5.0
#             n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
#             x = x * torch.clamp(max_norm / n, max=1.0)
#         return self.manifold.expmap0(x)
    
#     def shared_standardize(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
#         if not tensors:
#             return tensors
#         all_data = torch.cat([t.view(-1) for t in tensors], dim=0)
#         mean = torch.mean(all_data)
#         std = torch.std(all_data)
#         if std.item() == 0:
#             std = torch.tensor(1.0, device=all_data.device)
#         standardized = []
#         for t in tensors:
#             z = (t - mean) / std
#             z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
#             standardized.append(z)
#         return standardized

# class StructralSimilarity:
#     def __init__(self, mode="cosine"):
#         assert mode in ["cosine", "l2", "dot"], "Invalid similarity mode"
#         self.mode = mode

#     def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
#         x = np.asarray(scores, dtype=float)
#         if x.size == 0:
#             return x
#         lo, hi = np.percentile(x, 5), np.percentile(x, 95)
#         if hi - lo < 1e-8:
#             return np.full_like(x, 0.5)
#         y = (x - lo) / (hi - lo + 1e-8)
#         y = np.clip(y, 0.0, 1.0)
#         return y
    
#     def compute(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
#         if self.mode == "cosine":
#             return F.cosine_similarity(q, d, dim=-1)
#         if self.mode == "dot":
#             return torch.sum(q * d, dim=-1)
#         return -torch.norm(q - d, dim=-1)

#     def compute_listwise(self, query_list: List[torch.Tensor], doc_list: List[torch.Tensor]) -> torch.Tensor:
#         assert len(query_list) == len(doc_list)
#         sims = []
#         for q, d in zip(query_list, doc_list):
#             sim = self.compute(q.unsqueeze(0), d.unsqueeze(0)).squeeze()
#             sims.append(sim)
#         return torch.stack(sims)

# class HyperReranker:
#     def __init__(self, g: ig.Graph, save_dir: str = "./outputs/subgraph_analysis", dataset_name: str = "default"):
#         self.g = g
#         self.metrics_logger = MetricsLogger(save_dir=save_dir, dataset_name=dataset_name)
#         self.cfg = None


#     @staticmethod
#     def make_allow_ids(g: ig.Graph, seeds_vids: List[int],
#                          ppr_scores: Optional[np.ndarray], allow_topk: int) -> Optional[Set[int]]:
#         if ppr_scores is None or not allow_topk or allow_topk <= 0:
#             return None
#         ppr = np.nan_to_num(np.asarray(ppr_scores, float), nan=0.0, posinf=0.0, neginf=0.0)
#         topN = np.argsort(-ppr)[:allow_topk].tolist()
#         allow = set(topN)
#         for s in seeds_vids:
#             for nb in g.neighbors(s, mode="all"):
#                 allow.add(nb)
#         return allow

#     def build_pruned_subgraphs_for_topk_passages(
#         self,
#         topk_doc_ids: List[int],
#         topk_scores: List[float],
#         passage_node_idxs: List[int],
#         last_ppr_scores: Optional[np.ndarray],
#         cfg: SubgraphConfig,
#         head_k: int = 30,
#     ) -> Tuple[List[ig.Graph], List[int], List[int]]:
#         if not topk_doc_ids:
#             return [], [], []
#         self.cfg = cfg

#         sem = np.asarray(topk_scores, float)
#         head_idx = np.argsort(-sem)[: min(head_k, len(topk_doc_ids))]
#         head_doc_ids = [topk_doc_ids[i] for i in head_idx]
#         head_vids = IDMapper.docids_to_vids(head_doc_ids, passage_node_idxs)

#         allow_ids = self.make_allow_ids(self.g, head_vids, last_ppr_scores, cfg.allow_topk)
        
#         builder = SubgraphBuilder(self.g, cfg, node_scores=last_ppr_scores)
#         pruner = PPRPruner(self.g, last_ppr_scores, cfg)

#         subgraphs: List[ig.Graph] = []
        
#         for rid in head_vids:
#             cands = builder.collect_candidates(root=rid, allow_ids=allow_ids)
            
#             pruned = pruner.prune(root=rid, candidate_edges=cands) if cfg.subgraph_mode == "ppr" else cands
            
#             sg = builder.build_subgraph(pruned, root=rid)
#             try:
#                 row = compute_struct_metrics(
#                     sg,
#                     graph_role="subgraph",
#                     dataset_name=getattr(self, "dataset_name", "default") if hasattr(self, "dataset_name") else "default",
#                     query_id=str(getattr(self, "query_id", "unknown")) if hasattr(self, "query_id") else "unknown",
#                     root_global_id=rid,
#                     ppr_scores=last_ppr_scores ,
#                     exclude_virtual= False
#                 )
#                 row.update({
#                     "root_doc_vid": int(rid),
#                     "root_doc_id": int(head_doc_ids[len(subgraphs)]) if len(subgraphs) < len(head_doc_ids) else None,
#                     "subgraph_idx": int(len(subgraphs)),
#                     "collect_mode": self.cfg.subgraph_mode if hasattr(self, "cfg") else "unknown"
#                 })
#                 self.metrics_logger.write(row)
#             except Exception as e:
#                 logger.warning(f"[SubgraphMetrics] logging failed: {e}")

#             subgraphs.append(sg)
#             logger.debug(f"[Subgraph] mode={cfg.subgraph_mode} root={rid} cands={len(cands)} pruned={len(pruned)} v={sg.vcount()} e={sg.ecount()}")
#         return subgraphs, head_doc_ids, head_vids
    
#     def _struct_scores_tangent(
#         self,
#         query_graph: ig.Graph,
#         subgraphs: List[ig.Graph],
#         embedder: "HyperbolicEmbedder",
#         scorer: "StructralSimilarity",
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> np.ndarray:
#         with torch.no_grad():
#             tq = embedder.embed_and_logmap0(
#                 query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#             tdocs = [
#                 embedder.embed_and_logmap0(
#                     g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
#                     entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
#                     use_treerep_metric=use_treerep_metric
#                 ) for g in subgraphs
#             ]
        
#         try:
#             if not tdocs:
#                 return np.array([])
#             T = torch.stack(tdocs)
#             C = F.cosine_similarity(T.unsqueeze(1), T.unsqueeze(0), dim=-1)
#             mask = ~torch.eye(C.size(0), dtype=bool, device=C.device)
#             vals = C[mask]
#             logger.debug(
#                 "[EMB-DIAG][raw] tdocs_norm_mean=%.4f tdocs_std=%.4f | pairwise_cos_mean=%.4f p95=%.4f",
#                 T.norm(dim=1).mean().item(),
#                 T.std().item(),
#                 vals.mean().item(),
#                 vals.kthvalue(max(1, int(0.95 * len(vals))))[0].item()
#             )
#             cos_raw = scorer.compute_listwise([tq for _ in tdocs], tdocs)
#             q05, q50, q95 = torch.quantile(cos_raw, torch.tensor([0.05, 0.5, 0.95], device=cos_raw.device))
#             logger.debug(
#                 "[EMB-DIAG][raw] qvdoc_cos min=%.4f max=%.4f | p05=%.4f p50=%.4f p95=%.4f",
#                 cos_raw.min().item(), cos_raw.max().item(), q05.item(), q50.item(), q95.item()
#             )
#         except Exception as e:
#             logger.debug("[EMB-DIAG] raw diag skipped: %s", e)

#         tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
#         q_list = [tq_std for _ in range(len(tdocs_std))]

#         struct_head = scorer.compute_listwise(q_list, tdocs_std)
#         return _to_1d_numpy(struct_head)

#     def _struct_scores_lorentz(
#         self,
#         query_graph: ig.Graph,
#         subgraphs: List[ig.Graph],
#         embedder: "HyperbolicEmbedder",
#         temp: float = 2.0,
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> np.ndarray:
#         with torch.no_grad():
#             qL = embedder.embed_lorentz(
#                 query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#             dL = [
#                 embedder.embed_lorentz(
#                     g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
#                     entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
#                     use_treerep_metric=use_treerep_metric
#                 ) for g in subgraphs
#             ]

#             qL = qL if qL is not None else torch.zeros(embedder.embed_dim, device=embedder.device)
#             dL_tensors = []
#             for x in dL:
#                 if x is None:
#                     dL_tensors.append(torch.zeros(embedder.embed_dim, device=embedder.device))
#                 else:
#                     dL_tensors.append(x)

#             dists2 = torch.stack([
#                 embedder.manifold.dist2(qL.unsqueeze(0), x.unsqueeze(0)).squeeze()
#                 for x in dL_tensors
#             ])
#         d2 = _to_1d_numpy(dists2)
#         sim = -d2
#         if sim.size:
#             norm_sim = StructralSimilarity().normalize(sim)
#             sim = 1.0 / (1.0 + np.exp(-((norm_sim - 0.5) / (1.0 / temp))))
#         return sim

#     def _struct_scores_node_level(
#         self,
#         query_graph: ig.Graph,
#         subgraphs: List[ig.Graph],
#         embedder: "HyperbolicEmbedder",
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> np.ndarray:
#         with torch.no_grad():
#             q_node_embeddings = embedder.embed_nodes_lorentz(
#                 query_graph, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#             if q_node_embeddings.numel() == 0:
#                 return np.zeros(len(subgraphs))
            
#             struct_scores = []
#             for doc_g in subgraphs:
#                 d_node_embeddings = embedder.embed_nodes_lorentz(
#                     doc_g, ppr_scores=ppr_scores, passage_embeddings=passage_embeddings,
#                     entity_embeddings=entity_embeddings, use_semantic_features=use_semantic_features,
#                     use_treerep_metric=use_treerep_metric
#                 )
#                 if d_node_embeddings.numel() == 0:
#                     struct_scores.append(0.0)
#                     continue

#                 tq = embedder.logmap0_safe(q_node_embeddings)
#                 td = embedder.logmap0_safe(d_node_embeddings)
#                 similarity_matrix = F.cosine_similarity(tq.unsqueeze(1), td.unsqueeze(0), dim=-1)
#                 max_sim_per_query_node = torch.max(similarity_matrix, dim=1)[0]
#                 score = torch.mean(max_sim_per_query_node)
#                 struct_scores.append(score.item())
#         return np.array(struct_scores)
    
#     def rerank_with_structure(
#         self,
#         query_graph: Optional[ig.Graph],
#         passage_topk_ids: List[int],
#         passage_topk_scores: List[float],
#         passage_node_idxs: List[int],
#         last_ppr_scores: Optional[np.ndarray],
#         cfg: SubgraphConfig,
#         embedder: HyperbolicEmbedder,
#         scorer: StructralSimilarity,
#         passage_embeddings: np.ndarray,
#         entity_embeddings: np.ndarray,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False,
#         alpha: float = 0.8,
#         struct_k: int = 200,
#         device: str = "cpu",
#         save_cb: Optional[Callable[[ig.Graph, str, dict], None]] = None,
#         save_ctx: Optional[dict] = None,
#         sim_cac_mode: Literal["tangent", "lorentz", "node_level"] = "tangent",
#         lorentz_temp: float = 2.0,
#     ) -> Tuple[List[int], List[float]]:

#         passage_topk_ids = list(np.asarray(passage_topk_ids).tolist())
#         passage_topk_scores = list(np.asarray(passage_topk_scores, dtype=float).tolist())

#         if query_graph is None or len(passage_topk_ids) == 0:
#             return passage_topk_ids, passage_topk_scores

#         subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
#             passage_topk_ids, passage_topk_scores, passage_node_idxs,
#             last_ppr_scores, cfg, head_k=struct_k
#         )
        
#         if not subgraphs:
#             return passage_topk_ids, passage_topk_scores

#         if sim_cac_mode == "tangent":
#             struct_head = self._struct_scores_tangent(
#                 query_graph, subgraphs, embedder, scorer, ppr_scores=last_ppr_scores,
#                 passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#         elif sim_cac_mode == "lorentz":
#             struct_head = self._struct_scores_lorentz(
#                 query_graph, subgraphs, embedder, temp=lorentz_temp, ppr_scores=last_ppr_scores,
#                 passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#         elif sim_cac_mode == "node_level":
#             struct_head = self._struct_scores_node_level(
#                 query_graph, subgraphs, embedder, ppr_scores=last_ppr_scores,
#                 passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#         else:
#             raise ValueError(f"Unknown mode: {sim_cac_mode}")

#         if struct_head.size == 0:
#             return passage_topk_ids, passage_topk_scores

#         # sem = np.asarray(passage_topk_scores, float)
#         # head_idx = np.argsort(-sem)[:len(subgraphs)]
#         # sem_norm_head = scorer.normalize(sem[head_idx])
#         # str_norm_head = scorer.normalize(struct_head)

#         # final_head = alpha * sem_norm_head + (1 - alpha) * str_norm_head
#         sem = np.asarray(passage_topk_scores, float)
#         head_idx = np.argsort(-sem)[:len(subgraphs)]

#         # ① semantic [0,1]
#         sem_head = sem[head_idx]
#         sem_norm_head = (sem_head - sem_head.min()) / (sem_head.max() - sem_head.min() + 1e-9)

#         # ② structural [0,1]
#         s = np.asarray(struct_head, float)
#         s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
#         s = (s - s.min()) / (s.max() - s.min() + 1e-9)

#         # ③ 결합 (지금 코드 의미 그대로: alpha는 semantic 가중)
#         final_head = alpha * sem_norm_head + (1 - alpha) * s

#         head_pairs = sorted(
#             zip([passage_topk_ids[i] for i in head_idx], final_head),
#             key=lambda x: x[1], reverse=True
#         )

#         diag = {}
#         diag["struct_raw_min"] = float(np.min(struct_head)) if struct_head.size > 0 else 0.0
#         diag["struct_raw_max"] = float(np.max(struct_head)) if struct_head.size > 0 else 0.0
#         diag["struct_norm_min"] = float(np.min(s)) if s.size > 0 else 0.0
#         diag["struct_norm_max"] = float(np.max(s)) if s.size > 0 else 0.0
#         diag["sem_norm_min"] = float(np.min(sem_norm_head)) if sem_norm_head.size > 0 else 0.0
#         diag["sem_norm_max"] = float(np.max(sem_norm_head)) if sem_norm_head.size > 0 else 0.0

#         diag["final_min"] = float(np.min(final_head)) if final_head.size > 0 else 0.0
#         diag["final_max"] = float(np.max(final_head)) if final_head.size > 0 else 0.0
        
#         try:
#             sem_head = sem[head_idx]
#             diag['sem_struct_spearmanr'] = float(spearmanr(sem_head, struct_head).correlation)
#             diag['sem_struct_kendalltau'] = float(kendalltau(sem_head, struct_head).correlation)
#         except Exception:
#             diag['sem_struct_spearmanr'] = np.nan
#             diag['sem_struct_kendalltau'] = np.nan

#         diag_msg = f"[RERANK] head_k={struct_k} alpha={alpha} | {diag}"
#         if len(diag_msg) > 1000: diag_msg = diag_msg[:1000] + "..."
#         logger.info(diag_msg)
        
#         try:
#             head_diag_logger = MetricsLogger(save_dir="./outputs/diag", dataset_name="diag")
#             head_diag_logger.write({
#                 "dataset": getattr(self, "dataset_name", "default"),
#                 "query_id": getattr(self, "query_id", "unknown"),
#                 "graph_role": "head_diag",
#                 "struct_head_min": diag["struct_raw_min"],
#                 "struct_head_max": diag["struct_raw_max"],
#                 "sem_struct_spearmanr": diag["sem_struct_spearmanr"],
#                 "sem_struct_kendalltau": diag["sem_struct_kendalltau"],
#                 "alpha": float(alpha),
#                 "mode": str(sim_cac_mode),
#                 "struct_k": int(struct_k),
#             })
#         except Exception:
#             pass

#         orig_scores = dict(zip(passage_topk_ids, passage_topk_scores))
#         final_scores = {doc_id: score for doc_id, score in head_pairs}
        
#         merged_ids, merged_scores = [], []
        
#         for doc_id, score in head_pairs:
#             merged_ids.append(doc_id)
#             merged_scores.append(score)
            
#         remaining_ids = [did for did in passage_topk_ids if did not in final_scores]
#         remaining_pairs = sorted(
#             [(did, orig_scores[did]) for did in remaining_ids],
#             key=lambda x: x[1], reverse=True
#         )
        
#         for doc_id, score in remaining_pairs:
#             merged_ids.append(doc_id)
#             merged_scores.append(score)

#         return merged_ids, merged_scores

# ============================================================
# -----------------------------------------------------------
# ==============================================================
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

# TreeRep
from .TreeRep import TreeRep
import networkx as nx
from sklearn.manifold import MDS
import geoopt

logger = logging.getLogger(__name__)

import json, math
from typing import Mapping

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
    flow_quantile: float = 0.85
    per_node_top_r: int = 7
    ensure_connectivity: bool = True
    starter_beam: int = 8
    per_hop_topk: int = 10
    rank_metric: Literal["ppr","weight","deg"] = "ppr"
    subgraph_mode: Literal["bfs", "ppr"] = "ppr"

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
        mode = "OUT" if self.g.is_directed() else "ALL"
        K = max(1, int(self.cfg.per_hop_topk)) if self.cfg.per_hop_topk else None

        while q:
            u, depth = q.popleft()
            if depth >= self.cfg.max_depth:
                continue

            neighs = []
            for eid in self.g.incident(u, mode=mode):
                e = self.g.es[eid]
                if self.g.is_directed() and e.source != u:
                    continue
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
                if self.g.is_directed() and e.source != root:
                    continue
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
        
        # root는 이미 vid(정점 인덱스)이므로, 단순 포함 검사면 충분
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

    def collect_candidates(self, root: int, allow_ids: Optional[Set[int]]) -> List[Tuple[int,int,float]]:
        if self.cfg.subgraph_mode == "bfs":
            return self._collect_candidates_by_bfs(root, allow_ids)
        elif self.cfg.subgraph_mode == "ppr":
            return self._collect_candidates_by_ppr(root, allow_ids)
        else:
            raise ValueError(f"Unknown subgraph_mode: {self.cfg.subgraph_mode}")

    def build_subgraph(self, edges: List[Tuple[int,int,float]], root: Optional[int] = None) -> ig.Graph:
        sg = ig.Graph(directed=self.g.is_directed())
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
            if root in idx:
                sg["root_id"] = int(root)
            else:
                sg["root_id"] = -1
        
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

# ==============================================================


# # =========================
# # Hyperbolic Embedder
# # =========================
# class HyperbolicEmbedder:
#     def __init__(self, hr_instance, embed_dim=32, curvature=1.0, device="cpu"):
#         self.manifold = geoopt.manifolds.Lorentz(k=curvature)
#         self.embed_dim = embed_dim
#         self.device = device
#         self.hr = hr_instance
#         # TreeRep 실행을 위한 최소 노드 수 (권장: 3 또는 4)
#         self.min_nodes_for_treerep = 3
#         self.feature_extractors = {
#             # 필요 시 기본 유클리드 피처 추가 가능
#         }

#     # ---- Optional basic structural features (현재 미사용) ----
#     def _get_logdeg(self, g):
#         return np.log1p(np.asarray(g.degree(mode="all"), dtype=np.float32)).reshape(-1, 1)

#     def _get_depth(self, g):
#         root_local = 0
#         try:
#             gids = np.asarray(g.vs["global_id"])
#             rid = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
#             loc = np.where(gids == rid)[0]
#             if len(loc):
#                 root_local = int(loc[0])
#         except Exception:
#             pass

#         g_ud = g.as_undirected() if g.is_directed() else g
#         d = np.asarray(g_ud.shortest_paths(source=root_local)[0], dtype=np.float32)
#         finite = np.isfinite(d)
#         maxd = float(np.max(d[finite])) if finite.any() else 0.0
#         d[~finite] = maxd + 1.0
#         return (d / (maxd + 1e-6)).reshape(-1, 1)

#     def _get_leaf(self, g):
#         deg = np.asarray(g.degree(mode="all"), dtype=np.float32)
#         return (deg <= 1).astype(np.float32).reshape(-1, 1)

#     def _get_eig_centrality(self, g):
#         try:
#             eig = np.array(g.eigenvector_centrality()).astype(np.float32)
#             return (eig / (eig.max() + 1e-6)).reshape(-1, 1)
#         except Exception:
#             return np.zeros((g.vcount(), 1), dtype=np.float32)

#     def _get_clustering(self, g):
#         try:
#             clu = np.array(g.transitivity_local_undirected()).astype(np.float32)
#             return np.nan_to_num(clu, nan=0.0).reshape(-1, 1)
#         except Exception:
#             return np.zeros((g.vcount(), 1), dtype=np.float32)

#     # ---- TreeRep metric 계산 ----
#     def _get_treerep_metric(self, g) -> Tuple[np.ndarray, nx.Graph]:
#         n = g.vcount()
#         if n == 0:
#             return np.zeros((0, 0), dtype=np.float32), nx.Graph()

#         distances = np.array(
#             g.shortest_paths(weights='weight' if 'weight' in g.es.attribute_names() else None),
#             dtype=np.float32
#         )

#         if not np.isfinite(distances).all():
#             if np.isfinite(distances).any():
#                 mx = float(np.max(distances[np.isfinite(distances)]))
#             else:
#                 mx = 0.0
#             distances[~np.isfinite(distances)] = mx + 1.0

#         # --- 핵심 가드: 노드 수가 너무 작으면 TREEREP 생략 ---
#         if n < self.min_nodes_for_treerep:
#             # 작은 그래프는 그냥 기존 거리 행렬을 사용
#             return distances, nx.from_numpy_array(distances)

#         # --- TREEREP 시도: 실패 시 거리 행렬로 폴백 ---
#         try:
#             treerep_instance = TreeRep(d=distances)
#             treerep_instance.learn_tree()  # 여기서 IndexError 발생할 수 있음
#             learned_tree = treerep_instance.G
#         except Exception as e:
#             logger.warning(f"[TreeRep] learn_tree failed (n={n}): {e}. Falling back to plain distances.")
#             return distances, nx.from_numpy_array(distances)

#         # TREEREP로부터 트리 거리를 꺼내오고, 부족분은 원 거리로 보강
#         treerep_metric = np.zeros((n, n), dtype=np.float32)
#         if learned_tree.number_of_nodes() > 1:
#             try:
#                 tree_dist_dict = dict(nx.all_pairs_dijkstra_path_length(learned_tree, weight='weight'))
#                 for i in range(n):
#                     for j in range(n):
#                         if i in tree_dist_dict and j in tree_dist_dict[i]:
#                             treerep_metric[i, j] = float(tree_dist_dict[i][j])
#                         else:
#                             treerep_metric[i, j] = distances[i, j]
#             except Exception as e:
#                 logger.warning(f"[TreeRep] tree distance extraction failed: {e}. Using plain distances.")
#                 treerep_metric = distances
#         else:
#             treerep_metric = distances

#         # 안전 보정
#         finite = np.isfinite(treerep_metric)
#         if not finite.all():
#             if finite.any():
#                 mx = float(np.max(treerep_metric[finite]))
#             else:
#                 mx = 0.0
#             treerep_metric[~finite] = mx

#         treerep_metric = 0.5 * (treerep_metric + treerep_metric.T)
#         np.fill_diagonal(treerep_metric, 0.0)
#         treerep_metric = np.maximum(treerep_metric, 0.0)
#         return treerep_metric, learned_tree

#     # ---- TreeRep로부터 피처 생성 ----
#     def _treerep_features_from_tree(
#         self,
#         learned_tree: nx.Graph,
#         g,
#         ppr_scores: Optional[np.ndarray] = None,  # 호환성 유지(미사용)
#     ) -> np.ndarray:
#         """
#         반환: (N, 5) float32 포함 피처 (순서):
#         0 normalized_depth = depth / diameter (가중 길이 기준)
#         1 leaf = 리프 여부 (degree <= 1)
#         2 branching_factor_norm = (자식 수) 정규화
#         3 subtree_size_norm_log = log1p(subtree_size)/log1p(N)
#         4 parent_w = 이웃 간선 가중 최소값 (0-1 정규화)
#         """
#         n = g.vcount()
#         if n == 0:
#             return np.zeros((0, 5), dtype=np.float32)

#         gids = np.asarray(g.vs["global_id"])
#         # 루트(global id)
#         try:
#             root_gid = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
#         except Exception:
#             root_gid = int(gids[0]) if len(gids) else 0

#         # 트리 비어있으면 0
#         if learned_tree.number_of_nodes() == 0:
#             return np.zeros((n, 5), dtype=np.float32)

#         # ---------------------------
#         # 1) depth/leaf/parent_w 준비
#         # ---------------------------
#         depth = np.zeros(n, dtype=np.float32)
#         leaf = np.zeros(n, dtype=np.float32)
#         parent_w = np.zeros(n, dtype=np.float32)

#         # root로부터 가중 최단거리 (없으면 0)
#         if root_gid in learned_tree:
#             dist = dict(nx.single_source_dijkstra_path_length(learned_tree, root_gid, weight='weight'))
#         else:
#             dist = {}

#         for i in range(n):
#             u = int(gids[i])
#             depth[i] = float(dist.get(u, 0.0))
#             if u in learned_tree:
#                 deg_u = learned_tree.degree(u)
#                 leaf[i] = 1.0 if deg_u <= 1 else 0.0
#                 nbrs = list(learned_tree.neighbors(u))
#                 if nbrs:
#                     parent_w[i] = min(learned_tree[u][v].get('weight', 1.0) for v in nbrs)

#         # ---------------------------
#         # 2) parent/children, subtree, branching
#         # ---------------------------
#         parent = {}
#         children = {int(x): [] for x in learned_tree.nodes()}
#         hop_depth = {int(x): -1 for x in learned_tree.nodes()}

#         if root_gid in learned_tree:
#             from collections import deque
#             dq = deque([root_gid])
#             parent[root_gid] = -1
#             hop_depth[root_gid] = 0
#             while dq:
#                 u = dq.popleft()
#                 for v in learned_tree.neighbors(u):
#                     v = int(v)
#                     if hop_depth[v] != -1:
#                         continue
#                     parent[v] = int(u)
#                     children[int(u)].append(v)
#                     # 참고용
#                     hop_depth[v] = hop_depth[int(u)] + 1
#                     dq.append(v)

#         node_list = list(learned_tree.nodes())
#         N_all = max(1, len(node_list))

#         # subtree size (post-order)
#         subtree_size = {int(x): 1 for x in node_list}

#         def dfs_post(u: int) -> int:
#             s = 1
#             for v in children.get(u, []):
#                 s += dfs_post(v)
#             subtree_size[u] = s
#             return s

#         if root_gid in learned_tree:
#             dfs_post(int(root_gid))

#         # branching factor = 자식 수
#         branching = {int(x): max(0, len(children.get(int(x), []))) for x in node_list}

#         # ---------------------------
#         # 3) 정규화/스택
#         # ---------------------------
#         def _norm01(x: np.ndarray) -> np.ndarray:
#             x = np.asarray(x, dtype=np.float32)
#             lo, hi = float(np.min(x)), float(np.max(x))
#             if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
#                 return np.zeros_like(x, dtype=np.float32)
#             return (x - lo) / (hi - lo + 1e-12)

#         # normalized depth
#         diam = float(np.max(depth)) if np.isfinite(depth).any() else 0.0
#         norm_depth = (depth / (diam + 1e-6)) if diam > 0 else np.zeros_like(depth, dtype=np.float32)

#         # subtree size (log 정규화)
#         arr_subtree = np.zeros(n, dtype=np.float32)
#         for i in range(n):
#             u = int(gids[i])
#             arr_subtree[i] = float(subtree_size.get(u, 1))
#         subtree_log = np.log1p(arr_subtree) / max(1e-6, np.log1p(N_all))

#         # branching factor (정규화)
#         arr_branch = np.zeros(n, dtype=np.float32)
#         for i in range(n):
#             u = int(gids[i])
#             arr_branch[i] = float(branching.get(u, 0))
#         branch_norm = _norm01(arr_branch)

#         # parent_w (정규화)
#         parent_w_norm = _norm01(parent_w)

#         # 최종 스택: [normalized_depth, leaf, branching_factor_norm, subtree_size_norm_log, parent_w_norm]
#         feats = np.stack([
#             norm_depth.astype(np.float32),
#             leaf.astype(np.float32),
#             branch_norm.astype(np.float32),
#             subtree_log.astype(np.float32),
#             parent_w_norm.astype(np.float32),
#         ], axis=1)

#         return feats

#     # ---- 그래프를 유클리드 텐서로 변환(구조 + 의미) ----
#     def graph_to_tensor(
#         self,
#         g,
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ):
#         if g.vcount() == 0:
#             return torch.zeros(1, self.embed_dim, device=self.device)

#         feature_list = []

#         # 구조 피처: TreeRep 기반 (선택)
#         if use_treerep_metric:
#             treerep_metric, learned_tree = self._get_treerep_metric(g)
#             feats = self._treerep_features_from_tree(
#                 learned_tree,
#                 g,
#                 ppr_scores=ppr_scores  # ← 호환성 자리 유지
#             )
#             feature_list.append(feats)
#         else:
#             # 필요 시 기본 구조 피처 추가 가능
#             pass

#         # 의미 피처: passage/entity 임베딩 (선택)
#         if use_semantic_features and g.vcount() > 0:
#             semantic_features = []
#             global_ids = g.vs["global_id"]
#             for v_id in global_ids:
#                 node_key = self.hr.graph.vs[v_id]["name"]
#                 embed = np.zeros(self.hr.global_config.embedding_dim, dtype=np.float32)

#                 if node_key.startswith("chunk-") and passage_embeddings is not None:
#                     row = self.hr.chunk_embedding_store.get_row(node_key)
#                     if row and 'embedding' in row:
#                         embed = row['embedding']
#                 elif node_key.startswith("entity-") and entity_embeddings is not None:
#                     row = self.hr.entity_embedding_store.get_row(node_key)
#                     if row and 'embedding' in row:
#                         embed = row['embedding']

#                 semantic_features.append(embed)

#             if semantic_features:
#                 semantic_features = np.vstack(semantic_features)
#                 feature_list.append(semantic_features)

#         # 피처 스택/패딩/정규화
#         if not feature_list:
#             X = np.zeros((g.vcount(), self.embed_dim), dtype=np.float32)
#         else:
#             X = np.hstack(feature_list)

#         num_features = X.shape[1]
#         if num_features < self.embed_dim:
#             X = np.pad(X, ((0, 0), (0, self.embed_dim - num_features)), 'constant')
#         elif num_features > self.embed_dim:
#             X = X[:, :self.embed_dim]

#         if num_features > 0:
#             mu = X.mean(axis=0, keepdims=True)
#             sd = X.std(axis=0, keepdims=True)
#             sd = np.maximum(sd, 1e-3)
#             Z = np.clip((X - mu) / sd, -3.0, 3.0)
#         else:
#             Z = np.zeros((g.vcount(), self.embed_dim), dtype=np.float32)

#         return torch.tensor(Z, dtype=torch.float32, device=self.device)

#     # ---- 그래프 임베딩(Lorentz) + 풀링 ----
#     def embed_lorentz(
#         self,
#         g,
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> torch.Tensor:
#         if g.vcount() == 0:
#             return None

#         x = self.graph_to_tensor(
#             g,
#             ppr_scores=ppr_scores,
#             passage_embeddings=passage_embeddings,
#             entity_embeddings=entity_embeddings,
#             use_semantic_features=use_semantic_features,
#             use_treerep_metric=use_treerep_metric
#         )

#         with torch.no_grad():
#             # 노름 제한 후 로렌츠 expmap
#             if g.vcount() > 1:
#                 max_norm = 5.0
#                 n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
#                 x = x * torch.clamp(max_norm / n, max=1.0)

#             x_lorentz = self.manifold.expmap0(x)
#             x_lorentz = torch.nan_to_num(x_lorentz, nan=0.0, posinf=0.0, neginf=0.0)

#             # 가중 풀링(betweenness 기반) 또는 평균
#             if g.vcount() > 1:
#                 weights = np.array(g.betweenness()).astype(np.float32)
#                 weights = weights / (weights.sum() + 1e-6)
#                 weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
#                 pooled = (x_lorentz * weights).sum(dim=0)
#             else:
#                 pooled = x_lorentz.mean(dim=0)

#             pooled = self.manifold.projx(pooled)
#             if torch.linalg.norm(pooled).item() < 1e-8:
#                 pooled = pooled + 1e-3 * torch.randn_like(pooled)
#             pooled = self.manifold.projx(pooled)
#             return pooled

#     def logmap0_safe(self, lorentz_vec: torch.Tensor) -> torch.Tensor:
#         t = self.manifold.logmap0(lorentz_vec)
#         return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

#     # ---- 그래프 임베딩 → 탄젠트 로그맵 ----
#     def embed_and_logmap0(
#         self,
#         g,
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> torch.Tensor:
#         y = self.embed_lorentz(
#             g,
#             ppr_scores=ppr_scores,
#             passage_embeddings=passage_embeddings,
#             entity_embeddings=entity_embeddings,
#             use_semantic_features=use_semantic_features,
#             use_treerep_metric=use_treerep_metric
#         )
#         if y is None:
#             return torch.zeros(self.embed_dim, device=self.device)

#         if torch.linalg.norm(y).item() < 1e-8:
#             y = y + 1e-3 * torch.randn_like(y)
#         y = self.manifold.projx(y)

#         t = self.logmap0_safe(y)
#         if torch.linalg.norm(t).item() < 1e-8:
#             t[0] = 1e-3
#         return t

#     # ---- 노드 레벨 임베딩 (Lorentz) ----
#     def embed_nodes_lorentz(
#         self,
#         g,
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> torch.Tensor:
#         if g.vcount() == 0:
#             return torch.empty(0, self.embed_dim, device=self.device)

#         x = self.graph_to_tensor(
#             g,
#             ppr_scores=ppr_scores,
#             passage_embeddings=passage_embeddings,
#             entity_embeddings=entity_embeddings,
#             use_semantic_features=use_semantic_features,
#             use_treerep_metric=use_treerep_metric
#         )

#         with torch.no_grad():
#             max_norm = 5.0
#             n = torch.linalg.norm(x, dim=1, keepdim=True) + 1e-6
#             x = x * torch.clamp(max_norm / n, max=1.0)
#             return self.manifold.expmap0(x)

#     # ---- 여러 텐서 공통 표준화(진단용) ----
#     def shared_standardize(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
#         if not tensors:
#             return tensors
#         all_data = torch.cat([t.view(-1) for t in tensors], dim=0)
#         mean = torch.mean(all_data)
#         std = torch.std(all_data)
#         if std.item() == 0:
#             std = torch.tensor(1.0, device=all_data.device)

#         standardized = []
#         for t in tensors:
#             z = (t - mean) / std
#             z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
#             standardized.append(z)
#         return standardized


# # =========================
# # Structural Similarity
# # =========================
# class StructralSimilarity:
#     def __init__(self, mode="cosine"):
#         assert mode in ["cosine", "l2", "dot"], "Invalid similarity mode"
#         self.mode = mode

#     def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
#         x = np.asarray(scores, dtype=float)
#         if x.size == 0:
#             return x
#         lo, hi = np.percentile(x, 5), np.percentile(x, 95)
#         if hi - lo < 1e-8:
#             return np.full_like(x, 0.5)
#         y = (x - lo) / (hi - lo + 1e-8)
#         y = np.clip(y, 0.0, 1.0)
#         return y

#     def compute(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
#         if self.mode == "cosine":
#             return F.cosine_similarity(q, d, dim=-1)
#         if self.mode == "dot":
#             return torch.sum(q * d, dim=-1)
#         # l2
#         return -torch.norm(q - d, dim=-1)

#     def compute_listwise(self, query_list: List[torch.Tensor], doc_list: List[torch.Tensor]) -> torch.Tensor:
#         assert len(query_list) == len(doc_list)
#         sims = []
#         for q, d in zip(query_list, doc_list):
#             sim = self.compute(q.unsqueeze(0), d.unsqueeze(0)).squeeze()
#             sims.append(sim)
#         return torch.stack(sims)


# # =========================
# # Hyper Reranker
# # =========================
# class HyperReranker:
#     def __init__(self, g, save_dir: str = "./outputs/subgraph_analysis", dataset_name: str = "default"):
#         self.g = g
#         self.metrics_logger = MetricsLogger(save_dir=save_dir, dataset_name=dataset_name)
#         self.cfg = None

#     # ★★★ C. allow_ids: 전역 상위 ∩ 루트 2-hop 근방 (지역성 부여)
#     @staticmethod
#     def make_allow_ids(g, seeds_vids: List[int], ppr_scores: Optional[np.ndarray], allow_topk: int) -> Optional[Set[int]]:
#         if ppr_scores is None or not allow_topk or allow_topk <= 0:
#             return None
#         ppr = np.nan_to_num(np.asarray(ppr_scores, float), nan=0.0, posinf=0.0, neginf=0.0)
#         topN = np.argsort(-ppr)[:max(allow_topk*5, allow_topk)]
#         allow = set(int(i) for i in topN)

#         # seed 주변 2-hop 근방
#         g_ud = g.as_undirected() if g.is_directed() else g
#         local = set()
#         for s in seeds_vids:
#             d = np.array(g_ud.shortest_paths(source=s)[0])
#             local |= {i for i, dist in enumerate(d) if dist <= 2}
#         allow &= local

#         # seed 및 이웃은 항상 포함
#         for s in seeds_vids:
#             allow.add(s)
#             for nb in g.neighbors(s, mode="all"):
#                 allow.add(nb)
#         return allow

#     def build_pruned_subgraphs_for_topk_passages(
#         self,
#         topk_doc_ids: List[int],
#         topk_scores: List[float],
#         passage_node_idxs: List[int],
#         last_ppr_scores: Optional[np.ndarray],
#         cfg: "SubgraphConfig",
#         head_k: int = 30,
#     ) -> Tuple[List, List[int], List[int]]:
#         if not topk_doc_ids:
#             return [], [], []

#         self.cfg = cfg

#         sem = np.asarray(topk_scores, float)
#         head_idx = np.argsort(-sem)[: min(head_k, len(topk_doc_ids))]
#         head_doc_ids = [topk_doc_ids[i] for i in head_idx]
#         head_vids = IDMapper.docids_to_vids(head_doc_ids, passage_node_idxs)

#         allow_ids = self.make_allow_ids(self.g, head_vids, last_ppr_scores, cfg.allow_topk)

#         builder = SubgraphBuilder(self.g, cfg, node_scores=last_ppr_scores)
#         pruner = PPRPruner(self.g, last_ppr_scores, cfg)

#         subgraphs: List = []
#         # 진단: 루트별 노드집합 Jaccard 관측을 위한 저장
#         node_sets: List[Set[int]] = []

#         for rid in head_vids:
#             cands = builder.collect_candidates(root=rid, allow_ids=allow_ids)
#             pruned = pruner.prune(root=rid, candidate_edges=cands) if cfg.subgraph_mode == "ppr" else cands
#             sg = builder.build_subgraph(pruned, root=rid)

#             # 진단용 Jaccard: 서브그래프 노드 전역ID 집합 저장
#             node_sets.append(set([] if sg.vcount() == 0 else sg.vs["global_id"]))

#             # 메트릭 로깅(예외 방지)
#             try:
#                 row = compute_struct_metrics(
#                     sg,
#                     graph_role="subgraph",
#                     dataset_name=getattr(self, "dataset_name", "default") if hasattr(self, "dataset_name") else "default",
#                     query_id=str(getattr(self, "query_id", "unknown")) if hasattr(self, "query_id") else "unknown",
#                     root_global_id=rid,
#                     ppr_scores=last_ppr_scores,
#                     exclude_virtual=False,
#                 )
#                 row.update({
#                     "root_doc_vid": int(rid),
#                     "root_doc_id": int(head_doc_ids[len(subgraphs)]) if len(subgraphs) < len(head_doc_ids) else None,
#                     "subgraph_idx": int(len(subgraphs)),
#                     "collect_mode": self.cfg.subgraph_mode if hasattr(self, "cfg") else "unknown"
#                 })
#                 self.metrics_logger.write(row)
#             except Exception as e:
#                 logger.warning(f"[SubgraphMetrics] logging failed: {e}")

#             subgraphs.append(sg)
#             logger.debug(f"[Subgraph] mode={cfg.subgraph_mode} root={rid} cands={len(cands)} pruned={len(pruned)} v={sg.vcount()} e={sg.ecount()}")

#         # 진단 로깅: 루트 간 서브그래프 유사도(Jaccard) 평균
#         try:
#             jac_vals = []
#             for i in range(len(node_sets)):
#                 for j in range(i + 1, len(node_sets)):
#                     a, b = node_sets[i], node_sets[j]
#                     if not a and not b:
#                         continue
#                     inter = len(a & b)
#                     union = len(a | b)
#                     if union > 0:
#                         jac_vals.append(inter / union)
#             if jac_vals:
#                 head_diag_logger = MetricsLogger(save_dir="./outputs/diag", dataset_name="diag")
#                 head_diag_logger.write({
#                     "dataset": getattr(self, "dataset_name", "default"),
#                     "query_id": getattr(self, "query_id", "unknown"),
#                     "graph_role": "head_diag",
#                     "subgraph_nodeset_jaccard_mean": float(np.mean(jac_vals)),
#                     "subgraph_nodeset_jaccard_p90": float(np.percentile(jac_vals, 90)),
#                     "struct_k": int(head_k),
#                     "mode": str(cfg.subgraph_mode),
#                 })
#         except Exception:
#             pass

#         return subgraphs, head_doc_ids, head_vids

#     # ---- 구조 점수(탄젠트 공간) ----
#     def _struct_scores_tangent(
#         self,
#         query_graph,
#         subgraphs: List,
#         embedder: "HyperbolicEmbedder",
#         scorer: "StructralSimilarity",
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> np.ndarray:
#         with torch.no_grad():
#             tq = embedder.embed_and_logmap0(
#                 query_graph,
#                 ppr_scores=ppr_scores,
#                 passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )

#             tdocs = [
#                 embedder.embed_and_logmap0(
#                     g,
#                     ppr_scores=ppr_scores,
#                     passage_embeddings=passage_embeddings,
#                     entity_embeddings=entity_embeddings,
#                     use_semantic_features=use_semantic_features,
#                     use_treerep_metric=use_treerep_metric
#                 ) for g in subgraphs
#             ]

#             # 진단: pairwise 분포
#             try:
#                 if not tdocs:
#                     return np.array([])
#                 T = torch.stack(tdocs)
#                 C = F.cosine_similarity(T.unsqueeze(1), T.unsqueeze(0), dim=-1)
#                 mask = ~torch.eye(C.size(0), dtype=bool, device=C.device)
#                 vals = C[mask]
#                 logger.debug(
#                     "[EMB-DIAG][raw] tdocs_norm_mean=%.4f tdocs_std=%.4f | pairwise_cos_mean=%.4f p95=%.4f",
#                     T.norm(dim=1).mean().item(),
#                     T.std().item(),
#                     vals.mean().item(),
#                     vals.kthvalue(max(1, int(0.95 * len(vals))))[0].item()
#                 )
#                 cos_raw = scorer.compute_listwise([tq for _ in tdocs], tdocs)
#                 q05, q50, q95 = torch.quantile(cos_raw, torch.tensor([0.05, 0.5, 0.95], device=cos_raw.device))
#                 logger.debug(
#                     "[EMB-DIAG][raw] qvdoc_cos min=%.4f max=%.4f | p05=%.4f p50=%.4f p95=%.4f",
#                     cos_raw.min().item(),
#                     cos_raw.max().item(),
#                     q05.item(),
#                     q50.item(),
#                     q95.item()
#                 )
#             except Exception as e:
#                 logger.debug("[EMB-DIAG] raw diag skipped: %s", e)

#             # 공통 표준화 후 유사도
#             tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)
#             q_list = [tq_std for _ in range(len(tdocs_std))]
#             struct_head = scorer.compute_listwise(q_list, tdocs_std)
#             return _to_1d_numpy(struct_head)

#     # ---- 구조 점수(Lorentz 거리) ----
#     def _struct_scores_lorentz(
#         self,
#         query_graph,
#         subgraphs: List,
#         embedder: "HyperbolicEmbedder",
#         temp: float = 2.0,
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> np.ndarray:
#         with torch.no_grad():
#             qL = embedder.embed_lorentz(
#                 query_graph,
#                 ppr_scores=ppr_scores,
#                 passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#             dL = [
#                 embedder.embed_lorentz(
#                     g,
#                     ppr_scores=ppr_scores,
#                     passage_embeddings=passage_embeddings,
#                     entity_embeddings=entity_embeddings,
#                     use_semantic_features=use_semantic_features,
#                     use_treerep_metric=use_treerep_metric
#                 )
#                 for g in subgraphs
#             ]

#             qL = qL if qL is not None else torch.zeros(embedder.embed_dim, device=embedder.device)
#             dL_tensors = []
#             for x in dL:
#                 if x is None:
#                     dL_tensors.append(torch.zeros(embedder.embed_dim, device=embedder.device))
#                 else:
#                     dL_tensors.append(x)

#             dists2 = torch.stack([
#                 embedder.manifold.dist2(qL.unsqueeze(0), x.unsqueeze(0)).squeeze() for x in dL_tensors
#             ])
#             d2 = _to_1d_numpy(dists2)
#             sim = -d2
#             if sim.size:
#                 norm_sim = StructralSimilarity().normalize(sim)
#                 sim = 1.0 / (1.0 + np.exp(-((norm_sim - 0.5) / (1.0 / temp))))
#             return sim

#     # ---- 구조 점수(노드 레벨, Max-over-Query) ----
#     def _struct_scores_node_level(
#         self,
#         query_graph,
#         subgraphs: List,
#         embedder: "HyperbolicEmbedder",
#         ppr_scores: Optional[np.ndarray] = None,
#         passage_embeddings: Optional[np.ndarray] = None,
#         entity_embeddings: Optional[np.ndarray] = None,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False
#     ) -> np.ndarray:
#         with torch.no_grad():
#             q_node_embeddings = embedder.embed_nodes_lorentz(
#                 query_graph,
#                 ppr_scores=ppr_scores,
#                 passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#             if q_node_embeddings.numel() == 0:
#                 return np.zeros(len(subgraphs))

#             struct_scores = []
#             for doc_g in subgraphs:
#                 d_node_embeddings = embedder.embed_nodes_lorentz(
#                     doc_g,
#                     ppr_scores=ppr_scores,
#                     passage_embeddings=passage_embeddings,
#                     entity_embeddings=entity_embeddings,
#                     use_semantic_features=use_semantic_features,
#                     use_treerep_metric=use_treerep_metric
#                 )
#                 if d_node_embeddings.numel() == 0:
#                     struct_scores.append(0.0)
#                     continue

#                 # 탄젠트 공간에서 코사인 유사도
#                 tq = embedder.logmap0_safe(q_node_embeddings)
#                 td = embedder.logmap0_safe(d_node_embeddings)

#                 similarity_matrix = F.cosine_similarity(tq.unsqueeze(1), td.unsqueeze(0), dim=-1)
#                 max_sim_per_query_node = torch.max(similarity_matrix, dim=1)[0]
#                 score = torch.mean(max_sim_per_query_node)
#                 struct_scores.append(score.item())

#             return np.array(struct_scores)

#     # ---- 구조 기반 재랭킹 ----
#     def rerank_with_structure(
#         self,
#         query_graph: Optional["ig.Graph"],
#         passage_topk_ids: List[int],
#         passage_topk_scores: List[float],
#         passage_node_idxs: List[int],
#         last_ppr_scores: Optional[np.ndarray],
#         cfg: "SubgraphConfig",
#         embedder: HyperbolicEmbedder,
#         scorer: StructralSimilarity,
#         passage_embeddings: np.ndarray,
#         entity_embeddings: np.ndarray,
#         use_semantic_features: bool = True,
#         use_treerep_metric: bool = False,
#         alpha: float = 0.8,
#         struct_k: int = 200,
#         device: str = "cpu",
#         save_cb: Optional[Callable[[object, str, dict], None]] = None,
#         save_ctx: Optional[dict] = None,
#         sim_cac_mode: Literal["tangent", "lorentz", "node_level"] = "tangent",
#         lorentz_temp: float = 2.0,
#     ) -> Tuple[List[int], List[float]]:
#         passage_topk_ids = list(np.asarray(passage_topk_ids).tolist())
#         passage_topk_scores = list(np.asarray(passage_topk_scores, dtype=float).tolist())

#         if query_graph is None or len(passage_topk_ids) == 0:
#             return passage_topk_ids, passage_topk_scores

#         subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
#             passage_topk_ids,
#             passage_topk_scores,
#             passage_node_idxs,
#             last_ppr_scores,
#             cfg,
#             head_k=struct_k
#         )

#         if not subgraphs:
#             return passage_topk_ids, passage_topk_scores

#         # 구조 점수 계산
#         if sim_cac_mode == "tangent":
#             struct_head = self._struct_scores_tangent(
#                 query_graph, subgraphs, embedder, scorer,
#                 ppr_scores=last_ppr_scores,
#                 passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#         elif sim_cac_mode == "lorentz":
#             struct_head = self._struct_scores_lorentz(
#                 query_graph, subgraphs, embedder, temp=lorentz_temp,
#                 ppr_scores=last_ppr_scores,
#                 passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#             # 이미 0~1로 squashing 했음
#         elif sim_cac_mode == "node_level":
#             struct_head = self._struct_scores_node_level(
#                 query_graph, subgraphs, embedder,
#                 ppr_scores=last_ppr_scores,
#                 passage_embeddings=passage_embeddings,
#                 entity_embeddings=entity_embeddings,
#                 use_semantic_features=use_semantic_features,
#                 use_treerep_metric=use_treerep_metric
#             )
#         else:
#             raise ValueError(f"Unknown mode: {sim_cac_mode}")

#         if struct_head.size == 0:
#             return passage_topk_ids, passage_topk_scores

#         # semantic [0,1]
#         sem = np.asarray(passage_topk_scores, float)
#         head_idx = np.argsort(-sem)[:len(subgraphs)]  # semantic 상위 = subgraphs 개수만큼
#         sem_head = sem[head_idx]
#         sem_norm_head = (sem_head - sem_head.min()) / (sem_head.max() - sem_head.min() + 1e-9)

#         # structural [0,1]
#         s = np.asarray(struct_head, float)
#         s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
#         s = (s - s.min()) / (s.max() - s.min() + 1e-9)

#         # late fusion
#         final_head = alpha * sem_norm_head + (1 - alpha) * s

#         # head 영역 재정렬
#         head_pairs = sorted(
#             zip([passage_topk_ids[i] for i in head_idx], final_head),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         # 진단 로그
#         diag = {
#             "struct_raw_min": float(np.min(struct_head)) if struct_head.size > 0 else 0.0,
#             "struct_raw_max": float(np.max(struct_head)) if struct_head.size > 0 else 0.0,
#             "struct_norm_min": float(np.min(s)) if s.size > 0 else 0.0,
#             "struct_norm_max": float(np.max(s)) if s.size > 0 else 0.0,
#             "sem_norm_min": float(np.min(sem_norm_head)) if sem_norm_head.size > 0 else 0.0,
#             "sem_norm_max": float(np.max(sem_norm_head)) if sem_norm_head.size > 0 else 0.0,
#             "final_min": float(np.min(final_head)) if final_head.size > 0 else 0.0,
#             "final_max": float(np.max(final_head)) if final_head.size > 0 else 0.0
#         }
#         try:
#             sem_head = sem[head_idx]
#             diag['sem_struct_spearmanr'] = float(spearmanr(sem_head, struct_head).correlation)
#             diag['sem_struct_kendalltau'] = float(kendalltau(sem_head, struct_head).correlation)
#         except Exception:
#             diag['sem_struct_spearmanr'] = np.nan
#             diag['sem_struct_kendalltau'] = np.nan

#         diag_msg = f"[RERANK] head_k={struct_k} alpha={alpha} | {diag}"
#         if len(diag_msg) > 1000:
#             diag_msg = diag_msg[:1000] + "..."
#         logger.info(diag_msg)

#         try:
#             head_diag_logger = MetricsLogger(save_dir="./outputs/diag", dataset_name="diag")
#             head_diag_logger.write({
#                 "dataset": getattr(self, "dataset_name", "default"),
#                 "query_id": getattr(self, "query_id", "unknown"),
#                 "graph_role": "head_diag",
#                 "struct_head_min": diag["struct_raw_min"],
#                 "struct_head_max": diag["struct_raw_max"],
#                 "sem_struct_spearmanr": diag["sem_struct_spearmanr"],
#                 "sem_struct_kendalltau": diag["sem_struct_kendalltau"],
#                 "alpha": float(alpha),
#                 "mode": str(sim_cac_mode),
#                 "struct_k": int(struct_k),
#             })
#         except Exception:
#             pass

#         # 나머지 tail은 원래 semantic 점수 순으로 뒤에 유지
#         orig_scores = dict(zip(passage_topk_ids, passage_topk_scores))
#         final_scores = {doc_id: score for doc_id, score in head_pairs}

#         merged_ids, merged_scores = [], []
#         for doc_id, score in head_pairs:
#             merged_ids.append(doc_id)
#             merged_scores.append(score)

#         remaining_ids = [did for did in passage_topk_ids if did not in final_scores]
#         remaining_pairs = sorted(
#             [(did, orig_scores[did]) for did in remaining_ids],
#             key=lambda x: x[1],
#             reverse=True
#         )
#         for doc_id, score in remaining_pairs:
#             merged_ids.append(doc_id)
#             merged_scores.append(score)

#         return merged_ids, merged_scores


# ===============================================================
# -----------------------------------------------------------
# ==============================================================
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
            'eig_centrality': self._get_eig_centrality,
            # 'clustering': self._get_clustering,
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

    # ---------------------------------------------------------------------
    # 🌟 수정 1: TreeRep 거리 행렬 계산 로직 제거 및 함수명 변경
    # ---------------------------------------------------------------------
    def _get_treerep_tree(self, g: ig.Graph) -> Tuple[np.ndarray, nx.Graph]:
        """TreeRep 학습을 통해 트리를 얻고, TreeRep 거리 행렬 계산은 생략합니다."""
        if g.vcount() < 2:
            # 0 또는 1개의 노드를 가진 그래프는 거리를 계산할 필요가 없으며, 빈 트리를 반환합니다.
            return np.zeros((g.vcount(), g.vcount()), dtype=np.float32), nx.Graph()

        # 1. APSP 계산 (고비용)
        # igraph.shortest_paths()는 All-Pairs Shortest Path 행렬을 반환합니다.
        distances = np.array(g.shortest_paths(weights='weight' if 'weight' in g.es.attributes() else None))
        
        # 무한대 처리
        if np.isinf(distances).any():
            max_finite = np.max(distances[np.isfinite(distances)]) if np.isfinite(distances).any() else 0.0
            distances[np.isinf(distances)] = max_finite + 1.0

        # 2. TreeRep 학습 (고비용)
        # TreeRep 인스턴스 생성 및 최적의 트리 구조 학습
        try:
            treerep_instance = TreeRep(d=distances)
            treerep_instance.learn_tree()
            learned_tree = treerep_instance.G
        except Exception as e:
             # TreeRep 학습 실패 시, 빈 NetworkX 그래프를 반환하여 오류 회피
            logger.warning(f"TreeRep learning failed: {e}. Returning empty tree.")
            learned_tree = nx.Graph()


        # 3. 기존 코드의 'treerep_metric' (거리 행렬) 계산 로직 전체를 삭제했습니다. 
        #    - 목표는 트리 구조만 이용하는 것이므로, 불필요한 계산을 제거했습니다.

        # 거리 행렬 (distances)은 TreeRep 학습을 위한 입력으로만 사용되었으며, 
        # learned_tree는 피처 추출에 사용됩니다.
        return distances, learned_tree
    
    # ---------------------------------------------------------------------
    # 🌟 수정 2: 구조 피처 이름을 norm_depth 및 parent_w_norm으로 변경/확정
    # ---------------------------------------------------------------------
    def _treerep_features_from_tree(self, learned_tree: nx.Graph, g: ig.Graph) -> np.ndarray:
        n = g.vcount()
        if n == 0:
            return np.zeros((0, 8), dtype=np.float32)

        gids = np.asarray(g.vs["global_id"])

        # --- root 설정 (global_id 기준, 안전 가드) ---
        try:
            root = int(g["root_id"]) if "root_id" in g.attributes() else int(gids[0])
            if (len(gids) > 0) and (root not in set(gids)):
                root = int(gids[0])
        except Exception:
            root = int(gids[0]) if len(gids) else 0

        # 빠른 인덱싱용 매핑
        gid2loc = {int(x): i for i, x in enumerate(gids)}

        # --- 1) depth_norm ---
        depth = np.zeros(n, dtype=np.float32)
        if learned_tree.number_of_nodes() > 0 and root in learned_tree:
            dist = dict(nx.single_source_dijkstra_path_length(learned_tree, root, weight='weight'))
            for gid, i in gid2loc.items():
                depth[i] = float(dist.get(gid, 0.0))
        max_depth = float(depth.max())
        norm_depth = (depth / (max_depth + 1e-6)).astype(np.float32) if max_depth > 0 else depth

        # --- 2) leaf (차수 <= 1) ---
        deg_arr = np.zeros(n, dtype=np.float32)
        for gid, i in gid2loc.items():
            if gid in learned_tree:
                deg_arr[i] = float(learned_tree.degree(gid))
        leaf = (deg_arr <= 1.0).astype(np.float32)

        # --- 3) parent_w_norm (가까운 이웃 간선 가중치 최소값) ---
        parent_w = np.zeros(n, dtype=np.float32)
        for gid, i in gid2loc.items():
            if gid in learned_tree:
                nbrs = list(learned_tree.neighbors(gid))
                if nbrs:
                    parent_w[i] = min(learned_tree[gid][v].get('weight', 1.0) for v in nbrs)
        max_pw = float(parent_w.max())
        parent_w_norm = (parent_w / (max_pw + 1e-6)).astype(np.float32) if max_pw > 0 else parent_w

        # --- 4) is_root (루트 플래그) ---
        is_root = np.zeros(n, dtype=np.float32)
        if root in gid2loc:
            is_root[gid2loc[root]] = 1.0

        # ===== 추가 트리 기반 보강 피처 =====

        # 공통 준비: 루트 기준 방향 트리 (루트 컴포넌트만 포함)
        # learned_tree는 무방향일 수 있음 → bfs_tree로 방향성 부여(DiGraph)
        T = None
        if learned_tree.number_of_nodes() > 0 and root in learned_tree:
            try:
                T = nx.bfs_tree(learned_tree, root)  # DiGraph
            except Exception:
                T = None

        # --- 5) branch_factor_norm: 분기 강도 (deg 정규화) ---
        # (루트는 deg/max_deg, 그 외는 (deg-1)/(max_deg-1)로 해석 가능하지만
        #  간단히 deg/max_deg로 통일해도 작은 그래프에서 괜찮게 작동)
        branch_factor = deg_arr.copy()
        max_deg = float(branch_factor.max())
        branch_factor_norm = (branch_factor / (max_deg + 1e-6)).astype(np.float32) if max_deg > 0 else np.zeros(n, dtype=np.float32)

        # --- 6) subtree_size_norm: 서브트리 크기(루트 기준) ---
        subtree_size = np.zeros(n, dtype=np.float32)
        if T is not None:
            # 자식 목록 캐시
            children = {u: list(T.successors(u)) for u in T.nodes()}
            # 후위 순회로 서브트리 크기 누적
            order = list(nx.topological_sort(T)) if len(T) else []
            for u in reversed(order):
                sz = 1
                for c in children.get(u, []):
                    sz += int(subtree_size[gid2loc.get(c, -1)]) if c in gid2loc else 0
                if u in gid2loc:
                    subtree_size[gid2loc[u]] = float(sz)

            # 정규화 (해당 컴포넌트 크기 사용)
            comp_nodes = set(T.nodes())
            Ncomp = float(len(comp_nodes))
            if Ncomp > 0:
                subtree_size_norm = np.zeros(n, dtype=np.float32)
                for u in comp_nodes:
                    iu = gid2loc.get(u, None)
                    if iu is not None:
                        subtree_size_norm[iu] = subtree_size[iu] / Ncomp
            else:
                subtree_size_norm = np.zeros(n, dtype=np.float32)
        else:
            subtree_size_norm = np.zeros(n, dtype=np.float32)

        # --- 7) path_centrality_tree_norm: 트리 특화 betweenness 근사(선형) ---
        # 노드 u를 제거했을 때 생기는 컴포넌트 쌍 크기의 곱 합 Σ s*(N-s)
        path_cent = np.zeros(n, dtype=np.float32)
        if T is not None and len(T):
            comp_nodes = list(T.nodes())
            comp_set = set(comp_nodes)
            Ncomp = float(len(comp_nodes))
            if Ncomp >= 2:
                # 위에서 구한 subtree_size 이용
                # parent는 in_edges로 1개(루트 제외)
                parent_of = {}
                for u in comp_nodes:
                    preds = list(T.predecessors(u))
                    parent_of[u] = preds[0] if preds else None

                for u in comp_nodes:
                    iu = gid2loc.get(u, None)
                    if iu is None:
                        continue
                    # 자식 방향 분기들의 크기
                    parts = []
                    for c in T.successors(u):
                        ic = gid2loc.get(c, None)
                        if ic is not None:
                            sc = subtree_size[ic]
                            parts.append(sc)
                    # 부모 쪽(루트를 제외하고 나머지 한 덩어리)
                    pu = parent_of[u]
                    if pu is not None:
                        su = subtree_size[iu]
                        parts.append(Ncomp - su)
                    # Σ s*(N-s)
                    val = 0.0
                    for s in parts:
                        val += float(s * (Ncomp - s))
                    path_cent[iu] = val

                mx = float(path_cent.max())
                path_centrality_tree_norm = (path_cent / (mx + 1e-6)).astype(np.float32) if mx > 0 else np.zeros(n, dtype=np.float32)
            else:
                path_centrality_tree_norm = np.zeros(n, dtype=np.float32)
        else:
            path_centrality_tree_norm = np.zeros(n, dtype=np.float32)

        # --- 8) sibling_count_norm: 형제 수 정규화 ---
        sibling = np.zeros(n, dtype=np.float32)
        if T is not None and len(T):
            # 부모별 자식 수
            children = {u: list(T.successors(u)) for u in T.nodes()}
            # root의 형제개념은 "자기 형제"가 없으니 자식 수를 쓰거나 0으로 두는 선택지가 있음
            # 여기서는 일관성 있게: parent 있으면 (siblings = deg(parent)-1), root면 children(root) 사용
            for u in T.nodes():
                iu = gid2loc.get(u, None)
                if iu is None:
                    continue
                preds = list(T.predecessors(u))
                if preds:
                    p = preds[0]
                    siblings_cnt = max(0, len(children.get(p, [])) - 1)
                else:
                    siblings_cnt = len(children.get(u, []))  # root: 자식 수
                sibling[iu] = float(siblings_cnt)
            mxs = float(sibling.max())
            sibling_count_norm = (sibling / (mxs + 1e-6)).astype(np.float32) if mxs > 0 else np.zeros(n, dtype=np.float32)
        else:
            sibling_count_norm = np.zeros(n, dtype=np.float32)

        # ---- 최종 스택: (N, 8)
        # [0] depth_norm, [1] leaf, [2] parent_w_norm, [3] is_root,
        # [4] branch_factor_norm, [5] subtree_size_norm,
        # [6] path_centrality_tree_norm, [7] sibling_count_norm
        feats = np.stack([
            norm_depth.astype(np.float32),
            leaf.astype(np.float32),
            parent_w_norm.astype(np.float32),
            is_root.astype(np.float32),
            # branch_factor_norm.astype(np.float32),
            # subtree_size_norm.astype(np.float32),
            # path_centrality_tree_norm.astype(np.float32),
            # sibling_count_norm.astype(np.float32),
        ], axis=1)

        return feats.astype(np.float32)

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
            # 🌟 수정 3: 변경된 _get_treerep_tree 함수 호출에 맞춤
            # 1. TreeRep 기반 피처 추출 및 추가
            distances, learned_tree = self._get_treerep_tree(g) 
            feats_treerep = self._treerep_features_from_tree(learned_tree, g) 
            feature_list.append(feats_treerep)
            
            # 2. 🌟 일반 구조 피처(원본 그래프 기반) 추가
            for extractor_name, extractor_func in self.feature_extractors.items():
                # TreeRep 피처와 독립적으로 원본 그래프 g에서 추출
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
        
        # if num_features < self.embed_dim:
        #     X = np.pad(X, ((0, 0), (0, self.embed_dim - num_features)), 'constant')
        # elif num_features > self.embed_dim:
        #     X = X[:, :self.embed_dim]

        # if num_features > 0:
        #     mu = X.mean(axis=0, keepdims=True)
        #     sd = X.std(axis=0, keepdims=True)
        #     sd = np.maximum(sd, 1e-3)
        #     Z = np.clip((X - mu) / sd, -3.0, 3.0)
        # else:
        #     Z = np.zeros((g.vcount(), self.embed_dim), dtype=np.float32)
      # 🌟 A 해결책 적용: 피처 정규화 개선 (노이즈 증폭 방지)
        if num_features > 0:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True)
            
            # 🌟 분산이 0에 가까우면 Z를 0으로 설정
            if np.all(sd < 1e-6): 
                Z = np.zeros_like(X, dtype=np.float32)
            else:
                # 분산이 작은 경우 1e-6으로 클리핑 (1e-3 대신 더 작은 값 사용)
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
                # weights = np.array(g.betweenness()).astype(np.float32)
                # weights = weights / (weights.sum() + 1e-6)
                # weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
                # pooled = (x_lorentz * weights).sum(dim=0)

                # pooled = x_lorentz.mean(dim=0) 
                pooled, _ = torch.max(x_lorentz, dim=0, keepdim=False) # 👈 Max Pooling

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
    # def normalize(self, scores: List[float] | np.ndarray) -> np.ndarray:
    #     x = np.asarray(scores, dtype=float)
    #     if x.size == 0:
    #         return x
    #     lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        
    #     # 🌟 B 해결책 적용: 변별력 없을 때 0.5 대신 0을 반환 (로그 해석 명료화)
    #     range_val = hi - lo
    #     if range_val < 1e-9: 
    #         return np.zeros_like(x)
        
    #     y = (x - lo) / (range_val + 1e-9)
    #     y = np.clip(y, 0.0, 1.0)
    #     return y
        
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
        use_semantic_features: bool = True,
        use_treerep_metric: bool = False,
        alpha: float = 0.8,
        # 🌟 수정 4: struct_k의 기본값을 30으로 설정
        struct_k: int = 30, 
        device: str = "cpu",
        save_cb: Optional[Callable[[ig.Graph, str, dict], None]] = None,
        save_ctx: Optional[dict] = None,
        sim_cac_mode: Literal["tangent", "lorentz", "node_level"] = "tangent",
        lorentz_temp: float = 2.0,
    ) -> Tuple[List[int], List[float]]:

        passage_topk_ids = list(np.asarray(passage_topk_ids).tolist())
        passage_topk_scores = list(np.asarray(passage_topk_scores, dtype=float).tolist())

        if query_graph is None or len(passage_topk_ids) == 0:
            return passage_topk_ids, passage_topk_scores

        subgraphs, head_doc_ids, head_vids = self.build_pruned_subgraphs_for_topk_passages(
            passage_topk_ids, passage_topk_scores, passage_node_idxs,
            last_ppr_scores, cfg, head_k=struct_k
        )
        
        if not subgraphs:
            return passage_topk_ids, passage_topk_scores

        if sim_cac_mode == "tangent":
            struct_head = self._struct_scores_tangent(
                query_graph, subgraphs, embedder, scorer, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features,
                use_treerep_metric=use_treerep_metric
            )
        elif sim_cac_mode == "lorentz":
            struct_head = self._struct_scores_lorentz(
                query_graph, subgraphs, embedder, temp=lorentz_temp, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features,
                use_treerep_metric=use_treerep_metric
            )
        elif sim_cac_mode == "node_level":
            struct_head = self._struct_scores_node_level(
                query_graph, subgraphs, embedder, ppr_scores=last_ppr_scores,
                passage_embeddings=passage_embeddings, entity_embeddings=entity_embeddings,
                use_semantic_features=use_semantic_features,
                use_treerep_metric=use_treerep_metric
            )
        else:
            raise ValueError(f"Unknown mode: {sim_cac_mode}")

        if struct_head.size == 0:
            return passage_topk_ids, passage_topk_scores

        # sem = np.asarray(passage_topk_scores, float)
        # head_idx = np.argsort(-sem)[:len(subgraphs)]
        # sem_norm_head = scorer.normalize(sem[head_idx])
        # str_norm_head = scorer.normalize(struct_head)

        # final_head = alpha * sem_norm_head + (1 - alpha) * str_norm_head
        sem = np.asarray(passage_topk_scores, float)
        head_idx = np.argsort(-sem)[:len(subgraphs)]

        # ① semantic [0,1]
        sem_head = sem[head_idx]
        sem_norm_head = (sem_head - sem_head.min()) / (sem_head.max() - sem_head.min() + 1e-9)

        # ② structural [0,1]
        s = np.asarray(struct_head, float)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)

        # ③ 결합 (지금 코드 의미 그대로: alpha는 semantic 가중)
        final_head = alpha * sem_norm_head + (1 - alpha) * s

        head_pairs = sorted(
            zip([passage_topk_ids[i] for i in head_idx], final_head),
            key=lambda x: x[1], reverse=True
        )

        diag = {}
        diag["struct_raw_min"] = float(np.min(struct_head)) if struct_head.size > 0 else 0.0
        diag["struct_raw_max"] = float(np.max(struct_head)) if struct_head.size > 0 else 0.0
        diag["struct_norm_min"] = float(np.min(s)) if s.size > 0 else 0.0
        diag["struct_norm_max"] = float(np.max(s)) if s.size > 0 else 0.0
        diag["sem_norm_min"] = float(np.min(sem_norm_head)) if sem_norm_head.size > 0 else 0.0
        diag["sem_norm_max"] = float(np.max(sem_norm_head)) if sem_norm_head.size > 0 else 0.0

        diag["final_min"] = float(np.min(final_head)) if final_head.size > 0 else 0.0
        diag["final_max"] = float(np.max(final_head)) if final_head.size > 0 else 0.0
        
        try:
            sem_head = sem[head_idx]
            diag['sem_struct_spearmanr'] = float(spearmanr(sem_head, struct_head).correlation)
            diag['sem_struct_kendalltau'] = float(kendalltau(sem_head, struct_head).correlation)
        except Exception:
            diag['sem_struct_spearmanr'] = np.nan
            diag['sem_struct_kendalltau'] = np.nan

        diag_msg = f"[RERANK] head_k={struct_k} alpha={alpha} | {diag}"
        if len(diag_msg) > 1000: diag_msg = diag_msg[:1000] + "..."
        logger.info(diag_msg)
        
        try:
            head_diag_logger = MetricsLogger(save_dir="./outputs/diag", dataset_name="diag")
            head_diag_logger.write({
                "dataset": getattr(self, "dataset_name", "default"),
                "query_id": getattr(self, "query_id", "unknown"),
                "graph_role": "head_diag",
                "struct_head_min": diag["struct_raw_min"],
                "struct_head_max": diag["struct_raw_max"],
                "sem_struct_spearmanr": diag["sem_struct_spearmanr"],
                "sem_struct_kendalltau": diag["sem_struct_kendalltau"],
                "alpha": float(alpha),
                "mode": str(sim_cac_mode),
                "struct_k": int(struct_k),
            })
        except Exception:
            pass

        orig_scores = dict(zip(passage_topk_ids, passage_topk_scores))
        final_scores = {doc_id: score for doc_id, score in head_pairs}
        
        merged_ids, merged_scores = [], []
        
        for doc_id, score in head_pairs:
            merged_ids.append(doc_id)
            merged_scores.append(score)
            
        remaining_ids = [did for did in passage_topk_ids if did not in final_scores]
        remaining_pairs = sorted(
            [(did, orig_scores[did]) for did in remaining_ids],
            key=lambda x: x[1], reverse=True
        )
        
        for doc_id, score in remaining_pairs:
            merged_ids.append(doc_id)
            merged_scores.append(score)

        return merged_ids, merged_scores