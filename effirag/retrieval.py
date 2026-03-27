import multiprocessing as mp
import os
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import networkx as nx
import numpy as np

from .anchors import select_lexical_anchors
from .config import RetrievalConfig
from .embedding import cosine_similarity, encode_texts, rerank_sentences_by_embedding, topk_cosine_similarity
from .global_index import load_or_build_global_index, load_semantic_index
from .graph import build_document_entity_graph
from .registry import register_method
from .types import AnchorResult, RetrievalResult
from .utils import content_tokens

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

_GLOBAL_INDEX_MEMO = {}
_GLOBAL_SEMANTIC_MEMO = {}
_PPR_GRAPH_STRUCT_MEMO = {}
_PPR_PARALLEL_STRUCT = None


def _get_global_graph(cfg):
    corpus_path = str(getattr(cfg, "global_corpus_path", "") or "").strip()
    prebuilt_igraph_path = str(getattr(cfg, "prebuilt_igraph_path", "") or "").strip()
    if not corpus_path and not prebuilt_igraph_path:
        return None, {}

    cache_dir = str(getattr(cfg, "graph_cache_dir", "") or "outputs/index_cache").strip()
    force_rebuild = bool(getattr(cfg, "force_rebuild_graph_index", False))
    prebuilt_igraph_format = str(getattr(cfg, "prebuilt_igraph_format", "hipporag_pickle") or "hipporag_pickle").strip()
    prebuilt_entity_token_limit = int(getattr(cfg, "prebuilt_entity_token_limit", 6))
    openie_mode = str(getattr(cfg, "openie_mode", "llm") or "llm").strip().lower()
    openie_model_name = str(getattr(cfg, "openie_model_name", "") or "").strip()
    openie_text_max_chars = int(getattr(cfg, "openie_text_max_chars", 2200))
    openie_max_new_tokens = int(getattr(cfg, "openie_max_new_tokens", 256))
    openie_local_files_only = bool(getattr(cfg, "openie_local_files_only", True))
    openie_retry_attempts = int(getattr(cfg, "openie_retry_attempts", 3))
    openie_retry_backoff_sec = float(getattr(cfg, "openie_retry_backoff_sec", 0.2))
    openie_error_sample_limit = int(getattr(cfg, "openie_error_sample_limit", 20))
    openie_api_base_url = str(getattr(cfg, "openie_api_base_url", "") or "").strip()
    openie_api_key = str(getattr(cfg, "openie_api_key", "") or "").strip()
    openie_api_timeout_sec = float(getattr(cfg, "openie_api_timeout_sec", 120.0))
    openie_parallel_workers = int(getattr(cfg, "openie_parallel_workers", 4))
    openie_log_every = int(getattr(cfg, "openie_log_every", 200))
    embedding_enabled = bool(getattr(cfg, "embedding_enabled", False))
    embedding_model_name = str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2")
    embedding_batch_size = int(getattr(cfg, "embedding_batch_size", 16))
    embedding_max_length = int(getattr(cfg, "embedding_max_length", 256))
    embedding_text_max_chars = int(getattr(cfg, "embedding_text_max_chars", 600))
    memo_key = (
        str(Path(corpus_path).resolve()) if corpus_path else "",
        str(Path(prebuilt_igraph_path).resolve()) if prebuilt_igraph_path else "",
        prebuilt_igraph_format,
        prebuilt_entity_token_limit,
        embedding_enabled,
        embedding_model_name,
        embedding_batch_size,
        embedding_max_length,
        embedding_text_max_chars,
        str(Path(cache_dir).resolve()),
        openie_mode,
        openie_model_name,
        openie_text_max_chars,
        openie_max_new_tokens,
        openie_local_files_only,
        openie_retry_attempts,
        openie_retry_backoff_sec,
        openie_error_sample_limit,
        openie_api_base_url,
        openie_api_key,
        openie_api_timeout_sec,
        openie_parallel_workers,
        openie_log_every,
    )

    if (not force_rebuild) and memo_key in _GLOBAL_INDEX_MEMO:
        graph, meta = _GLOBAL_INDEX_MEMO[memo_key]
        meta = dict(meta)
        meta["memory_cache_hit"] = True
        return graph, meta

    graph, meta = load_or_build_global_index(
        corpus_path=corpus_path,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        prebuilt_igraph_path=prebuilt_igraph_path,
        prebuilt_igraph_format=prebuilt_igraph_format,
        prebuilt_entity_token_limit=prebuilt_entity_token_limit,
        embedding_enabled=embedding_enabled,
        embedding_model_name=embedding_model_name,
        embedding_batch_size=embedding_batch_size,
        embedding_max_length=embedding_max_length,
        embedding_text_max_chars=embedding_text_max_chars,
        openie_mode=openie_mode,
        openie_model_name=openie_model_name,
        openie_text_max_chars=openie_text_max_chars,
        openie_max_new_tokens=openie_max_new_tokens,
        openie_local_files_only=openie_local_files_only,
        openie_retry_attempts=openie_retry_attempts,
        openie_retry_backoff_sec=openie_retry_backoff_sec,
        openie_error_sample_limit=openie_error_sample_limit,
        openie_api_base_url=openie_api_base_url,
        openie_api_key=openie_api_key,
        openie_api_timeout_sec=openie_api_timeout_sec,
        openie_parallel_workers=openie_parallel_workers,
        openie_log_every=openie_log_every,
    )
    _GLOBAL_INDEX_MEMO[memo_key] = (graph, dict(meta))
    meta = dict(meta)
    meta["memory_cache_hit"] = False
    return graph, meta


def _get_graph_struct(g):
    key = id(g)
    cached = _PPR_GRAPH_STRUCT_MEMO.get(key)
    if cached is not None:
        return cached

    nodes = list(g.nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    neighbors = [[] for _ in range(len(nodes))]
    degrees = [0 for _ in range(len(nodes))]
    for idx, node in enumerate(nodes):
        nbrs = [node_to_idx[nbr] for nbr in g.neighbors(node) if nbr in node_to_idx]
        neighbors[idx] = nbrs
        degrees[idx] = len(nbrs)
    dangling = [idx for idx, deg in enumerate(degrees) if deg == 0]

    struct = {
        "nodes": nodes,
        "node_to_idx": node_to_idx,
        "neighbors": neighbors,
        "degrees": degrees,
        "dangling": dangling,
        "n": len(nodes),
    }
    _PPR_GRAPH_STRUCT_MEMO[key] = struct
    return struct


def _resolve_ppr_engine(cfg, graph_scope, graph):
    raw = str(getattr(cfg, "ppr_engine", "auto") or "auto").strip().lower()
    if raw in {"power", "mc"}:
        return raw

    # auto: keep small local graphs in exact mode, switch large global/prebuilt graphs to stochastic MC.
    if graph_scope in {"global_corpus", "prebuilt_igraph"} and graph.number_of_nodes() >= 50000:
        return "mc"
    return "power"


def _build_anchor_subgraph(g, anchors, hops, max_nodes):
    if int(hops) <= 0 or int(max_nodes) <= 0 or not anchors:
        return g

    visited = set()
    q = deque()
    for anchor in anchors:
        if anchor in g and anchor not in visited:
            visited.add(anchor)
            q.append((anchor, 0))

    while q and len(visited) < int(max_nodes):
        node, depth = q.popleft()
        if depth >= int(hops):
            continue
        for nbr in g.neighbors(node):
            if nbr in visited:
                continue
            visited.add(nbr)
            if len(visited) >= int(max_nodes):
                break
            q.append((nbr, depth + 1))

    if not visited or len(visited) >= g.number_of_nodes():
        return g
    return g.subgraph(list(visited)).copy()


def _power_ppr_idx(struct, source_idx, alpha, max_iter, tol):
    n = int(struct["n"])
    if n <= 0:
        return {}

    teleport = float(alpha)
    damping = 1.0 - float(alpha)
    neighbors = struct["neighbors"]
    degrees = struct["degrees"]
    dangling = struct["dangling"]

    ranks = [0.0 for _ in range(n)]
    ranks[source_idx] = 1.0

    for _ in range(max(1, int(max_iter))):
        new_ranks = [0.0 for _ in range(n)]
        new_ranks[source_idx] = teleport

        if dangling:
            dangling_mass = damping * sum(ranks[idx] for idx in dangling)
            if dangling_mass > 0.0:
                new_ranks[source_idx] += dangling_mass

        for node_idx, prev_score in enumerate(ranks):
            deg = degrees[node_idx]
            if deg <= 0 or prev_score <= 0.0:
                continue
            share = damping * prev_score / float(deg)
            for nbr_idx in neighbors[node_idx]:
                new_ranks[nbr_idx] += share

        err = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
        ranks = new_ranks
        if err < n * float(tol):
            break

    norm = sum(ranks)
    if norm <= 0.0:
        return {source_idx: 1.0}
    return {idx: (score / norm) for idx, score in enumerate(ranks) if score > 0.0}


def _mc_ppr_idx(struct, source_idx, alpha, num_walks, max_steps, seed, min_score):
    n = int(struct["n"])
    if n <= 0:
        return {}

    neighbors = struct["neighbors"]
    rng = random.Random(int(seed))
    walk_count = max(1, int(num_walks))
    step_cap = max(1, int(max_steps))
    teleport = float(alpha)

    visits = {}
    total = 0
    for _ in range(walk_count):
        cur = source_idx
        visits[cur] = visits.get(cur, 0) + 1
        total += 1
        for _ in range(step_cap):
            if rng.random() < teleport:
                cur = source_idx
            else:
                nbrs = neighbors[cur]
                if not nbrs:
                    cur = source_idx
                else:
                    cur = nbrs[rng.randrange(len(nbrs))]
            visits[cur] = visits.get(cur, 0) + 1
            total += 1

    if total <= 0:
        return {source_idx: 1.0}
    cutoff = max(0.0, float(min_score))
    return {idx: (cnt / float(total)) for idx, cnt in visits.items() if (cnt / float(total)) >= cutoff}


def _mc_task(payload):
    if _PPR_PARALLEL_STRUCT is None:
        raise RuntimeError("MC parallel worker has no graph structure.")
    source_idx, alpha, num_walks, max_steps, seed, min_score = payload
    return _mc_ppr_idx(
        struct=_PPR_PARALLEL_STRUCT,
        source_idx=int(source_idx),
        alpha=float(alpha),
        num_walks=int(num_walks),
        max_steps=int(max_steps),
        seed=int(seed),
        min_score=float(min_score),
    )


def _personalized_pagerank(g, source, alpha, max_iter=100, tol=1.0e-6, min_score=0.0):
    if source not in g:
        return {}
    struct = _get_graph_struct(g)
    source_idx = struct["node_to_idx"].get(source)
    if source_idx is None:
        return {}
    idx_scores = _power_ppr_idx(struct, source_idx=source_idx, alpha=alpha, max_iter=max_iter, tol=tol)
    nodes = struct["nodes"]
    cutoff = max(0.0, float(min_score))
    return {nodes[idx]: score for idx, score in idx_scores.items() if score >= cutoff}


def _compute_ppr_batch(g, sources, cfg, engine, run_seed, show_progress, desc):
    struct = _get_graph_struct(g)
    nodes = struct["nodes"]
    min_score = float(getattr(cfg, "ppr_min_score", 0.0))
    alpha = float(getattr(cfg, "ppr_alpha", 0.15))

    if not sources:
        return {}

    if engine == "mc":
        tasks = []
        seed_base = int(getattr(cfg, "random_seed", 42)) * 1000003 + int(run_seed) * 10007
        for s_idx, source in enumerate(sources):
            source_idx = struct["node_to_idx"].get(source)
            if source_idx is None:
                tasks.append(None)
                continue
            tasks.append(
                (
                    int(source_idx),
                    alpha,
                    int(getattr(cfg, "ppr_mc_walks", 512)),
                    int(getattr(cfg, "ppr_mc_max_steps", 24)),
                    seed_base + s_idx * 7919,
                    min_score,
                )
            )

        use_parallel = False
        ppr_workers = max(1, int(getattr(cfg, "ppr_parallel_workers", 1)))
        # Avoid nested process pools when dataset-level workers are already used.
        if int(getattr(cfg, "num_workers", 1)) <= 1 and ppr_workers > 1:
            try:
                use_parallel = mp.get_start_method(allow_none=True) == "fork"
            except Exception:
                use_parallel = False

        outputs = []
        if use_parallel:
            global _PPR_PARALLEL_STRUCT
            _PPR_PARALLEL_STRUCT = struct
            valid_tasks = [task for task in tasks if task is not None]
            with ProcessPoolExecutor(max_workers=ppr_workers) as ex:
                valid_outputs = list(
                    tqdm(
                        ex.map(_mc_task, valid_tasks),
                        total=len(valid_tasks),
                        desc=desc,
                        leave=False,
                        disable=not show_progress,
                    )
                )
            _PPR_PARALLEL_STRUCT = None
            it = iter(valid_outputs)
            for task in tasks:
                outputs.append({} if task is None else next(it))
        else:
            for task in tqdm(
                tasks,
                total=len(tasks),
                desc=desc,
                leave=False,
                disable=not show_progress,
            ):
                if task is None:
                    outputs.append({})
                else:
                    outputs.append(_mc_task(task) if _PPR_PARALLEL_STRUCT is not None else _mc_ppr_idx(struct, *task))

        mapped = {}
        for source, idx_scores in zip(sources, outputs):
            mapped[source] = {nodes[idx]: score for idx, score in idx_scores.items()}
        return mapped

    mapped = {}
    for source in tqdm(
        sources,
        total=len(sources),
        desc=desc,
        leave=False,
        disable=not show_progress,
    ):
        if source not in struct["node_to_idx"]:
            mapped[source] = {}
            continue
        idx_scores = _power_ppr_idx(
            struct=struct,
            source_idx=int(struct["node_to_idx"][source]),
            alpha=alpha,
            max_iter=int(getattr(cfg, "ppr_power_max_iter", 100)),
            tol=float(getattr(cfg, "ppr_power_tol", 1.0e-6)),
        )
        mapped[source] = {nodes[idx]: score for idx, score in idx_scores.items() if score >= min_score}
    return mapped


def _topk_nodes(scores, k):
    if k <= 0:
        return []
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def _stochastic_perturb_graph(g, rng, drop_prob):
    if float(drop_prob) <= 0.0:
        return g
    h = g.copy()
    if h.number_of_edges() == 0:
        return h

    for u, v in list(h.edges()):
        if rng.random() < drop_prob:
            h.remove_edge(u, v)

    if h.number_of_edges() == 0:
        return g.copy()
    return h


def _min_set_distance(g, node, seeds, tau):
    if not seeds:
        return tau

    best = tau
    for z in seeds:
        if z not in g or node not in g:
            continue
        try:
            dist = nx.shortest_path_length(g, node, z)
            if dist < best:
                best = dist
        except nx.NetworkXNoPath:
            continue
    return min(best, tau)


def _greedy_seed_set(g, candidates, weights, seed_k, tau):
    selected = set()
    remaining = set(candidates)

    while len(selected) < seed_k and remaining:
        best_seed = None
        best_obj = float("inf")

        for z in remaining:
            trial = selected | {z}
            obj = 0.0
            for u in candidates:
                obj += weights.get(u, 0.0) * _min_set_distance(g, u, trial, tau)
            if obj < best_obj:
                best_obj = obj
                best_seed = z

        if best_seed is None:
            break

        selected.add(best_seed)
        remaining.remove(best_seed)

    return selected


def _pair_support(anchor_scores, seed_scores, node):
    return sqrt(anchor_scores.get(node, 0.0) * seed_scores.get(node, 0.0))


def _ordered_unique(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalize_map(scores):
    if not scores:
        return {}
    vals = [float(v) for v in scores.values()]
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        if hi > 0.0:
            return {k: 1.0 for k in scores}
        return {k: 0.0 for k in scores}
    span = hi - lo
    return {k: (float(v) - lo) / span for k, v in scores.items()}


def _node_semantic_text(g, node):
    if node not in g:
        return ""
    data = g.nodes[node]
    node_type = str(data.get("node_type", "") or "")
    if node_type == "sentence":
        title = str(data.get("title", "") or "").strip()
        text = str(data.get("text", data.get("content", "")) or "").strip()
        if title:
            return f"Title: {title}\nPassage: {text}".strip()
        return text
    if node_type == "entity":
        token = str(data.get("token", "") or "").strip()
        content = str(data.get("content", "") or "").strip()
        label = token or content or str(node).split("::", 1)[-1]
        support = []
        for nbr in g.neighbors(node):
            nbr_data = g.nodes[nbr]
            if nbr_data.get("node_type") == "sentence":
                txt = str(nbr_data.get("text", "") or "").strip()
                if txt:
                    support.append(txt)
            if len(support) >= 2:
                break
        parts = [f"Entity: {label}"]
        if support:
            parts.append("Context: " + " ".join(support))
        return "\n".join(parts).strip()
    if node_type == "relation":
        subj = str(data.get("subject", "") or "").strip()
        pred = str(data.get("predicate", "") or "").strip()
        obj = str(data.get("object", "") or "").strip()
        return " ".join([x for x in [subj, pred, obj] if x]).strip()
    return str(node)


def _semantic_state_cache_key(global_index_meta):
    semantic_meta = (global_index_meta or {}).get("semantic_index", {}) or {}
    return (
        str(semantic_meta.get("index_dir", "")),
        str(semantic_meta.get("entity_embeddings_path", "")),
        str(semantic_meta.get("chunk_embeddings_path", "")),
        str(semantic_meta.get("entity_ids_path", "")),
        str(semantic_meta.get("chunk_ids_path", "")),
        str(semantic_meta.get("model_name", "")),
    )


def _get_global_semantic_state(global_index_meta):
    semantic_meta = (global_index_meta or {}).get("semantic_index", {}) or {}
    if not bool(semantic_meta.get("enabled", False)):
        return None
    key = _semantic_state_cache_key(global_index_meta)
    if key in _GLOBAL_SEMANTIC_MEMO:
        return _GLOBAL_SEMANTIC_MEMO[key]
    try:
        state = load_semantic_index(semantic_meta, mmap_mode="r")
    except Exception:
        state = None
    if state is not None:
        _GLOBAL_SEMANTIC_MEMO[key] = state
    return state


def _query_embedding(question, cfg):
    model_name = str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2")
    out = encode_texts(
        texts=[question],
        model_name=model_name,
        batch_size=int(getattr(cfg, "embedding_batch_size", 16)),
        max_length=int(getattr(cfg, "embedding_max_length", 256)),
        max_chars=int(getattr(cfg, "embedding_text_max_chars", 600)),
        instruction="Given a question, retrieve relevant phrases that are mentioned in this question.",
    )
    if not out.get("ok", False):
        return None, str(out.get("error", "query_embedding_failed"))
    vectors = out.get("vectors", []) or []
    if not vectors:
        return None, "query_embedding_empty"
    qvec = np.asarray(vectors[0], dtype=np.float32)
    norm = float(np.linalg.norm(qvec))
    if norm <= 0.0:
        return None, "query_embedding_zero_norm"
    return qvec / norm, ""


def _semantic_top_candidates(query_vec, semantic_state, cfg):
    if semantic_state is None or query_vec is None:
        return [], {}
    topn = max(0, int(getattr(cfg, "semantic_topn", 50)))
    if topn <= 0:
        return [], {}
    scan_bs = max(128, int(getattr(cfg, "semantic_scan_batch_size", 8192)))
    scores = {}
    ent_idx, ent_scores = topk_cosine_similarity(
        query_vector=query_vec,
        matrix=semantic_state.get("entity_embeddings"),
        topn=topn,
        scan_batch_size=scan_bs,
    )
    for idx, score in zip(ent_idx.tolist(), ent_scores.tolist()):
        if idx < 0 or idx >= len(semantic_state.get("entity_ids", [])):
            continue
        node = str(semantic_state["entity_ids"][idx])
        scores[node] = max(float(score), scores.get(node, -1.0))

    chunk_idx, chunk_scores = topk_cosine_similarity(
        query_vector=query_vec,
        matrix=semantic_state.get("chunk_embeddings"),
        topn=topn,
        scan_batch_size=scan_bs,
    )
    for idx, score in zip(chunk_idx.tolist(), chunk_scores.tolist()):
        if idx < 0 or idx >= len(semantic_state.get("chunk_ids", [])):
            continue
        node = str(semantic_state["chunk_ids"][idx])
        scores[node] = max(float(score), scores.get(node, -1.0))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    return [node for node, _ in ranked], {node: float(score) for node, score in ranked}


def _load_candidate_vectors(nodes, g, cfg, semantic_state):
    vectors = {}
    pending_nodes = []
    pending_texts = []
    for node in nodes:
        vec = None
        if semantic_state is not None and node in g:
            node_type = g.nodes[node].get("node_type")
            if node_type == "entity":
                idx = semantic_state.get("entity_id_to_idx", {}).get(str(node))
                if idx is not None:
                    vec = np.asarray(semantic_state["entity_embeddings"][idx], dtype=np.float32)
            elif node_type == "sentence":
                idx = semantic_state.get("chunk_id_to_idx", {}).get(str(node))
                if idx is not None:
                    vec = np.asarray(semantic_state["chunk_embeddings"][idx], dtype=np.float32)
        if vec is not None and vec.size > 0:
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vectors[node] = vec / norm
                continue

        text = _node_semantic_text(g, node)
        if text:
            pending_nodes.append(node)
            pending_texts.append(text)

    if pending_nodes:
        out = encode_texts(
            texts=pending_texts,
            model_name=str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2"),
            batch_size=int(getattr(cfg, "embedding_batch_size", 16)),
            max_length=int(getattr(cfg, "embedding_max_length", 256)),
            max_chars=int(getattr(cfg, "embedding_text_max_chars", 600)),
            instruction="",
        )
        if out.get("ok", False):
            for node, vec in zip(pending_nodes, out.get("vectors", []) or []):
                arr = np.asarray(vec, dtype=np.float32)
                if arr.size <= 0:
                    continue
                norm = float(np.linalg.norm(arr))
                if norm <= 0.0:
                    continue
                vectors[node] = arr / norm
            return vectors, ""
        return vectors, str(out.get("error", "candidate_embedding_failed"))
    return vectors, ""


def _entity_support_similarity(g, node, query_sim_map):
    if node not in g:
        return 0.0
    if g.nodes[node].get("node_type") != "entity":
        return float(query_sim_map.get(node, 0.0))

    best = 0.0
    for nbr in g.neighbors(node):
        nbr_type = g.nodes[nbr].get("node_type")
        if nbr_type == "sentence":
            best = max(best, float(query_sim_map.get(nbr, 0.0)))
        elif nbr_type == "relation":
            for nbr2 in g.neighbors(nbr):
                if g.nodes[nbr2].get("node_type") == "sentence":
                    best = max(best, float(query_sim_map.get(nbr2, 0.0)))
    return best


def _weighted_mean(score_map, keys, weight_map=None):
    if not keys:
        return 0.0
    if not weight_map:
        vals = [float(score_map.get(k, 0.0)) for k in keys]
        return float(sum(vals) / max(len(vals), 1))

    total_w = 0.0
    total_v = 0.0
    for key in keys:
        w = max(0.0, float(weight_map.get(key, 0.0)))
        v = float(score_map.get(key, 0.0))
        total_w += w
        total_v += w * v
    if total_w <= 0.0:
        vals = [float(score_map.get(k, 0.0)) for k in keys]
        return float(sum(vals) / max(len(vals), 1))
    return float(total_v / total_w)


def _seed_hybrid_scores(candidates, agg_scores, query_sim, anchor_sim, support_sim, cfg):
    graph_raw = {node: float(agg_scores.get(node, 0.0)) for node in candidates}
    graph_norm = _normalize_map(graph_raw)

    semantic_raw = {}
    for node in candidates:
        sem = max(float(query_sim.get(node, 0.0)), float(support_sim.get(node, 0.0)))
        semantic_raw[node] = 0.5 * (sem + 1.0)
    semantic_norm = _normalize_map(semantic_raw)

    anchor_raw = {node: 0.5 * (float(anchor_sim.get(node, 0.0)) + 1.0) for node in candidates}
    anchor_norm = _normalize_map(anchor_raw)

    w_sem = max(0.0, float(getattr(cfg, "seed_score_semantic_weight", 0.30)))
    w_graph = max(0.0, float(getattr(cfg, "seed_score_graph_weight", 0.50)))
    w_anchor = max(0.0, float(getattr(cfg, "seed_score_anchor_weight", 0.20)))
    norm = w_sem + w_graph + w_anchor
    if norm <= 0.0:
        w_graph = 1.0
        w_sem = 0.0
        w_anchor = 0.0
        norm = 1.0
    w_sem /= norm
    w_graph /= norm
    w_anchor /= norm

    score = {}
    for node in candidates:
        score[node] = (
            w_graph * float(graph_norm.get(node, 0.0))
            + w_sem * float(semantic_norm.get(node, 0.0))
            + w_anchor * float(anchor_norm.get(node, 0.0))
        )
    return score, graph_norm, semantic_norm, anchor_norm


def _bridge_utility(g, anchors, nodes, tau):
    if not anchors or len(anchors) < 2 or not nodes:
        return 0.0
    max_hops = max(1, int(tau))
    useful = 0
    for node in nodes:
        connected = 0
        for anchor in anchors:
            if node == anchor:
                connected += 1
                continue
            if node not in g or anchor not in g:
                continue
            try:
                d = nx.shortest_path_length(g, node, anchor)
            except Exception:
                continue
            if d <= max_hops:
                connected += 1
        if connected >= 2:
            useful += 1
    return float(useful / max(len(nodes), 1))


def _dispersion_penalty(g, seeds, tau):
    seeds = list(seeds or [])
    if len(seeds) <= 1:
        return 0.0
    cap = max(1, int(tau))
    pairs = 0
    total = 0.0
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a = seeds[i]
            b = seeds[j]
            if a not in g or b not in g:
                d = cap
            else:
                try:
                    d = nx.shortest_path_length(g, a, b)
                except Exception:
                    d = cap
            total += float(min(cap, d))
            pairs += 1
    if pairs <= 0:
        return 0.0
    return float((total / float(pairs)) / float(cap))


def _semantic_redundancy_penalty(seeds, node_vectors):
    items = [node for node in list(seeds or []) if node in node_vectors]
    if len(items) <= 1:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = cosine_similarity(node_vectors.get(items[i]), node_vectors.get(items[j]))
            total += max(0.0, float(sim))
            pairs += 1
    if pairs <= 0:
        return 0.0
    return float(total / float(pairs))


def _run_structural_connectivity(g, surrogate_universe, surrogate_weights, seeds, tau):
    if not surrogate_universe:
        return 0.0, 0.0, 0.0
    cap = max(1, int(tau))
    total_w = 0.0
    total_keep = 0.0
    loss = 0.0
    for node in surrogate_universe:
        w = max(0.0, float(surrogate_weights.get(node, 0.0)))
        if w <= 0.0:
            continue
        d = _min_set_distance(g, node, seeds, cap)
        keep = 1.0 - (float(d) / float(cap))
        total_w += w
        total_keep += w * keep
        loss += w * d
    if total_w <= 0.0:
        return 0.0, 0.0, loss
    base = float(total_keep / total_w)
    dispersion = _dispersion_penalty(g, seeds, cap)
    adjusted = max(0.0, base - 0.5 * dispersion)
    return adjusted, dispersion, loss


def _sentence_node_to_id(g, node):
    if g.nodes[node].get("node_type") != "sentence":
        return None
    return str(g.nodes[node].get("sentence_id", node))


def _build_single_corridor_payload(g, anchor, seed, corridor_id, corridor_score, node_scores):
    ordered_nodes = [node for node, _ in node_scores]
    local_nodes = set(ordered_nodes) | {anchor, seed}
    local_graph = g.subgraph(local_nodes).copy()

    path_nodes = []
    if anchor in local_graph and seed in local_graph:
        try:
            path_nodes = nx.shortest_path(local_graph, anchor, seed)
        except Exception:
            path_nodes = []

    main_sentence_nodes = [node for node in path_nodes if g.nodes[node].get("node_type") == "sentence"]
    if not main_sentence_nodes:
        main_sentence_nodes = [node for node, _ in node_scores if g.nodes[node].get("node_type") == "sentence"]

    connector_nodes = {node for node in path_nodes if g.nodes[node].get("node_type") != "sentence"}
    main_sentence_set = set(main_sentence_nodes)
    candidate_support_nodes = [
        node
        for node, _ in node_scores
        if g.nodes[node].get("node_type") == "sentence" and node not in main_sentence_set
    ]

    adjacent_support_nodes = []
    for node in candidate_support_nodes:
        nbrs = set(g.neighbors(node))
        if nbrs.intersection(main_sentence_set) or nbrs.intersection(connector_nodes):
            adjacent_support_nodes.append(node)

    support_sentence_nodes = _ordered_unique(adjacent_support_nodes + candidate_support_nodes)

    sentence_score_map = {}
    for node, score in node_scores:
        sid = _sentence_node_to_id(g, node)
        if sid is None:
            continue
        sentence_score_map[sid] = max(float(score), sentence_score_map.get(sid, 0.0))

    main_sentence_ids = _ordered_unique([_sentence_node_to_id(g, node) for node in main_sentence_nodes if node in g])
    main_sentence_ids = [sid for sid in main_sentence_ids if sid]

    support_sentence_ids = _ordered_unique([_sentence_node_to_id(g, node) for node in support_sentence_nodes if node in g])
    support_sentence_ids = [sid for sid in support_sentence_ids if sid and sid not in set(main_sentence_ids)]
    connector_adjacent_sentence_ids = _ordered_unique(
        [_sentence_node_to_id(g, node) for node in adjacent_support_nodes if node in g]
    )
    connector_adjacent_sentence_ids = [
        sid for sid in connector_adjacent_sentence_ids if sid and sid not in set(main_sentence_ids)
    ]

    return {
        "corridor_id": corridor_id,
        "corridor_score": float(corridor_score),
        "anchors": [str(anchor), str(seed)],
        "main_path_sentence_ids": main_sentence_ids,
        "support_sentence_ids": support_sentence_ids,
        "connector_adjacent_sentence_ids": connector_adjacent_sentence_ids,
        "sentence_score_map": sentence_score_map,
        "path_node_ids": [str(node) for node in path_nodes],
    }


def _build_corridor(
    g,
    anchors,
    seeds,
    anchor_scores_by_node,
    cfg,
    ppr_engine,
    pair_top_lp,
    corridor_top_bc,
    show_progress=False,
):
    if not anchors or not seeds:
        return g.subgraph(anchors + list(seeds)).copy(), [], {}, []

    seed_score_map = _compute_ppr_batch(
        g=g,
        sources=list(seeds),
        cfg=cfg,
        engine=ppr_engine,
        run_seed=911,
        show_progress=show_progress,
        desc="Corridor seed PPR",
    )

    pair_stats = []
    for anchor in tqdm(
        anchors,
        total=len(anchors),
        desc="Corridor pair score",
        leave=False,
        disable=not show_progress,
    ):
        a_scores = anchor_scores_by_node.get(anchor, {})
        for seed in seeds:
            z_scores = seed_score_map.get(seed, {})
            support_sum = 0.0
            for node in g.nodes:
                support_sum += _pair_support(a_scores, z_scores, node)
            pair_stats.append(((anchor, seed), support_sum))

    pair_stats.sort(key=lambda x: x[1], reverse=True)
    retained_pairs = [pair for pair, _ in pair_stats[:pair_top_lp]]
    pair_score_map = {pair: score for pair, score in pair_stats}

    corridor_nodes = set()
    sentence_scores = {}
    corridor_payloads = []

    for idx, (anchor, seed) in enumerate(
        tqdm(
            retained_pairs,
            total=len(retained_pairs),
            desc="Corridor compose",
            leave=False,
            disable=not show_progress,
        ),
        start=1,
    ):
        a_scores = anchor_scores_by_node.get(anchor, {})
        z_scores = seed_score_map.get(seed, {})
        node_scores = []

        for node in g.nodes:
            s = _pair_support(a_scores, z_scores, node)
            node_scores.append((node, s))

        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_node_scores = node_scores[:corridor_top_bc]
        for node, score in top_node_scores:
            corridor_nodes.add(node)
            sentence_scores[node] = max(sentence_scores.get(node, 0.0), score)

        corridor_nodes.add(anchor)
        corridor_nodes.add(seed)

        corridor_payloads.append(
            _build_single_corridor_payload(
                g=g,
                anchor=anchor,
                seed=seed,
                corridor_id=f"c{idx:02d}",
                corridor_score=pair_score_map.get((anchor, seed), 0.0),
                node_scores=top_node_scores,
            )
        )

    h = g.subgraph(corridor_nodes).copy()
    return h, retained_pairs, sentence_scores, corridor_payloads


def _can_remove_node(h, node, anchors, min_seed_keep, seeds):
    if node in anchors:
        return False

    trial = h.copy()
    trial.remove_node(node)

    if not anchors.issubset(trial.nodes()):
        return False

    components = list(nx.connected_components(trial))
    if not any(anchors.issubset(comp) for comp in components):
        return False

    kept_seed_count = len(seeds.intersection(set(trial.nodes())))
    return kept_seed_count >= min_seed_keep


def _greedy_trim(h, anchors, seeds, rho):
    anchors_set = set(anchors)
    min_seed_keep = max(1, int(round(rho * len(seeds)))) if seeds else 0

    changed = True
    while changed:
        changed = False

        leaf_like = [n for n in list(h.nodes()) if h.degree(n) <= 1 and n not in anchors_set]
        for node in leaf_like:
            if node not in h:
                continue
            if _can_remove_node(h, node, anchors_set, min_seed_keep, seeds):
                h.remove_node(node)
                changed = True

        for comp in list(nx.connected_components(h)):
            if not anchors_set.intersection(comp):
                h.remove_nodes_from(comp)
                changed = True

    return h


def _extract_sentence_payload(g, selected_nodes, sentence_scores):
    sentences = []
    for node in selected_nodes:
        if g.nodes[node].get("node_type") != "sentence":
            continue
        sid = str(g.nodes[node].get("sentence_id", node))
        text = str(g.nodes[node].get("text", ""))
        score = sentence_scores.get(node, 0.0)
        sentences.append((sid, text, score))

    sentences.sort(key=lambda x: x[2], reverse=True)
    sentence_ids = [sid for sid, _, _ in sentences]
    sentence_text = [text for _, text, _ in sentences]
    sentence_score_map = {sid: float(score) for sid, _, score in sentences}
    return sentence_ids, sentence_text, sentence_score_map


def _filter_corridor_payloads(corridors, selected_sentence_ids):
    selected = set(selected_sentence_ids)
    filtered = []

    for corridor in corridors:
        main_ids = [sid for sid in corridor.get("main_path_sentence_ids", []) if sid in selected]
        main_set = set(main_ids)
        support_ids = [
            sid
            for sid in corridor.get("support_sentence_ids", [])
            if sid in selected and sid not in main_set
        ]
        connector_adjacent_ids = [
            sid
            for sid in corridor.get("connector_adjacent_sentence_ids", [])
            if sid in selected and sid not in main_set
        ]

        if not main_ids and not support_ids:
            continue

        payload = dict(corridor)
        payload["main_path_sentence_ids"] = main_ids
        payload["support_sentence_ids"] = support_ids
        payload["connector_adjacent_sentence_ids"] = connector_adjacent_ids
        filtered.append(payload)

    return filtered


def _build_sentence_feature_table(sample, selected_sentence_ids, sentence_texts, sentence_score_map, corridors):
    question_tokens = set(content_tokens(sample.question))
    sentence_text_map = {sid: text for sid, text in zip(selected_sentence_ids, sentence_texts)}
    selected_set = set(selected_sentence_ids)

    ranked_corridors = sorted(corridors or [], key=lambda c: float(c.get("corridor_score", 0.0)), reverse=True)
    corridor_rank = {str(c.get("corridor_id", f"c{idx:02d}")): idx for idx, c in enumerate(ranked_corridors, start=1)}

    feature_table = {}
    for global_idx, sid in enumerate(selected_sentence_ids):
        feature_table[sid] = {
            "corridor_ids": [],
            "best_corridor_rank": None,
            "best_corridor_score": 0.0,
            "is_main_candidate": False,
            "is_support_candidate": False,
            "is_connector_adjacent": False,
            "query_overlap_score": 0.0,
            "locality_score": 1.0 / float(global_idx + 1),
            "base_retrieval_score": float(sentence_score_map.get(sid, 0.0)),
        }

    for idx, corridor in enumerate(ranked_corridors, start=1):
        cid = str(corridor.get("corridor_id", f"c{idx:02d}"))
        cscore = float(corridor.get("corridor_score", 0.0))
        main_ids = [sid for sid in corridor.get("main_path_sentence_ids", []) if sid in selected_set]
        support_ids = [sid for sid in corridor.get("support_sentence_ids", []) if sid in selected_set]
        connector_ids = [sid for sid in corridor.get("connector_adjacent_sentence_ids", []) if sid in selected_set]

        for mpos, sid in enumerate(main_ids):
            item = feature_table.get(sid)
            if item is None:
                continue
            if cid not in item["corridor_ids"]:
                item["corridor_ids"].append(cid)
            item["is_main_candidate"] = True
            item["best_corridor_score"] = max(item["best_corridor_score"], cscore)
            if item["best_corridor_rank"] is None or idx < item["best_corridor_rank"]:
                item["best_corridor_rank"] = idx
            item["locality_score"] = max(item["locality_score"], 1.0 / float((idx) * (mpos + 1)))

        for spos, sid in enumerate(support_ids):
            item = feature_table.get(sid)
            if item is None:
                continue
            if cid not in item["corridor_ids"]:
                item["corridor_ids"].append(cid)
            item["is_support_candidate"] = True
            item["best_corridor_score"] = max(item["best_corridor_score"], cscore)
            if item["best_corridor_rank"] is None or idx < item["best_corridor_rank"]:
                item["best_corridor_rank"] = idx
            item["locality_score"] = max(item["locality_score"], 0.75 / float((idx) * (spos + 1)))

        for sid in connector_ids:
            item = feature_table.get(sid)
            if item is None:
                continue
            item["is_connector_adjacent"] = True

    for sid, item in feature_table.items():
        sent = sentence_text_map.get(sid, "")
        sent_tokens = set(content_tokens(sent))
        item["query_overlap_score"] = float(len(question_tokens.intersection(sent_tokens)))
        item["corridor_ids"] = sorted(
            item["corridor_ids"],
            key=lambda cid: corridor_rank.get(cid, 10**9),
        )
        if item["best_corridor_rank"] is None and item["corridor_ids"]:
            item["best_corridor_rank"] = corridor_rank.get(item["corridor_ids"][0])
        if item["best_corridor_rank"] is None:
            item["best_corridor_rank"] = 10**9

    return feature_table


def run_graphrag_core(
    sample,
    cfg,
    method_name,
    stable_seed_selection,
    enable_trim,
):
    start = time.perf_counter()
    graph_scope = "query_context"
    global_index_meta = {}
    g = None

    has_global_corpus = bool(str(getattr(cfg, "global_corpus_path", "") or "").strip())
    has_prebuilt_igraph = bool(str(getattr(cfg, "prebuilt_igraph_path", "") or "").strip())
    if has_global_corpus or has_prebuilt_igraph:
        g, global_index_meta = _get_global_graph(cfg)
        graph_scope = "prebuilt_igraph" if has_prebuilt_igraph else "global_corpus"
    else:
        artifacts = build_document_entity_graph(sample)
        g = artifacts.graph

    anchors = select_lexical_anchors(sample, g, cfg.max_anchors)

    if not anchors:
        entities = [n for n in g.nodes if g.nodes[n].get("node_type") == "entity"]
        entities.sort(key=lambda n: g.degree(n), reverse=True)
        anchors = entities[: cfg.max_anchors]

    rng = random.Random(cfg.random_seed)
    per_anchor_runs = {a: [] for a in anchors}
    show_inner_progress = bool(getattr(cfg, "show_inner_progress", True))
    if int(getattr(cfg, "num_workers", 1)) > 1:
        show_inner_progress = False

    ppr_engine = _resolve_ppr_engine(cfg, graph_scope=graph_scope, graph=g)
    ppr_graph = g
    ppr_subgraph_applied = False
    if (
        bool(getattr(cfg, "ppr_subgraph_enable", True))
        and graph_scope in {"global_corpus", "prebuilt_igraph"}
        and anchors
    ):
        maybe_sub = _build_anchor_subgraph(
            g=g,
            anchors=anchors,
            hops=int(getattr(cfg, "ppr_subgraph_hops", 2)),
            max_nodes=int(getattr(cfg, "ppr_subgraph_max_nodes", 30000)),
        )
        if maybe_sub is not g:
            ppr_graph = maybe_sub
            ppr_subgraph_applied = True
            anchors = [a for a in anchors if a in ppr_graph]
            per_anchor_runs = {a: [] for a in anchors}

    n_runs = max(1, cfg.samples_per_anchor)
    for _ in tqdm(
        range(n_runs),
        total=n_runs,
        desc="PPR perturb",
        leave=False,
        disable=not show_inner_progress,
    ):
        run_seed = int(rng.random() * 10**9)
        if ppr_engine == "power":
            h = _stochastic_perturb_graph(ppr_graph, rng, cfg.edge_drop_prob)
            run_map = _compute_ppr_batch(
                g=h,
                sources=anchors,
                cfg=cfg,
                engine="power",
                run_seed=run_seed,
                show_progress=show_inner_progress,
                desc="Anchor PPR",
            )
        else:
            run_map = _compute_ppr_batch(
                g=ppr_graph,
                sources=anchors,
                cfg=cfg,
                engine="mc",
                run_seed=run_seed,
                show_progress=show_inner_progress,
                desc="Anchor PPR",
            )

        for anchor in anchors:
            per_anchor_runs[anchor].append(run_map.get(anchor, {}))

    semantic_diag = {
        "enabled": bool(getattr(cfg, "embedding_enabled", False)),
        "applied": False,
        "error": "",
        "model_name": str(getattr(cfg, "embedding_model_name", "") or ""),
        "semantic_topn": int(getattr(cfg, "semantic_topn", 50)),
        "semantic_candidate_union": bool(getattr(cfg, "semantic_candidate_union", True)),
        "index_available": False,
        "semantic_candidates": [],
        "semantic_candidate_scores": {},
        "query_embedding_dim": 0,
        "candidate_vector_count": 0,
    }

    query_vec = None
    semantic_state = None
    semantic_candidates = []
    semantic_candidate_scores = {}
    if semantic_diag["enabled"]:
        query_vec, qerr = _query_embedding(sample.question, cfg)
        if query_vec is None:
            semantic_diag["error"] = qerr
        else:
            semantic_diag["query_embedding_dim"] = int(query_vec.shape[0])
            if graph_scope in {"global_corpus", "prebuilt_igraph"}:
                semantic_state = _get_global_semantic_state(global_index_meta)
                semantic_diag["index_available"] = semantic_state is not None
            if semantic_state is not None:
                semantic_candidates, semantic_candidate_scores = _semantic_top_candidates(query_vec, semantic_state, cfg)
                semantic_diag["applied"] = True
            elif ppr_graph.number_of_nodes() <= 5000:
                local_nodes = [
                    n
                    for n in ppr_graph.nodes
                    if ppr_graph.nodes[n].get("node_type") in {"entity", "sentence"}
                ]
                local_vectors, local_err = _load_candidate_vectors(local_nodes, ppr_graph, cfg, semantic_state=None)
                if local_err:
                    semantic_diag["error"] = local_err
                else:
                    local_scores = {n: cosine_similarity(query_vec, vec) for n, vec in local_vectors.items()}
                    ranked = sorted(local_scores.items(), key=lambda x: x[1], reverse=True)
                    topn = max(0, int(getattr(cfg, "semantic_topn", 50)))
                    ranked = ranked[:topn]
                    semantic_candidates = [n for n, _ in ranked]
                    semantic_candidate_scores = {n: float(s) for n, s in ranked}
                    semantic_diag["applied"] = True
            else:
                semantic_diag["error"] = "semantic_index_unavailable_for_large_graph"
    semantic_diag["semantic_candidates"] = [str(n) for n in semantic_candidates]
    semantic_diag["semantic_candidate_scores"] = {str(k): float(v) for k, v in semantic_candidate_scores.items()}

    run_base = []
    num_runs = n_runs
    for run_id in tqdm(
        range(num_runs),
        total=num_runs,
        desc="Seed selection",
        leave=False,
        disable=not show_inner_progress,
    ):
        agg_scores = {}
        anchor_scores = {}

        for anchor in anchors:
            scores = per_anchor_runs[anchor][run_id]
            anchor_scores[anchor] = scores
            for node, score in scores.items():
                agg_scores[node] = agg_scores.get(node, 0.0) + score

        graph_candidates = [n for n in _topk_nodes(agg_scores, cfg.candidate_top_t) if n in ppr_graph]
        run_base.append(
            {
                "run_id": run_id,
                "anchor_scores": anchor_scores,
                "agg_scores": agg_scores,
                "graph_candidates": graph_candidates,
            }
        )

    semantic_union_enabled = bool(getattr(cfg, "semantic_candidate_union", True))
    candidate_universe = set(anchors)
    for run in run_base:
        candidate_universe.update(run.get("graph_candidates", []))
    if semantic_union_enabled:
        candidate_universe.update(semantic_candidates)
    candidate_universe = {node for node in candidate_universe if node in ppr_graph}

    node_vectors = {}
    vector_err = ""
    if semantic_diag["enabled"] and query_vec is not None and candidate_universe:
        node_vectors, vector_err = _load_candidate_vectors(
            nodes=sorted(candidate_universe),
            g=ppr_graph,
            cfg=cfg,
            semantic_state=semantic_state,
        )
        if vector_err and not semantic_diag["error"]:
            semantic_diag["error"] = vector_err
    semantic_diag["candidate_vector_count"] = int(len(node_vectors))

    query_sim_map = {}
    for node, vec in node_vectors.items():
        if query_vec is None:
            break
        query_sim_map[node] = cosine_similarity(query_vec, vec)
    for node, score in semantic_candidate_scores.items():
        query_sim_map[node] = max(float(score), float(query_sim_map.get(node, -1.0)))

    anchor_vecs = {anchor: node_vectors[anchor] for anchor in anchors if anchor in node_vectors}
    anchor_sim_map = {}
    for node in candidate_universe:
        vec = node_vectors.get(node)
        if vec is None or not anchor_vecs:
            anchor_sim_map[node] = 0.0
            continue
        anchor_sim_map[node] = max(cosine_similarity(vec, avec) for avec in anchor_vecs.values())

    support_sim_map = {}
    for node in candidate_universe:
        support_sim_map[node] = _entity_support_similarity(ppr_graph, node, query_sim_map)

    run_results = []
    for run in run_base:
        graph_candidates = list(run.get("graph_candidates", []))
        if semantic_union_enabled:
            candidates = _ordered_unique(graph_candidates + list(semantic_candidates))
        else:
            candidates = list(graph_candidates)
        candidates = [node for node in candidates if node in ppr_graph]

        if semantic_diag["enabled"] and query_vec is not None:
            seed_score_map, graph_norm, semantic_norm, anchor_norm = _seed_hybrid_scores(
                candidates=candidates,
                agg_scores=run.get("agg_scores", {}),
                query_sim=query_sim_map,
                anchor_sim=anchor_sim_map,
                support_sim=support_sim_map,
                cfg=cfg,
            )
        else:
            seed_score_map = {node: float(run["agg_scores"].get(node, 0.0)) for node in candidates}
            graph_norm = _normalize_map(seed_score_map)
            semantic_norm = {node: 0.0 for node in candidates}
            anchor_norm = {node: 0.0 for node in candidates}
        objective_weights = {node: float(seed_score_map.get(node, 0.0)) for node in candidates}
        if max(objective_weights.values(), default=0.0) <= 0.0:
            objective_weights = {node: float(run["agg_scores"].get(node, 0.0)) for node in candidates}

        seeds = _greedy_seed_set(ppr_graph, candidates, objective_weights, cfg.seed_k, cfg.tau)
        run_results.append(
            {
                "run_id": run["run_id"],
                "anchor_scores": run["anchor_scores"],
                "agg_scores": run["agg_scores"],
                "graph_candidates": graph_candidates,
                "semantic_candidates": list(semantic_candidates),
                "candidates": candidates,
                "seeds": seeds,
                "seed_score_map": seed_score_map,
                "seed_score_components": {
                    "graph": graph_norm,
                    "semantic": semantic_norm,
                    "anchor": anchor_norm,
                },
            }
        )

    if stable_seed_selection and run_results:
        surrogate_universe = set(anchors)
        for run in run_results:
            surrogate_universe.update(run["candidates"])

        surrogate_weights = {}
        for node in surrogate_universe:
            vals = [r["seed_score_map"].get(node, r["agg_scores"].get(node, 0.0)) for r in run_results]
            surrogate_weights[node] = sum(float(v) for v in vals) / max(len(vals), 1)
        if max(surrogate_weights.values(), default=0.0) <= 0.0:
            for node in surrogate_universe:
                vals = [r["agg_scores"].get(node, 0.0) for r in run_results]
                surrogate_weights[node] = sum(float(v) for v in vals) / max(len(vals), 1)

        if (not semantic_diag["enabled"]) or query_vec is None:
            best = None
            best_loss = float("inf")
            for run in run_results:
                seeds = run.get("seeds", set())
                loss = 0.0
                for node in surrogate_universe:
                    loss += surrogate_weights[node] * _min_set_distance(ppr_graph, node, seeds, cfg.tau)
                run["global_loss"] = float(loss)
                run["surrogate_loss"] = float(loss)
                run["hybrid_run_score"] = float(-loss)
                if loss < best_loss:
                    best_loss = loss
                    best = run
            chosen = best if best is not None else run_results[0]
        else:
            w_sem = max(0.0, float(getattr(cfg, "run_score_semantic_weight", 0.35)))
            w_anchor = max(0.0, float(getattr(cfg, "run_score_anchor_weight", 0.20)))
            w_struct = max(0.0, float(getattr(cfg, "run_score_structure_weight", 0.25)))
            w_bridge = max(0.0, float(getattr(cfg, "run_score_bridge_weight", 0.15)))
            w_redundancy = max(0.0, float(getattr(cfg, "run_score_redundancy_weight", 0.05)))
            total = w_sem + w_anchor + w_struct + w_bridge + w_redundancy
            if total <= 0.0:
                w_sem, w_anchor, w_struct, w_bridge, w_redundancy = 0.35, 0.20, 0.25, 0.15, 0.05
                total = 1.0
            w_sem /= total
            w_anchor /= total
            w_struct /= total
            w_bridge /= total
            w_redundancy /= total

            best = None
            best_score = float("-inf")
            for run in run_results:
                run_nodes = _ordered_unique(list(run.get("candidates", [])) + list(run.get("seeds", [])))
                node_weights = {n: float(run["seed_score_map"].get(n, run["agg_scores"].get(n, 0.0))) for n in run_nodes}

                semantic_raw = _weighted_mean(query_sim_map, run_nodes, node_weights)
                anchor_raw = _weighted_mean(anchor_sim_map, run_nodes, node_weights)
                semantic_cov = max(0.0, min(1.0, 0.5 * (semantic_raw + 1.0)))
                anchor_align = max(0.0, min(1.0, 0.5 * (anchor_raw + 1.0)))
                structural_conn, dispersion_penalty, structural_loss = _run_structural_connectivity(
                    ppr_graph,
                    surrogate_universe=surrogate_universe,
                    surrogate_weights=surrogate_weights,
                    seeds=run.get("seeds", set()),
                    tau=cfg.tau,
                )
                bridge = _bridge_utility(ppr_graph, anchors=anchors, nodes=run_nodes, tau=cfg.tau)
                redundancy = _semantic_redundancy_penalty(run.get("seeds", set()), node_vectors=node_vectors)

                run_score = (
                    w_sem * semantic_cov
                    + w_anchor * anchor_align
                    + w_struct * structural_conn
                    + w_bridge * bridge
                    - w_redundancy * redundancy
                )
                surrogate_loss = (
                    w_sem * (1.0 - semantic_cov)
                    + w_anchor * (1.0 - anchor_align)
                    + w_struct * (1.0 - structural_conn)
                    + w_bridge * (1.0 - bridge)
                    + w_redundancy * redundancy
                    + 0.25 * dispersion_penalty
                )

                run["run_score_components"] = {
                    "semantic_coverage": float(semantic_cov),
                    "anchor_alignment": float(anchor_align),
                    "structural_connectivity": float(structural_conn),
                    "bridge_utility": float(bridge),
                    "redundancy_penalty": float(redundancy),
                    "dispersion_penalty": float(dispersion_penalty),
                }
                run["hybrid_run_score"] = float(run_score)
                run["surrogate_loss"] = float(surrogate_loss)
                run["global_loss"] = float(structural_loss)
                if run_score > best_score:
                    best_score = run_score
                    best = run

            chosen = best if best is not None else run_results[0]
    else:
        chosen = (
            run_results[0]
            if run_results
            else {
                "run_id": 0,
                "anchor_scores": {},
                "graph_candidates": [],
                "semantic_candidates": [],
                "candidates": [],
                "seeds": set(),
                "agg_scores": {},
                "seed_score_map": {},
                "seed_score_components": {"graph": {}, "semantic": {}, "anchor": {}},
            }
        )

    corridor, retained_pairs, sentence_scores, corridor_payloads = _build_corridor(
        g=ppr_graph,
        anchors=anchors,
        seeds=chosen["seeds"],
        anchor_scores_by_node=chosen["anchor_scores"],
        cfg=cfg,
        ppr_engine=ppr_engine,
        pair_top_lp=cfg.pair_top_lp,
        corridor_top_bc=cfg.corridor_top_bc,
        show_progress=show_inner_progress,
    )

    final_graph = corridor
    if enable_trim and cfg.trim_on:
        final_graph = _greedy_trim(corridor.copy(), anchors=anchors, seeds=chosen["seeds"], rho=cfg.trim_rho)

    selected_nodes = set(final_graph.nodes())
    selected_sentence_ids, selected_sentences, selected_sentence_score_map = _extract_sentence_payload(
        g, selected_nodes, sentence_scores
    )
    embedding_diag = {
        "enabled": bool(getattr(cfg, "embedding_enabled", False)),
        "applied": False,
        "error": "",
        "model_name": str(getattr(cfg, "embedding_model_name", "") or ""),
        "weight": float(getattr(cfg, "embedding_weight", 0.35)),
        "rerank_topn": int(getattr(cfg, "embedding_rerank_topn", 80)),
        "batch_size": int(getattr(cfg, "embedding_batch_size", 16)),
        "max_length": int(getattr(cfg, "embedding_max_length", 256)),
        "max_chars": int(getattr(cfg, "embedding_text_max_chars", 600)),
        "head_size": 0,
    }
    if embedding_diag["enabled"] and selected_sentence_ids:
        rerank = rerank_sentences_by_embedding(
            question=sample.question,
            sentence_ids=selected_sentence_ids,
            sentence_texts=selected_sentences,
            base_score_map=selected_sentence_score_map,
            model_name=embedding_diag["model_name"],
            weight=embedding_diag["weight"],
            rerank_topn=embedding_diag["rerank_topn"],
            batch_size=embedding_diag["batch_size"],
            max_length=embedding_diag["max_length"],
            max_chars=embedding_diag["max_chars"],
        )
        selected_sentence_ids = list(rerank.get("ranked_sentence_ids", selected_sentence_ids))
        selected_sentences = list(rerank.get("ranked_sentence_texts", selected_sentences))
        embedding_diag["applied"] = bool(rerank.get("applied", False))
        embedding_diag["error"] = str(rerank.get("error", "") or "")
        embedding_diag["head_size"] = int(rerank.get("rerank_topn", 0))
        embedding_diag["similarity_by_sentence_id"] = rerank.get("similarity_by_sentence_id", {})
        embedding_diag["fused_score_by_sentence_id"] = rerank.get("fused_score_by_sentence_id", {})

    filtered_corridors = _filter_corridor_payloads(corridor_payloads, selected_sentence_ids)
    sentence_feature_table = _build_sentence_feature_table(
        sample=sample,
        selected_sentence_ids=selected_sentence_ids,
        sentence_texts=selected_sentences,
        sentence_score_map=selected_sentence_score_map,
        corridors=filtered_corridors,
    )

    anchor_results = []
    diag_topn = max(1, int(getattr(cfg, "anchor_diag_topn", 10)))
    diag_full = bool(getattr(cfg, "anchor_diag_store_full_scores", False))
    for anchor in tqdm(
        anchors,
        total=len(anchors),
        desc="Anchor diagnostics",
        leave=False,
        disable=not show_inner_progress,
    ):
        for sample_idx, scores in enumerate(per_anchor_runs.get(anchor, [])):
            top_candidates = _topk_nodes(scores, diag_topn)
            if diag_full:
                score_payload = {k: float(v) for k, v in scores.items() if v > 0.0}
            else:
                score_payload = {k: float(scores.get(k, 0.0)) for k in top_candidates}
            anchor_results.append(
                AnchorResult(
                    anchor=anchor,
                    scores=score_payload,
                    top_candidates=top_candidates,
                    sample_index=sample_idx,
                    metadata={"topn": diag_topn, "full_scores": diag_full},
                )
            )

    latency_ms = (time.perf_counter() - start) * 1000.0
    return RetrievalResult(
        sample_id=sample.qid,
        method=method_name,
        anchors=anchors,
        seeds=sorted(chosen["seeds"]),
        selected_nodes=sorted(selected_nodes),
        selected_sentence_ids=selected_sentence_ids,
        selected_sentences=selected_sentences,
        corridors=filtered_corridors,
        anchor_results=anchor_results,
        diagnostics={
            "graph_scope": graph_scope,
            "global_index": global_index_meta if graph_scope in {"global_corpus", "prebuilt_igraph"} else {},
            "ppr_engine_requested": str(getattr(cfg, "ppr_engine", "auto")),
            "ppr_engine_effective": ppr_engine,
            "ppr_subgraph_applied": bool(ppr_subgraph_applied),
            "ppr_graph_nodes": int(ppr_graph.number_of_nodes()),
            "ppr_graph_edges": int(ppr_graph.number_of_edges()),
            "best_run_id": chosen.get("run_id", 0),
            "best_run_score": float(chosen.get("hybrid_run_score", 0.0) or 0.0),
            "best_run_score_components": chosen.get("run_score_components", {}) or {},
            "best_seed_score_components": chosen.get("seed_score_components", {}) or {},
            "num_seeds": len(chosen["seeds"]),
            "num_candidates_graph": len(chosen.get("graph_candidates", []) or []),
            "num_candidates_semantic": len(chosen.get("semantic_candidates", []) or []),
            "num_candidates_union": len(chosen.get("candidates", []) or []),
            "retained_pairs": [[a, z] for a, z in retained_pairs],
            "num_corridors": len(filtered_corridors),
            "corridor_nodes_before_trim": corridor.number_of_nodes(),
            "corridor_nodes_after_trim": final_graph.number_of_nodes(),
            "sentence_scores": sentence_scores,
            "sentence_feature_table": sentence_feature_table,
            "semantic_selection": semantic_diag,
            "run_selection_mode": (
                "semantic_hybrid"
                if (stable_seed_selection and semantic_diag["enabled"] and query_vec is not None)
                else ("legacy_structural" if stable_seed_selection else "first_run")
            ),
            "run_pool": [
                {
                    "run_id": int(r.get("run_id", 0)),
                    "hybrid_run_score": float(r.get("hybrid_run_score", 0.0) or 0.0),
                    "surrogate_loss": float(r.get("surrogate_loss", 0.0) or 0.0),
                    "global_loss": float(r.get("global_loss", 0.0) or 0.0),
                    "num_graph_candidates": int(len(r.get("graph_candidates", []) or [])),
                    "num_semantic_candidates": int(len(r.get("semantic_candidates", []) or [])),
                    "num_candidates": int(len(r.get("candidates", []) or [])),
                    "num_seeds": int(len(r.get("seeds", []) or [])),
                    "run_score_components": r.get("run_score_components", {}) or {},
                }
                for r in run_results
            ],
            "embedding_rerank": embedding_diag,
            "stable_seed_selection": stable_seed_selection,
            "trim_enabled": enable_trim and cfg.trim_on,
            "anchor_diag_topn": int(diag_topn),
            "anchor_diag_store_full_scores": bool(diag_full),
        },
        latency_ms=latency_ms,
    )


@register_method("effirag")
def run_effirag(sample, cfg):
    return run_graphrag_core(
        sample=sample,
        cfg=cfg,
        method_name="effirag",
        stable_seed_selection=True,
        enable_trim=True,
    )
