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
    embedding_max_length = int(getattr(cfg, "embedding_max_length", 192))
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


def _get_distance_map(g, source, cutoff, distance_map_cache=None):
    if source not in g:
        return {}
    cap = max(1, int(cutoff))
    if isinstance(distance_map_cache, dict):
        key = (source, cap)
        cached = distance_map_cache.get(key)
        if isinstance(cached, dict):
            return cached
    try:
        dmap = nx.single_source_shortest_path_length(g, source, cutoff=cap)
    except Exception:
        dmap = {source: 0}
    if isinstance(distance_map_cache, dict):
        distance_map_cache[(source, cap)] = dmap
    return dmap


def _min_set_distance(g, node, seeds, tau, distance_map_cache=None):
    cap = max(1, int(tau))
    if not seeds or node not in g:
        return cap

    best = cap
    for z in seeds:
        if z not in g:
            continue
        if z == node:
            return 0
        dmap = _get_distance_map(g, z, cap, distance_map_cache=distance_map_cache)
        dist = int(dmap.get(node, cap))
        if dist < best:
            best = dist
            if best <= 0:
                break
    return min(best, cap)


def _greedy_seed_set(g, candidates, weights, seed_k, tau, distance_map_cache=None):
    selected = set()
    remaining = set(candidates)

    while len(selected) < seed_k and remaining:
        best_seed = None
        best_obj = float("inf")

        for z in remaining:
            trial = selected | {z}
            obj = 0.0
            for u in candidates:
                obj += weights.get(u, 0.0) * _min_set_distance(
                    g,
                    u,
                    trial,
                    tau,
                    distance_map_cache=distance_map_cache,
                )
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
        max_length=int(getattr(cfg, "embedding_max_length", 192)),
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


def _semantic_topk_candidates_by_type(query_vec, semantic_state, cfg):
    if semantic_state is None or query_vec is None:
        return {}, {}, {"entity_ms": 0.0, "chunk_ms": 0.0}

    topn_entity = max(
        0,
        int(
            getattr(
                cfg,
                "semantic_topn_entity",
                getattr(cfg, "semantic_topn", 50),
            )
        ),
    )
    topn_chunk = max(
        0,
        int(
            getattr(
                cfg,
                "semantic_topn_chunk",
                max(1, int(getattr(cfg, "semantic_topn", 50)) // 2),
            )
        ),
    )
    scan_bs = max(128, int(getattr(cfg, "semantic_scan_batch_size", 8192)))

    diag = {"entity_ms": 0.0, "chunk_ms": 0.0}
    entity_scores = {}
    if topn_entity > 0:
        ent_start = time.perf_counter()
        ent_idx, ent_scores = topk_cosine_similarity(
            query_vector=query_vec,
            matrix=semantic_state.get("entity_embeddings"),
            topn=topn_entity,
            scan_batch_size=scan_bs,
        )
        for idx, score in zip(ent_idx.tolist(), ent_scores.tolist()):
            if idx < 0 or idx >= len(semantic_state.get("entity_ids", [])):
                continue
            node = str(semantic_state["entity_ids"][idx])
            entity_scores[node] = max(float(score), float(entity_scores.get(node, -1.0)))
        diag["entity_ms"] = float((time.perf_counter() - ent_start) * 1000.0)

    chunk_scores = {}
    if topn_chunk > 0:
        chunk_start = time.perf_counter()
        chunk_idx, chunk_scores_arr = topk_cosine_similarity(
            query_vector=query_vec,
            matrix=semantic_state.get("chunk_embeddings"),
            topn=topn_chunk,
            scan_batch_size=scan_bs,
        )
        for idx, score in zip(chunk_idx.tolist(), chunk_scores_arr.tolist()):
            if idx < 0 or idx >= len(semantic_state.get("chunk_ids", [])):
                continue
            node = str(semantic_state["chunk_ids"][idx])
            chunk_scores[node] = max(float(score), float(chunk_scores.get(node, -1.0)))
        diag["chunk_ms"] = float((time.perf_counter() - chunk_start) * 1000.0)

    entity_scores = dict(sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:topn_entity])
    chunk_scores = dict(sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:topn_chunk])
    return entity_scores, chunk_scores, diag


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
            max_length=int(getattr(cfg, "embedding_max_length", 192)),
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


def _bridge_utility(g, anchors, nodes, tau, distance_map_cache=None):
    if not anchors or len(anchors) < 2 or not nodes:
        return 0.0
    max_hops = max(1, int(tau))
    anchor_maps = {
        anchor: _get_distance_map(g, anchor, max_hops, distance_map_cache=distance_map_cache)
        for anchor in anchors
        if anchor in g
    }
    useful = 0
    for node in nodes:
        connected = 0
        for anchor in anchors:
            if node == anchor:
                connected += 1
                continue
            if node not in g or anchor not in anchor_maps:
                continue
            d = int(anchor_maps[anchor].get(node, max_hops + 1))
            if d <= max_hops:
                connected += 1
        if connected >= 2:
            useful += 1
    return float(useful / max(len(nodes), 1))


def _dispersion_penalty(g, seeds, tau, distance_map_cache=None):
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
                dmap = _get_distance_map(g, a, cap, distance_map_cache=distance_map_cache)
                d = int(dmap.get(b, cap))
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


def _run_structural_connectivity(g, surrogate_universe, surrogate_weights, seeds, tau, distance_map_cache=None):
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
        d = _min_set_distance(g, node, seeds, cap, distance_map_cache=distance_map_cache)
        keep = 1.0 - (float(d) / float(cap))
        total_w += w
        total_keep += w * keep
        loss += w * d
    if total_w <= 0.0:
        return 0.0, 0.0, loss
    base = float(total_keep / total_w)
    dispersion = _dispersion_penalty(g, seeds, cap, distance_map_cache=distance_map_cache)
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


def _shortest_distance_with_cap(g, src, dst, cap):
    max_hops = max(1, int(cap))
    if src not in g or dst not in g:
        return max_hops
    try:
        d = nx.shortest_path_length(g, src, dst)
        return min(max_hops, int(d))
    except Exception:
        return max_hops


def _anchor_graph_reserve_candidates(g, anchor, topn, max_hops):
    k = max(0, int(topn))
    if k <= 0 or anchor not in g:
        return []

    cutoff = max(1, int(max_hops))
    try:
        dmap = nx.single_source_shortest_path_length(g, anchor, cutoff=cutoff)
    except Exception:
        dmap = {anchor: 0}

    scores = []
    for node, dist in dmap.items():
        if node == anchor:
            continue
        ntype = g.nodes[node].get("node_type")
        if ntype not in {"entity", "sentence"}:
            continue
        dist_term = 1.0 / float(1 + int(dist))
        degree_term = min(1.0, float(g.degree(node)) / 20.0)
        scores.append((node, 0.80 * dist_term + 0.20 * degree_term))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in scores[:k]]


def _build_anchor_proposals(g, anchors, semantic_entity_scores, semantic_chunk_scores, cfg):
    topn_entity = max(0, int(getattr(cfg, "semantic_topn_entity", getattr(cfg, "semantic_topn", 50))))
    topn_chunk = max(0, int(getattr(cfg, "semantic_topn_chunk", max(1, topn_entity // 2))))
    reserve_topn = max(0, int(getattr(cfg, "graph_reserve_topn", 15)))
    dist_cap = max(1, int(getattr(cfg, "tau", 4)))

    proposal_by_anchor = {}
    proposal_scores = {}
    semantic_scores = {}
    graph_reserve_union = set()
    semantic_scores.update({str(k): float(v) for k, v in (semantic_entity_scores or {}).items()})
    semantic_scores.update({str(k): float(v) for k, v in (semantic_chunk_scores or {}).items()})

    semantic_entities = list((semantic_entity_scores or {}).items())
    semantic_chunks = list((semantic_chunk_scores or {}).items())

    for anchor in anchors:
        try:
            dmap = nx.single_source_shortest_path_length(g, anchor, cutoff=dist_cap)
        except Exception:
            dmap = {}

        ent_ranked = []
        for node, raw_score in semantic_entities:
            if node not in g:
                continue
            bonus = 0.0
            if node in dmap:
                bonus = 1.0 - (float(min(dist_cap, int(dmap[node]))) / float(dist_cap))
            ent_ranked.append((node, 0.90 * float(raw_score) + 0.10 * bonus))

        chunk_ranked = []
        for node, raw_score in semantic_chunks:
            if node not in g:
                continue
            bonus = 0.0
            if node in dmap:
                bonus = 1.0 - (float(min(dist_cap, int(dmap[node]))) / float(dist_cap))
            chunk_ranked.append((node, 0.90 * float(raw_score) + 0.10 * bonus))

        ent_ranked.sort(key=lambda x: x[1], reverse=True)
        chunk_ranked.sort(key=lambda x: x[1], reverse=True)
        semantic_nodes = [n for n, _ in ent_ranked[:topn_entity]] + [n for n, _ in chunk_ranked[:topn_chunk]]
        reserve_nodes = _anchor_graph_reserve_candidates(
            g=g,
            anchor=anchor,
            topn=reserve_topn,
            max_hops=max(2, int(getattr(cfg, "tau", 4))),
        )
        graph_reserve_union.update([node for node in reserve_nodes if node in g])
        merged = _ordered_unique([anchor] + semantic_nodes + reserve_nodes)
        proposal_by_anchor[anchor] = merged

        for rank, node in enumerate(merged):
            base = float(semantic_scores.get(node, 0.0))
            if node in reserve_nodes:
                base = max(base, 0.05)
            rank_decay = 1.0 / float(rank + 1)
            proposal_scores[node] = max(float(proposal_scores.get(node, -1.0e9)), base + 0.02 * rank_decay)

    proposal_nodes = set()
    for vals in proposal_by_anchor.values():
        proposal_nodes.update(vals)
    proposal_nodes = {node for node in proposal_nodes if node in g}
    proposal_diag = {
        "proposal_entity_count": int(sum(1 for node in semantic_entity_scores.keys() if node in g)),
        "proposal_chunk_count": int(sum(1 for node in semantic_chunk_scores.keys() if node in g)),
        "graph_reserve_count": int(len(graph_reserve_union)),
        "union_candidate_count": int(len(proposal_nodes)),
    }
    return proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores, proposal_diag


def _build_reduced_subgraph_from_proposals(g, anchors, proposal_by_anchor, proposal_scores, cfg):
    if g is None or g.number_of_nodes() <= 0:
        return g, {"applied": False, "reason": "empty_graph"}

    reduced_max_nodes = max(200, int(getattr(cfg, "ppr_subgraph_max_nodes", 30000)))
    path_hops = max(2, int(getattr(cfg, "tau", 4)) + 1)
    path_links_per_anchor = max(1, int(getattr(cfg, "graph_reserve_topn", 15)) // 2)

    keep = set()
    for anchor in anchors:
        keep.add(anchor)
        anchor_props = list(proposal_by_anchor.get(anchor, []) or [])
        keep.update(anchor_props)
        for node in anchor_props[:path_links_per_anchor]:
            if node == anchor or anchor not in g or node not in g:
                continue
            dist = _shortest_distance_with_cap(g, anchor, node, path_hops)
            if dist >= path_hops:
                continue
            try:
                path = nx.shortest_path(g, anchor, node)
            except Exception:
                path = []
            if path:
                keep.update(path)

    # Keep lightweight connector coverage for relation nodes.
    connector = set()
    for node in list(keep):
        if node not in g:
            continue
        for nbr in g.neighbors(node):
            if g.nodes[nbr].get("node_type") == "relation":
                connector.add(nbr)
    keep.update(connector)
    keep = {node for node in keep if node in g}

    if len(keep) > reduced_max_nodes:
        mandatory = {a for a in anchors if a in g}
        budget = max(0, reduced_max_nodes - len(mandatory))
        ranked = []
        for node in keep:
            if node in mandatory:
                continue
            base = float(proposal_scores.get(node, 0.0))
            degree_term = min(1.0, float(g.degree(node)) / 25.0)
            ranked.append((node, base + 0.02 * degree_term))
        ranked.sort(key=lambda x: x[1], reverse=True)
        keep = set(mandatory)
        keep.update([node for node, _ in ranked[:budget]])

    if not keep:
        return g, {"applied": False, "reason": "empty_keep_set"}
    if len(keep) >= g.number_of_nodes():
        return g, {"applied": False, "reason": "no_reduction"}

    reduced = g.subgraph(sorted(keep)).copy()
    return reduced, {
        "applied": True,
        "nodes_before": int(g.number_of_nodes()),
        "edges_before": int(g.number_of_edges()),
        "nodes_after": int(reduced.number_of_nodes()),
        "edges_after": int(reduced.number_of_edges()),
        "max_nodes": int(reduced_max_nodes),
    }


def _phase2_pair_shortlist(shortlisted_runs, anchors, g, query_sim_map, support_sim_map, cfg):
    topb = max(1, int(getattr(cfg, "pair_shortlist_topb", 6)))
    cap = max(2, int(getattr(cfg, "tau", 4)) + 1)
    pair_scores = {}

    for run in shortlisted_runs:
        run_id = int(run.get("run_id", 0))
        seeds = list(run.get("seeds", []) or [])
        if not seeds:
            continue
        seed_scores = run.get("seed_score_map", {}) or {}
        for anchor in anchors:
            anchor_scores = (run.get("anchor_scores", {}) or {}).get(anchor, {}) or {}
            for seed in seeds:
                if anchor not in g or seed not in g:
                    continue
                dist = _shortest_distance_with_cap(g, anchor, seed, cap)
                dist_score = 1.0 - (float(dist) / float(cap))
                anchor_align = float(anchor_scores.get(seed, 0.0))
                seed_strength = float(seed_scores.get(seed, 0.0))
                semantic_rel = 0.5 * (
                    max(float(query_sim_map.get(seed, 0.0)), float(support_sim_map.get(seed, 0.0))) + 1.0
                )

                bridge_hits = 0
                for other_anchor in anchors:
                    if other_anchor == anchor:
                        continue
                    d_other = _shortest_distance_with_cap(g, other_anchor, seed, cap)
                    if d_other < cap:
                        bridge_hits += 1
                bridge_potential = float(bridge_hits) / float(max(1, len(anchors) - 1))

                score = (
                    0.30 * float(anchor_align)
                    + 0.25 * float(seed_strength)
                    + 0.20 * float(semantic_rel)
                    + 0.15 * float(dist_score)
                    + 0.10 * float(bridge_potential)
                )
                pair = (anchor, seed)
                prev = pair_scores.get(pair)
                payload = {
                    "anchor": anchor,
                    "seed": seed,
                    "run_id": run_id,
                    "pair_proxy_score": float(score),
                    "distance": int(dist),
                    "anchor_alignment": float(anchor_align),
                    "seed_strength": float(seed_strength),
                    "semantic_relevance": float(semantic_rel),
                    "bridge_potential": float(bridge_potential),
                }
                if prev is None or float(payload["pair_proxy_score"]) > float(prev["pair_proxy_score"]):
                    pair_scores[pair] = payload

    ranked = sorted(pair_scores.values(), key=lambda x: x["pair_proxy_score"], reverse=True)
    return ranked[:topb]


def _phase2_refine_pair_bounded_local(g, pair_item, run_by_id, query_sim_map, support_sim_map, cfg):
    anchor = pair_item["anchor"]
    seed = pair_item["seed"]
    run_id = int(pair_item.get("run_id", 0))
    run = run_by_id.get(run_id, {})
    anchor_scores = (run.get("anchor_scores", {}) or {}).get(anchor, {}) or {}

    local_hops = max(2, int(getattr(cfg, "tau", 4)) + 1)
    max_local_nodes = max(80, int(getattr(cfg, "corridor_top_bc", 20)) * 8)
    corridor_top_bc = max(1, int(getattr(cfg, "corridor_top_bc", 20)))

    a_nodes = {}
    z_nodes = {}
    try:
        a_nodes = nx.single_source_shortest_path_length(g, anchor, cutoff=local_hops)
    except Exception:
        a_nodes = {anchor: 0}
    try:
        z_nodes = nx.single_source_shortest_path_length(g, seed, cutoff=local_hops)
    except Exception:
        z_nodes = {seed: 0}

    local_nodes = set(a_nodes.keys()).union(z_nodes.keys())
    if anchor in g and seed in g:
        try:
            sp = nx.shortest_path(g, anchor, seed)
        except Exception:
            sp = []
        local_nodes.update(sp)
    local_nodes.add(anchor)
    local_nodes.add(seed)

    if len(local_nodes) > max_local_nodes:
        rank = []
        for node in local_nodes:
            da = a_nodes.get(node, local_hops)
            dz = z_nodes.get(node, local_hops)
            dist_term = 1.0 / float(1 + min(da, dz))
            sem_term = 0.5 * (
                max(float(query_sim_map.get(node, 0.0)), float(support_sim_map.get(node, 0.0))) + 1.0
            )
            rank.append((node, 0.70 * dist_term + 0.30 * sem_term))
        rank.sort(key=lambda x: x[1], reverse=True)
        kept = [node for node, _ in rank[: max_local_nodes]]
        local_nodes = set(kept)
        local_nodes.update([anchor, seed])

    local_nodes = {node for node in local_nodes if node in g}
    local_subgraph = g.subgraph(sorted(local_nodes)).copy()
    if local_subgraph.number_of_nodes() <= 0:
        return None

    local_ppr = _personalized_pagerank(
        local_subgraph,
        source=anchor,
        alpha=float(getattr(cfg, "ppr_alpha", 0.15)),
        max_iter=min(80, int(getattr(cfg, "ppr_power_max_iter", 100))),
        tol=max(1.0e-6, float(getattr(cfg, "ppr_power_tol", 1.0e-6))),
        min_score=max(0.0, float(getattr(cfg, "ppr_min_score", 0.0))),
    )
    local_ppr_norm = _normalize_map(local_ppr)
    anchor_graph_norm = _normalize_map(anchor_scores)

    reverse_raw = {}
    for node in local_subgraph.nodes:
        d = _shortest_distance_with_cap(local_subgraph, node, seed, local_hops)
        reverse_raw[node] = 1.0 - (float(d) / float(local_hops))
    reverse_norm = _normalize_map(reverse_raw)

    semantic_raw = {}
    for node in local_subgraph.nodes:
        semantic_raw[node] = 0.5 * (
            max(float(query_sim_map.get(node, 0.0)), float(support_sim_map.get(node, 0.0))) + 1.0
        )
    semantic_norm = _normalize_map(semantic_raw)

    node_scores = {}
    for node in local_subgraph.nodes:
        node_scores[node] = (
            0.40 * float(local_ppr_norm.get(node, 0.0))
            + 0.25 * float(reverse_norm.get(node, 0.0))
            + 0.20 * float(anchor_graph_norm.get(node, 0.0))
            + 0.15 * float(semantic_norm.get(node, 0.0))
        )

    ranked = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    top_node_scores = ranked[:corridor_top_bc]
    if not top_node_scores:
        return None

    payload = _build_single_corridor_payload(
        g=g,
        anchor=anchor,
        seed=seed,
        corridor_id=f"c{int(run_id):02d}_{anchor}_{seed}",
        corridor_score=float(pair_item.get("pair_proxy_score", 0.0)),
        node_scores=top_node_scores,
    )
    payload["pair_proxy_score"] = float(pair_item.get("pair_proxy_score", 0.0))
    payload["refine_mode"] = str(getattr(cfg, "phase2_refine_mode", "bounded_local") or "bounded_local")
    return {
        "pair": (anchor, seed),
        "payload": payload,
        "node_scores": top_node_scores,
    }


def _phase2_local_refinement(g, shortlisted_pairs, shortlisted_runs, query_sim_map, support_sim_map, cfg):
    run_by_id = {int(r.get("run_id", 0)): r for r in (shortlisted_runs or [])}
    sentence_scores = {}
    corridor_nodes = set()
    corridor_payloads = []
    retained_pairs = []

    for pair_item in shortlisted_pairs:
        refined = _phase2_refine_pair_bounded_local(
            g=g,
            pair_item=pair_item,
            run_by_id=run_by_id,
            query_sim_map=query_sim_map,
            support_sim_map=support_sim_map,
            cfg=cfg,
        )
        if refined is None:
            continue

        anchor, seed = refined["pair"]
        retained_pairs.append((anchor, seed))
        corridor_payloads.append(refined["payload"])
        corridor_nodes.update([anchor, seed])
        for node, score in refined["node_scores"]:
            corridor_nodes.add(node)
            if g.nodes[node].get("node_type") == "sentence":
                sentence_scores[node] = max(float(sentence_scores.get(node, 0.0)), float(score))

    if not corridor_nodes:
        corridor_nodes = set()
        for pair_item in shortlisted_pairs:
            corridor_nodes.add(pair_item["anchor"])
            corridor_nodes.add(pair_item["seed"])
        corridor_nodes = {node for node in corridor_nodes if node in g}

    if corridor_nodes:
        corridor_graph = g.subgraph(sorted(corridor_nodes)).copy()
    else:
        corridor_graph = g.subgraph([]).copy()
    return corridor_graph, retained_pairs, sentence_scores, corridor_payloads


def _rerank_corridors_hybrid(corridors, g, query_sim_map, support_sim_map, cfg):
    if not corridors:
        return []

    sid_to_node = {}
    for node in g.nodes:
        if g.nodes[node].get("node_type") != "sentence":
            continue
        sid = str(g.nodes[node].get("sentence_id", node))
        sid_to_node[sid] = node

    reuse_semantic = bool(getattr(cfg, "reuse_semantic_scores_in_final", True))
    redundancy_w = max(0.0, float(getattr(cfg, "run_score_redundancy_weight", 0.10)))

    base_rows = []
    for corridor in corridors:
        main_ids = list(corridor.get("main_path_sentence_ids", []) or [])
        support_ids = list(corridor.get("support_sentence_ids", []) or [])
        sent_ids = _ordered_unique(main_ids + support_ids)

        structural = 0.0
        if main_ids:
            structural += 0.6
        structural += 0.2 * min(1.0, float(len(main_ids)) / 2.0)
        structural += 0.2 * min(1.0, float(len(support_ids)) / 2.0)

        sem_vals = []
        support_vals = []
        for sid in sent_ids:
            node = sid_to_node.get(sid)
            if node is None:
                continue
            if reuse_semantic:
                sem = 0.5 * (float(query_sim_map.get(node, 0.0)) + 1.0)
            else:
                sem = 0.0
            sup = 0.5 * (float(support_sim_map.get(node, 0.0)) + 1.0)
            sem_vals.append(sem)
            support_vals.append(sup)
        semantic_rel = float(sum(sem_vals) / max(len(sem_vals), 1)) if sem_vals else 0.0
        answer_support = float(sum(support_vals) / max(len(support_vals), 1)) if support_vals else 0.0

        pair_proxy = float(corridor.get("pair_proxy_score", corridor.get("corridor_score", 0.0)) or 0.0)
        base_score = 0.45 * structural + 0.25 * semantic_rel + 0.20 * answer_support + 0.10 * pair_proxy
        base_rows.append(
            {
                "corridor": dict(corridor),
                "base_score": float(base_score),
                "sentence_set": set(sent_ids),
            }
        )

    base_rows.sort(key=lambda x: x["base_score"], reverse=True)
    selected_sets = []
    reranked = []
    for item in base_rows:
        overlap_penalty = 0.0
        for prev in selected_sets:
            inter = len(item["sentence_set"].intersection(prev))
            union = len(item["sentence_set"].union(prev))
            if union > 0:
                overlap_penalty = max(overlap_penalty, float(inter) / float(union))
        final_score = float(item["base_score"]) - float(redundancy_w) * float(overlap_penalty)
        payload = dict(item["corridor"])
        payload["corridor_score"] = float(final_score)
        payload["final_score_components"] = {
            "base_score": float(item["base_score"]),
            "redundancy_penalty": float(overlap_penalty),
        }
        reranked.append(payload)
        selected_sets.append(set(item["sentence_set"]))

    reranked.sort(key=lambda x: float(x.get("corridor_score", 0.0)), reverse=True)
    return reranked


def run_graphrag_core(
    sample,
    cfg,
    method_name,
    stable_seed_selection,
    enable_trim,
):
    start = time.perf_counter()
    stage_ms = {
        # detailed timings
        "query_embed_ms": 0.0,
        "semantic_lookup_entity_ms": 0.0,
        "semantic_lookup_chunk_ms": 0.0,
        "proposal_union_ms": 0.0,
        "proposal_subgraph_build_ms": 0.0,
        "phase1_ppr_ms": 0.0,
        "phase1_run_scoring_ms": 0.0,
        "phase2_pair_shortlist_ms": 0.0,
        "phase2_refine_ms": 0.0,
        "sentence_rerank_ms": 0.0,
        "render_ms": 0.0,
        # backward-compatible aliases
        "proposal_time_ms": 0.0,
        "phase1_ppr_time_ms": 0.0,
        "run_scoring_time_ms": 0.0,
        "phase2_refinement_time_ms": 0.0,
        "final_render_time_ms": 0.0,
    }
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

    anchors = select_lexical_anchors(sample, g, int(getattr(cfg, "max_anchors", 5)))
    if not anchors:
        entities = [n for n in g.nodes if g.nodes[n].get("node_type") == "entity"]
        entities.sort(key=lambda n: g.degree(n), reverse=True)
        anchors = entities[: int(getattr(cfg, "max_anchors", 5))]

    rng = random.Random(int(getattr(cfg, "random_seed", 42)))
    show_inner_progress = bool(getattr(cfg, "show_inner_progress", True))
    if int(getattr(cfg, "num_workers", 1)) > 1:
        show_inner_progress = False

    ppr_engine = _resolve_ppr_engine(cfg, graph_scope=graph_scope, graph=g)
    proposal_start = time.perf_counter()

    semantic_diag = {
        "enabled": bool(getattr(cfg, "embedding_enabled", False)),
        "applied": False,
        "error": "",
        "model_name": str(getattr(cfg, "embedding_model_name", "") or ""),
        "semantic_topn_entity": int(
            getattr(cfg, "semantic_topn_entity", getattr(cfg, "semantic_topn", 50))
        ),
        "semantic_topn_chunk": int(
            getattr(
                cfg,
                "semantic_topn_chunk",
                max(1, int(getattr(cfg, "semantic_topn", 50)) // 2),
            )
        ),
        "graph_reserve_topn": int(getattr(cfg, "graph_reserve_topn", 15)),
        "index_available": False,
        "query_embedding_dim": 0,
        "semantic_entity_candidates": [],
        "semantic_chunk_candidates": [],
        "semantic_entity_scores": {},
        "semantic_chunk_scores": {},
        "proposal_by_anchor": {},
        "proposal_candidate_count": 0,
        "candidate_vector_count": 0,
    }

    query_vec = None
    semantic_state = None
    semantic_entity_scores = {}
    semantic_chunk_scores = {}
    semantic_entity_lookup_mode = "disabled"
    semantic_chunk_lookup_mode = "disabled"
    query_embedding_recomputed = False
    candidate_similarity_recomputed_count = 0

    if semantic_diag["enabled"]:
        query_embed_start = time.perf_counter()
        query_vec, qerr = _query_embedding(sample.question, cfg)
        stage_ms["query_embed_ms"] = float((time.perf_counter() - query_embed_start) * 1000.0)
        query_embedding_recomputed = True
        if query_vec is None:
            semantic_diag["error"] = qerr
            semantic_entity_lookup_mode = "error"
            semantic_chunk_lookup_mode = "error"
        else:
            semantic_diag["query_embedding_dim"] = int(query_vec.shape[0])
            if graph_scope in {"global_corpus", "prebuilt_igraph"}:
                semantic_state = _get_global_semantic_state(global_index_meta)
                semantic_diag["index_available"] = semantic_state is not None

            if semantic_state is not None:
                semantic_entity_lookup_mode = "cache"
                semantic_chunk_lookup_mode = "cache"
                semantic_entity_scores, semantic_chunk_scores, semantic_lookup_diag = _semantic_topk_candidates_by_type(
                    query_vec=query_vec,
                    semantic_state=semantic_state,
                    cfg=cfg,
                )
                stage_ms["semantic_lookup_entity_ms"] = float(semantic_lookup_diag.get("entity_ms", 0.0))
                stage_ms["semantic_lookup_chunk_ms"] = float(semantic_lookup_diag.get("chunk_ms", 0.0))
                # Reuse offline support cache to cheaply expand entity proposals into chunk proposals.
                support_cache = semantic_state.get("entity_topk_chunks_cache", {}) if isinstance(semantic_state, dict) else {}
                if isinstance(support_cache, dict):
                    for ent_node, ent_score in list(semantic_entity_scores.items()):
                        linked_chunks = list(support_cache.get(str(ent_node), []) or [])
                        for rank, chunk_node in enumerate(linked_chunks[: int(getattr(cfg, "semantic_topn_chunk", 15))]):
                            bonus = float(ent_score) * (0.85 - 0.03 * float(rank))
                            if chunk_node in g:
                                semantic_chunk_scores[chunk_node] = max(
                                    float(semantic_chunk_scores.get(chunk_node, -1.0)),
                                    float(bonus),
                                )
                semantic_diag["applied"] = True
            elif g.number_of_nodes() <= 5000:
                semantic_entity_lookup_mode = "bruteforce"
                semantic_chunk_lookup_mode = "bruteforce"
                local_lookup_start = time.perf_counter()
                local_nodes = [n for n in g.nodes if g.nodes[n].get("node_type") in {"entity", "sentence"}]
                local_vectors, local_err = _load_candidate_vectors(local_nodes, g, cfg, semantic_state=None)
                if local_err:
                    semantic_diag["error"] = local_err
                    semantic_entity_lookup_mode = "error"
                    semantic_chunk_lookup_mode = "error"
                else:
                    local_scores = {n: cosine_similarity(query_vec, vec) for n, vec in local_vectors.items()}
                    ent = [(n, s) for n, s in local_scores.items() if g.nodes[n].get("node_type") == "entity"]
                    chk = [(n, s) for n, s in local_scores.items() if g.nodes[n].get("node_type") == "sentence"]
                    ent.sort(key=lambda x: x[1], reverse=True)
                    chk.sort(key=lambda x: x[1], reverse=True)
                    t_ent = max(0, int(getattr(cfg, "semantic_topn_entity", getattr(cfg, "semantic_topn", 50))))
                    t_chk = max(0, int(getattr(cfg, "semantic_topn_chunk", max(1, t_ent // 2))))
                    semantic_entity_scores = {n: float(s) for n, s in ent[:t_ent]}
                    semantic_chunk_scores = {n: float(s) for n, s in chk[:t_chk]}
                    semantic_diag["applied"] = True
                local_ms = float((time.perf_counter() - local_lookup_start) * 1000.0)
                stage_ms["semantic_lookup_entity_ms"] = float(local_ms * 0.5)
                stage_ms["semantic_lookup_chunk_ms"] = float(local_ms * 0.5)
            else:
                semantic_diag["error"] = "semantic_index_unavailable_for_large_graph"
                semantic_entity_lookup_mode = "unavailable"
                semantic_chunk_lookup_mode = "unavailable"

    proposal_union_start = time.perf_counter()
    proposal_by_anchor, proposal_nodes, proposal_scores, semantic_scores, proposal_diag = _build_anchor_proposals(
        g=g,
        anchors=anchors,
        semantic_entity_scores=semantic_entity_scores,
        semantic_chunk_scores=semantic_chunk_scores,
        cfg=cfg,
    )
    stage_ms["proposal_union_ms"] = float((time.perf_counter() - proposal_union_start) * 1000.0)
    proposal_subgraph_start = time.perf_counter()
    reduced_graph, reduced_diag = _build_reduced_subgraph_from_proposals(
        g=g,
        anchors=anchors,
        proposal_by_anchor=proposal_by_anchor,
        proposal_scores=proposal_scores,
        cfg=cfg,
    )
    stage_ms["proposal_subgraph_build_ms"] = float((time.perf_counter() - proposal_subgraph_start) * 1000.0)
    if reduced_graph is None or reduced_graph.number_of_nodes() <= 0:
        reduced_graph = g
        reduced_diag = {"applied": False, "reason": "fallback_to_full_graph"}

    anchors = [a for a in anchors if a in reduced_graph]
    if not anchors:
        entities = [n for n in reduced_graph.nodes if reduced_graph.nodes[n].get("node_type") == "entity"]
        entities.sort(key=lambda n: reduced_graph.degree(n), reverse=True)
        anchors = entities[: int(getattr(cfg, "max_anchors", 5))]

    semantic_diag["semantic_entity_candidates"] = [str(n) for n in semantic_entity_scores.keys()]
    semantic_diag["semantic_chunk_candidates"] = [str(n) for n in semantic_chunk_scores.keys()]
    semantic_diag["semantic_entity_scores"] = {str(k): float(v) for k, v in semantic_entity_scores.items()}
    semantic_diag["semantic_chunk_scores"] = {str(k): float(v) for k, v in semantic_chunk_scores.items()}
    semantic_diag["proposal_by_anchor"] = {
        str(anchor): [str(n) for n in list(nodes or [])]
        for anchor, nodes in proposal_by_anchor.items()
    }
    semantic_diag["proposal_candidate_count"] = int(len(proposal_nodes))
    semantic_diag["proposal_entity_count"] = int(proposal_diag.get("proposal_entity_count", 0))
    semantic_diag["proposal_chunk_count"] = int(proposal_diag.get("proposal_chunk_count", 0))
    semantic_diag["graph_reserve_count"] = int(proposal_diag.get("graph_reserve_count", 0))
    semantic_diag["union_candidate_count"] = int(proposal_diag.get("union_candidate_count", len(proposal_nodes)))
    semantic_diag["query_embedding_recomputed"] = bool(query_embedding_recomputed)
    semantic_diag["semantic_entity_lookup_mode"] = str(semantic_entity_lookup_mode)
    semantic_diag["semantic_chunk_lookup_mode"] = str(semantic_chunk_lookup_mode)
    stage_ms["proposal_time_ms"] = float((time.perf_counter() - proposal_start) * 1000.0)

    phase1_start = time.perf_counter()
    per_anchor_runs = {a: [] for a in anchors}
    n_runs = max(1, int(getattr(cfg, "samples_per_anchor", 3)))
    phase1_parallel_enabled = bool(getattr(cfg, "phase1_parallel_ppr", True))
    original_ppr_parallel_workers = int(getattr(cfg, "ppr_parallel_workers", 1))
    if not phase1_parallel_enabled:
        setattr(cfg, "ppr_parallel_workers", 1)
    for _ in tqdm(
        range(n_runs),
        total=n_runs,
        desc="Phase1 stochastic PPR",
        leave=False,
        disable=not show_inner_progress,
    ):
        run_seed = int(rng.random() * 10**9)
        if ppr_engine == "power":
            h = _stochastic_perturb_graph(reduced_graph, rng, float(getattr(cfg, "edge_drop_prob", 0.1)))
            run_map = _compute_ppr_batch(
                g=h,
                sources=anchors,
                cfg=cfg,
                engine="power",
                run_seed=run_seed,
                show_progress=show_inner_progress,
                desc="Phase1 Anchor PPR",
            )
        else:
            run_map = _compute_ppr_batch(
                g=reduced_graph,
                sources=anchors,
                cfg=cfg,
                engine="mc",
                run_seed=run_seed,
                show_progress=show_inner_progress,
                desc="Phase1 Anchor PPR",
            )
        for anchor in anchors:
            per_anchor_runs[anchor].append(run_map.get(anchor, {}))
    if not phase1_parallel_enabled:
        setattr(cfg, "ppr_parallel_workers", original_ppr_parallel_workers)
    stage_ms["phase1_ppr_ms"] = float((time.perf_counter() - phase1_start) * 1000.0)
    stage_ms["phase1_ppr_time_ms"] = float(stage_ms["phase1_ppr_ms"])

    run_score_start = time.perf_counter()
    run_base = []
    for run_id in range(n_runs):
        agg_scores = {}
        anchor_scores = {}
        for anchor in anchors:
            scores = per_anchor_runs.get(anchor, [{}])[run_id]
            anchor_scores[anchor] = scores
            for node, score in scores.items():
                agg_scores[node] = agg_scores.get(node, 0.0) + float(score)
        graph_candidates = [n for n in _topk_nodes(agg_scores, int(getattr(cfg, "candidate_top_t", 20))) if n in reduced_graph]
        run_base.append(
            {
                "run_id": run_id,
                "anchor_scores": anchor_scores,
                "agg_scores": agg_scores,
                "graph_candidates": graph_candidates,
            }
        )

    candidate_universe = set(anchors).union(set(proposal_nodes))
    for run in run_base:
        candidate_universe.update(run.get("graph_candidates", []))
    candidate_universe = {node for node in candidate_universe if node in reduced_graph}

    node_vectors = {}
    if semantic_diag["enabled"] and query_vec is not None and candidate_universe:
        node_vectors, vector_err = _load_candidate_vectors(
            nodes=sorted(candidate_universe),
            g=reduced_graph,
            cfg=cfg,
            semantic_state=semantic_state,
        )
        if vector_err and not semantic_diag["error"]:
            semantic_diag["error"] = vector_err
    semantic_diag["candidate_vector_count"] = int(len(node_vectors))

    query_sim_map = {}
    if query_vec is not None and node_vectors:
        vector_nodes = list(node_vectors.keys())
        vec_mat = np.stack([node_vectors[node] for node in vector_nodes], axis=0).astype(np.float32, copy=False)
        qvec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = np.matmul(vec_mat, qvec)
        for node, sim in zip(vector_nodes, sims.tolist()):
            query_sim_map[node] = float(sim)
        candidate_similarity_recomputed_count += int(len(vector_nodes))
    for node, score in semantic_scores.items():
        query_sim_map[node] = max(float(score), float(query_sim_map.get(node, -1.0)))

    anchor_vecs = {anchor: node_vectors[anchor] for anchor in anchors if anchor in node_vectors}
    anchor_sim_map = {node: 0.0 for node in candidate_universe}
    if anchor_vecs and candidate_universe:
        anchor_nodes = list(anchor_vecs.keys())
        anchor_mat = np.stack([anchor_vecs[node] for node in anchor_nodes], axis=0).astype(np.float32, copy=False)
        sim_nodes = [node for node in candidate_universe if node in node_vectors]
        if sim_nodes:
            cand_mat = np.stack([node_vectors[node] for node in sim_nodes], axis=0).astype(np.float32, copy=False)
            sim_mat = np.matmul(cand_mat, anchor_mat.T)
            max_sims = np.max(sim_mat, axis=1)
            for node, sim in zip(sim_nodes, max_sims.tolist()):
                anchor_sim_map[node] = float(sim)
            candidate_similarity_recomputed_count += int(len(sim_nodes) * len(anchor_nodes))

    support_sim_map = {}
    for node in candidate_universe:
        support_sim_map[node] = _entity_support_similarity(reduced_graph, node, query_sim_map)

    semantic_union_enabled = bool(getattr(cfg, "semantic_candidate_union", True))
    distance_map_cache = {}
    proposal_candidates_union = []
    for anchor in anchors:
        proposal_candidates_union.extend(list(proposal_by_anchor.get(anchor, []) or []))
    proposal_candidates_union = _ordered_unique(proposal_candidates_union)
    run_results = []
    for run in run_base:
        graph_candidates = list(run.get("graph_candidates", []))
        if semantic_union_enabled:
            candidates = _ordered_unique(graph_candidates + proposal_candidates_union)
        else:
            candidates = list(graph_candidates)
        candidates = [node for node in candidates if node in reduced_graph]
        if not candidates:
            candidates = list(anchors)

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
        seeds = _greedy_seed_set(
            reduced_graph,
            candidates,
            objective_weights,
            int(getattr(cfg, "seed_k", 4)),
            int(getattr(cfg, "tau", 4)),
            distance_map_cache=distance_map_cache,
        )
        if not seeds and candidates:
            fallback = sorted(candidates, key=lambda n: float(seed_score_map.get(n, 0.0)), reverse=True)
            seeds = set(fallback[: max(1, int(getattr(cfg, "seed_k", 4)))])

        run_results.append(
            {
                "run_id": int(run["run_id"]),
                "anchor_scores": run["anchor_scores"],
                "agg_scores": run["agg_scores"],
                "graph_candidates": graph_candidates,
                "semantic_candidates": proposal_candidates_union,
                "candidates": candidates,
                "seeds": set(seeds),
                "seed_score_map": seed_score_map,
                "seed_score_components": {
                    "graph": graph_norm,
                    "semantic": semantic_norm,
                    "anchor": anchor_norm,
                },
            }
        )

    run_score_sparse_topk = max(16, int(getattr(cfg, "run_score_sparse_topk", 96)))
    for run in run_results:
        full_nodes = _ordered_unique(list(run.get("candidates", [])) + list(run.get("seeds", [])))
        if not full_nodes:
            full_nodes = list(run.get("seeds", [])) or list(anchors)
        base_weights = {
            node: float(run.get("seed_score_map", {}).get(node, run.get("agg_scores", {}).get(node, 0.0)))
            for node in full_nodes
        }
        score_nodes = list(full_nodes)
        if len(score_nodes) > run_score_sparse_topk:
            ranked_nodes = sorted(score_nodes, key=lambda n: float(base_weights.get(n, 0.0)), reverse=True)
            score_nodes = _ordered_unique(ranked_nodes[:run_score_sparse_topk] + list(run.get("seeds", [])))
        node_weights = {node: float(base_weights.get(node, 0.0)) for node in score_nodes}
        w = np.asarray([max(0.0, float(node_weights.get(node, 0.0))) for node in score_nodes], dtype=np.float32)
        if w.size <= 0:
            w = np.ones((1,), dtype=np.float32)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w = np.ones_like(w, dtype=np.float32)
            w_sum = float(np.sum(w))

        sem_vec = np.asarray([float(query_sim_map.get(node, 0.0)) for node in score_nodes], dtype=np.float32)
        anc_vec = np.asarray([float(anchor_sim_map.get(node, 0.0)) for node in score_nodes], dtype=np.float32)
        semantic_raw = float(np.dot(w, sem_vec) / max(w_sum, 1.0e-8))
        anchor_raw = float(np.dot(w, anc_vec) / max(w_sum, 1.0e-8))

        run["score_nodes"] = score_nodes
        run["score_node_weights"] = node_weights
        run["semantic_coverage_pre"] = max(0.0, min(1.0, 0.5 * (semantic_raw + 1.0)))
        run["anchor_alignment_pre"] = max(0.0, min(1.0, 0.5 * (anchor_raw + 1.0)))

    if stable_seed_selection and run_results:
        surrogate_universe = set(anchors)
        for run in run_results:
            surrogate_universe.update(run.get("candidates", []))

        surrogate_weights = {}
        for node in surrogate_universe:
            vals = [r["seed_score_map"].get(node, r["agg_scores"].get(node, 0.0)) for r in run_results]
            surrogate_weights[node] = sum(float(v) for v in vals) / max(len(vals), 1)
        if max(surrogate_weights.values(), default=0.0) <= 0.0:
            for node in surrogate_universe:
                vals = [r["agg_scores"].get(node, 0.0) for r in run_results]
                surrogate_weights[node] = sum(float(v) for v in vals) / max(len(vals), 1)

        surrogate_focus_topk = max(32, int(getattr(cfg, "run_score_surrogate_topk", 192)))
        surrogate_focus = sorted(surrogate_weights.items(), key=lambda kv: float(kv[1]), reverse=True)
        surrogate_focus_nodes = [node for node, _ in surrogate_focus[:surrogate_focus_topk]]
        if not surrogate_focus_nodes:
            surrogate_focus_nodes = list(surrogate_universe)

        w_sem = max(0.0, float(getattr(cfg, "run_score_semantic_weight", 0.30)))
        w_anchor = max(0.0, float(getattr(cfg, "run_score_anchor_weight", 0.20)))
        w_struct = max(0.0, float(getattr(cfg, "run_score_structure_weight", 0.25)))
        w_bridge = max(0.0, float(getattr(cfg, "run_score_bridge_weight", 0.15)))
        w_redundancy = max(0.0, float(getattr(cfg, "run_score_redundancy_weight", 0.10)))
        total = w_sem + w_anchor + w_struct + w_bridge + w_redundancy
        if total <= 0.0:
            w_sem, w_anchor, w_struct, w_bridge, w_redundancy = 0.30, 0.20, 0.25, 0.15, 0.10
            total = 1.0
        w_sem /= total
        w_anchor /= total
        w_struct /= total
        w_bridge /= total
        w_redundancy /= total

        best = None
        best_score = float("-inf")
        structural_cache = {}
        redundancy_cache = {}
        bridge_cache = {}
        for run in run_results:
            run_nodes = list(run.get("score_nodes", []) or [])
            semantic_cov = float(run.get("semantic_coverage_pre", 0.0))
            anchor_align = float(run.get("anchor_alignment_pre", 0.0))
            seeds_key = tuple(sorted(run.get("seeds", set())))
            if seeds_key not in structural_cache:
                structural_cache[seeds_key] = _run_structural_connectivity(
                    reduced_graph,
                    surrogate_universe=surrogate_focus_nodes,
                    surrogate_weights=surrogate_weights,
                    seeds=run.get("seeds", set()),
                    tau=int(getattr(cfg, "tau", 4)),
                    distance_map_cache=distance_map_cache,
                )
            structural_conn, dispersion_penalty, structural_loss = structural_cache[seeds_key]

            bridge_key = tuple(run_nodes)
            if bridge_key not in bridge_cache:
                bridge_cache[bridge_key] = _bridge_utility(
                    reduced_graph,
                    anchors=anchors,
                    nodes=run_nodes,
                    tau=int(getattr(cfg, "tau", 4)),
                    distance_map_cache=distance_map_cache,
                )
            bridge = float(bridge_cache[bridge_key])

            if seeds_key not in redundancy_cache:
                redundancy_cache[seeds_key] = _semantic_redundancy_penalty(
                    run.get("seeds", set()),
                    node_vectors=node_vectors,
                )
            redundancy = float(redundancy_cache[seeds_key])
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
        for run in run_results:
            run["hybrid_run_score"] = float(sum(run.get("seed_score_map", {}).values()))
            run["surrogate_loss"] = float(-run["hybrid_run_score"])
            run["global_loss"] = 0.0
            run["run_score_components"] = {}
        chosen = run_results[0] if run_results else {
            "run_id": 0,
            "anchor_scores": {},
            "graph_candidates": [],
            "semantic_candidates": [],
            "candidates": [],
            "seeds": set(),
            "agg_scores": {},
            "seed_score_map": {},
            "seed_score_components": {"graph": {}, "semantic": {}, "anchor": {}},
            "run_score_components": {},
            "hybrid_run_score": 0.0,
            "surrogate_loss": 0.0,
            "global_loss": 0.0,
        }

    shortlist_k = max(1, int(getattr(cfg, "phase1_run_shortlist_topk", 2)))
    ranked_runs = sorted(run_results, key=lambda r: float(r.get("hybrid_run_score", 0.0)), reverse=True)
    shortlisted_runs = ranked_runs[:shortlist_k] if ranked_runs else [chosen]
    chosen = shortlisted_runs[0]
    stage_ms["phase1_run_scoring_ms"] = float((time.perf_counter() - run_score_start) * 1000.0)
    stage_ms["run_scoring_time_ms"] = float(stage_ms["phase1_run_scoring_ms"])

    phase2_start = time.perf_counter()
    pair_shortlist_start = time.perf_counter()
    pair_shortlist = _phase2_pair_shortlist(
        shortlisted_runs=shortlisted_runs,
        anchors=anchors,
        g=reduced_graph,
        query_sim_map=query_sim_map,
        support_sim_map=support_sim_map,
        cfg=cfg,
    )
    if not pair_shortlist and chosen.get("seeds"):
        fallback = []
        for anchor in anchors:
            for seed in list(chosen.get("seeds", [])):
                fallback.append(
                    {
                        "anchor": anchor,
                        "seed": seed,
                        "run_id": int(chosen.get("run_id", 0)),
                        "pair_proxy_score": float(chosen.get("seed_score_map", {}).get(seed, 0.0)),
                        "distance": int(_shortest_distance_with_cap(reduced_graph, anchor, seed, int(getattr(cfg, "tau", 4)) + 1)),
                        "anchor_alignment": float((chosen.get("anchor_scores", {}).get(anchor, {}) or {}).get(seed, 0.0)),
                        "seed_strength": float(chosen.get("seed_score_map", {}).get(seed, 0.0)),
                        "semantic_relevance": float(
                            0.5
                            * (
                                max(float(query_sim_map.get(seed, 0.0)), float(support_sim_map.get(seed, 0.0)))
                                + 1.0
                            )
                        ),
                        "bridge_potential": 0.0,
                    }
                )
        fallback.sort(key=lambda x: x["pair_proxy_score"], reverse=True)
        pair_shortlist = fallback[: max(1, int(getattr(cfg, "pair_shortlist_topb", 6)))]
    stage_ms["phase2_pair_shortlist_ms"] = float((time.perf_counter() - pair_shortlist_start) * 1000.0)

    phase2_refine_start = time.perf_counter()
    corridor, retained_pairs, sentence_scores, corridor_payloads = _phase2_local_refinement(
        g=reduced_graph,
        shortlisted_pairs=pair_shortlist,
        shortlisted_runs=shortlisted_runs,
        query_sim_map=query_sim_map,
        support_sim_map=support_sim_map,
        cfg=cfg,
    )
    stage_ms["phase2_refine_ms"] = float((time.perf_counter() - phase2_refine_start) * 1000.0)
    stage_ms["phase2_refinement_time_ms"] = float((time.perf_counter() - phase2_start) * 1000.0)

    final_start = time.perf_counter()
    corridor_payloads = _rerank_corridors_hybrid(
        corridors=corridor_payloads,
        g=reduced_graph,
        query_sim_map=query_sim_map,
        support_sim_map=support_sim_map,
        cfg=cfg,
    )

    final_graph = corridor
    if enable_trim and bool(getattr(cfg, "trim_on", True)):
        final_graph = _greedy_trim(
            corridor.copy(),
            anchors=anchors,
            seeds=set(chosen.get("seeds", set())),
            rho=float(getattr(cfg, "trim_rho", 0.6)),
        )

    selected_nodes = set(final_graph.nodes())
    selected_sentence_ids, selected_sentences, selected_sentence_score_map = _extract_sentence_payload(
        g,
        selected_nodes,
        sentence_scores,
    )

    sentence_rerank_enabled = bool(getattr(cfg, "sentence_rerank_enabled", True))
    embedding_diag = {
        "enabled": bool(getattr(cfg, "embedding_enabled", False) and sentence_rerank_enabled),
        "sentence_rerank_enabled": bool(sentence_rerank_enabled),
        "applied": False,
        "error": "",
        "model_name": str(getattr(cfg, "embedding_model_name", "") or ""),
        "weight": float(getattr(cfg, "embedding_weight", 0.35)),
        "rerank_topn": int(getattr(cfg, "embedding_rerank_topn", 80)),
        "batch_size": int(getattr(cfg, "embedding_batch_size", 16)),
        "max_length": int(getattr(cfg, "embedding_max_length", 192)),
        "max_chars": int(getattr(cfg, "embedding_text_max_chars", 600)),
        "head_size": 0,
    }
    sentence_rerank_semantic_calls = 0
    if embedding_diag["enabled"] and selected_sentence_ids:
        sentence_rerank_start = time.perf_counter()
        sentence_rerank_semantic_calls += 1
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
        stage_ms["sentence_rerank_ms"] = float((time.perf_counter() - sentence_rerank_start) * 1000.0)
    elif not bool(sentence_rerank_enabled):
        embedding_diag["error"] = "sentence_rerank_disabled_by_config"

    filtered_corridors = _filter_corridor_payloads(corridor_payloads, selected_sentence_ids)
    corridor_count_before_trim = int(len(corridor_payloads))
    corridor_count_after_trim = int(len(filtered_corridors))
    sentence_feature_table = _build_sentence_feature_table(
        sample=sample,
        selected_sentence_ids=selected_sentence_ids,
        sentence_texts=selected_sentences,
        sentence_score_map=selected_sentence_score_map,
        corridors=filtered_corridors,
    )
    final_total_ms = float((time.perf_counter() - final_start) * 1000.0)
    stage_ms["render_ms"] = max(0.0, float(final_total_ms) - float(stage_ms.get("sentence_rerank_ms", 0.0)))
    stage_ms["final_render_time_ms"] = float(stage_ms["render_ms"])

    anchor_results = []
    diag_topn = max(1, int(getattr(cfg, "anchor_diag_topn", 10)))
    diag_full = bool(getattr(cfg, "anchor_diag_store_full_scores", False))
    for anchor in anchors:
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
        seeds=sorted(chosen.get("seeds", set())),
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
            "phase1_parallel_ppr": bool(getattr(cfg, "phase1_parallel_ppr", True)),
            "phase2_refine_mode": str(getattr(cfg, "phase2_refine_mode", "bounded_local") or "bounded_local"),
            "phase2_bidirectional_full_ppr": bool(getattr(cfg, "phase2_bidirectional_full_ppr", False)),
            "reuse_semantic_scores_in_final": bool(getattr(cfg, "reuse_semantic_scores_in_final", True)),
            "ppr_graph_nodes": int(reduced_graph.number_of_nodes()),
            "ppr_graph_edges": int(reduced_graph.number_of_edges()),
            "proposal_reduced_subgraph": reduced_diag,
            "proposal_entity_count": int(proposal_diag.get("proposal_entity_count", 0)),
            "proposal_chunk_count": int(proposal_diag.get("proposal_chunk_count", 0)),
            "graph_reserve_count": int(proposal_diag.get("graph_reserve_count", 0)),
            "union_candidate_count": int(proposal_diag.get("union_candidate_count", len(proposal_nodes))),
            "proposal_subgraph_nodes": int(reduced_graph.number_of_nodes()),
            "proposal_subgraph_edges": int(reduced_graph.number_of_edges()),
            "phase1_run_count": int(n_runs),
            "selected_run_count": int(len(shortlisted_runs)),
            "phase2_refined_pair_count": int(len(retained_pairs)),
            "best_run_id": int(chosen.get("run_id", 0)),
            "best_run_score": float(chosen.get("hybrid_run_score", 0.0) or 0.0),
            "best_run_score_components": chosen.get("run_score_components", {}) or {},
            "best_seed_score_components": chosen.get("seed_score_components", {}) or {},
            "num_seeds": int(len(chosen.get("seeds", set()))),
            "num_candidates_graph": int(len(chosen.get("graph_candidates", []) or [])),
            "num_candidates_semantic": int(len(chosen.get("semantic_candidates", []) or [])),
            "num_candidates_union": int(len(chosen.get("candidates", []) or [])),
            "retained_pairs": [[a, z] for a, z in retained_pairs],
            "pair_shortlist": pair_shortlist,
            "num_corridors": int(len(filtered_corridors)),
            "corridor_count_before_trim": int(corridor_count_before_trim),
            "corridor_count_after_trim": int(corridor_count_after_trim),
            "corridor_nodes_before_trim": int(corridor.number_of_nodes()),
            "corridor_nodes_after_trim": int(final_graph.number_of_nodes()),
            "sentence_scores": sentence_scores,
            "sentence_feature_table": sentence_feature_table,
            "semantic_selection": semantic_diag,
            "query_embedding_recomputed": bool(query_embedding_recomputed),
            "semantic_entity_lookup_mode": str(semantic_entity_lookup_mode),
            "semantic_chunk_lookup_mode": str(semantic_chunk_lookup_mode),
            "candidate_similarity_recomputed_count": int(candidate_similarity_recomputed_count),
            "semantic_scores_reused_in_final": bool(getattr(cfg, "reuse_semantic_scores_in_final", True)),
            "sentence_rerank_semantic_calls": int(sentence_rerank_semantic_calls),
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
            "shortlisted_run_ids": [int(r.get("run_id", 0)) for r in shortlisted_runs],
            "embedding_rerank": embedding_diag,
            "stable_seed_selection": stable_seed_selection,
            "trim_enabled": bool(enable_trim and getattr(cfg, "trim_on", True)),
            "anchor_diag_topn": int(diag_topn),
            "anchor_diag_store_full_scores": bool(diag_full),
            "latency_breakdown_ms": stage_ms,
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
