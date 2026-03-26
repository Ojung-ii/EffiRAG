import multiprocessing as mp
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path

import networkx as nx

from .anchors import select_lexical_anchors
from .config import RetrievalConfig
from .embedding import rerank_sentences_by_embedding
from .global_index import load_or_build_global_index
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
    memo_key = (
        str(Path(corpus_path).resolve()) if corpus_path else "",
        str(Path(prebuilt_igraph_path).resolve()) if prebuilt_igraph_path else "",
        prebuilt_igraph_format,
        prebuilt_entity_token_limit,
        str(Path(cache_dir).resolve()),
        openie_mode,
        openie_model_name,
        openie_text_max_chars,
        openie_max_new_tokens,
        openie_local_files_only,
        openie_retry_attempts,
        openie_retry_backoff_sec,
        openie_error_sample_limit,
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
        openie_mode=openie_mode,
        openie_model_name=openie_model_name,
        openie_text_max_chars=openie_text_max_chars,
        openie_max_new_tokens=openie_max_new_tokens,
        openie_local_files_only=openie_local_files_only,
        openie_retry_attempts=openie_retry_attempts,
        openie_retry_backoff_sec=openie_retry_backoff_sec,
        openie_error_sample_limit=openie_error_sample_limit,
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

    run_results = []
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

        candidates = _topk_nodes(agg_scores, cfg.candidate_top_t)
        weights = {node: agg_scores[node] for node in candidates}
        seeds = _greedy_seed_set(ppr_graph, candidates, weights, cfg.seed_k, cfg.tau)

        run_results.append(
            {
                "run_id": run_id,
                "anchor_scores": anchor_scores,
                "agg_scores": agg_scores,
                "candidates": candidates,
                "seeds": seeds,
            }
        )

    if stable_seed_selection and run_results:
        surrogate_universe = set(anchors)
        for run in run_results:
            surrogate_universe.update(run["candidates"])

        surrogate_weights = {}
        for node in surrogate_universe:
            vals = [r["agg_scores"].get(node, 0.0) for r in run_results]
            surrogate_weights[node] = sum(vals) / max(len(vals), 1)

        best = None
        best_loss = float("inf")
        for run in run_results:
            seeds = run["seeds"]
            loss = 0.0
            for node in surrogate_universe:
                loss += surrogate_weights[node] * _min_set_distance(ppr_graph, node, seeds, cfg.tau)
            run["global_loss"] = loss
            if loss < best_loss:
                best_loss = loss
                best = run

        chosen = best if best is not None else run_results[0]
    else:
        chosen = (
            run_results[0]
            if run_results
            else {
                "run_id": 0,
                "anchor_scores": {},
                "candidates": [],
                "seeds": set(),
                "agg_scores": {},
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
            max_chars=600,
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
            "num_seeds": len(chosen["seeds"]),
            "retained_pairs": [[a, z] for a, z in retained_pairs],
            "num_corridors": len(filtered_corridors),
            "corridor_nodes_before_trim": corridor.number_of_nodes(),
            "corridor_nodes_after_trim": final_graph.number_of_nodes(),
            "sentence_scores": sentence_scores,
            "sentence_feature_table": sentence_feature_table,
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
