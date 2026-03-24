import random
import time
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

_GLOBAL_INDEX_MEMO = {}


def _get_global_graph(cfg):
    corpus_path = str(getattr(cfg, "global_corpus_path", "") or "").strip()
    if not corpus_path:
        return None, {}

    cache_dir = str(getattr(cfg, "graph_cache_dir", "") or "outputs/index_cache").strip()
    force_rebuild = bool(getattr(cfg, "force_rebuild_graph_index", False))
    openie_mode = str(getattr(cfg, "openie_mode", "llm") or "llm").strip().lower()
    openie_model_name = str(getattr(cfg, "openie_model_name", "") or "").strip()
    openie_text_max_chars = int(getattr(cfg, "openie_text_max_chars", 2200))
    openie_max_new_tokens = int(getattr(cfg, "openie_max_new_tokens", 256))
    memo_key = (
        str(Path(corpus_path).resolve()),
        str(Path(cache_dir).resolve()),
        openie_mode,
        openie_model_name,
        openie_text_max_chars,
        openie_max_new_tokens,
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
        openie_mode=openie_mode,
        openie_model_name=openie_model_name,
        openie_text_max_chars=openie_text_max_chars,
        openie_max_new_tokens=openie_max_new_tokens,
    )
    _GLOBAL_INDEX_MEMO[memo_key] = (graph, dict(meta))
    meta = dict(meta)
    meta["memory_cache_hit"] = False
    return graph, meta


def _personalized_pagerank(g, source, alpha):
    if source not in g:
        return {}

    # Personalized PageRank with no SciPy dependency.
    # alpha is teleport probability in this project config.
    nodes = list(g.nodes)
    if not nodes:
        return {}

    n = len(nodes)
    teleport = alpha
    damping = 1.0 - alpha
    tol = 1.0e-6
    max_iter = 100

    personalization = {node: 0.0 for node in nodes}
    personalization[source] = 1.0

    ranks = dict(personalization)
    dangling = [node for node in nodes if g.degree(node) == 0]

    for _ in range(max_iter):
        prev = ranks
        ranks = {node: teleport * personalization[node] for node in nodes}

        dangling_mass = damping * sum(prev[node] for node in dangling)
        if dangling_mass > 0.0:
            for node in nodes:
                ranks[node] += dangling_mass * personalization[node]

        for node in nodes:
            deg = g.degree(node)
            if deg == 0:
                continue
            share = damping * prev[node] / float(deg)
            for nbr in g.neighbors(node):
                ranks[nbr] += share

        err = sum(abs(ranks[node] - prev[node]) for node in nodes)
        if err < n * tol:
            break

    norm = sum(ranks.values())
    if norm <= 0.0:
        return {source: 1.0}
    return {node: score / norm for node, score in ranks.items()}


def _topk_nodes(scores, k):
    if k <= 0:
        return []
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def _stochastic_perturb_graph(g, rng, drop_prob):
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
    alpha,
    pair_top_lp,
    corridor_top_bc,
):
    if not anchors or not seeds:
        return g.subgraph(anchors + list(seeds)).copy(), [], {}, []

    seed_score_map = {}
    for seed in seeds:
        seed_score_map[seed] = _personalized_pagerank(g, seed, alpha=alpha)

    pair_stats = []
    for anchor in anchors:
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

    for idx, (anchor, seed) in enumerate(retained_pairs, start=1):
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

    if str(getattr(cfg, "global_corpus_path", "") or "").strip():
        g, global_index_meta = _get_global_graph(cfg)
        graph_scope = "global_corpus"
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

    for _ in range(max(1, cfg.samples_per_anchor)):
        h = _stochastic_perturb_graph(g, rng, cfg.edge_drop_prob)
        for anchor in anchors:
            per_anchor_runs[anchor].append(_personalized_pagerank(h, anchor, alpha=cfg.ppr_alpha))

    run_results = []
    num_runs = max(1, cfg.samples_per_anchor)
    for run_id in range(num_runs):
        agg_scores = {}
        anchor_scores = {}

        for anchor in anchors:
            scores = per_anchor_runs[anchor][run_id]
            anchor_scores[anchor] = scores
            for node, score in scores.items():
                agg_scores[node] = agg_scores.get(node, 0.0) + score

        candidates = _topk_nodes(agg_scores, cfg.candidate_top_t)
        weights = {node: agg_scores[node] for node in candidates}
        seeds = _greedy_seed_set(g, candidates, weights, cfg.seed_k, cfg.tau)

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
                loss += surrogate_weights[node] * _min_set_distance(g, node, seeds, cfg.tau)
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
        g=g,
        anchors=anchors,
        seeds=chosen["seeds"],
        anchor_scores_by_node=chosen["anchor_scores"],
        alpha=cfg.ppr_alpha,
        pair_top_lp=cfg.pair_top_lp,
        corridor_top_bc=cfg.corridor_top_bc,
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
    for anchor in anchors:
        for sample_idx, scores in enumerate(per_anchor_runs.get(anchor, [])):
            top_candidates = _topk_nodes(scores, 10)
            anchor_results.append(
                AnchorResult(
                    anchor=anchor,
                    scores={k: float(v) for k, v in scores.items() if v > 0.0},
                    top_candidates=top_candidates,
                    sample_index=sample_idx,
                    metadata={"topn": 10},
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
            "global_index": global_index_meta if graph_scope == "global_corpus" else {},
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
