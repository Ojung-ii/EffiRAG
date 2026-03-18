import random
import time
from math import sqrt

import networkx as nx

from .anchors import select_lexical_anchors
from .config import RetrievalConfig
from .graph import build_document_entity_graph
from .registry import register_method
from .types import AnchorResult, RetrievalResult


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
        return g.subgraph(anchors + list(seeds)).copy(), [], {}

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

    corridor_nodes = set()
    sentence_scores = {}

    for anchor, seed in retained_pairs:
        a_scores = anchor_scores_by_node.get(anchor, {})
        z_scores = seed_score_map.get(seed, {})
        node_scores = []

        for node in g.nodes:
            s = _pair_support(a_scores, z_scores, node)
            node_scores.append((node, s))

        node_scores.sort(key=lambda x: x[1], reverse=True)
        for node, score in node_scores[:corridor_top_bc]:
            corridor_nodes.add(node)
            sentence_scores[node] = max(sentence_scores.get(node, 0.0), score)

        corridor_nodes.add(anchor)
        corridor_nodes.add(seed)

    h = g.subgraph(corridor_nodes).copy()
    return h, retained_pairs, sentence_scores


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
    return sentence_ids, sentence_text


def run_graphrag_core(
    sample,
    cfg,
    method_name,
    stable_seed_selection,
    enable_trim,
):
    start = time.perf_counter()

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

    corridor, retained_pairs, sentence_scores = _build_corridor(
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
    selected_sentence_ids, selected_sentences = _extract_sentence_payload(g, selected_nodes, sentence_scores)

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
        anchor_results=anchor_results,
        diagnostics={
            "best_run_id": chosen.get("run_id", 0),
            "num_seeds": len(chosen["seeds"]),
            "retained_pairs": [[a, z] for a, z in retained_pairs],
            "corridor_nodes_before_trim": corridor.number_of_nodes(),
            "corridor_nodes_after_trim": final_graph.number_of_nodes(),
            "sentence_scores": sentence_scores,
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
