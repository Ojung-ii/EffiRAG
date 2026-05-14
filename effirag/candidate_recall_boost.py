from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import networkx as nx


def _ordered_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in list(items or []):
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _node_type(g, node: str) -> str:
    if node not in g:
        return ""
    return str((g.nodes[node] or {}).get("node_type", "") or "").strip().lower()


def _is_candidate_node(g, node: str) -> bool:
    ntype = _node_type(g, node)
    return ntype in {"entity", "sentence", "chunk", "passage", "document"}


def _node_text_signature(g, node: str) -> str:
    if node not in g:
        return str(node)
    data = g.nodes[node] or {}
    text = str(data.get("text", data.get("content", "")) or "").strip().lower()
    text = " ".join(text.split())
    if text:
        return text[:256]
    return str(node)


def _node_source_key(g, node: str) -> str:
    if node not in g:
        return ""
    data = g.nodes[node] or {}
    for key in ("source_id", "doc_id", "document_id", "title", "source", "doc_title"):
        value = str(data.get(key, "") or "").strip().lower()
        if value:
            return value
    return ""


def _node_entity_keys(g, node: str) -> List[str]:
    if node not in g:
        return []
    if _node_type(g, node) == "entity":
        return [str(node)]
    keys: List[str] = []
    try:
        for nbr in g.neighbors(node):
            if _node_type(g, nbr) == "entity":
                keys.append(str(nbr))
    except Exception:
        return []
    return _ordered_unique(keys[:4])


def _safe_shortest_path(g, src: str, dst: str, cutoff: int) -> List[str]:
    if src not in g or dst not in g:
        return []
    if src == dst:
        return [src]
    try:
        dmap = nx.single_source_shortest_path_length(g, src, cutoff=max(1, int(cutoff)))
    except Exception:
        return []
    if dst not in dmap:
        return []
    try:
        return [str(x) for x in nx.shortest_path(g, src, dst)]
    except Exception:
        return []


def _collect_path_candidates(
    g,
    anchors: Sequence[str],
    ranked_targets: Sequence[str],
    proposal_scores: Mapping[str, float],
    semantic_scores: Mapping[str, float],
    max_path_candidates: int,
    max_hops: int,
) -> Dict[str, float]:
    cap = max(0, int(max_path_candidates))
    if cap <= 0:
        return {}
    out: Dict[str, float] = {}
    for anchor in list(anchors or []):
        if anchor not in g:
            continue
        for target in list(ranked_targets or []):
            if len(out) >= cap:
                return out
            if target == anchor or target not in g:
                continue
            path = _safe_shortest_path(g, anchor, target, cutoff=max_hops)
            if len(path) < 3:
                continue
            plen = max(1, len(path) - 1)
            target_score = float(proposal_scores.get(target, semantic_scores.get(target, 0.0)))
            for step, node in enumerate(path[1:-1], start=1):
                if not _is_candidate_node(g, node):
                    continue
                midpoint_bonus = 1.0 - abs((float(step) / float(plen)) - 0.5)
                degree_bonus = min(1.0, float(g.degree(node)) / 20.0) if node in g else 0.0
                score = float(target_score) + 0.12 * float(midpoint_bonus) + 0.05 * float(degree_bonus)
                if score > float(out.get(node, -1.0e9)):
                    out[node] = float(score)
                if len(out) >= cap:
                    return out
    return out


def _collect_anchor_expansion_candidates(
    g,
    anchors: Sequence[str],
    semantic_scores: Mapping[str, float],
    max_hops: int,
) -> Dict[str, float]:
    hop_cap = max(0, int(max_hops))
    if hop_cap <= 0:
        return {}
    out: Dict[str, float] = {}
    for anchor in list(anchors or []):
        if anchor not in g:
            continue
        try:
            dmap = nx.single_source_shortest_path_length(g, anchor, cutoff=hop_cap)
        except Exception:
            continue
        for node, dist in dmap.items():
            node = str(node)
            if node == anchor or not _is_candidate_node(g, node):
                continue
            hop = int(dist)
            if hop <= 0 or hop > hop_cap:
                continue
            semantic = float(semantic_scores.get(node, 0.0))
            proximity = 1.0 - float(hop) / float(max(1, hop_cap + 1))
            degree_bonus = min(1.0, float(g.degree(node)) / 25.0) if node in g else 0.0
            score = float(semantic) + 0.10 * float(proximity) + 0.04 * float(degree_bonus)
            if score > float(out.get(node, -1.0e9)):
                out[node] = float(score)
    return out


def _select_diverse(
    g,
    ranked_nodes: Sequence[Tuple[str, float]],
    *,
    max_count: int,
    candidate_diversity_enabled: bool,
    entity_diversity_enabled: bool,
    source_diversity_enabled: bool,
    candidate_dedup_enabled: bool,
) -> Tuple[List[str], Dict[str, int]]:
    cap = max(0, int(max_count))
    if cap <= 0:
        return [], {"dedup_skipped": 0, "diversity_skipped": 0}

    source_cap = max(1, int(cap // 4))
    entity_cap = max(1, int(cap // 5))
    source_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    seen_sig = set()
    seen_nodes = set()
    selected: List[str] = []
    deferred: List[str] = []
    dedup_skipped = 0
    diversity_skipped = 0

    for node, _ in list(ranked_nodes or []):
        node = str(node)
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        if candidate_dedup_enabled:
            sig = _node_text_signature(g, node)
            if sig in seen_sig:
                dedup_skipped += 1
                continue
        else:
            sig = ""
        source_key = _node_source_key(g, node)
        entity_keys = _node_entity_keys(g, node)
        violates = False
        if candidate_diversity_enabled:
            if source_diversity_enabled and source_key and source_counts[source_key] >= source_cap:
                violates = True
            if entity_diversity_enabled and entity_keys:
                if all(entity_counts[key] >= entity_cap for key in entity_keys):
                    violates = True
        if violates:
            diversity_skipped += 1
            deferred.append(node)
            continue
        selected.append(node)
        if candidate_dedup_enabled and sig:
            seen_sig.add(sig)
        if source_key:
            source_counts[source_key] += 1
        for key in entity_keys:
            entity_counts[key] += 1
        if len(selected) >= cap:
            break

    if len(selected) < cap and deferred:
        for node in deferred:
            if node in selected:
                continue
            if candidate_dedup_enabled:
                sig = _node_text_signature(g, node)
                if sig in seen_sig:
                    continue
                if sig:
                    seen_sig.add(sig)
            selected.append(node)
            if len(selected) >= cap:
                break

    return selected[:cap], {"dedup_skipped": int(dedup_skipped), "diversity_skipped": int(diversity_skipped)}


def apply_candidate_recall_boost(
    *,
    g,
    anchors: Sequence[str],
    proposal_by_anchor: MutableMapping[str, List[str]],
    proposal_nodes: Iterable[str],
    proposal_scores: MutableMapping[str, float],
    semantic_scores: MutableMapping[str, float],
    cfg: Any,
) -> Tuple[Dict[str, List[str]], set[str], Dict[str, float], Dict[str, float], Dict[str, Any]]:
    flags = {
        "path_candidate_expansion_enabled": bool(getattr(cfg, "path_candidate_expansion_enabled", False)),
        "anchor_expansion_enabled": bool(getattr(cfg, "anchor_expansion_enabled", False)),
        "candidate_diversity_enabled": bool(getattr(cfg, "candidate_diversity_enabled", False)),
        "entity_diversity_enabled": bool(getattr(cfg, "entity_diversity_enabled", False)),
        "source_diversity_enabled": bool(getattr(cfg, "source_diversity_enabled", False)),
        "candidate_dedup_enabled": bool(getattr(cfg, "candidate_dedup_enabled", True)),
    }
    enabled = any(
        [
            flags["path_candidate_expansion_enabled"],
            flags["anchor_expansion_enabled"],
            flags["candidate_diversity_enabled"],
            flags["entity_diversity_enabled"],
            flags["source_diversity_enabled"],
        ]
    )
    if not enabled:
        return (
            dict(proposal_by_anchor),
            {str(x) for x in list(proposal_nodes or []) if str(x)},
            {str(k): float(v) for k, v in dict(proposal_scores or {}).items()},
            {str(k): float(v) for k, v in dict(semantic_scores or {}).items()},
            {
                "enabled": False,
                "applied": False,
                "reason": "disabled",
                **flags,
            },
        )

    anchors_in_graph = [str(a) for a in _ordered_unique(anchors) if str(a) in g]
    existing_nodes = [str(n) for n in _ordered_unique(proposal_nodes) if str(n) in g]
    ranked_existing = sorted(
        existing_nodes,
        key=lambda n: float(proposal_scores.get(n, semantic_scores.get(n, 0.0))),
        reverse=True,
    )
    max_expanded = max(0, int(getattr(cfg, "max_expanded_candidates", 64)))
    max_path = max(0, int(getattr(cfg, "max_path_candidates", 24)))
    max_bridge = max(0, int(getattr(cfg, "max_bridge_candidates", max_path)))
    max_hops = max(1, int(getattr(cfg, "max_anchor_expansion_hops", 1)))
    tau_hops = max(2, int(getattr(cfg, "tau", 4)) + 1)

    path_scores: Dict[str, float] = {}
    anchor_scores: Dict[str, float] = {}
    if flags["path_candidate_expansion_enabled"] and max_path > 0:
        target_limit = max(4, min(len(ranked_existing), max_path * 3))
        target_nodes = ranked_existing[:target_limit]
        path_scores = _collect_path_candidates(
            g,
            anchors_in_graph,
            target_nodes,
            proposal_scores,
            semantic_scores,
            max_path_candidates=min(max_path, max_bridge if max_bridge > 0 else max_path),
            max_hops=tau_hops,
        )
    if flags["anchor_expansion_enabled"] and max_hops > 0:
        anchor_scores = _collect_anchor_expansion_candidates(
            g,
            anchors_in_graph,
            semantic_scores,
            max_hops=max_hops,
        )

    combined_scores: Dict[str, float] = {}
    for node, score in list(path_scores.items()):
        combined_scores[str(node)] = max(float(combined_scores.get(str(node), -1.0e9)), float(score))
    for node, score in list(anchor_scores.items()):
        combined_scores[str(node)] = max(float(combined_scores.get(str(node), -1.0e9)), float(score))
    for node in list(existing_nodes):
        if node not in combined_scores:
            combined_scores[node] = float(proposal_scores.get(node, semantic_scores.get(node, 0.0)))

    ranked_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    selected_boost, skip_diag = _select_diverse(
        g,
        ranked_combined,
        max_count=max_expanded,
        candidate_diversity_enabled=flags["candidate_diversity_enabled"],
        entity_diversity_enabled=flags["entity_diversity_enabled"],
        source_diversity_enabled=flags["source_diversity_enabled"],
        candidate_dedup_enabled=flags["candidate_dedup_enabled"],
    )
    selected_boost_set = {str(node) for node in list(selected_boost or []) if str(node) in g}

    local_cap_base = max(8, int(getattr(cfg, "proposal_anchor_local_topn", 48)))
    local_cap = max(local_cap_base, max(8, max_expanded + 1))

    updated_by_anchor: Dict[str, List[str]] = {}
    target_anchor_keys = anchors_in_graph if anchors_in_graph else [str(a) for a in proposal_by_anchor.keys()]
    if not target_anchor_keys:
        target_anchor_keys = ["__global__"]
        proposal_by_anchor["__global__"] = []

    for anchor in target_anchor_keys:
        existing = list(proposal_by_anchor.get(anchor, []) or [])
        ranking: List[Tuple[str, float, int]] = []
        for idx, node in enumerate(existing):
            node_id = str(node)
            base = float(proposal_scores.get(node_id, semantic_scores.get(node_id, 0.0)))
            ranking.append((node_id, base + 0.0005 * (1.0 / float(idx + 1)), 0))
        for node_id in selected_boost_set:
            base = float(proposal_scores.get(node_id, semantic_scores.get(node_id, 0.0)))
            ranking.append((node_id, base + 0.03, 1))
        ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)
        ordered_nodes = _ordered_unique([node for node, _, _ in ranking if node in g and _is_candidate_node(g, node)])
        if anchor in g and _is_candidate_node(g, anchor):
            merged = [str(anchor)] + [node for node in ordered_nodes if node != str(anchor)]
        else:
            merged = list(ordered_nodes)
        updated_by_anchor[str(anchor)] = merged[:local_cap]

    updated_nodes = {str(x) for x in existing_nodes}
    updated_nodes.update({str(x) for x in selected_boost_set})
    for vals in updated_by_anchor.values():
        updated_nodes.update([str(v) for v in list(vals or []) if str(v) in g])

    updated_proposal_scores = {str(k): float(v) for k, v in dict(proposal_scores or {}).items()}
    updated_semantic_scores = {str(k): float(v) for k, v in dict(semantic_scores or {}).items()}
    for node in updated_nodes:
        node_score = float(combined_scores.get(node, proposal_scores.get(node, semantic_scores.get(node, 0.0))))
        updated_proposal_scores[node] = max(float(updated_proposal_scores.get(node, -1.0e9)), float(node_score))
        updated_semantic_scores[node] = max(float(updated_semantic_scores.get(node, -1.0e9)), float(node_score))

    diag = {
        "enabled": True,
        "applied": bool(len(selected_boost_set) > 0),
        "flags": dict(flags),
        "max_expanded_candidates": int(max_expanded),
        "max_path_candidates": int(max_path),
        "max_bridge_candidates": int(max_bridge),
        "max_anchor_expansion_hops": int(max_hops),
        "path_candidate_count": int(len(path_scores)),
        "anchor_expansion_candidate_count": int(len(anchor_scores)),
        "selected_boost_count": int(len(selected_boost_set)),
        "union_candidate_count_before": int(len(existing_nodes)),
        "union_candidate_count_after": int(len(updated_nodes)),
        "dedup_skipped": int(skip_diag.get("dedup_skipped", 0)),
        "diversity_skipped": int(skip_diag.get("diversity_skipped", 0)),
        "selected_boost_nodes": [str(x) for x in selected_boost[: min(len(selected_boost), 64)]],
    }
    return updated_by_anchor, updated_nodes, updated_proposal_scores, updated_semantic_scores, diag
