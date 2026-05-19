from __future__ import annotations

import hashlib
import os
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


def _safe_shortest_path_with_reach(
    g,
    src: str,
    dst: str,
    reachable_within_cutoff: Mapping[str, int] | None,
) -> List[str]:
    if src not in g or dst not in g:
        return []
    if src == dst:
        return [src]
    if reachable_within_cutoff is not None and dst not in reachable_within_cutoff:
        return []
    try:
        return [str(x) for x in nx.shortest_path(g, src, dst)]
    except Exception:
        return []


def _score_attachment_opt_enabled() -> bool:
    raw = str(os.environ.get("PHASE6T_SCORE_ATTACHMENT_OPT", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _score_attachment_stable_order_enabled() -> bool:
    raw = str(os.environ.get("PHASE6T_SCORE_ATTACHMENT_STABLE_ORDER", "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _score_attachment_stable_eps() -> float:
    raw = str(os.environ.get("PHASE6T_SCORE_ATTACHMENT_STABLE_EPS", "1e-12") or "").strip()
    try:
        val = float(raw)
    except Exception:
        return 1.0e-12
    if val <= 0.0:
        return 1.0e-12
    return float(val)


def _chunk_id_from_sentence_id(sentence_id: str) -> str:
    sid = str(sentence_id or "").strip()
    if not sid:
        return ""
    if sid.startswith("chunk::") and "::" in sid:
        left, right = sid.rsplit("::", 1)
        if right.isdigit():
            return left
    if sid.startswith("c::"):
        head, sep, tail = sid.rpartition(":")
        if sep and tail.isdigit():
            return head
    return ""


def _normalize_candidate_key(candidate_id: str) -> str:
    text = str(candidate_id or "").strip()
    if not text:
        return ""
    return text.lower()


def _node_source_title(g, node: str) -> str:
    if node not in g:
        return ""
    data = g.nodes[node] or {}
    for key in ("title", "doc_title", "source", "source_id", "doc_id", "document_id"):
        value = str(data.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _node_sentence_id(g, node: str) -> str:
    if node not in g:
        return ""
    ntype = _node_type(g, node)
    if ntype == "sentence":
        return str(node)
    return ""


def _node_chunk_id(g, node: str) -> str:
    node_id = str(node or "")
    if not node_id:
        return ""
    if node_id.startswith("chunk::") or node_id.startswith("c::"):
        return node_id
    if node not in g:
        return _chunk_id_from_sentence_id(node_id)
    ntype = _node_type(g, node)
    if ntype in {"chunk", "passage", "document"}:
        return node_id
    return _chunk_id_from_sentence_id(node_id)


def _node_text_hash(g, node: str) -> str:
    sig = _node_text_signature(g, node)
    if not sig:
        return ""
    try:
        return hashlib.sha1(sig.encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return ""


def _stable_tiebreak_key(g, node: str) -> Tuple[str, str, str, str]:
    node_id = str(node or "")
    source_key = _node_source_key(g, node_id) or ""
    chunk_id = _chunk_id_from_sentence_id(node_id)
    text_sig = _node_text_signature(g, node_id)
    text_hash = hashlib.sha1(text_sig.encode("utf-8", errors="ignore")).hexdigest() if text_sig else ""
    return (source_key, chunk_id, node_id, text_hash)


def _stable_score_key(g, node: str, score: float, eps: float) -> Tuple[int, str, str, str, str]:
    bucket = int(round(float(score) / float(eps)))
    source_key, chunk_id, node_id, text_hash = _stable_tiebreak_key(g, node)
    return (-bucket, source_key, chunk_id, node_id, text_hash)


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
    use_optimized = _score_attachment_opt_enabled()
    for anchor in list(anchors or []):
        if anchor not in g:
            continue
        anchor_reach = None
        if use_optimized:
            cutoff = max(1, int(max_hops))
            try:
                # Keep baseline shortest-path semantics (nx.shortest_path for each target)
                # but avoid repeatedly recomputing source reachability for cutoff filtering.
                anchor_reach = nx.single_source_shortest_path_length(g, anchor, cutoff=cutoff)
            except Exception:
                anchor_reach = {}
        for target in list(ranked_targets or []):
            if len(out) >= cap:
                return out
            if target == anchor or target not in g:
                continue
            if use_optimized:
                path = _safe_shortest_path_with_reach(g, anchor, target, anchor_reach)
            else:
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
    score_attachment_trace_fn=None,
) -> Tuple[Dict[str, List[str]], set[str], Dict[str, float], Dict[str, float], Dict[str, Any]]:
    def _score_trace(step: str, extra: Mapping[str, Any] | None = None, reset: bool = False) -> None:
        if not callable(score_attachment_trace_fn):
            return
        payload = dict(extra or {})
        try:
            score_attachment_trace_fn(step=str(step), extra=payload, reset=bool(reset))
        except TypeError:
            try:
                score_attachment_trace_fn(str(step), payload, bool(reset))
            except Exception:
                return
        except Exception:
            return

    flags = {
        "path_candidate_expansion_enabled": bool(getattr(cfg, "path_candidate_expansion_enabled", False)),
        "anchor_expansion_enabled": bool(getattr(cfg, "anchor_expansion_enabled", False)),
        "candidate_diversity_enabled": bool(getattr(cfg, "candidate_diversity_enabled", False)),
        "entity_diversity_enabled": bool(getattr(cfg, "entity_diversity_enabled", False)),
        "source_diversity_enabled": bool(getattr(cfg, "source_diversity_enabled", False)),
        "candidate_dedup_enabled": bool(getattr(cfg, "candidate_dedup_enabled", True)),
    }
    stable_order_enabled = _score_attachment_stable_order_enabled()
    stable_eps = _score_attachment_stable_eps()

    def _build_candidate_score_rows(
        *,
        candidate_nodes: Sequence[str],
        proposal_map: Mapping[str, float],
        semantic_map: Mapping[str, float],
        bridge_map: Mapping[str, float],
        corridor_map: Mapping[str, float],
        selected_boost_nodes: set[str],
        pre_score_nodes: set[str],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for node in list(candidate_nodes or []):
            node_id = str(node or "").strip()
            if not node_id:
                continue
            semantic_val = semantic_map.get(node_id, None)
            graph_val = proposal_map.get(node_id, None)
            bridge_val = bridge_map.get(node_id, None)
            corridor_val = corridor_map.get(node_id, None)

            semantic_missing = semantic_val is None
            graph_missing = graph_val is None
            bridge_missing = bridge_val is None
            corridor_missing = corridor_val is None

            final_val = graph_val if graph_val is not None else semantic_val
            if final_val is None:
                final_val = 0.0

            if node_id in bridge_map:
                source_stage = "bridge"
            elif node_id in corridor_map:
                source_stage = "corridor"
            elif node_id in proposal_map:
                source_stage = "graph"
            elif node_id in semantic_map:
                source_stage = "semantic"
            else:
                source_stage = "fallback"

            row = {
                "candidate_id": node_id,
                "candidate_key_raw": node_id,
                "candidate_key_normalized": _normalize_candidate_key(node_id),
                "candidate_type": (_node_type(g, node_id) or "unknown"),
                "source_title": _node_source_title(g, node_id),
                "chunk_id": _node_chunk_id(g, node_id),
                "sentence_id": _node_sentence_id(g, node_id),
                "text_hash": _node_text_hash(g, node_id),
                "source_stage": str(source_stage),
                "included_before_score_attachment": bool(node_id in pre_score_nodes),
                "included_after_score_attachment": True,
                "final_score": float(final_val),
                "semantic_score": (None if semantic_missing else float(semantic_val)),
                "graph_score": (None if graph_missing else float(graph_val)),
                "bridge_score": (None if bridge_missing else float(bridge_val)),
                "corridor_score": (None if corridor_missing else float(corridor_val)),
                "redundancy_score": None,
                "answerability_score": None,
                "score_source": {
                    "semantic": ("semantic_scores" if not semantic_missing else "none"),
                    "graph": ("proposal_scores" if not graph_missing else "none"),
                    "bridge": ("path_scores" if not bridge_missing else "none"),
                    "corridor": ("anchor_scores" if not corridor_missing else "none"),
                    "redundancy": "none",
                },
                "missing_flags": {
                    "semantic_missing": bool(semantic_missing),
                    "graph_missing": bool(graph_missing),
                    "bridge_missing": bool(bridge_missing),
                    "corridor_missing": bool(corridor_missing),
                    "redundancy_missing": True,
                },
                "selected_boost": bool(node_id in selected_boost_nodes),
                "score_merge_priority": [
                    "proposal_scores",
                    "semantic_scores",
                    "path_scores",
                    "anchor_scores",
                    "default_zero",
                ],
            }
            rows.append(row)

        if stable_order_enabled:
            rows.sort(
                key=lambda r: _stable_score_key(
                    g,
                    str(r.get("candidate_id", "")),
                    float(r.get("final_score", 0.0)),
                    stable_eps,
                )
            )
        else:
            rows.sort(
                key=lambda r: (
                    float(r.get("final_score", 0.0)),
                    str(r.get("candidate_id", "")),
                ),
                reverse=True,
            )
        for idx, row in enumerate(rows):
            row["rank_before_render"] = int(idx + 1)
        return rows

    enabled = any(
        [
            flags["path_candidate_expansion_enabled"],
            flags["anchor_expansion_enabled"],
            flags["candidate_diversity_enabled"],
            flags["entity_diversity_enabled"],
            flags["source_diversity_enabled"],
        ]
    )
    _score_trace("candidate_input_inspect_start", extra={})
    if not enabled:
        existing_nodes_short = [str(x) for x in _ordered_unique(proposal_nodes) if str(x)]
        _score_trace(
            "candidate_input_inspect_done",
            extra={
                "num_candidates": int(len(existing_nodes_short)),
                "candidate_type": str(type(proposal_nodes).__name__),
                "num_unique_candidate_ids": int(len(set(existing_nodes_short))),
            },
        )
        _score_trace("score_map_prepare_start", extra={})
        _score_trace(
            "score_map_prepare_done",
            extra={"num_score_maps": 2, "score_map_keys": ["proposal_scores", "semantic_scores"]},
        )
        _score_trace("semantic_score_index_build_start", extra={})
        _score_trace(
            "semantic_score_index_build_done",
            extra={
                "num_semantic_scores": int(len(semantic_scores or {})),
                "semantic_index_size": int(len(semantic_scores or {})),
                "semantic_index_rebuilt": False,
            },
        )
        _score_trace("graph_score_index_build_start", extra={})
        _score_trace(
            "graph_score_index_build_done",
            extra={
                "num_graph_scores": int(len(proposal_scores or {})),
                "graph_index_size": int(len(proposal_scores or {})),
                "graph_index_rebuilt": False,
            },
        )
        _score_trace("bridge_score_index_build_start", extra={})
        _score_trace(
            "bridge_score_index_build_done",
            extra={"num_bridge_scores": 0, "bridge_index_size": 0, "bridge_index_rebuilt": False},
        )
        _score_trace("corridor_score_index_build_start", extra={})
        _score_trace(
            "corridor_score_index_build_done",
            extra={"num_corridor_scores": 0, "corridor_index_size": 0, "corridor_index_rebuilt": False},
        )
        _score_trace("redundancy_feature_prepare_start", extra={})
        _score_trace("redundancy_feature_prepare_done", extra={})
        _score_trace("normalization_prepare_start", extra={})
        _score_trace("normalization_prepare_done", extra={})
        _score_trace("candidate_loop_start", extra={})
        _score_trace("candidate_semantic_attach_start", extra={})
        _score_trace("candidate_semantic_attach_done", extra={})
        _score_trace("candidate_graph_attach_start", extra={})
        _score_trace("candidate_graph_attach_done", extra={})
        _score_trace("candidate_bridge_attach_start", extra={})
        _score_trace("candidate_bridge_attach_done", extra={})
        _score_trace("candidate_corridor_attach_start", extra={})
        _score_trace("candidate_corridor_attach_done", extra={})
        _score_trace("candidate_redundancy_attach_start", extra={})
        _score_trace("candidate_redundancy_attach_done", extra={})
        _score_trace("candidate_object_update_start", extra={})
        _score_trace("candidate_object_update_done", extra={})
        _score_trace(
            "candidate_loop_done",
            extra={
                "num_candidates_processed": int(len(existing_nodes_short)),
                "num_semantic_lookups": 0,
                "num_graph_lookups": 0,
                "num_bridge_lookups": 0,
                "num_corridor_lookups": 0,
                "num_missing_scores": 0,
                "num_full_map_scans": 0,
            },
        )
        _score_trace(
            "score_attachment_done",
            extra={
                "num_candidates": int(len(existing_nodes_short)),
                "selected_boost_count": 0,
                "num_full_map_scans": 0,
            },
        )
        base_rows = _build_candidate_score_rows(
            candidate_nodes=list(existing_nodes_short),
            proposal_map={str(k): float(v) for k, v in dict(proposal_scores or {}).items()},
            semantic_map={str(k): float(v) for k, v in dict(semantic_scores or {}).items()},
            bridge_map={},
            corridor_map={},
            selected_boost_nodes=set(),
            pre_score_nodes=set(existing_nodes_short),
        )
        return (
            dict(proposal_by_anchor),
            {str(x) for x in list(proposal_nodes or []) if str(x)},
            {str(k): float(v) for k, v in dict(proposal_scores or {}).items()},
            {str(k): float(v) for k, v in dict(semantic_scores or {}).items()},
            {
                "enabled": False,
                "applied": False,
                "reason": "disabled",
                "score_attachment_opt_enabled": bool(_score_attachment_opt_enabled()),
                "score_attachment_stable_order_enabled": bool(stable_order_enabled),
                "score_attachment_stable_eps": float(stable_eps),
                "candidate_score_table": list(base_rows),
                **flags,
            },
        )

    raw_candidate_nodes = [str(n) for n in list(proposal_nodes or []) if str(n)]
    _score_trace(
        "candidate_input_inspect_done",
        extra={
            "num_candidates": int(len(raw_candidate_nodes)),
            "candidate_type": str(type(proposal_nodes).__name__),
            "num_unique_candidate_ids": int(len(set(raw_candidate_nodes))),
        },
    )

    _score_trace("score_map_prepare_start", extra={})
    score_maps = {
        "proposal_scores": {str(k): float(v) for k, v in dict(proposal_scores or {}).items()},
        "semantic_scores": {str(k): float(v) for k, v in dict(semantic_scores or {}).items()},
    }
    _score_trace(
        "score_map_prepare_done",
        extra={"num_score_maps": 2, "score_map_keys": sorted(list(score_maps.keys()))},
    )
    _score_trace("semantic_score_index_build_start", extra={})
    semantic_score_index = dict(score_maps["semantic_scores"])
    _score_trace(
        "semantic_score_index_build_done",
        extra={
            "num_semantic_scores": int(len(semantic_score_index)),
            "semantic_index_size": int(len(semantic_score_index)),
            "semantic_index_rebuilt": True,
        },
    )
    _score_trace("graph_score_index_build_start", extra={})
    graph_score_index = dict(score_maps["proposal_scores"])
    _score_trace(
        "graph_score_index_build_done",
        extra={
            "num_graph_scores": int(len(graph_score_index)),
            "graph_index_size": int(len(graph_score_index)),
            "graph_index_rebuilt": True,
        },
    )
    anchors_in_graph = [str(a) for a in _ordered_unique(anchors) if str(a) in g]
    if stable_order_enabled:
        anchors_in_graph = sorted(anchors_in_graph, key=lambda n: _stable_tiebreak_key(g, n))
    existing_nodes = [str(n) for n in _ordered_unique(proposal_nodes) if str(n) in g]
    if stable_order_enabled:
        existing_nodes = sorted(existing_nodes, key=lambda n: _stable_tiebreak_key(g, n))
        ranked_existing = sorted(
            existing_nodes,
            key=lambda n: _stable_score_key(g, n, float(graph_score_index.get(n, semantic_score_index.get(n, 0.0))), stable_eps),
        )
    else:
        ranked_existing = sorted(
            existing_nodes,
            key=lambda n: float(graph_score_index.get(n, semantic_score_index.get(n, 0.0))),
            reverse=True,
        )
    _score_trace(
        "bridge_score_prepare_start",
        extra={
            "num_anchors_in_graph": int(len(anchors_in_graph)),
            "num_existing_nodes": int(len(existing_nodes)),
            "num_ranked_existing": int(len(ranked_existing)),
        },
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
        _score_trace(
            "path_candidate_expansion_start",
            extra={
                "target_limit": int(target_limit),
                "max_path_candidates": int(min(max_path, max_bridge if max_bridge > 0 else max_path)),
                "max_hops": int(tau_hops),
            },
        )
        path_scores = _collect_path_candidates(
            g,
            anchors_in_graph,
            target_nodes,
            proposal_scores,
            semantic_scores,
            max_path_candidates=min(max_path, max_bridge if max_bridge > 0 else max_path),
            max_hops=tau_hops,
        )
        _score_trace(
            "path_candidate_expansion_done",
            extra={"path_candidate_count": int(len(path_scores))},
        )
    else:
        _score_trace("path_candidate_expansion_start", extra={"target_limit": 0, "max_path_candidates": 0, "max_hops": int(tau_hops)})
        _score_trace("path_candidate_expansion_done", extra={"path_candidate_count": 0})
    if flags["anchor_expansion_enabled"] and max_hops > 0:
        _score_trace(
            "anchor_candidate_expansion_start",
            extra={"max_hops": int(max_hops)},
        )
        anchor_scores = _collect_anchor_expansion_candidates(
            g,
            anchors_in_graph,
            semantic_scores,
            max_hops=max_hops,
        )
        _score_trace(
            "anchor_candidate_expansion_done",
            extra={"anchor_expansion_candidate_count": int(len(anchor_scores))},
        )
    else:
        _score_trace("anchor_candidate_expansion_start", extra={"max_hops": int(max_hops)})
        _score_trace("anchor_candidate_expansion_done", extra={"anchor_expansion_candidate_count": 0})
    _score_trace(
        "bridge_score_prepare_done",
        extra={
            "path_candidate_count": int(len(path_scores)),
            "anchor_expansion_candidate_count": int(len(anchor_scores)),
        },
    )

    _score_trace("bridge_score_index_build_start", extra={})
    bridge_score_index = dict(path_scores or {})
    _score_trace(
        "bridge_score_index_build_done",
        extra={
            "num_bridge_scores": int(len(bridge_score_index)),
            "bridge_index_size": int(len(bridge_score_index)),
            "bridge_index_rebuilt": True,
        },
    )
    _score_trace("corridor_score_index_build_start", extra={})
    corridor_score_index = dict(anchor_scores or {})
    _score_trace(
        "corridor_score_index_build_done",
        extra={
            "num_corridor_scores": int(len(corridor_score_index)),
            "corridor_index_size": int(len(corridor_score_index)),
            "corridor_index_rebuilt": True,
        },
    )

    combined_scores: Dict[str, float] = {}
    _score_trace("normalization_prepare_start", extra={})
    for node, score in list(path_scores.items()):
        combined_scores[str(node)] = max(float(combined_scores.get(str(node), -1.0e9)), float(score))
    for node, score in list(anchor_scores.items()):
        combined_scores[str(node)] = max(float(combined_scores.get(str(node), -1.0e9)), float(score))
    for node in list(existing_nodes):
        if node not in combined_scores:
            combined_scores[node] = float(graph_score_index.get(node, semantic_score_index.get(node, 0.0)))
    _score_trace("normalization_prepare_done", extra={})

    if stable_order_enabled:
        ranked_combined = sorted(
            combined_scores.items(),
            key=lambda x: _stable_score_key(g, str(x[0]), float(x[1]), stable_eps),
        )
    else:
        ranked_combined = sorted(
            combined_scores.items(),
            key=lambda x: (float(x[1]), str(x[0])),
            reverse=True,
        )
    _score_trace("redundancy_feature_prepare_start", extra={})
    selected_boost, skip_diag = _select_diverse(
        g,
        ranked_combined,
        max_count=max_expanded,
        candidate_diversity_enabled=flags["candidate_diversity_enabled"],
        entity_diversity_enabled=flags["entity_diversity_enabled"],
        source_diversity_enabled=flags["source_diversity_enabled"],
        candidate_dedup_enabled=flags["candidate_dedup_enabled"],
    )
    _score_trace(
        "redundancy_feature_prepare_done",
        extra={
            "dedup_skipped": int(skip_diag.get("dedup_skipped", 0)),
            "diversity_skipped": int(skip_diag.get("diversity_skipped", 0)),
        },
    )
    selected_boost_set = {str(node) for node in list(selected_boost or []) if str(node) in g}

    local_cap_base = max(8, int(getattr(cfg, "proposal_anchor_local_topn", 48)))
    local_cap = max(local_cap_base, max(8, max_expanded + 1))

    updated_by_anchor: Dict[str, List[str]] = {}
    target_anchor_keys = anchors_in_graph if anchors_in_graph else [str(a) for a in proposal_by_anchor.keys()]
    if stable_order_enabled:
        target_anchor_keys = sorted(list(target_anchor_keys or []), key=lambda n: _stable_tiebreak_key(g, n))
    if not target_anchor_keys:
        target_anchor_keys = ["__global__"]
        proposal_by_anchor["__global__"] = []

    num_candidates_processed = 0
    num_semantic_lookups = 0
    num_graph_lookups = 0
    num_bridge_lookups = 0
    num_corridor_lookups = 0
    num_missing_scores = 0
    num_full_map_scans = 0

    _score_trace("candidate_loop_start", extra={"num_candidate_loops": int(len(target_anchor_keys))})
    _score_trace("candidate_semantic_attach_start", extra={})
    ranking_by_anchor: Dict[str, List[Tuple[str, float, int]]] = {}
    for anchor in target_anchor_keys:
        existing = list(proposal_by_anchor.get(anchor, []) or [])
        ranking: List[Tuple[str, float, int]] = []
        for idx, node in enumerate(existing):
            node_id = str(node)
            gval = graph_score_index.get(node_id, None)
            sval = semantic_score_index.get(node_id, None)
            num_graph_lookups += 1
            num_semantic_lookups += 1
            if gval is None and sval is None:
                num_missing_scores += 1
                base = 0.0
            else:
                base = float(gval if gval is not None else sval)
            ranking.append((node_id, base + 0.0005 * (1.0 / float(idx + 1)), 0))
            num_candidates_processed += 1
        ranking_by_anchor[str(anchor)] = ranking
    _score_trace("candidate_semantic_attach_done", extra={})

    _score_trace("candidate_graph_attach_start", extra={})
    for anchor in target_anchor_keys:
        ranking = ranking_by_anchor.get(str(anchor), [])
        if stable_order_enabled:
            boost_iter = sorted(list(selected_boost_set), key=lambda n: _stable_tiebreak_key(g, n))
        else:
            # Keep deterministic iteration order shared across baseline/optimized paths.
            boost_iter = list(selected_boost)
        for node_id in boost_iter:
            gval = graph_score_index.get(node_id, None)
            sval = semantic_score_index.get(node_id, None)
            num_graph_lookups += 1
            num_semantic_lookups += 1
            if gval is None and sval is None:
                num_missing_scores += 1
                base = 0.0
            else:
                base = float(gval if gval is not None else sval)
            ranking.append((node_id, base + 0.03, 1))
            num_candidates_processed += 1
        ranking_by_anchor[str(anchor)] = ranking
    _score_trace("candidate_graph_attach_done", extra={})

    _score_trace("candidate_bridge_attach_start", extra={})
    _score_trace(
        "candidate_bridge_attach_done",
        extra={"num_bridge_scores": int(len(bridge_score_index)), "num_bridge_lookups": int(num_bridge_lookups)},
    )
    _score_trace("candidate_corridor_attach_start", extra={})
    _score_trace(
        "candidate_corridor_attach_done",
        extra={"num_corridor_scores": int(len(corridor_score_index)), "num_corridor_lookups": int(num_corridor_lookups)},
    )
    _score_trace("candidate_redundancy_attach_start", extra={})
    _score_trace("candidate_redundancy_attach_done", extra={"num_full_map_scans": int(num_full_map_scans)})
    _score_trace("candidate_object_update_start", extra={})
    updated_by_anchor = {}
    for anchor in target_anchor_keys:
        ranking = list(ranking_by_anchor.get(str(anchor), []))
        if stable_order_enabled:
            ranking.sort(
                key=lambda x: (
                    _stable_score_key(g, str(x[0]), float(x[1]), stable_eps),
                    -int(x[2]),
                ),
            )
        else:
            ranking.sort(key=lambda x: (float(x[1]), int(x[2]), str(x[0])), reverse=True)
        ordered_nodes = _ordered_unique([node for node, _, _ in ranking if node in g and _is_candidate_node(g, node)])
        if anchor in g and _is_candidate_node(g, anchor):
            merged = [str(anchor)] + [node for node in ordered_nodes if node != str(anchor)]
        else:
            merged = list(ordered_nodes)
        updated_by_anchor[str(anchor)] = merged[:local_cap]
    _score_trace("candidate_object_update_done", extra={})

    updated_nodes = {str(x) for x in existing_nodes}
    updated_nodes.update({str(x) for x in selected_boost_set})
    for vals in updated_by_anchor.values():
        updated_nodes.update([str(v) for v in list(vals or []) if str(v) in g])

    updated_proposal_scores = {str(k): float(v) for k, v in dict(proposal_scores or {}).items()}
    updated_semantic_scores = {str(k): float(v) for k, v in dict(semantic_scores or {}).items()}
    for node in updated_nodes:
        num_graph_lookups += 1
        num_semantic_lookups += 1
        gval = graph_score_index.get(node, None)
        sval = semantic_score_index.get(node, None)
        if gval is None and sval is None:
            num_missing_scores += 1
        node_score = float(combined_scores.get(node, gval if gval is not None else (sval if sval is not None else 0.0)))
        updated_proposal_scores[node] = max(float(updated_proposal_scores.get(node, -1.0e9)), float(node_score))
        updated_semantic_scores[node] = max(float(updated_semantic_scores.get(node, -1.0e9)), float(node_score))
    _score_trace(
        "candidate_loop_done",
        extra={
            "num_candidates_processed": int(num_candidates_processed),
            "num_semantic_lookups": int(num_semantic_lookups),
            "num_graph_lookups": int(num_graph_lookups),
            "num_bridge_lookups": int(num_bridge_lookups),
            "num_corridor_lookups": int(num_corridor_lookups),
            "num_missing_scores": int(num_missing_scores),
            "num_full_map_scans": int(num_full_map_scans),
        },
    )

    diag = {
        "enabled": True,
        "applied": bool(len(selected_boost_set) > 0),
        "flags": dict(flags),
        "score_attachment_opt_enabled": bool(_score_attachment_opt_enabled()),
        "score_attachment_stable_order_enabled": bool(stable_order_enabled),
        "score_attachment_stable_eps": float(stable_eps),
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
    diag["candidate_score_table"] = _build_candidate_score_rows(
        candidate_nodes=sorted(list(updated_nodes)),
        proposal_map=updated_proposal_scores,
        semantic_map=updated_semantic_scores,
        bridge_map=bridge_score_index,
        corridor_map=corridor_score_index,
        selected_boost_nodes=selected_boost_set,
        pre_score_nodes={str(x) for x in list(raw_candidate_nodes or []) if str(x)},
    )
    _score_trace(
        "score_attachment_done",
        extra={
            "num_candidates": int(len(updated_nodes)),
            "selected_boost_count": int(len(selected_boost_set)),
            "num_full_map_scans": int(num_full_map_scans),
        },
    )
    return updated_by_anchor, updated_nodes, updated_proposal_scores, updated_semantic_scores, diag
