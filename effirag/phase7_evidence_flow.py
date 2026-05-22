from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .anchors import select_lexical_anchors
from .embedding import encode_texts, topk_cosine_similarity
from .global_index import load_or_build_global_index, load_semantic_index
from .metrics import supporting_fact_match_details
from .phase7_config import Phase7Config, phase7_config_from_cfg
from .phase7_logging import get_phase7_logger
from .phase7_source_balanced import SOURCE_ORDER, disabled_source_balance_diagnostics, source_balanced_union
from .phase8_chunk_medoid import ChunkMedoidProposalResult, run_phase8_chunk_medoid_proposal
from .phase8_pamae_entity_seeding import PamaeProposalResult, run_phase8_pamae_proposal
from .registry import register_method
from .types import RetrievalResult
from .utils import content_tokens


@dataclass
class Phase7EvidenceAtom:
    atom_id: str
    source_id: str
    title: str
    text: str
    token_count: int

    semantic_score: float
    flow_score: float

    anchor_distance: float
    anchor_reachability: float
    bridge_entity_coverage: float
    relation_term_coverage: float

    source_key: str
    normalized_token_set: Set[str] = field(default_factory=set)
    entity_set: Set[str] = field(default_factory=set)
    source_order: int = 0

    query_entity_coverage: float = 0.0
    answerability_base: float = 0.0
    bridge_base: float = 0.0
    carrier_id: str = ""
    flow_rank: int = 0
    anchor_decay_score: float = 0.0
    corridor_score: float = 0.0
    corridor_anchor_coverage: float = 0.0
    corridor_seed_coverage: float = 0.0
    corridor_path_closure: float = 0.0
    corridor_degree_contrib: float = 0.0
    corridor_path_count: int = 0
    is_corridor_node: bool = False
    connected_anchor_ids: Set[str] = field(default_factory=set)
    connected_seed_ids: Set[str] = field(default_factory=set)
    corridor_path_ids: Set[str] = field(default_factory=set)
    source_tags: Set[str] = field(default_factory=set)
    proposal_source: str = ""


@dataclass
class Phase7ChainUnit:
    unit_id: str
    unit_type: str
    atoms: Tuple[Phase7EvidenceAtom, ...]
    source_ids: Tuple[str, ...]
    token_count: int
    base_score: float


_PHASE7_INDEX_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def _minmax_norm_map(values_by_key: Dict[str, float]) -> Dict[str, float]:
    if not values_by_key:
        return {}
    vals = [float(v) for v in values_by_key.values()]
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return {str(k): 0.0 for k in values_by_key}
    scale = float(hi - lo)
    return {str(k): float((float(v) - lo) / scale) for k, v in values_by_key.items()}


def _token_count(text: str) -> int:
    raw = [tok for tok in str(text or "").split() if tok.strip()]
    return int(len(raw))


def _entity_token(node_id: str) -> str:
    sid = str(node_id or "")
    if sid.startswith("e::"):
        return sid.split("::", 1)[1].strip().lower()
    return sid.strip().lower()


def _slot_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_anchor_nodes(anchors: Iterable[str], graph: nx.Graph) -> List[str]:
    out = []
    seen = set()
    for node in list(anchors or []):
        sid = str(node or "")
        if not sid or sid in seen or sid not in graph:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _build_index_state(cfg: Any, p7: Phase7Config) -> Dict[str, Any]:
    corpus_path = str(getattr(cfg, "global_corpus_path", "") or "").strip()
    cache_dir = str(getattr(cfg, "graph_cache_dir", "outputs/index_cache") or "outputs/index_cache").strip()
    prebuilt_igraph_path = str(getattr(cfg, "prebuilt_igraph_path", "") or "").strip()
    key = (
        corpus_path,
        cache_dir,
        prebuilt_igraph_path,
        str(getattr(cfg, "graph_mode", "entity_chunk_graph") or "entity_chunk_graph"),
        str(getattr(cfg, "index_chunk_unit", "passage") or "passage"),
        int(p7.carrier_chunk_size_sentences),
        int(p7.carrier_chunk_stride_sentences),
        str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2"),
        int(getattr(cfg, "embedding_batch_size", 16) or 16),
        int(getattr(cfg, "embedding_max_length", 256) or 256),
        int(getattr(cfg, "embedding_text_max_chars", 800) or 800),
        str(getattr(cfg, "openie_mode", "llm") or "llm"),
        bool(getattr(cfg, "phase7_build_sentence_layer", p7.build_sentence_layer)),
        bool(getattr(cfg, "phase7_require_sentence_layer", p7.require_sentence_layer)),
    )
    cached = _PHASE7_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    graph, meta = load_or_build_global_index(
        corpus_path=corpus_path,
        cache_dir=cache_dir,
        force_rebuild=bool(getattr(cfg, "force_rebuild_graph_index", False)),
        prebuilt_igraph_path=prebuilt_igraph_path,
        prebuilt_igraph_format=str(getattr(cfg, "prebuilt_igraph_format", "hipporag_pickle") or "hipporag_pickle"),
        prebuilt_entity_token_limit=int(getattr(cfg, "prebuilt_entity_token_limit", 6) or 6),
        embedding_enabled=True,
        embedding_model_name=str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2"),
        embedding_batch_size=int(getattr(cfg, "embedding_batch_size", 16) or 16),
        embedding_max_length=int(getattr(cfg, "embedding_max_length", 256) or 256),
        embedding_text_max_chars=int(getattr(cfg, "embedding_text_max_chars", 800) or 800),
        graph_mode=str(getattr(cfg, "graph_mode", "entity_chunk_graph") or "entity_chunk_graph"),
        index_chunk_unit=str(getattr(cfg, "index_chunk_unit", "passage") or "passage"),
        openie_mode=str(getattr(cfg, "openie_mode", "llm") or "llm"),
        openie_model_name=str(getattr(cfg, "openie_model_name", "Qwen/Qwen2.5-7B-Instruct") or "Qwen/Qwen2.5-7B-Instruct"),
        openie_text_max_chars=int(getattr(cfg, "openie_text_max_chars", 2200) or 2200),
        openie_max_new_tokens=int(getattr(cfg, "openie_max_new_tokens", 256) or 256),
        openie_local_files_only=bool(getattr(cfg, "openie_local_files_only", True)),
        openie_api_base_url=str(getattr(cfg, "openie_api_base_url", "") or ""),
        openie_api_key=str(getattr(cfg, "openie_api_key", "") or ""),
        openie_api_timeout_sec=float(getattr(cfg, "openie_api_timeout_sec", 120.0) or 120.0),
        openie_parallel_workers=int(getattr(cfg, "openie_parallel_workers", 6) or 6),
        openie_log_every=int(getattr(cfg, "openie_log_every", 200) or 200),
        openie_retry_attempts=int(getattr(cfg, "openie_retry_attempts", 3) or 3),
        openie_retry_backoff_sec=float(getattr(cfg, "openie_retry_backoff_sec", 0.2) or 0.2),
        openie_error_sample_limit=int(getattr(cfg, "openie_error_sample_limit", 20) or 20),
        passage_chunking_strategy="sentence_window",
        passage_chunk_size_sentences=int(p7.carrier_chunk_size_sentences),
        passage_chunk_stride_sentences=int(p7.carrier_chunk_stride_sentences),
        passage_chunk_min_sentences=max(1, min(3, int(p7.carrier_chunk_size_sentences))),
        passage_chunk_max_chars=int(getattr(cfg, "passage_chunk_max_chars", 900) or 900),
        force_sentence_layer=bool(
            getattr(cfg, "phase7_build_sentence_layer", p7.build_sentence_layer)
            or (p7.enabled and p7.require_sentence_layer and p7.atom_unit == "sentence")
        ),
    )

    semantic_meta = dict((meta or {}).get("semantic_index", {}) or {})
    semantic_state = load_semantic_index(semantic_meta, mmap_mode="r")

    sentence_nodes = []
    chunk_nodes = []
    sentence_id_to_node = {}
    for node in graph.nodes:
        ntype = str(graph.nodes[node].get("node_type", "") or "").strip().lower()
        if ntype == "sentence":
            sentence_nodes.append(str(node))
            sid = str(graph.nodes[node].get("sentence_id", "") or "").strip()
            if sid:
                sentence_id_to_node[sid] = str(node)
        elif ntype in {"chunk", "passage", "document"}:
            chunk_nodes.append(str(node))

    state = {
        "graph": graph,
        "meta": dict(meta or {}),
        "semantic_state": semantic_state if isinstance(semantic_state, dict) else {},
        "sentence_nodes": sentence_nodes,
        "chunk_nodes": chunk_nodes,
        "sentence_id_to_node": sentence_id_to_node,
    }
    _PHASE7_INDEX_CACHE[key] = state
    return state


def _question_embedding(question: str, cfg: Any) -> Optional[np.ndarray]:
    encoded = encode_texts(
        texts=[str(question or "")],
        model_name=str(getattr(cfg, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2"),
        batch_size=1,
        max_length=int(getattr(cfg, "query_embedding_max_length", 96) or 96),
        max_chars=int(getattr(cfg, "query_embedding_text_max_chars", 384) or 384),
        instruction="",
    )
    if not bool(encoded.get("ok", False)):
        return None
    vectors = list(encoded.get("vectors", []) or [])
    if not vectors:
        return None
    vec = np.asarray(vectors[0], dtype=np.float32)
    if vec.ndim != 1 or vec.size <= 0:
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return None
    return vec / norm


def _entity_neighbors(graph: nx.Graph, node: str) -> Set[str]:
    out = set()
    if node not in graph:
        return out
    for nbr in graph.neighbors(node):
        ntype = str(graph.nodes[nbr].get("node_type", "") or "").strip().lower()
        if ntype == "entity":
            out.add(_entity_token(str(nbr)))
    return out


def _sentence_node_to_id(graph: nx.Graph, node: str) -> str:
    sid = str(graph.nodes[node].get("sentence_id", "") or "").strip()
    if sid:
        return sid
    title = str(graph.nodes[node].get("title", "") or "").strip()
    idx = _safe_int(graph.nodes[node].get("sent_idx", 0), 0)
    if title:
        return f"{title}::{idx}"
    return str(node)


def _sentence_node_text(graph: nx.Graph, node: str) -> str:
    return str(graph.nodes[node].get("text", "") or "")


def _sentence_node_title(graph: nx.Graph, node: str) -> str:
    return str(graph.nodes[node].get("title", "") or "")


def _sentence_node_order(graph: nx.Graph, node: str) -> int:
    return _safe_int(graph.nodes[node].get("sent_idx", 0), 0)


def _candidate_chunk_neighbors(graph: nx.Graph, sentence_node: str) -> List[str]:
    chunks = []
    for nbr in graph.neighbors(sentence_node):
        ntype = str(graph.nodes[nbr].get("node_type", "") or "").strip().lower()
        if ntype in {"chunk", "passage", "document"}:
            chunks.append(str(nbr))
    chunks.sort(key=lambda n: _safe_int(graph.nodes[n].get("chunk_idx", 0), 0))
    return chunks


def _collect_sentence_nodes_from_chunks(graph: nx.Graph, chunk_nodes: Iterable[str]) -> Set[str]:
    out = set()
    for cnode in list(chunk_nodes or []):
        if cnode not in graph:
            continue
        for nbr in graph.neighbors(cnode):
            if str(graph.nodes[nbr].get("node_type", "") or "").strip().lower() == "sentence":
                out.add(str(nbr))
    return out


def _semantic_top_chunks(
    question_vec: Optional[np.ndarray],
    state: Dict[str, Any],
    top_t: int,
) -> Tuple[List[str], Dict[str, float]]:
    semantic_state = dict(state.get("semantic_state", {}) or {})
    chunk_ids = list(semantic_state.get("chunk_ids", []) or [])
    chunk_embeddings = semantic_state.get("chunk_embeddings", None)
    if question_vec is None or chunk_embeddings is None or not chunk_ids:
        return [], {}
    idx, scores = topk_cosine_similarity(
        query_vector=question_vec,
        matrix=chunk_embeddings,
        topn=max(1, int(top_t)),
        scan_batch_size=int(semantic_state.get("meta", {}).get("semantic_scan_batch_size", 8192) or 8192),
    )
    out_ids = []
    out_scores = {}
    for i, score in zip(idx.tolist(), scores.tolist()):
        if i < 0 or i >= len(chunk_ids):
            continue
        node = str(chunk_ids[i] or "")
        if not node:
            continue
        out_ids.append(node)
        out_scores[node] = float(score)
    return out_ids, out_scores


def _semantic_top_entities(
    question_vec: Optional[np.ndarray],
    state: Dict[str, Any],
    topn: int = 24,
) -> Tuple[List[str], Dict[str, float]]:
    semantic_state = dict(state.get("semantic_state", {}) or {})
    entity_ids = list(semantic_state.get("entity_ids", []) or [])
    entity_embeddings = semantic_state.get("entity_embeddings", None)
    if question_vec is None or entity_embeddings is None or not entity_ids:
        return [], {}
    idx, scores = topk_cosine_similarity(
        query_vector=question_vec,
        matrix=entity_embeddings,
        topn=max(1, int(topn)),
        scan_batch_size=8192,
    )
    out_ids = []
    out_scores = {}
    for i, score in zip(idx.tolist(), scores.tolist()):
        if i < 0 or i >= len(entity_ids):
            continue
        node = str(entity_ids[i] or "")
        if not node:
            continue
        out_ids.append(node)
        out_scores[node] = float(score)
    return out_ids, out_scores


def _build_local_graph(
    graph: nx.Graph,
    anchor_nodes: List[str],
    semantic_chunk_nodes: List[str],
    semantic_entity_nodes: List[str],
) -> Tuple[nx.Graph, Dict[str, Any]]:
    node_set = set(anchor_nodes)
    node_set.update(semantic_chunk_nodes)
    node_set.update(semantic_entity_nodes)

    for node in list(anchor_nodes):
        if node not in graph:
            continue
        node_set.update(str(n) for n in graph.neighbors(node))

    for node in list(semantic_chunk_nodes):
        if node not in graph:
            continue
        nbrs = [str(n) for n in graph.neighbors(node)]
        node_set.update(nbrs)
        for nbr in nbrs:
            if nbr in graph:
                node_set.update(str(n2) for n2 in graph.neighbors(nbr))

    for node in list(semantic_entity_nodes):
        if node not in graph:
            continue
        node_set.update(str(n) for n in graph.neighbors(node))

    if not node_set:
        return graph.subgraph([]).copy(), {"local_nodes": 0, "local_edges": 0}

    local_graph = graph.subgraph(list(node_set)).copy()
    return local_graph, {
        "local_nodes": int(local_graph.number_of_nodes()),
        "local_edges": int(local_graph.number_of_edges()),
    }


def _flow_scores(local_graph: nx.Graph, seed_nodes: List[str], p7: Phase7Config) -> Dict[str, float]:
    if local_graph.number_of_nodes() == 0:
        return {}
    valid_seeds = [n for n in list(seed_nodes or []) if n in local_graph]
    personalization = None
    if valid_seeds:
        personalization = {n: 0.0 for n in local_graph.nodes}
        inv = 1.0 / float(len(valid_seeds))
        for n in valid_seeds:
            personalization[n] = inv
    try:
        pr = nx.pagerank(
            local_graph,
            alpha=float(p7.flow_alpha),
            personalization=personalization,
            max_iter=int(p7.max_flow_iterations),
            tol=float(p7.flow_tolerance),
        )
    except Exception:
        pr = {str(n): 0.0 for n in local_graph.nodes}
    return {str(k): float(v) for k, v in pr.items()}


def _candidate_atoms_from_local_graph(
    question: str,
    local_graph: nx.Graph,
    flow_scores: Dict[str, float],
    semantic_chunk_scores: Dict[str, float],
    anchor_nodes: List[str],
    candidate_top_m: int,
    source_balanced_enabled: bool = False,
    source_balanced_top_m: Optional[int] = None,
    source_quotas: Optional[Dict[str, int]] = None,
    source_balanced_fill_remaining: bool = True,
) -> Tuple[List[Phase7EvidenceAtom], Dict[str, Any]]:
    query_tokens = set(content_tokens(question))
    anchor_tokens = set(_entity_token(a) for a in anchor_nodes)
    relation_terms = set(t for t in query_tokens if t not in anchor_tokens)

    bridge_entities = set()
    for node in local_graph.nodes:
        if str(local_graph.nodes[node].get("node_type", "") or "").strip().lower() != "entity":
            continue
        sent_neighbors = 0
        for nbr in local_graph.neighbors(node):
            if str(local_graph.nodes[nbr].get("node_type", "") or "").strip().lower() == "sentence":
                sent_neighbors += 1
                if sent_neighbors >= 2:
                    break
        if sent_neighbors >= 2:
            bridge_entities.add(_entity_token(str(node)))

    sentence_nodes = [str(n) for n in local_graph.nodes if str(local_graph.nodes[n].get("node_type", "") or "").strip().lower() == "sentence"]
    anchor_entity_nodes = set(
        str(a)
        for a in list(anchor_nodes or [])
        if str(a) in local_graph and str(local_graph.nodes[str(a)].get("node_type", "") or "").strip().lower() == "entity"
    )
    try:
        anchor_distance_map = (
            nx.multi_source_shortest_path_length(local_graph, list(anchor_entity_nodes), cutoff=8)
            if anchor_entity_nodes
            else {}
        )
    except Exception:
        anchor_distance_map = {}

    scored_rows: List[Dict[str, Any]] = []
    for node in sentence_nodes:
        carrier_nodes = _candidate_chunk_neighbors(local_graph, node)
        semantic_score = 0.0
        carrier_id = ""
        for cnode in carrier_nodes:
            s = float(semantic_chunk_scores.get(cnode, 0.0) or 0.0)
            if s > semantic_score:
                semantic_score = s
                carrier_id = str(cnode)
        flow_score = float(flow_scores.get(node, 0.0) or 0.0)
        blend = 0.55 * flow_score + 0.45 * semantic_score
        sid = _sentence_node_to_id(local_graph, node)
        title = _sentence_node_title(local_graph, node)
        entity_set = _entity_neighbors(local_graph, node)
        neighbor_ids = set(str(n) for n in local_graph.neighbors(node))
        direct_anchor_entity = bool(anchor_entity_nodes.intersection(neighbor_ids))
        title_key = str(title or "").strip().lower()
        entity_title_hit = bool(direct_anchor_entity or entity_set.intersection(anchor_tokens) or title_key in anchor_tokens)
        anchor_distance = anchor_distance_map.get(node, None)
        anchor_neighborhood_score = 0.0 if anchor_distance is None else float(1.0 / (1.0 + float(anchor_distance)))
        source_tags = set()
        if semantic_score > 0.0:
            source_tags.add("semantic")
        if entity_title_hit:
            source_tags.add("entity_title")
        if flow_score > 0.0:
            source_tags.add("graph_flow")
        if anchor_distance is not None:
            source_tags.add("anchor_neighborhood")
        scored_rows.append(
            {
                "node": str(node),
                "source_id": str(sid),
                "title": str(title),
                "proposal_score": float(blend),
                "semantic_score": float(semantic_score),
                "flow_score": float(flow_score),
                "carrier_id": str(carrier_id),
                "anchor_neighborhood_score": float(anchor_neighborhood_score),
                "entity_title_score": 1.0 if entity_title_hit else 0.0,
                "source_tags": source_tags,
            }
        )

    source_timing_ms: Dict[str, float] = {}
    t_src = time.perf_counter()
    ranked_semantic = sorted(
        [row for row in scored_rows if "semantic" in set(row.get("source_tags", set()) or set())],
        key=lambda row: (
            float(row.get("semantic_score", 0.0) or 0.0),
            float(row.get("proposal_score", 0.0) or 0.0),
            float(row.get("flow_score", 0.0) or 0.0),
            str(row.get("node", "") or ""),
        ),
        reverse=True,
    )
    source_timing_ms["semantic_candidates_ms"] = float((time.perf_counter() - t_src) * 1000.0)
    t_src = time.perf_counter()
    ranked_entity_title = sorted(
        [row for row in scored_rows if "entity_title" in set(row.get("source_tags", set()) or set())],
        key=lambda row: (
            float(row.get("entity_title_score", 0.0) or 0.0),
            float(row.get("proposal_score", 0.0) or 0.0),
            float(row.get("flow_score", 0.0) or 0.0),
            str(row.get("node", "") or ""),
        ),
        reverse=True,
    )
    source_timing_ms["entity_title_candidates_ms"] = float((time.perf_counter() - t_src) * 1000.0)
    t_src = time.perf_counter()
    ranked_graph_flow = sorted(
        [row for row in scored_rows if "graph_flow" in set(row.get("source_tags", set()) or set())],
        key=lambda row: (
            float(row.get("flow_score", 0.0) or 0.0),
            float(row.get("proposal_score", 0.0) or 0.0),
            float(row.get("semantic_score", 0.0) or 0.0),
            str(row.get("node", "") or ""),
        ),
        reverse=True,
    )
    source_timing_ms["graph_flow_candidates_ms"] = float((time.perf_counter() - t_src) * 1000.0)
    t_src = time.perf_counter()
    ranked_anchor_neighborhood = sorted(
        [row for row in scored_rows if "anchor_neighborhood" in set(row.get("source_tags", set()) or set())],
        key=lambda row: (
            float(row.get("anchor_neighborhood_score", 0.0) or 0.0),
            float(row.get("proposal_score", 0.0) or 0.0),
            float(row.get("flow_score", 0.0) or 0.0),
            str(row.get("node", "") or ""),
        ),
        reverse=True,
    )
    source_timing_ms["anchor_neighborhood_candidates_ms"] = float((time.perf_counter() - t_src) * 1000.0)

    if bool(source_balanced_enabled):
        head, source_balance_diag = source_balanced_union(
            ranked_candidates={
                "semantic": ranked_semantic,
                "entity_title": ranked_entity_title,
                "graph_flow": ranked_graph_flow,
                "anchor_neighborhood": ranked_anchor_neighborhood,
            },
            quotas=dict(source_quotas or {}),
            top_m=max(1, int(source_balanced_top_m or candidate_top_m)),
            fill_remaining=bool(source_balanced_fill_remaining),
        )
    else:
        scored_rows.sort(
            key=lambda row: (
                float(row.get("proposal_score", 0.0) or 0.0),
                float(row.get("flow_score", 0.0) or 0.0),
                float(row.get("semantic_score", 0.0) or 0.0),
                str(row.get("node", "") or ""),
            ),
            reverse=True,
        )
        head = scored_rows[: max(1, int(candidate_top_m))]
        source_balance_diag = disabled_source_balance_diagnostics(top_m=max(1, int(candidate_top_m)))

    source_balance_diag["source_membership_counts"] = {
        src: int(sum(1 for row in head if src in set(row.get("source_tags", set()) or set())))
        for src in SOURCE_ORDER
    }
    source_balance_diag["num_selected_candidates"] = int(len(head))
    source_balance_diag["source_candidate_counts"] = {
        "semantic": int(len(ranked_semantic)),
        "entity_title": int(len(ranked_entity_title)),
        "graph_flow": int(len(ranked_graph_flow)),
        "anchor_neighborhood": int(len(ranked_anchor_neighborhood)),
    }
    source_balance_diag["source_timing_ms"] = dict(source_timing_ms)
    flow_rank = {str(row.get("node", "") or ""): int(rank + 1) for rank, row in enumerate(head)}
    selected_by_source = dict(source_balance_diag.get("selected_by_source", {}) or {})

    atoms = []
    for row in head:
        node = str(row.get("node", "") or "")
        semantic_score = float(row.get("semantic_score", 0.0) or 0.0)
        flow_score = float(row.get("flow_score", 0.0) or 0.0)
        carrier_id = str(row.get("carrier_id", "") or "")
        text = _sentence_node_text(local_graph, node)
        title = _sentence_node_title(local_graph, node)
        sid = _sentence_node_to_id(local_graph, node)
        token_set = set(content_tokens(text))
        entity_set = _entity_neighbors(local_graph, node)

        if anchor_tokens:
            query_entity_coverage = float(len(entity_set.intersection(anchor_tokens)) / float(len(anchor_tokens)))
        else:
            query_entity_coverage = float(len(token_set.intersection(query_tokens)) / float(max(1, len(query_tokens))))
        if bridge_entities:
            bridge_entity_coverage = float(len(entity_set.intersection(bridge_entities)) / float(len(bridge_entities)))
        else:
            bridge_entity_coverage = 0.0
        if relation_terms:
            relation_term_coverage = float(len(token_set.intersection(relation_terms)) / float(len(relation_terms)))
        else:
            relation_term_coverage = 0.0

        atoms.append(
            Phase7EvidenceAtom(
                atom_id=str(node),
                source_id=str(sid),
                title=str(title),
                text=str(text),
                token_count=_token_count(text),
                semantic_score=float(semantic_score),
                flow_score=float(flow_score),
                anchor_distance=1.0e9,
                anchor_reachability=0.0,
                bridge_entity_coverage=float(bridge_entity_coverage),
                relation_term_coverage=float(relation_term_coverage),
                source_key=str(title),
                normalized_token_set=token_set,
                entity_set=entity_set,
                source_order=_sentence_node_order(local_graph, node),
                query_entity_coverage=float(query_entity_coverage),
                answerability_base=0.0,
                bridge_base=0.0,
                carrier_id=str(carrier_id),
                flow_rank=int(flow_rank.get(node, 0)),
                source_tags=set(str(x) for x in set(row.get("source_tags", set()) or set()) if str(x)),
                proposal_source=str(selected_by_source.get(str(sid), "")),
            )
        )

    return atoms, {
        "query_tokens": query_tokens,
        "anchor_tokens": anchor_tokens,
        "relation_terms": relation_terms,
        "bridge_entities": bridge_entities,
        "source_balance": dict(source_balance_diag),
    }


def _candidate_atoms_from_phase8_evidence(
    question: str,
    local_graph: nx.Graph,
    evidence_candidate_ids: List[str],
    evidence_source_tags: Dict[str, List[str]],
    flow_scores: Dict[str, float],
    anchor_nodes: List[str],
    candidate_top_m: int,
    proposal_tag: str = "phase8_pamae_seed",
    proposal_source: str = "phase8_pamae_seed",
) -> Tuple[List[Phase7EvidenceAtom], Dict[str, Any]]:
    query_tokens = set(content_tokens(question))
    anchor_tokens = set(_entity_token(a) for a in anchor_nodes)
    relation_terms = set(t for t in query_tokens if t not in anchor_tokens)

    bridge_entities = set()
    for node in local_graph.nodes:
        if str(local_graph.nodes[node].get("node_type", "") or "").strip().lower() != "entity":
            continue
        sent_neighbors = 0
        for nbr in local_graph.neighbors(node):
            if str(local_graph.nodes[nbr].get("node_type", "") or "").strip().lower() == "sentence":
                sent_neighbors += 1
                if sent_neighbors >= 2:
                    break
        if sent_neighbors >= 2:
            bridge_entities.add(_entity_token(str(node)))

    seen = set()
    ordered_nodes: List[str] = []
    for node in list(evidence_candidate_ids or []):
        sid = str(node)
        if sid in seen or sid not in local_graph:
            continue
        if str(local_graph.nodes[sid].get("node_type", "") or "").strip().lower() != "sentence":
            continue
        seen.add(sid)
        ordered_nodes.append(sid)
        if len(ordered_nodes) >= max(1, int(candidate_top_m)):
            break

    atoms: List[Phase7EvidenceAtom] = []
    total = max(1, len(ordered_nodes))
    flow_rank = {str(node): int(rank + 1) for rank, node in enumerate(ordered_nodes)}
    for rank, node in enumerate(ordered_nodes):
        text = _sentence_node_text(local_graph, node)
        title = _sentence_node_title(local_graph, node)
        sid = _sentence_node_to_id(local_graph, node)
        token_set = set(content_tokens(text))
        entity_set = _entity_neighbors(local_graph, node)
        carrier_nodes = _candidate_chunk_neighbors(local_graph, node)
        carrier_id = str(carrier_nodes[0]) if carrier_nodes else ""
        rank_score = float(1.0 - (float(rank) / float(total)))
        flow_score = float(flow_scores.get(node, 0.0) or 0.0)

        if anchor_tokens:
            query_entity_coverage = float(len(entity_set.intersection(anchor_tokens)) / float(len(anchor_tokens)))
        else:
            query_entity_coverage = float(len(token_set.intersection(query_tokens)) / float(max(1, len(query_tokens))))
        if bridge_entities:
            bridge_entity_coverage = float(len(entity_set.intersection(bridge_entities)) / float(len(bridge_entities)))
        else:
            bridge_entity_coverage = 0.0
        if relation_terms:
            relation_term_coverage = float(len(token_set.intersection(relation_terms)) / float(len(relation_terms)))
        else:
            relation_term_coverage = 0.0

        tags = set(str(x) for x in list(evidence_source_tags.get(str(node), []) or []) if str(x))
        tags.add(str(proposal_tag))
        atoms.append(
            Phase7EvidenceAtom(
                atom_id=str(node),
                source_id=str(sid),
                title=str(title),
                text=str(text),
                token_count=_token_count(text),
                semantic_score=float(max(rank_score, flow_score)),
                flow_score=float(flow_score),
                anchor_distance=1.0e9,
                anchor_reachability=0.0,
                bridge_entity_coverage=float(bridge_entity_coverage),
                relation_term_coverage=float(relation_term_coverage),
                source_key=str(title),
                normalized_token_set=token_set,
                entity_set=entity_set,
                source_order=_sentence_node_order(local_graph, node),
                query_entity_coverage=float(query_entity_coverage),
                answerability_base=0.0,
                bridge_base=0.0,
                carrier_id=str(carrier_id),
                flow_rank=int(flow_rank.get(node, 0)),
                source_tags=tags,
                proposal_source=str(proposal_source),
            )
        )

    return atoms, {
        "query_tokens": query_tokens,
        "anchor_tokens": anchor_tokens,
        "relation_terms": relation_terms,
        "bridge_entities": bridge_entities,
        "source_balance": disabled_source_balance_diagnostics(top_m=max(1, int(candidate_top_m))),
    }


def _assign_anchor_distances(
    local_graph: nx.Graph,
    atoms: List[Phase7EvidenceAtom],
    anchor_nodes: List[str],
) -> None:
    valid_anchors = [a for a in list(anchor_nodes or []) if a in local_graph]
    if not atoms:
        return
    if not valid_anchors:
        for atom in atoms:
            atom.anchor_distance = 1.0e9
            atom.anchor_reachability = 0.0
        return

    try:
        dmap = nx.multi_source_dijkstra_path_length(local_graph, valid_anchors, cutoff=8, weight=None)
    except Exception:
        dmap = {}

    for atom in atoms:
        d = dmap.get(atom.atom_id, None)
        if d is None:
            atom.anchor_distance = 1.0e9
            atom.anchor_reachability = 0.0
        else:
            atom.anchor_distance = float(d)
            atom.anchor_reachability = float(1.0 / (1.0 + float(d)))


def _fill_normalized_bases(atoms: List[Phase7EvidenceAtom]) -> None:
    semantic_norm = _minmax_norm_map({a.atom_id: a.semantic_score for a in atoms})
    flow_norm = _minmax_norm_map({a.atom_id: a.flow_score for a in atoms})
    qcov_norm = _minmax_norm_map({a.atom_id: a.query_entity_coverage for a in atoms})
    bridge_cov_norm = _minmax_norm_map({a.atom_id: a.bridge_entity_coverage for a in atoms})
    relation_cov_norm = _minmax_norm_map({a.atom_id: a.relation_term_coverage for a in atoms})
    reach_norm = _minmax_norm_map({a.atom_id: a.anchor_reachability for a in atoms})
    inv_dist_norm = _minmax_norm_map(
        {a.atom_id: (0.0 if a.anchor_distance >= 1.0e8 else 1.0 / (1.0 + float(a.anchor_distance))) for a in atoms}
    )
    for atom in atoms:
        aid = atom.atom_id
        atom.answerability_base = _clip01(
            _mean(
                [
                    semantic_norm.get(aid, 0.0),
                    flow_norm.get(aid, 0.0),
                    qcov_norm.get(aid, 0.0),
                ]
            )
        )
        atom.bridge_base = _clip01(
            _mean(
                [
                    bridge_cov_norm.get(aid, 0.0),
                    relation_cov_norm.get(aid, 0.0),
                    reach_norm.get(aid, 0.0),
                    inv_dist_norm.get(aid, 0.0),
                ]
            )
        )


def _corridor_anchor_rows(
    graph: nx.Graph,
    lexical_anchor_nodes: List[str],
    semantic_entity_nodes: List[str],
    semantic_entity_scores: Dict[str, float],
    atoms: List[Phase7EvidenceAtom],
    max_anchors: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    max_a = max(1, int(max_anchors))

    lexical_rows: List[Tuple[float, str]] = []
    for node in list(lexical_anchor_nodes or []):
        if node not in graph:
            continue
        ntype = _node_type(graph, node)
        if ntype != "entity":
            continue
        lexical_rows.append((1.0, str(node)))
    lexical_rows.sort(key=lambda x: (float(x[0]), str(x[1])), reverse=True)
    for score, node in lexical_rows:
        if node in seen:
            continue
        seen.add(node)
        rows.append(
            {
                "node_id": str(node),
                "node_type": str(_node_type(graph, node)),
                "source": "query_entity",
                "score": float(score),
            }
        )
        if len(rows) >= max_a:
            return rows

    semantic_rows: List[Tuple[float, str]] = []
    for node in list(semantic_entity_nodes or []):
        if node not in graph:
            continue
        ntype = _node_type(graph, node)
        if ntype != "entity":
            continue
        semantic_rows.append((float(semantic_entity_scores.get(node, 0.0) or 0.0), str(node)))
    semantic_rows.sort(key=lambda x: (float(x[0]), str(x[1])), reverse=True)
    for score, node in semantic_rows:
        if node in seen:
            continue
        seen.add(node)
        rows.append(
            {
                "node_id": str(node),
                "node_type": str(_node_type(graph, node)),
                "source": "semantic_anchor",
                "score": float(score),
            }
        )
        if len(rows) >= max_a:
            return rows

    atom_rows = sorted(
        list(atoms or []),
        key=lambda a: (float(a.answerability_base), float(a.flow_score), float(a.semantic_score), str(a.atom_id)),
        reverse=True,
    )
    for atom in atom_rows:
        node = str(atom.atom_id)
        if node not in graph or node in seen:
            continue
        seen.add(node)
        rows.append(
            {
                "node_id": str(node),
                "node_type": str(_node_type(graph, node)),
                "source": "semantic_anchor",
                "score": float(atom.answerability_base),
            }
        )
        if len(rows) >= max_a:
            return rows
    return rows


def _corridor_seed_rows(atoms: List[Phase7EvidenceAtom], max_seeds: int) -> List[Dict[str, Any]]:
    rows = sorted(
        list(atoms or []),
        key=lambda a: (float(a.answerability_base), float(a.flow_score), float(a.semantic_score), str(a.atom_id)),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    for atom in rows[: max(1, int(max_seeds))]:
        out.append(
            {
                "node_id": str(atom.atom_id),
                "node_type": "sentence",
                "score": float(atom.answerability_base),
            }
        )
    return out


def _corridor_filtered_graph(
    local_graph: nx.Graph,
    degree_cap: int,
    preserve_nodes: Set[str],
) -> nx.Graph:
    keep_nodes: List[str] = []
    cap = max(1, int(degree_cap))
    for node in list(local_graph.nodes):
        if str(node) in preserve_nodes:
            keep_nodes.append(str(node))
            continue
        if int(local_graph.degree(node)) <= cap:
            keep_nodes.append(str(node))
    # Keep a graph view to avoid expensive deep-copy on each query.
    return local_graph.subgraph(keep_nodes)


def _bounded_bfs_paths_to_targets(
    graph: nx.Graph,
    source: str,
    targets: Set[str],
    max_hops: int,
) -> Dict[str, List[str]]:
    src = str(source or "")
    if (not src) or (src not in graph):
        return {}
    pending = set(str(t) for t in list(targets or set()) if str(t) in graph)
    if not pending:
        return {}

    depth_limit = max(1, int(max_hops))
    parent: Dict[str, Optional[str]] = {src: None}
    depth: Dict[str, int] = {src: 0}
    q = deque([src])

    if src in pending:
        pending.remove(src)

    while q and pending:
        cur = str(q.popleft())
        cur_depth = int(depth.get(cur, 0))
        if cur_depth >= depth_limit:
            continue
        # Deterministic neighbor traversal is required to preserve path choices
        # across optimization variants and repeated runs.
        for nbr in sorted((str(n) for n in graph.neighbors(cur))):
            nb = str(nbr)
            if nb in parent:
                continue
            parent[nb] = cur
            depth[nb] = int(cur_depth + 1)
            if nb in pending:
                pending.remove(nb)
            q.append(nb)
            if not pending:
                break

    out: Dict[str, List[str]] = {}
    for target in set(str(t) for t in list(targets or set())):
        if target not in parent:
            continue
        path_rev: List[str] = []
        cur: Optional[str] = target
        while cur is not None:
            path_rev.append(str(cur))
            cur = parent.get(str(cur), None)
        path = list(reversed(path_rev))
        hops = max(0, len(path) - 1)
        if hops <= depth_limit:
            out[target] = path
    return out


def _extract_anchor_seed_corridors(
    local_graph: nx.Graph,
    anchor_rows: List[Dict[str, Any]],
    seed_rows: List[Dict[str, Any]],
    max_hops: int,
    max_paths_per_pair: int,
    degree_cap: int,
    max_pairs: int,
) -> Dict[str, Any]:
    # Preserve deterministic row order while removing duplicates.
    anchors: List[str] = []
    seen_anchors: Set[str] = set()
    for row in list(anchor_rows or []):
        node_id = str(row.get("node_id", "") or "")
        if not node_id or node_id in seen_anchors:
            continue
        seen_anchors.add(node_id)
        anchors.append(node_id)
    seeds: List[str] = []
    seen_seeds: Set[str] = set()
    for row in list(seed_rows or []):
        node_id = str(row.get("node_id", "") or "")
        if not node_id or node_id in seen_seeds:
            continue
        seen_seeds.add(node_id)
        seeds.append(node_id)
    preserve = set(anchors).union(set(seeds))
    graph_f = _corridor_filtered_graph(local_graph, degree_cap=degree_cap, preserve_nodes=preserve)

    pair_limit = max(1, int(max_pairs))
    pairs: List[Tuple[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for a in anchors:
        for s in seeds:
            key = (str(a), str(s))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append(key)
            if len(pairs) >= pair_limit:
                break
        if len(pairs) >= pair_limit:
            break

    node_to_anchors: Dict[str, Set[str]] = {}
    node_to_seeds: Dict[str, Set[str]] = {}
    node_to_paths: Dict[str, Set[str]] = {}
    path_lengths: Dict[str, int] = {}
    path_rows: List[Dict[str, Any]] = []

    paths_per_pair = max(1, int(max_paths_per_pair))
    hop_limit = max(1, int(max_hops))
    if paths_per_pair <= 0:
        pairs = []

    # Reuse a bounded BFS per anchor and stop early after all pair targets are found.
    # This avoids repeated pairwise shortest-path calls and avoids materializing full path
    # maps for all reachable nodes.
    pairs_by_anchor: Dict[str, List[Tuple[int, str]]] = {}
    for pair_idx, (a, s) in enumerate(pairs):
        pairs_by_anchor.setdefault(str(a), []).append((int(pair_idx), str(s)))

    for a in anchors:
        if a not in graph_f:
            continue
        anchor_pairs = sorted(
            list(pairs_by_anchor.get(str(a), []) or []),
            key=lambda x: (int(x[0]), str(x[1])),
        )
        if not anchor_pairs:
            continue
        target_ids = set(str(seed_id) for _pair_idx, seed_id in anchor_pairs)
        try:
            shortest_from_anchor = _bounded_bfs_paths_to_targets(
                graph=graph_f,
                source=str(a),
                targets=target_ids,
                max_hops=int(hop_limit),
            )
        except Exception:
            shortest_from_anchor = {}
        for pair_idx, s in anchor_pairs:
            if s not in graph_f:
                continue
            path = list(shortest_from_anchor.get(str(s), []) or [])
            if not path:
                continue
            hops = int(max(0, len(path) - 1))
            if hops > hop_limit:
                continue
            pid = f"pair_{pair_idx:03d}"
            path_lengths[pid] = int(hops)
            path_nodes = [str(n) for n in path]
            path_rows.append(
                {
                    "path_id": pid,
                    "anchor_id": str(a),
                    "seed_id": str(s),
                    "hops": int(hops),
                    "node_ids": path_nodes,
                }
            )
            for node in path_nodes:
                node_to_anchors.setdefault(node, set()).add(str(a))
                node_to_seeds.setdefault(node, set()).add(str(s))
                node_to_paths.setdefault(node, set()).add(str(pid))

    corridor_nodes = sorted(list(node_to_paths.keys()))
    corridor_sentence_nodes = [
        n for n in corridor_nodes if str(local_graph.nodes[n].get("node_type", "") or "").strip().lower() == "sentence"
    ]
    hop_vals = [int(v) for v in path_lengths.values()]
    avg_hops = float(sum(hop_vals) / float(len(hop_vals))) if hop_vals else None
    return {
        "graph_filtered_nodes": int(graph_f.number_of_nodes()),
        "graph_filtered_edges": int(graph_f.number_of_edges()),
        "anchor_seed_pairs": int(len(pairs)),
        "paths_found": int(len(path_rows)),
        "corridor_nodes": corridor_nodes,
        "corridor_sentence_nodes": corridor_sentence_nodes,
        "node_to_anchors": node_to_anchors,
        "node_to_seeds": node_to_seeds,
        "node_to_paths": node_to_paths,
        "path_lengths": path_lengths,
        "path_rows": path_rows,
        "avg_path_hops": avg_hops,
    }


def _assign_anchor_decay_scores(
    local_graph: nx.Graph,
    atoms: List[Phase7EvidenceAtom],
    anchor_nodes: List[str],
    gamma: float,
    max_hops: int,
) -> None:
    valid_anchors = [a for a in list(anchor_nodes or []) if a in local_graph]
    if not atoms or not valid_anchors:
        for atom in atoms:
            atom.anchor_decay_score = 0.0
        return
    try:
        dmap = nx.multi_source_shortest_path_length(local_graph, valid_anchors, cutoff=max(1, int(max_hops)))
    except Exception:
        dmap = {}
    g = max(0.0, float(gamma))
    for atom in atoms:
        d = dmap.get(atom.atom_id, None)
        if d is None:
            atom.anchor_decay_score = 0.0
        else:
            atom.anchor_decay_score = float(math.exp(-g * float(d)))


def _score_corridor_features(
    local_graph: nx.Graph,
    atoms: List[Phase7EvidenceAtom],
    corridor_info: Dict[str, Any],
    num_anchors: int,
    num_seeds: int,
) -> None:
    node_to_anchors = dict(corridor_info.get("node_to_anchors", {}) or {})
    node_to_seeds = dict(corridor_info.get("node_to_seeds", {}) or {})
    node_to_paths = dict(corridor_info.get("node_to_paths", {}) or {})
    path_lengths = dict(corridor_info.get("path_lengths", {}) or {})

    raw_scores: Dict[str, float] = {}
    for atom in list(atoms or []):
        aid = str(atom.atom_id)
        anchor_ids = set(node_to_anchors.get(aid, set()) or set())
        seed_ids = set(node_to_seeds.get(aid, set()) or set())
        path_ids = set(node_to_paths.get(aid, set()) or set())

        atom.connected_anchor_ids = set(anchor_ids)
        atom.connected_seed_ids = set(seed_ids)
        atom.corridor_path_ids = set(path_ids)
        atom.corridor_path_count = int(len(path_ids))
        atom.is_corridor_node = bool(path_ids)

        anchor_cov = float(len(anchor_ids) / float(max(1, int(num_anchors))))
        seed_cov = float(len(seed_ids) / float(max(1, int(num_seeds))))
        dist_term = float(0.0 if atom.anchor_distance >= 1.0e8 else 1.0 / (1.0 + float(atom.anchor_distance)))
        path_closure = _mean([1.0 / (1.0 + float(path_lengths.get(pid, 0))) for pid in path_ids]) if path_ids else 0.0
        deg = int(local_graph.degree(aid)) if aid in local_graph else 0
        deg_norm = float(min(1.0, float(len(path_ids)) / math.sqrt(1.0 + float(max(0, deg))))) if path_ids else 0.0
        is_member = 1.0 if path_ids else 0.0

        atom.corridor_anchor_coverage = _clip01(anchor_cov)
        atom.corridor_seed_coverage = _clip01(seed_cov)
        atom.corridor_path_closure = _clip01(path_closure)
        atom.corridor_degree_contrib = _clip01(deg_norm)

        raw = _mean(
            [
                is_member,
                atom.corridor_anchor_coverage,
                atom.corridor_seed_coverage,
                _clip01(dist_term),
                atom.corridor_path_closure,
                atom.corridor_degree_contrib,
            ]
        )
        raw_scores[aid] = float(raw)

    norm = _minmax_norm_map(raw_scores)
    for atom in list(atoms or []):
        atom.corridor_score = _clip01(norm.get(atom.atom_id, 0.0))


def _token_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a.union(b)
    if not union:
        return 0.0
    return float(len(a.intersection(b)) / float(len(union)))


def _conditional_novelty_q(
    cand: Phase7EvidenceAtom,
    selected_anchor_coverage: Set[str],
    selected_corridor_paths: Set[str],
    selected_sources: Set[str],
) -> float:
    new_anchor_cov = 0.0
    if cand.connected_anchor_ids:
        new_anchor_cov = float(
            len(set(cand.connected_anchor_ids).difference(selected_anchor_coverage))
            / float(max(1, len(set(cand.connected_anchor_ids))))
        )
    new_corridor_cov = 0.0
    if cand.corridor_path_ids:
        new_corridor_cov = float(
            len(set(cand.corridor_path_ids).difference(selected_corridor_paths))
            / float(max(1, len(set(cand.corridor_path_ids))))
        )
    new_source_cov = 1.0 if (cand.source_key and cand.source_key not in selected_sources) else 0.0
    return _clip01(_mean([new_anchor_cov, new_corridor_cov, new_source_cov]))


def _atom_title_key(atom: Phase7EvidenceAtom) -> str:
    return _slot_key(atom.title or atom.source_key)


def _canonical_source_ids(atoms: Iterable[Phase7EvidenceAtom]) -> Tuple[str, ...]:
    return tuple(sorted(str(a.source_id) for a in list(atoms or []) if str(a.source_id)))


def _unit_id_for(atoms: Iterable[Phase7EvidenceAtom]) -> str:
    return "||".join(_canonical_source_ids(atoms))


def _explicit_transition_pair(a: Phase7EvidenceAtom, b: Phase7EvidenceAtom) -> bool:
    title_a = _atom_title_key(a)
    title_b = _atom_title_key(b)
    ents_a = set(str(x) for x in set(a.entity_set or set()) if str(x))
    ents_b = set(str(x) for x in set(b.entity_set or set()) if str(x))
    if title_a and title_a in ents_b:
        return True
    if title_b and title_b in ents_a:
        return True
    text_a = str(a.text or "").lower()
    text_b = str(b.text or "").lower()
    if title_a and len(title_a) >= 4 and title_a in text_b:
        return True
    if title_b and len(title_b) >= 4 and title_b in text_a:
        return True
    return False


def _same_title_pair(a: Phase7EvidenceAtom, b: Phase7EvidenceAtom) -> bool:
    ta = _atom_title_key(a)
    tb = _atom_title_key(b)
    return bool(ta and tb and ta == tb)


def _same_carrier_pair(a: Phase7EvidenceAtom, b: Phase7EvidenceAtom) -> bool:
    ca = str(a.carrier_id or "")
    cb = str(b.carrier_id or "")
    return bool(ca and cb and ca == cb)


def _strict_pair_type(a: Phase7EvidenceAtom, b: Phase7EvidenceAtom, p7: Phase7Config) -> str:
    if bool(p7.chain_unit_use_explicit_transition) and _explicit_transition_pair(a, b):
        return "explicit_transition_pair"
    if bool(p7.chain_unit_use_same_title) and _same_title_pair(a, b):
        return "same_title_pair"
    if bool(p7.chain_unit_use_same_carrier) and _same_carrier_pair(a, b):
        return "same_carrier_pair"
    return ""


def _unit_type_priority(unit_type: str) -> int:
    return {
        "explicit_transition_pair": 0,
        "same_title_pair": 1,
        "same_carrier_pair": 2,
        "single_atom": 3,
    }.get(str(unit_type), 9)


def _chain_unit_preview(units: List[Phase7ChainUnit], limit: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for unit in list(units or []):
        if len(unit.atoms) <= 1:
            continue
        rows.append(
            {
                "unit_id": str(unit.unit_id),
                "unit_type": str(unit.unit_type),
                "source_ids": list(unit.source_ids),
                "titles": [str(atom.title or "") for atom in unit.atoms],
                "token_count": int(unit.token_count),
                "base_score": float(unit.base_score),
            }
        )
        if len(rows) >= int(limit):
            break
    return rows


def _build_chain_units(atoms: List[Phase7EvidenceAtom], p7: Phase7Config) -> Tuple[List[Phase7ChainUnit], Dict[str, Any]]:
    t0 = time.perf_counter()
    units: List[Phase7ChainUnit] = []
    seen: Set[str] = set()

    for atom in list(atoms or []):
        uid = _unit_id_for([atom])
        if uid in seen:
            continue
        seen.add(uid)
        units.append(
            Phase7ChainUnit(
                unit_id=str(uid),
                unit_type="single_atom",
                atoms=(atom,),
                source_ids=(str(atom.source_id),),
                token_count=int(atom.token_count),
                base_score=float(0.55 * atom.answerability_base + 0.45 * atom.corridor_score),
            )
        )

    pair_rows: List[Tuple[int, float, str, str, Phase7EvidenceAtom, Phase7EvidenceAtom]] = []
    max_unit_size = max(1, int(p7.chain_unit_max_unit_size))
    if max_unit_size >= 2:
        atom_list = list(atoms or [])
        for i, a in enumerate(atom_list):
            for b in atom_list[i + 1 :]:
                unit_type = _strict_pair_type(a, b, p7)
                if not unit_type:
                    continue
                pair_score = float(
                    _mean(
                        [
                            0.55 * a.answerability_base + 0.45 * a.corridor_score,
                            0.55 * b.answerability_base + 0.45 * b.corridor_score,
                        ]
                    )
                )
                pair_rows.append(
                    (
                        _unit_type_priority(unit_type),
                        -pair_score,
                        _unit_id_for([a, b]),
                        unit_type,
                        a,
                        b,
                    )
                )
    pair_rows.sort(key=lambda row: (int(row[0]), float(row[1]), str(row[2])))
    max_pairs = max(0, int(p7.chain_unit_max_pair_units))
    for _prio, neg_score, uid, unit_type, a, b in pair_rows[:max_pairs]:
        if uid in seen:
            continue
        seen.add(uid)
        units.append(
            Phase7ChainUnit(
                unit_id=str(uid),
                unit_type=str(unit_type),
                atoms=(a, b),
                source_ids=_canonical_source_ids([a, b]),
                token_count=int(a.token_count + b.token_count),
                base_score=float(-neg_score),
            )
        )

    pair_units = [u for u in units if len(u.atoms) > 1]
    diag = {
        "num_single_units": int(sum(1 for u in units if u.unit_type == "single_atom")),
        "num_pair_units": int(len(pair_units)),
        "num_explicit_transition_units": int(sum(1 for u in pair_units if u.unit_type == "explicit_transition_pair")),
        "num_same_title_units": int(sum(1 for u in pair_units if u.unit_type == "same_title_pair")),
        "num_same_carrier_units": int(sum(1 for u in pair_units if u.unit_type == "same_carrier_pair")),
        "num_selected_units": 0,
        "num_selected_atoms": 0,
        "unit_preview": _chain_unit_preview(units),
        "chain_unit_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    return units, diag


def _chain_graph_from_units(units: List[Phase7ChainUnit]) -> nx.Graph:
    graph = nx.Graph()
    for unit in list(units or []):
        ids = [str(sid) for sid in unit.source_ids]
        for sid in ids:
            graph.add_node(sid)
        if len(ids) >= 2:
            graph.add_edge(ids[0], ids[1], unit_type=str(unit.unit_type))
    return graph


def _unit_connects_to_selected(unit: Phase7ChainUnit, selected: List[Phase7EvidenceAtom], p7: Phase7Config) -> bool:
    if not selected:
        return False
    for atom in list(unit.atoms or ()):
        for sel in selected:
            if _strict_pair_type(atom, sel, p7):
                return True
    return False


def _unit_chain_base(unit_type: str) -> float:
    return {
        "explicit_transition_pair": 1.0,
        "same_title_pair": 0.7,
        "same_carrier_pair": 0.5,
        "single_atom": 0.0,
    }.get(str(unit_type), 0.0)


def _unit_delta(
    unit: Phase7ChainUnit,
    selected: List[Phase7EvidenceAtom],
    selected_query_tokens: Set[str],
    selected_bridge_entities: Set[str],
    selected_anchor_coverage: Set[str],
    selected_corridor_paths: Set[str],
    selected_sources: Set[str],
    max_selected_atoms: int,
    max_context_tokens: int,
    selected_token_count: int,
    p7: Phase7Config,
) -> Dict[str, Any]:
    selected_ids = set(str(a.source_id) for a in selected)
    new_atoms = [a for a in list(unit.atoms or ()) if str(a.source_id) not in selected_ids]
    if not new_atoms:
        return {"final_gain": -1.0e9, "new_atoms": []}
    remaining_atoms = max(1, int(max_selected_atoms) - int(len(selected)))
    remaining_tokens = max(1, int(max_context_tokens) - int(selected_token_count))
    new_atom_count = int(len(new_atoms))
    new_token_count = int(sum(int(a.token_count) for a in new_atoms))
    if new_atom_count > remaining_atoms or new_token_count > remaining_tokens:
        return {"final_gain": -1.0e9, "new_atoms": new_atoms, "budget_blocked": True}

    atom_terms = [
        _objective_delta(
            cand=atom,
            selected=selected,
            selected_query_tokens=selected_query_tokens,
            selected_bridge_entities=selected_bridge_entities,
            selected_anchor_coverage=selected_anchor_coverage,
            selected_corridor_paths=selected_corridor_paths,
            selected_sources=selected_sources,
            objective_mode="a_plus_bq_minus_r",
            lambda_bridge=float(p7.lambda_bridge),
            lambda_decay=float(p7.lambda_decay),
            lambda_bq=float(p7.lambda_bq),
            mu_redundancy=float(p7.mu_redundancy),
            conditional_redundancy_enabled=False,
        )
        for atom in new_atoms
    ]
    a_unit = _clip01(_mean(t.get("A", 0.0) for t in atom_terms))
    r_unit = _clip01(_mean(t.get("R", 0.0) for t in atom_terms))
    connect_bonus = 0.15 if _unit_connects_to_selected(unit, selected, p7) else 0.0
    c_unit = _clip01(_unit_chain_base(unit.unit_type) + connect_bonus)
    cost_penalty = _clip01(0.15 * _mean([new_atom_count / float(remaining_atoms), new_token_count / float(remaining_tokens)]))
    gain = float(a_unit + c_unit - r_unit - cost_penalty)
    return {
        "final_gain": float(gain),
        "A_unit": float(a_unit),
        "C_unit": float(c_unit),
        "R_unit": float(r_unit),
        "cost_penalty": float(cost_penalty),
        "new_atoms": list(new_atoms),
        "new_atom_count": int(new_atom_count),
        "new_token_count": int(new_token_count),
        "unit_type": str(unit.unit_type),
    }


def _chain_feasibility_diagnostics(
    atoms: List[Phase7EvidenceAtom],
    units: List[Phase7ChainUnit],
    gold_unit_ids: Set[str],
    selected_ids: Set[str],
    max_selected_atoms: int,
    max_context_tokens: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidate_by_id = {str(a.source_id): a for a in list(atoms or [])}
    candidate_ids = set(candidate_by_id.keys())
    gold_ids = set(str(x) for x in set(gold_unit_ids or set()) if str(x))
    present_gold = sorted(gold_ids.intersection(candidate_ids))
    candidate_partial = bool(present_gold)
    candidate_full = bool(gold_ids) and gold_ids.issubset(candidate_ids)
    candidate_recall = float(len(present_gold) / float(max(1, len(gold_ids)))) if gold_ids else 0.0

    unit_graph = _chain_graph_from_units(units)
    graph_connected = False
    if candidate_full:
        if len(gold_ids) <= 1:
            graph_connected = True
        else:
            try:
                graph_connected = bool(nx.is_connected(unit_graph.subgraph(list(gold_ids))))
            except Exception:
                graph_connected = False

    minimal_atoms = int(len(gold_ids)) if candidate_full else 0
    minimal_tokens = int(sum(candidate_by_id[sid].token_count for sid in gold_ids)) if candidate_full else 0
    budget_feasible = bool(
        candidate_full
        and minimal_atoms <= int(max_selected_atoms)
        and minimal_tokens <= int(max_context_tokens)
    )
    unit_covered_ids: Set[str] = set()
    for unit in list(units or []):
        unit_covered_ids.update(str(sid) for sid in unit.source_ids)
    chain_unit_oracle_feasible = bool(
        budget_feasible
        and gold_ids.issubset(unit_covered_ids)
        and (graph_connected or len(gold_ids) <= 1)
    )
    selected_full = bool(gold_ids) and gold_ids.issubset(set(selected_ids or set()))
    carrier_ids = set(candidate_by_id[sid].carrier_id for sid in gold_ids if sid in candidate_by_id and candidate_by_id[sid].carrier_id)
    carrier_feasible_but_atom_failed = bool(budget_feasible and len(carrier_ids) == 1 and not selected_full)

    feasibility = {
        "candidate_gold_partial": bool(candidate_partial),
        "candidate_gold_full": bool(candidate_full),
        "candidate_gold_recall": float(candidate_recall),
        "candidate_graph_gold_connected": bool(graph_connected),
        "minimal_gold_chain_atoms": int(minimal_atoms),
        "minimal_gold_chain_tokens": int(minimal_tokens),
        "budget_feasible_gold_chain": bool(budget_feasible),
        "chain_unit_oracle_feasible": bool(chain_unit_oracle_feasible),
        "carrier_feasible_but_atom_failed": bool(carrier_feasible_but_atom_failed),
    }
    selected_unit_ids = set(str(sid) for sid in set(selected_ids or set()) if str(sid))
    gold_units_selected = int(sum(1 for unit in units if set(unit.source_ids).intersection(gold_ids).intersection(selected_unit_ids)))
    gold_eval = {
        "gold_unit_partial": bool(gold_ids.intersection(unit_covered_ids)),
        "gold_unit_full": bool(gold_ids) and gold_ids.issubset(unit_covered_ids),
        "num_gold_atoms_covered_by_units": int(len(gold_ids.intersection(unit_covered_ids))),
        "gold_units_selected": int(gold_units_selected),
    }
    return feasibility, gold_eval


def _objective_delta(
    cand: Phase7EvidenceAtom,
    selected: List[Phase7EvidenceAtom],
    selected_query_tokens: Set[str],
    selected_bridge_entities: Set[str],
    selected_anchor_coverage: Set[str],
    selected_corridor_paths: Set[str],
    selected_sources: Set[str],
    objective_mode: str,
    lambda_bridge: float,
    lambda_decay: float,
    lambda_bq: float,
    mu_redundancy: float,
    conditional_redundancy_enabled: bool,
) -> Dict[str, float]:
    cand_query_cov = set(cand.normalized_token_set)
    new_query = cand_query_cov.difference(selected_query_tokens)
    novelty = float(len(new_query) / float(max(1, len(cand_query_cov)))) if cand_query_cov else 0.0
    a_gain = _clip01(cand.answerability_base * (0.7 + 0.3 * novelty))

    new_bridge = cand.entity_set.difference(selected_bridge_entities)
    bridge_novelty = float(len(new_bridge) / float(max(1, len(cand.entity_set)))) if cand.entity_set else 0.0
    connectivity = 0.0
    if selected:
        connectivity = max(
            (_token_jaccard(cand.entity_set, s.entity_set) for s in selected),
            default=0.0,
        )
    b_gain = _clip01(_mean([cand.bridge_base, bridge_novelty, connectivity]))

    max_tok_j = 0.0
    same_source = 0.0
    if selected:
        max_tok_j = max(
            (_token_jaccard(cand.normalized_token_set, s.normalized_token_set) for s in selected),
            default=0.0,
        )
        same_source = 0.2 if any(cand.source_key and cand.source_key == s.source_key for s in selected) else 0.0
    r_pen = _clip01(max_tok_j + same_source)
    rq_pen = float(r_pen)
    novelty_q = 0.0
    if bool(conditional_redundancy_enabled):
        novelty_q = float(
            _conditional_novelty_q(
                cand=cand,
                selected_anchor_coverage=selected_anchor_coverage,
                selected_corridor_paths=selected_corridor_paths,
                selected_sources=selected_sources,
            )
        )
        rq_pen = float(r_pen * (1.0 - novelty_q))

    d_gain = _clip01(cand.anchor_decay_score)
    bq_gain = _clip01(cand.corridor_score)
    bq_eff_gain = _clip01(float(bq_gain) * math.sqrt(_clip01(a_gain)))
    mode = str(objective_mode or "normalized_equal_weight").strip().lower()
    if mode == "a_only":
        gain = float(a_gain)
    elif mode == "a_plus_b":
        gain = float(a_gain + float(lambda_bridge) * b_gain)
    elif mode == "a_minus_r":
        gain = float(a_gain - float(mu_redundancy) * r_pen)
    elif mode == "a_plus_decay_minus_r":
        gain = float(a_gain + float(lambda_decay) * d_gain - float(mu_redundancy) * r_pen)
    elif mode == "a_plus_bq_minus_r":
        gain = float(a_gain + float(lambda_bq) * bq_gain - float(mu_redundancy) * r_pen)
    elif mode == "a_plus_bq_eff_minus_r":
        gain = float(a_gain + float(lambda_bq) * bq_eff_gain - float(mu_redundancy) * r_pen)
    elif mode == "a_plus_bq_minus_rq":
        gain = float(a_gain + float(lambda_bq) * bq_gain - float(mu_redundancy) * rq_pen)
    else:
        # normalized_equal_weight and a_plus_b_minus_r both map to A + lambda*B - mu*R.
        gain = float(a_gain + float(lambda_bridge) * b_gain - float(mu_redundancy) * r_pen)
    return {
        "final_gain": float(gain),
        "A": float(a_gain),
        "D": float(d_gain),
        "B": float(b_gain),
        "Bq": float(bq_gain),
        "Bq_eff": float(bq_eff_gain),
        "R": float(r_pen),
        "Rq": float(rq_pen),
        "novelty_q": float(novelty_q),
    }


def _deterministic_render_order(selected: List[Phase7EvidenceAtom]) -> List[Phase7EvidenceAtom]:
    return sorted(
        list(selected or []),
        key=lambda a: (
            float(a.anchor_distance if a.anchor_distance < 1.0e8 else 1.0e9),
            -float(a.bridge_base),
            -float(a.answerability_base),
            int(a.source_order),
            str(a.atom_id),
        ),
    )


def _support_metrics_eval_only(sample: Any, ids: List[str], texts: List[str], graph_mode: str) -> Dict[str, Optional[float]]:
    sf = list(getattr(sample, "supporting_facts", []) or [])
    if not sf:
        return {
            "recall": None,
            "precision": None,
            "f1": None,
        }
    detail = supporting_fact_match_details(
        sample=sample,
        unit_ids=ids,
        unit_texts=texts,
        graph_mode=graph_mode,
    )
    recall = _safe_float(detail.get("recall", 0.0), 0.0)
    precision = _safe_float(detail.get("precision", 0.0), 0.0)
    f1 = 0.0
    if (recall + precision) > 0.0:
        f1 = float(2.0 * recall * precision / (recall + precision))
    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
    }


def _count_nodes_by_type(graph: nx.Graph) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in graph.nodes:
        node_type = str(graph.nodes[node].get("node_type", "") or "").strip().lower()
        counts[node_type] = int(counts.get(node_type, 0) + 1)
    return counts


def _count_sentence_edge_stats(graph: nx.Graph) -> Dict[str, int]:
    sentence_entity_edges = 0
    sentence_carrier_edges = 0
    for u, v in graph.edges:
        tu = str(graph.nodes[u].get("node_type", "") or "").strip().lower()
        tv = str(graph.nodes[v].get("node_type", "") or "").strip().lower()
        if {tu, tv} == {"sentence", "entity"}:
            sentence_entity_edges += 1
        if ("sentence" in {tu, tv}) and ({tu, tv}.intersection({"carrier", "chunk", "passage", "document"})):
            sentence_carrier_edges += 1
    return {
        "num_sentence_entity_edges": int(sentence_entity_edges),
        "num_sentence_carrier_edges": int(sentence_carrier_edges),
    }


def _node_type(graph: nx.Graph, node: str) -> str:
    if node not in graph:
        return "missing"
    return str(graph.nodes[node].get("node_type", "") or "").strip().lower() or "unknown"


def _unit_consistency_report(cfg: Any, p7: Phase7Config, graph: nx.Graph) -> Dict[str, Any]:
    graph_mode = str(getattr(cfg, "graph_mode", "entity_chunk_graph") or "entity_chunk_graph").strip().lower()
    index_chunk_unit = str(getattr(cfg, "index_chunk_unit", "passage") or "passage").strip().lower()
    candidate_node_type = "sentence"
    selector_node_type = "sentence"
    render_node_type = "sentence"
    gold_matching_unit = "sentence" if graph_mode == "current_entity_graph" else "sentence_or_chunk_title_fallback"
    node_counts = _count_nodes_by_type(graph)
    available_node_types = sorted([str(k) for k in node_counts.keys()])
    warnings: List[str] = []

    sentence_nodes = int(node_counts.get("sentence", 0))
    chunk_nodes = int(node_counts.get("chunk", 0) + node_counts.get("passage", 0) + node_counts.get("document", 0))

    if candidate_node_type == "sentence" and sentence_nodes <= 0:
        warnings.append(
            "selector_node_type=sentence but graph contains zero sentence nodes; selection/render may fail."
        )
    if graph_mode == "entity_chunk_graph" and index_chunk_unit == "passage" and sentence_nodes <= 0:
        warnings.append(
            "entity_chunk_graph+passage index without sentence nodes can break sentence-level matching."
        )
    if chunk_nodes <= 0:
        warnings.append("graph contains zero carrier/chunk nodes; phase1 semantic seeding may be ineffective.")

    mismatch = bool(len(warnings) > 0)
    return {
        "graph_mode": str(graph_mode),
        "index_chunk_unit": str(index_chunk_unit),
        "candidate_node_type": str(candidate_node_type),
        "selector_node_type": str(selector_node_type),
        "render_node_type": str(render_node_type),
        "gold_matching_unit": str(gold_matching_unit),
        "available_node_types": list(available_node_types),
        "node_type_counts": {str(k): int(v) for k, v in sorted(node_counts.items())},
        "warnings": list(warnings),
        "mismatch": bool(mismatch),
    }


def _phase7_index_summary(graph: nx.Graph, p7: Phase7Config) -> Dict[str, Any]:
    node_counts = _count_nodes_by_type(graph)
    edge_counts = _count_sentence_edge_stats(graph)
    return {
        "phase7_index_summary": {
            "phase7_atom_unit": str(p7.atom_unit),
            "phase7_carrier_unit": str(p7.carrier_unit),
            "num_sentence_nodes": int(node_counts.get("sentence", 0)),
            "num_carrier_nodes": int(
                node_counts.get("carrier", 0)
                + node_counts.get("chunk", 0)
                + node_counts.get("passage", 0)
                + node_counts.get("document", 0)
            ),
            "num_entity_nodes": int(node_counts.get("entity", 0)),
            "num_sentence_entity_edges": int(edge_counts.get("num_sentence_entity_edges", 0)),
            "num_sentence_carrier_edges": int(edge_counts.get("num_sentence_carrier_edges", 0)),
            "node_type_counts": {str(k): int(v) for k, v in sorted(node_counts.items())},
        }
    }


def validate_phase7_index_compatibility(graph: nx.Graph, p7: Phase7Config) -> Tuple[bool, str]:
    node_counts = _count_nodes_by_type(graph)
    sentence_nodes = int(node_counts.get("sentence", 0))
    if p7.enabled and str(p7.atom_unit) == "sentence" and sentence_nodes <= 0:
        return (
            False,
            "Phase7 requires sentence nodes, but the loaded index contains zero sentence nodes. "
            "Rebuild the index with phase7_build_sentence_layer=true and phase7_atom_unit=sentence. "
            "Do not run Phase7 with passage-only entity_chunk_graph indexes.",
        )
    return True, ""


def _resolved_output_dir(cfg: Any) -> str:
    out_dir = str(getattr(cfg, "_resolved_output_dir", "") or "").strip()
    if out_dir:
        return out_dir
    return str(getattr(cfg, "output_dir", "outputs/phase7_evidence_flow") or "outputs/phase7_evidence_flow")


def _phase1_seed_rows(
    graph: nx.Graph,
    sem_chunks: List[str],
    sem_chunk_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_source_ids: Set[str] = set()
    for chunk_node in list(sem_chunks or []):
        if chunk_node not in graph:
            continue
        seed_score = float(sem_chunk_scores.get(chunk_node, 0.0) or 0.0)
        chunk_type = _node_type(graph, chunk_node)
        chunk_text = str(graph.nodes[chunk_node].get("text", "") or "")
        chunk_title = str(graph.nodes[chunk_node].get("title", "") or "")
        sent_neighbors = _collect_sentence_nodes_from_chunks(graph, [chunk_node])
        if sent_neighbors:
            ordered_sent_neighbors = sorted(
                list(sent_neighbors),
                key=lambda node: (
                    str(graph.nodes[node].get("title", "") or ""),
                    _safe_int(graph.nodes[node].get("sent_idx", 0), 0),
                    str(node),
                ),
            )
            for sent_node in ordered_sent_neighbors:
                source_id = _sentence_node_to_id(graph, sent_node)
                if source_id in seen_source_ids:
                    continue
                seen_source_ids.add(source_id)
                rows.append(
                    {
                        "node_id": str(sent_node),
                        "source_id": str(source_id),
                        "node_type": "sentence",
                        "carrier_id": str(chunk_node),
                        "title": _sentence_node_title(graph, sent_node),
                        "text": _sentence_node_text(graph, sent_node),
                        "score": float(seed_score),
                    }
                )
        else:
            source_id = str(chunk_node)
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            rows.append(
                {
                    "node_id": str(chunk_node),
                    "source_id": str(source_id),
                    "node_type": str(chunk_type),
                    "carrier_id": str(chunk_node),
                    "title": str(chunk_title),
                    "text": str(chunk_text),
                    "score": float(seed_score),
                }
            )
    rows.sort(
        key=lambda row: (
            float(row.get("score", 0.0)),
            str(row.get("title", "")),
            str(row.get("source_id", "")),
        ),
        reverse=True,
    )
    return rows


def _select_from_rows_with_budget(
    rows: List[Dict[str, Any]],
    max_selected_atoms: int,
    max_context_tokens: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    selected_tokens = 0
    max_sel = max(1, int(max_selected_atoms))
    max_tok = max(1, int(max_context_tokens))
    for row in list(rows or []):
        if len(out) >= max_sel:
            break
        text = str(row.get("text", "") or "")
        tok = _token_count(text)
        if selected_tokens + tok > max_tok:
            continue
        selected_tokens += int(tok)
        out.append(dict(row))
    return out


@register_method("phase7_evidence_flow")
def run_phase7_evidence_flow(sample: Any, cfg: Any) -> RetrievalResult:
    p7 = phase7_config_from_cfg(cfg)
    phase8_pamae_enabled = bool(getattr(cfg, "phase8_pamae_enabled", False))
    phase8_chunk_enabled = bool(getattr(cfg, "phase8_chunk_medoid_enabled", False))
    phase8_enabled = bool(phase8_pamae_enabled or phase8_chunk_enabled)
    candidate_top_m_eff = max(
        1,
        int(
            getattr(cfg, "phase8_chunk_seed_total_candidate_cap", 160)
            if phase8_chunk_enabled
            else getattr(cfg, "phase8_seed_total_candidate_cap", 160)
            if phase8_pamae_enabled
            else (p7.source_balanced_candidate_top_m if bool(p7.source_balanced_proposal_enabled) else p7.candidate_top_m)
        ),
    )
    max_selected_atoms_eff = max(1, int(p7.max_selected_atoms))
    max_context_tokens_eff = max(32, int(p7.max_context_tokens))
    out_dir = _resolved_output_dir(cfg)
    logger = get_phase7_logger(
        output_dir=out_dir,
        dataset=str(getattr(cfg, "dataset", "") or ""),
        limit=int(getattr(cfg, "limit", 0) or 0),
        config_path=str(getattr(cfg, "_config_path", "") or ""),
        cfg=p7,
    )

    stage_ms: Dict[str, float] = {}
    stage_rows: List[Dict[str, Any]] = []
    phase8_result: Optional[PamaeProposalResult] = None
    phase8_chunk_result: Optional[ChunkMedoidProposalResult] = None
    t_total = time.perf_counter()

    def _stage_begin() -> float:
        return time.perf_counter()

    def _stage_end(name: str, t0: float, **kwargs) -> None:
        elapsed = float((time.perf_counter() - t0) * 1000.0)
        stage_ms[name] = elapsed
        row = {
            "query_id": str(getattr(sample, "qid", "") or ""),
            "stage": str(name),
            "elapsed_ms": elapsed,
            "num_nodes": int(kwargs.get("num_nodes", 0) or 0),
            "num_edges": int(kwargs.get("num_edges", 0) or 0),
            "num_candidates_in": int(kwargs.get("num_candidates_in", 0) or 0),
            "num_candidates_out": int(kwargs.get("num_candidates_out", 0) or 0),
        }
        stage_rows.append(row)

    # query embedding
    t0 = _stage_begin()
    qvec = _question_embedding(str(getattr(sample, "question", "") or ""), cfg)
    _stage_end("query_embedding", t0)

    # index + anchors
    state = _build_index_state(cfg, p7)
    graph = state["graph"]
    graph_mode = str((state.get("meta", {}) or {}).get("build_config", {}).get("graph_mode", "entity_chunk_graph"))
    unit_consistency = _unit_consistency_report(cfg=cfg, p7=p7, graph=graph)

    t0 = _stage_begin()
    index_summary = _phase7_index_summary(graph, p7)
    logger.log_index_summary(index_summary)
    compat_ok, compat_msg = validate_phase7_index_compatibility(graph, p7)
    _stage_end(
        "index_validation",
        t0,
        num_nodes=int(graph.number_of_nodes()),
        num_edges=int(graph.number_of_edges()),
        num_candidates_in=0,
        num_candidates_out=0,
    )
    if not compat_ok:
        raise RuntimeError(str(compat_msg))
    if bool(unit_consistency.get("mismatch", False)) and bool(p7.diagnostics_fail_on_unit_mismatch):
        raise RuntimeError(
            "Phase7 unit mismatch detected (fail_on_unit_mismatch=true): "
            + "; ".join([str(x) for x in list(unit_consistency.get("warnings", []) or [])])
        )

    t0 = _stage_begin()
    anchors = select_lexical_anchors(sample, graph, max_anchors=max(1, int(getattr(cfg, "max_anchors", 5) or 5)), cfg=cfg)
    anchor_nodes = _normalize_anchor_nodes(anchors, graph)
    _stage_end("anchor_extraction", t0, num_nodes=len(anchor_nodes), num_edges=0)

    # semantic anchor retrieval
    t0 = _stage_begin()
    sem_chunks, sem_chunk_scores = _semantic_top_chunks(qvec, state, top_t=int(p7.semantic_anchor_top_t))
    sem_entities, sem_entity_scores = _semantic_top_entities(qvec, state, topn=max(8, int(p7.semantic_anchor_top_t // 4)))
    sem_chunks = [node for node in sem_chunks if node in graph]
    sem_entities = [node for node in sem_entities if node in graph]
    phase1_seed_rows = _phase1_seed_rows(graph=graph, sem_chunks=sem_chunks, sem_chunk_scores=sem_chunk_scores)
    _stage_end(
        "semantic_anchor_retrieval",
        t0,
        num_nodes=int(len(sem_chunks) + len(sem_entities)),
        num_edges=0,
        num_candidates_in=0,
        num_candidates_out=int(len(sem_chunks)),
    )

    # local proposal graph build
    t0 = _stage_begin()
    local_graph, local_diag = _build_local_graph(
        graph=graph,
        anchor_nodes=anchor_nodes,
        semantic_chunk_nodes=sem_chunks,
        semantic_entity_nodes=sem_entities,
    )
    _stage_end(
        "local_graph_build",
        t0,
        num_nodes=int(local_diag.get("local_nodes", 0)),
        num_edges=int(local_diag.get("local_edges", 0)),
    )

    # graph flow
    t0 = _stage_begin()
    flow = _flow_scores(local_graph, seed_nodes=(anchor_nodes + sem_chunks + sem_entities), p7=p7)
    _stage_end(
        "graph_flow",
        t0,
        num_nodes=int(local_graph.number_of_nodes()),
        num_edges=int(local_graph.number_of_edges()),
    )

    if phase8_chunk_enabled:
        t0 = _stage_begin()
        phase8_chunk_result = run_phase8_chunk_medoid_proposal(
            query=str(getattr(sample, "question", "") or ""),
            graph_index=state,
            embeddings={
                "query_embedding": qvec,
                "semantic_chunks": list(sem_chunks),
                "semantic_entities": list(sem_entities),
                "anchor_nodes": list(anchor_nodes),
                "flow_scores": dict(flow),
            },
            config=cfg,
            query_embedding=qvec,
        )
        refined_chunk_ids = [str(seed.chunk_id) for seed in list(phase8_chunk_result.refined_medoid_set or [])]
        bridge_entity_ids = sorted(
            {
                str(entity_id)
                for seed in list(phase8_chunk_result.refined_medoid_set or [])
                for entity_id in list(seed.bridge_entities or [])
                if str(entity_id) in graph
            }
        )
        phase8_candidate_ids = [str(cid) for cid in list(phase8_chunk_result.evidence_candidate_ids or []) if str(cid) in graph]
        _stage_end(
            "phase8_chunk_medoid_proposal",
            t0,
            num_nodes=int(len(phase8_chunk_result.chunk_universe)),
            num_edges=0,
            num_candidates_in=int(len(phase8_chunk_result.chunk_universe)),
            num_candidates_out=int(len(phase8_candidate_ids)),
        )

        t0 = _stage_begin()
        local_graph, local_diag = _build_local_graph(
            graph=graph,
            anchor_nodes=anchor_nodes,
            semantic_chunk_nodes=list(dict.fromkeys(refined_chunk_ids + phase8_candidate_ids)),
            semantic_entity_nodes=bridge_entity_ids,
        )
        _stage_end(
            "phase8_local_graph_build",
            t0,
            num_nodes=int(local_diag.get("local_nodes", 0)),
            num_edges=int(local_diag.get("local_edges", 0)),
        )

        t0 = _stage_begin()
        flow = _flow_scores(local_graph, seed_nodes=(anchor_nodes + bridge_entity_ids + refined_chunk_ids + phase8_candidate_ids), p7=p7)
        _stage_end(
            "phase8_graph_flow",
            t0,
            num_nodes=int(local_graph.number_of_nodes()),
            num_edges=int(local_graph.number_of_edges()),
        )
        sem_chunks = [sid for sid in refined_chunk_ids if sid in graph]
        sem_entities = [sid for sid in bridge_entity_ids if sid in graph]
        sem_chunk_scores = {
            str(seed.chunk_id): float(seed.score)
            for seed in list(phase8_chunk_result.refined_medoid_set or [])
            if str(seed.chunk_id) in graph
        }
        phase1_seed_rows = [
            {
                "node_id": str(seed.chunk_id),
                "source_id": str(seed.chunk_id),
                "node_type": str((graph.nodes[str(seed.chunk_id)].get("node_type", "") if str(seed.chunk_id) in graph else "") or "chunk"),
                "carrier_id": str(seed.chunk_id),
                "title": str((graph.nodes[str(seed.chunk_id)].get("title", "") if str(seed.chunk_id) in graph else "") or ""),
                "text": str((graph.nodes[str(seed.chunk_id)].get("text", "") if str(seed.chunk_id) in graph else "") or str(seed.chunk_id)),
                "score": float(seed.score),
            }
            for seed in list(phase8_chunk_result.refined_medoid_set or [])
        ]

    elif phase8_pamae_enabled:
        t0 = _stage_begin()
        phase8_result = run_phase8_pamae_proposal(
            query=str(getattr(sample, "question", "") or ""),
            graph_index=state,
            embeddings={
                "query_embedding": qvec,
                "semantic_chunks": list(sem_chunks),
                "semantic_entities": list(sem_entities),
                "anchor_nodes": list(anchor_nodes),
                "flow_scores": dict(flow),
            },
            config=cfg,
            query_embedding=qvec,
            anchors=anchor_nodes,
        )
        refined_seed_ids = [str(seed.entity_id) for seed in list(phase8_result.refined_medoid_set or [])]
        phase8_candidate_ids = [str(cid) for cid in list(phase8_result.evidence_candidate_ids or []) if str(cid) in graph]
        _stage_end(
            "phase8_pamae_proposal",
            t0,
            num_nodes=int(len(phase8_result.entity_universe)),
            num_edges=0,
            num_candidates_in=int(len(phase8_result.entity_universe)),
            num_candidates_out=int(len(phase8_candidate_ids)),
        )

        t0 = _stage_begin()
        local_graph, local_diag = _build_local_graph(
            graph=graph,
            anchor_nodes=anchor_nodes,
            semantic_chunk_nodes=phase8_candidate_ids,
            semantic_entity_nodes=refined_seed_ids,
        )
        _stage_end(
            "phase8_local_graph_build",
            t0,
            num_nodes=int(local_diag.get("local_nodes", 0)),
            num_edges=int(local_diag.get("local_edges", 0)),
        )

        t0 = _stage_begin()
        flow = _flow_scores(local_graph, seed_nodes=(anchor_nodes + refined_seed_ids + phase8_candidate_ids), p7=p7)
        _stage_end(
            "phase8_graph_flow",
            t0,
            num_nodes=int(local_graph.number_of_nodes()),
            num_edges=int(local_graph.number_of_edges()),
        )
        sem_entities = [sid for sid in refined_seed_ids if sid in graph]
        sem_entity_scores = {
            str(seed.entity_id): float(seed.score)
            for seed in list(phase8_result.refined_medoid_set or [])
            if str(seed.entity_id) in graph
        }
        phase1_seed_rows = [
            {
                "node_id": str(seed.entity_id),
                "source_id": str(seed.entity_id),
                "node_type": "entity",
                "carrier_id": "",
                "title": str((graph.nodes[str(seed.entity_id)].get("title", "") if str(seed.entity_id) in graph else "") or ""),
                "text": str((graph.nodes[str(seed.entity_id)].get("name", "") if str(seed.entity_id) in graph else "") or str(seed.entity_id)),
                "score": float(seed.score),
            }
            for seed in list(phase8_result.refined_medoid_set or [])
        ]

    # candidate truncation (top-M atoms)
    t0 = _stage_begin()
    local_node_type_counts = _count_nodes_by_type(local_graph)
    candidate_source_breakdown = {
        "sentence": int(local_node_type_counts.get("sentence", 0)),
        "carrier": int(local_node_type_counts.get("carrier", 0)),
        "chunk": int(local_node_type_counts.get("chunk", 0)),
        "passage": int(local_node_type_counts.get("passage", 0)),
        "entity": int(local_node_type_counts.get("entity", 0)),
    }
    num_candidate_atoms_before_truncation = int(candidate_source_breakdown["sentence"])
    if phase8_chunk_enabled and phase8_chunk_result is not None:
        phase8_tags = dict((phase8_chunk_result.diagnostics or {}).get("evidence_source_tags", {}) or {})
        atoms, aux = _candidate_atoms_from_phase8_evidence(
            question=str(getattr(sample, "question", "") or ""),
            local_graph=local_graph,
            evidence_candidate_ids=list(phase8_chunk_result.evidence_candidate_ids or []),
            evidence_source_tags=phase8_tags,
            flow_scores=flow,
            anchor_nodes=anchor_nodes,
            candidate_top_m=int(candidate_top_m_eff),
            proposal_tag="phase8_chunk_medoid",
            proposal_source="phase8_chunk_medoid",
        )
        aux["source_balance"] = {
            **dict(aux.get("source_balance", {}) or {}),
            "enabled": False,
            "phase8_chunk_medoid_enabled": True,
            "num_phase8_chunk_evidence_candidates": int(len(phase8_chunk_result.evidence_candidate_ids or [])),
        }
    elif phase8_pamae_enabled and phase8_result is not None:
        phase8_tags = dict((phase8_result.diagnostics or {}).get("evidence_source_tags", {}) or {})
        atoms, aux = _candidate_atoms_from_phase8_evidence(
            question=str(getattr(sample, "question", "") or ""),
            local_graph=local_graph,
            evidence_candidate_ids=list(phase8_result.evidence_candidate_ids or []),
            evidence_source_tags=phase8_tags,
            flow_scores=flow,
            anchor_nodes=anchor_nodes,
            candidate_top_m=int(candidate_top_m_eff),
        )
        aux["source_balance"] = {
            **dict(aux.get("source_balance", {}) or {}),
            "enabled": False,
            "phase8_pamae_enabled": True,
            "num_phase8_evidence_candidates": int(len(phase8_result.evidence_candidate_ids or [])),
        }
    else:
        atoms, aux = _candidate_atoms_from_local_graph(
            question=str(getattr(sample, "question", "") or ""),
            local_graph=local_graph,
            flow_scores=flow,
            semantic_chunk_scores=sem_chunk_scores,
            anchor_nodes=anchor_nodes,
            candidate_top_m=int(candidate_top_m_eff),
            source_balanced_enabled=bool(p7.source_balanced_proposal_enabled),
            source_balanced_top_m=int(candidate_top_m_eff),
            source_quotas={
                "semantic": int(p7.source_quota_semantic),
                "entity_title": int(p7.source_quota_entity_title),
                "graph_flow": int(p7.source_quota_graph_flow),
                "anchor_neighborhood": int(p7.source_quota_anchor_neighborhood),
            },
            source_balanced_fill_remaining=bool(p7.source_balanced_fill_remaining),
        )
    source_balance_diag = dict(aux.get("source_balance", {}) or {})
    source_timing_ms = dict(source_balance_diag.get("source_timing_ms", {}) or {})
    stage_ms["source_balanced_union"] = float(source_balance_diag.get("source_balanced_union_ms", 0.0) or 0.0)
    stage_ms["entity_title_retrieval"] = float(source_timing_ms.get("entity_title_candidates_ms", 0.0) or 0.0)
    stage_ms["anchor_neighborhood"] = float(source_timing_ms.get("anchor_neighborhood_candidates_ms", 0.0) or 0.0)
    _stage_end(
        "candidate_truncation",
        t0,
        num_nodes=int(local_graph.number_of_nodes()),
        num_edges=int(local_graph.number_of_edges()),
        num_candidates_in=int(num_candidate_atoms_before_truncation),
        num_candidates_out=int(len(atoms)),
    )

    if not atoms and not bool(p7.allow_empty_candidates_for_debug):
        raise RuntimeError(
            "Phase7 candidate_generation produced zero sentence candidates "
            "(FLOW_RETURNED_NO_SENTENCE_CANDIDATES). "
            "Run with phase7_allow_empty_candidates_for_debug=true only for diagnostics."
        )

    # feature extraction (single-pass distances + normalized bases)
    t0 = _stage_begin()
    _assign_anchor_distances(local_graph, atoms, anchor_nodes)
    _fill_normalized_bases(atoms)
    _stage_end(
        "feature_extraction",
        t0,
        num_nodes=int(local_graph.number_of_nodes()),
        num_edges=int(local_graph.number_of_edges()),
        num_candidates_in=int(len(atoms)),
        num_candidates_out=int(len(atoms)),
    )

    # anchor-distance decay feature (feature-only, no pruning)
    t0 = _stage_begin()
    if bool(p7.anchor_decay_enabled):
        _assign_anchor_decay_scores(
            local_graph=local_graph,
            atoms=atoms,
            anchor_nodes=anchor_nodes,
            gamma=float(p7.anchor_decay_gamma),
            max_hops=int(p7.anchor_decay_max_hops),
        )
    else:
        for atom in atoms:
            atom.anchor_decay_score = 0.0
    _stage_end(
        "anchor_distance_bfs",
        t0,
        num_nodes=int(local_graph.number_of_nodes()),
        num_edges=int(local_graph.number_of_edges()),
        num_candidates_in=int(len(atoms)),
        num_candidates_out=int(len(atoms)),
    )

    # anchor-seed corridor utility feature (feature-only, no pinning/fallback)
    corridor_diag: Dict[str, Any] = {
        "corridor_enabled": bool(p7.corridor_enabled),
        "num_corridor_anchors": 0,
        "num_corridor_seeds": 0,
        "num_anchor_seed_pairs": 0,
        "num_paths_found": 0,
        "num_corridor_nodes": 0,
        "num_corridor_sentence_nodes": 0,
        "avg_corridor_path_length": None,
        "corridor_feature_ms": 0.0,
    }
    corridor_anchor_rows: List[Dict[str, Any]] = []
    corridor_seed_rows: List[Dict[str, Any]] = []
    corridor_info: Dict[str, Any] = {}
    t_corridor_start = time.perf_counter()
    t0 = _stage_begin()
    if bool(p7.corridor_enabled):
        corridor_anchor_rows = _corridor_anchor_rows(
            graph=local_graph,
            lexical_anchor_nodes=anchor_nodes,
            semantic_entity_nodes=sem_entities,
            semantic_entity_scores=sem_entity_scores,
            atoms=atoms,
            max_anchors=int(p7.corridor_max_anchors),
        )
        corridor_seed_rows = _corridor_seed_rows(
            atoms=atoms,
            max_seeds=int(p7.corridor_max_seeds),
        )
        corridor_info = _extract_anchor_seed_corridors(
            local_graph=local_graph,
            anchor_rows=corridor_anchor_rows,
            seed_rows=corridor_seed_rows,
            max_hops=int(p7.corridor_max_hops),
            max_paths_per_pair=int(p7.corridor_max_paths_per_pair),
            degree_cap=int(p7.corridor_degree_cap),
            max_pairs=int(p7.corridor_max_pairs),
        )
    _stage_end(
        "corridor_extraction",
        t0,
        num_nodes=int(local_graph.number_of_nodes()),
        num_edges=int(local_graph.number_of_edges()),
        num_candidates_in=int(len(atoms)),
        num_candidates_out=int(len(corridor_info.get("corridor_sentence_nodes", []) or [])),
    )

    t0 = _stage_begin()
    if bool(p7.corridor_enabled):
        _score_corridor_features(
            local_graph=local_graph,
            atoms=atoms,
            corridor_info=corridor_info,
            num_anchors=max(1, len(corridor_anchor_rows)),
            num_seeds=max(1, len(corridor_seed_rows)),
        )
    else:
        for atom in atoms:
            atom.corridor_score = 0.0
            atom.corridor_anchor_coverage = 0.0
            atom.corridor_seed_coverage = 0.0
            atom.corridor_path_closure = 0.0
            atom.corridor_degree_contrib = 0.0
            atom.corridor_path_count = 0
            atom.is_corridor_node = False
            atom.connected_anchor_ids = set()
            atom.connected_seed_ids = set()
            atom.corridor_path_ids = set()
    _stage_end(
        "bq_feature_scoring",
        t0,
        num_nodes=int(local_graph.number_of_nodes()),
        num_edges=int(local_graph.number_of_edges()),
        num_candidates_in=int(len(atoms)),
        num_candidates_out=int(len(atoms)),
    )
    corridor_diag = {
        "corridor_enabled": bool(p7.corridor_enabled),
        "num_corridor_anchors": int(len(corridor_anchor_rows)),
        "num_corridor_seeds": int(len(corridor_seed_rows)),
        "num_anchor_seed_pairs": int(corridor_info.get("anchor_seed_pairs", 0) or 0),
        "num_paths_found": int(corridor_info.get("paths_found", 0) or 0),
        "num_corridor_nodes": int(len(corridor_info.get("corridor_nodes", []) or [])),
        "num_corridor_sentence_nodes": int(len(corridor_info.get("corridor_sentence_nodes", []) or [])),
        "avg_corridor_path_length": corridor_info.get("avg_path_hops", None),
        "corridor_feature_ms": float((time.perf_counter() - t_corridor_start) * 1000.0),
    }

    t0 = _stage_begin()
    if bool(p7.chain_unit_enabled) or str(p7.objective_mode).strip().lower() == "chain_unit" or bool(p7.diagnostics_enabled):
        chain_units, chain_unit_diag = _build_chain_units(atoms=atoms, p7=p7)
    else:
        chain_units, chain_unit_diag = [], {
            "num_single_units": 0,
            "num_pair_units": 0,
            "num_explicit_transition_units": 0,
            "num_same_title_units": 0,
            "num_same_carrier_units": 0,
            "num_selected_units": 0,
            "num_selected_atoms": 0,
            "chain_unit_ms": 0.0,
        }
    _stage_end(
        "chain_unit_build",
        t0,
        num_nodes=int(local_graph.number_of_nodes()),
        num_edges=int(local_graph.number_of_edges()),
        num_candidates_in=int(len(atoms)),
        num_candidates_out=int(len(chain_units)),
    )
    chain_unit_diag["chain_unit_ms"] = float(stage_ms.get("chain_unit_build", 0.0))

    # single budgeted marginal selection
    t0 = _stage_begin()
    selected: List[Phase7EvidenceAtom] = []
    selected_units: List[Phase7ChainUnit] = []
    selected_query_tokens: Set[str] = set()
    selected_bridge_entities: Set[str] = set()
    selected_anchor_coverage: Set[str] = set()
    selected_corridor_paths: Set[str] = set()
    selected_sources: Set[str] = set()
    selected_token_count = 0
    answerability_gain_total = 0.0
    decay_gain_total = 0.0
    corridor_gain_total = 0.0
    bq_eff_gain_total = 0.0
    bridge_gain_total = 0.0
    redundancy_penalty_total = 0.0
    conditional_redundancy_penalty_total = 0.0
    objective_gain_total = 0.0
    budget_filtered_candidates = 0
    selected_feature_breakdown: List[Dict[str, Any]] = []
    conditional_redundancy_ms = 0.0

    if str(p7.objective_mode).strip().lower() == "chain_unit" and bool(p7.chain_unit_enabled):
        while len(selected) < int(max_selected_atoms_eff):
            best_unit = None
            best_terms: Dict[str, Any] = {}
            best_gain = float(p7.min_positive_gain)
            for unit in chain_units:
                t_cr = time.perf_counter()
                terms = _unit_delta(
                    unit=unit,
                    selected=selected,
                    selected_query_tokens=selected_query_tokens,
                    selected_bridge_entities=selected_bridge_entities,
                    selected_anchor_coverage=selected_anchor_coverage,
                    selected_corridor_paths=selected_corridor_paths,
                    selected_sources=selected_sources,
                    max_selected_atoms=int(max_selected_atoms_eff),
                    max_context_tokens=int(max_context_tokens_eff),
                    selected_token_count=int(selected_token_count),
                    p7=p7,
                )
                conditional_redundancy_ms += float((time.perf_counter() - t_cr) * 1000.0)
                if bool(terms.get("budget_blocked", False)):
                    budget_filtered_candidates += 1
                gain = float(terms.get("final_gain", 0.0))
                if gain > best_gain:
                    best_gain = float(gain)
                    best_unit = unit
                    best_terms = dict(terms)
            if best_unit is None:
                break
            selected_units.append(best_unit)
            new_atoms = list(best_terms.get("new_atoms", []) or [])
            for atom in new_atoms:
                selected.append(atom)
                selected_token_count += int(atom.token_count)
                selected_query_tokens.update(atom.normalized_token_set)
                selected_bridge_entities.update(atom.entity_set)
                selected_anchor_coverage.update(set(atom.connected_anchor_ids))
                selected_corridor_paths.update(set(atom.corridor_path_ids))
                if atom.source_key:
                    selected_sources.add(str(atom.source_key))
            answerability_gain_total += float(best_terms.get("A_unit", 0.0))
            corridor_gain_total += float(best_terms.get("C_unit", 0.0))
            redundancy_penalty_total += float(best_terms.get("R_unit", 0.0))
            objective_gain_total += float(best_terms.get("final_gain", best_gain))
            selected_feature_breakdown.append(
                {
                    "node_id": str(best_unit.unit_id),
                    "atom_id": str(best_unit.unit_id),
                    "unit_id": str(best_unit.unit_id),
                    "unit_type": str(best_unit.unit_type),
                    "unit_source_ids": list(best_unit.source_ids),
                    "A": float(best_terms.get("A_unit", 0.0)),
                    "D": 0.0,
                    "Bq": 0.0,
                    "Bq_eff": 0.0,
                    "C_unit": float(best_terms.get("C_unit", 0.0)),
                    "R": float(best_terms.get("R_unit", 0.0)),
                    "Rq": float(best_terms.get("R_unit", 0.0)),
                    "cost_penalty": float(best_terms.get("cost_penalty", 0.0)),
                    "final_gain": float(best_terms.get("final_gain", best_gain)),
                    "is_corridor_node": any(bool(a.is_corridor_node) for a in new_atoms),
                    "num_new_atoms": int(len(new_atoms)),
                    "new_token_count": int(best_terms.get("new_token_count", 0)),
                }
            )
    else:
        while len(selected) < int(max_selected_atoms_eff):
            best_atom = None
            best_gain = float(p7.min_positive_gain)
            best_terms: Dict[str, float] = {}
            for atom in atoms:
                if atom.atom_id in {s.atom_id for s in selected}:
                    continue
                if selected_token_count + int(atom.token_count) > int(max_context_tokens_eff):
                    budget_filtered_candidates += 1
                    continue
                t_cr = time.perf_counter()
                terms = _objective_delta(
                    cand=atom,
                    selected=selected,
                    selected_query_tokens=selected_query_tokens,
                    selected_bridge_entities=selected_bridge_entities,
                    selected_anchor_coverage=selected_anchor_coverage,
                    selected_corridor_paths=selected_corridor_paths,
                    selected_sources=selected_sources,
                    objective_mode=str(p7.objective_mode),
                    lambda_bridge=float(p7.lambda_bridge),
                    lambda_decay=float(p7.lambda_decay),
                    lambda_bq=float(p7.lambda_bq),
                    mu_redundancy=float(p7.mu_redundancy),
                    conditional_redundancy_enabled=bool(p7.conditional_redundancy_enabled),
                )
                conditional_redundancy_ms += float((time.perf_counter() - t_cr) * 1000.0)
                gain = float(terms.get("final_gain", 0.0))
                if gain > best_gain:
                    best_gain = float(gain)
                    best_atom = atom
                    best_terms = dict(terms)
            if best_atom is None:
                break
            selected.append(best_atom)
            selected_token_count += int(best_atom.token_count)
            selected_query_tokens.update(best_atom.normalized_token_set)
            selected_bridge_entities.update(best_atom.entity_set)
            selected_anchor_coverage.update(set(best_atom.connected_anchor_ids))
            selected_corridor_paths.update(set(best_atom.corridor_path_ids))
            if best_atom.source_key:
                selected_sources.add(str(best_atom.source_key))
            answerability_gain_total += float(best_terms.get("A", 0.0))
            decay_gain_total += float(best_terms.get("D", 0.0))
            corridor_gain_total += float(best_terms.get("Bq", 0.0))
            bq_eff_gain_total += float(best_terms.get("Bq_eff", 0.0))
            bridge_gain_total += float(best_terms.get("B", 0.0))
            redundancy_penalty_total += float(best_terms.get("R", 0.0))
            conditional_redundancy_penalty_total += float(best_terms.get("Rq", 0.0))
            objective_gain_total += float(best_terms.get("final_gain", best_gain))
            selected_feature_breakdown.append(
                {
                    "node_id": str(best_atom.source_id),
                    "atom_id": str(best_atom.atom_id),
                    "A": float(best_terms.get("A", 0.0)),
                    "D": float(best_terms.get("D", 0.0)),
                    "Bq": float(best_terms.get("Bq", 0.0)),
                    "Bq_eff": float(best_terms.get("Bq_eff", 0.0)),
                    "R": float(best_terms.get("R", 0.0)),
                    "Rq": float(best_terms.get("Rq", 0.0)),
                    "final_gain": float(best_terms.get("final_gain", best_gain)),
                    "is_corridor_node": bool(best_atom.is_corridor_node),
                    "unit_type": "single_atom",
                    "unit_source_ids": [str(best_atom.source_id)],
                }
            )

    # Anti-heuristic mainline contract:
    # Phase II performs feature extraction only; no second-stage refinement/pruning.
    phase2_drop_count = 0
    phase2_drop_reasons: List[str] = []
    _stage_end(
        "marginal_selection",
        t0,
        num_nodes=int(local_graph.number_of_nodes()),
        num_edges=int(local_graph.number_of_edges()),
        num_candidates_in=int(len(atoms)),
        num_candidates_out=int(len(selected)),
    )
    stage_ms["conditional_redundancy"] = float(conditional_redundancy_ms)
    stage_ms["chain_unit_selection"] = (
        float(stage_ms.get("marginal_selection", 0.0))
        if str(p7.objective_mode).strip().lower() == "chain_unit" and bool(p7.chain_unit_enabled)
        else 0.0
    )
    chain_unit_diag["num_selected_units"] = int(len(selected_units) if selected_units else len(selected))
    chain_unit_diag["num_selected_atoms"] = int(len(selected))
    stage_rows.append(
        {
            "query_id": str(getattr(sample, "qid", "") or ""),
            "stage": "conditional_redundancy",
            "elapsed_ms": float(conditional_redundancy_ms),
            "num_nodes": int(local_graph.number_of_nodes()),
            "num_edges": int(local_graph.number_of_edges()),
            "num_candidates_in": int(len(atoms)),
            "num_candidates_out": int(len(selected)),
        }
    )
    stage_rows.append(
        {
            "query_id": str(getattr(sample, "qid", "") or ""),
            "stage": "chain_unit_selection",
            "elapsed_ms": float(stage_ms.get("chain_unit_selection", 0.0)),
            "num_nodes": int(local_graph.number_of_nodes()),
            "num_edges": int(local_graph.number_of_edges()),
            "num_candidates_in": int(len(chain_units)),
            "num_candidates_out": int(len(selected_units)),
        }
    )

    # deterministic rendering order (selection invariant)
    t0 = _stage_begin()
    selected_ordered = _deterministic_render_order(selected)
    _stage_end(
        "rendering",
        t0,
        num_nodes=0,
        num_edges=0,
        num_candidates_in=int(len(selected)),
        num_candidates_out=int(len(selected_ordered)),
    )

    stage_ms["total_retrieval"] = float((time.perf_counter() - t_total) * 1000.0)
    stage_rows.append(
        {
            "query_id": str(getattr(sample, "qid", "") or ""),
            "stage": "total_retrieval",
            "elapsed_ms": float(stage_ms["total_retrieval"]),
            "num_nodes": int(local_graph.number_of_nodes()),
            "num_edges": int(local_graph.number_of_edges()),
            "num_candidates_in": int(len(atoms)),
            "num_candidates_out": int(len(selected_ordered)),
        }
    )

    candidate_ids = [a.source_id for a in atoms]
    candidate_texts = [a.text for a in atoms]
    selected_ids = [a.source_id for a in selected_ordered]
    selected_texts = [a.text for a in selected_ordered]
    phase1_seed_ids = [str(row.get("source_id", "")) for row in phase1_seed_rows]
    phase1_seed_texts = [str(row.get("text", "") or "") for row in phase1_seed_rows]
    phase1_seed_node_types = [str(row.get("node_type", "")) for row in phase1_seed_rows]
    phase1_seed_scores = [float(row.get("score", 0.0) or 0.0) for row in phase1_seed_rows]
    phase2_candidate_node_types = [_node_type(local_graph, a.atom_id) for a in atoms]
    phase2_candidate_scores = [float(0.55 * a.answerability_base + 0.45 * a.bridge_base) for a in atoms]
    selected_node_types = [_node_type(local_graph, a.atom_id) for a in selected_ordered]
    selected_scores = [float(0.55 * a.answerability_base + 0.45 * a.bridge_base) for a in selected_ordered]

    gold_unit_ids: Set[str] = set()
    for title, sent_idx in list(getattr(sample, "supporting_facts", []) or []):
        t = str(title or "").strip()
        idx = _safe_int(sent_idx, 0)
        if t:
            gold_unit_ids.add(f"{t}::{idx}")
    phase1_gold_candidate_ids = [a.source_id for a in atoms if a.source_id in gold_unit_ids]
    corridor_gold_candidate_ids = [a.source_id for a in atoms if (a.source_id in gold_unit_ids and bool(a.is_corridor_node))]
    selected_gold_candidate_ids = [a.source_id for a in selected_ordered if a.source_id in gold_unit_ids]
    chain_feasibility_diag, chain_unit_gold_eval_only = _chain_feasibility_diagnostics(
        atoms=atoms,
        units=chain_units,
        gold_unit_ids=gold_unit_ids,
        selected_ids=set(selected_ids),
        max_selected_atoms=int(max_selected_atoms_eff),
        max_context_tokens=int(max_context_tokens_eff),
    )
    selected_pair_unit_rate = float(
        sum(1 for u in selected_units if len(u.atoms) > 1) / float(max(1, len(selected_units)))
    ) if selected_units else 0.0
    selected_explicit_transition_unit_rate = float(
        sum(1 for u in selected_units if u.unit_type == "explicit_transition_pair") / float(max(1, len(selected_units)))
    ) if selected_units else 0.0
    selected_same_title_unit_rate = float(
        sum(1 for u in selected_units if u.unit_type == "same_title_pair") / float(max(1, len(selected_units)))
    ) if selected_units else 0.0
    selected_same_carrier_unit_rate = float(
        sum(1 for u in selected_units if u.unit_type == "same_carrier_pair") / float(max(1, len(selected_units)))
    ) if selected_units else 0.0
    selected_chain_units = [
        {
            "unit_id": str(unit.unit_id),
            "unit_type": str(unit.unit_type),
            "source_ids": list(unit.source_ids),
            "token_count": int(unit.token_count),
            "base_score": float(unit.base_score),
        }
        for unit in selected_units
    ]
    source_gold_hit_rates_eval_only: Dict[str, float] = {}
    source_gold_hit_any_eval_only: Dict[str, bool] = {}
    for source_name in SOURCE_ORDER:
        source_atoms = [a for a in atoms if source_name in set(a.source_tags or set())]
        source_gold_atoms = [a for a in source_atoms if a.source_id in gold_unit_ids]
        source_gold_hit_rates_eval_only[source_name] = (
            float(len(source_gold_atoms) / float(len(source_atoms))) if source_atoms else 0.0
        )
        source_gold_hit_any_eval_only[source_name] = bool(source_gold_atoms)
    source_balance_diag["source_gold_hit_rates_eval_only"] = dict(source_gold_hit_rates_eval_only)
    source_balance_diag["source_gold_hit_any_eval_only"] = dict(source_gold_hit_any_eval_only)

    ranked_by_bq = sorted(
        list(atoms or []),
        key=lambda a: (float(a.corridor_score), float(a.answerability_base), str(a.source_id)),
        reverse=True,
    )
    ranked_by_a = sorted(
        list(atoms or []),
        key=lambda a: (float(a.answerability_base), float(a.semantic_score), str(a.source_id)),
        reverse=True,
    )
    ranked_by_final = []
    for atom in list(atoms or []):
        terms = _objective_delta(
            cand=atom,
            selected=[],
            selected_query_tokens=set(),
            selected_bridge_entities=set(),
            selected_anchor_coverage=set(),
            selected_corridor_paths=set(),
            selected_sources=set(),
            objective_mode=str(p7.objective_mode),
            lambda_bridge=float(p7.lambda_bridge),
            lambda_decay=float(p7.lambda_decay),
            lambda_bq=float(p7.lambda_bq),
            mu_redundancy=float(p7.mu_redundancy),
            conditional_redundancy_enabled=bool(p7.conditional_redundancy_enabled),
        )
        ranked_by_final.append((atom, float(terms.get("final_gain", 0.0))))
    ranked_by_final.sort(key=lambda x: (float(x[1]), float(x[0].corridor_score), str(x[0].source_id)), reverse=True)

    gold_bq_scores = [float(a.corridor_score) for a in atoms if a.source_id in gold_unit_ids]
    distractor_bq_scores = [float(a.corridor_score) for a in atoms if a.source_id not in gold_unit_ids]
    gold_best_rank_by_bq = None
    for i, atom in enumerate(ranked_by_bq, start=1):
        if atom.source_id in gold_unit_ids:
            gold_best_rank_by_bq = int(i)
            break
    gold_best_rank_by_final = None
    for i, (atom, _score) in enumerate(ranked_by_final, start=1):
        if atom.source_id in gold_unit_ids:
            gold_best_rank_by_final = int(i)
            break
    corridor_gold_diag_eval_only = {
        "num_gold_candidates_in_phase1": int(len(phase1_gold_candidate_ids)),
        "num_gold_candidates_on_corridor": int(len(corridor_gold_candidate_ids)),
        "num_gold_candidates_selected": int(len(selected_gold_candidate_ids)),
        "gold_best_rank_by_A": None,
        "gold_best_rank_by_Bq": int(gold_best_rank_by_bq) if gold_best_rank_by_bq is not None else None,
        "gold_best_rank_by_final_score": int(gold_best_rank_by_final) if gold_best_rank_by_final is not None else None,
        "distractor_avg_Bq": float(_mean(distractor_bq_scores)) if distractor_bq_scores else None,
        "gold_avg_Bq": float(_mean(gold_bq_scores)) if gold_bq_scores else None,
    }
    for i, atom in enumerate(ranked_by_a, start=1):
        if atom.source_id in gold_unit_ids:
            corridor_gold_diag_eval_only["gold_best_rank_by_A"] = int(i)
            break

    candidate_sf = _support_metrics_eval_only(sample, candidate_ids, candidate_texts, graph_mode=graph_mode)
    selected_sf = _support_metrics_eval_only(sample, selected_ids, selected_texts, graph_mode=graph_mode)
    rendered_sf = _support_metrics_eval_only(sample, selected_ids, selected_texts, graph_mode=graph_mode)
    selected_avg_a = float(_mean([x.get("A", 0.0) for x in selected_feature_breakdown]))
    selected_avg_bq = float(_mean([x.get("Bq", 0.0) for x in selected_feature_breakdown]))
    selected_avg_bq_eff = float(_mean([x.get("Bq_eff", 0.0) for x in selected_feature_breakdown]))
    selected_avg_r = float(_mean([x.get("R", 0.0) for x in selected_feature_breakdown]))
    low_a_high_bq_selected_count = int(
        sum(
            1
            for x in selected_feature_breakdown
            if _safe_float(x.get("A", 0.0), 0.0) < 0.25
            and _safe_float(x.get("Bq", 0.0), 0.0) > 0.75
        )
    )

    bottleneck_flags = []
    if stage_ms.get("total_retrieval", 0.0) > float(p7.bottleneck_total_retrieval_ms):
        bottleneck_flags.append("slow_total_retrieval")
    if stage_ms.get("graph_flow", 0.0) > float(p7.bottleneck_graph_flow_ms):
        bottleneck_flags.append("slow_graph_flow")
    if stage_ms.get("local_graph_build", 0.0) > float(p7.bottleneck_local_graph_build_ms):
        bottleneck_flags.append("slow_local_graph_build")
    if stage_ms.get("feature_extraction", 0.0) > float(p7.bottleneck_feature_extraction_ms):
        bottleneck_flags.append("slow_feature_extraction")
    if stage_ms.get("marginal_selection", 0.0) > float(p7.bottleneck_marginal_selection_ms):
        bottleneck_flags.append("slow_marginal_selection")
    if stage_ms.get("corridor_extraction", 0.0) > 1000.0:
        bottleneck_flags.append("slow_corridor_extraction")
    if stage_ms.get("anchor_distance_bfs", 0.0) > 500.0:
        bottleneck_flags.append("slow_anchor_distance_bfs")
    if int(local_graph.number_of_nodes()) > int(p7.bottleneck_local_graph_nodes):
        bottleneck_flags.append("large_local_graph_nodes")
    if int(local_graph.number_of_edges()) > int(p7.bottleneck_local_graph_edges):
        bottleneck_flags.append("large_local_graph_edges")
    if int(len(atoms)) > int(p7.bottleneck_candidate_atoms):
        bottleneck_flags.append("too_many_candidates")
    if not atoms:
        bottleneck_flags.append("empty_candidate_pool")
    if not selected_ordered:
        bottleneck_flags.append("empty_selected_atoms")

    cand_sf_recall = candidate_sf.get("recall")
    sel_sf_recall = selected_sf.get("recall")
    empty_candidate_reason = None
    if not atoms:
        empty_candidate_reason = "NO_SENTENCE_NODES" if int(index_summary["phase7_index_summary"]["num_sentence_nodes"]) <= 0 else "FLOW_RETURNED_NO_SENTENCE_CANDIDATES"
    elif not selected_ordered:
        if int(budget_filtered_candidates) > 0:
            empty_candidate_reason = "ALL_CANDIDATES_FILTERED_BY_TOKEN_BUDGET"
        else:
            empty_candidate_reason = "NO_POSITIVE_MARGINAL_GAIN"

    if isinstance(cand_sf_recall, float) and cand_sf_recall < 0.05:
        bottleneck_flags.append("low_candidate_sf_recall_eval_only")
    if isinstance(cand_sf_recall, float) and isinstance(sel_sf_recall, float):
        if cand_sf_recall >= 0.40 and (cand_sf_recall - sel_sf_recall) >= 0.20:
            bottleneck_flags.append("candidate_to_selected_sf_collapse_eval_only")

    trace = {
        "query_id": str(getattr(sample, "qid", "") or ""),
        "question": str(getattr(sample, "question", "") or ""),
        "phase7_variant": str(p7.variant),
        "phase7_enable_phase2_refinement": bool(p7.enable_phase2_refinement),
        "phase7_objective_mode": str(p7.objective_mode),
        "phase7_lambda_bridge": float(p7.lambda_bridge),
        "phase7_lambda_decay": float(p7.lambda_decay),
        "phase7_lambda_bq": float(p7.lambda_bq),
        "phase7_mu_redundancy": float(p7.mu_redundancy),
        "phase7_conditional_redundancy_enabled": bool(p7.conditional_redundancy_enabled),
        "phase7_corridor_enabled": bool(p7.corridor_enabled),
        "phase7_anchor_decay_enabled": bool(p7.anchor_decay_enabled),
        "phase7_source_balanced_proposal_enabled": bool(p7.source_balanced_proposal_enabled),
        "phase7_source_balanced_candidate_top_m": int(p7.source_balanced_candidate_top_m),
        "phase7_source_balanced_fill_remaining": bool(p7.source_balanced_fill_remaining),
        "phase7_source_quota_semantic": int(p7.source_quota_semantic),
        "phase7_source_quota_entity_title": int(p7.source_quota_entity_title),
        "phase7_source_quota_graph_flow": int(p7.source_quota_graph_flow),
        "phase7_source_quota_anchor_neighborhood": int(p7.source_quota_anchor_neighborhood),
        "phase8_pamae_enabled": bool(phase8_pamae_enabled),
        "phase8_pamae_proposal_mode": str(getattr(cfg, "phase8_pamae_proposal_mode", "entity_seed_refine") or "entity_seed_refine"),
        "phase8_diagnostics": dict((phase8_result.diagnostics if phase8_result is not None else {}) or {}),
        "phase8_chunk_medoid_enabled": bool(phase8_chunk_enabled),
        "phase8_chunk_medoid_proposal_mode": str(getattr(cfg, "phase8_chunk_medoid_proposal_mode", "carrier_seed_bridge_refine") or "carrier_seed_bridge_refine"),
        "phase8_chunk_diagnostics": dict((phase8_chunk_result.diagnostics if phase8_chunk_result is not None else {}) or {}),
        "phase7_chain_unit_enabled": bool(p7.chain_unit_enabled),
        "phase7_chain_unit_max_pair_units": int(p7.chain_unit_max_pair_units),
        "phase7_chain_unit_use_same_title": bool(p7.chain_unit_use_same_title),
        "phase7_chain_unit_use_explicit_transition": bool(p7.chain_unit_use_explicit_transition),
        "phase7_chain_unit_use_same_carrier": bool(p7.chain_unit_use_same_carrier),
        "phase7_chain_unit_use_shared_entity": bool(p7.chain_unit_use_shared_entity),
        "phase7_chain_unit_max_unit_size": int(p7.chain_unit_max_unit_size),
        "num_query_anchors": int(len(anchor_nodes)),
        "num_semantic_anchor_atoms": int(len(sem_chunks)),
        "num_phase1_seeds": int(len(phase1_seed_ids)),
        "num_phase1_candidates": int(len(atoms)),
        "num_phase2_candidates": int(len(atoms) - int(phase2_drop_count)),
        "phase2_drop_count": int(phase2_drop_count),
        "phase2_drop_reasons": list(phase2_drop_reasons),
        "num_local_graph_nodes": int(local_graph.number_of_nodes()),
        "num_local_graph_edges": int(local_graph.number_of_edges()),
        "phase7_index_has_sentence_nodes": bool(int(index_summary["phase7_index_summary"]["num_sentence_nodes"]) > 0),
        "num_sentence_nodes_total": int(index_summary["phase7_index_summary"]["num_sentence_nodes"]),
        "num_candidate_atoms_before_truncation": int(num_candidate_atoms_before_truncation),
        "num_candidate_atoms_after_truncation": int(len(atoms)),
        "num_candidate_atoms": int(len(atoms)),
        "num_selected_atoms": int(len(selected_ordered)),
        "selected_token_count": int(sum(a.token_count for a in selected_ordered)),
        "rendered_context_tokens": int(sum(a.token_count for a in selected_ordered)),
        "empty_candidate_reason": empty_candidate_reason,
        "unit_consistency": dict(unit_consistency),
        "candidate_sf_recall_eval_only": candidate_sf.get("recall"),
        "selected_sf_recall_eval_only": selected_sf.get("recall"),
        "rendered_sf_recall_eval_only": rendered_sf.get("recall"),
        "phase1_gold_hit_eval_only": bool((candidate_sf.get("recall") or 0.0) > 0.0),
        "phase2_gold_hit_eval_only": bool((candidate_sf.get("recall") or 0.0) > 0.0),
        "selected_gold_hit_eval_only": bool((selected_sf.get("recall") or 0.0) > 0.0),
        "rendered_gold_hit_eval_only": bool((rendered_sf.get("recall") or 0.0) > 0.0),
        "candidate_sf_precision_eval_only": candidate_sf.get("precision"),
        "selected_sf_precision_eval_only": selected_sf.get("precision"),
        "rendered_sf_precision_eval_only": rendered_sf.get("precision"),
        "selected_to_rendered_match": None,
        "answerability_gain_total": float(answerability_gain_total),
        "decay_gain_total": float(decay_gain_total),
        "corridor_gain_total": float(corridor_gain_total),
        "bq_eff_gain_total": float(bq_eff_gain_total),
        "bridge_gain_total": float(bridge_gain_total),
        "redundancy_penalty_total": float(redundancy_penalty_total),
        "conditional_redundancy_penalty_total": float(conditional_redundancy_penalty_total),
        "final_objective_gain_total": float(objective_gain_total),
        "selected_avg_A": float(selected_avg_a),
        "selected_avg_Bq": float(selected_avg_bq),
        "selected_avg_Bq_eff": float(selected_avg_bq_eff),
        "selected_avg_R": float(selected_avg_r),
        "low_A_high_Bq_selected_count": int(low_a_high_bq_selected_count),
        "selected_atom_ids": [a.atom_id for a in selected_ordered],
        "selected_titles": [a.title for a in selected_ordered],
        "phase7_corridor_diagnostics": dict(corridor_diag),
        "phase7_source_balance_diagnostics": dict(source_balance_diag),
        "candidate_chain_feasibility": dict(chain_feasibility_diag),
        "candidate_gold_partial": bool(chain_feasibility_diag.get("candidate_gold_partial", False)),
        "candidate_gold_full": bool(chain_feasibility_diag.get("candidate_gold_full", False)),
        "candidate_gold_recall": float(chain_feasibility_diag.get("candidate_gold_recall", 0.0) or 0.0),
        "candidate_graph_gold_connected": bool(chain_feasibility_diag.get("candidate_graph_gold_connected", False)),
        "minimal_gold_chain_atoms": int(chain_feasibility_diag.get("minimal_gold_chain_atoms", 0) or 0),
        "minimal_gold_chain_tokens": int(chain_feasibility_diag.get("minimal_gold_chain_tokens", 0) or 0),
        "budget_feasible_gold_chain": bool(chain_feasibility_diag.get("budget_feasible_gold_chain", False)),
        "chain_unit_oracle_feasible": bool(chain_feasibility_diag.get("chain_unit_oracle_feasible", False)),
        "carrier_feasible_but_atom_failed": bool(chain_feasibility_diag.get("carrier_feasible_but_atom_failed", False)),
        "chain_unit_diagnostics": dict(chain_unit_diag),
        "chain_unit_gold_eval_only": dict(chain_unit_gold_eval_only),
        "gold_unit_partial": bool(chain_unit_gold_eval_only.get("gold_unit_partial", False)),
        "gold_unit_full": bool(chain_unit_gold_eval_only.get("gold_unit_full", False)),
        "gold_units_selected": int(chain_unit_gold_eval_only.get("gold_units_selected", 0) or 0),
        "selected_chain_units": list(selected_chain_units),
        "selected_pair_unit_rate": float(selected_pair_unit_rate),
        "selected_explicit_transition_unit_rate": float(selected_explicit_transition_unit_rate),
        "selected_same_title_unit_rate": float(selected_same_title_unit_rate),
        "selected_same_carrier_unit_rate": float(selected_same_carrier_unit_rate),
        "corridor_anchor_nodes": list(corridor_anchor_rows),
        "corridor_seed_nodes": list(corridor_seed_rows),
        "selected_evidence_feature_breakdown": list(selected_feature_breakdown),
        "corridor_gold_diagnostics_eval_only": dict(corridor_gold_diag_eval_only),
        "corridor_gold_hit_eval_only": bool(int(corridor_gold_diag_eval_only.get("num_gold_candidates_on_corridor", 0) or 0) > 0),
        "stage_ms": {
            "query_embedding": float(stage_ms.get("query_embedding", 0.0)),
            "index_validation": float(stage_ms.get("index_validation", 0.0)),
            "anchor_extraction": float(stage_ms.get("anchor_extraction", 0.0)),
            "semantic_anchor_retrieval": float(stage_ms.get("semantic_anchor_retrieval", 0.0)),
            "local_graph_build": float(stage_ms.get("local_graph_build", 0.0)),
            "graph_flow": float(stage_ms.get("graph_flow", 0.0)),
            "phase8_pamae_proposal": float(stage_ms.get("phase8_pamae_proposal", 0.0)),
            "phase8_chunk_medoid_proposal": float(stage_ms.get("phase8_chunk_medoid_proposal", 0.0)),
            "phase8_local_graph_build": float(stage_ms.get("phase8_local_graph_build", 0.0)),
            "phase8_graph_flow": float(stage_ms.get("phase8_graph_flow", 0.0)),
            "source_balanced_union": float(stage_ms.get("source_balanced_union", 0.0)),
            "entity_title_retrieval": float(stage_ms.get("entity_title_retrieval", 0.0)),
            "anchor_neighborhood": float(stage_ms.get("anchor_neighborhood", 0.0)),
            "candidate_truncation": float(stage_ms.get("candidate_truncation", 0.0)),
            "chain_unit_build": float(stage_ms.get("chain_unit_build", 0.0)),
            "feature_extraction": float(stage_ms.get("feature_extraction", 0.0)),
            "corridor_extraction": float(stage_ms.get("corridor_extraction", 0.0)),
            "anchor_distance_bfs": float(stage_ms.get("anchor_distance_bfs", 0.0)),
            "bq_feature_scoring": float(stage_ms.get("bq_feature_scoring", 0.0)),
            "conditional_redundancy": float(stage_ms.get("conditional_redundancy", 0.0)),
            "marginal_selection": float(stage_ms.get("marginal_selection", 0.0)),
            "chain_unit_selection": float(stage_ms.get("chain_unit_selection", 0.0)),
            "rendering": float(stage_ms.get("rendering", 0.0)),
            "total_retrieval": float(stage_ms.get("total_retrieval", 0.0)),
        },
        "bottleneck_flags": list(bottleneck_flags),
    }
    logger.log_query(query_trace=trace, stage_rows=stage_rows)

    diagnostics = {
        "graph_mode": str(graph_mode),
        "selected_unit_type": (
            "sentence"
            if selected_node_types and all(str(t) == "sentence" for t in selected_node_types)
            else ("mixed" if selected_node_types else "sentence")
        ),
        "phase7_enabled": True,
        "phase7_method": "phase7_evidence_flow",
        "phase7_atom_unit": str(p7.atom_unit),
        "phase7_carrier_chunk_size_sentences": int(p7.carrier_chunk_size_sentences),
        "phase7_carrier_chunk_stride_sentences": int(p7.carrier_chunk_stride_sentences),
        "phase7_semantic_anchor_top_t": int(p7.semantic_anchor_top_t),
        "phase7_candidate_top_m": int(p7.candidate_top_m),
        "phase7_flow_alpha": float(p7.flow_alpha),
        "phase7_max_selected_atoms": int(p7.max_selected_atoms),
        "phase7_max_context_tokens": int(p7.max_context_tokens),
        "phase7_query_trace_path": str(Path(out_dir) / "phase7_query_trace.jsonl"),
        "phase7_stage_timing_path": str(Path(out_dir) / "phase7_stage_timing.jsonl"),
        "phase7_variant": str(p7.variant),
        "phase7_enable_phase2_refinement": bool(p7.enable_phase2_refinement),
        "phase7_objective_mode": str(p7.objective_mode),
        "phase7_lambda_bridge": float(p7.lambda_bridge),
        "phase7_lambda_decay": float(p7.lambda_decay),
        "phase7_lambda_bq": float(p7.lambda_bq),
        "phase7_mu_redundancy": float(p7.mu_redundancy),
        "phase7_conditional_redundancy_enabled": bool(p7.conditional_redundancy_enabled),
        "phase7_corridor_enabled": bool(p7.corridor_enabled),
        "phase7_anchor_decay_enabled": bool(p7.anchor_decay_enabled),
        "phase7_source_balanced_proposal_enabled": bool(p7.source_balanced_proposal_enabled),
        "phase7_source_balanced_candidate_top_m": int(p7.source_balanced_candidate_top_m),
        "phase7_source_balanced_fill_remaining": bool(p7.source_balanced_fill_remaining),
        "phase7_source_quota_semantic": int(p7.source_quota_semantic),
        "phase7_source_quota_entity_title": int(p7.source_quota_entity_title),
        "phase7_source_quota_graph_flow": int(p7.source_quota_graph_flow),
        "phase7_source_quota_anchor_neighborhood": int(p7.source_quota_anchor_neighborhood),
        "phase8_pamae_enabled": bool(phase8_pamae_enabled),
        "phase8_pamae_proposal_mode": str(getattr(cfg, "phase8_pamae_proposal_mode", "entity_seed_refine") or "entity_seed_refine"),
        "phase8_diagnostics": dict((phase8_result.diagnostics if phase8_result is not None else {}) or {}),
        "phase8_query_trace_path": str(Path(out_dir) / "phase8_query_trace.jsonl"),
        "phase8_stage_timing_path": str(Path(out_dir) / "phase8_stage_timing.jsonl"),
        "phase8_seed_trace_path": str(Path(out_dir) / "phase8_seed_trace.jsonl"),
        "phase8_evidence_trace_path": str(Path(out_dir) / "phase8_evidence_trace.jsonl"),
        "phase8_chunk_medoid_enabled": bool(phase8_chunk_enabled),
        "phase8_chunk_medoid_proposal_mode": str(getattr(cfg, "phase8_chunk_medoid_proposal_mode", "carrier_seed_bridge_refine") or "carrier_seed_bridge_refine"),
        "phase8_chunk_diagnostics": dict((phase8_chunk_result.diagnostics if phase8_chunk_result is not None else {}) or {}),
        "phase8_chunk_query_trace_path": str(Path(out_dir) / "phase8_chunk_query_trace.jsonl"),
        "phase8_chunk_stage_timing_path": str(Path(out_dir) / "phase8_chunk_stage_timing.jsonl"),
        "phase8_chunk_seed_trace_path": str(Path(out_dir) / "phase8_chunk_seed_trace.jsonl"),
        "phase8_chunk_evidence_trace_path": str(Path(out_dir) / "phase8_chunk_evidence_trace.jsonl"),
        "phase7_chain_unit_enabled": bool(p7.chain_unit_enabled),
        "phase7_chain_unit_max_pair_units": int(p7.chain_unit_max_pair_units),
        "phase7_chain_unit_use_same_title": bool(p7.chain_unit_use_same_title),
        "phase7_chain_unit_use_explicit_transition": bool(p7.chain_unit_use_explicit_transition),
        "phase7_chain_unit_use_same_carrier": bool(p7.chain_unit_use_same_carrier),
        "phase7_chain_unit_use_shared_entity": bool(p7.chain_unit_use_shared_entity),
        "phase7_chain_unit_max_unit_size": int(p7.chain_unit_max_unit_size),
        "phase7_candidate_top_m_effective": int(candidate_top_m_eff),
        "phase7_max_selected_atoms_effective": int(max_selected_atoms_eff),
        "phase7_max_context_tokens_effective": int(max_context_tokens_eff),
        "phase7_diagnostics_enabled": bool(p7.diagnostics_enabled),
        "phase7_diagnostics_dump_text": bool(p7.diagnostics_dump_text),
        "phase7_diagnostics_dump_context": bool(p7.diagnostics_dump_context),
        "phase7_diagnostics_dump_scores": bool(p7.diagnostics_dump_scores),
        "phase7_diagnostics_fail_on_unit_mismatch": bool(p7.diagnostics_fail_on_unit_mismatch),
        "latency_breakdown_ms": {
            "query_embed_ms": float(stage_ms.get("query_embedding", 0.0)),
            "semantic_lookup_chunk_ms": float(stage_ms.get("semantic_anchor_retrieval", 0.0)),
            "phase8_pamae_proposal_ms": float(stage_ms.get("phase8_pamae_proposal", 0.0)),
            "phase8_chunk_medoid_proposal_ms": float(stage_ms.get("phase8_chunk_medoid_proposal", 0.0)),
            "phase8_local_graph_build_ms": float(stage_ms.get("phase8_local_graph_build", 0.0)),
            "phase8_graph_flow_ms": float(stage_ms.get("phase8_graph_flow", 0.0)),
            "source_balanced_union_ms": float(stage_ms.get("source_balanced_union", 0.0)),
            "entity_title_retrieval_ms": float(stage_ms.get("entity_title_retrieval", 0.0)),
            "anchor_neighborhood_ms": float(stage_ms.get("anchor_neighborhood", 0.0)),
            "proposal_subgraph_build_ms": float(stage_ms.get("local_graph_build", 0.0)),
            "phase1_ppr_ms": float(stage_ms.get("graph_flow", 0.0)),
            "chain_unit_build_ms": float(stage_ms.get("chain_unit_build", 0.0)),
            "feature_extraction_ms": float(stage_ms.get("feature_extraction", 0.0)),
            "corridor_extraction_ms": float(stage_ms.get("corridor_extraction", 0.0)),
            "anchor_distance_bfs_ms": float(stage_ms.get("anchor_distance_bfs", 0.0)),
            "bq_feature_scoring_ms": float(stage_ms.get("bq_feature_scoring", 0.0)),
            "conditional_redundancy_ms": float(stage_ms.get("conditional_redundancy", 0.0)),
            "marginal_selection_ms": float(stage_ms.get("marginal_selection", 0.0)),
            "chain_unit_selection_ms": float(stage_ms.get("chain_unit_selection", 0.0)),
            "render_ms": float(stage_ms.get("rendering", 0.0)),
            "retrieval_total_ms": float(stage_ms.get("total_retrieval", 0.0)),
        },
        "stage_ms": dict(trace.get("stage_ms", {})),
        "bottleneck_flags": list(bottleneck_flags),
        "candidate_sf_recall_eval_only": trace["candidate_sf_recall_eval_only"],
        "selected_sf_recall_eval_only": trace["selected_sf_recall_eval_only"],
        "rendered_sf_recall_eval_only": trace["rendered_sf_recall_eval_only"],
        "candidate_sf_precision_eval_only": trace["candidate_sf_precision_eval_only"],
        "selected_sf_precision_eval_only": trace["selected_sf_precision_eval_only"],
        "rendered_sf_precision_eval_only": trace["rendered_sf_precision_eval_only"],
        "answerability_gain_total": float(answerability_gain_total),
        "decay_gain_total": float(decay_gain_total),
        "corridor_gain_total": float(corridor_gain_total),
        "bq_eff_gain_total": float(bq_eff_gain_total),
        "bridge_gain_total": float(bridge_gain_total),
        "redundancy_penalty_total": float(redundancy_penalty_total),
        "conditional_redundancy_penalty_total": float(conditional_redundancy_penalty_total),
        "final_objective_gain_total": float(objective_gain_total),
        "selected_avg_A": float(selected_avg_a),
        "selected_avg_Bq": float(selected_avg_bq),
        "selected_avg_Bq_eff": float(selected_avg_bq_eff),
        "selected_avg_R": float(selected_avg_r),
        "low_A_high_Bq_selected_count": int(low_a_high_bq_selected_count),
        "phase7_corridor_diagnostics": dict(corridor_diag),
        "phase7_source_balance_diagnostics": dict(source_balance_diag),
        "candidate_chain_feasibility": dict(chain_feasibility_diag),
        "chain_unit_diagnostics": dict(chain_unit_diag),
        "chain_unit_gold_eval_only": dict(chain_unit_gold_eval_only),
        "selected_chain_units": list(selected_chain_units),
        "selected_pair_unit_rate": float(selected_pair_unit_rate),
        "selected_explicit_transition_unit_rate": float(selected_explicit_transition_unit_rate),
        "selected_same_title_unit_rate": float(selected_same_title_unit_rate),
        "selected_same_carrier_unit_rate": float(selected_same_carrier_unit_rate),
        "selected_evidence_feature_breakdown": list(selected_feature_breakdown),
        "corridor_gold_diagnostics_eval_only": dict(corridor_gold_diag_eval_only),
        "corridor_gold_hit_eval_only": bool(int(corridor_gold_diag_eval_only.get("num_gold_candidates_on_corridor", 0) or 0) > 0),
        "corridor_anchor_nodes": list(corridor_anchor_rows),
        "corridor_seed_nodes": list(corridor_seed_rows),
        "num_query_anchors": int(len(anchor_nodes)),
        "num_semantic_anchor_atoms": int(len(sem_chunks)),
        "num_local_graph_nodes": int(local_graph.number_of_nodes()),
        "num_local_graph_edges": int(local_graph.number_of_edges()),
        "num_candidate_atoms": int(len(atoms)),
        "num_selected_atoms": int(len(selected_ordered)),
        "selected_token_count": int(sum(a.token_count for a in selected_ordered)),
        "num_candidate_atoms_before_truncation": int(num_candidate_atoms_before_truncation),
        "num_candidate_atoms_after_truncation": int(len(atoms)),
        "num_phase1_candidates": int(len(atoms)),
        "num_phase2_candidates": int(len(atoms) - int(phase2_drop_count)),
        "phase2_drop_count": int(phase2_drop_count),
        "phase2_drop_reasons": list(phase2_drop_reasons),
        "phase7_index_has_sentence_nodes": bool(int(index_summary["phase7_index_summary"]["num_sentence_nodes"]) > 0),
        "num_sentence_nodes_total": int(index_summary["phase7_index_summary"]["num_sentence_nodes"]),
        "candidate_source_breakdown": dict(candidate_source_breakdown),
        "unit_consistency": dict(unit_consistency),
        "phase1_seed_ids": list(phase1_seed_ids),
        "phase1_seed_node_types": list(phase1_seed_node_types),
        "phase1_seed_scores": list(phase1_seed_scores),
        "phase1_seed_texts": list(phase1_seed_texts),
        "phase2_candidate_ids": list(candidate_ids),
        "phase2_candidate_node_types": list(phase2_candidate_node_types),
        "phase2_candidate_scores": list(phase2_candidate_scores),
        "phase2_candidate_texts": list(candidate_texts),
        "selected_evidence_ids": list(selected_ids),
        "selected_evidence_node_types": list(selected_node_types),
        "selected_evidence_scores": list(selected_scores),
        "selected_evidence_texts": list(selected_texts),
        "empty_candidate_reason": empty_candidate_reason,
        "selected_text_map": {a.source_id: a.text for a in selected_ordered},
        "phase7_candidate_atoms": [
            {
                "atom_id": str(a.atom_id),
                "source_id": str(a.source_id),
                "title": str(a.title),
                "text": str(a.text),
                "token_count": int(a.token_count),
                "semantic_score": float(a.semantic_score),
                "flow_score": float(a.flow_score),
                "answerability_base": float(a.answerability_base),
                "bridge_base": float(a.bridge_base),
                "anchor_decay_score": float(a.anchor_decay_score),
                "corridor_score": float(a.corridor_score),
                "corridor_anchor_coverage": float(a.corridor_anchor_coverage),
                "corridor_seed_coverage": float(a.corridor_seed_coverage),
                "corridor_path_closure": float(a.corridor_path_closure),
                "corridor_degree_contrib": float(a.corridor_degree_contrib),
                "corridor_path_count": int(a.corridor_path_count),
                "is_corridor_node": bool(a.is_corridor_node),
                "anchor_distance": float(a.anchor_distance),
                "anchor_reachability": float(a.anchor_reachability),
                "carrier_id": str(a.carrier_id),
                "flow_rank": int(a.flow_rank),
                "source_tags": sorted(str(x) for x in set(a.source_tags or set())),
                "proposal_source": str(a.proposal_source),
            }
            for a in atoms
        ],
        "phase7_selected_atoms": [
            {
                "atom_id": str(a.atom_id),
                "source_id": str(a.source_id),
                "title": str(a.title),
                "text": str(a.text),
                "token_count": int(a.token_count),
                "answerability_base": float(a.answerability_base),
                "bridge_base": float(a.bridge_base),
                "anchor_decay_score": float(a.anchor_decay_score),
                "corridor_score": float(a.corridor_score),
                "is_corridor_node": bool(a.is_corridor_node),
                "anchor_distance": float(a.anchor_distance),
                "carrier_id": str(a.carrier_id),
                "source_tags": sorted(str(x) for x in set(a.source_tags or set())),
                "proposal_source": str(a.proposal_source),
            }
            for a in selected_ordered
        ],
        "phase7_aux": {
            "query_terms": sorted(list(aux.get("query_tokens", set()) or set())),
            "anchor_terms": sorted(list(aux.get("anchor_tokens", set()) or set())),
            "relation_terms": sorted(list(aux.get("relation_terms", set()) or set())),
        },
    }

    return RetrievalResult(
        sample_id=str(getattr(sample, "qid", "") or ""),
        method="phase7_evidence_flow",
        anchors=[str(a) for a in anchor_nodes],
        seeds=[str(s) for s in sem_chunks],
        selected_nodes=[str(a.atom_id) for a in selected_ordered],
        selected_sentence_ids=[str(a.source_id) for a in selected_ordered],
        selected_sentences=[str(a.text) for a in selected_ordered],
        candidate_sentence_ids=[str(a.source_id) for a in atoms],
        candidate_sentences=[str(a.text) for a in atoms],
        corridors=[
            {
                "corridor_id": f"carrier::{idx}",
                "carrier_id": str(a.carrier_id),
                "sentence_ids": [str(a.source_id)],
                "score": float(a.flow_score),
            }
            for idx, a in enumerate(selected_ordered, start=1)
        ],
        anchor_results=[],
        diagnostics=diagnostics,
        latency_ms=float(stage_ms.get("total_retrieval", 0.0)),
    )
