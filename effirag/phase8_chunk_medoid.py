from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from .embedding import encode_texts, topk_cosine_similarity
from .phase8_pamae_entity_seeding import (
    _as_graph,
    _cfg,
    _cfg_bool,
    _clip01,
    _cosine,
    _mean,
    _node_type,
    _normalize_vector,
    _query_embedding,
    _safe_float,
    _safe_int,
    _semantic_state,
)
from .utils import content_tokens


@dataclass
class ChunkCandidate:
    chunk_id: str
    title: Optional[str]
    text: str
    carrier_id: Optional[str]
    embedding_id: Optional[str]
    query_relevance: float
    token_count: int
    linked_entities: list[str]
    source_tags: set[str]
    embedding: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


@dataclass
class ChunkMedoidSeed:
    chunk_id: str
    score: float
    sample_id: int
    refined_from: Optional[str]
    bridge_entities: list[str]
    diagnostics: dict


@dataclass
class ChunkMedoidProposalResult:
    chunk_universe: list[ChunkCandidate]
    initial_medoid_sets: list[list[ChunkMedoidSeed]]
    best_medoid_set: list[ChunkMedoidSeed]
    refined_medoid_set: list[ChunkMedoidSeed]
    evidence_candidate_ids: list[str]
    diagnostics: dict


_LAST_DIAGNOSTICS: Dict[str, Any] = {}


def get_last_phase8_chunk_diagnostics() -> Dict[str, Any]:
    return dict(_LAST_DIAGNOSTICS)


def _stable_seed(config: Any, query_embedding: Any = None) -> int:
    base = _safe_int(_cfg(config, "phase8_chunk_medoid_random_seed", 42), 42)
    query_key = str(_cfg(config, "_phase8_query_key", "") or "")
    if not query_key and query_embedding is not None:
        vec = _normalize_vector(query_embedding)
        if vec is not None:
            query_key = ",".join(f"{float(x):.4f}" for x in vec[: min(16, vec.size)])
    digest = hashlib.sha1(f"{base}:{query_key}".encode("utf-8")).hexdigest()[:8]
    return int((base + int(digest, 16)) % (2**32 - 1))


def _source_flags(config: Any) -> Dict[str, bool]:
    raw = _cfg(config, "phase8_chunk_universe_sources", None)
    if isinstance(raw, dict):
        return {
            "semantic": bool(raw.get("semantic", True)),
            "source_balanced": bool(raw.get("source_balanced", True)),
            "graph_flow": bool(raw.get("graph_flow", True)),
            "title_entity_lookup": bool(raw.get("title_entity_lookup", True)),
        }
    return {
        "semantic": _cfg_bool(config, "phase8_chunk_source_semantic", True),
        "source_balanced": _cfg_bool(config, "phase8_chunk_source_balanced", True),
        "graph_flow": _cfg_bool(config, "phase8_chunk_source_graph_flow", True),
        "title_entity_lookup": _cfg_bool(config, "phase8_chunk_source_title_entity_lookup", True),
    }


def _sentence_sort_key(graph: nx.Graph, node: str) -> Tuple[str, int, str]:
    data = graph.nodes[node] if node in graph else {}
    return (
        str(data.get("title", "") or ""),
        _safe_int(data.get("sent_idx", data.get("sentence_idx", 0)), 0),
        str(node),
    )


def _node_title(graph: nx.Graph, node: str) -> Optional[str]:
    if str(node) not in graph:
        return None
    title = str(graph.nodes[str(node)].get("title", "") or "").strip()
    return title or None


def _node_text(graph: nx.Graph, node: str) -> str:
    if str(node) not in graph:
        return ""
    data = graph.nodes[str(node)]
    text = str(data.get("text", "") or data.get("content", "") or data.get("name", "") or "")
    if text:
        return text
    if _node_type(graph, str(node)) in {"chunk", "passage", "document"}:
        parts = []
        for sent in _sentences_for_carrier(graph, str(node), limit=16):
            stext = str(graph.nodes[sent].get("text", "") or "")
            if stext:
                parts.append(stext)
        return " ".join(parts)
    return ""


def _token_count(text: str) -> int:
    return int(len([tok for tok in str(text or "").split() if tok.strip()]))


def _sentence_source_id(graph: nx.Graph, node: str) -> str:
    sid = str(graph.nodes[str(node)].get("sentence_id", "") or "").strip() if str(node) in graph else ""
    if sid:
        return sid
    title = str(_node_title(graph, str(node)) or "").strip()
    idx = _safe_int(graph.nodes[str(node)].get("sent_idx", 0), 0) if str(node) in graph else 0
    return f"{title}::{idx}" if title else str(node)


def _sentences_for_carrier(graph: nx.Graph, node: str, limit: int) -> List[str]:
    if str(node) not in graph:
        return []
    ntype = _node_type(graph, str(node))
    if ntype == "sentence":
        return [str(node)]
    out = [
        str(n)
        for n in graph.neighbors(str(node))
        if _node_type(graph, str(n)) == "sentence"
    ]
    out.sort(key=lambda n: _sentence_sort_key(graph, n))
    return out[: max(1, int(limit))]


def _carrier_for_sentence(graph: nx.Graph, node: str) -> Optional[str]:
    if str(node) not in graph:
        return None
    if _node_type(graph, str(node)) in {"chunk", "passage", "document"}:
        return str(node)
    carriers = [
        str(n)
        for n in graph.neighbors(str(node))
        if _node_type(graph, str(n)) in {"chunk", "passage", "document"}
    ]
    carriers.sort(key=lambda n: (_safe_int(graph.nodes[n].get("chunk_idx", 0), 0), str(n)))
    return carriers[0] if carriers else None


def _linked_entities(graph: nx.Graph, node: str, limit: int = 32) -> List[str]:
    if str(node) not in graph:
        return []
    out: Set[str] = set()
    frontier = [str(node)]
    if _node_type(graph, str(node)) in {"chunk", "passage", "document"}:
        frontier.extend(_sentences_for_carrier(graph, str(node), limit=12))
    for src in frontier:
        if src not in graph:
            continue
        for nbr in graph.neighbors(src):
            if _node_type(graph, str(nbr)) == "entity":
                out.add(str(nbr))
            if len(out) >= max(1, int(limit)):
                break
        if len(out) >= max(1, int(limit)):
            break
    return sorted(out)


def _chunk_embedding(chunk_id: str, semantic: Dict[str, Any]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    chunk_id_to_idx = dict(semantic.get("chunk_id_to_idx", {}) or {})
    embeddings = semantic.get("chunk_embeddings")
    if embeddings is None:
        return None, None
    cid = str(chunk_id or "")
    idx = _safe_int(chunk_id_to_idx.get(cid, -1), -1)
    if idx < 0:
        ids = list(semantic.get("chunk_ids", []) or [])
        try:
            idx = ids.index(cid)
        except Exception:
            idx = -1
    try:
        if idx < 0 or idx >= len(embeddings):
            return None, None
        vec = _normalize_vector(embeddings[idx])
    except Exception:
        return None, None
    if vec is None:
        return None, None
    return cid, vec


def _text_embedding(text: str, config: Any) -> Optional[np.ndarray]:
    encoded = encode_texts(
        texts=[str(text or "")],
        model_name=str(_cfg(config, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2"),
        batch_size=1,
        max_length=max(1, _safe_int(_cfg(config, "embedding_max_length", 320), 320)),
        max_chars=max(1, _safe_int(_cfg(config, "embedding_text_max_chars", 900), 900)),
        instruction="",
    )
    if not bool(encoded.get("ok", False)):
        return None
    vectors = list(encoded.get("vectors", []) or [])
    if not vectors:
        return None
    return _normalize_vector(vectors[0])


def _best_chunk_embedding(
    chunk_id: str,
    graph: nx.Graph,
    semantic: Dict[str, Any],
    config: Any,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    embedding_id, vec = _chunk_embedding(chunk_id, semantic)
    if vec is not None:
        return embedding_id, vec
    text = _node_text(graph, chunk_id)
    if text:
        vec = _text_embedding(text, config)
        if vec is not None:
            return f"{chunk_id}::text", vec
    return None, None


def _semantic_chunk_ids(graph: nx.Graph, semantic: Dict[str, Any]) -> List[str]:
    out = [
        str(cid)
        for cid in list(semantic.get("chunk_ids", []) or [])
        if str(cid) in graph and _node_type(graph, str(cid)) in {"chunk", "passage", "document", "sentence"}
    ]
    if out:
        return out
    return [
        str(node)
        for node in graph.nodes
        if _node_type(graph, str(node)) in {"chunk", "passage", "document"}
    ]


def _candidate_node_for_unit(graph: nx.Graph, node: str, unit: str) -> Optional[str]:
    sid = str(node or "")
    if not sid or sid not in graph:
        return None
    ntype = _node_type(graph, sid)
    if str(unit or "carrier").strip().lower() == "sentence":
        if ntype == "sentence":
            return sid
        sentences = _sentences_for_carrier(graph, sid, limit=1)
        return sentences[0] if sentences else None
    if ntype in {"chunk", "passage", "document"}:
        return sid
    if ntype == "sentence":
        return _carrier_for_sentence(graph, sid) or sid
    return None


def _add_candidate(
    rows: Dict[str, Dict[str, Any]],
    chunk_id: str,
    graph: nx.Graph,
    semantic: Dict[str, Any],
    config: Any,
    source: str,
    query_relevance: float = 0.0,
) -> None:
    cid = str(chunk_id or "")
    if not cid or cid not in graph:
        return
    if _node_type(graph, cid) not in {"chunk", "passage", "document", "sentence"}:
        return
    row = rows.setdefault(
        cid,
        {
            "chunk_id": cid,
            "query_relevance": 0.0,
            "source_tags": set(),
            "embedding_id": None,
            "embedding": None,
        },
    )
    row["query_relevance"] = max(float(row.get("query_relevance", 0.0) or 0.0), float(query_relevance))
    row["source_tags"].add(str(source))
    if row.get("embedding") is None:
        embedding_id, vec = _best_chunk_embedding(cid, graph, semantic, config)
        row["embedding_id"] = embedding_id
        row["embedding"] = vec


def _chunks_from_entities(graph: nx.Graph, entities: Iterable[str], limit_per_entity: int = 8) -> List[str]:
    out: List[str] = []
    seen = set()
    for entity in list(entities or []):
        eid = str(entity or "")
        if not eid or eid not in graph:
            continue
        local = []
        for nbr in graph.neighbors(eid):
            ntype = _node_type(graph, str(nbr))
            if ntype in {"chunk", "passage", "document"}:
                local.append(str(nbr))
            elif ntype == "sentence":
                carrier = _carrier_for_sentence(graph, str(nbr))
                local.append(str(carrier or nbr))
        local = [cid for cid in local if cid in graph]
        local.sort(key=lambda cid: (_safe_int(graph.nodes[cid].get("chunk_idx", 0), 0), str(cid)))
        for cid in local[: max(1, int(limit_per_entity))]:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


def build_query_chunk_universe(query, graph_index, embeddings, config) -> list[ChunkCandidate]:
    t0 = time.perf_counter()
    graph = _as_graph(graph_index)
    semantic = _semantic_state(graph_index, embeddings)
    qvec = _query_embedding(str(query or ""), embeddings, config)
    top_n = max(1, _safe_int(_cfg(config, "phase8_chunk_universe_top_n", 2000), 2000))
    min_n = max(0, _safe_int(_cfg(config, "phase8_chunk_universe_min_n", 200), 200))
    unit = str(_cfg(config, "phase8_chunk_universe_unit", "carrier") or "carrier").strip().lower()
    flags = _source_flags(config)
    rows: Dict[str, Dict[str, Any]] = {}
    counts = {
        "semantic": 0,
        "source_balanced": 0,
        "graph_flow": 0,
        "title_entity_lookup": 0,
    }

    chunk_ids = _semantic_chunk_ids(graph, semantic)
    if flags["semantic"] and qvec is not None and semantic.get("chunk_embeddings") is not None:
        semantic_ids = list(semantic.get("chunk_ids", []) or [])
        top_k = min(max(top_n, min_n), len(semantic_ids))
        if top_k > 0:
            try:
                idx, scores = topk_cosine_similarity(
                    query_vector=qvec,
                    matrix=semantic.get("chunk_embeddings"),
                    topn=top_k,
                    scan_batch_size=max(1, _safe_int(_cfg(config, "semantic_scan_batch_size", 8192), 8192)),
                )
                for i, score in zip(idx.tolist(), scores.tolist()):
                    if i < 0 or i >= len(semantic_ids):
                        continue
                    cid = _candidate_node_for_unit(graph, str(semantic_ids[i]), unit=unit)
                    if cid:
                        _add_candidate(
                            rows,
                            cid,
                            graph,
                            semantic,
                            config,
                            "semantic",
                            query_relevance=_clip01((float(score) + 1.0) / 2.0),
                        )
                counts["semantic"] = int(sum(1 for r in rows.values() if "semantic" in set(r.get("source_tags", set()))))
            except Exception:
                counts["semantic"] = 0

    if flags["source_balanced"] and isinstance(embeddings, dict):
        balanced_chunks: Set[str] = set()
        for key in ("semantic_chunks", "phase7_candidate_chunks", "candidate_chunks"):
            for raw in list(embeddings.get(key, []) or []):
                cid = _candidate_node_for_unit(graph, str(raw), unit=unit)
                if cid:
                    balanced_chunks.add(str(cid))
        for eid in list(embeddings.get("semantic_entities", []) or []) + list(embeddings.get("anchor_nodes", []) or []):
            for cid in _chunks_from_entities(graph, [str(eid)], limit_per_entity=4):
                cand_id = _candidate_node_for_unit(graph, cid, unit=unit)
                if cand_id:
                    balanced_chunks.add(str(cand_id))
        for cid in sorted(balanced_chunks):
            _add_candidate(rows, cid, graph, semantic, config, "source_balanced", query_relevance=0.25)
        counts["source_balanced"] = int(sum(1 for r in rows.values() if "source_balanced" in set(r.get("source_tags", set()))))

    flow_scores: Dict[str, float] = {}
    if isinstance(embeddings, dict):
        flow_scores = dict(embeddings.get("flow_scores", {}) or {})
    if flags["graph_flow"] and flow_scores:
        chunk_flow: Dict[str, float] = {}
        for node, score in flow_scores.items():
            sid = str(node)
            cid = _candidate_node_for_unit(graph, sid, unit=unit)
            if cid:
                chunk_flow[cid] = max(float(chunk_flow.get(cid, 0.0) or 0.0), _safe_float(score, 0.0))
            elif sid in graph and _node_type(graph, sid) == "entity":
                for linked in _chunks_from_entities(graph, [sid], limit_per_entity=4):
                    cid2 = _candidate_node_for_unit(graph, linked, unit=unit)
                    if cid2:
                        chunk_flow[cid2] = max(float(chunk_flow.get(cid2, 0.0) or 0.0), _safe_float(score, 0.0))
        ranked = sorted(chunk_flow.items(), key=lambda x: (float(x[1]), str(x[0])), reverse=True)
        max_flow = max([abs(float(v)) for _cid, v in ranked[:top_n]] or [1.0])
        for cid, score in ranked[:top_n]:
            _add_candidate(rows, cid, graph, semantic, config, "graph_flow", query_relevance=_clip01(float(score) / float(max_flow or 1.0)))
        counts["graph_flow"] = int(sum(1 for r in rows.values() if "graph_flow" in set(r.get("source_tags", set()))))

    query_tokens = set(content_tokens(str(query or "")))
    if flags["title_entity_lookup"] and query_tokens:
        matched_entities = []
        for node in graph.nodes:
            if _node_type(graph, str(node)) != "entity":
                continue
            data = graph.nodes[str(node)]
            text = " ".join(
                str(data.get(key, "") or "")
                for key in ("name", "title", "text", "label")
                if str(data.get(key, "") or "").strip()
            )
            toks = set(content_tokens(text))
            if toks and query_tokens.intersection(toks):
                matched_entities.append(str(node))
        for cid in _chunks_from_entities(graph, matched_entities[:512], limit_per_entity=4):
            cand_id = _candidate_node_for_unit(graph, cid, unit=unit)
            if cand_id:
                linked = _linked_entities(graph, cand_id)
                rel = 0.0
                if linked:
                    rel = min(1.0, float(len(set(linked).intersection(set(matched_entities)))) / float(max(1, len(linked))))
                _add_candidate(rows, cand_id, graph, semantic, config, "title_entity_lookup", query_relevance=max(0.15, rel))
        for cid in chunk_ids:
            if len(rows) >= top_n:
                break
            title = str(_node_title(graph, cid) or "")
            text = f"{title} {_node_text(graph, cid)[:240]}"
            toks = set(content_tokens(text))
            if toks and query_tokens.intersection(toks):
                overlap = float(len(query_tokens.intersection(toks))) / float(max(1, len(query_tokens)))
                _add_candidate(rows, cid, graph, semantic, config, "title_entity_lookup", query_relevance=_clip01(overlap))
        counts["title_entity_lookup"] = int(sum(1 for r in rows.values() if "title_entity_lookup" in set(r.get("source_tags", set()))))

    if len(rows) < min_n:
        seen = set(rows)
        filler: List[Tuple[float, str]] = []
        for cid in chunk_ids:
            if cid in seen:
                continue
            text = f"{_node_title(graph, cid) or ''} {_node_text(graph, cid)[:280]}"
            toks = set(content_tokens(text))
            overlap = float(len(query_tokens.intersection(toks))) / float(max(1, len(query_tokens)))
            token_pen = min(1.0, float(_token_count(text)) / 600.0)
            filler.append((float(overlap - 0.02 * token_pen), str(cid)))
        filler.sort(key=lambda x: (float(x[0]), str(x[1])), reverse=True)
        for _score, cid in filler[: max(0, min_n - len(rows))]:
            _add_candidate(rows, cid, graph, semantic, config, "source_balanced", query_relevance=0.05)

    out: List[ChunkCandidate] = []
    for cid, row in rows.items():
        embedding = row.get("embedding")
        qrel = float(row.get("query_relevance", 0.0) or 0.0)
        if qvec is not None and embedding is not None:
            qrel = max(qrel, _clip01((_cosine(qvec, embedding) + 1.0) / 2.0))
        text = _node_text(graph, cid)
        carrier_id = str(cid) if _node_type(graph, cid) in {"chunk", "passage", "document"} else (_carrier_for_sentence(graph, cid) or "")
        out.append(
            ChunkCandidate(
                chunk_id=str(cid),
                title=_node_title(graph, cid),
                text=str(text),
                carrier_id=str(carrier_id or "") or None,
                embedding_id=row.get("embedding_id"),
                query_relevance=float(qrel),
                token_count=int(_token_count(text)),
                linked_entities=list(_linked_entities(graph, cid)),
                source_tags=set(str(x) for x in set(row.get("source_tags", set()) or set()) if str(x)),
                embedding=embedding,
            )
        )

    def _rank_key(candidate: ChunkCandidate) -> Tuple[float, float, float, str]:
        source_bonus = min(1.0, float(len(candidate.source_tags)) / 4.0)
        token_pen = min(1.0, float(candidate.token_count) / 700.0)
        score = 0.78 * candidate.query_relevance + 0.14 * source_bonus - 0.08 * token_pen
        return (float(score), float(candidate.query_relevance), -float(candidate.token_count), str(candidate.chunk_id))

    out.sort(key=_rank_key, reverse=True)
    out = out[:top_n]
    diag = {
        "stage": "chunk_universe",
        "num_chunks": int(len(out)),
        "num_semantic_chunks": int(counts["semantic"]),
        "num_source_balanced_chunks": int(counts["source_balanced"]),
        "num_graph_flow_chunks": int(counts["graph_flow"]),
        "num_title_entity_lookup_chunks": int(counts["title_entity_lookup"]),
        "avg_query_relevance": float(_mean([c.query_relevance for c in out])),
        "max_query_relevance": float(max([c.query_relevance for c in out] or [0.0])),
        "avg_token_count": float(_mean([float(c.token_count) for c in out])),
        "candidate_gold_partial_eval_only": False,
        "candidate_gold_full_eval_only": False,
        "candidate_gold_recall_eval_only": 0.0,
        "chunk_universe_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    _LAST_DIAGNOSTICS["chunk_universe"] = diag
    return out


def _candidate_by_id(universe: Sequence[ChunkCandidate]) -> Dict[str, ChunkCandidate]:
    return {str(c.chunk_id): c for c in list(universe or [])}


def _candidate_distance(a: ChunkCandidate, b: ChunkCandidate) -> float:
    if str(a.chunk_id) == str(b.chunk_id):
        return 0.0
    av = _normalize_vector(a.embedding)
    bv = _normalize_vector(b.embedding)
    if av is None or bv is None or av.shape != bv.shape:
        return 1.0
    return float(max(0.0, min(2.0, 1.0 - float(np.dot(av, bv)))))


def _distance_matrix(candidates: Sequence[ChunkCandidate]) -> np.ndarray:
    n = len(candidates)
    dmat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _candidate_distance(candidates[i], candidates[j])
            dmat[i, j] = d
            dmat[j, i] = d
    return dmat


def _normalize_weights(values: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray([max(0.0, float(v)) for v in list(values or [])], dtype=np.float64)
    if arr.size <= 0:
        return None
    total = float(arr.sum())
    if total <= 1.0e-12:
        return None
    return arr / total


def _weighted_sample_indices(
    candidates: Sequence[ChunkCandidate],
    sample_size: int,
    rng: np.random.Generator,
    mode: str,
) -> List[int]:
    n = len(candidates)
    if n <= 0:
        return []
    size = min(max(1, int(sample_size)), n)
    weights = None
    if str(mode or "query_weighted").strip().lower() == "query_weighted":
        weights = _normalize_weights([0.001 + max(0.0, c.query_relevance) for c in candidates])
    idx = rng.choice(np.arange(n), size=size, replace=False, p=weights)
    return [int(i) for i in idx.tolist()]


def _medoid_cost(dmat: np.ndarray, medoids: Sequence[int], weights: np.ndarray) -> float:
    if dmat.size <= 0 or not medoids:
        return 0.0
    m = np.asarray(list(medoids), dtype=np.int64)
    min_dist = np.min(dmat[:, m], axis=1)
    return float(np.dot(min_dist, weights))


def _greedy_initial_medoids(dmat: np.ndarray, weights: np.ndarray, relevances: np.ndarray, k: int) -> List[int]:
    n = int(dmat.shape[0])
    if n <= 0:
        return []
    kk = min(max(1, int(k)), n)
    first = int(np.argmax(relevances + 0.05 * weights))
    medoids = [first]
    while len(medoids) < kk:
        current = np.min(dmat[:, np.asarray(medoids, dtype=np.int64)], axis=1)
        scores = current * (0.35 + 0.65 * np.clip(relevances, 0.0, 1.0))
        for midx in medoids:
            scores[midx] = -1.0
        medoids.append(int(np.argmax(scores)))
    return medoids


def _pam_approx_medoids(sample: Sequence[ChunkCandidate], k: int) -> Tuple[List[int], Dict[str, float]]:
    t0 = time.perf_counter()
    n = len(sample)
    if n <= 0:
        return [], {"cost": 0.0, "iterations": 0, "kmedoids_ms": 0.0}
    kk = min(max(1, int(k)), n)
    dmat = _distance_matrix(sample)
    weights = _normalize_weights([0.001 + max(0.0, c.query_relevance) for c in sample])
    if weights is None:
        weights = np.ones(n, dtype=np.float64) / float(n)
    rel = np.asarray([_clip01(c.query_relevance) for c in sample], dtype=np.float64)
    medoids = _greedy_initial_medoids(dmat=dmat, weights=weights, relevances=rel, k=kk)
    current_cost = _medoid_cost(dmat, medoids, weights)
    iterations = 0
    for _ in range(8):
        iterations += 1
        best_cost = current_cost
        best_swap: Optional[Tuple[int, int]] = None
        medoid_set = set(medoids)
        for pos, _old_idx in enumerate(list(medoids)):
            for cand_idx in range(n):
                if cand_idx in medoid_set:
                    continue
                trial = list(medoids)
                trial[pos] = int(cand_idx)
                trial_cost = _medoid_cost(dmat, trial, weights)
                if trial_cost + 1.0e-9 < best_cost:
                    best_cost = float(trial_cost)
                    best_swap = (int(pos), int(cand_idx))
        if best_swap is None:
            break
        medoids[best_swap[0]] = best_swap[1]
        current_cost = float(best_cost)
    medoids = sorted(set(int(x) for x in medoids), key=lambda i: (-float(rel[i]), str(sample[i].chunk_id)))
    return medoids, {
        "cost": float(current_cost),
        "iterations": int(iterations),
        "kmedoids_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def _seed_set_diversity(seeds: Sequence[ChunkMedoidSeed], universe_by_id: Dict[str, ChunkCandidate]) -> float:
    if len(seeds) < 2:
        return 0.0
    vals = []
    for i in range(len(seeds)):
        a = universe_by_id.get(str(seeds[i].chunk_id))
        if a is None:
            continue
        for j in range(i + 1, len(seeds)):
            b = universe_by_id.get(str(seeds[j].chunk_id))
            if b is None:
                continue
            vals.append(min(1.0, _candidate_distance(a, b) / 2.0))
    return float(_mean(vals))


def _seed_set_components(
    seeds: Sequence[ChunkMedoidSeed],
    universe: Sequence[ChunkCandidate],
) -> Dict[str, float]:
    by_id = _candidate_by_id(universe)
    seed_candidates = [by_id[str(seed.chunk_id)] for seed in list(seeds or []) if str(seed.chunk_id) in by_id]
    if not seed_candidates:
        return {
            "score": 0.0,
            "relevance": 0.0,
            "coverage": 0.0,
            "diversity": 0.0,
            "redundancy": 0.0,
            "token_cost": 0.0,
        }
    relevance = _clip01(_mean([c.query_relevance for c in seed_candidates]))
    weights = _normalize_weights([0.001 + max(0.0, c.query_relevance) for c in universe])
    if weights is None:
        weights = np.ones(len(universe), dtype=np.float64) / float(max(1, len(universe)))
    min_dist = []
    for cand in universe:
        min_dist.append(min((_candidate_distance(cand, seed) for seed in seed_candidates), default=1.0))
    weighted_avg_dist = float(np.dot(np.asarray(min_dist, dtype=np.float64), weights)) if min_dist else 1.0
    coverage = _clip01(1.0 - min(1.0, weighted_avg_dist / 2.0))
    diversity = _seed_set_diversity(list(seeds or []), by_id)
    pair_sim = []
    for i in range(len(seed_candidates)):
        for j in range(i + 1, len(seed_candidates)):
            pair_sim.append(_clip01(1.0 - min(1.0, _candidate_distance(seed_candidates[i], seed_candidates[j]) / 2.0)))
    redundancy = float(max(pair_sim) if pair_sim else 0.0)
    token_cost = _clip01(_mean([min(1.0, float(c.token_count) / 700.0) for c in seed_candidates]))
    score = float(relevance + coverage + diversity - redundancy - token_cost)
    return {
        "score": float(score),
        "relevance": float(relevance),
        "coverage": float(coverage),
        "diversity": float(diversity),
        "redundancy": float(redundancy),
        "token_cost": float(token_cost),
    }


def sample_chunk_medoid_seed_sets(C_q, query_embedding, config) -> list[list[ChunkMedoidSeed]]:
    t_sampling = time.perf_counter()
    universe = list(C_q or [])
    k = max(1, _safe_int(_cfg(config, "phase8_chunk_medoid_k", 5), 5))
    per_k = max(1, _safe_int(_cfg(config, "phase8_chunk_medoid_sample_size_per_k", 40), 40))
    sample_size = max(1, int(k * per_k))
    num_samples = max(1, _safe_int(_cfg(config, "phase8_chunk_medoid_num_samples", 5), 5))
    mode = str(_cfg(config, "phase8_chunk_medoid_sampling", "query_weighted") or "query_weighted").strip().lower()
    seed = _stable_seed(config, query_embedding=query_embedding)
    rng = np.random.default_rng(seed)
    sampled_unique: Set[str] = set()
    seed_sets: List[List[ChunkMedoidSeed]] = []
    kmedoids_ms_total = 0.0
    universe_by_id = _candidate_by_id(universe)

    for sample_id in range(num_samples):
        sample_indices = _weighted_sample_indices(universe, sample_size=sample_size, rng=rng, mode=mode)
        sampled_unique.update(str(universe[i].chunk_id) for i in sample_indices)
        sample = [universe[i] for i in sample_indices]
        medoid_indices, medoid_diag = _pam_approx_medoids(sample, k=k)
        kmedoids_ms_total += float(medoid_diag.get("kmedoids_ms", 0.0) or 0.0)
        seeds: List[ChunkMedoidSeed] = []
        for rank, idx in enumerate(medoid_indices):
            cand = sample[int(idx)]
            token_pen = min(1.0, float(cand.token_count) / 700.0)
            score = _clip01(0.85 * cand.query_relevance - 0.15 * token_pen)
            seeds.append(
                ChunkMedoidSeed(
                    chunk_id=str(cand.chunk_id),
                    score=float(score),
                    sample_id=int(sample_id),
                    refined_from=None,
                    bridge_entities=list(cand.linked_entities),
                    diagnostics={
                        "sample_id": int(sample_id),
                        "rank": int(rank),
                        "title": str(cand.title or ""),
                        "query_relevance": float(cand.query_relevance),
                        "token_count": int(cand.token_count),
                        "source_tags": sorted(str(x) for x in set(cand.source_tags or set())),
                        "kmedoids_cost": float(medoid_diag.get("cost", 0.0)),
                    },
                )
            )
        seed_sets.append(seeds)

    seed_rows = []
    for sample_id, seeds in enumerate(seed_sets):
        comp = _seed_set_components(seeds, universe=universe)
        seed_rows.append(
            {
                "stage": "chunk_seed_set_candidate",
                "sample_id": int(sample_id),
                "seed_chunk_ids": [str(s.chunk_id) for s in seeds],
                "seed_titles": [
                    str((universe_by_id.get(str(s.chunk_id)) or ChunkCandidate(str(s.chunk_id), None, "", None, None, 0.0, 0, [], set())).title or "")
                    for s in seeds
                ],
                "seed_mean_relevance": float(comp.get("relevance", 0.0)),
                "seed_coverage": float(comp.get("coverage", 0.0)),
                "seed_diversity": float(comp.get("diversity", 0.0)),
                "seed_token_cost": float(comp.get("token_cost", 0.0)),
                "seed_gold_hit_eval_only": False,
            }
        )
    _LAST_DIAGNOSTICS["chunk_medoid_sampling"] = {
        "stage": "chunk_medoid_sampling",
        "k": int(k),
        "sample_size": int(min(sample_size, len(universe))),
        "num_samples": int(num_samples),
        "sampling_mode": str(mode),
        "sample_seed": int(seed),
        "num_unique_sampled_chunks": int(len(sampled_unique)),
        "sampling_ms": float((time.perf_counter() - t_sampling) * 1000.0),
        "kmedoids_ms": float(kmedoids_ms_total),
        "best_seed_score": 0.0,
        "best_seed_relevance": 0.0,
        "best_seed_coverage": 0.0,
        "best_seed_diversity": 0.0,
        "best_seed_token_cost": 0.0,
        "seed_gold_hit_eval_only": False,
    }
    _LAST_DIAGNOSTICS["chunk_seed_set_candidates"] = seed_rows
    return seed_sets


def select_best_chunk_medoid_seed_set(seed_sets, C_q, query_embedding, config) -> list[ChunkMedoidSeed]:
    del query_embedding
    del config
    t0 = time.perf_counter()
    universe = list(C_q or [])
    candidate_sets = [list(s or []) for s in list(seed_sets or []) if list(s or [])]
    if not candidate_sets:
        _LAST_DIAGNOSTICS["chunk_seed_selection"] = {
            "stage": "chunk_seed_selection",
            "best_sample_id": -1,
            "best_seed_chunk_ids": [],
            "best_seed_titles": [],
            "best_seed_score": 0.0,
            "best_seed_relevance": 0.0,
            "best_seed_coverage": 0.0,
            "best_seed_diversity": 0.0,
            "best_seed_redundancy": 0.0,
            "best_seed_token_cost": 0.0,
            "seed_gold_hit_eval_only": False,
            "chunk_seed_selection_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return []
    by_id = _candidate_by_id(universe)
    scored: List[Tuple[float, int, List[ChunkMedoidSeed], Dict[str, float]]] = []
    for idx, seeds in enumerate(candidate_sets):
        comp = _seed_set_components(seeds, universe=universe)
        scored.append((float(comp.get("score", 0.0)), int(idx), seeds, comp))
    scored.sort(key=lambda row: (float(row[0]), -int(row[1])), reverse=True)
    _score, best_idx, best, comp = scored[0]
    best_out: List[ChunkMedoidSeed] = []
    for seed in best:
        diag = dict(seed.diagnostics or {})
        diag.update(
            {
                "best_seed_set": True,
                "best_seed_score": float(comp.get("score", 0.0)),
                "best_seed_relevance": float(comp.get("relevance", 0.0)),
                "best_seed_coverage": float(comp.get("coverage", 0.0)),
                "best_seed_diversity": float(comp.get("diversity", 0.0)),
                "best_seed_redundancy": float(comp.get("redundancy", 0.0)),
                "best_seed_token_cost": float(comp.get("token_cost", 0.0)),
            }
        )
        best_out.append(
            ChunkMedoidSeed(
                chunk_id=str(seed.chunk_id),
                score=float(seed.score),
                sample_id=int(seed.sample_id),
                refined_from=seed.refined_from,
                bridge_entities=list(seed.bridge_entities or []),
                diagnostics=diag,
            )
        )
    _LAST_DIAGNOSTICS["chunk_seed_selection"] = {
        "stage": "chunk_seed_selection",
        "best_sample_id": int(best_idx),
        "best_seed_chunk_ids": [str(s.chunk_id) for s in best_out],
        "best_seed_titles": [str(by_id[str(s.chunk_id)].title or "") for s in best_out if str(s.chunk_id) in by_id],
        "best_seed_score": float(comp.get("score", 0.0)),
        "best_seed_relevance": float(comp.get("relevance", 0.0)),
        "best_seed_coverage": float(comp.get("coverage", 0.0)),
        "best_seed_diversity": float(comp.get("diversity", 0.0)),
        "best_seed_redundancy": float(comp.get("redundancy", 0.0)),
        "best_seed_token_cost": float(comp.get("token_cost", 0.0)),
        "seed_gold_hit_eval_only": False,
        "chunk_seed_selection_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    sampling_diag = dict(_LAST_DIAGNOSTICS.get("chunk_medoid_sampling", {}) or {})
    if sampling_diag:
        sampling_diag.update(
            {
                "best_seed_score": float(comp.get("score", 0.0)),
                "best_seed_relevance": float(comp.get("relevance", 0.0)),
                "best_seed_coverage": float(comp.get("coverage", 0.0)),
                "best_seed_diversity": float(comp.get("diversity", 0.0)),
                "best_seed_token_cost": float(comp.get("token_cost", 0.0)),
                "seed_gold_hit_eval_only": False,
            }
        )
        _LAST_DIAGNOSTICS["chunk_medoid_sampling"] = sampling_diag
    return best_out


def _candidate_for_chunk(
    chunk_id: str,
    universe_by_id: Dict[str, ChunkCandidate],
    graph: nx.Graph,
    semantic: Dict[str, Any],
    config: Any,
    query_embedding: Any,
    source: str,
) -> Optional[ChunkCandidate]:
    cid = str(chunk_id or "")
    if cid in universe_by_id:
        base = universe_by_id[cid]
        tags = set(base.source_tags or set())
        tags.add(str(source))
        return ChunkCandidate(
            chunk_id=str(base.chunk_id),
            title=base.title,
            text=str(base.text),
            carrier_id=base.carrier_id,
            embedding_id=base.embedding_id,
            query_relevance=float(base.query_relevance),
            token_count=int(base.token_count),
            linked_entities=list(base.linked_entities),
            source_tags=tags,
            embedding=base.embedding,
        )
    if not cid or cid not in graph or _node_type(graph, cid) not in {"chunk", "passage", "document", "sentence"}:
        return None
    embedding_id, vec = _best_chunk_embedding(cid, graph, semantic, config)
    qrel = 0.0
    qvec = _normalize_vector(query_embedding)
    if qvec is not None and vec is not None:
        qrel = _clip01((_cosine(qvec, vec) + 1.0) / 2.0)
    text = _node_text(graph, cid)
    carrier_id = str(cid) if _node_type(graph, cid) in {"chunk", "passage", "document"} else (_carrier_for_sentence(graph, cid) or "")
    return ChunkCandidate(
        chunk_id=str(cid),
        title=_node_title(graph, cid),
        text=str(text),
        carrier_id=str(carrier_id or "") or None,
        embedding_id=embedding_id,
        query_relevance=float(qrel),
        token_count=int(_token_count(text)),
        linked_entities=list(_linked_entities(graph, cid)),
        source_tags={str(source)},
        embedding=vec,
    )


def _assigned_chunks_by_seed(
    seeds: Sequence[ChunkMedoidSeed],
    universe: Sequence[ChunkCandidate],
) -> Dict[str, Set[str]]:
    by_id = _candidate_by_id(universe)
    seed_cands = [by_id[str(seed.chunk_id)] for seed in list(seeds or []) if str(seed.chunk_id) in by_id]
    out: Dict[str, Set[str]] = {str(seed.chunk_id): set() for seed in list(seeds or [])}
    if not seed_cands:
        return out
    for cand in list(universe or []):
        best_seed = min(seed_cands, key=lambda seed: (_candidate_distance(cand, seed), str(seed.chunk_id)))
        out.setdefault(str(best_seed.chunk_id), set()).add(str(cand.chunk_id))
    return out


def _entity_degree(graph: nx.Graph, entity_id: str) -> int:
    if str(entity_id) not in graph:
        return 0
    return int(graph.degree(str(entity_id)))


def _filtered_bridge_entities(
    graph: nx.Graph,
    entities: Sequence[str],
    max_entities: int,
    degree_cap: int,
) -> List[str]:
    rows = []
    for entity in list(entities or []):
        eid = str(entity or "")
        if not eid or eid not in graph or _node_type(graph, eid) != "entity":
            continue
        degree = _entity_degree(graph, eid)
        if degree > max(1, int(degree_cap)):
            continue
        rows.append((degree, eid))
    rows.sort(key=lambda x: (int(x[0]), str(x[1])))
    return [eid for _degree, eid in rows[: max(1, int(max_entities))]]


def _bridge_connectivity(candidate: ChunkCandidate, bridge_entities: Sequence[str]) -> float:
    bridges = set(str(x) for x in list(bridge_entities or []) if str(x))
    if not bridges:
        return 0.0
    linked = set(str(x) for x in list(candidate.linked_entities or []) if str(x))
    return _clip01(float(len(linked.intersection(bridges))) / float(max(1, len(bridges))))


def _chunk_graph_connectivity(
    graph: nx.Graph,
    chunk_id: str,
    other_chunk_ids: Sequence[str],
    max_hops: int,
) -> float:
    cid = str(chunk_id or "")
    if not cid or cid not in graph:
        return 0.0
    vals = []
    cutoff = max(1, int(max_hops))
    for other in list(other_chunk_ids or [])[:32]:
        oid = str(other or "")
        if not oid or oid not in graph or oid == cid:
            continue
        try:
            dist = nx.shortest_path_length(graph, cid, oid)
        except Exception:
            continue
        if int(dist) <= cutoff:
            vals.append(1.0 / (1.0 + float(dist)))
    return _clip01(max(vals) if vals else 0.0)


def _redundancy_to_other_chunks(
    candidate: ChunkCandidate,
    other_seed_ids: Sequence[str],
    candidates_by_id: Dict[str, ChunkCandidate],
) -> float:
    vals = []
    for sid in list(other_seed_ids or []):
        other = candidates_by_id.get(str(sid))
        if other is None:
            continue
        dist = _candidate_distance(candidate, other)
        vals.append(_clip01(1.0 - min(1.0, dist / 2.0)))
    return float(max(vals) if vals else 0.0)


def _hub_entity_penalty(graph: nx.Graph, candidate: ChunkCandidate, degree_cap: int) -> float:
    degrees = [
        min(1.0, float(_entity_degree(graph, eid)) / float(max(1, int(degree_cap))))
        for eid in list(candidate.linked_entities or [])
        if str(eid) in graph and _node_type(graph, str(eid)) == "entity"
    ]
    return _clip01(_mean(degrees))


def refine_chunk_medoids_via_bridge_entities(
    best_seed_set,
    C_q,
    graph_index,
    query_embedding,
    config,
) -> list[ChunkMedoidSeed]:
    t0 = time.perf_counter()
    enabled = _cfg_bool(config, "phase8_chunk_bridge_refine_enabled", False)
    before = list(best_seed_set or [])
    if not before:
        _LAST_DIAGNOSTICS["chunk_bridge_refinement"] = {
            "stage": "chunk_bridge_refinement",
            "enabled": bool(enabled),
            "num_refined_seeds": 0,
            "num_changed_seeds": 0,
            "avg_bridge_entities_per_seed": 0.0,
            "avg_refine_candidate_count": 0.0,
            "before_seed_gold_hit_eval_only": False,
            "after_seed_gold_hit_eval_only": False,
            "before_candidate_gold_recall_eval_only": 0.0,
            "after_candidate_gold_recall_eval_only": 0.0,
            "bridge_refinement_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return []
    if not enabled:
        _LAST_DIAGNOSTICS["chunk_bridge_refinement"] = {
            "stage": "chunk_bridge_refinement",
            "enabled": False,
            "num_refined_seeds": int(len(before)),
            "num_changed_seeds": 0,
            "avg_bridge_entities_per_seed": float(_mean([len(list(s.bridge_entities or [])) for s in before])),
            "avg_refine_candidate_count": 0.0,
            "before_seed_gold_hit_eval_only": False,
            "after_seed_gold_hit_eval_only": False,
            "before_candidate_gold_recall_eval_only": 0.0,
            "after_candidate_gold_recall_eval_only": 0.0,
            "bridge_refinement_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return before

    graph = _as_graph(graph_index)
    semantic = _semantic_state(graph_index, graph_index if isinstance(graph_index, dict) else {})
    universe = list(C_q or [])
    universe_by_id = _candidate_by_id(universe)
    assigned = _assigned_chunks_by_seed(before, universe)
    max_entities = max(1, _safe_int(_cfg(config, "phase8_chunk_bridge_max_entities_per_seed", 8), 8))
    degree_cap = max(1, _safe_int(_cfg(config, "phase8_chunk_bridge_entity_degree_cap", 100), 100))
    chunks_per_entity = max(1, _safe_int(_cfg(config, "phase8_chunk_bridge_max_chunks_per_entity", 4), 4))
    max_candidates = max(1, _safe_int(_cfg(config, "phase8_chunk_bridge_max_refine_candidates_per_seed", 64), 64))
    hops = max(1, _safe_int(_cfg(config, "phase8_chunk_bridge_refine_hops", 1), 1))
    candidates_seen: Dict[str, ChunkCandidate] = dict(universe_by_id)
    refined: List[ChunkMedoidSeed] = []
    bridge_counts: List[int] = []
    candidate_counts: List[int] = []
    changed = 0

    for seed in before:
        seed_id = str(seed.chunk_id)
        seed_cand = universe_by_id.get(seed_id)
        raw_bridge_entities = list(seed.bridge_entities or [])
        if seed_cand is not None:
            raw_bridge_entities.extend(list(seed_cand.linked_entities or []))
        bridge_entities = _filtered_bridge_entities(
            graph,
            raw_bridge_entities,
            max_entities=max_entities,
            degree_cap=degree_cap,
        )
        bridge_counts.append(int(len(bridge_entities)))
        raw_candidate_ids = set(assigned.get(seed_id, set()))
        raw_candidate_ids.add(seed_id)
        raw_candidate_ids.update(_chunks_from_entities(graph, bridge_entities, limit_per_entity=chunks_per_entity))
        candidate_rows: List[ChunkCandidate] = []
        for cid in sorted(raw_candidate_ids):
            cand = _candidate_for_chunk(
                cid,
                universe_by_id=universe_by_id,
                graph=graph,
                semantic=semantic,
                config=config,
                query_embedding=query_embedding,
                source="bridge_refine",
            )
            if cand is None:
                continue
            candidates_seen[str(cand.chunk_id)] = cand
            candidate_rows.append(cand)
        candidate_rows.sort(
            key=lambda c: (
                float(c.query_relevance),
                float(_bridge_connectivity(c, bridge_entities)),
                -float(c.token_count),
                str(c.chunk_id),
            ),
            reverse=True,
        )
        candidate_rows = candidate_rows[:max_candidates]
        candidate_counts.append(int(len(candidate_rows)))
        other_seed_ids = [str(s.chunk_id) for s in before if str(s.chunk_id) != seed_id]
        best_cand = candidates_seen.get(seed_id)
        best_score = -1.0e9
        for cand in candidate_rows:
            bridge_conn = _bridge_connectivity(cand, bridge_entities)
            medoid_conn = _chunk_graph_connectivity(graph, str(cand.chunk_id), other_seed_ids, max_hops=hops + 2)
            redundancy = _redundancy_to_other_chunks(cand, other_seed_ids, candidates_seen)
            novelty = _clip01(1.0 - redundancy)
            token_pen = min(1.0, float(cand.token_count) / 700.0)
            hub_pen = _hub_entity_penalty(graph, cand, degree_cap=degree_cap)
            score = float(cand.query_relevance + bridge_conn + medoid_conn + novelty - redundancy - token_pen - hub_pen)
            current_id = str(best_cand.chunk_id if best_cand is not None else "")
            if score > best_score or (abs(score - best_score) <= 1.0e-12 and str(cand.chunk_id) < current_id):
                best_score = score
                best_cand = cand
        if best_cand is None:
            refined.append(seed)
            continue
        if str(best_cand.chunk_id) != seed_id:
            changed += 1
        diag = dict(seed.diagnostics or {})
        diag.update(
            {
                "refined": True,
                "refined_from": str(seed_id),
                "refine_candidate_count": int(len(candidate_rows)),
                "bridge_entity_count": int(len(bridge_entities)),
                "refine_score": float(best_score),
                "query_relevance": float(best_cand.query_relevance),
                "token_count": int(best_cand.token_count),
            }
        )
        refined.append(
            ChunkMedoidSeed(
                chunk_id=str(best_cand.chunk_id),
                score=float(best_score),
                sample_id=int(seed.sample_id),
                refined_from=(str(seed_id) if str(best_cand.chunk_id) != seed_id else None),
                bridge_entities=list(_filtered_bridge_entities(graph, list(best_cand.linked_entities or bridge_entities), max_entities, degree_cap)),
                diagnostics=diag,
            )
        )

    _LAST_DIAGNOSTICS["chunk_bridge_refinement"] = {
        "stage": "chunk_bridge_refinement",
        "enabled": True,
        "num_refined_seeds": int(len(refined)),
        "num_changed_seeds": int(changed),
        "avg_bridge_entities_per_seed": float(_mean(bridge_counts)),
        "avg_refine_candidate_count": float(_mean(candidate_counts)),
        "before_seed_gold_hit_eval_only": False,
        "after_seed_gold_hit_eval_only": False,
        "before_candidate_gold_recall_eval_only": 0.0,
        "after_candidate_gold_recall_eval_only": 0.0,
        "bridge_refinement_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    return refined


def _sentences_for_title(graph: nx.Graph, title: str, limit: int) -> List[str]:
    key = str(title or "").strip().lower()
    if not key:
        return []
    rows = [
        str(n)
        for n in graph.nodes
        if _node_type(graph, str(n)) == "sentence"
        and str(graph.nodes[n].get("title", "") or "").strip().lower() == key
    ]
    rows.sort(key=lambda n: _sentence_sort_key(graph, n))
    return rows[: max(1, int(limit))]


def _sentence_neighbors_for_entity(graph: nx.Graph, entity_id: str, limit: int) -> List[str]:
    if str(entity_id) not in graph:
        return []
    out = [
        str(n)
        for n in graph.neighbors(str(entity_id))
        if _node_type(graph, str(n)) == "sentence"
    ]
    out.sort(key=lambda n: _sentence_sort_key(graph, n))
    return out[: max(1, int(limit))]


def _carrier_neighbors_for_entity(graph: nx.Graph, entity_id: str, limit: int) -> List[str]:
    if str(entity_id) not in graph:
        return []
    out = [
        str(n)
        for n in graph.neighbors(str(entity_id))
        if _node_type(graph, str(n)) in {"chunk", "passage", "document"}
    ]
    for sent in _sentence_neighbors_for_entity(graph, str(entity_id), limit=max(1, int(limit)) * 2):
        carrier = _carrier_for_sentence(graph, sent)
        if carrier:
            out.append(str(carrier))
    seen = set()
    uniq = []
    for cid in out:
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(cid)
    uniq.sort(key=lambda n: (_safe_int(graph.nodes[n].get("chunk_idx", 0), 0), str(n)))
    return uniq[: max(1, int(limit))]


def _path_sentence_evidence(graph: nx.Graph, path: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for node in list(path or []):
        sid = str(node)
        if sid not in graph:
            continue
        if _node_type(graph, sid) == "sentence":
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
            continue
        for sent in _sentences_for_carrier(graph, sid, limit=2):
            if sent not in seen:
                seen.add(sent)
                out.append(sent)
    out.sort(key=lambda n: _sentence_sort_key(graph, n))
    return out


def _shortest_path_sentences(
    graph: nx.Graph,
    pairs: Sequence[Tuple[str, str]],
    max_hops: int,
    max_pairs: int,
) -> List[str]:
    out: List[str] = []
    seen = set()
    count = 0
    for src, dst in list(pairs or []):
        if count >= max(0, int(max_pairs)):
            break
        if str(src) not in graph or str(dst) not in graph or str(src) == str(dst):
            continue
        try:
            path = nx.shortest_path(graph, str(src), str(dst))
        except Exception:
            continue
        hops = max(0, len(path) - 1)
        if hops > max(1, int(max_hops)):
            continue
        count += 1
        for sent in _path_sentence_evidence(graph, path):
            if sent not in seen:
                seen.add(sent)
                out.append(sent)
    return out


def _add_evidence_candidate(
    graph: nx.Graph,
    out: List[str],
    tags: Dict[str, Set[str]],
    node: str,
    tag: str,
    cap: int,
) -> bool:
    sid = str(node or "")
    if not sid or sid not in graph or _node_type(graph, sid) != "sentence":
        return False
    tags.setdefault(sid, set()).add(str(tag))
    tags.setdefault(sid, set()).add("phase8_chunk_medoid")
    if sid not in out and len(out) < max(1, int(cap)):
        out.append(sid)
        return True
    return False


def build_chunk_medoid_evidence_candidates(refined_chunk_seeds, graph_index, config) -> list[str]:
    t0 = time.perf_counter()
    graph = _as_graph(graph_index)
    seeds = [s for s in list(refined_chunk_seeds or []) if str(getattr(s, "chunk_id", "")) in graph]
    cap = max(1, _safe_int(_cfg(config, "phase8_chunk_seed_total_candidate_cap", 160), 160))
    top_sentences = max(1, _safe_int(_cfg(config, "phase8_chunk_seed_top_sentences_per_carrier", 3), 3))
    top_entities = max(1, _safe_int(_cfg(config, "phase8_chunk_seed_top_entities_per_seed", 8), 8))
    top_atoms = max(1, _safe_int(_cfg(config, "phase8_chunk_seed_top_atoms_per_entity", 3), 3))
    top_carriers = max(0, _safe_int(_cfg(config, "phase8_chunk_seed_top_carriers_per_entity", 2), 2))
    out: List[str] = []
    source_tags: Dict[str, Set[str]] = {}
    counts = {
        "num_seed_sentence_atoms": 0,
        "num_bridge_entity_atoms": 0,
        "num_bridge_entity_carriers": 0,
        "num_same_title_atoms": 0,
        "num_same_carrier_atoms": 0,
        "num_seed_seed_bridge_atoms": 0,
    }

    seed_sentence_nodes: List[str] = []
    for seed in seeds:
        seed_id = str(seed.chunk_id)
        carrier = seed_id if _node_type(graph, seed_id) in {"chunk", "passage", "document"} else (_carrier_for_sentence(graph, seed_id) or seed_id)
        for sent in _sentences_for_carrier(graph, carrier, limit=top_sentences):
            seed_sentence_nodes.append(sent)
            if _add_evidence_candidate(graph, out, source_tags, sent, "phase8_chunk_seed_sentence", cap):
                counts["num_seed_sentence_atoms"] += 1

        bridge_entities = _filtered_bridge_entities(
            graph,
            list(seed.bridge_entities or []) + _linked_entities(graph, seed_id),
            max_entities=top_entities,
            degree_cap=max(1, _safe_int(_cfg(config, "phase8_chunk_bridge_entity_degree_cap", 100), 100)),
        )
        for entity_id in bridge_entities:
            for sent in _sentence_neighbors_for_entity(graph, entity_id, limit=top_atoms):
                if _add_evidence_candidate(graph, out, source_tags, sent, "phase8_chunk_bridge_entity_atom", cap):
                    counts["num_bridge_entity_atoms"] += 1
            for carrier_id in _carrier_neighbors_for_entity(graph, entity_id, limit=top_carriers):
                for sent in _sentences_for_carrier(graph, carrier_id, limit=top_atoms):
                    if _add_evidence_candidate(graph, out, source_tags, sent, "phase8_chunk_bridge_entity_carrier", cap):
                        counts["num_bridge_entity_carriers"] += 1

        for sent in list(seed_sentence_nodes)[-top_sentences:]:
            title = str(_node_title(graph, sent) or "")
            for same_title in _sentences_for_title(graph, title, limit=top_atoms):
                if _add_evidence_candidate(graph, out, source_tags, same_title, "phase8_chunk_same_title", cap):
                    counts["num_same_title_atoms"] += 1
            same_carrier = _carrier_for_sentence(graph, sent)
            if same_carrier:
                for same_sent in _sentences_for_carrier(graph, same_carrier, limit=top_atoms):
                    if _add_evidence_candidate(graph, out, source_tags, same_sent, "phase8_chunk_same_carrier", cap):
                        counts["num_same_carrier_atoms"] += 1
        if len(out) >= cap:
            break

    seed_ids = [str(seed.chunk_id) for seed in seeds]
    seed_pairs = []
    for i, src in enumerate(seed_ids):
        for dst in seed_ids[i + 1 :]:
            seed_pairs.append((src, dst))
    for sent in _shortest_path_sentences(graph, seed_pairs, max_hops=3, max_pairs=10):
        if _add_evidence_candidate(graph, out, source_tags, sent, "phase8_chunk_seed_seed_bridge", cap):
            counts["num_seed_seed_bridge_atoms"] += 1

    diag = {
        "stage": "chunk_medoid_evidence_proposal",
        **counts,
        "num_final_evidence_candidates": int(len(out)),
        "candidate_gold_partial_eval_only": False,
        "candidate_gold_full_eval_only": False,
        "candidate_gold_recall_eval_only": 0.0,
        "candidate_oracle_F1_eval_only": 0.0,
        "evidence_proposal_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    _LAST_DIAGNOSTICS["chunk_medoid_evidence_proposal"] = diag
    _LAST_DIAGNOSTICS["evidence_source_tags"] = {
        str(cid): sorted(str(t) for t in set(source_tags.get(str(cid), set()) or set()))
        for cid in out
    }
    return out


def run_phase8_chunk_medoid_proposal(
    query: str,
    graph_index: Any,
    embeddings: Any,
    config: Any,
    query_embedding: Any = None,
) -> ChunkMedoidProposalResult:
    payload = dict(embeddings or {}) if isinstance(embeddings, dict) else {}
    if query_embedding is not None and "query_embedding" not in payload:
        payload["query_embedding"] = query_embedding
    _LAST_DIAGNOSTICS.clear()
    universe = build_query_chunk_universe(query, graph_index=graph_index, embeddings=payload, config=config)
    initial = sample_chunk_medoid_seed_sets(universe, query_embedding=payload.get("query_embedding"), config=config)
    best = select_best_chunk_medoid_seed_set(initial, C_q=universe, query_embedding=payload.get("query_embedding"), config=config)
    refined = refine_chunk_medoids_via_bridge_entities(
        best,
        C_q=universe,
        graph_index=graph_index,
        query_embedding=payload.get("query_embedding"),
        config=config,
    )
    evidence = build_chunk_medoid_evidence_candidates(
        refined_chunk_seeds=refined,
        graph_index=graph_index,
        config=config,
    )
    diagnostics = get_last_phase8_chunk_diagnostics()
    return ChunkMedoidProposalResult(
        chunk_universe=list(universe),
        initial_medoid_sets=list(initial),
        best_medoid_set=list(best),
        refined_medoid_set=list(refined),
        evidence_candidate_ids=list(evidence),
        diagnostics=dict(diagnostics),
    )
