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
