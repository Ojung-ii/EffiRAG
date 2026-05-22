from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from .embedding import encode_texts, topk_cosine_similarity
from .utils import content_tokens


@dataclass
class EntityCandidate:
    entity_id: str
    name: str
    title: Optional[str]
    embedding_id: Optional[str]
    query_relevance: float
    graph_score: float
    degree: int
    source_tags: set[str]
    embedding: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


@dataclass
class MedoidSeed:
    entity_id: str
    score: float
    source_sample_id: int
    refined_from: Optional[str]
    diagnostics: dict


@dataclass
class PamaeProposalResult:
    entity_universe: list[EntityCandidate]
    initial_medoid_sets: list[list[MedoidSeed]]
    best_medoid_set: list[MedoidSeed]
    refined_medoid_set: list[MedoidSeed]
    evidence_candidate_ids: list[str]
    diagnostics: dict


_LAST_DIAGNOSTICS: Dict[str, Any] = {}


def get_last_phase8_diagnostics() -> Dict[str, Any]:
    return dict(_LAST_DIAGNOSTICS)


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


def _cfg(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _cfg_bool(config: Any, key: str, default: bool) -> bool:
    value = _cfg(config, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _normalize_vector(vec: Any) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1 or arr.size <= 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        return None
    return arr / norm


def _cosine(a: Any, b: Any) -> float:
    av = _normalize_vector(a)
    bv = _normalize_vector(b)
    if av is None or bv is None or av.shape != bv.shape:
        return 0.0
    return float(np.dot(av, bv))


def _as_graph(graph_index: Any) -> nx.Graph:
    if isinstance(graph_index, nx.Graph):
        return graph_index
    if isinstance(graph_index, dict):
        graph = graph_index.get("graph")
        if isinstance(graph, nx.Graph):
            return graph
    raise TypeError("phase8 graph_index must be a networkx graph or an index state containing 'graph'")


def _semantic_state(graph_index: Any, embeddings: Any) -> Dict[str, Any]:
    if isinstance(embeddings, dict) and "entity_ids" in embeddings:
        return dict(embeddings)
    if isinstance(graph_index, dict):
        semantic = graph_index.get("semantic_state", {})
        if isinstance(semantic, dict):
            return dict(semantic)
    return {}


def _query_embedding(query: str, embeddings: Any, config: Any) -> Optional[np.ndarray]:
    if isinstance(embeddings, dict):
        for key in ("query_embedding", "qvec", "query_vec"):
            if key in embeddings:
                vec = _normalize_vector(embeddings.get(key))
                if vec is not None:
                    return vec
    encoded = encode_texts(
        texts=[str(query or "")],
        model_name=str(_cfg(config, "embedding_model_name", "nvidia/NV-Embed-v2") or "nvidia/NV-Embed-v2"),
        batch_size=1,
        max_length=max(1, _safe_int(_cfg(config, "query_embedding_max_length", 96), 96)),
        max_chars=max(1, _safe_int(_cfg(config, "query_embedding_text_max_chars", 384), 384)),
        instruction="",
    )
    if not bool(encoded.get("ok", False)):
        return None
    vectors = list(encoded.get("vectors", []) or [])
    if not vectors:
        return None
    return _normalize_vector(vectors[0])


def _node_type(graph: nx.Graph, node: str) -> str:
    if node not in graph:
        return ""
    return str(graph.nodes[node].get("node_type", "") or "").strip().lower()


def _entity_name(entity_id: str, graph: nx.Graph) -> str:
    if entity_id in graph:
        data = dict(graph.nodes[entity_id] or {})
        for key in ("name", "text", "title", "label"):
            value = str(data.get(key, "") or "").strip()
            if value:
                return value
    sid = str(entity_id or "")
    if sid.startswith("e::"):
        return sid.split("::", 1)[1].strip()
    return sid


def _entity_title(entity_id: str, graph: nx.Graph) -> Optional[str]:
    if entity_id not in graph:
        return None
    title = str(graph.nodes[entity_id].get("title", "") or "").strip()
    return title or None


def _entity_token(entity_id: str) -> str:
    sid = str(entity_id or "")
    if sid.startswith("e::"):
        return sid.split("::", 1)[1].strip().lower()
    return sid.strip().lower()


def _entity_embedding(entity_id: str, semantic: Dict[str, Any]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    id_to_idx = dict(semantic.get("entity_id_to_idx", {}) or {})
    embeddings = semantic.get("entity_embeddings")
    if embeddings is None or entity_id not in id_to_idx:
        return None, None
    idx = _safe_int(id_to_idx.get(entity_id, -1), -1)
    try:
        if idx < 0 or idx >= len(embeddings):
            return None, None
        vec = _normalize_vector(embeddings[idx])
    except Exception:
        return None, None
    if vec is None:
        return None, None
    return str(entity_id), vec


def _linked_chunk_embeddings(entity_id: str, graph: nx.Graph, semantic: Dict[str, Any], max_chunks: int = 8) -> Optional[np.ndarray]:
    chunk_id_to_idx = dict(semantic.get("chunk_id_to_idx", {}) or {})
    chunk_embeddings = semantic.get("chunk_embeddings")
    if chunk_embeddings is None:
        return None
    chunks: List[str] = []
    entity_to_chunks = dict(semantic.get("entity_to_chunks", {}) or {})
    for cid in list(entity_to_chunks.get(entity_id, []) or []):
        sid = str(cid)
        if sid and sid in chunk_id_to_idx:
            chunks.append(sid)
    if not chunks and entity_id in graph:
        for nbr in graph.neighbors(entity_id):
            ntype = _node_type(graph, str(nbr))
            if ntype in {"chunk", "passage", "document", "sentence"} and str(nbr) in chunk_id_to_idx:
                chunks.append(str(nbr))
            if len(chunks) >= max_chunks:
                break
    vecs = []
    for cid in chunks[: max(1, int(max_chunks))]:
        idx = _safe_int(chunk_id_to_idx.get(cid, -1), -1)
        try:
            if 0 <= idx < len(chunk_embeddings):
                vec = _normalize_vector(chunk_embeddings[idx])
                if vec is not None:
                    vecs.append(vec)
        except Exception:
            continue
    if not vecs:
        return None
    return _normalize_vector(np.mean(np.vstack(vecs), axis=0))


def _entity_name_embedding(entity_id: str, graph: nx.Graph, config: Any) -> Optional[np.ndarray]:
    encoded = encode_texts(
        texts=[_entity_name(entity_id, graph)],
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


def _best_entity_embedding(entity_id: str, graph: nx.Graph, semantic: Dict[str, Any], config: Any) -> Tuple[Optional[str], Optional[np.ndarray]]:
    embedding_id, vec = _entity_embedding(entity_id, semantic)
    if vec is not None:
        return embedding_id, vec
    pooled = _linked_chunk_embeddings(entity_id, graph, semantic)
    if pooled is not None:
        return f"{entity_id}::linked_carrier_pool", pooled
    name_vec = _entity_name_embedding(entity_id, graph, config)
    if name_vec is not None:
        return f"{entity_id}::name", name_vec
    return None, None


def _add_candidate(
    rows: Dict[str, Dict[str, Any]],
    entity_id: str,
    graph: nx.Graph,
    semantic: Dict[str, Any],
    config: Any,
    source: str,
    query_relevance: float = 0.0,
    graph_score: float = 0.0,
) -> None:
    eid = str(entity_id or "")
    if not eid or eid not in graph or _node_type(graph, eid) != "entity":
        return
    row = rows.setdefault(
        eid,
        {
            "entity_id": eid,
            "query_relevance": 0.0,
            "graph_score": 0.0,
            "source_tags": set(),
            "embedding_id": None,
            "embedding": None,
        },
    )
    row["query_relevance"] = max(float(row.get("query_relevance", 0.0) or 0.0), float(query_relevance))
    row["graph_score"] = max(float(row.get("graph_score", 0.0) or 0.0), float(graph_score))
    row["source_tags"].add(str(source))
    if row.get("embedding") is None:
        embedding_id, vec = _best_entity_embedding(eid, graph, semantic, config)
        row["embedding_id"] = embedding_id
        row["embedding"] = vec


def _source_flags(config: Any) -> Dict[str, bool]:
    raw = _cfg(config, "phase8_entity_sources", None)
    if isinstance(raw, dict):
        return {
            "semantic": bool(raw.get("semantic", True)),
            "title_lookup": bool(raw.get("title_lookup", True)),
            "graph_flow": bool(raw.get("graph_flow", True)),
            "source_balanced": bool(raw.get("source_balanced", True)),
        }
    return {
        "semantic": _cfg_bool(config, "phase8_entity_source_semantic", True),
        "title_lookup": _cfg_bool(config, "phase8_entity_source_title_lookup", True),
        "graph_flow": _cfg_bool(config, "phase8_entity_source_graph_flow", True),
        "source_balanced": _cfg_bool(config, "phase8_entity_source_balanced", True),
    }


def build_query_entity_universe(query, graph_index, embeddings, config) -> list[EntityCandidate]:
    t0 = time.perf_counter()
    graph = _as_graph(graph_index)
    semantic = _semantic_state(graph_index, embeddings)
    qvec = _query_embedding(str(query or ""), embeddings, config)
    top_n = max(1, _safe_int(_cfg(config, "phase8_entity_universe_top_n", 2000), 2000))
    min_n = max(0, _safe_int(_cfg(config, "phase8_entity_universe_min_n", 200), 200))
    degree_cap = max(1, _safe_int(_cfg(config, "phase8_entity_degree_cap", 200), 200))
    flags = _source_flags(config)
    rows: Dict[str, Dict[str, Any]] = {}
    counts = {
        "semantic": 0,
        "title_lookup": 0,
        "graph_flow": 0,
        "source_balanced": 0,
    }

    entity_ids = [str(e) for e in list(semantic.get("entity_ids", []) or []) if str(e) in graph]
    if not entity_ids:
        entity_ids = [str(n) for n in graph.nodes if _node_type(graph, str(n)) == "entity"]

    if flags["semantic"] and qvec is not None and semantic.get("entity_embeddings") is not None:
        try:
            idx, scores = topk_cosine_similarity(
                query_vector=qvec,
                matrix=semantic.get("entity_embeddings"),
                topn=min(max(top_n, min_n), len(entity_ids)),
                scan_batch_size=max(1, _safe_int(_cfg(config, "semantic_scan_batch_size", 8192), 8192)),
            )
            semantic_ids = list(semantic.get("entity_ids", []) or [])
            for i, score in zip(idx.tolist(), scores.tolist()):
                if i < 0 or i >= len(semantic_ids):
                    continue
                eid = str(semantic_ids[i])
                _add_candidate(rows, eid, graph, semantic, config, "semantic", query_relevance=_clip01((float(score) + 1.0) / 2.0))
            counts["semantic"] = int(sum(1 for r in rows.values() if "semantic" in set(r.get("source_tags", set()))))
        except Exception:
            counts["semantic"] = 0

    query_tokens = set(content_tokens(str(query or "")))
    if flags["title_lookup"] and query_tokens:
        for eid in entity_ids:
            name_tokens = set(content_tokens(_entity_name(eid, graph)))
            title_tokens = set(content_tokens(_entity_title(eid, graph) or ""))
            overlap = len(query_tokens.intersection(name_tokens.union(title_tokens)))
            if overlap <= 0:
                continue
            denom = max(1, min(len(query_tokens), max(1, len(name_tokens.union(title_tokens)))))
            _add_candidate(
                rows,
                eid,
                graph,
                semantic,
                config,
                "title_lookup",
                query_relevance=_clip01(float(overlap) / float(denom)),
            )
        counts["title_lookup"] = int(sum(1 for r in rows.values() if "title_lookup" in set(r.get("source_tags", set()))))

    flow_scores = {}
    if isinstance(embeddings, dict):
        flow_scores = dict(embeddings.get("flow_scores", {}) or {})
    if flags["graph_flow"] and flow_scores:
        flow_entity_rows = [
            (str(eid), _safe_float(score, 0.0))
            for eid, score in flow_scores.items()
            if str(eid) in graph and _node_type(graph, str(eid)) == "entity"
        ]
        flow_entity_rows.sort(key=lambda x: (float(x[1]), str(x[0])), reverse=True)
        max_flow = max([abs(float(s)) for _eid, s in flow_entity_rows[:top_n]] or [1.0])
        for eid, score in flow_entity_rows[:top_n]:
            _add_candidate(
                rows,
                eid,
                graph,
                semantic,
                config,
                "graph_flow",
                graph_score=_clip01(float(score) / float(max_flow or 1.0)),
            )
        counts["graph_flow"] = int(sum(1 for r in rows.values() if "graph_flow" in set(r.get("source_tags", set()))))

    if flags["source_balanced"]:
        balanced_entities: Set[str] = set()
        if isinstance(embeddings, dict):
            for key in ("semantic_entities", "anchor_nodes", "seed_entities"):
                for eid in list(embeddings.get(key, []) or []):
                    if str(eid) in graph and _node_type(graph, str(eid)) == "entity":
                        balanced_entities.add(str(eid))
            for chunk in list(embeddings.get("semantic_chunks", []) or []):
                if str(chunk) not in graph:
                    continue
                for nbr in graph.neighbors(str(chunk)):
                    if _node_type(graph, str(nbr)) == "entity":
                        balanced_entities.add(str(nbr))
        for eid in sorted(balanced_entities):
            _add_candidate(rows, eid, graph, semantic, config, "source_balanced", query_relevance=0.25)
        counts["source_balanced"] = int(sum(1 for r in rows.values() if "source_balanced" in set(r.get("source_tags", set()))))

    num_high_degree_filtered = 0
    out: List[EntityCandidate] = []
    for eid, row in rows.items():
        degree = int(graph.degree(eid)) if eid in graph else 0
        if degree > degree_cap:
            num_high_degree_filtered += 1
            continue
        vec = row.get("embedding")
        qrel = float(row.get("query_relevance", 0.0) or 0.0)
        if qvec is not None and vec is not None:
            qrel = max(qrel, _clip01((_cosine(qvec, vec) + 1.0) / 2.0))
        out.append(
            EntityCandidate(
                entity_id=str(eid),
                name=_entity_name(eid, graph),
                title=_entity_title(eid, graph),
                embedding_id=row.get("embedding_id"),
                query_relevance=float(qrel),
                graph_score=float(row.get("graph_score", 0.0) or 0.0),
                degree=int(degree),
                source_tags=set(str(x) for x in set(row.get("source_tags", set()) or set()) if str(x)),
                embedding=vec,
            )
        )

    if len(out) < min_n:
        seen = {c.entity_id for c in out}
        filler: List[Tuple[float, str]] = []
        for eid in entity_ids:
            if eid in seen:
                continue
            degree = int(graph.degree(eid)) if eid in graph else 0
            if degree > degree_cap:
                num_high_degree_filtered += 1
                continue
            name_tokens = set(content_tokens(_entity_name(eid, graph)))
            token_overlap = float(len(query_tokens.intersection(name_tokens))) / float(max(1, len(query_tokens)))
            filler.append((float(token_overlap - 0.001 * degree), eid))
        filler.sort(key=lambda x: (float(x[0]), str(x[1])), reverse=True)
        for _score, eid in filler[: max(0, min_n - len(out))]:
            embedding_id, vec = _best_entity_embedding(eid, graph, semantic, config)
            qrel = 0.0
            if qvec is not None and vec is not None:
                qrel = _clip01((_cosine(qvec, vec) + 1.0) / 2.0)
            out.append(
                EntityCandidate(
                    entity_id=str(eid),
                    name=_entity_name(eid, graph),
                    title=_entity_title(eid, graph),
                    embedding_id=embedding_id,
                    query_relevance=float(qrel),
                    graph_score=0.0,
                    degree=int(graph.degree(eid)) if eid in graph else 0,
                    source_tags={"source_balanced"},
                    embedding=vec,
                )
            )

    def _rank_key(candidate: EntityCandidate) -> Tuple[float, float, float, str]:
        source_bonus = min(1.0, float(len(candidate.source_tags)) / 4.0)
        degree_pen = min(1.0, float(candidate.degree) / float(max(1, degree_cap)))
        score = 0.65 * candidate.query_relevance + 0.25 * candidate.graph_score + 0.10 * source_bonus - 0.08 * degree_pen
        return (float(score), float(candidate.query_relevance), -float(candidate.degree), str(candidate.entity_id))

    out.sort(key=_rank_key, reverse=True)
    out = out[:top_n]
    diag = {
        "stage": "entity_universe",
        "num_entities": int(len(out)),
        "num_semantic_entities": int(counts["semantic"]),
        "num_title_lookup_entities": int(counts["title_lookup"]),
        "num_graph_flow_entities": int(counts["graph_flow"]),
        "num_source_balanced_entities": int(counts["source_balanced"]),
        "avg_query_relevance": float(_mean([c.query_relevance for c in out])),
        "max_query_relevance": float(max([c.query_relevance for c in out] or [0.0])),
        "avg_entity_degree": float(_mean([float(c.degree) for c in out])),
        "num_high_degree_filtered": int(num_high_degree_filtered),
        "gold_entity_hit_eval_only": False,
        "num_gold_entities_in_universe_eval_only": 0,
        "universe_build_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    _LAST_DIAGNOSTICS["entity_universe"] = diag
    return out
