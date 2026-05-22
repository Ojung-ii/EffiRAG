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


def _stable_seed(config: Any, query_embedding: Any = None) -> int:
    base = _safe_int(_cfg(config, "phase8_pamae_random_seed", 42), 42)
    query_key = str(_cfg(config, "_phase8_query_key", "") or "")
    if not query_key and query_embedding is not None:
        vec = _normalize_vector(query_embedding)
        if vec is not None:
            query_key = ",".join(f"{float(x):.4f}" for x in vec[: min(16, vec.size)])
    digest = hashlib.sha1(f"{base}:{query_key}".encode("utf-8")).hexdigest()[:8]
    return int((base + int(digest, 16)) % (2**32 - 1))


def _candidate_by_id(universe: Sequence[EntityCandidate]) -> Dict[str, EntityCandidate]:
    return {str(c.entity_id): c for c in list(universe or [])}


def _candidate_distance(a: EntityCandidate, b: EntityCandidate) -> float:
    if str(a.entity_id) == str(b.entity_id):
        return 0.0
    av = _normalize_vector(a.embedding)
    bv = _normalize_vector(b.embedding)
    if av is None or bv is None or av.shape != bv.shape:
        return 1.0
    return float(max(0.0, min(2.0, 1.0 - float(np.dot(av, bv)))))


def _distance_matrix(candidates: Sequence[EntityCandidate]) -> np.ndarray:
    n = len(candidates)
    dmat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _candidate_distance(candidates[i], candidates[j])
            dmat[i, j] = d
            dmat[j, i] = d
    return dmat


def _normalize_weights(values: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray([max(0.0, float(v)) for v in values], dtype=np.float64)
    if arr.size <= 0:
        return None
    total = float(arr.sum())
    if total <= 1.0e-12:
        return None
    return arr / total


def _weighted_sample_indices(
    candidates: Sequence[EntityCandidate],
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


def _pam_approx_medoids(sample: Sequence[EntityCandidate], k: int) -> Tuple[List[int], Dict[str, float]]:
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
        for pos, old_idx in enumerate(list(medoids)):
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
    medoids = sorted(set(int(x) for x in medoids), key=lambda i: (-float(rel[i]), str(sample[i].entity_id)))
    return medoids, {
        "cost": float(current_cost),
        "iterations": int(iterations),
        "kmedoids_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def _seed_set_diversity(seeds: Sequence[MedoidSeed], universe_by_id: Dict[str, EntityCandidate]) -> float:
    if len(seeds) < 2:
        return 0.0
    vals = []
    for i in range(len(seeds)):
        a = universe_by_id.get(str(seeds[i].entity_id))
        if a is None:
            continue
        for j in range(i + 1, len(seeds)):
            b = universe_by_id.get(str(seeds[j].entity_id))
            if b is None:
                continue
            vals.append(min(1.0, _candidate_distance(a, b) / 2.0))
    return float(_mean(vals))


def sample_medoid_seed_sets(U_q, query_embedding, config) -> list[list[MedoidSeed]]:
    t_sampling = time.perf_counter()
    universe = list(U_q or [])
    k = max(1, _safe_int(_cfg(config, "phase8_pamae_k", 5), 5))
    per_k = max(1, _safe_int(_cfg(config, "phase8_pamae_sample_size_per_k", 40), 40))
    sample_size = max(1, int(k * per_k))
    num_samples = max(1, _safe_int(_cfg(config, "phase8_pamae_num_samples", 5), 5))
    mode = str(_cfg(config, "phase8_pamae_sampling", "query_weighted") or "query_weighted").strip().lower()
    seed = _stable_seed(config, query_embedding=query_embedding)
    rng = np.random.default_rng(seed)
    sampled_unique: Set[str] = set()
    seed_sets: List[List[MedoidSeed]] = []
    kmedoids_ms_total = 0.0
    universe_by_id = _candidate_by_id(universe)

    for sample_id in range(num_samples):
        sample_indices = _weighted_sample_indices(universe, sample_size=sample_size, rng=rng, mode=mode)
        sampled_unique.update(str(universe[i].entity_id) for i in sample_indices)
        sample = [universe[i] for i in sample_indices]
        medoid_indices, medoid_diag = _pam_approx_medoids(sample, k=k)
        kmedoids_ms_total += float(medoid_diag.get("kmedoids_ms", 0.0) or 0.0)
        seeds: List[MedoidSeed] = []
        for rank, idx in enumerate(medoid_indices):
            cand = sample[int(idx)]
            degree_pen = min(1.0, float(cand.degree) / float(max(1, _safe_int(_cfg(config, "phase8_entity_degree_cap", 200), 200))))
            score = _clip01(0.75 * cand.query_relevance + 0.25 * cand.graph_score - 0.10 * degree_pen)
            seeds.append(
                MedoidSeed(
                    entity_id=str(cand.entity_id),
                    score=float(score),
                    source_sample_id=int(sample_id),
                    refined_from=None,
                    diagnostics={
                        "sample_id": int(sample_id),
                        "rank": int(rank),
                        "name": str(cand.name),
                        "query_relevance": float(cand.query_relevance),
                        "graph_score": float(cand.graph_score),
                        "degree": int(cand.degree),
                        "source_tags": sorted(str(x) for x in set(cand.source_tags or set())),
                        "kmedoids_cost": float(medoid_diag.get("cost", 0.0)),
                    },
                )
            )
        seed_sets.append(seeds)

    seed_rows = []
    for sample_id, seeds in enumerate(seed_sets):
        seed_rows.append(
            {
                "stage": "seed_set_candidate",
                "sample_id": int(sample_id),
                "seed_entity_ids": [str(s.entity_id) for s in seeds],
                "seed_names": [str((universe_by_id.get(str(s.entity_id)) or EntityCandidate(str(s.entity_id), str(s.entity_id), None, None, 0.0, 0.0, 0, set())).name) for s in seeds],
                "seed_mean_relevance": float(_mean([(universe_by_id.get(str(s.entity_id)).query_relevance if universe_by_id.get(str(s.entity_id)) else 0.0) for s in seeds])),
                "seed_diversity": float(_seed_set_diversity(seeds, universe_by_id)),
                "seed_avg_degree": float(_mean([(universe_by_id.get(str(s.entity_id)).degree if universe_by_id.get(str(s.entity_id)) else 0) for s in seeds])),
                "seed_gold_hit_eval_only": False,
            }
        )
    _LAST_DIAGNOSTICS["pamae_sampling"] = {
        "stage": "pamae_sampling",
        "k": int(k),
        "sample_size": int(min(sample_size, len(universe))),
        "num_samples": int(num_samples),
        "sampling_mode": str(mode),
        "sample_seed": int(seed),
        "num_unique_sampled_entities": int(len(sampled_unique)),
        "sampling_ms": float((time.perf_counter() - t_sampling) * 1000.0),
        "kmedoids_ms": float(kmedoids_ms_total),
    }
    _LAST_DIAGNOSTICS["seed_set_candidates"] = seed_rows
    return seed_sets


def _seed_set_components(
    seeds: Sequence[MedoidSeed],
    universe: Sequence[EntityCandidate],
    config: Any,
) -> Dict[str, float]:
    by_id = _candidate_by_id(universe)
    seed_candidates = [by_id[str(s.entity_id)] for s in list(seeds or []) if str(s.entity_id) in by_id]
    if not seed_candidates:
        return {
            "score": 0.0,
            "mean_relevance": 0.0,
            "coverage": 0.0,
            "diversity": 0.0,
            "hub_penalty": 0.0,
        }
    mean_relevance = _clip01(_mean([c.query_relevance for c in seed_candidates]))
    weights = _normalize_weights([0.001 + max(0.0, c.query_relevance) for c in universe])
    if weights is None:
        weights = np.ones(len(universe), dtype=np.float64) / float(max(1, len(universe)))
    min_dist = []
    for cand in universe:
        min_dist.append(min((_candidate_distance(cand, seed) for seed in seed_candidates), default=1.0))
    weighted_avg_dist = float(np.dot(np.asarray(min_dist, dtype=np.float64), weights)) if min_dist else 1.0
    coverage = _clip01(1.0 - min(1.0, weighted_avg_dist / 2.0))
    diversity = _seed_set_diversity(list(seeds or []), by_id)
    degree_cap = max(1, _safe_int(_cfg(config, "phase8_entity_degree_cap", 200), 200))
    hub_penalty = _clip01(_mean([min(1.0, float(c.degree) / float(degree_cap)) for c in seed_candidates]))
    score = float(mean_relevance + coverage + diversity - hub_penalty)
    return {
        "score": float(score),
        "mean_relevance": float(mean_relevance),
        "coverage": float(coverage),
        "diversity": float(diversity),
        "hub_penalty": float(hub_penalty),
    }


def select_best_seed_set(seed_sets, U_q, query_embedding, config) -> list[MedoidSeed]:
    t0 = time.perf_counter()
    universe = list(U_q or [])
    candidate_sets = [list(s or []) for s in list(seed_sets or []) if list(s or [])]
    if not candidate_sets:
        _LAST_DIAGNOSTICS["best_seed_selection"] = {
            "stage": "best_seed_selection",
            "best_sample_id": -1,
            "best_seed_entity_ids": [],
            "best_seed_names": [],
            "best_seed_score": 0.0,
            "best_mean_relevance": 0.0,
            "best_coverage": 0.0,
            "best_diversity": 0.0,
            "best_hub_penalty": 0.0,
            "seed_gold_hit_eval_only": False,
            "seed_evidence_gold_hit_eval_only": False,
            "seed_selection_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return []
    by_id = _candidate_by_id(universe)
    scored: List[Tuple[float, int, List[MedoidSeed], Dict[str, float]]] = []
    for idx, seeds in enumerate(candidate_sets):
        comp = _seed_set_components(seeds, universe=universe, config=config)
        scored.append((float(comp.get("score", 0.0)), int(idx), seeds, comp))
    scored.sort(key=lambda row: (float(row[0]), -int(row[1])), reverse=True)
    _score, best_idx, best, comp = scored[0]
    best_out: List[MedoidSeed] = []
    for seed in best:
        diag = dict(seed.diagnostics or {})
        diag.update(
            {
                "best_seed_set": True,
                "best_seed_score": float(comp.get("score", 0.0)),
                "best_mean_relevance": float(comp.get("mean_relevance", 0.0)),
                "best_coverage": float(comp.get("coverage", 0.0)),
                "best_diversity": float(comp.get("diversity", 0.0)),
                "best_hub_penalty": float(comp.get("hub_penalty", 0.0)),
            }
        )
        best_out.append(
            MedoidSeed(
                entity_id=str(seed.entity_id),
                score=float(seed.score),
                source_sample_id=int(seed.source_sample_id),
                refined_from=seed.refined_from,
                diagnostics=diag,
            )
        )
    _LAST_DIAGNOSTICS["best_seed_selection"] = {
        "stage": "best_seed_selection",
        "best_sample_id": int(best_idx),
        "best_seed_entity_ids": [str(s.entity_id) for s in best_out],
        "best_seed_names": [str(by_id[str(s.entity_id)].name) for s in best_out if str(s.entity_id) in by_id],
        "best_seed_score": float(comp.get("score", 0.0)),
        "best_mean_relevance": float(comp.get("mean_relevance", 0.0)),
        "best_coverage": float(comp.get("coverage", 0.0)),
        "best_diversity": float(comp.get("diversity", 0.0)),
        "best_hub_penalty": float(comp.get("hub_penalty", 0.0)),
        "seed_gold_hit_eval_only": False,
        "seed_evidence_gold_hit_eval_only": False,
        "seed_selection_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    return best_out


def _graph_and_semantic(G_q: Any) -> Tuple[nx.Graph, Dict[str, Any]]:
    graph = _as_graph(G_q)
    semantic = _semantic_state(G_q, G_q if isinstance(G_q, dict) else {})
    return graph, semantic


def _candidate_for_entity(
    entity_id: str,
    universe_by_id: Dict[str, EntityCandidate],
    graph: nx.Graph,
    semantic: Dict[str, Any],
    config: Any,
    query_embedding: Any,
    source: str,
) -> Optional[EntityCandidate]:
    eid = str(entity_id or "")
    if eid in universe_by_id:
        return universe_by_id[eid]
    if not eid or eid not in graph or _node_type(graph, eid) != "entity":
        return None
    degree = int(graph.degree(eid)) if eid in graph else 0
    embedding_id, vec = _best_entity_embedding(eid, graph, semantic, config)
    qrel = 0.0
    qvec = _normalize_vector(query_embedding)
    if qvec is not None and vec is not None:
        qrel = _clip01((_cosine(qvec, vec) + 1.0) / 2.0)
    return EntityCandidate(
        entity_id=str(eid),
        name=_entity_name(eid, graph),
        title=_entity_title(eid, graph),
        embedding_id=embedding_id,
        query_relevance=float(qrel),
        graph_score=0.0,
        degree=int(degree),
        source_tags={str(source)},
        embedding=vec,
    )


def _assigned_entities_by_seed(
    seeds: Sequence[MedoidSeed],
    universe: Sequence[EntityCandidate],
) -> Dict[str, Set[str]]:
    by_id = _candidate_by_id(universe)
    seed_cands = [by_id[str(seed.entity_id)] for seed in list(seeds or []) if str(seed.entity_id) in by_id]
    out: Dict[str, Set[str]] = {str(seed.entity_id): set() for seed in list(seeds or [])}
    if not seed_cands:
        return out
    for cand in list(universe or []):
        best_seed = min(seed_cands, key=lambda seed: (_candidate_distance(cand, seed), str(seed.entity_id)))
        out.setdefault(str(best_seed.entity_id), set()).add(str(cand.entity_id))
    return out


def _entity_neighborhood(
    graph: nx.Graph,
    start: str,
    hops: int,
    degree_cap: int,
    limit: int,
) -> Set[str]:
    src = str(start or "")
    if not src or src not in graph:
        return set()
    max_hops = max(0, int(hops))
    cap = max(1, int(degree_cap))
    frontier = [(src, 0)]
    seen = {src}
    out: Set[str] = set()
    while frontier and len(out) < max(1, int(limit)):
        node, depth = frontier.pop(0)
        if depth >= max_hops:
            continue
        for nbr in sorted(str(n) for n in graph.neighbors(node)):
            if nbr in seen:
                continue
            seen.add(nbr)
            ntype = _node_type(graph, nbr)
            if ntype == "entity":
                if int(graph.degree(nbr)) <= cap:
                    out.add(nbr)
                if len(out) >= max(1, int(limit)):
                    break
            if int(graph.degree(nbr)) <= cap or ntype in {"sentence", "chunk", "passage", "document"}:
                frontier.append((nbr, depth + 1))
        if len(out) >= max(1, int(limit)):
            break
    return out


def _connectivity_to_anchors_and_seeds(
    graph: nx.Graph,
    entity_id: str,
    anchors: Sequence[str],
    seed_ids: Sequence[str],
    max_hops: int,
) -> float:
    targets = [str(x) for x in list(anchors or []) + list(seed_ids or []) if str(x) and str(x) in graph and str(x) != str(entity_id)]
    if not targets or str(entity_id) not in graph:
        return 0.0
    vals = []
    cutoff = max(1, int(max_hops))
    for target in targets[:32]:
        try:
            d = nx.shortest_path_length(graph, str(entity_id), str(target))
        except Exception:
            continue
        if int(d) <= cutoff:
            vals.append(1.0 / (1.0 + float(d)))
    return _clip01(max(vals) if vals else 0.0)


def _local_evidence_support(graph: nx.Graph, entity_id: str) -> float:
    if str(entity_id) not in graph:
        return 0.0
    sent = 0
    carrier = 0
    for nbr in graph.neighbors(str(entity_id)):
        ntype = _node_type(graph, str(nbr))
        if ntype == "sentence":
            sent += 1
        elif ntype in {"chunk", "passage", "document"}:
            carrier += 1
    return _clip01((float(sent) + 0.5 * float(carrier)) / 8.0)


def _redundancy_to_other_seeds(
    candidate: EntityCandidate,
    other_seed_ids: Sequence[str],
    candidates_by_id: Dict[str, EntityCandidate],
) -> float:
    vals = []
    for sid in list(other_seed_ids or []):
        other = candidates_by_id.get(str(sid))
        if other is None:
            continue
        dist = _candidate_distance(candidate, other)
        vals.append(_clip01(1.0 - min(1.0, dist / 2.0)))
    return float(max(vals) if vals else 0.0)


def refine_medoid_seeds(best_seed_set, U_q, G_q, query_embedding, anchors, config) -> list[MedoidSeed]:
    t0 = time.perf_counter()
    enabled = _cfg_bool(config, "phase8_refine_enabled", True)
    before = list(best_seed_set or [])
    if not before:
        _LAST_DIAGNOSTICS["refinement"] = {
            "stage": "refinement",
            "enabled": bool(enabled),
            "num_refined_seeds": 0,
            "num_changed_seeds": 0,
            "avg_refine_candidate_count": 0.0,
            "before_seed_score": 0.0,
            "after_seed_score": 0.0,
            "before_seed_gold_hit_eval_only": False,
            "after_seed_gold_hit_eval_only": False,
            "before_seed_evidence_gold_hit_eval_only": False,
            "after_seed_evidence_gold_hit_eval_only": False,
            "refinement_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return []
    if not enabled:
        _LAST_DIAGNOSTICS["refinement"] = {
            "stage": "refinement",
            "enabled": False,
            "num_refined_seeds": int(len(before)),
            "num_changed_seeds": 0,
            "avg_refine_candidate_count": 0.0,
            "before_seed_score": float(_seed_set_components(before, list(U_q or []), config).get("score", 0.0)),
            "after_seed_score": float(_seed_set_components(before, list(U_q or []), config).get("score", 0.0)),
            "before_seed_gold_hit_eval_only": False,
            "after_seed_gold_hit_eval_only": False,
            "before_seed_evidence_gold_hit_eval_only": False,
            "after_seed_evidence_gold_hit_eval_only": False,
            "refinement_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return before

    graph, semantic = _graph_and_semantic(G_q)
    universe = list(U_q or [])
    universe_by_id = _candidate_by_id(universe)
    assigned = _assigned_entities_by_seed(before, universe)
    hops = max(0, _safe_int(_cfg(config, "phase8_refine_hops", 2), 2))
    max_candidates = max(1, _safe_int(_cfg(config, "phase8_refine_max_candidates_per_seed", 128), 128))
    degree_cap = max(1, _safe_int(_cfg(config, "phase8_refine_degree_cap", 100), 100))
    candidates_seen: Dict[str, EntityCandidate] = dict(universe_by_id)
    refined: List[MedoidSeed] = []
    candidate_counts: List[int] = []
    changed = 0

    for seed in before:
        seed_id = str(seed.entity_id)
        raw_candidate_ids = set(assigned.get(seed_id, set()))
        raw_candidate_ids.add(seed_id)
        raw_candidate_ids.update(_entity_neighborhood(graph, seed_id, hops=hops, degree_cap=degree_cap, limit=max_candidates))
        candidate_rows: List[EntityCandidate] = []
        for eid in sorted(raw_candidate_ids):
            cand = _candidate_for_entity(
                eid,
                universe_by_id=universe_by_id,
                graph=graph,
                semantic=semantic,
                config=config,
                query_embedding=query_embedding,
                source="refine_neighborhood",
            )
            if cand is None or int(cand.degree) > degree_cap:
                continue
            candidates_seen[str(cand.entity_id)] = cand
            candidate_rows.append(cand)
        candidate_rows.sort(
            key=lambda c: (
                float(c.query_relevance),
                float(c.graph_score),
                -float(c.degree),
                str(c.entity_id),
            ),
            reverse=True,
        )
        candidate_rows = candidate_rows[:max_candidates]
        candidate_counts.append(int(len(candidate_rows)))
        other_seed_ids = [str(s.entity_id) for s in before if str(s.entity_id) != seed_id]
        best_cand = candidates_seen.get(seed_id)
        best_score = -1.0e9
        for cand in candidate_rows:
            connectivity = _connectivity_to_anchors_and_seeds(
                graph=graph,
                entity_id=str(cand.entity_id),
                anchors=list(anchors or []),
                seed_ids=other_seed_ids,
                max_hops=max(1, hops),
            )
            support = _local_evidence_support(graph, str(cand.entity_id))
            hub = min(1.0, float(cand.degree) / float(max(1, degree_cap)))
            redundancy = _redundancy_to_other_seeds(cand, other_seed_ids, candidates_seen)
            score = float(cand.query_relevance + connectivity + support - hub - redundancy)
            if score > best_score or (abs(score - best_score) <= 1.0e-12 and str(cand.entity_id) < str(best_cand.entity_id if best_cand else "")):
                best_score = score
                best_cand = cand
        if best_cand is None:
            refined.append(seed)
            continue
        if str(best_cand.entity_id) != seed_id:
            changed += 1
        diag = dict(seed.diagnostics or {})
        diag.update(
            {
                "refined": True,
                "refined_from": str(seed_id),
                "refine_candidate_count": int(len(candidate_rows)),
                "refine_score": float(best_score),
                "query_relevance": float(best_cand.query_relevance),
                "degree": int(best_cand.degree),
            }
        )
        refined.append(
            MedoidSeed(
                entity_id=str(best_cand.entity_id),
                score=float(best_score),
                source_sample_id=int(seed.source_sample_id),
                refined_from=(str(seed_id) if str(best_cand.entity_id) != seed_id else None),
                diagnostics=diag,
            )
        )

    before_comp = _seed_set_components(before, universe=universe, config=config)
    after_universe = list(candidates_seen.values())
    after_comp = _seed_set_components(refined, universe=after_universe, config=config)
    _LAST_DIAGNOSTICS["refinement"] = {
        "stage": "refinement",
        "enabled": True,
        "num_refined_seeds": int(len(refined)),
        "num_changed_seeds": int(changed),
        "avg_refine_candidate_count": float(_mean(candidate_counts)),
        "before_seed_score": float(before_comp.get("score", 0.0)),
        "after_seed_score": float(after_comp.get("score", 0.0)),
        "before_seed_gold_hit_eval_only": False,
        "after_seed_gold_hit_eval_only": False,
        "before_seed_evidence_gold_hit_eval_only": False,
        "after_seed_evidence_gold_hit_eval_only": False,
        "refinement_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    return refined
