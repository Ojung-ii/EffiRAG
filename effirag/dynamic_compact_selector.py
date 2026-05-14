from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .utils import content_tokens


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]*", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


@dataclass(frozen=True)
class DynamicCompactCandidate:
    candidate_id: str
    text: str
    semantic_score: float = 0.0
    bridge_score: float = 0.0
    path_score: float = 0.0
    rank: int = 0
    source_doc_id: str = ""
    entity_ids: Tuple[str, ...] = field(default_factory=tuple)
    coverage_terms: Tuple[str, ...] = field(default_factory=tuple)
    token_count: int = 0


@dataclass(frozen=True)
class DynamicCompactSelection:
    selected_ids: List[str]
    score_by_id: Dict[str, float]
    diagnostics: Dict[str, Any]


def _terms(text: str) -> Set[str]:
    out: Set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "").lower()):
        tok = raw.strip("-_")
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        out.add(tok)
    return out


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


def _candidate_from_mapping(raw: Mapping[str, Any]) -> DynamicCompactCandidate:
    candidate_id = str(raw.get("candidate_id", raw.get("id", "")) or "")
    text = str(raw.get("text", "") or "")
    raw_entities = raw.get("entity_ids", ()) or ()
    if isinstance(raw_entities, str):
        entities = (raw_entities,)
    else:
        entities = tuple(str(x) for x in raw_entities if str(x or "").strip())
    raw_terms = raw.get("coverage_terms", ()) or ()
    if isinstance(raw_terms, str):
        coverage_terms = tuple(sorted(_terms(raw_terms)))
    else:
        coverage_terms = tuple(str(x).lower() for x in raw_terms if str(x or "").strip())
    token_count = _safe_int(raw.get("token_count", 0), 0)
    if token_count <= 0:
        token_count = len(content_tokens(text))
    return DynamicCompactCandidate(
        candidate_id=candidate_id,
        text=text,
        semantic_score=_safe_float(raw.get("semantic_score", raw.get("score", 0.0)), 0.0),
        bridge_score=_safe_float(raw.get("bridge_score", 0.0), 0.0),
        path_score=_safe_float(raw.get("path_score", 0.0), 0.0),
        rank=_safe_int(raw.get("rank", 0), 0),
        source_doc_id=str(raw.get("source_doc_id", "") or ""),
        entity_ids=entities,
        coverage_terms=coverage_terms,
        token_count=int(token_count),
    )


def _normalize_candidates(candidates: Sequence[DynamicCompactCandidate | Mapping[str, Any]]) -> List[DynamicCompactCandidate]:
    out: List[DynamicCompactCandidate] = []
    seen: Set[str] = set()
    for idx, raw in enumerate(candidates):
        item = raw if isinstance(raw, DynamicCompactCandidate) else _candidate_from_mapping(raw)
        cid = str(item.candidate_id or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if item.rank <= 0:
            item = DynamicCompactCandidate(
                candidate_id=item.candidate_id,
                text=item.text,
                semantic_score=item.semantic_score,
                bridge_score=item.bridge_score,
                path_score=item.path_score,
                rank=idx + 1,
                source_doc_id=item.source_doc_id,
                entity_ids=item.entity_ids,
                coverage_terms=item.coverage_terms,
                token_count=item.token_count,
            )
        out.append(item)
    return out


def _minmax(values: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 1.0
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) <= 1.0e-12:
        return lo, lo + 1.0
    return lo, hi


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(value) - float(lo)) / (float(hi) - float(lo))))


def _coverage_terms(candidate: DynamicCompactCandidate, query_terms: Set[str]) -> Set[str]:
    explicit = set(str(x).lower() for x in candidate.coverage_terms if str(x or "").strip())
    entity_terms = set()
    for ent in candidate.entity_ids:
        entity_terms.update(_terms(ent))
    source_terms = _terms(candidate.source_doc_id)
    text_terms = _terms(candidate.text)
    query_overlap = text_terms.intersection(query_terms)

    terms = set()
    terms.update(explicit)
    terms.update(entity_terms)
    terms.update(source_terms)
    terms.update(query_overlap)
    if not terms:
        # Fallback remains answer/gold-free: it only uses salient lexical terms from the candidate text.
        terms.update(sorted(text_terms)[:8])
    return terms


def _text_terms(candidate: DynamicCompactCandidate) -> Set[str]:
    terms = _terms(candidate.text)
    terms.update(_terms(candidate.source_doc_id))
    return terms


def select_dynamic_compact_evidence(
    candidates: Sequence[DynamicCompactCandidate | Mapping[str, Any]],
    *,
    question_text: str = "",
    max_render_topn: int = 24,
    min_render_topn: int = 6,
    target_prompt_tokens: int = 600,
    max_prompt_tokens: int = 700,
    coverage_gain_enabled: bool = True,
    redundancy_penalty_enabled: bool = True,
    bridge_preserve_enabled: bool = True,
    path_preserve_enabled: bool = True,
    adaptive_stop_enabled: bool = True,
    coverage_gain_threshold: float = 0.05,
    bridge_score_threshold: float = 0.35,
    redundancy_threshold: float = 0.62,
    marginal_gain_threshold: float = 0.08,
) -> DynamicCompactSelection:
    """Greedy dataset-agnostic compact evidence selector.

    The selector intentionally uses only query/candidate/retrieval-derived signals.
    It does not inspect dataset names, gold supports, or answer strings.
    """

    items = _normalize_candidates(candidates)
    if not items:
        return DynamicCompactSelection(
            selected_ids=[],
            score_by_id={},
            diagnostics={
                "enabled": True,
                "num_candidates": 0,
                "num_selected": 0,
                "prompt_tokens": 0,
                "early_stop_reason": "no_candidates",
                "coverage_gain_sum": 0.0,
                "redundancy_penalty_sum": 0.0,
                "bridge_preserved": False,
                "bridge_preserve_rate": 0.0,
            },
        )

    max_n = max(1, int(max_render_topn or 1))
    min_n = max(0, min(int(min_render_topn or 0), max_n))
    target_tokens = max(1, int(target_prompt_tokens or max_prompt_tokens or 700))
    hard_token_budget = max(target_tokens, int(max_prompt_tokens or target_tokens))
    query_terms = _terms(question_text)
    sem_lo, sem_hi = _minmax(item.semantic_score for item in items)

    coverage_by_id = {item.candidate_id: _coverage_terms(item, query_terms) for item in items}
    text_terms_by_id = {item.candidate_id: _text_terms(item) for item in items}
    by_id = {item.candidate_id: item for item in items}

    selected: List[str] = []
    selected_text_terms: List[Set[str]] = []
    selected_sources: Dict[str, int] = {}
    covered_terms: Set[str] = set()
    score_by_id: Dict[str, float] = {}
    coverage_gain_sum = 0.0
    redundancy_penalty_sum = 0.0
    bridge_preserved_count = 0
    estimated_prompt_tokens = 0
    early_stop_reason = "max_render_topn"
    remaining = list(items)

    def estimate_added_tokens(item: DynamicCompactCandidate) -> int:
        # Include a small per-item rendering overhead so the compact stop rule better tracks prompt cost.
        return max(1, int(item.token_count or len(content_tokens(item.text)))) + 8

    def candidate_score(item: DynamicCompactCandidate) -> Tuple[float, Dict[str, float]]:
        cid = item.candidate_id
        semantic = _norm(float(item.semantic_score), sem_lo, sem_hi)
        bridge = max(0.0, min(1.0, float(item.bridge_score)))
        path = max(0.0, min(1.0, float(item.path_score)))
        cov_terms = coverage_by_id.get(cid, set())
        new_terms = cov_terms.difference(covered_terms)
        coverage_gain = 0.0
        if coverage_gain_enabled and cov_terms:
            coverage_gain = min(1.0, float(len(new_terms)) / float(max(1, min(8, len(cov_terms)))))

        redundancy = 0.0
        if redundancy_penalty_enabled and selected_text_terms:
            cand_terms = text_terms_by_id.get(cid, set())
            if cand_terms:
                redundancy = max(
                    (len(cand_terms.intersection(prev)) / float(max(1, len(cand_terms.union(prev)))))
                    for prev in selected_text_terms
                )
            source_repeats = int(selected_sources.get(item.source_doc_id, 0)) if item.source_doc_id else 0
            if source_repeats > 0:
                redundancy = min(1.0, redundancy + 0.12 * float(source_repeats))
            if redundancy >= float(redundancy_threshold):
                redundancy = min(1.0, redundancy + 0.08)

        score = (
            0.48 * semantic
            + 0.24 * bridge
            + 0.12 * path
            + 0.34 * coverage_gain
            - 0.28 * redundancy
            + 0.04 * (1.0 / float(max(1, item.rank)))
        )
        return float(score), {
            "semantic": float(semantic),
            "bridge": float(bridge),
            "path": float(path),
            "coverage_gain": float(coverage_gain),
            "redundancy": float(redundancy),
            "new_terms": float(len(new_terms)),
        }

    while remaining and len(selected) < max_n:
        scored = []
        for item in remaining:
            score, components = candidate_score(item)
            scored.append((score, -int(item.rank), item.candidate_id, components))
        scored.sort(reverse=True)
        best_score, _neg_rank, best_id, best_components = scored[0]
        best = by_id[best_id]
        added_tokens = estimate_added_tokens(best)
        would_exceed = (estimated_prompt_tokens + added_tokens) > hard_token_budget
        if would_exceed and len(selected) >= min_n:
            early_stop_reason = "max_prompt_tokens"
            break

        needs_bridge = bool(
            bridge_preserve_enabled
            and bridge_preserved_count == 0
            and float(best_components.get("bridge", 0.0)) >= float(bridge_score_threshold)
        )
        needs_path = bool(
            path_preserve_enabled
            and not any(float(candidate_score(by_id[sid])[1].get("path", 0.0)) >= float(bridge_score_threshold) for sid in selected)
            and float(best_components.get("path", 0.0)) >= float(bridge_score_threshold)
        )
        below_gain = float(best_score) < float(marginal_gain_threshold)
        coverage_saturated = bool(
            coverage_gain_enabled
            and len(selected) >= min_n
            and float(best_components.get("coverage_gain", 0.0)) < float(coverage_gain_threshold)
            and not needs_bridge
            and not needs_path
        )
        if adaptive_stop_enabled and len(selected) >= min_n and (below_gain or coverage_saturated):
            early_stop_reason = "marginal_gain" if below_gain else "coverage_saturated"
            break

        selected.append(best_id)
        selected_text_terms.append(text_terms_by_id.get(best_id, set()))
        covered_terms.update(coverage_by_id.get(best_id, set()))
        selected_sources[best.source_doc_id] = int(selected_sources.get(best.source_doc_id, 0)) + 1
        score_by_id[best_id] = float(best_score)
        coverage_gain_sum += float(best_components.get("coverage_gain", 0.0))
        redundancy_penalty_sum += float(best_components.get("redundancy", 0.0))
        if float(best_components.get("bridge", 0.0)) >= float(bridge_score_threshold):
            bridge_preserved_count += 1
        estimated_prompt_tokens += int(added_tokens)
        remaining = [item for item in remaining if item.candidate_id != best_id]

    if len(selected) >= max_n:
        early_stop_reason = "max_render_topn"
    elif not remaining and early_stop_reason == "max_render_topn":
        early_stop_reason = "candidate_exhausted"

    diagnostics = {
        "enabled": True,
        "num_candidates": int(len(items)),
        "num_selected": int(len(selected)),
        "prompt_tokens": int(estimated_prompt_tokens),
        "target_prompt_tokens": int(target_tokens),
        "max_prompt_tokens": int(hard_token_budget),
        "early_stop_reason": str(early_stop_reason),
        "coverage_gain_sum": float(coverage_gain_sum),
        "redundancy_penalty_sum": float(redundancy_penalty_sum),
        "bridge_preserved": bool(bridge_preserved_count > 0),
        "bridge_preserved_count": int(bridge_preserved_count),
        "bridge_preserve_rate": float(bridge_preserved_count / max(1, len(selected))),
        "covered_term_count": int(len(covered_terms)),
        "adaptive_stop": bool(adaptive_stop_enabled and early_stop_reason in {"marginal_gain", "coverage_saturated"}),
    }
    return DynamicCompactSelection(selected_ids=list(selected), score_by_id=score_by_id, diagnostics=diagnostics)
