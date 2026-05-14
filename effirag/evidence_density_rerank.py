from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .utils import content_tokens


def _ordered_unique(ids: Sequence[str], texts: Sequence[str]) -> Tuple[List[str], List[str]]:
    seen = set()
    out_ids: List[str] = []
    out_texts: List[str] = []
    for idx, sid in enumerate(list(ids or [])):
        token = str(sid or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out_ids.append(token)
        out_texts.append(str(texts[idx] if idx < len(texts) else ""))
    return out_ids, out_texts


def _normalize(values: Mapping[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    scores = [float(v) for v in values.values()]
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        if hi > 0.0:
            return {str(k): 1.0 for k in values.keys()}
        return {str(k): 0.0 for k in values.keys()}
    span = float(hi - lo)
    return {str(k): float((float(v) - lo) / span) for k, v in values.items()}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(len(a.intersection(b)) / float(union))


def _source_key(sentence_id: str) -> str:
    raw = str(sentence_id or "")
    if "::" in raw:
        return raw.split("::", 1)[0].strip().lower()
    return raw.strip().lower()


def _token_penalty(token_count: int) -> float:
    # Sentence-level evidence above ~80 tokens is treated as high-cost.
    return min(1.0, float(max(1, int(token_count))) / 80.0)


def rerank_evidence_density(
    *,
    question_text: str,
    selected_sentence_ids: Sequence[str],
    selected_sentences: Sequence[str],
    sentence_feature_table: Mapping[str, Mapping[str, Any]] | None,
    cfg: Any,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    enabled = bool(getattr(cfg, "evidence_density_rerank_enabled", False))
    base_ids, base_texts = _ordered_unique(list(selected_sentence_ids or []), list(selected_sentences or []))
    if (not enabled) or (not base_ids):
        return list(base_ids), list(base_texts), {
            "enabled": bool(enabled),
            "applied": False,
            "reason": "disabled_or_empty",
        }

    features = dict(sentence_feature_table or {})
    text_map = {sid: txt for sid, txt in zip(base_ids, base_texts)}
    token_map = {sid: set(content_tokens(text_map.get(sid, ""))) for sid in base_ids}
    token_count_map = {sid: len(token_map.get(sid, set())) for sid in base_ids}
    source_map = {sid: _source_key(sid) for sid in base_ids}
    query_tokens = set(content_tokens(str(question_text or "")))

    semantic_raw = {
        sid: float((features.get(sid, {}) or {}).get("base_retrieval_score", 0.0))
        for sid in base_ids
    }
    semantic_norm = _normalize(semantic_raw)

    bridge_path_enabled = bool(getattr(cfg, "bridge_path_utility_enabled", False))
    bridge_path_score: Dict[str, float] = {}
    for sid in base_ids:
        feat = dict(features.get(sid, {}) or {})
        rank = int(feat.get("best_corridor_rank", 10**9) or 10**9)
        rank_bonus = 0.0 if rank >= 10**8 else 1.0 / float(rank + 1)
        corr_score = max(0.0, float(feat.get("best_corridor_score", 0.0)))
        utility = (
            0.40 * float(1.0 if feat.get("is_main_candidate", False) else 0.0)
            + 0.25 * float(1.0 if feat.get("is_support_candidate", False) else 0.0)
            + 0.20 * float(1.0 if feat.get("is_connector_adjacent", False) else 0.0)
            + 0.10 * float(rank_bonus)
            + 0.05 * float(corr_score)
        )
        bridge_path_score[sid] = float(max(0.0, min(1.0, utility))) if bridge_path_enabled else 0.0

    coverage_enabled = bool(getattr(cfg, "coverage_gain_enabled", False))
    redundancy_enabled = bool(getattr(cfg, "redundancy_penalty_enabled", False))
    token_penalty_enabled = bool(getattr(cfg, "token_cost_penalty_enabled", False))
    budget_awareness_enabled = bool(getattr(cfg, "density_budget_awareness_enabled", False))

    w_sem = max(0.0, float(getattr(cfg, "density_semantic_weight", 0.42)))
    w_bridge = max(0.0, float(getattr(cfg, "density_bridge_path_weight", 0.24)))
    w_cov = max(0.0, float(getattr(cfg, "density_coverage_weight", 0.24)))
    w_red = max(0.0, float(getattr(cfg, "density_redundancy_weight", 0.18)))
    w_tok = max(0.0, float(getattr(cfg, "density_token_cost_weight", 0.22)))

    max_selected = max(1, int(getattr(cfg, "max_selected_candidates", len(base_ids))))
    max_rendered = max(1, int(getattr(cfg, "max_rendered_candidates", len(base_ids))))
    max_count = min(len(base_ids), max_selected, max_rendered)
    max_tokens = max(1, int(getattr(cfg, "max_rendered_tokens", 340)))
    min_keep = max(1, int(getattr(cfg, "min_render_topn", 1)))

    remaining = list(base_ids)
    selected: List[str] = []
    selected_sources: Set[str] = set()
    covered_query_tokens: Set[str] = set()
    covered_content_tokens: Set[str] = set()
    covered_corridors: Set[str] = set()
    total_tokens = 0

    score_trace: Dict[str, Dict[str, float]] = {}
    budget_skips = 0

    while remaining and len(selected) < max_count:
        best_sid = None
        best_score = float("-inf")
        best_components: Dict[str, float] = {}

        for sid in list(remaining):
            sent_tokens = token_map.get(sid, set())
            sent_token_count = int(token_count_map.get(sid, 0))
            feat = dict(features.get(sid, {}) or {})

            if budget_awareness_enabled and len(selected) >= min_keep:
                if (total_tokens + sent_token_count) > max_tokens:
                    continue

            semantic_component = float(semantic_norm.get(sid, 0.0))
            bridge_component = float(bridge_path_score.get(sid, 0.0))

            query_gain = 0.0
            lexical_gain = 0.0
            source_gain = 0.0
            corridor_gain = 0.0
            if coverage_enabled:
                if query_tokens:
                    query_gain = float(len((sent_tokens.intersection(query_tokens)) - covered_query_tokens)) / float(
                        max(1, len(query_tokens))
                    )
                lexical_new = [tok for tok in sent_tokens if len(tok) >= 4 and tok not in covered_content_tokens]
                lexical_gain = float(min(4, len(lexical_new))) / 4.0
                source_gain = 1.0 if source_map.get(sid, "") and source_map[sid] not in selected_sources else 0.0
                corridor_ids = list(feat.get("corridor_ids", []) or [])
                new_corridors = [cid for cid in corridor_ids if str(cid) not in covered_corridors]
                corridor_gain = 1.0 if new_corridors else 0.0
            coverage_component = (
                0.45 * query_gain + 0.30 * lexical_gain + 0.15 * source_gain + 0.10 * corridor_gain
            ) if coverage_enabled else 0.0

            redundancy_component = 0.0
            if redundancy_enabled and selected:
                max_overlap = 0.0
                for prev_sid in selected:
                    prev_tokens = token_map.get(prev_sid, set())
                    max_overlap = max(max_overlap, _jaccard(sent_tokens, prev_tokens))
                same_source = 1.0 if source_map.get(sid, "") and source_map[sid] in selected_sources else 0.0
                redundancy_component = 0.80 * float(max_overlap) + 0.20 * float(same_source)

            token_cost_component = _token_penalty(sent_token_count) if token_penalty_enabled else 0.0

            score = (
                w_sem * semantic_component
                + w_bridge * bridge_component
                + w_cov * coverage_component
                - w_red * redundancy_component
                - w_tok * token_cost_component
            )

            if score > best_score:
                best_score = float(score)
                best_sid = sid
                best_components = {
                    "semantic": float(semantic_component),
                    "bridge_path": float(bridge_component),
                    "coverage_gain": float(coverage_component),
                    "redundancy_penalty": float(redundancy_component),
                    "token_cost_penalty": float(token_cost_component),
                    "final_score": float(score),
                }

        if best_sid is None:
            # Budget-aware skip path: no remaining candidate can fit.
            budget_skips += int(len(remaining))
            break

        remaining.remove(best_sid)
        selected.append(best_sid)
        total_tokens += int(token_count_map.get(best_sid, 0))
        selected_sources.add(source_map.get(best_sid, ""))
        best_tokens = token_map.get(best_sid, set())
        covered_query_tokens.update(best_tokens.intersection(query_tokens))
        covered_content_tokens.update(best_tokens)
        for cid in list((features.get(best_sid, {}) or {}).get("corridor_ids", []) or []):
            covered_corridors.add(str(cid))
        score_trace[best_sid] = dict(best_components)

    selected_texts = [text_map.get(sid, "") for sid in selected]
    diagnostics = {
        "enabled": True,
        "applied": bool(selected != base_ids),
        "candidate_count_before": int(len(base_ids)),
        "selected_count_after": int(len(selected)),
        "tokens_after": int(total_tokens),
        "max_rendered_tokens": int(max_tokens),
        "max_selected_candidates": int(max_selected),
        "max_rendered_candidates": int(max_rendered),
        "coverage_gain_enabled": bool(coverage_enabled),
        "bridge_path_utility_enabled": bool(bridge_path_enabled),
        "redundancy_penalty_enabled": bool(redundancy_enabled),
        "token_cost_penalty_enabled": bool(token_penalty_enabled),
        "density_budget_awareness_enabled": bool(budget_awareness_enabled),
        "weights": {
            "density_semantic_weight": float(w_sem),
            "density_bridge_path_weight": float(w_bridge),
            "density_coverage_weight": float(w_cov),
            "density_redundancy_weight": float(w_red),
            "density_token_cost_weight": float(w_tok),
        },
        "budget_skips": int(budget_skips),
        "score_trace": dict(score_trace),
    }
    return selected, selected_texts, diagnostics
