from __future__ import annotations

from dataclasses import dataclass
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
    nums = [float(v) for v in values.values()]
    lo = min(nums)
    hi = max(nums)
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


def _bridge_path_utility(feat: Mapping[str, Any], enabled: bool) -> float:
    if not enabled:
        return 0.0
    rank = int(feat.get("best_corridor_rank", 10**9) or 10**9)
    rank_bonus = 0.0 if rank >= 10**8 else 1.0 / float(rank + 1)
    corr_score = max(0.0, float(feat.get("best_corridor_score", 0.0)))
    utility = (
        0.35 * float(1.0 if feat.get("is_main_candidate", False) else 0.0)
        + 0.25 * float(1.0 if feat.get("is_support_candidate", False) else 0.0)
        + 0.20 * float(1.0 if feat.get("is_connector_adjacent", False) else 0.0)
        + 0.15 * float(rank_bonus)
        + 0.05 * float(corr_score)
    )
    return float(max(0.0, min(1.0, utility)))


def _token_penalty(token_count: int) -> float:
    return min(1.0, float(max(1, int(token_count))) / 80.0)


@dataclass(frozen=True)
class SeedSetCandidate:
    seed_set_id: str
    sentence_ids: Tuple[str, ...]
    recall_proxy: float
    bridge_path_proxy: float
    diversity_proxy: float
    stability_proxy: float
    redundancy_penalty: float
    compactness_penalty: float


def resolve_dynamic_weights(
    *,
    base_weights: Mapping[str, float],
    graph_dispersion: float,
    coverage_saturation: float,
    redundancy_ratio: float,
    dynamic_control_enabled: bool,
) -> Dict[str, float]:
    weights = {
        "recall": float(base_weights.get("recall", 0.34)),
        "bridge_path": float(base_weights.get("bridge_path", 0.22)),
        "diversity": float(base_weights.get("diversity", 0.16)),
        "stability": float(base_weights.get("stability", 0.12)),
        "redundancy": float(base_weights.get("redundancy", 0.16)),
        "cost": float(base_weights.get("cost", 0.12)),
    }
    if not dynamic_control_enabled:
        return weights

    if float(graph_dispersion) >= 0.55:
        weights["bridge_path"] += 0.10
    if float(coverage_saturation) <= 0.40:
        weights["recall"] += 0.10
    if float(redundancy_ratio) >= 0.45:
        weights["redundancy"] += 0.10
    if float(redundancy_ratio) <= 0.25 and float(coverage_saturation) >= 0.60:
        weights["cost"] += 0.06

    return weights


def score_seed_set(
    seed_set: SeedSetCandidate,
    *,
    weights: Mapping[str, float],
    use_stability: bool,
    use_bridge_path: bool,
) -> float:
    score = (
        float(weights.get("recall", 0.0)) * float(seed_set.recall_proxy)
        + float(weights.get("diversity", 0.0)) * float(seed_set.diversity_proxy)
    )
    if use_bridge_path:
        score += float(weights.get("bridge_path", 0.0)) * float(seed_set.bridge_path_proxy)
    if use_stability:
        score += float(weights.get("stability", 0.0)) * float(seed_set.stability_proxy)
    score -= float(weights.get("redundancy", 0.0)) * float(seed_set.redundancy_penalty)
    score -= float(weights.get("cost", 0.0)) * float(seed_set.compactness_penalty)
    return float(score)


def select_best_seed_set(
    seed_sets: Sequence[SeedSetCandidate],
    *,
    weights: Mapping[str, float],
    use_stability: bool,
    use_bridge_path: bool,
) -> Tuple[SeedSetCandidate | None, Dict[str, float]]:
    if not seed_sets:
        return None, {}
    score_map: Dict[str, float] = {}
    best = None
    best_score = float("-inf")
    for item in list(seed_sets):
        sc = score_seed_set(item, weights=weights, use_stability=use_stability, use_bridge_path=use_bridge_path)
        score_map[item.seed_set_id] = float(sc)
        if sc > best_score:
            best_score = float(sc)
            best = item
    return best, score_map


def _safe_slice(nodes: Sequence[str], k: int) -> List[str]:
    return [str(x) for x in list(nodes or [])[: max(1, int(k))]]


def _build_seed_set_candidates(
    *,
    sentence_ids: Sequence[str],
    question_tokens: Set[str],
    token_map: Mapping[str, Set[str]],
    token_count_map: Mapping[str, int],
    source_map: Mapping[str, str],
    semantic_norm: Mapping[str, float],
    bridge_path_scores: Mapping[str, float],
    feature_table: Mapping[str, Mapping[str, Any]],
    seed_topk: int,
    bridge_path_enabled: bool,
    stability_enabled: bool,
) -> List[SeedSetCandidate]:
    if not sentence_ids:
        return []

    ids = [str(x) for x in list(sentence_ids or []) if str(x)]
    ranked_sem = sorted(ids, key=lambda sid: float(semantic_norm.get(sid, 0.0)), reverse=True)
    ranked_bridge = sorted(ids, key=lambda sid: float(bridge_path_scores.get(sid, 0.0)), reverse=True)
    ranked_query = sorted(
        ids,
        key=lambda sid: float(len(token_map.get(sid, set()).intersection(question_tokens))),
        reverse=True,
    )
    ranked_local = sorted(
        ids,
        key=lambda sid: float((feature_table.get(sid, {}) or {}).get("locality_score", 0.0)),
        reverse=True,
    )

    source_buckets: Dict[str, List[str]] = {}
    for sid in ids:
        skey = str(source_map.get(sid, "") or "")
        source_buckets.setdefault(skey, []).append(sid)
    diverse_source: List[str] = []
    for skey in sorted(source_buckets.keys()):
        group = sorted(source_buckets[skey], key=lambda sid: float(semantic_norm.get(sid, 0.0)), reverse=True)
        if group:
            diverse_source.append(group[0])
    if len(diverse_source) < max(1, int(seed_topk)):
        for sid in ranked_sem:
            if sid not in diverse_source:
                diverse_source.append(sid)
            if len(diverse_source) >= int(seed_topk):
                break

    corridor_leaders: List[str] = []
    corridor_best: Dict[str, Tuple[float, str]] = {}
    for sid in ids:
        feat = dict(feature_table.get(sid, {}) or {})
        for cid in list(feat.get("corridor_ids", []) or []):
            score = float(semantic_norm.get(sid, 0.0)) + 0.35 * float(bridge_path_scores.get(sid, 0.0))
            prev = corridor_best.get(str(cid))
            if prev is None or score > float(prev[0]):
                corridor_best[str(cid)] = (score, sid)
    for _cid, (_score, sid) in sorted(corridor_best.items(), key=lambda kv: kv[1][0], reverse=True):
        corridor_leaders.append(str(sid))
        if len(corridor_leaders) >= int(seed_topk):
            break
    if not corridor_leaders:
        corridor_leaders = _safe_slice(ranked_bridge, seed_topk)

    raw_sets = [
        ("semantic_view", _safe_slice(ranked_sem, seed_topk)),
        ("bridge_view", _safe_slice(ranked_bridge, seed_topk)),
        ("coverage_view", _safe_slice(ranked_query, seed_topk)),
        ("stability_view", _safe_slice(ranked_local, seed_topk)),
        ("source_diverse_view", _safe_slice(diverse_source, seed_topk)),
        ("corridor_view", _safe_slice(corridor_leaders, seed_topk)),
    ]

    out: List[SeedSetCandidate] = []
    seen_signatures: Set[Tuple[str, ...]] = set()
    for seed_set_id, raw_ids in raw_sets:
        unique_ids = tuple(x for x in raw_ids if x)
        signature = tuple(unique_ids)
        if (not unique_ids) or signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        local_tokens = set()
        local_sources = set()
        local_corridors = set()
        local_bridge = 0.0
        local_sem = 0.0
        local_stability = 0.0
        local_redundancy = 0.0
        local_cost = 0.0
        prev_sets: List[Set[str]] = []
        for rank_idx, sid in enumerate(unique_ids, start=1):
            terms = set(token_map.get(sid, set()))
            local_tokens.update(terms)
            skey = str(source_map.get(sid, "") or "")
            if skey:
                local_sources.add(skey)
            feat = dict(feature_table.get(sid, {}) or {})
            local_corridors.update(str(cid) for cid in list(feat.get("corridor_ids", []) or []))
            local_bridge += float(bridge_path_scores.get(sid, 0.0))
            local_sem += float(semantic_norm.get(sid, 0.0))
            if stability_enabled:
                local_stability += float(1.0 / float(rank_idx + int(feat.get("best_corridor_rank", 0) or 0) + 1))
            local_cost += _token_penalty(int(token_count_map.get(sid, 0)))
            cur = set(token_map.get(sid, set()))
            if prev_sets:
                local_redundancy += max(_jaccard(cur, p) for p in prev_sets)
            prev_sets.append(cur)

        q_cover = float(len(local_tokens.intersection(question_tokens))) / float(max(1, len(question_tokens)))
        source_cover = float(len(local_sources)) / float(max(1, len(unique_ids)))
        corridor_cover = float(len(local_corridors)) / float(max(1, len(unique_ids)))
        semantic_proxy = float(local_sem) / float(max(1, len(unique_ids)))
        recall_proxy = float(0.45 * q_cover + 0.35 * semantic_proxy + 0.10 * source_cover + 0.10 * corridor_cover)
        bridge_proxy = float(local_bridge) / float(max(1, len(unique_ids))) if bridge_path_enabled else 0.0
        diversity_proxy = float(min(1.0, 0.60 * source_cover + 0.40 * corridor_cover))
        stability_proxy = float(min(1.0, local_stability / float(max(1, len(unique_ids))))) if stability_enabled else 0.0
        redundancy_penalty = float(min(1.0, local_redundancy / float(max(1, len(unique_ids) - 1))))
        compactness_penalty = float(min(1.0, local_cost / float(max(1, len(unique_ids)))))

        out.append(
            SeedSetCandidate(
                seed_set_id=str(seed_set_id),
                sentence_ids=unique_ids,
                recall_proxy=float(recall_proxy),
                bridge_path_proxy=float(bridge_proxy),
                diversity_proxy=float(diversity_proxy),
                stability_proxy=float(stability_proxy),
                redundancy_penalty=float(redundancy_penalty),
                compactness_penalty=float(compactness_penalty),
            )
        )
    return out


def apply_gl_rcedr(
    *,
    question_text: str,
    selected_sentence_ids: Sequence[str],
    selected_sentences: Sequence[str],
    sentence_feature_table: Mapping[str, Mapping[str, Any]] | None,
    cfg: Any,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    enabled = bool(getattr(cfg, "gl_rcedr_enabled", False))
    base_ids, base_texts = _ordered_unique(list(selected_sentence_ids or []), list(selected_sentences or []))
    if (not enabled) or (not base_ids):
        return list(base_ids), list(base_texts), {
            "enabled": bool(enabled),
            "applied": False,
            "reason": "disabled_or_empty",
        }

    feature_table = dict(sentence_feature_table or {})
    text_map = {sid: txt for sid, txt in zip(base_ids, base_texts)}
    token_map = {sid: set(content_tokens(text_map.get(sid, ""))) for sid in base_ids}
    token_count_map = {sid: len(token_map.get(sid, set())) for sid in base_ids}
    source_map = {sid: _source_key(sid) for sid in base_ids}
    question_tokens = set(content_tokens(str(question_text or "")))

    semantic_raw = {sid: float((feature_table.get(sid, {}) or {}).get("base_retrieval_score", 0.0)) for sid in base_ids}
    semantic_norm = _normalize(semantic_raw)

    bridge_path_enabled = bool(getattr(cfg, "gl_rcedr_bridge_path_enabled", True))
    bridge_path_scores = {
        sid: _bridge_path_utility(dict(feature_table.get(sid, {}) or {}), bridge_path_enabled)
        for sid in base_ids
    }

    corridor_groups = []
    for sid in base_ids:
        feat = dict(feature_table.get(sid, {}) or {})
        corridor_groups.extend([str(cid) for cid in list(feat.get("corridor_ids", []) or []) if str(cid)])
    graph_dispersion = float(len(set(corridor_groups)) / float(max(1, len(base_ids))))
    covered_q = set()
    for sid in sorted(base_ids, key=lambda x: float(semantic_norm.get(x, 0.0)), reverse=True)[: max(4, int(len(base_ids) * 0.4))]:
        covered_q.update(token_map.get(sid, set()).intersection(question_tokens))
    coverage_saturation = float(len(covered_q) / float(max(1, len(question_tokens))))
    redundancy_vals: List[float] = []
    sorted_ids = sorted(base_ids, key=lambda x: float(semantic_norm.get(x, 0.0)), reverse=True)
    for i in range(len(sorted_ids)):
        a = sorted_ids[i]
        for j in range(i + 1, min(len(sorted_ids), i + 4)):
            b = sorted_ids[j]
            redundancy_vals.append(_jaccard(token_map.get(a, set()), token_map.get(b, set())))
    redundancy_ratio = float(sum(redundancy_vals) / float(max(1, len(redundancy_vals))))

    dynamic_control_enabled = bool(getattr(cfg, "gl_rcedr_dynamic_control_enabled", True))
    stability_enabled = bool(getattr(cfg, "gl_rcedr_stability_enabled", True))
    density_first_ablation = bool(getattr(cfg, "gl_rcedr_density_first_ablation_enabled", False))
    base_weights = {
        "recall": float(getattr(cfg, "gl_rcedr_recall_weight", 0.34)),
        "bridge_path": float(getattr(cfg, "gl_rcedr_bridge_path_weight", 0.22)),
        "diversity": float(getattr(cfg, "gl_rcedr_diversity_weight", 0.16)),
        "stability": float(getattr(cfg, "gl_rcedr_stability_weight", 0.12)),
        "redundancy": float(getattr(cfg, "gl_rcedr_redundancy_weight", 0.16)),
        "cost": float(getattr(cfg, "gl_rcedr_cost_weight", 0.12)),
    }
    dynamic_weights = resolve_dynamic_weights(
        base_weights=base_weights,
        graph_dispersion=graph_dispersion,
        coverage_saturation=coverage_saturation,
        redundancy_ratio=redundancy_ratio,
        dynamic_control_enabled=dynamic_control_enabled,
    )

    seed_topk = max(2, int(getattr(cfg, "gl_rcedr_seed_topk", 6)))
    seed_sets = _build_seed_set_candidates(
        sentence_ids=base_ids,
        question_tokens=question_tokens,
        token_map=token_map,
        token_count_map=token_count_map,
        source_map=source_map,
        semantic_norm=semantic_norm,
        bridge_path_scores=bridge_path_scores,
        feature_table=feature_table,
        seed_topk=seed_topk,
        bridge_path_enabled=bridge_path_enabled,
        stability_enabled=stability_enabled,
    )
    best_seed_set, seed_scores = select_best_seed_set(
        seed_sets,
        weights=dynamic_weights,
        use_stability=stability_enabled,
        use_bridge_path=bridge_path_enabled,
    )

    core_preserve_threshold = float(getattr(cfg, "gl_rcedr_core_preserve_threshold", 0.56))
    bridge_preserve_threshold = float(getattr(cfg, "gl_rcedr_bridge_preserve_threshold", 0.46))
    coverage_preserve_threshold = float(getattr(cfg, "gl_rcedr_coverage_preserve_threshold", 0.38))
    max_selected = max(1, int(getattr(cfg, "gl_rcedr_max_selected_candidates", 24)))
    max_rendered = max(1, int(getattr(cfg, "gl_rcedr_max_rendered_candidates", 24)))
    max_count = min(max_selected, max_rendered, len(base_ids))
    max_tokens = max(1, int(getattr(cfg, "gl_rcedr_max_rendered_tokens", 360)))
    min_keep = max(1, int(getattr(cfg, "min_render_topn", 6)))

    ranked_ids = sorted(base_ids, key=lambda sid: float(semantic_norm.get(sid, 0.0)), reverse=True)
    seed_boost = set(best_seed_set.sentence_ids if best_seed_set is not None else ())

    query_cover_map: Dict[str, float] = {}
    for sid in base_ids:
        q_cover = float(len(token_map.get(sid, set()).intersection(question_tokens))) / float(max(1, len(question_tokens)))
        query_cover_map[sid] = q_cover

    preserve_scores: Dict[str, float] = {}
    for sid in ranked_ids:
        preserve = (
            0.45 * float(semantic_norm.get(sid, 0.0))
            + 0.30 * float(query_cover_map.get(sid, 0.0))
            + 0.25 * float(bridge_path_scores.get(sid, 0.0))
        )
        if sid in seed_boost:
            preserve += 0.08
        preserve_scores[sid] = float(min(1.2, preserve))

    core_ids: List[str] = []
    if not density_first_ablation:
        for sid in ranked_ids:
            if sid in seed_boost:
                core_ids.append(sid)
                continue
            if float(preserve_scores.get(sid, 0.0)) >= core_preserve_threshold:
                core_ids.append(sid)
                continue
            if float(bridge_path_scores.get(sid, 0.0)) >= bridge_preserve_threshold:
                core_ids.append(sid)
                continue
            if float(query_cover_map.get(sid, 0.0)) >= coverage_preserve_threshold:
                core_ids.append(sid)
                continue
    core_ids = [sid for sid in core_ids if sid in base_ids]

    selected: List[str] = []
    selected_tokens = 0
    selected_sources: Set[str] = set()
    covered_q_terms: Set[str] = set()
    covered_content_terms: Set[str] = set()

    for sid in core_ids:
        if sid in selected:
            continue
        tok = int(token_count_map.get(sid, 0))
        if len(selected) >= min_keep or (len(selected) > 0 and selected_tokens > 0):
            if selected_tokens + tok > max_tokens and len(selected) >= min_keep:
                continue
        selected.append(sid)
        selected_tokens += tok
        if str(source_map.get(sid, "") or ""):
            selected_sources.add(str(source_map.get(sid, "")))
        covered_q_terms.update(token_map.get(sid, set()).intersection(question_tokens))
        covered_content_terms.update(token_map.get(sid, set()))
        if len(selected) >= max_count:
            break

    remaining = [sid for sid in ranked_ids if sid not in selected]
    score_trace: Dict[str, Dict[str, float]] = {}
    while remaining and len(selected) < max_count:
        best_sid = None
        best_score = float("-inf")
        best_components = {}
        for sid in list(remaining):
            sent_terms = token_map.get(sid, set())
            sent_tok = int(token_count_map.get(sid, 0))
            if selected_tokens + sent_tok > max_tokens and len(selected) >= min_keep:
                continue

            query_gain = float(len((sent_terms.intersection(question_tokens)) - covered_q_terms)) / float(max(1, len(question_tokens)))
            lexical_gain = float(
                len([tok for tok in sent_terms if len(tok) >= 4 and tok not in covered_content_terms])
            ) / float(max(1, min(8, len(sent_terms) or 1)))
            marginal_coverage = min(1.0, 0.60 * query_gain + 0.40 * lexical_gain)

            redundancy = 0.0
            if selected:
                redundancy = max(
                    _jaccard(sent_terms, token_map.get(prev, set()))
                    for prev in selected
                )
                if str(source_map.get(sid, "") or "") in selected_sources:
                    redundancy = min(1.0, redundancy + 0.12)

            recall_preserve = preserve_scores.get(sid, 0.0)
            bridge_score = bridge_path_scores.get(sid, 0.0) if bridge_path_enabled else 0.0
            token_cost = _token_penalty(sent_tok)
            if density_first_ablation:
                token_cost = min(1.0, token_cost + 0.20)
                recall_preserve *= 0.80

            score = (
                float(dynamic_weights.get("recall", 0.0)) * float(recall_preserve)
                + float(dynamic_weights.get("bridge_path", 0.0)) * float(bridge_score)
                + float(dynamic_weights.get("diversity", 0.0)) * float(marginal_coverage)
                - float(dynamic_weights.get("redundancy", 0.0)) * float(redundancy)
                - float(dynamic_weights.get("cost", 0.0)) * float(token_cost)
            )
            if score > best_score:
                best_score = float(score)
                best_sid = sid
                best_components = {
                    "recall_preserve": float(recall_preserve),
                    "bridge_path": float(bridge_score),
                    "marginal_coverage": float(marginal_coverage),
                    "redundancy": float(redundancy),
                    "token_cost": float(token_cost),
                    "score": float(score),
                }
        if best_sid is None:
            break
        remaining.remove(best_sid)
        selected.append(best_sid)
        selected_tokens += int(token_count_map.get(best_sid, 0))
        if str(source_map.get(best_sid, "") or ""):
            selected_sources.add(str(source_map.get(best_sid, "")))
        covered_q_terms.update(token_map.get(best_sid, set()).intersection(question_tokens))
        covered_content_terms.update(token_map.get(best_sid, set()))
        score_trace[best_sid] = dict(best_components)

    selected = selected[:max_count]
    selected_texts = [text_map.get(sid, "") for sid in selected]
    selected_set = set(selected)
    rendered_tokens = int(sum(int(token_count_map.get(sid, 0)) for sid in selected))
    selected_core_count = int(len([sid for sid in selected if sid in set(core_ids)]))
    diagnostics = {
        "enabled": True,
        "applied": bool(selected != base_ids),
        "reason": "ok",
        "candidate_count_before": int(len(base_ids)),
        "selected_count_after": int(len(selected)),
        "selected_core_count": int(selected_core_count),
        "tokens_after": int(rendered_tokens),
        "max_rendered_tokens": int(max_tokens),
        "query_state": {
            "graph_dispersion": float(graph_dispersion),
            "coverage_saturation": float(coverage_saturation),
            "redundancy_ratio": float(redundancy_ratio),
        },
        "dynamic_control_enabled": bool(dynamic_control_enabled),
        "stability_enabled": bool(stability_enabled),
        "bridge_path_enabled": bool(bridge_path_enabled),
        "density_first_ablation_enabled": bool(density_first_ablation),
        "weights": dict(dynamic_weights),
        "seed_set_candidates": int(len(seed_sets)),
        "selected_seed_set_id": str(best_seed_set.seed_set_id) if best_seed_set is not None else "",
        "selected_seed_set": list(best_seed_set.sentence_ids) if best_seed_set is not None else [],
        "seed_set_scores": dict(seed_scores),
        "core_preserve_threshold": float(core_preserve_threshold),
        "bridge_preserve_threshold": float(bridge_preserve_threshold),
        "coverage_preserve_threshold": float(coverage_preserve_threshold),
        "max_selected_candidates": int(max_selected),
        "max_rendered_candidates": int(max_rendered),
        "score_trace": dict(score_trace),
        "core_sentence_ids": sorted([sid for sid in selected_set if sid in set(core_ids)]),
    }
    return selected, selected_texts, diagnostics
