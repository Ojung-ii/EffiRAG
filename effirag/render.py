from .types import RenderedContext
from .metrics import supporting_fact_match_details
from .utils import content_tokens


def _sample_sentence_lookup(sample):
    lookup = {}
    for doc in sample.contexts:
        for sent_idx, sent in enumerate(doc.sentences):
            sid = "%s::%d" % (doc.title, sent_idx)
            lookup[sid] = sent
    return lookup


def _retrieval_graph_mode(retrieval_result):
    diagnostics = (getattr(retrieval_result, "diagnostics", {}) or {})
    mode = str(diagnostics.get("graph_mode", "current_entity_graph") or "current_entity_graph").strip().lower()
    if mode not in {"current_entity_graph", "entity_chunk_graph"}:
        mode = "current_entity_graph"
    return mode


def _selected_unit_type(retrieval_result):
    diagnostics = (getattr(retrieval_result, "diagnostics", {}) or {})
    unit_type = str(diagnostics.get("selected_unit_type", diagnostics.get("selected_text_unit_type", "sentence")) or "sentence").strip().lower()
    if unit_type not in {"sentence", "chunk"}:
        unit_type = "sentence"
    return unit_type


def _corridor_get(corridor, key, legacy_key):
    return corridor.get(key, corridor.get(legacy_key, [])) or []


def _is_chunk_like_id(unit_id):
    sid = str(unit_id or "")
    return sid.startswith("chunk::")


def _build_retrieval_text_lookup(sample, retrieval_result):
    lookup = {}
    ids = list(getattr(retrieval_result, "selected_sentence_ids", []) or [])
    texts = list(getattr(retrieval_result, "selected_sentences", []) or [])
    for sid, text in zip(ids, texts):
        sid = str(sid or "")
        txt = str(text or "").strip()
        if sid and txt:
            lookup[sid] = txt

    diagnostics = (getattr(retrieval_result, "diagnostics", {}) or {})
    diag_map = diagnostics.get("selected_text_map", {}) or {}
    if isinstance(diag_map, dict):
        for sid, text in diag_map.items():
            sid = str(sid or "")
            txt = str(text or "").strip()
            if sid and txt and sid not in lookup:
                lookup[sid] = txt

    sentence_lookup = _sample_sentence_lookup(sample)
    for sid in ids:
        sid = str(sid or "")
        if (not sid) or (sid in lookup) or _is_chunk_like_id(sid):
            continue
        txt = str(sentence_lookup.get(sid, "") or "").strip()
        if txt:
            lookup[sid] = txt
    return lookup


def _ordered_unique(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _token_set(text):
    return set(content_tokens(text or ""))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    union = a.union(b)
    if not union:
        return 0.0
    return float(len(a.intersection(b)) / len(union))


def _unit_title(unit_id):
    sid = str(unit_id or "")
    if sid.startswith("chunk::"):
        parts = sid.split("::")
        if len(parts) >= 3:
            return str(parts[1])
    if "::" in sid:
        return str(sid.rsplit("::", 1)[0])
    return sid


def _feature_role(feat, query_overlap_threshold=1.0):
    overlap = float((feat or {}).get("query_overlap_score", 0.0) or 0.0)
    if overlap >= float(query_overlap_threshold):
        return "query"
    if bool((feat or {}).get("is_connector_adjacent", False)) or bool((feat or {}).get("is_support_candidate", False)):
        return "bridge"
    return "answer"


def _role_tag_for_sentence(sid, feat, answer_bearing_sentence_scores):
    src = dict(feat or {})
    overlap = float(src.get("query_overlap_score", 0.0) or 0.0)
    if overlap >= 1.0:
        return "[QUERY]", False
    if bool(src.get("is_connector_adjacent", False)):
        return "[BRIDGE]", False
    if bool(src.get("is_support_candidate", False)):
        return "[SUPPORT]", False
    if sid in answer_bearing_sentence_scores:
        return "[LOCAL]", False
    return "[EVIDENCE]", True


def _safe_ratio(numer, denom):
    try:
        n = float(numer)
        d = float(denom)
    except Exception:
        return 0.0
    if d <= 0.0:
        return 0.0
    return float(n / d)


def _approx_token_count(text):
    return int(len(content_tokens(text or "")))


def _short_line(text, max_words=16, max_chars=140):
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return "-"
    words = raw.split()
    if len(words) > int(max_words):
        raw = " ".join(words[: int(max_words)]).strip() + " ..."
    if len(raw) > int(max_chars):
        raw = raw[: int(max_chars) - 3].rstrip() + "..."
    return raw


def _int_default(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _bool_default(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _is_support_like(feat):
    src = feat or {}
    return bool(src.get("is_support_candidate", False) or src.get("is_connector_adjacent", False))


def _parse_order_strategy_flags(order_strategy, retrieval_result):
    diagnostics = (getattr(retrieval_result, "diagnostics", {}) or {})
    raw = str(order_strategy or "score").strip().lower()
    tokens = [tok.strip() for tok in raw.split("+") if tok.strip()]
    base = tokens[0] if tokens else "score"
    flags = set(tokens[1:] if len(tokens) > 1 else [])
    base_strategies = {"score", "retrieval", "corridor_rank", "query_bridge_answer", "qba"}
    if base not in base_strategies:
        flags.add(base)
        base = "score"

    if base in {"relation_off", "relation_disabled", "relation_ordering_off"}:
        base = "score"
    elif base in {"relation_on", "relation_enabled", "relation_ordering_on"}:
        base = "query_bridge_answer"

    if _bool_default(diagnostics.get("final_top_slice_reorder_enabled", False), False):
        flags.add("top_slice_reorder")
    if _bool_default(diagnostics.get("answer_support_pinning_enabled", False), False):
        flags.add("answer_support_pin")
    if _bool_default(diagnostics.get("oracle_support_injection_enabled", False), False):
        flags.add("oracle_support")

    if "relation_on" in flags or "relation_enabled" in flags or "relation_ordering_on" in flags:
        base = "query_bridge_answer"
    if "relation_off" in flags or "relation_disabled" in flags or "relation_ordering_off" in flags:
        base = "score"

    top_slice_topk = max(1, _int_default(diagnostics.get("final_top_slice_reorder_topk", 4), 4))
    support_pin_min = max(0, _int_default(diagnostics.get("answer_support_pinning_min", 1), 1))
    raw_focus_top_bundle_only_n = 0
    if "raw_focus_top1_bundle_only" in flags or "top1_answer_bearing_bundle_only" in flags:
        raw_focus_top_bundle_only_n = 1
    elif "raw_focus_top2_bundle_only" in flags or "top2_answer_bearing_bundle_only" in flags:
        raw_focus_top_bundle_only_n = 2
    else:
        for tok in flags:
            if tok.startswith("raw_focus_top") and tok.endswith("_bundle_only"):
                core = tok[len("raw_focus_top") : -len("_bundle_only")]
                if core.isdigit():
                    raw_focus_top_bundle_only_n = max(raw_focus_top_bundle_only_n, int(core))
    return base, flags, top_slice_topk, support_pin_min, int(max(0, raw_focus_top_bundle_only_n))


def _corridor_sentence_ids(corridor):
    return _ordered_unique(
        list(_corridor_get(corridor, "main_path_unit_ids", "main_path_sentence_ids"))
        + list(_corridor_get(corridor, "support_unit_ids", "support_sentence_ids"))
        + list(_corridor_get(corridor, "connector_adjacent_unit_ids", "connector_adjacent_sentence_ids"))
    )


def _corridor_answer_bearing_stats(corridor):
    comp = dict((corridor or {}).get("final_score_components", {}) or {})
    path_complete_score = float(comp.get("path_complete_score", comp.get("path_complete", 0.0)) or 0.0)
    bridge_answer_pair_retention = float(comp.get("bridge_answer_pair_retention", 0.0) or 0.0)
    answer_side_density = float(comp.get("answer_side_density", 0.0) or 0.0)
    bridge_purity = float(comp.get("bridge_purity", 0.0) or 0.0)
    path_preserve_applied = bool(comp.get("path_preserve_applied", False))

    score = (
        0.42 * max(0.0, min(1.0, path_complete_score))
        + 0.33 * max(0.0, min(1.0, bridge_answer_pair_retention))
        + 0.20 * max(0.0, min(1.0, answer_side_density))
        + 0.05 * float(1.0 if path_preserve_applied else 0.0)
    )
    is_answer_bearing = bool(
        score >= 0.58
        and bridge_answer_pair_retention >= 0.40
        and answer_side_density >= 0.45
        and (path_complete_score >= 0.45 or bridge_purity >= 0.35)
    )
    return {
        "score": float(max(0.0, min(1.0, score))),
        "path_complete_score": float(path_complete_score),
        "bridge_answer_pair_retention": float(bridge_answer_pair_retention),
        "answer_side_density": float(answer_side_density),
        "bridge_purity": float(bridge_purity),
        "is_answer_bearing": bool(is_answer_bearing),
    }


def _build_answer_bearing_maps(retrieval_result):
    ranked_corridors = sorted(
        list(getattr(retrieval_result, "corridors", []) or []),
        key=lambda c: float((c or {}).get("corridor_score", 0.0)),
        reverse=True,
    )
    answer_bearing_corridor_scores = {}
    answer_bearing_corridor_rank = {}
    answer_bearing_sentence_scores = {}
    answer_bearing_sentence_bundle_rank = {}
    answer_bearing_corridor_order = []

    for rank, corridor in enumerate(ranked_corridors, start=1):
        cid = str((corridor or {}).get("corridor_id", f"c{rank:02d}") or f"c{rank:02d}")
        stats = _corridor_answer_bearing_stats(corridor)
        if not bool(stats.get("is_answer_bearing", False)):
            continue
        score = float(stats.get("score", 0.0))
        answer_bearing_corridor_scores[cid] = float(score)
        answer_bearing_corridor_rank[cid] = int(rank)
        answer_bearing_corridor_order.append((cid, score))
        for sid in _corridor_sentence_ids(corridor):
            sid = str(sid or "")
            if not sid:
                continue
            prev = float(answer_bearing_sentence_scores.get(sid, -1.0))
            if score > prev:
                answer_bearing_sentence_scores[sid] = float(score)
            prev_rank = int(answer_bearing_sentence_bundle_rank.get(sid, 10**9))
            if int(rank) < prev_rank:
                answer_bearing_sentence_bundle_rank[sid] = int(rank)

    answer_bearing_corridor_order = sorted(
        answer_bearing_corridor_order,
        key=lambda x: (float(x[1]), -float(answer_bearing_corridor_rank.get(x[0], 10**9))),
        reverse=True,
    )
    return {
        "corridor_scores": dict(answer_bearing_corridor_scores),
        "corridor_rank": dict(answer_bearing_corridor_rank),
        "corridor_order": [cid for cid, _ in answer_bearing_corridor_order],
        "sentence_scores": dict(answer_bearing_sentence_scores),
        "sentence_bundle_rank": dict(answer_bearing_sentence_bundle_rank),
    }


def _apply_raw_focus_front_order(
    selected_ids,
    features,
    score_at_pick,
    base_scores,
    retrieval_rank,
    answer_bearing_sentence_scores,
):
    ids = list(selected_ids or [])
    if not ids:
        return ids, {"applied": False, "promoted": 0}
    if not answer_bearing_sentence_scores:
        return ids, {"applied": False, "promoted": 0}

    pos_before = {sid: idx for idx, sid in enumerate(ids)}
    ranked = sorted(
        ids,
        key=lambda sid: (
            max(
                1,
                int((features.get(sid, {}) or {}).get("best_corridor_rank", 10**9) or 10**9)
                - (1 if sid in answer_bearing_sentence_scores else 0),
            ),
            0 if sid in answer_bearing_sentence_scores else 1,
            -float(answer_bearing_sentence_scores.get(sid, 0.0)),
            -float(score_at_pick.get(sid, base_scores.get(sid, 0.0))),
            retrieval_rank.get(sid, 10**9),
        ),
    )
    pos_after = {sid: idx for idx, sid in enumerate(ranked)}
    promoted = int(
        sum(
            1
            for sid in ranked
            if sid in answer_bearing_sentence_scores and pos_before.get(sid, 10**9) > pos_after.get(sid, 10**9)
        )
    )
    changed = bool(ranked != ids)
    return ranked, {"applied": bool(changed), "promoted": int(promoted if changed else 0)}


def _apply_answer_bearing_dedup(
    selected_ids,
    candidate_ids,
    candidate_text_map,
    answer_bearing_sentence_scores,
    max_n,
):
    ids = list(selected_ids or [])
    answer_set = set(answer_bearing_sentence_scores.keys())
    if not ids or not answer_set:
        return ids, {"applied": False, "removed": 0, "candidate_count": 0, "refilled": 0}

    token_cache = {}

    def _tokens_for(sid):
        if sid not in token_cache:
            token_cache[sid] = _token_set(candidate_text_map.get(sid, ""))
        return token_cache[sid]

    kept = []
    kept_set = set()
    kept_answer_ids = []
    removed = 0
    candidate_count = 0
    for sid in ids:
        sid = str(sid or "")
        if not sid:
            continue
        if sid not in answer_set:
            kept.append(sid)
            kept_set.add(sid)
            continue
        candidate_count += 1
        toks = _tokens_for(sid)
        redundancy = 0.0
        same_title = False
        for prev_sid in kept_answer_ids:
            redundancy = max(redundancy, _jaccard(toks, _tokens_for(prev_sid)))
            if _unit_title(prev_sid) == _unit_title(sid):
                same_title = True
        if redundancy >= 0.82 or (same_title and redundancy >= 0.65):
            removed += 1
            continue
        kept.append(sid)
        kept_set.add(sid)
        kept_answer_ids.append(sid)

    refilled = 0
    for sid in list(candidate_ids or []):
        if len(kept) >= int(max_n):
            break
        sid = str(sid or "")
        if not sid or sid in kept_set:
            continue
        if sid in answer_set:
            toks = _tokens_for(sid)
            redundancy = 0.0
            same_title = False
            for prev_sid in kept_answer_ids:
                redundancy = max(redundancy, _jaccard(toks, _tokens_for(prev_sid)))
                if _unit_title(prev_sid) == _unit_title(sid):
                    same_title = True
            if redundancy >= 0.82 or (same_title and redundancy >= 0.65):
                continue
            kept_answer_ids.append(sid)
        kept.append(sid)
        kept_set.add(sid)
        refilled += 1

    if len(kept) > int(max_n):
        kept = kept[: int(max_n)]
    return kept, {
        "applied": bool(removed > 0),
        "removed": int(max(0, removed)),
        "candidate_count": int(max(0, candidate_count)),
        "refilled": int(max(0, refilled)),
    }


def _apply_top_slice_reorder(selected_ids, features, score_at_pick, base_scores, retrieval_rank, topk):
    head_size = min(len(selected_ids), max(1, int(topk)))
    if head_size <= 1:
        return list(selected_ids), {"applied": False, "head_size": int(head_size), "reordered": 0}
    head = list(selected_ids[:head_size])
    tail = list(selected_ids[head_size:])
    order_map = {sid: idx for idx, sid in enumerate(head)}
    ranked = sorted(
        head,
        key=lambda sid: (
            -int(1 if _is_support_like(features.get(sid, {})) else 0),
            -int(1 if bool((features.get(sid, {}) or {}).get("is_main_candidate", False)) else 0),
            -float((features.get(sid, {}) or {}).get("query_overlap_score", 0.0)),
            -float(score_at_pick.get(sid, base_scores.get(sid, 0.0))),
            retrieval_rank.get(sid, 10**9),
            order_map.get(sid, 10**9),
        ),
    )
    out = ranked + tail
    changed = int(sum(1 for idx, sid in enumerate(head) if ranked[idx] != sid))
    return out, {"applied": bool(changed > 0), "head_size": int(head_size), "reordered": int(changed)}


def _apply_answer_support_pinning(
    selected_ids,
    candidate_ids,
    features,
    score_at_pick,
    base_scores,
    retrieval_rank,
    min_support,
):
    target = max(0, int(min_support))
    if target <= 0:
        return list(selected_ids), {"applied": False, "added": 0, "replaced": 0, "target": 0}

    out = list(selected_ids)

    def _support_count(ids):
        return sum(1 for sid in ids if _is_support_like(features.get(sid, {})))

    support_now = _support_count(out)
    need = max(0, target - support_now)
    if need <= 0:
        return out, {"applied": False, "added": 0, "replaced": 0, "target": int(target)}

    support_pool = [
        sid
        for sid in list(candidate_ids or [])
        if sid not in set(out) and _is_support_like(features.get(sid, {}))
    ]
    support_pool = sorted(
        support_pool,
        key=lambda sid: (
            float(score_at_pick.get(sid, base_scores.get(sid, 0.0))),
            -retrieval_rank.get(sid, 10**9),
        ),
        reverse=True,
    )

    added = 0
    replaced = 0
    for sid in support_pool:
        if need <= 0:
            break
        replace_idx = None
        for idx in range(len(out) - 1, -1, -1):
            if not _is_support_like(features.get(out[idx], {})):
                replace_idx = idx
                break
        if replace_idx is None:
            break
        out[replace_idx] = sid
        need -= 1
        replaced += 1
        added += 1

    out = _ordered_unique(out)
    return out, {
        "applied": bool(added > 0),
        "added": int(added),
        "replaced": int(replaced),
        "target": int(target),
    }


def _gold_support_unit_pairs(sample, retrieval_result):
    graph_mode = _retrieval_graph_mode(retrieval_result)
    sentence_lookup = _sample_sentence_lookup(sample)
    pairs = []
    seen = set()
    for title, sent_idx in list(getattr(sample, "supporting_facts", []) or []):
        t = str(title or "").strip()
        if not t:
            continue
        try:
            idx = int(sent_idx)
        except Exception:
            idx = 0
        if graph_mode == "entity_chunk_graph":
            unit_id = f"chunk::{t}::0"
        else:
            unit_id = f"{t}::{idx}"
        if unit_id in seen:
            continue
        seen.add(unit_id)
        text = str(sentence_lookup.get(f"{t}::{idx}", "") or "").strip()
        if not text:
            text = str(sentence_lookup.get(f"{t}::0", "") or "").strip()
        if not text:
            text = t
        pairs.append((unit_id, text))
    return pairs


def _apply_oracle_support_injection(selected_ids, candidate_text_map, sample, retrieval_result, max_n):
    oracle_pairs = _gold_support_unit_pairs(sample, retrieval_result)
    if not oracle_pairs:
        return list(selected_ids), {"applied": False, "injected": 0}

    for sid, text in oracle_pairs:
        if sid and text and sid not in candidate_text_map:
            candidate_text_map[sid] = text

    out = list(selected_ids)
    oracle_ids = [sid for sid, _ in oracle_pairs if sid]
    oracle_set = set(oracle_ids)
    injected = 0
    for sid in reversed(oracle_ids):
        if sid in out:
            continue
        if len(out) < int(max_n):
            out.insert(0, sid)
            injected += 1
            continue
        replace_idx = None
        for idx in range(len(out) - 1, -1, -1):
            if out[idx] not in oracle_set:
                replace_idx = idx
                break
        if replace_idx is None:
            continue
        out[replace_idx] = sid
        injected += 1

    out = _ordered_unique(out)
    if len(out) > int(max_n):
        out = out[: int(max_n)]
    return out, {"applied": bool(injected > 0), "injected": int(injected)}


def _gold_support_unit_ids(sample, retrieval_result):
    return {sid for sid, _txt in _gold_support_unit_pairs(sample, retrieval_result) if sid}


def _attach_stagewise_render_diagnostics(sample, retrieval_result, rendered):
    diagnostics = dict((getattr(retrieval_result, "diagnostics", {}) or {}))
    funnel = dict((diagnostics.get("stagewise_loss_funnel", {}) or {}))
    if not funnel:
        funnel = {}

    graph_mode = _retrieval_graph_mode(retrieval_result)
    retrieval_match = supporting_fact_match_details(
        sample=sample,
        unit_ids=list(getattr(retrieval_result, "selected_sentence_ids", []) or []),
        unit_texts=list(getattr(retrieval_result, "selected_sentences", []) or []),
        graph_mode=graph_mode,
    )
    rendered_match = supporting_fact_match_details(
        sample=sample,
        unit_ids=list(getattr(rendered, "sentence_ids", []) or []),
        unit_texts=list(getattr(rendered, "sentences", []) or []),
        graph_mode=graph_mode,
    )

    gold_total = int(rendered_match.get("gold_total", 0))
    rendered_hit_count = int(rendered_match.get("matched_gold_total", 0))
    final_hit_count = int(retrieval_match.get("matched_gold_total", 0))

    funnel["rendered_hit_count"] = int(rendered_hit_count)
    funnel["rendered_hit_rate"] = float(_safe_ratio(rendered_hit_count, gold_total))
    funnel["rendered_retention"] = float(_safe_ratio(rendered_hit_count, final_hit_count))
    funnel["rendered_hit_unit_ids"] = list(rendered_match.get("matched_gold_sentence_ids", []) or [])
    funnel["rendered_missed_unit_ids"] = list(rendered_match.get("missed_gold_sentence_ids", []) or [])
    rendered_meta = dict(getattr(rendered, "metadata", {}) or {})
    for key in (
        "path_bundle_count",
        "avg_chunks_per_bundle",
        "bridge_answer_adjacency_rate",
        "first_complete_path_rank",
        "bundle_dedup_ratio",
        "answer_path_coverage",
        "conversion_after_path_bundle",
        "summary_token_count",
        "path_focus_count",
        "derivation_prompt_activation",
        "answer_chain_readability_score",
        "raw_focus_front_applied",
        "raw_focus_front_promoted",
        "raw_focus_dedup_applied",
        "raw_focus_dedup_removed",
        "raw_focus_dedup_candidate_count",
        "raw_focus_dedup_refilled",
        "raw_focus_top_bundle_only_n",
        "raw_focus_top_bundle_only_applied",
        "raw_focus_scaffold_light_enabled",
        "raw_focus_answer_bearing_chunk_count",
        "raw_focus_answer_bearing_bundle_count",
        "raw_focus_first_answer_bearing_chunk_rank",
        "raw_focus_first_answer_bearing_bundle_rank",
        "generation_intervention_enabled",
        "generation_intervention_focus_count",
        "generation_intervention_quote_then_answer",
        "generation_intervention_grounded_answer",
        "generation_intervention_evidence_focus",
        "generation_intervention_ab_chain",
        "generation_intervention_ab_wording",
    ):
        if key in rendered_meta:
            try:
                funnel[key] = float(rendered_meta.get(key, 0.0) or 0.0)
            except Exception:
                funnel[key] = 0.0
            diagnostics[key] = funnel[key]
    diagnostics["stagewise_loss_funnel"] = funnel
    retrieval_result.diagnostics = diagnostics

    meta = dict(getattr(rendered, "metadata", {}) or {})
    meta.update(
        {
            "stagewise_rendered_hit_count": int(rendered_hit_count),
            "stagewise_rendered_hit_rate": float(funnel.get("rendered_hit_rate", 0.0)),
            "stagewise_rendered_retention": float(funnel.get("rendered_retention", 0.0)),
        }
    )
    rendered.metadata = meta
    return rendered


def _extract_selected_pairs(sample, retrieval_result):
    lookup = _build_retrieval_text_lookup(sample, retrieval_result)
    ids = list(retrieval_result.selected_sentence_ids or [])
    texts = list(retrieval_result.selected_sentences or [])
    if len(texts) != len(ids):
        texts = [lookup.get(sid, "") for sid in ids]
    pairs = [(sid, txt) for sid, txt in zip(ids, texts) if sid and txt]
    return pairs


def _build_fallback_feature_table(sample, retrieval_result, candidate_ids, candidate_text_map):
    question_tokens = _token_set(sample.question)
    corridors = sorted(
        retrieval_result.corridors or [],
        key=lambda c: float(c.get("corridor_score", 0.0)),
        reverse=True,
    )
    corridor_rank = {str(c.get("corridor_id", f"c{idx:02d}")): idx for idx, c in enumerate(corridors, start=1)}

    feature_table = {}
    for ridx, sid in enumerate(candidate_ids):
        sent_tokens = _token_set(candidate_text_map.get(sid, ""))
        feature_table[sid] = {
            "corridor_ids": [],
            "best_corridor_rank": 10**9,
            "best_corridor_score": 0.0,
            "is_main_candidate": False,
            "is_support_candidate": False,
            "is_connector_adjacent": False,
            "query_overlap_score": float(len(question_tokens.intersection(sent_tokens))),
            "locality_score": 1.0 / float(ridx + 1),
            "base_retrieval_score": 1.0 / float(ridx + 1),
        }

    for idx, corridor in enumerate(corridors, start=1):
        cid = str(corridor.get("corridor_id", f"c{idx:02d}"))
        cscore = float(corridor.get("corridor_score", 0.0))
        score_map = corridor.get("unit_score_map", corridor.get("sentence_score_map", {})) or {}
        main_ids = _corridor_get(corridor, "main_path_unit_ids", "main_path_sentence_ids")
        support_ids = _corridor_get(corridor, "support_unit_ids", "support_sentence_ids")
        connector_ids = _corridor_get(corridor, "connector_adjacent_unit_ids", "connector_adjacent_sentence_ids")

        for mpos, sid in enumerate(main_ids):
            if sid not in feature_table:
                continue
            item = feature_table[sid]
            if cid not in item["corridor_ids"]:
                item["corridor_ids"].append(cid)
            item["is_main_candidate"] = True
            item["best_corridor_rank"] = min(item["best_corridor_rank"], idx)
            item["best_corridor_score"] = max(item["best_corridor_score"], cscore)
            item["base_retrieval_score"] = max(item["base_retrieval_score"], float(score_map.get(sid, 0.0)))
            item["locality_score"] = max(item["locality_score"], 1.0 / float((idx) * (mpos + 1)))

        for spos, sid in enumerate(support_ids):
            if sid not in feature_table:
                continue
            item = feature_table[sid]
            if cid not in item["corridor_ids"]:
                item["corridor_ids"].append(cid)
            item["is_support_candidate"] = True
            item["best_corridor_rank"] = min(item["best_corridor_rank"], idx)
            item["best_corridor_score"] = max(item["best_corridor_score"], cscore)
            item["base_retrieval_score"] = max(item["base_retrieval_score"], float(score_map.get(sid, 0.0)))
            item["locality_score"] = max(item["locality_score"], 0.75 / float((idx) * (spos + 1)))

        for sid in connector_ids:
            if sid in feature_table:
                feature_table[sid]["is_connector_adjacent"] = True

    for sid, item in feature_table.items():
        item["corridor_ids"] = sorted(item["corridor_ids"], key=lambda x: corridor_rank.get(x, 10**9))
        if item["best_corridor_rank"] >= 10**9 and item["corridor_ids"]:
            item["best_corridor_rank"] = corridor_rank.get(item["corridor_ids"][0], 10**9)

    return feature_table


def _get_sentence_feature_table(sample, retrieval_result, candidate_ids, candidate_text_map):
    raw = (retrieval_result.diagnostics or {}).get("text_unit_feature_table", (retrieval_result.diagnostics or {}).get("sentence_feature_table", {})) or {}
    if not isinstance(raw, dict):
        raw = {}

    normalized = {}
    for sid in candidate_ids:
        src = raw.get(sid, {}) if isinstance(raw.get(sid, {}), dict) else {}
        normalized[sid] = {
            "corridor_ids": [str(x) for x in (src.get("corridor_ids", []) or [])],
            "best_corridor_rank": int(src.get("best_corridor_rank", 10**9) or 10**9),
            "best_corridor_score": float(src.get("best_corridor_score", 0.0)),
            "is_main_candidate": bool(src.get("is_main_candidate", False)),
            "is_support_candidate": bool(src.get("is_support_candidate", False)),
            "is_connector_adjacent": bool(src.get("is_connector_adjacent", False)),
            "query_overlap_score": float(src.get("query_overlap_score", 0.0)),
            "locality_score": float(src.get("locality_score", 0.0)),
            "base_retrieval_score": float(src.get("base_retrieval_score", 0.0)),
        }

    has_usable = any(v.get("corridor_ids") for v in normalized.values())
    if has_usable:
        return normalized
    return _build_fallback_feature_table(sample, retrieval_result, candidate_ids, candidate_text_map)


def _corridor_aware_linear_score(
    feat,
    alpha,
    beta,
    gamma_main,
    delta_support,
    eta_connector,
    zeta_query,
    xi_locality,
):
    return (
        alpha * float(feat.get("base_retrieval_score", 0.0))
        + beta * float(feat.get("best_corridor_score", 0.0))
        + gamma_main * (1.0 if feat.get("is_main_candidate") else 0.0)
        + delta_support * (1.0 if feat.get("is_support_candidate") else 0.0)
        + eta_connector * (1.0 if feat.get("is_connector_adjacent") else 0.0)
        + zeta_query * float(feat.get("query_overlap_score", 0.0))
        + xi_locality * float(feat.get("locality_score", 0.0))
    )


def order_corridor_sentences(corridor, sentence_map=None):
    main_ids = _ordered_unique(_corridor_get(corridor, "main_path_unit_ids", "main_path_sentence_ids"))
    main_ids = [sid for sid in main_ids if sid in sentence_map]
    if main_ids:
        return main_ids

    support_ids = _ordered_unique(_corridor_get(corridor, "support_unit_ids", "support_sentence_ids"))
    support_ids = [sid for sid in support_ids if sid in sentence_map]
    score_map = corridor.get("sentence_score_map", {}) or {}
    if score_map:
        support_ids = sorted(support_ids, key=lambda sid: float(score_map.get(sid, 0.0)), reverse=True)
    return support_ids


def select_support_sentences(corridor, main_sentence_ids, max_support_per_corridor, sentence_map=None):
    if max_support_per_corridor is not None and max_support_per_corridor <= 0:
        return []

    main_set = set(main_sentence_ids)
    support_ids = _ordered_unique(_corridor_get(corridor, "support_unit_ids", "support_sentence_ids"))
    support_ids = [sid for sid in support_ids if sid not in main_set and sid in sentence_map]
    if max_support_per_corridor is None:
        return support_ids
    return support_ids[:max_support_per_corridor]


def _rank_sentence_ids(
    candidate_ids,
    sentence_map,
    sentence_score_map,
    question_tokens,
    selected_ids=None,
):
    selected_ids = selected_ids or []
    selected_token_sets = [_token_set(sentence_map.get(sid, "")) for sid in selected_ids if sid in sentence_map]

    scored = []
    seen = set()
    for idx, sid in enumerate(candidate_ids):
        if sid in seen or sid not in sentence_map:
            continue
        seen.add(sid)

        sent = sentence_map.get(sid, "")
        sent_tokens = _token_set(sent)
        retrieval_score = float((sentence_score_map or {}).get(sid, 0.0))
        query_overlap = float(len(sent_tokens.intersection(question_tokens)))
        locality = 1.0 / float(idx + 1)
        redundancy = 0.0
        if selected_token_sets:
            redundancy = max(_jaccard(sent_tokens, prev) for prev in selected_token_sets)

        # Lightweight utility: retrieval + query overlap + locality - redundancy.
        utility = retrieval_score + 0.35 * query_overlap + 0.20 * locality - 0.30 * redundancy
        scored.append((sid, utility, idx))

    scored.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    return [sid for sid, _, _ in scored]


def _prefer_non_duplicate(candidate_ids, used_ids):
    dedup = [sid for sid in candidate_ids if sid not in used_ids]
    return dedup if dedup else candidate_ids


def build_corridor_payload(retrieval_result, graph=None, sentence_map=None):
    sentence_map = sentence_map or {}
    raw_corridors = retrieval_result.corridors or []
    corridors = []

    for item in raw_corridors:
        if not isinstance(item, dict):
            continue
        corridors.append(
            {
                "corridor_id": str(item.get("corridor_id", "")),
                "corridor_score": float(item.get("corridor_score", 0.0)),
                "anchors": [str(x) for x in (item.get("anchors", []) or [])],
                "main_path_sentence_ids": [str(x) for x in (item.get("main_path_sentence_ids", []) or [])],
                "support_sentence_ids": [str(x) for x in (item.get("support_sentence_ids", []) or [])],
                "connector_adjacent_sentence_ids": [
                    str(x)
                    for x in (
                        item.get("connector_adjacent_sentence_ids", item.get("connector_adjacent_unit_ids", []))
                        or []
                    )
                ],
                "sentence_score_map": {str(k): float(v) for k, v in (item.get("sentence_score_map", {}) or {}).items()},
            }
        )

    if not corridors:
        corridors = [
            {
                "corridor_id": "c01",
                "corridor_score": 0.0,
                "anchors": ["-", "-"],
                "main_path_sentence_ids": list(retrieval_result.selected_sentence_ids or []),
                "support_sentence_ids": [],
                "sentence_score_map": {},
            }
        ]

    corridors.sort(key=lambda c: c.get("corridor_score", 0.0), reverse=True)
    return corridors


def pack_corridors_with_budget(
    corridors,
    sentence_map,
    question_text,
    max_corridors_in_context,
    max_main_sentences_per_corridor,
    max_support_per_corridor,
    max_total_sentences,
):
    ranked_corridors = sorted(corridors, key=lambda c: float(c.get("corridor_score", 0.0)), reverse=True)
    selected_corridors = ranked_corridors[:max_corridors_in_context]
    truncated_corridors = max(0, len(ranked_corridors) - len(selected_corridors))
    question_tokens = _token_set(question_text)

    packed = []
    used_sentence_ids = set()
    global_selected_ids = []
    for idx, corridor in enumerate(selected_corridors, start=1):
        score_map = corridor.get("sentence_score_map", {}) or {}

        main_candidates = order_corridor_sentences(corridor, sentence_map=sentence_map)
        main_candidates = _prefer_non_duplicate(main_candidates, used_sentence_ids)
        ranked_main = _rank_sentence_ids(
            candidate_ids=main_candidates,
            sentence_map=sentence_map,
            sentence_score_map=score_map,
            question_tokens=question_tokens,
            selected_ids=global_selected_ids,
        )
        main_ids = ranked_main[:max_main_sentences_per_corridor]

        support_candidates = select_support_sentences(
            corridor,
            main_sentence_ids=main_ids,
            max_support_per_corridor=None,
            sentence_map=sentence_map,
        )
        connector_candidates = _ordered_unique(_corridor_get(corridor, "connector_adjacent_unit_ids", "connector_adjacent_sentence_ids"))
        connector_candidates = [sid for sid in connector_candidates if sid not in set(main_ids) and sid in sentence_map]
        support_candidates = _ordered_unique(connector_candidates + support_candidates)
        support_candidates = _prefer_non_duplicate(support_candidates, used_sentence_ids)
        support_score_map = dict(score_map)
        for sid in connector_candidates:
            support_score_map[sid] = float(support_score_map.get(sid, 0.0)) + 0.10
        ranked_support = _rank_sentence_ids(
            candidate_ids=support_candidates,
            sentence_map=sentence_map,
            sentence_score_map=support_score_map,
            question_tokens=question_tokens,
            selected_ids=global_selected_ids + main_ids,
        )
        support_ids = ranked_support[:max_support_per_corridor]

        if not main_ids and not support_ids:
            truncated_corridors += 1
            continue

        packed.append(
            {
                "corridor_id": corridor.get("corridor_id", f"c{idx:02d}"),
                "corridor_score": float(corridor.get("corridor_score", 0.0)),
                "anchors": corridor.get("anchors", ["-", "-"]),
                "main_sentence_ids": list(main_ids),
                "support_sentence_ids": list(support_ids),
            }
        )
        for sid in main_ids + support_ids:
            used_sentence_ids.add(sid)
            global_selected_ids.append(sid)

    def _total_sent_count(payloads):
        return sum(len(c.get("main_sentence_ids", [])) + len(c.get("support_sentence_ids", [])) for c in payloads)

    # Corridor-first overflow handling: drop low-ranked corridors before sentence truncation.
    while len(packed) > 1 and _total_sent_count(packed) > max_total_sentences:
        packed.pop()
        truncated_corridors += 1

    truncated_sentences = 0
    while packed and _total_sent_count(packed) > max_total_sentences:
        changed = False

        # Last resort #1: remove support from low-ranked corridor first.
        for corridor in reversed(packed):
            if corridor.get("support_sentence_ids"):
                corridor["support_sentence_ids"].pop()
                truncated_sentences += 1
                changed = True
                break
        if changed:
            continue

        # Last resort #2: trim lower-ranked main sentences.
        for corridor in reversed(packed):
            main_ids = corridor.get("main_sentence_ids", [])
            if len(main_ids) > 1:
                main_ids.pop()
                truncated_sentences += 1
                changed = True
                break
        if changed:
            continue

        # Hard fallback: keep at least one sentence if possible.
        for corridor in reversed(packed):
            main_ids = corridor.get("main_sentence_ids", [])
            if main_ids:
                main_ids.pop()
                truncated_sentences += 1
                changed = True
                break

        if not changed:
            break

    kept = []
    for corridor in packed:
        if corridor.get("main_sentence_ids") or corridor.get("support_sentence_ids"):
            kept.append(corridor)
        else:
            truncated_corridors += 1

    rendered_sentence_ids = []
    rendered_corridor_ids = []
    for corridor in kept:
        rendered_corridor_ids.append(corridor.get("corridor_id", ""))
        rendered_sentence_ids.extend(corridor.get("main_sentence_ids", []))
        rendered_sentence_ids.extend(corridor.get("support_sentence_ids", []))

    return kept, rendered_sentence_ids, rendered_corridor_ids, truncated_corridors, truncated_sentences


def _parse_path_bundle_flags(order_strategy):
    raw = str(order_strategy or "").strip().lower()
    tokens = [tok.strip() for tok in raw.split("+") if tok.strip()]
    flags = set(tokens)
    focus_n = 0
    if "path_bundle_top1_focus" in flags or "top1_path_focus" in flags:
        focus_n = 1
    elif "path_bundle_top2_focus" in flags or "top2_path_focus" in flags:
        focus_n = 2
    else:
        for tok in flags:
            if tok.startswith("path_bundle_top") and tok.endswith("_focus"):
                core = tok[len("path_bundle_top") : -len("_focus")]
                if core.isdigit():
                    focus_n = max(focus_n, int(core))
    return {
        "ordered": not bool(
            "path_bundle_unordered" in flags or "bundle_unordered" in flags or "unordered" in flags
        ),
        "dedup": bool("path_bundle_dedup" in flags or "bundle_dedup" in flags),
        "compact_lite": bool("path_bundle_compactlite" in flags or "compact_lite" in flags),
        "chain_summary": bool("path_bundle_chain_summary" in flags or "chain_summary" in flags),
        "derivation_prompt": bool("path_bundle_derivation_prompt" in flags or "derivation_prompt" in flags),
        "focus_n": int(max(0, focus_n)),
    }


def _clip_path_bundles_to_budget(bundles, max_total_sentences):
    max_total = max(1, int(max_total_sentences))
    kept = []
    rendered_sentence_ids = []
    truncated_sentences = 0
    truncated_bundles = 0
    remaining = int(max_total)

    for bundle in list(bundles or []):
        if remaining <= 0:
            truncated_bundles += 1
            continue
        anchors = list(bundle.get("anchor_sentence_ids", []) or [])
        bridges = list(bundle.get("bridge_sentence_ids", []) or [])
        answers = list(bundle.get("answer_sentence_ids", []) or [])
        ordered_ids = list(bundle.get("ordered_sentence_ids", []) or [])
        min_keep = int(bool(anchors)) + int(bool(bridges)) + int(bool(answers))
        if min_keep > remaining:
            truncated_bundles += 1
            continue

        chosen = []
        if anchors:
            chosen.append(anchors[0])
        if bridges:
            chosen.append(bridges[0])
        if answers:
            chosen.append(answers[0])
        chosen = _ordered_unique(chosen)

        extra_budget = max(0, remaining - len(chosen))
        for sid in ordered_ids:
            if sid in set(chosen):
                continue
            if extra_budget <= 0:
                truncated_sentences += 1
                continue
            chosen.append(sid)
            extra_budget -= 1

        if not chosen:
            truncated_bundles += 1
            continue

        chosen_set = set(chosen)
        kept_bundle = dict(bundle)
        kept_bundle["ordered_sentence_ids"] = list(chosen)
        kept_bundle["anchor_sentence_ids"] = [sid for sid in anchors if sid in chosen_set]
        kept_bundle["bridge_sentence_ids"] = [sid for sid in bridges if sid in chosen_set]
        kept_bundle["answer_sentence_ids"] = [sid for sid in answers if sid in chosen_set]
        kept.append(kept_bundle)

        rendered_sentence_ids.extend(chosen)
        remaining -= len(chosen)

    rendered_sentence_ids = _ordered_unique(rendered_sentence_ids)
    return kept, rendered_sentence_ids, truncated_bundles, truncated_sentences


def render_path_bundle_context(
    sample,
    retrieval_result,
    max_corridors_in_context,
    max_main_sentences_per_corridor,
    max_support_per_corridor,
    max_total_sentences,
    order_strategy,
):
    sentence_map = _build_retrieval_text_lookup(sample, retrieval_result)
    corridors = build_corridor_payload(retrieval_result, graph=None, sentence_map=sentence_map)
    ranked_corridors = sorted(corridors, key=lambda c: float(c.get("corridor_score", 0.0)), reverse=True)
    flags = _parse_path_bundle_flags(order_strategy)
    compact_lite = bool(flags.get("compact_lite", False))
    dedup_enabled = bool(flags.get("dedup", False))
    chain_summary_enabled = bool(flags.get("chain_summary", False))
    derivation_prompt_enabled = bool(flags.get("derivation_prompt", False))
    focus_n = int(flags.get("focus_n", 0))

    max_corridors = max(1, int(max_corridors_in_context))
    max_main = max(1, int(max_main_sentences_per_corridor))
    max_support = max(0, int(max_support_per_corridor))
    if compact_lite:
        max_corridors = max(1, min(max_corridors, 2))
        max_main = max(1, min(max_main, 2))
        max_support = max(1, min(max_support, 2))
    if focus_n > 0:
        max_corridors = max(1, min(max_corridors, focus_n))
    selected_corridors = ranked_corridors[:max_corridors]
    truncated_corridors = max(0, len(ranked_corridors) - len(selected_corridors))

    question_tokens = _token_set(sample.question)
    bundles = []
    assigned_bridge_answer_ids = set()
    dedup_removed_count = 0
    dedup_candidate_count = 0
    for idx, corridor in enumerate(selected_corridors, start=1):
        score_map = corridor.get("sentence_score_map", {}) or {}
        main_candidates = order_corridor_sentences(corridor, sentence_map=sentence_map)
        ranked_main = _rank_sentence_ids(
            candidate_ids=main_candidates,
            sentence_map=sentence_map,
            sentence_score_map=score_map,
            question_tokens=question_tokens,
            selected_ids=[],
        )
        anchor_ids = ranked_main[:max_main]

        connector_ids = _ordered_unique(_corridor_get(corridor, "connector_adjacent_unit_ids", "connector_adjacent_sentence_ids"))
        connector_ids = [sid for sid in connector_ids if sid in sentence_map and sid not in set(anchor_ids)]
        bridge_score_map = dict(score_map)
        for sid in connector_ids:
            bridge_score_map[sid] = float(bridge_score_map.get(sid, 0.0)) + 0.10
        ranked_bridge = _rank_sentence_ids(
            candidate_ids=connector_ids,
            sentence_map=sentence_map,
            sentence_score_map=bridge_score_map,
            question_tokens=question_tokens,
            selected_ids=anchor_ids,
        )
        bridge_budget = max(1, int(max_support / 2)) if max_support > 0 else 0
        bridge_ids = ranked_bridge[:bridge_budget]

        support_candidates = select_support_sentences(
            corridor,
            main_sentence_ids=anchor_ids,
            max_support_per_corridor=None,
            sentence_map=sentence_map,
        )
        answer_candidates = [sid for sid in support_candidates if sid not in set(anchor_ids) and sid not in set(bridge_ids)]
        ranked_answer = _rank_sentence_ids(
            candidate_ids=answer_candidates,
            sentence_map=sentence_map,
            sentence_score_map=score_map,
            question_tokens=question_tokens,
            selected_ids=anchor_ids + bridge_ids,
        )
        answer_budget = max(0, int(max_support) - len(bridge_ids))
        answer_ids = ranked_answer[:answer_budget]
        if not answer_ids and ranked_answer and max_support > 0:
            answer_ids = ranked_answer[:1]

        if dedup_enabled:
            before = len(bridge_ids) + len(answer_ids)
            dedup_candidate_count += before
            bridge_ids = [sid for sid in bridge_ids if sid not in assigned_bridge_answer_ids]
            answer_ids = [sid for sid in answer_ids if sid not in assigned_bridge_answer_ids]
            after = len(bridge_ids) + len(answer_ids)
            dedup_removed_count += max(0, before - after)
            for sid in bridge_ids + answer_ids:
                assigned_bridge_answer_ids.add(sid)

        ordered_ids = _ordered_unique(anchor_ids + bridge_ids + answer_ids)
        if not ordered_ids:
            truncated_corridors += 1
            continue
        bundles.append(
            {
                "bundle_id": f"b{idx:02d}",
                "corridor_id": str(corridor.get("corridor_id", f"c{idx:02d}") or f"c{idx:02d}"),
                "corridor_score": float(corridor.get("corridor_score", 0.0)),
                "anchors": list(corridor.get("anchors", ["-", "-"]) or ["-", "-"]),
                "anchor_sentence_ids": list(anchor_ids),
                "bridge_sentence_ids": list(bridge_ids),
                "answer_sentence_ids": list(answer_ids),
                "ordered_sentence_ids": list(ordered_ids),
            }
        )

    max_total = max_total_sentences if max_total_sentences is not None else max(1, (max_corridors * (max_main + max_support)))
    max_total = max(1, int(max_total))
    kept_bundles, rendered_sentence_ids, truncated_bundle_count, truncated_sentences = _clip_path_bundles_to_budget(
        bundles=bundles,
        max_total_sentences=max_total,
    )
    truncated_corridors += int(truncated_bundle_count)

    lines = [f"Question: {sample.question}", ""]
    if derivation_prompt_enabled:
        lines.extend(
            [
                "[Evidence Derivation Guide]",
                "1) Verify anchor evidence in each path bundle.",
                "2) Verify bridge evidence that links anchor and answer-side facts.",
                "3) Verify answer-facing evidence that supports the final span.",
                "4) Derive the final answer only if 1-3 are consistent.",
                "",
            ]
        )
    rendered_corridor_ids = []
    complete_path_rank = 0
    adjacency_hits = 0
    adjacency_total = 0
    bundles_with_answer = 0
    total_bundle_chunks = 0
    summary_lines = []
    for idx, bundle in enumerate(kept_bundles, start=1):
        anchors = list(bundle.get("anchors", ["-", "-"]) or ["-", "-"])
        if len(anchors) >= 2:
            anchor_text = f"{anchors[0]} ↔ {anchors[1]}"
        elif anchors:
            anchor_text = str(anchors[0])
        else:
            anchor_text = "-"
        lines.append(
            f"[PathBundle {idx} | score={float(bundle.get('corridor_score', 0.0)):.3f} | anchors={anchor_text}]"
        )
        if chain_summary_enabled:
            anchor_sid = next(iter(list(bundle.get("anchor_sentence_ids", []) or [])), "")
            bridge_sid = next(iter(list(bundle.get("bridge_sentence_ids", []) or [])), "")
            answer_sid = next(iter(list(bundle.get("answer_sentence_ids", []) or [])), "")
            anchor_hint = _short_line(sentence_map.get(anchor_sid, ""), max_words=14, max_chars=100)
            bridge_hint = _short_line(sentence_map.get(bridge_sid, ""), max_words=14, max_chars=100)
            answer_hint = _short_line(sentence_map.get(answer_sid, ""), max_words=14, max_chars=100)
            summary_line = (
                f"Summary: anchor={anchor_hint} | bridge={bridge_hint} | answer={answer_hint}"
            )
            lines.append(summary_line)
            summary_lines.append(summary_line)
        role_blocks = (
            ("Anchor", list(bundle.get("anchor_sentence_ids", []) or [])),
            ("Bridge", list(bundle.get("bridge_sentence_ids", []) or [])),
            ("Answer", list(bundle.get("answer_sentence_ids", []) or [])),
        )
        for role_name, sid_list in role_blocks:
            if not sid_list:
                continue
            lines.append(f"{role_name}:")
            for sid in sid_list:
                lines.append(f"- {sentence_map.get(sid, '')}")
        lines.append("")

        rendered_corridor_ids.append(str(bundle.get("corridor_id", "")))
        ordered_ids = list(bundle.get("ordered_sentence_ids", []) or [])
        total_bundle_chunks += int(len(ordered_ids))
        has_anchor = bool(bundle.get("anchor_sentence_ids", []))
        has_bridge = bool(bundle.get("bridge_sentence_ids", []))
        has_answer = bool(bundle.get("answer_sentence_ids", []))
        if has_answer:
            bundles_with_answer += 1
        if complete_path_rank <= 0 and has_anchor and has_bridge and has_answer:
            complete_path_rank = int(idx)
        if has_bridge and has_answer:
            adjacency_total += 1
            bridge_set = set(bundle.get("bridge_sentence_ids", []) or [])
            answer_set = set(bundle.get("answer_sentence_ids", []) or [])
            last_bridge_pos = max((pos for pos, sid in enumerate(ordered_ids) if sid in bridge_set), default=-1)
            first_answer_pos = min((pos for pos, sid in enumerate(ordered_ids) if sid in answer_set), default=10**9)
            if first_answer_pos == last_bridge_pos + 1:
                adjacency_hits += 1

    while lines and not lines[-1].strip():
        lines.pop()

    rendered_sentences = [sentence_map.get(sid, "") for sid in rendered_sentence_ids if sentence_map.get(sid, "")]
    bundle_count = int(len(kept_bundles))
    avg_chunks_per_bundle = float(_safe_ratio(total_bundle_chunks, max(1, bundle_count)))
    bridge_answer_adjacency_rate = float(_safe_ratio(adjacency_hits, max(1, adjacency_total)))
    bundle_dedup_ratio = float(_safe_ratio(dedup_removed_count, max(1, dedup_candidate_count)))
    answer_path_coverage = float(_safe_ratio(bundles_with_answer, max(1, bundle_count)))
    summary_token_count = float(sum(_approx_token_count(line) for line in summary_lines))
    path_focus_count = float(min(bundle_count, focus_n)) if focus_n > 0 else 0.0
    readability_role_signal = 1.0 if bundle_count > 0 else 0.0
    readability_order_signal = 1.0 if complete_path_rank > 0 else 0.0
    readability_adj_signal = max(0.0, min(1.0, bridge_answer_adjacency_rate))
    readability_concise_signal = max(0.0, min(1.0, 1.0 - max(0.0, avg_chunks_per_bundle - 2.5) / 3.0))
    answer_chain_readability_score = (
        0.20 * readability_role_signal
        + 0.20 * readability_order_signal
        + 0.20 * readability_adj_signal
        + 0.15 * readability_concise_signal
        + 0.15 * float(1.0 if chain_summary_enabled else 0.0)
        + 0.10 * float(1.0 if derivation_prompt_enabled else 0.0)
    )
    answer_chain_readability_score = float(max(0.0, min(1.0, answer_chain_readability_score)))

    return RenderedContext(
        sample_id=sample.qid,
        method=retrieval_result.method,
        text="\n".join(lines),
        sentences=rendered_sentences,
        sentence_ids=rendered_sentence_ids,
        truncated=(truncated_corridors > 0 or truncated_sentences > 0),
        render_mode="path_bundle",
        rendered_corridor_ids=_ordered_unique(rendered_corridor_ids),
        truncated_corridor_count=int(truncated_corridors),
        truncated_sentence_count=int(truncated_sentences),
        retrieval_selected_sentence_ids=list(retrieval_result.selected_sentence_ids or []),
        metadata={
            "selected_unit_type": _selected_unit_type(retrieval_result),
            "render_variant": "path_structured_render",
            "path_bundle_ordered_enabled": bool(flags.get("ordered", True)),
            "path_bundle_dedup_enabled": bool(dedup_enabled),
            "path_bundle_compact_lite_enabled": bool(compact_lite),
            "path_bundle_chain_summary_enabled": bool(chain_summary_enabled),
            "path_bundle_derivation_prompt_enabled": bool(derivation_prompt_enabled),
            "path_bundle_focus_n": int(max(0, focus_n)),
            "path_bundle_count": float(bundle_count),
            "path_block_count": float(bundle_count),
            "avg_chunks_per_bundle": float(avg_chunks_per_bundle),
            "bridge_answer_adjacency_rate": float(bridge_answer_adjacency_rate),
            "first_complete_path_rank": float(complete_path_rank),
            "bundle_dedup_ratio": float(bundle_dedup_ratio),
            "answer_path_coverage": float(answer_path_coverage),
            "summary_token_count": float(summary_token_count),
            "path_focus_count": float(path_focus_count),
            "derivation_prompt_activation": float(1.0 if derivation_prompt_enabled else 0.0),
            "answer_chain_readability_score": float(answer_chain_readability_score),
            "conversion_after_path_bundle": 0.0,
        },
    )


def _sentence_id_parts(sentence_id):
    sid = str(sentence_id or "")
    if "::" not in sid:
        return sid, None
    title, idx_str = sid.rsplit("::", 1)
    try:
        return title, int(idx_str)
    except Exception:
        return title, None


def _context_doc_sentence_map(sample):
    docs = {}
    for doc in sample.contexts:
        title = str(doc.title or "")
        if not title:
            continue
        existing = docs.get(title)
        current = list(doc.sentences or [])
        if existing is None or len(current) > len(existing):
            docs[title] = current
    return docs


def _extract_chunk_window_sentences(doc_sentences, title, center_idx, before_n, after_n):
    if not doc_sentences:
        return []
    if center_idx is None:
        center_idx = 0
    center_idx = max(0, min(int(center_idx), len(doc_sentences) - 1))
    start = max(0, int(center_idx) - max(0, int(before_n)))
    end = min(len(doc_sentences) - 1, int(center_idx) + max(0, int(after_n)))
    pairs = []
    for idx in range(start, end + 1):
        text = str(doc_sentences[idx] or "").strip()
        if not text:
            continue
        sid = f"{title}::{idx}"
        pairs.append((sid, text))
    return pairs


def _ordered_corridors(corridors):
    ranked = []
    for idx, corridor in enumerate(corridors or [], start=1):
        if not isinstance(corridor, dict):
            continue
        cid = str(corridor.get("corridor_id", f"c{idx:02d}") or f"c{idx:02d}")
        ranked.append((cid, corridor))
    ranked.sort(key=lambda x: float((x[1] or {}).get("corridor_score", 0.0)), reverse=True)
    return ranked


def _build_sentence_to_corridor_maps(corridors):
    sent_to_corridors = {}
    corridor_rank = {}
    corridor_support_counts = {}
    corridor_bridge_flags = {}
    for rank, (cid, corridor) in enumerate(_ordered_corridors(corridors), start=1):
        corridor_rank[cid] = rank
        support_ids = list((corridor or {}).get("support_sentence_ids", []) or [])
        connector_ids = list((corridor or {}).get("connector_adjacent_sentence_ids", []) or [])
        main_ids = list((corridor or {}).get("main_path_sentence_ids", []) or [])
        corridor_support_counts[cid] = len(_ordered_unique(support_ids + connector_ids))
        anchors = list((corridor or {}).get("anchors", []) or [])
        corridor_bridge_flags[cid] = bool(len([a for a in anchors if str(a or "").strip()]) >= 2)
        for sid in _ordered_unique(main_ids + support_ids + connector_ids):
            sent_to_corridors.setdefault(str(sid), []).append(cid)
    return sent_to_corridors, corridor_rank, corridor_support_counts, corridor_bridge_flags


def _select_excerpt_sentences_from_packages(
    packages,
    base_sentence_ids,
    max_excerpt_sentences,
    dedup_enabled,
):
    max_excerpt = max(0, int(max_excerpt_sentences))
    if max_excerpt <= 0:
        return [], 0, 0

    used = set(base_sentence_ids or [])
    kept_pairs = []
    kept_package_count = 0
    candidate_sentence_count = 0

    for pkg in packages:
        pkg_pairs = list((pkg or {}).get("pairs", []) or [])
        pkg_added = False
        for sid, sent in pkg_pairs:
            sid = str(sid or "")
            text = str(sent or "").strip()
            if not sid or not text:
                continue
            if dedup_enabled and sid in used:
                continue
            candidate_sentence_count += 1
            if len(kept_pairs) >= max_excerpt:
                continue
            kept_pairs.append((sid, text))
            pkg_added = True
            if dedup_enabled:
                used.add(sid)
        if pkg_added:
            kept_package_count += 1

    return kept_pairs, kept_package_count, candidate_sentence_count


def _build_sentence_backfill_packages(
    sample,
    retrieval_result,
    base_sentence_ids,
    rendered_corridor_ids,
    max_per_corridor,
    before_n,
    after_n,
):
    doc_map = _context_doc_sentence_map(sample)
    sentence_rank = {sid: idx for idx, sid in enumerate(base_sentence_ids or [])}
    ranked_corridors = _ordered_corridors(retrieval_result.corridors or [])
    corridor_map = {cid: corridor for cid, corridor in ranked_corridors}
    selected_corridors = [cid for cid in (rendered_corridor_ids or []) if cid in corridor_map]
    if not selected_corridors:
        selected_corridors = [cid for cid, _ in ranked_corridors[:3]]

    packages = []
    per_corr = max(1, int(max_per_corridor))
    for cid in selected_corridors:
        corridor = corridor_map.get(cid, {})
        candidates = _ordered_unique(
            list((corridor or {}).get("main_path_sentence_ids", []) or [])
            + list((corridor or {}).get("support_sentence_ids", []) or [])
        )
        if not candidates:
            continue
        candidates.sort(key=lambda sid: sentence_rank.get(sid, 10**9))
        centers = candidates[:per_corr]
        for center_sid in centers:
            title, center_idx = _sentence_id_parts(center_sid)
            doc_sentences = doc_map.get(title, [])
            excerpt_pairs = _extract_chunk_window_sentences(
                doc_sentences=doc_sentences,
                title=title,
                center_idx=center_idx,
                before_n=before_n,
                after_n=after_n,
            )
            if excerpt_pairs:
                packages.append(
                    {
                        "package_id": f"{cid}:{center_sid}",
                        "pairs": excerpt_pairs,
                    }
                )

    if packages:
        return packages

    # Corridor map may be sparse for some runs; fall back to sentence-centered backfill.
    for idx, sid in enumerate(list(base_sentence_ids or [])[: max(1, per_corr * 2)]):
        title, center_idx = _sentence_id_parts(sid)
        doc_sentences = doc_map.get(title, [])
        excerpt_pairs = _extract_chunk_window_sentences(
            doc_sentences=doc_sentences,
            title=title,
            center_idx=center_idx,
            before_n=before_n,
            after_n=after_n,
        )
        if excerpt_pairs:
            packages.append(
                {
                    "package_id": f"fallback:{idx}:{sid}",
                    "pairs": excerpt_pairs,
                }
            )
    return packages


def _build_corridor_lift_packages(
    sample,
    retrieval_result,
    rendered_corridor_ids,
    before_n,
    after_n,
    top_chunks_per_corridor,
):
    doc_map = _context_doc_sentence_map(sample)
    question_tokens = _token_set(sample.question)
    ranked_corridors = _ordered_corridors(retrieval_result.corridors or [])
    corridor_map = {cid: corridor for cid, corridor in ranked_corridors}
    selected_corridors = [cid for cid in (rendered_corridor_ids or []) if cid in corridor_map]
    if not selected_corridors:
        selected_corridors = [cid for cid, _ in ranked_corridors[:3]]

    packages = []
    topn = max(1, int(top_chunks_per_corridor))
    for cid in selected_corridors:
        corridor = corridor_map.get(cid, {})
        sentence_ids = _ordered_unique(
            list((corridor or {}).get("main_path_sentence_ids", []) or [])
            + list((corridor or {}).get("support_sentence_ids", []) or [])
        )
        if not sentence_ids:
            continue
        by_title = {}
        for sid in sentence_ids:
            title, sent_idx = _sentence_id_parts(sid)
            if sent_idx is None:
                continue
            by_title.setdefault(title, []).append(sent_idx)

        ranked_titles = []
        for title, idx_list in by_title.items():
            doc_sentences = doc_map.get(title, [])
            if not doc_sentences:
                continue
            title_tokens = _token_set(title)
            center = sorted(idx_list)[len(idx_list) // 2]
            center_sent = str(doc_sentences[min(center, len(doc_sentences) - 1)] or "")
            overlap = len(question_tokens.intersection(title_tokens.union(_token_set(center_sent))))
            score = float(len(idx_list)) + 0.1 * float(overlap)
            ranked_titles.append((title, center, score))

        ranked_titles.sort(key=lambda x: x[2], reverse=True)
        for title, center_idx, _ in ranked_titles[:topn]:
            excerpt_pairs = _extract_chunk_window_sentences(
                doc_sentences=doc_map.get(title, []),
                title=title,
                center_idx=center_idx,
                before_n=before_n,
                after_n=after_n,
            )
            if excerpt_pairs:
                packages.append(
                    {
                        "package_id": f"{cid}:{title}:{center_idx}",
                        "pairs": excerpt_pairs,
                    }
                )
    return packages


def _build_package_score_packages(
    sample,
    retrieval_result,
    base_sentence_ids,
    before_n,
    after_n,
    top_k_packages,
    w_answer,
    w_bridge,
    w_support,
    w_chunk_grounding,
    w_redundancy,
):
    doc_map = _context_doc_sentence_map(sample)
    sent_to_corridors, corridor_rank, corridor_support_counts, corridor_bridge_flags = _build_sentence_to_corridor_maps(
        retrieval_result.corridors or []
    )
    question_tokens = _token_set(sample.question)
    sentence_token_cache = {}

    candidates = []
    for rank, sid in enumerate(base_sentence_ids or [], start=1):
        title, sent_idx = _sentence_id_parts(sid)
        if sent_idx is None:
            continue
        doc_sentences = doc_map.get(title, [])
        if not doc_sentences:
            continue
        excerpt_pairs = _extract_chunk_window_sentences(
            doc_sentences=doc_sentences,
            title=title,
            center_idx=sent_idx,
            before_n=before_n,
            after_n=after_n,
        )
        if not excerpt_pairs:
            continue
        excerpt_text = " ".join([txt for _, txt in excerpt_pairs])
        excerpt_tokens = _token_set(excerpt_text)
        sentence_token_cache[sid] = excerpt_tokens
        cids = sent_to_corridors.get(sid, [])
        best_rank = min([corridor_rank.get(cid, 10**9) for cid in cids], default=10**9)
        answer_score = 1.0 / float(rank)
        bridge_score = max([1.0 if corridor_bridge_flags.get(cid, False) else 0.0 for cid in cids], default=0.0)
        support_score = 0.0
        if cids:
            support_score = max([float(corridor_support_counts.get(cid, 0)) for cid in cids]) / 4.0
        grounding_score = 0.0
        if question_tokens:
            grounding_score = float(len(question_tokens.intersection(excerpt_tokens)) / len(question_tokens))
        base_score = (
            float(w_answer) * answer_score
            + float(w_bridge) * bridge_score
            + float(w_support) * support_score
            + float(w_chunk_grounding) * grounding_score
            + (0.05 / float(best_rank + 1 if best_rank < 10**9 else 1000))
        )
        candidates.append(
            {
                "sid": sid,
                "score": float(base_score),
                "pairs": excerpt_pairs,
            }
        )

    if not candidates:
        return []

    k = max(1, int(top_k_packages))
    selected = []
    selected_token_sets = []
    remaining = list(candidates)
    while remaining and len(selected) < k:
        best = None
        best_score = None
        for cand in remaining:
            c_tokens = sentence_token_cache.get(cand["sid"], set())
            redundancy = 0.0
            if selected_token_sets:
                redundancy = max(_jaccard(c_tokens, prev) for prev in selected_token_sets)
            score = float(cand.get("score", 0.0)) - float(w_redundancy) * redundancy
            if best is None or score > best_score:
                best = cand
                best_score = score
        if best is None:
            break
        selected.append(
            {
                "package_id": f"pkg:{best['sid']}",
                "pairs": list(best.get("pairs", []) or []),
            }
        )
        selected_token_sets.append(sentence_token_cache.get(best["sid"], set()))
        remaining = [x for x in remaining if x.get("sid") != best.get("sid")]
    return selected


def _apply_chunk_grounding(
    sample,
    retrieval_result,
    rendered,
    max_context_sentences,
    chunk_grounding_mode,
    chunk_excerpt_max_per_corridor,
    chunk_excerpt_window_sentences_before,
    chunk_excerpt_window_sentences_after,
    chunk_excerpt_max_total_sentences,
    chunk_excerpt_dedup_enabled,
    chunk_grounding_top_corridor_chunks,
    chunk_grounding_top_k_packages,
    package_score_answer_weight,
    package_score_bridge_weight,
    package_score_support_weight,
    package_score_chunk_grounding_weight,
    package_score_redundancy_weight,
):
    sentence_map = _sample_sentence_lookup(sample)
    base_ids = list(rendered.sentence_ids or [])
    if not base_ids:
        meta = dict(rendered.metadata or {})
        meta.update(
            {
                "chunk_grounding_enabled": True,
                "chunk_grounding_mode": str(chunk_grounding_mode or ""),
                "chunk_excerpts_used": 0,
                "chunk_excerpt_sentence_count": 0,
                "evidence_package_count": 0,
                "chunk_excerpt_truncated_count": 0,
            }
        )
        rendered.metadata = meta
        return rendered

    extra_budget = max(0, int(chunk_excerpt_max_total_sentences))
    if max_context_sentences is not None:
        allowed = max(0, int(max_context_sentences) - len(base_ids))
        extra_budget = min(extra_budget, allowed)
    if extra_budget <= 0:
        meta = dict(rendered.metadata or {})
        meta.update(
            {
                "chunk_grounding_enabled": True,
                "chunk_grounding_mode": str(chunk_grounding_mode or ""),
                "chunk_excerpts_used": 0,
                "chunk_excerpt_sentence_count": 0,
                "evidence_package_count": 0,
                "chunk_excerpt_truncated_count": 0,
            }
        )
        rendered.metadata = meta
        return rendered

    mode = str(chunk_grounding_mode or "sentence_backfill").strip().lower()
    if mode == "corridor_lift":
        packages = _build_corridor_lift_packages(
            sample=sample,
            retrieval_result=retrieval_result,
            rendered_corridor_ids=rendered.rendered_corridor_ids,
            before_n=chunk_excerpt_window_sentences_before,
            after_n=chunk_excerpt_window_sentences_after,
            top_chunks_per_corridor=chunk_grounding_top_corridor_chunks,
        )
    elif mode == "package_score":
        packages = _build_package_score_packages(
            sample=sample,
            retrieval_result=retrieval_result,
            base_sentence_ids=base_ids,
            before_n=chunk_excerpt_window_sentences_before,
            after_n=chunk_excerpt_window_sentences_after,
            top_k_packages=chunk_grounding_top_k_packages,
            w_answer=package_score_answer_weight,
            w_bridge=package_score_bridge_weight,
            w_support=package_score_support_weight,
            w_chunk_grounding=package_score_chunk_grounding_weight,
            w_redundancy=package_score_redundancy_weight,
        )
    else:
        packages = _build_sentence_backfill_packages(
            sample=sample,
            retrieval_result=retrieval_result,
            base_sentence_ids=base_ids,
            rendered_corridor_ids=rendered.rendered_corridor_ids,
            max_per_corridor=chunk_excerpt_max_per_corridor,
            before_n=chunk_excerpt_window_sentences_before,
            after_n=chunk_excerpt_window_sentences_after,
        )

    excerpt_pairs, package_count, candidate_sentence_count = _select_excerpt_sentences_from_packages(
        packages=packages,
        base_sentence_ids=base_ids,
        max_excerpt_sentences=extra_budget,
        dedup_enabled=bool(chunk_excerpt_dedup_enabled),
    )

    if not excerpt_pairs:
        meta = dict(rendered.metadata or {})
        meta.update(
            {
                "chunk_grounding_enabled": True,
                "chunk_grounding_mode": mode,
                "chunk_excerpts_used": 0,
                "chunk_excerpt_sentence_count": 0,
                "evidence_package_count": 0,
                "chunk_excerpt_truncated_count": max(0, int(candidate_sentence_count)),
            }
        )
        rendered.metadata = meta
        return rendered

    extra_ids = [sid for sid, _ in excerpt_pairs]
    extra_sentences = [txt for _, txt in excerpt_pairs]
    rendered.sentence_ids = list(base_ids) + extra_ids
    rendered.sentences = [sentence_map.get(sid, "") for sid in base_ids] + extra_sentences

    lines = []
    if str(rendered.text or "").strip():
        lines.append(str(rendered.text))
        lines.append("")
    lines.append("[Grounded Chunk Excerpts]")
    for idx, (sid, sent) in enumerate(excerpt_pairs, start=1):
        lines.append(f"[G{idx}] ({sid}) {sent}")
    rendered.text = "\n".join(lines)

    meta = dict(rendered.metadata or {})
    meta.update(
        {
            "chunk_grounding_enabled": True,
            "chunk_grounding_mode": mode,
            "chunk_excerpts_used": int(package_count),
            "chunk_excerpt_sentence_count": int(len(excerpt_pairs)),
            "evidence_package_count": int(package_count),
            "chunk_excerpt_truncated_count": max(0, int(candidate_sentence_count - len(excerpt_pairs))),
        }
    )
    rendered.metadata = meta
    return rendered


def render_flat_context(sample, retrieval_result, max_context_sentences):
    max_n = max(1, int(max_context_sentences))
    lookup = _build_retrieval_text_lookup(sample, retrieval_result)

    ids = list(retrieval_result.selected_sentence_ids)
    texts = list(retrieval_result.selected_sentences)

    if len(texts) != len(ids):
        texts = [lookup.get(sid, "") for sid in ids]

    pairs = [(sid, txt) for sid, txt in zip(ids, texts) if sid and txt]

    truncated = len(pairs) > max_n
    truncated_sentence_count = max(0, len(pairs) - max_n)
    pairs = pairs[:max_n]

    lines = []
    for i, (sid, sent) in enumerate(pairs, start=1):
        lines.append("[%d] (%s) %s" % (i, sid, sent))

    text = "\n".join(lines)
    return RenderedContext(
        sample_id=sample.qid,
        method=retrieval_result.method,
        text=text,
        sentences=[sent for _, sent in pairs],
        sentence_ids=[sid for sid, _ in pairs],
        truncated=truncated,
        render_mode="flat",
        rendered_corridor_ids=[],
        truncated_corridor_count=0,
        truncated_sentence_count=truncated_sentence_count,
        retrieval_selected_sentence_ids=ids,
    )


def render_corridor_context(
    sample,
    retrieval_result,
    max_corridors_in_context,
    max_main_sentences_per_corridor,
    max_support_per_corridor,
    max_total_sentences,
):
    sentence_map = _build_retrieval_text_lookup(sample, retrieval_result)
    corridors = build_corridor_payload(retrieval_result, graph=None, sentence_map=sentence_map)

    packed, rendered_sentence_ids, rendered_corridor_ids, truncated_corridors, truncated_sentences = pack_corridors_with_budget(
        corridors=corridors,
        sentence_map=sentence_map,
        question_text=sample.question,
        max_corridors_in_context=max_corridors_in_context,
        max_main_sentences_per_corridor=max_main_sentences_per_corridor,
        max_support_per_corridor=max_support_per_corridor,
        max_total_sentences=max_total_sentences,
    )

    lines = [f"Question: {sample.question}", ""]

    for idx, corridor in enumerate(packed, start=1):
        anchors = corridor.get("anchors", ["-", "-"])
        if len(anchors) >= 2:
            anchor_text = f"{anchors[0]} ↔ {anchors[1]}"
        elif anchors:
            anchor_text = anchors[0]
        else:
            anchor_text = "-"

        lines.append(
            f"[Corridor {idx} | score={corridor.get('corridor_score', 0.0):.3f} | anchors={anchor_text}]"
        )

        for sent_idx, sid in enumerate(corridor.get("main_sentence_ids", []), start=1):
            lines.append(f"{sent_idx}. {sentence_map.get(sid, '')}")

        support_ids = corridor.get("support_sentence_ids", [])
        if support_ids:
            lines.append("Support:")
            for sid in support_ids:
                lines.append(f"- {sentence_map.get(sid, '')}")

        lines.append("")

    while lines and not lines[-1].strip():
        lines.pop()

    text = "\n".join(lines)
    rendered_sentences = [sentence_map.get(sid, "") for sid in rendered_sentence_ids if sentence_map.get(sid, "")]

    return RenderedContext(
        sample_id=sample.qid,
        method=retrieval_result.method,
        text=text,
        sentences=rendered_sentences,
        sentence_ids=rendered_sentence_ids,
        truncated=(truncated_corridors > 0 or truncated_sentences > 0),
        render_mode="corridor",
        rendered_corridor_ids=rendered_corridor_ids,
        truncated_corridor_count=truncated_corridors,
        truncated_sentence_count=truncated_sentences,
        retrieval_selected_sentence_ids=list(retrieval_result.selected_sentence_ids or []),
        metadata={"selected_unit_type": _selected_unit_type(retrieval_result)},
    )


def render_corridor_aware_flat_context(
    sample,
    retrieval_result,
    max_context_sentences,
    alpha,
    beta,
    gamma_main,
    delta_support,
    eta_connector,
    zeta_query,
    xi_locality,
    lambda_redundancy,
    top_corridors,
    max_sentences,
    reserve_top_corridor,
    order_strategy,
):
    pairs = _extract_selected_pairs(sample, retrieval_result)
    if not pairs:
        return RenderedContext(
            sample_id=sample.qid,
            method=retrieval_result.method,
            text="",
            sentences=[],
            sentence_ids=[],
            truncated=False,
            render_mode="corridor_aware_flat",
            rendered_corridor_ids=[],
            truncated_corridor_count=0,
            truncated_sentence_count=0,
            retrieval_selected_sentence_ids=list(retrieval_result.selected_sentence_ids or []),
            metadata={"selected_unit_type": _selected_unit_type(retrieval_result)},
        )

    candidate_ids = [sid for sid, _ in pairs]
    candidate_text_map = {sid: sent for sid, sent in pairs}
    retrieval_rank = {sid: idx for idx, sid in enumerate(candidate_ids)}
    features = _get_sentence_feature_table(sample, retrieval_result, candidate_ids, candidate_text_map)

    all_corridor_ids = []
    for feat in features.values():
        all_corridor_ids.extend(feat.get("corridor_ids", []))
    all_corridor_ids = _ordered_unique(all_corridor_ids)

    if top_corridors is not None and int(top_corridors) > 0:
        keep_rank = int(top_corridors)
        filtered_ids = [
            sid
            for sid in candidate_ids
            if int(features.get(sid, {}).get("best_corridor_rank", 10**9)) <= keep_rank
        ]
        if filtered_ids:
            candidate_ids = filtered_ids

    max_n = int(max_sentences)
    if max_context_sentences is not None:
        max_n = min(max_n, int(max_context_sentences))
    max_n = max(1, max_n)

    token_cache = {sid: _token_set(candidate_text_map.get(sid, "")) for sid in candidate_ids}
    base_scores = {
        sid: _corridor_aware_linear_score(
            features.get(sid, {}),
            alpha=alpha,
            beta=beta,
            gamma_main=gamma_main,
            delta_support=delta_support,
            eta_connector=eta_connector,
            zeta_query=zeta_query,
            xi_locality=xi_locality,
        )
        for sid in candidate_ids
    }

    selected_ids = []
    selected_tokens = []
    score_at_pick = {}
    remaining = list(candidate_ids)
    selected_title_counts = {}
    selected_role_counts = {"query": 0, "bridge": 0, "answer": 0}

    def _select_score(sid):
        feat = features.get(sid, {}) or {}
        role = _feature_role(feat, query_overlap_threshold=1.0)
        redundancy = 0.0
        if selected_tokens:
            redundancy = max(_jaccard(token_cache.get(sid, set()), prev) for prev in selected_tokens)
        score = float(base_scores.get(sid, 0.0)) - float(lambda_redundancy) * redundancy

        # Bridge-priority boost (connector-side evidence), preserving backward compatibility via eta_connector.
        if bool(feat.get("is_connector_adjacent", False)):
            score += 0.25 * max(0.0, float(eta_connector))

        # Diversity control: avoid over-selecting the same title/chunk family.
        title = _unit_title(sid)
        title_repeats = int(selected_title_counts.get(title, 0))
        score -= 0.20 * max(0.0, float(lambda_redundancy)) * float(title_repeats)

        # Keep answer-side and connector-side evidence balanced.
        balance_w = 0.10 * max(0.0, float(eta_connector) + float(delta_support))
        bridge_count = int(selected_role_counts.get("bridge", 0))
        answer_count = int(selected_role_counts.get("answer", 0))
        if role == "bridge":
            score -= balance_w * float(max(0, bridge_count - answer_count))
        elif role == "answer":
            score -= balance_w * float(max(0, answer_count - bridge_count - 1))

        return float(score), role, title

    def _pick_best(eligible_ids):
        best_sid = None
        best_score = None
        best_role = "answer"
        best_title = ""
        for sid in eligible_ids:
            score, role, title = _select_score(sid)
            if (
                (best_sid is None)
                or (score > best_score)
                or (
                    abs(score - best_score) <= 1.0e-12
                    and retrieval_rank.get(sid, 10**9) < retrieval_rank.get(best_sid, 10**9)
                )
            ):
                best_sid = sid
                best_score = score
                best_role = role
                best_title = title
        return best_sid, float(best_score or 0.0), best_role, best_title

    def _accept(sid, score, role, title):
        selected_ids.append(sid)
        selected_tokens.append(token_cache.get(sid, set()))
        score_at_pick[sid] = float(score)
        selected_title_counts[title] = int(selected_title_counts.get(title, 0)) + 1
        selected_role_counts[role] = int(selected_role_counts.get(role, 0)) + 1
        if sid in remaining:
            remaining.remove(sid)

    if reserve_top_corridor:
        corridor_rank_map = {}
        for sid in candidate_ids:
            feat = features.get(sid, {}) or {}
            rank = int(feat.get("best_corridor_rank", 10**9) or 10**9)
            for cid in feat.get("corridor_ids", []) or []:
                corridor_rank_map[cid] = min(int(corridor_rank_map.get(cid, 10**9)), rank)
        corridor_order = sorted(corridor_rank_map.keys(), key=lambda cid: corridor_rank_map.get(cid, 10**9))
        quota_corridors = int(top_corridors) if (top_corridors is not None and int(top_corridors) > 0) else len(corridor_order)
        for cid in corridor_order[:quota_corridors]:
            if len(selected_ids) >= max_n:
                break
            eligible = [sid for sid in remaining if cid in (features.get(sid, {}) or {}).get("corridor_ids", [])]
            if not eligible:
                continue
            sid, score, role, title = _pick_best(eligible)
            if sid is None:
                continue
            _accept(sid, score, role, title)

    while remaining and len(selected_ids) < max_n:
        best_sid, best_score, best_role, best_title = _pick_best(remaining)
        if best_sid is None:
            break
        _accept(best_sid, best_score, best_role, best_title)

    strategy, strategy_flags, top_slice_topk, support_pin_min, raw_focus_top_bundle_only_n = _parse_order_strategy_flags(
        order_strategy=order_strategy,
        retrieval_result=retrieval_result,
    )
    raw_focus_front_enabled = bool("raw_focus_front" in strategy_flags)
    raw_focus_dedup_enabled = bool("raw_focus_dedup" in strategy_flags)
    raw_focus_scaffold_ab1_enabled = bool("raw_focus_scaffold_ab1" in strategy_flags)
    raw_focus_scaffold_ab2_enabled = bool("raw_focus_scaffold_ab2" in strategy_flags)
    raw_focus_scaffold_ab3_enabled = bool("raw_focus_scaffold_ab3" in strategy_flags)
    answer_cue_highlight_enabled = bool("answer_cue_highlight" in strategy_flags)
    corridor_grouped_flat_enabled = bool("corridor_grouped_flat" in strategy_flags)
    role_tagged_compact_enabled = bool("role_tagged_compact" in strategy_flags)
    gen_quote_then_answer_light_enabled = bool("gen_quote_then_answer_light" in strategy_flags)
    gen_grounded_answer_light_enabled = bool("gen_grounded_answer_light" in strategy_flags)
    gen_evidence_focus_light_enabled = bool("gen_evidence_focus_light" in strategy_flags)
    gen_evidence_verify_light_enabled = bool("gen_evidence_verify_light" in strategy_flags)
    gen_light_ab_chain_enabled = bool("gen_light_ab_chain" in strategy_flags)
    gen_light_ab_wording_enabled = bool("gen_light_ab_wording" in strategy_flags)
    raw_focus_scaffold_light_enabled = bool(
        ("raw_focus_scaffold_light" in strategy_flags)
        or raw_focus_scaffold_ab1_enabled
        or raw_focus_scaffold_ab2_enabled
        or raw_focus_scaffold_ab3_enabled
    )
    raw_focus_scaffold_text = ""
    if raw_focus_scaffold_ab1_enabled:
        raw_focus_scaffold_text = "Use the earliest linked evidence chain to answer."
    elif raw_focus_scaffold_ab2_enabled:
        raw_focus_scaffold_text = "Answer only after connecting entity evidence and answer-bearing evidence."
    elif raw_focus_scaffold_ab3_enabled:
        raw_focus_scaffold_text = "Prefer the evidence chain that directly supports the answer entity."
    elif raw_focus_scaffold_light_enabled:
        raw_focus_scaffold_text = "Use the earliest evidence chain that links entity, bridge, and answer."
    answer_bearing_maps = _build_answer_bearing_maps(retrieval_result)
    answer_bearing_sentence_scores = dict(answer_bearing_maps.get("sentence_scores", {}) or {})
    answer_bearing_corridor_scores = dict(answer_bearing_maps.get("corridor_scores", {}) or {})

    if strategy == "retrieval":
        selected_ids = sorted(selected_ids, key=lambda sid: retrieval_rank.get(sid, 10**9))
    elif strategy == "corridor_rank":
        selected_ids = sorted(
            selected_ids,
            key=lambda sid: (
                int(features.get(sid, {}).get("best_corridor_rank", 10**9)),
                -float(score_at_pick.get(sid, base_scores.get(sid, 0.0))),
                retrieval_rank.get(sid, 10**9),
            ),
        )
    elif strategy in {"query_bridge_answer", "qba"}:
        role_priority = {"query": 0, "bridge": 1, "answer": 2}
        selected_ids = sorted(
            selected_ids,
            key=lambda sid: (
                int((features.get(sid, {}) or {}).get("best_corridor_rank", 10**9) or 10**9),
                int(role_priority.get(_feature_role(features.get(sid, {}) or {}, query_overlap_threshold=1.0), 3)),
                -float(score_at_pick.get(sid, base_scores.get(sid, 0.0))),
                retrieval_rank.get(sid, 10**9),
            ),
        )
    else:  # score
        selected_ids = sorted(
            selected_ids,
            key=lambda sid: (
                float(score_at_pick.get(sid, base_scores.get(sid, 0.0))),
                -retrieval_rank.get(sid, 10**9),
            ),
            reverse=True,
        )

    top_slice_diag = {"applied": False, "head_size": 0, "reordered": 0}
    if "top_slice_reorder" in strategy_flags:
        selected_ids, top_slice_diag = _apply_top_slice_reorder(
            selected_ids=selected_ids,
            features=features,
            score_at_pick=score_at_pick,
            base_scores=base_scores,
            retrieval_rank=retrieval_rank,
            topk=top_slice_topk,
        )

    support_pin_diag = {"applied": False, "added": 0, "replaced": 0, "target": int(max(0, support_pin_min))}
    if "answer_support_pin" in strategy_flags:
        selected_ids, support_pin_diag = _apply_answer_support_pinning(
            selected_ids=selected_ids,
            candidate_ids=candidate_ids,
            features=features,
            score_at_pick=score_at_pick,
            base_scores=base_scores,
            retrieval_rank=retrieval_rank,
            min_support=support_pin_min,
        )

    oracle_injection_diag = {"applied": False, "injected": 0}
    if "oracle_support" in strategy_flags:
        selected_ids, oracle_injection_diag = _apply_oracle_support_injection(
            selected_ids=selected_ids,
            candidate_text_map=candidate_text_map,
            sample=sample,
            retrieval_result=retrieval_result,
            max_n=max_n,
        )

    top_bundle_only_diag = {"enabled": False, "applied": False, "n": 0, "kept_corridors": 0}
    if int(raw_focus_top_bundle_only_n) > 0:
        top_bundle_only_diag["enabled"] = True
        top_bundle_only_diag["n"] = int(raw_focus_top_bundle_only_n)
        answer_bearing_order = list(answer_bearing_maps.get("corridor_order", []) or [])
        keep_corridors = set(answer_bearing_order[: int(raw_focus_top_bundle_only_n)])
        if keep_corridors:
            filtered = []
            for sid in selected_ids:
                cids = list((features.get(sid, {}) or {}).get("corridor_ids", []) or [])
                if any(cid in keep_corridors for cid in cids):
                    filtered.append(sid)
            if filtered:
                selected_ids = list(filtered[:max_n])
                top_bundle_only_diag["applied"] = True
                top_bundle_only_diag["kept_corridors"] = int(len(keep_corridors))

    raw_focus_diag = {"applied": False, "promoted": 0}
    if raw_focus_front_enabled:
        selected_ids, raw_focus_diag = _apply_raw_focus_front_order(
            selected_ids=selected_ids,
            features=features,
            score_at_pick=score_at_pick,
            base_scores=base_scores,
            retrieval_rank=retrieval_rank,
            answer_bearing_sentence_scores=answer_bearing_sentence_scores,
        )

    raw_focus_dedup_diag = {"applied": False, "removed": 0, "candidate_count": 0, "refilled": 0}
    if raw_focus_dedup_enabled:
        selected_ids, raw_focus_dedup_diag = _apply_answer_bearing_dedup(
            selected_ids=selected_ids,
            candidate_ids=candidate_ids,
            candidate_text_map=candidate_text_map,
            answer_bearing_sentence_scores=answer_bearing_sentence_scores,
            max_n=max_n,
        )

    corridor_group_diag = {
        "enabled": bool(corridor_grouped_flat_enabled),
        "applied": False,
        "group_count": 0,
        "avg_items_per_group": 0.0,
    }
    grouped_order = []
    if corridor_grouped_flat_enabled and selected_ids:
        corridor_rank_map = {}
        corridor_first_pos = {}
        sid_group = {}
        sid_order_map = {sid: idx for idx, sid in enumerate(selected_ids)}
        for sid in selected_ids:
            feat = dict(features.get(sid, {}) or {})
            corr_ids = list(feat.get("corridor_ids", []) or [])
            rank = int(feat.get("best_corridor_rank", 10**9) or 10**9)
            primary_cid = corr_ids[0] if corr_ids else "__ungrouped__"
            sid_group[sid] = str(primary_cid)
            if primary_cid not in corridor_rank_map:
                corridor_rank_map[primary_cid] = rank
            else:
                corridor_rank_map[primary_cid] = min(int(corridor_rank_map[primary_cid]), int(rank))
            if primary_cid not in corridor_first_pos:
                corridor_first_pos[primary_cid] = int(sid_order_map[sid])

        group_to_ids = {}
        for sid in selected_ids:
            gid = sid_group.get(sid, "__ungrouped__")
            group_to_ids.setdefault(gid, []).append(sid)
        sorted_groups = sorted(
            list(group_to_ids.keys()),
            key=lambda gid: (int(corridor_rank_map.get(gid, 10**9)), int(corridor_first_pos.get(gid, 10**9))),
        )
        regrouped_ids = []
        grouped_order = []
        for gid in sorted_groups:
            g_items = list(group_to_ids.get(gid, []) or [])
            if not g_items:
                continue
            grouped_order.append((str(gid), list(g_items)))
            regrouped_ids.extend(g_items)
        if regrouped_ids:
            corridor_group_diag["applied"] = list(regrouped_ids) != list(selected_ids)
            selected_ids = list(regrouped_ids[:max_n])
            corridor_group_diag["group_count"] = int(len(grouped_order))
            corridor_group_diag["avg_items_per_group"] = float(
                _safe_ratio(len(selected_ids), max(1, int(corridor_group_diag["group_count"])))
            )
            # Keep grouped_order aligned with potentially clipped selected_ids.
            selected_set = set(selected_ids)
            grouped_order = [
                (gid, [sid for sid in sids if sid in selected_set])
                for gid, sids in grouped_order
                if any(sid in selected_set for sid in sids)
            ]

    lines = []
    selected_sentences = []
    answer_cue_highlight_count = 0
    role_tag_applied_count = 0
    role_unknown_count = 0
    role_total_count = 0

    def _render_line(idx, sid):
        nonlocal answer_cue_highlight_count, role_tag_applied_count, role_unknown_count, role_total_count
        sent = candidate_text_map.get(sid, "")
        if not sent:
            return None
        display_sent = str(sent)
        feat = dict(features.get(sid, {}) or {})
        if role_tagged_compact_enabled:
            tag, is_unknown = _role_tag_for_sentence(sid, feat, answer_bearing_sentence_scores)
            if tag:
                display_sent = f"{tag} {display_sent}"
                role_tag_applied_count += 1
            role_total_count += 1
            if bool(is_unknown):
                role_unknown_count += 1
        if answer_cue_highlight_enabled:
            marker = ""
            if sid in answer_bearing_sentence_scores:
                marker = "[ANSWER-CUE] "
            elif _is_support_like(feat):
                marker = "[SUPPORT] "
            if marker:
                display_sent = f"{marker}{display_sent}"
                answer_cue_highlight_count += 1
        selected_sentences.append(sent)
        return "[%d] (%s) %s" % (idx, sid, display_sent)

    if corridor_grouped_flat_enabled and grouped_order:
        line_idx = 1
        for group_idx, (gid, sids) in enumerate(grouped_order, start=1):
            lines.append(f"[Corridor {group_idx} | id={gid}]")
            for sid in list(sids or []):
                row = _render_line(line_idx, sid)
                if row is None:
                    continue
                lines.append(f"- {row}")
                line_idx += 1
            lines.append("")
        while lines and not str(lines[-1]).strip():
            lines.pop()
    else:
        for idx, sid in enumerate(selected_ids, start=1):
            row = _render_line(idx, sid)
            if row is None:
                continue
            lines.append(row)

    rendered_corridor_ids = _ordered_unique(
        [cid for sid in selected_ids for cid in (features.get(sid, {}).get("corridor_ids", []) or [])]
    )
    selected_answer_bearing_chunk_count = int(sum(1 for sid in selected_ids if sid in answer_bearing_sentence_scores))
    selected_answer_bearing_bundle_count = int(sum(1 for cid in rendered_corridor_ids if cid in answer_bearing_corridor_scores))
    first_answer_bearing_chunk_rank = 0
    for idx, sid in enumerate(selected_ids, start=1):
        if sid in answer_bearing_sentence_scores:
            first_answer_bearing_chunk_rank = int(idx)
            break
    first_answer_bearing_bundle_rank = 0
    for idx, cid in enumerate(rendered_corridor_ids, start=1):
        if cid in answer_bearing_corridor_scores:
            first_answer_bearing_bundle_rank = int(idx)
            break

    generation_focus_line_ranks = []
    if gen_evidence_focus_light_enabled:
        for idx, sid in enumerate(selected_ids, start=1):
            if sid in answer_bearing_sentence_scores:
                generation_focus_line_ranks.append(int(idx))
            if len(generation_focus_line_ranks) >= 2:
                break
        if not generation_focus_line_ranks and selected_ids:
            generation_focus_line_ranks = [1]

    generation_intervention_variant = ""
    generation_intervention_text = ""
    if gen_quote_then_answer_light_enabled:
        generation_intervention_variant = "quote_then_answer_light"
        generation_intervention_text = (
            "First identify one short supporting phrase from the context, then output only the final answer span."
        )
    elif gen_grounded_answer_light_enabled:
        generation_intervention_variant = "grounded_answer_light"
        generation_intervention_text = (
            "Answer using the wording most directly supported by the earliest answer-bearing evidence. "
            "If multiple candidates exist, choose the one directly supported by context."
        )
    elif gen_evidence_focus_light_enabled:
        generation_intervention_variant = "evidence_focus_light"
        if generation_focus_line_ranks:
            line_refs = ", ".join([f"[{int(x)}]" for x in generation_focus_line_ranks[:2]])
            generation_intervention_text = (
                f"Prioritize evidence lines {line_refs} when deciding the answer; output only the final answer span."
            )
        else:
            generation_intervention_text = (
                "Prioritize the earliest answer-bearing evidence when deciding the answer; output only the final answer span."
            )
    elif gen_evidence_verify_light_enabled:
        generation_intervention_variant = "evidence_verify_light"
        generation_intervention_text = (
            "Briefly verify your answer is explicitly supported by context before finalizing. "
            "If unsupported, choose the best supported answer span from context."
        )

    if generation_intervention_text and gen_light_ab_chain_enabled:
        generation_intervention_text = (
            generation_intervention_text.rstrip() + " Prefer a consistent entity-to-answer chain from those lines."
        )
    if generation_intervention_text and gen_light_ab_wording_enabled:
        generation_intervention_text = (
            generation_intervention_text.rstrip() + " Prefer the exact surface form stated in those lines when possible."
        )

    truncated_corridors = max(0, len(all_corridor_ids) - len(rendered_corridor_ids))
    truncated_sentences = max(0, len(candidate_ids) - len(selected_ids))

    return RenderedContext(
        sample_id=sample.qid,
        method=retrieval_result.method,
        text="\n".join(lines),
        sentences=selected_sentences,
        sentence_ids=selected_ids,
        truncated=(truncated_corridors > 0 or truncated_sentences > 0),
        render_mode="corridor_aware_flat",
        rendered_corridor_ids=rendered_corridor_ids,
        truncated_corridor_count=truncated_corridors,
        truncated_sentence_count=truncated_sentences,
        retrieval_selected_sentence_ids=list(retrieval_result.selected_sentence_ids or []),
        metadata={
            "selected_unit_type": _selected_unit_type(retrieval_result),
            "render_variant": "corridor_grouped_flat" if corridor_grouped_flat_enabled else ("role_tagged_compact_render" if role_tagged_compact_enabled else "corridor_aware_flat"),
            "final_order_strategy": str(strategy),
            "strategy_flags": sorted(list(strategy_flags)),
            "top_slice_reorder_applied": bool(top_slice_diag.get("applied", False)),
            "top_slice_reorder_head_size": int(top_slice_diag.get("head_size", 0)),
            "top_slice_reorder_reordered": int(top_slice_diag.get("reordered", 0)),
            "answer_support_pinning_applied": bool(support_pin_diag.get("applied", False)),
            "answer_support_pinning_target": int(support_pin_diag.get("target", 0)),
            "answer_support_pinning_added": int(support_pin_diag.get("added", 0)),
            "oracle_support_injection_applied": bool(oracle_injection_diag.get("applied", False)),
            "oracle_support_injected": int(oracle_injection_diag.get("injected", 0)),
            "raw_focus_front_enabled": bool(raw_focus_front_enabled),
            "raw_focus_front_applied": bool(raw_focus_diag.get("applied", False)),
            "raw_focus_front_promoted": int(raw_focus_diag.get("promoted", 0)),
            "raw_focus_dedup_enabled": bool(raw_focus_dedup_enabled),
            "raw_focus_dedup_applied": bool(raw_focus_dedup_diag.get("applied", False)),
            "raw_focus_dedup_removed": int(raw_focus_dedup_diag.get("removed", 0)),
            "raw_focus_dedup_candidate_count": int(raw_focus_dedup_diag.get("candidate_count", 0)),
            "raw_focus_dedup_refilled": int(raw_focus_dedup_diag.get("refilled", 0)),
            "corridor_grouped_flat_enabled": bool(corridor_grouped_flat_enabled),
            "corridor_grouped_flat_applied": bool(corridor_group_diag.get("applied", False)),
            "corridor_group_count": int(corridor_group_diag.get("group_count", 0)),
            "avg_items_per_corridor": float(corridor_group_diag.get("avg_items_per_group", 0.0)),
            "role_tagged_compact_enabled": bool(role_tagged_compact_enabled),
            "role_tagged_compact_applied": bool(role_tag_applied_count > 0),
            "role_tag_applied_count": int(role_tag_applied_count),
            "role_unknown_count": int(role_unknown_count),
            "role_total_count": int(role_total_count),
            "role_unknown_rate": float(_safe_ratio(role_unknown_count, max(1, role_total_count))),
            "raw_focus_top_bundle_only_n": int(raw_focus_top_bundle_only_n),
            "raw_focus_top_bundle_only_applied": bool(top_bundle_only_diag.get("applied", False)),
            "raw_focus_scaffold_light_enabled": bool(raw_focus_scaffold_light_enabled),
            "raw_focus_scaffold_light_text": str(raw_focus_scaffold_text),
            "raw_focus_answer_bearing_chunk_count": int(selected_answer_bearing_chunk_count),
            "raw_focus_answer_bearing_bundle_count": int(selected_answer_bearing_bundle_count),
            "raw_focus_first_answer_bearing_chunk_rank": int(first_answer_bearing_chunk_rank),
            "raw_focus_first_answer_bearing_bundle_rank": int(first_answer_bearing_bundle_rank),
            "answer_cue_highlight_enabled": bool(answer_cue_highlight_enabled),
            "answer_cue_highlight_applied": bool(answer_cue_highlight_count > 0),
            "answer_cue_highlight_count": int(answer_cue_highlight_count),
            "generation_intervention_enabled": bool(generation_intervention_text),
            "generation_intervention_variant": str(generation_intervention_variant),
            "generation_intervention_text": str(generation_intervention_text),
            "generation_intervention_quote_then_answer": bool(gen_quote_then_answer_light_enabled),
            "generation_intervention_grounded_answer": bool(gen_grounded_answer_light_enabled),
            "generation_intervention_evidence_focus": bool(gen_evidence_focus_light_enabled),
            "generation_intervention_evidence_verify": bool(gen_evidence_verify_light_enabled),
            "generation_intervention_focus_count": int(len(generation_focus_line_ranks)),
            "generation_intervention_focus_line_ranks": list(generation_focus_line_ranks),
            "generation_intervention_ab_chain": bool(gen_light_ab_chain_enabled),
            "generation_intervention_ab_wording": bool(gen_light_ab_wording_enabled),
        },
    )


def render_context(
    sample,
    retrieval_result,
    max_context_sentences,
    render_mode="flat",
    max_corridors_in_context=3,
    max_main_sentences_per_corridor=3,
    max_support_per_corridor=2,
    max_total_sentences=12,
    chunk_grounding_enabled=False,
    chunk_grounding_mode="sentence_backfill",
    chunk_excerpt_max_per_corridor=1,
    chunk_excerpt_window_sentences_before=1,
    chunk_excerpt_window_sentences_after=1,
    chunk_excerpt_max_total_sentences=4,
    chunk_excerpt_dedup_enabled=True,
    chunk_grounding_top_corridor_chunks=2,
    chunk_grounding_top_k_packages=4,
    package_score_answer_weight=0.50,
    package_score_bridge_weight=0.20,
    package_score_support_weight=0.15,
    package_score_chunk_grounding_weight=0.15,
    package_score_redundancy_weight=0.10,
    alpha=1.0,
    beta=0.35,
    gamma_main=0.45,
    delta_support=0.20,
    eta_connector=0.20,
    zeta_query=0.12,
    xi_locality=0.08,
    lambda_redundancy=0.25,
    top_corridors=3,
    max_sentences=10,
    reserve_top_corridor=False,
    order_strategy="score",
):
    mode = str(render_mode or "flat").strip().lower()
    rendered = None
    if mode == "corridor_aware_flat":
        rendered = render_corridor_aware_flat_context(
            sample,
            retrieval_result,
            max_context_sentences=max_context_sentences,
            alpha=float(alpha),
            beta=float(beta),
            gamma_main=float(gamma_main),
            delta_support=float(delta_support),
            eta_connector=float(eta_connector),
            zeta_query=float(zeta_query),
            xi_locality=float(xi_locality),
            lambda_redundancy=float(lambda_redundancy),
            top_corridors=int(top_corridors),
            max_sentences=int(max_sentences),
            reserve_top_corridor=bool(reserve_top_corridor),
            order_strategy=order_strategy,
        )
    elif mode == "path_bundle":
        max_corridors = max(1, int(max_corridors_in_context))
        max_main = max(1, int(max_main_sentences_per_corridor))
        max_support = max(0, int(max_support_per_corridor))
        max_total = max_total_sentences if max_total_sentences is not None else max_context_sentences
        if max_context_sentences is not None:
            max_total = min(int(max_total), int(max_context_sentences))
        max_total = max(1, int(max_total))
        rendered = render_path_bundle_context(
            sample,
            retrieval_result,
            max_corridors_in_context=max_corridors,
            max_main_sentences_per_corridor=max_main,
            max_support_per_corridor=max_support,
            max_total_sentences=max_total,
            order_strategy=order_strategy,
        )
    elif mode == "corridor":
        max_corridors = max(1, int(max_corridors_in_context))
        max_main = max(1, int(max_main_sentences_per_corridor))
        max_support = max(0, int(max_support_per_corridor))
        max_total = max_total_sentences if max_total_sentences is not None else max_context_sentences
        if max_context_sentences is not None:
            max_total = min(int(max_total), int(max_context_sentences))
        max_total = max(1, int(max_total))
        rendered = render_corridor_context(
            sample,
            retrieval_result,
            max_corridors_in_context=max_corridors,
            max_main_sentences_per_corridor=max_main,
            max_support_per_corridor=max_support,
            max_total_sentences=max_total,
        )
    else:
        rendered = render_flat_context(sample, retrieval_result, max_context_sentences=max_context_sentences)

    meta = dict(rendered.metadata or {})
    if not str(meta.get("render_variant", "") or "").strip():
        meta["render_variant"] = str(rendered.render_mode or mode or "flat")
    rendered.metadata = meta

    if _retrieval_graph_mode(retrieval_result) == "entity_chunk_graph":
        meta = dict(rendered.metadata or {})
        meta.update({
            "chunk_grounding_enabled": False,
            "native_chunk_context": bool(_selected_unit_type(retrieval_result) == "chunk"),
        })
        rendered.metadata = meta
        return _attach_stagewise_render_diagnostics(sample=sample, retrieval_result=retrieval_result, rendered=rendered)

    if not bool(chunk_grounding_enabled):
        return _attach_stagewise_render_diagnostics(sample=sample, retrieval_result=retrieval_result, rendered=rendered)

    rendered = _apply_chunk_grounding(
        sample=sample,
        retrieval_result=retrieval_result,
        rendered=rendered,
        max_context_sentences=max_context_sentences,
        chunk_grounding_mode=chunk_grounding_mode,
        chunk_excerpt_max_per_corridor=chunk_excerpt_max_per_corridor,
        chunk_excerpt_window_sentences_before=chunk_excerpt_window_sentences_before,
        chunk_excerpt_window_sentences_after=chunk_excerpt_window_sentences_after,
        chunk_excerpt_max_total_sentences=chunk_excerpt_max_total_sentences,
        chunk_excerpt_dedup_enabled=chunk_excerpt_dedup_enabled,
        chunk_grounding_top_corridor_chunks=chunk_grounding_top_corridor_chunks,
        chunk_grounding_top_k_packages=chunk_grounding_top_k_packages,
        package_score_answer_weight=package_score_answer_weight,
        package_score_bridge_weight=package_score_bridge_weight,
        package_score_support_weight=package_score_support_weight,
        package_score_chunk_grounding_weight=package_score_chunk_grounding_weight,
        package_score_redundancy_weight=package_score_redundancy_weight,
    )
    return _attach_stagewise_render_diagnostics(sample=sample, retrieval_result=retrieval_result, rendered=rendered)
