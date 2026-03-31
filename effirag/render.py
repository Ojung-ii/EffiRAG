import re
from .types import RenderedContext
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


def _is_chunk_like_id(unit_id):
    sid = str(unit_id or "")
    return sid.startswith("chunk::")


def _split_chunk_text_to_sentences(text):
    raw = str(text or "").strip()
    if not raw:
        return []
    normalized = re.sub(r"\s+", " ", raw)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", normalized) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\s*\n+\s*", raw) if p.strip()]
    if len(parts) <= 1:
        parts = [normalized]
    return parts


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
        score_map = corridor.get("sentence_score_map", {}) or {}
        main_ids = corridor.get("main_path_sentence_ids", []) or []
        support_ids = corridor.get("support_sentence_ids", []) or []
        connector_ids = corridor.get("connector_adjacent_sentence_ids", []) or []

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
    raw = (retrieval_result.diagnostics or {}).get("sentence_feature_table", {}) or {}
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
    main_ids = _ordered_unique(corridor.get("main_path_sentence_ids", []) or [])
    main_ids = [sid for sid in main_ids if sid in sentence_map]
    if main_ids:
        return main_ids

    support_ids = _ordered_unique(corridor.get("support_sentence_ids", []) or [])
    support_ids = [sid for sid in support_ids if sid in sentence_map]
    score_map = corridor.get("sentence_score_map", {}) or {}
    if score_map:
        support_ids = sorted(support_ids, key=lambda sid: float(score_map.get(sid, 0.0)), reverse=True)
    return support_ids


def select_support_sentences(corridor, main_sentence_ids, max_support_per_corridor, sentence_map=None):
    if max_support_per_corridor is not None and max_support_per_corridor <= 0:
        return []

    main_set = set(main_sentence_ids)
    support_ids = _ordered_unique(corridor.get("support_sentence_ids", []) or [])
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
        support_candidates = _prefer_non_duplicate(support_candidates, used_sentence_ids)
        ranked_support = _rank_sentence_ids(
            candidate_ids=support_candidates,
            sentence_map=sentence_map,
            sentence_score_map=score_map,
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
        main_ids = list((corridor or {}).get("main_path_sentence_ids", []) or [])
        corridor_support_counts[cid] = len(support_ids)
        anchors = list((corridor or {}).get("anchors", []) or [])
        corridor_bridge_flags[cid] = bool(len([a for a in anchors if str(a or "").strip()]) >= 2)
        for sid in _ordered_unique(main_ids + support_ids):
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


def _build_chunk_native_package_score_packages(
    sample,
    retrieval_result,
    base_sentence_ids,
    base_text_lookup,
    top_k_packages,
    w_answer,
    w_bridge,
    w_support,
    w_chunk_grounding,
    w_redundancy,
):
    question_tokens = _token_set(sample.question)
    feature_table = _get_sentence_feature_table(sample, retrieval_result, base_sentence_ids, base_text_lookup)
    sentence_token_cache = {}
    candidates = []

    for rank, sid in enumerate(base_sentence_ids or [], start=1):
        sid = str(sid or "")
        if not _is_chunk_like_id(sid):
            continue
        chunk_text = str(base_text_lookup.get(sid, "") or "").strip()
        if not chunk_text:
            continue
        chunk_sents = _split_chunk_text_to_sentences(chunk_text)
        if not chunk_sents:
            continue
        scored_sents = []
        for idx, sent in enumerate(chunk_sents):
            sent_tokens = _token_set(sent)
            overlap = 0.0
            if question_tokens:
                overlap = float(len(question_tokens.intersection(sent_tokens)) / max(1, len(question_tokens)))
            prior = 1.0 / float(idx + 1)
            sent_score = overlap + 0.15 * prior
            scored_sents.append((sent_score, idx, sent, sent_tokens))
        scored_sents.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scored_sents[:2]
        excerpt_pairs = [(f"{sid}#e{idx}", sent) for _, idx, sent, _ in top_sentences]
        if not excerpt_pairs:
            continue
        excerpt_tokens = set()
        for _, _, _, toks in top_sentences:
            excerpt_tokens.update(toks)
        sentence_token_cache[sid] = excerpt_tokens

        feat = feature_table.get(sid, {}) or {}
        answer_score = 1.0 / float(rank)
        bridge_score = 1.0 if bool(feat.get("is_connector_adjacent", False)) else 0.0
        support_score = 1.0 if bool(feat.get("is_support_candidate", False)) else 0.0
        grounding_score = 0.0
        if question_tokens:
            grounding_score = float(len(question_tokens.intersection(excerpt_tokens)) / len(question_tokens))
        base_score = (
            float(w_answer) * answer_score
            + float(w_bridge) * bridge_score
            + float(w_support) * support_score
            + float(w_chunk_grounding) * grounding_score
        )
        candidates.append({
            "sid": sid,
            "score": float(base_score),
            "pairs": excerpt_pairs,
        })

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
        selected.append({
            "package_id": f"pkg:{best['sid']}",
            "pairs": list(best.get("pairs", []) or []),
        })
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
    chunk_like_base = any(_is_chunk_like_id(sid) for sid in base_ids)
    if chunk_like_base:
        if mode != "package_score":
            meta = dict(rendered.metadata or {})
            meta.update({
                "chunk_grounding_enabled": False,
                "chunk_grounding_skipped": True,
                "chunk_grounding_skip_reason": "entity_chunk_graph_only_supports_package_score",
            })
            rendered.metadata = meta
            return rendered
        base_text_lookup = _build_retrieval_text_lookup(sample, retrieval_result)
        packages = _build_chunk_native_package_score_packages(
            sample=sample,
            retrieval_result=retrieval_result,
            base_sentence_ids=base_ids,
            base_text_lookup=base_text_lookup,
            top_k_packages=chunk_grounding_top_k_packages,
            w_answer=package_score_answer_weight,
            w_bridge=package_score_bridge_weight,
            w_support=package_score_support_weight,
            w_chunk_grounding=package_score_chunk_grounding_weight,
            w_redundancy=package_score_redundancy_weight,
        )
    elif mode == "corridor_lift":
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
    if chunk_like_base:
        base_lookup = _build_retrieval_text_lookup(sample, retrieval_result)
        rendered.sentences = [base_lookup.get(sid, "") for sid in base_ids] + extra_sentences
    else:
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

    if reserve_top_corridor:
        top_candidates = [sid for sid in remaining if int(features.get(sid, {}).get("best_corridor_rank", 10**9)) == 1]
        if top_candidates:
            top_candidates.sort(
                key=lambda sid: (
                    base_scores.get(sid, 0.0),
                    -retrieval_rank.get(sid, 10**9),
                ),
                reverse=True,
            )
            sid = top_candidates[0]
            selected_ids.append(sid)
            selected_tokens.append(token_cache.get(sid, set()))
            score_at_pick[sid] = base_scores.get(sid, 0.0)
            remaining = [x for x in remaining if x != sid]

    while remaining and len(selected_ids) < max_n:
        best_sid = None
        best_score = None
        for sid in remaining:
            redundancy = 0.0
            if selected_tokens:
                redundancy = max(_jaccard(token_cache.get(sid, set()), prev) for prev in selected_tokens)
            score = float(base_scores.get(sid, 0.0)) - float(lambda_redundancy) * redundancy
            if best_sid is None or score > best_score:
                best_sid = sid
                best_score = score
        if best_sid is None:
            break
        selected_ids.append(best_sid)
        selected_tokens.append(token_cache.get(best_sid, set()))
        score_at_pick[best_sid] = float(best_score or 0.0)
        remaining = [x for x in remaining if x != best_sid]

    strategy = str(order_strategy or "score").strip().lower()
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
    else:  # score
        selected_ids = sorted(
            selected_ids,
            key=lambda sid: (
                float(score_at_pick.get(sid, base_scores.get(sid, 0.0))),
                -retrieval_rank.get(sid, 10**9),
            ),
            reverse=True,
        )

    lines = []
    selected_sentences = []
    for idx, sid in enumerate(selected_ids, start=1):
        sent = candidate_text_map.get(sid, "")
        if not sent:
            continue
        selected_sentences.append(sent)
        lines.append("[%d] (%s) %s" % (idx, sid, sent))

    rendered_corridor_ids = _ordered_unique(
        [cid for sid in selected_ids for cid in (features.get(sid, {}).get("corridor_ids", []) or [])]
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

    if _retrieval_graph_mode(retrieval_result) == "entity_chunk_graph":
        if not bool(chunk_grounding_enabled):
            meta = dict(rendered.metadata or {})
            meta.update({
                "chunk_grounding_enabled": False,
                "chunk_grounding_skipped": True,
                "chunk_grounding_skip_reason": "entity_chunk_graph_chunk_package_disabled",
            })
            rendered.metadata = meta
            return rendered
        if str(chunk_grounding_mode or "").strip().lower() != "package_score":
            meta = dict(rendered.metadata or {})
            meta.update({
                "chunk_grounding_enabled": False,
                "chunk_grounding_skipped": True,
                "chunk_grounding_skip_reason": "entity_chunk_graph_only_supports_package_score",
            })
            rendered.metadata = meta
            return rendered

    if not bool(chunk_grounding_enabled):
        return rendered

    return _apply_chunk_grounding(
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
