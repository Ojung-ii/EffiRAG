from .types import RenderedContext
from .utils import content_tokens


def _sample_sentence_lookup(sample):
    lookup = {}
    for doc in sample.contexts:
        for sent_idx, sent in enumerate(doc.sentences):
            sid = "%s::%d" % (doc.title, sent_idx)
            lookup[sid] = sent
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
    lookup = _sample_sentence_lookup(sample)
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


def render_flat_context(sample, retrieval_result, max_context_sentences):
    max_n = max(1, int(max_context_sentences))
    lookup = _sample_sentence_lookup(sample)

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
    sentence_map = _sample_sentence_lookup(sample)
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
    if mode == "corridor_aware_flat":
        return render_corridor_aware_flat_context(
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

    if mode == "corridor":
        max_corridors = max(1, int(max_corridors_in_context))
        max_main = max(1, int(max_main_sentences_per_corridor))
        max_support = max(0, int(max_support_per_corridor))
        max_total = max_total_sentences if max_total_sentences is not None else max_context_sentences
        if max_context_sentences is not None:
            max_total = min(int(max_total), int(max_context_sentences))
        max_total = max(1, int(max_total))
        return render_corridor_context(
            sample,
            retrieval_result,
            max_corridors_in_context=max_corridors,
            max_main_sentences_per_corridor=max_main,
            max_support_per_corridor=max_support,
            max_total_sentences=max_total,
        )

    return render_flat_context(sample, retrieval_result, max_context_sentences=max_context_sentences)
