from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple


SOURCE_ORDER: Tuple[str, ...] = (
    "semantic",
    "entity_title",
    "graph_flow",
    "anchor_neighborhood",
)


def canonical_sentence_id(candidate: Mapping[str, Any]) -> str:
    sid = str(candidate.get("source_id", "") or "").strip()
    if sid:
        return sid
    node_id = str(candidate.get("node", "") or candidate.get("node_id", "") or "").strip()
    return node_id


def _copy_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(candidate or {})
    tags = set(str(x) for x in list(out.get("source_tags", []) or []) if str(x))
    out["source_tags"] = tags
    return out


def _proposal_key(candidate: Mapping[str, Any]) -> Tuple[float, float, float, str]:
    return (
        float(candidate.get("proposal_score", 0.0) or 0.0),
        float(candidate.get("flow_score", 0.0) or 0.0),
        float(candidate.get("semantic_score", 0.0) or 0.0),
        str(candidate.get("node", "") or candidate.get("node_id", "") or ""),
    )


def source_balanced_union(
    ranked_candidates: Mapping[str, Iterable[Mapping[str, Any]]],
    quotas: Mapping[str, int],
    top_m: int,
    fill_remaining: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build a deterministic source-balanced proposal union.

    The function is intentionally source-agnostic: callers provide ranked
    candidates for each existing source, and this helper only enforces fixed
    quotas plus optional score-based backfill.
    """
    t0 = time.perf_counter()
    limit = max(1, int(top_m))
    quota_by_source = {src: max(0, int(quotas.get(src, 0) or 0)) for src in SOURCE_ORDER}

    by_sid: Dict[str, Dict[str, Any]] = {}
    normalized_ranked: Dict[str, List[Dict[str, Any]]] = {src: [] for src in SOURCE_ORDER}

    for source in SOURCE_ORDER:
        for raw in list(ranked_candidates.get(source, []) or []):
            row = _copy_candidate(raw)
            sid = canonical_sentence_id(row)
            if not sid:
                continue
            row.setdefault("source_id", sid)
            row["source_tags"].add(source)
            if sid not in by_sid:
                by_sid[sid] = row
            else:
                by_sid[sid]["source_tags"].update(row.get("source_tags", set()) or set())
                if _proposal_key(row) > _proposal_key(by_sid[sid]):
                    keep_tags = set(by_sid[sid].get("source_tags", set()) or set())
                    by_sid[sid].update(row)
                    by_sid[sid]["source_tags"] = keep_tags.union(row.get("source_tags", set()) or set())
            normalized_ranked[source].append(by_sid[sid])

    selected: List[Dict[str, Any]] = []
    seen = set()
    selected_by_counts = {src: 0 for src in SOURCE_ORDER}
    selected_by_source: Dict[str, str] = {}

    for source in SOURCE_ORDER:
        quota = int(quota_by_source.get(source, 0) or 0)
        if quota <= 0:
            continue
        source_count = 0
        for row in normalized_ranked.get(source, []):
            sid = canonical_sentence_id(row)
            if not sid or sid in seen:
                continue
            selected.append(row)
            seen.add(sid)
            selected_by_source[sid] = source
            source_count += 1
            selected_by_counts[source] = source_count
            if len(selected) >= limit or source_count >= quota:
                break
        if len(selected) >= limit:
            break

    fill_count = 0
    if fill_remaining and len(selected) < limit:
        pooled = sorted(list(by_sid.values()), key=_proposal_key, reverse=True)
        for row in pooled:
            sid = canonical_sentence_id(row)
            if not sid or sid in seen:
                continue
            selected.append(row)
            seen.add(sid)
            selected_by_source[sid] = "fill_remaining"
            fill_count += 1
            if len(selected) >= limit:
                break

    selected = selected[:limit]
    selected_ids = [canonical_sentence_id(row) for row in selected]
    membership_counts = {
        src: int(sum(1 for row in selected if src in set(row.get("source_tags", set()) or set())))
        for src in SOURCE_ORDER
    }
    quota_fill_rates = {
        src: (
            float(selected_by_counts.get(src, 0)) / float(quota_by_source[src])
            if int(quota_by_source[src]) > 0
            else 0.0
        )
        for src in SOURCE_ORDER
    }
    diagnostics = {
        "enabled": True,
        "top_m": int(limit),
        "quotas": dict(quota_by_source),
        "fill_remaining": bool(fill_remaining),
        "num_unique_candidates": int(len(by_sid)),
        "num_selected_candidates": int(len(selected)),
        "selected_ids": selected_ids,
        "selected_by_source": dict(selected_by_source),
        "selected_by_counts": dict(selected_by_counts),
        "fill_remaining_count": int(fill_count),
        "source_membership_counts": dict(membership_counts),
        "quota_fill_rates": dict(quota_fill_rates),
        "source_balanced_union_ms": float((time.perf_counter() - t0) * 1000.0),
    }
    return selected, diagnostics


def disabled_source_balance_diagnostics(top_m: int) -> Dict[str, Any]:
    return {
        "enabled": False,
        "top_m": int(max(1, int(top_m))),
        "quotas": {src: 0 for src in SOURCE_ORDER},
        "fill_remaining": False,
        "num_unique_candidates": 0,
        "num_selected_candidates": 0,
        "selected_ids": [],
        "selected_by_source": {},
        "selected_by_counts": {src: 0 for src in SOURCE_ORDER},
        "fill_remaining_count": 0,
        "source_membership_counts": {src: 0 for src in SOURCE_ORDER},
        "quota_fill_rates": {src: 0.0 for src in SOURCE_ORDER},
        "source_balanced_union_ms": 0.0,
    }

