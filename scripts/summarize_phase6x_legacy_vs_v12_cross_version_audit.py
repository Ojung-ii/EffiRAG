#!/usr/bin/env python3
"""Summarize PHASE6X legacy-vs-v12 cross-version stagewise audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from effirag.evidence_flow_audit import load_samples_for_dataset, stage_metrics
from effirag.utils import safe_div


STAGE_ORDER = [
    "S1_proposal_candidates",
    "S2_phase1_shortlist",
    "S3_local_refinement_or_corridor",
    "S4_sentence_candidates",
    "S5_final_selected_or_renderable",
    "S6_rendered_context",
    "S7_generation_output",
]


YES_NO_RE = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had)\b", flags=re.IGNORECASE)
DATE_NUM_RE = re.compile(
    r"\b(\d{4}|how many|how much|number of|year|date|population|age|older|younger|first|last)\b",
    flags=re.IGNORECASE,
)
ENTITY_WORD_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
COMPARISON_RE = re.compile(
    r"\b(which|more|less|larger|smaller|earlier|later|older|younger|highest|lowest|first|last|before|after)\b",
    flags=re.IGNORECASE,
)


def _safe_text(v: Any) -> str:
    return str(v or "").strip()


def _safe_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _safe_int(v: Any, d: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(d)


def _mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in list(values or [])]
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        token = _safe_text(value)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_for_match(text: str) -> str:
    return _normalize_space(text).lower()


def _normalize_sentence_support_id(sentence_id: str) -> str:
    sid = _safe_text(sentence_id)
    if not sid:
        return ""
    parts = sid.split("::")
    if len(parts) >= 3 and parts[0] in {"chunk", "sentence"}:
        return f"{parts[1]}::{parts[2]}"
    return sid


def _infer_query_type(question: str) -> str:
    q = _safe_text(question)
    ql = q.lower()
    if not ql:
        return "other"
    if YES_NO_RE.search(ql):
        return "yes_no"
    if COMPARISON_RE.search(ql):
        return "comparison"
    if DATE_NUM_RE.search(ql):
        return "date_or_number"
    entity_like = ENTITY_WORD_RE.findall(q)
    if len(entity_like) >= 2 or " of " in ql:
        return "bridge_entity"
    if any(tok in ql for tok in ("who", "where", "organization", "person", "city", "country", "state", "located")):
        return "person_location_org"
    return "other"


def _parse_unit_id(unit_id: str) -> Tuple[str, Optional[int]]:
    sid = _safe_text(unit_id)
    if not sid:
        return "", None
    parts = sid.split("::")
    if len(parts) >= 3 and parts[0] in {"chunk", "sentence"}:
        title = _safe_text(parts[1])
        try:
            idx = int(parts[2])
        except Exception:
            idx = None
        return title, idx
    if len(parts) >= 2:
        title = _safe_text(parts[0])
        try:
            idx = int(parts[1])
        except Exception:
            idx = None
        return title, idx
    return sid, None


def _answer_aliases(sample: Any) -> List[str]:
    aliases: List[str] = []
    answer = _safe_text(getattr(sample, "answer", ""))
    if answer:
        aliases.append(answer)
    meta = dict(getattr(sample, "metadata", {}) or {})
    for key in ("answer_aliases", "aliases", "normalized_aliases", "answer_alias"):
        raw = meta.get(key, [])
        if isinstance(raw, (str, bytes)):
            raw = [str(raw)]
        for value in list(raw or []):
            v = _safe_text(value)
            if v:
                aliases.append(v)
    out: List[str] = []
    seen = set()
    for alias in aliases:
        norm = _normalize_for_match(alias)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(alias)
    return out


def _answer_sufficiency_metrics(sample: Any, texts: Sequence[str]) -> Dict[str, Any]:
    aliases = _answer_aliases(sample)
    norm_aliases = [_normalize_for_match(x) for x in aliases if _normalize_for_match(x)]
    sent_norm = [_normalize_for_match(x) for x in list(texts or [])]
    bearing_indices: List[int] = []
    if norm_aliases:
        for i, sentence in enumerate(sent_norm):
            if any(alias in sentence for alias in norm_aliases):
                bearing_indices.append(i)
    answer_hit = 1.0 if bearing_indices else 0.0
    first_rank = int(bearing_indices[0]) if bearing_indices else -1
    bearing_count = int(len(bearing_indices))
    sent_count = int(len(sent_norm))
    density = float(bearing_count / float(max(1, sent_count))) if sent_count > 0 else 0.0
    early_hit = 1.0 if any(i < 3 for i in bearing_indices) else 0.0
    return {
        "answer_string_hit": float(answer_hit),
        "answer_string_rank": float(first_rank),
        "answer_bearing_sentence_count": int(bearing_count),
        "answer_bearing_density": float(density),
        "answer_context_early_hit": float(early_hit),
    }


def _token_count_estimate(texts: Sequence[str]) -> int:
    total = 0
    for txt in list(texts or []):
        total += max(1, len(_safe_text(txt).split()))
    return int(total)


def _count_units(unit_ids: Sequence[str]) -> Dict[str, int]:
    entity = 0
    chunk = 0
    sentence = 0
    for sid in list(unit_ids or []):
        s = _safe_text(sid)
        if not s:
            continue
        if s.startswith("e::"):
            entity += 1
            continue
        if s.startswith("chunk::") or s.startswith("c::"):
            chunk += 1
            continue
        if "::" in s:
            sentence += 1
            continue
        sentence += 1
    return {
        "entity_node_count": int(entity),
        "chunk_node_count": int(chunk),
        "sentence_node_count": int(sentence),
    }


def _gold_support_ids(sample: Any) -> List[str]:
    out: List[str] = []
    for title, sent_idx in list(getattr(sample, "supporting_facts", []) or []):
        t = _safe_text(title)
        if not t:
            continue
        try:
            idx = int(sent_idx)
        except Exception:
            idx = 0
        out.append(f"{t}::{idx}")
    return _ordered_unique(out)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aset = {str(x) for x in list(a or []) if str(x)}
    bset = {str(x) for x in list(b or []) if str(x)}
    if not aset and not bset:
        return 1.0
    den = len(aset.union(bset))
    if den <= 0:
        return 0.0
    return float(len(aset.intersection(bset)) / float(den))


@dataclass
class RunBundle:
    run_name: str
    method_label: str
    dataset: str
    profile: str
    config_path: str
    query_path: Path
    summary_path: Path
    summary: Dict[str, Any]
    rows: List[Dict[str, Any]]


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, Mapping):
            out.append(dict(obj))
    return out


def _find_latest_file(root: Path, filename: str) -> Optional[Path]:
    candidates = sorted(root.rglob(filename))
    if not candidates:
        return None
    return candidates[-1]


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
    f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
    sf_p = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
    sf_r = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
    sf_f1 = (2.0 * sf_p * sf_r / (sf_p + sf_r)) if (sf_p + sf_r) > 0 else 0.0
    prompt_tokens = _safe_float(
        summary.get("avg_context_tokens", summary.get("prompt_tokens_avg", summary.get("avg_prompt_tokens", 0.0))),
        0.0,
    )
    return {
        "EM": em,
        "F1": f1,
        "Recall@5": _safe_float(summary.get("recall_at_5", summary.get("R@5", 0.0)), 0.0),
        "avg_context_tokens": prompt_tokens,
        "F1_per_1k_context_tokens": (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0,
        "SF_precision": sf_p,
        "SF_recall": sf_r,
        "SF_f1": sf_f1,
        "answer_string_hit": _safe_float(summary.get("answer_surface_present", 0.0), 0.0),
        "answer_bearing_density": _safe_float(summary.get("answer_surface_token_density", 0.0), 0.0),
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
    }


def _query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    m = dict(row.get("metrics", {}) or {})
    e = dict(row.get("efficiency", {}) or {})
    gdiag = dict(row.get("generation_diagnostics", {}) or {})
    prompt_tokens = _safe_float(gdiag.get("prompt_tokens", row.get("prompt_tokens", 0.0)), 0.0)
    f1 = _safe_float(m.get("f1", 0.0), 0.0)
    return {
        "em": _safe_float(m.get("em", 0.0), 0.0),
        "f1": f1,
        "sf_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
        "sf_precision": _safe_float(m.get("supporting_fact_precision", 0.0), 0.0),
        "retrieval_ms": _safe_float(e.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(e.get("generation_latency_ms", 0.0), 0.0),
        "total_ms": _safe_float(e.get("total_latency_ms", 0.0), 0.0),
        "prompt_tokens": prompt_tokens,
        "f1_per_1k": (f1 * 1000.0 / prompt_tokens) if prompt_tokens > 0 else 0.0,
    }


def _stage_unavailable(stage_name: str) -> Dict[str, Any]:
    return {
        "stage": stage_name,
        "stage_available": False,
        "candidate_count": None,
        "entity_node_count": None,
        "chunk_node_count": None,
        "sentence_node_count": None,
        "atom_count": None,
        "estimated_token_count": None,
        "SF_recall": None,
        "SF_precision": None,
        "SF_f1": None,
        "answer_string_hit": None,
        "answer_string_rank": None,
        "answer_bearing_sentence_count": None,
        "answer_bearing_density": None,
        "matched_gold_sentence_ids": [],
        "missing_gold_sentence_ids": [],
        "unit_ids": [],
        "texts": [],
    }


def _stage_available(
    sample: Any,
    stage_name: str,
    unit_ids: Sequence[str],
    texts: Sequence[str],
    graph_mode: str,
    *,
    atom_count: Optional[int] = None,
    entity_node_count: Optional[int] = None,
    chunk_node_count: Optional[int] = None,
    sentence_node_count: Optional[int] = None,
) -> Dict[str, Any]:
    ids = _ordered_unique([_safe_text(x) for x in list(unit_ids or []) if _safe_text(x)])
    text_map: Dict[str, str] = {}
    for idx, sid in enumerate(list(unit_ids or [])):
        ss = _safe_text(sid)
        if not ss or ss in text_map:
            continue
        txt = str(texts[idx] if idx < len(list(texts or [])) else "")
        text_map[ss] = txt
    aligned_texts = [str(text_map.get(sid, "") or "") for sid in ids]
    st = stage_metrics(sample, ids, aligned_texts, graph_mode)
    ans = _answer_sufficiency_metrics(sample, aligned_texts)
    gold_ids = _gold_support_ids(sample)
    matched = list(st.get("matched_gold_ids", []) or [])
    missing = [gid for gid in gold_ids if gid not in set(matched)]
    counts = _count_units(ids)
    return {
        "stage": stage_name,
        "stage_available": True,
        "candidate_count": int(len(ids)),
        "entity_node_count": int(entity_node_count if entity_node_count is not None else counts["entity_node_count"]),
        "chunk_node_count": int(chunk_node_count if chunk_node_count is not None else counts["chunk_node_count"]),
        "sentence_node_count": int(sentence_node_count if sentence_node_count is not None else counts["sentence_node_count"]),
        "atom_count": int(atom_count if atom_count is not None else len(ids)),
        "estimated_token_count": int(st.get("tokens", _token_count_estimate(aligned_texts))),
        "SF_recall": float(st.get("sf_R", 0.0)),
        "SF_precision": float(st.get("sf_P", 0.0)),
        "SF_f1": float(st.get("sf_F1", 0.0)),
        "answer_string_hit": float(ans["answer_string_hit"]),
        "answer_string_rank": float(ans["answer_string_rank"]),
        "answer_bearing_sentence_count": int(ans["answer_bearing_sentence_count"]),
        "answer_bearing_density": float(ans["answer_bearing_density"]),
        "matched_gold_sentence_ids": list(matched),
        "missing_gold_sentence_ids": list(missing),
        "unit_ids": ids,
        "texts": aligned_texts,
    }


def _extract_legacy_stage_dict(sample: Any, row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    ret = dict((row.get("retrieval", {}) or {}))
    diag = dict((ret.get("diagnostics", {}) or {}))
    graph_mode = _safe_text(diag.get("graph_mode", "entity_chunk_graph")) or "entity_chunk_graph"
    selected_text_map = dict(diag.get("selected_text_map", {}) or {})

    out: Dict[str, Dict[str, Any]] = {}

    # S1: proposal candidates (legacy often lacks sentence-level IDs)
    proposal_count = None
    if ("proposal_entity_count" in diag) or ("proposal_chunk_count" in diag):
        proposal_count = int(_safe_int(diag.get("proposal_entity_count", 0), 0) + _safe_int(diag.get("proposal_chunk_count", 0), 0))
    s1 = _stage_unavailable("S1_proposal_candidates")
    if proposal_count is not None:
        s1.update(
            {
                "candidate_count": int(proposal_count),
                "entity_node_count": int(_safe_int(diag.get("proposal_entity_count", 0), 0)),
                "chunk_node_count": int(_safe_int(diag.get("proposal_chunk_count", 0), 0)),
            }
        )
    out["S1_proposal_candidates"] = s1

    # S2: phase1 shortlist (legacy usually run-level only)
    s2 = _stage_unavailable("S2_phase1_shortlist")
    if "phase1_run_count" in diag:
        s2["candidate_count"] = int(_safe_int(diag.get("phase1_run_count", 0), 0))
    out["S2_phase1_shortlist"] = s2

    # S3: local refinement / corridor
    corridor_ids: List[str] = []
    for c in list(ret.get("corridors", []) or []):
        if not isinstance(c, Mapping):
            continue
        for k in (
            "main_path_unit_ids",
            "support_unit_ids",
            "connector_adjacent_unit_ids",
            "main_path_sentence_ids",
            "support_sentence_ids",
            "connector_adjacent_sentence_ids",
        ):
            for sid in list(c.get(k, []) or []):
                ss = _safe_text(sid)
                if ss:
                    corridor_ids.append(ss)
    corridor_ids = _ordered_unique(corridor_ids)
    corridor_texts = [str(selected_text_map.get(sid, "") or "") for sid in corridor_ids]
    if corridor_ids:
        out["S3_local_refinement_or_corridor"] = _stage_available(
            sample,
            "S3_local_refinement_or_corridor",
            corridor_ids,
            corridor_texts,
            graph_mode,
            atom_count=0,
        )
    else:
        out["S3_local_refinement_or_corridor"] = _stage_unavailable("S3_local_refinement_or_corridor")

    # S4: sentence candidates
    sentence_table = dict(diag.get("sentence_feature_table", {}) or {})
    sentence_ids = _ordered_unique([_safe_text(k) for k in sentence_table.keys() if _safe_text(k)])
    sentence_texts = [str(selected_text_map.get(sid, "") or "") for sid in sentence_ids]
    if sentence_ids:
        out["S4_sentence_candidates"] = _stage_available(
            sample,
            "S4_sentence_candidates",
            sentence_ids,
            sentence_texts,
            graph_mode,
            atom_count=0,
        )
    else:
        out["S4_sentence_candidates"] = _stage_unavailable("S4_sentence_candidates")

    # S5: final selected (legacy selected sentence/context units)
    s5_ids = list(ret.get("selected_sentence_ids", []) or [])
    s5_texts = list(ret.get("selected_sentences", []) or [])
    if s5_ids:
        out["S5_final_selected_or_renderable"] = _stage_available(
            sample,
            "S5_final_selected_or_renderable",
            s5_ids,
            s5_texts,
            graph_mode,
            atom_count=0,
        )
    else:
        out["S5_final_selected_or_renderable"] = _stage_unavailable("S5_final_selected_or_renderable")

    # S6: rendered context
    rendered = dict((row.get("rendered", {}) or {}))
    s6_ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
    s6_texts = list(rendered.get("sentences", []) or [])
    if s6_ids:
        out["S6_rendered_context"] = _stage_available(
            sample,
            "S6_rendered_context",
            s6_ids,
            s6_texts,
            graph_mode,
            atom_count=0,
        )
    else:
        out["S6_rendered_context"] = _stage_unavailable("S6_rendered_context")

    # S7 generation output
    s7 = _stage_unavailable("S7_generation_output")
    m = dict(row.get("metrics", {}) or {})
    pred = _safe_text(row.get("prediction", ""))
    if pred or bool(row.get("qa_executed", False)):
        ans = _answer_sufficiency_metrics(sample, [pred] if pred else [])
        s7.update(
            {
                "stage_available": True,
                "candidate_count": 1,
                "estimated_token_count": int(_token_count_estimate([pred] if pred else [])),
                "SF_recall": float(m.get("rendered_supporting_fact_recall", m.get("supporting_fact_recall", 0.0))),
                "SF_precision": float(m.get("rendered_supporting_fact_precision", m.get("supporting_fact_precision", 0.0))),
                "SF_f1": float(m.get("rendered_supporting_fact_f1", 0.0)),
                "answer_string_hit": float(ans["answer_string_hit"]),
                "answer_string_rank": float(ans["answer_string_rank"]),
                "answer_bearing_sentence_count": int(ans["answer_bearing_sentence_count"]),
                "answer_bearing_density": float(ans["answer_bearing_density"]),
            }
        )
    out["S7_generation_output"] = s7
    return out


def _extract_v12_stage_dict(sample: Any, row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    ret = dict((row.get("retrieval", {}) or {}))
    diag = dict((ret.get("diagnostics", {}) or {}))
    graph_mode = _safe_text(diag.get("graph_mode", "current_entity_graph")) or "current_entity_graph"
    out: Dict[str, Dict[str, Any]] = {}

    # S1 proposal candidates
    s1_ids = list(diag.get("phase6u_proposal_candidate_unit_ids", []) or [])
    s1_texts = list(diag.get("phase6u_proposal_candidate_texts", []) or [])
    if s1_ids:
        out["S1_proposal_candidates"] = _stage_available(
            sample,
            "S1_proposal_candidates",
            s1_ids,
            s1_texts,
            graph_mode,
            atom_count=0,
            entity_node_count=_safe_int(diag.get("proposal_entity_count", 0), 0),
            chunk_node_count=_safe_int(diag.get("proposal_chunk_count", 0), 0),
        )
    else:
        out["S1_proposal_candidates"] = _stage_unavailable("S1_proposal_candidates")

    # S2 phase1 shortlist
    s2_ids = list(diag.get("phase6u_phase1_shortlist_unit_ids", []) or [])
    s2_texts = list(diag.get("phase6u_phase1_shortlist_texts", []) or [])
    if s2_ids:
        out["S2_phase1_shortlist"] = _stage_available(
            sample,
            "S2_phase1_shortlist",
            s2_ids,
            s2_texts,
            graph_mode,
            atom_count=0,
        )
    else:
        out["S2_phase1_shortlist"] = _stage_unavailable("S2_phase1_shortlist")

    # S3 local refinement / corridor
    s3_ids = list(diag.get("phase6u_corridor_after_trim_unit_ids", []) or [])
    s3_texts = list(diag.get("phase6u_corridor_after_trim_texts", []) or [])
    if s3_ids:
        out["S3_local_refinement_or_corridor"] = _stage_available(
            sample,
            "S3_local_refinement_or_corridor",
            s3_ids,
            s3_texts,
            graph_mode,
            atom_count=0,
        )
    else:
        out["S3_local_refinement_or_corridor"] = _stage_unavailable("S3_local_refinement_or_corridor")

    # S4 sentence candidates
    s4_ids = list(diag.get("phase6u_sentence_candidates_after_text_rerank_ids", []) or [])
    s4_texts = list(diag.get("phase6u_sentence_candidates_after_text_rerank_texts", []) or [])
    if s4_ids:
        out["S4_sentence_candidates"] = _stage_available(
            sample,
            "S4_sentence_candidates",
            s4_ids,
            s4_texts,
            graph_mode,
            atom_count=0,
        )
    else:
        out["S4_sentence_candidates"] = _stage_unavailable("S4_sentence_candidates")

    # S5 ABR selected evidence
    s5_ids = list(diag.get("phase6u_abr_selected_evidence_ids", []) or ret.get("selected_sentence_ids", []) or [])
    s5_texts = list(diag.get("phase6u_abr_selected_evidence_texts", []) or ret.get("selected_sentences", []) or [])
    atom_count = _safe_int((diag.get("unified_acr_rcedr_diag", {}) or {}).get("num_selected_atoms", len(s5_ids)), len(s5_ids))
    if s5_ids:
        out["S5_final_selected_or_renderable"] = _stage_available(
            sample,
            "S5_final_selected_or_renderable",
            s5_ids,
            s5_texts,
            graph_mode,
            atom_count=atom_count,
        )
    else:
        out["S5_final_selected_or_renderable"] = _stage_unavailable("S5_final_selected_or_renderable")

    # S6 rendered
    rendered = dict((row.get("rendered", {}) or {}))
    s6_ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
    s6_texts = list(rendered.get("sentences", []) or [])
    if s6_ids:
        out["S6_rendered_context"] = _stage_available(
            sample,
            "S6_rendered_context",
            s6_ids,
            s6_texts,
            graph_mode,
            atom_count=0,
        )
    else:
        out["S6_rendered_context"] = _stage_unavailable("S6_rendered_context")

    # S7 generation output
    s7 = _stage_unavailable("S7_generation_output")
    m = dict(row.get("metrics", {}) or {})
    pred = _safe_text(row.get("prediction", ""))
    if pred or bool(row.get("qa_executed", False)):
        ans = _answer_sufficiency_metrics(sample, [pred] if pred else [])
        s7.update(
            {
                "stage_available": True,
                "candidate_count": 1,
                "estimated_token_count": int(_token_count_estimate([pred] if pred else [])),
                "SF_recall": float(m.get("rendered_supporting_fact_recall", m.get("supporting_fact_recall", 0.0))),
                "SF_precision": float(m.get("rendered_supporting_fact_precision", m.get("supporting_fact_precision", 0.0))),
                "SF_f1": float(m.get("rendered_supporting_fact_f1", 0.0)),
                "answer_string_hit": float(ans["answer_string_hit"]),
                "answer_string_rank": float(ans["answer_string_rank"]),
                "answer_bearing_sentence_count": int(ans["answer_bearing_sentence_count"]),
                "answer_bearing_density": float(ans["answer_bearing_density"]),
            }
        )
    out["S7_generation_output"] = s7
    return out


def _transfer_deltas(stage_map: Mapping[str, Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    def sf(stage: str) -> Optional[float]:
        item = dict(stage_map.get(stage, {}) or {})
        if not bool(item.get("stage_available", False)):
            return None
        v = item.get("SF_recall", None)
        if v is None:
            return None
        return float(v)

    def ah(stage: str) -> Optional[float]:
        item = dict(stage_map.get(stage, {}) or {})
        if not bool(item.get("stage_available", False)):
            return None
        v = item.get("answer_string_hit", None)
        if v is None:
            return None
        return float(v)

    def diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(b - a)

    s1 = sf("S1_proposal_candidates")
    s2 = sf("S2_phase1_shortlist")
    s3 = sf("S3_local_refinement_or_corridor")
    s4 = sf("S4_sentence_candidates")
    s5 = sf("S5_final_selected_or_renderable")
    s6 = sf("S6_rendered_context")
    a1 = ah("S1_proposal_candidates")
    a2 = ah("S2_phase1_shortlist")
    a3 = ah("S3_local_refinement_or_corridor")
    a4 = ah("S4_sentence_candidates")
    a5 = ah("S5_final_selected_or_renderable")
    a6 = ah("S6_rendered_context")
    candidate_to_rendered_sf = None
    if s1 is not None and s6 is not None and s1 > 0:
        candidate_to_rendered_sf = float(safe_div(s6, s1))
    candidate_to_rendered_answer = None
    if a1 is not None and a6 is not None and a1 > 0:
        candidate_to_rendered_answer = float(safe_div(a6, a1))
    sentence_to_rendered_answer = None
    if a4 is not None and a6 is not None and a4 > 0:
        sentence_to_rendered_answer = float(safe_div(a6, a4))
    return {
        "proposal_to_phase1_SF_delta": diff(s1, s2),
        "phase1_to_refinement_SF_delta": diff(s2, s3),
        "refinement_to_sentence_SF_delta": diff(s3, s4),
        "sentence_to_final_SF_delta": diff(s4, s5),
        "final_to_rendered_SF_delta": diff(s5, s6),
        "proposal_to_phase1_answer_hit_delta": diff(a1, a2),
        "phase1_to_refinement_answer_hit_delta": diff(a2, a3),
        "refinement_to_sentence_answer_hit_delta": diff(a3, a4),
        "sentence_to_final_answer_hit_delta": diff(a4, a5),
        "final_to_rendered_answer_hit_delta": diff(a5, a6),
        "candidate_to_rendered_SF_transfer_rate": candidate_to_rendered_sf,
        "candidate_to_rendered_answer_transfer_rate": candidate_to_rendered_answer,
        "sentence_to_rendered_answer_transfer_rate": sentence_to_rendered_answer,
    }


def _failure_stage(stage_map: Mapping[str, Mapping[str, Any]]) -> str:
    # Earliest stage where SF recall drops from previous available stage.
    prev_name = None
    prev_r = None
    for stage in STAGE_ORDER:
        item = dict(stage_map.get(stage, {}) or {})
        if not bool(item.get("stage_available", False)):
            continue
        cur = item.get("SF_recall", None)
        if cur is None:
            continue
        cur_r = float(cur)
        if prev_r is not None and (cur_r + 1e-12) < prev_r:
            return stage
        prev_r = cur_r
        prev_name = stage
    if prev_name is None:
        return "unknown"
    return "no_stage_drop"


def _paired_case_classification(legacy: Mapping[str, Any], v12: Mapping[str, Any]) -> str:
    legacy_f1 = float(legacy.get("legacy_F1", 0.0))
    v12_f1 = float(v12.get("v12_F1", 0.0))
    legacy_sf = float(legacy.get("legacy_rendered_SF_recall", 0.0))
    v12_sf = float(v12.get("v12_rendered_SF_recall", 0.0))
    legacy_hit = float(legacy.get("legacy_answer_string_hit", 0.0))
    v12_hit = float(v12.get("v12_answer_string_hit", 0.0))
    if legacy_f1 > (v12_f1 + 0.10) and legacy_sf <= (v12_sf + 1e-12):
        return "legacy_high_F1_low_SF"
    if v12_sf >= (legacy_sf - 1e-12) and legacy_f1 > (v12_f1 + 0.10):
        return "v12_high_SF_low_F1"
    if legacy_f1 > (v12_f1 + 0.10):
        return "legacy_better_answer_sufficient"
    if v12_f1 > (legacy_f1 + 0.10):
        return "v12_better_compact"
    if legacy_f1 >= 0.5 and v12_f1 >= 0.5:
        return "both_good"
    if legacy_f1 < 0.1 and v12_f1 < 0.1:
        return "both_bad"
    if (legacy_f1 > v12_f1 and legacy_hit < v12_hit) or (v12_f1 > legacy_f1 and v12_hit < legacy_hit):
        return "mixed"
    return "neutral"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _write_examples_md(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for i, row in enumerate(rows, start=1):
        lines.append(f"## {i}. {_safe_text(row.get('dataset'))} / {_safe_text(row.get('sample_id'))}")
        lines.append(f"- question: {_safe_text(row.get('question'))}")
        lines.append(f"- gold_answer: {_safe_text(row.get('gold_answer'))}")
        lines.append(f"- legacy_prediction: {_safe_text(row.get('legacy_prediction'))}")
        lines.append(f"- v12_prediction: {_safe_text(row.get('v12_prediction'))}")
        lines.append(
            f"- legacy_F1/v12_F1: {_fmt(row.get('legacy_F1'))} / {_fmt(row.get('v12_F1'))} | legacy_SF_R/v12_SF_R: {_fmt(row.get('legacy_rendered_SF_recall'))} / {_fmt(row.get('v12_rendered_SF_recall'))}"
        )
        lines.append(
            f"- legacy_answer_hit/v12_answer_hit: {_fmt(row.get('legacy_answer_string_hit'))} / {_fmt(row.get('v12_answer_string_hit'))} | rendered_jaccard: {_fmt(row.get('rendered_sentence_jaccard'))}"
        )
        lines.append(
            f"- legacy_failure_stage: {_safe_text(row.get('legacy_failure_stage'))} | v12_failure_stage: {_safe_text(row.get('v12_failure_stage'))}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage_available_value(stage_map: Mapping[str, Mapping[str, Any]], stage: str, key: str) -> Optional[float]:
    item = dict(stage_map.get(stage, {}) or {})
    if not bool(item.get("stage_available", False)):
        return None
    v = item.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _load_run_bundle(
    out_root: Path,
    run_item: Mapping[str, Any],
) -> Tuple[Optional[RunBundle], Optional[str]]:
    run_name = _safe_text(run_item.get("run_name"))
    run_root = Path(_safe_text(run_item.get("run_root")))
    dataset = _safe_text(run_item.get("dataset"))
    method_label = _safe_text(run_item.get("method_label"))
    profile = _safe_text(run_item.get("profile"))
    config_path = _safe_text(run_item.get("config_path"))
    if not run_root.exists():
        return None, f"{run_name}: run_root_missing"
    qpath = _find_latest_file(run_root, "rag_query_results.jsonl")
    spath = _find_latest_file(run_root, "rag_summary.json")
    if qpath is None or spath is None:
        return None, f"{run_name}: missing_output_files"
    try:
        rows = _load_jsonl(qpath)
        summary = _load_json(spath)
    except Exception as exc:
        return None, f"{run_name}: parse_error:{exc}"
    return (
        RunBundle(
            run_name=run_name,
            method_label=method_label,
            dataset=dataset,
            profile=profile,
            config_path=config_path,
            query_path=qpath,
            summary_path=spath,
            summary=summary,
            rows=rows,
        ),
        None,
    )


def build_audit(out_root: Path, datasets: Sequence[str], top_k: int) -> Dict[str, Any]:
    manifest_path = out_root / "phase6x_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    run_items = list(manifest.get("runs", []) or [])
    run_bundles: List[RunBundle] = []
    missing_outputs = 0
    parse_errors = 0
    for item in run_items:
        bundle, err = _load_run_bundle(out_root, item)
        if bundle is None:
            if err and "parse_error" in err:
                parse_errors += 1
            else:
                missing_outputs += 1
            continue
        run_bundles.append(bundle)

    sample_maps = {d: load_samples_for_dataset(d) for d in datasets}
    # rows grouped by dataset/method
    grouped: Dict[Tuple[str, str], RunBundle] = {}
    for b in run_bundles:
        grouped[(b.dataset, b.method_label)] = b
    run_level_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    for b in run_bundles:
        s = dict(b.summary or {})
        run_level_metrics[(b.dataset, b.method_label)] = {
            "retrieval_ms": _safe_float(s.get("retrieval_ms", s.get("retrieval_latency_ms", 0.0)), 0.0),
            "generation_ms": _safe_float(s.get("generation_ms", s.get("generation_latency_ms", 0.0)), 0.0),
            "total_ms": _safe_float(s.get("total_ms", s.get("total_latency_ms", 0.0)), 0.0),
        }

    per_query_pairs: List[Dict[str, Any]] = []
    stage_rows_by_dataset: List[Dict[str, Any]] = []
    failure_stage_rows: List[Dict[str, Any]] = []

    normalization_errors = 0
    sample_id_mismatch = 0

    for dataset in datasets:
        legacy_bundle = grouped.get((dataset, "legacy"))
        v12_bundle = grouped.get((dataset, "v12"))
        if legacy_bundle is None or v12_bundle is None:
            continue
        legacy_by_id = {str(r.get("sample_id", "")): r for r in legacy_bundle.rows}
        v12_by_id = {str(r.get("sample_id", "")): r for r in v12_bundle.rows}
        legacy_ids = [_safe_text(r.get("sample_id")) for r in legacy_bundle.rows if _safe_text(r.get("sample_id"))]
        v12_ids = [_safe_text(r.get("sample_id")) for r in v12_bundle.rows if _safe_text(r.get("sample_id"))]
        if legacy_ids != v12_ids:
            sample_id_mismatch += 1
        paired_ids = [sid for sid in legacy_ids if sid in v12_by_id]

        for sid in paired_ids:
            sample = sample_maps.get(dataset, {}).get(sid)
            if sample is None:
                normalization_errors += 1
                continue
            lrow = legacy_by_id.get(sid, {})
            vrow = v12_by_id.get(sid, {})

            try:
                legacy_stage_map = _extract_legacy_stage_dict(sample, lrow)
                v12_stage_map = _extract_v12_stage_dict(sample, vrow)
            except Exception:
                normalization_errors += 1
                continue

            legacy_qm = _query_metrics(lrow)
            v12_qm = _query_metrics(vrow)
            legacy_render = dict((lrow.get("rendered", {}) or {}))
            v12_render = dict((vrow.get("rendered", {}) or {}))

            legacy_s6 = dict(legacy_stage_map.get("S6_rendered_context", {}) or {})
            v12_s6 = dict(v12_stage_map.get("S6_rendered_context", {}) or {})
            legacy_s5 = dict(legacy_stage_map.get("S5_final_selected_or_renderable", {}) or {})
            v12_s5 = dict(v12_stage_map.get("S5_final_selected_or_renderable", {}) or {})

            legacy_transfers = _transfer_deltas(legacy_stage_map)
            v12_transfers = _transfer_deltas(v12_stage_map)
            legacy_failure = _failure_stage(legacy_stage_map)
            v12_failure = _failure_stage(v12_stage_map)

            row = {
                "dataset": dataset,
                "sample_id": sid,
                "question": _safe_text(getattr(sample, "question", lrow.get("question", ""))),
                "gold_answer": _safe_text(getattr(sample, "answer", lrow.get("answer", ""))),
                "query_type": _infer_query_type(_safe_text(getattr(sample, "question", lrow.get("question", "")))),
                "legacy_prediction": _safe_text(lrow.get("prediction", "")),
                "v12_prediction": _safe_text(vrow.get("prediction", "")),
                "legacy_EM": float(legacy_qm["em"]),
                "v12_EM": float(v12_qm["em"]),
                "legacy_F1": float(legacy_qm["f1"]),
                "v12_F1": float(v12_qm["f1"]),
                "legacy_rendered_tokens": float(legacy_qm["prompt_tokens"]),
                "v12_rendered_tokens": float(v12_qm["prompt_tokens"]),
                "legacy_F1_per_1k": float(legacy_qm["f1_per_1k"]),
                "v12_F1_per_1k": float(v12_qm["f1_per_1k"]),
                "legacy_rendered_SF_recall": float(legacy_s6.get("SF_recall", lrow.get("metrics", {}).get("rendered_supporting_fact_recall", 0.0)) or 0.0),
                "v12_rendered_SF_recall": float(v12_s6.get("SF_recall", vrow.get("metrics", {}).get("rendered_supporting_fact_recall", 0.0)) or 0.0),
                "legacy_answer_string_hit": float(legacy_s6.get("answer_string_hit", 0.0) or 0.0),
                "v12_answer_string_hit": float(v12_s6.get("answer_string_hit", 0.0) or 0.0),
                "legacy_answer_bearing_density": float(legacy_s6.get("answer_bearing_density", 0.0) or 0.0),
                "v12_answer_bearing_density": float(v12_s6.get("answer_bearing_density", 0.0) or 0.0),
                "legacy_failure_stage": str(legacy_failure),
                "v12_failure_stage": str(v12_failure),
                "selected_sentence_jaccard": float(
                    _jaccard(
                        list((lrow.get("retrieval", {}) or {}).get("selected_sentence_ids", []) or []),
                        list((vrow.get("retrieval", {}) or {}).get("selected_sentence_ids", []) or []),
                    )
                ),
                "rendered_sentence_jaccard": float(
                    _jaccard(
                        list(lrow.get("rendered_sentence_ids", []) or legacy_render.get("sentence_ids", []) or []),
                        list(vrow.get("rendered_sentence_ids", []) or v12_render.get("sentence_ids", []) or []),
                    )
                ),
                "legacy_transfers": dict(legacy_transfers),
                "v12_transfers": dict(v12_transfers),
                "legacy_stage_metrics": {s: {k: v for k, v in dict(legacy_stage_map.get(s, {})).items() if k not in {"unit_ids", "texts"}} for s in STAGE_ORDER},
                "v12_stage_metrics": {s: {k: v for k, v in dict(v12_stage_map.get(s, {})).items() if k not in {"unit_ids", "texts"}} for s in STAGE_ORDER},
            }

            # classification
            cls = _paired_case_classification(row, row)
            row["case_classification"] = cls
            per_query_pairs.append(row)

            for method_label, stage_map, transfers, failure in [
                ("legacy", legacy_stage_map, legacy_transfers, legacy_failure),
                ("v12", v12_stage_map, v12_transfers, v12_failure),
            ]:
                failure_stage_rows.append(
                    {
                        "dataset": dataset,
                        "method": method_label,
                        "sample_id": sid,
                        "failure_stage": failure,
                    }
                )
                for stage in STAGE_ORDER:
                    item = dict(stage_map.get(stage, {}) or {})
                    stage_rows_by_dataset.append(
                        {
                            "dataset": dataset,
                            "method": method_label,
                            "stage": stage,
                            "stage_available": bool(item.get("stage_available", False)),
                            "candidate_count": item.get("candidate_count"),
                            "entity_node_count": item.get("entity_node_count"),
                            "chunk_node_count": item.get("chunk_node_count"),
                            "sentence_node_count": item.get("sentence_node_count"),
                            "atom_count": item.get("atom_count"),
                            "estimated_token_count": item.get("estimated_token_count"),
                            "SF_recall": item.get("SF_recall"),
                            "SF_precision": item.get("SF_precision"),
                            "SF_f1": item.get("SF_f1"),
                            "answer_string_hit": item.get("answer_string_hit"),
                            "answer_bearing_density": item.get("answer_bearing_density"),
                        }
                    )

    # Aggregate stagewise by dataset/method/stage
    agg_stage: List[Dict[str, Any]] = []
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in stage_rows_by_dataset:
        buckets[(str(r["dataset"]), str(r["method"]), str(r["stage"]))].append(r)
    for (dataset, method, stage), rows in sorted(buckets.items()):
        avail_rows = [x for x in rows if bool(x.get("stage_available", False))]
        def _m(key: str) -> Optional[float]:
            vals = [float(x.get(key)) for x in avail_rows if x.get(key) is not None]
            if not vals:
                return None
            return float(sum(vals) / float(len(vals)))
        agg_stage.append(
            {
                "dataset": dataset,
                "method": method,
                "stage": stage,
                "queries": int(len(rows)),
                "stage_available_queries": int(len(avail_rows)),
                "candidate_count": _m("candidate_count"),
                "estimated_token_count": _m("estimated_token_count"),
                "SF_recall": _m("SF_recall"),
                "SF_precision": _m("SF_precision"),
                "SF_f1": _m("SF_f1"),
                "answer_string_hit": _m("answer_string_hit"),
                "answer_bearing_density": _m("answer_bearing_density"),
            }
        )

    # Main QA by dataset/method
    qa_rows: List[Dict[str, Any]] = []
    by_dataset_method: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for q in per_query_pairs:
        by_dataset_method[(str(q["dataset"]), "legacy")].append(q)
        by_dataset_method[(str(q["dataset"]), "v12")].append(q)
    for (dataset, method), rows in sorted(by_dataset_method.items()):
        run_ms = dict(run_level_metrics.get((dataset, method), {}) or {})
        if method == "legacy":
            f1s = [float(r["legacy_F1"]) for r in rows]
            ems = [float(r["legacy_EM"]) for r in rows]
            toks = [float(r["legacy_rendered_tokens"]) for r in rows]
            sf_r = [float(r["legacy_rendered_SF_recall"]) for r in rows]
            sf_p = [float((r.get("legacy_stage_metrics", {}).get("S6_rendered_context", {}) or {}).get("SF_precision", 0.0) or 0.0) for r in rows]
            ah = [float(r["legacy_answer_string_hit"]) for r in rows]
            abd = [float(r["legacy_answer_bearing_density"]) for r in rows]
        else:
            f1s = [float(r["v12_F1"]) for r in rows]
            ems = [float(r["v12_EM"]) for r in rows]
            toks = [float(r["v12_rendered_tokens"]) for r in rows]
            sf_r = [float(r["v12_rendered_SF_recall"]) for r in rows]
            sf_p = [float((r.get("v12_stage_metrics", {}).get("S6_rendered_context", {}) or {}).get("SF_precision", 0.0) or 0.0) for r in rows]
            ah = [float(r["v12_answer_string_hit"]) for r in rows]
            abd = [float(r["v12_answer_bearing_density"]) for r in rows]
        sf_f1_vals = []
        for p, rr in zip(sf_p, sf_r):
            if (p + rr) <= 0:
                sf_f1_vals.append(0.0)
            else:
                sf_f1_vals.append((2.0 * p * rr) / (p + rr))
        tok_mean = _mean(toks)
        f1_mean = _mean(f1s)
        qa_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "EM": _mean(ems),
                "F1": f1_mean,
                "avg_context_tokens": tok_mean,
                "F1_per_1k_context_tokens": (f1_mean * 1000.0 / tok_mean) if tok_mean > 0 else 0.0,
                "SF_precision": _mean(sf_p),
                "SF_recall": _mean(sf_r),
                "SF_f1": _mean(sf_f1_vals),
                "answer_string_hit": _mean(ah),
                "answer_bearing_density": _mean(abd),
                "retrieval_ms": _safe_float(run_ms.get("retrieval_ms", 0.0), 0.0),
                "generation_ms": _safe_float(run_ms.get("generation_ms", 0.0), 0.0),
                "total_ms": _safe_float(run_ms.get("total_ms", 0.0), 0.0),
                "num_queries": len(rows),
            }
        )

    # failure distribution
    failure_dist: List[Dict[str, Any]] = []
    f_buckets: Dict[Tuple[str, str, str], int] = Counter()
    f_total: Dict[Tuple[str, str], int] = Counter()
    for r in failure_stage_rows:
        k = (str(r["dataset"]), str(r["method"]))
        f_total[k] += 1
        f_buckets[(str(r["dataset"]), str(r["method"]), str(r["failure_stage"]))] += 1
    for (dataset, method, stage), cnt in sorted(f_buckets.items()):
        total = max(1, f_total[(dataset, method)])
        failure_dist.append(
            {
                "dataset": dataset,
                "method": method,
                "failure_stage": stage,
                "count": int(cnt),
                "share": float(safe_div(cnt, total)),
            }
        )

    # examples
    legacy_better = [r for r in per_query_pairs if r.get("case_classification") == "legacy_better_answer_sufficient"]
    v12_better = [r for r in per_query_pairs if r.get("case_classification") == "v12_better_compact"]
    legacy_high = [r for r in per_query_pairs if r.get("case_classification") == "legacy_high_F1_low_SF"]
    v12_high = [r for r in per_query_pairs if r.get("case_classification") == "v12_high_SF_low_F1"]
    legacy_better = sorted(legacy_better, key=lambda r: float(r["legacy_F1"] - r["v12_F1"]), reverse=True)[:top_k]
    v12_better = sorted(v12_better, key=lambda r: float(r["v12_F1"] - r["legacy_F1"]), reverse=True)[:top_k]
    legacy_high = sorted(legacy_high, key=lambda r: float(r["legacy_F1"] - r["v12_F1"]), reverse=True)[:top_k]
    v12_high = sorted(v12_high, key=lambda r: float(r["legacy_F1"] - r["v12_F1"]), reverse=True)[:top_k]

    # recommendation logic
    # Q1: Phase1 candidate ceiling gap
    q1_signal = 0.0
    q2_signal = 0.0
    for dataset in datasets:
        legacy_s1 = next((r for r in agg_stage if r["dataset"] == dataset and r["method"] == "legacy" and r["stage"] == "S1_proposal_candidates"), None)
        v12_s1 = next((r for r in agg_stage if r["dataset"] == dataset and r["method"] == "v12" and r["stage"] == "S1_proposal_candidates"), None)
        legacy_s5 = next((r for r in agg_stage if r["dataset"] == dataset and r["method"] == "legacy" and r["stage"] == "S5_final_selected_or_renderable"), None)
        v12_s5 = next((r for r in agg_stage if r["dataset"] == dataset and r["method"] == "v12" and r["stage"] == "S5_final_selected_or_renderable"), None)
        legacy_s6 = next((r for r in agg_stage if r["dataset"] == dataset and r["method"] == "legacy" and r["stage"] == "S6_rendered_context"), None)
        v12_s6 = next((r for r in agg_stage if r["dataset"] == dataset and r["method"] == "v12" and r["stage"] == "S6_rendered_context"), None)
        if legacy_s1 and v12_s1:
            l = legacy_s1.get("SF_recall")
            r = v12_s1.get("SF_recall")
            if l is not None and r is not None:
                q1_signal += float(l) - float(r)
        if legacy_s5 and v12_s5 and legacy_s6 and v12_s6:
            l5 = legacy_s5.get("SF_recall")
            v5 = v12_s5.get("SF_recall")
            l6 = legacy_s6.get("SF_recall")
            v6 = v12_s6.get("SF_recall")
            if None not in (l5, v5, l6, v6):
                q2_signal += float((v5 - v6) - (l5 - l6))

    if q1_signal > 0.08:
        recommendation = "phase1_candidate_or_run_seed_recovery"
    elif q2_signal > 0.05:
        recommendation = "final_selection_or_rendering_redesign"
    else:
        # answer sufficiency lens
        avg_legacy_f1 = _mean([r["F1"] for r in qa_rows if r["method"] == "legacy"])
        avg_v12_f1 = _mean([r["F1"] for r in qa_rows if r["method"] == "v12"])
        avg_legacy_sf = _mean([r["SF_recall"] for r in qa_rows if r["method"] == "legacy"])
        avg_v12_sf = _mean([r["SF_recall"] for r in qa_rows if r["method"] == "v12"])
        avg_legacy_ah = _mean([r["answer_string_hit"] for r in qa_rows if r["method"] == "legacy"])
        avg_v12_ah = _mean([r["answer_string_hit"] for r in qa_rows if r["method"] == "v12"])
        if (avg_legacy_f1 > avg_v12_f1) and (avg_legacy_sf <= avg_v12_sf) and (avg_legacy_ah > avg_v12_ah):
            recommendation = "answer_sufficiency_over_sf_recall"
        else:
            recommendation = "mixed_signal_additional_stage_audit_needed"

    # validation payload
    completed_run_names = len(run_bundles)
    scheduled_runs = len(run_items)
    incomplete_attempts = max(0, scheduled_runs - completed_run_names)
    validation = {
        "scheduled_runs": int(scheduled_runs),
        "completed_run_names": int(completed_run_names),
        "incomplete_attempts": int(incomplete_attempts),
        "missing_outputs": int(missing_outputs),
        "parse_errors": int(parse_errors),
        "sample_id_mismatch": int(sample_id_mismatch),
        "normalization_errors": int(normalization_errors),
        "legacy_missing_stage_fields": bool(
            any(
                r["method"] == "legacy"
                and r["stage"] in {"S1_proposal_candidates", "S2_phase1_shortlist"}
                and (not bool(r.get("stage_available", False)))
                for r in stage_rows_by_dataset
            )
        ),
    }

    # Output files
    norm_root = out_root / "normalized"
    ex_root = out_root / "examples"
    _write_jsonl(norm_root / "legacy_vs_v12_stagewise_by_query.jsonl", per_query_pairs)
    _write_csv(
        norm_root / "legacy_vs_v12_stagewise_by_dataset.csv",
        agg_stage,
        [
            "dataset",
            "method",
            "stage",
            "queries",
            "stage_available_queries",
            "candidate_count",
            "estimated_token_count",
            "SF_recall",
            "SF_precision",
            "SF_f1",
            "answer_string_hit",
            "answer_bearing_density",
        ],
    )
    _write_csv(
        norm_root / "failure_stage_distribution.csv",
        failure_dist,
        ["dataset", "method", "failure_stage", "count", "share"],
    )

    _write_examples_md(ex_root / "legacy_better_examples_top20.md", "Legacy Better Answer-Sufficient Examples", legacy_better)
    _write_examples_md(ex_root / "v12_better_examples_top20.md", "V12 Better Compact Examples", v12_better)
    _write_examples_md(
        ex_root / "legacy_high_F1_low_SF_examples_top20.md",
        "Legacy High-F1 Low-SF Examples",
        legacy_high,
    )
    _write_examples_md(
        ex_root / "v12_high_SF_low_F1_examples_top20.md",
        "V12 High-SF Low-F1 Examples",
        v12_high,
    )

    summary_payload = {
        "phase": "phase6x_legacy_vs_v12_cross_version_audit",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root.resolve()),
        "datasets": list(datasets),
        "validation": dict(validation),
        "qa_metrics": list(qa_rows),
        "stagewise_metrics": list(agg_stage),
        "failure_stage_distribution": list(failure_dist),
        "case_distribution": dict(Counter([str(r.get("case_classification", "")) for r in per_query_pairs])),
        "recommendation": str(recommendation),
    }
    _write_json(out_root / "phase6x_legacy_vs_v12_cross_version_audit_summary.json", summary_payload)
    _write_json(out_root / "phase6x_artifact_validation.json", validation)

    # Markdown summary
    md_lines: List[str] = [
        "# PHASE6X Legacy-vs-v12 Cross-Version Stagewise Audit",
        "",
        "## 1. Goal",
        "Compare true legacy (commit 56779c0 lineage) vs latest unified v12 with paired query IDs.",
        "",
        "## 2. Code Versions and Configs",
        f"- legacy_commit: {_safe_text(manifest.get('legacy_commit', ''))}",
        f"- latest_commit: {_safe_text(manifest.get('latest_commit', ''))}",
        f"- legacy_hotpotqa_config_path: {_safe_text(manifest.get('legacy_hotpotqa_config_path', ''))}",
        f"- legacy_2wiki_config_path: {_safe_text(manifest.get('legacy_2wiki_config_path', ''))}",
        f"- latest_hotpotqa_config_path: {_safe_text(manifest.get('latest_hotpotqa_config_path', ''))}",
        f"- latest_2wiki_config_path: {_safe_text(manifest.get('latest_2wiki_config_path', ''))}",
        "",
        "## 3. Artifact Validation",
        f"- scheduled_runs={validation['scheduled_runs']}, completed_run_names={validation['completed_run_names']}, incomplete_attempts={validation['incomplete_attempts']}",
        f"- missing_outputs={validation['missing_outputs']}, parse_errors={validation['parse_errors']}, sample_id_mismatch={validation['sample_id_mismatch']}, normalization_errors={validation['normalization_errors']}",
        f"- legacy_missing_stage_fields={validation['legacy_missing_stage_fields']}",
        "",
        "## 4. Main QA Comparison",
        "| dataset | method | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | SF_precision | SF_recall | SF_f1 | answer_string_hit | answer_bearing_density | retrieval_ms | generation_ms | total_ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in qa_rows:
        md_lines.append(
            f"| {r['dataset']} | {r['method']} | {_fmt(r['EM'])} | {_fmt(r['F1'])} | {_fmt(r['avg_context_tokens'],2)} | {_fmt(r['F1_per_1k_context_tokens'])} | {_fmt(r['SF_precision'])} | {_fmt(r['SF_recall'])} | {_fmt(r['SF_f1'])} | {_fmt(r['answer_string_hit'])} | {_fmt(r['answer_bearing_density'])} | {_fmt(r['retrieval_ms'],2)} | {_fmt(r['generation_ms'],2)} | {_fmt(r['total_ms'],2)} |"
        )
    md_lines += [
        "",
        "## 5. Stagewise Evidence Transfer",
        "| dataset | method | stage | available_queries | candidate_count | estimated_token_count | SF_recall | SF_precision | answer_string_hit | answer_bearing_density |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_stage:
        md_lines.append(
            f"| {r['dataset']} | {r['method']} | {r['stage']} | {int(r['stage_available_queries'])} | {_fmt(r['candidate_count'],2)} | {_fmt(r['estimated_token_count'],2)} | {_fmt(r['SF_recall'])} | {_fmt(r['SF_precision'])} | {_fmt(r['answer_string_hit'])} | {_fmt(r['answer_bearing_density'])} |"
        )
    md_lines += [
        "",
        "## 6. Candidate Ceiling Comparison",
        "- See S1/S2 rows in stagewise table above.",
        "",
        "## 7. Final Selection/Rendering Comparison",
        "- Compare S5 vs S6 deltas and failure-stage distribution.",
        "",
        "## 8. Legacy High-F1 Low-SF Cases",
        f"- `{ex_root / 'legacy_high_F1_low_SF_examples_top20.md'}`",
        "",
        "## 9. V12 High-SF Low-F1 Cases",
        f"- `{ex_root / 'v12_high_SF_low_F1_examples_top20.md'}`",
        "",
        "## 10. Failure Stage Distribution",
        "| dataset | method | failure_stage | count | share |",
        "|---|---|---|---:|---:|",
    ]
    for r in failure_dist:
        md_lines.append(
            f"| {r['dataset']} | {r['method']} | {r['failure_stage']} | {int(r['count'])} | {_fmt(r['share'])} |"
        )
    md_lines += [
        "",
        "## 11. Recommendation",
        f"- {recommendation}",
    ]
    (out_root / "PHASE6X_LEGACY_VS_V12_CROSS_VERSION_AUDIT_SUMMARY.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PHASE6X legacy-vs-v12 cross-version audit outputs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    build_audit(out_root=out_root, datasets=datasets, top_k=max(1, int(args.top_k)))
    print(out_root / "PHASE6X_LEGACY_VS_V12_CROSS_VERSION_AUDIT_SUMMARY.md")
    print(out_root / "phase6x_legacy_vs_v12_cross_version_audit_summary.json")
    print(out_root / "phase6x_artifact_validation.json")
    print(out_root / "normalized" / "legacy_vs_v12_stagewise_by_query.jsonl")
    print(out_root / "normalized" / "legacy_vs_v12_stagewise_by_dataset.csv")
    print(out_root / "normalized" / "failure_stage_distribution.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
