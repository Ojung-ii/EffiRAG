#!/usr/bin/env python3
"""Summarize PHASE6W unit-transition and atomization audit runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts
from effirag.evidence_flow_audit import load_samples_for_dataset, stage_metrics, token_count_for_texts


STAGES: List[Tuple[str, str]] = [
    ("S0_anchors", "anchors"),
    ("S1_proposal_candidates", "proposal_candidates"),
    ("S2_phase1_run_seed_shortlist", "phase1_run_seed_shortlist"),
    ("S3_local_corridor_candidates", "local_corridor_candidates"),
    ("S4_sentence_candidates", "sentence_candidates"),
    ("S5_evidence_atoms", "evidence_atoms"),
    ("S6_ABR_selected", "ABR_selected"),
    ("S7_rendered_context", "rendered_context"),
]

STAGE_INDEX = {k: i for i, (k, _label) in enumerate(STAGES)}
STAGE_KEY_FROM_LABEL = {label: key for key, label in STAGES}

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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(x) for x in values) / float(len(values)))


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


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


def _extract_selected_ids(row: Mapping[str, Any]) -> List[str]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    ids = list(row.get("retrieval_selected_sentence_ids", []) or retrieval.get("selected_sentence_ids", []) or [])
    return [_normalize_sentence_support_id(x) for x in _ordered_unique(ids)]


def _extract_rendered_ids(row: Mapping[str, Any]) -> List[str]:
    rendered = dict((row.get("rendered", {}) or {}))
    ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
    return [_normalize_sentence_support_id(x) for x in _ordered_unique(ids)]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(_ordered_unique(a))
    sb = set(_ordered_unique(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa.intersection(sb)) / float(len(sa.union(sb))))


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
    f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
    recall5 = _safe_float(summary.get("recall_at_5", summary.get("Recall@5", 0.0)), 0.0)
    sf_p = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
    sf_r = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
    sf_f1 = (2.0 * sf_p * sf_r / (sf_p + sf_r)) if (sf_p + sf_r) > 0 else 0.0
    ctx = _safe_float(summary.get("avg_context_tokens", summary.get("prompt_tokens_avg", 0.0)), 0.0)
    f1k = (f1 * 1000.0 / ctx) if ctx > 0 else 0.0
    return {
        "Recall@5": recall5,
        "EM": em,
        "F1": f1,
        "avg_context_tokens": ctx,
        "F1_per_1k_context_tokens": f1k,
        "supporting_fact_precision": sf_p,
        "supporting_fact_recall": sf_r,
        "supporting_fact_f1": sf_f1,
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
        "answer_string_hit": _safe_float(summary.get("answer_surface_present", 0.0), 0.0),
        "answer_string_rank_avg": _safe_float(summary.get("answer_surface_position_avg", -1.0), -1.0),
        "answer_bearing_density": _safe_float(summary.get("answer_surface_token_density", 0.0), 0.0),
        "answer_context_early_hit": _safe_float(summary.get("answer_surface_early_hit", 0.0), 0.0),
    }


def _query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    m = dict(row.get("metrics", {}) or {})
    e = dict(row.get("efficiency", {}) or {})
    gdiag = dict(row.get("generation_diagnostics", {}) or {})
    return {
        "f1": _safe_float(m.get("f1", 0.0), 0.0),
        "em": _safe_float(m.get("em", 0.0), 0.0),
        "sf_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
        "sf_precision": _safe_float(m.get("supporting_fact_precision", 0.0), 0.0),
        "retrieval_ms": _safe_float(e.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(e.get("generation_latency_ms", 0.0), 0.0),
        "total_ms": _safe_float(e.get("total_latency_ms", 0.0), 0.0),
        "prompt_tokens": _safe_float(gdiag.get("prompt_tokens", 0.0), 0.0),
        "answer_string_hit": _safe_float(m.get("answer_surface_present", 0.0), 0.0),
        "answer_string_rank": _safe_float(m.get("answer_surface_position", -1.0), -1.0),
        "answer_bearing_density": _safe_float(m.get("answer_surface_token_density", 0.0), 0.0),
        "answer_context_early_hit": _safe_float(m.get("answer_surface_early_hit", 0.0), 0.0),
    }


def _stage_latency_ms(stage_label: str, lat: Mapping[str, Any]) -> float:
    if stage_label == "anchors":
        return _safe_float(lat.get("query_preprocess_ms", 0.0), 0.0)
    if stage_label == "proposal_candidates":
        return _safe_float(
            lat.get("proposal_union_total_ms", lat.get("proposal_union_ms", lat.get("proposal_time_ms", 0.0))),
            0.0,
        )
    if stage_label == "phase1_run_seed_shortlist":
        return (
            _safe_float(lat.get("phase1_run_generation_ms", 0.0), 0.0)
            + _safe_float(lat.get("phase1_run_scoring_ms", 0.0), 0.0)
            + _safe_float(lat.get("phase1_run_shortlist_ms", 0.0), 0.0)
        )
    if stage_label == "local_corridor_candidates":
        return (
            _safe_float(lat.get("local_graph_construction_ms", 0.0), 0.0)
            + _safe_float(lat.get("ppr_time_ms", 0.0), 0.0)
            + _safe_float(lat.get("corridor_candidate_generation_ms", 0.0), 0.0)
        )
    if stage_label == "sentence_candidates":
        return (
            _safe_float(lat.get("final_candidate_packaging_ms", 0.0), 0.0)
            + _safe_float(lat.get("sentence_rerank_ms", 0.0), 0.0)
        )
    if stage_label == "evidence_atoms":
        return _safe_float(lat.get("atomization_ms", 0.0), 0.0)
    if stage_label == "ABR_selected":
        return _safe_float(lat.get("unified_acr_rcedr_ms", 0.0), 0.0)
    if stage_label == "rendered_context":
        return _safe_float(lat.get("render_ms", 0.0), 0.0)
    return 0.0


def _corridor_ids_texts(corridors: Any) -> Tuple[List[str], List[str]]:
    unit_ids: List[str] = []
    unit_texts: List[str] = []
    for item in list(corridors or []):
        if not isinstance(item, Mapping):
            continue
        for key in (
            "main_path_unit_ids",
            "support_unit_ids",
            "connector_adjacent_unit_ids",
            "main_path_sentence_ids",
            "support_sentence_ids",
            "connector_adjacent_sentence_ids",
        ):
            for sid in list(item.get(key, []) or []):
                s = _safe_text(sid)
                if s:
                    unit_ids.append(s)
                    unit_texts.append("")
    ids = _ordered_unique(unit_ids)
    text_by_id: Dict[str, str] = {}
    for sid, txt in zip(unit_ids, unit_texts):
        if sid not in text_by_id:
            text_by_id[sid] = _safe_text(txt)
    texts = [text_by_id.get(sid, "") for sid in ids]
    return ids, texts


def _gold_ids_from_sample(sample: Any) -> List[str]:
    ids: List[str] = []
    for title, sent_idx in list(getattr(sample, "supporting_facts", []) or []):
        t = _safe_text(title)
        if not t:
            continue
        try:
            idx = int(sent_idx)
        except Exception:
            idx = 0
        ids.append(f"{t}::{idx}")
    return sorted(set(ids))


def _answer_aliases(sample: Any) -> List[str]:
    aliases: List[str] = []
    answer = _safe_text(getattr(sample, "answer", ""))
    if answer:
        aliases.append(answer)
    for key in ("answer_aliases", "aliases", "normalized_aliases"):
        for value in list(getattr(sample, key, []) or []):
            v = _safe_text(value)
            if v:
                aliases.append(v)
    # De-dup on normalized surface.
    norm_seen = set()
    out: List[str] = []
    for alias in aliases:
        norm = _normalize_for_match(alias)
        if not norm or norm in norm_seen:
            continue
        norm_seen.add(norm)
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


def _stage_payloads(row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    diag = dict((retrieval.get("diagnostics", {}) or {}))
    lat = dict((row.get("latency_breakdown_ms", {}) or {}))
    rendered = dict((row.get("rendered", {}) or {}))
    funnel = dict((diag.get("stagewise_loss_funnel", {}) or {}))
    unified_diag = dict((diag.get("unified_acr_rcedr_diag", {}) or {}))

    proposal_ids = list(diag.get("phase6u_proposal_candidate_unit_ids", []) or [])
    proposal_texts = list(diag.get("phase6u_proposal_candidate_texts", []) or [])
    phase1_ids = list(diag.get("phase6u_phase1_shortlist_unit_ids", []) or [])
    phase1_texts = list(diag.get("phase6u_phase1_shortlist_texts", []) or [])
    corridor_before_ids = list(diag.get("phase6u_corridor_before_trim_unit_ids", []) or [])
    corridor_before_texts = list(diag.get("phase6u_corridor_before_trim_texts", []) or [])
    if not corridor_before_ids:
        corridor_before_ids, corridor_before_texts = _corridor_ids_texts(retrieval.get("corridors", []))
    sent_after_ids = list(diag.get("phase6u_sentence_candidates_after_text_rerank_ids", []) or [])
    sent_after_texts = list(diag.get("phase6u_sentence_candidates_after_text_rerank_texts", []) or [])
    atoms_ids = list(diag.get("phase6u_evidence_atoms_before_abr_ids", []) or [])
    atoms_texts = list(diag.get("phase6u_evidence_atoms_before_abr_texts", []) or [])
    abr_ids = list(diag.get("phase6u_abr_selected_evidence_ids", []) or [])
    abr_texts = list(diag.get("phase6u_abr_selected_evidence_texts", []) or [])
    rendered_ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
    rendered_texts = list(rendered.get("sentences", []) or [])

    return {
        "S0_anchors": {
            "label": "anchors",
            "ids": [str(x) for x in list(retrieval.get("anchors", []) or []) if _safe_text(x)],
            "texts": [],
            "candidate_count": int(len(list(retrieval.get("anchors", []) or []))),
            "override_sf_recall": _safe_float(funnel.get("anchor_hit_rate", 0.0), 0.0),
            "latency_ms": _stage_latency_ms("anchors", lat),
            "entity_node_count_hint": int(len(list(retrieval.get("anchors", []) or []))),
        },
        "S1_proposal_candidates": {
            "label": "proposal_candidates",
            "ids": list(proposal_ids),
            "texts": list(proposal_texts),
            "candidate_count": _safe_int(diag.get("num_candidates_union", len(proposal_ids)), len(proposal_ids)),
            "latency_ms": _stage_latency_ms("proposal_candidates", lat),
            "entity_node_count_hint": _safe_int(diag.get("proposal_entity_count", 0), 0),
            "chunk_node_count_hint": _safe_int(diag.get("proposal_chunk_count", 0), 0),
        },
        "S2_phase1_run_seed_shortlist": {
            "label": "phase1_run_seed_shortlist",
            "ids": list(phase1_ids),
            "texts": list(phase1_texts),
            "candidate_count": _safe_int(diag.get("num_seed_candidates", len(phase1_ids)), len(phase1_ids)),
            "latency_ms": _stage_latency_ms("phase1_run_seed_shortlist", lat),
        },
        "S3_local_corridor_candidates": {
            "label": "local_corridor_candidates",
            "ids": list(corridor_before_ids),
            "texts": list(corridor_before_texts),
            "candidate_count": _safe_int(diag.get("corridor_count_before_trim", len(corridor_before_ids)), len(corridor_before_ids)),
            "latency_ms": _stage_latency_ms("local_corridor_candidates", lat),
        },
        "S4_sentence_candidates": {
            "label": "sentence_candidates",
            "ids": list(sent_after_ids),
            "texts": list(sent_after_texts),
            "candidate_count": int(len(list(sent_after_ids))),
            "latency_ms": _stage_latency_ms("sentence_candidates", lat),
        },
        "S5_evidence_atoms": {
            "label": "evidence_atoms",
            "ids": list(atoms_ids),
            "texts": list(atoms_texts),
            "candidate_count": _safe_int(unified_diag.get("num_atoms", len(atoms_ids)), len(atoms_ids)),
            "latency_ms": _stage_latency_ms("evidence_atoms", lat),
            "atom_count_hint": _safe_int(unified_diag.get("num_atoms", len(atoms_ids)), len(atoms_ids)),
        },
        "S6_ABR_selected": {
            "label": "ABR_selected",
            "ids": list(abr_ids if abr_ids else list(diag.get("selected_text_unit_ids", []) or [])),
            "texts": list(abr_texts if abr_texts else list(diag.get("selected_texts", []) or [])),
            "candidate_count": _safe_int(unified_diag.get("num_selected_atoms", len(abr_ids)), len(abr_ids)),
            "latency_ms": _stage_latency_ms("ABR_selected", lat),
            "atom_count_hint": _safe_int(unified_diag.get("num_selected_atoms", len(abr_ids)), len(abr_ids)),
        },
        "S7_rendered_context": {
            "label": "rendered_context",
            "ids": list(rendered_ids),
            "texts": list(rendered_texts),
            "candidate_count": int(len(list(rendered_ids))),
            "latency_ms": _stage_latency_ms("rendered_context", lat),
            "rendered_sentence_count_hint": int(len(list(rendered_ids))),
        },
    }


def _unit_counts_for_stage(stage_key: str, payload: Mapping[str, Any]) -> Dict[str, int]:
    ids = [_safe_text(x) for x in list(payload.get("ids", []) or []) if _safe_text(x)]
    entity_count = 0
    chunk_count = 0
    sentence_count = 0
    for sid in ids:
        if sid.startswith("chunk::"):
            chunk_count += 1
        elif sid.startswith("sentence::"):
            sentence_count += 1
        elif "::" in sid:
            sentence_count += 1
        else:
            entity_count += 1

    if stage_key == "S0_anchors":
        entity_count = max(entity_count, _safe_int(payload.get("entity_node_count_hint", entity_count), entity_count))
    if stage_key == "S1_proposal_candidates":
        entity_hint = _safe_int(payload.get("entity_node_count_hint", 0), 0)
        chunk_hint = _safe_int(payload.get("chunk_node_count_hint", 0), 0)
        entity_count = max(entity_count, entity_hint)
        chunk_count = max(chunk_count, chunk_hint)

    atom_count = _safe_int(payload.get("atom_count_hint", 0), 0)
    rendered_sentence_count = _safe_int(payload.get("rendered_sentence_count_hint", 0), 0)
    if stage_key == "S5_evidence_atoms":
        atom_count = max(atom_count, _safe_int(payload.get("candidate_count", 0), 0))
    if stage_key == "S6_ABR_selected":
        atom_count = max(atom_count, _safe_int(payload.get("candidate_count", 0), 0))
    if stage_key == "S7_rendered_context":
        rendered_sentence_count = max(rendered_sentence_count, len(ids))

    return {
        "entity_node_count": int(entity_count),
        "chunk_node_count": int(chunk_count),
        "sentence_node_count": int(sentence_count),
        "atom_count": int(atom_count),
        "rendered_sentence_count": int(rendered_sentence_count),
    }


def _profile_role(profile: str, baseline_profile: str) -> str:
    if profile == baseline_profile:
        return "baseline"
    if profile.endswith("_no_atomization"):
        return "no_atomization"
    if profile.endswith("_atom_span3"):
        return "atom_span3"
    if profile.endswith("_render_package"):
        return "render_package"
    return "variant"


def _run_name_for_profile(dataset: str, profile: str) -> str:
    p = _safe_text(profile)
    if p.endswith("_no_atomization"):
        return f"{dataset}_no_atomization"
    if p.endswith("_atom_span3"):
        return f"{dataset}_atom_span3"
    if p.endswith("_render_package"):
        return f"{dataset}_render_package"
    return f"{dataset}_baseline"


@dataclass
class RunData:
    run_name: str
    dataset: str
    profile: str
    role: str
    summary: Dict[str, Any]
    query_rows: List[Dict[str, Any]]


def _load_runs(out_root: Path) -> Tuple[Dict[str, Any], List[RunData]]:
    report = collect_phase6s_artifacts(out_root)
    scheduled = {_safe_text(r.get("run_name")): dict(r) for r in list(report.get("scheduled_runs", []) or [])}
    runs: List[RunData] = []
    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        meta = scheduled.get(run_name, {})
        dataset = _safe_text(meta.get("dataset", item.get("dataset", "")))
        profile = _safe_text(meta.get("profile", item.get("profile", "")))
        role = _safe_text(meta.get("role", ""))
        summary = dict(item.get("summary", {}) or {})
        qpath = Path(_safe_text(item.get("query_path"))).resolve()
        qrows = _read_jsonl(qpath)
        runs.append(RunData(run_name, dataset, profile, role, summary, qrows))
    return report, runs


def _pair_baseline_variant(
    dataset: str,
    baseline: RunData,
    variant: RunData,
    variant_profile: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bmap = {_safe_text(r.get("sample_id")): r for r in baseline.query_rows}
    vmap = {_safe_text(r.get("sample_id")): r for r in variant.query_rows}
    ordered_ids = [_safe_text(r.get("sample_id")) for r in baseline.query_rows if _safe_text(r.get("sample_id"))]

    paired: List[Dict[str, Any]] = []
    improved: List[Dict[str, Any]] = []
    regressed: List[Dict[str, Any]] = []

    for sid in ordered_ids:
        b = bmap.get(sid)
        v = vmap.get(sid)
        if b is None or v is None:
            continue

        bm = _query_metrics(b)
        vm = _query_metrics(v)
        b_sel = _extract_selected_ids(b)
        v_sel = _extract_selected_ids(v)
        b_rnd = _extract_rendered_ids(b)
        v_rnd = _extract_rendered_ids(v)

        b_ctx = _safe_text(dict(b.get("rendered", {}) or {}).get("text", ""))
        v_ctx = _safe_text(dict(v.get("rendered", {}) or {}).get("text", ""))
        question = _safe_text(b.get("question", ""))

        row = {
            "dataset": dataset,
            "sample_id": sid,
            "question": question,
            "query_type": _infer_query_type(question),
            "variant_profile": variant_profile,
            "baseline_prediction": _safe_text(b.get("prediction", "")),
            "variant_prediction": _safe_text(v.get("prediction", "")),
            "baseline_f1": bm["f1"],
            "variant_f1": vm["f1"],
            "delta_f1": float(vm["f1"] - bm["f1"]),
            "baseline_sf_recall": bm["sf_recall"],
            "variant_sf_recall": vm["sf_recall"],
            "delta_sf_recall": float(vm["sf_recall"] - bm["sf_recall"]),
            "baseline_sf_precision": bm["sf_precision"],
            "variant_sf_precision": vm["sf_precision"],
            "baseline_prompt_tokens": bm["prompt_tokens"],
            "variant_prompt_tokens": vm["prompt_tokens"],
            "baseline_retrieval_ms": bm["retrieval_ms"],
            "variant_retrieval_ms": vm["retrieval_ms"],
            "baseline_total_ms": bm["total_ms"],
            "variant_total_ms": vm["total_ms"],
            "baseline_answer_string_hit": bm["answer_string_hit"],
            "variant_answer_string_hit": vm["answer_string_hit"],
            "delta_answer_string_hit": float(vm["answer_string_hit"] - bm["answer_string_hit"]),
            "baseline_answer_bearing_density": bm["answer_bearing_density"],
            "variant_answer_bearing_density": vm["answer_bearing_density"],
            "delta_answer_bearing_density": float(vm["answer_bearing_density"] - bm["answer_bearing_density"]),
            "selected_sentence_jaccard": _jaccard(b_sel, v_sel),
            "rendered_sentence_jaccard": _jaccard(b_rnd, v_rnd),
            "prompt_hash_changed": bool(_hash_text(question + "\n" + b_ctx) != _hash_text(question + "\n" + v_ctx)),
            "context_hash_changed": bool(_hash_text(b_ctx) != _hash_text(v_ctx)),
            "baseline_selected_sentence_ids": list(b_sel),
            "variant_selected_sentence_ids": list(v_sel),
            "baseline_rendered_sentence_ids": list(b_rnd),
            "variant_rendered_sentence_ids": list(v_rnd),
        }
        paired.append(row)

        if row["delta_f1"] >= 0.05 or row["delta_answer_string_hit"] > 0.0:
            improved.append(row)
        if row["delta_f1"] <= -0.05:
            regressed.append(row)

    improved = sorted(
        improved,
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_sf_recall", 0.0), 0.0)),
        reverse=True,
    )[: max(1, top_k)]
    regressed = sorted(
        regressed,
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_sf_recall", 0.0), 0.0)),
    )[: max(1, top_k)]
    return paired, improved, regressed


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_examples_md(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    lines: List[str] = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for i, row in enumerate(rows, start=1):
        lines.append(f"## {i}. {_safe_text(row.get('dataset'))} / {_safe_text(row.get('sample_id'))}")
        lines.append(f"- variant_profile: {_safe_text(row.get('variant_profile'))}")
        lines.append(f"- query_type: {_safe_text(row.get('query_type'))}")
        lines.append(f"- question: {_safe_text(row.get('question'))}")
        lines.append(
            f"- delta_f1: {_fmt(row.get('delta_f1'))}, delta_sf_recall: {_fmt(row.get('delta_sf_recall'))}, delta_answer_string_hit: {_fmt(row.get('delta_answer_string_hit'))}"
        )
        lines.append(
            f"- selected_jaccard/rendered_jaccard: {_fmt(row.get('selected_sentence_jaccard'))} / {_fmt(row.get('rendered_sentence_jaccard'))}"
        )
        lines.append(f"- baseline_prediction: {_safe_text(row.get('baseline_prediction'))}")
        lines.append(f"- variant_prediction: {_safe_text(row.get('variant_prediction'))}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(
    *,
    out_root: Path,
    datasets: Sequence[str],
    baseline_profile: str,
    variant_profiles: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    report, runs = _load_runs(out_root)
    run_index = {(r.dataset, r.profile): r for r in runs}

    # Build unit-transition by-query rows.
    stage_rows: List[Dict[str, Any]] = []
    sample_maps: Dict[str, Dict[str, Any]] = {}

    for run in runs:
        if datasets and run.dataset not in set(datasets):
            continue
        if run.dataset not in sample_maps:
            sample_maps[run.dataset] = load_samples_for_dataset(run.dataset)
        sample_map = sample_maps[run.dataset]

        for qrow in run.query_rows:
            sample_id = _safe_text(qrow.get("sample_id"))
            sample = sample_map.get(sample_id)
            if sample is None:
                continue
            retrieval = dict((qrow.get("retrieval", {}) or {}))
            diag = dict((retrieval.get("diagnostics", {}) or {}))
            graph_mode = _safe_text(diag.get("graph_mode", "current_entity_graph"))
            gold_ids = _gold_ids_from_sample(sample)
            question = _safe_text(getattr(sample, "question", qrow.get("question", "")))
            qtype = _infer_query_type(question)
            payloads = _stage_payloads(qrow)

            per_stage: Dict[str, Dict[str, Any]] = {}
            ordered_recalls: List[float] = []
            ordered_answer_hits: List[float] = []
            for stage_key, stage_label in STAGES:
                payload = dict(payloads.get(stage_key, {}) or {})
                ids = list(payload.get("ids", []) or [])
                texts = list(payload.get("texts", []) or [])
                metrics = stage_metrics(sample, ids, texts, graph_mode)
                matched_ids = list(metrics.get("matched_gold_ids", []) or [])
                matched_set = set([_safe_text(x) for x in matched_ids if _safe_text(x)])
                missing_ids = sorted([sid for sid in gold_ids if sid not in matched_set])
                sf_p = _safe_float(metrics.get("sf_P", 0.0), 0.0)
                sf_r = _safe_float(metrics.get("sf_R", 0.0), 0.0)
                sf_f1 = _safe_float(metrics.get("sf_F1", 0.0), 0.0)
                if stage_key == "S0_anchors":
                    override_r = _safe_float(payload.get("override_sf_recall", 0.0), 0.0)
                    if override_r > 0.0:
                        sf_r = float(override_r)
                        sf_p = float(override_r)
                        sf_f1 = float(override_r)

                answer_metrics = _answer_sufficiency_metrics(sample, texts)
                unit_counts = _unit_counts_for_stage(stage_key, payload)

                row = {
                    "dataset": run.dataset,
                    "profile": run.profile,
                    "run_name": run.run_name,
                    "profile_role": _profile_role(run.profile, baseline_profile),
                    "sample_id": sample_id,
                    "question": question,
                    "query_type": qtype,
                    "stage": stage_label,
                    "stage_key": stage_key,
                    "candidate_count": int(payload.get("candidate_count", len(ids))),
                    "sentence_candidate_count": int(len(ids)),
                    "estimated_token_count": int(
                        token_count_for_texts(texts) if texts else int(metrics.get("tokens", 0) or 0)
                    ),
                    "entity_node_count": int(unit_counts["entity_node_count"]),
                    "chunk_node_count": int(unit_counts["chunk_node_count"]),
                    "sentence_node_count": int(unit_counts["sentence_node_count"]),
                    "atom_count": int(unit_counts["atom_count"]),
                    "rendered_sentence_count": int(unit_counts["rendered_sentence_count"]),
                    "supporting_fact_recall": float(sf_r),
                    "supporting_fact_precision": float(sf_p),
                    "supporting_fact_f1": float(sf_f1),
                    "matched_gold_sentence_ids": list(matched_ids),
                    "missing_gold_sentence_ids": list(missing_ids),
                    "stage_latency_ms": float(payload.get("latency_ms", 0.0)),
                    "answer_string_hit": float(answer_metrics["answer_string_hit"]),
                    "answer_string_rank": float(answer_metrics["answer_string_rank"]),
                    "answer_bearing_sentence_count": int(answer_metrics["answer_bearing_sentence_count"]),
                    "answer_bearing_density": float(answer_metrics["answer_bearing_density"]),
                    "answer_context_early_hit": float(answer_metrics["answer_context_early_hit"]),
                }
                per_stage[stage_key] = row
                ordered_recalls.append(float(sf_r))
                ordered_answer_hits.append(float(answer_metrics["answer_string_hit"]))

            # Transition deltas (query-level, attached to each stage row for easier aggregation).
            rec = ordered_recalls + [0.0] * max(0, len(STAGES) - len(ordered_recalls))
            ah = ordered_answer_hits + [0.0] * max(0, len(STAGES) - len(ordered_answer_hits))
            chunk_to_sentence = float(rec[STAGE_INDEX["S4_sentence_candidates"]] - rec[STAGE_INDEX["S3_local_corridor_candidates"]])
            sentence_to_atom = float(rec[STAGE_INDEX["S5_evidence_atoms"]] - rec[STAGE_INDEX["S4_sentence_candidates"]])
            atom_to_selected = float(rec[STAGE_INDEX["S6_ABR_selected"]] - rec[STAGE_INDEX["S5_evidence_atoms"]])
            selected_to_rendered = float(rec[STAGE_INDEX["S7_rendered_context"]] - rec[STAGE_INDEX["S6_ABR_selected"]])
            answer_hit_delta = {
                "chunk_to_sentence": float(ah[STAGE_INDEX["S4_sentence_candidates"]] - ah[STAGE_INDEX["S3_local_corridor_candidates"]]),
                "sentence_to_atom": float(ah[STAGE_INDEX["S5_evidence_atoms"]] - ah[STAGE_INDEX["S4_sentence_candidates"]]),
                "atom_to_selected": float(ah[STAGE_INDEX["S6_ABR_selected"]] - ah[STAGE_INDEX["S5_evidence_atoms"]]),
                "selected_to_rendered": float(ah[STAGE_INDEX["S7_rendered_context"]] - ah[STAGE_INDEX["S6_ABR_selected"]]),
            }

            for stage_key, _stage_label in STAGES:
                row = per_stage.get(stage_key)
                if row is None:
                    continue
                row["chunk_to_sentence_recall_delta"] = float(chunk_to_sentence)
                row["sentence_to_atom_recall_delta"] = float(sentence_to_atom)
                row["atom_to_selected_recall_delta"] = float(atom_to_selected)
                row["selected_to_rendered_recall_delta"] = float(selected_to_rendered)
                row["answer_hit_delta_by_transition"] = dict(answer_hit_delta)
                stage_rows.append(row)

    _write_jsonl(out_root / "unit_transition_by_query.jsonl", stage_rows)

    # Dataset/Profile/Stage aggregation.
    agg_buckets: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        agg_buckets[(_safe_text(row.get("dataset")), _safe_text(row.get("profile")), _safe_text(row.get("stage")))].append(row)

    unit_by_dataset_rows: List[Dict[str, Any]] = []
    for dataset, profile, stage in sorted(agg_buckets.keys()):
        group = agg_buckets[(dataset, profile, stage)]
        unit_by_dataset_rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "stage": stage,
                "candidate_count": _mean([_safe_float(r.get("candidate_count", 0.0), 0.0) for r in group]),
                "entity_node_count": _mean([_safe_float(r.get("entity_node_count", 0.0), 0.0) for r in group]),
                "chunk_node_count": _mean([_safe_float(r.get("chunk_node_count", 0.0), 0.0) for r in group]),
                "sentence_node_count": _mean([_safe_float(r.get("sentence_node_count", 0.0), 0.0) for r in group]),
                "atom_count": _mean([_safe_float(r.get("atom_count", 0.0), 0.0) for r in group]),
                "rendered_sentence_count": _mean([_safe_float(r.get("rendered_sentence_count", 0.0), 0.0) for r in group]),
                "supporting_fact_recall": _mean([_safe_float(r.get("supporting_fact_recall", 0.0), 0.0) for r in group]),
                "supporting_fact_precision": _mean([_safe_float(r.get("supporting_fact_precision", 0.0), 0.0) for r in group]),
                "supporting_fact_f1": _mean([_safe_float(r.get("supporting_fact_f1", 0.0), 0.0) for r in group]),
                "answer_string_hit": _mean([_safe_float(r.get("answer_string_hit", 0.0), 0.0) for r in group]),
                "answer_string_rank_avg": _mean([_safe_float(r.get("answer_string_rank", -1.0), -1.0) for r in group]),
                "answer_bearing_density": _mean([_safe_float(r.get("answer_bearing_density", 0.0), 0.0) for r in group]),
                "answer_context_early_hit": _mean([_safe_float(r.get("answer_context_early_hit", 0.0), 0.0) for r in group]),
                "chunk_to_sentence_recall_delta": _mean([_safe_float(r.get("chunk_to_sentence_recall_delta", 0.0), 0.0) for r in group]),
                "sentence_to_atom_recall_delta": _mean([_safe_float(r.get("sentence_to_atom_recall_delta", 0.0), 0.0) for r in group]),
                "atom_to_selected_recall_delta": _mean([_safe_float(r.get("atom_to_selected_recall_delta", 0.0), 0.0) for r in group]),
                "selected_to_rendered_recall_delta": _mean([_safe_float(r.get("selected_to_rendered_recall_delta", 0.0), 0.0) for r in group]),
                "stage_latency_ms": _mean([_safe_float(r.get("stage_latency_ms", 0.0), 0.0) for r in group]),
                "num_queries": int(len(group)),
            }
        )

    _write_csv(
        out_root / "unit_transition_by_dataset.csv",
        unit_by_dataset_rows,
        fieldnames=[
            "dataset",
            "profile",
            "stage",
            "candidate_count",
            "entity_node_count",
            "chunk_node_count",
            "sentence_node_count",
            "atom_count",
            "rendered_sentence_count",
            "supporting_fact_recall",
            "supporting_fact_precision",
            "supporting_fact_f1",
            "answer_string_hit",
            "answer_string_rank_avg",
            "answer_bearing_density",
            "answer_context_early_hit",
            "chunk_to_sentence_recall_delta",
            "sentence_to_atom_recall_delta",
            "atom_to_selected_recall_delta",
            "selected_to_rendered_recall_delta",
            "stage_latency_ms",
            "num_queries",
        ],
    )

    # Atomization diagnostics by-query.
    atom_diag_rows: List[Dict[str, Any]] = []
    for run in runs:
        if datasets and run.dataset not in set(datasets):
            continue
        for qrow in run.query_rows:
            retrieval = dict((qrow.get("retrieval", {}) or {}))
            diag = dict((retrieval.get("diagnostics", {}) or {}))
            unified_diag = dict((diag.get("unified_acr_rcedr_diag", {}) or {}))
            rendered = dict((qrow.get("rendered", {}) or {}))
            meta = dict((rendered.get("metadata", {}) or {}))
            atom_diag_rows.append(
                {
                    "dataset": run.dataset,
                    "profile": run.profile,
                    "sample_id": _safe_text(qrow.get("sample_id")),
                    "atomization_enabled": bool(unified_diag.get("atomization_enabled", True)),
                    "atom_span_max_sentences": _safe_int(unified_diag.get("atom_span_max_sentences", 2), 2),
                    "selection_unit": _safe_text(unified_diag.get("selection_unit", "atom")),
                    "num_input_candidates": _safe_int(unified_diag.get("num_input_candidates", 0), 0),
                    "num_atoms": _safe_int(unified_diag.get("num_atoms", 0), 0),
                    "num_selected_atoms": _safe_int(unified_diag.get("num_selected_atoms", 0), 0),
                    "avg_atom_sentences": _safe_float(unified_diag.get("avg_atom_sentences", 0.0), 0.0),
                    "avg_atom_tokens": _safe_float(unified_diag.get("avg_atom_tokens", 0.0), 0.0),
                    "selected_tokens": _safe_int(unified_diag.get("selected_tokens", 0), 0),
                    "objective_eval_calls": _safe_int(unified_diag.get("objective_eval_calls", 0), 0),
                    "selection_ms": _safe_float(unified_diag.get("selection_ms", 0.0), 0.0),
                    "rendered_sentence_count": len(list(rendered.get("sentence_ids", []) or [])),
                    "top_slice_reorder_applied": bool(meta.get("top_slice_reorder_applied", False)),
                    "top_slice_reorder_changed_order": bool(meta.get("top_slice_reorder_changed_order", False)),
                    "minimal_package_applied": bool(meta.get("minimal_package_applied", False)),
                    "num_extra_package_sentences": _safe_int(meta.get("num_extra_package_sentences", 0), 0),
                    "extra_sentence_token_count": _safe_int(meta.get("extra_sentence_token_count", 0), 0),
                    "budget_skip_count": _safe_int(meta.get("budget_skip_count", 0), 0),
                    "minimal_package_dominant_skip_reason": _safe_text(
                        meta.get("minimal_package_dominant_skip_reason", meta.get("dominant_skip_reason", ""))
                    ),
                }
            )

    _write_csv(
        out_root / "atomization_diagnostics.csv",
        atom_diag_rows,
        fieldnames=[
            "dataset",
            "profile",
            "sample_id",
            "atomization_enabled",
            "atom_span_max_sentences",
            "selection_unit",
            "num_input_candidates",
            "num_atoms",
            "num_selected_atoms",
            "avg_atom_sentences",
            "avg_atom_tokens",
            "selected_tokens",
            "objective_eval_calls",
            "selection_ms",
            "rendered_sentence_count",
            "top_slice_reorder_applied",
            "top_slice_reorder_changed_order",
            "minimal_package_applied",
            "num_extra_package_sentences",
            "extra_sentence_token_count",
            "budget_skip_count",
            "minimal_package_dominant_skip_reason",
        ],
    )

    # Main results and pair deltas.
    rows_main: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    no_atom_examples: List[Dict[str, Any]] = []
    atom3_examples: List[Dict[str, Any]] = []
    render_pkg_examples: List[Dict[str, Any]] = []
    regression_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        base = run_index.get((dataset, baseline_profile))
        if base is None:
            continue
        bm = _summary_metrics(base.summary)
        rows_main.append(
            {
                "dataset": dataset,
                "profile": baseline_profile,
                "role": "baseline",
                **bm,
            }
        )
        for profile in variant_profiles:
            run = run_index.get((dataset, profile))
            if run is None:
                continue
            vm = _summary_metrics(run.summary)
            rows_main.append({"dataset": dataset, "profile": profile, "role": "variant", **vm})
            base_ctx = bm["avg_context_tokens"]
            base_ret = bm["retrieval_ms"]
            delta_rows.append(
                {
                    "dataset": dataset,
                    "variant_profile": profile,
                    "delta_EM": float(vm["EM"] - bm["EM"]),
                    "delta_F1": float(vm["F1"] - bm["F1"]),
                    "delta_F1_per_1k_context_tokens": float(vm["F1_per_1k_context_tokens"] - bm["F1_per_1k_context_tokens"]),
                    "delta_SF_precision": float(vm["supporting_fact_precision"] - bm["supporting_fact_precision"]),
                    "delta_SF_recall": float(vm["supporting_fact_recall"] - bm["supporting_fact_recall"]),
                    "delta_avg_context_tokens": float(vm["avg_context_tokens"] - bm["avg_context_tokens"]),
                    "delta_retrieval_ms": float(vm["retrieval_ms"] - bm["retrieval_ms"]),
                    "delta_total_ms": float(vm["total_ms"] - bm["total_ms"]),
                    "delta_answer_string_hit": float(vm["answer_string_hit"] - bm["answer_string_hit"]),
                    "delta_answer_string_rank_avg": float(vm["answer_string_rank_avg"] - bm["answer_string_rank_avg"]),
                    "delta_answer_bearing_density": float(vm["answer_bearing_density"] - bm["answer_bearing_density"]),
                    "delta_avg_context_tokens_pct": float(((vm["avg_context_tokens"] - base_ctx) / base_ctx) * 100.0) if base_ctx > 0 else 0.0,
                    "delta_retrieval_ms_pct": float(((vm["retrieval_ms"] - base_ret) / base_ret) * 100.0) if base_ret > 0 else 0.0,
                }
            )

            pair_rows, improved, regressed = _pair_baseline_variant(
                dataset=dataset,
                baseline=base,
                variant=run,
                variant_profile=profile,
                top_k=top_k,
            )
            paired_rows.extend(pair_rows)
            regression_rows.extend(regressed)
            if profile.endswith("_no_atomization"):
                no_atom_examples.extend(improved)
            elif profile.endswith("_atom_span3"):
                atom3_examples.extend(improved)
            elif profile.endswith("_render_package"):
                render_pkg_examples.extend(improved)

    _write_jsonl(out_root / "paired_query_comparison.jsonl", paired_rows)

    no_atom_examples = sorted(
        no_atom_examples,
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_answer_string_hit", 0.0), 0.0)),
        reverse=True,
    )[: max(1, top_k)]
    atom3_examples = sorted(
        atom3_examples,
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_answer_string_hit", 0.0), 0.0)),
        reverse=True,
    )[: max(1, top_k)]
    render_pkg_examples = sorted(
        render_pkg_examples,
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_answer_string_hit", 0.0), 0.0)),
        reverse=True,
    )[: max(1, top_k)]
    regression_rows = sorted(
        regression_rows,
        key=lambda r: (_safe_float(r.get("delta_f1", 0.0), 0.0), _safe_float(r.get("delta_sf_recall", 0.0), 0.0)),
    )[: max(1, top_k)]

    _write_examples_md(out_root / "no_atomization_examples_top20.md", "No-Atomization Examples (Top 20)", no_atom_examples)
    _write_examples_md(out_root / "atom_span3_examples_top20.md", "Atom-Span3 Examples (Top 20)", atom3_examples)
    _write_examples_md(out_root / "render_package_examples_top20.md", "Render-Package Examples (Top 20)", render_pkg_examples)
    _write_examples_md(out_root / "regression_examples_top20.md", "Regression Examples (Top 20)", regression_rows)

    # Query type breakdown.
    qtype_bucket: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        if datasets and run.dataset not in set(datasets):
            continue
        for q in run.query_rows:
            qtype = _infer_query_type(_safe_text(q.get("question", "")))
            qtype_bucket[(run.dataset, run.profile, qtype)].append(q)

    qtype_rows: List[Dict[str, Any]] = []
    best_by_dataset_qtype: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for (dataset, profile, qtype), rows_q in qtype_bucket.items():
        f1s = [_safe_float((r.get("metrics", {}) or {}).get("f1", 0.0), 0.0) for r in rows_q]
        sf_rs = [_safe_float((r.get("metrics", {}) or {}).get("supporting_fact_recall", 0.0), 0.0) for r in rows_q]
        ahs = [_safe_float((r.get("metrics", {}) or {}).get("answer_surface_present", 0.0), 0.0) for r in rows_q]
        dens = [_safe_float((r.get("metrics", {}) or {}).get("answer_surface_token_density", 0.0), 0.0) for r in rows_q]
        row = {
            "dataset": dataset,
            "profile": profile,
            "query_type": qtype,
            "count": int(len(rows_q)),
            "F1": float(_mean(f1s)),
            "supporting_fact_recall": float(_mean(sf_rs)),
            "answer_string_hit": float(_mean(ahs)),
            "answer_bearing_density": float(_mean(dens)),
            "best_profile_by_type": "",
        }
        qtype_rows.append(row)
        key = (dataset, qtype)
        cur_best = best_by_dataset_qtype.get(key)
        if cur_best is None or row["F1"] > cur_best[1]:
            best_by_dataset_qtype[key] = (profile, float(row["F1"]))

    for row in qtype_rows:
        row["best_profile_by_type"] = best_by_dataset_qtype.get((row["dataset"], row["query_type"]), ("", 0.0))[0]

    _write_csv(
        out_root / "query_type_breakdown.csv",
        sorted(qtype_rows, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("query_type")), _safe_text(r.get("profile")))),
        fieldnames=[
            "dataset",
            "profile",
            "query_type",
            "count",
            "F1",
            "supporting_fact_recall",
            "answer_string_hit",
            "answer_bearing_density",
            "best_profile_by_type",
        ],
    )

    # Decision.
    decision = "hold"
    # Reject if any variant violates guardrail strongly.
    reject = False
    for d in delta_rows:
        if _safe_float(d.get("delta_F1", 0.0), 0.0) < -0.02:
            reject = True
        if _safe_float(d.get("delta_avg_context_tokens_pct", 0.0), 0.0) > 20.0:
            reject = True
        if _safe_float(d.get("delta_retrieval_ms_pct", 0.0), 0.0) > 20.0:
            reject = True
    if reject:
        decision = "reject"
    else:
        # Strong accept if one variant has robust non-regressive lift.
        by_variant: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for d in delta_rows:
            by_variant[_safe_text(d.get("variant_profile"))].append(d)
        for profile, rows_p in by_variant.items():
            if not rows_p:
                continue
            avg_df1 = _mean([_safe_float(x.get("delta_F1", 0.0), 0.0) for x in rows_p])
            min_df1 = min([_safe_float(x.get("delta_F1", 0.0), 0.0) for x in rows_p])
            avg_df1k = _mean([_safe_float(x.get("delta_F1_per_1k_context_tokens", 0.0), 0.0) for x in rows_p])
            avg_dctx_pct = _mean([_safe_float(x.get("delta_avg_context_tokens_pct", 0.0), 0.0) for x in rows_p])
            if avg_df1 >= 0.02 and min_df1 >= -0.01 and avg_df1k >= 0.0 and avg_dctx_pct <= 10.0:
                decision = f"strong_accept:{profile}"
                break
        if decision == "hold":
            # Mild accept candidate if at least non-regressive average.
            for profile, rows_p in by_variant.items():
                avg_df1 = _mean([_safe_float(x.get("delta_F1", 0.0), 0.0) for x in rows_p])
                min_df1 = min([_safe_float(x.get("delta_F1", 0.0), 0.0) for x in rows_p]) if rows_p else 0.0
                if avg_df1 >= 0.0 and min_df1 >= -0.01:
                    decision = f"accept_candidate:{profile}"
                    break

    summary = {
        "phase": "phase6w_unit_transition_atomization",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "datasets": list(datasets),
        "baseline_profile": baseline_profile,
        "variant_profiles": list(variant_profiles),
        "artifact_counts": dict((report.get("counts", {}) or {})),
        "rows_main": rows_main,
        "delta_rows": delta_rows,
        "unit_transition_rows_count": int(len(stage_rows)),
        "unit_transition_by_dataset_count": int(len(unit_by_dataset_rows)),
        "atomization_diagnostics_count": int(len(atom_diag_rows)),
        "query_type_breakdown_count": int(len(qtype_rows)),
        "paired_query_count": int(len(paired_rows)),
        "decision": decision,
    }
    _write_json(out_root / "phase6w_unit_transition_atomization_summary.json", summary)

    # Markdown.
    counts = dict((report.get("counts", {}) or {}))
    lines: List[str] = []
    lines.append("# PHASE6W Unit Transition & Atomization Audit")
    lines.append("")
    lines.append("## 1. Goal")
    lines.append("Audit whether unit transitions (sentence -> atom -> selected -> rendered) are preserving answer-enabling evidence.")
    lines.append("")
    lines.append("## 2. Artifact Validation")
    lines.append(
        "- scheduled_runs={scheduled_runs}, completed_run_names={completed_run_names}, incomplete_attempts={incomplete_attempts}, extra_not_in_manifest={extra_not_in_manifest}, multi_attempt_run_names={multi_attempt_run_names}, parse_errors={parse_errors}".format(
            scheduled_runs=counts.get("scheduled_runs", "n/a"),
            completed_run_names=counts.get("completed_run_names", "n/a"),
            incomplete_attempts=counts.get("incomplete_attempts", "n/a"),
            extra_not_in_manifest=counts.get("extra_not_in_manifest", "n/a"),
            multi_attempt_run_names=counts.get("multi_attempt_run_names", "n/a"),
            parse_errors=counts.get("parse_errors", "n/a"),
        )
    )
    lines.append("")

    lines.append("## 3. Paired Main Results")
    lines.append("| dataset | profile | role | Recall@5 | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | supporting_fact_precision | supporting_fact_recall | supporting_fact_f1 | retrieval_ms | generation_ms | total_ms | answer_string_hit | answer_bearing_density |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows_main, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {role} | {r5} | {em} | {f1} | {ctx} | {f1k} | {sp} | {sr} | {sf1} | {rm} | {gm} | {tm} | {ahit} | {aden} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("profile")),
                role=_safe_text(row.get("role")),
                r5=_fmt(row.get("Recall@5")),
                em=_fmt(row.get("EM")),
                f1=_fmt(row.get("F1")),
                ctx=_fmt(row.get("avg_context_tokens"), 2),
                f1k=_fmt(row.get("F1_per_1k_context_tokens")),
                sp=_fmt(row.get("supporting_fact_precision")),
                sr=_fmt(row.get("supporting_fact_recall")),
                sf1=_fmt(row.get("supporting_fact_f1")),
                rm=_fmt(row.get("retrieval_ms"), 2),
                gm=_fmt(row.get("generation_ms"), 2),
                tm=_fmt(row.get("total_ms"), 2),
                ahit=_fmt(row.get("answer_string_hit")),
                aden=_fmt(row.get("answer_bearing_density")),
            )
        )
    lines.append("")

    lines.append("## 4. Delta vs Baseline")
    lines.append("| dataset | variant_profile | ΔEM | ΔF1 | ΔF1_per_1k | ΔSF_precision | ΔSF_recall | Δavg_context_tokens | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(delta_rows, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("variant_profile")))):
        lines.append(
            "| {dataset} | {profile} | {dem} | {df1} | {df1k} | {dsp} | {dsr} | {dctx} | {drm} | {dtm} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("variant_profile")),
                dem=_fmt(row.get("delta_EM")),
                df1=_fmt(row.get("delta_F1")),
                df1k=_fmt(row.get("delta_F1_per_1k_context_tokens")),
                dsp=_fmt(row.get("delta_SF_precision")),
                dsr=_fmt(row.get("delta_SF_recall")),
                dctx=_fmt(row.get("delta_avg_context_tokens"), 2),
                drm=_fmt(row.get("delta_retrieval_ms"), 2),
                dtm=_fmt(row.get("delta_total_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 5. Unit Transition Diagnostics")
    lines.append("| dataset | profile | stage | candidate_count | sentence_node_count | atom_count | rendered_sentence_count | SF_recall | chunk_to_sentence_delta | sentence_to_atom_delta | atom_to_selected_delta | selected_to_rendered_delta | latency_ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(unit_by_dataset_rows, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("profile")), _safe_text(r.get("stage")))):
        lines.append(
            "| {dataset} | {profile} | {stage} | {cand} | {sent} | {atom} | {rend} | {sfr} | {d1} | {d2} | {d3} | {d4} | {lat} |".format(
                dataset=_safe_text(row.get("dataset")),
                profile=_safe_text(row.get("profile")),
                stage=_safe_text(row.get("stage")),
                cand=_fmt(row.get("candidate_count"), 2),
                sent=_fmt(row.get("sentence_node_count"), 2),
                atom=_fmt(row.get("atom_count"), 2),
                rend=_fmt(row.get("rendered_sentence_count"), 2),
                sfr=_fmt(row.get("supporting_fact_recall")),
                d1=_fmt(row.get("chunk_to_sentence_recall_delta")),
                d2=_fmt(row.get("sentence_to_atom_recall_delta")),
                d3=_fmt(row.get("atom_to_selected_recall_delta")),
                d4=_fmt(row.get("selected_to_rendered_recall_delta")),
                lat=_fmt(row.get("stage_latency_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 6. Atomization Diagnostics")
    lines.append("| dataset | profile | atomization_enabled | atom_span_max_sentences | avg_num_atoms | avg_selected_atoms | avg_atom_sentences | avg_atom_tokens | minimal_package_apply_rate | dominant_skip_reason |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    atom_group: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in atom_diag_rows:
        atom_group[(_safe_text(row.get("dataset")), _safe_text(row.get("profile")))].append(row)
    for dataset, profile in sorted(atom_group.keys()):
        g = atom_group[(dataset, profile)]
        apply_rate = _mean([1.0 if bool(x.get("minimal_package_applied", False)) else 0.0 for x in g])
        reasons: Dict[str, int] = defaultdict(int)
        for x in g:
            reason = _safe_text(x.get("minimal_package_dominant_skip_reason"))
            if reason:
                reasons[reason] += 1
        dom_reason = "n/a"
        if reasons:
            dom_reason = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        lines.append(
            "| {dataset} | {profile} | {atom_on} | {span} | {num_atoms} | {num_sel} | {avg_sent} | {avg_tok} | {apply_rate} | {reason} |".format(
                dataset=dataset,
                profile=profile,
                atom_on=_fmt(_mean([1.0 if bool(x.get("atomization_enabled", True)) else 0.0 for x in g])),
                span=_fmt(_mean([_safe_float(x.get("atom_span_max_sentences", 0), 0.0) for x in g]), 2),
                num_atoms=_fmt(_mean([_safe_float(x.get("num_atoms", 0.0), 0.0) for x in g]), 2),
                num_sel=_fmt(_mean([_safe_float(x.get("num_selected_atoms", 0.0), 0.0) for x in g]), 2),
                avg_sent=_fmt(_mean([_safe_float(x.get("avg_atom_sentences", 0.0), 0.0) for x in g]), 2),
                avg_tok=_fmt(_mean([_safe_float(x.get("avg_atom_tokens", 0.0), 0.0) for x in g]), 2),
                apply_rate=_fmt(apply_rate),
                reason=dom_reason,
            )
        )
    lines.append("")

    lines.append("## 7. Query-Type Breakdown")
    lines.append("| dataset | query_type | profile | count | F1 | SF_recall | answer_string_hit | answer_bearing_density | best_profile_by_type |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for row in sorted(qtype_rows, key=lambda r: (_safe_text(r.get("dataset")), _safe_text(r.get("query_type")), _safe_text(r.get("profile")))):
        lines.append(
            "| {dataset} | {qtype} | {profile} | {count} | {f1} | {sfr} | {ahit} | {dens} | {best} |".format(
                dataset=_safe_text(row.get("dataset")),
                qtype=_safe_text(row.get("query_type")),
                profile=_safe_text(row.get("profile")),
                count=_safe_int(row.get("count", 0), 0),
                f1=_fmt(row.get("F1")),
                sfr=_fmt(row.get("supporting_fact_recall")),
                ahit=_fmt(row.get("answer_string_hit")),
                dens=_fmt(row.get("answer_bearing_density")),
                best=_safe_text(row.get("best_profile_by_type")),
            )
        )
    lines.append("")

    lines.append("## 8. No-Atomization Analysis")
    lines.append(f"- examples: `{out_root / 'no_atomization_examples_top20.md'}`")
    lines.append("")
    lines.append("## 9. Atom-Span-3 Analysis")
    lines.append(f"- examples: `{out_root / 'atom_span3_examples_top20.md'}`")
    lines.append("")
    lines.append("## 10. Render-Package Analysis")
    lines.append(f"- examples: `{out_root / 'render_package_examples_top20.md'}`")
    lines.append("")
    lines.append("## 11. Decision")
    lines.append(f"- {decision}")
    lines.append("")
    lines.append(f"- regressions: `{out_root / 'regression_examples_top20.md'}`")
    lines.append("")

    (out_root / "PHASE6W_UNIT_TRANSITION_ATOMIZATION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", required=True)
    p.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    p.add_argument("--baseline-profile", default="unified_acr_rcedr_v12")
    p.add_argument(
        "--variant-profiles",
        default="unified_acr_rcedr_v12_no_atomization,unified_acr_rcedr_v12_atom_span3,unified_acr_rcedr_v12_render_package",
    )
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root).resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    variant_profiles = [x.strip() for x in str(args.variant_profiles).split(",") if x.strip()]
    summary = _build_summary(
        out_root=out_root,
        datasets=datasets,
        baseline_profile=_safe_text(args.baseline_profile),
        variant_profiles=variant_profiles,
        top_k=max(1, int(args.top_k)),
    )
    print(out_root / "PHASE6W_UNIT_TRANSITION_ATOMIZATION_SUMMARY.md")
    print(out_root / "phase6w_unit_transition_atomization_summary.json")
    print(out_root / "unit_transition_by_query.jsonl")
    print(out_root / "unit_transition_by_dataset.csv")
    print(out_root / "atomization_diagnostics.csv")
    print(out_root / "query_type_breakdown.csv")
    print(out_root / "no_atomization_examples_top20.md")
    print(out_root / "atom_span3_examples_top20.md")
    print(out_root / "render_package_examples_top20.md")
    print(out_root / "regression_examples_top20.md")
    _ = summary


if __name__ == "__main__":
    main()
