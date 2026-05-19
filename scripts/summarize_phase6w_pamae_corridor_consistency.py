#!/usr/bin/env python3
"""Summarize PHASE6W-P PAMAE corridor consistency audit runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts
from effirag.evidence_flow_audit import load_samples_for_dataset, stage_metrics
from effirag.utils import safe_div


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


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(float(x) for x in vals) / float(len(vals)))


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


def _parse_unit_id(sentence_id: str) -> Tuple[str, Optional[int]]:
    sid = _safe_text(sentence_id)
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


def _stable_corridor_id(dataset: str, sample_id: str, anchor_id: str, seed_id: str, rank: int) -> str:
    base = f"{_safe_text(dataset)}::{_safe_text(sample_id)}::{_safe_text(anchor_id)}::{_safe_text(seed_id)}::{int(rank)}"
    return "c_" + hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, Mapping):
            rows.append(dict(obj))
    return rows


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
    seen = set()
    out: List[str] = []
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
        "answer_bearing_density": _safe_float(summary.get("answer_surface_token_density", 0.0), 0.0),
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
    }


@dataclass
class RunData:
    run_name: str
    dataset: str
    profile: str
    summary: Dict[str, Any]
    query_rows: List[Dict[str, Any]]


def _load_runs(out_root: Path, datasets: Sequence[str], profile: str) -> Tuple[Dict[str, Any], List[RunData]]:
    report = collect_phase6s_artifacts(out_root)
    scheduled = {_safe_text(r.get("run_name")): dict(r) for r in list(report.get("scheduled_runs", []) or [])}
    runs: List[RunData] = []
    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        meta = scheduled.get(run_name, {})
        dataset = _safe_text(meta.get("dataset", item.get("dataset", "")))
        run_profile = _safe_text(meta.get("profile", item.get("profile", "")))
        if dataset not in set(datasets):
            continue
        if run_profile != profile:
            continue
        summary = dict(item.get("summary", {}) or {})
        qpath = Path(_safe_text(item.get("query_path"))).resolve()
        qrows = _read_jsonl(qpath)
        runs.append(RunData(run_name, dataset, run_profile, summary, qrows))
    return report, runs


def _fallback_corridor_rows(dataset: str, sample_id: str, retrieval_corridors: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, c in enumerate(list(retrieval_corridors or []), start=1):
        cid = _safe_text(c.get("corridor_id"))
        anchors = list(c.get("anchors", []) or [])
        anchor_id = _safe_text(anchors[0]) if len(anchors) >= 1 else ""
        seed_id = _safe_text(anchors[1]) if len(anchors) >= 2 else ""
        if not cid:
            cid = _stable_corridor_id(dataset, sample_id, anchor_id, seed_id, idx)
        main_ids = [str(x) for x in list(c.get("main_path_unit_ids", c.get("main_path_sentence_ids", [])) or []) if str(x)]
        support_ids = [str(x) for x in list(c.get("support_unit_ids", c.get("support_sentence_ids", [])) or []) if str(x)]
        connector_ids = [str(x) for x in list(c.get("connector_adjacent_unit_ids", c.get("connector_adjacent_sentence_ids", [])) or []) if str(x)]
        sentence_ids = _ordered_unique(main_ids + support_ids + connector_ids)
        path_nodes = [str(x) for x in list(c.get("path_node_ids", []) or []) if str(x)]
        corridor_nodes = _ordered_unique(path_nodes + sentence_ids)
        entity_nodes = [n for n in corridor_nodes if n.startswith("e::")]
        chunk_nodes = [n for n in corridor_nodes if n.startswith("chunk::")]
        anchor_title, _ = _parse_unit_id(anchor_id)
        seed_title, _ = _parse_unit_id(seed_id)
        rows.append(
            {
                "corridor_id": cid,
                "anchor_id": anchor_id,
                "seed_id": seed_id,
                "anchor_title_or_entity": anchor_title or anchor_id,
                "seed_title_or_entity": seed_title or seed_id,
                "corridor_node_ids": corridor_nodes,
                "corridor_entity_node_ids": entity_nodes,
                "corridor_chunk_node_ids": chunk_nodes,
                "corridor_sentence_ids": sentence_ids,
                "main_path_unit_ids": list(main_ids),
                "support_unit_ids": list(support_ids),
                "connector_adjacent_unit_ids": list(connector_ids),
                "corridor_node_count_before_trim": int(len(sentence_ids)),
                "corridor_node_count_after_trim": int(len(sentence_ids)),
                "corridor_score": _safe_float(c.get("corridor_score", 0.0), 0.0),
                "corridor_rank": int(idx),
                "corridor_source_stage": _safe_text(c.get("refine_mode", "phase2_local_refinement")) or "phase2_local_refinement",
            }
        )
    return rows


def _corridor_ids_from_entry(entry: Mapping[str, Any]) -> List[str]:
    ids = list(entry.get("origin_corridor_ids", []) or [])
    if not ids:
        ids = list(entry.get("corridor_ids", []) or [])
    best = _safe_text(entry.get("best_corridor_id", ""))
    if best:
        ids = [best] + list(ids)
    return _ordered_unique([_safe_text(x) for x in ids])


def _best_corridor_id(entry: Mapping[str, Any]) -> str:
    ids = _corridor_ids_from_entry(entry)
    return ids[0] if ids else ""


def _dominant_share(corridor_ids: Sequence[str]) -> float:
    ids = [_safe_text(x) for x in list(corridor_ids or []) if _safe_text(x)]
    if not ids:
        return 0.0
    counts: Dict[str, int] = defaultdict(int)
    for cid in ids:
        counts[cid] += 1
    return float(max(counts.values()) / float(len(ids)))


def _coverage_ratio(num: int, den: int) -> float:
    return float(safe_div(float(num), float(max(1, den))))


def _path_completeness_score(
    rows: Sequence[Mapping[str, Any]],
    *,
    use_bridge_from_feature: bool = True,
) -> Dict[str, Any]:
    anchor_side_hit = False
    seed_side_hit = False
    bridge_node_hit = False
    for row in list(rows or []):
        if bool(row.get("is_main_candidate", False)):
            anchor_side_hit = True
        if bool(row.get("is_support_candidate", False)):
            seed_side_hit = True
        if use_bridge_from_feature and bool(row.get("is_connector_adjacent", False)):
            bridge_node_hit = True
        try:
            bridge_score = float(row.get("bridge_score", 0.0) or 0.0)
        except Exception:
            bridge_score = 0.0
        if bridge_score > 0.0:
            bridge_node_hit = True
    score = float((float(anchor_side_hit) + float(seed_side_hit) + float(bridge_node_hit)) / 3.0)
    return {
        "anchor_side_hit": int(anchor_side_hit),
        "seed_side_hit": int(seed_side_hit),
        "bridge_node_hit": int(bridge_node_hit),
        "anchor_seed_pair_covered": int(anchor_side_hit and seed_side_hit),
        "path_completeness_score": float(score),
    }


def _q75(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(float(x) for x in values)
    idx = int(math.ceil(0.75 * len(s))) - 1
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])


def _q25(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(float(x) for x in values)
    idx = int(math.floor(0.25 * (len(s) - 1)))
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])


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
    lines: List[str] = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for i, row in enumerate(rows, start=1):
        lines.append(f"## {i}. {_safe_text(row.get('dataset'))} / {_safe_text(row.get('sample_id'))}")
        lines.append(f"- question: {_safe_text(row.get('question'))}")
        lines.append(
            f"- F1/EM: {_fmt(row.get('f1'))} / {_fmt(row.get('em'))}, SF_recall: {_fmt(row.get('sf_recall'))}"
        )
        lines.append(
            f"- selected_path_completeness: {_fmt(row.get('selected_path_completeness_score'))}, rendered_path_completeness: {_fmt(row.get('rendered_path_completeness_score'))}"
        )
        lines.append(
            f"- selected_disconnected_corridor_count/rendered_disconnected_corridor_count: {int(row.get('selected_disconnected_corridor_count', 0))} / {int(row.get('rendered_disconnected_corridor_count', 0))}"
        )
        lines.append(
            f"- dominant_corridor_selected_share/rendered_share: {_fmt(row.get('dominant_corridor_selected_share'))} / {_fmt(row.get('dominant_corridor_rendered_share'))}"
        )
        lines.append(
            f"- selected_candidate_corridors: {_safe_text(','.join(row.get('selected_corridor_ids', [])[:8]))}"
        )
        lines.append(
            f"- rendered_candidate_corridors: {_safe_text(','.join(row.get('rendered_corridor_ids', [])[:8]))}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build(out_root: Path, datasets: Sequence[str], profile: str, top_k: int) -> Dict[str, Any]:
    report, runs = _load_runs(out_root, datasets, profile)
    run_by_dataset = {r.dataset: r for r in runs}
    sample_maps = {d: load_samples_for_dataset(d) for d in datasets}

    by_query: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    qtype_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        run = run_by_dataset.get(dataset)
        if run is None:
            continue
        sample_map = sample_maps.get(dataset, {})
        summary_main = _summary_metrics(run.summary)
        per_dataset_query: List[Dict[str, Any]] = []

        for row in run.query_rows:
            sample_id = _safe_text(row.get("sample_id"))
            sample = sample_map.get(sample_id)
            if sample is None:
                continue
            question = _safe_text(getattr(sample, "question", row.get("question", "")))
            retrieval = dict((row.get("retrieval", {}) or {}))
            diag = dict((retrieval.get("diagnostics", {}) or {}))
            graph_mode = _safe_text(diag.get("graph_mode", "current_entity_graph"))

            # Corridor candidates (phase2 output snapshot).
            corridor_rows = list(diag.get("phase6w_corridor_candidates_before_trim", []) or [])
            if not corridor_rows:
                corridor_rows = _fallback_corridor_rows(dataset, sample_id, list(retrieval.get("corridors", []) or []))
            corridor_ids = [_safe_text(c.get("corridor_id")) for c in corridor_rows if _safe_text(c.get("corridor_id"))]

            # Sentence candidates before ABR.
            sentence_rows = list(diag.get("phase6w_sentence_candidates_before_abr", []) or [])
            if not sentence_rows:
                sid_list = list(diag.get("phase6u_sentence_candidates_after_text_rerank_ids", []) or [])
                txt_list = list(diag.get("phase6u_sentence_candidates_after_text_rerank_texts", []) or [])
                for sid, txt in zip(sid_list, txt_list):
                    title, sent_idx = _parse_unit_id(str(sid))
                    sentence_rows.append(
                        {
                            "sentence_id": str(sid),
                            "title_or_source": str(title),
                            "sentence_idx": sent_idx,
                            "text": str(txt or ""),
                            "origin_corridor_ids": [],
                            "best_corridor_id": "",
                            "is_main_candidate": False,
                            "is_support_candidate": False,
                            "is_connector_adjacent": False,
                            "query_overlap_score": 0.0,
                            "semantic_query_score": 0.0,
                            "base_retrieval_score": 0.0,
                        }
                    )
            sentence_row_by_id = {_safe_text(s.get("sentence_id")): dict(s) for s in sentence_rows if _safe_text(s.get("sentence_id"))}
            sentence_feature_table = dict(diag.get("phase6w_sentence_feature_table", {}) or {})

            # Atom candidates before ABR and selected atoms.
            atom_rows = list(diag.get("phase6w_evidence_atoms_before_abr", []) or [])
            if not atom_rows:
                aid = list(diag.get("phase6u_evidence_atoms_before_abr_ids", []) or [])
                atx = list(diag.get("phase6u_evidence_atoms_before_abr_texts", []) or [])
                for sid, txt in zip(aid, atx):
                    atom_rows.append(
                        {
                            "atom_id": f"atom::{sid}",
                            "source_sentence_ids": [str(sid)],
                            "sentence_id": str(sid),
                            "text": str(txt or ""),
                            "origin_corridor_ids": [],
                            "best_corridor_id": "",
                            "atom_token_count": 0,
                            "atom_score_before_abr": 0.0,
                            "answerability_score": 0.0,
                            "bridge_score": 0.0,
                        }
                    )
            selected_atom_rows = list(diag.get("phase6w_abr_selected_evidence", []) or [])
            if not selected_atom_rows:
                sid = list(diag.get("phase6u_abr_selected_evidence_ids", []) or [])
                stx = list(diag.get("phase6u_abr_selected_evidence_texts", []) or [])
                for idx, (sent_id, txt) in enumerate(zip(sid, stx), start=1):
                    selected_atom_rows.append(
                        {
                            "selected_rank": int(idx),
                            "atom_id": f"atom::{sent_id}",
                            "source_sentence_ids": [str(sent_id)],
                            "sentence_id": str(sent_id),
                            "text": str(txt or ""),
                            "origin_corridor_ids": [],
                            "best_corridor_id": "",
                            "abr_delta_score": 0.0,
                            "bridge_score": 0.0,
                        }
                    )

            atom_candidate_corridors = set()
            for a in atom_rows:
                best = _best_corridor_id(a)
                if best:
                    atom_candidate_corridors.add(best)
            sentence_candidate_corridors = set()
            for s in sentence_rows:
                best = _best_corridor_id(s)
                if best:
                    sentence_candidate_corridors.add(best)
            selected_corridors = set()
            selected_corridor_ids_in_order: List[str] = []
            for s in selected_atom_rows:
                best = _best_corridor_id(s)
                if best:
                    selected_corridors.add(best)
                    selected_corridor_ids_in_order.append(best)

            # Rendered evidence units mapped back to corridor ids.
            rendered = dict((row.get("rendered", {}) or {}))
            rendered_ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
            rendered_texts = list(rendered.get("sentences", []) or [])
            selected_texts = list(retrieval.get("selected_sentences", []) or [])
            selected_ids = list(retrieval.get("selected_sentence_ids", []) or [])

            rendered_rows: List[Dict[str, Any]] = []
            rendered_corridor_ids_in_order: List[str] = []
            atom_by_sentence: Dict[str, Dict[str, Any]] = {}
            for a in selected_atom_rows:
                sent = _safe_text(a.get("sentence_id"))
                if not sent:
                    sents = list(a.get("source_sentence_ids", []) or [])
                    sent = _safe_text(sents[0] if sents else "")
                if sent and sent not in atom_by_sentence:
                    atom_by_sentence[sent] = dict(a)

            for idx, sid in enumerate(rendered_ids, start=1):
                sid_s = _safe_text(sid)
                if not sid_s:
                    continue
                srow = dict(sentence_row_by_id.get(sid_s, {}) or {})
                if not srow and sid_s in sentence_feature_table:
                    f = dict(sentence_feature_table.get(sid_s, {}) or {})
                    srow = {
                        "sentence_id": sid_s,
                        "origin_corridor_ids": list(f.get("corridor_ids", []) or []),
                        "best_corridor_id": _safe_text((list(f.get("corridor_ids", []) or []) or [""])[0]),
                        "is_main_candidate": bool(f.get("is_main_candidate", False)),
                        "is_support_candidate": bool(f.get("is_support_candidate", False)),
                        "is_connector_adjacent": bool(f.get("is_connector_adjacent", False)),
                        "bridge_score": 0.0,
                    }
                atom_row = atom_by_sentence.get(sid_s, {})
                origin_ids = _corridor_ids_from_entry(srow)
                best_cid = _best_corridor_id(srow)
                if not best_cid:
                    best_cid = _best_corridor_id(atom_row)
                if best_cid:
                    rendered_corridor_ids_in_order.append(best_cid)
                rendered_rows.append(
                    {
                        "render_rank": int(idx),
                        "sentence_id": sid_s,
                        "atom_id": _safe_text(atom_row.get("atom_id", "")),
                        "origin_corridor_ids": list(origin_ids),
                        "best_corridor_id": best_cid,
                        "is_main_candidate": bool(srow.get("is_main_candidate", False)),
                        "is_support_candidate": bool(srow.get("is_support_candidate", False)),
                        "is_connector_adjacent": bool(srow.get("is_connector_adjacent", False)),
                        "bridge_score": _safe_float(atom_row.get("bridge_score", 0.0), 0.0),
                        "text": str(rendered_texts[idx - 1] if idx - 1 < len(rendered_texts) else ""),
                    }
                )

            rendered_corridors = {cid for cid in rendered_corridor_ids_in_order if cid}

            # Corridor consistency metrics.
            num_candidate_corridors = int(len(corridor_ids))
            num_corridors_with_sentence_candidates = int(len(sentence_candidate_corridors))
            num_corridors_represented_in_atoms = int(len(atom_candidate_corridors))
            num_corridors_represented_in_selected = int(len(selected_corridors))
            num_corridors_represented_in_rendered = int(len(rendered_corridors))
            selected_cov = _coverage_ratio(
                num_corridors_represented_in_selected,
                max(1, num_corridors_represented_in_atoms),
            )
            rendered_cov = _coverage_ratio(
                num_corridors_represented_in_rendered,
                max(1, num_corridors_represented_in_atoms),
            )
            selected_dom_share = _dominant_share(selected_corridor_ids_in_order)
            rendered_dom_share = _dominant_share(rendered_corridor_ids_in_order)
            selected_disconnected = max(0, int(len(selected_corridors)) - 1)
            rendered_disconnected = max(0, int(len(rendered_corridors)) - 1)

            selected_enriched_rows: List[Dict[str, Any]] = []
            for a in selected_atom_rows:
                sent = _safe_text(a.get("sentence_id"))
                if not sent:
                    src = list(a.get("source_sentence_ids", []) or [])
                    sent = _safe_text(src[0] if src else "")
                srow = dict(sentence_row_by_id.get(sent, {}) or {})
                selected_enriched_rows.append(
                    {
                        "sentence_id": sent,
                        "is_main_candidate": bool(srow.get("is_main_candidate", False)),
                        "is_support_candidate": bool(srow.get("is_support_candidate", False)),
                        "is_connector_adjacent": bool(srow.get("is_connector_adjacent", False)),
                        "bridge_score": _safe_float(a.get("bridge_score", 0.0), 0.0),
                    }
                )
            selected_path = _path_completeness_score(selected_enriched_rows)
            rendered_path = _path_completeness_score(rendered_rows)

            # Stage/offline metrics.
            sentence_candidate_ids = [_safe_text(s.get("sentence_id")) for s in sentence_rows if _safe_text(s.get("sentence_id"))]
            sentence_candidate_texts = [str(s.get("text", "") or "") for s in sentence_rows]
            selected_ids_norm = [_normalize_sentence_support_id(x) for x in _ordered_unique(selected_ids)]
            rendered_ids_norm = [_normalize_sentence_support_id(x) for x in _ordered_unique(rendered_ids)]
            selected_text_map = {str(sid): str(txt or "") for sid, txt in zip(selected_ids, selected_texts)}
            rendered_text_map = {str(sid): str(txt or "") for sid, txt in zip(rendered_ids, rendered_texts)}
            selected_texts_norm = [selected_text_map.get(sid, "") for sid in selected_ids]
            rendered_texts_norm = [rendered_text_map.get(sid, "") for sid in rendered_ids]

            candidate_stage = stage_metrics(sample, sentence_candidate_ids, sentence_candidate_texts, graph_mode)
            selected_stage = stage_metrics(sample, selected_ids_norm, selected_texts_norm, graph_mode)
            rendered_stage = stage_metrics(sample, rendered_ids_norm, rendered_texts_norm, graph_mode)
            cand_ans = _answer_sufficiency_metrics(sample, sentence_candidate_texts)
            sel_ans = _answer_sufficiency_metrics(sample, selected_texts_norm)
            ren_ans = _answer_sufficiency_metrics(sample, rendered_texts_norm)
            qm = _query_metrics(row)

            qrow = {
                "dataset": dataset,
                "profile": run.profile,
                "run_name": run.run_name,
                "sample_id": sample_id,
                "question": question,
                "query_type": _infer_query_type(question),
                "f1": float(qm["f1"]),
                "em": float(qm["em"]),
                "sf_recall": float(qm["sf_recall"]),
                "sf_precision": float(qm["sf_precision"]),
                "retrieval_ms": float(qm["retrieval_ms"]),
                "generation_ms": float(qm["generation_ms"]),
                "total_ms": float(qm["total_ms"]),
                "prompt_tokens": float(qm["prompt_tokens"]),
                "num_candidate_corridors": num_candidate_corridors,
                "num_corridors_with_sentence_candidates": num_corridors_with_sentence_candidates,
                "num_corridors_represented_in_atoms": num_corridors_represented_in_atoms,
                "num_corridors_represented_in_selected": num_corridors_represented_in_selected,
                "num_corridors_represented_in_rendered": num_corridors_represented_in_rendered,
                "selected_corridor_coverage_ratio": float(selected_cov),
                "rendered_corridor_coverage_ratio": float(rendered_cov),
                "selected_same_corridor_ratio": float(selected_dom_share),
                "rendered_same_corridor_ratio": float(rendered_dom_share),
                "selected_disconnected_corridor_count": int(selected_disconnected),
                "rendered_disconnected_corridor_count": int(rendered_disconnected),
                "dominant_corridor_selected_share": float(selected_dom_share),
                "dominant_corridor_rendered_share": float(rendered_dom_share),
                "selected_path_completeness_score": float(selected_path["path_completeness_score"]),
                "rendered_path_completeness_score": float(rendered_path["path_completeness_score"]),
                "selected_anchor_side_hit": int(selected_path["anchor_side_hit"]),
                "selected_seed_side_hit": int(selected_path["seed_side_hit"]),
                "selected_bridge_node_hit": int(selected_path["bridge_node_hit"]),
                "selected_anchor_seed_pair_covered": int(selected_path["anchor_seed_pair_covered"]),
                "rendered_anchor_side_hit": int(rendered_path["anchor_side_hit"]),
                "rendered_seed_side_hit": int(rendered_path["seed_side_hit"]),
                "rendered_bridge_node_hit": int(rendered_path["bridge_node_hit"]),
                "rendered_anchor_seed_pair_covered": int(rendered_path["anchor_seed_pair_covered"]),
                "candidate_corridor_SF_recall": float(candidate_stage.get("sf_R", 0.0)),
                "selected_corridor_SF_recall": float(selected_stage.get("sf_R", 0.0)),
                "rendered_corridor_SF_recall": float(rendered_stage.get("sf_R", 0.0)),
                "candidate_corridor_SF_precision": float(candidate_stage.get("sf_P", 0.0)),
                "selected_corridor_SF_precision": float(selected_stage.get("sf_P", 0.0)),
                "rendered_corridor_SF_precision": float(rendered_stage.get("sf_P", 0.0)),
                "candidate_corridor_answer_string_hit": float(cand_ans["answer_string_hit"]),
                "selected_answer_string_hit": float(sel_ans["answer_string_hit"]),
                "rendered_answer_string_hit": float(ren_ans["answer_string_hit"]),
                "candidate_answer_bearing_density": float(cand_ans["answer_bearing_density"]),
                "selected_answer_bearing_density": float(sel_ans["answer_bearing_density"]),
                "rendered_answer_bearing_density": float(ren_ans["answer_bearing_density"]),
                "selected_corridor_ids": sorted(list(selected_corridors)),
                "rendered_corridor_ids": sorted(list(rendered_corridors)),
                "candidate_corridor_ids": sorted(list(atom_candidate_corridors)),
                "corridor_rows": corridor_rows,
            }
            by_query.append(qrow)
            per_dataset_query.append(qrow)

        # Dataset aggregates.
        if not per_dataset_query:
            continue
        ds = {
            "dataset": dataset,
            "profile": run.profile,
            **summary_main,
            "answer_string_hit": float(_mean([r["rendered_answer_string_hit"] for r in per_dataset_query])),
            "answer_bearing_density": float(_mean([r["rendered_answer_bearing_density"] for r in per_dataset_query])),
            "num_candidate_corridors_avg": float(_mean([r["num_candidate_corridors"] for r in per_dataset_query])),
            "num_corridors_represented_in_selected_avg": float(
                _mean([r["num_corridors_represented_in_selected"] for r in per_dataset_query])
            ),
            "num_corridors_represented_in_rendered_avg": float(
                _mean([r["num_corridors_represented_in_rendered"] for r in per_dataset_query])
            ),
            "selected_corridor_coverage_ratio_avg": float(
                _mean([r["selected_corridor_coverage_ratio"] for r in per_dataset_query])
            ),
            "rendered_corridor_coverage_ratio_avg": float(
                _mean([r["rendered_corridor_coverage_ratio"] for r in per_dataset_query])
            ),
            "selected_disconnected_corridor_count_avg": float(
                _mean([r["selected_disconnected_corridor_count"] for r in per_dataset_query])
            ),
            "rendered_disconnected_corridor_count_avg": float(
                _mean([r["rendered_disconnected_corridor_count"] for r in per_dataset_query])
            ),
            "dominant_corridor_selected_share_avg": float(
                _mean([r["dominant_corridor_selected_share"] for r in per_dataset_query])
            ),
            "dominant_corridor_rendered_share_avg": float(
                _mean([r["dominant_corridor_rendered_share"] for r in per_dataset_query])
            ),
            "selected_path_completeness_score_avg": float(
                _mean([r["selected_path_completeness_score"] for r in per_dataset_query])
            ),
            "rendered_path_completeness_score_avg": float(
                _mean([r["rendered_path_completeness_score"] for r in per_dataset_query])
            ),
            "candidate_corridor_SF_recall_avg": float(_mean([r["candidate_corridor_SF_recall"] for r in per_dataset_query])),
            "selected_corridor_SF_recall_avg": float(_mean([r["selected_corridor_SF_recall"] for r in per_dataset_query])),
            "rendered_corridor_SF_recall_avg": float(_mean([r["rendered_corridor_SF_recall"] for r in per_dataset_query])),
            "num_queries": int(len(per_dataset_query)),
        }
        dataset_rows.append(ds)

        # Query-type breakdown.
        qtype_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for q in per_dataset_query:
            qtype_groups[str(q.get("query_type", "other"))].append(q)
        for qtype, group in sorted(qtype_groups.items()):
            qtype_rows.append(
                {
                    "dataset": dataset,
                    "query_type": qtype,
                    "count": int(len(group)),
                    "F1": float(_mean([r["f1"] for r in group])),
                    "SF_recall": float(_mean([r["sf_recall"] for r in group])),
                    "answer_string_hit": float(_mean([r["rendered_answer_string_hit"] for r in group])),
                    "answer_bearing_density": float(_mean([r["rendered_answer_bearing_density"] for r in group])),
                    "selected_path_completeness_score": float(_mean([r["selected_path_completeness_score"] for r in group])),
                    "selected_disconnected_corridor_count": float(
                        _mean([r["selected_disconnected_corridor_count"] for r in group])
                    ),
                    "dominant_corridor_selected_share": float(_mean([r["dominant_corridor_selected_share"] for r in group])),
                }
            )

    # High/low examples.
    high_examples = sorted(
        by_query,
        key=lambda r: (
            float(r.get("f1", 0.0)),
            float(r.get("selected_path_completeness_score", 0.0)),
            -float(r.get("selected_disconnected_corridor_count", 0.0)),
        ),
        reverse=True,
    )[: max(1, int(top_k))]
    low_examples = sorted(
        by_query,
        key=lambda r: (
            float(r.get("f1", 0.0)),
            -float(r.get("selected_path_completeness_score", 0.0)),
            float(r.get("selected_disconnected_corridor_count", 0.0)),
        ),
    )[: max(1, int(top_k))]

    # Coherence interpretation.
    f1_vals = [float(r.get("f1", 0.0)) for r in by_query]
    q75 = _q75(f1_vals)
    q25 = _q25(f1_vals)
    high_grp = [r for r in by_query if float(r.get("f1", 0.0)) >= q75]
    low_grp = [r for r in by_query if float(r.get("f1", 0.0)) <= q25]
    high_path = _mean([r["selected_path_completeness_score"] for r in high_grp]) if high_grp else 0.0
    low_path = _mean([r["selected_path_completeness_score"] for r in low_grp]) if low_grp else 0.0
    high_disc = _mean([r["selected_disconnected_corridor_count"] for r in high_grp]) if high_grp else 0.0
    low_disc = _mean([r["selected_disconnected_corridor_count"] for r in low_grp]) if low_grp else 0.0
    corridor_consistency_gap = float(high_path - low_path)
    corridor_fragmentation_gap = float(low_disc - high_disc)
    if corridor_consistency_gap >= 0.05 and corridor_fragmentation_gap >= 0.25:
        recommendation = "corridor_consistency_likely_matters"
    else:
        recommendation = "corridor_consistency_not_primary"

    # Q1..Q6 answers.
    q1 = "likely_yes" if _mean([r.get("candidate_corridor_SF_recall", 0.0) for r in by_query]) > 0.45 else "unclear_or_no"
    q2 = "mostly_preserved" if _mean([r.get("selected_corridor_coverage_ratio", 0.0) for r in by_query]) >= 0.60 else "scattered"
    q3 = "yes" if corridor_consistency_gap > 0.0 else "no_clear_signal"
    q4 = "yes" if corridor_fragmentation_gap > 0.0 else "no_clear_signal"
    q5 = "yes" if abs(
        _mean([r.get("selected_path_completeness_score", 0.0) for r in by_query if r.get("dataset") == "hotpotqa"])
        - _mean([r.get("selected_path_completeness_score", 0.0) for r in by_query if r.get("dataset") == "2wikimultihopqa"])
    ) >= 0.05 else "minor_difference"
    q6 = "weak_candidate_source" if q2 == "scattered" else "reasoning_object_partially_preserved"

    summary = {
        "phase": "phase6w_pamae_corridor_consistency",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root.resolve()),
        "datasets": list(datasets),
        "profile": profile,
        "artifact_validation": {
            "scheduled_runs": int(len(list(report.get("scheduled_runs", []) or []))),
            "completed_run_names": int(len(list(report.get("completed_runs", []) or []))),
            "incomplete_attempts": int(len(list(report.get("incomplete_attempts", []) or []))),
            "extra_not_in_manifest": int(len(list(report.get("extra_not_in_manifest", []) or []))),
            "multi_attempt_run_names": int(len(list(report.get("multi_attempt_run_names", []) or []))),
            "parse_errors": int(len(list(report.get("parse_errors", []) or []))),
        },
        "dataset_metrics": list(dataset_rows),
        "query_type_breakdown": list(qtype_rows),
        "global_coherence_stats": {
            "q75_f1_threshold": float(q75),
            "q25_f1_threshold": float(q25),
            "high_group_size": int(len(high_grp)),
            "low_group_size": int(len(low_grp)),
            "high_group_selected_path_completeness": float(high_path),
            "low_group_selected_path_completeness": float(low_path),
            "high_group_selected_disconnected_corridor_count": float(high_disc),
            "low_group_selected_disconnected_corridor_count": float(low_disc),
            "corridor_consistency_gap": float(corridor_consistency_gap),
            "corridor_fragmentation_gap": float(corridor_fragmentation_gap),
        },
        "key_questions": {
            "Q1_phase2_contains_answer_or_support_evidence": str(q1),
            "Q2_abr_preserves_refined_corridor_or_scatter": str(q2),
            "Q3_high_f1_associated_with_path_completeness": str(q3),
            "Q4_low_f1_associated_with_fragmentation": str(q4),
            "Q5_hotpot_vs_2wiki_corridor_requirement_difference": str(q5),
            "Q6_phase2_role_final_reasoning_object_or_weak_source": str(q6),
        },
        "recommendation": str(recommendation),
        "decision": "audit_completed",
    }

    # Outputs.
    _write_jsonl(out_root / "corridor_consistency_by_query.jsonl", by_query)
    _write_csv(
        out_root / "corridor_consistency_by_dataset.csv",
        dataset_rows,
        [
            "dataset",
            "profile",
            "Recall@5",
            "EM",
            "F1",
            "avg_context_tokens",
            "F1_per_1k_context_tokens",
            "supporting_fact_precision",
            "supporting_fact_recall",
            "supporting_fact_f1",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
            "answer_string_hit",
            "answer_bearing_density",
            "num_candidate_corridors_avg",
            "num_corridors_represented_in_selected_avg",
            "num_corridors_represented_in_rendered_avg",
            "selected_corridor_coverage_ratio_avg",
            "rendered_corridor_coverage_ratio_avg",
            "selected_disconnected_corridor_count_avg",
            "rendered_disconnected_corridor_count_avg",
            "dominant_corridor_selected_share_avg",
            "dominant_corridor_rendered_share_avg",
            "selected_path_completeness_score_avg",
            "rendered_path_completeness_score_avg",
            "candidate_corridor_SF_recall_avg",
            "selected_corridor_SF_recall_avg",
            "rendered_corridor_SF_recall_avg",
            "num_queries",
        ],
    )
    _write_csv(
        out_root / "query_type_corridor_breakdown.csv",
        qtype_rows,
        [
            "dataset",
            "query_type",
            "count",
            "F1",
            "SF_recall",
            "answer_string_hit",
            "answer_bearing_density",
            "selected_path_completeness_score",
            "selected_disconnected_corridor_count",
            "dominant_corridor_selected_share",
        ],
    )
    _write_examples_md(
        out_root / "high_f1_corridor_examples_top20.md",
        "High-F1 Corridor-Coherent Examples",
        high_examples,
    )
    _write_examples_md(
        out_root / "low_f1_corridor_fragmentation_examples_top20.md",
        "Low-F1 Corridor-Fragmented Examples",
        low_examples,
    )

    md_lines: List[str] = [
        "# PHASE6W-P PAMAE Corridor Consistency Audit",
        "",
        "## 1. Goal",
        "Audit whether ABR preserves Phase-2 refined local corridor coherence or selects fragmented atoms.",
        "",
        "## 2. Artifact Validation",
        f"- scheduled_runs={summary['artifact_validation']['scheduled_runs']}, completed_run_names={summary['artifact_validation']['completed_run_names']}, incomplete_attempts={summary['artifact_validation']['incomplete_attempts']}, extra_not_in_manifest={summary['artifact_validation']['extra_not_in_manifest']}, multi_attempt_run_names={summary['artifact_validation']['multi_attempt_run_names']}, parse_errors={summary['artifact_validation']['parse_errors']}",
        "",
        "## 3. Main QA Results",
        "| dataset | profile | Recall@5 | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | supporting_fact_precision | supporting_fact_recall | supporting_fact_f1 | retrieval_ms | generation_ms | total_ms | answer_string_hit | answer_bearing_density |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in dataset_rows:
        md_lines.append(
            f"| {ds['dataset']} | {ds['profile']} | {_fmt(ds['Recall@5'])} | {_fmt(ds['EM'])} | {_fmt(ds['F1'])} | {_fmt(ds['avg_context_tokens'],2)} | {_fmt(ds['F1_per_1k_context_tokens'])} | {_fmt(ds['supporting_fact_precision'])} | {_fmt(ds['supporting_fact_recall'])} | {_fmt(ds['supporting_fact_f1'])} | {_fmt(ds['retrieval_ms'],2)} | {_fmt(ds['generation_ms'],2)} | {_fmt(ds['total_ms'],2)} | {_fmt(ds['answer_string_hit'])} | {_fmt(ds['answer_bearing_density'])} |"
        )
    md_lines += [
        "",
        "## 4. Corridor Preservation Metrics",
        "| dataset | num_candidate_corridors_avg | selected_corridor_coverage_ratio_avg | rendered_corridor_coverage_ratio_avg |",
        "|---|---:|---:|---:|",
    ]
    for ds in dataset_rows:
        md_lines.append(
            f"| {ds['dataset']} | {_fmt(ds['num_candidate_corridors_avg'],2)} | {_fmt(ds['selected_corridor_coverage_ratio_avg'])} | {_fmt(ds['rendered_corridor_coverage_ratio_avg'])} |"
        )
    md_lines += [
        "",
        "## 5. Corridor Fragmentation Metrics",
        "| dataset | selected_disconnected_corridor_count_avg | rendered_disconnected_corridor_count_avg | dominant_corridor_selected_share_avg | dominant_corridor_rendered_share_avg |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in dataset_rows:
        md_lines.append(
            f"| {ds['dataset']} | {_fmt(ds['selected_disconnected_corridor_count_avg'],2)} | {_fmt(ds['rendered_disconnected_corridor_count_avg'],2)} | {_fmt(ds['dominant_corridor_selected_share_avg'])} | {_fmt(ds['dominant_corridor_rendered_share_avg'])} |"
        )
    md_lines += [
        "",
        "## 6. Path Completeness Metrics",
        "| dataset | selected_path_completeness_score_avg | rendered_path_completeness_score_avg | candidate_corridor_SF_recall_avg | selected_corridor_SF_recall_avg | rendered_corridor_SF_recall_avg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds in dataset_rows:
        md_lines.append(
            f"| {ds['dataset']} | {_fmt(ds['selected_path_completeness_score_avg'])} | {_fmt(ds['rendered_path_completeness_score_avg'])} | {_fmt(ds['candidate_corridor_SF_recall_avg'])} | {_fmt(ds['selected_corridor_SF_recall_avg'])} | {_fmt(ds['rendered_corridor_SF_recall_avg'])} |"
        )
    md_lines += [
        "",
        "## 7. Query-Type Breakdown",
        "| dataset | query_type | count | F1 | SF_recall | answer_string_hit | selected_path_completeness_score | selected_disconnected_corridor_count |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for qr in qtype_rows:
        md_lines.append(
            f"| {qr['dataset']} | {qr['query_type']} | {qr['count']} | {_fmt(qr['F1'])} | {_fmt(qr['SF_recall'])} | {_fmt(qr['answer_string_hit'])} | {_fmt(qr['selected_path_completeness_score'])} | {_fmt(qr['selected_disconnected_corridor_count'],2)} |"
        )
    md_lines += [
        "",
        "## 8. High-F1 Corridor-Coherent Examples",
        f"- `{out_root / 'high_f1_corridor_examples_top20.md'}`",
        "",
        "## 9. Low-F1 Corridor-Fragmented Examples",
        f"- `{out_root / 'low_f1_corridor_fragmentation_examples_top20.md'}`",
        "",
        "## 10. Interpretation",
        f"- Q1 (Phase2 corridor evidence present): {q1}",
        f"- Q2 (ABR corridor preservation vs scatter): {q2}",
        f"- Q3 (High-F1 ↔ path completeness): {q3}",
        f"- Q4 (Low-F1 ↔ fragmentation): {q4}",
        f"- Q5 (Hotpot vs 2Wiki corridor requirement difference): {q5}",
        f"- Q6 (Phase2 as reasoning object or weak source): {q6}",
        "",
        "## 11. Recommendation",
        f"- {recommendation}",
    ]
    (out_root / "PHASE6W_PAMAE_CORRIDOR_CONSISTENCY_SUMMARY.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    _write_json(out_root / "phase6w_pamae_corridor_consistency_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PHASE6W-P PAMAE corridor consistency audit outputs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    parser.add_argument("--profile", default="unified_acr_rcedr_v12")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    _build(
        out_root=out_root,
        datasets=datasets,
        profile=str(args.profile).strip(),
        top_k=max(1, int(args.top_k)),
    )
    print(out_root / "PHASE6W_PAMAE_CORRIDOR_CONSISTENCY_SUMMARY.md")
    print(out_root / "phase6w_pamae_corridor_consistency_summary.json")
    print(out_root / "corridor_consistency_by_query.jsonl")
    print(out_root / "corridor_consistency_by_dataset.csv")
    print(out_root / "query_type_corridor_breakdown.csv")
    print(out_root / "high_f1_corridor_examples_top20.md")
    print(out_root / "low_f1_corridor_fragmentation_examples_top20.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
