#!/usr/bin/env python3
"""Summarize PHASE6U stagewise evidence audit (read-only retrieval diagnostics)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts

from effirag.evidence_flow_audit import load_samples_for_dataset, stage_metrics, token_count_for_texts


STAGES: List[Tuple[str, str]] = [
    ("anchors", "anchors"),
    ("proposal_candidates", "proposal candidates"),
    ("phase1_run_seed_shortlist", "phase1 run / seed shortlist"),
    ("anchor_seed_pair_shortlist", "anchor-seed pair shortlist"),
    ("local_corridor_candidates", "local corridor candidates"),
    ("corridor_after_rerank_trim", "corridor after rerank / trim"),
    ("sentence_candidates_before_text_rerank", "sentence candidates before text rerank"),
    ("sentence_candidates_after_text_rerank", "sentence candidates after text rerank"),
    ("evidence_atoms_before_abr_selector", "evidence atoms before ABR selector"),
    ("abr_selected_evidence", "ABR-selected evidence"),
    ("rendered_compact_context", "rendered compact context"),
]

DROP_CLASS_BY_STAGE: Dict[str, str] = {
    "anchors": "phase1_candidate_loss",
    "proposal_candidates": "phase1_candidate_loss",
    "phase1_run_seed_shortlist": "run_seed_shortlist_loss",
    "anchor_seed_pair_shortlist": "run_seed_shortlist_loss",
    "local_corridor_candidates": "local_corridor_loss",
    "corridor_after_rerank_trim": "local_corridor_loss",
    "sentence_candidates_before_text_rerank": "sentence_extraction_loss",
    "sentence_candidates_after_text_rerank": "text_rerank_loss",
    "evidence_atoms_before_abr_selector": "abr_selection_loss",
    "abr_selected_evidence": "abr_selection_loss",
    "rendered_compact_context": "rendering_loss",
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


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


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _fmt(value: Any, nd: int = 4) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return "n/a"


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


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        v = _safe_text(value)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


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


def _stage_latency_ms(stage_key: str, lat: Mapping[str, Any]) -> float:
    if stage_key == "anchors":
        return _safe_float(lat.get("query_preprocess_ms", 0.0), 0.0)
    if stage_key == "proposal_candidates":
        return _safe_float(
            lat.get("proposal_union_total_ms", lat.get("proposal_union_ms", lat.get("proposal_time_ms", 0.0))),
            0.0,
        )
    if stage_key == "phase1_run_seed_shortlist":
        return (
            _safe_float(lat.get("phase1_run_generation_ms", 0.0), 0.0)
            + _safe_float(lat.get("phase1_run_scoring_ms", 0.0), 0.0)
            + _safe_float(lat.get("phase1_run_shortlist_ms", 0.0), 0.0)
        )
    if stage_key == "anchor_seed_pair_shortlist":
        return (
            _safe_float(lat.get("pair_construction_ms", 0.0), 0.0)
            + _safe_float(lat.get("pair_scoring_ms", 0.0), 0.0)
            + _safe_float(lat.get("pair_shortlist_ms", 0.0), 0.0)
        )
    if stage_key == "local_corridor_candidates":
        return (
            _safe_float(lat.get("local_graph_construction_ms", 0.0), 0.0)
            + _safe_float(lat.get("ppr_time_ms", 0.0), 0.0)
            + _safe_float(lat.get("corridor_candidate_generation_ms", 0.0), 0.0)
        )
    if stage_key == "corridor_after_rerank_trim":
        return (
            _safe_float(lat.get("corridor_feature_extraction_ms", 0.0), 0.0)
            + _safe_float(lat.get("corridor_scoring_ms", 0.0), 0.0)
            + _safe_float(lat.get("corridor_shortlist_ms", 0.0), 0.0)
        )
    if stage_key == "sentence_candidates_before_text_rerank":
        return _safe_float(lat.get("final_candidate_packaging_ms", lat.get("render_context_ms", 0.0)), 0.0)
    if stage_key == "sentence_candidates_after_text_rerank":
        return _safe_float(lat.get("sentence_rerank_ms", 0.0), 0.0)
    if stage_key == "evidence_atoms_before_abr_selector":
        return 0.0
    if stage_key == "abr_selected_evidence":
        return _safe_float(lat.get("unified_acr_rcedr_ms", 0.0), 0.0)
    if stage_key == "rendered_compact_context":
        return _safe_float(lat.get("render_ms", 0.0), 0.0)
    return 0.0


def _build_stage_payloads(row: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
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
    pair_ids = list(diag.get("phase6u_pair_shortlist_unit_ids", []) or [])
    pair_texts = list(diag.get("phase6u_pair_shortlist_texts", []) or [])
    corridor_before_ids = list(diag.get("phase6u_corridor_before_trim_unit_ids", []) or [])
    corridor_before_texts = list(diag.get("phase6u_corridor_before_trim_texts", []) or [])
    corridor_after_ids = list(diag.get("phase6u_corridor_after_trim_unit_ids", []) or [])
    corridor_after_texts = list(diag.get("phase6u_corridor_after_trim_texts", []) or [])
    if not corridor_after_ids:
        corridor_after_ids, corridor_after_texts = _corridor_ids_texts(retrieval.get("corridors", []))
    if not corridor_before_ids:
        corridor_before_ids = list(corridor_after_ids)
        corridor_before_texts = list(corridor_after_texts)

    pre_rerank_ids = list(diag.get("phase6u_sentence_candidates_before_text_rerank_ids", []) or [])
    pre_rerank_texts = list(diag.get("phase6u_sentence_candidates_before_text_rerank_texts", []) or [])
    post_rerank_ids = list(diag.get("phase6u_sentence_candidates_after_text_rerank_ids", []) or [])
    post_rerank_texts = list(diag.get("phase6u_sentence_candidates_after_text_rerank_texts", []) or [])
    atoms_pre_ids = list(diag.get("phase6u_evidence_atoms_before_abr_ids", []) or [])
    atoms_pre_texts = list(diag.get("phase6u_evidence_atoms_before_abr_texts", []) or [])
    abr_ids = list(diag.get("phase6u_abr_selected_evidence_ids", []) or [])
    abr_texts = list(diag.get("phase6u_abr_selected_evidence_texts", []) or [])
    rendered_ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
    rendered_texts = list(rendered.get("sentences", []) or [])

    payloads: Dict[str, Dict[str, Any]] = {
        "anchors": {
            "ids": [],
            "texts": [],
            "candidate_count": int(len(list(retrieval.get("anchors", []) or []))),
            "sentence_candidate_count": 0,
            "latency_ms": _stage_latency_ms("anchors", lat),
            "override_sf_recall": _safe_float(funnel.get("anchor_hit_rate", 0.0), 0.0),
        },
        "proposal_candidates": {
            "ids": list(proposal_ids),
            "texts": list(proposal_texts),
            "candidate_count": _safe_int(diag.get("num_candidates_union", len(proposal_ids)), len(proposal_ids)),
            "sentence_candidate_count": int(len(list(proposal_ids))),
            "latency_ms": _stage_latency_ms("proposal_candidates", lat),
        },
        "phase1_run_seed_shortlist": {
            "ids": list(phase1_ids),
            "texts": list(phase1_texts),
            "candidate_count": _safe_int(diag.get("num_seed_candidates", len(phase1_ids)), len(phase1_ids)),
            "sentence_candidate_count": int(len(list(phase1_ids))),
            "latency_ms": _stage_latency_ms("phase1_run_seed_shortlist", lat),
        },
        "anchor_seed_pair_shortlist": {
            "ids": list(pair_ids),
            "texts": list(pair_texts),
            "candidate_count": int(len(list(diag.get("pair_shortlist", []) or []))),
            "sentence_candidate_count": int(len(list(pair_ids))),
            "latency_ms": _stage_latency_ms("anchor_seed_pair_shortlist", lat),
        },
        "local_corridor_candidates": {
            "ids": list(corridor_before_ids),
            "texts": list(corridor_before_texts),
            "candidate_count": _safe_int(diag.get("corridor_count_before_trim", len(corridor_before_ids)), len(corridor_before_ids)),
            "sentence_candidate_count": int(len(list(corridor_before_ids))),
            "latency_ms": _stage_latency_ms("local_corridor_candidates", lat),
        },
        "corridor_after_rerank_trim": {
            "ids": list(corridor_after_ids),
            "texts": list(corridor_after_texts),
            "candidate_count": _safe_int(diag.get("corridor_count_after_trim", len(corridor_after_ids)), len(corridor_after_ids)),
            "sentence_candidate_count": int(len(list(corridor_after_ids))),
            "latency_ms": _stage_latency_ms("corridor_after_rerank_trim", lat),
        },
        "sentence_candidates_before_text_rerank": {
            "ids": list(pre_rerank_ids),
            "texts": list(pre_rerank_texts),
            "candidate_count": _safe_int(diag.get("embedding_rerank_candidate_count", len(pre_rerank_ids)), len(pre_rerank_ids)),
            "sentence_candidate_count": int(len(list(pre_rerank_ids))),
            "latency_ms": _stage_latency_ms("sentence_candidates_before_text_rerank", lat),
        },
        "sentence_candidates_after_text_rerank": {
            "ids": list(post_rerank_ids),
            "texts": list(post_rerank_texts),
            "candidate_count": int(len(list(post_rerank_ids))),
            "sentence_candidate_count": int(len(list(post_rerank_ids))),
            "latency_ms": _stage_latency_ms("sentence_candidates_after_text_rerank", lat),
        },
        "evidence_atoms_before_abr_selector": {
            "ids": list(atoms_pre_ids),
            "texts": list(atoms_pre_texts),
            "candidate_count": _safe_int(unified_diag.get("num_atoms", len(atoms_pre_ids)), len(atoms_pre_ids)),
            "sentence_candidate_count": int(len(list(atoms_pre_ids))),
            "latency_ms": _stage_latency_ms("evidence_atoms_before_abr_selector", lat),
        },
        "abr_selected_evidence": {
            "ids": list(abr_ids if abr_ids else list(diag.get("selected_text_unit_ids", []) or [])),
            "texts": list(abr_texts if abr_texts else list(diag.get("selected_texts", []) or [])),
            "candidate_count": _safe_int(unified_diag.get("num_selected_atoms", len(abr_ids)), len(abr_ids)),
            "sentence_candidate_count": int(len(list(abr_ids if abr_ids else list(diag.get("selected_text_unit_ids", []) or [])))),
            "latency_ms": _stage_latency_ms("abr_selected_evidence", lat),
        },
        "rendered_compact_context": {
            "ids": list(rendered_ids),
            "texts": list(rendered_texts),
            "candidate_count": int(len(list(rendered_ids))),
            "sentence_candidate_count": int(len(list(rendered_ids))),
            "latency_ms": _stage_latency_ms("rendered_compact_context", lat),
        },
    }
    return payloads


def _largest_drop(stages: Sequence[Mapping[str, Any]]) -> Tuple[str, float, float, List[str]]:
    largest = 0.0
    largest_stage = ""
    prev_r = None
    prev_val = 0.0
    cur_val = 0.0
    missing: List[str] = []
    for stage in stages:
        r = _safe_float(stage.get("supporting_fact_recall", 0.0), 0.0)
        if prev_r is not None:
            delta = r - prev_r
            if delta < largest:
                largest = float(delta)
                largest_stage = _safe_text(stage.get("stage"))
                prev_val = float(prev_r)
                cur_val = float(r)
                missing = list(stage.get("missing_gold_sentence_ids", []) or [])
        prev_r = r
    if not largest_stage:
        largest_stage = _safe_text(stages[-1].get("stage")) if stages else ""
    return largest_stage, prev_val, cur_val, missing


def _load_run_query_path(run: Mapping[str, Any]) -> Path | None:
    query_path = _safe_text(run.get("query_path"))
    if query_path:
        p = Path(query_path)
        if p.exists():
            return p
    return None


def _build_stage_rows(report: Mapping[str, Any], dataset_filter: Sequence[str], profile_filter: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sample_maps: Dict[str, Dict[str, Any]] = {}
    allowed_datasets = set([_safe_text(x) for x in dataset_filter if _safe_text(x)])

    for run in list(report.get("completed_runs", []) or []):
        dataset = _safe_text(run.get("dataset"))
        profile = _safe_text(run.get("profile"))
        run_name = _safe_text(run.get("run_name"))
        if allowed_datasets and dataset not in allowed_datasets:
            continue
        if profile_filter and profile != profile_filter:
            continue

        qpath = _load_run_query_path(run)
        if qpath is None:
            continue
        qrows = _read_jsonl(qpath)
        if dataset not in sample_maps:
            sample_maps[dataset] = load_samples_for_dataset(dataset)
        sample_map = sample_maps[dataset]

        for qrow in qrows:
            sample_id = _safe_text(qrow.get("sample_id"))
            sample = sample_map.get(sample_id)
            graph_mode = _safe_text(
                (((qrow.get("retrieval") or {}).get("diagnostics") or {}).get("graph_mode", "current_entity_graph"))
            )
            gold_ids = _gold_ids_from_sample(sample)

            payloads = _build_stage_payloads(qrow)
            per_query_stage_rows: List[Dict[str, Any]] = []
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
                if stage_key == "anchors":
                    # Anchors are entity-level (not sentence-unit IDs); use funnel recall as best-effort signal.
                    override_r = _safe_float(payload.get("override_sf_recall", 0.0), 0.0)
                    if override_r > 0.0:
                        sf_r = float(override_r)
                        sf_p = float(override_r)
                        sf_f1 = float(override_r)
                row_item = {
                    "run_name": run_name,
                    "dataset": dataset,
                    "profile": profile,
                    "sample_id": sample_id,
                    "stage": stage_key,
                    "stage_label": stage_label,
                    "candidate_count": int(payload.get("candidate_count", len(ids))),
                    "sentence_candidate_count": int(payload.get("sentence_candidate_count", len(ids))),
                    "estimated_token_count": int(
                        token_count_for_texts(texts) if texts else int(metrics.get("tokens", 0) or 0)
                    ),
                    "supporting_fact_recall": float(sf_r),
                    "supporting_fact_precision": float(sf_p),
                    "supporting_fact_f1": float(sf_f1),
                    "matched_gold_sentence_ids": list(matched_ids),
                    "missing_gold_sentence_ids": list(missing_ids),
                    "stage_latency_ms": float(payload.get("latency_ms", 0.0)),
                }
                per_query_stage_rows.append(row_item)

            largest_stage, prev_r, cur_r, missing = _largest_drop(per_query_stage_rows)
            for item in per_query_stage_rows:
                item["largest_drop_stage"] = largest_stage
                item["largest_drop_prev_stage_recall"] = float(prev_r)
                item["largest_drop_current_stage_recall"] = float(cur_r)
                item["largest_drop_missing_gold_ids"] = list(missing)
                item["bottleneck_classification"] = DROP_CLASS_BY_STAGE.get(
                    _safe_text(item.get("largest_drop_stage")), "phase1_candidate_loss"
                )
                rows.append(item)
    return rows


def _aggregate_dataset_stage(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(_safe_text(row.get("dataset")), _safe_text(row.get("stage")))].append(row)

    out: List[Dict[str, Any]] = []
    for dataset in sorted({k[0] for k in buckets.keys()}):
        prev_recall = None
        for stage_key, _stage_label in STAGES:
            group = list(buckets.get((dataset, stage_key), []))
            if not group:
                continue
            rec = _mean([_safe_float(x.get("supporting_fact_recall", 0.0), 0.0) for x in group])
            prec = _mean([_safe_float(x.get("supporting_fact_precision", 0.0), 0.0) for x in group])
            f1 = _mean([_safe_float(x.get("supporting_fact_f1", 0.0), 0.0) for x in group])
            cand = _mean([_safe_float(x.get("candidate_count", 0.0), 0.0) for x in group])
            sent = _mean([_safe_float(x.get("sentence_candidate_count", 0.0), 0.0) for x in group])
            tok = _mean([_safe_float(x.get("estimated_token_count", 0.0), 0.0) for x in group])
            lat = _mean([_safe_float(x.get("stage_latency_ms", 0.0), 0.0) for x in group])
            delta = float(rec - prev_recall) if prev_recall is not None else 0.0
            out.append(
                {
                    "dataset": dataset,
                    "stage": stage_key,
                    "candidate_count": float(cand),
                    "sentence_candidate_count": float(sent),
                    "estimated_token_count": float(tok),
                    "supporting_fact_recall": float(rec),
                    "supporting_fact_precision": float(prec),
                    "supporting_fact_f1": float(f1),
                    "delta_from_prev": float(delta),
                    "latency_ms": float(lat),
                    "num_queries": int(len(group)),
                }
            )
            prev_recall = float(rec)
    return out


def _build_drop_examples(rows: Sequence[Mapping[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    per_query_best: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (_safe_text(row.get("dataset")), _safe_text(row.get("sample_id")))
        stage = _safe_text(row.get("stage"))
        if stage != _safe_text(row.get("largest_drop_stage")):
            continue
        prev_r = _safe_float(row.get("largest_drop_prev_stage_recall", 0.0), 0.0)
        cur_r = _safe_float(row.get("largest_drop_current_stage_recall", 0.0), 0.0)
        drop = float(cur_r - prev_r)
        cur_best = per_query_best.get(key)
        if cur_best is None or drop < _safe_float(cur_best.get("_drop", 0.0), 0.0):
            item = dict(row)
            item["_drop"] = float(drop)
            per_query_best[key] = item

    examples = sorted(per_query_best.values(), key=lambda x: _safe_float(x.get("_drop", 0.0), 0.0))[: int(limit)]
    out: List[Dict[str, Any]] = []
    for item in examples:
        out.append(
            {
                "dataset": _safe_text(item.get("dataset")),
                "sample_id": _safe_text(item.get("sample_id")),
                "largest_drop_stage": _safe_text(item.get("largest_drop_stage")),
                "prev_stage_recall": _safe_float(item.get("largest_drop_prev_stage_recall", 0.0), 0.0),
                "current_stage_recall": _safe_float(item.get("largest_drop_current_stage_recall", 0.0), 0.0),
                "missing_gold_ids": list(item.get("largest_drop_missing_gold_ids", []) or []),
                "bottleneck_classification": _safe_text(item.get("bottleneck_classification")),
            }
        )
    return out


def _mk_markdown(
    report: Mapping[str, Any],
    by_dataset_stage: Sequence[Mapping[str, Any]],
    drop_examples: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
) -> str:
    counts = dict(report.get("counts", {}) or {})
    lines: List[str] = []
    lines.append("# PHASE6U Stagewise Evidence Audit Summary")
    lines.append("")
    lines.append("## 1. Executed Runs")
    lines.append("")
    lines.append(
        f"- scheduled_runs: {int(counts.get('scheduled_runs', 0))}, "
        f"completed_run_names: {int(counts.get('completed_run_names', 0))}, "
        f"incomplete_attempts: {int(counts.get('incomplete_attempts', 0))}"
    )
    lines.append("- profile: `unified_acr_rcedr_v12`")
    lines.append("- datasets: `hotpotqa`, `2wikimultihopqa`")
    lines.append("")

    lines.append("## 2. Stagewise Table (Dataset x Stage)")
    lines.append("")
    lines.append(
        "| dataset | stage | candidate_count | SF_recall | SF_precision | SF_f1 | delta_from_prev | latency_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in by_dataset_stage:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("stage")),
                _fmt(row.get("candidate_count"), 2),
                _fmt(row.get("supporting_fact_recall"), 4),
                _fmt(row.get("supporting_fact_precision"), 4),
                _fmt(row.get("supporting_fact_f1"), 4),
                _fmt(row.get("delta_from_prev"), 4),
                _fmt(row.get("latency_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 3. Largest Drop Examples (Top 20)")
    lines.append("")
    lines.append(
        "| dataset | sample_id | largest_drop_stage | prev_stage_recall | current_stage_recall | missing_gold_ids |"
    )
    lines.append("|---|---|---|---:|---:|---|")
    for item in drop_examples:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(item.get("dataset")),
                _safe_text(item.get("sample_id")),
                _safe_text(item.get("largest_drop_stage")),
                _fmt(item.get("prev_stage_recall"), 4),
                _fmt(item.get("current_stage_recall"), 4),
                ", ".join([_safe_text(x) for x in list(item.get("missing_gold_ids", []) or [])]),
            )
        )
    lines.append("")

    lines.append("## 4. Bottleneck Classification")
    lines.append("")
    cls_counts: Dict[str, int] = defaultdict(int)
    seen = set()
    for row in stage_rows:
        qk = (_safe_text(row.get("dataset")), _safe_text(row.get("sample_id")))
        if qk in seen:
            continue
        seen.add(qk)
        cls_counts[_safe_text(row.get("bottleneck_classification"))] += 1
    for key in [
        "phase1_candidate_loss",
        "run_seed_shortlist_loss",
        "local_corridor_loss",
        "sentence_extraction_loss",
        "text_rerank_loss",
        "abr_selection_loss",
        "rendering_loss",
    ]:
        lines.append(f"- {key}: {int(cls_counts.get(key, 0))}")
    lines.append("")

    lines.append("## 5. Notes")
    lines.append("")
    lines.append(
        "- This branch is audit-only. Retrieval behavior, scoring, ranking, ABR selector, token budget, and rendering are unchanged."
    )
    lines.append(
        "- `anchors` stage is entity-level; sentence-level SF metrics use best-effort proxy from stagewise funnel recall."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PHASE6U stagewise evidence audit")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--profile", default="unified_acr_rcedr_v12")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    report = collect_phase6s_artifacts(out_root)
    datasets = [_safe_text(x) for x in _safe_text(args.datasets).replace(",", " ").split() if _safe_text(x)]
    stage_rows = _build_stage_rows(report, dataset_filter=datasets, profile_filter=_safe_text(args.profile))
    by_dataset_stage = _aggregate_dataset_stage(stage_rows)
    drop_examples = _build_drop_examples(stage_rows, limit=20)

    stagewise_query_path = out_root / "stagewise_recall_by_query.jsonl"
    stagewise_csv_path = out_root / "stagewise_recall_by_dataset.csv"
    drop_md_path = out_root / "drop_examples_top20.md"
    summary_json_path = out_root / "phase6u_stagewise_evidence_audit_summary.json"
    summary_md_path = out_root / "PHASE6U_STAGEWISE_EVIDENCE_AUDIT_SUMMARY.md"

    _write_jsonl(stagewise_query_path, stage_rows)
    _write_csv(
        stagewise_csv_path,
        by_dataset_stage,
        fieldnames=[
            "dataset",
            "stage",
            "candidate_count",
            "sentence_candidate_count",
            "estimated_token_count",
            "supporting_fact_recall",
            "supporting_fact_precision",
            "supporting_fact_f1",
            "delta_from_prev",
            "latency_ms",
            "num_queries",
        ],
    )

    drop_lines = [
        "# PHASE6U Largest Drop Examples (Top 20)",
        "",
        "| dataset | sample_id | largest_drop_stage | prev_stage_recall | current_stage_recall | missing_gold_ids | bottleneck_classification |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in drop_examples:
        drop_lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("sample_id")),
                _safe_text(row.get("largest_drop_stage")),
                _fmt(row.get("prev_stage_recall"), 4),
                _fmt(row.get("current_stage_recall"), 4),
                ", ".join([_safe_text(x) for x in list(row.get("missing_gold_ids", []) or [])]),
                _safe_text(row.get("bottleneck_classification")),
            )
        )
    drop_md_path.write_text("\n".join(drop_lines) + "\n", encoding="utf-8")

    summary_json = {
        "phase": "phase6u_stagewise_evidence_audit",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "profile": _safe_text(args.profile),
        "datasets": list(datasets),
        "artifact_counts": dict(report.get("counts", {}) or {}),
        "stagewise_rows_count": int(len(stage_rows)),
        "dataset_stage_rows_count": int(len(by_dataset_stage)),
        "drop_examples_count": int(len(drop_examples)),
        "by_dataset_stage": list(by_dataset_stage),
        "drop_examples_top20": list(drop_examples),
        "outputs": {
            "stagewise_recall_by_query_jsonl": str(stagewise_query_path),
            "stagewise_recall_by_dataset_csv": str(stagewise_csv_path),
            "drop_examples_top20_md": str(drop_md_path),
        },
    }
    _write_json(summary_json_path, summary_json)

    summary_md = _mk_markdown(report, by_dataset_stage, drop_examples, stage_rows)
    summary_md_path.write_text(summary_md, encoding="utf-8")

    print(str(summary_md_path))
    print(str(summary_json_path))
    print(str(stagewise_query_path))
    print(str(stagewise_csv_path))
    print(str(drop_md_path))


if __name__ == "__main__":
    main()
