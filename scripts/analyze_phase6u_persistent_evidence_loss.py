#!/usr/bin/env python3
"""PHASE6U-1 Persistent Evidence Loss Attribution (post-processing only)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from validate_phase6s_artifacts import collect_phase6s_artifacts
except ModuleNotFoundError:  # pragma: no cover - import path differs under pytest module import
    from scripts.validate_phase6s_artifacts import collect_phase6s_artifacts


CANONICAL_STAGE_ORDER: List[str] = [
    "anchors",
    "proposal_candidates",
    "phase1_run_seed_shortlist",
    "anchor_seed_pair_shortlist",
    "local_corridor_candidates",
    "corridor_after_rerank_trim",
    "sentence_candidates_before_text_rerank",
    "sentence_candidates_after_text_rerank",
    "evidence_atoms_before_abr",
    "abr_selected_evidence",
    "rendered_compact_context",
]

ALIASES: Dict[str, str] = {
    "proposal": "proposal_candidates",
    "proposal_candidates": "proposal_candidates",
    "phase1_shortlist": "phase1_run_seed_shortlist",
    "phase1_run_seed_shortlist": "phase1_run_seed_shortlist",
    "pair_shortlist": "anchor_seed_pair_shortlist",
    "anchor_seed_pair_shortlist": "anchor_seed_pair_shortlist",
    "local_corridor": "local_corridor_candidates",
    "local_corridor_candidates": "local_corridor_candidates",
    "trimmed_corridor": "corridor_after_rerank_trim",
    "corridor_after_rerank_trim": "corridor_after_rerank_trim",
    "sentence_before_rerank": "sentence_candidates_before_text_rerank",
    "sentence_candidates_before_text_rerank": "sentence_candidates_before_text_rerank",
    "sentence_after_rerank": "sentence_candidates_after_text_rerank",
    "sentence_candidates_after_text_rerank": "sentence_candidates_after_text_rerank",
    "atoms": "evidence_atoms_before_abr",
    "evidence_atoms_before_abr": "evidence_atoms_before_abr",
    "evidence_atoms_before_abr_selector": "evidence_atoms_before_abr",
    "abr_selected": "abr_selected_evidence",
    "abr_selected_evidence": "abr_selected_evidence",
    "rendered": "rendered_compact_context",
    "rendered_compact_context": "rendered_compact_context",
}

TRANSIENT_CLASS = "pair_shortlist_transient_loss"
RETAINED_CLASS = "retained_to_rendered"

LOSS_CLASSES: List[str] = [
    "never_retrieved",
    "phase1_proposal_loss",
    "run_seed_shortlist_persistent_loss",
    "pair_shortlist_transient_loss",
    "local_corridor_persistent_loss",
    "corridor_rerank_trim_persistent_loss",
    "sentence_extraction_persistent_loss",
    "text_rerank_persistent_loss",
    "abr_selection_persistent_loss",
    "rendering_persistent_loss",
    "retained_to_rendered",
]

PERSISTENT_CLASSES = {
    "phase1_proposal_loss",
    "run_seed_shortlist_persistent_loss",
    "local_corridor_persistent_loss",
    "corridor_rerank_trim_persistent_loss",
    "sentence_extraction_persistent_loss",
    "text_rerank_persistent_loss",
    "abr_selection_persistent_loss",
    "rendering_persistent_loss",
}


@dataclass
class SupportTrace:
    dataset: str
    profile: str
    sample_id: str
    gold_id: str
    stage_presence: Dict[str, bool]
    first_seen_stage: str
    first_lost_stage: str
    recovered_after_loss: bool
    final_present: bool
    final_missing: bool
    persistent_loss_stage: str
    transient_loss_stages: List[str]
    loss_class: str
    question: str
    answer: str


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


def normalize_stage_name(stage: str) -> str:
    s = _safe_text(stage)
    return ALIASES.get(s, s)


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
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _find_input_stagewise_jsonl(input_root: Path) -> Path:
    candidates = [
        input_root / "stagewise_recall_by_query.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    discovered = sorted(input_root.rglob("stagewise_recall_by_query.jsonl"))
    if discovered:
        return discovered[-1]
    raise FileNotFoundError(f"stagewise_recall_by_query.jsonl not found under {input_root}")


def _load_query_meta(input_root: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    report = collect_phase6s_artifacts(input_root)
    meta: Dict[Tuple[str, str], Dict[str, str]] = {}
    for run in list(report.get("completed_runs", []) or []):
        dataset = _safe_text(run.get("dataset"))
        qpath = _safe_text(run.get("query_path"))
        if not qpath:
            continue
        path = Path(qpath)
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            sample_id = _safe_text(row.get("sample_id"))
            if not sample_id:
                continue
            key = (dataset, sample_id)
            if key not in meta:
                meta[key] = {
                    "question": _safe_text(row.get("question")),
                    "answer": _safe_text(row.get("answer")),
                }
    return meta


def _stage_presence_for_gold(
    stage_rows_by_name: Mapping[str, Mapping[str, Any]],
    gold_id: str,
) -> Dict[str, bool]:
    presence: Dict[str, bool] = {}
    gid = _safe_text(gold_id)
    for stage in CANONICAL_STAGE_ORDER:
        row = dict(stage_rows_by_name.get(stage, {}) or {})
        matched = set([_safe_text(x) for x in list(row.get("matched_gold_sentence_ids", []) or []) if _safe_text(x)])
        presence[stage] = bool(gid in matched)
    return presence


def _transitions(presence: Mapping[str, bool]) -> List[Tuple[str, str, bool, bool]]:
    out: List[Tuple[str, str, bool, bool]] = []
    for i in range(1, len(CANONICAL_STAGE_ORDER)):
        prev_stage = CANONICAL_STAGE_ORDER[i - 1]
        cur_stage = CANONICAL_STAGE_ORDER[i]
        out.append((prev_stage, cur_stage, bool(presence.get(prev_stage, False)), bool(presence.get(cur_stage, False))))
    return out


def _first_seen_stage(presence: Mapping[str, bool]) -> str:
    for stage in CANONICAL_STAGE_ORDER:
        if bool(presence.get(stage, False)):
            return stage
    return ""


def _first_lost_stage_and_recovery(presence: Mapping[str, bool]) -> Tuple[str, bool, List[str]]:
    first_lost = ""
    recovered_after_loss = False
    transient_stages: List[str] = []
    for i, (_prev, cur, prev_v, cur_v) in enumerate(_transitions(presence)):
        if prev_v and (not cur_v):
            later_stages = CANONICAL_STAGE_ORDER[i + 1 :]
            recovered = any(bool(presence.get(s, False)) for s in later_stages)
            if not first_lost:
                first_lost = cur
                recovered_after_loss = bool(recovered)
            if recovered:
                transient_stages.append(cur)
    return first_lost, recovered_after_loss, transient_stages


def _persistent_loss_stage(presence: Mapping[str, bool]) -> str:
    if bool(presence.get("rendered_compact_context", False)):
        return ""
    first_seen = _first_seen_stage(presence)
    if not first_seen:
        return ""
    start = CANONICAL_STAGE_ORDER.index(first_seen)
    for idx in range(start + 1, len(CANONICAL_STAGE_ORDER)):
        stage = CANONICAL_STAGE_ORDER[idx]
        if bool(presence.get(stage, False)):
            continue
        later = CANONICAL_STAGE_ORDER[idx:]
        if not any(bool(presence.get(s, False)) for s in later):
            return stage
    return ""


def classify_support_presence(presence: Mapping[str, bool]) -> str:
    final_present = bool(presence.get("rendered_compact_context", False))
    any_present = any(bool(presence.get(s, False)) for s in CANONICAL_STAGE_ORDER)

    pair_transient = (
        bool(presence.get("phase1_run_seed_shortlist", False))
        and (not bool(presence.get("anchor_seed_pair_shortlist", False)))
        and any(bool(presence.get(s, False)) for s in CANONICAL_STAGE_ORDER[4:])
    )
    if final_present:
        if pair_transient:
            return TRANSIENT_CLASS
        return RETAINED_CLASS

    if not any_present:
        return "never_retrieved"

    if bool(presence.get("abr_selected_evidence", False)) and (not final_present):
        return "rendering_persistent_loss"

    if (
        bool(presence.get("sentence_candidates_after_text_rerank", False))
        or bool(presence.get("evidence_atoms_before_abr", False))
    ) and (not bool(presence.get("abr_selected_evidence", False))):
        return "abr_selection_persistent_loss"

    if bool(presence.get("sentence_candidates_before_text_rerank", False)) and (
        not bool(presence.get("sentence_candidates_after_text_rerank", False))
    ):
        return "text_rerank_persistent_loss"

    if (
        bool(presence.get("corridor_after_rerank_trim", False))
        or bool(presence.get("local_corridor_candidates", False))
    ) and (not bool(presence.get("sentence_candidates_before_text_rerank", False))):
        return "sentence_extraction_persistent_loss"

    if bool(presence.get("local_corridor_candidates", False)) and (
        not bool(presence.get("corridor_after_rerank_trim", False))
    ):
        return "corridor_rerank_trim_persistent_loss"

    if bool(presence.get("proposal_candidates", False)) and (
        not bool(presence.get("phase1_run_seed_shortlist", False))
    ):
        return "run_seed_shortlist_persistent_loss"

    if bool(presence.get("phase1_run_seed_shortlist", False)) and (
        not bool(presence.get("local_corridor_candidates", False))
    ):
        return "local_corridor_persistent_loss"

    if bool(presence.get("anchors", False)) and (not bool(presence.get("proposal_candidates", False))):
        return "phase1_proposal_loss"

    if pair_transient:
        return TRANSIENT_CLASS

    return "never_retrieved"


def _build_support_traces(
    rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str],
    profile: str,
    query_meta: Mapping[Tuple[str, str], Mapping[str, str]],
) -> Tuple[List[SupportTrace], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    allowed_datasets = set([_safe_text(x) for x in datasets if _safe_text(x)])

    for row in rows:
        dataset = _safe_text(row.get("dataset"))
        sample_id = _safe_text(row.get("sample_id"))
        row_profile = _safe_text(row.get("profile"))
        if allowed_datasets and dataset not in allowed_datasets:
            continue
        if profile and row_profile != profile:
            continue
        stage = normalize_stage_name(_safe_text(row.get("stage")))
        if stage not in CANONICAL_STAGE_ORDER:
            continue
        grouped[(dataset, row_profile, sample_id)][stage] = dict(row)

    support_traces: List[SupportTrace] = []
    query_stage_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for (dataset, row_profile, sample_id), stage_rows in grouped.items():
        # Fill missing stages as empty rows.
        for stage in CANONICAL_STAGE_ORDER:
            if stage not in stage_rows:
                stage_rows[stage] = {
                    "dataset": dataset,
                    "profile": row_profile,
                    "sample_id": sample_id,
                    "stage": stage,
                    "matched_gold_sentence_ids": [],
                    "missing_gold_sentence_ids": [],
                    "supporting_fact_recall": 0.0,
                }
            query_stage_rows[(dataset, sample_id)].append(dict(stage_rows[stage]))

        gold_ids: List[str] = []
        for stage in CANONICAL_STAGE_ORDER:
            r = dict(stage_rows.get(stage, {}) or {})
            gold_ids.extend([_safe_text(x) for x in list(r.get("matched_gold_sentence_ids", []) or []) if _safe_text(x)])
            gold_ids.extend([_safe_text(x) for x in list(r.get("missing_gold_sentence_ids", []) or []) if _safe_text(x)])
        gold_ids = sorted(set(gold_ids))

        question = _safe_text((query_meta.get((dataset, sample_id), {}) or {}).get("question"))
        answer = _safe_text((query_meta.get((dataset, sample_id), {}) or {}).get("answer"))

        for gold_id in gold_ids:
            presence = _stage_presence_for_gold(stage_rows, gold_id)
            first_seen = _first_seen_stage(presence)
            first_lost, recovered_after_loss, transient_stages = _first_lost_stage_and_recovery(presence)
            final_present = bool(presence.get("rendered_compact_context", False))
            persistent_stage = _persistent_loss_stage(presence)
            loss_class = classify_support_presence(presence)
            support_traces.append(
                SupportTrace(
                    dataset=dataset,
                    profile=row_profile,
                    sample_id=sample_id,
                    gold_id=gold_id,
                    stage_presence=dict(presence),
                    first_seen_stage=first_seen,
                    first_lost_stage=first_lost,
                    recovered_after_loss=bool(recovered_after_loss),
                    final_present=bool(final_present),
                    final_missing=bool(not final_present),
                    persistent_loss_stage=persistent_stage,
                    transient_loss_stages=list(transient_stages),
                    loss_class=loss_class,
                    question=question,
                    answer=answer,
                )
            )
    return support_traces, query_stage_rows


def _support_trace_to_row(trace: SupportTrace) -> Dict[str, Any]:
    row = {
        "dataset": trace.dataset,
        "profile": trace.profile,
        "sample_id": trace.sample_id,
        "question": trace.question,
        "answer": trace.answer,
        "gold_id": trace.gold_id,
        "loss_class": trace.loss_class,
        "first_seen_stage": trace.first_seen_stage,
        "first_lost_stage": trace.first_lost_stage,
        "recovered_after_loss": trace.recovered_after_loss,
        "final_present": trace.final_present,
        "final_missing": trace.final_missing,
        "persistent_loss_stage": trace.persistent_loss_stage,
        "transient_loss_stages": list(trace.transient_loss_stages),
    }
    for stage in CANONICAL_STAGE_ORDER:
        row[f"present_{stage}"] = bool(trace.stage_presence.get(stage, False))
    return row


def _build_query_rows(traces: Sequence[SupportTrace]) -> List[Dict[str, Any]]:
    by_query: Dict[Tuple[str, str, str], List[SupportTrace]] = defaultdict(list)
    for t in traces:
        by_query[(t.dataset, t.profile, t.sample_id)].append(t)

    out: List[Dict[str, Any]] = []
    for (dataset, profile, sample_id), group in sorted(by_query.items()):
        gold_total = len(group)
        rendered_hit = sum(1 for t in group if t.final_present)
        rendered_recall = float(rendered_hit / gold_total) if gold_total else 0.0
        transient_count = sum(1 for t in group if t.loss_class == TRANSIENT_CLASS or len(t.transient_loss_stages) > 0)
        persistent_group = [t for t in group if t.loss_class in PERSISTENT_CLASSES]
        persistent_count = len(persistent_group)
        class_counts = Counter([t.loss_class for t in persistent_group])
        primary = class_counts.most_common(1)[0][0] if class_counts else (TRANSIENT_CLASS if transient_count > 0 else RETAINED_CLASS)

        stage_counts = Counter([_safe_text(t.persistent_loss_stage) for t in persistent_group if _safe_text(t.persistent_loss_stage)])
        largest_stage = stage_counts.most_common(1)[0][0] if stage_counts else ""

        question = _safe_text(group[0].question if group else "")
        answer = _safe_text(group[0].answer if group else "")

        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "sample_id": sample_id,
                "question": question,
                "answer": answer,
                "gold_total": int(gold_total),
                "rendered_hit_count": int(rendered_hit),
                "rendered_recall": float(rendered_recall),
                "num_persistent_losses": int(persistent_count),
                "num_transient_losses": int(transient_count),
                "primary_persistent_loss_class": primary,
                "largest_persistent_loss_stage": largest_stage,
                "transient_pair_loss_count": int(sum(1 for t in group if t.loss_class == TRANSIENT_CLASS)),
                "abr_loss_count": int(sum(1 for t in group if t.loss_class == "abr_selection_persistent_loss")),
                "phase1_loss_count": int(
                    sum(1 for t in group if t.loss_class in {"phase1_proposal_loss", "run_seed_shortlist_persistent_loss"})
                ),
            }
        )
    return out


def _build_support_class_distribution(traces: Sequence[SupportTrace]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[SupportTrace]] = defaultdict(list)
    for t in traces:
        by_dataset[t.dataset].append(t)
    for dataset in sorted(by_dataset.keys()):
        group = by_dataset[dataset]
        total = len(group)
        counts = Counter([t.loss_class for t in group])
        for cls in LOSS_CLASSES:
            count = int(counts.get(cls, 0))
            share = float(count / total) if total else 0.0
            out.append({"dataset": dataset, "loss_class": cls, "count": count, "share": share})
    return out


def _build_query_bottleneck_distribution(query_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_dataset[_safe_text(row.get("dataset"))].append(row)
    for dataset in sorted(by_dataset.keys()):
        group = by_dataset[dataset]
        total = len(group)
        counts = Counter([_safe_text(r.get("primary_persistent_loss_class")) for r in group])
        for cls, count in sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0])):
            out.append(
                {
                    "dataset": dataset,
                    "primary_bottleneck": cls,
                    "query_count": int(count),
                    "share": float(count / total) if total else 0.0,
                }
            )
    return out


def _build_stage_transition_recovery(traces: Sequence[SupportTrace]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[SupportTrace]] = defaultdict(list)
    for t in traces:
        by_dataset[t.dataset].append(t)

    for dataset in sorted(by_dataset.keys()):
        group = by_dataset[dataset]
        raw_drop = Counter()
        recovered = Counter()
        persistent = Counter()
        for t in group:
            presence = t.stage_presence
            for i, (_prev_stage, cur_stage, prev_v, cur_v) in enumerate(_transitions(presence)):
                if prev_v and (not cur_v):
                    raw_drop[cur_stage] += 1
                    later_stages = CANONICAL_STAGE_ORDER[i + 1 :]
                    if any(bool(presence.get(s, False)) for s in later_stages):
                        recovered[cur_stage] += 1
                    else:
                        persistent[cur_stage] += 1
        for stage in CANONICAL_STAGE_ORDER[1:]:
            r = int(raw_drop.get(stage, 0))
            rec = int(recovered.get(stage, 0))
            per = int(persistent.get(stage, 0))
            rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "raw_drop_count": r,
                    "recovered_later_count": rec,
                    "persistent_drop_count": per,
                    "recovery_rate_after_drop": float(rec / r) if r > 0 else 0.0,
                }
            )
    return rows


def _build_stagewise_avg_with_persistence(
    stage_rows: Sequence[Mapping[str, Any]],
    traces: Sequence[SupportTrace],
    transition_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_ds_stage: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        stage = normalize_stage_name(_safe_text(row.get("stage")))
        if stage not in CANONICAL_STAGE_ORDER:
            continue
        by_ds_stage[(_safe_text(row.get("dataset")), stage)].append(row)

    total_support_by_dataset: Dict[str, int] = Counter([t.dataset for t in traces])
    transition_map: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in transition_rows:
        transition_map[(_safe_text(row.get("dataset")), _safe_text(row.get("stage")))] = row

    for dataset in sorted({k[0] for k in by_ds_stage.keys()}):
        for stage in CANONICAL_STAGE_ORDER:
            group = by_ds_stage.get((dataset, stage), [])
            if not group:
                continue
            avg_recall = _mean([_safe_float(x.get("supporting_fact_recall", 0.0), 0.0) for x in group])
            avg_prec = _mean([_safe_float(x.get("supporting_fact_precision", 0.0), 0.0) for x in group])
            avg_f1 = _mean([_safe_float(x.get("supporting_fact_f1", 0.0), 0.0) for x in group])
            transition = dict(transition_map.get((dataset, stage), {}) or {})
            persistent_drop = _safe_int(transition.get("persistent_drop_count", 0), 0)
            raw_drop = _safe_int(transition.get("raw_drop_count", 0), 0)
            support_total = int(total_support_by_dataset.get(dataset, 0))
            out.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "supporting_fact_recall_avg": float(avg_recall),
                    "supporting_fact_precision_avg": float(avg_prec),
                    "supporting_fact_f1_avg": float(avg_f1),
                    "persistent_loss_after_stage": float(persistent_drop / support_total) if support_total > 0 else 0.0,
                    "recovery_rate_after_drop": float(
                        _safe_float(transition.get("recovery_rate_after_drop", 0.0), 0.0)
                    )
                    if raw_drop > 0
                    else 0.0,
                }
            )
    return out


def _pick_examples(
    traces: Sequence[SupportTrace],
    *,
    target_classes: Sequence[str],
    top_k: int,
) -> List[SupportTrace]:
    subset = [t for t in traces if t.loss_class in set(target_classes)]
    subset.sort(
        key=lambda t: (
            t.dataset,
            t.sample_id,
            0 if t.loss_class in PERSISTENT_CLASSES else 1,
            CANONICAL_STAGE_ORDER.index(t.persistent_loss_stage)
            if t.persistent_loss_stage in CANONICAL_STAGE_ORDER
            else 999,
            t.gold_id,
        )
    )
    return subset[: max(0, int(top_k))]


def _examples_markdown(title: str, traces: Sequence[SupportTrace]) -> str:
    lines = [f"# {title}", "", "| dataset | sample_id | gold_id | loss_class | first_seen | first_lost | persistent_stage | transient_stages | final_present | question |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in traces:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                t.dataset,
                t.sample_id,
                t.gold_id,
                t.loss_class,
                t.first_seen_stage,
                t.first_lost_stage,
                t.persistent_loss_stage,
                ",".join(t.transient_loss_stages),
                str(t.final_present).lower(),
                t.question.replace("|", " "),
            )
        )
    return "\n".join(lines) + "\n"


def _summary_markdown(
    input_root: Path,
    output_root: Path,
    support_dist: Sequence[Mapping[str, Any]],
    query_dist: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    stagewise_aug: Sequence[Mapping[str, Any]],
    abr_examples: Sequence[SupportTrace],
    phase1_examples: Sequence[SupportTrace],
    transient_examples: Sequence[SupportTrace],
) -> str:
    lines: List[str] = []
    lines.append("# PHASE6U Persistent Evidence Loss Attribution")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This analysis separates transient stage drops from persistent final evidence loss without changing retrieval behavior."
    )
    lines.append(f"- input_root: `{input_root}`")
    lines.append(f"- output_root: `{output_root}`")
    lines.append("")

    lines.append("## 2. Dataset-Level Persistent Loss Distribution")
    lines.append("")
    lines.append("| dataset | loss_class | count | share |")
    lines.append("|---|---|---:|---:|")
    for row in support_dist:
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("loss_class")),
                int(row.get("count", 0)),
                _fmt(row.get("share"), 4),
            )
        )
    lines.append("")

    lines.append("## 3. Query-Level Bottleneck Distribution")
    lines.append("")
    lines.append("| dataset | primary_bottleneck | query_count | share |")
    lines.append("|---|---|---:|---:|")
    for row in query_dist:
        lines.append(
            "| {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("primary_bottleneck")),
                int(row.get("query_count", 0)),
                _fmt(row.get("share"), 4),
            )
        )
    lines.append("")

    lines.append("## 4. Stage Transition Recovery")
    lines.append("")
    lines.append("| dataset | stage | raw_drop_count | recovered_later_count | persistent_drop_count | recovery_rate_after_drop |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in transitions:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("stage")),
                int(row.get("raw_drop_count", 0)),
                int(row.get("recovered_later_count", 0)),
                int(row.get("persistent_drop_count", 0)),
                _fmt(row.get("recovery_rate_after_drop"), 4),
            )
        )
    lines.append("")

    lines.append("## 5. ABR Selection Loss Examples")
    lines.append("")
    for t in abr_examples[:10]:
        lines.append(
            f"- {t.dataset} / {t.sample_id} / {t.gold_id} "
            f"(first_seen={t.first_seen_stage}, first_lost={t.first_lost_stage}, persistent={t.persistent_loss_stage})"
        )
    lines.append("")

    lines.append("## 6. Phase-1 Persistent Loss Examples")
    lines.append("")
    for t in phase1_examples[:10]:
        lines.append(
            f"- {t.dataset} / {t.sample_id} / {t.gold_id} "
            f"(class={t.loss_class}, first_seen={t.first_seen_stage}, persistent={t.persistent_loss_stage})"
        )
    lines.append("")

    lines.append("## 7. Transient Pair-Shortlist Drops")
    lines.append("")
    for t in transient_examples[:10]:
        lines.append(
            f"- {t.dataset} / {t.sample_id} / {t.gold_id} "
            f"(transient_stages={','.join(t.transient_loss_stages)}, final_present={str(t.final_present).lower()})"
        )
    lines.append("")

    lines.append("## 8. Interpretation")
    lines.append("")
    lines.append(
        "Use persistent-drop-heavy stages as the next improvement target; treat pair-shortlist-only drops with later recovery as transient."
    )
    lines.append("")

    lines.append("## 9. Recommendation")
    lines.append("")
    # Simple heuristic recommendation
    class_totals = Counter()
    for row in support_dist:
        cls = _safe_text(row.get("loss_class"))
        class_totals[cls] += int(row.get("count", 0))
    dominant = class_totals.most_common(1)[0][0] if class_totals else "unknown"
    if dominant == "abr_selection_persistent_loss":
        rec = "Target ABR selector persistent loss next (no code change in this step)."
    elif dominant in {"run_seed_shortlist_persistent_loss", "phase1_proposal_loss"}:
        rec = "Target Phase-1 shortlist persistent loss next (no code change in this step)."
    elif dominant == "rendering_persistent_loss":
        rec = "Target rendering-stage loss next (no code change in this step)."
    else:
        rec = "Dominant persistent-loss class is mixed; inspect transition recovery and top examples before implementation."
    lines.append(f"- dominant_support_loss_class: `{dominant}`")
    lines.append(f"- recommendation: {rec}")
    lines.append("")

    lines.append("## Appendix: Stagewise Average Recall with Persistence Overlay")
    lines.append("")
    lines.append("| dataset | stage | SF_recall_avg | SF_precision_avg | SF_f1_avg | persistent_loss_after_stage | recovery_rate_after_drop |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in stagewise_aug:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                _safe_text(row.get("dataset")),
                _safe_text(row.get("stage")),
                _fmt(row.get("supporting_fact_recall_avg"), 4),
                _fmt(row.get("supporting_fact_precision_avg"), 4),
                _fmt(row.get("supporting_fact_f1_avg"), 4),
                _fmt(row.get("persistent_loss_after_stage"), 4),
                _fmt(row.get("recovery_rate_after_drop"), 4),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_analysis(
    *,
    input_root: Path,
    output_root: Path,
    datasets: Sequence[str],
    profile: str,
    top_k: int,
) -> Dict[str, Any]:
    stagewise_path = _find_input_stagewise_jsonl(input_root)
    rows = _read_jsonl(stagewise_path)
    query_meta = _load_query_meta(input_root)

    traces, query_stage_rows = _build_support_traces(
        rows,
        datasets=datasets,
        profile=profile,
        query_meta=query_meta,
    )
    support_rows = [_support_trace_to_row(t) for t in traces]
    query_rows = _build_query_rows(traces)

    support_dist = _build_support_class_distribution(traces)
    query_dist = _build_query_bottleneck_distribution(query_rows)
    transition_rows = _build_stage_transition_recovery(traces)

    # Flatten all stage rows for stagewise average overlays.
    flat_stage_rows: List[Mapping[str, Any]] = []
    for stage_list in query_stage_rows.values():
        flat_stage_rows.extend(stage_list)
    stagewise_aug = _build_stagewise_avg_with_persistence(flat_stage_rows, traces, transition_rows)

    abr_examples = _pick_examples(
        traces,
        target_classes=["abr_selection_persistent_loss"],
        top_k=top_k,
    )
    phase1_examples = _pick_examples(
        traces,
        target_classes=["phase1_proposal_loss", "run_seed_shortlist_persistent_loss"],
        top_k=top_k,
    )
    transient_examples = _pick_examples(
        traces,
        target_classes=[TRANSIENT_CLASS],
        top_k=top_k,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "persistent_loss_by_support_id.jsonl", support_rows)
    _write_jsonl(output_root / "persistent_loss_by_query.jsonl", query_rows)
    _write_csv(
        output_root / "persistent_loss_by_dataset.csv",
        support_dist,
        fieldnames=["dataset", "loss_class", "count", "share"],
    )
    _write_csv(
        output_root / "stage_transition_recovery.csv",
        transition_rows,
        fieldnames=[
            "dataset",
            "stage",
            "raw_drop_count",
            "recovered_later_count",
            "persistent_drop_count",
            "recovery_rate_after_drop",
        ],
    )

    (output_root / "abr_loss_examples_top20.md").write_text(
        _examples_markdown("PHASE6U ABR Persistent Loss Examples", abr_examples),
        encoding="utf-8",
    )
    (output_root / "phase1_persistent_loss_examples_top20.md").write_text(
        _examples_markdown("PHASE6U Phase1 Persistent Loss Examples", phase1_examples),
        encoding="utf-8",
    )
    (output_root / "transient_pair_loss_examples_top20.md").write_text(
        _examples_markdown("PHASE6U Transient Pair-Shortlist Loss Examples", transient_examples),
        encoding="utf-8",
    )

    summary_md = _summary_markdown(
        input_root=input_root,
        output_root=output_root,
        support_dist=support_dist,
        query_dist=query_dist,
        transitions=transition_rows,
        stagewise_aug=stagewise_aug,
        abr_examples=abr_examples,
        phase1_examples=phase1_examples,
        transient_examples=transient_examples,
    )
    (output_root / "PHASE6U_PERSISTENT_EVIDENCE_LOSS_SUMMARY.md").write_text(summary_md, encoding="utf-8")

    summary_json = {
        "phase": "phase6u_1_persistent_evidence_loss_attribution",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "datasets": list(datasets),
        "profile": profile,
        "stagewise_source": str(stagewise_path),
        "support_rows_count": int(len(support_rows)),
        "query_rows_count": int(len(query_rows)),
        "support_loss_distribution": support_dist,
        "query_bottleneck_distribution": query_dist,
        "stage_transition_recovery": transition_rows,
        "stagewise_avg_with_persistence": stagewise_aug,
        "top_examples": {
            "abr_loss_examples": [_support_trace_to_row(t) for t in abr_examples],
            "phase1_persistent_loss_examples": [_support_trace_to_row(t) for t in phase1_examples],
            "transient_pair_loss_examples": [_support_trace_to_row(t) for t in transient_examples],
        },
        "outputs": {
            "summary_md": str(output_root / "PHASE6U_PERSISTENT_EVIDENCE_LOSS_SUMMARY.md"),
            "summary_json": str(output_root / "phase6u_persistent_evidence_loss_summary.json"),
            "persistent_loss_by_support_id_jsonl": str(output_root / "persistent_loss_by_support_id.jsonl"),
            "persistent_loss_by_query_jsonl": str(output_root / "persistent_loss_by_query.jsonl"),
            "persistent_loss_by_dataset_csv": str(output_root / "persistent_loss_by_dataset.csv"),
            "stage_transition_recovery_csv": str(output_root / "stage_transition_recovery.csv"),
            "abr_loss_examples_md": str(output_root / "abr_loss_examples_top20.md"),
            "phase1_persistent_loss_examples_md": str(output_root / "phase1_persistent_loss_examples_top20.md"),
            "transient_pair_loss_examples_md": str(output_root / "transient_pair_loss_examples_top20.md"),
        },
    }
    _write_json(output_root / "phase6u_persistent_evidence_loss_summary.json", summary_json)
    return summary_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze PHASE6U persistent evidence loss")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--profile", default="unified_acr_rcedr_v12")
    ap.add_argument("--top-k", type=int, default=20)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    datasets = [_safe_text(x) for x in _safe_text(args.datasets).replace(",", " ").split() if _safe_text(x)]
    summary = run_analysis(
        input_root=input_root,
        output_root=output_root,
        datasets=datasets,
        profile=_safe_text(args.profile),
        top_k=max(1, int(args.top_k)),
    )
    print(summary["outputs"]["summary_md"])
    print(summary["outputs"]["summary_json"])
    print(summary["outputs"]["persistent_loss_by_support_id_jsonl"])
    print(summary["outputs"]["persistent_loss_by_query_jsonl"])
    print(summary["outputs"]["persistent_loss_by_dataset_csv"])
    print(summary["outputs"]["stage_transition_recovery_csv"])
    print(summary["outputs"]["abr_loss_examples_md"])
    print(summary["outputs"]["phase1_persistent_loss_examples_md"])
    print(summary["outputs"]["transient_pair_loss_examples_md"])


if __name__ == "__main__":
    main()
