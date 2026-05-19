#!/usr/bin/env python3
"""Summarize PHASE6X early-collapse recovery paired run."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts

from effirag.evidence_flow_audit import load_samples_for_dataset, stage_metrics
from summarize_phase6u_stagewise_evidence_audit import _build_stage_payloads


STAGE_KEYS = [
    ("S1_proposal", "proposal_candidates"),
    ("S2_phase1", "phase1_run_seed_shortlist"),
    ("S3_local_refinement", "local_corridor_candidates"),
    ("S4_sentence", "sentence_candidates_after_text_rerank"),
    ("S5_selected", "abr_selected_evidence"),
    ("S6_rendered", "rendered_compact_context"),
]


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
    for x in values:
        s = _safe_text(x)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(_ordered_unique(a))
    sb = set(_ordered_unique(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa.intersection(sb)) / float(len(sa.union(sb)))
    )


def _normalize_for_match(text: str) -> str:
    return " ".join(str(text or "").strip().split()).lower()


def _normalize_sentence_support_id(sentence_id: str) -> str:
    sid = _safe_text(sentence_id)
    if not sid:
        return ""
    parts = sid.split("::")
    if len(parts) >= 3 and parts[0] in {"chunk", "sentence"}:
        return f"{parts[1]}::{parts[2]}"
    return sid


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


def _answer_metrics(sample: Any, texts: Sequence[str]) -> Dict[str, float]:
    aliases = _answer_aliases(sample)
    norm_aliases = [_normalize_for_match(x) for x in aliases if _normalize_for_match(x)]
    sent_norm = [_normalize_for_match(x) for x in list(texts or [])]
    hit_indices: List[int] = []
    if norm_aliases:
        for i, s in enumerate(sent_norm):
            if any(alias in s for alias in norm_aliases):
                hit_indices.append(i)
    hit = 1.0 if hit_indices else 0.0
    rank = float(hit_indices[0]) if hit_indices else -1.0
    count = float(len(hit_indices))
    density = float(count / float(max(1, len(sent_norm)))) if sent_norm else 0.0
    return {
        "answer_hit": hit,
        "answer_rank": rank,
        "answer_count": count,
        "answer_density": density,
    }


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


@dataclass
class RunData:
    run_name: str
    dataset: str
    profile: str
    role: str
    summary: Dict[str, Any]
    query_rows: List[Dict[str, Any]]


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
        "answer_string_hit": _safe_float(summary.get("answer_surface_present", 0.0), 0.0),
        "answer_bearing_density": _safe_float(summary.get("answer_surface_token_density", 0.0), 0.0),
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
    }


def _query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    m = dict(row.get("metrics", {}) or {})
    e = dict(row.get("efficiency", {}) or {})
    g = dict(row.get("generation_diagnostics", {}) or {})
    return {
        "f1": _safe_float(m.get("f1", 0.0), 0.0),
        "em": _safe_float(m.get("em", 0.0), 0.0),
        "sf_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
        "sf_precision": _safe_float(m.get("supporting_fact_precision", 0.0), 0.0),
        "retrieval_ms": _safe_float(e.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(e.get("generation_latency_ms", 0.0), 0.0),
        "total_ms": _safe_float(e.get("total_latency_ms", 0.0), 0.0),
        "prompt_tokens": _safe_float(g.get("prompt_tokens", 0.0), 0.0),
        "answer_string_hit": _safe_float(m.get("answer_surface_present", 0.0), 0.0),
        "answer_bearing_density": _safe_float(m.get("answer_surface_token_density", 0.0), 0.0),
    }


def _extract_selected_ids(row: Mapping[str, Any]) -> List[str]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    ids = list(row.get("retrieval_selected_sentence_ids", []) or retrieval.get("selected_sentence_ids", []) or [])
    return [_normalize_sentence_support_id(x) for x in _ordered_unique(ids)]


def _extract_rendered_ids(row: Mapping[str, Any]) -> List[str]:
    rendered = dict((row.get("rendered", {}) or {}))
    ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
    return [_normalize_sentence_support_id(x) for x in _ordered_unique(ids)]


def _load_runs(out_root: Path, datasets: Sequence[str], profiles: Sequence[str]) -> Tuple[Dict[str, Any], List[RunData]]:
    report = collect_phase6s_artifacts(out_root)
    scheduled = {_safe_text(r.get("run_name")): dict(r) for r in list(report.get("scheduled_runs", []) or [])}
    dataset_set = set(datasets)
    profile_set = set(profiles)
    runs: List[RunData] = []
    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        meta = scheduled.get(run_name, {})
        dataset = _safe_text(meta.get("dataset", item.get("dataset", "")))
        profile = _safe_text(meta.get("profile", item.get("profile", "")))
        role = _safe_text(meta.get("role", ""))
        if dataset not in dataset_set or profile not in profile_set:
            continue
        qpath = Path(_safe_text(item.get("query_path"))).resolve()
        qrows = _read_jsonl(qpath)
        runs.append(
            RunData(
                run_name=run_name,
                dataset=dataset,
                profile=profile,
                role=role,
                summary=dict(item.get("summary", {}) or {}),
                query_rows=qrows,
            )
        )
    return report, runs


def _stage_aggregate(run: RunData) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    sample_map = load_samples_for_dataset(run.dataset)
    stage_acc: Dict[str, Dict[str, List[float]]] = {}
    query_rows: List[Dict[str, Any]] = []
    for short_name, _ in STAGE_KEYS:
        stage_acc[short_name] = {
            "sf_recall": [],
            "sf_precision": [],
            "sf_f1": [],
            "answer_hit": [],
            "answer_density": [],
            "candidate_count": [],
        }

    for row in run.query_rows:
        sid = _safe_text(row.get("sample_id"))
        sample = sample_map.get(sid)
        if sample is None:
            continue
        graph_mode = _safe_text(
            ((row.get("retrieval") or {}).get("diagnostics") or {}).get("graph_mode", "current_entity_graph")
        )
        payloads = _build_stage_payloads(row)

        query_entry: Dict[str, Any] = {
            "dataset": run.dataset,
            "profile": run.profile,
            "sample_id": sid,
            "question": _safe_text(row.get("question", "")),
        }
        for short_name, stage_key in STAGE_KEYS:
            payload = dict(payloads.get(stage_key, {}) or {})
            ids = list(payload.get("ids", []) or [])
            texts = list(payload.get("texts", []) or [])
            metric = stage_metrics(sample, ids, texts, graph_mode)
            sf_r = _safe_float(metric.get("sf_R", 0.0), 0.0)
            sf_p = _safe_float(metric.get("sf_P", 0.0), 0.0)
            sf_f = _safe_float(metric.get("sf_F1", 0.0), 0.0)
            ans = _answer_metrics(sample, texts)
            stage_acc[short_name]["sf_recall"].append(sf_r)
            stage_acc[short_name]["sf_precision"].append(sf_p)
            stage_acc[short_name]["sf_f1"].append(sf_f)
            stage_acc[short_name]["answer_hit"].append(_safe_float(ans.get("answer_hit", 0.0), 0.0))
            stage_acc[short_name]["answer_density"].append(_safe_float(ans.get("answer_density", 0.0), 0.0))
            stage_acc[short_name]["candidate_count"].append(float(len(ids)))
            query_entry[f"{short_name}_SF_recall"] = sf_r
            query_entry[f"{short_name}_answer_hit"] = _safe_float(ans.get("answer_hit", 0.0), 0.0)
        query_rows.append(query_entry)

    stage_out: Dict[str, Dict[str, float]] = {}
    for short_name, vals in stage_acc.items():
        stage_out[short_name] = {
            "SF_recall": _mean(vals["sf_recall"]),
            "SF_precision": _mean(vals["sf_precision"]),
            "SF_f1": _mean(vals["sf_f1"]),
            "answer_hit": _mean(vals["answer_hit"]),
            "answer_density": _mean(vals["answer_density"]),
            "candidate_count": _mean(vals["candidate_count"]),
        }
    return stage_out, query_rows


def _transfer_deltas(stage_out: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    def s(name: str, key: str) -> float:
        return _safe_float(dict(stage_out.get(name, {}) or {}).get(key, 0.0), 0.0)

    return {
        "S1_to_S2_SF_delta": s("S2_phase1", "SF_recall") - s("S1_proposal", "SF_recall"),
        "S2_to_S3_SF_delta": s("S3_local_refinement", "SF_recall") - s("S2_phase1", "SF_recall"),
        "S3_to_S4_SF_delta": s("S4_sentence", "SF_recall") - s("S3_local_refinement", "SF_recall"),
        "S4_to_S5_SF_delta": s("S5_selected", "SF_recall") - s("S4_sentence", "SF_recall"),
        "S1_to_S2_answer_hit_delta": s("S2_phase1", "answer_hit") - s("S1_proposal", "answer_hit"),
        "S2_to_S3_answer_hit_delta": s("S3_local_refinement", "answer_hit") - s("S2_phase1", "answer_hit"),
        "S3_to_S4_answer_hit_delta": s("S4_sentence", "answer_hit") - s("S3_local_refinement", "answer_hit"),
        "S4_to_S5_answer_hit_delta": s("S5_selected", "answer_hit") - s("S4_sentence", "answer_hit"),
    }


def _build_paired_rows(
    dataset: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    bmap = {_safe_text(r.get("sample_id")): r for r in baseline_rows}
    vmap = {_safe_text(r.get("sample_id")): r for r in variant_rows}
    ordered = [_safe_text(r.get("sample_id")) for r in baseline_rows if _safe_text(r.get("sample_id"))]
    paired_rows: List[Dict[str, Any]] = []
    reserve_rows: List[Dict[str, Any]] = []

    for sid in ordered:
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
        vdiag = dict(((v.get("retrieval") or {}).get("diagnostics") or {}))
        reserve_ids = [_normalize_sentence_support_id(x) for x in list(vdiag.get("phase1_reserve_unit_ids", []) or [])]
        rendered_hit = bool(set(reserve_ids).intersection(set(v_rnd))) if reserve_ids else False

        d_f1 = float(vm["f1"] - bm["f1"])
        if d_f1 > 0.05:
            cls = "improved"
        elif d_f1 < -0.05:
            cls = "regressed"
        else:
            cls = "neutral"

        paired_rows.append(
            {
                "dataset": dataset,
                "sample_id": sid,
                "question": _safe_text(v.get("question", b.get("question", ""))),
                "baseline_prediction": _safe_text(b.get("prediction", "")),
                "variant_prediction": _safe_text(v.get("prediction", "")),
                "baseline_em": bm["em"],
                "variant_em": vm["em"],
                "baseline_f1": bm["f1"],
                "variant_f1": vm["f1"],
                "delta_f1": d_f1,
                "baseline_sf_recall": bm["sf_recall"],
                "variant_sf_recall": vm["sf_recall"],
                "delta_sf_recall": float(vm["sf_recall"] - bm["sf_recall"]),
                "baseline_sf_precision": bm["sf_precision"],
                "variant_sf_precision": vm["sf_precision"],
                "baseline_retrieval_ms": bm["retrieval_ms"],
                "variant_retrieval_ms": vm["retrieval_ms"],
                "baseline_total_ms": bm["total_ms"],
                "variant_total_ms": vm["total_ms"],
                "selected_sentence_jaccard": _jaccard(b_sel, v_sel),
                "rendered_sentence_jaccard": _jaccard(b_rnd, v_rnd),
                "classification": cls,
            }
        )

        reserve_rows.append(
            {
                "dataset": dataset,
                "sample_id": sid,
                "num_phase1_shortlist_before": _safe_int(vdiag.get("num_phase1_shortlist_before", 0), 0),
                "num_phase1_shortlist_after": _safe_int(vdiag.get("num_phase1_shortlist_after", 0), 0),
                "phase1_reserve_added": bool(vdiag.get("phase1_reserve_added", False)),
                "phase1_reserve_seed_or_run_id": _safe_int(vdiag.get("phase1_reserve_seed_or_run_id", -1), -1),
                "phase1_reserve_reason": _safe_text(vdiag.get("phase1_reserve_reason", "")),
                "phase1_reserve_score": _safe_float(vdiag.get("phase1_reserve_score", 0.0), 0.0),
                "phase1_reserve_reaches_phase2": bool(vdiag.get("phase1_reserve_reaches_phase2", False)),
                "phase1_reserve_has_sentence_candidates": bool(vdiag.get("phase1_reserve_has_sentence_candidates", False)),
                "phase1_reserve_selected_by_ABR": bool(vdiag.get("phase1_reserve_selected_by_ABR", False)),
                "phase1_reserve_rendered": bool(rendered_hit),
            }
        )
    return paired_rows, reserve_rows


def _markdown_examples(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        return "\n".join(lines) + "\n"
    lines.append("| dataset | sample_id | baseline_f1 | variant_f1 | delta_f1 | delta_sf_recall | selected_jaccard | rendered_jaccard |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {dataset} | {sample_id} | {b:.4f} | {v:.4f} | {df:+.4f} | {ds:+.4f} | {sj:.4f} | {rj:.4f} |".format(
                dataset=_safe_text(row.get("dataset", "")),
                sample_id=_safe_text(row.get("sample_id", "")),
                b=_safe_float(row.get("baseline_f1", 0.0), 0.0),
                v=_safe_float(row.get("variant_f1", 0.0), 0.0),
                df=_safe_float(row.get("delta_f1", 0.0), 0.0),
                ds=_safe_float(row.get("delta_sf_recall", 0.0), 0.0),
                sj=_safe_float(row.get("selected_sentence_jaccard", 0.0), 0.0),
                rj=_safe_float(row.get("rendered_sentence_jaccard", 0.0), 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PHASE6X early-collapse recovery run.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--baseline-profile", default="unified_acr_rcedr_v12")
    ap.add_argument("--variant-profile", default="unified_acr_rcedr_v12_early_collapse_recovery")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    profiles = [args.baseline_profile, args.variant_profile]

    report, runs = _load_runs(out_root, datasets=datasets, profiles=profiles)

    by_dataset_profile: Dict[Tuple[str, str], RunData] = {}
    for r in runs:
        by_dataset_profile[(r.dataset, r.profile)] = r

    stage_csv_rows: List[Dict[str, Any]] = []
    paired_rows_all: List[Dict[str, Any]] = []
    reserve_rows_all: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        b = by_dataset_profile.get((dataset, args.baseline_profile))
        v = by_dataset_profile.get((dataset, args.variant_profile))
        if b is None or v is None:
            continue

        b_metrics = _summary_metrics(b.summary)
        v_metrics = _summary_metrics(v.summary)
        b_stage, _ = _stage_aggregate(b)
        v_stage, _ = _stage_aggregate(v)
        b_delta = _transfer_deltas(b_stage)
        v_delta = _transfer_deltas(v_stage)

        for prof, stage_out in ((args.baseline_profile, b_stage), (args.variant_profile, v_stage)):
            for short_name, _ in STAGE_KEYS:
                s = dict(stage_out.get(short_name, {}) or {})
                stage_csv_rows.append(
                    {
                        "dataset": dataset,
                        "profile": prof,
                        "stage": short_name,
                        "candidate_count": _safe_float(s.get("candidate_count", 0.0), 0.0),
                        "SF_recall": _safe_float(s.get("SF_recall", 0.0), 0.0),
                        "SF_precision": _safe_float(s.get("SF_precision", 0.0), 0.0),
                        "SF_f1": _safe_float(s.get("SF_f1", 0.0), 0.0),
                        "answer_hit": _safe_float(s.get("answer_hit", 0.0), 0.0),
                        "answer_bearing_density": _safe_float(s.get("answer_density", 0.0), 0.0),
                    }
                )

        paired_rows, reserve_rows = _build_paired_rows(dataset, b.query_rows, v.query_rows)
        paired_rows_all.extend(paired_rows)
        reserve_rows_all.extend(reserve_rows)

        reserve_trigger_rows = [r for r in reserve_rows if bool(r.get("phase1_reserve_added", False))]
        summary_rows.append(
            {
                "dataset": dataset,
                "baseline": b_metrics,
                "variant": v_metrics,
                "delta": {k: float(v_metrics.get(k, 0.0) - b_metrics.get(k, 0.0)) for k in b_metrics.keys()},
                "baseline_stage": b_stage,
                "variant_stage": v_stage,
                "baseline_stage_delta": b_delta,
                "variant_stage_delta": v_delta,
                "reserve_metrics": {
                    "reserve_trigger_rate": _mean([1.0 if bool(r.get("phase1_reserve_added", False)) else 0.0 for r in reserve_rows]),
                    "reserve_phase2_survival_rate": _mean([1.0 if bool(r.get("phase1_reserve_reaches_phase2", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_sentence_candidate_rate": _mean([1.0 if bool(r.get("phase1_reserve_has_sentence_candidates", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_ABR_selected_rate": _mean([1.0 if bool(r.get("phase1_reserve_selected_by_ABR", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                    "reserve_rendered_rate": _mean([1.0 if bool(r.get("phase1_reserve_rendered", False)) else 0.0 for r in reserve_trigger_rows]) if reserve_trigger_rows else 0.0,
                },
            }
        )

    improved = sorted(
        [r for r in paired_rows_all if _safe_float(r.get("delta_f1", 0.0), 0.0) > 0],
        key=lambda x: _safe_float(x.get("delta_f1", 0.0), 0.0),
        reverse=True,
    )[: max(1, int(args.top_k))]
    regressed = sorted(
        [r for r in paired_rows_all if _safe_float(r.get("delta_f1", 0.0), 0.0) < 0],
        key=lambda x: _safe_float(x.get("delta_f1", 0.0), 0.0),
    )[: max(1, int(args.top_k))]

    _write_jsonl(out_root / "paired_query_comparison.jsonl", paired_rows_all)
    _write_csv(
        out_root / "early_collapse_stagewise_metrics.csv",
        stage_csv_rows,
        [
            "dataset",
            "profile",
            "stage",
            "candidate_count",
            "SF_recall",
            "SF_precision",
            "SF_f1",
            "answer_hit",
            "answer_bearing_density",
        ],
    )
    _write_csv(
        out_root / "reserve_diagnostics.csv",
        reserve_rows_all,
        [
            "dataset",
            "sample_id",
            "num_phase1_shortlist_before",
            "num_phase1_shortlist_after",
            "phase1_reserve_added",
            "phase1_reserve_seed_or_run_id",
            "phase1_reserve_reason",
            "phase1_reserve_score",
            "phase1_reserve_reaches_phase2",
            "phase1_reserve_has_sentence_candidates",
            "phase1_reserve_selected_by_ABR",
            "phase1_reserve_rendered",
        ],
    )

    (out_root / "improved_examples_top20.md").write_text(
        _markdown_examples(improved, "PHASE6X Improved Examples Top20"),
        encoding="utf-8",
    )
    (out_root / "regression_examples_top20.md").write_text(
        _markdown_examples(regressed, "PHASE6X Regression Examples Top20"),
        encoding="utf-8",
    )

    decision = "hold"
    if summary_rows:
        avg_delta_f1 = _mean([_safe_float(r.get("delta", {}).get("F1", 0.0), 0.0) for r in summary_rows])
        max_drop = min([_safe_float(r.get("delta", {}).get("F1", 0.0), 0.0) for r in summary_rows])
        avg_token_delta = _mean([_safe_float(r.get("delta", {}).get("avg_context_tokens", 0.0), 0.0) for r in summary_rows])
        avg_ret_delta = _mean([_safe_float(r.get("delta", {}).get("retrieval_ms", 0.0), 0.0) for r in summary_rows])
        if avg_delta_f1 >= 0.02 and max_drop >= -0.01 and avg_token_delta <= 10.0 and avg_ret_delta <= 20.0:
            decision = "strong_accept"
        elif max_drop <= -0.02:
            decision = "reject"
        else:
            decision = "hold"
    else:
        avg_delta_f1 = 0.0
        max_drop = 0.0
        avg_token_delta = 0.0
        avg_ret_delta = 0.0

    summary_json = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_root": str(out_root),
        "artifact_validation": {
            "scheduled_runs": int(_safe_int(report.get("scheduled_runs_count", 0), 0)),
            "completed_runs": int(_safe_int(report.get("completed_runs_count", 0), 0)),
            "incomplete_attempts": int(_safe_int(report.get("incomplete_attempts_count", 0), 0)),
            "extra_not_in_manifest": int(_safe_int(report.get("extra_not_in_manifest_count", 0), 0)),
            "multi_attempt_run_names": int(_safe_int(report.get("multi_attempt_run_names_count", 0), 0)),
            "parse_errors": int(_safe_int(report.get("parse_errors_count", 0), 0)),
        },
        "datasets": summary_rows,
        "aggregates": {
            "avg_delta_f1": avg_delta_f1,
            "max_dataset_drop_f1": max_drop,
            "avg_delta_context_tokens": avg_token_delta,
            "avg_delta_retrieval_ms": avg_ret_delta,
            "paired_query_count": int(len(paired_rows_all)),
        },
        "decision": decision,
    }
    _write_json(out_root / "phase6x_early_collapse_recovery_summary.json", summary_json)

    lines: List[str] = []
    lines.append("# PHASE6X Early-Collapse Recovery Summary")
    lines.append("")
    lines.append("## 1. Goal")
    lines.append("Test one dataset-agnostic reserve candidate (`reserve=1`) to reduce Phase1/Phase2 early collapse without changing ABR or context budget.")
    lines.append("")
    lines.append("## 2. Artifact Validation")
    av = summary_json["artifact_validation"]
    lines.append(f"- scheduled_runs: {av['scheduled_runs']}")
    lines.append(f"- completed_runs: {av['completed_runs']}")
    lines.append(f"- incomplete_attempts: {av['incomplete_attempts']}")
    lines.append(f"- extra_not_in_manifest: {av['extra_not_in_manifest']}")
    lines.append(f"- multi_attempt_run_names: {av['multi_attempt_run_names']}")
    lines.append(f"- parse_errors: {av['parse_errors']}")
    lines.append("")
    lines.append("## 3. Paired Main Results")
    lines.append("| dataset | profile | EM | F1 | Recall@5 | SF_recall | SF_precision | SF_f1 | avg_context_tokens | F1_per_1k_context_tokens | answer_string_hit | answer_bearing_density | retrieval_ms | generation_ms | total_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        dataset = _safe_text(row.get("dataset", ""))
        for prof_key in ("baseline", "variant"):
            prof_name = args.baseline_profile if prof_key == "baseline" else args.variant_profile
            m = dict(row.get(prof_key, {}) or {})
            lines.append(
                "| {d} | {p} | {em} | {f1} | {r5} | {sfr} | {sfp} | {sff} | {ctx} | {f1k} | {ah} | {ad} | {rm} | {gm} | {tm} |".format(
                    d=dataset,
                    p=prof_name,
                    em=_fmt(m.get("EM"), 4),
                    f1=_fmt(m.get("F1"), 4),
                    r5=_fmt(m.get("Recall@5"), 4),
                    sfr=_fmt(m.get("supporting_fact_recall"), 4),
                    sfp=_fmt(m.get("supporting_fact_precision"), 4),
                    sff=_fmt(m.get("supporting_fact_f1"), 4),
                    ctx=_fmt(m.get("avg_context_tokens"), 2),
                    f1k=_fmt(m.get("F1_per_1k_context_tokens"), 4),
                    ah=_fmt(m.get("answer_string_hit"), 4),
                    ad=_fmt(m.get("answer_bearing_density"), 4),
                    rm=_fmt(m.get("retrieval_ms"), 2),
                    gm=_fmt(m.get("generation_ms"), 2),
                    tm=_fmt(m.get("total_ms"), 2),
                )
            )
    lines.append("")
    lines.append("## 4. Delta vs Baseline")
    lines.append("| dataset | ΔEM | ΔF1 | ΔSF_recall | ΔSF_precision | Δavg_context_tokens | ΔF1_per_1k_context_tokens | Δanswer_string_hit | Δanswer_bearing_density | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        d = dict(row.get("delta", {}) or {})
        lines.append(
            "| {dataset} | {em} | {f1} | {sfr} | {sfp} | {ctx} | {f1k} | {ah} | {ad} | {rm} | {tm} |".format(
                dataset=_safe_text(row.get("dataset", "")),
                em=_fmt(d.get("EM"), 4),
                f1=_fmt(d.get("F1"), 4),
                sfr=_fmt(d.get("supporting_fact_recall"), 4),
                sfp=_fmt(d.get("supporting_fact_precision"), 4),
                ctx=_fmt(d.get("avg_context_tokens"), 2),
                f1k=_fmt(d.get("F1_per_1k_context_tokens"), 4),
                ah=_fmt(d.get("answer_string_hit"), 4),
                ad=_fmt(d.get("answer_bearing_density"), 4),
                rm=_fmt(d.get("retrieval_ms"), 2),
                tm=_fmt(d.get("total_ms"), 2),
            )
        )
    lines.append("")
    lines.append("## 5. Stagewise Evidence Transfer")
    for row in summary_rows:
        lines.append(f"- {row['dataset']} baseline deltas: {row['baseline_stage_delta']}")
        lines.append(f"- {row['dataset']} variant deltas: {row['variant_stage_delta']}")
    lines.append("")
    lines.append("## 6. Phase1 Reserve Diagnostics")
    lines.append("| dataset | reserve_trigger_rate | reserve_phase2_survival_rate | reserve_sentence_candidate_rate | reserve_ABR_selected_rate | reserve_rendered_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        r = dict(row.get("reserve_metrics", {}) or {})
        lines.append(
            "| {dataset} | {t} | {p2} | {sc} | {abr} | {rend} |".format(
                dataset=_safe_text(row.get("dataset", "")),
                t=_fmt(r.get("reserve_trigger_rate"), 4),
                p2=_fmt(r.get("reserve_phase2_survival_rate"), 4),
                sc=_fmt(r.get("reserve_sentence_candidate_rate"), 4),
                abr=_fmt(r.get("reserve_ABR_selected_rate"), 4),
                rend=_fmt(r.get("reserve_rendered_rate"), 4),
            )
        )
    lines.append("")
    lines.append("## 7. Candidate Ceiling Recovery")
    lines.append("See `early_collapse_stagewise_metrics.csv` for S1~S6 per-profile aggregates and transfer deltas.")
    lines.append("")
    lines.append("## 8. Token/Latency Guardrail")
    lines.append(f"- avg_delta_f1: {_fmt(avg_delta_f1, 4)}")
    lines.append(f"- max_dataset_drop_f1: {_fmt(max_drop, 4)}")
    lines.append(f"- avg_delta_context_tokens: {_fmt(avg_token_delta, 2)}")
    lines.append(f"- avg_delta_retrieval_ms: {_fmt(avg_ret_delta, 2)}")
    lines.append("")
    lines.append("## 9. Improved Examples")
    lines.append(f"- `improved_examples_top20.md` ({len(improved)} rows)")
    lines.append("")
    lines.append("## 10. Regression Examples")
    lines.append(f"- `regression_examples_top20.md` ({len(regressed)} rows)")
    lines.append("")
    lines.append("## 11. Decision")
    lines.append(f"- {decision}")
    lines.append("")

    (out_root / "PHASE6X_EARLY_COLLAPSE_RECOVERY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
