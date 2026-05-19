#!/usr/bin/env python3
"""PHASE6V semantic-sufficiency case audit from existing paired outputs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts

from effirag.eval.metrics_legacy import normalize_answer_legacy


def _safe_text(v: Any) -> str:
    return str(v or "").strip()


def _safe_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


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


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(obj) if isinstance(obj, Mapping) else {}


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


@dataclass
class RunData:
    run_name: str
    dataset: str
    profile: str
    summary_path: Path
    query_path: Path
    summary: Dict[str, Any]
    rows: List[Dict[str, Any]]


def _infer_dataset(run_name: str, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    d = _safe_text(summary.get("dataset", ""))
    if d:
        return d
    if rows:
        d = _safe_text(rows[0].get("dataset", ""))
        if d:
            return d
    n = run_name.lower()
    if "2wiki" in n:
        return "2wikimultihopqa"
    if "hotpot" in n:
        return "hotpotqa"
    return ""


def _infer_profile(run_name: str, summary: Mapping[str, Any]) -> str:
    p = _safe_text(summary.get("profile", ""))
    if p:
        return p
    n = run_name.lower()
    if "semantic_sufficiency" in n:
        return "unified_acr_rcedr_v12_semantic_sufficiency"
    if "semantic_contract" in n:
        return "unified_acr_rcedr_v12_semantic_contract"
    return "unified_acr_rcedr_v12"


def _scan_runs_fallback(input_root: Path) -> List[RunData]:
    out: List[RunData] = []
    qa_root = input_root / "qa_runs"
    if not qa_root.exists():
        return out

    for run_dir in sorted(p for p in qa_root.iterdir() if p.is_dir()):
        run_name = run_dir.name
        summary_paths = sorted(run_dir.rglob("rag_summary.json"))
        query_paths = sorted(run_dir.rglob("rag_query_results.jsonl"))
        if not summary_paths or not query_paths:
            continue
        summary_path = summary_paths[-1]
        query_path = query_paths[-1]
        summary = _read_json(summary_path)
        rows = _read_jsonl(query_path)
        if not rows:
            continue
        dataset = _infer_dataset(run_name, rows, summary)
        profile = _infer_profile(run_name, summary)
        out.append(
            RunData(
                run_name=run_name,
                dataset=dataset,
                profile=profile,
                summary_path=summary_path,
                query_path=query_path,
                summary=summary,
                rows=rows,
            )
        )
    return out


def _load_runs(input_root: Path) -> List[RunData]:
    report = collect_phase6s_artifacts(input_root)
    scheduled = {_safe_text(r.get("run_name")): dict(r) for r in list(report.get("scheduled_runs", []) or [])}
    runs: List[RunData] = []

    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        if not run_name:
            continue
        sch = scheduled.get(run_name, {})
        query_path = Path(_safe_text(item.get("query_path"))).resolve()
        summary_path = Path(_safe_text(item.get("summary_path"))).resolve()
        rows = _read_jsonl(query_path)
        summary = _read_json(summary_path)
        if not rows:
            continue
        dataset = _safe_text(sch.get("dataset", "")) or _infer_dataset(run_name, rows, summary)
        profile = _safe_text(sch.get("profile", "")) or _infer_profile(run_name, summary)
        runs.append(
            RunData(
                run_name=run_name,
                dataset=dataset,
                profile=profile,
                summary_path=summary_path,
                query_path=query_path,
                summary=summary,
                rows=rows,
            )
        )

    if runs:
        return runs
    return _scan_runs_fallback(input_root)


def _extract_ids(row: Mapping[str, Any], rendered: bool = False) -> List[str]:
    if rendered:
        ids = list(row.get("rendered_sentence_ids", []) or [])
        if not ids:
            ids = list(((row.get("rendered") or {}).get("sentence_ids", []) or []))
    else:
        ids = list(row.get("retrieval_selected_sentence_ids", []) or [])
    return [_safe_text(x) for x in ids if _safe_text(x)]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _answer_candidates(row: Mapping[str, Any]) -> Tuple[str, List[str]]:
    ans = normalize_answer_legacy(_safe_text(row.get("answer", "")))
    aliases: List[str] = []
    for a in list(row.get("gold_answers", []) or []):
        na = normalize_answer_legacy(_safe_text(a))
        if na and na != ans and na not in aliases:
            aliases.append(na)
    return ans, aliases


def _answer_sufficiency_from_row(row: Mapping[str, Any]) -> Dict[str, float]:
    rendered = dict(row.get("rendered", {}) or {})
    sentences = list(rendered.get("sentences", []) or [])
    sent_norm = [normalize_answer_legacy(_safe_text(s)) for s in sentences]
    answer_norm, alias_norms = _answer_candidates(row)

    def _contains(s: str, target: str) -> bool:
        return bool(s and target and target in s)

    hit_indices: List[int] = []
    alias_hit_indices: List[int] = []
    for i, s in enumerate(sent_norm):
        if _contains(s, answer_norm):
            hit_indices.append(i)
        elif any(_contains(s, a) for a in alias_norms):
            hit_indices.append(i)
            alias_hit_indices.append(i)

    hit = 1.0 if hit_indices else 0.0
    alias_hit = 1.0 if alias_hit_indices else 0.0
    rank = float(hit_indices[0] + 1) if hit_indices else -1.0
    count = float(len(hit_indices))
    density = float(count / len(sentences)) if sentences else 0.0
    early_hit = 1.0 if any(idx < 3 for idx in hit_indices) else 0.0

    # Optional equivalent evidence proxy: answer-surface hit OR strong question overlap.
    eq_proxy = hit
    if eq_proxy == 0.0:
        q_tokens = set(normalize_answer_legacy(_safe_text(row.get("question", ""))).split())
        q_tokens = {t for t in q_tokens if len(t) >= 4}
        for s in sent_norm:
            if not q_tokens:
                continue
            toks = set(s.split())
            overlap = len(toks & q_tokens) / max(1, len(q_tokens))
            if overlap >= 0.35:
                eq_proxy = 1.0
                break

    return {
        "answer_string_hit": hit,
        "answer_alias_hit": alias_hit,
        "answer_string_rank": rank,
        "answer_bearing_sentence_count": count,
        "answer_bearing_density": density,
        "answer_context_early_hit": early_hit,
        "equivalent_evidence_proxy_hit": eq_proxy,
    }


def _query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    m = dict(row.get("metrics", {}) or {})
    gd = dict(row.get("generation_diagnostics", {}) or {})
    out = {
        "EM": _safe_float(m.get("em", 0.0), 0.0),
        "F1": _safe_float(m.get("f1", 0.0), 0.0),
        "SF_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
        "SF_precision": _safe_float(m.get("supporting_fact_precision", 0.0), 0.0),
        "context_tokens": _safe_float(gd.get("prompt_tokens", 0.0), 0.0),
    }
    out.update(_answer_sufficiency_from_row(row))
    return out


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
    em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
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
    }


def _rank_delta_direction(base_rank: float, sem_rank: float) -> Tuple[bool, bool]:
    # lower rank is better; -1 means absent.
    if base_rank < 0 and sem_rank < 0:
        return False, False
    if base_rank < 0 and sem_rank >= 0:
        return True, False
    if base_rank >= 0 and sem_rank < 0:
        return False, True
    # both present
    if sem_rank < base_rank:
        return True, False
    if sem_rank > base_rank:
        return False, True
    return False, False


def _classify_case(delta_f1: float, delta_sf: float, delta_hit: float, delta_density: float, rank_worse: bool) -> str:
    f1_th = 0.05
    sf_th = 0.10
    if delta_f1 >= f1_th and delta_sf <= 0.0:
        return "improved_answer_sufficiency"
    if delta_sf >= sf_th:
        return "sf_recall_gain"
    if delta_f1 <= -f1_th and (delta_hit < 0.0 or delta_density < 0.0 or rank_worse):
        return "semantic_distractor_regression"
    if abs(delta_f1) < f1_th and abs(delta_sf) < sf_th:
        return "neutral"
    return "mixed"


def _write_examples_md(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for i, r in enumerate(rows, 1):
        lines.append(f"## {i}. {r.get('dataset')} / {r.get('sample_id')}")
        lines.append(f"- question: {r.get('question','')}")
        lines.append(
            f"- baseline_F1={_fmt(r.get('baseline_F1'))}, semantic_F1={_fmt(r.get('semantic_F1'))}, "
            f"delta_F1={_fmt(r.get('delta_F1'))}"
        )
        lines.append(
            f"- baseline_SF_recall={_fmt(r.get('baseline_SF_recall'))}, semantic_SF_recall={_fmt(r.get('semantic_SF_recall'))}, "
            f"delta_SF_recall={_fmt(r.get('delta_SF_recall'))}"
        )
        lines.append(
            f"- answer_hit: {_fmt(r.get('baseline_answer_string_hit'))} -> {_fmt(r.get('semantic_answer_string_hit'))}, "
            f"density: {_fmt(r.get('baseline_answer_bearing_density'))} -> {_fmt(r.get('semantic_answer_bearing_density'))}"
        )
        lines.append(f"- baseline_prediction: {r.get('baseline_prediction','')}")
        lines.append(f"- semantic_prediction: {r.get('semantic_prediction','')}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--profiles", default="unified_acr_rcedr_v12,unified_acr_rcedr_v12_semantic_sufficiency")
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    wanted_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    wanted_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    base_profile = "unified_acr_rcedr_v12"
    sem_profile = "unified_acr_rcedr_v12_semantic_sufficiency"

    runs = _load_runs(input_root)
    runs = [r for r in runs if r.dataset in wanted_datasets and r.profile in wanted_profiles]

    by_ds_prof: Dict[Tuple[str, str], RunData] = {}
    for r in runs:
        key = (r.dataset, r.profile)
        prev = by_ds_prof.get(key)
        if prev is None or len(r.rows) > len(prev.rows):
            by_ds_prof[key] = r

    results_rows: List[Dict[str, Any]] = []
    deltas_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    case_dist_rows: List[Dict[str, Any]] = []

    improved_examples: List[Dict[str, Any]] = []
    regress_examples: List[Dict[str, Any]] = []

    dataset_answer_delta: Dict[str, Dict[str, float]] = {}

    for ds in wanted_datasets:
        base_run = by_ds_prof.get((ds, base_profile))
        sem_run = by_ds_prof.get((ds, sem_profile))
        if not base_run or not sem_run:
            continue

        base_m = _summary_metrics(base_run.summary)
        sem_m = _summary_metrics(sem_run.summary)

        results_rows.append({"dataset": ds, "profile": base_profile, **base_m})
        results_rows.append({"dataset": ds, "profile": sem_profile, **sem_m})

        deltas_rows.append(
            {
                "dataset": ds,
                "profile": sem_profile,
                "delta_F1": float(sem_m["F1"] - base_m["F1"]),
                "delta_SF_recall": float(sem_m["supporting_fact_recall"] - base_m["supporting_fact_recall"]),
                "delta_SF_precision": float(sem_m["supporting_fact_precision"] - base_m["supporting_fact_precision"]),
                "delta_context_tokens": float(sem_m["avg_context_tokens"] - base_m["avg_context_tokens"]),
                "delta_retrieval_ms": float(sem_m["retrieval_ms"] - base_m["retrieval_ms"]),
                "delta_total_ms": float(sem_m["total_ms"] - base_m["total_ms"]),
            }
        )

        base_map = {_safe_text(r.get("sample_id")): r for r in base_run.rows if _safe_text(r.get("sample_id"))}
        sem_map = {_safe_text(r.get("sample_id")): r for r in sem_run.rows if _safe_text(r.get("sample_id"))}
        ordered_ids = [_safe_text(r.get("sample_id")) for r in base_run.rows if _safe_text(r.get("sample_id"))]

        dist = {
            "improved_answer_sufficiency": 0,
            "sf_recall_gain": 0,
            "semantic_distractor_regression": 0,
            "neutral": 0,
            "mixed": 0,
        }

        d_hit: List[float] = []
        d_rank: List[float] = []
        d_density: List[float] = []
        d_early: List[float] = []

        for sid in ordered_ids:
            b = base_map.get(sid)
            s = sem_map.get(sid)
            if b is None or s is None:
                continue

            bm = _query_metrics(b)
            sm = _query_metrics(s)
            delta_f1 = float(sm["F1"] - bm["F1"])
            delta_sf = float(sm["SF_recall"] - bm["SF_recall"])
            delta_hit = float(sm["answer_string_hit"] - bm["answer_string_hit"])
            delta_rank = float(sm["answer_string_rank"] - bm["answer_string_rank"])
            delta_density = float(sm["answer_bearing_density"] - bm["answer_bearing_density"])
            delta_early = float(sm["answer_context_early_hit"] - bm["answer_context_early_hit"])

            rank_improved, rank_worse = _rank_delta_direction(bm["answer_string_rank"], sm["answer_string_rank"])
            cls = _classify_case(delta_f1, delta_sf, delta_hit, delta_density, rank_worse)
            dist[cls] = int(dist.get(cls, 0) + 1)

            row = {
                "dataset": ds,
                "sample_id": sid,
                "question": _safe_text(b.get("question", "")),
                "gold_answer": _safe_text(b.get("answer", "")),
                "baseline_prediction": _safe_text(b.get("prediction", "")),
                "semantic_prediction": _safe_text(s.get("prediction", "")),
                "baseline_EM": bm["EM"],
                "semantic_EM": sm["EM"],
                "baseline_F1": bm["F1"],
                "semantic_F1": sm["F1"],
                "delta_F1": delta_f1,
                "baseline_SF_recall": bm["SF_recall"],
                "semantic_SF_recall": sm["SF_recall"],
                "delta_SF_recall": delta_sf,
                "baseline_SF_precision": bm["SF_precision"],
                "semantic_SF_precision": sm["SF_precision"],
                "baseline_context_tokens": bm["context_tokens"],
                "semantic_context_tokens": sm["context_tokens"],
                "baseline_selected_sentence_ids": _extract_ids(b, rendered=False),
                "semantic_selected_sentence_ids": _extract_ids(s, rendered=False),
                "baseline_rendered_sentence_ids": _extract_ids(b, rendered=True),
                "semantic_rendered_sentence_ids": _extract_ids(s, rendered=True),
                "selected_sentence_jaccard": _jaccard(_extract_ids(b, False), _extract_ids(s, False)),
                "rendered_sentence_jaccard": _jaccard(_extract_ids(b, True), _extract_ids(s, True)),
                "baseline_answer_string_hit": bm["answer_string_hit"],
                "semantic_answer_string_hit": sm["answer_string_hit"],
                "baseline_answer_alias_hit": bm["answer_alias_hit"],
                "semantic_answer_alias_hit": sm["answer_alias_hit"],
                "baseline_answer_string_rank": bm["answer_string_rank"],
                "semantic_answer_string_rank": sm["answer_string_rank"],
                "baseline_answer_bearing_sentence_count": bm["answer_bearing_sentence_count"],
                "semantic_answer_bearing_sentence_count": sm["answer_bearing_sentence_count"],
                "baseline_answer_bearing_density": bm["answer_bearing_density"],
                "semantic_answer_bearing_density": sm["answer_bearing_density"],
                "baseline_answer_context_early_hit": bm["answer_context_early_hit"],
                "semantic_answer_context_early_hit": sm["answer_context_early_hit"],
                "baseline_equivalent_evidence_proxy_hit": bm["equivalent_evidence_proxy_hit"],
                "semantic_equivalent_evidence_proxy_hit": sm["equivalent_evidence_proxy_hit"],
                "case_type": cls,
                "rank_improved": rank_improved,
                "rank_worse": rank_worse,
                "delta_answer_string_hit": delta_hit,
                "delta_answer_string_rank": delta_rank,
                "delta_answer_bearing_density": delta_density,
                "delta_answer_context_early_hit": delta_early,
            }
            paired_rows.append(row)

            if cls == "improved_answer_sufficiency":
                improved_examples.append(row)
            if cls == "semantic_distractor_regression":
                regress_examples.append(row)

            d_hit.append(delta_hit)
            d_rank.append(delta_rank)
            d_density.append(delta_density)
            d_early.append(delta_early)

        total = sum(dist.values())
        for k, c in dist.items():
            case_dist_rows.append(
                {
                    "dataset": ds,
                    "case_type": k,
                    "count": c,
                    "share": (float(c) / float(total)) if total > 0 else 0.0,
                }
            )

        dataset_answer_delta[ds] = {
            "delta_answer_string_hit_avg": _mean(d_hit),
            "delta_answer_string_rank_avg": _mean(d_rank),
            "delta_answer_bearing_density_avg": _mean(d_density),
            "delta_answer_context_early_hit_avg": _mean(d_early),
        }

    improved_examples = sorted(improved_examples, key=lambda r: (r["delta_F1"], r["delta_answer_bearing_density"]), reverse=True)[: args.top_k]
    regress_examples = sorted(regress_examples, key=lambda r: (r["delta_F1"], r["delta_answer_bearing_density"]))[: args.top_k]

    _write_jsonl(out_root / "paired_case_comparison.jsonl", paired_rows)
    _write_csv(out_root / "dataset_case_distribution.csv", case_dist_rows, ["dataset", "case_type", "count", "share"])
    _write_examples_md(
        out_root / "improved_answer_sufficiency_examples_top20.md",
        "Improved Answer-Sufficiency Examples Top20",
        improved_examples,
    )
    _write_examples_md(
        out_root / "semantic_distractor_regression_examples_top20.md",
        "Semantic Distractor Regression Examples Top20",
        regress_examples,
    )

    # Decision logic for Stage A.
    delta_by_ds = {d["dataset"]: d for d in deltas_rows}
    two = delta_by_ds.get("2wikimultihopqa")
    hot = delta_by_ds.get("hotpotqa")

    decision = "hold_for_debug"
    reasons: List[str] = []
    if not paired_rows:
        decision = "hold_for_debug"
        reasons.append("No paired rows available.")
    else:
        two_signal = False
        hot_catastrophic = False
        if two is not None:
            d2 = dataset_answer_delta.get("2wikimultihopqa", {})
            two_signal = (
                _safe_float(two.get("delta_F1"), 0.0) > 0.015
                and (
                    _safe_float(d2.get("delta_answer_string_hit_avg"), 0.0) > 0.0
                    or _safe_float(d2.get("delta_answer_bearing_density_avg"), 0.0) > 0.0
                    or _safe_float(d2.get("delta_answer_string_rank_avg"), 0.0) < 0.0
                )
            )
        if hot is not None:
            hot_catastrophic = _safe_float(hot.get("delta_F1"), 0.0) <= -0.03

        if two_signal and not hot_catastrophic:
            decision = "proceed_to_n200"
            reasons.append("2Wiki F1 gain aligns with answer-sufficiency deltas; Hotpot regression not catastrophic.")
        elif two is not None and hot is not None and _safe_float(two.get("delta_F1"), 0.0) <= 0.0 and _safe_float(hot.get("delta_F1"), 0.0) < 0.0:
            decision = "reject_semantic_sufficiency"
            reasons.append("No positive 2Wiki signal and Hotpot regression present.")
        else:
            decision = "hold_for_debug"
            reasons.append("Signal is mixed or not strongly tied to answer-sufficiency metrics.")

    payload = {
        "phase": "phase6v_semantic_sufficiency_case_audit",
        "input_root": str(input_root),
        "output_root": str(out_root),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profiles": wanted_profiles,
        "datasets": wanted_datasets,
        "results": results_rows,
        "deltas_vs_baseline": deltas_rows,
        "dataset_answer_sufficiency_deltas": dataset_answer_delta,
        "case_distribution": case_dist_rows,
        "decision": decision,
        "decision_reasons": reasons,
    }
    _write_json(out_root / "phase6v_semantic_sufficiency_case_audit.json", payload)

    md: List[str] = []
    md.append("# PHASE6V Semantic-Sufficiency Case Audit")
    md.append("")
    md.append("## 1. Goal")
    md.append("- Re-audit paired n=100 baseline vs semantic_sufficiency to separate answer-sufficiency gains from SF-recall changes.")
    md.append("")
    md.append("## 2. Dataset-Level Summary")
    md.append("| dataset | profile | Recall@5 | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | SF_precision | SF_recall | SF_f1 | retrieval_ms | generation_ms | total_ms |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results_rows, key=lambda x: (x["dataset"], x["profile"])):
        md.append(
            "| {d} | {p} | {r5} | {em} | {f1} | {ctx} | {f1k} | {sfp} | {sfr} | {sff1} | {ret} | {gen} | {tot} |".format(
                d=r["dataset"],
                p=r["profile"],
                r5=_fmt(r["Recall@5"], 4),
                em=_fmt(r["EM"], 4),
                f1=_fmt(r["F1"], 4),
                ctx=_fmt(r["avg_context_tokens"], 3),
                f1k=_fmt(r["F1_per_1k_context_tokens"], 4),
                sfp=_fmt(r["supporting_fact_precision"], 4),
                sfr=_fmt(r["supporting_fact_recall"], 4),
                sff1=_fmt(r["supporting_fact_f1"], 4),
                ret=_fmt(r["retrieval_ms"], 2),
                gen=_fmt(r["generation_ms"], 2),
                tot=_fmt(r["total_ms"], 2),
            )
        )

    md.append("")
    md.append("## 3. Case Type Distribution")
    md.append("| dataset | case_type | count | share |")
    md.append("|---|---:|---:|---:|")
    for r in sorted(case_dist_rows, key=lambda x: (x["dataset"], x["case_type"])):
        md.append(f"| {r['dataset']} | {r['case_type']} | {r['count']} | {_fmt(r['share'], 4)} |")

    md.append("")
    md.append("## 4. 2Wiki Improved Answer-Sufficiency Examples")
    md.append(f"- See `{(out_root / 'improved_answer_sufficiency_examples_top20.md').name}`")

    md.append("")
    md.append("## 5. HotpotQA Regression Examples")
    md.append(f"- See `{(out_root / 'semantic_distractor_regression_examples_top20.md').name}`")

    md.append("")
    md.append("## 6. Answer-Sufficiency Metric Deltas")
    md.append("| dataset | Δanswer_string_hit_avg | Δanswer_string_rank_avg | Δanswer_bearing_density_avg | Δanswer_context_early_hit_avg |")
    md.append("|---|---:|---:|---:|---:|")
    for ds in sorted(dataset_answer_delta.keys()):
        d = dataset_answer_delta[ds]
        md.append(
            "| {ds} | {hit} | {rank} | {dens} | {early} |".format(
                ds=ds,
                hit=_fmt(d.get("delta_answer_string_hit_avg"), 4),
                rank=_fmt(d.get("delta_answer_string_rank_avg"), 4),
                dens=_fmt(d.get("delta_answer_bearing_density_avg"), 4),
                early=_fmt(d.get("delta_answer_context_early_hit_avg"), 4),
            )
        )

    md.append("")
    md.append("## 7. Interpretation")
    for reason in reasons:
        md.append(f"- {reason}")

    md.append("")
    md.append("## 8. Recommendation")
    md.append(f"- {decision}")

    (out_root / "PHASE6V_SEMANTIC_SUFFICIENCY_CASE_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
