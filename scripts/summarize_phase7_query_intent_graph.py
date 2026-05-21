#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from effirag.phase7_eval_utils import canonical_support_key, coverage_stats, normalize_evidence_id, normalized_evidence_set


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


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = str(line or "").strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _mean(vals: Iterable[float]) -> float:
    items = [float(x) for x in list(vals or [])]
    if not items:
        return 0.0
    return float(sum(items) / float(len(items)))


def _sf_f1(r: float, p: float) -> float:
    rr = _safe_float(r, 0.0)
    pp = _safe_float(p, 0.0)
    if (rr + pp) <= 0.0:
        return 0.0
    return float(2.0 * rr * pp / (rr + pp))


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "null"
    return f"{_safe_float(v, 0.0):.{nd}f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    s = "| " + " | ".join(["---"] * len(headers)) + " |"
    b = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([h, s] + b)


def _qid(row: Dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("qid", row.get("sample_id", ""))) or "").strip()


def _dedupe_last(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_qid: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        qid = _qid(row)
        if not qid:
            continue
        if qid not in by_qid:
            order.append(qid)
        by_qid[qid] = row
    return [by_qid[q] for q in order if q in by_qid]


def _oracle_summary_row(run_dir: Path, dataset: str) -> Dict[str, Any]:
    payload = _read_json(run_dir / "oracle_context_results.json")
    for row in list(payload.get("summary_rows", []) or []):
        if str(row.get("dataset", "")) == str(dataset) and str(row.get("prompt_mode", "")) == "current_phase7":
            return dict(row)
    return {}


def _gold_ids_from_diag(row: Dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in list(row.get("gold_supporting_facts", []) or []):
        if isinstance(item, dict):
            uid = str(item.get("unit_id", "") or "").strip()
            if not uid:
                uid = canonical_support_key(item.get("title", ""), item.get("sent_idx", -1))
            nid = normalize_evidence_id(uid)
        else:
            nid = normalize_evidence_id(item)
        if nid:
            ids.add(nid)
    return ids


def _coverage_from_diag(diag_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = _dedupe_last(diag_rows)
    n_gold = 0
    p1_partial = 0.0
    p1_full = 0.0
    p1_recall = 0.0
    sel_partial = 0.0
    sel_full = 0.0
    sel_recall = 0.0
    ren_partial = 0.0
    ren_full = 0.0
    ren_recall = 0.0
    q_with_gold = 0
    q_no_gold = 0
    q_with_phase1 = 0
    q_with_selected = 0
    q_with_rendered = 0
    for row in rows:
        gold_ids = _gold_ids_from_diag(row)
        if not gold_ids:
            q_no_gold += 1
            continue
        q_with_gold += 1
        n_gold += 1
        p1_ids = normalized_evidence_set(row.get("phase1_seed_ids", []) or [])
        sel_ids = normalized_evidence_set(row.get("selected_evidence_ids", []) or [])
        ren_ids = normalized_evidence_set(row.get("rendered_evidence_ids", []) or [])
        if p1_ids:
            q_with_phase1 += 1
        if sel_ids:
            q_with_selected += 1
        if ren_ids:
            q_with_rendered += 1
        p1_cov = coverage_stats(gold_ids, p1_ids)
        sel_cov = coverage_stats(gold_ids, sel_ids)
        ren_cov = coverage_stats(gold_ids, ren_ids)
        p1_partial += p1_cov["partial_hit"]
        p1_full += p1_cov["full_coverage"]
        p1_recall += p1_cov["gold_recall"]
        sel_partial += sel_cov["partial_hit"]
        sel_full += sel_cov["full_coverage"]
        sel_recall += sel_cov["gold_recall"]
        ren_partial += ren_cov["partial_hit"]
        ren_full += ren_cov["full_coverage"]
        ren_recall += ren_cov["gold_recall"]
    denom = float(max(1, n_gold))
    return {
        "phase1_partial_gold_hit": float(p1_partial / denom),
        "phase1_full_gold_coverage": float(p1_full / denom),
        "phase1_gold_recall": float(p1_recall / denom),
        "selected_partial_gold_hit": float(sel_partial / denom),
        "selected_full_gold_coverage": float(sel_full / denom),
        "selected_gold_recall": float(sel_recall / denom),
        "rendered_partial_gold_hit": float(ren_partial / denom),
        "rendered_full_gold_coverage": float(ren_full / denom),
        "rendered_gold_recall": float(ren_recall / denom),
        "num_queries_with_gold_support": int(q_with_gold),
        "num_queries_missing_gold_support": int(q_no_gold),
        "num_queries_with_phase1_candidates": int(q_with_phase1),
        "num_queries_with_selected_context": int(q_with_selected),
        "num_queries_with_rendered_context": int(q_with_rendered),
    }


def _coverage_from_oracle_gap_trace(og_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = _dedupe_last(og_rows)
    n_gold = 0
    p1_partial = 0.0
    p1_full = 0.0
    p1_recall = 0.0
    sel_partial = 0.0
    sel_full = 0.0
    sel_recall = 0.0
    ren_partial = 0.0
    ren_full = 0.0
    ren_recall = 0.0
    for row in rows:
        gold = dict(row.get("gold_survival_eval_only", {}) or {})
        gtot = _safe_int(gold.get("gold_support_count", 0), 0)
        if gtot <= 0:
            continue
        n_gold += 1
        g_p1 = _safe_int(gold.get("gold_in_phase1", 0), 0)
        g_sel = _safe_int(gold.get("gold_selected", 0), 0)
        g_ren = _safe_int(gold.get("gold_rendered", 0), 0)
        p1_partial += 1.0 if g_p1 > 0 else 0.0
        p1_full += 1.0 if g_p1 >= gtot else 0.0
        p1_recall += float(g_p1 / float(gtot))
        sel_partial += 1.0 if g_sel > 0 else 0.0
        sel_full += 1.0 if g_sel >= gtot else 0.0
        sel_recall += float(g_sel / float(gtot))
        ren_partial += 1.0 if g_ren > 0 else 0.0
        ren_full += 1.0 if g_ren >= gtot else 0.0
        ren_recall += float(g_ren / float(gtot))
    denom = float(max(1, n_gold))
    return {
        "phase1_partial_gold_hit": float(p1_partial / denom),
        "phase1_full_gold_coverage": float(p1_full / denom),
        "phase1_gold_recall": float(p1_recall / denom),
        "selected_partial_gold_hit": float(sel_partial / denom),
        "selected_full_gold_coverage": float(sel_full / denom),
        "selected_gold_recall": float(sel_recall / denom),
        "rendered_partial_gold_hit": float(ren_partial / denom),
        "rendered_full_gold_coverage": float(ren_full / denom),
        "rendered_gold_recall": float(ren_recall / denom),
        "num_queries_with_gold_support": int(n_gold),
        "num_queries_missing_gold_support": 0,
        "num_queries_with_phase1_candidates": 0,
        "num_queries_with_selected_context": 0,
        "num_queries_with_rendered_context": 0,
    }


def _contrib_and_timing(qtrace_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = _dedupe_last(qtrace_rows)
    src_entity_count: List[float] = []
    src_entity_hit: List[float] = []
    src_relation_count: List[float] = []
    src_relation_hit: List[float] = []
    src_answer_count: List[float] = []
    src_answer_hit: List[float] = []
    src_graph_count: List[float] = []
    src_graph_hit: List[float] = []
    intent_entity_cov: List[float] = []
    intent_relation_cov: List[float] = []
    intent_answer_cov: List[float] = []
    t_intent: List[float] = []
    t_sem_anchor: List[float] = []
    t_local_graph: List[float] = []
    t_graph_flow: List[float] = []
    t_corridor: List[float] = []
    t_feat_aq: List[float] = []
    t_feat_rq: List[float] = []
    t_marginal: List[float] = []
    t_retrieval: List[float] = []

    for row in rows:
        contrib = dict(row.get("candidate_source_contribution_eval_only", {}) or {})
        c_entity = _safe_float(contrib.get("gold_found_by_entity_title", 0.0), 0.0)
        c_relation = _safe_float(contrib.get("gold_found_by_relation_cue", 0.0), 0.0)
        c_answer = _safe_float(contrib.get("gold_found_by_answer_type", 0.0), 0.0)
        c_graph = _safe_float(contrib.get("gold_found_by_graph_flow", 0.0), 0.0)
        src_entity_count.append(c_entity)
        src_relation_count.append(c_relation)
        src_answer_count.append(c_answer)
        src_graph_count.append(c_graph)
        src_entity_hit.append(1.0 if c_entity > 0.0 else 0.0)
        src_relation_hit.append(1.0 if c_relation > 0.0 else 0.0)
        src_answer_hit.append(1.0 if c_answer > 0.0 else 0.0)
        src_graph_hit.append(1.0 if c_graph > 0.0 else 0.0)

        for feat in list(row.get("selected_evidence_feature_breakdown", []) or []):
            intent_entity_cov.append(_safe_float(feat.get("intent_entity_coverage", 0.0), 0.0))
            intent_relation_cov.append(_safe_float(feat.get("intent_relation_cue_coverage", 0.0), 0.0))
            intent_answer_cov.append(_safe_float(feat.get("intent_answer_type_compatibility", 0.0), 0.0))

        stage_ms = dict(row.get("stage_ms", {}) or {})
        t_intent.append(
            _safe_float(stage_ms.get("query_intent_extraction", 0.0), 0.0)
            + _safe_float(stage_ms.get("intent_candidate_union", 0.0), 0.0)
        )
        t_sem_anchor.append(_safe_float(stage_ms.get("semantic_anchor_retrieval", 0.0), 0.0))
        t_local_graph.append(_safe_float(stage_ms.get("local_graph_build", 0.0), 0.0))
        t_graph_flow.append(_safe_float(stage_ms.get("graph_flow", 0.0), 0.0))
        t_corridor.append(_safe_float(stage_ms.get("corridor_extraction", 0.0), 0.0))
        t_feat_aq.append(_safe_float(stage_ms.get("feature_Aq_scoring", stage_ms.get("feature_A_scoring", 0.0)), 0.0))
        t_feat_rq.append(
            _safe_float(
                stage_ms.get(
                    "feature_Rq_scoring",
                    stage_ms.get("conditional_redundancy", stage_ms.get("feature_R_scoring", 0.0)),
                ),
                0.0,
            )
        )
        t_marginal.append(_safe_float(stage_ms.get("marginal_selection", 0.0), 0.0))
        t_retrieval.append(_safe_float(stage_ms.get("total_retrieval", 0.0), 0.0))

    return {
        "intent_entity_coverage": _mean(intent_entity_cov),
        "intent_relation_cue_coverage": _mean(intent_relation_cov),
        "intent_answer_type_coverage": _mean(intent_answer_cov),
        "entity_source_gold_count": _mean(src_entity_count),
        "entity_source_gold_hit_rate": _mean(src_entity_hit),
        "relation_source_gold_count": _mean(src_relation_count),
        "relation_source_gold_hit_rate": _mean(src_relation_hit),
        "answer_type_source_gold_count": _mean(src_answer_count),
        "answer_type_source_gold_hit_rate": _mean(src_answer_hit),
        "graph_flow_gold_count": _mean(src_graph_count),
        "graph_flow_gold_hit_rate": _mean(src_graph_hit),
        "query_intent_extraction_ms": _mean(t_intent),
        "semantic_anchor_retrieval_ms": _mean(t_sem_anchor),
        "local_graph_build_ms": _mean(t_local_graph),
        "graph_flow_ms": _mean(t_graph_flow),
        "corridor_extraction_ms": _mean(t_corridor),
        "feature_Aq_scoring_ms": _mean(t_feat_aq),
        "feature_Rq_scoring_ms": _mean(t_feat_rq),
        "marginal_selection_ms": _mean(t_marginal),
        "total_retrieval_ms_trace": _mean(t_retrieval),
    }


def _oracle_query_avg_f1(run_dir: Path, field: str) -> float:
    payload = _read_json(run_dir / "oracle_context_results.json")
    qrows = _dedupe_last(list(payload.get("query_rows", []) or []))
    vals: List[float] = []
    for row in qrows:
        vals.append(_safe_float(row.get(field, 0.0), 0.0))
    return _mean(vals)


def _failure_counts(run_dir: Path) -> Dict[str, int]:
    payload = _read_json(run_dir / "phase7_failure_attribution_report_repaired.json")
    if not payload:
        payload = _read_json(run_dir / "phase7_failure_attribution_report.json")
    attributions = list(payload.get("attributions", []) or [])
    cnt = Counter()
    for a in attributions:
        primary = str(a.get("primary_stage", "OTHER") or "OTHER")
        secondary = str(a.get("secondary_stage", "") or "")
        cnt[primary] += 1
        if secondary == "SEMANTIC_ANCHOR_FAILED":
            cnt["SEMANTIC_ANCHOR_FAILED"] += 1
    out = {
        "n_failed": int(len(attributions)),
        "PHASE1_NOT_FOUND": int(cnt.get("PHASE1_NOT_FOUND", 0)),
        "SEMANTIC_ANCHOR_FAILED": int(cnt.get("SEMANTIC_ANCHOR_FAILED", 0)),
        "PRUNING_FAILED": int(
            cnt.get("PRUNING_FAILED", 0)
            + cnt.get("PRUNED_INVALID", 0)
            + cnt.get("PRUNED_DEDUP", 0)
            + cnt.get("PRUNED_DOMINANCE", 0)
        ),
        "FINAL_SELECTION_FAILED": int(cnt.get("FINAL_SELECTION_FAILED", 0)),
        "RENDERING_ID_MISMATCH": int(cnt.get("RENDERING_ID_MISMATCH", 0)),
        "SELECTED_NOT_SUFFICIENT": int(cnt.get("SELECTED_NOT_SUFFICIENT", 0)),
        "QA_PROMPT_FAILED_WITH_GOLD_CONTEXT": int(cnt.get("QA_PROMPT_FAILED_WITH_GOLD_CONTEXT", 0)),
        "QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT": int(cnt.get("QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT", 0)),
    }
    tracked = sum(v for k, v in out.items() if k != "n_failed")
    out["OTHER"] = max(0, int(out["n_failed"] - tracked))
    return out


def _sanity_warnings(row: Dict[str, Any]) -> List[str]:
    warns: List[str] = []
    if row["phase1_full_gold_coverage"] == 0.0 and row["phase1_partial_gold_hit"] > 0.0:
        warns.append("phase1_full_all_zero_but_partial_positive")
    if row["selected_full_gold_coverage"] > row["phase1_full_gold_coverage"] + 1e-9:
        warns.append("selected_full_gt_phase1_full")
    if row["rendered_full_gold_coverage"] > row["selected_full_gold_coverage"] + 1e-9:
        warns.append("rendered_full_gt_selected_full")
    if row["selected_partial_gold_hit"] > row["phase1_partial_gold_hit"] + 1e-9:
        warns.append("selected_partial_gt_phase1_partial")
    if row.get("gold_support_context_F1", None) in (None, 0.0):
        warns.append("gold_oracle_f1_missing_or_zero")
    for key in (
        "entity_source_gold_hit_rate",
        "relation_source_gold_hit_rate",
        "answer_type_source_gold_hit_rate",
        "graph_flow_gold_hit_rate",
    ):
        if _safe_float(row.get(key, 0.0), 0.0) > 1.0 + 1e-9:
            warns.append(f"{key}_gt_1")
    return warns


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Phase7 query-intent graph experiment.")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional markdown output path. If empty, write query_intent_graph_summary.md under root.",
    )
    ap.add_argument(
        "--repair-diagnostics",
        action="store_true",
        help="When enabled, recompute coverage metrics from phase7_diagnostics.jsonl if oracle-gap trace is empty/missing.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Missing root: {root}")

    rows: List[Dict[str, Any]] = []
    global_warnings: List[str] = []
    valid_datasets = {"hotpotqa", "2wikimultihopqa"}
    for profile_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "logs"]):
        profile = profile_dir.name
        for variant_dir in sorted([p for p in profile_dir.iterdir() if p.is_dir()]):
            variant = variant_dir.name
            for dataset_dir in sorted([p for p in variant_dir.iterdir() if p.is_dir()]):
                dataset = dataset_dir.name
                if dataset not in valid_datasets:
                    continue
                rag = _read_json(dataset_dir / "rag_summary.json")
                if not rag:
                    continue
                qtrace = _read_jsonl(dataset_dir / "phase7_query_trace.jsonl")
                diag_rows = _read_jsonl(dataset_dir / "phase7_diagnostics.jsonl")
                og_rows = _read_jsonl(dataset_dir / "phase7_oracle_gap_trace.jsonl")
                oracle = _oracle_summary_row(dataset_dir, dataset=dataset)

                coverage_source = "oracle_gap_trace"
                if args.repair_diagnostics:
                    og_n = len(_dedupe_last(og_rows))
                    diag_n = len(_dedupe_last(diag_rows))
                    if og_rows and og_n >= max(1, diag_n):
                        coverage = _coverage_from_oracle_gap_trace(og_rows)
                    else:
                        coverage = _coverage_from_diag(diag_rows)
                        coverage_source = "diagnostics_fallback"
                else:
                    coverage = _coverage_from_oracle_gap_trace(og_rows)

                extras = _contrib_and_timing(qtrace)
                fails = _failure_counts(dataset_dir)

                sf_r = _safe_float(rag.get("supporting_fact_recall", 0.0), 0.0)
                sf_p = _safe_float(rag.get("supporting_fact_precision", 0.0), 0.0)
                selected_f1 = _safe_float(oracle.get("selected_context_F1", 0.0), 0.0) if oracle else _oracle_query_avg_f1(dataset_dir, "selected_context_f1")
                gold_oracle_f1 = _safe_float(oracle.get("gold_support_F1", 0.0), 0.0) if oracle else _oracle_query_avg_f1(dataset_dir, "gold_support_context_f1")
                selected_plus_gold_f1 = _safe_float(oracle.get("selected_plus_gold_F1", 0.0), 0.0) if oracle else _oracle_query_avg_f1(dataset_dir, "selected_plus_gold_context_f1")
                oracle_gap = float(gold_oracle_f1 - selected_f1)

                row: Dict[str, Any] = {
                    "dataset": dataset,
                    "profile": profile,
                    "variant": variant,
                    "n": int(_safe_int(rag.get("n_samples", 0), 0) or len(_read_jsonl(dataset_dir / "rag_query_results.jsonl"))),
                    "EM": _safe_float(rag.get("em", 0.0), 0.0),
                    "F1": _safe_float(rag.get("f1", 0.0), 0.0),
                    "SF_R": sf_r,
                    "SF_P": sf_p,
                    "SF_F1": _sf_f1(sf_r, sf_p),
                    "avg_tokens": _safe_float(rag.get("prompt_tokens_avg", 0.0), 0.0),
                    "F1_per_1k": (
                        float(_safe_float(rag.get("f1", 0.0), 0.0) / max(1.0, _safe_float(rag.get("prompt_tokens_avg", 0.0), 0.0)) * 1000.0)
                        if _safe_float(rag.get("prompt_tokens_avg", 0.0), 0.0) > 0
                        else 0.0
                    ),
                    "retrieval_ms": _safe_float(rag.get("retrieval_latency_ms", 0.0), 0.0),
                    "generation_ms": _safe_float(rag.get("total_latency_ms", 0.0), 0.0) - _safe_float(rag.get("retrieval_latency_ms", 0.0), 0.0),
                    "total_ms": _safe_float(rag.get("total_latency_ms", 0.0), 0.0),
                    "selected_context_F1": selected_f1,
                    "gold_support_context_F1": gold_oracle_f1,
                    "selected_plus_gold_F1": selected_plus_gold_f1,
                    "oracle_gap": oracle_gap,
                    "coverage_source": coverage_source,
                    "run_dir": str(dataset_dir.resolve()),
                }
                row.update(coverage)
                row.update(extras)
                row.update(fails)
                warnings = _sanity_warnings(row)
                row["sanity_warnings"] = warnings
                if warnings:
                    global_warnings.append(f"{dataset}/{profile}/{variant}: {', '.join(warnings)}")
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No runs found under: {root}")

    rows_sorted = sorted(rows, key=lambda x: (x["dataset"], x["profile"], x["variant"]))
    main_rows: List[List[str]] = []
    oracle_rows: List[List[str]] = []
    fail_rows: List[List[str]] = []
    contrib_rows: List[List[str]] = []
    timing_rows: List[List[str]] = []
    sanity_rows: List[List[str]] = []
    for r in rows_sorted:
        main_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                str(r["variant"]),
                _fmt(r["EM"], 4),
                _fmt(r["F1"], 4),
                _fmt(r["SF_R"], 4),
                _fmt(r["SF_P"], 4),
                _fmt(r["avg_tokens"], 2),
                _fmt(r["retrieval_ms"], 2),
                _fmt(r["total_ms"], 2),
            ]
        )
        oracle_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                str(r["variant"]),
                _fmt(r["phase1_partial_gold_hit"], 4),
                _fmt(r["phase1_full_gold_coverage"], 4),
                _fmt(r["phase1_gold_recall"], 4),
                _fmt(r["selected_partial_gold_hit"], 4),
                _fmt(r["selected_full_gold_coverage"], 4),
                _fmt(r["selected_gold_recall"], 4),
                _fmt(r["rendered_partial_gold_hit"], 4),
                _fmt(r["rendered_full_gold_coverage"], 4),
                _fmt(r["rendered_gold_recall"], 4),
                _fmt(r["selected_context_F1"], 4),
                _fmt(r["gold_support_context_F1"], 4),
                _fmt(r["oracle_gap"], 4),
            ]
        )
        fail_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                str(r["variant"]),
                str(int(r["n_failed"])),
                str(int(r["PHASE1_NOT_FOUND"])),
                str(int(r["SEMANTIC_ANCHOR_FAILED"])),
                str(int(r["PRUNING_FAILED"])),
                str(int(r["FINAL_SELECTION_FAILED"])),
                str(int(r["RENDERING_ID_MISMATCH"])),
                str(int(r["SELECTED_NOT_SUFFICIENT"])),
                str(int(r["QA_PROMPT_FAILED_WITH_GOLD_CONTEXT"])),
                str(int(r["QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT"])),
                str(int(r["OTHER"])),
            ]
        )
        contrib_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                str(r["variant"]),
                _fmt(r["entity_source_gold_count"], 4),
                _fmt(r["entity_source_gold_hit_rate"], 4),
                _fmt(r["relation_source_gold_count"], 4),
                _fmt(r["relation_source_gold_hit_rate"], 4),
                _fmt(r["answer_type_source_gold_count"], 4),
                _fmt(r["answer_type_source_gold_hit_rate"], 4),
                _fmt(r["graph_flow_gold_count"], 4),
                _fmt(r["graph_flow_gold_hit_rate"], 4),
            ]
        )
        timing_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                str(r["variant"]),
                _fmt(r["query_intent_extraction_ms"], 2),
                _fmt(r["semantic_anchor_retrieval_ms"], 2),
                _fmt(r["local_graph_build_ms"], 2),
                _fmt(r["corridor_extraction_ms"], 2),
                _fmt(r["feature_Aq_scoring_ms"], 2),
                _fmt(r["feature_Rq_scoring_ms"], 2),
                _fmt(r["retrieval_ms"], 2),
            ]
        )
        sanity_rows.append(
            [
                str(r["dataset"]),
                str(r["profile"]),
                str(r["variant"]),
                str(int(r["num_queries_with_gold_support"])),
                str(int(r["num_queries_missing_gold_support"])),
                str(int(r["num_queries_with_phase1_candidates"])),
                str(int(r["num_queries_with_selected_context"])),
                str(int(r["num_queries_with_rendered_context"])),
                str(r["coverage_source"]),
                ", ".join(list(r.get("sanity_warnings", []) or [])) if r.get("sanity_warnings") else "-",
            ]
        )

    md: List[str] = []
    md.append("# Query-Intent Graph Summary (Repaired)" if args.repair_diagnostics else "# Query-Intent Graph Summary")
    md.append("")
    md.append("## Main Result Table")
    md.append(
        _table(
            ["dataset", "profile", "variant", "EM", "F1", "SF-R", "SF-P", "avg_tokens", "retrieval_ms", "total_ms"],
            main_rows,
        )
    )
    md.append("")
    md.append("## Oracle Gap Table")
    md.append(
        _table(
            [
                "dataset",
                "profile",
                "variant",
                "phase1_partial",
                "phase1_full",
                "phase1_gold_recall",
                "selected_partial",
                "selected_full",
                "selected_gold_recall",
                "rendered_partial",
                "rendered_full",
                "rendered_gold_recall",
                "selected_F1",
                "gold_oracle_F1",
                "oracle_gap",
            ],
            oracle_rows,
        )
    )
    md.append("")
    md.append("## Failure Attribution Table")
    md.append(
        _table(
            [
                "dataset",
                "profile",
                "variant",
                "n_failed",
                "PHASE1_NOT_FOUND",
                "SEMANTIC_ANCHOR_FAILED",
                "PRUNING_FAILED",
                "FINAL_SELECTION_FAILED",
                "RENDERING_ID_MISMATCH",
                "SELECTED_NOT_SUFFICIENT",
                "QA_PROMPT_FAILED_WITH_GOLD_CONTEXT",
                "QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT",
                "OTHER",
            ],
            fail_rows,
        )
    )
    md.append("")
    md.append("## Candidate Source Contribution Table")
    md.append(
        _table(
            [
                "dataset",
                "profile",
                "variant",
                "entity_source_gold_count",
                "entity_source_gold_hit_rate",
                "relation_source_gold_count",
                "relation_source_gold_hit_rate",
                "answer_type_source_gold_count",
                "answer_type_source_gold_hit_rate",
                "graph_flow_gold_count",
                "graph_flow_gold_hit_rate",
            ],
            contrib_rows,
        )
    )
    md.append("")
    md.append("## Timing Table")
    md.append(
        _table(
            [
                "dataset",
                "profile",
                "variant",
                "intent_ms",
                "semantic_anchor_ms",
                "local_graph_ms",
                "corridor_ms",
                "feature_Aq_ms",
                "feature_Rq_ms",
                "retrieval_ms",
            ],
            timing_rows,
        )
    )
    md.append("")
    md.append("## Sanity & Coverage Source")
    md.append(
        _table(
            [
                "dataset",
                "profile",
                "variant",
                "num_queries_with_gold_support",
                "num_queries_missing_gold_support",
                "num_queries_with_phase1_candidates",
                "num_queries_with_selected_context",
                "num_queries_with_rendered_context",
                "coverage_source",
                "warnings",
            ],
            sanity_rows,
        )
    )
    if global_warnings:
        md.append("")
        md.append("## Warnings")
        for w in global_warnings:
            md.append(f"- {w}")

    out_md = Path(args.out).resolve() if str(args.out or "").strip() else (root / "query_intent_graph_summary.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")

    if args.repair_diagnostics:
        out_json = root / "query_intent_graph_summary_repaired.json"
    else:
        out_json = root / "query_intent_graph_summary.json"
    out_json.write_text(json.dumps({"rows": rows_sorted}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
