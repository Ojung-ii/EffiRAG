#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


STAGES = [
    "phase1",
    "after_invalid_filter",
    "after_dedup",
    "after_dominance",
    "feature_ready",
    "selected",
    "rendered",
]


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _fmt4(x: float) -> str:
    return f"{float(x):.4f}"


def _fmt2(x: float) -> str:
    return f"{float(x):.2f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def _merge_jsonl_files(paths: Iterable[Path], out_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(list(paths or [])):
        rows.extend(_read_jsonl(p))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _prediction_rates(preds: List[str]) -> Dict[str, float]:
    n = max(1, len(preds))
    norms = [str(p or "").strip().lower() for p in preds]
    return {
        "no_rate": float(sum(1 for p in norms if p == "no") / float(n)),
        "insufficient_rate": float(sum(1 for p in norms if p == "insufficient information") / float(n)),
    }


def _pick_oracle_mode_rows(rows: List[Dict[str, Any]], preferred_mode: str = "phase7_short") -> Dict[Tuple[str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        ds = str(r.get("dataset", "") or "")
        qid = str(r.get("query_id", "") or "")
        mode = str(r.get("prompt_mode", "") or "")
        if ds and qid and mode:
            grouped[(ds, qid)][mode] = dict(r)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, modes in grouped.items():
        if preferred_mode in modes:
            out[key] = dict(modes[preferred_mode])
        elif "current_phase7" in modes:
            out[key] = dict(modes["current_phase7"])
        elif modes:
            out[key] = dict(next(iter(modes.values())))
    return out


def _infer_primary_stage(trace: Dict[str, Any], oracle: Dict[str, Any], metric_row: Dict[str, Any]) -> Tuple[str, str, str]:
    counts = dict(trace.get("stage_counts", {}) or {})
    gold = dict(trace.get("gold_survival_eval_only", {}) or {})
    qcons = dict(trace.get("query_constraints", {}) or {})
    unit_consistency = dict(trace.get("unit_consistency", {}) or {})
    feature = dict(trace.get("feature_rank_diagnostics_eval_only", {}) or trace.get("corridor_gold_diagnostics_eval_only", {}) or {})
    pruning = dict(trace.get("pruning_diagnostics_eval_only", {}) or {})
    render_diag = dict(trace.get("rendering_diagnostics", {}) or {})

    selected_f1 = _safe_float(oracle.get("selected_context_f1", metric_row.get("f1", 0.0)), 0.0)
    gold_f1 = _safe_float(oracle.get("gold_support_context_f1", 0.0), 0.0)
    phase1_oracle_f1 = _safe_float(oracle.get("phase1_oracle_context_f1", 0.0), 0.0)
    plus_f1 = _safe_float(oracle.get("selected_plus_gold_context_f1", 0.0), 0.0)
    best_f1 = _safe_float(oracle.get("best_candidate_oracle_context_f1", 0.0), 0.0)

    if bool(unit_consistency.get("mismatch", False)):
        return "INDEX_OR_GRAPH_INVALID", "unit_consistency.mismatch=true", "high"
    if _safe_int(trace.get("num_sentence_nodes_total", 1), 1) <= 0:
        return "INDEX_OR_GRAPH_INVALID", "num_sentence_nodes_total<=0", "high"
    if _safe_int(gold.get("gold_support_count", 0), 0) > 0 and len(list(qcons.get("query_entities", []) or [])) == 0:
        return "ANCHOR_EXTRACTION_FAILED", "query_entities empty while gold supports exist", "medium"
    if _safe_int(trace.get("num_semantic_anchor_atoms", 0), 0) <= 0 and _safe_int(gold.get("gold_in_phase1", 0), 0) <= 0:
        return "SEMANTIC_ANCHOR_FAILED", "num_semantic_anchor_atoms=0 and gold_in_phase1=0", "high"
    if _safe_int(counts.get("phase1_candidates", 0), 0) <= 0 or _safe_int(gold.get("gold_in_phase1", 0), 0) <= 0:
        return "PHASE1_NOT_FOUND", "gold_in_phase1=0", "high"

    if _safe_int(gold.get("gold_after_invalid_filter", 0), 0) < _safe_int(gold.get("gold_in_phase1", 0), 0):
        return "PRUNED_INVALID", "gold dropped in safe_pruning_invalid", "high"
    if _safe_int(gold.get("gold_after_dedup", 0), 0) < _safe_int(gold.get("gold_after_invalid_filter", 0), 0):
        return "PRUNED_DEDUP", "gold dropped in safe_pruning_dedup", "high"
    if _safe_int(gold.get("gold_after_dominance", 0), 0) < _safe_int(gold.get("gold_after_dedup", 0), 0):
        return "PRUNED_DOMINANCE", "gold dropped in safe_pruning_dominance", "high"

    if _safe_int(gold.get("gold_feature_ready", 0), 0) > 0 and _safe_int(gold.get("gold_selected", 0), 0) <= 0:
        rank_a = feature.get("gold_best_rank_by_A", None)
        rank_bq = feature.get("gold_best_rank_by_Bq", None)
        gold_avg_bq = feature.get("gold_avg_Bq", None)
        dist_avg_bq = feature.get("distractor_avg_Bq", None)
        if isinstance(rank_a, int) and rank_a > 24:
            return "FEATURE_A_RANKING_FAILED", f"gold_best_rank_by_A={rank_a}", "medium"
        if (
            isinstance(rank_bq, int)
            and isinstance(gold_avg_bq, (int, float))
            and isinstance(dist_avg_bq, (int, float))
            and float(dist_avg_bq) > float(gold_avg_bq)
        ):
            return "FEATURE_BQ_RANKING_FAILED", "distractor_avg_Bq > gold_avg_Bq", "medium"
        gold_best_r = feature.get("gold_best_R", None)
        selected_avg_r = feature.get("selected_avg_R", None)
        if isinstance(gold_best_r, (int, float)) and isinstance(selected_avg_r, (int, float)) and float(gold_best_r) > float(selected_avg_r) + 0.2:
            return "REDUNDANCY_OVERPENALIZED", "gold_best_R much larger than selected_avg_R", "low"
        if _safe_int(counts.get("selected_atoms", 0), 0) >= 8 and plus_f1 > selected_f1 + 0.15:
            return "BUDGET_TOO_SMALL", "selected_plus_gold improves while selected atom count is saturated", "medium"
        return "FINAL_SELECTION_FAILED", "gold feature-ready but not selected", "high"

    if not bool(render_diag.get("selected_to_rendered_match", True)):
        return "RENDERING_ID_MISMATCH", "selected_to_rendered_match=false", "high"

    if plus_f1 > selected_f1 + 0.15:
        if _safe_int(counts.get("selected_atoms", 0), 0) >= 8:
            return "BUDGET_TOO_SMALL", "selected_plus_gold gain indicates budget bottleneck", "medium"
        return "RENDERING_CONTEXT_INSUFFICIENT", "selected_plus_gold gain indicates context insufficiency", "medium"

    if gold_f1 <= 0.0:
        return "QA_PROMPT_FAILED_WITH_GOLD_CONTEXT", "gold_support_context_f1<=0", "high"
    if phase1_oracle_f1 > selected_f1 + 0.2 or best_f1 > selected_f1 + 0.2:
        return "FINAL_SELECTION_FAILED", "oracle from candidate pool outperforms selected context", "medium"

    pred = str(metric_row.get("prediction", "") or "")
    if not pred.strip() or "\n" in pred:
        return "GENERATION_FORMAT_FAILURE", "empty/multiline prediction", "medium"
    return "UNKNOWN", "no dominant signal", "low"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Phase7 oracle-gap decomposition outputs.")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/phase7_evidence_flow/oracle_gap_decomposition/phase7_failure_attribution_report.md")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    merged_gap_trace = _merge_jsonl_files(
        [p for p in root.rglob("phase7_oracle_gap_trace.jsonl") if p.resolve() != (root / "phase7_oracle_gap_trace.jsonl").resolve()],
        root / "phase7_oracle_gap_trace.jsonl",
    )
    merged_stage_events = _merge_jsonl_files(
        [p for p in root.rglob("phase7_stage_events.jsonl") if p.resolve() != (root / "phase7_stage_events.jsonl").resolve()],
        root / "phase7_stage_events.jsonl",
    )

    oracle_replay_jsonl = root / "phase7_oracle_replay_results.jsonl"
    oracle_rows = _read_jsonl(oracle_replay_jsonl)
    if not oracle_rows:
        oracle_payload = _read_json(root / "oracle_context_results.json", {})
        oracle_rows = list(oracle_payload.get("query_rows", []) or [])
        with oracle_replay_jsonl.open("w", encoding="utf-8") as f:
            for row in oracle_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rag_rows: List[Dict[str, Any]] = []
    for p in root.rglob("rag_query_results.jsonl"):
        rag_rows.extend(_read_jsonl(p))

    metric_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rag_rows:
        ds = str(row.get("dataset", "") or "")
        qid = str(row.get("sample_id", "") or "")
        if not ds or not qid:
            continue
        metric_by_key[(ds, qid)] = {
            "dataset": ds,
            "query_id": qid,
            "question": str(row.get("question", "") or ""),
            "gold_answer": str(row.get("answer", "") or ""),
            "prediction": str(row.get("prediction", "") or ""),
            "em": _safe_float((row.get("metrics", {}) or {}).get("em", 0.0), 0.0),
            "f1": _safe_float((row.get("metrics", {}) or {}).get("f1", 0.0), 0.0),
        }

    trace_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in merged_gap_trace:
        ds = str(row.get("dataset", "") or "")
        qid = str(row.get("query_id", "") or row.get("qid", "") or "")
        if ds and qid:
            trace_by_key[(ds, qid)] = dict(row)

    oracle_by_key = _pick_oracle_mode_rows(oracle_rows, preferred_mode="phase7_short")

    all_keys = sorted(set(trace_by_key.keys()).union(set(metric_by_key.keys())).union(set(oracle_by_key.keys())))
    attribution_rows: List[Dict[str, Any]] = []
    for key in all_keys:
        ds, qid = key
        trace = dict(trace_by_key.get(key, {}) or {})
        metric_row = dict(metric_by_key.get(key, {}) or {})
        oracle = dict(oracle_by_key.get(key, {}) or {})

        sel_f1 = _safe_float(oracle.get("selected_context_f1", metric_row.get("f1", 0.0)), 0.0)
        if sel_f1 >= 1.0:
            continue
        primary, evidence, conf = _infer_primary_stage(trace=trace, oracle=oracle, metric_row=metric_row)
        attribution_rows.append(
            {
                "dataset": ds,
                "query_id": qid,
                "question": str(metric_row.get("question", trace.get("question", "")) or ""),
                "gold_answer": str(metric_row.get("gold_answer", trace.get("gold_answer", "")) or ""),
                "prediction": str(metric_row.get("prediction", "")),
                "selected_context_f1": float(sel_f1),
                "gold_support_context_f1": _safe_float(oracle.get("gold_support_context_f1", 0.0), 0.0),
                "selected_plus_gold_context_f1": _safe_float(oracle.get("selected_plus_gold_context_f1", 0.0), 0.0),
                "best_candidate_oracle_context_f1": _safe_float(oracle.get("best_candidate_oracle_context_f1", 0.0), 0.0),
                "primary_stage": str(primary),
                "secondary_stage": "",
                "evidence": str(evidence),
                "confidence": str(conf),
            }
        )

    # stage survival summary
    survival_rows: List[List[str]] = []
    by_ds = defaultdict(list)
    for row in merged_gap_trace:
        by_ds[str(row.get("dataset", "") or "")].append(row)
    for ds, rows in sorted(by_ds.items()):
        n = max(1, len(rows))
        for st in STAGES:
            if st == "phase1":
                gold_key = "gold_in_phase1"
                equiv_key = "equiv_in_phase1"
                cand_key = "phase1_candidates"
            elif st == "after_invalid_filter":
                gold_key = "gold_after_invalid_filter"
                equiv_key = "equiv_in_phase1"
                cand_key = "after_invalid_filter"
            elif st == "after_dedup":
                gold_key = "gold_after_dedup"
                equiv_key = "equiv_in_phase1"
                cand_key = "after_dedup"
            elif st == "after_dominance":
                gold_key = "gold_after_dominance"
                equiv_key = "equiv_in_phase1"
                cand_key = "after_dominance"
            elif st == "feature_ready":
                gold_key = "gold_feature_ready"
                equiv_key = "equiv_feature_ready"
                cand_key = "feature_ready_candidates"
            elif st == "selected":
                gold_key = "gold_selected"
                equiv_key = "equiv_selected"
                cand_key = "selected_atoms"
            else:
                gold_key = "gold_rendered"
                equiv_key = "equiv_rendered"
                cand_key = "rendered_atoms"

            partial = 0
            full = 0
            equiv = 0
            cvals: List[float] = []
            for r in rows:
                gold = dict(r.get("gold_survival_eval_only", {}) or {})
                eq = dict(r.get("equivalent_survival_eval_only", {}) or {})
                counts = dict(r.get("stage_counts", {}) or {})
                gtot = _safe_int(gold.get("gold_support_count", 0), 0)
                gval = _safe_int(gold.get(gold_key, 0), 0)
                eval_ = _safe_int(eq.get(equiv_key, 0), 0)
                if gval > 0:
                    partial += 1
                if gtot > 0 and gval >= gtot:
                    full += 1
                if eval_ > 0:
                    equiv += 1
                cvals.append(float(_safe_int(counts.get(cand_key, 0), 0)))
            survival_rows.append(
                [
                    ds,
                    st,
                    _fmt4(partial / float(n)),
                    _fmt4(full / float(n)),
                    _fmt4(equiv / float(n)),
                    _fmt2(sum(cvals) / float(max(1, len(cvals)))),
                ]
            )

    # oracle upper bound table
    oracle_table_rows: List[List[str]] = []
    ctx_fields = [
        "selected_context",
        "gold_support_context",
        "phase1_oracle_context",
        "feature_ready_oracle_context",
        "selected_plus_gold_context",
        "best_candidate_oracle_context",
    ]
    grouped_oracle = defaultdict(list)
    for row in oracle_rows:
        grouped_oracle[(str(row.get("dataset", "") or ""), str(row.get("prompt_mode", "") or ""))].append(row)
    for (ds, mode), rows in sorted(grouped_oracle.items()):
        for ctx in ctx_fields:
            f1s = [_safe_float(r.get(f"{ctx}_f1", 0.0), 0.0) for r in rows]
            ems = [_safe_float(r.get(f"{ctx}_em", 0.0), 0.0) for r in rows]
            preds = [str(r.get(f"{ctx}_prediction", "") or "") for r in rows]
            rates = _prediction_rates(preds)
            oracle_table_rows.append(
                [
                    ds,
                    ctx,
                    mode,
                    _fmt4(sum(ems) / float(max(1, len(ems)))),
                    _fmt4(sum(f1s) / float(max(1, len(f1s)))),
                    _fmt4(rates["no_rate"]),
                    _fmt4(rates["insufficient_rate"]),
                ]
            )

    # failure attribution summary
    cnt_by_ds_stage: Dict[str, Counter] = defaultdict(Counter)
    for row in attribution_rows:
        cnt_by_ds_stage[str(row["dataset"])][str(row["primary_stage"])] += 1
    fail_tbl_rows: List[List[str]] = []
    for ds, counter in sorted(cnt_by_ds_stage.items()):
        total = max(1, sum(counter.values()))
        for st, cnt in counter.most_common():
            fail_tbl_rows.append([ds, st, str(int(cnt)), _fmt4(cnt / float(total))])

    # bottleneck summary from trace timing
    bottleneck_rows: List[List[str]] = []
    for ds, rows in sorted(by_ds.items()):
        stage_to_vals: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            tms = dict(r.get("timing_ms", {}) or {})
            for k, v in tms.items():
                stage_to_vals[str(k)].append(_safe_float(v, 0.0))
        for stage, vals in sorted(stage_to_vals.items()):
            if not vals:
                continue
            sorted_vals = sorted(vals)
            p95_idx = int(max(0, min(len(sorted_vals) - 1, round(0.95 * (len(sorted_vals) - 1)))))
            p99_idx = int(max(0, min(len(sorted_vals) - 1, round(0.99 * (len(sorted_vals) - 1)))))
            p95 = float(sorted_vals[p95_idx])
            p99 = float(sorted_vals[p99_idx])
            flag = (
                (stage == "graph_flow" and p95 > 2000.0)
                or (stage == "corridor_extraction" and p95 > 1000.0)
                or (stage == "marginal_selection" and p95 > 500.0)
                or (stage == "rendering" and p95 > 300.0)
                or (stage == "qa_generation" and p95 > 3000.0)
            )
            bottleneck_rows.append([ds, stage, _fmt2(p95), _fmt2(p99), "true" if flag else "false"])

    summary_md = []
    summary_md.append("# Phase7 Oracle Gap Summary")
    summary_md.append("")
    summary_md.append("## Oracle Upper-Bound Table")
    summary_md.append(
        _table(
            ["dataset", "context_type", "prompt_mode", "EM", "F1", "no_rate", "insufficient_rate"],
            oracle_table_rows,
        )
    )
    summary_md.append("")
    summary_md.append("## Stage Survival Table")
    summary_md.append(
        _table(
            ["dataset", "stage", "partial_gold_hit", "full_gold_coverage", "equivalent_hit", "avg_candidates"],
            survival_rows,
        )
    )
    summary_md.append("")
    summary_md.append("## Failure Attribution Table")
    summary_md.append(_table(["dataset", "primary_stage", "count", "percentage"], fail_tbl_rows))
    summary_md.append("")
    summary_md.append("## Bottleneck Timing Table")
    summary_md.append(_table(["dataset", "stage", "p95_ms", "p99_ms", "bottleneck"], bottleneck_rows))
    (root / "phase7_oracle_gap_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    # failure attribution detail report
    fail_md = []
    fail_md.append("# Phase7 Failure Attribution Report")
    fail_md.append("")
    fail_md.append(_table(["dataset", "primary_stage", "count", "percentage"], fail_tbl_rows))
    fail_md.append("")
    fail_md.append("## Sample Cases")
    for ds in sorted(set([str(r.get("dataset", "")) for r in attribution_rows])):
        fail_md.append(f"### {ds}")
        ds_rows = [r for r in attribution_rows if str(r.get("dataset", "")) == ds]
        by_stage = defaultdict(list)
        for r in ds_rows:
            by_stage[str(r.get("primary_stage", "UNKNOWN"))].append(r)
        for stage, srows in sorted(by_stage.items()):
            fail_md.append(f"- {stage}")
            for ex in sorted(srows, key=lambda x: (float(x.get("selected_context_f1", 0.0)), str(x.get("query_id", ""))))[:3]:
                fail_md.append(
                    f"  - {ex['query_id']} | f1={_fmt4(ex['selected_context_f1'])} | gold={_fmt4(ex['gold_support_context_f1'])} "
                    f"| plus={_fmt4(ex['selected_plus_gold_context_f1'])} | evidence={ex['evidence']} | confidence={ex['confidence']}"
                )
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(fail_md), encoding="utf-8")

    # final report
    top_stage_global = "UNKNOWN"
    if attribution_rows:
        top_stage_global = Counter([str(r.get("primary_stage", "UNKNOWN")) for r in attribution_rows]).most_common(1)[0][0]

    final_md = []
    final_md.append("# Phase7 Oracle Gap Final Report")
    final_md.append("")
    final_md.append("## Diagnosis Answers")
    final_md.append(f"1. Is Phase7 mainly proposal-limited? {'yes' if top_stage_global in {'PHASE1_NOT_FOUND', 'SEMANTIC_ANCHOR_FAILED', 'ANCHOR_EXTRACTION_FAILED'} else 'partially'}")
    final_md.append(f"2. Is Phase7 mainly pruning-limited? {'yes' if top_stage_global.startswith('PRUNED_') else 'no'}")
    final_md.append(f"3. Is Phase7 mainly feature-ranking-limited? {'yes' if top_stage_global in {'FEATURE_A_RANKING_FAILED', 'FEATURE_BQ_RANKING_FAILED', 'REDUNDANCY_OVERPENALIZED'} else 'partially'}")
    final_md.append(f"4. Is Phase7 mainly final-selection-limited? {'yes' if top_stage_global == 'FINAL_SELECTION_FAILED' else 'partially'}")
    final_md.append(f"5. Is Phase7 mainly budget-limited? {'yes' if top_stage_global == 'BUDGET_TOO_SMALL' else 'partially'}")
    final_md.append(f"6. Is Phase7 mainly rendering/context-sufficiency-limited? {'yes' if top_stage_global in {'RENDERING_CONTEXT_INSUFFICIENT', 'RENDERING_ID_MISMATCH'} else 'partially'}")
    final_md.append(f"7. Is Phase7 mainly QA-prompt/generation-limited? {'yes' if top_stage_global in {'QA_PROMPT_FAILED_WITH_GOLD_CONTEXT', 'QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT', 'GENERATION_FORMAT_FAILURE'} else 'partially'}")
    final_md.append(f"8. Largest gap stage: {top_stage_global}")
    final_md.append(f"9. Next stage to improve: {top_stage_global}")
    final_md.append("10. Evidence: see failure attribution distribution and oracle upper-bound gains.")
    final_md.append("")
    final_md.append("## Recommendation Table")
    final_md.append(_table(["finding", "evidence", "recommended next action"], [[top_stage_global, "dominant failure attribution", "focus next method update on this stage"]]))
    (root / "phase7_oracle_gap_final_report.md").write_text("\n".join(final_md), encoding="utf-8")

    # structured dump
    (root / "phase7_failure_attribution_report.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "num_gap_trace_rows": len(merged_gap_trace),
                "num_stage_event_rows": len(merged_stage_events),
                "num_oracle_rows": len(oracle_rows),
                "num_failed_attributions": len(attribution_rows),
                "top_stage_global": top_stage_global,
                "attributions": attribution_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "root": str(root),
                "oracle_gap_summary": str(root / "phase7_oracle_gap_summary.md"),
                "failure_report": str(out_path),
                "final_report": str(root / "phase7_oracle_gap_final_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
