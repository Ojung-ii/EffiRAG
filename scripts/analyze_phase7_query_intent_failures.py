#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from effirag.phase7_eval_utils import canonical_support_key, coverage_stats, normalize_evidence_id, normalized_evidence_set


TARGET_STAGES = [
    "PHASE1_NOT_FOUND",
    "SEMANTIC_ANCHOR_FAILED",
    "PRUNING_FAILED",
    "FINAL_SELECTION_FAILED",
    "RENDERING_ID_MISMATCH",
    "SELECTED_NOT_SUFFICIENT",
    "QA_PROMPT_FAILED_WITH_GOLD_CONTEXT",
    "QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT",
]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = str(line or "").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


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


def _table(headers: List[str], rows: List[List[str]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    s = "| " + " | ".join(["---"] * len(headers)) + " |"
    b = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([h, s] + b)


def _fmt(v: Any, nd: int = 4) -> str:
    return f"{_safe_float(v, 0.0):.{nd}f}"


def _qid(row: Dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("qid", row.get("sample_id", ""))) or "").strip()


def _dedupe_last(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_qid: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = _qid(row)
        if qid:
            by_qid[qid] = row
    return by_qid


def _gold_ids(diag_row: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for item in list(diag_row.get("gold_supporting_facts", []) or []):
        if isinstance(item, dict):
            uid = str(item.get("unit_id", "") or "").strip()
            if not uid:
                uid = canonical_support_key(item.get("title", ""), item.get("sent_idx", -1))
            nid = normalize_evidence_id(uid)
        else:
            nid = normalize_evidence_id(item)
        if nid:
            out.add(nid)
    return out


def _oracle_query_rows_by_qid(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    payload = _read_json(run_dir / "oracle_context_results.json")
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in list(payload.get("query_rows", []) or []):
        if str(row.get("prompt_mode", "") or "") != "current_phase7":
            continue
        qid = _qid(row)
        if qid:
            grouped[qid] = dict(row)
    if grouped:
        return grouped
    # fallback to any prompt mode rows
    for row in list(payload.get("query_rows", []) or []):
        qid = _qid(row)
        if qid and qid not in grouped:
            grouped[qid] = dict(row)
    return grouped


def _rag_rows_by_qid(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "rag_query_results.jsonl"):
        qid = str(row.get("sample_id", "") or "").strip()
        if qid:
            out[qid] = row
    return out


def _map_pruning_subtype(qtrace_row: Dict[str, Any]) -> str:
    reasons = list(qtrace_row.get("phase2_drop_reasons", []) or [])
    joined = " ".join([str(x).upper() for x in reasons])
    if "INVALID" in joined or "MISSING_REQUIRED_FIELDS" in joined or "TOKENIZATION_ERROR" in joined:
        return "INVALID"
    if "DEDUP" in joined or "DUPLICATE" in joined:
        return "DEDUP"
    if "DOMINANCE" in joined or "DOMINATED" in joined:
        return "DOMINANCE"
    return "UNKNOWN"


def _classify(
    *,
    dataset: str,
    profile: str,
    variant: str,
    qid: str,
    diag_row: Dict[str, Any],
    qtrace_row: Dict[str, Any],
    oracle_row: Dict[str, Any],
    rag_row: Dict[str, Any],
) -> Dict[str, Any]:
    gold = _gold_ids(diag_row)
    p1 = normalized_evidence_set(diag_row.get("phase1_seed_ids", []) or [])
    p2 = normalized_evidence_set(diag_row.get("phase2_candidate_ids", []) or [])
    sel = normalized_evidence_set(diag_row.get("selected_evidence_ids", []) or [])
    ren = normalized_evidence_set(diag_row.get("rendered_evidence_ids", []) or [])

    p1_cov = coverage_stats(gold, p1)
    p2_cov = coverage_stats(gold, p2)
    sel_cov = coverage_stats(gold, sel)
    ren_cov = coverage_stats(gold, ren)

    selected_f1 = _safe_float(
        oracle_row.get("selected_context_f1", diag_row.get("query_f1", (rag_row.get("metrics", {}) or {}).get("f1", 0.0))),
        0.0,
    )
    gold_f1 = _safe_float(oracle_row.get("gold_support_context_f1", 0.0), 0.0)
    phase1_oracle_f1 = _safe_float(oracle_row.get("phase1_oracle_context_f1", 0.0), 0.0)
    plus_f1 = _safe_float(oracle_row.get("selected_plus_gold_context_f1", 0.0), 0.0)

    contrib = dict(qtrace_row.get("candidate_source_contribution_eval_only", {}) or {})
    sem_gold = _safe_float(contrib.get("gold_found_by_semantic", 0.0), 0.0)
    ent_gold = _safe_float(contrib.get("gold_found_by_entity_title", 0.0), 0.0)
    rel_gold = _safe_float(contrib.get("gold_found_by_relation_cue", 0.0), 0.0)
    ans_gold = _safe_float(contrib.get("gold_found_by_answer_type", 0.0), 0.0)
    graph_gold = _safe_float(contrib.get("gold_found_by_graph_flow", 0.0), 0.0)

    selected_to_rendered_match = bool(diag_row.get("selected_to_rendered_match", True))
    selected_atoms = _safe_int(qtrace_row.get("num_selected_atoms", 0), 0)
    atom_cap = _safe_int({"balanced_384_8": 8, "legacy_512_10": 10}.get(profile, 0), 0)
    gain_plus = float(plus_f1 - selected_f1)

    primary = "OTHER"
    secondary = ""
    subtype = "UNKNOWN"
    confidence = "low"
    evidence: List[str] = []

    if not gold:
        primary = "OTHER"
        subtype = "NO_GOLD_SUPPORT_LABELS"
        confidence = "low"
        evidence.append("gold_supporting_facts empty")
    elif p1_cov["gold_in_stage"] <= 0.0:
        primary = "PHASE1_NOT_FOUND"
        confidence = "high"
        evidence.append("gold absent from phase1 candidates")
        if sem_gold <= 0.0 and ent_gold <= 0.0:
            secondary = "SEMANTIC_ANCHOR_FAILED"
            subtype = "NO_SEMANTIC_OR_ENTITY_TITLE_GOLD"
            evidence.append("semantic/entity source could not recover gold")
        else:
            subtype = "PHASE1_OTHER_SOURCE_MISS"
    elif p2_cov["gold_in_stage"] <= 0.0:
        primary = "PRUNING_FAILED"
        secondary = ""
        subtype = _map_pruning_subtype(qtrace_row)
        confidence = "high"
        evidence.append("gold present in phase1 but absent by phase2/feature-ready")
    elif sel_cov["gold_in_stage"] <= 0.0:
        primary = "FINAL_SELECTION_FAILED"
        confidence = "high"
        evidence.append("gold feature-ready but not selected")
        if atom_cap > 0 and selected_atoms >= atom_cap and gain_plus > 0.10:
            secondary = "BUDGET_TOO_SMALL"
            subtype = "ATOM_CAP_SATURATED"
            evidence.append("selected atoms saturated and selected_plus_gold gain positive")
        else:
            subtype = "MARGINAL_GAIN_OR_RANKING"
    elif (not selected_to_rendered_match) or (sel_cov["gold_in_stage"] > 0.0 and ren_cov["gold_in_stage"] <= 0.0):
        primary = "RENDERING_ID_MISMATCH"
        subtype = "SELECTED_NOT_RENDERED"
        confidence = "high"
        evidence.append("selected/rendered mismatch or rendered lost selected gold")
    elif ren_cov["full_coverage"] < 1.0 and gain_plus > 0.05:
        primary = "SELECTED_NOT_SUFFICIENT"
        subtype = "PARTIAL_CONTEXT_MISSING_KEY_SUPPORT"
        confidence = "medium"
        evidence.append("selected_plus_gold improves over selected context")
    elif gold_f1 <= 0.0:
        primary = "QA_PROMPT_FAILED_WITH_GOLD_CONTEXT"
        subtype = "GOLD_CONTEXT_STILL_FAILS"
        confidence = "high"
        evidence.append("gold support context f1 <= 0")
    elif ren_cov["full_coverage"] >= 1.0 and selected_f1 < 1.0:
        primary = "QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT"
        subtype = "RENDERED_FULL_BUT_QA_WRONG"
        confidence = "medium"
        evidence.append("rendered full coverage but QA mismatch")
    else:
        primary = "OTHER"
        subtype = "UNRESOLVED"
        confidence = "low"
        evidence.append("no dominant failure signal")

    if primary == "PHASE1_NOT_FOUND" and secondary == "SEMANTIC_ANCHOR_FAILED":
        # keep explicit bucket visibility as requested
        pass

    return {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "query_id": qid,
        "question": str(diag_row.get("question", rag_row.get("question", "")) or ""),
        "gold_answer": str(diag_row.get("gold_answer", rag_row.get("answer", "")) or ""),
        "prediction": str(diag_row.get("prediction", rag_row.get("prediction", "")) or ""),
        "selected_context_f1": float(selected_f1),
        "gold_support_context_f1": float(gold_f1),
        "phase1_oracle_context_f1": float(phase1_oracle_f1),
        "selected_plus_gold_context_f1": float(plus_f1),
        "phase1_gold_recall": float(p1_cov["gold_recall"]),
        "selected_gold_recall": float(sel_cov["gold_recall"]),
        "rendered_gold_recall": float(ren_cov["gold_recall"]),
        "phase1_partial": float(p1_cov["partial_hit"]),
        "phase1_full": float(p1_cov["full_coverage"]),
        "selected_partial": float(sel_cov["partial_hit"]),
        "selected_full": float(sel_cov["full_coverage"]),
        "rendered_partial": float(ren_cov["partial_hit"]),
        "rendered_full": float(ren_cov["full_coverage"]),
        "primary_stage": primary,
        "secondary_stage": secondary,
        "subtype": subtype,
        "confidence": confidence,
        "evidence": "; ".join(evidence),
        "source_gold_counts": {
            "semantic": sem_gold,
            "entity_title": ent_gold,
            "relation_cue": rel_gold,
            "answer_type": ans_gold,
            "graph_flow": graph_gold,
        },
    }


def _stage_count_rows(records: List[Dict[str, Any]]) -> Counter:
    c = Counter()
    for row in records:
        primary = str(row.get("primary_stage", "OTHER") or "OTHER")
        c[primary] += 1
        if str(row.get("secondary_stage", "") or "") == "SEMANTIC_ANCHOR_FAILED":
            c["SEMANTIC_ANCHOR_FAILED"] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze failure shifts for query-intent graph experiment.")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--repair-diagnostics", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Missing root: {root}")

    counts_by_key: Dict[Tuple[str, str, str], Counter] = {}
    records_by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

    valid_datasets = {"hotpotqa", "2wikimultihopqa"}
    for profile_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "logs"]):
        profile = profile_dir.name
        for variant_dir in sorted([p for p in profile_dir.iterdir() if p.is_dir()]):
            variant = variant_dir.name
            for dataset_dir in sorted([p for p in variant_dir.iterdir() if p.is_dir()]):
                dataset = dataset_dir.name
                if dataset not in valid_datasets:
                    continue
                if args.repair_diagnostics:
                    diag_by_qid = _dedupe_last(_read_jsonl(dataset_dir / "phase7_diagnostics.jsonl"))
                    qtrace_by_qid = _dedupe_last(_read_jsonl(dataset_dir / "phase7_query_trace.jsonl"))
                    oracle_by_qid = _oracle_query_rows_by_qid(dataset_dir)
                    rag_by_qid = _rag_rows_by_qid(dataset_dir)
                    qids = sorted(set(diag_by_qid.keys()) | set(rag_by_qid.keys()))
                    records: List[Dict[str, Any]] = []
                    for qid in qids:
                        diag_row = dict(diag_by_qid.get(qid, {}) or {})
                        qtrace_row = dict(qtrace_by_qid.get(qid, {}) or {})
                        oracle_row = dict(oracle_by_qid.get(qid, {}) or {})
                        rag_row = dict(rag_by_qid.get(qid, {}) or {})
                        f1 = _safe_float(
                            oracle_row.get(
                                "selected_context_f1",
                                diag_row.get("query_f1", (rag_row.get("metrics", {}) or {}).get("f1", 0.0)),
                            ),
                            0.0,
                        )
                        if f1 >= 1.0:
                            continue
                        records.append(
                            _classify(
                                dataset=dataset,
                                profile=profile,
                                variant=variant,
                                qid=qid,
                                diag_row=diag_row,
                                qtrace_row=qtrace_row,
                                oracle_row=oracle_row,
                                rag_row=rag_row,
                            )
                        )
                    report = {
                        "dataset": dataset,
                        "profile": profile,
                        "variant": variant,
                        "n_failed": len(records),
                        "attributions": records,
                    }
                    (dataset_dir / "phase7_failure_attribution_report_repaired.json").write_text(
                        json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    counts = _stage_count_rows(records)
                    counts_by_key[(dataset, profile, variant)] = counts
                    records_by_key[(dataset, profile, variant)] = records
                else:
                    report_json = dataset_dir / "phase7_failure_attribution_report.json"
                    if not report_json.exists():
                        continue
                    payload = _read_json(report_json)
                    records = list(payload.get("attributions", []) or [])
                    counts = _stage_count_rows(records)
                    counts_by_key[(dataset, profile, variant)] = counts
                    records_by_key[(dataset, profile, variant)] = records

    if not counts_by_key:
        raise RuntimeError(f"No failure attribution data found under: {root}")

    rows: List[List[str]] = []
    for (dataset, profile, variant), c in sorted(counts_by_key.items(), key=lambda x: x[0]):
        n = sum(v for k, v in c.items() if k != "SEMANTIC_ANCHOR_FAILED")
        if n <= 0:
            n = sum(c.values())
        row = [dataset, profile, variant, str(n)]
        for st in TARGET_STAGES:
            row.append(str(_safe_int(c.get(st, 0), 0)))
        other = n - sum(_safe_int(c.get(st, 0), 0) for st in TARGET_STAGES)
        row.append(str(max(0, other)))
        rows.append(row)

    shift_rows: List[List[str]] = []
    by_dp: Dict[Tuple[str, str], Dict[str, Counter]] = defaultdict(dict)
    for (dataset, profile, variant), c in counts_by_key.items():
        by_dp[(dataset, profile)][variant] = c

    for (dataset, profile), variants in sorted(by_dp.items()):
        base = variants.get("baseline_bq")
        if base is None:
            continue
        for variant in ("intent_p1", "intent_p1_aq", "intent_p1_aq_rq"):
            cur = variants.get(variant)
            if cur is None:
                continue
            d_phase1 = _safe_int(cur.get("PHASE1_NOT_FOUND", 0), 0) - _safe_int(base.get("PHASE1_NOT_FOUND", 0), 0)
            d_anchor = _safe_int(cur.get("SEMANTIC_ANCHOR_FAILED", 0), 0) - _safe_int(base.get("SEMANTIC_ANCHOR_FAILED", 0), 0)
            d_final = _safe_int(cur.get("FINAL_SELECTION_FAILED", 0), 0) - _safe_int(base.get("FINAL_SELECTION_FAILED", 0), 0)
            d_sel_ins = _safe_int(cur.get("SELECTED_NOT_SUFFICIENT", 0), 0) - _safe_int(base.get("SELECTED_NOT_SUFFICIENT", 0), 0)
            d_prompt = _safe_int(cur.get("QA_PROMPT_FAILED_WITH_GOLD_CONTEXT", 0), 0) - _safe_int(base.get("QA_PROMPT_FAILED_WITH_GOLD_CONTEXT", 0), 0)
            transition_flags = []
            if d_anchor < 0 and d_final > 0:
                transition_flags.append("SEMANTIC_ANCHOR_FAILED→FINAL_SELECTION_FAILED")
            if d_final < 0 and d_sel_ins > 0:
                transition_flags.append("FINAL_SELECTION_FAILED→SELECTED_NOT_SUFFICIENT")
            if d_sel_ins < 0 and d_prompt > 0:
                transition_flags.append("SELECTED_NOT_SUFFICIENT→QA_PROMPT_FAILED")
            shift_rows.append(
                [
                    dataset,
                    profile,
                    variant,
                    str(d_phase1),
                    str(d_anchor),
                    str(d_final),
                    str(d_sel_ins),
                    str(d_prompt),
                    ", ".join(transition_flags) if transition_flags else "-",
                ]
            )

    md: List[str] = []
    md.append("# Query-Intent Failure Analysis (Repaired)" if args.repair_diagnostics else "# Query-Intent Failure Analysis")
    md.append("")
    md.append("## Failure Count Table")
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
            rows,
        )
    )
    md.append("")
    md.append("## Downstream Shift (vs baseline_bq)")
    if shift_rows:
        md.append(
            _table(
                [
                    "dataset",
                    "profile",
                    "variant",
                    "ΔPHASE1_NOT_FOUND",
                    "ΔSEMANTIC_ANCHOR_FAILED",
                    "ΔFINAL_SELECTION_FAILED",
                    "ΔSELECTED_NOT_SUFFICIENT",
                    "ΔQA_PROMPT_FAILED_WITH_GOLD_CONTEXT",
                    "transition_flags",
                ],
                shift_rows,
            )
        )
    else:
        md.append("- baseline 대비 비교 가능한 variant가 없습니다.")

    md.append("")
    md.append("## Representative Cases")
    for key in sorted(records_by_key.keys()):
        dataset, profile, variant = key
        records = records_by_key[key]
        md.append(f"### {dataset} / {profile} / {variant}")
        by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_stage[str(record.get("primary_stage", "OTHER"))].append(record)
        for stage in TARGET_STAGES + ["OTHER"]:
            items = sorted(
                list(by_stage.get(stage, []) or []),
                key=lambda r: (_safe_float(r.get("selected_context_f1", 0.0), 0.0), str(r.get("query_id", ""))),
            )[:2]
            if not items:
                continue
            md.append(f"- {stage}:")
            for ex in items:
                md.append(
                    "  - "
                    + f"{ex.get('query_id', '')} | f1={_fmt(ex.get('selected_context_f1', 0.0), 4)} "
                    + f"| gold={_fmt(ex.get('gold_support_context_f1', 0.0), 4)} "
                    + f"| plus={_fmt(ex.get('selected_plus_gold_context_f1', 0.0), 4)} "
                    + f"| subtype={ex.get('subtype', 'UNKNOWN')} | evidence={ex.get('evidence', '')}"
                )

    out_path = Path(args.out).resolve() if str(args.out or "").strip() else (root / "query_intent_failure_analysis.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")

    if args.repair_diagnostics:
        json_out = root / "query_intent_failure_analysis_repaired.json"
    else:
        json_out = root / "query_intent_failure_analysis.json"
    json_out.write_text(
        json.dumps(
            {
                "rows": rows,
                "shift_rows": shift_rows,
                "records_by_run": {
                    f"{k[0]}::{k[1]}::{k[2]}": v for k, v in records_by_key.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
