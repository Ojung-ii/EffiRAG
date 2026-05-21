#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


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


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in list(values or [])]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def _mean_optional(values: Iterable[Any]) -> Optional[float]:
    seq: List[float] = []
    for v in list(values or []):
        if v is None:
            continue
        try:
            seq.append(float(v))
        except Exception:
            continue
    if not seq:
        return None
    return float(sum(seq) / float(len(seq)))


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
            line = str(line or "").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _summary_from_rows(rows: List[Dict[str, Any]], rag_summary: Dict[str, Any]) -> Dict[str, Any]:
    n = int(len(rows))
    dataset = str((rows[0].get("dataset", "") if rows else rag_summary.get("dataset", "")) or "")
    variant = str((rows[0].get("phase7_variant", "") if rows else rag_summary.get("phase7_variant", "")) or "")

    em = _safe_float(rag_summary.get("em", _mean([_safe_float(r.get("query_em", 0.0), 0.0) for r in rows])), 0.0)
    f1 = _safe_float(rag_summary.get("f1", _mean([_safe_float(r.get("query_f1", 0.0), 0.0) for r in rows])), 0.0)
    sf_recall = _safe_float(
        rag_summary.get("rendered_supporting_fact_recall", rag_summary.get("supporting_fact_recall", 0.0)),
        0.0,
    )
    sf_precision = _safe_float(
        rag_summary.get("rendered_supporting_fact_precision", rag_summary.get("supporting_fact_precision", 0.0)),
        0.0,
    )

    context_tokens = [_safe_float(r.get("context_tokens", 0.0), 0.0) for r in rows]
    avg_context_tokens = _mean(context_tokens)
    f1_per_1k = float((f1 * 1000.0 / avg_context_tokens) if avg_context_tokens > 0.0 else 0.0)

    phase1_gold_hit_rate = _mean([1.0 if bool(r.get("gold_hit_phase1", False)) else 0.0 for r in rows])
    phase2_gold_hit_rate = _mean([1.0 if bool(r.get("gold_hit_phase2", False)) else 0.0 for r in rows])
    selected_gold_hit_rate = _mean([1.0 if bool(r.get("gold_hit_selected", False)) else 0.0 for r in rows])
    rendered_gold_hit_rate = _mean([1.0 if bool(r.get("gold_hit_rendered", False)) else 0.0 for r in rows])
    corridor_gold_hit_rate = _mean([1.0 if bool(r.get("corridor_gold_hit", False)) else 0.0 for r in rows])

    selected_to_rendered_match_rate = _mean(
        [1.0 if bool(r.get("selected_to_rendered_match", False)) else 0.0 for r in rows]
    )
    missing_render_text_rate = _mean([1.0 if bool(r.get("missing_render_text", False)) else 0.0 for r in rows])
    empty_context_rate = _mean([1.0 if bool(r.get("empty_context", False)) else 0.0 for r in rows])

    avg_phase1_seed_count = _mean([float(len(list(r.get("phase1_seed_ids", []) or []))) for r in rows])
    avg_phase2_candidate_count = _mean([float(len(list(r.get("phase2_candidate_ids", []) or []))) for r in rows])
    avg_selected_evidence_count = _mean([float(len(list(r.get("selected_evidence_ids", []) or []))) for r in rows])
    avg_rendered_evidence_count = _mean([float(len(list(r.get("rendered_evidence_ids", []) or []))) for r in rows])

    avg_retrieval_ms = _mean([_safe_float(r.get("retrieval_ms", 0.0), 0.0) for r in rows])
    avg_generation_ms = _mean([_safe_float(r.get("generation_ms", 0.0), 0.0) for r in rows])
    avg_total_ms = _mean([_safe_float(r.get("total_ms", 0.0), 0.0) for r in rows])
    gold_avg_bq = _mean_optional(
        [
            (r.get("corridor_gold_diagnostics_eval_only", {}) or {}).get("gold_avg_Bq", None)
            for r in rows
        ]
    )
    distractor_avg_bq = _mean_optional(
        [
            (r.get("corridor_gold_diagnostics_eval_only", {}) or {}).get("distractor_avg_Bq", None)
            for r in rows
        ]
    )
    gold_rank_by_bq = _mean_optional(
        [
            (r.get("corridor_gold_diagnostics_eval_only", {}) or {}).get("gold_best_rank_by_Bq", None)
            for r in rows
        ]
    )
    gold_rank_by_a = _mean_optional(
        [
            (r.get("corridor_gold_diagnostics_eval_only", {}) or {}).get("gold_best_rank_by_A", None)
            for r in rows
        ]
    )
    paths_found_avg = _mean_optional(
        [
            (r.get("phase7_corridor_diagnostics", {}) or {}).get("num_paths_found", None)
            for r in rows
        ]
    )
    corridor_feature_ms = [
        _safe_float((r.get("phase7_corridor_diagnostics", {}) or {}).get("corridor_feature_ms", 0.0), 0.0)
        for r in rows
    ]
    corridor_ms_p95 = float(np.percentile(corridor_feature_ms, 95)) if corridor_feature_ms else 0.0

    return {
        "dataset": dataset,
        "variant": variant,
        "n": int(n),
        "EM": float(em),
        "F1": float(f1),
        "SF_recall": float(sf_recall),
        "SF_precision": float(sf_precision),
        "avg_context_tokens": float(avg_context_tokens),
        "F1_per_1k": float(f1_per_1k),
        "phase1_gold_hit_rate": float(phase1_gold_hit_rate),
        "phase2_gold_hit_rate": float(phase2_gold_hit_rate),
        "selected_gold_hit_rate": float(selected_gold_hit_rate),
        "rendered_gold_hit_rate": float(rendered_gold_hit_rate),
        "corridor_gold_hit_rate": float(corridor_gold_hit_rate),
        "selected_to_rendered_match_rate": float(selected_to_rendered_match_rate),
        "missing_render_text_rate": float(missing_render_text_rate),
        "empty_context_rate": float(empty_context_rate),
        "avg_phase1_seed_count": float(avg_phase1_seed_count),
        "avg_phase2_candidate_count": float(avg_phase2_candidate_count),
        "avg_selected_evidence_count": float(avg_selected_evidence_count),
        "avg_rendered_evidence_count": float(avg_rendered_evidence_count),
        "avg_retrieval_ms": float(avg_retrieval_ms),
        "avg_generation_ms": float(avg_generation_ms),
        "avg_total_ms": float(avg_total_ms),
        "gold_avg_Bq": None if gold_avg_bq is None else float(gold_avg_bq),
        "distractor_avg_Bq": None if distractor_avg_bq is None else float(distractor_avg_bq),
        "gold_rank_by_Bq": None if gold_rank_by_bq is None else float(gold_rank_by_bq),
        "gold_rank_by_A": None if gold_rank_by_a is None else float(gold_rank_by_a),
        "paths_found_avg": None if paths_found_avg is None else float(paths_found_avg),
        "corridor_ms_p95": float(corridor_ms_p95),
    }


def _discover_summaries(compare_root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(compare_root.rglob("diagnostic_summary.json")):
        row = _read_json(path, {})
        if not isinstance(row, dict):
            continue
        row = dict(row)
        row["_summary_path"] = str(path.resolve())
        out.append(row)
    return out


def _write_compare_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "variant",
        "n",
        "EM",
        "F1",
        "SF_recall",
        "SF_precision",
        "avg_context_tokens",
        "F1_per_1k",
        "phase1_gold_hit_rate",
        "phase2_gold_hit_rate",
        "selected_gold_hit_rate",
        "rendered_gold_hit_rate",
        "selected_to_rendered_match_rate",
        "retrieval_ms",
        "total_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in list(rows or []):
            writer.writerow(
                {
                    "dataset": str(row.get("dataset", "") or ""),
                    "variant": str(row.get("variant", "") or ""),
                    "n": int(_safe_int(row.get("n", 0), 0)),
                    "EM": f"{_safe_float(row.get('EM', 0.0), 0.0):.4f}",
                    "F1": f"{_safe_float(row.get('F1', 0.0), 0.0):.4f}",
                    "SF_recall": f"{_safe_float(row.get('SF_recall', 0.0), 0.0):.4f}",
                    "SF_precision": f"{_safe_float(row.get('SF_precision', 0.0), 0.0):.4f}",
                    "avg_context_tokens": f"{_safe_float(row.get('avg_context_tokens', 0.0), 0.0):.2f}",
                    "F1_per_1k": f"{_safe_float(row.get('F1_per_1k', 0.0), 0.0):.4f}",
                    "phase1_gold_hit_rate": f"{_safe_float(row.get('phase1_gold_hit_rate', 0.0), 0.0):.4f}",
                    "phase2_gold_hit_rate": f"{_safe_float(row.get('phase2_gold_hit_rate', 0.0), 0.0):.4f}",
                    "selected_gold_hit_rate": f"{_safe_float(row.get('selected_gold_hit_rate', 0.0), 0.0):.4f}",
                    "rendered_gold_hit_rate": f"{_safe_float(row.get('rendered_gold_hit_rate', 0.0), 0.0):.4f}",
                    "selected_to_rendered_match_rate": f"{_safe_float(row.get('selected_to_rendered_match_rate', 0.0), 0.0):.4f}",
                    "retrieval_ms": f"{_safe_float(row.get('avg_retrieval_ms', 0.0), 0.0):.2f}",
                    "total_ms": f"{_safe_float(row.get('avg_total_ms', 0.0), 0.0):.2f}",
                }
            )


def _failure_label(row: Dict[str, Any]) -> str:
    if not bool(row.get("gold_hit_phase1", False)):
        return "no_phase1_candidate"
    if bool(row.get("gold_hit_phase2", False)) and not bool(row.get("gold_hit_selected", False)):
        return "answer_evidence_not_selected"
    if not bool(row.get("selected_to_rendered_match", True)):
        return "selected_not_rendered"
    if bool(row.get("gold_hit_phase1", False)) and not bool(row.get("gold_hit_rendered", False)):
        return "bridge_entity_miss"
    if bool(row.get("gold_hit_rendered", False)) and _safe_float(row.get("query_f1", 0.0), 0.0) <= 0.0:
        return "generation_failure"
    return "unknown"


def _failure_diagnosis(label: str) -> str:
    if label == "no_phase1_candidate":
        return "Phase I seed set에서 gold supporting fact를 회수하지 못했습니다."
    if label == "answer_evidence_not_selected":
        return "Phase II 후보에는 신호가 있지만 최종 selection 단계에서 누락되었습니다."
    if label == "selected_not_rendered":
        return "선택된 증거와 렌더링된 증거 ID 정합성이 깨졌습니다."
    if label == "bridge_entity_miss":
        return "초기 시드는 맞췄지만 bridge/조합 증거가 최종 컨텍스트까지 유지되지 않았습니다."
    if label == "generation_failure":
        return "렌더링 증거는 있으나 생성 단계에서 정답을 생성하지 못했습니다."
    return "자동 규칙으로 원인을 확정하기 어려워 수동 점검이 필요합니다."


def _format_stage_items(ids: List[Any], node_types: List[Any], scores: List[Any], texts: List[Any]) -> List[str]:
    out: List[str] = []
    n = max(len(list(ids or [])), len(list(node_types or [])), len(list(scores or [])), len(list(texts or [])))
    for i in range(n):
        sid = str(ids[i]) if i < len(ids) else ""
        ntype = str(node_types[i]) if i < len(node_types) else ""
        score = scores[i] if i < len(scores) else None
        text = str(texts[i]) if i < len(texts) else ""
        sscore = "null" if score is None else f"{_safe_float(score, 0.0):.4f}"
        text_short = " ".join(str(text or "").split())
        if len(text_short) > 300:
            text_short = text_short[:297] + "..."
        out.append(f"- {sid} | {ntype} | {sscore} | {text_short}")
    if not out:
        out.append("- (none)")
    return out


def _write_failure_cases_md(path: Path, rows: List[Dict[str, Any]], top_k: int) -> None:
    filtered: List[Dict[str, Any]] = []
    for row in list(rows or []):
        cond = (
            _safe_float(row.get("query_f1", 0.0), 0.0) <= 0.0
            or (bool(row.get("gold_hit_phase1", False)) and not bool(row.get("gold_hit_rendered", False)))
            or (bool(row.get("gold_hit_phase2", False)) and not bool(row.get("gold_hit_selected", False)))
            or (not bool(row.get("selected_to_rendered_match", True)))
        )
        if cond:
            filtered.append(row)

    def _priority(row: Dict[str, Any]) -> tuple:
        return (
            0 if not bool(row.get("selected_to_rendered_match", True)) else 1,
            0 if (bool(row.get("gold_hit_phase2", False)) and not bool(row.get("gold_hit_selected", False))) else 1,
            0 if (bool(row.get("gold_hit_phase1", False)) and not bool(row.get("gold_hit_rendered", False))) else 1,
            _safe_float(row.get("query_f1", 0.0), 0.0),
            str(row.get("qid", "")),
        )

    filtered = sorted(filtered, key=_priority)[: max(1, int(top_k))]

    lines: List[str] = []
    lines.append("# Phase7 Failure Cases")
    lines.append("")
    for row in filtered:
        qid = str(row.get("qid", "") or "")
        label = _failure_label(row)
        lines.append(f"## Case {qid}")
        lines.append("")
        lines.append("Question:")
        lines.append(str(row.get("question", "") or ""))
        lines.append("")
        lines.append("Gold answer:")
        lines.append(str(row.get("gold_answer", "") or ""))
        lines.append("")
        lines.append("Prediction:")
        lines.append(str(row.get("prediction", "") or ""))
        lines.append("")
        lines.append("Gold supporting facts:")
        gsf = list(row.get("gold_supporting_facts", []) or [])
        if gsf:
            for item in gsf:
                title = str((item or {}).get("title", "") or "")
                sent_idx = _safe_int((item or {}).get("sent_idx", 0), 0)
                unit_id = str((item or {}).get("unit_id", "") or f"{title}::{sent_idx}")
                lines.append(f"- {unit_id} ({title}, sent={sent_idx})")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("Phase I seeds:")
        lines.extend(
            _format_stage_items(
                list(row.get("phase1_seed_ids", []) or []),
                list(row.get("phase1_seed_node_types", []) or []),
                list(row.get("phase1_seed_scores", []) or []),
                list(row.get("phase1_seed_texts", []) or []),
            )
        )
        lines.append("")
        lines.append("Phase II candidates:")
        lines.extend(
            _format_stage_items(
                list(row.get("phase2_candidate_ids", []) or []),
                list(row.get("phase2_candidate_node_types", []) or []),
                list(row.get("phase2_candidate_scores", []) or []),
                list(row.get("phase2_candidate_texts", []) or []),
            )
        )
        lines.append("")
        lines.append("Selected evidence:")
        lines.extend(
            _format_stage_items(
                list(row.get("selected_evidence_ids", []) or []),
                list(row.get("selected_evidence_node_types", []) or []),
                list(row.get("selected_evidence_scores", []) or []),
                list(row.get("selected_evidence_texts", []) or []),
            )
        )
        lines.append("")
        lines.append("Rendered context:")
        rctx = str(row.get("rendered_context", "") or "")
        lines.append(rctx if rctx else "(empty)")
        lines.append("")
        lines.append("Failure label:")
        lines.append(f"- {label}")
        lines.append("")
        lines.append("Brief diagnosis:")
        lines.append(f"- {_failure_diagnosis(label)}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Phase7 diagnostics JSONL.")
    ap.add_argument("--diag-jsonl", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output diagnostic_summary.json path.")
    ap.add_argument("--compare-root", type=str, default="", help="Optional root directory to collect diagnostic summaries.")
    ap.add_argument("--compare-csv", type=str, default="", help="Optional output CSV path for variant comparison.")
    ap.add_argument("--failure-md", type=str, default="", help="Optional markdown output for failure cases.")
    ap.add_argument("--failure-top-k", type=int, default=20)
    args = ap.parse_args()

    diag_path = Path(args.diag_jsonl).resolve()
    out_path = Path(args.out).resolve()
    rows = _read_jsonl(diag_path)
    rag_summary = _read_json(diag_path.parent / "rag_summary.json", {})
    summary = _summary_from_rows(rows=rows, rag_summary=rag_summary)
    summary["diag_jsonl"] = str(diag_path)
    summary["rag_summary_path"] = str((diag_path.parent / "rag_summary.json").resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    compare_rows: List[Dict[str, Any]] = []
    if args.compare_root:
        compare_rows.extend(_discover_summaries(Path(args.compare_root).resolve()))
    else:
        compare_rows.append(summary)

    compare_rows = sorted(
        compare_rows,
        key=lambda r: (str(r.get("dataset", "")), str(r.get("variant", "")), int(_safe_int(r.get("n", 0), 0))),
    )

    if args.compare_csv:
        _write_compare_csv(Path(args.compare_csv).resolve(), compare_rows)
    if args.failure_md:
        _write_failure_cases_md(Path(args.failure_md).resolve(), rows=rows, top_k=max(1, int(args.failure_top_k)))

    print(f"[phase7-diag] summary: {out_path}")
    if args.compare_csv:
        print(f"[phase7-diag] compare_csv: {Path(args.compare_csv).resolve()}")
    if args.failure_md:
        print(f"[phase7-diag] failure_md: {Path(args.failure_md).resolve()}")


if __name__ == "__main__":
    main()
