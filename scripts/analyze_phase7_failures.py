#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FOCUS_VARIANTS = [
    "baseline_a_minus_r",
    "corridor_a_plus_bq_minus_r",
    "corridor_conditional_r_a_plus_bq_minus_rq",
]


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


def _extract_runs(root: Path) -> List[Tuple[str, str, str, Path]]:
    out: List[Tuple[str, str, str, Path]] = []
    for diag in sorted(root.glob("*/*/*/phase7_diagnostics.jsonl")):
        try:
            profile = diag.parts[-4]
            variant = diag.parts[-3]
            dataset = diag.parts[-2]
        except Exception:
            continue
        out.append((profile, variant, dataset, diag))
    return out


def _failure_subtype(row: Dict[str, Any]) -> str:
    gold_phase1 = bool(row.get("gold_hit_phase1", False))
    gold_selected = bool(row.get("gold_hit_selected", False))
    gold_rendered = bool(row.get("gold_hit_rendered", False))
    corridor_gold = bool(row.get("corridor_gold_hit", False))
    qf1 = _safe_float(row.get("query_f1", 0.0), 0.0)

    corridor_diag = dict(row.get("corridor_gold_diagnostics_eval_only", {}) or {})
    gold_bq = corridor_diag.get("gold_avg_Bq", None)
    dis_bq = corridor_diag.get("distractor_avg_Bq", None)

    if not gold_phase1:
        return "NOT_IN_PHASE1"
    if gold_phase1 and not corridor_gold:
        return "NOT_ON_CORRIDOR"
    if corridor_gold and not gold_selected:
        if gold_bq is not None and dis_bq is not None and _safe_float(dis_bq, 0.0) > _safe_float(gold_bq, 0.0):
            return "DISTRACTOR_HIGH_BQ"
        selected_fb = list(row.get("selected_evidence_feature_breakdown", []) or [])
        if any(_safe_float(item.get("R", 0.0), 0.0) > (_safe_float(item.get("Bq", 0.0), 0.0) + 0.15) for item in selected_fb):
            return "REDUNDANCY_DROPPED_CORRIDOR_GOLD"
        return "ON_CORRIDOR_NOT_SELECTED"
    if gold_rendered and qf1 <= 0.0:
        return "QA_FAILED_WITH_EVIDENCE"
    if not gold_rendered and gold_selected:
        return "ON_CORRIDOR_NOT_SELECTED"
    return "ON_CORRIDOR_NOT_SELECTED"


def _is_failure(row: Dict[str, Any]) -> bool:
    return (
        _safe_float(row.get("query_f1", 0.0), 0.0) <= 0.0
        or (bool(row.get("gold_hit_phase1", False)) and not bool(row.get("gold_hit_rendered", False)))
        or (bool(row.get("corridor_gold_hit", False)) and not bool(row.get("gold_hit_selected", False)))
    )


def _selected_evidence_lines(row: Dict[str, Any]) -> List[str]:
    ids = list(row.get("selected_evidence_ids", []) or [])
    types = list(row.get("selected_evidence_node_types", []) or [])
    scores = list(row.get("selected_evidence_scores", []) or [])
    texts = list(row.get("selected_evidence_texts", []) or [])
    feat = list(row.get("selected_evidence_feature_breakdown", []) or [])
    feat_map = {str(x.get("node_id", "")): dict(x) for x in feat}
    out: List[str] = []
    n = max(len(ids), len(types), len(scores), len(texts))
    for i in range(n):
        sid = str(ids[i]) if i < len(ids) else ""
        ntype = str(types[i]) if i < len(types) else ""
        s = scores[i] if i < len(scores) else None
        text = str(texts[i]) if i < len(texts) else ""
        text = " ".join(text.split())
        if len(text) > 240:
            text = text[:237] + "..."
        fb = feat_map.get(sid, {})
        out.append(
            "- "
            + f"{sid} | {ntype} | score={_safe_float(s, 0.0):.4f} | "
            + f"A={_safe_float(fb.get('A', 0.0), 0.0):.3f}, Bq={_safe_float(fb.get('Bq', 0.0), 0.0):.3f}, "
            + f"R={_safe_float(fb.get('R', 0.0), 0.0):.3f}, Rq={_safe_float(fb.get('Rq', 0.0), 0.0):.3f}, "
            + f"final={_safe_float(fb.get('final_gain', 0.0), 0.0):.3f} | {text}"
        )
    if not out:
        out.append("- (none)")
    return out


def _sort_key(row: Dict[str, Any]) -> Tuple:
    subtype = _failure_subtype(row)
    subtype_rank = {
        "NOT_IN_PHASE1": 0,
        "NOT_ON_CORRIDOR": 1,
        "ON_CORRIDOR_NOT_SELECTED": 2,
        "REDUNDANCY_DROPPED_CORRIDOR_GOLD": 3,
        "DISTRACTOR_HIGH_BQ": 4,
        "QA_FAILED_WITH_EVIDENCE": 5,
    }.get(subtype, 9)
    return (
        subtype_rank,
        _safe_float(row.get("query_f1", 0.0), 0.0),
        str(row.get("qid", "")),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Phase7 corridor experiment failures.")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    runs = _extract_runs(root)
    if not runs:
        raise FileNotFoundError(f"No phase7_diagnostics.jsonl found under: {root}")

    lines: List[str] = []
    subtype_dist: Dict[str, Counter] = defaultdict(Counter)

    lines.append("# Phase7 Corridor-Bq Failure Case Analysis")
    lines.append("")

    for profile, variant, dataset, diag_path in runs:
        if variant not in FOCUS_VARIANTS:
            continue
        rows = _read_jsonl(diag_path)
        failures = [r for r in rows if _is_failure(r)]
        failures = sorted(failures, key=_sort_key)[: max(1, int(args.top_k))]

        lines.append(f"## {profile} | {variant} | {dataset}")
        lines.append("")
        lines.append(f"- total_queries: {len(rows)}")
        lines.append(f"- selected_failures: {len(failures)}")
        lines.append("")

        for row in failures:
            subtype = _failure_subtype(row)
            subtype_dist[f"{profile}/{variant}/{dataset}"][subtype] += 1
            qid = str(row.get("qid", ""))
            corridor_diag = dict(row.get("corridor_gold_diagnostics_eval_only", {}) or {})
            lines.append(f"### Case {qid}")
            lines.append("")
            lines.append(f"- failure_subtype: {subtype}")
            lines.append(f"- question: {str(row.get('question', '') or '').strip()}")
            lines.append(f"- gold_answer: {str(row.get('gold_answer', '') or '').strip()}")
            lines.append(f"- prediction: {str(row.get('prediction', '') or '').strip()}")
            lines.append(f"- phase1_gold_present: {bool(row.get('gold_hit_phase1', False))}")
            lines.append(f"- gold_on_corridor: {bool(row.get('corridor_gold_hit', False))}")
            lines.append(f"- selected_gold_present: {bool(row.get('gold_hit_selected', False))}")
            lines.append(f"- rendered_gold_present: {bool(row.get('gold_hit_rendered', False))}")
            lines.append(f"- selected_to_rendered_match: {bool(row.get('selected_to_rendered_match', False))}")
            lines.append(
                "- gold_best_rank_by_A/Bq/final: "
                + f"{corridor_diag.get('gold_best_rank_by_A', None)} / "
                + f"{corridor_diag.get('gold_best_rank_by_Bq', None)} / "
                + f"{corridor_diag.get('gold_best_rank_by_final_score', None)}"
            )
            lines.append(
                "- gold_avg_Bq_vs_distractor_avg_Bq: "
                + f"{corridor_diag.get('gold_avg_Bq', None)} vs {corridor_diag.get('distractor_avg_Bq', None)}"
            )
            lines.append("- gold_support_facts:")
            gsf = list(row.get("gold_supporting_facts", []) or [])
            if gsf:
                for sf in gsf:
                    lines.append(
                        f"  - {str(sf.get('unit_id', '') or '')}"
                    )
            else:
                lines.append("  - (none)")
            lines.append("- selected_evidence:")
            for item in _selected_evidence_lines(row):
                lines.append(f"  {item}")
            lines.append("")

    lines.append("## Failure Subtype Distribution")
    lines.append("")
    for key in sorted(subtype_dist.keys()):
        lines.append(f"### {key}")
        for subtype, cnt in subtype_dist[key].most_common():
            lines.append(f"- {subtype}: {cnt}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    dist_json = out_path.with_suffix(".json")
    payload = {k: dict(v) for k, v in subtype_dist.items()}
    dist_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[phase7-failures] {out_path}")
    print(f"[phase7-failures] {dist_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
