#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


VARIANTS = [
    "baseline_bq",
    "source_balanced_128",
    "source_balanced_160",
    "chain_unit_selection",
]
DATASETS = ["hotpotqa", "2wikimultihopqa"]
PROFILES = ["legacy_512_10", "balanced_384_8"]


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _failure_stage(row: Dict[str, Any]) -> str:
    cand = _safe_float(row.get("candidate_sf_recall_eval_only", row.get("candidate_gold_recall", 0.0)), 0.0)
    sel = _safe_float(row.get("selected_sf_recall_eval_only", 0.0), 0.0)
    if cand <= 0.0:
        return "PHASE1_NOT_FOUND"
    if sel <= 0.0:
        return "FINAL_SELECTION_FAILED"
    if sel < 0.999999:
        return "SELECTED_NOT_SUFFICIENT"
    return "QA_PROMPT_FAILED"


def _prediction_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(run_dir / "rag_query_results.jsonl"):
        qid = str(row.get("sample_id", row.get("query_id", "")) or "")
        if qid:
            out[qid] = row
    return out


def _rows_for(run_dir: Path) -> List[Dict[str, Any]]:
    traces = _read_jsonl(run_dir / "phase7_query_trace.jsonl")
    preds = _prediction_map(run_dir)
    rows: List[Dict[str, Any]] = []
    for trace in traces:
        qid = str(trace.get("query_id", "") or "")
        pred = preds.get(qid, {})
        generation = dict(pred.get("generation", {}) or {})
        metrics = dict(pred.get("metrics", {}) or {})
        row = dict(trace)
        row["prediction"] = generation.get("prediction", pred.get("prediction", ""))
        row["gold_answer"] = pred.get("answer", pred.get("gold_answer", ""))
        row["gold_support"] = pred.get("supporting_facts", pred.get("gold_supporting_facts", []))
        row["answer_f1"] = metrics.get("f1", pred.get("f1", 0.0))
        row["failure_stage"] = "SOLVED" if _safe_float(row.get("answer_f1", 0.0), 0.0) >= 0.999999 else _failure_stage(trace)
        rows.append(row)
    return rows


def _candidate_goldish(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    goldish: List[Dict[str, Any]] = []
    selected_ids = set(str(x) for x in list(row.get("selected_evidence_ids", row.get("selected_atom_ids", [])) or []))
    for cand in list(row.get("phase7_candidate_atoms", []) or []):
        sid = str(cand.get("source_id", "") or "")
        tags = list(cand.get("source_tags", []) or [])
        if sid in selected_ids or "entity_title" in tags or "graph_flow" in tags or "anchor_neighborhood" in tags:
            goldish.append(
                {
                    "source_id": sid,
                    "title": cand.get("title", ""),
                    "source_tags": tags,
                    "text": str(cand.get("text", "") or "")[:180],
                }
            )
        if len(goldish) >= 8:
            break
    return goldish


def _selected_units(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    units = list(row.get("selected_chain_units", []) or [])
    if units:
        return units[:6]
    return [
        {
            "unit_id": item.get("unit_id", item.get("node_id", "")),
            "unit_type": item.get("unit_type", "single_atom"),
            "source_ids": item.get("unit_source_ids", []),
        }
        for item in list(row.get("selected_evidence_feature_breakdown", []) or [])[:6]
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/phase7_evidence_flow/chain_unit_selection_experiment")
    args = ap.parse_args()
    root = Path(args.root)

    all_rows: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    lines = ["# Phase7 Chain-Unit Selection Failure Analysis", ""]
    table_rows: List[List[str]] = []
    for profile in PROFILES:
        for dataset in DATASETS:
            for variant in VARIANTS:
                run_dir = root / profile / variant / dataset
                rows = _rows_for(run_dir)
                all_rows[(profile, dataset, variant)] = rows
                counts = Counter(str(r.get("failure_stage", "OTHER")) for r in rows)
                table_rows.append(
                    [
                        profile,
                        dataset,
                        variant,
                        str(sum(v for k, v in counts.items() if k != "SOLVED")),
                        str(counts.get("PHASE1_NOT_FOUND", 0)),
                        str(counts.get("FINAL_SELECTION_FAILED", 0)),
                        str(counts.get("SELECTED_NOT_SUFFICIENT", 0)),
                        str(counts.get("QA_PROMPT_FAILED", 0)),
                        str(counts.get("SOLVED", 0)),
                    ]
                )

    lines.extend(
        [
            "## Failure Counts",
            "| profile | dataset | variant | n_failed | PHASE1_NOT_FOUND | FINAL_SELECTION_FAILED | SELECTED_NOT_SUFFICIENT | QA_PROMPT_FAILED | SOLVED |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend("| " + " | ".join(row) + " |" for row in table_rows)
    lines.append("")

    lines.extend(
        [
            "## Failure Shifts",
            "| profile | dataset | variant | PHASE1_NOT_FOUND_to_FINAL_SELECTION | PHASE1_NOT_FOUND_to_SELECTED_NOT_SUFFICIENT | FINAL_SELECTION_to_SELECTED_NOT_SUFFICIENT | top_transitions |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for profile in PROFILES:
        for dataset in DATASETS:
            base = {
                str(r.get("query_id", "")): str(r.get("failure_stage", "OTHER"))
                for r in all_rows.get((profile, dataset, "baseline_bq"), [])
            }
            for variant in ("source_balanced_128", "source_balanced_160", "chain_unit_selection"):
                var = {
                    str(r.get("query_id", "")): str(r.get("failure_stage", "OTHER"))
                    for r in all_rows.get((profile, dataset, variant), [])
                }
                transitions = Counter()
                for qid, bstage in base.items():
                    if qid in var and bstage != var[qid]:
                        transitions[(bstage, var[qid])] += 1
                top = ", ".join(f"{a}->{b}:{n}" for (a, b), n in transitions.most_common(5))
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            profile,
                            dataset,
                            variant,
                            str(transitions.get(("PHASE1_NOT_FOUND", "FINAL_SELECTION_FAILED"), 0)),
                            str(transitions.get(("PHASE1_NOT_FOUND", "SELECTED_NOT_SUFFICIENT"), 0)),
                            str(transitions.get(("FINAL_SELECTION_FAILED", "SELECTED_NOT_SUFFICIENT"), 0)),
                            top or "-",
                        ]
                    )
                    + " |"
                )
    lines.append("")

    lines.append("## Representative Cases")
    for profile in PROFILES:
        for dataset in DATASETS:
            for variant in VARIANTS:
                rows = [r for r in all_rows.get((profile, dataset, variant), []) if str(r.get("failure_stage")) != "SOLVED"][:2]
                lines.append(f"### {profile} / {dataset} / {variant}")
                if not rows:
                    lines.append("- No representative failures.")
                    continue
                for row in rows:
                    feasibility = dict(row.get("candidate_chain_feasibility", {}) or {})
                    chain_diag = dict(row.get("chain_unit_diagnostics", {}) or {})
                    chain_gold = dict(row.get("chain_unit_gold_eval_only", {}) or {})
                    lines.extend(
                        [
                            f"- {row.get('failure_stage', 'OTHER')} / {row.get('query_id', '')}",
                            f"  - query: {row.get('question', '')}",
                            f"  - gold answer: {row.get('gold_answer', '')}",
                            f"  - prediction: {row.get('prediction', '')}",
                            f"  - gold support: {row.get('gold_support', [])}",
                            f"  - candidate feasibility: {feasibility}",
                            f"  - chain units: singles={chain_diag.get('num_single_units', 0)}, pairs={chain_diag.get('num_pair_units', 0)}, explicit={chain_diag.get('num_explicit_transition_units', 0)}, same_title={chain_diag.get('num_same_title_units', 0)}, carrier={chain_diag.get('num_same_carrier_units', 0)}",
                            f"  - gold unit eval: {chain_gold}",
                            f"  - candidate source tags: {_candidate_goldish(row)}",
                            f"  - selected units: {_selected_units(row)}",
                            f"  - selected evidence: {row.get('selected_evidence_ids', row.get('selected_atom_ids', []))}",
                            f"  - failure attribution: {row.get('failure_stage', 'OTHER')}",
                        ]
                    )
                lines.append("")

    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "chain_unit_selection_failure_analysis.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
