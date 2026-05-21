#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _fmt4(x: float) -> str:
    return f"{float(x):.4f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _load_diag_by_dataset(root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    replay_summary = _read_json(root / "qa_prompt_replay_summary.json", {})
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    datasets = dict((replay_summary.get("datasets", {}) or {}))
    for ds, payload in datasets.items():
        run_dir = Path(str((payload or {}).get("source_run_dir", "") or "")).resolve()
        diag_rows = _read_jsonl(run_dir / "phase7_diagnostics.jsonl")
        out[str(ds)] = {str(r.get("qid", "") or ""): dict(r) for r in diag_rows if str(r.get("qid", "") or "")}
    return out


def _oracle_row(
    oracle_rows_by_key: Dict[Tuple[str, str, str], Dict[str, Any]],
    dataset: str,
    qid: str,
    prompt_mode: str,
) -> Dict[str, Any]:
    direct = dict(oracle_rows_by_key.get((dataset, qid, prompt_mode), {}) or {})
    if direct:
        return direct
    # Fallback to any prompt mode row for this query if requested prompt mode is unavailable.
    for (ds, oqid, _mode), row in oracle_rows_by_key.items():
        if ds == dataset and oqid == qid:
            return dict(row)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Phase7 context sufficiency failure taxonomy.")
    ap.add_argument("--root", type=str, default="outputs/phase7_evidence_flow/qa_prompt_diagnosis")
    ap.add_argument(
        "--out",
        type=str,
        default="outputs/phase7_evidence_flow/qa_prompt_diagnosis/failure_taxonomy_report.md",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    prompt_rows = _read_jsonl(root / "qa_prompt_replay_results.jsonl")
    prompt_summary = _read_json(root / "qa_prompt_replay_summary.json", {})
    oracle_payload = _read_json(root / "oracle_context_results.json", {})
    oracle_rows = list(oracle_payload.get("query_rows", []) or [])
    oracle_summary_rows = list(oracle_payload.get("summary_rows", []) or [])

    if not prompt_rows:
        raise FileNotFoundError(f"Missing prompt replay rows: {root / 'qa_prompt_replay_results.jsonl'}")

    diag_by_dataset = _load_diag_by_dataset(root)

    by_ds_mode_qid: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in prompt_rows:
        ds = str(row.get("dataset", "") or "")
        mode = str(row.get("prompt_mode", "") or "")
        qid = str(row.get("query_id", "") or "")
        by_ds_mode_qid[(ds, mode, qid)] = dict(row)

    oracle_by_ds_qid_mode: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in oracle_rows:
        ds = str(row.get("dataset", "") or "")
        qid = str(row.get("query_id", "") or "")
        mode = str(row.get("prompt_mode", "") or "")
        oracle_by_ds_qid_mode[(ds, qid, mode)] = dict(row)

    categories = [
        "NOT_IN_PHASE1",
        "IN_PHASE1_NOT_SELECTED",
        "SELECTED_NOT_SUFFICIENT",
        "QA_FAILED_WITH_SUFFICIENT_CONTEXT",
        "FORMAT_OR_PROMPT_FAILURE",
    ]

    failures_by_ds_mode: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for (ds, mode, qid), row in by_ds_mode_qid.items():
        f1 = _safe_float(row.get("f1", 0.0), 0.0)
        if f1 >= 1.0:
            continue

        diag = dict((diag_by_dataset.get(ds, {}) or {}).get(qid, {}) or {})
        oracle_row = _oracle_row(
            oracle_rows_by_key=oracle_by_ds_qid_mode,
            dataset=ds,
            qid=qid,
            prompt_mode=mode,
        )
        selected_best = _safe_float(oracle_row.get("selected_context_f1", 0.0), 0.0)
        gold_best = _safe_float(oracle_row.get("gold_support_context_f1", 0.0), 0.0)
        phase1_best = _safe_float(oracle_row.get("phase1_oracle_context_f1", 0.0), 0.0)
        plus_best = _safe_float(oracle_row.get("selected_plus_gold_context_f1", 0.0), 0.0)

        gold_hit_phase1 = bool(diag.get("gold_hit_phase1", False))
        gold_hit_rendered = bool(diag.get("gold_hit_rendered", False))
        fmt_violation = bool(row.get("format_violation", False))

        if not gold_hit_phase1:
            category = "NOT_IN_PHASE1"
        elif gold_hit_phase1 and (not gold_hit_rendered):
            category = "IN_PHASE1_NOT_SELECTED"
        elif fmt_violation:
            category = "FORMAT_OR_PROMPT_FAILURE"
        else:
            if selected_best > f1 + 1e-6:
                category = "QA_FAILED_WITH_SUFFICIENT_CONTEXT"
            elif (plus_best > f1 + 1e-6) or (phase1_best > f1 + 1e-6) or (gold_best > f1 + 1e-6):
                category = "SELECTED_NOT_SUFFICIENT"
            else:
                category = "SELECTED_NOT_SUFFICIENT"

        sublabels: List[str] = []
        pred_norm = str(row.get("prediction_norm", "") or "")
        if category == "NOT_IN_PHASE1":
            if int(diag.get("phase1_drop_count", 0) or 0) > 0:
                sublabels.append("TARGET_ENTITY_MISSING")
            else:
                sublabels.append("RELATION_CUE_MISSING")
        if category == "IN_PHASE1_NOT_SELECTED":
            corr = dict(diag.get("corridor_gold_diagnostics_eval_only", {}) or {})
            gold_bq = corr.get("gold_avg_Bq", None)
            dist_bq = corr.get("distractor_avg_Bq", None)
            if isinstance(gold_bq, (int, float)) and isinstance(dist_bq, (int, float)) and float(dist_bq) > float(gold_bq):
                sublabels.append("DISTRACTOR_HIGH_BQ")
            else:
                sublabels.append("REDUNDANCY_DROPPED_GOLD")
        if pred_norm == "no" and f1 <= 0.0:
            sublabels.append("WRONG_YES_NO_DEFAULT")
        if (plus_best - selected_best) >= 0.2:
            sublabels.append("SPARSE_CONTEXT_COREFERENCE")

        failures_by_ds_mode[(ds, mode)].append(
            {
                "query_id": qid,
                "prompt_mode": mode,
                "question": str(row.get("question", "") or ""),
                "prediction": str(row.get("prediction", "") or ""),
                "gold_answer": str(row.get("gold_answer", "") or ""),
                "baseline_f1": float(f1),
                "category": category,
                "sublabels": sublabels,
                "selected_best_f1": float(selected_best),
                "gold_best_f1": float(gold_best),
                "phase1_best_f1": float(phase1_best),
                "selected_plus_gold_best_f1": float(plus_best),
                "gold_hit_phase1": gold_hit_phase1,
                "gold_hit_rendered": gold_hit_rendered,
            }
        )

    # Aggregate tables
    taxonomy_rows_md: List[List[str]] = []
    for (ds, mode), rows in sorted(failures_by_ds_mode.items()):
        cnt = Counter([str(r.get("category", "")) for r in rows])
        taxonomy_rows_md.append(
            [
                ds,
                mode,
                str(int(cnt.get("NOT_IN_PHASE1", 0))),
                str(int(cnt.get("IN_PHASE1_NOT_SELECTED", 0))),
                str(int(cnt.get("SELECTED_NOT_SUFFICIENT", 0))),
                str(int(cnt.get("QA_FAILED_WITH_SUFFICIENT_CONTEXT", 0))),
                str(int(cnt.get("FORMAT_OR_PROMPT_FAILURE", 0))),
            ]
        )

    # prompt comparison table from replay summary
    prompt_rows_md: List[List[str]] = []
    datasets_payload = dict((prompt_summary.get("datasets", {}) or {}))
    for ds, payload in sorted(datasets_payload.items()):
        mode_map = dict((payload.get("prompt_modes", {}) or {}))
        for mode in ["current_phase7", "lightrag_short", "phase7_short"]:
            m = dict(mode_map.get(mode, {}) or {})
            prompt_rows_md.append(
                [
                    str(ds),
                    mode,
                    _fmt4(m.get("EM", 0.0)),
                    _fmt4(m.get("F1", 0.0)),
                    _fmt4(m.get("no_rate", 0.0)),
                    _fmt4(m.get("insufficient_information_rate", 0.0)),
                    _fmt4(m.get("format_violation_rate", 0.0)),
                ]
            )

    # oracle summary table
    oracle_rows_md: List[List[str]] = []
    for row in oracle_summary_rows:
        oracle_rows_md.append(
            [
                str(row.get("dataset", "")),
                str(row.get("prompt_mode", "")),
                _fmt4(row.get("selected_context_F1", 0.0)),
                _fmt4(row.get("gold_support_F1", 0.0)),
                _fmt4(row.get("phase1_oracle_F1", 0.0)),
                _fmt4(row.get("selected_plus_gold_F1", 0.0)),
            ]
        )

    md: List[str] = []
    md.append("# Phase7 Context Sufficiency Final Report")
    md.append("")
    md.append("## Prompt Comparison")
    md.append(
        _table(
            ["dataset", "prompt_mode", "EM", "F1", "no_rate", "insufficient_rate", "format_violation_rate"],
            prompt_rows_md,
        )
    )
    md.append("")
    md.append("## Oracle Context Result")
    md.append(
        _table(
            [
                "dataset",
                "prompt_mode",
                "selected_F1",
                "gold_support_F1",
                "phase1_oracle_F1",
                "selected_plus_gold_F1",
            ],
            oracle_rows_md,
        )
    )
    md.append("")
    md.append("## Failure Taxonomy")
    md.append(
        _table(
            [
                "dataset",
                "prompt_mode",
                "NOT_IN_PHASE1",
                "IN_PHASE1_NOT_SELECTED",
                "SELECTED_NOT_SUFFICIENT",
                "QA_FAILED_WITH_SUFFICIENT_CONTEXT",
                "FORMAT_OR_PROMPT_FAILURE",
            ],
            taxonomy_rows_md,
        )
    )

    md.append("")
    md.append("## Representative Cases")
    for (ds, mode), rows in sorted(failures_by_ds_mode.items()):
        md.append(f"### {ds} / {mode}")
        by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_cat[str(r.get("category", ""))].append(r)
        for cat in categories:
            items = sorted(
                list(by_cat.get(cat, []) or []),
                key=lambda r: (float(r.get("baseline_f1", 0.0)), str(r.get("query_id", ""))),
            )[:2]
            if not items:
                continue
            md.append(f"- {cat}:")
            for ex in items:
                sub = ",".join(list(ex.get("sublabels", []) or []))
                md.append(
                    f"  - {ex['query_id']} | f1={_fmt4(ex['baseline_f1'])} | selected={_fmt4(ex['selected_best_f1'])} | "
                    f"gold={_fmt4(ex['gold_best_f1'])} | plus_gold={_fmt4(ex['selected_plus_gold_best_f1'])} | sublabels={sub}"
                )

    # concise interpretation
    interp_lines: List[str] = []
    for (ds, mode), rows in sorted(failures_by_ds_mode.items()):
        cnt = Counter([str(r.get("category", "")) for r in rows])
        total = max(1, len(rows))
        top_cat, top_n = ("NONE", 0)
        if cnt:
            top_cat, top_n = cnt.most_common(1)[0]
        interp_lines.append(
            f"- {ds}/{mode}: dominant failure = {top_cat} ({top_n}/{total}, {_fmt4(top_n/float(total))})."
        )

    md.append("")
    md.append("## Final Interpretation")
    md.extend(interp_lines)
    md.append("- Prompt-only changes should be interpreted together with oracle gaps, not as retrieval replacement.")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")

    # also write canonical final report path requested in instructions
    final_report_path = root / "phase7_context_sufficiency_final_report.md"
    final_report_path.write_text("\n".join(md), encoding="utf-8")

    payload = {
        "root": str(root),
        "failure_counts": {
            f"{ds}::{mode}": Counter([r["category"] for r in rows])
            for (ds, mode), rows in failures_by_ds_mode.items()
        },
        "failures": {f"{ds}::{mode}": rows for (ds, mode), rows in failures_by_ds_mode.items()},
    }
    (root / "failure_taxonomy_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(out_path), "final_report": str(final_report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
