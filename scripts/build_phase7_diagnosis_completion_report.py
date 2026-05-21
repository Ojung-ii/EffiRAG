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


def _fmt4(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "0.0000"


def _fmt2(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "0.00"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def _collect_failure_counts(payload: Dict[str, Any]) -> Dict[Tuple[str, str], Counter]:
    out: Dict[Tuple[str, str], Counter] = {}
    raw = dict(payload.get("failure_counts", {}) or {})
    for k, v in raw.items():
        ds, mode = str(k), "unknown"
        if "::" in ds:
            ds, mode = ds.split("::", 1)
        out[(str(ds), str(mode))] = Counter(dict(v or {}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Phase7 diagnosis completion report.")
    ap.add_argument("--root", type=str, default="outputs/phase7_evidence_flow/qa_prompt_diagnosis")
    ap.add_argument(
        "--equivalence-json",
        type=str,
        default="outputs/phase7_evidence_flow/optimization/corridor_equivalence_report.json",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="outputs/phase7_evidence_flow/qa_prompt_diagnosis/phase7_diagnosis_completion_report.md",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    eq_json = Path(args.equivalence_json).resolve()
    out_path = Path(args.out).resolve()

    eq_payload = _read_json(eq_json, {})
    replay_summary = _read_json(root / "qa_prompt_replay_summary.json", {})
    oracle_payload = _read_json(root / "oracle_context_results.json", {})
    taxonomy_payload = _read_json(root / "failure_taxonomy_report.json", {})

    # 18.1 corridor optimization status
    eq_rows_md: List[List[str]] = []
    eq_rows_data: List[Dict[str, Any]] = []
    for ds, row in sorted(dict(eq_payload or {}).items()):
        if not isinstance(row, dict):
            continue
        sel = float(row.get("selected_id_match_rate", 0.0) or 0.0)
        rnd = float(row.get("rendered_hash_match_rate", 0.0) or 0.0)
        old_p95 = float(row.get("old_corridor_extraction_p95_ms", 0.0) or 0.0)
        new_p95 = float(row.get("new_corridor_extraction_p95_ms", 0.0) or 0.0)
        status = "PASS" if (sel >= 0.95 and rnd >= 0.95) else "FAIL"
        eq_rows_data.append(
            {
                "dataset": ds,
                "selected_id_match_rate": sel,
                "rendered_hash_match_rate": rnd,
                "old_p95_ms": old_p95,
                "new_p95_ms": new_p95,
                "status": status,
            }
        )
        eq_rows_md.append([str(ds), _fmt4(sel), _fmt4(rnd), _fmt2(old_p95), _fmt2(new_p95), status])

    # 18.2 prompt replay
    prompt_rows_md: List[List[str]] = []
    replay_datasets = dict((replay_summary.get("datasets", {}) or {}))
    for ds, payload in sorted(replay_datasets.items()):
        mode_map = dict((payload.get("prompt_modes", {}) or {}))
        for mode in ["current_phase7", "lightrag_short", "phase7_short"]:
            m = dict(mode_map.get(mode, {}) or {})
            prompt_rows_md.append(
                [
                    str(ds),
                    str(mode),
                    _fmt4(m.get("EM", 0.0)),
                    _fmt4(m.get("F1", 0.0)),
                    _fmt4(m.get("no_rate", 0.0)),
                    _fmt4(m.get("insufficient_information_rate", 0.0)),
                    _fmt4(m.get("format_violation_rate", 0.0)),
                ]
            )

    # 18.3 oracle context
    oracle_rows_md: List[List[str]] = []
    for row in list(oracle_payload.get("summary_rows", []) or []):
        oracle_rows_md.append(
            [
                str(row.get("dataset", "")),
                str(row.get("prompt_mode", "")),
                _fmt4(row.get("selected_context_F1", 0.0)),
                _fmt4(row.get("gold_support_F1", 0.0)),
                _fmt4(row.get("selected_plus_gold_F1", 0.0)),
            ]
        )

    # 18.4 failure taxonomy
    failure_counts = _collect_failure_counts(dict(taxonomy_payload or {}))
    taxonomy_rows_md: List[List[str]] = []
    agg_ds: Dict[str, Counter] = defaultdict(Counter)
    for (ds, mode), cnt in sorted(failure_counts.items()):
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
        agg_ds[ds].update(cnt)

    # 18.5 interpretation
    interp_lines: List[str] = []
    total_cnt = Counter()
    for ds, cnt in sorted(agg_ds.items()):
        total = int(sum(cnt.values()) or 0)
        if total <= 0:
            continue
        total_cnt.update(cnt)
        interp_lines.append(
            f"- {ds}: NOT_IN_PHASE1={cnt.get('NOT_IN_PHASE1',0)}, IN_PHASE1_NOT_SELECTED={cnt.get('IN_PHASE1_NOT_SELECTED',0)}, "
            f"SELECTED_NOT_SUFFICIENT={cnt.get('SELECTED_NOT_SUFFICIENT',0)}, QA_FAILED_WITH_SUFFICIENT_CONTEXT={cnt.get('QA_FAILED_WITH_SUFFICIENT_CONTEXT',0)}, "
            f"FORMAT_OR_PROMPT_FAILURE={cnt.get('FORMAT_OR_PROMPT_FAILURE',0)}."
        )

    all_fail = int(sum(total_cnt.values()) or 0)
    proposal_ratio = float(total_cnt.get("NOT_IN_PHASE1", 0) / all_fail) if all_fail > 0 else 0.0
    final_sel_ratio = float(total_cnt.get("IN_PHASE1_NOT_SELECTED", 0) / all_fail) if all_fail > 0 else 0.0
    suff_ratio = float(total_cnt.get("SELECTED_NOT_SUFFICIENT", 0) / all_fail) if all_fail > 0 else 0.0
    prompt_ratio = float(
        (total_cnt.get("QA_FAILED_WITH_SUFFICIENT_CONTEXT", 0) + total_cnt.get("FORMAT_OR_PROMPT_FAILURE", 0)) / all_fail
    ) if all_fail > 0 else 0.0

    md: List[str] = []
    md.append("# Phase7 Diagnosis Completion Report")
    md.append("")
    md.append("## 18.1 Corridor Optimization Status")
    md.append(
        _table(
            ["dataset", "selected_id_match_rate", "rendered_hash_match_rate", "old_p95_ms", "new_p95_ms", "status"],
            eq_rows_md,
        )
    )
    md.append("")
    md.append("## 18.2 Prompt Replay Result")
    md.append(
        _table(
            ["dataset", "prompt_mode", "EM", "F1", "no_rate", "insufficient_rate", "format_violation_rate"],
            prompt_rows_md,
        )
    )
    md.append("")
    md.append("## 18.3 Oracle Context Result")
    md.append(
        _table(
            ["dataset", "prompt_mode", "selected_context_F1", "gold_support_F1", "selected_plus_gold_F1"],
            oracle_rows_md,
        )
    )
    md.append("")
    md.append("## 18.4 Failure Taxonomy")
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
    md.append("## 18.5 Final Interpretation")
    md.extend(interp_lines)
    md.append(f"- Proposal-limited signal (NOT_IN_PHASE1 ratio): {_fmt4(proposal_ratio)}")
    md.append(f"- Final-selection-limited signal (IN_PHASE1_NOT_SELECTED ratio): {_fmt4(final_sel_ratio)}")
    md.append(f"- Context-sufficiency-limited signal (SELECTED_NOT_SUFFICIENT ratio): {_fmt4(suff_ratio)}")
    md.append(f"- Prompt-limited signal (QA_FAILED+FORMAT ratio): {_fmt4(prompt_ratio)}")
    md.append("- Prompt change alone explains legacy gap: likely NO if oracle gold-support F1 remains much higher than selected-context F1.")
    md.append("- Recommended next change: improve final evidence selection/context sufficiency first; keep prompt changes secondary and global.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"report": str(out_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

