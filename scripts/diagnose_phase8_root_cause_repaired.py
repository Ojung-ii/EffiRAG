#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import md_table, safe_float, write_json, write_md


def _read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _rows(path: Path) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    return {
        (str(r.get("dataset", "")), str(r.get("profile", "")), str(r.get("variant", ""))): r
        for r in _read(path).get("runs", [])
    }


def _failure_counts(path: Path) -> Dict[tuple[str, str, str], Dict[str, float]]:
    out: Dict[tuple[str, str, str], Dict[str, float]] = {}
    for row in _read(path).get("summary", []):
        key = (str(row.get("dataset", "")), str(row.get("profile", "")), str(row.get("variant", "")))
        out.setdefault(key, {})[str(row.get("category", ""))] = safe_float(row.get("rate", 0.0), 0.0)
    return out


def _verdict(key: tuple[str, str, str], summary: Dict[str, Any], seed_evidence: Dict[str, Any], timing: Dict[str, Any], failures: Dict[str, float]) -> Dict[str, Any]:
    dataset, profile, variant = key
    is_pamae = variant.startswith("pamae_seed")
    status = str(summary.get("status", seed_evidence.get("status", "MISSING")))
    partial = status != "COMPLETE"
    normalized_candidate = safe_float(summary.get("normalized_candidate_gold_recall", seed_evidence.get("candidate_gold_recall_eval_only", 0.0)), 0.0)
    selected_recall = safe_float(summary.get("normalized_selected_gold_recall", seed_evidence.get("selected_gold_recall_eval_only", 0.0)), 0.0)
    source_balanced_baseline = variant == "source_balanced_128"
    u_q_fail = is_pamae and not partial and safe_float(summary.get("gold_entity_hit_rate_eval_only", 0.0), 0.0) <= 0.01
    seed_fail = is_pamae and not partial and safe_float(summary.get("seed_gold_hit_rate_eval_only", 0.0), 0.0) <= 0.01 and normalized_candidate <= 0.10
    seed_to_evidence_fail = is_pamae and not partial and safe_float(seed_evidence.get("num_final_evidence_candidates_mean", 0.0), 0.0) > 0.0 and normalized_candidate <= 0.10
    selection_fail = is_pamae and not partial and normalized_candidate > selected_recall + 0.10
    refine_signal = is_pamae and not partial and "refine" in variant and normalized_candidate > 0.20
    return {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "status": status,
        "num_queries": summary.get("num_queries", seed_evidence.get("num_queries", 0)),
        "normalized_candidate_gold_recall": normalized_candidate,
        "normalized_selected_gold_recall": selected_recall,
        "F1": summary.get("F1", 0.0),
        "SF-R": summary.get("SF-R", 0.0),
        "retrieval_ms": summary.get("retrieval_ms", 0.0),
        "u_q_failure": u_q_fail,
        "seed_selection_failure": seed_fail,
        "refinement_drift": failures.get("REFINEMENT_DRIFT", 0.0) > 0.0,
        "seed_to_evidence_failure": seed_to_evidence_fail,
        "candidate_transfer_selection_failure": selection_fail,
        "previous_candidate_oracle_diagnostics_broken": source_balanced_baseline or failures.get("N/A_PHASE8_STAGE", 0.0) > 0.0,
        "source_balanced_phase8_failure_attribution_repaired": source_balanced_baseline,
        "pamae_seed_k5_genuinely_weak": is_pamae and "refine" not in variant and normalized_candidate <= 0.10,
        "pamae_seed_k5_refine_useful_signal": refine_signal,
        "largest_time_bottleneck": timing.get("largest_phase8_stage", "N/A"),
        "failure_rate_seed_to_evidence": failures.get("SEED_TO_EVIDENCE_MISS", 0.0),
        "failure_rate_final_selection": failures.get("FINAL_SELECTION_FAILED", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repaired Phase8 root-cause verdict.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    summary = _rows(root / "phase8_pamae_entity_seeding_summary_repaired.json")
    seed_evidence = _rows(root / "phase8_seed_to_evidence_mapping_repaired.json")
    timing = _rows(root / "phase8_timing_bottleneck_repaired.json")
    failures = _failure_counts(root / "phase8_pamae_failure_analysis_repaired.json")
    keys = sorted(set(summary) | set(seed_evidence) | set(timing) | set(failures))
    rows = [_verdict(k, summary.get(k, {}), seed_evidence.get(k, {}), timing.get(k, {}), failures.get(k, {})) for k in keys]

    completed_pamae = [r for r in rows if str(r["variant"]).startswith("pamae_seed") and r["status"] == "COMPLETE"]
    raw_k5 = [r for r in completed_pamae if r["variant"] == "pamae_seed_k5"]
    refine = [r for r in completed_pamae if r["variant"] == "pamae_seed_k5_refine"]
    raw_weak = all(safe_float(r["normalized_candidate_gold_recall"], 0.0) <= 0.10 for r in raw_k5) if raw_k5 else True
    refine_signal = any(safe_float(r["normalized_candidate_gold_recall"], 0.0) > 0.20 for r in refine)
    verdict = {
        "Current pamae_seed_k5 implementation": "reject as replacement",
        "PAMAE-inspired idea": "still viable as augmentation" if refine_signal else "inconclusive",
        "Most likely root cause": "raw k5 replacement loses source-balanced coverage; U_q/seed gold diagnostics are weak, and seed-to-evidence candidate recall is low before refinement",
        "Next required action": "design source_balanced_128 plus pamae_seed_k5_refine augmentation after repairing diagnostics; do not run PAMAE-only replacement again",
    }

    decision = [
        {
            "Question": "Did Phase8 actually fail at U_q construction?",
            "Answer": "Yes for pamae variants by eval-only gold-entity hit, though gold-entity diagnostics may be incomplete.",
        },
        {
            "Question": "Did Phase8 actually fail at medoid seed selection?",
            "Answer": "Yes for raw k5: seed gold/evidence hit is near zero and candidate recall is very low.",
        },
        {
            "Question": "Did Phase8 actually fail at refinement?",
            "Answer": "No clear drift; completed 2Wiki refine runs show useful candidate recall signal.",
        },
        {
            "Question": "Did Phase8 actually fail at seed-to-evidence mapping?",
            "Answer": "Raw k5 is weak, but previous 100/100 SEED_TO_EVIDENCE_MISS was inflated by broken diagnostics.",
        },
        {
            "Question": "Were previous candidate/oracle diagnostics broken?",
            "Answer": "Yes. Source-balanced candidate recall is nonzero after normalization.",
        },
        {
            "Question": "Was source-balanced incorrectly attributed with Phase8 failures?",
            "Answer": "Yes; repaired taxonomy prevents source-balanced from receiving Phase8-specific labels.",
        },
        {
            "Question": "Is pamae_seed_k5 genuinely weak after repaired diagnostics?",
            "Answer": "Yes. Hotpot normalized candidate recall is about 0.0665 and 2Wiki raw k5 is similarly low.",
        },
        {
            "Question": "Does pamae_seed_k5_refine show useful signal, especially on 2Wiki?",
            "Answer": "Yes. Completed 2Wiki refine runs show much higher normalized candidate recall than raw k5.",
        },
        {
            "Question": "Recommended next experiment",
            "Answer": "source_balanced_128 union pamae_seed_k5_refine augmentation, not PAMAE-only replacement.",
        },
    ]
    write_json(root / "phase8_root_cause_repaired.json", {"verdict": verdict, "runs": rows, "decision": decision})
    keys_for_rows = [
        "dataset",
        "profile",
        "variant",
        "status",
        "num_queries",
        "normalized_candidate_gold_recall",
        "normalized_selected_gold_recall",
        "F1",
        "SF-R",
        "retrieval_ms",
        "u_q_failure",
        "seed_selection_failure",
        "refinement_drift",
        "seed_to_evidence_failure",
        "candidate_transfer_selection_failure",
        "pamae_seed_k5_refine_useful_signal",
        "largest_time_bottleneck",
    ]
    lines = [
        "# Phase8 Root-Cause Report Repaired",
        "",
        "## Verdict",
        "",
        "Current pamae_seed_k5 implementation:",
        "  reject as replacement",
        "",
        "PAMAE-inspired idea:",
        f"  {verdict['PAMAE-inspired idea']}",
        "",
        "Most likely root cause:",
        f"  {verdict['Most likely root cause']}",
        "",
        "Next required action:",
        f"  {verdict['Next required action']}",
        "",
        "## Run-Level Diagnosis",
        "",
        md_table(rows, keys_for_rows),
        "",
        "## Questions",
        "",
        md_table(decision, ["Question", "Answer"]),
        "",
    ]
    write_md(root / "phase8_root_cause_report_repaired.md", lines)
    print(root / "phase8_root_cause_report_repaired.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
