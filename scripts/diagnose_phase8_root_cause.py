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


def _by_key(rows: List[Dict[str, Any]]) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    return {
        (str(r.get("dataset", "")), str(r.get("profile", "")), str(r.get("variant", ""))): r
        for r in rows
    }


def _answer_bool(value: bool) -> str:
    return "yes" if value else "no"


def _verdict_for(key: tuple[str, str, str], entity: Dict[str, Any], seed: Dict[str, Any], evidence: Dict[str, Any], timing: Dict[str, Any], idnorm: Dict[str, Any]) -> Dict[str, Any]:
    dataset, profile, variant = key
    is_pamae = variant.startswith("pamae")
    num_queries = int(
        max(
            safe_float(entity.get("num_queries", 0.0), 0.0),
            safe_float(seed.get("num_queries", 0.0), 0.0),
            safe_float(evidence.get("num_queries", 0.0), 0.0),
            safe_float(timing.get("num_queries", 0.0), 0.0),
            safe_float(idnorm.get("num_queries", 0.0), 0.0),
        )
    )
    partial_run = num_queries < 10
    uq_miss = is_pamae and safe_float(entity.get("gold_entity_hit_rate_eval_only", 0.0), 0.0) <= 0.01
    seed_miss = is_pamae and safe_float(seed.get("seed_gold_hit_rate_eval_only", 0.0), 0.0) <= 0.01
    evidence_miss = is_pamae and safe_float(evidence.get("num_final_evidence_candidates_mean", 0.0), 0.0) > 0.0 and safe_float(evidence.get("candidate_gold_recall_mean_eval_only", 0.0), 0.0) <= 0.10
    selection_miss = is_pamae and safe_float(evidence.get("candidate_gold_recall_mean_eval_only", 0.0), 0.0) > 0.0 and safe_float(evidence.get("selected_pamae_source_rate", 0.0), 0.0) <= 0.01
    refinement_drift = is_pamae and safe_float(seed.get("after_seed_evidence_gold_hit_rate_eval_only", 0.0), 0.0) < safe_float(seed.get("before_seed_evidence_gold_hit_rate_eval_only", 0.0), 0.0)
    diag_broken = safe_float(idnorm.get("phase8_logging_mismatch_count", 0.0), 0.0) > 0.0
    replacement = is_pamae and "pamae" in variant and safe_float(evidence.get("selected_pamae_source_rate", 0.0), 0.0) >= 0.0
    largest_stage = str(timing.get("largest_phase8_stage", ""))
    recommendation = "Stop Phase8 and keep Phase7 source-balanced"
    if partial_run:
        recommendation = "Partial run; do not interpret as evidence"
        uq_miss = False
        seed_miss = False
        evidence_miss = False
        selection_miss = False
        refinement_drift = False
    elif diag_broken and not is_pamae:
        recommendation = "Fix diagnostic id normalization"
    elif evidence_miss:
        recommendation = "Fix seed-to-evidence mapping before any more full experiments"
    elif uq_miss and seed_miss:
        recommendation = "Fix U_q construction and gold-entity diagnostics"
    elif seed_miss:
        recommendation = "Fix seed scoring/sampling"
    elif selection_miss:
        recommendation = "Change Phase8 from replacement to source-balanced augmentation or fix selection transfer"
    return {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "num_queries": num_queries,
        "partial_run": partial_run,
        "u_q_miss": uq_miss,
        "seed_selection_miss": seed_miss,
        "refinement_drift": refinement_drift,
        "seed_to_evidence_miss": evidence_miss,
        "diagnostic_id_normalization_broken": diag_broken,
        "phase8_replaced_source_balanced": replacement,
        "generated_candidates_not_selected": selection_miss,
        "largest_time_bottleneck": largest_stage,
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate Phase8 diagnostics into a root-cause verdict.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()

    entity = _by_key(_read(root / "phase8_entity_universe_debug.json").get("runs", []))
    seed = _by_key(_read(root / "phase8_seed_quality_debug.json").get("runs", []))
    evidence = _by_key(_read(root / "phase8_seed_to_evidence_mapping.json").get("runs", []))
    timing = _by_key(_read(root / "phase8_timing_bottleneck.json").get("runs", []))
    idnorm = _by_key(_read(root / "phase8_id_normalization.json").get("runs", []))
    keys = sorted(set(entity) | set(seed) | set(evidence) | set(timing) | set(idnorm))

    verdicts = [
        _verdict_for(
            key,
            entity.get(key, {}),
            seed.get(key, {}),
            evidence.get(key, {}),
            timing.get(key, {}),
            idnorm.get(key, {}),
        )
        for key in keys
    ]

    decision_table = []
    for v in verdicts:
        if v.get("partial_run"):
            decision_table.append(
                {
                    "Condition": f"{v['dataset']}/{v['profile']}/{v['variant']} partial run",
                    "Evidence": f"Only {v['num_queries']} query rows are available.",
                    "Recommended action": "Do not use this run for method conclusions.",
                }
            )
            continue
        if v["variant"] == "source_balanced_128":
            if v["diagnostic_id_normalization_broken"]:
                decision_table.append(
                    {
                        "Condition": f"{v['dataset']}/{v['profile']} source-balanced oracle appears zero in phase8 trace",
                        "Evidence": "Phase7 candidate recall is nonzero but phase8_query_trace logs zero.",
                        "Recommended action": "Fix diagnostic id normalization/logging; do not treat source-balanced oracle as zero.",
                    }
                )
            continue
        if v["seed_to_evidence_miss"]:
            decision_table.append(
                {
                    "Condition": f"{v['dataset']}/{v['profile']} Phase8 candidates miss gold evidence",
                    "Evidence": "Final evidence candidates exist, but candidate_gold_recall remains near zero.",
                    "Recommended action": "Fix seed-to-evidence mapping; inspect entity-id to sentence/carrier joins.",
                }
            )
        if v["u_q_miss"]:
            decision_table.append(
                {
                    "Condition": f"{v['dataset']}/{v['profile']} U_q gold hit near zero",
                    "Evidence": "Large entity universe with zero eval-only gold entity hit.",
                    "Recommended action": "Check U_q construction and gold-entity diagnostic mapping.",
                }
            )
        if v["largest_time_bottleneck"]:
            decision_table.append(
                {
                    "Condition": f"{v['dataset']}/{v['profile']} runtime bottleneck",
                    "Evidence": f"Largest Phase8 stage is {v['largest_time_bottleneck']}.",
                    "Recommended action": f"Optimize {v['largest_time_bottleneck']} after correctness is repaired.",
                }
            )

    overall = {
        "current_pamae_seed_k5_implementation": "reject as mainline",
        "pamae_inspired_idea": "inconclusive",
        "most_likely_root_cause": "implementation/integration failure in seed-to-evidence mapping, compounded by diagnostic logging mismatch for candidate oracle metrics",
        "next_required_fix": "repair diagnostics first, then repair seed-to-evidence mapping; only then consider source-balanced plus PAMAE augmentation",
    }
    out = {"overall_verdict": overall, "runs": verdicts, "decision_table": decision_table}
    write_json(root / "phase8_root_cause.json", out)

    keys_for_verdict = [
        "dataset",
        "profile",
        "variant",
        "num_queries",
        "partial_run",
        "u_q_miss",
        "seed_selection_miss",
        "refinement_drift",
        "seed_to_evidence_miss",
        "diagnostic_id_normalization_broken",
        "phase8_replaced_source_balanced",
        "generated_candidates_not_selected",
        "largest_time_bottleneck",
        "recommendation",
    ]
    lines = [
        "# Phase8 Root-Cause Report",
        "",
        "## Verdict",
        "",
        "Current pamae_seed_k5 implementation:",
        "  reject as mainline.",
        "",
        "PAMAE-inspired idea:",
        "  inconclusive.",
        "",
        "Most likely root cause:",
        "  implementation/integration failure in seed-to-evidence mapping, with a separate diagnostic logging/id-normalization bug in the Phase8 summary trace.",
        "",
        "Next required fix:",
        "  repair diagnostics first, then repair seed-to-evidence mapping. Do not run more full experiments until candidate oracle metrics and Phase8 candidate transfer are trustworthy.",
        "",
        "## Run-Level Verdicts",
        "",
        md_table(verdicts, keys_for_verdict),
        "",
        "## Decision Table",
        "",
        md_table(decision_table, ["Condition", "Evidence", "Recommended action"]),
        "",
        "## Explicit Answers",
        "",
    ]
    questions = [
        ("Did Phase8 fail because U_q missed relevant entities?", any(v["u_q_miss"] for v in verdicts if v["variant"].startswith("pamae"))),
        ("Did Phase8 fail because medoid seed selection missed relevant entities?", any(v["seed_selection_miss"] for v in verdicts if v["variant"].startswith("pamae"))),
        ("Did Phase8 fail because refinement caused drift?", any(v["refinement_drift"] for v in verdicts if v["variant"].startswith("pamae"))),
        ("Did Phase8 fail because seed-to-evidence mapping failed?", any(v["seed_to_evidence_miss"] for v in verdicts if v["variant"].startswith("pamae"))),
        ("Did Phase8 fail because diagnostics/id normalization is broken?", any(v["diagnostic_id_normalization_broken"] for v in verdicts)),
        ("Did Phase8 fail because PAMAE proposal replaced source-balanced candidates?", any(v["phase8_replaced_source_balanced"] for v in verdicts if v["variant"].startswith("pamae"))),
        ("Did Phase8 fail because generated candidates were not selected?", any(v["generated_candidates_not_selected"] for v in verdicts if v["variant"].startswith("pamae"))),
    ]
    for question, answer in questions:
        lines.append(f"- {question} {_answer_bool(answer)}.")
    bottlenecks = sorted({v["largest_time_bottleneck"] for v in verdicts if v["largest_time_bottleneck"]})
    lines.append(f"- What is the largest time bottleneck? {', '.join(bottlenecks) if bottlenecks else 'unknown'}.")
    lines.append("- Should Phase8 be discarded, repaired, augmented, or postponed? Reject current PAMAE-only implementation as mainline; repair diagnostics and seed-to-evidence mapping, and consider source-balanced plus PAMAE augmentation only after that.")
    lines.append("")
    lines.extend(
        [
            "## Failure Type Distinction",
            "",
            "- Implementation failure: seed-to-evidence mapping emits many candidates but they do not cover gold/equivalent evidence.",
            "- Diagnostic failure: source-balanced candidate oracle metrics were written as zero in `phase8_query_trace` despite nonzero Phase7 candidate recall.",
            "- Method failure: not proven; PAMAE-only replacement is currently not viable, but the idea remains inconclusive because diagnostics and mapping are faulty.",
            "- Integration failure: Phase8 replaces the Phase7 source-balanced proposal path in pamae variants, so useful baseline candidates are not preserved.",
            "- Timing bottleneck: entity universe construction and evidence proposal dominate Phase8 added cost, with sampling/k-medoids also material.",
            "",
        ]
    )
    write_md(root / "phase8_root_cause_report.md", lines)
    print(root / "phase8_root_cause_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
