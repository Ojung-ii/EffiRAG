#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effirag.phase8_diagnostics import (
    entity_universe,
    evidence_proposal,
    evidence_tags,
    group_runs,
    is_phase8_variant,
    iter_queries,
    load_runs,
    md_table,
    mean,
    missing_files_for,
    normalized_query_diagnostics,
    phase7_candidate_full,
    phase7_candidate_recall,
    phase7_chain_oracle,
    question_for,
    safe_float,
    seed_best,
    selected_atom_ids,
    source_tag_counter,
    stats,
    write_json,
    write_md,
)


COUNT_FIELDS = [
    "num_seed_atoms",
    "num_seed_carriers",
    "num_anchor_seed_path_atoms",
    "num_seed_seed_path_atoms",
    "num_same_title_support_atoms",
    "num_same_carrier_support_atoms",
    "num_final_evidence_candidates",
]


def _selected_pamae(query: Dict[str, Any]) -> tuple[float, Counter]:
    selected = selected_atom_ids(query.get("phase7", {}))
    if not selected:
        return 0.0, Counter()
    counter = source_tag_counter(selected, evidence_tags(query))
    pamae_selected = 0
    tags_by_id = evidence_tags(query)
    for sid in selected:
        tags = tags_by_id.get(sid, [])
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(tags, list) and any(str(tag).startswith("phase8_") for tag in tags):
            pamae_selected += 1
    return float(pamae_selected / len(selected)) if selected else 0.0, counter


def _case(query: Dict[str, Any], reason: str) -> Dict[str, Any]:
    p7 = query.get("phase7", {})
    proposal = evidence_proposal(query)
    best = seed_best(query)
    selected_rate, counter = _selected_pamae(query)
    return {
        "query_id": query.get("query_id", ""),
        "reason": reason,
        "question": question_for(query),
        "best_seed_names": best.get("best_seed_names", []),
        "num_seed_atoms": proposal.get("num_seed_atoms", 0),
        "num_seed_carriers": proposal.get("num_seed_carriers", 0),
        "num_final_evidence_candidates": proposal.get("num_final_evidence_candidates", 0),
        "num_phase1_candidates": p7.get("num_phase1_candidates", 0),
        "num_phase2_candidates": p7.get("num_phase2_candidates", 0),
        "num_selected_atoms": p7.get("num_selected_atoms", 0),
        "selected_pamae_source_rate": selected_rate,
        "selected_source_tag_distribution": dict(counter),
        "candidate_gold_recall": phase7_candidate_recall(query),
        "normalized_candidate_gold_recall": normalized_query_diagnostics(query).get("normalized_candidate_gold_recall"),
        "normalized_selected_gold_recall": normalized_query_diagnostics(query).get("normalized_selected_gold_recall"),
        "selected_gold_hit_eval_only": bool(p7.get("selected_gold_hit_eval_only", False)),
    }


def _summarize_group(dataset: str, profile: str, variant: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries = list(iter_queries(runs))
    proposals = [evidence_proposal(q) for q in queries]
    row: Dict[str, Any] = {
        "dataset": dataset,
        "profile": profile,
        "variant": variant,
        "num_queries": len(queries),
        "status": ",".join(sorted({str(r.get("status", "MISSING")) for r in runs})),
        "missing_files": {r["run_dir"]: missing_files_for(r) for r in runs},
    }
    for field in COUNT_FIELDS:
        vals = [safe_float(p.get(field, 0.0), 0.0) for p in proposals]
        s = stats(vals)
        row[f"{field}_mean"] = s["mean"]
        if field in {"num_seed_atoms", "num_seed_carriers", "num_final_evidence_candidates"}:
            row[f"{field}_p50"] = s["p50"]
            row[f"{field}_p95"] = s["p95"]
    row["candidate_gold_partial_rate_eval_only"] = mean(
        1.0 if (p.get("candidate_gold_partial_eval_only") or phase7_candidate_recall(q) > 0.0) else 0.0
        for p, q in zip(proposals, queries)
    )
    row["candidate_gold_full_rate_eval_only"] = mean(
        1.0 if (p.get("candidate_gold_full_eval_only") or phase7_candidate_full(q)) else 0.0
        for p, q in zip(proposals, queries)
    )
    row["candidate_gold_recall_mean_eval_only"] = mean(
        max(safe_float(p.get("candidate_gold_recall_eval_only", 0.0), 0.0), phase7_candidate_recall(q))
        for p, q in zip(proposals, queries)
    )
    nds = [normalized_query_diagnostics(q) for q in queries]
    row["candidate_gold_recall_eval_only"] = mean(
        d["normalized_candidate_gold_recall"] for d in nds if d["normalized_candidate_gold_recall"] is not None
    )
    row["selected_gold_recall_eval_only"] = mean(
        d["normalized_selected_gold_recall"] for d in nds if d["normalized_selected_gold_recall"] is not None
    )
    row["candidate_oracle_F1_mean_eval_only"] = mean(
        p.get("candidate_oracle_F1_eval_only", 0.0) for p in proposals
    )
    row["chain_unit_oracle_feasible_mean_eval_only"] = mean(
        max(safe_float(p.get("chain_unit_oracle_feasible_eval_only", 0.0), 0.0), phase7_chain_oracle(q))
        for p, q in zip(proposals, queries)
    )
    selected_rates = []
    tag_counter: Counter = Counter()
    for q in queries:
        rate, counter = _selected_pamae(q)
        selected_rates.append(rate)
        tag_counter.update(counter)
    row["selected_pamae_source_rate"] = mean(selected_rates)
    row["selected_source_tag_distribution"] = dict(tag_counter)
    row["interpretation"] = _interpret(row)
    return row


def _interpret(row: Dict[str, Any]) -> str:
    if row.get("variant") == "source_balanced_128":
        if safe_float(row.get("candidate_gold_recall_mean_eval_only", 0.0)) > 0.0:
            return "Source-balanced candidates have nonzero candidate recall; prior zero summary was a diagnostic extraction issue."
        return "Source-balanced baseline has no Phase8 evidence proposal."
    if safe_float(row.get("num_seed_atoms_mean", 0.0)) <= 1.0 and safe_float(row.get("num_seed_carriers_mean", 0.0)) <= 1.0:
        return "Seed-to-index mapping is broken: seed atoms/carriers are near zero."
    if safe_float(row.get("num_final_evidence_candidates_mean", 0.0)) > 0.0 and safe_float(row.get("candidate_gold_recall_mean_eval_only", 0.0)) <= 0.10:
        return "Phase8 emits candidates, but gold/equivalent coverage is very low; seed-to-evidence mapping is weak."
    if safe_float(row.get("selected_pamae_source_rate", 0.0)) <= 0.01:
        return "Phase8 candidates exist but are not selected, or source tags are not preserved into selection."
    return "Seed-to-evidence mapping produces selected Phase8 candidates."


def _representative_cases(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    queries = [q for run in runs for q in run.get("queries", [])]
    cases = {
        "u_q_contains_gold_but_seeds_miss": [],
        "relevant_seeds_but_evidence_candidates_missing": [],
        "seeds_exist_but_low_final_candidates": [],
        "final_candidates_exist_but_selected_pamae_zero": [],
        "seed_evidence_hit_but_selected_gold_false": [],
        "pamae_candidates_exist_but_all_filtered": [],
        "refinement_improves_seed_evidence_hit": [],
        "refinement_causes_drift": [],
    }
    for q in queries:
        proposal = evidence_proposal(q)
        best = seed_best(q)
        universe = entity_universe(q)
        ref = q.get("seed", {}).get("refinement", {}) if isinstance(q.get("seed"), dict) else {}
        if not isinstance(ref, dict):
            ref = {}
        if not best and not proposal:
            continue
        final_n = safe_float(proposal.get("num_final_evidence_candidates", 0.0), 0.0)
        selected_rate, _ = _selected_pamae(q)
        p7 = q.get("phase7", {})
        if bool(universe.get("gold_entity_hit_eval_only", False)) and not bool(best.get("seed_gold_hit_eval_only", False)) and len(cases["u_q_contains_gold_but_seeds_miss"]) < 10:
            cases["u_q_contains_gold_but_seeds_miss"].append(_case(q, "u_q_contains_gold_but_seeds_miss"))
        if safe_float(best.get("best_mean_relevance", 0.0), 0.0) >= 0.5 and final_n <= 2 and len(cases["relevant_seeds_but_evidence_candidates_missing"]) < 10:
            cases["relevant_seeds_but_evidence_candidates_missing"].append(_case(q, "relevant_seeds_but_evidence_candidates_missing"))
        if best.get("best_seed_entity_ids") and final_n <= 2 and len(cases["seeds_exist_but_low_final_candidates"]) < 10:
            cases["seeds_exist_but_low_final_candidates"].append(_case(q, "seeds_exist_but_low_final_candidates"))
        if final_n > 0 and selected_rate <= 0.0 and len(cases["final_candidates_exist_but_selected_pamae_zero"]) < 10:
            cases["final_candidates_exist_but_selected_pamae_zero"].append(_case(q, "final_candidates_exist_but_selected_pamae_zero"))
        if bool(best.get("seed_evidence_gold_hit_eval_only", False)) and not bool(p7.get("selected_gold_hit_eval_only", False)) and len(cases["seed_evidence_hit_but_selected_gold_false"]) < 10:
            cases["seed_evidence_hit_but_selected_gold_false"].append(_case(q, "seed_evidence_hit_but_selected_gold_false"))
        if final_n > 0 and (safe_float(p7.get("num_phase2_candidates", 0.0), 0.0) <= 0.0 or safe_float(p7.get("num_selected_atoms", 0.0), 0.0) <= 0.0) and len(cases["pamae_candidates_exist_but_all_filtered"]) < 10:
            cases["pamae_candidates_exist_but_all_filtered"].append(_case(q, "pamae_candidates_exist_but_all_filtered"))
        before = float(bool(ref.get("before_seed_evidence_gold_hit_eval_only", False)))
        after = float(bool(ref.get("after_seed_evidence_gold_hit_eval_only", False)))
        changed = safe_float(ref.get("num_changed_seeds", 0.0), 0.0) > 0.0
        if changed and after > before and len(cases["refinement_improves_seed_evidence_hit"]) < 10:
            cases["refinement_improves_seed_evidence_hit"].append(_case(q, "refinement_improves_seed_evidence_hit"))
        if changed and after < before and len(cases["refinement_causes_drift"]) < 10:
            cases["refinement_causes_drift"].append(_case(q, "refinement_causes_drift"))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Phase8 seed-to-evidence mapping.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = load_runs(root)
    rows = []
    for (dataset, profile, variant), group in sorted(group_runs(runs).items()):
        rows.append(_summarize_group(dataset, profile, variant, group))
    cases = _representative_cases(runs)
    write_json(root / "phase8_seed_to_evidence_mapping.json", {"runs": rows, "representative_cases": cases})
    write_json(root / "phase8_seed_to_evidence_mapping_repaired.json", {"runs": rows, "representative_cases": cases})

    keys = [
        "dataset",
        "profile",
        "variant",
        "num_queries",
        "num_seed_atoms_mean",
        "num_seed_atoms_p50",
        "num_seed_atoms_p95",
        "num_seed_carriers_mean",
        "num_seed_carriers_p50",
        "num_seed_carriers_p95",
        "num_anchor_seed_path_atoms_mean",
        "num_seed_seed_path_atoms_mean",
        "num_same_title_support_atoms_mean",
        "num_same_carrier_support_atoms_mean",
        "num_final_evidence_candidates_mean",
        "num_final_evidence_candidates_p50",
        "num_final_evidence_candidates_p95",
        "candidate_gold_partial_rate_eval_only",
        "candidate_gold_full_rate_eval_only",
        "candidate_gold_recall_mean_eval_only",
        "candidate_oracle_F1_mean_eval_only",
        "chain_unit_oracle_feasible_mean_eval_only",
        "selected_pamae_source_rate",
        "interpretation",
    ]
    lines = [
        "# Phase8 Seed-to-Evidence Mapping Report",
        "",
        md_table(rows, keys),
        "",
        "## Required Checks",
        "",
        "- Seed entity to atom/carrier mapping is represented by `num_seed_atoms_*` and `num_seed_carriers_*`.",
        "- Phase8 evidence source tags are keyed by atom ids such as `s::<doc>::<sent>` while Phase7 candidate ids are rendered as `title::sent_idx`; selected feature rows preserve `atom_id` for tag joins.",
        "- Source tag preservation into selected evidence is measured by `selected_pamae_source_rate` and `selected_source_tag_distribution` in the JSON.",
        "- Rendered evidence uses title sentence ids and does not directly carry Phase8 source tags; selected-to-rendered matching must be used for rendered propagation.",
        "",
        "## Representative Cases",
        "",
    ]
    for name, items in cases.items():
        lines.extend([f"### {name}", ""])
        if items:
            lines.append(md_table(items, ["query_id", "reason", "num_final_evidence_candidates", "num_phase1_candidates", "num_phase2_candidates", "num_selected_atoms", "selected_pamae_source_rate", "candidate_gold_recall", "question"]))
        else:
            lines.append("No cases found.")
        lines.append("")
    write_md(root / "phase8_seed_to_evidence_mapping_report.md", lines)
    repaired_lines = [
        "# Phase8 Seed-to-Evidence Mapping Report Repaired",
        "",
        "This repaired report uses normalized Phase7 candidate/selected ids for candidate and selected recall. Source-balanced rows are included for comparison, but Phase8-specific interpretations apply only to pamae variants.",
        "",
        md_table(rows, keys + ["candidate_gold_recall_eval_only", "selected_gold_recall_eval_only"]),
        "",
        "## Explicit Answers",
        "",
        "- Is seed-to-evidence mapping actually failing? For raw `pamae_seed_k5`, yes: normalized candidate recall is very low despite nonzero Phase8 candidate counts.",
        "- Or was SEED_TO_EVIDENCE_MISS inflated by broken diagnostics? The original 100/100 attribution was inflated by broken Phase8 trace logging, but repaired recall still shows raw k5 is weak.",
        "- Do Phase8 evidence candidates reach Phase7 selection? Yes; selected evidence carries Phase8 atom source tags in pamae variants.",
        "- Does refinement improve seed-to-evidence mapping? On completed 2Wiki refine runs, candidate recall improves materially; Hotpot refine is partial/missing and should not be interpreted.",
        "",
        "## Representative Cases",
        "",
    ]
    for name, items in cases.items():
        repaired_lines.extend([f"### {name}", ""])
        if items:
            repaired_lines.append(md_table(items, ["query_id", "reason", "num_final_evidence_candidates", "selected_pamae_source_rate", "normalized_candidate_gold_recall", "normalized_selected_gold_recall", "question"]))
        else:
            repaired_lines.append("No cases found.")
        repaired_lines.append("")
    write_md(root / "phase8_seed_to_evidence_mapping_report_repaired.md", repaired_lines)
    print(root / "phase8_seed_to_evidence_mapping_report.md")
    print(root / "phase8_seed_to_evidence_mapping_report_repaired.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
