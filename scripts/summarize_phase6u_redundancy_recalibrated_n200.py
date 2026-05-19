#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

from scripts.summarize_phase6u_abr_objective_redesign import build_summary


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def _find_latest_query_jsonl(out_root: Path, run_name: str) -> Path | None:
    paths = sorted((out_root / "qa_runs" / run_name).glob("**/rag_query_results.jsonl"))
    return paths[-1] if paths else None


def _aggregate_redundancy_reasons(query_path: Path) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    rows = 0
    relaxed_sum = 0.0
    before_sum = 0.0
    after_sum = 0.0
    applied_sum = 0.0

    for raw in query_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        diag = (
            obj.get("retrieval", {})
            .get("diagnostics", {})
            .get("unified_acr_rcedr_diag", {})
        )
        if not isinstance(diag, dict):
            continue
        rows += 1
        relaxed_sum += _to_float(diag.get("redundancy_relaxed_count", diag.get("redundancy_recalibrated_count")))
        before_sum += _to_float(diag.get("redundancy_before_avg"))
        after_sum += _to_float(diag.get("redundancy_after_avg"))
        applied_sum += _to_float(diag.get("redundancy_recalibration_applied_avg"))
        reasons = diag.get("redundancy_recalibration_reasons")
        if isinstance(reasons, dict):
            for k, v in reasons.items():
                counts[str(k)] += int(_to_float(v))

    denom = float(rows) if rows > 0 else 1.0
    return {
        "rows": rows,
        "redundancy_relaxed_count_avg": relaxed_sum / denom,
        "redundancy_before_avg": before_sum / denom,
        "redundancy_after_avg": after_sum / denom,
        "redundancy_recalibration_applied_avg": applied_sum / denom,
        "redundancy_relaxation_reason_counts": dict(counts),
    }


def _decision(summary: Mapping[str, Any], baseline_profile: str, variant_profile: str) -> str:
    rows = list(summary.get("result_rows", []) or [])
    deltas = list(summary.get("deltas", []) or [])

    by_ds = {d.get("dataset"): d for d in deltas if d.get("variant_profile") == variant_profile}
    if not by_ds:
        return "hold"

    # Reject gates
    for ds, d in by_ds.items():
        if _to_float(d.get("delta_F1")) < -0.02:
            return "reject"
        if _to_float(d.get("delta_SF_recall")) < -0.03:
            return "reject"

    # Token/retrieval percentage guards
    base_rows = {r.get("dataset"): r for r in rows if r.get("profile") == baseline_profile}
    var_rows = {r.get("dataset"): r for r in rows if r.get("profile") == variant_profile}
    for ds in by_ds.keys():
        b = base_rows.get(ds, {})
        v = var_rows.get(ds, {})
        b_ctx = _to_float(b.get("avg_context_tokens"))
        v_ctx = _to_float(v.get("avg_context_tokens"))
        b_ret = _to_float(b.get("retrieval_ms"))
        v_ret = _to_float(v.get("retrieval_ms"))
        if b_ctx > 0 and ((v_ctx - b_ctx) / b_ctx) > 0.20:
            return "reject"
        if b_ret > 0 and ((v_ret - b_ret) / b_ret) > 0.20:
            return "reject"

    d_f1_vals = [_to_float(by_ds[ds].get("delta_F1")) for ds in by_ds]
    d_sr_vals = [_to_float(by_ds[ds].get("delta_SF_recall")) for ds in by_ds]
    avg_df1 = sum(d_f1_vals) / max(1, len(d_f1_vals))
    avg_dsr = sum(d_sr_vals) / max(1, len(d_sr_vals))

    # strong accept
    all_nonneg_f1 = all(v >= 0.0 for v in d_f1_vals)
    if all_nonneg_f1 and avg_df1 >= 0.02 and avg_dsr >= 0.03:
        # tighter guard rails
        for ds in by_ds.keys():
            b = base_rows.get(ds, {})
            v = var_rows.get(ds, {})
            b_ctx = _to_float(b.get("avg_context_tokens"))
            v_ctx = _to_float(v.get("avg_context_tokens"))
            b_ret = _to_float(b.get("retrieval_ms"))
            v_ret = _to_float(v.get("retrieval_ms"))
            if b_ctx > 0 and ((v_ctx - b_ctx) / b_ctx) > 0.10:
                break
            if b_ret > 0 and ((v_ret - b_ret) / b_ret) > 0.15:
                break
        else:
            return "strong_accept"

    # weak accept
    improved_count = sum(1 for v in d_f1_vals if v > 0.0)
    neutral_or_better = all(v >= -0.01 for v in d_f1_vals) and all(v >= -0.01 for v in d_sr_vals)
    if improved_count >= 1 and neutral_or_better and avg_df1 > 0 and avg_dsr >= 0:
        return "accept"

    # hold
    if all(v >= -0.02 for v in d_f1_vals) and all(v >= -0.03 for v in d_sr_vals):
        return "hold"

    return "reject"


def build_n200_summary(
    out_root: Path,
    datasets: List[str],
    baseline_profile: str,
    variant_profile: str,
    top_k: int,
) -> Dict[str, Any]:
    # Build generic paired summary first (also writes paired_query_comparison.jsonl / diagnostics csv / examples)
    generic = build_summary(
        out_root=out_root,
        datasets=datasets,
        baseline_profile=baseline_profile,
        variant_profiles=[variant_profile],
        top_k=top_k,
    )

    rows = list(generic.get("result_rows", []) or [])
    deltas = [d for d in list(generic.get("deltas", []) or []) if d.get("variant_profile") == variant_profile]

    # Aggregate redundancy reason counts from variant runs
    reason_by_dataset: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("profile") != variant_profile:
            continue
        run_name = str(r.get("run_name") or "")
        dataset = str(r.get("dataset") or "")
        q = _find_latest_query_jsonl(out_root, run_name)
        if q is None:
            reason_by_dataset[dataset] = {
                "available": False,
                "redundancy_relaxation_reason_counts": {},
            }
            continue
        agg = _aggregate_redundancy_reasons(q)
        agg["available"] = True
        reason_by_dataset[dataset] = agg

    decision = _decision(generic, baseline_profile, variant_profile)

    summary = {
        "phase": "phase6u_redundancy_recalibrated_n200",
        "out_root": str(out_root),
        "datasets": datasets,
        "baseline_profile": baseline_profile,
        "variant_profile": variant_profile,
        "result_rows": rows,
        "deltas": deltas,
        "redundancy_recalibration_diagnostics": reason_by_dataset,
        "decision": decision,
    }

    # Write required JSON
    (out_root / "phase6u_redundancy_recalibrated_n200_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Write required Markdown
    lines: List[str] = []
    lines.append("# PHASE6U Redundancy-Recalibrated ABR N200 Summary")
    lines.append("")
    lines.append("## 1. Goal")
    lines.append("Paired n=200 stability check for unified_acr_rcedr_v12_redundancy_recalibrated versus unified_acr_rcedr_v12.")
    lines.append("")
    lines.append("## 2. Artifact Validation")
    vpath = out_root / "phase6u_artifact_validation.json"
    if vpath.exists():
        try:
            vobj = json.loads(vpath.read_text(encoding="utf-8"))
            c = dict(vobj.get("counts", {}) or {})
            lines.append(
                f"- scheduled_runs={c.get('scheduled_runs','n/a')}, completed_run_names={c.get('completed_run_names','n/a')}, "
                f"incomplete_attempts={c.get('incomplete_attempts','n/a')}, extra_not_in_manifest={c.get('extra_not_in_manifest','n/a')}, "
                f"multi_attempt_run_names={c.get('multi_attempt_run_names','n/a')}, parse_errors={c.get('parse_errors','n/a')}"
            )
        except Exception:
            lines.append("- diagnostic_unavailable")
    else:
        lines.append("- diagnostic_unavailable")
    lines.append("")

    lines.append("## 3. Paired Main Results")
    lines.append("| dataset | profile | role | Recall@5 | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | supporting_fact_precision | supporting_fact_recall | supporting_fact_f1 | retrieval_ms | generation_ms | total_ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (str(x.get("dataset")), str(x.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {role} | {r5} | {em} | {f1} | {ctx} | {f1k} | {sp} | {sr} | {sf1} | {rm} | {gm} | {tm} |".format(
                dataset=r.get("dataset"),
                profile=r.get("profile"),
                role=r.get("role"),
                r5=_fmt(r.get("Recall@5"), 4),
                em=_fmt(r.get("EM"), 4),
                f1=_fmt(r.get("F1"), 4),
                ctx=_fmt(r.get("avg_context_tokens"), 3),
                f1k=_fmt(r.get("F1_per_1k_context_tokens"), 4),
                sp=_fmt(r.get("supporting_fact_precision"), 4),
                sr=_fmt(r.get("supporting_fact_recall"), 4),
                sf1=_fmt(r.get("supporting_fact_f1"), 4),
                rm=_fmt(r.get("retrieval_ms"), 2),
                gm=_fmt(r.get("generation_ms"), 2),
                tm=_fmt(r.get("total_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 4. Delta vs Baseline")
    lines.append("| dataset | variant_profile | ΔEM | ΔF1 | ΔSF_recall | ΔSF_precision | Δavg_context_tokens | Δretrieval_ms | Δtotal_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in sorted(deltas, key=lambda x: str(x.get("dataset"))):
        lines.append(
            "| {dataset} | {profile} | {dem} | {df1} | {dsr} | {dsp} | {dctx} | {drm} | {dtm} |".format(
                dataset=d.get("dataset"),
                profile=d.get("variant_profile"),
                dem=_fmt(d.get("delta_EM"), 4),
                df1=_fmt(d.get("delta_F1"), 4),
                dsr=_fmt(d.get("delta_SF_recall"), 4),
                dsp=_fmt(d.get("delta_SF_precision"), 4),
                dctx=_fmt(d.get("delta_avg_context_tokens"), 3),
                drm=_fmt(d.get("delta_retrieval_ms"), 2),
                dtm=_fmt(d.get("delta_total_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 5. ABR-stage Diagnostics")
    lines.append("| dataset | profile | sentence_candidate_SF_recall | evidence_atom_SF_recall | ABR_selected_SF_recall | rendered_SF_recall | selected_atoms_avg | selected_tokens_avg | objective_eval_calls_avg | unified_acr_rcedr_ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (str(x.get("dataset")), str(x.get("profile")))):
        lines.append(
            "| {dataset} | {profile} | {s1} | {s2} | {s3} | {s4} | {a} | {t} | {o} | {u} |".format(
                dataset=r.get("dataset"),
                profile=r.get("profile"),
                s1=_fmt(r.get("sentence_candidate_SF_recall"), 4),
                s2=_fmt(r.get("evidence_atom_SF_recall"), 4),
                s3=_fmt(r.get("ABR_selected_SF_recall"), 4),
                s4=_fmt(r.get("rendered_SF_recall"), 4),
                a=_fmt(r.get("selected_atoms_avg"), 2),
                t=_fmt(r.get("selected_tokens_avg"), 2),
                o=_fmt(r.get("objective_eval_calls_avg"), 2),
                u=_fmt(r.get("unified_acr_rcedr_ms"), 2),
            )
        )
    lines.append("")

    lines.append("## 6. Redundancy Recalibration Diagnostics")
    for ds in datasets:
        info = reason_by_dataset.get(ds, {"available": False})
        lines.append(f"- {ds}: available={str(info.get('available', False)).lower()}")
        if info.get("available"):
            lines.append(
                "  - redundancy_relaxed_count_avg={relaxed}, redundancy_before_avg={before}, redundancy_after_avg={after}, redundancy_recalibration_applied_avg={applied}".format(
                    relaxed=_fmt(info.get("redundancy_relaxed_count_avg"), 2),
                    before=_fmt(info.get("redundancy_before_avg"), 4),
                    after=_fmt(info.get("redundancy_after_avg"), 4),
                    applied=_fmt(info.get("redundancy_recalibration_applied_avg"), 4),
                )
            )
            lines.append(f"  - redundancy_relaxation_reason_counts={json.dumps(info.get('redundancy_relaxation_reason_counts', {}), ensure_ascii=False)}")
        else:
            lines.append("  - diagnostic_unavailable")
    lines.append("")

    lines.append("## 7. Token/Latency Guardrail")
    lines.append("See Delta vs Baseline table.")
    lines.append("")

    lines.append("## 8. Improved Examples")
    lines.append("See redundancy_recalibrated_examples_top20.md")
    lines.append("")

    lines.append("## 9. Regression Examples")
    lines.append("See regression_examples_top20.md")
    lines.append("")

    lines.append("## 10. Decision")
    lines.append(f"- decision: `{decision}`")

    (out_root / "PHASE6U_REDUNDANCY_RECALIBRATED_N200_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PHASE6U redundancy-recalibrated n200 experiment.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    ap.add_argument("--baseline-profile", default="unified_acr_rcedr_v12")
    ap.add_argument("--variant-profile", default="unified_acr_rcedr_v12_redundancy_recalibrated")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]

    build_n200_summary(
        out_root=out_root,
        datasets=datasets,
        baseline_profile=str(args.baseline_profile).strip(),
        variant_profile=str(args.variant_profile).strip(),
        top_k=max(1, int(args.top_k)),
    )

    print(out_root / "PHASE6U_REDUNDANCY_RECALIBRATED_N200_SUMMARY.md")
    print(out_root / "phase6u_redundancy_recalibrated_n200_summary.json")


if __name__ == "__main__":
    main()
