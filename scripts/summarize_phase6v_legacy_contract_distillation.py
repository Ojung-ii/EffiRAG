#!/usr/bin/env python3
"""Summarize PHASE6V legacy-contract distillation paired runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from validate_phase6s_artifacts import collect_phase6s_artifacts


def _safe_text(v: Any) -> str:
    return str(v or "").strip()


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


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(float(x) for x in vals) / float(len(vals)))


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, Mapping):
            out.append(dict(obj))
    return out


@dataclass
class RunData:
    run_name: str
    dataset: str
    profile: str
    role: str
    summary: Dict[str, Any]
    query_rows: List[Dict[str, Any]]


def _load_completed_runs(out_root: Path) -> List[RunData]:
    report = collect_phase6s_artifacts(out_root)
    scheduled = {_safe_text(r.get("run_name")): dict(r) for r in list(report.get("scheduled_runs", []) or [])}
    runs: List[RunData] = []
    for item in list(report.get("completed_runs", []) or []):
        run_name = _safe_text(item.get("run_name"))
        meta = scheduled.get(run_name, {})
        dataset = _safe_text(meta.get("dataset", item.get("dataset", "")))
        profile = _safe_text(meta.get("profile", item.get("profile", "")))
        role = _safe_text(meta.get("role", ""))
        summary = dict(item.get("summary", {}) or {})
        qpath = Path(_safe_text(item.get("query_path"))).resolve()
        qrows = _read_jsonl(qpath)
        runs.append(RunData(run_name, dataset, profile, role, summary, qrows))
    return runs


def _summary_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    em = _safe_float(summary.get("em", summary.get("EM", 0.0)), 0.0)
    f1 = _safe_float(summary.get("f1", summary.get("F1", 0.0)), 0.0)
    recall5 = _safe_float(summary.get("recall_at_5", summary.get("Recall@5", 0.0)), 0.0)
    sf_p = _safe_float(summary.get("supporting_fact_precision", 0.0), 0.0)
    sf_r = _safe_float(summary.get("supporting_fact_recall", 0.0), 0.0)
    sf_f1 = (2.0 * sf_p * sf_r / (sf_p + sf_r)) if (sf_p + sf_r) > 0 else 0.0
    ctx = _safe_float(summary.get("avg_context_tokens", summary.get("prompt_tokens_avg", 0.0)), 0.0)
    f1k = (f1 * 1000.0 / ctx) if ctx > 0 else 0.0
    return {
        "Recall@5": recall5,
        "EM": em,
        "F1": f1,
        "avg_context_tokens": ctx,
        "F1_per_1k_context_tokens": f1k,
        "supporting_fact_precision": sf_p,
        "supporting_fact_recall": sf_r,
        "supporting_fact_f1": sf_f1,
        "retrieval_ms": _safe_float(summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)), 0.0),
        "generation_ms": _safe_float(summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(summary.get("total_ms", summary.get("total_latency_ms", 0.0)), 0.0),
        "answer_string_hit": _safe_float(summary.get("answer_surface_present", 0.0), 0.0),
        "answer_string_rank_avg": _safe_float(summary.get("answer_surface_position_avg", -1.0), -1.0),
        "answer_bearing_density": _safe_float(summary.get("answer_surface_token_density", 0.0), 0.0),
    }


def _query_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    m = dict(row.get("metrics", {}) or {})
    e = dict(row.get("efficiency", {}) or {})
    return {
        "f1": _safe_float(m.get("f1", 0.0), 0.0),
        "em": _safe_float(m.get("em", 0.0), 0.0),
        "sf_recall": _safe_float(m.get("supporting_fact_recall", 0.0), 0.0),
        "sf_precision": _safe_float(m.get("supporting_fact_precision", 0.0), 0.0),
        "retrieval_ms": _safe_float(e.get("retrieval_latency_ms", row.get("latency_ms", 0.0)), 0.0),
        "total_ms": _safe_float(e.get("total_latency_ms", 0.0), 0.0),
        "answer_string_hit": _safe_float(m.get("answer_surface_present", 0.0), 0.0),
        "answer_string_rank": _safe_float(m.get("answer_surface_position", -1.0), -1.0),
        "answer_bearing_density": _safe_float(m.get("answer_surface_token_density", 0.0), 0.0),
    }


def _contract_diag_for_run(run: RunData) -> Dict[str, Optional[float]]:
    reorder_applied: List[float] = []
    reorder_changed: List[float] = []
    minimal_applied: List[float] = []
    num_extra: List[float] = []
    extra_tokens: List[float] = []
    budget_skip: List[float] = []
    token_delta: List[float] = []

    for row in run.query_rows:
        rendered_meta = dict((row.get("rendered") or {}).get("metadata", {}) or {})
        render_diag = dict(rendered_meta.get("render_diagnostics", {}) or {})
        reorder_applied.append(1.0 if bool(rendered_meta.get("top_slice_reorder_applied", False)) else 0.0)
        reorder_changed.append(1.0 if bool(rendered_meta.get("top_slice_reorder_changed_order", False)) else 0.0)
        minimal_applied.append(1.0 if bool(rendered_meta.get("minimal_package_applied", False)) else 0.0)
        num_extra.append(_safe_float(rendered_meta.get("num_extra_package_sentences", render_diag.get("num_context_sentences", 0)), 0.0))
        extra_tokens.append(_safe_float(rendered_meta.get("extra_sentence_token_count", 0.0), 0.0))
        budget_skip.append(_safe_float(rendered_meta.get("budget_skip_count", render_diag.get("budget_pruned_count", 0)), 0.0))
        before = _safe_float(rendered_meta.get("rendered_token_count_before_contract", 0.0), 0.0)
        after = _safe_float(rendered_meta.get("rendered_token_count_after_contract", render_diag.get("estimated_actual_prompt_tokens", 0.0)), 0.0)
        token_delta.append(float(after - before))

    return {
        "top_slice_reorder_rate": (_mean(reorder_applied) if reorder_applied else None),
        "top_slice_reorder_changed_order_rate": (_mean(reorder_changed) if reorder_changed else None),
        "minimal_package_apply_rate": (_mean(minimal_applied) if minimal_applied else None),
        "avg_extra_sentences": (_mean(num_extra) if num_extra else None),
        "avg_extra_tokens": (_mean(extra_tokens) if extra_tokens else None),
        "budget_skip_rate": (_mean([1.0 if x > 0 else 0.0 for x in budget_skip]) if budget_skip else None),
        "avg_context_token_delta": (_mean(token_delta) if token_delta else None),
    }


def _pair_baseline_variant(
    dataset: str,
    baseline: RunData,
    variant: RunData,
    variant_label: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bmap = {_safe_text(r.get("sample_id")): r for r in baseline.query_rows}
    vmap = {_safe_text(r.get("sample_id")): r for r in variant.query_rows}
    ordered = [_safe_text(r.get("sample_id")) for r in baseline.query_rows if _safe_text(r.get("sample_id"))]
    paired: List[Dict[str, Any]] = []
    improved: List[Dict[str, Any]] = []
    regressed: List[Dict[str, Any]] = []

    for sid in ordered:
        b = bmap.get(sid)
        v = vmap.get(sid)
        if b is None or v is None:
            continue
        bm = _query_metrics(b)
        vm = _query_metrics(v)
        delta_f1 = float(vm["f1"] - bm["f1"])
        delta_sf = float(vm["sf_recall"] - bm["sf_recall"])
        delta_hit = float(vm["answer_string_hit"] - bm["answer_string_hit"])
        row = {
            "dataset": dataset,
            "sample_id": sid,
            "variant": variant_label,
            "question": _safe_text(b.get("question", "")),
            "baseline_f1": bm["f1"],
            "variant_f1": vm["f1"],
            "delta_f1": delta_f1,
            "baseline_sf_recall": bm["sf_recall"],
            "variant_sf_recall": vm["sf_recall"],
            "delta_sf_recall": delta_sf,
            "baseline_answer_string_hit": bm["answer_string_hit"],
            "variant_answer_string_hit": vm["answer_string_hit"],
            "delta_answer_string_hit": delta_hit,
            "baseline_prediction": _safe_text(b.get("prediction", "")),
            "variant_prediction": _safe_text(v.get("prediction", "")),
            "baseline_selected_sentence_ids": list(b.get("retrieval_selected_sentence_ids", []) or []),
            "variant_selected_sentence_ids": list(v.get("retrieval_selected_sentence_ids", []) or []),
            "baseline_rendered_sentence_ids": list(((b.get("rendered") or {}).get("sentence_ids", []) or [])),
            "variant_rendered_sentence_ids": list(((v.get("rendered") or {}).get("sentence_ids", []) or [])),
        }
        paired.append(row)
        if delta_f1 >= 0.10 or delta_hit > 0:
            improved.append(row)
        if delta_f1 <= -0.10:
            regressed.append(row)

    improved = sorted(improved, key=lambda r: (r["delta_f1"], r["delta_sf_recall"]), reverse=True)[:20]
    regressed = sorted(regressed, key=lambda r: (r["delta_f1"], r["delta_sf_recall"]))[:20]
    return paired, improved, regressed


def _write_examples_md(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    lines: List[str] = [f"# {title}", ""]
    if not rows:
        lines.append("No examples.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for i, row in enumerate(rows, start=1):
        lines.append(f"## {i}. {row.get('dataset')} / {row.get('sample_id')}")
        lines.append(f"- question: {row.get('question','')}")
        lines.append(f"- delta_f1: {_fmt(row.get('delta_f1'), 4)}, delta_sf_recall: {_fmt(row.get('delta_sf_recall'), 4)}")
        lines.append(f"- baseline_prediction: {row.get('baseline_prediction','')}")
        lines.append(f"- variant_prediction: {row.get('variant_prediction','')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    runs = _load_completed_runs(out_root)

    by_ds_profile: Dict[Tuple[str, str], RunData] = {}
    for run in runs:
        by_ds_profile[(run.dataset, run.profile)] = run

    datasets = sorted({run.dataset for run in runs})
    profiles = [
        "unified_acr_rcedr_v12",
        "unified_acr_rcedr_v12_semantic_sufficiency",
        "unified_acr_rcedr_v12_semantic_contract",
    ]

    result_rows: List[Dict[str, Any]] = []
    contract_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        for prof in profiles:
            run = by_ds_profile.get((ds, prof))
            if run is None:
                continue
            metrics = _summary_metrics(run.summary)
            contract_diag = _contract_diag_for_run(run)
            row = {"dataset": ds, "profile": prof, **metrics, **contract_diag}
            result_rows.append(row)
            contract_rows.append(
                {
                    "dataset": ds,
                    "profile": prof,
                    "top_slice_reorder_rate": contract_diag.get("top_slice_reorder_rate"),
                    "top_slice_reorder_changed_order_rate": contract_diag.get("top_slice_reorder_changed_order_rate"),
                    "minimal_package_apply_rate": contract_diag.get("minimal_package_apply_rate"),
                    "avg_extra_sentences": contract_diag.get("avg_extra_sentences"),
                    "avg_extra_tokens": contract_diag.get("avg_extra_tokens"),
                    "budget_skip_rate": contract_diag.get("budget_skip_rate"),
                    "avg_context_token_delta": contract_diag.get("avg_context_token_delta"),
                }
            )

    # Paired comparison against baseline for semantic_sufficiency and semantic_contract.
    paired_rows: List[Dict[str, Any]] = []
    improved_examples: List[Dict[str, Any]] = []
    regression_examples: List[Dict[str, Any]] = []
    for ds in datasets:
        base = by_ds_profile.get((ds, "unified_acr_rcedr_v12"))
        if base is None:
            continue
        for prof, label in [
            ("unified_acr_rcedr_v12_semantic_sufficiency", "semantic_sufficiency"),
            ("unified_acr_rcedr_v12_semantic_contract", "semantic_contract"),
        ]:
            var = by_ds_profile.get((ds, prof))
            if var is None:
                continue
            paired, imp, reg = _pair_baseline_variant(ds, base, var, label)
            paired_rows.extend(paired)
            improved_examples.extend(imp)
            regression_examples.extend(reg)

    improved_examples = sorted(improved_examples, key=lambda r: r.get("delta_f1", 0.0), reverse=True)[:20]
    regression_examples = sorted(regression_examples, key=lambda r: r.get("delta_f1", 0.0))[:20]

    _write_jsonl(out_root / "paired_query_comparison.jsonl", paired_rows)
    _write_csv(
        out_root / "contract_diagnostics.csv",
        contract_rows,
        [
            "dataset",
            "profile",
            "top_slice_reorder_rate",
            "top_slice_reorder_changed_order_rate",
            "minimal_package_apply_rate",
            "avg_extra_sentences",
            "avg_extra_tokens",
            "budget_skip_rate",
            "avg_context_token_delta",
        ],
    )
    _write_examples_md(out_root / "answer_sufficiency_examples_top20.md", "Answer Sufficiency Improved Examples Top20", improved_examples)
    _write_examples_md(out_root / "regression_examples_top20.md", "Regression Examples Top20", regression_examples)

    # aggregate deltas vs baseline
    delta_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        base_row = next((r for r in result_rows if r["dataset"] == ds and r["profile"] == "unified_acr_rcedr_v12"), None)
        if not base_row:
            continue
        for prof in ["unified_acr_rcedr_v12_semantic_sufficiency", "unified_acr_rcedr_v12_semantic_contract"]:
            var_row = next((r for r in result_rows if r["dataset"] == ds and r["profile"] == prof), None)
            if not var_row:
                continue
            delta_rows.append(
                {
                    "dataset": ds,
                    "profile": prof,
                    "delta_F1": float(var_row["F1"] - base_row["F1"]),
                    "delta_SF_recall": float(var_row["supporting_fact_recall"] - base_row["supporting_fact_recall"]),
                    "delta_avg_context_tokens": float(var_row["avg_context_tokens"] - base_row["avg_context_tokens"]),
                    "delta_retrieval_ms": float(var_row["retrieval_ms"] - base_row["retrieval_ms"]),
                    "delta_total_ms": float(var_row["total_ms"] - base_row["total_ms"]),
                    "delta_answer_string_hit": float(var_row["answer_string_hit"] - base_row["answer_string_hit"]),
                    "delta_answer_bearing_density": float(var_row["answer_bearing_density"] - base_row["answer_bearing_density"]),
                }
            )

    avg_f1_delta_contract = _mean([r["delta_F1"] for r in delta_rows if r["profile"] == "unified_acr_rcedr_v12_semantic_contract"]) if delta_rows else 0.0
    avg_token_delta_contract = _mean([r["delta_avg_context_tokens"] for r in delta_rows if r["profile"] == "unified_acr_rcedr_v12_semantic_contract"]) if delta_rows else 0.0
    if avg_f1_delta_contract >= 0.02 and avg_token_delta_contract <= 10.0:
        decision = "accept"
    elif avg_f1_delta_contract <= -0.02:
        decision = "reject"
    else:
        decision = "hold"

    summary_payload = {
        "phase": "phase6v_legacy_contract_distillation",
        "out_root": str(out_root),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": result_rows,
        "deltas_vs_baseline": delta_rows,
        "decision": decision,
    }
    _write_json(out_root / "phase6v_legacy_contract_distillation_summary.json", summary_payload)

    md: List[str] = []
    md.append("# PHASE6V Legacy Contract Distillation Summary")
    md.append("")
    md.append("## 1. Goal")
    md.append("- Distill legacy evidence delivery behaviors into unified, budget-preserving contract flags.")
    md.append("")
    md.append("## 2. Legacy Signals Distilled")
    md.append("- Top-slice reorder (set-invariant ordering only)")
    md.append("- Minimal package rendering (bounded local context expansion)")
    md.append("")
    md.append("## 3. Artifact Validation")
    md.append(f"- phase6v_artifact_validation.json: {'present' if (out_root / 'phase6v_artifact_validation.json').exists() else 'missing'}")
    md.append("")
    md.append("## 4. Paired Main Results")
    md.append("| dataset | profile | Recall@5 | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | supporting_fact_precision | supporting_fact_recall | supporting_fact_f1 | retrieval_ms | generation_ms | total_ms |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(result_rows, key=lambda x: (x["dataset"], x["profile"])):
        md.append(
            "| {dataset} | {profile} | {r5} | {em} | {f1} | {ctx} | {f1k} | {sfp} | {sfr} | {sff1} | {ret} | {gen} | {tot} |".format(
                dataset=r["dataset"],
                profile=r["profile"],
                r5=_fmt(r["Recall@5"], 4),
                em=_fmt(r["EM"], 4),
                f1=_fmt(r["F1"], 4),
                ctx=_fmt(r["avg_context_tokens"], 3),
                f1k=_fmt(r["F1_per_1k_context_tokens"], 4),
                sfp=_fmt(r["supporting_fact_precision"], 4),
                sfr=_fmt(r["supporting_fact_recall"], 4),
                sff1=_fmt(r["supporting_fact_f1"], 4),
                ret=_fmt(r["retrieval_ms"], 2),
                gen=_fmt(r["generation_ms"], 2),
                tot=_fmt(r["total_ms"], 2),
            )
        )

    md.append("")
    md.append("## 5. Delta vs Baseline")
    md.append("| dataset | profile | ΔF1 | ΔSF_recall | Δavg_context_tokens | Δretrieval_ms | Δtotal_ms | Δanswer_string_hit | Δanswer_bearing_density |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in sorted(delta_rows, key=lambda x: (x["dataset"], x["profile"])):
        md.append(
            "| {dataset} | {profile} | {f1} | {sfr} | {ctx} | {ret} | {tot} | {hit} | {dens} |".format(
                dataset=d["dataset"],
                profile=d["profile"],
                f1=_fmt(d["delta_F1"], 4),
                sfr=_fmt(d["delta_SF_recall"], 4),
                ctx=_fmt(d["delta_avg_context_tokens"], 3),
                ret=_fmt(d["delta_retrieval_ms"], 2),
                tot=_fmt(d["delta_total_ms"], 2),
                hit=_fmt(d["delta_answer_string_hit"], 4),
                dens=_fmt(d["delta_answer_bearing_density"], 4),
            )
        )

    md.append("")
    md.append("## 6. Answer-Sufficiency Metrics")
    md.append("- See result table columns: `answer_string_hit`, `answer_string_rank_avg`, `answer_bearing_density` in JSON summary.")
    md.append("")
    md.append("## 7. Contract Diagnostics")
    md.append("| dataset | profile | top_slice_reorder_rate | top_slice_reorder_changed_order_rate | minimal_package_apply_rate | avg_extra_sentences | avg_extra_tokens | budget_skip_rate | avg_context_token_delta |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(contract_rows, key=lambda x: (x["dataset"], x["profile"])):
        md.append(
            "| {dataset} | {profile} | {a} | {b} | {c} | {d} | {e} | {f} | {g} |".format(
                dataset=r["dataset"],
                profile=r["profile"],
                a=_fmt(r["top_slice_reorder_rate"], 4),
                b=_fmt(r["top_slice_reorder_changed_order_rate"], 4),
                c=_fmt(r["minimal_package_apply_rate"], 4),
                d=_fmt(r["avg_extra_sentences"], 3),
                e=_fmt(r["avg_extra_tokens"], 3),
                f=_fmt(r["budget_skip_rate"], 4),
                g=_fmt(r["avg_context_token_delta"], 3),
            )
        )

    md.append("")
    md.append("## 8. Improved Examples")
    md.append(f"- See `{(out_root / 'answer_sufficiency_examples_top20.md').name}`")
    md.append("")
    md.append("## 9. Regression Examples")
    md.append(f"- See `{(out_root / 'regression_examples_top20.md').name}`")
    md.append("")
    md.append("## 10. Decision")
    md.append(f"- {decision}")

    (out_root / "PHASE6V_LEGACY_CONTRACT_DISTILLATION_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
