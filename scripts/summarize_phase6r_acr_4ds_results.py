#!/usr/bin/env python3
"""Summarize Phase-6R ACR v1.1 beam3 4DS QA outputs into markdown and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from scripts.compare_qa_pareto_profiles import (
    _find_retrieval_aggregate_csv,
    _prepare_retrieval_index,
    _read_csv,
    _safe_float,
    _safe_int,
    _safe_text,
    _tokens,
)


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]
DEFAULT_PROFILES = [
    "legacy_sota",
    "unified_large",
    "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
    "unified_acr_v1",
    "unified_acr_v11",
    "unified_acr_v11_beam3",
]
BASELINES = ["dense", "hipporag2", "raptor", "bm25"]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return dict(obj or {}) if isinstance(obj, Mapping) else {}
    except Exception:
        return {}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row or {}))


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def _metric(summary: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in summary:
            try:
                return float(summary.get(key))
            except Exception:
                continue
    return float(default)


def _collect_qa_records(input_root: Path, datasets: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for manifest_path in sorted(input_root.rglob("run_manifest.json")):
        manifest = _read_json(manifest_path)
        profile = _safe_text(manifest.get("profile"))
        profile_name = _safe_text(manifest.get("profile_name"))
        if not profile:
            profile = "legacy_sota" if profile_name else _safe_text(manifest_path.parent.name)
        for record in list(manifest.get("records", []) or []):
            if not isinstance(record, Mapping):
                continue
            dataset = _safe_text(record.get("dataset"))
            if dataset not in datasets:
                continue
            summary_path = _safe_text(record.get("summary_path"))
            if not summary_path:
                continue
            summary = _read_json(Path(summary_path))
            if not summary:
                continue
            prompt_tokens = _metric(summary, "prompt_tokens_avg", "avg_context_tokens", default=0.0)
            f1 = _metric(summary, "f1", "F1", default=0.0)
            rows.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "EM": _metric(summary, "em", "EM", default=0.0),
                    "F1": f1,
                    "prompt_tokens_avg": prompt_tokens,
                    "completion_tokens_avg": _metric(summary, "completion_tokens_avg", default=0.0),
                    "retrieval_ms": _metric(summary, "retrieval_ms", "retrieval_latency_ms", default=0.0),
                    "generation_ms": _metric(summary, "generation_ms", "generation_latency_ms", default=0.0),
                    "total_ms": _metric(summary, "total_ms", "total_latency_ms", default=0.0),
                    "recall_at_5": _metric(summary, "recall_at_5", "R@5", "Recall@5", default=0.0),
                    "supporting_fact_precision": _metric(
                        summary, "supporting_fact_precision", "sf_precision", default=0.0
                    ),
                    "supporting_fact_recall": _metric(summary, "supporting_fact_recall", "sf_recall", default=0.0),
                    "supporting_fact_f1": _metric(summary, "supporting_fact_f1", "sf_f1", default=0.0),
                    "rendered_sf_P": _metric(
                        summary, "rendered_supporting_fact_precision", "rendered_sf_P", default=0.0
                    ),
                    "rendered_sf_R": _metric(summary, "rendered_supporting_fact_recall", "rendered_sf_R", default=0.0),
                    "rendered_sf_F1": _metric(summary, "rendered_supporting_fact_f1", "rendered_sf_F1", default=0.0),
                    "rendered_tokens": _metric(summary, "rendered_tokens", "rendered_tokens_avg", default=0.0),
                    "summary_path": summary_path,
                    "qa_executed_samples": _safe_int(summary.get("qa_executed_samples", 0), 0),
                }
            )
    return rows


def _aggregate_rows(
    qa_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    profiles: Sequence[str],
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in qa_rows:
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))
        if dataset in datasets and profile in profiles:
            buckets.setdefault((dataset, profile), []).append(row)

    metric_keys = [
        "EM",
        "F1",
        "prompt_tokens_avg",
        "completion_tokens_avg",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "recall_at_5",
        "supporting_fact_precision",
        "supporting_fact_recall",
        "supporting_fact_f1",
        "rendered_sf_P",
        "rendered_sf_R",
        "rendered_sf_F1",
        "rendered_tokens",
    ]
    out: List[Dict[str, Any]] = []
    for dataset in datasets:
        for profile in profiles:
            group = list(buckets.get((dataset, profile), []) or [])
            if not group:
                out.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "qa_num_runs": 0,
                        "qa_num_queries": 0,
                        "qa_available": False,
                        **{k: 0.0 for k in metric_keys},
                    }
                )
                continue
            agg: Dict[str, Any] = {
                "dataset": dataset,
                "profile": profile,
                "qa_num_runs": int(len(group)),
                "qa_num_queries": int(sum(_safe_int(r.get("qa_executed_samples", 0), 0) for r in group)),
                "qa_available": True,
            }
            for k in metric_keys:
                vals = [_safe_float(r.get(k, 0.0), 0.0) for r in group]
                agg[k] = float(sum(vals) / max(1, len(vals)))
            out.append(agg)
    return out


def _profile_role(profile: str) -> str:
    mapping = {
        "legacy_sota": "heuristic teacher/reference",
        "unified_large": "primary baseline",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge": "compact reference",
        "unified_acr_v1": "ACR v1",
        "unified_acr_v11": "ACR v1.1",
        "unified_acr_v11_beam3": "ACR v1.1 beam3 (main candidate)",
        "unified_acr_v1_beam3": "optional beam3 reference",
    }
    return mapping.get(profile, "candidate")


def _normalize_baseline_entry(raw: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "f1": _metric(raw, "f1", "F1", default=0.0),
        "tokens": _metric(
            raw,
            "prompt_tokens_avg",
            "avg_context_tokens",
            "context_tokens_avg",
            "context_tokens",
            "tokens",
            default=0.0,
        ),
    }


def _load_native_baseline_json(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    data = _read_json(path)
    if not data:
        return out

    dataset_keys = [k for k in data.keys() if isinstance(data.get(k), Mapping)]
    for dataset in dataset_keys:
        ds_obj = data.get(dataset)
        if not isinstance(ds_obj, Mapping):
            continue
        ds_map: Dict[str, Dict[str, float]] = {}
        for base in BASELINES:
            base_obj = ds_obj.get(base)
            if isinstance(base_obj, Mapping):
                ds_map[base] = _normalize_baseline_entry(base_obj)
        if ds_map:
            out[_safe_text(dataset)] = ds_map
    return out


def _load_hipporag2_baselines(root: Path, datasets: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for dataset in datasets:
        path = root / dataset / f"{dataset}_eval_summary_results.json"
        summary = _read_json(path)
        if not summary:
            continue
        out[dataset] = {
            "f1": _metric(summary, "f1", "F1", default=0.0),
            "tokens": _metric(
                summary,
                "prompt_tokens_avg",
                "avg_context_tokens",
                "context_tokens_avg",
                "context_tokens",
                "tokens",
                default=0.0,
            ),
        }
    return out


def _merge_retrieval_metrics(
    rows: List[Dict[str, Any]],
    retrieval_index: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> None:
    for row in rows:
        key = (_safe_text(row.get("dataset")), _safe_text(row.get("profile")))
        r = dict(retrieval_index.get(key, {}) or {})
        row["rendered_tokens"] = _safe_float(
            row.get("rendered_tokens", 0.0),
            0.0,
        ) or _safe_float(r.get("rendered_tokens", r.get("rendered_tokens_avg", 0.0)), 0.0)
        row["rendered_sf_R"] = _safe_float(
            row.get("rendered_sf_R", 0.0),
            0.0,
        ) or _safe_float(r.get("rendered_sf_R", 0.0), 0.0)
        row["rendered_sf_P"] = _safe_float(
            row.get("rendered_sf_P", 0.0),
            0.0,
        ) or _safe_float(r.get("rendered_sf_P", 0.0), 0.0)
        row["rendered_sf_F1"] = _safe_float(
            row.get("rendered_sf_F1", 0.0),
            0.0,
        ) or _safe_float(r.get("rendered_sf_F1", 0.0), 0.0)


def _baseline_relative(
    rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    profiles: Sequence[str],
    baseline_profile: str = "unified_large",
) -> List[Dict[str, Any]]:
    by_key = {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): dict(r) for r in rows}
    out: List[Dict[str, Any]] = []
    for dataset in datasets:
        base = by_key.get((dataset, baseline_profile), {})
        base_f1 = _safe_float(base.get("F1", 0.0), 0.0)
        base_tok = _safe_float(base.get("prompt_tokens_avg", 0.0), 0.0)
        base_f1k = (
            (base_f1 * 1000.0 / base_tok) if base_tok > 0 else 0.0
        )
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            f1 = _safe_float(row.get("F1", 0.0), 0.0)
            tok = _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0)
            f1k = (f1 * 1000.0 / tok) if tok > 0 else 0.0
            out.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "delta_F1_vs_unified_large": f1 - base_f1,
                    "delta_tokens_vs_unified_large": tok - base_tok,
                    "delta_F1_per_1k_vs_unified_large": f1k - base_f1k,
                }
            )
    return out


def _yn(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _compare_to_baseline(
    f1: float,
    tok: float,
    baseline: Mapping[str, float] | None,
) -> Tuple[bool | None, bool | None]:
    if not baseline:
        return (None, None)
    base_f1 = _safe_float(baseline.get("f1", 0.0), 0.0)
    base_tok = _safe_float(baseline.get("tokens", 0.0), 0.0)
    f1_cmp = (f1 > base_f1) if base_f1 > 0.0 else None
    tok_cmp = (tok < base_tok) if base_tok > 0.0 else None
    return (f1_cmp, tok_cmp)


def _make_markdown(
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    baseline_rel: Sequence[Mapping[str, Any]],
    baseline_index: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> str:
    by_key = {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): dict(r) for r in rows}
    rel_key = {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): dict(r) for r in baseline_rel}

    lines: List[str] = []
    lines.append("# Phase-6R ACR v1.1 Beam3 4DS Validation")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This phase validates whether unified_acr_v11_beam3 maintains QA-token Pareto gains beyond 2DS smoke, across HotpotQA, 2WikiMultihopQA, MuSiQue, and PopQA."
    )
    lines.append("")
    lines.append("## 2. Compared Methods")
    lines.append("")
    lines.append("| profile | role |")
    lines.append("|---|---|")
    for profile in profiles:
        lines.append(f"| {profile} | {_profile_role(profile)} |")
    lines.append("")

    lines.append("## 3. QA Results")
    lines.append("")
    lines.append(
        "| dataset | method | EM | F1 | avg_context_tokens | F1_per_1k_context_tokens | retrieval_ms | generation_ms | total_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            f1 = _safe_float(row.get("F1", 0.0), 0.0)
            tok = _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0)
            f1k = (f1 * 1000.0 / tok) if tok > 0 else 0.0
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("EM", 0.0)),
                    _fmt(f1),
                    _fmt(tok, 1),
                    _fmt(f1k),
                    _fmt(row.get("retrieval_ms", 0.0), 1),
                    _fmt(row.get("generation_ms", 0.0), 1),
                    _fmt(row.get("total_ms", 0.0), 1),
                )
            )
    lines.append("")

    lines.append("## 4. Baseline-relative Results")
    lines.append("")
    lines.append(
        "| dataset | method | delta_F1_vs_unified_large | delta_tokens_vs_unified_large | delta_F1_per_1k_vs_unified_large |"
    )
    lines.append("|---|---|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = rel_key.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("delta_F1_vs_unified_large", 0.0)),
                    _fmt(row.get("delta_tokens_vs_unified_large", 0.0), 1),
                    _fmt(row.get("delta_F1_per_1k_vs_unified_large", 0.0)),
                )
            )
    lines.append("")

    lines.append("## 5. Native Baseline Comparison")
    lines.append("")
    lines.append(
        "| dataset | method | beats_dense_F1 | beats_hipporag2_F1 | lower_tokens_than_dense | lower_tokens_than_hipporag2 |"
    )
    lines.append("|---|---|---|---|---|---|")
    for dataset in datasets:
        ds_base = dict(baseline_index.get(dataset, {}) or {})
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            f1 = _safe_float(row.get("F1", 0.0), 0.0)
            tok = _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0)
            dense_cmp = _compare_to_baseline(f1, tok, ds_base.get("dense"))
            hippo_cmp = _compare_to_baseline(f1, tok, ds_base.get("hipporag2"))
            lines.append(
                f"| {dataset} | {profile} | {_yn(dense_cmp[0])} | {_yn(hippo_cmp[0])} | {_yn(dense_cmp[1])} | {_yn(hippo_cmp[1])} |"
            )
    lines.append("")

    lines.append("## 6. Legacy Reference Gap")
    lines.append("")
    lines.append("Legacy is treated as a heuristic teacher/reference, not the direct optimization target.")
    lines.append("")
    lines.append("| dataset | method | F1_gap_to_legacy | token_ratio_to_legacy | F1_per_token_ratio_to_legacy |")
    lines.append("|---|---|---:|---:|---:|")
    for dataset in datasets:
        legacy = by_key.get((dataset, "legacy_sota"), {})
        legacy_f1 = _safe_float(legacy.get("F1", 0.0), 0.0)
        legacy_tok = _safe_float(legacy.get("prompt_tokens_avg", 0.0), 0.0)
        legacy_f1k = (legacy_f1 * 1000.0 / legacy_tok) if legacy_tok > 0 else 0.0
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            f1 = _safe_float(row.get("F1", 0.0), 0.0)
            tok = _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0)
            f1k = (f1 * 1000.0 / tok) if tok > 0 else 0.0
            gap = f1 - legacy_f1 if legacy_f1 > 0 else 0.0
            tok_ratio = (tok / legacy_tok) if legacy_tok > 0 else 0.0
            eff_ratio = (f1k / legacy_f1k) if legacy_f1k > 0 else 0.0
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(gap),
                    _fmt(tok_ratio),
                    _fmt(eff_ratio),
                )
            )
    lines.append("")

    lines.append("## 7. Decision")
    lines.append("")
    lines.append("- Does ACR v1.1 beam3 remain the main candidate?")
    lines.append("- Which datasets pass QA-token Pareto against unified_large?")
    lines.append("- Should we scale to n=500/n=1000?")
    lines.append("- Which failure cases need targeted diagnosis?")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase-6R 4DS QA runs.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--input-root", required=True, help="Root containing qa_runs/*/run_manifest.json")
    parser.add_argument(
        "--retrieval-audit-root",
        default="",
        help="Optional retrieval audit root containing aggregate_by_profile_dataset.csv",
    )
    parser.add_argument(
        "--baseline-table-source",
        default="native_baseline_table",
        help="Baseline source label (for manifest only).",
    )
    parser.add_argument(
        "--native-baseline-json",
        default="",
        help="Optional JSON file containing dataset->baseline->{f1,tokens} entries.",
    )
    parser.add_argument(
        "--hipporag2-summary-root",
        default="/home/ojungii/HippoRAG2/outputs",
        help="HippoRAG2 output root for dataset summary JSONs.",
    )
    parser.add_argument("--output", required=True, help="Output markdown path.")
    args = parser.parse_args()

    datasets = _tokens(args.datasets)
    profiles = _tokens(args.profiles)
    input_root = Path(args.input_root).resolve()
    output_md = Path(args.output).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = _collect_qa_records(input_root=input_root, datasets=datasets)
    rows = _aggregate_rows(qa_rows=raw_rows, datasets=datasets, profiles=profiles)

    retrieval_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    retrieval_csv_path = ""
    if str(args.retrieval_audit_root or "").strip():
        try:
            retrieval_csv = _find_retrieval_aggregate_csv(Path(args.retrieval_audit_root).resolve())
            retrieval_rows = _read_csv(retrieval_csv)
            retrieval_index = _prepare_retrieval_index(retrieval_rows)
            retrieval_csv_path = str(retrieval_csv)
        except Exception:
            retrieval_index = {}
            retrieval_csv_path = ""
    if retrieval_index:
        _merge_retrieval_metrics(rows=rows, retrieval_index=retrieval_index)

    baseline_index: Dict[str, Dict[str, Dict[str, float]]] = {}
    native_path = Path(str(args.native_baseline_json or "").strip()) if str(args.native_baseline_json or "").strip() else None
    if native_path and native_path.exists():
        baseline_index = _load_native_baseline_json(native_path)
    # Always try to overlay HippoRAG2 summaries when available.
    hippo_root = Path(args.hipporag2_summary_root).resolve()
    hippo_map = _load_hipporag2_baselines(hippo_root, datasets=datasets)
    for dataset, metrics in hippo_map.items():
        baseline_index.setdefault(dataset, {})
        baseline_index[dataset]["hipporag2"] = dict(metrics)

    baseline_rel_rows = _baseline_relative(rows=rows, datasets=datasets, profiles=profiles)

    joined_csv = output_md.parent / "phase6r_joined_results.csv"
    rel_csv = output_md.parent / "phase6r_baseline_relative.csv"
    _write_csv(
        joined_csv,
        rows,
        [
            "dataset",
            "profile",
            "qa_num_runs",
            "qa_num_queries",
            "qa_available",
            "EM",
            "F1",
            "prompt_tokens_avg",
            "completion_tokens_avg",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
            "recall_at_5",
            "supporting_fact_precision",
            "supporting_fact_recall",
            "supporting_fact_f1",
            "rendered_sf_P",
            "rendered_sf_R",
            "rendered_sf_F1",
            "rendered_tokens",
        ],
    )
    _write_csv(
        rel_csv,
        baseline_rel_rows,
        [
            "dataset",
            "profile",
            "delta_F1_vs_unified_large",
            "delta_tokens_vs_unified_large",
            "delta_F1_per_1k_vs_unified_large",
        ],
    )

    md = _make_markdown(
        datasets=datasets,
        profiles=profiles,
        rows=rows,
        baseline_rel=baseline_rel_rows,
        baseline_index=baseline_index,
    )
    output_md.write_text(md + "\n", encoding="utf-8")

    manifest: MutableMapping[str, Any] = {
        "datasets": list(datasets),
        "profiles": list(profiles),
        "input_root": str(input_root),
        "retrieval_audit_root": str(args.retrieval_audit_root or ""),
        "retrieval_aggregate_csv": retrieval_csv_path,
        "baseline_table_source": str(args.baseline_table_source),
        "native_baseline_json": str(native_path.resolve()) if native_path and native_path.exists() else "",
        "hipporag2_summary_root": str(hippo_root),
        "output_summary_md": str(output_md),
        "output_joined_csv": str(joined_csv),
        "output_baseline_relative_csv": str(rel_csv),
        "num_raw_rows": int(len(raw_rows)),
        "num_aggregated_rows": int(len(rows)),
    }
    (output_md.parent / "phase6r_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(str(output_md))


if __name__ == "__main__":
    main()
