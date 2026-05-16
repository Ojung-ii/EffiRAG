#!/usr/bin/env python3
"""Build QA-aware Pareto comparison report for Phase-6N profiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa"]
DEFAULT_PROFILES = [
    "legacy_sota",
    "unified_large",
    "unified_gl_rcedr_v1",
    "unified_gl_rcedr_v1_support_span_contract",
    "unified_gl_rcedr_v1_support_span_contract_no_cap",
    "unified_gl_rcedr_v1_adaptive_support_span",
    "unified_gl_rcedr_v1_adaptive_support_span_no_bridge",
    "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty",
    "unified_gl_rcedr_v1_adaptive_support_span_span40",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return dict(obj or {}) if isinstance(obj, Mapping) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row or {}))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row or {}))


def _profile_role(profile: str) -> str:
    role_map = {
        "legacy_sota": "heuristic reference",
        "unified_large": "clean baseline",
        "unified_gl_rcedr_v1": "render-preserving baseline",
        "unified_gl_rcedr_v1_support_span_contract": "hard support span",
        "unified_gl_rcedr_v1_support_span_contract_no_cap": "best retrieval-only support span",
        "unified_gl_rcedr_v1_adaptive_support_span": "main adaptive candidate",
        "unified_gl_rcedr_v1_adaptive_support_span_no_bridge": "high-density ablation",
        "unified_gl_rcedr_v1_adaptive_support_span_no_length_penalty": "recall-heavy ablation",
        "unified_gl_rcedr_v1_adaptive_support_span_span40": "capped adaptive ablation",
        "unified_gl_rcedr_v1_sentence_contract": "phase6k sentence contract (optional)",
    }
    return role_map.get(profile, "candidate")


def _find_retrieval_aggregate_csv(root: Path) -> Path:
    candidates = [
        root / "aggregate_by_profile_dataset.csv",
        root / "audit" / "aggregate_by_profile_dataset.csv",
        root / "champion_audit" / "aggregate_by_profile_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find aggregate_by_profile_dataset.csv under: {root}. "
        "Expected one of root/, root/audit/, root/champion_audit/."
    )


def _collect_qa_summary_rows(input_root: Path, datasets: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_manifest_paths = sorted(input_root.rglob("run_manifest.json"))
    for manifest_path in run_manifest_paths:
        manifest = _read_json(manifest_path)
        profile = _safe_text(manifest.get("profile"))
        profile_name = _safe_text(manifest.get("profile_name"))
        if not profile and profile_name:
            # legacy runner typically uses profile_name in manifest.
            profile = "legacy_sota"
        if not profile:
            # fallback to parent folder name
            profile = _safe_text(manifest_path.parent.name)
        records = list(manifest.get("records", []) or [])
        for record in records:
            if not isinstance(record, Mapping):
                continue
            dataset = _safe_text(record.get("dataset"))
            if dataset not in datasets:
                continue
            summary_path = _safe_text(record.get("summary_path"))
            if not summary_path:
                continue
            spath = Path(summary_path)
            summary = _read_json(spath)
            if not summary:
                continue
            prompt_tokens = _safe_float(summary.get("prompt_tokens_avg", summary.get("avg_context_tokens", 0.0)), 0.0)
            f1 = _safe_float(summary.get("f1", 0.0), 0.0)
            sf_f1_k_prompt = float((f1 * 1000.0) / prompt_tokens) if prompt_tokens > 0 else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "EM": _safe_float(summary.get("em", 0.0), 0.0),
                    "F1": f1,
                    "prompt_tokens_avg": prompt_tokens,
                    "completion_tokens_avg": _safe_float(summary.get("completion_tokens_avg", 0.0), 0.0),
                    "retrieval_ms": _safe_float(
                        summary.get("retrieval_ms", summary.get("retrieval_latency_ms", 0.0)),
                        0.0,
                    ),
                    "generation_ms": _safe_float(
                        summary.get("generation_ms", summary.get("generation_latency_ms", 0.0)),
                        0.0,
                    ),
                    "total_ms": _safe_float(
                        summary.get("total_ms", summary.get("total_latency_ms", 0.0)),
                        0.0,
                    ),
                    "sf_F1_per_1k_prompt_tokens": sf_f1_k_prompt,
                    "summary_path": str(spath),
                    "qa_executed_samples": _safe_int(summary.get("qa_executed_samples", 0), 0),
                }
            )
    return rows


def _aggregate_mean(rows: Sequence[Mapping[str, Any]], value_keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in value_keys:
        vals = [_safe_float(r.get(key, 0.0), 0.0) for r in rows]
        out[key] = float(sum(vals) / max(1, len(vals)))
    return out


def _summarize_qa_rows(qa_rows: Sequence[Mapping[str, Any]], datasets: Sequence[str], profiles: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in qa_rows:
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))
        if dataset not in datasets or profile not in profiles:
            continue
        buckets.setdefault((dataset, profile), []).append(row)

    out: List[Dict[str, Any]] = []
    metric_keys = [
        "EM",
        "F1",
        "prompt_tokens_avg",
        "completion_tokens_avg",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "sf_F1_per_1k_prompt_tokens",
    ]
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
                        "EM": 0.0,
                        "F1": 0.0,
                        "prompt_tokens_avg": 0.0,
                        "completion_tokens_avg": 0.0,
                        "retrieval_ms": 0.0,
                        "generation_ms": 0.0,
                        "total_ms": 0.0,
                        "sf_F1_per_1k_prompt_tokens": 0.0,
                        "qa_available": False,
                    }
                )
                continue
            means = _aggregate_mean(group, metric_keys)
            out.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "qa_num_runs": int(len(group)),
                    "qa_num_queries": int(sum(_safe_int(r.get("qa_executed_samples", 0), 0) for r in group)),
                    "qa_available": True,
                    **means,
                }
            )
    return out


def _prepare_retrieval_index(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))
        if not dataset or not profile:
            continue
        index[(dataset, profile)] = dict(row)
    return index


def _dominates(a: Mapping[str, Any], b: Mapping[str, Any], maximize: Sequence[str], minimize: Sequence[str]) -> bool:
    ge_all = True
    better_any = False
    for key in maximize:
        av = _safe_float(a.get(key, 0.0), 0.0)
        bv = _safe_float(b.get(key, 0.0), 0.0)
        if av < bv:
            ge_all = False
            break
        if av > bv:
            better_any = True
    if not ge_all:
        return False
    for key in minimize:
        av = _safe_float(a.get(key, 0.0), 0.0)
        bv = _safe_float(b.get(key, 0.0), 0.0)
        if av > bv:
            ge_all = False
            break
        if av < bv:
            better_any = True
    return ge_all and better_any


def _pareto_frontier(rows: Sequence[Mapping[str, Any]], maximize: Sequence[str], minimize: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    data = [dict(r) for r in rows]
    for i, row in enumerate(data):
        dominated = False
        for j, other in enumerate(data):
            if i == j:
                continue
            if _dominates(other, row, maximize=maximize, minimize=minimize):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return out


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "0.0000"


def _build_markdown(
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
    merged_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
) -> str:
    by_key = {(_safe_text(r.get("dataset")), _safe_text(r.get("profile"))): dict(r) for r in merged_rows}
    frontier_set = {
        (_safe_text(r.get("dataset")), _safe_text(r.get("profile")), _safe_text(r.get("frontier_name")))
        for r in frontier_rows
    }

    lines: List[str] = []
    lines.append("# Phase-6N QA-aware Pareto Validation")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This experiment validates adaptive support span profiles using QA F1, token efficiency, and retrieval evidence density jointly, instead of relying only on retrieval-only metrics."
    )
    lines.append("")

    lines.append("## 2. Compared Profiles")
    lines.append("")
    lines.append("| profile | role |")
    lines.append("|---|---|")
    for profile in profiles:
        lines.append(f"| {profile} | {_profile_role(profile)} |")
    lines.append("")

    lines.append("## 3. QA Results")
    lines.append("")
    lines.append(
        "| dataset | profile | EM | F1 | prompt_tokens_avg | completion_tokens_avg | retrieval_ms | generation_ms | total_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("EM", 0.0)),
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("prompt_tokens_avg", 0.0), 1),
                    _fmt(row.get("completion_tokens_avg", 0.0), 1),
                    _fmt(row.get("retrieval_ms", 0.0), 1),
                    _fmt(row.get("generation_ms", 0.0), 1),
                    _fmt(row.get("total_ms", 0.0), 1),
                )
            )
    lines.append("")

    lines.append("## 4. Retrieval + Density Results")
    lines.append("")
    lines.append(
        "| dataset | profile | rendered_sf_R | rendered_sf_P | rendered_tokens | rendered_sf_F1_per_1k_tokens | render_drop_rate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("rendered_sf_R", 0.0)),
                    _fmt(row.get("rendered_sf_P", 0.0)),
                    _fmt(row.get("rendered_tokens", 0.0), 1),
                    _fmt(row.get("rendered_sf_F1_per_1k_tokens", 0.0)),
                    _fmt(row.get("render_drop_rate", 0.0)),
                )
            )
    lines.append("")

    lines.append("## 5. QA-token Pareto")
    lines.append("")
    lines.append("| dataset | profile | F1 | prompt_tokens_avg | F1_per_1k_prompt | pareto_status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for dataset in datasets:
        for profile in profiles:
            row = by_key.get((dataset, profile), {})
            key = (dataset, profile, "qa_f1_vs_prompt_tokens")
            pareto_status = "frontier" if key in frontier_set else "dominated"
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    profile,
                    _fmt(row.get("F1", 0.0)),
                    _fmt(row.get("prompt_tokens_avg", 0.0), 1),
                    _fmt(row.get("sf_F1_per_1k_prompt_tokens", 0.0)),
                    pareto_status,
                )
            )
    lines.append("")

    lines.append("## 6. Decision")
    lines.append("")
    lines.append("- Which profile is QA champion?")
    lines.append("- Which profile is efficiency champion?")
    lines.append("- Which profile is Pareto candidate?")
    lines.append("- Should we do fine-grained improvement or return to bottleneck redesign?")
    lines.append("")

    lines.append("## 7. Recommendation")
    lines.append("")
    lines.append("- TBD")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare QA-aware Pareto profiles.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    p.add_argument("--input-root", required=True, help="Root containing qa_runs/*/run_manifest.json")
    p.add_argument("--retrieval-audit-root", required=True, help="Root containing aggregate_by_profile_dataset.csv")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    datasets = _tokens(args.datasets)
    profiles = _tokens(args.profiles)
    input_root = Path(args.input_root).resolve()
    retrieval_root = Path(args.retrieval_audit_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_raw_rows = _collect_qa_summary_rows(input_root=input_root, datasets=datasets)
    qa_rows = _summarize_qa_rows(qa_rows=qa_raw_rows, datasets=datasets, profiles=profiles)

    retrieval_csv = _find_retrieval_aggregate_csv(retrieval_root)
    retrieval_rows = _read_csv(retrieval_csv)
    retrieval_index = _prepare_retrieval_index(retrieval_rows)

    merged_rows: List[Dict[str, Any]] = []
    for row in qa_rows:
        dataset = _safe_text(row.get("dataset"))
        profile = _safe_text(row.get("profile"))
        r = dict(retrieval_index.get((dataset, profile), {}) or {})
        merged = {
            "dataset": dataset,
            "profile": profile,
            "EM": _safe_float(row.get("EM", 0.0), 0.0),
            "F1": _safe_float(row.get("F1", 0.0), 0.0),
            "prompt_tokens_avg": _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0),
            "completion_tokens_avg": _safe_float(row.get("completion_tokens_avg", 0.0), 0.0),
            "retrieval_ms": _safe_float(row.get("retrieval_ms", 0.0), 0.0),
            "generation_ms": _safe_float(row.get("generation_ms", 0.0), 0.0),
            "total_ms": _safe_float(row.get("total_ms", 0.0), 0.0),
            "sf_F1_per_1k_prompt_tokens": _safe_float(row.get("sf_F1_per_1k_prompt_tokens", 0.0), 0.0),
            "qa_num_runs": _safe_int(row.get("qa_num_runs", 0), 0),
            "qa_num_queries": _safe_int(row.get("qa_num_queries", 0), 0),
            "qa_available": bool(row.get("qa_available", False)),
            "rendered_sf_R": _safe_float(r.get("rendered_sf_R", 0.0), 0.0),
            "rendered_sf_P": _safe_float(r.get("rendered_sf_P", 0.0), 0.0),
            "rendered_sf_F1": _safe_float(r.get("rendered_sf_F1", 0.0), 0.0),
            "rendered_sf_F1_per_1k_tokens": _safe_float(r.get("rendered_sf_F1_per_1k_tokens", 0.0), 0.0),
            "rendered_tokens": _safe_float(r.get("rendered_tokens", r.get("rendered_tokens_avg", 0.0)), 0.0),
            "avg_item_tokens": _safe_float(
                r.get("avg_item_tokens", r.get("avg_tokens_per_item", r.get("rendered_tokens_avg", 0.0))),
                0.0,
            ),
            "chunk_like_rate": _safe_float(r.get("chunk_like_rate", 0.0), 0.0),
            "render_drop_rate": _safe_float(r.get("render_drop_rate", r.get("selector_drop_rate", 0.0)), 0.0),
            "generation_ready_rate": _safe_float(r.get("generation_ready_rate", 0.0), 0.0),
        }
        merged_rows.append(merged)

    qa_pareto_rows: List[Dict[str, Any]] = []
    frontier_rows: List[Dict[str, Any]] = []

    frontier_specs = [
        ("qa_f1_vs_prompt_tokens", ["F1"], ["prompt_tokens_avg"]),
        ("qa_f1_vs_rendered_tokens", ["F1"], ["rendered_tokens"]),
        ("qa_f1_vs_sf_f1_per_1k_prompt", ["F1", "sf_F1_per_1k_prompt_tokens"], []),
        ("rendered_sf_R_vs_rendered_tokens", ["rendered_sf_R"], ["rendered_tokens"]),
        ("rendered_sf_f1_per_1k_tokens_vs_qa_f1", ["rendered_sf_F1_per_1k_tokens", "F1"], []),
    ]

    for dataset in datasets:
        ds_rows = [dict(r) for r in merged_rows if _safe_text(r.get("dataset")) == dataset]
        for frontier_name, maximize, minimize in frontier_specs:
            # For QA frontiers, ignore rows without QA.
            if frontier_name.startswith("qa_f1"):
                base_rows = [r for r in ds_rows if bool(r.get("qa_available", False)) and _safe_int(r.get("qa_num_queries", 0), 0) > 0]
            else:
                base_rows = list(ds_rows)
            if not base_rows:
                continue
            frontier = _pareto_frontier(base_rows, maximize=maximize, minimize=minimize)
            for row in frontier:
                frontier_rows.append(
                    {
                        "dataset": dataset,
                        "profile": _safe_text(row.get("profile")),
                        "frontier_name": frontier_name,
                        "maximize": ",".join(maximize),
                        "minimize": ",".join(minimize),
                        "F1": _safe_float(row.get("F1", 0.0), 0.0),
                        "prompt_tokens_avg": _safe_float(row.get("prompt_tokens_avg", 0.0), 0.0),
                        "rendered_tokens": _safe_float(row.get("rendered_tokens", 0.0), 0.0),
                        "sf_F1_per_1k_prompt_tokens": _safe_float(row.get("sf_F1_per_1k_prompt_tokens", 0.0), 0.0),
                        "rendered_sf_R": _safe_float(row.get("rendered_sf_R", 0.0), 0.0),
                        "rendered_sf_F1_per_1k_tokens": _safe_float(row.get("rendered_sf_F1_per_1k_tokens", 0.0), 0.0),
                    }
                )

    qa_pareto_rows = sorted(merged_rows, key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("profile"))))
    frontier_rows = sorted(frontier_rows, key=lambda x: (_safe_text(x.get("dataset")), _safe_text(x.get("frontier_name")), _safe_text(x.get("profile"))))

    qa_pareto_csv = output_dir / "qa_pareto_by_profile.csv"
    frontier_csv = output_dir / "pareto_frontier.csv"
    summary_md = output_dir / "PHASE6N_QA_PARETO_SUMMARY.md"

    qa_fields = [
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
        "sf_F1_per_1k_prompt_tokens",
        "rendered_sf_R",
        "rendered_sf_P",
        "rendered_sf_F1",
        "rendered_sf_F1_per_1k_tokens",
        "rendered_tokens",
        "avg_item_tokens",
        "chunk_like_rate",
        "render_drop_rate",
        "generation_ready_rate",
    ]
    frontier_fields = [
        "dataset",
        "profile",
        "frontier_name",
        "maximize",
        "minimize",
        "F1",
        "prompt_tokens_avg",
        "rendered_tokens",
        "sf_F1_per_1k_prompt_tokens",
        "rendered_sf_R",
        "rendered_sf_F1_per_1k_tokens",
    ]
    _write_csv(qa_pareto_csv, qa_pareto_rows, qa_fields)
    _write_csv(frontier_csv, frontier_rows, frontier_fields)

    md = _build_markdown(
        datasets=datasets,
        profiles=profiles,
        merged_rows=qa_pareto_rows,
        frontier_rows=frontier_rows,
    )
    summary_md.write_text(md + "\n", encoding="utf-8")

    run_manifest = {
        "datasets": list(datasets),
        "profiles": list(profiles),
        "input_root": str(input_root),
        "retrieval_audit_root": str(retrieval_root),
        "retrieval_aggregate_csv": str(retrieval_csv),
        "output_dir": str(output_dir),
        "num_qa_rows_raw": int(len(qa_raw_rows)),
        "num_qa_rows_merged": int(len(qa_pareto_rows)),
        "num_frontier_rows": int(len(frontier_rows)),
    }
    (output_dir / "phase6n_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(str(output_dir))


if __name__ == "__main__":
    main()
