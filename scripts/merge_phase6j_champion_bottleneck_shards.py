#!/usr/bin/env python3
"""
Merge Phase-6J shard outputs into a single report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from effirag.champion_bottleneck_audit import (
    aggregate_query_rows_by_profile_dataset,
    build_phase6j_markdown,
    champion_summary_rows,
    parse_key_value_items,
    summarize_bottleneck_taxonomy,
    summarize_token_decomposition,
    summarize_unit_contract,
    write_csv,
    write_jsonl,
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _tokens(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = _safe_text(line)
            if not line:
                continue
            try:
                rows.append(dict(json.loads(line)))
            except Exception:
                continue
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _dedup_query_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in list(rows or []):
        dataset = _safe_text(row.get("dataset", ""))
        profile = _safe_text(row.get("profile", ""))
        qid = _safe_text(row.get("qid", ""))
        key = (dataset, profile, qid)
        if key not in out:
            out[key] = dict(row)
    return list(out.values())


def _aggregate_overlap_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        dataset = _safe_text(row.get("dataset", ""))
        profile = _safe_text(row.get("profile", ""))
        if profile == "legacy_sota":
            continue
        buckets[(dataset, profile)].append(row)

    out: List[Dict[str, Any]] = []
    for (dataset, profile), group in sorted(buckets.items()):
        stage_counter: Dict[str, int] = defaultdict(int)
        for item in group:
            stage_counter[_safe_text(item.get("dominant_lost_stage", "unknown"))] += 1
        dominant_stage = "unknown"
        if stage_counter:
            dominant_stage = sorted(stage_counter.items(), key=lambda x: (-x[1], x[0]))[0][0]
        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "num_queries": int(len(group)),
                "legacy_candidate_overlap": float(
                    sum(_safe_float(item.get("legacy_candidate_overlap", 0.0), 0.0) for item in group) / max(1, len(group))
                ),
                "legacy_selected_overlap": float(
                    sum(_safe_float(item.get("legacy_selected_overlap", 0.0), 0.0) for item in group) / max(1, len(group))
                ),
                "legacy_rendered_overlap": float(
                    sum(_safe_float(item.get("legacy_rendered_overlap", 0.0), 0.0) for item in group) / max(1, len(group))
                ),
                "dominant_lost_stage": dominant_stage,
            }
        )
    return out


def _dedup_utility_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in list(rows or []):
        dataset = _safe_text(row.get("dataset", ""))
        profile = _safe_text(row.get("profile", ""))
        key = (dataset, profile)
        if key not in out:
            out[key] = dict(row)
    return list(out.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Phase-6J shard outputs.")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa"])
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=[
            "legacy_sota",
            "unified_large",
            "unified_candidate_recall_boost_v1",
            "unified_candidate_recall_boost_density_rerank_v1",
            "unified_gl_rcedr_v1",
            "unified_gl_rcedr_v2",
        ],
    )
    parser.add_argument("--roots", nargs="+", required=True, help="Profile roots in key=path format.")
    parser.add_argument("--query-jsonls", nargs="+", required=True, help="Shard query_level_bottlenecks.jsonl files.")
    parser.add_argument(
        "--utility-csvs",
        nargs="*",
        default=[],
        help="Shard utility_component_by_profile.csv files.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    datasets = [d for d in _tokens(args.datasets) if d]
    profiles = [p for p in _tokens(args.profiles) if p]
    roots = parse_key_value_items(args.roots)

    query_rows: List[Dict[str, Any]] = []
    for path_text in list(args.query_jsonls or []):
        query_rows.extend(_read_jsonl(Path(path_text)))
    query_rows = _dedup_query_rows(query_rows)

    utility_summary_rows: List[Dict[str, Any]] = []
    for path_text in list(args.utility_csvs or []):
        utility_summary_rows.extend(_read_csv(Path(path_text)))
    utility_summary_rows = _dedup_utility_summary(utility_summary_rows)

    aggregate_rows = aggregate_query_rows_by_profile_dataset(query_rows)
    token_rows = summarize_token_decomposition(query_rows)
    unit_rows = summarize_unit_contract(query_rows)
    taxonomy_rows = summarize_bottleneck_taxonomy(query_rows)
    overlap_rows = _aggregate_overlap_rows(query_rows)
    champions = champion_summary_rows(
        aggregate_rows=aggregate_rows,
        datasets=datasets,
        profiles=profiles,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "query_level_bottlenecks.jsonl", query_rows)
    write_csv(
        output_dir / "aggregate_by_profile_dataset.csv",
        aggregate_rows,
        fieldnames=list(aggregate_rows[0].keys()) if aggregate_rows else ["dataset", "profile", "num_queries"],
    )
    write_csv(
        output_dir / "legacy_overlap_by_query.csv",
        [
            {
                "dataset": _safe_text(row.get("dataset", "")),
                "qid": _safe_text(row.get("qid", "")),
                "profile": _safe_text(row.get("profile", "")),
                "legacy_rendered_sf_R": _safe_float(row.get("legacy_rendered_sf_R", 0.0), 0.0),
                "profile_candidate_sf_R": _safe_float(row.get("candidate_sf_R", 0.0), 0.0),
                "profile_selected_sf_R": _safe_float(row.get("selected_sf_R", 0.0), 0.0),
                "profile_rendered_sf_R": _safe_float(row.get("rendered_sf_R", 0.0), 0.0),
                "legacy_candidate_overlap": _safe_float(row.get("legacy_candidate_overlap", 0.0), 0.0),
                "legacy_selected_overlap": _safe_float(row.get("legacy_selected_overlap", 0.0), 0.0),
                "legacy_rendered_overlap": _safe_float(row.get("legacy_rendered_overlap", 0.0), 0.0),
                "dominant_lost_stage": _safe_text(row.get("dominant_lost_stage", "unknown")),
            }
            for row in query_rows
            if _safe_text(row.get("profile", "")) != "legacy_sota"
        ],
        fieldnames=[
            "dataset",
            "qid",
            "profile",
            "legacy_rendered_sf_R",
            "profile_candidate_sf_R",
            "profile_selected_sf_R",
            "profile_rendered_sf_R",
            "legacy_candidate_overlap",
            "legacy_selected_overlap",
            "legacy_rendered_overlap",
            "dominant_lost_stage",
        ],
    )
    write_csv(
        output_dir / "token_decomposition_by_profile.csv",
        token_rows,
        fieldnames=list(token_rows[0].keys()) if token_rows else ["dataset", "profile"],
    )
    write_csv(
        output_dir / "unit_contract_by_profile.csv",
        unit_rows,
        fieldnames=list(unit_rows[0].keys()) if unit_rows else ["dataset", "profile"],
    )
    write_csv(
        output_dir / "utility_component_by_profile.csv",
        utility_summary_rows,
        fieldnames=list(utility_summary_rows[0].keys()) if utility_summary_rows else [
            "dataset",
            "profile",
            "num_scored_units",
            "avg_G_selected",
            "avg_B_selected",
            "avg_R_selected",
            "avg_C_selected",
            "avg_G_rejected",
            "avg_C_rejected",
        ],
    )
    write_csv(
        output_dir / "bottleneck_taxonomy_by_profile.csv",
        taxonomy_rows,
        fieldnames=list(taxonomy_rows[0].keys()) if taxonomy_rows else ["dataset", "profile"],
    )

    summary_md = build_phase6j_markdown(
        datasets=datasets,
        profiles=profiles,
        roots=roots,
        champion_rows=champions,
        aggregate_rows=aggregate_rows,
        overlap_rows=overlap_rows,
        token_rows=token_rows,
        unit_rows=unit_rows,
        utility_rows=utility_summary_rows,
        taxonomy_rows=taxonomy_rows,
    )
    (output_dir / "PHASE6J_CHAMPION_BOTTLENECK_AUDIT.md").write_text(summary_md, encoding="utf-8")
    (output_dir / "audit_run_manifest.json").write_text(
        json.dumps(
            {
                "datasets": datasets,
                "profiles": profiles,
                "roots": {k: str(v) for k, v in roots.items()},
                "query_jsonls": [str(Path(x)) for x in list(args.query_jsonls or [])],
                "utility_csvs": [str(Path(x)) for x in list(args.utility_csvs or [])],
                "num_query_rows_after_dedup": int(len(query_rows)),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()

