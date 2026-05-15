#!/usr/bin/env python3
"""
Phase-6J champion-based bottleneck audit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from effirag.champion_bottleneck_audit import (
    aggregate_query_rows_by_profile_dataset,
    build_phase6j_markdown,
    candidate_provenance_available,
    champion_summary_rows,
    classify_bottleneck_taxonomy,
    density_metrics_from_row,
    extract_utility_component_rows,
    legacy_overlap_for_query,
    load_optional_qa_index,
    merge_optional_qa_metrics,
    parse_key_value_items,
    read_jsonl,
    stage_from_row,
    summarize_bottleneck_taxonomy,
    summarize_token_decomposition,
    summarize_unit_contract,
    summarize_utility_component_rows,
    token_decomposition_from_row,
    unit_contract_from_row,
    write_csv,
    write_jsonl,
)
from effirag.evidence_flow_audit import DATASET_ORDER, find_latest_query_results, load_samples_for_dataset, stage_metrics
from effirag.utils import safe_div


def _tokens(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


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


def _reconcile_decomposition_with_prompt_override(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt_tokens = _safe_int(row.get("prompt_tokens", 0), 0)
    evidence_tokens = _safe_int(row.get("evidence_tokens", 0), 0)
    metadata_tokens = _safe_int(row.get("metadata_tokens", 0), 0)
    separator_tokens = _safe_int(row.get("separator_tokens", 0), 0)
    instruction_tokens = _safe_int(row.get("instruction_tokens", 0), 0)
    total = int(evidence_tokens + metadata_tokens + separator_tokens + instruction_tokens)
    gap = int(prompt_tokens - total)
    gap_abs = abs(gap)
    gap_ratio = float(safe_div(float(gap_abs), float(max(1, prompt_tokens))))
    row["decomposition_sum_tokens"] = int(total)
    row["decomposition_gap_tokens"] = int(gap)
    row["decomposition_gap_ratio"] = float(gap_ratio)
    row["decomposition_warning"] = bool(prompt_tokens > 0 and (gap_abs > 64 or gap_ratio > 0.15))
    return row


def _aggregate_overlap_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        buckets[(str(row.get("dataset", "")), str(row.get("profile", "")))].append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, profile), group in sorted(buckets.items()):
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
                "dominant_lost_stage": max(
                    ((str(item.get("dominant_lost_stage", "")), idx) for idx, item in enumerate(group)),
                    key=lambda x: sum(1 for y in group if str(y.get("dominant_lost_stage", "")) == x[0]),
                )[0]
                if group
                else "unknown",
            }
        )
    return out


def _build_query_row(
    *,
    dataset: str,
    profile: str,
    row: Mapping[str, Any],
    sample: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    qid = str(row.get("sample_id", "") or "")

    candidate_ids, candidate_texts, graph_mode = stage_from_row(row, "candidate")
    selected_ids, selected_texts, _ = stage_from_row(row, "selected")
    rendered_ids, rendered_texts, _ = stage_from_row(row, "rendered")

    candidate_stage = stage_metrics(sample, candidate_ids, candidate_texts, graph_mode)
    selected_stage = stage_metrics(sample, selected_ids, selected_texts, graph_mode)
    rendered_stage = stage_metrics(sample, rendered_ids, rendered_texts, graph_mode)

    if (not candidate_provenance_available(row)) and int(candidate_stage.get("count", 0)) <= 0:
        candidate_stage = dict(candidate_stage)
        candidate_stage["id_mapping_available"] = False

    decomp = token_decomposition_from_row(row)
    unit = unit_contract_from_row(row)
    rendered = dict((row.get("rendered", {}) or {}))
    rendered_meta = dict((rendered.get("metadata", {}) or {}))
    density = density_metrics_from_row(
        row,
        prompt_tokens=int(decomp.get("prompt_tokens", 0)),
        rendered_sf_r=float(rendered_stage.get("sf_R", 0.0)),
        rendered_sf_p=float(rendered_stage.get("sf_P", 0.0)),
        rendered_sf_f1=float(rendered_stage.get("sf_F1", 0.0)),
    )
    metrics = dict((row.get("metrics", {}) or {}))
    generation_diag = dict((row.get("generation_diagnostics", {}) or {}))

    candidate_sf_r = float(candidate_stage.get("sf_R", 0.0))
    selected_sf_r = float(selected_stage.get("sf_R", 0.0))
    candidate_to_selected_drop = max(0.0, candidate_sf_r - selected_sf_r)
    selector_drop_rate = float(safe_div(candidate_to_selected_drop, max(1.0e-12, candidate_sf_r)))

    query_row: Dict[str, Any] = {
        "dataset": dataset,
        "qid": qid,
        "profile": profile,
        "gold_support_count": int(rendered_stage.get("gold_support_count", selected_stage.get("gold_support_count", 0))),
        "candidate_ids": list(candidate_stage.get("predicted_ids", []) or []),
        "selected_ids": list(selected_stage.get("predicted_ids", []) or []),
        "rendered_ids": list(rendered_stage.get("predicted_ids", []) or []),
        "candidate_matched_gold_ids": list(candidate_stage.get("matched_gold_ids", []) or []),
        "selected_matched_gold_ids": list(selected_stage.get("matched_gold_ids", []) or []),
        "rendered_matched_gold_ids": list(rendered_stage.get("matched_gold_ids", []) or []),
        "candidate_pool_size": int(candidate_stage.get("count", 0)),
        "candidate_tokens": int(candidate_stage.get("tokens", 0)),
        "candidate_sf_P": float(candidate_stage.get("sf_P", 0.0)),
        "candidate_sf_R": float(candidate_stage.get("sf_R", 0.0)),
        "candidate_sf_F1": float(candidate_stage.get("sf_F1", 0.0)),
        "candidate_sf_F1_per_1k_tokens": float(candidate_stage.get("sf_F1_per_1k_tokens", 0.0)),
        "selected_count": int(selected_stage.get("count", 0)),
        "selected_tokens": int(selected_stage.get("tokens", 0)),
        "selected_sf_P": float(selected_stage.get("sf_P", 0.0)),
        "selected_sf_R": float(selected_stage.get("sf_R", 0.0)),
        "selected_sf_F1": float(selected_stage.get("sf_F1", 0.0)),
        "candidate_to_selected_drop": float(candidate_to_selected_drop),
        "selector_drop_rate": float(selector_drop_rate),
        "rendered_count": int(rendered_stage.get("count", 0)),
        "rendered_tokens": int(rendered_stage.get("tokens", 0)),
        "rendered_sf_P": float(rendered_stage.get("sf_P", 0.0)),
        "rendered_sf_R": float(rendered_stage.get("sf_R", 0.0)),
        "rendered_sf_F1": float(rendered_stage.get("sf_F1", 0.0)),
        "rendered_sf_F1_per_1k_tokens": float(rendered_stage.get("sf_F1_per_1k_tokens", 0.0)),
        "rendered_sentence_count": int(unit.get("rendered_sentence_count", 0)),
        "rendered_unit_type": str(unit.get("rendered_unit_type", "")),
        "is_sentence_level": bool(unit.get("is_sentence_level", False)),
        "is_chunk_like": bool(unit.get("is_chunk_like", False)),
        "chunk_like_rate": float(unit.get("chunk_like_rate", 0.0)),
        "avg_tokens_per_rendered_item": float(unit.get("avg_tokens_per_rendered_item", 0.0)),
        "avg_tokens_per_item": float(unit.get("avg_tokens_per_item", 0.0)),
        "max_item_tokens": int(unit.get("max_item_tokens", 0)),
        "prompt_tokens": int(decomp.get("prompt_tokens", 0)),
        "evidence_tokens": int(decomp.get("evidence_tokens", 0)),
        "metadata_tokens": int(decomp.get("metadata_tokens", 0)),
        "instruction_tokens": int(decomp.get("instruction_tokens", 0)),
        "separator_tokens": int(decomp.get("separator_tokens", 0)),
        "decomposition_sum_tokens": int(decomp.get("decomposition_sum_tokens", 0)),
        "decomposition_gap_tokens": int(decomp.get("decomposition_gap_tokens", 0)),
        "decomposition_gap_ratio": float(decomp.get("decomposition_gap_ratio", 0.0)),
        "decomposition_warning": bool(decomp.get("decomposition_warning", False)),
        "redundancy_rate": float(density.get("redundancy_rate", 0.0)),
        "source_repetition_rate": float(density.get("source_repetition_rate", 0.0)),
        "entity_repetition_rate": float(density.get("entity_repetition_rate", 0.0)),
        "coverage_gain_per_token": float(density.get("coverage_gain_per_token", 0.0)),
        "sf_F1_per_1k_prompt": float(density.get("sf_F1_per_1k_prompt", 0.0)),
        "QA_EM": float(_safe_float(metrics.get("em", metrics.get("EM", 0.0)), 0.0)),
        "QA_F1": float(_safe_float(metrics.get("f1", metrics.get("F1", 0.0)), 0.0)),
        "completion_tokens": int(_safe_int(generation_diag.get("completion_tokens", 0), 0)),
        "answer_error_type_if_available": str(generation_diag.get("answer_error_type", "")),
        "sentence_contract_render_enabled": bool(rendered_meta.get("sentence_contract_render_enabled", False)),
        "selected_item_count": int(_safe_int(rendered_meta.get("selected_item_count", selected_stage.get("count", 0)), 0)),
        "rendered_item_count": int(_safe_int(rendered_meta.get("rendered_item_count", rendered_stage.get("count", 0)), 0)),
        "selected_to_rendered_preservation_rate": float(
            _safe_float(rendered_meta.get("selected_to_rendered_preservation_rate", 0.0), 0.0)
        ),
        "render_drop_rate": float(_safe_float(rendered_meta.get("render_drop_rate", 0.0), 0.0)),
        "truncated_item_rate": float(_safe_float(rendered_meta.get("truncated_item_rate", 0.0), 0.0)),
        "minimal_span_fallback_rate": float(_safe_float(rendered_meta.get("minimal_span_fallback_rate", 0.0), 0.0)),
        "candidate_contains_legacy_evidence": False,
        "legacy_candidate_overlap": 0.0,
        "legacy_selected_overlap": 0.0,
        "legacy_rendered_overlap": 0.0,
        "candidate_overlap_with_legacy": 0.0,
        "selected_overlap_with_legacy": 0.0,
        "rendered_overlap_with_legacy": 0.0,
        "dominant_lost_stage": "unknown",
        "legacy_rendered_sf_R": 0.0,
    }

    utility_rows = extract_utility_component_rows(
        dataset=dataset,
        profile=profile,
        qid=qid,
        row=row,
        selected_ids=query_row["selected_ids"],
    )
    return query_row, utility_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-6J champion-based bottleneck audit.")
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
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Profile roots in key=path format.",
    )
    parser.add_argument(
        "--qa-jsons",
        nargs="*",
        default=[],
        help="Optional dataset=qa_json mappings. Empty values are ignored.",
    )
    parser.add_argument("--output-dir", default="outputs/phase6j_champion_bottleneck_audit")
    args = parser.parse_args()

    requested_datasets = set(_tokens(args.datasets))
    datasets = [dataset for dataset in DATASET_ORDER if dataset in requested_datasets]
    if not datasets:
        raise ValueError(f"No valid dataset requested. got={sorted(requested_datasets)}")

    profiles = [str(p).strip() for p in list(args.profiles or []) if str(p).strip()]
    if not profiles:
        raise ValueError("No profiles requested.")

    roots = parse_key_value_items(args.roots)
    missing_profiles = [profile for profile in profiles if profile not in roots]
    if missing_profiles:
        raise ValueError(f"Missing --roots mapping for profiles: {missing_profiles}")

    qa_json_map = parse_key_value_items(args.qa_jsons)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_maps: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        sample_maps[dataset] = load_samples_for_dataset(dataset)

    qa_profile_index: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    qa_dataset_index: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for dataset in datasets:
        qa_path = qa_json_map.get(dataset)
        if not qa_path:
            continue
        profile_map, dataset_map = load_optional_qa_index(qa_path, dataset)
        qa_profile_index.update(profile_map)
        for key, rows in dataset_map.items():
            qa_dataset_index[key].extend(list(rows or []))

    query_rows: List[Dict[str, Any]] = []
    utility_component_rows: List[Dict[str, Any]] = []
    query_sources: Dict[str, Dict[str, str]] = defaultdict(dict)
    row_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for dataset in datasets:
        sample_map = sample_maps.get(dataset, {})
        for profile in profiles:
            root = roots[profile]
            query_path = find_latest_query_results(root, dataset)
            query_sources[dataset][profile] = str(query_path or "")
            if not query_path:
                continue
            for raw_row in read_jsonl(query_path):
                qid = str(raw_row.get("sample_id", "") or "")
                sample = sample_map.get(qid)
                query_row, utility_rows = _build_query_row(
                    dataset=dataset,
                    profile=profile,
                    row=raw_row,
                    sample=sample,
                )
                query_row = merge_optional_qa_metrics(
                    dataset=dataset,
                    profile=profile,
                    qid=qid,
                    base_row=query_row,
                    qa_profile_index=qa_profile_index,
                    qa_dataset_index=qa_dataset_index,
                )
                query_row = _reconcile_decomposition_with_prompt_override(query_row)
                query_rows.append(query_row)
                utility_component_rows.extend(utility_rows)
                row_by_key[(dataset, profile, qid)] = query_row

    legacy_overlap_by_query: List[Dict[str, Any]] = []
    for dataset in datasets:
        qids = sorted({qid for ds, _profile, qid in row_by_key.keys() if ds == dataset})
        for qid in qids:
            legacy_row = row_by_key.get((dataset, "legacy_sota", qid))
            if not legacy_row:
                continue
            legacy_ids = list(
                legacy_row.get("rendered_ids", legacy_row.get("rendered_matched_gold_ids", [])) or []
            )
            legacy_rendered_sf_r = float(_safe_float(legacy_row.get("rendered_sf_R", 0.0), 0.0))
            for profile in profiles:
                row = row_by_key.get((dataset, profile, qid))
                if not row:
                    continue
                overlap = legacy_overlap_for_query(
                    legacy_rendered_ids=legacy_ids,
                    candidate_ids=row.get("candidate_ids", []),
                    selected_ids=row.get("selected_ids", []),
                    rendered_ids=row.get("rendered_ids", []),
                )
                row["legacy_rendered_sf_R"] = float(legacy_rendered_sf_r)
                row["legacy_candidate_overlap"] = float(overlap["legacy_candidate_overlap"])
                row["legacy_selected_overlap"] = float(overlap["legacy_selected_overlap"])
                row["legacy_rendered_overlap"] = float(overlap["legacy_rendered_overlap"])
                row["candidate_overlap_with_legacy"] = float(overlap["legacy_candidate_overlap"])
                row["selected_overlap_with_legacy"] = float(overlap["legacy_selected_overlap"])
                row["rendered_overlap_with_legacy"] = float(overlap["legacy_rendered_overlap"])
                row["dominant_lost_stage"] = str(overlap["dominant_lost_stage"])
                row["candidate_contains_legacy_evidence"] = bool(overlap["legacy_candidate_overlap"] > 0.0)
                row["candidate_to_selected_drop"] = max(
                    0.0,
                    _safe_float(row.get("candidate_sf_R", 0.0), 0.0) - _safe_float(row.get("selected_sf_R", 0.0), 0.0),
                )
                row["selector_drop_rate"] = float(
                    safe_div(
                        float(row["candidate_to_selected_drop"]),
                        float(max(1.0e-12, _safe_float(row.get("candidate_sf_R", 0.0), 0.0))),
                    )
                )
                taxonomy = classify_bottleneck_taxonomy(row)
                row.update(taxonomy)
                if profile != "legacy_sota":
                    legacy_overlap_by_query.append(
                        {
                            "dataset": dataset,
                            "qid": qid,
                            "profile": profile,
                            "legacy_rendered_sf_R": float(legacy_rendered_sf_r),
                            "profile_candidate_sf_R": float(_safe_float(row.get("candidate_sf_R", 0.0), 0.0)),
                            "profile_selected_sf_R": float(_safe_float(row.get("selected_sf_R", 0.0), 0.0)),
                            "profile_rendered_sf_R": float(_safe_float(row.get("rendered_sf_R", 0.0), 0.0)),
                            "legacy_candidate_overlap": float(overlap["legacy_candidate_overlap"]),
                            "legacy_selected_overlap": float(overlap["legacy_selected_overlap"]),
                            "legacy_rendered_overlap": float(overlap["legacy_rendered_overlap"]),
                            "dominant_lost_stage": str(overlap["dominant_lost_stage"]),
                        }
                    )

    # Fallback taxonomy when legacy row is missing for some query/profile.
    for row in query_rows:
        if "dominant_bottleneck" not in row:
            row.update(classify_bottleneck_taxonomy(row))

    aggregate_rows = aggregate_query_rows_by_profile_dataset(query_rows)
    token_rows = summarize_token_decomposition(query_rows)
    unit_rows = summarize_unit_contract(query_rows)
    utility_rows = summarize_utility_component_rows(utility_component_rows)
    taxonomy_rows = summarize_bottleneck_taxonomy(query_rows)
    overlap_aggregate_rows = _aggregate_overlap_rows(legacy_overlap_by_query)
    champions = champion_summary_rows(
        aggregate_rows=aggregate_rows,
        datasets=datasets,
        profiles=profiles,
    )

    write_jsonl(output_dir / "query_level_bottlenecks.jsonl", query_rows)
    sentence_contract_rows: List[Dict[str, Any]] = []
    for row in query_rows:
        if not bool(row.get("sentence_contract_render_enabled", False)):
            continue
        sentence_contract_rows.append(
            {
                "dataset": str(row.get("dataset", "")),
                "qid": str(row.get("qid", "")),
                "profile": str(row.get("profile", "")),
                "selected_item_count": int(_safe_int(row.get("selected_item_count", 0), 0)),
                "rendered_item_count": int(_safe_int(row.get("rendered_item_count", row.get("rendered_count", 0)), 0)),
                "render_drop_rate": float(_safe_float(row.get("render_drop_rate", 0.0), 0.0)),
                "selected_to_rendered_preservation_rate": float(
                    _safe_float(row.get("selected_to_rendered_preservation_rate", 0.0), 0.0)
                ),
                "avg_item_tokens": float(_safe_float(row.get("avg_tokens_per_item", 0.0), 0.0)),
                "max_item_tokens": int(_safe_int(row.get("max_item_tokens", 0), 0)),
                "chunk_like_rate": float(_safe_float(row.get("chunk_like_rate", 0.0), 0.0)),
                "sentence_level_rate": float(
                    _safe_float(row.get("sentence_level_rate", 1.0 - _safe_float(row.get("chunk_like_rate", 0.0), 0.0)), 0.0)
                ),
                "metadata_tokens": int(_safe_int(row.get("metadata_tokens", 0), 0)),
                "evidence_text_tokens": int(_safe_int(row.get("evidence_tokens", 0), 0)),
                "separator_tokens": int(_safe_int(row.get("separator_tokens", 0), 0)),
                "truncated_item_rate": float(_safe_float(row.get("truncated_item_rate", 0.0), 0.0)),
                "minimal_span_fallback_rate": float(_safe_float(row.get("minimal_span_fallback_rate", 0.0), 0.0)),
            }
        )
    if sentence_contract_rows:
        write_jsonl(output_dir / "query_level_sentence_contract.jsonl", sentence_contract_rows)
    write_csv(
        output_dir / "aggregate_by_profile_dataset.csv",
        aggregate_rows,
        fieldnames=list(aggregate_rows[0].keys()) if aggregate_rows else [
            "dataset",
            "profile",
            "num_queries",
            "candidate_sf_R",
            "selected_sf_R",
            "rendered_sf_R",
        ],
    )
    write_csv(
        output_dir / "legacy_overlap_by_query.csv",
        legacy_overlap_by_query,
        fieldnames=list(legacy_overlap_by_query[0].keys()) if legacy_overlap_by_query else [
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
        fieldnames=list(token_rows[0].keys()) if token_rows else [
            "dataset",
            "profile",
            "prompt_tokens",
            "evidence_tokens",
            "metadata_tokens",
            "instruction_tokens",
            "separator_tokens",
            "avg_tokens_per_item",
            "decomposition_warning_rate",
        ],
    )
    write_csv(
        output_dir / "unit_contract_by_profile.csv",
        unit_rows,
        fieldnames=list(unit_rows[0].keys()) if unit_rows else [
            "dataset",
            "profile",
            "sentence_level_rate",
            "chunk_like_rate",
            "avg_item_tokens",
            "max_item_tokens",
        ],
    )
    write_csv(
        output_dir / "utility_component_by_profile.csv",
        utility_rows,
        fieldnames=list(utility_rows[0].keys()) if utility_rows else [
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
        fieldnames=list(taxonomy_rows[0].keys()) if taxonomy_rows else [
            "dataset",
            "profile",
            "candidate_bottleneck",
            "selection_bottleneck",
            "unit_bottleneck",
            "density_bottleneck",
            "generation_bottleneck",
            "dominant_bottleneck",
        ],
    )

    summary_md = build_phase6j_markdown(
        datasets=datasets,
        profiles=profiles,
        roots=roots,
        champion_rows=champions,
        aggregate_rows=aggregate_rows,
        overlap_rows=overlap_aggregate_rows,
        token_rows=token_rows,
        unit_rows=unit_rows,
        utility_rows=utility_rows,
        taxonomy_rows=taxonomy_rows,
    )
    (output_dir / "PHASE6J_CHAMPION_BOTTLENECK_AUDIT.md").write_text(summary_md, encoding="utf-8")

    run_manifest = {
        "datasets": datasets,
        "profiles": profiles,
        "roots": {k: str(v) for k, v in roots.items()},
        "qa_jsons": {k: str(v) for k, v in qa_json_map.items()},
        "query_sources": query_sources,
        "output_dir": str(output_dir),
        "num_query_rows": int(len(query_rows)),
        "num_utility_rows": int(len(utility_component_rows)),
    }
    (output_dir / "audit_run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
