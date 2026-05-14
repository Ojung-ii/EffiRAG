#!/usr/bin/env python3
"""
Phase-6E retrieval-only evidence-flow audit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from effirag.evidence_flow_audit import (
    DATASET_ORDER,
    aggregate_by_profile_dataset,
    classify_failure_type,
    find_latest_query_results,
    infer_lost_stage,
    jaccard,
    load_samples_for_dataset,
    overlap_ratio,
    parse_roots,
    stage_metrics,
)


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
            line = str(line or "").strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _stage_from_row(row: Mapping[str, Any], stage: str) -> Tuple[List[str], List[str], str]:
    retrieval = dict((row.get("retrieval", {}) or {}))
    rendered = dict((row.get("rendered", {}) or {}))
    diagnostics = dict((retrieval.get("diagnostics", {}) or {}))
    graph_mode = str(diagnostics.get("graph_mode", "current_entity_graph") or "current_entity_graph")

    if stage == "candidate":
        ids = list(retrieval.get("candidate_sentence_ids", []) or [])
        texts = list(retrieval.get("candidate_sentences", []) or [])
        if (not ids) and diagnostics.get("candidate_text_unit_ids"):
            ids = list(diagnostics.get("candidate_text_unit_ids", []) or [])
            texts = list(diagnostics.get("candidate_texts", []) or [])
        return [str(x) for x in ids], [str(x) for x in texts], graph_mode

    if stage == "selected":
        ids = list(retrieval.get("selected_sentence_ids", []) or [])
        texts = list(retrieval.get("selected_sentences", []) or [])
        return [str(x) for x in ids], [str(x) for x in texts], graph_mode

    if stage == "rendered":
        ids = list(row.get("rendered_sentence_ids", []) or rendered.get("sentence_ids", []) or [])
        texts = list(rendered.get("sentences", []) or [])
        return [str(x) for x in ids], [str(x) for x in texts], graph_mode

    raise ValueError(f"Unsupported stage: {stage}")


def _candidate_provenance_available(row: Mapping[str, Any]) -> bool:
    retrieval = dict((row.get("retrieval", {}) or {}))
    diagnostics = dict((retrieval.get("diagnostics", {}) or {}))
    direct_ids = list(retrieval.get("candidate_sentence_ids", []) or [])
    diag_ids = list(diagnostics.get("candidate_text_unit_ids", []) or [])
    if direct_ids or diag_ids:
        return True
    try:
        union_count = int(diagnostics.get("num_candidates_union", 0) or 0)
    except Exception:
        union_count = 0
    # If candidate pool is non-empty but ids are absent, provenance is unavailable.
    if union_count > 0:
        return False
    return True


def _audit_single_row(dataset: str, profile: str, row: Mapping[str, Any], sample) -> Dict[str, Any]:
    qid = str(row.get("sample_id", "") or "")

    candidate_ids, candidate_texts, graph_mode = _stage_from_row(row, "candidate")
    selected_ids, selected_texts, _ = _stage_from_row(row, "selected")
    rendered_ids, rendered_texts, _ = _stage_from_row(row, "rendered")

    candidate_stage = stage_metrics(sample, candidate_ids, candidate_texts, graph_mode)
    selected_stage = stage_metrics(sample, selected_ids, selected_texts, graph_mode)
    rendered_stage = stage_metrics(sample, rendered_ids, rendered_texts, graph_mode)
    if (not _candidate_provenance_available(row)) and int(candidate_stage.get("count", 0)) <= 0:
        candidate_stage = dict(candidate_stage)
        candidate_stage["id_mapping_available"] = False

    failure_type = classify_failure_type(
        candidate_stage=candidate_stage,
        selected_stage=selected_stage,
        rendered_stage=rendered_stage,
    )

    rendered_set = set(rendered_stage.get("predicted_ids", []) or [])
    selected_set = set(selected_stage.get("predicted_ids", []) or [])
    selected_to_rendered = jaccard(selected_set, rendered_set)

    return {
        "dataset": dataset,
        "qid": qid,
        "profile": profile,
        "gold_support_count": int(rendered_stage.get("gold_support_count", selected_stage.get("gold_support_count", 0))),
        "candidate_ids": list(candidate_stage.get("predicted_ids", []) or []),
        "candidate_matched_gold_ids": list(candidate_stage.get("matched_gold_ids", []) or []),
        "selected_ids": list(selected_stage.get("predicted_ids", []) or []),
        "selected_matched_gold_ids": list(selected_stage.get("matched_gold_ids", []) or []),
        "rendered_ids": list(rendered_stage.get("predicted_ids", []) or []),
        "rendered_matched_gold_ids": list(rendered_stage.get("matched_gold_ids", []) or []),
        "candidate_sf_P": float(candidate_stage.get("sf_P", 0.0)),
        "candidate_sf_R": float(candidate_stage.get("sf_R", 0.0)),
        "candidate_sf_F1": float(candidate_stage.get("sf_F1", 0.0)),
        "candidate_support_hit": int(candidate_stage.get("support_hit", 0)),
        "candidate_tokens": int(candidate_stage.get("tokens", 0)),
        "candidate_sf_F1_per_1k_tokens": float(candidate_stage.get("sf_F1_per_1k_tokens", 0.0)),
        "candidate_count": int(candidate_stage.get("count", 0)),
        "selected_sf_P": float(selected_stage.get("sf_P", 0.0)),
        "selected_sf_R": float(selected_stage.get("sf_R", 0.0)),
        "selected_sf_F1": float(selected_stage.get("sf_F1", 0.0)),
        "selected_support_hit": int(selected_stage.get("support_hit", 0)),
        "selected_tokens": int(selected_stage.get("tokens", 0)),
        "selected_sf_F1_per_1k_tokens": float(selected_stage.get("sf_F1_per_1k_tokens", 0.0)),
        "selected_count": int(selected_stage.get("count", 0)),
        "rendered_sf_P": float(rendered_stage.get("sf_P", 0.0)),
        "rendered_sf_R": float(rendered_stage.get("sf_R", 0.0)),
        "rendered_sf_F1": float(rendered_stage.get("sf_F1", 0.0)),
        "rendered_support_hit": int(rendered_stage.get("support_hit", 0)),
        "rendered_tokens": int(rendered_stage.get("tokens", 0)),
        "rendered_sf_F1_per_1k_tokens": float(rendered_stage.get("sf_F1_per_1k_tokens", 0.0)),
        "rendered_count": int(rendered_stage.get("count", 0)),
        "selected_to_rendered_jaccard": float(selected_to_rendered),
        "extra_tokens_after_selector": int(max(0, int(rendered_stage.get("tokens", 0)) - int(selected_stage.get("tokens", 0)))),
        "extra_sentences_after_selector": int(max(0, int(rendered_stage.get("count", 0)) - int(selected_stage.get("count", 0)))),
        "failure_type": str(failure_type),
        "id_mapping_available": bool(candidate_stage.get("id_mapping_available", False)),
    }


def _dominant_lost_stage(rows: List[Dict[str, Any]]) -> str:
    counts = Counter([str(row.get("lost_stage", "")) for row in rows])
    if not counts:
        return "unknown"
    return str(counts.most_common(1)[0][0] or "unknown")


def _build_legacy_overlap(rows: List[Dict[str, Any]], legacy_profile: str = "legacy_sota") -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        by_key[(str(row.get("dataset", "")), str(row.get("profile", "")), str(row.get("qid", "")))] = row

    datasets = sorted({str(row.get("dataset", "")) for row in rows})
    profiles = sorted({str(row.get("profile", "")) for row in rows if str(row.get("profile", "")) != legacy_profile})
    qids = sorted({str(row.get("qid", "")) for row in rows})

    out: List[Dict[str, Any]] = []
    for dataset in datasets:
        for qid in qids:
            legacy_row = by_key.get((dataset, legacy_profile, qid))
            if not legacy_row:
                continue
            legacy_rendered_ids = list(
                legacy_row.get("rendered_matched_gold_ids", legacy_row.get("rendered_ids", [])) or []
            )
            legacy_rendered_sf_r = float(legacy_row.get("rendered_sf_R", 0.0) or 0.0)
            for profile in profiles:
                row = by_key.get((dataset, profile, qid))
                if not row:
                    continue
                lost_stage = infer_lost_stage(
                    legacy_rendered_sf_r=legacy_rendered_sf_r,
                    profile_candidate_sf_r=float(row.get("candidate_sf_R", 0.0) or 0.0),
                    profile_selected_sf_r=float(row.get("selected_sf_R", 0.0) or 0.0),
                    profile_rendered_sf_r=float(row.get("rendered_sf_R", 0.0) or 0.0),
                )
                out.append(
                    {
                        "dataset": dataset,
                        "qid": qid,
                        "profile": profile,
                        "legacy_rendered_sf_R": legacy_rendered_sf_r,
                        "profile_candidate_sf_R": float(row.get("candidate_sf_R", 0.0) or 0.0),
                        "profile_selected_sf_R": float(row.get("selected_sf_R", 0.0) or 0.0),
                        "profile_rendered_sf_R": float(row.get("rendered_sf_R", 0.0) or 0.0),
                        "legacy_rendered_ids": "|".join(legacy_rendered_ids),
                        "profile_candidate_overlap_with_legacy": float(
                            overlap_ratio(legacy_rendered_ids, row.get("candidate_matched_gold_ids", []) or [])
                        ),
                        "profile_selected_overlap_with_legacy": float(
                            overlap_ratio(legacy_rendered_ids, row.get("selected_matched_gold_ids", []) or [])
                        ),
                        "profile_rendered_overlap_with_legacy": float(
                            overlap_ratio(legacy_rendered_ids, row.get("rendered_matched_gold_ids", []) or [])
                        ),
                        "lost_stage": str(lost_stage),
                    }
                )
    return out


def _summary_markdown(
    aggregate_rows: List[Dict[str, Any]],
    overlap_rows: List[Dict[str, Any]],
    datasets: List[str],
    profiles: List[str],
    roots: Mapping[str, Path],
) -> str:
    lines: List[str] = []
    lines.append("# Phase-6E Retrieval-Only Evidence Flow Audit")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This audit stops generation and diagnoses where support evidence is lost in the retrieval pipeline: "
        "candidate retrieval, selector/reranking, or final rendering."
    )
    lines.append("")
    lines.append("generation skipped: true")
    lines.append("")
    lines.append("## 2. Compared Profiles")
    lines.append("")
    lines.append("| profile | role |")
    lines.append("|---|---|")
    for profile in profiles:
        role = "evidence-density reference" if profile == "legacy_sota" else "unified/dynamic candidate"
        lines.append(f"| {profile} | {role} |")
    lines.append("")
    lines.append("## 3. Aggregate Evidence Flow")
    lines.append("")
    lines.append(
        "| dataset | profile | candidate_sf_R | selected_sf_R | rendered_sf_R | "
        "candidate_sf_P | selected_sf_P | rendered_sf_P | rendered_tokens | rendered_sf_F1_per_1k_tokens |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = next(
                (
                    item
                    for item in aggregate_rows
                    if str(item.get("dataset", "")) == dataset and str(item.get("profile", "")) == profile
                ),
                None,
            )
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | "
                f"{float(row.get('candidate_sf_R', 0.0)):.4f} | "
                f"{float(row.get('selected_sf_R', 0.0)):.4f} | "
                f"{float(row.get('rendered_sf_R', 0.0)):.4f} | "
                f"{float(row.get('candidate_sf_P', 0.0)):.4f} | "
                f"{float(row.get('selected_sf_P', 0.0)):.4f} | "
                f"{float(row.get('rendered_sf_P', 0.0)):.4f} | "
                f"{float(row.get('rendered_tokens_avg', 0.0)):.1f} | "
                f"{float(row.get('rendered_sf_F1_per_1k_tokens', 0.0)):.3f} |"
            )
    lines.append("")
    lines.append("## 4. Failure Taxonomy")
    lines.append("")
    lines.append(
        "| dataset | profile | retrieval_miss_rate | selector_drop_rate | render_drop_rate | low_density_rate | generation_ready_rate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for profile in profiles:
            row = next(
                (
                    item
                    for item in aggregate_rows
                    if str(item.get("dataset", "")) == dataset and str(item.get("profile", "")) == profile
                ),
                None,
            )
            if not row:
                continue
            lines.append(
                f"| {dataset} | {profile} | "
                f"{float(row.get('retrieval_miss_rate', 0.0)):.4f} | "
                f"{float(row.get('selector_drop_rate', 0.0)):.4f} | "
                f"{float(row.get('render_drop_rate', 0.0)):.4f} | "
                f"{float(row.get('low_density_rate', 0.0)):.4f} | "
                f"{float(row.get('generation_ready_rate', 0.0)):.4f} |"
            )
    lines.append("")
    lines.append("## 5. Legacy Overlap Analysis")
    lines.append("")
    lines.append(
        "| dataset | profile | avg_candidate_overlap_with_legacy | "
        "avg_selected_overlap_with_legacy | avg_rendered_overlap_with_legacy | dominant_lost_stage |"
    )
    lines.append("|---|---|---:|---:|---:|---|")
    overlap_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in overlap_rows:
        overlap_group[(str(row.get("dataset", "")), str(row.get("profile", "")))].append(row)
    for dataset in datasets:
        for profile in profiles:
            if profile == "legacy_sota":
                continue
            group = overlap_group.get((dataset, profile), [])
            if not group:
                continue
            cand = sum(float(r.get("profile_candidate_overlap_with_legacy", 0.0)) for r in group) / max(1, len(group))
            sel = sum(float(r.get("profile_selected_overlap_with_legacy", 0.0)) for r in group) / max(1, len(group))
            ren = sum(float(r.get("profile_rendered_overlap_with_legacy", 0.0)) for r in group) / max(1, len(group))
            lines.append(
                f"| {dataset} | {profile} | {cand:.4f} | {sel:.4f} | {ren:.4f} | {_dominant_lost_stage(group)} |"
            )
    lines.append("")
    lines.append("## 6. Diagnosis")
    lines.append("")
    lines.append("Decision tree:")
    lines.append("- If candidate_sf_R is much lower than legacy: retrieval/candidate generation is the bottleneck.")
    lines.append("- If candidate_sf_R is close but selected_sf_R drops: selector/reranker is the bottleneck.")
    lines.append("- If selected_sf_R is close but rendered_sf_R drops: renderer is the bottleneck.")
    lines.append("- If rendered_sf_R is close but QA F1 is low: generation/prompting is the bottleneck.")
    lines.append("")
    lines.append("## 7. Inputs")
    lines.append("")
    for profile in profiles:
        lines.append(f"- {profile}: {str(roots.get(profile, Path('')))}")
    lines.append("")
    lines.append(
        f"_Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} UTC from retrieval-only evidence-flow audit._"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit retrieval-only evidence flow by profile/dataset.")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa"])
    parser.add_argument("--profiles", nargs="+", default=["legacy_sota", "unified_large", "unified_dynamic_compact_v1"])
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Profile roots in key=path format, e.g. legacy_sota=outputs/... unified_large=outputs/...",
    )
    parser.add_argument("--output-dir", default="outputs/phase6e_evidence_flow_audit")
    args = parser.parse_args()

    requested_datasets = set(_tokens(args.datasets))
    datasets = [dataset for dataset in DATASET_ORDER if dataset in requested_datasets]
    if not datasets:
        raise ValueError(f"No valid dataset requested. got={sorted(requested_datasets)}")

    profiles = [str(p).strip() for p in list(args.profiles or []) if str(p).strip()]
    if not profiles:
        raise ValueError("No profiles requested.")

    roots = parse_roots(args.roots)
    missing_profiles = [profile for profile in profiles if profile not in roots]
    if missing_profiles:
        raise ValueError(f"Missing --roots mapping for profiles: {missing_profiles}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_maps: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        sample_maps[dataset] = load_samples_for_dataset(dataset)

    audit_rows: List[Dict[str, Any]] = []
    run_sources: Dict[str, Dict[str, str]] = defaultdict(dict)

    for dataset in datasets:
        sample_map = sample_maps.get(dataset, {})
        for profile in profiles:
            root = roots[profile]
            query_path = find_latest_query_results(root, dataset)
            run_sources[dataset][profile] = str(query_path or "")
            if not query_path:
                continue
            query_rows = _read_jsonl(query_path)
            for row in query_rows:
                qid = str(row.get("sample_id", "") or "")
                sample = sample_map.get(qid)
                audit_rows.append(_audit_single_row(dataset=dataset, profile=profile, row=row, sample=sample))

    aggregate_rows = aggregate_by_profile_dataset(audit_rows)
    legacy_overlap_rows = _build_legacy_overlap(audit_rows, legacy_profile="legacy_sota")

    _write_jsonl(output_dir / "audit_rows.jsonl", audit_rows)
    if aggregate_rows:
        _write_csv(
            output_dir / "aggregate_by_profile_dataset.csv",
            aggregate_rows,
            fieldnames=list(aggregate_rows[0].keys()),
        )
    else:
        _write_csv(
            output_dir / "aggregate_by_profile_dataset.csv",
            [],
            fieldnames=[
                "dataset",
                "profile",
                "num_queries",
                "candidate_sf_P",
                "candidate_sf_R",
                "candidate_sf_F1",
                "candidate_tokens_avg",
                "candidate_sf_F1_per_1k_tokens",
                "selected_sf_P",
                "selected_sf_R",
                "selected_sf_F1",
                "selected_tokens_avg",
                "selected_sf_F1_per_1k_tokens",
                "rendered_sf_P",
                "rendered_sf_R",
                "rendered_sf_F1",
                "rendered_tokens_avg",
                "rendered_sf_F1_per_1k_tokens",
                "retrieval_miss_rate",
                "selector_drop_rate",
                "render_drop_rate",
                "low_density_rate",
                "generation_ready_rate",
            ],
        )
    if legacy_overlap_rows:
        _write_csv(
            output_dir / "legacy_overlap_by_query.csv",
            legacy_overlap_rows,
            fieldnames=list(legacy_overlap_rows[0].keys()),
        )
    else:
        _write_csv(
            output_dir / "legacy_overlap_by_query.csv",
            [],
            fieldnames=[
                "dataset",
                "qid",
                "profile",
                "legacy_rendered_sf_R",
                "profile_candidate_sf_R",
                "profile_selected_sf_R",
                "profile_rendered_sf_R",
                "legacy_rendered_ids",
                "profile_candidate_overlap_with_legacy",
                "profile_selected_overlap_with_legacy",
                "profile_rendered_overlap_with_legacy",
                "lost_stage",
            ],
        )

    summary_md = _summary_markdown(
        aggregate_rows=aggregate_rows,
        overlap_rows=legacy_overlap_rows,
        datasets=datasets,
        profiles=profiles,
        roots=roots,
    )
    (output_dir / "PHASE6E_EVIDENCE_FLOW_AUDIT_SUMMARY.md").write_text(summary_md, encoding="utf-8")

    run_meta = {
        "datasets": datasets,
        "profiles": profiles,
        "roots": {k: str(v) for k, v in roots.items()},
        "query_sources": run_sources,
        "output_dir": str(output_dir),
    }
    (output_dir / "audit_run_manifest.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
