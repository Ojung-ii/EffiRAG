#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.grouped_profiles import (
    COPY_SPAN_INSTRUCTION_GROUPED_V0,
    COPY_SPAN_INSTRUCTION_GROUPED_V1,
    COPY_SPAN_INSTRUCTION_MINIMAL_V1,
    INSTRUCTION_GROUPED_PROFILE_NAMES,
    REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
    apply_grouped_profile,
)

WEIGHT_PARITY_ABS_TOL = 1e-9
RECALL_PARITY_ABS_TOL = 1e-12


def _strip(v: Any) -> str:
    return str(v or "").strip()


def _bool(v: Any) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    d = float(denom)
    if abs(d) <= 1e-12:
        return float(default)
    return float(numer) / d


def _parse_csv(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        rr = csv.DictReader(f, delimiter="\t")
        for row in rr:
            rows.append({str(k): str(v or "") for k, v in dict(row).items()})
    return rows


def _load_yaml(path: Path) -> Dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload or {}), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return int(c)


def _find_latest_summary(out_dir: Path, dataset: str) -> Optional[Path]:
    ds_dir = out_dir / dataset
    if not ds_dir.exists():
        return None
    cands = sorted(ds_dir.glob("*/rag_summary.json"))
    if not cands:
        return None
    return cands[-1]


def _is_complete(summary_path: Path, expected_samples: int) -> bool:
    if not summary_path.exists():
        return False
    query_path = summary_path.with_name("rag_query_results.jsonl")
    if not query_path.exists():
        return False
    return _count_jsonl_lines(query_path) >= int(expected_samples)


def _find_dataset_path(qa_root: Path, dataset: str) -> Path:
    p = qa_root / f"{dataset}.json"
    if p.exists():
        return p
    fallback = REPO_ROOT / "data" / f"{dataset}.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"QA file not found for dataset={dataset}")


def _find_corpus_path(corpus_root: Path, dataset: str) -> Path:
    p = corpus_root / f"{dataset}_corpus.json"
    if p.exists():
        return p
    fallback = REPO_ROOT / "data" / f"{dataset}_corpus.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Corpus file not found for dataset={dataset}")


def _find_source_row(round_root: Path, dataset: str) -> Dict[str, str]:
    rows = _read_tsv(round_root / "run_records.tsv")
    hits = [
        r
        for r in rows
        if _strip(r.get("dataset")) == dataset
        and _strip(r.get("variant")) == REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN
        and _strip(r.get("status")) in {"ok", "skipped"}
    ]
    if not hits:
        raise FileNotFoundError(
            f"source run not found for dataset={dataset}, variant={REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN}"
        )
    hits.sort(key=lambda x: (_strip(x.get("end_ts")), _strip(x.get("start_ts"))))
    return hits[-1]


def _extract_metric(summary: Mapping[str, Any], key: str) -> Any:
    if key == "avg_context_tokens":
        if "avg_context_tokens" in summary:
            return summary.get("avg_context_tokens")
        return summary.get("prompt_tokens_avg")
    return summary.get(key)


def _derive_timing_fields(summary: Mapping[str, Any]) -> Dict[str, Any]:
    retrieval_ms = _safe_float(summary.get("retrieval_ms"))
    if retrieval_ms is None:
        retrieval_ms = _safe_float(summary.get("retrieval_latency_ms")) or 0.0
    generation_ms = _safe_float(summary.get("generation_ms"))
    if generation_ms is None:
        generation_ms = _safe_float(summary.get("generation_latency_ms")) or 0.0
    total_ms = _safe_float(summary.get("total_ms"))
    if total_ms is None:
        total_ms = _safe_float(summary.get("total_latency_ms")) or 0.0
    render_ms = _safe_float(summary.get("render_ms")) or 0.0

    source_breakdown = dict(summary.get("retrieval_source_breakdown", {}) or {})
    precomputed = int(source_breakdown.get("precomputed", 0) or 0)
    on_the_fly = int(source_breakdown.get("on_the_fly", 0) or 0)
    total_sources = max(1, precomputed + on_the_fly)
    precomputed_ratio = float(precomputed) / float(total_sources)
    precomputed_used = bool(summary.get("precomputed_retrieval_used", False))
    precomputed_only = precomputed_used and precomputed > 0 and on_the_fly == 0

    total_ge_generation = bool(total_ms + 1e-9 >= generation_ms)
    total_ge_render = bool(total_ms + 1e-9 >= render_ms)
    total_ge_retrieval = bool(total_ms + 1e-9 >= retrieval_ms)
    total_ge_retrieval_plus_generation = bool(total_ms + 1e-9 >= (retrieval_ms + generation_ms))

    if precomputed_only:
        latency_mode = "precomputed_retrieval_latency_reused"
        checks_passed = bool(total_ge_generation and total_ge_render)
        note = (
            "retrieval_ms is loaded from precomputed retrieval rows; "
            "total_ms/generation_ms are measured in this replay run."
        )
    else:
        latency_mode = "current_run_measured"
        checks_passed = bool(
            total_ge_generation
            and total_ge_render
            and total_ge_retrieval
            and total_ge_retrieval_plus_generation
        )
        note = "all latency components are measured in the current run."

    return {
        "precomputed_retrieval_used": precomputed_used,
        "retrieval_source_precomputed_ratio": precomputed_ratio,
        "latency_mode": latency_mode,
        "latency_check_total_ge_generation": total_ge_generation,
        "latency_check_total_ge_retrieval": total_ge_retrieval,
        "latency_check_total_ge_render": total_ge_render,
        "latency_check_total_ge_retrieval_plus_generation": total_ge_retrieval_plus_generation,
        "latency_checks_passed": checks_passed,
        "latency_note": note,
    }


def _load_sample_ids(query_path: Path) -> List[str]:
    out: List[str] = []
    if not query_path.exists():
        return out
    with query_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            sid = obj.get("sample_id")
            if sid is None:
                sid = obj.get("sample_index")
            if sid is None:
                continue
            out.append(str(sid))
    return out


@dataclass
class StageSpec:
    stage: str
    n_samples: int


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retargeted grouped-profile round for copy-span instruction reference.")
    p.add_argument("--source-round-root", default="outputs/prompt_interface_bridge_resolution_round/20260428_120903")
    p.add_argument("--out-root", default="outputs/copy_span_instruction_grouped_round")
    p.add_argument("--timestamp", default="")
    p.add_argument("--datasets", default="hotpotqa,2wikimultihopqa")
    p.add_argument(
        "--profiles",
        default="light_separator_copy_span_instruction,copy_span_instruction_grouped_v0,copy_span_instruction_grouped_v1,copy_span_instruction_minimal_v1",
    )
    p.add_argument("--run-stage-b", default="true")
    p.add_argument("--stage-a-samples", type=int, default=100)
    p.add_argument("--stage-b-samples", type=int, default=1000)
    p.add_argument("--qa-root", default=str(REPO_ROOT / "data" / "qa"))
    p.add_argument("--corpus-root", default=str(REPO_ROOT / "data"))
    p.add_argument("--graph-cache-dir", default=str(REPO_ROOT / "outputs" / "index_cache"))
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--llm-base-url", default="http://localhost:8011/v1")
    p.add_argument("--llm-api-key", default="EMPTY")
    p.add_argument("--skip-completed", default="true")
    return p


def _run_once(cfg_path: Path, dataset: str, n_samples: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "effirag.run_rag",
        "--config",
        str(cfg_path),
        "--dataset",
        str(dataset),
        "--limit",
        str(n_samples),
    ]
    return int(subprocess.run(cmd, check=False).returncode)


def _summary_metrics(summary: Mapping[str, Any], cfg: Mapping[str, Any], stage: str, profile: str, dataset: str) -> Dict[str, Any]:
    sf_p = _extract_metric(summary, "supporting_fact_precision")
    sf_f1 = _extract_metric(summary, "supporting_fact_f1")
    rendered_sf_p = _extract_metric(summary, "rendered_supporting_fact_precision")
    rendered_sf_f1 = _extract_metric(summary, "rendered_supporting_fact_f1")
    prompt_tokens_avg = _extract_metric(summary, "prompt_tokens_avg")
    completion_tokens_avg = _extract_metric(summary, "completion_tokens_avg")
    answer_surface_token_density = _extract_metric(summary, "answer_surface_token_density")
    em = _extract_metric(summary, "em")
    f1 = _extract_metric(summary, "f1")
    abgf = _extract_metric(summary, "abgf_support_present_generation_fail")

    pt = _safe_float(prompt_tokens_avg) or 0.0
    ct = _safe_float(completion_tokens_avg) or 0.0
    tt = pt + ct
    sf_p_f = _safe_float(sf_p) or 0.0
    sf_f1_f = _safe_float(sf_f1) or 0.0
    em_f = _safe_float(em) or 0.0
    f1_f = _safe_float(f1) or 0.0
    abgf_f = _safe_float(abgf) or 0.0

    token_fields = {
        "prompt_tokens_avg": pt,
        "completion_tokens_avg": ct,
        "total_tokens_avg": tt,
        "answer_surface_token_density": _safe_float(answer_surface_token_density),
        "sf_P_per_1k_prompt_tokens": 1000.0 * _safe_div(sf_p_f, pt, 0.0),
        "sf_F1_per_1k_prompt_tokens": 1000.0 * _safe_div(sf_f1_f, pt, 0.0),
        "em_per_1k_prompt_tokens": 1000.0 * _safe_div(em_f, pt, 0.0),
        "f1_per_1k_prompt_tokens": 1000.0 * _safe_div(f1_f, pt, 0.0),
        "abgf_per_1k_prompt_tokens": 1000.0 * _safe_div(abgf_f, pt, 0.0),
        "sf_P_per_1k_total_tokens": 1000.0 * _safe_div(sf_p_f, tt, 0.0),
        "sf_F1_per_1k_total_tokens": 1000.0 * _safe_div(sf_f1_f, tt, 0.0),
        "em_per_1k_total_tokens": 1000.0 * _safe_div(em_f, tt, 0.0),
        "f1_per_1k_total_tokens": 1000.0 * _safe_div(f1_f, tt, 0.0),
        "abgf_per_1k_total_tokens": 1000.0 * _safe_div(abgf_f, tt, 0.0),
        "em_per_100ms": _extract_metric(summary, "em_per_100ms"),
        "f1_per_100ms": _extract_metric(summary, "f1_per_100ms"),
    }
    timing_fields = _derive_timing_fields(summary)

    out = {
        "stage": stage,
        "dataset": dataset,
        "profile": profile,
        "Recall@1": _extract_metric(summary, "recall_at_1"),
        "Recall@5": _extract_metric(summary, "recall_at_5"),
        "Recall@10": _extract_metric(summary, "recall_at_10"),
        "EM": em,
        "F1": f1,
        "ABGF": abgf,
        "sf_P": sf_p,
        "sf_F1": sf_f1,
        "supporting_fact_precision": sf_p,
        "supporting_fact_f1": sf_f1,
        "rendered_sf_P": rendered_sf_p,
        "rendered_sf_F1": rendered_sf_f1,
        "rendered_supporting_fact_precision": rendered_sf_p,
        "rendered_supporting_fact_f1": rendered_sf_f1,
        "output_overlap": _extract_metric(summary, "output_overlap_answer_bearing"),
        "minimal_support_subset_coverage": _extract_metric(summary, "minimal_support_subset_coverage"),
        "equivalent_evidence_coverage": _extract_metric(summary, "equivalent_evidence_coverage"),
        "avg_context_tokens": _extract_metric(summary, "avg_context_tokens"),
        "retrieval_ms": _extract_metric(summary, "retrieval_ms"),
        "generation_ms": _extract_metric(summary, "generation_ms"),
        "total_ms": _extract_metric(summary, "total_ms"),
        "final_order_strategy": cfg.get("order_strategy"),
        "precomputed_retrieval_strict": cfg.get("precomputed_retrieval_strict"),
        "retrieval_source_counts": summary.get("retrieval_source_breakdown", {}),
        "prompt_variant": cfg.get("prompt_variant"),
        "added_instruction": cfg.get("added_instruction"),
    }
    out.update(token_fields)
    out.update(timing_fields)
    return out


def _mk_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(str(h) for h in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> None:
    args = _build_parser().parse_args()

    stamp = _strip(args.timestamp) or datetime.now().strftime("%Y%m%d_%H%M%S")
    round_root = Path(args.out_root).resolve() / stamp
    cfg_root = round_root / "configs"
    run_root = round_root / "runs"
    report_root = round_root / "reports"
    log_root = round_root / "logs"
    for p in (cfg_root, run_root, report_root, log_root):
        p.mkdir(parents=True, exist_ok=True)

    source_round_root = Path(args.source_round_root).resolve()
    qa_root = Path(args.qa_root).resolve()
    corpus_root = Path(args.corpus_root).resolve()
    datasets = _parse_csv(args.datasets)
    profiles = _parse_csv(args.profiles)
    skip_completed = _bool(args.skip_completed)

    stages: List[StageSpec] = [StageSpec(stage="stage_a_smoke", n_samples=int(args.stage_a_samples))]
    if _bool(args.run_stage_b):
        stages.append(StageSpec(stage="stage_b_full", n_samples=int(args.stage_b_samples)))

    # Ensure profile names are explicit and non-ambiguous.
    ambiguous = {
        "copy_span_grouped_v0",
        "copy_span_grouped_v1",
        "copy_span_minimal_v1",
        "copy_span_grouped_no_dataset_toggle",
    }
    for p in profiles:
        if p in ambiguous:
            raise SystemExit(f"Ambiguous profile name is not allowed in retargeted round: {p}")

    allowed_profiles = set(INSTRUCTION_GROUPED_PROFILE_NAMES) | {REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN}
    unknown_profiles = [p for p in profiles if p not in allowed_profiles]
    if unknown_profiles:
        raise SystemExit(f"Unsupported profiles: {unknown_profiles}")

    run_manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for stage_spec in stages:
        for dataset in datasets:
            qa_path = _find_dataset_path(qa_root, dataset)
            corpus_path = _find_corpus_path(corpus_root, dataset)
            source_row = _find_source_row(source_round_root, dataset)
            source_cfg_path = Path(_strip(source_row.get("config_path"))).resolve()
            source_summary_path = Path(_strip(source_row.get("summary_path"))).resolve()
            source_query_path = Path(_strip(source_row.get("precomputed_retrieval_path"))).resolve()
            if not source_query_path.exists():
                source_query_path = source_summary_path.with_name("rag_query_results.jsonl")
            if not source_query_path.exists():
                raise FileNotFoundError(f"Missing source precomputed retrieval: {source_query_path}")
            source_cfg = _load_yaml(source_cfg_path)

            for profile in profiles:
                cfg_path = cfg_root / stage_spec.stage / profile / dataset / "rag.yaml"
                out_dir = run_root / stage_spec.stage / profile / "rag"
                out_dir.mkdir(parents=True, exist_ok=True)
                cfg_path.parent.mkdir(parents=True, exist_ok=True)

                cfg = dict(source_cfg)
                cfg["dataset"] = dataset
                cfg["split"] = "validation"
                cfg["data_path"] = str(qa_path)
                cfg["global_corpus_path"] = str(corpus_path)
                cfg["graph_cache_dir"] = str(Path(args.graph_cache_dir).resolve())
                cfg["output_dir"] = str(out_dir)
                cfg["timestamp_output"] = True
                cfg["limit"] = int(stage_spec.n_samples)
                cfg["run_qa"] = True
                cfg["retrieval_only"] = False
                cfg["generator"] = "vllm"
                cfg["model_name"] = str(args.model_name)
                cfg["llm_base_url"] = str(args.llm_base_url)
                cfg["llm_api_key"] = str(args.llm_api_key)
                cfg["openie_mode"] = "llm"
                cfg["openie_model_name"] = str(args.model_name)
                cfg["openie_local_files_only"] = True
                cfg["openie_api_base_url"] = str(args.llm_base_url)
                cfg["openie_api_key"] = str(args.llm_api_key)
                cfg["evaluator_mode"] = "hipporag2_parity"
                cfg["num_workers"] = 1
                cfg["force_rebuild_graph_index"] = False
                cfg["canonical_variant_name"] = str(profile)
                cfg["selected_query_count"] = int(stage_spec.n_samples)
                cfg["precomputed_retrieval_path"] = str(source_query_path)
                cfg["precomputed_retrieval_strict"] = True

                cfg = apply_grouped_profile(
                    cfg,
                    dataset=dataset,
                    profile_name=profile,
                    reference_profile=REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
                )
                _dump_yaml(cfg_path, cfg)

                existing_summary = _find_latest_summary(out_dir, dataset)
                if skip_completed and existing_summary is not None and _is_complete(existing_summary, stage_spec.n_samples):
                    run_manifest.append(
                        {
                            "stage": stage_spec.stage,
                            "profile": profile,
                            "dataset": dataset,
                            "n_samples": int(stage_spec.n_samples),
                            "config_path": str(cfg_path),
                            "output_dir": str(out_dir),
                            "exit_code": 0,
                            "summary_path": str(existing_summary),
                            "source_config_path": str(source_cfg_path),
                            "source_summary_path": str(source_summary_path),
                            "source_query_path": str(source_query_path),
                            "status": "skipped",
                        }
                    )
                    continue

                t0 = time.time()
                rc = _run_once(cfg_path=cfg_path, dataset=dataset, n_samples=stage_spec.n_samples)
                elapsed = time.time() - t0
                latest_summary = _find_latest_summary(out_dir, dataset)
                status = "ok" if (rc == 0 and latest_summary is not None) else "failed"

                entry = {
                    "stage": stage_spec.stage,
                    "profile": profile,
                    "dataset": dataset,
                    "n_samples": int(stage_spec.n_samples),
                    "config_path": str(cfg_path),
                    "output_dir": str(out_dir),
                    "exit_code": int(rc),
                    "summary_path": str(latest_summary) if latest_summary is not None else "",
                    "source_config_path": str(source_cfg_path),
                    "source_summary_path": str(source_summary_path),
                    "source_query_path": str(source_query_path),
                    "elapsed_sec": round(float(elapsed), 3),
                    "status": status,
                }
                run_manifest.append(entry)
                if status != "ok":
                    failures.append(entry)

    _write_json(round_root / "run_manifest.json", run_manifest)
    _write_json(round_root / "failures.json", {"count": len(failures), "items": failures})
    if failures:
        raise SystemExit(f"run failures detected: {len(failures)}")

    # Collect metrics and deltas.
    metric_rows: List[Dict[str, Any]] = []
    index: Dict[tuple, Dict[str, Any]] = {}
    for row in run_manifest:
        if row.get("status") not in {"ok", "skipped"}:
            continue
        summary_path = Path(_strip(row.get("summary_path")))
        cfg_path = Path(_strip(row.get("config_path")))
        if not summary_path.exists() or not cfg_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = _load_yaml(cfg_path)
        mrow = _summary_metrics(
            summary=summary,
            cfg=cfg,
            stage=_strip(row.get("stage")),
            profile=_strip(row.get("profile")),
            dataset=_strip(row.get("dataset")),
        )
        metric_rows.append(mrow)
        index[(mrow["stage"], mrow["dataset"], mrow["profile"])] = mrow

    delta_rows: List[Dict[str, Any]] = []
    for m in metric_rows:
        stage = str(m["stage"])
        dataset = str(m["dataset"])
        profile = str(m["profile"])
        if profile == REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN:
            continue
        ref = index.get((stage, dataset, REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN))
        if ref is None:
            continue
        delta_rows.append(
            {
                "stage": stage,
                "dataset": dataset,
                "profile": profile,
                "delta_em": (_safe_float(m.get("EM")) or 0.0) - (_safe_float(ref.get("EM")) or 0.0),
                "delta_f1": (_safe_float(m.get("F1")) or 0.0) - (_safe_float(ref.get("F1")) or 0.0),
                "delta_abgf": (_safe_float(m.get("ABGF")) or 0.0) - (_safe_float(ref.get("ABGF")) or 0.0),
                "delta_overlap": (_safe_float(m.get("output_overlap")) or 0.0)
                - (_safe_float(ref.get("output_overlap")) or 0.0),
                "delta_ctx_tokens_pct": (
                    100.0
                    * (
                        (_safe_float(m.get("avg_context_tokens")) or 0.0)
                        - (_safe_float(ref.get("avg_context_tokens")) or 0.0)
                    )
                    / max(1e-12, (_safe_float(ref.get("avg_context_tokens")) or 1.0))
                ),
                "delta_retrieval_ms_pct": (
                    100.0
                    * (
                        (_safe_float(m.get("retrieval_ms")) or 0.0)
                        - (_safe_float(ref.get("retrieval_ms")) or 0.0)
                    )
                    / max(1e-12, (_safe_float(ref.get("retrieval_ms")) or 1.0))
                ),
                "delta_equiv_evidence": (_safe_float(m.get("equivalent_evidence_coverage")) or 0.0)
                - (_safe_float(ref.get("equivalent_evidence_coverage")) or 0.0),
            }
        )

    _write_json(report_root / "metric_rows.json", metric_rows)
    _write_json(report_root / "delta_rows.json", delta_rows)

    # v0 weight parity (config-level parity against reference profile in this round).
    v0_weight_parity: Dict[str, Any] = {"profiles": {}, "all_passed": True}
    weight_keys = [
        "seed_score_semantic_weight",
        "seed_score_graph_weight",
        "seed_score_anchor_weight",
        "run_score_semantic_weight",
        "run_score_anchor_weight",
        "run_score_structure_weight",
        "run_score_bridge_weight",
        "run_score_redundancy_weight",
        "embedding_weight",
        "alpha",
        "beta",
        "gamma_main",
        "delta_support",
        "eta_connector",
        "zeta_query",
        "xi_locality",
        "lambda_redundancy",
    ]
    for ds in datasets:
        ref_row = index.get(("stage_a_smoke", ds, REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN))
        v0_row = index.get(("stage_a_smoke", ds, COPY_SPAN_INSTRUCTION_GROUPED_V0))
        if ref_row is None or v0_row is None:
            v0_weight_parity["profiles"][ds] = {"missing": True, "passed": False}
            v0_weight_parity["all_passed"] = False
            continue
        ref_cfg_path = next(
            Path(r["config_path"])
            for r in run_manifest
            if r["stage"] == "stage_a_smoke"
            and r["dataset"] == ds
            and r["profile"] == REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN
        )
        v0_cfg_path = next(
            Path(r["config_path"])
            for r in run_manifest
            if r["stage"] == "stage_a_smoke" and r["dataset"] == ds and r["profile"] == COPY_SPAN_INSTRUCTION_GROUPED_V0
        )
        ref_cfg = _load_yaml(ref_cfg_path)
        v0_cfg = _load_yaml(v0_cfg_path)
        key_rows = []
        passed = True
        for k in weight_keys:
            rv = _safe_float(ref_cfg.get(k))
            vv = _safe_float(v0_cfg.get(k))
            eq = (rv is not None and vv is not None and abs(rv - vv) <= WEIGHT_PARITY_ABS_TOL)
            if not eq:
                passed = False
            key_rows.append({"key": k, "reference": rv, "v0": vv, "equal": eq})
        v0_weight_parity["profiles"][ds] = {"passed": passed, "rows": key_rows}
        if not passed:
            v0_weight_parity["all_passed"] = False
    _write_json(report_root / "v0_weight_parity.json", v0_weight_parity)

    # v0 recall parity on stage A + sample id alignment.
    v0_recall_parity: Dict[str, Any] = {"datasets": {}, "all_passed": True}
    for ds in datasets:
        ref_m = index.get(("stage_a_smoke", ds, REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN))
        v0_m = index.get(("stage_a_smoke", ds, COPY_SPAN_INSTRUCTION_GROUPED_V0))
        if ref_m is None or v0_m is None:
            v0_recall_parity["datasets"][ds] = {"missing": True, "passed": False}
            v0_recall_parity["all_passed"] = False
            continue
        deltas = {
            "delta_r1": (_safe_float(v0_m.get("Recall@1")) or 0.0) - (_safe_float(ref_m.get("Recall@1")) or 0.0),
            "delta_r5": (_safe_float(v0_m.get("Recall@5")) or 0.0) - (_safe_float(ref_m.get("Recall@5")) or 0.0),
            "delta_r10": (_safe_float(v0_m.get("Recall@10")) or 0.0) - (_safe_float(ref_m.get("Recall@10")) or 0.0),
        }
        ref_summary = next(
            Path(r["summary_path"])
            for r in run_manifest
            if r["stage"] == "stage_a_smoke"
            and r["dataset"] == ds
            and r["profile"] == REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN
        )
        v0_summary = next(
            Path(r["summary_path"])
            for r in run_manifest
            if r["stage"] == "stage_a_smoke" and r["dataset"] == ds and r["profile"] == COPY_SPAN_INSTRUCTION_GROUPED_V0
        )
        ref_ids = _load_sample_ids(ref_summary.with_name("rag_query_results.jsonl"))
        v0_ids = _load_sample_ids(v0_summary.with_name("rag_query_results.jsonl"))
        same_count = len(ref_ids) == len(v0_ids)
        same_ids = set(ref_ids) == set(v0_ids)
        same_order = same_count and all(a == b for a, b in zip(ref_ids, v0_ids))
        passed = (
            abs(deltas["delta_r1"]) <= RECALL_PARITY_ABS_TOL
            and abs(deltas["delta_r5"]) <= RECALL_PARITY_ABS_TOL
            and abs(deltas["delta_r10"]) <= RECALL_PARITY_ABS_TOL
            and same_count
            and same_ids
            and same_order
        )
        v0_recall_parity["datasets"][ds] = {
            **deltas,
            "same_sample_count": same_count,
            "same_sample_ids": same_ids,
            "same_sample_order": same_order,
            "passed": passed,
        }
        if not passed:
            v0_recall_parity["all_passed"] = False
    _write_json(report_root / "v0_recall_parity_stage_a.json", v0_recall_parity)

    # markdown summary
    rows = []
    for m in metric_rows:
        rows.append(
            [
                m["stage"],
                m["dataset"],
                m["profile"],
                f"{(_safe_float(m.get('Recall@1')) or 0.0):.6f}",
                f"{(_safe_float(m.get('Recall@5')) or 0.0):.6f}",
                f"{(_safe_float(m.get('Recall@10')) or 0.0):.6f}",
                f"{(_safe_float(m.get('EM')) or 0.0):.6f}",
                f"{(_safe_float(m.get('F1')) or 0.0):.6f}",
                f"{(_safe_float(m.get('ABGF')) or 0.0):.6f}",
                f"{(_safe_float(m.get('avg_context_tokens')) or 0.0):.3f}",
                f"{(_safe_float(m.get('retrieval_ms')) or 0.0):.3f}",
                f"{(_safe_float(m.get('generation_ms')) or 0.0):.3f}",
                f"{(_safe_float(m.get('total_ms')) or 0.0):.3f}",
                str(m.get("final_order_strategy")),
                str(m.get("precomputed_retrieval_strict")),
            ]
        )
    metric_table = _mk_markdown_table(
        headers=[
            "stage",
            "dataset",
            "profile",
            "Recall@1",
            "Recall@5",
            "Recall@10",
            "EM",
            "F1",
            "ABGF",
            "avg_context_tokens",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
            "final_order_strategy",
            "precomputed_retrieval_strict",
        ],
        rows=rows,
    )

    drows = []
    for d in delta_rows:
        drows.append(
            [
                d["stage"],
                d["dataset"],
                d["profile"],
                f"{d['delta_em']:.6f}",
                f"{d['delta_f1']:.6f}",
                f"{d['delta_abgf']:.6f}",
                f"{d['delta_overlap']:.6f}",
                f"{d['delta_ctx_tokens_pct']:.6f}",
                f"{d['delta_retrieval_ms_pct']:.6f}",
                f"{d['delta_equiv_evidence']:.6f}",
            ]
        )
    delta_table = _mk_markdown_table(
        headers=[
            "stage",
            "dataset",
            "profile",
            "delta_em",
            "delta_f1",
            "delta_abgf",
            "delta_overlap",
            "delta_ctx_tokens_pct",
            "delta_retrieval_ms_pct",
            "delta_equiv_evidence",
        ],
        rows=drows,
    )

    # threshold checks for stage_b on hotpot/2wiki
    target_datasets = {"hotpotqa", "2wikimultihopqa"}
    stage_b_grouped_rows = [
        d
        for d in delta_rows
        if d.get("stage") == "stage_b_full"
        and d.get("dataset") in target_datasets
        and d.get("profile") == COPY_SPAN_INSTRUCTION_GROUPED_V1
    ]
    stage_b_minimal_rows = [
        d
        for d in delta_rows
        if d.get("stage") == "stage_b_full"
        and d.get("dataset") in target_datasets
        and d.get("profile") == COPY_SPAN_INSTRUCTION_MINIMAL_V1
    ]

    stage_b_threshold: Dict[str, Any] = {"available": False}
    if stage_b_grouped_rows:
        n_rows = max(1, len(stage_b_grouped_rows))
        avg_f1_drop = -sum(float(r["delta_f1"]) for r in stage_b_grouped_rows) / n_rows
        worst_f1_drop = max([-float(r["delta_f1"]) for r in stage_b_grouped_rows] + [0.0])
        avg_em_drop = -sum(float(r["delta_em"]) for r in stage_b_grouped_rows) / n_rows
        avg_ctx_increase_pct = sum(float(r["delta_ctx_tokens_pct"]) for r in stage_b_grouped_rows) / n_rows
        avg_retrieval_increase_pct = sum(float(r["delta_retrieval_ms_pct"]) for r in stage_b_grouped_rows) / n_rows
        worst_equiv_evidence_drop = max([-float(r["delta_equiv_evidence"]) for r in stage_b_grouped_rows] + [0.0])

        checks = {
            "avg_f1_drop_le_0.01": avg_f1_drop <= 0.01,
            "worst_f1_drop_le_0.02": worst_f1_drop <= 0.02,
            "avg_em_drop_le_0.01": avg_em_drop <= 0.01,
            "avg_context_tokens_increase_pct_le_5": avg_ctx_increase_pct <= 5.0,
            "avg_retrieval_ms_increase_pct_le_5": avg_retrieval_increase_pct <= 5.0,
            "worst_equivalent_evidence_drop_le_0.03": worst_equiv_evidence_drop <= 0.03,
        }
        stage_b_threshold = {
            "available": True,
            "profile": COPY_SPAN_INSTRUCTION_GROUPED_V1,
            "datasets": sorted(target_datasets),
            "summary": {
                "avg_f1_drop": avg_f1_drop,
                "worst_f1_drop": worst_f1_drop,
                "avg_em_drop": avg_em_drop,
                "avg_context_tokens_increase_pct": avg_ctx_increase_pct,
                "avg_retrieval_ms_increase_pct": avg_retrieval_increase_pct,
                "worst_equivalent_evidence_drop": worst_equiv_evidence_drop,
            },
            "checks": checks,
            "overall_passed": all(bool(v) for v in checks.values()),
        }

        if stage_b_minimal_rows:
            n_min = max(1, len(stage_b_minimal_rows))
            stage_b_threshold["minimal_v1_summary"] = {
                "avg_f1_drop": -sum(float(r["delta_f1"]) for r in stage_b_minimal_rows) / n_min,
                "avg_em_drop": -sum(float(r["delta_em"]) for r in stage_b_minimal_rows) / n_min,
            }

        _write_json(report_root / "stage_b_threshold_check.json", stage_b_threshold)

    stage_a_rows = [r for r in rows if r[0] == "stage_a_smoke"]
    stage_b_rows_table = [r for r in rows if r[0] == "stage_b_full"]
    stage_a_table = _mk_markdown_table(
        headers=[
            "stage",
            "dataset",
            "profile",
            "Recall@1",
            "Recall@5",
            "Recall@10",
            "EM",
            "F1",
            "ABGF",
            "avg_context_tokens",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
            "final_order_strategy",
            "precomputed_retrieval_strict",
        ],
        rows=stage_a_rows,
    )
    stage_b_table = _mk_markdown_table(
        headers=[
            "stage",
            "dataset",
            "profile",
            "Recall@1",
            "Recall@5",
            "Recall@10",
            "EM",
            "F1",
            "ABGF",
            "avg_context_tokens",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
            "final_order_strategy",
            "precomputed_retrieval_strict",
        ],
        rows=stage_b_rows_table,
    )

    stage_b_diag_rows = []
    for m in metric_rows:
        if m.get("stage") != "stage_b_full":
            continue
        stage_b_diag_rows.append(
            [
                str(m.get("dataset")),
                str(m.get("profile")),
                f"{(_safe_float(m.get('sf_P')) or 0.0):.6f}",
                f"{(_safe_float(m.get('sf_F1')) or 0.0):.6f}",
                f"{(_safe_float(m.get('prompt_tokens_avg')) or 0.0):.3f}",
                f"{(_safe_float(m.get('completion_tokens_avg')) or 0.0):.3f}",
                f"{(_safe_float(m.get('sf_F1_per_1k_prompt_tokens')) or 0.0):.6f}",
                str(m.get("latency_mode")),
                str(m.get("latency_checks_passed")),
            ]
        )
    stage_b_diag_table = _mk_markdown_table(
        headers=[
            "dataset",
            "profile",
            "sf_P",
            "sf_F1",
            "prompt_tokens_avg",
            "completion_tokens_avg",
            "sf_F1_per_1k_prompt_tokens",
            "latency_mode",
            "latency_checks_passed",
        ],
        rows=stage_b_diag_rows,
    )

    recommendation = "v0 parity가 불충분하여 v1 해석 보류"
    if bool(v0_weight_parity.get("all_passed")) and bool(v0_recall_parity.get("all_passed")):
        if stage_b_threshold.get("available"):
            if bool(stage_b_threshold.get("overall_passed")):
                recommendation = "grouped_v1이 copy-span instruction 기준 단순화 후보로 수용 가능"
            else:
                recommendation = "grouped_v1은 기준 미달이므로 reference 유지, 추가 보정/분석 필요"
        else:
            recommendation = "Stage B 미실행 상태이므로 v1 승격 보류"

    summary_md = (
        "# Copy-Span Instruction Grouped Round Summary\n\n"
        "## Reference Definition\n"
        "- method_name: `copy-span`\n"
        "- reference_profile: `light_separator_copy_span_instruction`\n"
        "- reference_label: `copy-span/light_separator_copy_span_instruction`\n"
        "- previous_reference_profile: `champion_reconfirm`\n"
        "- retarget_reason: `generation_interface_sota`\n\n"
        "## Retarget Rationale\n"
        "- 기존 grouped Phase-2는 `champion_reconfirm` 기준으로 유효했지만 strongest copy-span interface 보존 주장에는 불충분했습니다.\n"
        "- 이 라운드는 동일 grouped 설계를 copy-span instruction 인터페이스로 재타깃합니다.\n\n"
        "## Profiles\n"
        f"- reference: `{REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN}`\n"
        f"- grouped_v0: `{COPY_SPAN_INSTRUCTION_GROUPED_V0}`\n"
        f"- grouped_v1: `{COPY_SPAN_INSTRUCTION_GROUPED_V1}`\n"
        f"- minimal_v1: `{COPY_SPAN_INSTRUCTION_MINIMAL_V1}`\n\n"
        "## Dataset Locks\n"
        "- hotpotqa: retrieval_objective_mode=p3_answer_preserve_guarded_hotpot, precomputed_retrieval_strict=true\n"
        "- 2wikimultihopqa: final_top_slice_reorder_enabled=true, precomputed_retrieval_strict=true\n\n"
        "## Expanded v0 Weights\n"
        "- seed: semantic=0.35, graph=0.45, anchor=0.20\n"
        "- run: semantic=0.30, anchor=0.20, structure=0.25, bridge=0.15, redundancy=0.10\n"
        "- render: alpha=1.0, beta=0.35, gamma_main=0.45, delta_support=0.20, eta_connector=0.20, zeta_query=0.12, xi_locality=0.08, lambda_redundancy=0.25\n\n"
        "## v0 Parity Checks\n"
        f"- expanded-weight parity (abs_tol={WEIGHT_PARITY_ABS_TOL}): `{v0_weight_parity.get('all_passed')}`\n"
        f"- stage_a recall/sample parity: `{v0_recall_parity.get('all_passed')}`\n\n"
        "## Stage A Metrics\n"
        + stage_a_table
        + "\n\n## Stage B Metrics\n"
        + stage_b_table
        + "\n\n## Stage B SF/Token/Timing Diagnostics\n"
        + stage_b_diag_table
        + "\n\n## Delta from Reference\n"
        + delta_table
        + "\n\n## Acceptance Threshold Check\n"
        + "```json\n"
        + json.dumps(stage_b_threshold, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        "## Recommendation\n"
        f"- {recommendation}\n"
    )
    (round_root / "summary.md").write_text(summary_md, encoding="utf-8")

    print(str(round_root))


if __name__ == "__main__":
    main()
