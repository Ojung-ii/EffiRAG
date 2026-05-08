#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.canonical_objective import (
    CANONICAL_COPY_SPAN_MODE,
    CANONICAL_NO_BRIDGE_COMPLETENESS_MODE,
    CANONICAL_NO_DATASET_GUARD_MODE,
    CANONICAL_NO_PAIR_COVERAGE_MODE,
    CANONICAL_NO_RUN_OPTIONAL_MODE,
    CANONICAL_NO_SEED_OPTIONAL_MODE,
    CANONICAL_OBJECTIVE_MODES,
    CANONICAL_RENDER_CORE_ONLY_MODE,
)
from effirag.grouped_profiles import (
    COPY_SPAN_INSTRUCTION_GROUPED_V1,
    INSTRUCTION_GROUPED_PROFILE_NAMES,
    REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
    apply_grouped_profile,
)
from effirag.pipeline_trace import trace_active_pipeline


def _parse_csv(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _bool(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload or {}), sort_keys=False, allow_unicode=False), encoding="utf-8")


def _mk_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(str(h) for h in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _build_profile_cfg(dataset: str, profile: str) -> Dict[str, Any]:
    grouped_names = set(INSTRUCTION_GROUPED_PROFILE_NAMES) | {REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN}

    if profile in grouped_names:
        cfg = apply_grouped_profile(
            {},
            dataset=dataset,
            profile_name=profile,
            reference_profile=REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
        )
        cfg["dataset"] = dataset
        return cfg

    if profile in set(CANONICAL_OBJECTIVE_MODES):
        cfg = apply_grouped_profile(
            {},
            dataset=dataset,
            profile_name=COPY_SPAN_INSTRUCTION_GROUPED_V1,
            reference_profile=REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
        )
        cfg["dataset"] = dataset
        cfg["retrieval_objective_mode"] = profile
        cfg["method_profile"] = profile
        cfg["profile_family"] = "canonical_pipeline_simplification"
        cfg["grouped_profile_version"] = "phase3_canonical"
        return cfg

    raise ValueError(f"Unsupported profile: {profile}")


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


def _trace_stage_a(
    datasets: Sequence[str],
    profiles: Sequence[str],
    cfg_root: Path,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        for profile in profiles:
            cfg = _build_profile_cfg(dataset, profile)
            cfg_path = cfg_root / "stage_a_smoke" / profile / dataset / "rag.yaml"
            _dump_yaml(cfg_path, cfg)
            trace = trace_active_pipeline(cfg, dataset=dataset, profile=profile)
            rows.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "config_path": str(cfg_path),
                    "trace": trace,
                }
            )

    key_flags = [
        "bridge_candidate_induction_enabled",
        "role_aware_chunk_scoring_enabled",
        "coverage_selection_enabled",
        "corridor_answer_preserve_enabled",
        "corridor_answer_preserve_guarded_hotpot_enabled",
        "corridor_answer_preserve_confidence_gated_enabled",
        "corridor_path_preserve_enabled",
    ]

    parity: List[Dict[str, Any]] = []
    index = {(r["dataset"], r["profile"]): r for r in rows}
    for dataset in datasets:
        ref = index.get((dataset, COPY_SPAN_INSTRUCTION_GROUPED_V1))
        can = index.get((dataset, CANONICAL_COPY_SPAN_MODE))
        if ref is None or can is None:
            parity.append(
                {
                    "dataset": dataset,
                    "available": False,
                    "flags_equal": False,
                    "details": {},
                }
            )
            continue

        ref_active = ref["trace"].get("active_flags", {})
        ref_inactive = ref["trace"].get("inactive_flags", {})
        can_active = can["trace"].get("active_flags", {})
        can_inactive = can["trace"].get("inactive_flags", {})

        detail = {}
        eq_all = True
        for key in key_flags:
            ref_val = bool(ref_active.get(key, False) or (key not in ref_inactive)) if key not in ref_active else True
            if key in ref_inactive:
                ref_val = False
            can_val = bool(can_active.get(key, False) or (key not in can_inactive)) if key not in can_active else True
            if key in can_inactive:
                can_val = False
            same = ref_val == can_val
            if not same:
                eq_all = False
            detail[key] = {
                "grouped_v1": ref_val,
                "canonical_copy_span": can_val,
                "equal": same,
            }

        parity.append(
            {
                "dataset": dataset,
                "available": True,
                "flags_equal": eq_all,
                "details": detail,
                "grouped_mode": ref["trace"].get("retrieval_objective_mode"),
                "canonical_mode": can["trace"].get("retrieval_objective_mode"),
            }
        )

    all_equal = all(bool(p.get("flags_equal")) for p in parity if p.get("available"))
    return {
        "status": "ok",
        "datasets": list(datasets),
        "profiles": list(profiles),
        "rows": rows,
        "parity": parity,
        "all_flag_parity": bool(all_equal),
    }


def _stage_placeholder(stage_name: str, reason: str, profiles: Sequence[str], datasets: Sequence[str], n_samples: int) -> Dict[str, Any]:
    return {
        "stage": stage_name,
        "status": "not_run",
        "reason": reason,
        "profiles": list(profiles),
        "datasets": list(datasets),
        "n_samples": int(n_samples),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-3 canonical pipeline simplification helper.")
    p.add_argument("--out-root", default="outputs/canonical_pipeline_simplification_phase3")
    p.add_argument("--timestamp", default="")
    p.add_argument("--datasets", default="hotpotqa,2wikimultihopqa,musique,popqa")
    p.add_argument("--trace-profiles", default="light_separator_copy_span_instruction,copy_span_instruction_grouped_v1,canonical_copy_span")
    p.add_argument("--stage-a-samples", type=int, default=100)
    p.add_argument("--stage-b-samples", type=int, default=1000)
    p.add_argument("--stage-c-samples", type=int, default=1000)
    p.add_argument("--run-stage-b", default="false")
    p.add_argument("--run-stage-c", default="false")
    p.add_argument("--run-smoke", default="false")
    args = p.parse_args()

    stamp = str(args.timestamp or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    round_root = Path(args.out_root).resolve() / stamp
    report_root = round_root / "reports"
    cfg_root = round_root / "configs"
    run_root = round_root / "runs"
    report_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    datasets = _parse_csv(args.datasets)
    trace_profiles = _parse_csv(args.trace_profiles)

    stage_a = _trace_stage_a(datasets=datasets, profiles=trace_profiles, cfg_root=cfg_root)
    _write_json(report_root / "active_pipeline_trace.json", stage_a)
    _write_json(report_root / "canonical_parity_stage_a.json", stage_a)

    stage_b_profiles = [
        REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
        COPY_SPAN_INSTRUCTION_GROUPED_V1,
        CANONICAL_COPY_SPAN_MODE,
    ]
    stage_c_profiles = [
        CANONICAL_COPY_SPAN_MODE,
        CANONICAL_NO_SEED_OPTIONAL_MODE,
        CANONICAL_NO_RUN_OPTIONAL_MODE,
        CANONICAL_NO_PAIR_COVERAGE_MODE,
        CANONICAL_NO_BRIDGE_COMPLETENESS_MODE,
        CANONICAL_RENDER_CORE_ONLY_MODE,
        CANONICAL_NO_DATASET_GUARD_MODE,
    ]

    if _bool(args.run_stage_b):
        stage_b = {
            "stage": "stage_b_full",
            "status": "planned",
            "message": "stage_b execution is enabled, but this helper only prepares configs by default.",
            "profiles": stage_b_profiles,
            "datasets": datasets,
            "n_samples": int(args.stage_b_samples),
        }
    else:
        stage_b = _stage_placeholder(
            "stage_b_full",
            "run_stage_b=false",
            stage_b_profiles,
            datasets,
            int(args.stage_b_samples),
        )

    if _bool(args.run_stage_c):
        stage_c = {
            "stage": "stage_c_ablation",
            "status": "planned",
            "message": "stage_c execution is enabled, but this helper only prepares configs by default.",
            "profiles": stage_c_profiles,
            "datasets": datasets,
            "n_samples": int(args.stage_c_samples),
        }
    else:
        stage_c = _stage_placeholder(
            "stage_c_ablation",
            "run_stage_c=false",
            stage_c_profiles,
            datasets,
            int(args.stage_c_samples),
        )

    _write_json(report_root / "canonical_parity_stage_b.json", stage_b)
    _write_json(report_root / "minimal_ablation_stage_c.json", stage_c)

    parity_rows = []
    for row in stage_a.get("parity", []):
        parity_rows.append(
            [
                row.get("dataset"),
                row.get("grouped_mode"),
                row.get("canonical_mode"),
                row.get("flags_equal"),
            ]
        )

    summary_lines = []
    summary_lines.append("# Canonical Pipeline Simplification (Phase-3)")
    summary_lines.append("")
    summary_lines.append("## Stage A")
    summary_lines.append(f"- status: `{stage_a.get('status')}`")
    summary_lines.append(f"- all_flag_parity: `{stage_a.get('all_flag_parity')}`")
    summary_lines.append("")
    summary_lines.append(
        _mk_table(
            ["dataset", "grouped_v1_mode", "canonical_mode", "key_flag_parity"],
            parity_rows,
        )
    )
    summary_lines.append("")
    summary_lines.append("## Stage B")
    summary_lines.append(f"- status: `{stage_b.get('status')}`")
    summary_lines.append(f"- reason/message: `{stage_b.get('reason', stage_b.get('message', ''))}`")
    summary_lines.append("")
    summary_lines.append("## Stage C")
    summary_lines.append(f"- status: `{stage_c.get('status')}`")
    summary_lines.append(f"- reason/message: `{stage_c.get('reason', stage_c.get('message', ''))}`")
    summary_lines.append("")
    summary_lines.append("## Artifacts")
    summary_lines.append(f"- reports: `{report_root}`")
    summary_lines.append(f"- configs: `{cfg_root}`")

    (round_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(str(round_root))


if __name__ == "__main__":
    main()
