#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _default_config_path(dataset: str) -> Path:
    return Path("configs/main_config/phase7_evidence_flow") / f"{dataset}.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase7 evidence flow with controlled variants.")
    ap.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "2wikimultihopqa", "musique", "popqa"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=[
            "full",
            "phase1_only",
            "no_phase2_refinement",
            "phase2_budget_x2",
            "phase2_budget_x3",
            "legacy_compatible_render",
        ],
    )
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--python-bin", type=str, default=os.environ.get("PYTHON_BIN", "/home/ojungii/miniconda3/envs/effirag/bin/python"))
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--diagnostics-max-examples", type=int, default=100)
    ap.add_argument("--diagnostics-fail-on-unit-mismatch", action="store_true")
    ap.add_argument("--phase7-enable-phase2-refinement", action="store_true")
    ap.add_argument(
        "--phase7-objective-mode",
        type=str,
        default="normalized_equal_weight",
        choices=[
            "normalized_equal_weight",
            "a_only",
            "a_plus_b",
            "a_minus_r",
            "a_plus_b_minus_r",
            "a_plus_decay_minus_r",
            "a_plus_bq_minus_r",
            "a_plus_bq_minus_rq",
            "aq_plus_bq_minus_r",
            "aq_plus_bq_minus_rq",
        ],
    )
    ap.add_argument("--phase7-lambda-bridge", type=float, default=1.0)
    ap.add_argument("--phase7-lambda-decay", type=float, default=1.0)
    ap.add_argument("--phase7-lambda-bq", type=float, default=1.0)
    ap.add_argument("--phase7-mu-redundancy", type=float, default=1.0)
    ap.add_argument("--phase7-conditional-redundancy-enabled", action="store_true")
    ap.add_argument("--phase7-corridor-enabled", action="store_true")
    ap.add_argument("--phase7-corridor-max-anchors", type=int, default=4)
    ap.add_argument("--phase7-corridor-max-seeds", type=int, default=16)
    ap.add_argument("--phase7-corridor-max-hops", type=int, default=3)
    ap.add_argument("--phase7-corridor-max-paths-per-pair", type=int, default=1)
    ap.add_argument("--phase7-corridor-degree-cap", type=int, default=100)
    ap.add_argument("--phase7-corridor-max-pairs", type=int, default=64)
    ap.add_argument("--phase7-anchor-decay-enabled", action="store_true")
    ap.add_argument("--phase7-anchor-decay-gamma", type=float, default=0.7)
    ap.add_argument("--phase7-anchor-decay-max-hops", type=int, default=4)
    ap.add_argument("--phase7-query-intent-enabled", action="store_true")
    ap.add_argument("--phase7-intent-phase1-enabled", action="store_true")
    ap.add_argument("--phase7-intent-max-relation-candidates", type=int, default=32)
    ap.add_argument("--phase7-intent-max-answer-type-candidates", type=int, default=32)
    ap.add_argument("--phase7-intent-max-entity-candidates", type=int, default=32)
    ap.add_argument("--phase7-intent-candidate-top-m", type=int, default=128)
    ap.add_argument("--phase7-max-context-tokens", type=int, default=192)
    ap.add_argument("--phase7-max-selected-atoms", type=int, default=8)
    ap.add_argument("--phase7-candidate-top-m", type=int, default=96)
    ap.add_argument("--timestamp-output", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve() if str(args.config).strip() else (repo_root / _default_config_path(args.dataset))
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cmd = [
        str(args.python_bin),
        "-m",
        "effirag.run_rag",
        "--config",
        str(config_path),
        "--dataset",
        str(args.dataset),
        "--limit",
        str(int(args.limit)),
        "--phase7-variant",
        str(args.variant),
        "--phase7-enable-phase2-refinement",
        "true" if bool(args.phase7_enable_phase2_refinement) else "false",
        "--phase7-objective-mode",
        str(args.phase7_objective_mode),
        "--phase7-lambda-bridge",
        str(float(args.phase7_lambda_bridge)),
        "--phase7-lambda-decay",
        str(float(args.phase7_lambda_decay)),
        "--phase7-lambda-bq",
        str(float(args.phase7_lambda_bq)),
        "--phase7-mu-redundancy",
        str(float(args.phase7_mu_redundancy)),
        "--phase7-conditional-redundancy-enabled",
        "true" if bool(args.phase7_conditional_redundancy_enabled) else "false",
        "--phase7-corridor-enabled",
        "true" if bool(args.phase7_corridor_enabled) else "false",
        "--phase7-corridor-max-anchors",
        str(int(args.phase7_corridor_max_anchors)),
        "--phase7-corridor-max-seeds",
        str(int(args.phase7_corridor_max_seeds)),
        "--phase7-corridor-max-hops",
        str(int(args.phase7_corridor_max_hops)),
        "--phase7-corridor-max-paths-per-pair",
        str(int(args.phase7_corridor_max_paths_per_pair)),
        "--phase7-corridor-degree-cap",
        str(int(args.phase7_corridor_degree_cap)),
        "--phase7-corridor-max-pairs",
        str(int(args.phase7_corridor_max_pairs)),
        "--phase7-anchor-decay-enabled",
        "true" if bool(args.phase7_anchor_decay_enabled) else "false",
        "--phase7-anchor-decay-gamma",
        str(float(args.phase7_anchor_decay_gamma)),
        "--phase7-anchor-decay-max-hops",
        str(int(args.phase7_anchor_decay_max_hops)),
        "--phase7-query-intent-enabled",
        "true" if bool(args.phase7_query_intent_enabled) else "false",
        "--phase7-intent-phase1-enabled",
        "true" if bool(args.phase7_intent_phase1_enabled) else "false",
        "--phase7-intent-max-relation-candidates",
        str(int(args.phase7_intent_max_relation_candidates)),
        "--phase7-intent-max-answer-type-candidates",
        str(int(args.phase7_intent_max_answer_type_candidates)),
        "--phase7-intent-max-entity-candidates",
        str(int(args.phase7_intent_max_entity_candidates)),
        "--phase7-intent-candidate-top-m",
        str(int(args.phase7_intent_candidate_top_m)),
        "--phase7-max-context-tokens",
        str(int(args.phase7_max_context_tokens)),
        "--phase7-max-selected-atoms",
        str(int(args.phase7_max_selected_atoms)),
        "--phase7-candidate-top-m",
        str(int(args.phase7_candidate_top_m)),
        "--output-dir",
        str(output_root),
        "--timestamp-output",
        "true" if bool(args.timestamp_output) else "false",
    ]

    if args.variant == "legacy_compatible_render":
        cmd.extend(["--render-mode", "flat"])

    if args.diagnostics:
        cmd.extend(
            [
                "--phase7-diagnostics-enabled",
                "true",
                "--phase7-diagnostics-max-examples-to-dump",
                str(max(1, int(args.diagnostics_max_examples))),
                "--phase7-diagnostics-dump-text",
                "true",
                "--phase7-diagnostics-dump-context",
                "true",
                "--phase7-diagnostics-dump-scores",
                "true",
                "--phase7-diagnostics-fail-on-unit-mismatch",
                "true" if bool(args.diagnostics_fail_on_unit_mismatch) else "false",
            ]
        )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    print("[phase7-run] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)

    if args.diagnostics:
        diag_jsonl = output_root / "phase7_diagnostics.jsonl"
        if not diag_jsonl.exists():
            raise FileNotFoundError(f"Diagnostics JSONL not found: {diag_jsonl}")
        summary_json = output_root / "diagnostic_summary.json"
        compare_csv = output_root.parents[1] / "variant_comparison.csv" if len(output_root.parents) >= 2 else output_root / "variant_comparison.csv"
        analyze_cmd = [
            str(args.python_bin),
            "scripts/analyze_phase7_diagnostics.py",
            "--diag-jsonl",
            str(diag_jsonl),
            "--out",
            str(summary_json),
            "--compare-root",
            str(output_root.parents[1] if len(output_root.parents) >= 2 else output_root),
            "--compare-csv",
            str(compare_csv),
        ]
        if args.dataset == "2wikimultihopqa":
            analyze_cmd.extend(
                [
                    "--failure-md",
                    str(output_root / "failure_cases.md"),
                    "--failure-top-k",
                    "20",
                ]
            )
        print("[phase7-analyze] " + " ".join(analyze_cmd))
        subprocess.run(analyze_cmd, check=True, cwd=str(repo_root), env=env)


if __name__ == "__main__":
    main()
