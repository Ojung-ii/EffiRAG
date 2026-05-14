#!/usr/bin/env python3
"""
Run 4-dataset copy-span-instruction SOTA config pack.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_ROOT = REPO_ROOT / "configs" / "SOTA_config" / "copy_span_instruction_4ds"
DATASET_ORDER = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_manifest(profile_root: Path) -> Dict:
    manifest_path = profile_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path}. "
            "Run scripts/update_sota_config_copy_span_instruction_4ds.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _tokens(values: Iterable[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for token in str(value).replace(",", " ").split():
            token = token.strip()
            if token:
                out.append(token)
    return out


def _find_latest_summary(output_dir: Path, dataset: str) -> str:
    ds_dir = output_dir / dataset
    if not ds_dir.exists():
        return ""
    summaries = sorted(ds_dir.rglob("rag_summary.json"))
    return str(summaries[-1]) if summaries else ""


def main() -> None:
    p = argparse.ArgumentParser(description="Run 4-dataset SOTA config pack.")
    p.add_argument("--mode", choices=["locked_precomputed", "on_the_fly"], default="locked_precomputed")
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique", "popqa"])
    p.add_argument("--sample-size", "--limit", dest="limit", type=int, default=0, help="Override sample limit when >0.")
    p.add_argument("--profile-root", default=str(DEFAULT_PROFILE_ROOT))
    p.add_argument(
        "--output-root",
        default="",
        help="Output root directory. Default: outputs/sota_config_runs/copy_span_instruction_4ds/<timestamp>_<mode>",
    )
    p.add_argument("--python-bin", default=sys.executable)
    args = p.parse_args()

    profile_root = Path(args.profile_root).resolve()
    manifest = _load_manifest(profile_root)

    dataset_tokens = _tokens(args.datasets)
    datasets = [d for d in DATASET_ORDER if d in dataset_tokens]
    if not datasets:
        raise ValueError(f"No valid dataset requested. got={dataset_tokens}")

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        output_root = (REPO_ROOT / "outputs" / "sota_config_runs" / "copy_span_instruction_4ds" / f"{_ts()}_{args.mode}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_records: List[Dict] = []

    for ds in datasets:
        cfg_path = profile_root / args.mode / f"{ds}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file missing for {ds}: {cfg_path}")

        ds_out = output_root / ds
        cmd = [
            args.python_bin,
            "-m",
            "effirag.run_rag",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(ds_out),
            "--timestamp-output",
            "true",
        ]
        if int(args.limit or 0) > 0:
            cmd.extend(["--limit", str(int(args.limit))])

        print(f"\n=== [{ds}] running")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)

        summary_path = _find_latest_summary(ds_out, ds)
        observed = {}
        expected = {}
        if summary_path:
            try:
                sobj = json.loads(Path(summary_path).read_text(encoding="utf-8"))
                observed = {
                    "f1": float(sobj.get("f1", 0.0) or 0.0),
                    "em": float(sobj.get("em", 0.0) or 0.0),
                    "supporting_fact_recall": float(sobj.get("supporting_fact_recall", 0.0) or 0.0),
                    "supporting_fact_precision": float(sobj.get("supporting_fact_precision", 0.0) or 0.0),
                    "supporting_fact_f1": float(sobj.get("supporting_fact_f1", 0.0) or 0.0),
                    "prompt_tokens_avg": float(sobj.get("prompt_tokens_avg", 0.0) or 0.0),
                    "avg_context_tokens": float(sobj.get("avg_context_tokens", 0.0) or 0.0),
                    "retrieval_latency_ms": float(sobj.get("retrieval_latency_ms", 0.0) or 0.0),
                    "total_latency_ms": float(sobj.get("total_latency_ms", 0.0) or 0.0),
                }
            except Exception:
                observed = {}
        if args.mode == "locked_precomputed":
            expected = dict((manifest.get("datasets", {}) or {}).get(ds, {}).get("expected_metrics_locked", {}) or {})
            if observed and expected:
                f1_delta = float(observed.get("f1", 0.0)) - float(expected.get("f1", 0.0))
                em_delta = float(observed.get("em", 0.0)) - float(expected.get("em", 0.0))
                sf_delta = float(observed.get("supporting_fact_recall", 0.0)) - float(
                    expected.get("supporting_fact_recall", 0.0)
                )
                print(
                    f"[{ds}] observed vs expected | "
                    f"F1 {observed.get('f1', 0.0):.6f} (delta {f1_delta:+.6f}), "
                    f"EM {observed.get('em', 0.0):.6f} (delta {em_delta:+.6f}), "
                    f"SF_recall {observed.get('supporting_fact_recall', 0.0):.6f} (delta {sf_delta:+.6f})"
                )

        run_records.append(
            {
                "dataset": ds,
                "mode": args.mode,
                "config_path": str(cfg_path),
                "exit_code": int(proc.returncode),
                "summary_path": summary_path,
                "observed_metrics": observed,
                "expected_metrics_locked": expected if args.mode == "locked_precomputed" else {},
            }
        )

    run_manifest = {
        "profile_name": manifest.get("profile_name", "copy_span_instruction_4ds"),
        "mode": args.mode,
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_root": str(output_root),
        "records": run_records,
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nRun manifest: {manifest_path}")

    failed = [record for record in run_records if int(record.get("exit_code", 0)) != 0]
    if failed:
        labels = ", ".join(f"{record['dataset']}={record['exit_code']}" for record in failed)
        raise SystemExit(f"Legacy SOTA run failed for: {labels}")


if __name__ == "__main__":
    main()
