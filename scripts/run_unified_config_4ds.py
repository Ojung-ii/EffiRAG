#!/usr/bin/env python3
"""
Run strict unified copy-span-instruction config profiles.
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
DEFAULT_PROFILE_ROOT = REPO_ROOT / "configs" / "main_config" / "copy_span_instruction_unified"
DATASET_ORDER = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from effirag.unified_copy_span_policy import unified_profile_dir


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def _observed_metrics(summary_path: str) -> Dict[str, float]:
    if not summary_path:
        return {}
    try:
        obj = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    keys = [
        "f1",
        "em",
        "supporting_fact_recall",
        "supporting_fact_precision",
        "supporting_fact_f1",
        "prompt_tokens_avg",
        "avg_context_tokens",
        "retrieval_latency_ms",
        "total_latency_ms",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        if key not in obj:
            continue
        try:
            out[key] = float(obj.get(key, 0.0) or 0.0)
        except Exception:
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict unified copy-span config pack.")
    parser.add_argument("--profile", default="unified_medium")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique", "popqa"])
    parser.add_argument("--sample-size", "--limit", dest="limit", type=int, default=0)
    parser.add_argument("--profile-root", default=str(DEFAULT_PROFILE_ROOT))
    parser.add_argument(
        "--output-root",
        default="",
        help="Output root. Default: outputs/main_config_runs/copy_span_instruction_unified/<timestamp>_<profile>",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    profile_dir_name = unified_profile_dir(args.profile)
    profile_root = Path(args.profile_root).resolve()
    profile_dir = profile_root / profile_dir_name
    if not profile_dir.exists():
        raise FileNotFoundError(
            f"unified profile config directory not found: {profile_dir}. "
            "Run scripts/update_unified_config_copy_span_instruction_4ds.py first."
        )

    requested = set(_tokens(args.datasets))
    datasets = [dataset for dataset in DATASET_ORDER if dataset in requested]
    if not datasets:
        raise ValueError(f"No valid dataset requested. got={sorted(requested)}")

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        output_root = (
            REPO_ROOT
            / "outputs"
            / "main_config_runs"
            / "copy_span_instruction_unified"
            / f"{_ts()}_{profile_dir_name}"
        ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_records: List[Dict] = []
    for dataset in datasets:
        cfg_path = profile_dir / f"{dataset}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file missing for {dataset}: {cfg_path}")

        ds_out = output_root / dataset
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

        print(f"\n=== [{dataset}] running strict unified profile={profile_dir_name}")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        summary_path = _find_latest_summary(ds_out, dataset)
        observed = _observed_metrics(summary_path)
        if observed:
            print(
                f"[{dataset}] F1={observed.get('f1', 0.0):.6f} "
                f"EM={observed.get('em', 0.0):.6f} "
                f"SF_recall={observed.get('supporting_fact_recall', 0.0):.6f}"
            )

        run_records.append(
            {
                "dataset": dataset,
                "profile": profile_dir_name,
                "config_path": str(cfg_path),
                "exit_code": int(proc.returncode),
                "summary_path": summary_path,
                "observed_metrics": observed,
            }
        )

    run_manifest = {
        "profile_family": "copy_span_instruction_unified",
        "profile": profile_dir_name,
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
        raise SystemExit(f"Unified run failed for: {labels}")


if __name__ == "__main__":
    main()
