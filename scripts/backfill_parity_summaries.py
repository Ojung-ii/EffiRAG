#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _experiment_family(summary_path: Path) -> str:
    s = str(summary_path)
    parts = summary_path.parts
    if "/outputs/rag_experiments/" in s:
        i = parts.index("rag_experiments")
        family = parts[i + 1] if i + 1 < len(parts) else "unknown"
        return f"rag_experiments/{family}"
    if "/outputs/rag/" in s:
        return "rag"
    if "/outputs/final_eval/" in s:
        return "final_eval"
    if "/outputs/rag_quality_track_probe/" in s:
        return "rag_quality_track_probe"
    if "/outputs/rag_quality_track/" in s:
        return "rag_quality_track"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill parity summary sidecars from recomputed parity report.")
    parser.add_argument(
        "--recomputed-json",
        type=str,
        default="outputs/profiling/eval_parity/recomputed_all_summary.json",
    )
    parser.add_argument(
        "--sidecar-name",
        type=str,
        default="rag_summary_hipporag2_parity.json",
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="outputs/profiling/eval_parity/parity_backfill_manifest.json",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default="outputs/profiling/eval_parity/parity_backfill_delta_report.md",
    )
    args = parser.parse_args()

    recomputed_path = Path(args.recomputed_json).resolve()
    payload = json.loads(recomputed_path.read_text(encoding="utf-8"))
    rows: List[dict] = payload.get("rows", [])

    updated_rows = []
    for row in rows:
        summary_path = Path(str(row.get("summary_path", ""))).resolve()
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        legacy_em = _safe_float(row.get("legacy_EM", summary.get("em", 0.0)), 0.0)
        legacy_f1 = _safe_float(row.get("legacy_F1", summary.get("f1", 0.0)), 0.0)
        parity_em = _safe_float(row.get("parity_EM", legacy_em), legacy_em)
        parity_f1 = _safe_float(row.get("parity_F1", legacy_f1), legacy_f1)

        parity_summary = dict(summary)
        parity_summary["evaluator_mode"] = "hipporag2_parity"
        parity_summary["legacy_em"] = legacy_em
        parity_summary["legacy_f1"] = legacy_f1
        parity_summary["parity_em"] = parity_em
        parity_summary["parity_f1"] = parity_f1
        parity_summary["delta_em_parity_minus_legacy"] = parity_em - legacy_em
        parity_summary["delta_f1_parity_minus_legacy"] = parity_f1 - legacy_f1
        # parity view: keep canonical fields aligned to parity values.
        parity_summary["em"] = parity_em
        parity_summary["f1"] = parity_f1

        sidecar_path = summary_path.with_name(args.sidecar_name)
        sidecar_path.write_text(json.dumps(parity_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        row_out = dict(row)
        row_out["parity_sidecar_path"] = str(sidecar_path)
        row_out["experiment_family"] = _experiment_family(summary_path)
        updated_rows.append(row_out)

    # Build aggregate deltas by family and dataset
    family_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n_runs": 0.0, "delta_em_sum": 0.0, "delta_f1_sum": 0.0})
    dataset_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n_runs": 0.0, "delta_em_sum": 0.0, "delta_f1_sum": 0.0})
    for r in updated_rows:
        fam = str(r.get("experiment_family", "other"))
        ds = str(r.get("dataset", ""))
        de = _safe_float(r.get("delta_EM", 0.0), 0.0)
        df = _safe_float(r.get("delta_F1", 0.0), 0.0)

        family_stats[fam]["n_runs"] += 1.0
        family_stats[fam]["delta_em_sum"] += de
        family_stats[fam]["delta_f1_sum"] += df

        dataset_stats[ds]["n_runs"] += 1.0
        dataset_stats[ds]["delta_em_sum"] += de
        dataset_stats[ds]["delta_f1_sum"] += df

    manifest = {
        "recomputed_json": str(recomputed_path),
        "n_runs_input": len(rows),
        "n_runs_backfilled": len(updated_rows),
        "sidecar_name": args.sidecar_name,
        "rows": updated_rows,
        "family_stats": {
            k: {
                "n_runs": int(v["n_runs"]),
                "avg_delta_em": (v["delta_em_sum"] / v["n_runs"]) if v["n_runs"] > 0 else 0.0,
                "avg_delta_f1": (v["delta_f1_sum"] / v["n_runs"]) if v["n_runs"] > 0 else 0.0,
            }
            for k, v in sorted(family_stats.items())
        },
        "dataset_stats": {
            k: {
                "n_runs": int(v["n_runs"]),
                "avg_delta_em": (v["delta_em_sum"] / v["n_runs"]) if v["n_runs"] > 0 else 0.0,
                "avg_delta_f1": (v["delta_f1_sum"] / v["n_runs"]) if v["n_runs"] > 0 else 0.0,
            }
            for k, v in sorted(dataset_stats.items())
        },
    }

    manifest_path = Path(args.manifest_json).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report
    lines = []
    lines.append("# Parity Backfill Delta Report")
    lines.append("")
    lines.append(f"- recomputed_json: `{recomputed_path}`")
    lines.append(f"- n_runs_input: {len(rows)}")
    lines.append(f"- n_runs_backfilled: {len(updated_rows)}")
    lines.append(f"- sidecar_name: `{args.sidecar_name}`")
    lines.append("")

    lines.append("## Family Delta")
    lines.append("")
    lines.append("| experiment_family | n_runs | avg_delta_EM | avg_delta_F1 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for fam, st in manifest["family_stats"].items():
        lines.append(
            f"| {fam} | {int(st['n_runs'])} | {float(st['avg_delta_em']):+.6f} | {float(st['avg_delta_f1']):+.6f} |"
        )
    lines.append("")

    lines.append("## Dataset Delta")
    lines.append("")
    lines.append("| dataset | n_runs | avg_delta_EM | avg_delta_F1 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for ds, st in manifest["dataset_stats"].items():
        lines.append(
            f"| {ds} | {int(st['n_runs'])} | {float(st['avg_delta_em']):+.6f} | {float(st['avg_delta_f1']):+.6f} |"
        )
    lines.append("")

    changed = [r for r in updated_rows if abs(_safe_float(r.get("delta_EM", 0.0))) > 1e-12 or abs(_safe_float(r.get("delta_F1", 0.0))) > 1e-12]
    lines.append("## Non-zero Delta Runs")
    lines.append("")
    if not changed:
        lines.append("- none")
    else:
        lines.append("| dataset | variant | delta_EM | delta_F1 | summary_path | parity_sidecar_path |")
        lines.append("| --- | --- | ---: | ---: | --- | --- |")
        for r in changed:
            lines.append(
                f"| {r.get('dataset','')} | {r.get('variant','')} | "
                f"{_safe_float(r.get('delta_EM',0.0)):+.6f} | {_safe_float(r.get('delta_F1',0.0)):+.6f} | "
                f"{r.get('summary_path','')} | {r.get('parity_sidecar_path','')} |"
            )

    report_path = Path(args.report_md).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
