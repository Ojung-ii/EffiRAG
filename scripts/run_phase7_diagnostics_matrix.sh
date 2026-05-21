#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_diag}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}"

DATASETS=("hotpotqa" "2wikimultihopqa")
VARIANTS=("full" "phase1_only" "no_phase2_refinement" "phase2_budget_x2" "legacy_compatible_render")

for ds in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    out_dir="${OUT_ROOT}/${ds}/${variant}"
    mkdir -p "${out_dir}"
    echo "[phase7-diag] dataset=${ds} variant=${variant} limit=${LIMIT} out=${out_dir}"
    PYTHONPATH=. "${PYTHON_BIN}" scripts/run_phase7_evidence_flow.py \
      --dataset "${ds}" \
      --limit "${LIMIT}" \
      --variant "${variant}" \
      --diagnostics \
      --diagnostics-max-examples "${LIMIT}" \
      --output-root "${out_dir}"
  done
done

# 2Wiki full failure-case dump at dataset root
PYTHONPATH=. "${PYTHON_BIN}" scripts/analyze_phase7_diagnostics.py \
  --diag-jsonl "${OUT_ROOT}/2wikimultihopqa/full/phase7_diagnostics.jsonl" \
  --out "${OUT_ROOT}/2wikimultihopqa/full/diagnostic_summary.json" \
  --compare-root "${OUT_ROOT}" \
  --compare-csv "${OUT_ROOT}/variant_comparison.csv" \
  --failure-md "${OUT_ROOT}/2wikimultihopqa/failure_cases.md" \
  --failure-top-k 20

PYTHONPATH=. "${PYTHON_BIN}" scripts/build_phase7_diagnostic_report.py \
  --root "${OUT_ROOT}"

echo "[phase7-diag] done: ${OUT_ROOT}"

