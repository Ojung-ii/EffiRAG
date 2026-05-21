#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/chunk_ablation_n100}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/logs"

run_one() {
  local dataset="$1"
  local chunk_size="$2"
  local config="configs/main_config/phase7_evidence_flow/${dataset}.yaml"
  local out_dir="${OUT_ROOT}/${dataset}/carrier_chunk_${chunk_size}"
  mkdir -p "${out_dir}"
  echo "[phase7-chunk-ablation] dataset=${dataset} chunk_size=${chunk_size} limit=${LIMIT}"
  PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "${config}" \
    --limit "${LIMIT}" \
    --output-dir "${out_dir}" \
    --timestamp-output true \
    --phase7-carrier-chunk-size-sentences "${chunk_size}" \
    --phase7-carrier-chunk-stride-sentences 1 \
    2>&1 | tee "${OUT_ROOT}/logs/${dataset}_chunk${chunk_size}.log"
}

for dataset in hotpotqa 2wikimultihopqa; do
  for chunk_size in 1 2 3 4; do
    run_one "${dataset}" "${chunk_size}"
  done
done

PYTHONPATH=. "${PYTHON}" scripts/summarize_phase7_results.py \
  --output-root "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/summarize.log"

echo "[phase7-chunk-ablation] done: ${OUT_ROOT}"

