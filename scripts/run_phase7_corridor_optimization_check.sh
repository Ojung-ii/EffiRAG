#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
LIMIT="${LIMIT:-20}"

# Keep retrieval path deterministic/offline-friendly in this diagnostic check.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

OLD_ROOT="${OLD_ROOT:-outputs/phase7_evidence_flow/optimization/equivalence_old/legacy_512_10/corridor_a_plus_bq_minus_r}"
NEW_ROOT="${NEW_ROOT:-outputs/phase7_evidence_flow/optimization/equivalence_new/legacy_512_10/corridor_a_plus_bq_minus_r}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/optimization}"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${NEW_ROOT}/hotpotqa" "${NEW_ROOT}/2wikimultihopqa"
cd "${REPO_ROOT}"

echo "[phase7-opt-check] audit"
PYTHONPATH=. "${PYTHON}" scripts/audit_phase7_method.py \
  2>&1 | tee "${LOG_DIR}/audit_phase7_method.log"

echo "[phase7-opt-check] run NEW hotpotqa"
PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
  --config configs/main_config/phase7_evidence_flow/hotpotqa.yaml \
  --dataset hotpotqa \
  --limit "${LIMIT}" \
  --output-dir "${NEW_ROOT}/hotpotqa" \
  --timestamp-output false \
  --run-qa false \
  --phase7-enable-phase2-refinement false \
  --phase7-objective-mode a_plus_bq_minus_r \
  --phase7-lambda-bridge 1.0 \
  --phase7-lambda-decay 1.0 \
  --phase7-lambda-bq 1.0 \
  --phase7-mu-redundancy 1.0 \
  --phase7-conditional-redundancy-enabled false \
  --phase7-corridor-enabled true \
  --phase7-anchor-decay-enabled false \
  --phase7-corridor-max-anchors 4 \
  --phase7-corridor-max-seeds 16 \
  --phase7-corridor-max-hops 3 \
  --phase7-corridor-max-paths-per-pair 1 \
  --phase7-corridor-degree-cap 100 \
  --phase7-corridor-max-pairs 64 \
  --phase7-candidate-top-m 96 \
  --phase7-max-context-tokens 512 \
  --phase7-max-selected-atoms 10 \
  --phase7-diagnostics-enabled true \
  --phase7-diagnostics-max-examples-to-dump "${LIMIT}" \
  --phase7-diagnostics-dump-text true \
  --phase7-diagnostics-dump-context true \
  --phase7-diagnostics-dump-scores true \
  --phase7-diagnostics-fail-on-unit-mismatch true \
  >"${LOG_DIR}/equivalence_new_hotpotqa.log" 2>&1 &


echo "[phase7-opt-check] run NEW 2wikimultihopqa"
PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
  --config configs/main_config/phase7_evidence_flow/2wikimultihopqa.yaml \
  --dataset 2wikimultihopqa \
  --limit "${LIMIT}" \
  --output-dir "${NEW_ROOT}/2wikimultihopqa" \
  --timestamp-output false \
  --run-qa false \
  --phase7-enable-phase2-refinement false \
  --phase7-objective-mode a_plus_bq_minus_r \
  --phase7-lambda-bridge 1.0 \
  --phase7-lambda-decay 1.0 \
  --phase7-lambda-bq 1.0 \
  --phase7-mu-redundancy 1.0 \
  --phase7-conditional-redundancy-enabled false \
  --phase7-corridor-enabled true \
  --phase7-anchor-decay-enabled false \
  --phase7-corridor-max-anchors 4 \
  --phase7-corridor-max-seeds 16 \
  --phase7-corridor-max-hops 3 \
  --phase7-corridor-max-paths-per-pair 1 \
  --phase7-corridor-degree-cap 100 \
  --phase7-corridor-max-pairs 64 \
  --phase7-candidate-top-m 96 \
  --phase7-max-context-tokens 512 \
  --phase7-max-selected-atoms 10 \
  --phase7-diagnostics-enabled true \
  --phase7-diagnostics-max-examples-to-dump "${LIMIT}" \
  --phase7-diagnostics-dump-text true \
  --phase7-diagnostics-dump-context true \
  --phase7-diagnostics-dump-scores true \
  --phase7-diagnostics-fail-on-unit-mismatch true \
  >"${LOG_DIR}/equivalence_new_2wiki.log" 2>&1 &

wait

echo "[phase7-opt-check] equivalence compare"
PYTHONPATH=. "${PYTHON}" scripts/check_phase7_corridor_equivalence.py \
  --old-root "${OLD_ROOT}" \
  --new-root "${NEW_ROOT}" \
  --datasets "hotpotqa,2wikimultihopqa" \
  --out "${OUT_ROOT}/corridor_equivalence_report.md" \
  2>&1 | tee "${LOG_DIR}/corridor_equivalence_compare.log"

echo "[phase7-opt-check] timing summary"
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase7_stage_timing.py \
  --root "${NEW_ROOT}" \
  --out "${OUT_ROOT}/stage_timing_summary.md" \
  2>&1 | tee "${LOG_DIR}/stage_timing_summary.log"

echo "[phase7-opt-check] done"
