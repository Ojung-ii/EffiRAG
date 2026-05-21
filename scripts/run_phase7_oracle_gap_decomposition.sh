#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ojungii/EffiRAG"
PYTHON="${PYTHON:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
LIMIT="${LIMIT:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/phase7_evidence_flow/oracle_gap_decomposition}"
DATASETS="${DATASETS:-hotpotqa,2wikimultihopqa}"
PROFILE="${PROFILE:-legacy_512_10}"
VARIANT="${VARIANT:-corridor_a_plus_bq_minus_r}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[phase7-oracle-gap] audit"
PYTHONPATH=. "${PYTHON}" scripts/audit_phase7_method.py \
  2>&1 | tee "${OUT_ROOT}/logs/audit_phase7_method.log"

echo "[phase7-oracle-gap] py_compile"
PYTHONPATH=. "${PYTHON}" -m py_compile effirag/*.py \
  2>&1 | tee "${OUT_ROOT}/logs/py_compile.log"

if [[ "${PROFILE}" == "balanced_384_8" ]]; then
  MAX_TOK=384
  MAX_ATOMS=8
else
  MAX_TOK=512
  MAX_ATOMS=10
fi

IFS=',' read -r -a DS_ARR <<< "${DATASETS}"

for ds in "${DS_ARR[@]}"; do
  ds="$(echo "${ds}" | xargs)"
  [[ -z "${ds}" ]] && continue
  RUN_OUT="${OUT_ROOT}/${PROFILE}/${VARIANT}/${ds}"
  mkdir -p "${RUN_OUT}"
  echo "[phase7-oracle-gap] run ${ds} -> ${RUN_OUT}"
  PYTHONPATH=. "${PYTHON}" -m effirag.run_rag \
    --config "configs/main_config/phase7_evidence_flow/${ds}.yaml" \
    --dataset "${ds}" \
    --limit "${LIMIT}" \
    --output-dir "${RUN_OUT}" \
    --timestamp-output false \
    --phase7-variant "full" \
    --phase7-enable-phase2-refinement false \
    --phase7-objective-mode "a_plus_bq_minus_r" \
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
    --phase7-max-context-tokens "${MAX_TOK}" \
    --phase7-max-selected-atoms "${MAX_ATOMS}" \
    --qa-prompt-mode phase7_short \
    --phase7-diagnostics-enabled true \
    --phase7-diagnostics-max-examples-to-dump "${LIMIT}" \
    --phase7-diagnostics-dump-text true \
    --phase7-diagnostics-dump-context true \
    --phase7-diagnostics-dump-scores true \
    --phase7-diagnostics-fail-on-unit-mismatch true \
    2>&1 | tee "${OUT_ROOT}/logs/run_${PROFILE}_${VARIANT}_${ds}.log"
done

echo "[phase7-oracle-gap] oracle context replay"
PYTHONPATH=. "${PYTHON}" scripts/run_phase7_oracle_context_qa.py \
  --root "${OUT_ROOT}" \
  --out "${OUT_ROOT}/phase7_oracle_context_report.md" \
  --limit "${LIMIT}" \
  --datasets "${DATASETS}" \
  --context-source-roots "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/logs/oracle_context_replay.log"

echo "[phase7-oracle-gap] analyze oracle gap"
PYTHONPATH=. "${PYTHON}" scripts/analyze_phase7_oracle_gap.py \
  --root "${OUT_ROOT}" \
  --out "${OUT_ROOT}/phase7_failure_attribution_report.md" \
  2>&1 | tee "${OUT_ROOT}/logs/analyze_oracle_gap.log"

echo "[phase7-oracle-gap] summarize stage timing"
PYTHONPATH=. "${PYTHON}" scripts/summarize_phase7_stage_timing.py \
  --root "${OUT_ROOT}" \
  --out "${OUT_ROOT}/phase7_stage_timing_summary.md" \
  2>&1 | tee "${OUT_ROOT}/logs/stage_timing_summary.log"

echo "[phase7-oracle-gap] done"
