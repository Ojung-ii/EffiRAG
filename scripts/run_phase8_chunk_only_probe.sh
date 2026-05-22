#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ojungii/EffiRAG}"
OUT_ROOT="${OUT_ROOT:-outputs/phase8_chunk_medoid}"
LIMIT="${LIMIT:-10}"
HOTPOT_GPU="${HOTPOT_GPU:-0}"
TWIKI_GPU="${TWIKI_GPU:-1}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}/logs"

RUN_ID_BASE="${RUN_ID_BASE:-$(date -u +%Y%m%dT%H%M%S_%N)}"

echo "[phase8-chunk-probe] starting hotpotqa/legacy_512_10 and 2wikimultihopqa/balanced_384_8 limit=${LIMIT}"

CUDA_VISIBLE_DEVICES="${HOTPOT_GPU}" \
DATASET=hotpotqa \
PROFILE=legacy_512_10 \
LIMIT="${LIMIT}" \
OUT_ROOT="${OUT_ROOT}" \
RUN_ID="${RUN_ID_BASE}_hotpot_legacy" \
RETRIEVAL_ONLY="${RETRIEVAL_ONLY}" \
PHASE8_CHUNK_VARIANTS=chunk_pamae_k5 \
bash scripts/run_phase8_chunk_medoid_experiment.sh &
PID_HOTPOT=$!

CUDA_VISIBLE_DEVICES="${TWIKI_GPU}" \
DATASET=2wikimultihopqa \
PROFILE=balanced_384_8 \
LIMIT="${LIMIT}" \
OUT_ROOT="${OUT_ROOT}" \
RUN_ID="${RUN_ID_BASE}_2wiki_balanced" \
RETRIEVAL_ONLY="${RETRIEVAL_ONLY}" \
PHASE8_CHUNK_VARIANTS=chunk_pamae_k5 \
bash scripts/run_phase8_chunk_medoid_experiment.sh &
PID_TWIKI=$!

wait "${PID_HOTPOT}"
wait "${PID_TWIKI}"

python scripts/summarize_phase8_chunk_medoid.py \
  --root "${OUT_ROOT}" \
  --latest-only \
  --dedupe-query-id \
  --expected-limit "${LIMIT}"

python scripts/analyze_phase8_chunk_medoid_failures.py \
  --root "${OUT_ROOT}" \
  --latest-only \
  --dedupe-query-id \
  --expected-limit "${LIMIT}"

echo "[phase8-chunk-probe] complete"
