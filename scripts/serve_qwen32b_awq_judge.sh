#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8012}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-32b-awq-judge}"

vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --quantization awq \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --trust-remote-code
