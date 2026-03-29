#!/bin/bash
set -euo pipefail

# Ordered runner requested in spec:
# 1) Stage-1 F1 boost (context/render expansion)
# 2) Stage-2 semantic-aware selection ablation

LIMIT="${LIMIT:-200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"

# Stage 1 defaults: hotpotqa + 2wikimultihopqa, QA mode
STAGE1_DATASETS="${STAGE1_DATASETS:-hotpotqa,2wikimultihopqa}"

# Stage 2 defaults: hotpotqa + 2wikimultihopqa, QA mode
STAGE2_DATASETS="${STAGE2_DATASETS:-hotpotqa,2wikimultihopqa}"
STAGE2_RETRIEVAL_ONLY="${STAGE2_RETRIEVAL_ONLY:-false}"
INCLUDE_MUSIQUE_RETRIEVAL="${INCLUDE_MUSIQUE_RETRIEVAL:-true}"

echo "[stage12] Step 1/2: run_f1_boost_qa.sh"
LIMIT="${LIMIT}" \
DATASETS="${STAGE1_DATASETS}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_f1_boost_qa.sh

echo "[stage12] Step 2/2: run_semantic_selection_ablation.sh"
LIMIT="${LIMIT}" \
DATASETS="${STAGE2_DATASETS}" \
RETRIEVAL_ONLY="${STAGE2_RETRIEVAL_ONLY}" \
INCLUDE_MUSIQUE_RETRIEVAL="${INCLUDE_MUSIQUE_RETRIEVAL}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/run_semantic_selection_ablation.sh

echo "[stage12] complete"
