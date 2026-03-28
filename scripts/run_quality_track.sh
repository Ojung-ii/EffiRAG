#!/bin/bash
set -euo pipefail

MODE="${1:-quality_variant}"
LIMIT="${LIMIT:-100}"
DATASET="${DATASET:-2wikimultihopqa}"
DATA_PATH="${DATA_PATH:-data/qa/2wikimultihopqa.json}"
GLOBAL_CORPUS_PATH="${GLOBAL_CORPUS_PATH:-/home/ojungii/HippoRAG2/dataset/2wikimultihopqa_corpus.json}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rag_quality_track}"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-false}"

BASE_CONFIG="configs/rag_quality_profile.yaml"
EXTRA_ARGS=()

case "${MODE}" in
  speed_default)
    # Final operating speed preset: aggressive + sentence_rerank off.
    BASE_CONFIG="configs/rag_speed_profile.yaml"
    ;;
  quality_variant)
    # Final operating quality preset: top1corr_t1 on top of aggressive core.
    BASE_CONFIG="configs/rag_quality_profile.yaml"
    ;;
  ablation_off)
    # Final reference/ablation preset: proposal union off.
    BASE_CONFIG="configs/rag_quality_off_ablation.yaml"
    ;;
  baseline)
    # Legacy alias -> quality_variant.
    BASE_CONFIG="configs/rag_quality_profile.yaml"
    ;;
  baseline_aggressive)
    # Legacy alias -> speed_default.
    BASE_CONFIG="configs/rag_speed_profile.yaml"
    ;;
  off_ablation)
    # Legacy alias -> ablation_off.
    BASE_CONFIG="configs/rag_quality_off_ablation.yaml"
    ;;
  top1corr_t1)
    # Legacy alias -> quality_variant.
    BASE_CONFIG="configs/rag_quality_profile.yaml"
    ;;
  anchor_boost_a)
    # Experiment A: single-variable change (max_anchors 2 -> 3)
    EXTRA_ARGS+=(--max-anchors 3 --samples-per-anchor 1)
    ;;
  proposal_expand_b1)
    # Experiment B1 (conservative): +5 entity, +5 chunk, +5 reserve
    EXTRA_ARGS+=(--semantic-topn-entity 25 --semantic-topn-chunk 15 --graph-reserve-topn 15)
    ;;
  proposal_expand_b2)
    # Experiment B2 (medium): +10 entity, +5 chunk, +5 reserve
    EXTRA_ARGS+=(--semantic-topn-entity 30 --semantic-topn-chunk 15 --graph-reserve-topn 15)
    ;;
  reserve_boost_c1)
    # Experiment C1: reserve-only boost (20/10/15)
    EXTRA_ARGS+=(--semantic-topn-entity 20 --semantic-topn-chunk 10 --graph-reserve-topn 15)
    ;;
  chunk_reserve_boost_c2)
    # Experiment C2: chunk+reserve boost (20/15/15), run only if C1 passes
    EXTRA_ARGS+=(--semantic-topn-entity 20 --semantic-topn-chunk 15 --graph-reserve-topn 15)
    ;;
  top1corr_t2)
    # Legacy exploratory preset (balanced top-5 correction).
    BASE_CONFIG="configs/rag_quality_top1corr_t2.yaml"
    ;;
  top1corr_t3)
    # Legacy exploratory preset (aggressive top-1 bias).
    BASE_CONFIG="configs/rag_quality_top1corr_t3.yaml"
    ;;
  proposal_expand_b)
    # Backward-compatible alias for B2
    EXTRA_ARGS+=(--semantic-topn-entity 30 --semantic-topn-chunk 15 --graph-reserve-topn 15)
    ;;
  proposal_expand_b5)
    # Backward-compatible alias for B1
    EXTRA_ARGS+=(--semantic-topn-entity 25 --semantic-topn-chunk 15 --graph-reserve-topn 15)
    ;;
  proposal_expand_b10)
    # Backward-compatible alias for B2
    EXTRA_ARGS+=(--semantic-topn-entity 30 --semantic-topn-chunk 15 --graph-reserve-topn 15)
    ;;
  pruning_relax_c)
    # Experiment C: relax early pruning (after B passes)
    EXTRA_ARGS+=(--candidate-top-t 24 --seed-k 5 --pair-top-lp 4)
    ;;
  run_boost_d)
    # Experiment D: increase stochastic runs (last step)
    EXTRA_ARGS+=(--max-anchors 3 --samples-per-anchor 2)
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Supported final modes: speed_default | quality_variant | ablation_off" >&2
    echo "Legacy modes: baseline | baseline_aggressive | top1corr_t1 | top1corr_t2 | top1corr_t3 | reserve_boost_c1 | chunk_reserve_boost_c2 | anchor_boost_a | proposal_expand_b1 | proposal_expand_b2 | proposal_expand_b | proposal_expand_b5 | proposal_expand_b10 | pruning_relax_c | run_boost_d | off_ablation" >&2
    exit 1
    ;;
esac

PROFILE_OUTPUT="outputs/profiling/2wiki_quality_track_${MODE}_q${LIMIT}.jsonl"

echo "Running quality-track mode=${MODE} limit=${LIMIT}"
echo "Config: ${BASE_CONFIG}"
echo "Profile output: ${PROFILE_OUTPUT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m effirag.run_rag \
  --config "${BASE_CONFIG}" \
  --dataset "${DATASET}" \
  --data-path "${DATA_PATH}" \
  --global-corpus-path "${GLOBAL_CORPUS_PATH}" \
  --graph-cache-dir "${GRAPH_CACHE_DIR}" \
  --force-rebuild-graph-index false \
  --embedding-enabled true \
  --embedding-model-name nvidia/NV-Embed-v2 \
  --generator openai_compat \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --llm-base-url http://localhost:8011/v1 \
  --llm-api-key EMPTY \
  --limit "${LIMIT}" \
  --retrieval-only "${RETRIEVAL_ONLY}" \
  --output-dir "${OUTPUT_DIR}" \
  --profile-stages true \
  --profile-output "${PROFILE_OUTPUT}" \
  "${EXTRA_ARGS[@]}"
