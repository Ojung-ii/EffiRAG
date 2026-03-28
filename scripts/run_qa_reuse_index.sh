#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

DATASETS=(
  "2wikimultihopqa:/home/ojungii/EffiRAG/data/qa/2wikimultihopqa.json:/home/ojungii/EffiRAG/data/2wikimultihopqa_corpus.json"
  # "hotpotqa:/home/ojungii/EffiRAG/data/qa/hotpotqa.json:/home/ojungii/EffiRAG/data/hotpotqa_corpus.json"
  # "musique:/home/ojungii/EffiRAG/data/qa/musique.json:/home/ojungii/EffiRAG/data/musique_corpus.json"
  # "popqa:/home/ojungii/EffiRAG/data/qa/popqa.json:/home/ojungii/EffiRAG/data/popqa_corpus.json"
)

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
PROFILE_CONFIG="${PROFILE_CONFIG:-configs/rag_speed_profile.yaml}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
STRICT_REUSE="${STRICT_REUSE:-true}"
LIMIT="${LIMIT:-}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EffiRAG QA 실행 - 기존 인덱스/임베딩 강제 재사용${NC}"
echo -e "${BLUE}profile=${PROFILE_CONFIG}${NC}"
echo -e "${BLUE}graph_cache_dir=${GRAPH_CACHE_DIR}${NC}"
echo -e "${BLUE}========================================${NC}"

for dataset_info in "${DATASETS[@]}"; do
  IFS=':' read -r dataset_name qa_path corpus_path <<< "$dataset_info"

  echo ""
  echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 시작: ${dataset_name}${NC}"
  echo "  qa_path: $qa_path"
  echo "  corpus_path: $corpus_path"

  corpus_stem="$(basename "${corpus_path}" .json)"
  latest_meta=""
  latest_meta_mtime=0
  shopt -s nullglob
  for search_root in "${GRAPH_CACHE_DIR}" "${GRAPH_CACHE_DIR}/${dataset_name}"; do
    for cand_meta in "${search_root}/${corpus_stem}_llm_"*/meta.json; do
      [ -f "${cand_meta}" ] || continue
      cand_mtime="$(stat -c %Y "${cand_meta}" 2>/dev/null || echo 0)"
      if [ "${cand_mtime}" -ge "${latest_meta_mtime}" ]; then
        latest_meta_mtime="${cand_mtime}"
        latest_meta="${cand_meta}"
      fi
    done
  done
  shopt -u nullglob

  if [ -z "${latest_meta}" ]; then
    echo -e "${RED}[오류] ${dataset_name}: 재사용 가능한 meta.json이 없습니다.${NC}"
    echo -e "${YELLOW}먼저 인덱싱을 1회 완료하세요: scripts/index_all_datasets.sh${NC}"
    exit 1
  fi

  index_dir="$(dirname "${latest_meta}")"
  if [ ! -f "${index_dir}/graph.gpickle" ]; then
    echo -e "${RED}[오류] ${dataset_name}: ${index_dir} 에 graph.gpickle이 없어 재사용 불가합니다.${NC}"
    echo -e "${YELLOW}인덱싱이 중간에 끊긴 상태입니다. 해당 데이터셋만 run_index를 다시 완료하세요.${NC}"
    exit 1
  fi

  # 기존 semantic 아티팩트가 있어야 재임베딩 없이 재사용됩니다.
  for f in semantic_entities_ids.json semantic_entities_embeddings.f16.npy semantic_chunks_ids.json semantic_chunks_embeddings.f16.npy; do
    if [ ! -f "${index_dir}/${f}" ]; then
      echo -e "${RED}[오류] ${dataset_name}: ${index_dir}/${f} 없음 (semantic 캐시 미완성).${NC}"
      echo -e "${YELLOW}먼저 동일 코퍼스로 run_index를 끝까지 1회 완료하세요.${NC}"
      exit 1
    fi
  done

  mapfile -t META_FIELDS < <(${PYTHON_BIN} - <<PY
import json
from pathlib import Path
meta_path = Path(r'''${latest_meta}''')
corpus_path = Path(r'''${corpus_path}''').resolve()
meta = json.loads(meta_path.read_text(encoding='utf-8'))
bc = dict(meta.get('build_config', {}) or {})
fp = dict(meta.get('fingerprint', {}) or {})
st = corpus_path.stat()
fp_match = (
    str(fp.get('path', '')) == str(corpus_path)
    and int(fp.get('size', -1)) == int(st.st_size)
    and int(fp.get('mtime_ns', -1)) == int(st.st_mtime_ns)
)
vals = [
    str(bc.get('embedding_model_name', 'nvidia/NV-Embed-v2')),
    str(int(bc.get('embedding_batch_size', 16))),
    str(int(bc.get('embedding_max_length', 192))),
    str(int(bc.get('embedding_text_max_chars', 600))),
    str(bc.get('openie_mode', 'llm')),
    str(bc.get('openie_model_name', 'Qwen/Qwen2.5-7B-Instruct')),
    str(int(bc.get('openie_text_max_chars', 2200))),
    str(int(bc.get('openie_max_new_tokens', 256))),
    'true' if bool(bc.get('openie_local_files_only', True)) else 'false',
    str(bc.get('openie_api_base_url', '')),
    str(float(bc.get('openie_api_timeout_sec', 120.0))),
    str(int(bc.get('openie_retry_attempts', 3))),
    str(float(bc.get('openie_retry_backoff_sec', 0.2))),
    str(int(bc.get('openie_error_sample_limit', 20))),
    'true' if fp_match else 'false',
]
for v in vals:
    print(v)
PY
)

  EMB_MODEL="${META_FIELDS[0]:-nvidia/NV-Embed-v2}"
  EMB_BATCH="${META_FIELDS[1]:-16}"
  EMB_MAXLEN="${META_FIELDS[2]:-192}"
  EMB_MAXCHARS="${META_FIELDS[3]:-600}"
  OPENIE_MODE="${META_FIELDS[4]:-llm}"
  OPENIE_MODEL="${META_FIELDS[5]:-Qwen/Qwen2.5-7B-Instruct}"
  OPENIE_TEXT="${META_FIELDS[6]:-2200}"
  OPENIE_NEWTOK="${META_FIELDS[7]:-256}"
  OPENIE_LOCAL="${META_FIELDS[8]:-true}"
  OPENIE_BASE="${META_FIELDS[9]:-}"
  OPENIE_TIMEOUT="${META_FIELDS[10]:-120.0}"
  OPENIE_RETRY="${META_FIELDS[11]:-3}"
  OPENIE_BACKOFF="${META_FIELDS[12]:-0.2}"
  OPENIE_ERR_LIMIT="${META_FIELDS[13]:-20}"
  FP_MATCH="${META_FIELDS[14]:-false}"

  if [ "${FP_MATCH}" != "true" ]; then
    echo -e "${YELLOW}[경고] ${dataset_name}: corpus fingerprint가 캐시 메타와 다릅니다.${NC}"
    echo -e "${YELLOW}       (파일이 바뀌었으면 재인덱싱이 정상 동작입니다)${NC}"
    if [ "${STRICT_REUSE}" = "true" ]; then
      echo -e "${RED}STRICT_REUSE=true 이므로 종료합니다. (의도적 재빌드면 STRICT_REUSE=false로 실행)${NC}"
      exit 1
    fi
  fi

  echo "  reuse_index_dir: ${index_dir}"
  dataset_cache_dir="$(dirname "${index_dir}")"
  echo "  reuse_cache_dir: ${dataset_cache_dir}"
  echo "  embedding_model: ${EMB_MODEL}"
  echo "  embedding_batch/maxlen/maxchars: ${EMB_BATCH}/${EMB_MAXLEN}/${EMB_MAXCHARS}"
  LIMIT_ARGS=()
  if [ -n "${LIMIT}" ]; then
    LIMIT_ARGS+=(--limit "${LIMIT}")
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" ${PYTHON_BIN} -m effirag.run_rag \
    --config "${PROFILE_CONFIG}" \
    --dataset "${dataset_name}" \
    --data-path "${qa_path}" \
    --global-corpus-path "${corpus_path}" \
    --graph-cache-dir "${dataset_cache_dir}" \
    --force-rebuild-graph-index false \
    --run-qa true \
    --embedding-enabled true \
    --embedding-model-name "${EMB_MODEL}" \
    --embedding-batch-size "${EMB_BATCH}" \
    --embedding-max-length "${EMB_MAXLEN}" \
    --embedding-text-max-chars "${EMB_MAXCHARS}" \
    --openie-mode "${OPENIE_MODE}" \
    --openie-model-name "${OPENIE_MODEL}" \
    --openie-text-max-chars "${OPENIE_TEXT}" \
    --openie-max-new-tokens "${OPENIE_NEWTOK}" \
    --openie-local-files-only "${OPENIE_LOCAL}" \
    --openie-api-base-url "${OPENIE_BASE}" \
    --openie-api-timeout-sec "${OPENIE_TIMEOUT}" \
    --openie-retry-attempts "${OPENIE_RETRY}" \
    --openie-retry-backoff-sec "${OPENIE_BACKOFF}" \
    --openie-error-sample-limit "${OPENIE_ERR_LIMIT}" \
    "${LIMIT_ARGS[@]}" \
    --render-mode corridor_aware_flat \
    --generator openai_compat \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --llm-base-url "${LLM_BASE_URL}" \
    --llm-api-key "${LLM_API_KEY}" \
    --llm-max-new-tokens 64

  if [ $? -eq 0 ]; then
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 완료: ${dataset_name}${NC}"
  else
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] 실패: ${dataset_name}${NC}"
    exit 1
  fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}모든 데이터셋 QA 실행 완료!${NC}"
echo -e "${BLUE}========================================${NC}"
