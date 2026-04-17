#!/bin/bash

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DATASETS=(
  "2wikimultihopqa:data/2wikimultihopqa_corpus.json"
  "hotpotqa:data/hotpotqa_corpus.json"
  "musique:data/musique_corpus.json"
  "popqa:data/popqa_corpus.json"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EffiRAG 인덱싱 - 전체 데이터셋${NC}"
echo -e "${BLUE}vLLM 서버: http://localhost:8011/v1${NC}"
echo -e "${BLUE}========================================${NC}"

for dataset_info in "${DATASETS[@]}"; do
  IFS=':' read -r dataset_name corpus_path <<< "$dataset_info"
  
  echo ""
  echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 시작: ${dataset_name}${NC}"
  echo "  corpus_path: $corpus_path"
  
  CUDA_VISIBLE_DEVICES=0 python3 -m effirag.run_index \
    --dataset "$dataset_name" \
    --corpus-path "$corpus_path" \
    --embedding-batch-size 16 \
    --embedding-max-length 192 \
    --embedding-text-max-chars 600 \
    --semantic-scan-batch-size 8192 \
    --openie-mode llm \
    --openie-model-name Qwen/Qwen2.5-7B-Instruct \
    --openie-text-max-chars 2200 \
    --openie-max-new-tokens 256 \
    --openie-api-base-url http://localhost:8011/v1 \
    --openie-parallel-workers 6
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] 완료: ${dataset_name}${NC}"
  else
    echo -e "\033[0;31m[$(date '+%Y-%m-%d %H:%M:%S')] 실패: ${dataset_name}${NC}"
    exit 1
  fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}모든 데이터셋 인덱싱 완료!${NC}"
echo -e "${BLUE}========================================${NC}"
