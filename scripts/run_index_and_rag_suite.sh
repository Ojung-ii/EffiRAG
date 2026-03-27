#!/usr/bin/env bash
set -euo pipefail

# Always execute from project root so `python -m effirag...` resolves to local source.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

# Keep tqdm readable by silencing tokenizers fork-parallelism warnings.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -x "/home/ojungii/miniconda3/envs/effirag/bin/python" ]]; then
  python_bin="/home/ojungii/miniconda3/envs/effirag/bin/python"
else
  python_bin="python3"
fi
datasets_csv="hotpotqa,musique,2wikimultihopqa,popqa"

# Input roots
# - data_root: <dataset>.json or <dataset>.jsonl
# - corpus_root: <dataset>_corpus.json or <dataset>_corpus.jsonl
data_root="data/qa"
corpus_root="/home/ojungii/HippoRAG2/dataset"

# Output/cache
index_cache_dir="outputs/index_cache"
output_root="outputs/index_and_rag_$(date -u +%Y%m%d_%H%M%S)"

# RAG run options
split="validation"
limit=""
method="effirag"
generator="heuristic"
run_qa="true"
model_name=""
llm_base_url="http://localhost:8011/v1"
llm_api_key="EMPTY"
llm_timeout_sec="120"
llm_max_new_tokens="64"

# Indexing/OpenIE options
force_rebuild_index="false"
openie_mode="llm"
openie_model_name="Qwen/Qwen2.5-7B-Instruct"
openie_text_max_chars="2200"
openie_max_new_tokens="256"
openie_local_files_only="true"
openie_retry_attempts="3"
openie_retry_backoff_sec="0.2"
openie_error_sample_limit="20"
openie_api_base_url=""
openie_api_key=""
openie_api_timeout_sec="120"
openie_parallel_workers="8"
openie_log_every="200"

# Embedding/Semantic options
embedding_enabled="true"
embedding_model_name="nvidia/NV-Embed-v2"
embedding_batch_size="8"
embedding_max_length="256"
embedding_text_max_chars="600"
embedding_rerank_topn="80"
semantic_topn="50"
semantic_candidate_union="true"
run_score_semantic_weight="0.35"
run_score_anchor_weight="0.20"
run_score_structure_weight="0.25"
run_score_bridge_weight="0.15"
run_score_redundancy_weight="0.05"
seed_score_semantic_weight="0.30"
seed_score_graph_weight="0.50"
seed_score_anchor_weight="0.20"
samples_per_anchor="8"
ppr_parallel_workers="1"
ppr_mc_walks="512"
ppr_mc_max_steps="24"
require_gpu_for_embedding="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin) python_bin="$2"; shift 2 ;;
    --datasets) datasets_csv="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --corpus-root) corpus_root="$2"; shift 2 ;;
    --index-cache-dir) index_cache_dir="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --split) split="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    --generator) generator="$2"; shift 2 ;;
    --run-qa) run_qa="$2"; shift 2 ;;
    --model-name) model_name="$2"; shift 2 ;;
    --llm-base-url) llm_base_url="$2"; shift 2 ;;
    --llm-api-key) llm_api_key="$2"; shift 2 ;;
    --llm-timeout-sec) llm_timeout_sec="$2"; shift 2 ;;
    --llm-max-new-tokens) llm_max_new_tokens="$2"; shift 2 ;;

    --force-rebuild-index) force_rebuild_index="$2"; shift 2 ;;
    --openie-mode) openie_mode="$2"; shift 2 ;;
    --openie-model-name) openie_model_name="$2"; shift 2 ;;
    --openie-text-max-chars) openie_text_max_chars="$2"; shift 2 ;;
    --openie-max-new-tokens) openie_max_new_tokens="$2"; shift 2 ;;
    --openie-local-files-only) openie_local_files_only="$2"; shift 2 ;;
    --openie-retry-attempts) openie_retry_attempts="$2"; shift 2 ;;
    --openie-retry-backoff-sec) openie_retry_backoff_sec="$2"; shift 2 ;;
    --openie-error-sample-limit) openie_error_sample_limit="$2"; shift 2 ;;
    --openie-api-base-url) openie_api_base_url="$2"; shift 2 ;;
    --openie-api-key) openie_api_key="$2"; shift 2 ;;
    --openie-api-timeout-sec) openie_api_timeout_sec="$2"; shift 2 ;;
    --openie-parallel-workers) openie_parallel_workers="$2"; shift 2 ;;
    --openie-log-every) openie_log_every="$2"; shift 2 ;;

    --embedding-enabled) embedding_enabled="$2"; shift 2 ;;
    --embedding-model-name) embedding_model_name="$2"; shift 2 ;;
    --embedding-batch-size) embedding_batch_size="$2"; shift 2 ;;
    --embedding-max-length) embedding_max_length="$2"; shift 2 ;;
    --embedding-text-max-chars) embedding_text_max_chars="$2"; shift 2 ;;
    --embedding-rerank-topn) embedding_rerank_topn="$2"; shift 2 ;;
    --semantic-topn) semantic_topn="$2"; shift 2 ;;
    --semantic-candidate-union) semantic_candidate_union="$2"; shift 2 ;;
    --run-score-semantic-weight) run_score_semantic_weight="$2"; shift 2 ;;
    --run-score-anchor-weight) run_score_anchor_weight="$2"; shift 2 ;;
    --run-score-structure-weight) run_score_structure_weight="$2"; shift 2 ;;
    --run-score-bridge-weight) run_score_bridge_weight="$2"; shift 2 ;;
    --run-score-redundancy-weight) run_score_redundancy_weight="$2"; shift 2 ;;
    --seed-score-semantic-weight) seed_score_semantic_weight="$2"; shift 2 ;;
    --seed-score-graph-weight) seed_score_graph_weight="$2"; shift 2 ;;
    --seed-score-anchor-weight) seed_score_anchor_weight="$2"; shift 2 ;;
    --samples-per-anchor) samples_per_anchor="$2"; shift 2 ;;
    --ppr-parallel-workers) ppr_parallel_workers="$2"; shift 2 ;;
    --ppr-mc-walks) ppr_mc_walks="$2"; shift 2 ;;
    --ppr-mc-max-steps) ppr_mc_max_steps="$2"; shift 2 ;;
    --require-gpu-for-embedding) require_gpu_for_embedding="$2"; shift 2 ;;

    --help|-h)
      cat <<'USAGE'
Usage: scripts/run_index_and_rag_suite.sh [options]

Defaults:
  --datasets hotpotqa,musique,2wikimultihopqa,popqa
  --data-root data/qa
  --corpus-root /home/ojungii/HippoRAG2/dataset
  --index-cache-dir outputs/index_cache
  --output-root outputs/index_and_rag_<UTC timestamp>

Common options:
  --limit <n>
  --generator heuristic|hf|openai_compat|vllm|oracle
  --model-name <name>
  --openie-retry-attempts <n>      (default: 3)
  --embedding-rerank-topn <n>      (default: 80)
  --samples-per-anchor <n>         (default: 8)
  --ppr-parallel-workers <n>       (default: 1)
  --ppr-mc-walks <n>               (default: 512)
  --ppr-mc-max-steps <n>           (default: 24)
  --force-rebuild-index true|false (default: false)
  --require-gpu-for-embedding true|false (default: true)
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Try: $0 --help" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${output_root}" "${index_cache_dir}"
IFS=',' read -r -a datasets <<< "${datasets_csv}"

if ! effirag_module_path="$("${python_bin}" -c 'import os, effirag; print(os.path.abspath(effirag.__file__))' 2>/dev/null)"; then
  echo "[suite][error] failed to import effirag with python_bin=${python_bin}" >&2
  echo "[suite][hint] run from project env and verify: ${python_bin} -m effirag.run_index --help" >&2
  exit 1
fi
if [[ "${effirag_module_path}" != "${project_root}/effirag/"* ]]; then
  echo "[suite][error] unexpected effirag module path: ${effirag_module_path}" >&2
  echo "[suite][hint] expected local path under ${project_root}/effirag" >&2
  echo "[suite][hint] set --python-bin to the environment where local EffiRAG is installed, or run from this repo." >&2
  exit 1
fi

if [[ "${embedding_enabled}" == "true" ]]; then
  if ! "${python_bin}" -c "import einops" >/dev/null 2>&1; then
    echo "[suite][error] embedding_enabled=true but '${python_bin}' env is missing dependency: einops" >&2
    echo "[suite][hint] install first: ${python_bin} -m pip install einops" >&2
    exit 1
  fi
  if [[ "${embedding_model_name,,}" == *"nv-embed"* ]]; then
    if ! "${python_bin}" - <<'PY' >/dev/null 2>&1
import sys
try:
    import transformers
except Exception:
    raise SystemExit(1)
major = int(str(transformers.__version__).split(".")[0])
raise SystemExit(0 if major < 5 else 2)
PY
    then
      tf_ver="$("${python_bin}" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'unknown')"
      echo "[suite][error] NV-Embed-v2 is incompatible with transformers>=5 in this pipeline (detected: ${tf_ver})" >&2
      echo "[suite][hint] use transformers 4.45.x (e.g. ${python_bin} -m pip install \"transformers>=4.45.2,<5\")" >&2
      exit 1
    fi
  fi
  if [[ "${require_gpu_for_embedding}" == "true" ]]; then
    if ! "${python_bin}" - <<'PY' >/dev/null 2>&1
import sys
import torch
sys.exit(0 if (torch.cuda.is_available() and int(torch.cuda.device_count()) > 0) else 1)
PY
    then
      probe="$("${python_bin}" - <<'PY'
import json
try:
    import torch
    print(json.dumps({
        "torch": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }))
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
PY
)"
      echo "[suite][error] embedding_enabled=true but CUDA is not available in python_bin=${python_bin}" >&2
      echo "[suite][probe] ${probe}" >&2
      echo "[suite][hint] run with a CUDA-enabled env, or pass --require-gpu-for-embedding false to allow CPU fallback." >&2
      exit 1
    fi
  fi
fi

echo "[suite] datasets: ${datasets[*]}"
echo "[suite] project_root: ${project_root}"
echo "[suite] python_bin: ${python_bin}"
echo "[suite] effirag_module: ${effirag_module_path}"
echo "[suite] data_root: ${data_root}"
echo "[suite] corpus_root: ${corpus_root}"
echo "[suite] index_cache_dir: ${index_cache_dir}"
echo "[suite] output_root: ${output_root}"

for dataset in "${datasets[@]}"; do
  corpus_path=""
  data_path=""

  if [[ -f "${corpus_root}/${dataset}_corpus.json" ]]; then
    corpus_path="${corpus_root}/${dataset}_corpus.json"
  elif [[ -f "${corpus_root}/${dataset}_corpus.jsonl" ]]; then
    corpus_path="${corpus_root}/${dataset}_corpus.jsonl"
  else
    echo "[suite][skip] dataset=${dataset} corpus not found at ${corpus_root}/${dataset}_corpus.(json|jsonl)" >&2
    continue
  fi

  if [[ -f "${data_root}/${dataset}.json" ]]; then
    data_path="${data_root}/${dataset}.json"
  elif [[ -f "${data_root}/${dataset}.jsonl" ]]; then
    data_path="${data_root}/${dataset}.jsonl"
  else
    echo "[suite][warn] dataset=${dataset} qa file not found in data_root. run_rag auto-discovery will be used."
  fi

  echo "[suite][index] dataset=${dataset} corpus=${corpus_path}"
  "${python_bin}" -m effirag.run_index \
    --corpus-path "${corpus_path}" \
    --cache-dir "${index_cache_dir}" \
    --force-rebuild "${force_rebuild_index}" \
    --embedding-enabled "${embedding_enabled}" \
    --embedding-model-name "${embedding_model_name}" \
    --embedding-batch-size "${embedding_batch_size}" \
    --embedding-max-length "${embedding_max_length}" \
    --embedding-text-max-chars "${embedding_text_max_chars}" \
    --openie-mode "${openie_mode}" \
    --openie-model-name "${openie_model_name}" \
    --openie-text-max-chars "${openie_text_max_chars}" \
    --openie-max-new-tokens "${openie_max_new_tokens}" \
    --openie-local-files-only "${openie_local_files_only}" \
    --openie-retry-attempts "${openie_retry_attempts}" \
    --openie-retry-backoff-sec "${openie_retry_backoff_sec}" \
    --openie-error-sample-limit "${openie_error_sample_limit}" \
    --openie-api-base-url "${openie_api_base_url}" \
    --openie-api-key "${openie_api_key}" \
    --openie-api-timeout-sec "${openie_api_timeout_sec}" \
    --openie-parallel-workers "${openie_parallel_workers}" \
    --openie-log-every "${openie_log_every}"

  out_dir="${output_root}/${dataset}"
  mkdir -p "${out_dir}"

  echo "[suite][rag] dataset=${dataset} out=${out_dir}"
  rag_cmd=(
    "${python_bin}" -m effirag.run_rag
    --dataset "${dataset}"
    --split "${split}"
    --method "${method}"
    --output-dir "${out_dir}"
    --timestamp-output false
    --global-corpus-path "${corpus_path}"
    --graph-cache-dir "${index_cache_dir}"
    --force-rebuild-graph-index false
    --embedding-enabled "${embedding_enabled}"
    --embedding-model-name "${embedding_model_name}"
    --openie-mode "${openie_mode}"
    --openie-model-name "${openie_model_name}"
    --openie-text-max-chars "${openie_text_max_chars}"
    --openie-max-new-tokens "${openie_max_new_tokens}"
    --openie-local-files-only "${openie_local_files_only}"
    --openie-retry-attempts "${openie_retry_attempts}"
    --openie-retry-backoff-sec "${openie_retry_backoff_sec}"
    --openie-error-sample-limit "${openie_error_sample_limit}"
    --openie-api-base-url "${openie_api_base_url}"
    --openie-api-key "${openie_api_key}"
    --openie-api-timeout-sec "${openie_api_timeout_sec}"
    --openie-parallel-workers "${openie_parallel_workers}"
    --openie-log-every "${openie_log_every}"
    --embedding-rerank-topn "${embedding_rerank_topn}"
    --embedding-batch-size "${embedding_batch_size}"
    --embedding-max-length "${embedding_max_length}"
    --embedding-text-max-chars "${embedding_text_max_chars}"
    --semantic-topn "${semantic_topn}"
    --semantic-candidate-union "${semantic_candidate_union}"
    --run-score-semantic-weight "${run_score_semantic_weight}"
    --run-score-anchor-weight "${run_score_anchor_weight}"
    --run-score-structure-weight "${run_score_structure_weight}"
    --run-score-bridge-weight "${run_score_bridge_weight}"
    --run-score-redundancy-weight "${run_score_redundancy_weight}"
    --seed-score-semantic-weight "${seed_score_semantic_weight}"
    --seed-score-graph-weight "${seed_score_graph_weight}"
    --seed-score-anchor-weight "${seed_score_anchor_weight}"
    --samples-per-anchor "${samples_per_anchor}"
    --ppr-parallel-workers "${ppr_parallel_workers}"
    --ppr-mc-walks "${ppr_mc_walks}"
    --ppr-mc-max-steps "${ppr_mc_max_steps}"
    --run-qa "${run_qa}"
    --generator "${generator}"
    --model-name "${model_name}"
    --llm-base-url "${llm_base_url}"
    --llm-api-key "${llm_api_key}"
    --llm-timeout-sec "${llm_timeout_sec}"
    --llm-max-new-tokens "${llm_max_new_tokens}"
  )

  if [[ -n "${data_path}" ]]; then
    rag_cmd+=(--data-path "${data_path}")
  fi
  if [[ -n "${limit}" ]]; then
    rag_cmd+=(--limit "${limit}")
  fi

  "${rag_cmd[@]}"
done

echo "[suite] all done"
