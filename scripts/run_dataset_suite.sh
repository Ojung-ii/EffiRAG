#!/usr/bin/env bash
set -euo pipefail

datasets_csv="hotpotqa,musique,2wikimultihopqa,popqa"
data_root=""
output_root="outputs/dataset_suite_$(date -u +%Y%m%d_%H%M%S)"
split="validation"
limit=""
global_corpus_root=""
graph_cache_dir="outputs/index_cache"
force_rebuild_graph_index="false"
openie_mode="llm"
openie_model_name="Qwen/Qwen2.5-7B-Instruct"
openie_text_max_chars=2200
openie_max_new_tokens=256
openie_local_files_only="true"
openie_retry_attempts=3
openie_retry_backoff_sec=0.2
openie_error_sample_limit=20
embedding_enabled="false"
embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
embedding_weight=0.35
embedding_rerank_topn=80
embedding_batch_size=16
embedding_max_length=256

method="effirag"
generator="hf"
model_name="Qwen/Qwen2.5-7B-Instruct"
run_qa="true"
render_mode=""
max_corridors_in_context=2
max_main_sentences_per_corridor=2
max_support_per_corridor=1
max_total_sentences=12
alpha=1.0
beta=0.35
gamma_main=0.45
delta_support=0.20
eta_connector=0.20
zeta_query=0.12
xi_locality=0.08
lambda_redundancy=0.25
top_corridors=3
max_sentences=10
reserve_top_corridor="false"
order_strategy="score"

# p053 defaults from latest hotpotqa grid run.
max_anchors=6
samples_per_anchor=8
candidate_top_t=20
seed_k=4
pair_top_lp=3
corridor_top_bc=20
trim_on="true"
trim_rho=0.6
tau=4
max_context_sentences=10

ppr_alpha=0.15
edge_drop_prob=0.1
random_seed=42
num_workers=1
measure_gpu_peak="false"
measure_cpu_ram="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      datasets_csv="$2"
      shift 2
      ;;
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --split)
      split="$2"
      shift 2
      ;;
    --limit)
      limit="$2"
      shift 2
      ;;
    --global-corpus-root)
      global_corpus_root="$2"
      shift 2
      ;;
    --graph-cache-dir)
      graph_cache_dir="$2"
      shift 2
      ;;
    --force-rebuild-graph-index)
      force_rebuild_graph_index="$2"
      shift 2
      ;;
    --openie-mode)
      openie_mode="$2"
      shift 2
      ;;
    --openie-model-name)
      openie_model_name="$2"
      shift 2
      ;;
    --openie-text-max-chars)
      openie_text_max_chars="$2"
      shift 2
      ;;
    --openie-max-new-tokens)
      openie_max_new_tokens="$2"
      shift 2
      ;;
    --openie-local-files-only)
      openie_local_files_only="$2"
      shift 2
      ;;
    --openie-retry-attempts)
      openie_retry_attempts="$2"
      shift 2
      ;;
    --openie-retry-backoff-sec)
      openie_retry_backoff_sec="$2"
      shift 2
      ;;
    --openie-error-sample-limit)
      openie_error_sample_limit="$2"
      shift 2
      ;;
    --embedding-enabled)
      embedding_enabled="$2"
      shift 2
      ;;
    --embedding-model-name)
      embedding_model_name="$2"
      shift 2
      ;;
    --embedding-weight)
      embedding_weight="$2"
      shift 2
      ;;
    --embedding-rerank-topn)
      embedding_rerank_topn="$2"
      shift 2
      ;;
    --embedding-batch-size)
      embedding_batch_size="$2"
      shift 2
      ;;
    --embedding-max-length)
      embedding_max_length="$2"
      shift 2
      ;;
    --method)
      method="$2"
      shift 2
      ;;
    --generator)
      generator="$2"
      shift 2
      ;;
    --model-name)
      model_name="$2"
      shift 2
      ;;
    --run-qa)
      run_qa="$2"
      shift 2
      ;;
    --render-mode)
      render_mode="$2"
      shift 2
      ;;
    --max-corridors-in-context)
      max_corridors_in_context="$2"
      shift 2
      ;;
    --max-main-sentences-per-corridor)
      max_main_sentences_per_corridor="$2"
      shift 2
      ;;
    --max-support-per-corridor)
      max_support_per_corridor="$2"
      shift 2
      ;;
    --max-total-sentences)
      max_total_sentences="$2"
      shift 2
      ;;
    --alpha)
      alpha="$2"
      shift 2
      ;;
    --beta)
      beta="$2"
      shift 2
      ;;
    --gamma-main)
      gamma_main="$2"
      shift 2
      ;;
    --delta-support)
      delta_support="$2"
      shift 2
      ;;
    --eta-connector)
      eta_connector="$2"
      shift 2
      ;;
    --zeta-query)
      zeta_query="$2"
      shift 2
      ;;
    --xi-locality)
      xi_locality="$2"
      shift 2
      ;;
    --lambda-redundancy)
      lambda_redundancy="$2"
      shift 2
      ;;
    --top-corridors)
      top_corridors="$2"
      shift 2
      ;;
    --max-sentences)
      max_sentences="$2"
      shift 2
      ;;
    --reserve-top-corridor)
      reserve_top_corridor="$2"
      shift 2
      ;;
    --order-strategy)
      order_strategy="$2"
      shift 2
      ;;
    --max-anchors)
      max_anchors="$2"
      shift 2
      ;;
    --samples-per-anchor)
      samples_per_anchor="$2"
      shift 2
      ;;
    --candidate-top-t)
      candidate_top_t="$2"
      shift 2
      ;;
    --seed-k)
      seed_k="$2"
      shift 2
      ;;
    --pair-top-lp)
      pair_top_lp="$2"
      shift 2
      ;;
    --corridor-top-bc)
      corridor_top_bc="$2"
      shift 2
      ;;
    --trim-on)
      trim_on="$2"
      shift 2
      ;;
    --trim-rho)
      trim_rho="$2"
      shift 2
      ;;
    --tau)
      tau="$2"
      shift 2
      ;;
    --max-context-sentences)
      max_context_sentences="$2"
      shift 2
      ;;
    --num-workers)
      num_workers="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: scripts/run_dataset_suite.sh [options]

Options:
  --datasets <csv>                Dataset list (default: hotpotqa,musique,2wikimultihopqa,popqa)
  --data-root <dir>               Directory containing <dataset>.json or <dataset>.jsonl
  --output-root <dir>             Root output dir (default: outputs/dataset_suite_<timestamp>)
  --split <name>                  Split name (default: validation)
  --limit <int>                   Optional sample cap (omit for full split)
  --global-corpus-root <dir>      Directory containing <dataset>_corpus.json for global KG mode
  --graph-cache-dir <dir>         Graph cache dir (default: outputs/index_cache)
  --force-rebuild-graph-index <bool>  Force rebuild global KG index (default: false)
  --openie-mode <mode>            llm|lexical (default: llm)
  --openie-model-name <hf-model>  OpenIE model (default: Qwen/Qwen2.5-7B-Instruct)
  --openie-text-max-chars <int>   OpenIE input clip length per sentence (default: 2200)
  --openie-max-new-tokens <int>   OpenIE generation cap (default: 256)
  --openie-local-files-only <bool> Prefer local HF cache only (default: true)
  --openie-retry-attempts <int>   OpenIE runtime retry attempts per sentence (default: 3)
  --openie-retry-backoff-sec <f>  OpenIE retry backoff seconds (default: 0.2)
  --openie-error-sample-limit <n> Store up to n OpenIE error samples in stats (default: 20)
  --embedding-enabled <bool>      Enable embedding rerank in retrieval (default: false)
  --embedding-model-name <hf-model> Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
  --embedding-weight <float>      Fusion weight for embedding similarity (default: 0.35)
  --embedding-rerank-topn <int>   Rerank first N retrieved sentences (default: 80)
  --embedding-batch-size <int>    Embedding encode batch size (default: 16)
  --embedding-max-length <int>    Embedding tokenizer max length (default: 256)
  --model-name <hf-model>         HF model (default: Qwen/Qwen2.5-7B-Instruct)
  --generator <name>              Generator: heuristic|oracle|hf (default: hf)
  --method <name>                 Retrieval method: effirag|naive_graphrag (default: effirag)
  --run-qa <bool>                 Whether to run QA (default: true)
  --render-mode <mode>            flat|corridor|corridor_aware_flat (default: method-aware auto)
  --max-corridors-in-context <n>  Corridor budget (default: 2)
  --max-main-sentences-per-corridor <n>  Main path budget (default: 2)
  --max-support-per-corridor <n>  Support budget (default: 1)
  --max-total-sentences <n>       Total sentence budget (default: 12)
  --alpha <float>                 corridor-aware-flat base retrieval weight
  --beta <float>                  corridor-aware-flat corridor score weight
  --gamma-main <float>            corridor-aware-flat main candidate bonus
  --delta-support <float>         corridor-aware-flat support candidate bonus
  --eta-connector <float>         corridor-aware-flat connector-adjacent bonus
  --zeta-query <float>            corridor-aware-flat query overlap weight
  --xi-locality <float>           corridor-aware-flat locality weight
  --lambda-redundancy <float>     corridor-aware-flat redundancy penalty
  --top-corridors <int>           corridor-aware-flat candidate corridor cutoff
  --max-sentences <int>           corridor-aware-flat max kept sentences
  --reserve-top-corridor <bool>   corridor-aware-flat keep >=1 from top corridor
  --order-strategy <mode>         score|retrieval|corridor_rank

Hyperparameters can also be overridden with:
  --max-anchors --samples-per-anchor --candidate-top-t --seed-k --pair-top-lp
  --corridor-top-bc --trim-on --trim-rho --tau --max-context-sentences --num-workers
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Try: scripts/run_dataset_suite.sh --help" >&2
      exit 1
      ;;
  esac
done

IFS=',' read -r -a datasets <<< "${datasets_csv}"
mkdir -p "${output_root}"

echo "[suite] datasets: ${datasets[*]}"
echo "[suite] output root: ${output_root}"

for dataset in "${datasets[@]}"; do
  data_path=""
  global_corpus_path=""
  if [[ -n "${data_root}" ]]; then
    if [[ -f "${data_root}/${dataset}.json" ]]; then
      data_path="${data_root}/${dataset}.json"
    elif [[ -f "${data_root}/${dataset}.jsonl" ]]; then
      data_path="${data_root}/${dataset}.jsonl"
    fi
  fi
  if [[ -n "${global_corpus_root}" ]]; then
    if [[ -f "${global_corpus_root}/${dataset}_corpus.json" ]]; then
      global_corpus_path="${global_corpus_root}/${dataset}_corpus.json"
    elif [[ -f "${global_corpus_root}/${dataset}_corpus.jsonl" ]]; then
      global_corpus_path="${global_corpus_root}/${dataset}_corpus.jsonl"
    fi
  fi

  out_dir="${output_root}/${dataset}"
  cmd=(
    python3 -m effirag.run_rag
    --dataset "${dataset}"
    --split "${split}"
    --method "${method}"
    --output-dir "${out_dir}"
    --global-corpus-path "${global_corpus_path}"
    --graph-cache-dir "${graph_cache_dir}"
    --force-rebuild-graph-index "${force_rebuild_graph_index}"
    --openie-mode "${openie_mode}"
    --openie-model-name "${openie_model_name}"
    --openie-text-max-chars "${openie_text_max_chars}"
    --openie-max-new-tokens "${openie_max_new_tokens}"
    --openie-local-files-only "${openie_local_files_only}"
    --openie-retry-attempts "${openie_retry_attempts}"
    --openie-retry-backoff-sec "${openie_retry_backoff_sec}"
    --openie-error-sample-limit "${openie_error_sample_limit}"
    --embedding-enabled "${embedding_enabled}"
    --embedding-model-name "${embedding_model_name}"
    --embedding-weight "${embedding_weight}"
    --embedding-rerank-topn "${embedding_rerank_topn}"
    --embedding-batch-size "${embedding_batch_size}"
    --embedding-max-length "${embedding_max_length}"
    --max-anchors "${max_anchors}"
    --samples-per-anchor "${samples_per_anchor}"
    --num-workers "${num_workers}"
    --candidate-top-t "${candidate_top_t}"
    --seed-k "${seed_k}"
    --pair-top-lp "${pair_top_lp}"
    --corridor-top-bc "${corridor_top_bc}"
    --trim-on "${trim_on}"
    --trim-rho "${trim_rho}"
    --ppr-alpha "${ppr_alpha}"
    --tau "${tau}"
    --edge-drop-prob "${edge_drop_prob}"
    --random-seed "${random_seed}"
    --run-qa "${run_qa}"
    --generator "${generator}"
    --model-name "${model_name}"
    --max-context-sentences "${max_context_sentences}"
    --max-corridors-in-context "${max_corridors_in_context}"
    --max-main-sentences-per-corridor "${max_main_sentences_per_corridor}"
    --max-support-per-corridor "${max_support_per_corridor}"
    --max-total-sentences "${max_total_sentences}"
    --alpha "${alpha}"
    --beta "${beta}"
    --gamma-main "${gamma_main}"
    --delta-support "${delta_support}"
    --eta-connector "${eta_connector}"
    --zeta-query "${zeta_query}"
    --xi-locality "${xi_locality}"
    --lambda-redundancy "${lambda_redundancy}"
    --top-corridors "${top_corridors}"
    --max-sentences "${max_sentences}"
    --reserve-top-corridor "${reserve_top_corridor}"
    --order-strategy "${order_strategy}"
    --measure-gpu-peak "${measure_gpu_peak}"
    --measure-cpu-ram "${measure_cpu_ram}"
  )
  if [[ -n "${render_mode}" ]]; then
    cmd+=(--render-mode "${render_mode}")
  fi

  if [[ -n "${data_path}" ]]; then
    cmd+=(--data-path "${data_path}")
  fi
  if [[ -n "${limit}" ]]; then
    cmd+=(--limit "${limit}")
  fi

  echo "[suite] running dataset=${dataset} data_path=${data_path:-<auto>} global_corpus_path=${global_corpus_path:-<none>}"
  "${cmd[@]}"
done

echo "[suite] done"
