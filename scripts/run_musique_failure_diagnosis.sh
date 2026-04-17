#!/bin/bash
set -euo pipefail

# Next-method-design family: MuSiQue failure-mode diagnosis.
# Frozen operating configs/scripts are not mutated.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-qa}"  # qa | retrieval_only | both
LIMIT="${LIMIT:-200}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-outputs/index_cache}"
GENERATOR="${GENERATOR:-openai_compat}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8011/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

DATASET_NAME="${DATASET_NAME:-musique}"
DATA_PATH="${DATA_PATH:-data/qa/musique.json}"
CORPUS_PATH="${CORPUS_PATH:-data/musique_corpus.json}"

PROFILE_ROOT="${PROFILE_ROOT:-outputs/profiling/next_method_design/musique_failure_diagnosis}"
mkdir -p "${PROFILE_ROOT}"

find_latest_summary() {
  local out_dir="$1"
  local dataset="$2"
  find "${out_dir}/${dataset}" -type f -name rag_summary.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
}

GIT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
GIT_TAG="$(git -C "${REPO_ROOT}" tag --points-at HEAD | head -n 1 || true)"
if [ -z "${GIT_TAG}" ]; then
  GIT_TAG="$(git -C "${REPO_ROOT}" tag --list 'freeze-*' | sort | tail -n 1)"
fi

run_kinds=()
case "${MODE}" in
  qa)
    run_kinds=(qa)
    ;;
  retrieval|retrieval_only)
    run_kinds=(retrieval_only)
    ;;
  both)
    run_kinds=(qa retrieval_only)
    ;;
  *)
    echo "[ERROR] MODE must be qa|retrieval_only|both, got: ${MODE}" >&2
    exit 1
    ;;
esac

for run_kind in "${run_kinds[@]}"; do
  if [ "${run_kind}" = "qa" ]; then
    RETRIEVAL_ONLY="false"
    RUN_QA="true"
    CFG="configs/rag_nextdesign_musique_diagnosis_baseline.yaml"
    OUTPUT_ROOT="${OUTPUT_ROOT_QA:-outputs/rag_experiments/next_method_design/musique_failure_diagnosis}"
  else
    RETRIEVAL_ONLY="true"
    RUN_QA="false"
    CFG="configs/retrieval_nextdesign_musique_diagnosis_baseline.yaml"
    OUTPUT_ROOT="${OUTPUT_ROOT_RETR:-outputs/retrieval_experiments/next_method_design/musique_failure_diagnosis}"
  fi

  if [ ! -f "${CFG}" ]; then
    echo "[ERROR] Missing config: ${CFG}" >&2
    exit 1
  fi

  OUT_DIR="${OUTPUT_ROOT}/baseline"
  PROFILE_OUT="${PROFILE_ROOT}/nextdesign_musique_failure_${run_kind}_q${LIMIT}_${RUN_STAMP}.jsonl"

  mkdir -p "${OUTPUT_ROOT}" "${OUT_DIR}"

  echo "[nextdesign:musique_failure] run_kind=${run_kind} limit=${LIMIT} config=${CFG}"

  ARGS=(
    --config "${CFG}"
    --dataset "${DATASET_NAME}"
    --data-path "${DATA_PATH}"
    --global-corpus-path "${CORPUS_PATH}"
    --graph-cache-dir "${GRAPH_CACHE_DIR}"
    --force-rebuild-graph-index false
    --run-qa "${RUN_QA}"
    --retrieval-only "${RETRIEVAL_ONLY}"
    --embedding-enabled true
    --embedding-model-name nvidia/NV-Embed-v2
    --generator "${GENERATOR}"
    --model-name "${MODEL_NAME}"
    --limit "${LIMIT}"
    --output-dir "${OUT_DIR}"
    --profile-stages true
    --profile-output "${PROFILE_OUT}"
    --llm-max-new-tokens 64
  )

  if [ "${GENERATOR}" = "openai_compat" ]; then
    ARGS+=(
      --llm-base-url "${LLM_BASE_URL}"
      --llm-api-key "${LLM_API_KEY}"
    )
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m effirag.run_rag "${ARGS[@]}"

  SUMMARY_PATH="$(find_latest_summary "${OUT_DIR}" "${DATASET_NAME}")"
  if [ -z "${SUMMARY_PATH}" ]; then
    echo "[ERROR] Missing rag_summary.json for musique baseline (${run_kind})" >&2
    exit 1
  fi

  DIAG_JSON="${PROFILE_ROOT}/nextdesign_musique_failure_${run_kind}_summary_q${LIMIT}_${RUN_STAMP}.json"
  DIAG_MD="${PROFILE_ROOT}/nextdesign_musique_failure_${run_kind}_summary_q${LIMIT}_${RUN_STAMP}.md"

  "${PYTHON_BIN}" - <<'PY' "${SUMMARY_PATH}" "${PROFILE_OUT}" "${DIAG_JSON}" "${DIAG_MD}" "${run_kind}" "${GIT_BRANCH}" "${GIT_COMMIT}" "${GIT_TAG}"
import json
from pathlib import Path
import statistics
import sys

summary_path = Path(sys.argv[1])
profile_path = Path(sys.argv[2])
out_json = Path(sys.argv[3])
out_md = Path(sys.argv[4])
run_kind = sys.argv[5]
git_branch = sys.argv[6]
git_commit = sys.argv[7]
git_tag = sys.argv[8]

summary = json.loads(summary_path.read_text(encoding='utf-8'))
rows = []
if profile_path.exists():
    for line in profile_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)


def _i(v, d=0):
    try:
        return int(v)
    except Exception:
        return int(d)


def mean_field(key, default=0.0):
    if not rows:
        return float(default)
    vals = [_f(r.get(key, default), default) for r in rows]
    return float(sum(vals) / max(len(vals), 1))


def rate(pred):
    if not rows:
        return 0.0
    hit = 0
    for r in rows:
        if pred(r):
            hit += 1
    return float(hit) / float(len(rows))

proposal_coverage = rate(lambda r: _i(r.get('union_candidate_count', 0)) > 0 and _i(r.get('proposal_subgraph_nodes', 0)) > 0)
proposal_size = mean_field('union_candidate_count', 0.0)
proposal_subgraph_nodes = mean_field('proposal_subgraph_nodes', 0.0)
proposal_subgraph_edges = mean_field('proposal_subgraph_edges', 0.0)
phase1_run_diversity = mean_field('phase1_run_count', 0.0)
selected_run_rate = rate(lambda r: _i(r.get('selected_run_count', 0)) > 0)
pair_shortlist_coverage = rate(lambda r: _i(r.get('phase2_refined_pair_count', 0)) > 0)

proposal_stage_drop_rate = rate(lambda r: _i(r.get('union_candidate_count', 0)) <= 0 or _i(r.get('proposal_subgraph_nodes', 0)) <= 0)
run_selection_stage_drop_rate = rate(
    lambda r: (_i(r.get('phase1_run_count', 0)) > 0) and (_i(r.get('selected_run_count', 0)) <= 0 or _i(r.get('seed_count', 0)) <= 0)
)
pair_refinement_stage_drop_rate = rate(lambda r: _i(r.get('phase2_refined_pair_count', 0)) <= 0)
final_selection_stage_drop_rate = rate(
    lambda r: _i(r.get('corridor_count_after_trim', 0)) <= 0 or _i(r.get('rendered_sentence_count', 0)) <= 0
)

payload = {
    'experiment_family': 'next_method_design',
    'question_being_answered': 'Where does MuSiQue fail in the retrieval-to-generation pipeline?',
    'dataset': str(summary.get('dataset', 'musique')),
    'run_kind': run_kind,
    'n_samples': int(_f(summary.get('n_samples', len(rows)), len(rows))),
    'supporting_fact_recall': _f(summary.get('supporting_fact_recall', 0.0), 0.0),
    'rendered_recall': _f(summary.get('rendered_supporting_fact_recall', summary.get('supporting_fact_recall', 0.0)), 0.0),
    'em': _f(summary.get('em', 0.0), 0.0),
    'f1': _f(summary.get('f1', 0.0), 0.0),
    'retrieval_ms': _f(summary.get('retrieval_latency_ms', 0.0), 0.0),
    'generation_ms': _f(summary.get('generation_ms', 0.0), 0.0),
    'total_ms': _f(summary.get('total_latency_ms', 0.0), 0.0),
    'proposal_coverage': proposal_coverage,
    'proposal_size_mean': proposal_size,
    'proposal_subgraph_nodes_mean': proposal_subgraph_nodes,
    'proposal_subgraph_edges_mean': proposal_subgraph_edges,
    'phase1_run_diversity_mean': phase1_run_diversity,
    'selected_run_rate': selected_run_rate,
    'pair_shortlist_coverage': pair_shortlist_coverage,
    'truncated_corridors_avg': _f(summary.get('truncated_corridors_avg', summary.get('truncated_corridor_count_avg', 0.0)), 0.0),
    'truncated_sentences_avg': _f(summary.get('truncated_sentences_avg', summary.get('truncated_sentence_count_avg', 0.0)), 0.0),
    'fallback_rate': _f(summary.get('fallback_rate', 0.0), 0.0),
    'proposal_stage_drop_rate': proposal_stage_drop_rate,
    'run_selection_stage_drop_rate': run_selection_stage_drop_rate,
    'pair_refinement_stage_drop_rate': pair_refinement_stage_drop_rate,
    'final_selection_stage_drop_rate': final_selection_stage_drop_rate,
    'summary_path': str(summary_path.resolve()),
    'profile_path': str(profile_path.resolve()),
    'git_branch': git_branch,
    'git_commit': git_commit,
    'git_tag': git_tag,
}

out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

lines = []
lines.append('# Next Method Design: MuSiQue Failure Diagnosis')
lines.append('')
lines.append('## Metadata')
lines.append(f"- experiment_family: `next_method_design`")
lines.append(f"- run_kind: `{run_kind}`")
lines.append(f"- dataset: `{payload['dataset']}`")
lines.append(f"- n_samples: `{payload['n_samples']}`")
lines.append(f"- git_branch: `{git_branch}`")
lines.append(f"- git_commit: `{git_commit}`")
lines.append(f"- git_tag: `{git_tag}`")
lines.append('')
lines.append('## Aggregate Metrics')
lines.append('')
lines.append('| metric | value |')
lines.append('| --- | ---: |')
for key in [
    'supporting_fact_recall', 'rendered_recall', 'em', 'f1',
    'retrieval_ms', 'generation_ms', 'total_ms',
    'proposal_coverage', 'proposal_size_mean',
    'proposal_subgraph_nodes_mean', 'proposal_subgraph_edges_mean',
    'phase1_run_diversity_mean', 'selected_run_rate', 'pair_shortlist_coverage',
    'truncated_corridors_avg', 'truncated_sentences_avg', 'fallback_rate',
    'proposal_stage_drop_rate', 'run_selection_stage_drop_rate',
    'pair_refinement_stage_drop_rate', 'final_selection_stage_drop_rate',
]:
    v = payload.get(key)
    if isinstance(v, float):
        lines.append(f"| {key} | {v:.4f} |")
    else:
        lines.append(f"| {key} | {v} |")

lines.append('')
lines.append('## Stage-Failure Interpretation')
lines.append('')

if payload['proposal_stage_drop_rate'] > 0.25:
    lines.append('- Proposal stage drop is high; candidate universe is often too narrow for MuSiQue.')
else:
    lines.append('- Proposal stage drop is not dominant; later-stage decisions are likely more critical.')

if payload['run_selection_stage_drop_rate'] > 0.25:
    lines.append('- Run selection drop is high; best-run objective may miss bridge-complete hypotheses.')
else:
    lines.append('- Run selection drop is moderate/low; run-level objective is not the only bottleneck.')

if payload['pair_refinement_stage_drop_rate'] > 0.25:
    lines.append('- Pair-refinement drop is high; shortlist/refinement path likely loses supporting paths.')
else:
    lines.append('- Pair-refinement drop is moderate/low.')

if payload['final_selection_stage_drop_rate'] > 0.25:
    lines.append('- Final corridor selection drop is high; useful evidence is being filtered late.')
else:
    lines.append('- Final corridor selection drop is moderate/low.')

if payload['rendered_recall'] >= payload['supporting_fact_recall'] - 0.01:
    lines.append('- Render loss appears limited; retrieval/selection ceiling is likely the primary bottleneck.')
else:
    lines.append('- Render stage may contribute to loss; consider render policy after retrieval objective fixes.')

lines.append('')
lines.append('## Artifacts')
lines.append('')
lines.append(f"- summary_path: `{payload['summary_path']}`")
lines.append(f"- profile_path: `{payload['profile_path']}`")

out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

  cp -f "${DIAG_JSON}" "${PROFILE_ROOT}/nextdesign_musique_failure_${run_kind}_summary_latest.json"
  cp -f "${DIAG_MD}" "${PROFILE_ROOT}/nextdesign_musique_failure_${run_kind}_summary_latest.md"

  echo "[nextdesign:musique_failure] run_kind=${run_kind} done"
  echo "  diagnosis_json=${DIAG_JSON}"
  echo "  diagnosis_md=${DIAG_MD}"
done
