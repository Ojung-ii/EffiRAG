# Phase5A Score Simplification Report

## Branch
- `refactor/phase5a-score-simplification-diagnostics`

## Commit Hash
- before changes: `8f207f5`
- after changes: not committed yet

## Changed Files
- `effirag/config.py`
- `effirag/run_rag.py`
- `effirag/retrieval.py`
- `scripts/compare_phase5_score_ablation.py`
- `tests/test_phase5_score_ablation.py`
- `docs/phase5a_score_simplification_diagnostics.md`

## New Config Flags (default all off)
- `ablation_no_pair_semantic: False`
- `ablation_no_pair_bridge: False`
- `ablation_no_final_text_rerank: False`
- `score_component_trace_enabled: False`
- `score_component_trace_topn: 20`

Deferred placeholders (default off):
- `ablation_no_local_semantic`
- `ablation_no_top1_correction`
- `ablation_no_run_pre_semantic`
- `ablation_no_run_semantic_coverage`
- `ablation_no_anchor_alignment_in_pair`

## Default Behavior Statement
- No ablation flag is enabled by default.
- Pair score default formula remains explicit and unchanged.
- Final text rerank default path remains unchanged unless `ablation_no_final_text_rerank=true`.
- Score trace is emitted only when `score_component_trace_enabled=true`.

## Unit Tests
- `PYTHONPATH=. /home/ojungii/miniconda3/envs/effirag/bin/python -m pytest -q tests/test_phase5_score_ablation.py`
  - result: `13 passed`
- `PYTHONPATH=. /home/ojungii/miniconda3/envs/effirag/bin/python -m pytest -q tests/test_retrieval.py tests/test_canonical_scoring.py tests/test_canonical_mainline_pruning.py tests/test_canonical_copy_span_objective.py tests/test_copy_span_mainline_invariance.py tests/test_pipeline_trace.py tests/test_phase5_score_ablation.py`
  - result: `45 passed`
- `PYTHONPATH=. /home/ojungii/miniconda3/envs/effirag/bin/python -m pytest -q tests`
  - result: `82 passed`

## Default No-Flag Parity Status
Completed:
1. unit-level invariance check: trace on/off keeps selected sentence IDs and corridor IDs unchanged
2. default pair-score formula exactness test
3. rerank ablation gate test (default path calls rerank, ablation path skips rerank)

Pending:
1. dataset-level no-flag parity run (`n=100`) vs Phase4 canonical round
2. large-scale ablation smoke/full experiments

## Decision Overlap Tooling
Added script:
- `scripts/compare_phase5_score_ablation.py`

Self-compare sanity check:
- baseline==variant input produced top1/jaccard/EM/F1 deltas = zero as expected.

## Smoke / Experiment Commands
Environment:
```bash
cd /home/ojungii/EffiRAG
export BASE_CONFIG=${BASE_CONFIG:-configs/canonical/rag_canonical_copy_span_phase4.yaml}
export OUT_ROOT=${OUT_ROOT:-/home/ojungii/EffiRAG/outputs/phase5a_score_simplification_diagnostics/$(date +%Y%m%d_%H%M%S)}
export LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8011/v1}
export LLM_MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export API_KEY=${API_KEY:-EMPTY}
mkdir -p "$OUT_ROOT"
```

No-flag baseline (`n=100`):
```bash
for DATASET in hotpotqa 2wikimultihopqa; do
  python -m effirag.run_rag \
    --config "$BASE_CONFIG" \
    --dataset "$DATASET" \
    --limit 100 \
    --method effirag \
    --output-dir "$OUT_ROOT/no_flag_baseline" \
    --timestamp-output true \
    --run-qa true \
    --generator openai_compat \
    --llm-base-url "$LLM_BASE_URL" \
    --model-name "$LLM_MODEL" \
    --llm-api-key "$API_KEY"
done
```

Ablation example (`no_pair_semantic`):
```bash
python -m effirag.run_rag \
  --config "$BASE_CONFIG" \
  --dataset hotpotqa \
  --limit 100 \
  --method effirag \
  --output-dir "$OUT_ROOT/no_pair_semantic" \
  --timestamp-output true \
  --run-qa true \
  --generator openai_compat \
  --llm-base-url "$LLM_BASE_URL" \
  --model-name "$LLM_MODEL" \
  --llm-api-key "$API_KEY" \
  --ablation-no-pair-semantic true \
  --score-component-trace-enabled true
```

Compare baseline vs variant:
```bash
BASELINE_JSONL=$(find "$OUT_ROOT/no_flag_baseline" -path "*hotpotqa*" -name rag_query_results.jsonl | head -1)
VARIANT_JSONL=$(find "$OUT_ROOT/no_pair_semantic" -path "*hotpotqa*" -name rag_query_results.jsonl | head -1)
python scripts/compare_phase5_score_ablation.py \
  --baseline "$BASELINE_JSONL" \
  --variant "$VARIANT_JSONL" \
  --output "$OUT_ROOT/compare_hotpotqa_no_pair_semantic.json" \
  --markdown-output "$OUT_ROOT/compare_hotpotqa_no_pair_semantic.md"
```

## Recommendation
1. keep default path unchanged in Phase5A (done)
2. run no-flag parity `n=100` vs Phase4 reference before interpreting ablations
3. use overlap metrics + EM/F1 deltas to decide Phase5B/5C candidate removals
