# EffiRAG Experimental Add-ons (Stage 1 + Stage 2)

이 문서는 frozen 운영 설정을 변경하지 않고, 실험용 파일만 추가한 항목을 정리합니다.

## Scope Guard
- frozen 운영 config/script/README/final reports는 수정하지 않음
- 추가된 것은 `exp_*` config, 신규 scripts, 신규 실험 report 생성 경로만 포함

## Added Configs
### Stage 1: F1 boost (context/render expansion)
- `configs/rag_exp_f1_baseline.yaml`
- `configs/rag_exp_f1_variant_a_support.yaml`
- `configs/rag_exp_f1_variant_b_support_main.yaml`
- `configs/rag_exp_f1_variant_c_support_corridor.yaml`
- `configs/rag_exp_f1_variant_b_support_main_connector.yaml` (optional)

(미러/호환용 retrieval config)
- `configs/retrieval_exp_f1_baseline.yaml`
- `configs/retrieval_exp_f1_variant_a_support.yaml`
- `configs/retrieval_exp_f1_variant_b_support_main.yaml`
- `configs/retrieval_exp_f1_variant_c_support_corridor.yaml`

### Stage 2: Semantic-aware selection ablation
- `configs/rag_exp_semantic_ablation_baseline.yaml`
- `configs/rag_exp_semantic_ablation_run_score.yaml`
- `configs/rag_exp_semantic_ablation_seed_score.yaml`
- `configs/rag_exp_semantic_ablation_union.yaml`

(미러/호환용 retrieval config)
- `configs/retrieval_exp_semantic_ablation_baseline.yaml`
- `configs/retrieval_exp_semantic_ablation_run_score.yaml`
- `configs/retrieval_exp_semantic_ablation_seed_score.yaml`
- `configs/retrieval_exp_semantic_ablation_union.yaml`

## Added Scripts
- `scripts/run_f1_boost_qa.sh`
- `scripts/run_semantic_selection_ablation.sh`
- `scripts/run_semantic_reindex.sh`
- `scripts/run_stage12_experiments.sh`
- `scripts/summarize_experiment_report.py`

## Run Order
1. Stage 1: context/render 확장 (`run_f1_boost_qa.sh`)
2. Stage 2: semantic-aware selection ablation (`run_semantic_selection_ablation.sh`)
3. 필요 시 semantic 전용 reindex (`run_semantic_reindex.sh`)

## Command Examples
### Stage 1 (QA, 기본 hotpotqa + 2wikimultihopqa, LIMIT=200)
```bash
cd /home/ojungii/EffiRAG
CUDA_VISIBLE_DEVICES=0 LIMIT=200 bash scripts/run_f1_boost_qa.sh
```

### Stage 2 (QA)
```bash
cd /home/ojungii/EffiRAG
CUDA_VISIBLE_DEVICES=0 LIMIT=200 RETRIEVAL_ONLY=false bash scripts/run_semantic_selection_ablation.sh
```

### Stage 2 (retrieval-only, MuSiQue 포함)
```bash
cd /home/ojungii/EffiRAG
CUDA_VISIBLE_DEVICES=0 LIMIT=200 RETRIEVAL_ONLY=true INCLUDE_MUSIQUE_RETRIEVAL=true \
  bash scripts/run_semantic_selection_ablation.sh
```

### Ordered runner (Stage1 -> Stage2)
```bash
cd /home/ojungii/EffiRAG
CUDA_VISIBLE_DEVICES=0 LIMIT=200 bash scripts/run_stage12_experiments.sh
```

### Optional semantic reindex (isolated cache namespace)
```bash
cd /home/ojungii/EffiRAG
CUDA_VISIBLE_DEVICES=0 CACHE_ROOT=outputs/index_cache_semantic_exp FORCE_REBUILD=false \
  bash scripts/run_semantic_reindex.sh
```

## Report Outputs
### Stage 1
- Raw profile JSONL: `outputs/profiling/f1_boost/*_q{LIMIT}_{timestamp}.jsonl`
- Summary JSON/MD: `outputs/profiling/f1_boost/f1_boost_qa_summary_q{LIMIT}_{timestamp}.{json,md}`
- Interpretation MD: `outputs/profiling/f1_boost/f1_boost_qa_interpretation_q{LIMIT}_{timestamp}.md`
- Latest symlink-like copies:
  - `outputs/profiling/f1_boost/f1_boost_qa_summary_latest.json`
  - `outputs/profiling/f1_boost/f1_boost_qa_summary_latest.md`
  - `outputs/profiling/f1_boost/f1_boost_qa_interpretation_latest.md`

### Stage 2
- Raw profile JSONL: `outputs/profiling/semantic_selection/*_q{LIMIT}_{timestamp}.jsonl`
- Summary JSON/MD: `outputs/profiling/semantic_selection/semantic_selection_{qa|retrieval_only}_summary_q{LIMIT}_{timestamp}.{json,md}`
- Interpretation MD: `outputs/profiling/semantic_selection/semantic_selection_{qa|retrieval_only}_interpretation_q{LIMIT}_{timestamp}.md`
- Latest copies:
  - `outputs/profiling/semantic_selection/semantic_selection_{qa|retrieval_only}_summary_latest.{json,md}`
  - `outputs/profiling/semantic_selection/semantic_selection_{qa|retrieval_only}_interpretation_latest.md`

## Evaluation Questions Covered
1. EM 유지 + F1 개선에 context/support budget 확장이 유효한가?
2. semantic-aware run/seed/union이 retrieval coverage ceiling 개선에 유효한가?
3. 두 축 중 즉각 효과가 더 큰 축은 무엇인가?
4. 다음 operating point 후보는 무엇인가?

실행 후 `*_summary_latest.md` + `*_interpretation_latest.md`를 기준으로 답을 확정하면 됩니다.
