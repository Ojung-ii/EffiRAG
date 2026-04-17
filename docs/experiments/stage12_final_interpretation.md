# Stage12 Final Interpretation (Template + Smoke Run Notes)

이 문서는 Stage 1/2 실험 결과 해석용 템플릿입니다.

## Current Status
- Full 200-query 실험: pending
- Smoke validation (limit=1): completed

Generated smoke reports:
- `outputs/profiling/f1_boost/f1_boost_qa_summary_latest.md`
- `outputs/profiling/f1_boost/f1_boost_qa_interpretation_latest.md`
- `outputs/profiling/semantic_selection/semantic_selection_retrieval_only_summary_latest.md`
- `outputs/profiling/semantic_selection/semantic_selection_retrieval_only_interpretation_latest.md`

## Q1. EM 유지 + F1 개선을 위해 support/corridor budget 확장이 유효한가?
- 판정: TBD (200-query 결과 필요)
- 근거 파일: Stage1 summary/interpretation latest

## Q2. retrieval coverage ceiling 개선에 semantic-aware run/seed/union이 유효한가?
- 판정: TBD (200-query 결과 필요)
- 근거 파일: Stage2 summary/interpretation latest

## Q3. 즉각적인 효과가 더 큰 축은?
- 판정: TBD
- 비교 기준: ΔRecall@20, Δrendered_recall, ΔEM/F1, Δtotal_ms

## Q4. 다음 기본 operating point 후보는?
- 임시 후보 규칙:
  - Recall@20 / rendered_recall 개선
  - EM/F1 비하락(또는 개선)
  - total_ms 증가폭 관리 가능
- 최종 후보: TBD (200-query 완료 후 확정)

## Recommended Next Command (full experiment)
```bash
cd /path/to/EffiRAG
CUDA_VISIBLE_DEVICES=0 LIMIT=200 bash scripts/run_stage12_experiments.sh
```
