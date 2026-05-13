# Copy-Span SOTA Lock (Mainline)

이 문서는 EffiRAG 메인 결과를 `copy-span`(=`champion_reconfirm`)으로 고정하기 위한 불변 계약(invariance contract)입니다.

## 1) SOTA Definition

- Mainline SOTA profile: `champion_reconfirm` (`copy-span`)
- Lock-in round root: `outputs/light_separator_lockin_round/20260428_004715`
- Decision rule: 메인 표는 `champion_reconfirm` 기준으로 보고, `light_separator_render`는 부가/분석 결과로 분리

Canonical config files:
- `outputs/light_separator_lockin_round/20260428_004715/configs/s0/champion_reconfirm/hotpotqa/rag.yaml`
- `outputs/light_separator_lockin_round/20260428_004715/configs/s0/champion_reconfirm/2wikimultihopqa/rag.yaml`
- `outputs/light_separator_lockin_round/20260428_004715/configs/s0/champion_reconfirm/musique/rag.yaml`
- `outputs/light_separator_lockin_round/20260428_004715/configs/s0/champion_reconfirm/popqa/rag.yaml`

Champion lineage (reference):
- `outputs/retrieval_metric_hotpot_2wiki_round_rerun_strict/20260422_231715/configs/s1/champion_config/hotpotqa/rag.yaml`
- `outputs/retrieval_metric_hotpot_2wiki_round_rerun_strict/20260422_231715/configs/s1/champion_config/2wikimultihopqa/rag.yaml`

## 2) Global Locked Settings (Do Not Change)

아래는 네 데이터셋 공통 고정값이다.

### Retrieval/Seed Weights

- `run_score_semantic_weight=0.30`
- `run_score_anchor_weight=0.20`
- `run_score_structure_weight=0.25`
- `run_score_bridge_weight=0.15`
- `run_score_redundancy_weight=0.10`
- `seed_score_semantic_weight=0.35`
- `seed_score_graph_weight=0.45`
- `seed_score_anchor_weight=0.20`
- `embedding_weight=0.35`

### Render Blend Weights

- `alpha=1.0`
- `beta=0.35`
- `gamma_main=0.45`
- `delta_support=0.20`
- `eta_connector=0.20`
- `zeta_query=0.12`
- `xi_locality=0.08`
- `lambda_redundancy=0.25`

### Budget/Shape

- `max_context_sentences=8`
- `max_total_sentences=8`
- `top_corridors=3`
- `delivery_mode=sentence_compressed`
- `render_mode=corridor_aware_flat`
- `order_strategy=score` (중요: `score+light_separator_render` 금지)

### Disabled Modules (Stay Disabled)

- `top1_correction_enabled=false`
- `run_light_rerank_enabled=false`
- `corridor_compact_shaping_enabled=false`
- `corridor_bridge_purity_shaping_enabled=false`
- `chunk_grounding_enabled=false`
- `hybrid_anchor_recall_enabled=false`
- `role_aware_chunk_scoring_enabled=false`
- `coverage_selection_enabled=false`

## 3) Dataset-Specific Locked Toggles

- `hotpotqa`
  - `retrieval_objective_mode=p3_answer_preserve_guarded_hotpot`
  - `answer_support_pinning_enabled=true`
  - `final_top_slice_reorder_enabled=false`
  - precomputed retrieval strict 사용
- `2wikimultihopqa`
  - `retrieval_objective_mode=baseline`
  - `answer_support_pinning_enabled=false`
  - `final_top_slice_reorder_enabled=true`
  - precomputed retrieval strict 사용
- `musique`
  - `retrieval_objective_mode=baseline`
  - `answer_support_pinning_enabled=false`
  - `final_top_slice_reorder_enabled=false`
  - on-the-fly retrieval (champion precomputed 없음)
- `popqa`
  - `retrieval_objective_mode=baseline`
  - `answer_support_pinning_enabled=false`
  - `final_top_slice_reorder_enabled=false`
  - on-the-fly retrieval (champion precomputed 없음)

## 4) Runtime Semantics Freeze (Code-Level)

수치를 안 바꿔도 아래 동작이 바뀌면 결과가 달라질 수 있으므로 고정한다.

- Objective profile 적용 체인 유지:
  - `_resolve_retrieval_objective_flags`
  - `_apply_connector_objective_profile`
  - call order 포함 고정
  - code refs: `effirag/retrieval.py:5619`, `effirag/retrieval.py:5620`
- Delivery mode 기반 유효값 보정 로직 유지:
  - `eff_chunk_grounding_enabled` / package weight 보정
  - code ref: `effirag/run_rag.py:949-976`

## 5) Refactor Policy

### Allowed (Safe)

- 공통 함수 추출(정규화/가중합/bool 파싱)
- 비활성 모듈 fast-path 추가(동일 출력 보장 시)
- 로깅/진단 키 정리(기존 키 유지 조건)
- 설정 로딩 코드 단순화(최종 유효 config 동일 조건)

### Forbidden (Breaks SOTA Contract)

- 고정 가중치/토글/예산 값 변경
- `order_strategy=score` 변경
- objective profile 적용 순서/분기 의미 변경
- precomputed strict 처리 규칙 변경(hotpot/2wiki)
- 락인 검증 지표 키 삭제(`retrieval_invariance_pass` 등)

## 6) Invariance Validation Checklist

PR 머지 전 아래를 만족해야 한다.

1. Config parity
- canonical YAML 대비 key/value 동일(예외: output path/timestamp)

2. Retrieval invariance
- hotpotqa/2wikimultihopqa: `recall_at_1/5/10` delta = 0

3. Main QA guard
- EM/F1 하락 금지(메인 리포트 기준)

4. Source-path guard
- hotpotqa/2wiki: `precomputed_retrieval_strict=true` 유지
- musique/popqa: fallback 동작 유지

5. Strategy guard
- metadata 상 `final_order_strategy=score` 유지

## 7) Notes

- `light_separator_render`는 별도 분석/부록 후보로 취급한다.
- 메인 SOTA 문구는 반드시 `copy-span (champion_reconfirm)`로 표기한다.
