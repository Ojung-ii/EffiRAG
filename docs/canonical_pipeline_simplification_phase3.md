# Canonical Pipeline Simplification (Phase-3)

## Motivation
기존 EffiRAG 구현은 실험 라운드를 거치며 objective mode, guard 변형, corridor shaping 분기가 누적되었습니다.
Phase-3의 목표는 계수를 감추는 것이 아니라, 실제 실행 경로를 **single-path canonical pipeline**으로 정리해 구현 복잡도를 낮추는 것입니다.

## What Is Preserved
다음 핵심 철학은 유지합니다.

1. Query-conditioned proposal graph construction
2. Stochastic multi-run PPR over the reduced graph
3. Bounded local evidence subgraph refinement
4. Compact evidence rendering
5. Copy-span answer generation interface

또한 copy-span instruction SOTA 레퍼런스(`light_separator_copy_span_instruction`) 동작 보존이 우선입니다.

## What Is Simplified
Phase-3에서는 아래 구조를 도입했습니다.

1. `canonical_copy_span` objective mode 추가
2. canonical mode의 dataset-aware 동작(Hotpot guarded / others baseline) 명시화
3. legacy objective alias/분기 해석을 `legacy_objectives.py` 경계로 분리
4. 활성/비활성 플래그와 0/비0 가중치를 추적하는 `pipeline_trace.py` 추가
5. 위험한 항목은 즉시 제거하지 않고 ablation mode로만 노출

## Canonical vs Legacy Boundary
- Canonical main path: `canonical_copy_span`
- Optional canonical ablation hooks:
  - `canonical_no_seed_optional`
  - `canonical_no_run_optional`
  - `canonical_no_pair_coverage`
  - `canonical_no_bridge_completeness`
  - `canonical_render_core_only`
  - `canonical_no_dataset_guard`
- Legacy modes: path-preserve/compact-lite/confidence-gated 포함 historical objective modes

## Acceptance Criteria
1. **Parity first**:
   - canonical path가 grouped_v1/reference의 핵심 objective flag 동작을 재현해야 함
   - 기존 invariance 테스트를 깨지 않아야 함
2. **Ablation-based pruning second**:
   - optional term 제거는 Stage-C 임계치 통과 후에만 mainline 반영

## Current Decision
- `canonical_copy_span`는 Phase-3 main candidate로 추가되었습니다.
- minimal ablation modes는 실험 훅으로만 추가되었고, 기본 mainline으로 merge하지 않았습니다.
- 실제 pruning 결정은 Stage-B/Stage-C 결과(성능/latency/parity) 확인 후 진행합니다.

## Phase-3.5 Note
- Phase-3.5에서는 Stage-C(HotpotQA/2Wiki) 무변화 결과를 근거로 canonical mainline에서 no-op optional scoring 가지를 제거합니다.
- 핵심 안전장치(dataset guard, bridge induction, redundancy, copy-span interface)는 유지합니다.
- MuSiQue/PopQA 일반화 이슈는 별도 `Phase-4`로 분리합니다.

## Phase-4 Note
- Phase-4는 성능 튜닝이 아니라 correctness/invariant 정합성 정리 단계입니다.
- PPR graph-struct cache 오염 가능성 제거, canonical pruned-weight 강제, config unknown-key/audit safety를 코드로 강제합니다.
