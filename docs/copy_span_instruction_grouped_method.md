# Copy-Span Instruction Grouped Method

## Background
- 이전 Phase-2 grouped 실험은 `champion_reconfirm`를 기준으로 수행되었습니다.
- SOTA reference reconciliation 이후, 더 강한 copy-span 인터페이스 기준은 `light_separator_copy_span_instruction`으로 확인되었습니다.

## Important Framing
- 기존 Phase-2 결과는 **무효가 아니라**, `champion_reconfirm` 프로토콜 하에서 유효합니다.
- 다만 그 결과만으로는 strongest copy-span 결과에 대한 단순화 보존을 주장하기에 불충분합니다.

## Retargeted Scope
이 브랜치(`experiment/copy-span-grouped-from-copy-instruction`)는 grouped 설계를 유지한 채, 기준 레퍼런스를 아래로 재타깃합니다.

- method_name: `copy-span`
- reference_profile: `light_separator_copy_span_instruction`
- reference_label: `copy-span/light_separator_copy_span_instruction`
- previous_reference_profile: `champion_reconfirm`
- retarget_reason: `generation_interface_sota`

## Interface Contract (Must Preserve)
- `order_strategy=score+light_separator_render`
- `prompt_variant=light_separator_copy_span_instruction`
- `added_instruction=copy_span_only`

## Retargeted Profiles
- `copy_span_instruction_grouped_v0`
- `copy_span_instruction_grouped_v1`
- `copy_span_instruction_minimal_v1`
- `copy_span_instruction_no_dataset_toggle` (explicit ablation)

`copy_span_instruction_grouped_v0`는 레퍼런스와의 **대수적 동치(parity)** 확인용이며, 성능 개선용이 아닙니다.

## Acceptance
- 먼저 v0 parity(가중치/Recall/sample alignment)를 통과해야 합니다.
- v1/minimal 평가는 v0 parity 통과 후에만 해석합니다.

