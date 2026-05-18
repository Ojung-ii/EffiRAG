# Phase-6T Latency Triage Audit

| item | value |
|---|---|
| status | pass |
| profiles | unified_acr_rcedr_v12_sota_contract_fast, unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank, unified_acr_rcedr_v12_sota_contract_fast_rerank5, unified_acr_rcedr_v12_sota_contract_fast_proposal_ultralight |
| datasets | hotpotqa, 2wikimultihopqa |

## Per Profile/Dataset Checks

| profile | dataset | unified_selector | gl_rcedr | answerability_selection | prompt_variant | order_strategy | render_mode | delivery_mode | max_context | max_total | max_sentences | top_corridors | max_corridors | bridge_induction | target_prompt_tokens | max_prompt_tokens | rerank_topn | max_anchors | samples_per_anchor | ppr_mc_walks | corridor_top_bc | ppr_subgraph_max_nodes |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unified_acr_rcedr_v12_sota_contract_fast | hotpotqa | True | False | False | light_separator_copy_span_instruction | score+light_separator_render | corridor_aware_flat | sentence_compressed | 8 | 8 | 8 | 3 | 3 | False | 600 | 700 | 20 | 3 | 3 | 256 | 10 | 15000 |
| unified_acr_rcedr_v12_sota_contract_fast | 2wikimultihopqa | True | False | False | light_separator_copy_span_instruction | score+light_separator_render | corridor_aware_flat | sentence_compressed | 8 | 8 | 8 | 3 | 3 | False | 600 | 700 | 20 | 3 | 3 | 256 | 10 | 15000 |
| unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank | hotpotqa | True | False | False | light_separator_copy_span_instruction | score+light_separator_render | corridor_aware_flat | sentence_compressed | 8 | 8 | 8 | 3 | 3 | False | 600 | 700 | 0 | 3 | 3 | 256 | 10 | 15000 |
| unified_acr_rcedr_v12_sota_contract_fast_no_sentence_rerank | 2wikimultihopqa | True | False | False | light_separator_copy_span_instruction | score+light_separator_render | corridor_aware_flat | sentence_compressed | 8 | 8 | 8 | 3 | 3 | False | 600 | 700 | 0 | 3 | 3 | 256 | 10 | 15000 |
| unified_acr_rcedr_v12_sota_contract_fast_rerank5 | hotpotqa | True | False | False | light_separator_copy_span_instruction | score+light_separator_render | corridor_aware_flat | sentence_compressed | 8 | 8 | 8 | 3 | 3 | False | 600 | 700 | 5 | 3 | 3 | 256 | 10 | 15000 |
| unified_acr_rcedr_v12_sota_contract_fast_rerank5 | 2wikimultihopqa | True | False | False | light_separator_copy_span_instruction | score+light_separator_render | corridor_aware_flat | sentence_compressed | 8 | 8 | 8 | 3 | 3 | False | 600 | 700 | 5 | 3 | 3 | 256 | 10 | 15000 |
| unified_acr_rcedr_v12_sota_contract_fast_proposal_ultralight | hotpotqa | True | False | False | light_separator_copy_span_instruction | score+light_separator_render | corridor_aware_flat | sentence_compressed | 8 | 8 | 8 | 3 | 3 | False | 600 | 700 | 20 | 2 | 2 | 128 | 6 | 8000 |
| unified_acr_rcedr_v12_sota_contract_fast_proposal_ultralight | 2wikimultihopqa | True | False | False | light_separator_copy_span_instruction | score+light_separator_render | corridor_aware_flat | sentence_compressed | 8 | 8 | 8 | 3 | 3 | False | 600 | 700 | 20 | 2 | 2 | 128 | 6 | 8000 |

## Warnings

- none

## Errors

- none

