# Phase8 ID Normalization Report

| dataset | profile | variant | num_queries | phase7_candidate_gold_recall_mean | phase8_query_candidate_gold_recall_mean | normalized_candidate_recall_mean | normalized_selected_recall_mean | normalized_rendered_recall_mean | sf_recall_mean | phase8_logging_mismatch_count | sf_positive_candidate_zero_warning_count | selected_context_positive_selected_zero_warning_count | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 0.0325 | 0.0000 | 0.0325 | 0.0225 | 0.0225 | 0.0225 | 7 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 0.3625 | 0.0000 | 0.3625 | 0.2375 | 0.2375 | 0.2375 | 61 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 0.5475 | 0.0000 | 0.5475 | 0.3425 | 0.3425 | 0.3425 | 84 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 0.0325 | 0.0000 | 0.0325 | 0.0225 | 0.0225 | 0.0225 | 7 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 0.3625 | 0.0000 | 0.3625 | 0.2425 | 0.2425 | 0.2425 | 61 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 0.5475 | 0.0000 | 0.5475 | 0.3725 | 0.3725 | 0.3725 | 84 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 0.0665 | 0.0000 | 0.0665 | 0.0600 | 0.0600 | 0.0600 | 12 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | No primary id-normalization warning for this group. |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 0.7433 | 0.0000 | 0.7433 | 0.4523 | 0.4523 | 0.4523 | 92 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 0.0665 | 0.0000 | 0.0665 | 0.0600 | 0.0600 | 0.0600 | 12 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | No primary id-normalization warning for this group. |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 0.7433 | 0.0000 | 0.7433 | 0.5040 | 0.5040 | 0.5040 | 92 | 0 | 0 | phase8_query_trace logs candidate recall as zero while Phase7 normalized diagnostics find candidates; diagnostic extraction is broken. |

## Warnings

No warnings.

## Checks

- Candidate ids are normalized with `normalize_evidence_id`.
- Gold support ids are normalized with `canonical_support_key(title, sent_idx)`.
- HTML entities are unescaped before comparison, e.g. `&amp;` becomes `&`.
- `s::` atom ids are kept as atom ids; title sentence ids are compared to gold support ids.
- The current source-balanced zero values in `phase8_query_trace` are diagnostic logging mismatches, not evidence that source-balanced candidates lack oracle coverage.

## Phase8 Logging Mismatch Examples

| query_id | phase7_candidate_gold_recall | phase8_query_candidate_gold_recall | normalized_candidate_recall | sf_recall |
| --- | --- | --- | --- | --- |
| 0dfe41f60bdc11eba7f7acde48001122 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| 462bb642099211ebbdb0ac1f6bf848b6 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| 4d3ab509099c11ebbdb0ac1f6bf848b6 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| 63240f22089d11ebbd78ac1f6bf848b6 | 0.2500 | 0.0000 | 0.2500 | 0.2500 |
| a1d9a65c0bd911eba7f7acde48001122 | 0.5000 | 0.0000 | 0.5000 | 0.0000 |
| c9a769c608be11ebbd8aac1f6bf848b6 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| f5e3b9ca0bdb11eba7f7acde48001122 | 0.5000 | 0.0000 | 0.5000 | 0.0000 |
| 006d81bc0bde11eba7f7acde48001122 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| 01309e5008bd11ebbd89ac1f6bf848b6 | 0.2500 | 0.0000 | 0.2500 | 0.2500 |
| 037da85d08c611ebbd90ac1f6bf848b6 | 1.0000 | 0.0000 | 1.0000 | 0.5000 |
