# Phase8 Entity Universe Debug Report

This report reads existing traces only. Missing trace files are recorded in the JSON output.

| dataset | profile | variant | num_queries | entity_universe_size_mean | entity_universe_size_p50 | entity_universe_size_p95 | avg_query_relevance_mean | max_query_relevance_mean | avg_entity_degree_mean | num_high_degree_filtered_mean | gold_entity_hit_rate_eval_only | num_gold_entities_in_universe_eval_only_mean | num_semantic_entities_mean | num_title_lookup_entities_mean | num_graph_flow_entities_mean | num_source_balanced_entities_mean | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 2000.0000 | 2000.0000 | 2000.0000 | 0.7376 | 0.9941 | 11.3085 | 394.7000 | 0.0000 | 0.0000 | 2000.0000 | 6.6800 | 1829.4700 | 1219.9600 | U_q is broad but eval-only gold entity hit is near zero; U_q construction or gold-entity diagnostic mapping is a primary suspect. |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 2000.0000 | 2000.0000 | 2000.0000 | 0.7376 | 0.9941 | 11.3085 | 394.7000 | 0.0000 | 0.0000 | 2000.0000 | 6.6800 | 1829.4700 | 1219.9600 | U_q is broad but eval-only gold entity hit is near zero; U_q construction or gold-entity diagnostic mapping is a primary suspect. |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Phase8 disabled baseline; U_q is not expected. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 2000.0000 | 2000.0000 | 2000.0000 | 0.7376 | 0.9941 | 11.3085 | 394.7000 | 0.0000 | 0.0000 | 2000.0000 | 6.6800 | 1829.4700 | 1219.9600 | U_q is broad but eval-only gold entity hit is near zero; U_q construction or gold-entity diagnostic mapping is a primary suspect. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 2000.0000 | 2000.0000 | 2000.0000 | 0.7376 | 0.9941 | 11.3085 | 394.7000 | 0.0000 | 0.0000 | 2000.0000 | 6.6800 | 1829.4700 | 1219.9600 | U_q is broad but eval-only gold entity hit is near zero; U_q construction or gold-entity diagnostic mapping is a primary suspect. |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Phase8 disabled baseline; U_q is not expected. |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 2000.0000 | 2000.0000 | 2000.0000 | 0.7303 | 1.0000 | 12.5950 | 616.6900 | 0.0000 | 0.0000 | 2000.0000 | 9.7200 | 1730.4500 | 1306.2500 | U_q is broad but eval-only gold entity hit is near zero; U_q construction or gold-entity diagnostic mapping is a primary suspect. |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | U_q was not built or was not logged. |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Phase8 disabled baseline; U_q is not expected. |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 2000.0000 | 2000.0000 | 2000.0000 | 0.7303 | 1.0000 | 12.5950 | 616.6900 | 0.0000 | 0.0000 | 2000.0000 | 9.7200 | 1730.4500 | 1306.2500 | U_q is broad but eval-only gold entity hit is near zero; U_q construction or gold-entity diagnostic mapping is a primary suspect. |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 1 | 2000.0000 | 2000.0000 | 2000.0000 | 0.7749 | 1.0000 | 9.9230 | 772.0000 | 0.0000 | 0.0000 | 2000.0000 | 4.0000 | 2000.0000 | 1686.0000 | U_q is broad but eval-only gold entity hit is near zero; U_q construction or gold-entity diagnostic mapping is a primary suspect. |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Phase8 disabled baseline; U_q is not expected. |

## Interpretation Rules

- Near-zero `gold_entity_hit_rate_eval_only` marks U_q construction or gold-entity diagnostic mapping as suspicious.
- Large `entity_universe_size_mean` with near-zero gold hits means the universe is broad but not landing on answer-relevant entities.
- Zero `num_source_balanced_entities_mean` in pamae variants means the proposal is not benefiting from source-balanced candidates.

