# Phase8 Seed Quality Debug Report

| dataset | profile | variant | num_queries | k | sample_size | num_samples | best_seed_score_mean | best_seed_relevance_mean | best_seed_coverage_mean | best_seed_diversity_mean | best_seed_hub_penalty_mean | seed_gold_hit_rate_eval_only | seed_evidence_gold_hit_rate_eval_only | num_refined_seeds_mean | num_changed_seeds_mean | changed_seed_rate | before_seed_gold_hit_rate_eval_only | after_seed_gold_hit_rate_eval_only | before_seed_evidence_gold_hit_rate_eval_only | after_seed_evidence_gold_hit_rate_eval_only | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 5.0000 | 200.0000 | 5.0000 | 1.7340 | 0.7500 | 0.8753 | 0.1449 | 0.0361 | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing. |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 5.0000 | 200.0000 | 5.0000 | 1.7340 | 0.7500 | 0.8753 | 0.1449 | 0.0361 | 0.0000 | 0.0000 | 5.0000 | 4.9900 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing. |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Phase8 disabled baseline; seed quality is not applicable. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 5.0000 | 200.0000 | 5.0000 | 1.7340 | 0.7500 | 0.8753 | 0.1449 | 0.0361 | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 5.0000 | 200.0000 | 5.0000 | 1.7340 | 0.7500 | 0.8753 | 0.1449 | 0.0361 | 0.0000 | 0.0000 | 5.0000 | 4.9900 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing. |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Phase8 disabled baseline; seed quality is not applicable. |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 5.0000 | 200.0000 | 5.0000 | 1.7270 | 0.7445 | 0.8703 | 0.1541 | 0.0418 | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing. |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing. |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Phase8 disabled baseline; seed quality is not applicable. |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 5.0000 | 200.0000 | 5.0000 | 1.7270 | 0.7445 | 0.8703 | 0.1541 | 0.0418 | 0.0000 | 0.0000 | 5.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing. |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 1 | 5.0000 | 200.0000 | 5.0000 | 1.7741 | 0.7924 | 0.8904 | 0.1193 | 0.0280 | 0.0000 | 0.0000 | 5.0000 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Best medoid seeds rarely hit eval-only gold entities; seed selection or U_q/gold mapping is failing. |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Phase8 disabled baseline; seed quality is not applicable. |

## Representative Cases

### low_seed_relevance

| query_id | reason | best_seed_relevance | best_hub_penalty | num_changed_seeds | question |
| --- | --- | --- | --- | --- | --- |
| 5ab57fc4554299488d4d99c0 | low_seed_relevance | 0.6537 | 0.0310 | 0.0000 | In which county is the town in which Raymond Robertsen was born ? |
| 5ab57fc4554299488d4d99c0 | low_seed_relevance | 0.6537 | 0.0310 | 0.0000 | In which county is the town in which Raymond Robertsen was born ? |
| 5a8e41735542995085b37406 | low_seed_relevance | 0.6610 | 0.0300 | 0.0000 | What is the county seat of the county in which Keokuk Falls, Oklahoma is loated? |
| 5a8e41735542995085b37406 | low_seed_relevance | 0.6610 | 0.0300 | 0.0000 | What is the county seat of the county in which Keokuk Falls, Oklahoma is loated? |
| 5a797bd95542994bb9457016 | low_seed_relevance | 0.6644 | 0.0360 | 0.0000 | What age was Georgia Middleman when she started singing in the seventh-most populated city in the United States? |
| 5a797bd95542994bb9457016 | low_seed_relevance | 0.6644 | 0.0360 | 0.0000 | What age was Georgia Middleman when she started singing in the seventh-most populated city in the United States? |
| 5a84fd085542994c784ddaba | low_seed_relevance | 0.6751 | 0.0420 | 0.0000 | Devil's Food is a singles compilation by an American rock and roll band that has also been known to play country shows under what? |
| 5a84fd085542994c784ddaba | low_seed_relevance | 0.6751 | 0.0420 | 0.0000 | Devil's Food is a singles compilation by an American rock and roll band that has also been known to play country shows under what? |
| 5ab3a83d5542992ade7c6e08 | low_seed_relevance | 0.6782 | 0.0380 | 0.0000 | Faith Goldy got fired after an interview she gave on what production site edited by Andrew Anglin? |
| 5ab3a83d5542992ade7c6e08 | low_seed_relevance | 0.6782 | 0.0380 | 0.0000 | Faith Goldy got fired after an interview she gave on what production site edited by Andrew Anglin? |

### high_hub_penalty

| query_id | reason | best_seed_relevance | best_hub_penalty | num_changed_seeds | question |
| --- | --- | --- | --- | --- | --- |
| 5a7732dc55429972597f149b | high_hub_penalty | 0.7952 | 0.0880 | 0.0000 | What 1944 Bollywood film was the mother of Bollywood actor Govinda in? |
| 5a7732dc55429972597f149b | high_hub_penalty | 0.7952 | 0.0880 | 0.0000 | What 1944 Bollywood film was the mother of Bollywood actor Govinda in? |
| 5a7a3a945542996a35c17147 | high_hub_penalty | 0.7573 | 0.0760 | 0.0000 | One Raffles Place is one of the tallest skyscrapers in the city of Singapore and tallest in the wolrd outside North America until it was succeeded by a Building in city? |
| 5a7a3a945542996a35c17147 | high_hub_penalty | 0.7573 | 0.0760 | 0.0000 | One Raffles Place is one of the tallest skyscrapers in the city of Singapore and tallest in the wolrd outside North America until it was succeeded by a Building in city? |
| 5a7b3fc155429931da12ca3a | high_hub_penalty | 0.7767 | 0.0750 | 0.0000 | Wexner Graduate Fellowships are given to students who show an ability of an individual to "lead" or guide who? |
| 5ae40b2b55429970de88d8b3 | high_hub_penalty | 0.7486 | 0.0750 | 0.0000 | In between  Polytechnic University of the Philippines and California Polytechnic State University which was founded as a vocational high school? |
| 5a7b3fc155429931da12ca3a | high_hub_penalty | 0.7767 | 0.0750 | 0.0000 | Wexner Graduate Fellowships are given to students who show an ability of an individual to "lead" or guide who? |
| 5ae40b2b55429970de88d8b3 | high_hub_penalty | 0.7486 | 0.0750 | 0.0000 | In between  Polytechnic University of the Philippines and California Polytechnic State University which was founded as a vocational high school? |
| fcdafe320bdb11eba7f7acde48001122 | high_hub_penalty | 0.7507 | 0.0730 | 0.0000 | Where was the director of film The Circus Cyclone born? |
| fcdafe320bdb11eba7f7acde48001122 | high_hub_penalty | 0.7507 | 0.0730 | 5.0000 | Where was the director of film The Circus Cyclone born? |

### refinement_decreased_hit

No cases found.

### refinement_improved_hit

No cases found.

