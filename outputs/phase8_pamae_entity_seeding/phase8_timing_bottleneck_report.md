# Phase8 Timing Bottleneck Report

## Compact Timing

| dataset | profile | variant | num_queries | entity_universe_ms_mean | sampling_ms_mean | kmedoids_ms_mean | seed_selection_ms_mean | refinement_ms_mean | evidence_proposal_ms_mean | feature_construction_ms_mean | selection_ms_mean | generation_ms_mean | total_retrieval_ms_mean | total_ms_mean | largest_phase8_stage | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 3053.1479 | 1651.4936 | 1639.8560 | 664.6839 | 259.8904 | 2281.6021 | 519.8867 | 48.8243 | 82.3221 | 21118.1878 | 21380.5359 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 2906.4565 | 1814.2138 | 1801.5453 | 723.9981 | 1511.9678 | 3434.5648 | 705.7424 | 68.6282 | 68.8501 | 25024.7380 | 25302.7599 | evidence_proposal_ms | Evidence proposal is the largest Phase8 stage. |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 730.2614 | 68.1681 | 217.1810 | 15344.7971 | 15755.9385 | entity_universe_ms | Baseline timing; Phase8 stages should be zero. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 2935.6602 | 1738.4477 | 1725.7087 | 685.8017 | 271.0581 | 2296.1980 | 522.7841 | 53.8677 | 73.7303 | 20713.1164 | 20964.9706 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 2966.1339 | 1689.5240 | 1676.9227 | 690.9465 | 1475.0237 | 3548.3968 | 717.4380 | 77.0351 | 77.1176 | 25158.7831 | 25449.7509 | evidence_proposal_ms | Evidence proposal is the largest Phase8 stage. |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 775.0696 | 80.8277 | 206.4730 | 15469.2437 | 15880.7765 | entity_universe_ms | Baseline timing; Phase8 stages should be zero. |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 4544.5553 | 1665.4302 | 1653.4578 | 668.6663 | 261.2972 | 4408.0646 | 984.5577 | 82.5270 | 65.8371 | 35980.1614 | 36360.1573 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1504.7662 | 112.2913 | 175.5726 | 24289.6216 | 24836.1877 | entity_universe_ms | Baseline timing; Phase8 stages should be zero. |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 4690.6177 | 1786.1395 | 1773.1556 | 709.9138 | 281.0666 | 4511.9614 | 996.1127 | 85.0658 | 73.0515 | 36169.6391 | 36554.3341 | entity_universe_ms | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 1 | 4560.5199 | 1604.9105 | 1594.7569 | 720.1384 | 3938.2122 | 5812.1430 | 1349.4523 | 117.8569 | 781.7042 | 72163.1359 | 73329.6133 | evidence_proposal_ms | Evidence proposal is the largest Phase8 stage. |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1484.8971 | 120.5532 | 182.3595 | 24257.6635 | 24813.7115 | entity_universe_ms | Baseline timing; Phase8 stages should be zero. |

## Detailed Timing

| dataset | profile | variant | num_queries | entity_universe_ms_mean | entity_universe_ms_p50 | entity_universe_ms_p95 | entity_universe_ms_max | sampling_ms_mean | sampling_ms_p50 | sampling_ms_p95 | sampling_ms_max | kmedoids_ms_mean | kmedoids_ms_p50 | kmedoids_ms_p95 | kmedoids_ms_max | seed_selection_ms_mean | seed_selection_ms_p50 | seed_selection_ms_p95 | seed_selection_ms_max | refinement_ms_mean | refinement_ms_p50 | refinement_ms_p95 | refinement_ms_max | evidence_proposal_ms_mean | evidence_proposal_ms_p50 | evidence_proposal_ms_p95 | evidence_proposal_ms_max | feature_construction_ms_mean | feature_construction_ms_p50 | feature_construction_ms_p95 | feature_construction_ms_max | selection_ms_mean | selection_ms_p50 | selection_ms_p95 | selection_ms_max | rendering_ms_mean | rendering_ms_p50 | rendering_ms_p95 | rendering_ms_max | generation_ms_mean | generation_ms_p50 | generation_ms_p95 | generation_ms_max | total_retrieval_ms_mean | total_retrieval_ms_p50 | total_retrieval_ms_p95 | total_retrieval_ms_max | total_ms_mean | total_ms_p50 | total_ms_p95 | total_ms_max | largest_phase8_stage | largest_phase8_stage_ms | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5 | 100 | 3053.1479 | 2940.1181 | 3847.9233 | 4995.5769 | 1651.4936 | 1597.0515 | 1997.3210 | 2392.5579 | 1639.8560 | 1581.8406 | 1978.0202 | 2378.4591 | 664.6839 | 634.3617 | 870.1281 | 944.5261 | 259.8904 | 253.7965 | 339.3499 | 402.8790 | 2281.6021 | 2215.7903 | 2968.8273 | 3197.4941 | 519.8867 | 491.5191 | 715.3607 | 994.7211 | 48.8243 | 45.3757 | 70.2412 | 132.7166 | 0.1102 | 0.0403 | 0.0703 | 6.5027 | 82.3221 | 45.4549 | 93.7896 | 3003.9590 | 21118.1878 | 20819.3155 | 23675.8720 | 48977.9594 | 21380.5359 | 21037.4372 | 23866.9245 | 52106.7806 | entity_universe_ms | 3053.1479 | Entity universe construction is the largest Phase8 stage. |
| 2wikimultihopqa | balanced_384_8 | pamae_seed_k5_refine | 100 | 2906.4565 | 2803.1215 | 3607.2537 | 4323.5655 | 1814.2138 | 1758.5884 | 2324.1513 | 2511.7332 | 1801.5453 | 1744.3346 | 2303.5772 | 2501.5656 | 723.9981 | 684.1845 | 942.1482 | 1082.0158 | 1511.9678 | 1331.2398 | 2585.8011 | 3787.2277 | 3434.5648 | 3390.1596 | 4085.2628 | 4635.1248 | 705.7424 | 678.9655 | 926.5922 | 2135.0293 | 68.6282 | 66.7047 | 95.7866 | 152.1089 | 0.0504 | 0.0476 | 0.0796 | 0.0977 | 68.8501 | 49.1304 | 158.3527 | 576.3818 | 25024.7380 | 24916.8215 | 26390.6726 | 62216.8773 | 25302.7599 | 25206.6659 | 26651.5371 | 63003.7946 | evidence_proposal_ms | 3434.5648 | Evidence proposal is the largest Phase8 stage. |
| 2wikimultihopqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 730.2614 | 705.7257 | 1011.4506 | 1299.7259 | 68.1681 | 62.3222 | 111.0514 | 179.0561 | 0.0394 | 0.0362 | 0.0614 | 0.1498 | 217.1810 | 49.2422 | 1299.0027 | 2672.6288 | 15344.7971 | 10867.1458 | 40059.8281 | 152350.8966 | 15755.9385 | 11093.3544 | 41644.8714 | 155292.4647 | entity_universe_ms | 0.0000 | Baseline timing; Phase8 stages should be zero. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5 | 100 | 2935.6602 | 2827.8146 | 3828.7434 | 4990.0329 | 1738.4477 | 1643.4741 | 2305.4941 | 2963.7740 | 1725.7087 | 1628.8148 | 2295.2638 | 2952.7624 | 685.8017 | 635.2850 | 953.7183 | 1526.8081 | 271.0581 | 261.4284 | 370.5807 | 572.2835 | 2296.1980 | 2240.0755 | 2996.7866 | 3885.4320 | 522.7841 | 487.3677 | 810.7811 | 1885.9787 | 53.8677 | 51.7044 | 79.5858 | 127.1142 | 0.0510 | 0.0481 | 0.0714 | 0.1210 | 73.7303 | 52.0126 | 94.5417 | 1610.9251 | 20713.1164 | 20619.7804 | 22821.7559 | 46906.8885 | 20964.9706 | 20886.8411 | 23152.2051 | 48730.3322 | entity_universe_ms | 2935.6602 | Entity universe construction is the largest Phase8 stage. |
| 2wikimultihopqa | legacy_512_10 | pamae_seed_k5_refine | 100 | 2966.1339 | 2845.0424 | 3881.8032 | 4533.7871 | 1689.5240 | 1616.0767 | 2191.9940 | 2567.9686 | 1676.9227 | 1603.1662 | 2178.4638 | 2557.0159 | 690.9465 | 648.9266 | 951.4372 | 1091.0529 | 1475.0237 | 1288.2664 | 2659.5025 | 3169.5798 | 3548.3968 | 3467.4580 | 4346.8337 | 4825.8584 | 717.4380 | 681.4125 | 1031.9950 | 1224.3562 | 77.0351 | 74.1902 | 115.6791 | 146.4561 | 0.0562 | 0.0521 | 0.0814 | 0.1144 | 77.1176 | 57.4386 | 144.7774 | 645.1737 | 25158.7831 | 24905.8252 | 27322.9798 | 57857.3482 | 25449.7509 | 25179.7803 | 27593.5552 | 58698.8880 | evidence_proposal_ms | 3548.3968 | Evidence proposal is the largest Phase8 stage. |
| 2wikimultihopqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 775.0696 | 749.4537 | 1079.9615 | 1163.8909 | 80.8277 | 76.3580 | 120.0633 | 180.3212 | 0.0477 | 0.0457 | 0.0660 | 0.1267 | 206.4730 | 49.3991 | 1351.5395 | 2661.6711 | 15469.2437 | 11189.7444 | 41368.6908 | 153569.4267 | 15880.7765 | 11481.1540 | 42868.0181 | 156430.2449 | entity_universe_ms | 0.0000 | Baseline timing; Phase8 stages should be zero. |
| hotpotqa | balanced_384_8 | pamae_seed_k5 | 100 | 4544.5553 | 4508.2442 | 5533.4389 | 5765.5836 | 1665.4302 | 1628.2448 | 2010.5769 | 2085.2934 | 1653.4578 | 1615.5143 | 1999.9230 | 2072.2971 | 668.6663 | 627.0831 | 851.1370 | 1225.6181 | 261.2972 | 260.3309 | 327.6132 | 394.1500 | 4408.0646 | 4376.5488 | 5560.1603 | 6164.4554 | 984.5577 | 999.0268 | 1351.3176 | 1701.7333 | 82.5270 | 76.3498 | 133.3781 | 196.5185 | 0.0517 | 0.0488 | 0.0818 | 0.0992 | 65.8371 | 48.8947 | 119.2272 | 922.9416 | 35980.1614 | 36461.7534 | 41001.2919 | 64894.7808 | 36360.1573 | 36825.5476 | 41691.1746 | 66320.7877 | entity_universe_ms | 4544.5553 | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | balanced_384_8 | pamae_seed_k5_refine | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | entity_universe_ms | 0.0000 | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | balanced_384_8 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1504.7662 | 1494.3699 | 1910.7685 | 2290.6990 | 112.2913 | 107.4115 | 153.6340 | 256.7215 | 0.0470 | 0.0440 | 0.0663 | 0.1307 | 175.5726 | 52.6944 | 1248.8586 | 1936.9484 | 24289.6216 | 20884.4395 | 46961.1344 | 163104.5846 | 24836.1877 | 21380.9213 | 48596.5844 | 165456.2174 | entity_universe_ms | 0.0000 | Baseline timing; Phase8 stages should be zero. |
| hotpotqa | legacy_512_10 | pamae_seed_k5 | 100 | 4690.6177 | 4646.5805 | 5688.9838 | 7236.1115 | 1786.1395 | 1719.1799 | 2281.4616 | 2590.2140 | 1773.1556 | 1707.0017 | 2267.8126 | 2569.0369 | 709.9138 | 670.0654 | 955.6930 | 1210.8400 | 281.0666 | 261.0778 | 412.6621 | 517.1125 | 4511.9614 | 4446.9956 | 5732.5265 | 6485.6542 | 996.1127 | 955.9292 | 1415.4244 | 3169.2310 | 85.0658 | 79.4750 | 125.3525 | 163.0708 | 0.0547 | 0.0507 | 0.0836 | 0.1480 | 73.0515 | 58.3716 | 138.8733 | 689.5588 | 36169.6391 | 36564.8091 | 40311.5195 | 69310.8728 | 36554.3341 | 36927.9666 | 40683.0050 | 70367.0869 | entity_universe_ms | 4690.6177 | Entity universe construction is the largest Phase8 stage. |
| hotpotqa | legacy_512_10 | pamae_seed_k5_refine | 1 | 4560.5199 | 4560.5199 | 4560.5199 | 4560.5199 | 1604.9105 | 1604.9105 | 1604.9105 | 1604.9105 | 1594.7569 | 1594.7569 | 1594.7569 | 1594.7569 | 720.1384 | 720.1384 | 720.1384 | 720.1384 | 3938.2122 | 3938.2122 | 3938.2122 | 3938.2122 | 5812.1430 | 5812.1430 | 5812.1430 | 5812.1430 | 1349.4523 | 1349.4523 | 1349.4523 | 1349.4523 | 117.8569 | 117.8569 | 117.8569 | 117.8569 | 0.1059 | 0.1059 | 0.1059 | 0.1059 | 781.7042 | 781.7042 | 781.7042 | 781.7042 | 72163.1359 | 72163.1359 | 72163.1359 | 72163.1359 | 73329.6133 | 73329.6133 | 73329.6133 | 73329.6133 | evidence_proposal_ms | 5812.1430 | Evidence proposal is the largest Phase8 stage. |
| hotpotqa | legacy_512_10 | source_balanced_128 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1484.8971 | 1464.0864 | 1984.8939 | 2163.1147 | 120.5532 | 111.8511 | 196.4504 | 266.3173 | 0.0518 | 0.0455 | 0.0844 | 0.1672 | 182.3595 | 59.9905 | 1264.9191 | 1931.8678 | 24257.6635 | 20324.7035 | 52960.9489 | 168921.8817 | 24813.7115 | 20788.9039 | 54471.7853 | 171287.0328 | entity_universe_ms | 0.0000 | Baseline timing; Phase8 stages should be zero. |

## Explicit Answers

### 2wikimultihopqa/balanced_384_8/pamae_seed_k5

- Entity universe dominant: True
- K-medoids expensive: True
- Evidence proposal expensive: True
- Refinement meaningful cost: False
- Optimize first: entity_universe_ms

### 2wikimultihopqa/balanced_384_8/pamae_seed_k5_refine

- Entity universe dominant: False
- K-medoids expensive: True
- Evidence proposal expensive: True
- Refinement meaningful cost: True
- Optimize first: evidence_proposal_ms

### 2wikimultihopqa/legacy_512_10/pamae_seed_k5

- Entity universe dominant: True
- K-medoids expensive: True
- Evidence proposal expensive: True
- Refinement meaningful cost: False
- Optimize first: entity_universe_ms

### 2wikimultihopqa/legacy_512_10/pamae_seed_k5_refine

- Entity universe dominant: False
- K-medoids expensive: True
- Evidence proposal expensive: True
- Refinement meaningful cost: True
- Optimize first: evidence_proposal_ms

### hotpotqa/balanced_384_8/pamae_seed_k5

- Entity universe dominant: True
- K-medoids expensive: False
- Evidence proposal expensive: True
- Refinement meaningful cost: False
- Optimize first: entity_universe_ms

### hotpotqa/balanced_384_8/pamae_seed_k5_refine

- Entity universe dominant: True
- K-medoids expensive: False
- Evidence proposal expensive: False
- Refinement meaningful cost: False
- Optimize first: entity_universe_ms

### hotpotqa/legacy_512_10/pamae_seed_k5

- Entity universe dominant: True
- K-medoids expensive: False
- Evidence proposal expensive: True
- Refinement meaningful cost: False
- Optimize first: entity_universe_ms

### hotpotqa/legacy_512_10/pamae_seed_k5_refine

- Entity universe dominant: False
- K-medoids expensive: False
- Evidence proposal expensive: True
- Refinement meaningful cost: True
- Optimize first: evidence_proposal_ms

