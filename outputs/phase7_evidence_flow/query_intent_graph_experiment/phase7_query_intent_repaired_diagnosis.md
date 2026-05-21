# Phase7 Query-Intent Repaired Diagnosis

## Repair Verdict
1. Were previous full coverage metrics broken? Yes
2. Were previous failure attributions over-collapsed? Partially

## Variant Delta vs baseline_bq (mean across dataset/profile pairs)
| variant | Δphase1_full | Δselected_full | Δoracle_gap (negative is better) | ΔF1 |
| --- | --- | --- | --- | --- |
| intent_p1 | 0.0000 | 0.0000 | -0.0024 | 0.0110 |
| intent_p1_aq | 0.0000 | -0.0100 | -0.0086 | 0.0086 |
| intent_p1_aq_rq | -0.0900 | -0.0175 | 0.0195 | -0.0323 |

## Explicit Answers
3. Does query intent actually improve Phase1 full-chain coverage? No (mean Δ≈0)
4. Does query intent actually improve selected/rendered full coverage? No overall (mean Δ<=0)
5. Does query intent reduce oracle gap? Only partially
6. Is intent_p1 still the safest improvement? Yes for Hotpot F1, but coverage gains are limited in repaired metrics.
7. Is Aq useful after repaired diagnostics? Mixed: small F1 gains in some cells, but selected_full generally down
8. Is Rq still risky? Yes (notable regressions in hotpot legacy_512_10 intent_p1_aq_rq)
9. What should be the next method direction? Prioritize Phase1 full-chain proposal redesign (evidence packet / anchor robustness) before heavier Aq/Rq tuning.

## Notes
- Repaired summaries use diagnostics fallback when oracle-gap trace rows are missing/incomplete.
- Failure attribution now separates PHASE1_NOT_FOUND vs SEMANTIC_ANCHOR_FAILED (secondary subtype).
- Candidate source metrics are split into count and hit-rate to avoid ambiguity.