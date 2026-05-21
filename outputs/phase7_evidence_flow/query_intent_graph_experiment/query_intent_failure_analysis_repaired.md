# Query-Intent Failure Analysis (Repaired)

## Failure Count Table
| dataset | profile | variant | n_failed | PHASE1_NOT_FOUND | SEMANTIC_ANCHOR_FAILED | PRUNING_FAILED | FINAL_SELECTION_FAILED | RENDERING_ID_MISMATCH | SELECTED_NOT_SUFFICIENT | QA_PROMPT_FAILED_WITH_GOLD_CONTEXT | QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT | OTHER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | baseline_bq | 94 | 17 | 17 | 5 | 11 | 0 | 21 | 31 | 0 | 0 |
| 2wikimultihopqa | balanced_384_8 | intent_p1 | 92 | 17 | 17 | 3 | 16 | 0 | 19 | 30 | 0 | 0 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq | 91 | 17 | 17 | 3 | 15 | 0 | 16 | 32 | 0 | 0 |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq_rq | 93 | 17 | 17 | 3 | 15 | 0 | 18 | 32 | 0 | 0 |
| 2wikimultihopqa | legacy_512_10 | baseline_bq | 94 | 17 | 17 | 5 | 8 | 0 | 16 | 35 | 2 | 0 |
| 2wikimultihopqa | legacy_512_10 | intent_p1 | 94 | 17 | 17 | 3 | 12 | 0 | 14 | 36 | 2 | 0 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq | 94 | 17 | 17 | 3 | 13 | 0 | 14 | 34 | 1 | 0 |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq_rq | 94 | 17 | 17 | 3 | 14 | 0 | 14 | 34 | 0 | 0 |
| hotpotqa | balanced_384_8 | baseline_bq | 76 | 5 | 5 | 5 | 12 | 0 | 19 | 21 | 3 | 6 |
| hotpotqa | balanced_384_8 | intent_p1 | 76 | 5 | 5 | 4 | 13 | 0 | 17 | 21 | 3 | 8 |
| hotpotqa | balanced_384_8 | intent_p1_aq | 78 | 5 | 5 | 4 | 14 | 0 | 15 | 22 | 3 | 10 |
| hotpotqa | balanced_384_8 | intent_p1_aq_rq | 77 | 5 | 5 | 4 | 13 | 0 | 15 | 22 | 3 | 10 |
| hotpotqa | legacy_512_10 | baseline_bq | 74 | 5 | 5 | 5 | 8 | 0 | 19 | 22 | 5 | 5 |
| hotpotqa | legacy_512_10 | intent_p1 | 72 | 5 | 5 | 4 | 8 | 0 | 17 | 23 | 4 | 6 |
| hotpotqa | legacy_512_10 | intent_p1_aq | 73 | 5 | 5 | 4 | 11 | 0 | 15 | 25 | 2 | 6 |
| hotpotqa | legacy_512_10 | intent_p1_aq_rq | 85 | 61 | 23 | 0 | 3 | 0 | 3 | 6 | 4 | 0 |

## Downstream Shift (vs baseline_bq)
| dataset | profile | variant | ΔPHASE1_NOT_FOUND | ΔSEMANTIC_ANCHOR_FAILED | ΔFINAL_SELECTION_FAILED | ΔSELECTED_NOT_SUFFICIENT | ΔQA_PROMPT_FAILED_WITH_GOLD_CONTEXT | transition_flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimultihopqa | balanced_384_8 | intent_p1 | 0 | 0 | 5 | -2 | -1 | - |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq | 0 | 0 | 4 | -5 | 1 | SELECTED_NOT_SUFFICIENT→QA_PROMPT_FAILED |
| 2wikimultihopqa | balanced_384_8 | intent_p1_aq_rq | 0 | 0 | 4 | -3 | 1 | SELECTED_NOT_SUFFICIENT→QA_PROMPT_FAILED |
| 2wikimultihopqa | legacy_512_10 | intent_p1 | 0 | 0 | 4 | -2 | 1 | SELECTED_NOT_SUFFICIENT→QA_PROMPT_FAILED |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq | 0 | 0 | 5 | -2 | -1 | - |
| 2wikimultihopqa | legacy_512_10 | intent_p1_aq_rq | 0 | 0 | 6 | -2 | -1 | - |
| hotpotqa | balanced_384_8 | intent_p1 | 0 | 0 | 1 | -2 | 0 | - |
| hotpotqa | balanced_384_8 | intent_p1_aq | 0 | 0 | 2 | -4 | 1 | SELECTED_NOT_SUFFICIENT→QA_PROMPT_FAILED |
| hotpotqa | balanced_384_8 | intent_p1_aq_rq | 0 | 0 | 1 | -4 | 1 | SELECTED_NOT_SUFFICIENT→QA_PROMPT_FAILED |
| hotpotqa | legacy_512_10 | intent_p1 | 0 | 0 | 0 | -2 | 1 | SELECTED_NOT_SUFFICIENT→QA_PROMPT_FAILED |
| hotpotqa | legacy_512_10 | intent_p1_aq | 0 | 0 | 3 | -4 | 3 | SELECTED_NOT_SUFFICIENT→QA_PROMPT_FAILED |
| hotpotqa | legacy_512_10 | intent_p1_aq_rq | 56 | 18 | -5 | -16 | -16 | - |

## Representative Cases
### 2wikimultihopqa / balanced_384_8 / baseline_bq
- PHASE1_NOT_FOUND:
  - 2dc690ba0bdc11eba7f7acde48001122 | f1=0.0000 | gold=0.7500 | plus=0.7500 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 50e4a65c0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.7692 | plus=0.7692 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 33f51d7e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 435f65fa0baf11ebab90acde48001122 | f1=0.0000 | gold=0.8000 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 076288460bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 265daf200bdc11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 037da85d08c611ebbd90ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 0ce9a92008ed11ebbda7ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 006d81bc0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 01309e5008bd11ebbd89ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- OTHER:
  - 0083038e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.5000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 551e024408a611ebbd7fac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### 2wikimultihopqa / balanced_384_8 / intent_p1
- PHASE1_NOT_FOUND:
  - 2dc690ba0bdc11eba7f7acde48001122 | f1=0.0000 | gold=0.7500 | plus=0.7500 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 50e4a65c0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.7692 | plus=0.7692 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 33f51d7e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 435f65fa0baf11ebab90acde48001122 | f1=0.0000 | gold=0.8000 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 076288460bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 265daf200bdc11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 037da85d08c611ebbd90ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 0ce9a92008ed11ebbda7ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 006d81bc0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 01309e5008bd11ebbd89ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- OTHER:
  - 0083038e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.5000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 551e024408a611ebbd7fac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### 2wikimultihopqa / balanced_384_8 / intent_p1_aq
- PHASE1_NOT_FOUND:
  - 2dc690ba0bdc11eba7f7acde48001122 | f1=0.0000 | gold=0.7500 | plus=0.7500 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 50e4a65c0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.7692 | plus=0.7692 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 33f51d7e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 435f65fa0baf11ebab90acde48001122 | f1=0.0000 | gold=0.8000 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 076288460bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 265daf200bdc11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 037da85d08c611ebbd90ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 1dfaa6200bdd11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 006d81bc0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 01309e5008bd11ebbd89ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- OTHER:
  - 0083038e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.5000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 551e024408a611ebbd7fac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### 2wikimultihopqa / balanced_384_8 / intent_p1_aq_rq
- PHASE1_NOT_FOUND:
  - 2dc690ba0bdc11eba7f7acde48001122 | f1=0.0000 | gold=0.7500 | plus=0.7500 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 50e4a65c0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.7692 | plus=0.7692 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 33f51d7e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 435f65fa0baf11ebab90acde48001122 | f1=0.0000 | gold=0.8000 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 076288460bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 265daf200bdc11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 037da85d08c611ebbd90ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 0dc9691b087911ebbd67ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 006d81bc0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 01309e5008bd11ebbd89ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- OTHER:
  - 0083038e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.5000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 551e024408a611ebbd7fac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### 2wikimultihopqa / legacy_512_10 / baseline_bq
- PHASE1_NOT_FOUND:
  - 2dc690ba0bdc11eba7f7acde48001122 | f1=0.0000 | gold=0.7500 | plus=0.7500 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 50e4a65c0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.7692 | plus=0.7692 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 33f51d7e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 435f65fa0baf11ebab90acde48001122 | f1=0.0000 | gold=0.8000 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 265daf200bdc11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
  - 3bb9c0740bb011ebab90acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 0ce9a92008ed11ebbda7ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 1dfaa6200bdd11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 006d81bc0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 01309e5008bd11ebbd89ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 0083038e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.5000 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 9fe6a6760baf11ebab90acde48001122 | f1=0.3333 | gold=1.0000 | plus=0.3333 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 3ce92df80bde11eba7f7acde48001122 | f1=0.0000 | gold=0.2857 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 551e024408a611ebbd7fac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### 2wikimultihopqa / legacy_512_10 / intent_p1
- PHASE1_NOT_FOUND:
  - 2dc690ba0bdc11eba7f7acde48001122 | f1=0.0000 | gold=0.7500 | plus=0.7500 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 50e4a65c0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.7692 | plus=0.7692 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 33f51d7e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 435f65fa0baf11ebab90acde48001122 | f1=0.0000 | gold=0.8000 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 076288460bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 265daf200bdc11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 0ce9a92008ed11ebbda7ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 1dfaa6200bdd11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 006d81bc0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 01309e5008bd11ebbd89ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 0083038e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.5000 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 9fe6a6760baf11ebab90acde48001122 | f1=0.3333 | gold=1.0000 | plus=0.3333 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 551e024408a611ebbd7fac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 62a9cd640bb011ebab90acde48001122 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### 2wikimultihopqa / legacy_512_10 / intent_p1_aq
- PHASE1_NOT_FOUND:
  - 2dc690ba0bdc11eba7f7acde48001122 | f1=0.0000 | gold=0.7500 | plus=0.7500 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 50e4a65c0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.7692 | plus=0.7692 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 33f51d7e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 435f65fa0baf11ebab90acde48001122 | f1=0.0000 | gold=0.8000 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 076288460bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 265daf200bdc11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 0ce9a92008ed11ebbda7ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 1dfaa6200bdd11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 006d81bc0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 01309e5008bd11ebbd89ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 9fe6a6760baf11ebab90acde48001122 | f1=0.3333 | gold=1.0000 | plus=0.3333 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 0083038e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.5000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 3ce92df80bde11eba7f7acde48001122 | f1=0.0000 | gold=0.2857 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### 2wikimultihopqa / legacy_512_10 / intent_p1_aq_rq
- PHASE1_NOT_FOUND:
  - 2dc690ba0bdc11eba7f7acde48001122 | f1=0.0000 | gold=0.7500 | plus=0.7500 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 50e4a65c0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.7692 | plus=0.7692 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 33f51d7e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 435f65fa0baf11ebab90acde48001122 | f1=0.0000 | gold=0.8000 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 076288460bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 265daf200bdc11eba7f7acde48001122 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 0ce9a92008ed11ebbda7ac1f6bf848b6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 1dfaa6200bdd11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 006d81bc0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 01309e5008bd11ebbd89ac1f6bf848b6 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- OTHER:
  - 0083038e0bde11eba7f7acde48001122 | f1=0.0000 | gold=0.5000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 3ce92df80bde11eba7f7acde48001122 | f1=0.0000 | gold=0.2857 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### hotpotqa / balanced_384_8 / baseline_bq
- PHASE1_NOT_FOUND:
  - 5a8a48ee55429930ff3c0d66 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 5ab3b42c5542992ade7c6e4f | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 5a7732dc55429972597f149b | f1=0.0000 | gold=0.5000 | plus=0.5000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 5a85c3225542992a431d1b95 | f1=0.0000 | gold=0.8571 | plus=0.4000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 5a7b1c0a55429931da12c9c9 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
  - 5a7b88b45542997c3ec97201 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
- SELECTED_NOT_SUFFICIENT:
  - 5a73987c55429978a71e9039 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 5a77aa095542995d83181260 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 5a717d4c5542994082a3e854 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 5a749af055429979e28829b7 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 5a72b2dc5542992359bc3173 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 5a79fc8b5542994f819ef114 | f1=0.5000 | gold=1.0000 | plus=0.5000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 5a7a3a945542996a35c17147 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 5a8f5b6f554299458435d5e7 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### hotpotqa / balanced_384_8 / intent_p1
- PHASE1_NOT_FOUND:
  - 5a8a48ee55429930ff3c0d66 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 5ab3b42c5542992ade7c6e4f | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 5a7732dc55429972597f149b | f1=0.0000 | gold=0.5000 | plus=0.5000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 5abcf14f55429959677d6b5b | f1=0.0000 | gold=0.5714 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 5a7b1c0a55429931da12c9c9 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
  - 5a7b88b45542997c3ec97201 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
- SELECTED_NOT_SUFFICIENT:
  - 5a73987c55429978a71e9039 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 5a77aa095542995d83181260 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 5a717d4c5542994082a3e854 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 5a749af055429979e28829b7 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 5a72b2dc5542992359bc3173 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 5a79fc8b5542994f819ef114 | f1=0.5000 | gold=1.0000 | plus=0.5000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 5a7a3a945542996a35c17147 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 5a7d3470554299452d57bb56 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### hotpotqa / balanced_384_8 / intent_p1_aq
- PHASE1_NOT_FOUND:
  - 5a8a48ee55429930ff3c0d66 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 5ab3b42c5542992ade7c6e4f | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 5a7732dc55429972597f149b | f1=0.0000 | gold=0.5000 | plus=0.5000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 5abcf14f55429959677d6b5b | f1=0.0000 | gold=0.5714 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 5a790d8f554299029c4b5eec | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 5a7b1c0a55429931da12c9c9 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 5a77aa095542995d83181260 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 5a83411655429966c78a6b5d | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 5a717d4c5542994082a3e854 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 5a73987c55429978a71e9039 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 5a72b2dc5542992359bc3173 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 5a79fc8b5542994f819ef114 | f1=0.5000 | gold=1.0000 | plus=0.5000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 5a7a3a945542996a35c17147 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 5a7d3470554299452d57bb56 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### hotpotqa / balanced_384_8 / intent_p1_aq_rq
- PHASE1_NOT_FOUND:
  - 5a8a48ee55429930ff3c0d66 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 5ab3b42c5542992ade7c6e4f | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 5a7732dc55429972597f149b | f1=0.0000 | gold=0.5000 | plus=0.5000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 5abcf14f55429959677d6b5b | f1=0.0000 | gold=0.5714 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 5a7b1c0a55429931da12c9c9 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
  - 5a7b88b45542997c3ec97201 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
- SELECTED_NOT_SUFFICIENT:
  - 5a73987c55429978a71e9039 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 5a77aa095542995d83181260 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 5a717d4c5542994082a3e854 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 5a749af055429979e28829b7 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 5a72b2dc5542992359bc3173 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 5a79fc8b5542994f819ef114 | f1=0.5000 | gold=1.0000 | plus=0.5000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 5a7a3a945542996a35c17147 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 5a7d3470554299452d57bb56 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### hotpotqa / legacy_512_10 / baseline_bq
- PHASE1_NOT_FOUND:
  - 5a8a48ee55429930ff3c0d66 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 5ab3b42c5542992ade7c6e4f | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 5a7732dc55429972597f149b | f1=0.0000 | gold=0.5000 | plus=0.5000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 5a85c3225542992a431d1b95 | f1=0.0000 | gold=0.8571 | plus=0.5000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 5a7b88b45542997c3ec97201 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 5a7f33115542992e7d278c84 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
- SELECTED_NOT_SUFFICIENT:
  - 5a77aa095542995d83181260 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 5a790d8f554299029c4b5eec | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 5a717d4c5542994082a3e854 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 5a73987c55429978a71e9039 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 5a72b2dc5542992359bc3173 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 5ab69f9a554299710c8d1ef8 | f1=0.0000 | gold=0.3636 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 5a7a3a945542996a35c17147 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 5a8f5b6f554299458435d5e7 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### hotpotqa / legacy_512_10 / intent_p1
- PHASE1_NOT_FOUND:
  - 5a8a48ee55429930ff3c0d66 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 5ab3b42c5542992ade7c6e4f | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 5a7732dc55429972597f149b | f1=0.0000 | gold=0.5000 | plus=0.5000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 5abcf14f55429959677d6b5b | f1=0.0000 | gold=0.5714 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 5a7b88b45542997c3ec97201 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 5a7f33115542992e7d278c84 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
- SELECTED_NOT_SUFFICIENT:
  - 5a77aa095542995d83181260 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 5a790d8f554299029c4b5eec | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 5a717d4c5542994082a3e854 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 5a73987c55429978a71e9039 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 5ab69f9a554299710c8d1ef8 | f1=0.0000 | gold=0.3636 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 5a79fc8b5542994f819ef114 | f1=0.5000 | gold=1.0000 | plus=0.5000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 5a7a3a945542996a35c17147 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 5a88a1bb5542997e5c09a64c | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### hotpotqa / legacy_512_10 / intent_p1_aq
- PHASE1_NOT_FOUND:
  - 5a8a48ee55429930ff3c0d66 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
  - 5ab3b42c5542992ade7c6e4f | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- PRUNING_FAILED:
  - 5a7732dc55429972597f149b | f1=0.0000 | gold=0.5000 | plus=0.5000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
  - 5abcf14f55429959677d6b5b | f1=0.0000 | gold=0.5714 | plus=0.0000 | subtype=UNKNOWN | evidence=gold present in phase1 but absent by phase2/feature-ready
- FINAL_SELECTION_FAILED:
  - 5a790d8f554299029c4b5eec | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=MARGINAL_GAIN_OR_RANKING | evidence=gold feature-ready but not selected
  - 5a7b1c0a55429931da12c9c9 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 5a77aa095542995d83181260 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 5a77cb335542997042120b3a | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 5a717d4c5542994082a3e854 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 5a73987c55429978a71e9039 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 5ab69f9a554299710c8d1ef8 | f1=0.0000 | gold=0.3636 | plus=0.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 5abff9905542997d64295980 | f1=0.6667 | gold=0.0833 | plus=0.6667 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 5a7a3a945542996a35c17147 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 5a88a1bb5542997e5c09a64c | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
### hotpotqa / legacy_512_10 / intent_p1_aq_rq
- PHASE1_NOT_FOUND:
  - 5a717d4c5542994082a3e854 | f1=0.0000 | gold=0.0000 | plus=0.8000 | subtype=PHASE1_OTHER_SOURCE_MISS | evidence=gold absent from phase1 candidates
  - 5a77322055429972597f1494 | f1=0.0000 | gold=0.0000 | plus=1.0000 | subtype=NO_SEMANTIC_OR_ENTITY_TITLE_GOLD | evidence=gold absent from phase1 candidates; semantic/entity source could not recover gold
- FINAL_SELECTION_FAILED:
  - 5a85c3225542992a431d1b95 | f1=0.0000 | gold=0.8571 | plus=0.4000 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
  - 5a8835dc5542994846c1ce2b | f1=0.0000 | gold=0.0000 | plus=0.8571 | subtype=ATOM_CAP_SATURATED | evidence=gold feature-ready but not selected; selected atoms saturated and selected_plus_gold gain positive
- SELECTED_NOT_SUFFICIENT:
  - 5a7b1c0a55429931da12c9c9 | f1=0.0000 | gold=0.6667 | plus=0.6667 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
  - 5a84574455429933447460e6 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=PARTIAL_CONTEXT_MISSING_KEY_SUPPORT | evidence=selected_plus_gold improves over selected context
- QA_PROMPT_FAILED_WITH_GOLD_CONTEXT:
  - 5a749af055429979e28829b7 | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
  - 5a7542425542996c70cfaecc | f1=0.0000 | gold=0.0000 | plus=0.0000 | subtype=GOLD_CONTEXT_STILL_FAILS | evidence=gold support context f1 <= 0
- QA_PROMPT_FAILED_WITH_SUFFICIENT_CONTEXT:
  - 5a714dea5542994082a3e7a9 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
  - 5a8481945542997175ce1ed3 | f1=0.0000 | gold=1.0000 | plus=1.0000 | subtype=RENDERED_FULL_BUT_QA_WRONG | evidence=rendered full coverage but QA mismatch
- OTHER:
  - 5a7a3a945542996a35c17147 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal
  - 5a8f5b6f554299458435d5e7 | f1=0.0000 | gold=1.0000 | plus=0.0000 | subtype=UNRESOLVED | evidence=no dominant failure signal