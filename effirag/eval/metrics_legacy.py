from __future__ import annotations

import re
import string


def normalize_answer_legacy(text: str) -> str:
    value = str(text or "")
    value = value.lower()
    value = " ".join(value.split())
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = " ".join(value.split())
    return value


def qa_exact_match_legacy(prediction: str, gold_answer: str) -> float:
    return float(normalize_answer_legacy(prediction) == normalize_answer_legacy(gold_answer))


def qa_f1_legacy(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_answer_legacy(prediction).split()
    gold_tokens = normalize_answer_legacy(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for tok in pred_tokens:
        common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)
