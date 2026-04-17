from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List


def normalize_answer_parity(answer: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(answer or "")))))


def _ensure_answer_list(gold_answers) -> List[str]:
    if gold_answers is None:
        return [""]
    if isinstance(gold_answers, str):
        return [gold_answers]
    if isinstance(gold_answers, (list, tuple, set)):
        out = [str(x) for x in gold_answers]
        return out if out else [""]
    return [str(gold_answers)]


def qa_exact_match_parity(prediction: str, gold_answers) -> float:
    pred = normalize_answer_parity(prediction)
    best = 0.0
    for gold in _ensure_answer_list(gold_answers):
        if normalize_answer_parity(gold) == pred:
            best = 1.0
            break
    return float(best)


def qa_f1_single_parity(prediction: str, gold: str) -> float:
    predicted_tokens = normalize_answer_parity(prediction).split()
    gold_tokens = normalize_answer_parity(gold).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2.0 * (precision * recall) / (precision + recall)


def qa_f1_parity(prediction: str, gold_answers) -> float:
    best = 0.0
    for gold in _ensure_answer_list(gold_answers):
        score = qa_f1_single_parity(prediction, gold)
        if score > best:
            best = score
    return float(best)

