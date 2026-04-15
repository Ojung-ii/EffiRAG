from __future__ import annotations

from .eval.metrics_legacy import (
    normalize_answer_legacy as _normalize_answer,
    qa_exact_match_legacy,
    qa_f1_legacy,
)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(qa_exact_match_legacy(prediction, ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    return float(qa_f1_legacy(prediction, ground_truth))
