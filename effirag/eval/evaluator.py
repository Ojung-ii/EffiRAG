from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .metrics_hipporag2_parity import qa_exact_match_parity, qa_f1_parity
from .metrics_legacy import qa_exact_match_legacy, qa_f1_legacy


VALID_EVALUATOR_MODES = ("legacy", "hipporag2_parity")


def _to_answer_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        out = [str(x).strip() for x in value if str(x).strip()]
        return out
    text = str(value).strip()
    return [text] if text else []


def extract_gold_answers(sample) -> List[str]:
    answers: List[str] = []
    if sample is None:
        return answers

    answers.extend(_to_answer_list(getattr(sample, "answer", "")))
    metadata = getattr(sample, "metadata", {}) or {}
    for key in ("possible_answers", "answers", "answer_aliases", "o_aliases"):
        answers.extend(_to_answer_list(metadata.get(key)))

    seen = set()
    deduped = []
    for ans in answers:
        if ans in seen:
            continue
        seen.add(ans)
        deduped.append(ans)
    return deduped


@dataclass
class QAEvalResult:
    em: float
    f1: float
    gold_answers: List[str]
    evaluator_mode: str


class QAEvaluator:
    def __init__(self, mode: str = "legacy"):
        mode_norm = str(mode or "legacy").strip().lower()
        if mode_norm not in VALID_EVALUATOR_MODES:
            raise ValueError(f"Unsupported evaluator mode: {mode}. Expected one of {VALID_EVALUATOR_MODES}")
        self.mode = mode_norm

    def evaluate(self, prediction: str, sample) -> QAEvalResult:
        gold_answers = extract_gold_answers(sample)
        if not gold_answers:
            gold_answers = [""]

        if self.mode == "legacy":
            em = qa_exact_match_legacy(prediction, gold_answers[0])
            f1 = qa_f1_legacy(prediction, gold_answers[0])
        else:
            em = qa_exact_match_parity(prediction, gold_answers)
            f1 = qa_f1_parity(prediction, gold_answers)
        return QAEvalResult(
            em=float(em),
            f1=float(f1),
            gold_answers=list(gold_answers),
            evaluator_mode=self.mode,
        )

    def evaluate_batch(self, predictions: Sequence[str], gold_answers_batch: Sequence[Sequence[str]]) -> Dict[str, float]:
        if len(predictions) != len(gold_answers_batch):
            raise ValueError("predictions and gold_answers_batch must have same length")
        if not predictions:
            return {"ExactMatch": 0.0, "F1": 0.0, "count": 0.0}

        total_em = 0.0
        total_f1 = 0.0
        for pred, gold_list in zip(predictions, gold_answers_batch):
            if self.mode == "legacy":
                first_gold = list(gold_list)[0] if gold_list else ""
                total_em += qa_exact_match_legacy(pred, first_gold)
                total_f1 += qa_f1_legacy(pred, first_gold)
            else:
                total_em += qa_exact_match_parity(pred, gold_list)
                total_f1 += qa_f1_parity(pred, gold_list)
        n = float(len(predictions))
        return {
            "ExactMatch": float(total_em / n),
            "F1": float(total_f1 / n),
            "count": n,
        }

