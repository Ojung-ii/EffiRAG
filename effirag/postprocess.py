from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .eval.metrics_hipporag2_parity import normalize_answer_parity
from .utils import content_tokens

MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)
DATE_RE = re.compile(
    rf"\b(?:{MONTH_PATTERN}\s+\d{{1,2}}(?:,\s*\d{{4}})?|{MONTH_PATTERN}\s+\d{{4}}|\d{{4}})\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d[\d,./-]*\b")
QUOTE_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")
CAP_SEQ_RE = re.compile(
    r"\b(?:[A-Z][\w'`-]*)(?:\s+(?:[A-Z][\w'`-]*|of|the|and|for|in|on|at|de|la)){0,6}\b"
)


@dataclass
class EvidenceCandidate:
    span: str
    sentence_idx: int
    score: float
    type_match: bool


def detect_answer_type(question: str) -> str:
    q = str(question or "").strip().lower()
    if not q:
        return "entity"
    if q.startswith(
        ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ")
    ):
        return "yesno"
    if q.startswith("when ") or " what year" in q or " date " in q:
        return "date"
    if q.startswith("who "):
        return "person"
    if q.startswith("where ") or " which country" in q or " which city" in q or "location" in q:
        return "location"
    if "how many" in q or "how much" in q or "number of" in q:
        return "number"
    return "entity"


def _clean_span(text: str) -> str:
    s = " ".join(str(text or "").strip().split())
    return s.strip(" .,;:!?()[]{}\"'")


def _type_match(span: str, answer_type: str) -> bool:
    s = str(span or "").strip()
    if not s:
        return False
    if answer_type == "yesno":
        return s.lower() in {"yes", "no"}
    if answer_type == "date":
        return bool(DATE_RE.search(s))
    if answer_type == "number":
        return bool(NUMBER_RE.search(s))
    if answer_type in {"person", "location", "entity"}:
        return bool(CAP_SEQ_RE.search(s))
    return True


def _prediction_supported(prediction: str, sentences: Sequence[str]) -> bool:
    pred = _clean_span(prediction)
    if not pred:
        return False
    pred_norm = normalize_answer_parity(pred)
    pred_tokens = set(content_tokens(pred))
    if not pred_tokens:
        return False
    for sent in list(sentences or []):
        text = str(sent or "")
        text_norm = normalize_answer_parity(text)
        if pred_norm and pred_norm in text_norm:
            return True
        text_tokens = set(content_tokens(text))
        if not text_tokens:
            continue
        overlap = len(pred_tokens.intersection(text_tokens))
        if overlap >= max(1, int(round(0.8 * len(pred_tokens)))):
            return True
    return False


def _extract_candidates(
    question: str,
    prediction: str,
    sentences: Sequence[str],
    answer_type: str,
) -> List[EvidenceCandidate]:
    q_tokens = set(content_tokens(question))
    p_tokens = set(content_tokens(prediction))
    out: List[EvidenceCandidate] = []
    seen = set()

    for idx, sent in enumerate(list(sentences or []), start=1):
        text = str(sent or "")
        spans = []

        if answer_type == "date":
            spans.extend([m.group(0) for m in DATE_RE.finditer(text)])
        if answer_type == "number":
            spans.extend([m.group(0) for m in NUMBER_RE.finditer(text)])
        if answer_type in {"person", "location", "entity"}:
            spans.extend([m.group(0) for m in CAP_SEQ_RE.finditer(text)])
        for m in QUOTE_RE.finditer(text):
            for g in m.groups():
                if g:
                    spans.append(g)

        spans.append(text)

        for raw in spans:
            span = _clean_span(raw)
            if not span:
                continue
            key = normalize_answer_parity(span)
            if not key or key in seen:
                continue
            seen.add(key)

            span_tokens = set(content_tokens(span))
            if not span_tokens and answer_type not in {"date", "number"}:
                continue

            type_match = _type_match(span, answer_type)
            pred_overlap = 0.0
            if p_tokens and span_tokens:
                pred_overlap = float(len(p_tokens.intersection(span_tokens)) / max(1, len(p_tokens)))
            question_overlap = 0.0
            if q_tokens and span_tokens:
                question_overlap = float(len(q_tokens.intersection(span_tokens)) / max(1, len(span_tokens)))

            early_bonus = 1.0 / float(idx)
            score = 0.0
            score += 0.45 * (1.0 if type_match else 0.0)
            score += 0.30 * pred_overlap
            score += 0.20 * early_bonus
            score += 0.05 * question_overlap
            out.append(EvidenceCandidate(span=span, sentence_idx=int(idx), score=float(score), type_match=bool(type_match)))

    out.sort(key=lambda c: (c.score, -c.sentence_idx, len(c.span)), reverse=True)
    return out


def _pick_best_candidate(cands: Sequence[EvidenceCandidate], require_type_match: bool = False) -> str:
    for cand in list(cands or []):
        if require_type_match and not cand.type_match:
            continue
        if len(cand.span) > 160:
            continue
        return str(cand.span)
    return ""


def _normalize_to_evidence(prediction: str, cands: Sequence[EvidenceCandidate], answer_type: str) -> str:
    pred = _clean_span(prediction)
    if not pred:
        return pred
    pred_norm = normalize_answer_parity(pred)
    best = pred

    for cand in list(cands or []):
        cand_norm = normalize_answer_parity(cand.span)
        if not cand_norm:
            continue
        if pred_norm and pred_norm in cand_norm and len(cand.span) > len(best) and len(cand.span) <= 80:
            best = cand.span

    if answer_type == "date":
        year = re.search(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b", pred)
        if year:
            y = year.group(1)
            for cand in list(cands or []):
                if y in cand.span and len(cand.span) >= len(best):
                    best = cand.span
                    break
    return _clean_span(best)


def apply_answer_realization_variant(
    *,
    prediction: str,
    question: str,
    sentences: Sequence[str],
    variant: str,
) -> Dict[str, object]:
    initial = _clean_span(prediction)
    answer_type = detect_answer_type(question)
    final = str(initial)
    cands = _extract_candidates(question=question, prediction=initial, sentences=sentences, answer_type=answer_type)

    if variant == "answer_normalization_light":
        final = _normalize_to_evidence(initial, cands, answer_type)
    elif variant == "answer_verification_light":
        if not _prediction_supported(initial, sentences):
            replacement = _pick_best_candidate(cands, require_type_match=False)
            if replacement:
                final = replacement
        final = _normalize_to_evidence(final, cands, answer_type)
    elif variant == "answer_type_aware_extraction":
        if answer_type != "yesno":
            replacement = _pick_best_candidate(cands, require_type_match=True) or _pick_best_candidate(cands)
            if replacement:
                final = replacement
        final = _normalize_to_evidence(final, cands, answer_type)

    final = _clean_span(final)
    if answer_type == "yesno":
        low = final.lower()
        if low.startswith("yes"):
            final = "yes"
        elif low.startswith("no"):
            final = "no"

    evidence_supported = _prediction_supported(final, sentences)
    type_match = _type_match(final, answer_type) if answer_type != "yesno" else (final in {"yes", "no"})

    return {
        "initial_prediction": initial,
        "final_prediction": final,
        "qa_utilization_variant": str(variant),
        "qa_utilization_applied": bool(True),
        "qa_utilization_changed": bool(final != initial),
        "qa_utilization_candidate_count": int(len(cands)),
        "evidence_supported_answer": bool(evidence_supported),
        "answer_type": str(answer_type),
        "answer_type_match": bool(type_match),
    }
