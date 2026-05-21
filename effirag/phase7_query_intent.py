from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from .utils import content_tokens


RELATION_CUE_LEXICON: Sequence[str] = (
    "father",
    "mother",
    "spouse",
    "wife",
    "husband",
    "son",
    "daughter",
    "child",
    "born",
    "birth",
    "date of birth",
    "place of birth",
    "death",
    "place of death",
    "nationality",
    "country",
    "same nationality",
    "director",
    "performer",
    "composer",
    "author",
    "located in",
    "capital",
)


@dataclass
class QueryIntentGraph:
    target_entities: List[str]
    relation_cues: List[str]
    answer_type_cues: List[str]
    comparison_markers: List[str]
    constraint_slots: List[str]
    intent_scores: Dict[str, float]


def _normalize_token(text: str) -> str:
    return str(text or "").strip().lower()


def _dedup_sorted(values: Iterable[str]) -> List[str]:
    return sorted({str(v or "").strip() for v in list(values or []) if str(v or "").strip()})


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _detect_answer_type_cues(question: str) -> List[str]:
    q = _normalize_token(question)
    out: List[str] = []
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ", "can ")):
        out.append("yes_no")
    if q.startswith("who "):
        out.append("person")
    if q.startswith("where "):
        out.append("place")
    if q.startswith("when "):
        out.append("date")
    if " which country" in f" {q}" or q.startswith("which country"):
        out.append("country")
    if " what year" in f" {q}" or q.startswith("what year"):
        out.append("year")
    if " how many" in f" {q}" or " number of " in f" {q}":
        out.append("number")
    if " organization " in f" {q}" or " company " in f" {q}":
        out.append("organization")
    if " title " in f" {q}" or q.startswith("which title"):
        out.append("title")
    if not out:
        out.append("unknown")
    return _dedup_sorted(out)


def _detect_relation_cues(question: str) -> List[str]:
    q = _normalize_token(question)
    found: List[str] = []
    for cue in RELATION_CUE_LEXICON:
        c = _normalize_token(cue)
        if _contains_phrase(q, c):
            found.append(c)
    return _dedup_sorted(found)


def _detect_comparison_markers(question: str) -> List[str]:
    q = _normalize_token(question)
    markers = []
    for m in ("same", "both", "compare", "difference", "older", "younger", "more", "less"):
        if _contains_phrase(q, m):
            markers.append(m)
    return _dedup_sorted(markers)


def _intent_scores(question: str, relation_cues: Sequence[str], comparison_markers: Sequence[str]) -> Dict[str, float]:
    q = _normalize_token(question)
    comparison_score = 0.0
    if comparison_markers:
        comparison_score = min(1.0, 0.25 + 0.15 * float(len(comparison_markers)))
    bridge_score = 0.0
    if " and " in f" {q} " or " both " in f" {q} ":
        bridge_score += 0.35
    if relation_cues:
        bridge_score += min(0.45, 0.15 * float(len(relation_cues)))
    bridge_score = min(1.0, bridge_score)
    attribute_lookup_score = min(1.0, 0.2 * float(len(relation_cues)))
    temporal_score = 0.0
    if any(x in q for x in ("when", "year", "date", "born", "death")):
        temporal_score = 0.8
    return {
        "comparison_score": float(comparison_score),
        "bridge_score": float(bridge_score),
        "attribute_lookup_score": float(attribute_lookup_score),
        "temporal_score": float(temporal_score),
    }


def extract_query_intent_graph(
    *,
    question: str,
    query_entities: Sequence[str],
    candidate_titles: Sequence[str] | None = None,
) -> QueryIntentGraph:
    entities = _dedup_sorted([_normalize_token(x) for x in list(query_entities or []) if _normalize_token(x)])
    relation_cues = _detect_relation_cues(str(question or ""))
    answer_type_cues = _detect_answer_type_cues(str(question or ""))
    comparison_markers = _detect_comparison_markers(str(question or ""))
    slots: List[str] = []
    slots.extend([f"entity:{x}" for x in entities])
    slots.extend([f"relation:{x}" for x in relation_cues])
    slots.extend([f"answer_type:{x}" for x in answer_type_cues])
    slots.extend([f"cmp:{x}" for x in comparison_markers])
    if candidate_titles:
        title_tokens = [_normalize_token(x) for x in list(candidate_titles or [])]
        for tok in title_tokens:
            if tok and tok in entities:
                slots.append(f"title_entity:{tok}")
    slots = _dedup_sorted(slots)
    scores = _intent_scores(str(question or ""), relation_cues, comparison_markers)
    return QueryIntentGraph(
        target_entities=entities,
        relation_cues=relation_cues,
        answer_type_cues=answer_type_cues,
        comparison_markers=comparison_markers,
        constraint_slots=slots,
        intent_scores=scores,
    )


def sentence_relation_cue_coverage(text: str, relation_cues: Sequence[str]) -> float:
    cues = [c for c in list(relation_cues or []) if str(c or "").strip()]
    if not cues:
        return 0.0
    low = _normalize_token(text)
    hit = sum(1 for cue in cues if _contains_phrase(low, _normalize_token(cue)))
    return float(hit / float(max(1, len(cues))))


def sentence_entity_coverage(*, title: str, text: str, target_entities: Sequence[str]) -> float:
    ents = [e for e in list(target_entities or []) if str(e or "").strip()]
    if not ents:
        return 0.0
    low_title = _normalize_token(title)
    tok = set(content_tokens(text)).union(set(content_tokens(title)))
    hit = 0
    for ent in ents:
        e = _normalize_token(ent)
        if (e in tok) or _contains_phrase(low_title, e):
            hit += 1
    return float(hit / float(max(1, len(ents))))


def sentence_answer_type_compatibility(*, text: str, answer_type_cues: Sequence[str]) -> float:
    cues = {str(c or "").strip().lower() for c in list(answer_type_cues or []) if str(c or "").strip()}
    if not cues or cues == {"unknown"}:
        return 0.0
    t = str(text or "")
    tl = t.lower()
    score = 0.0
    if "yes_no" in cues:
        # yes/no questions often require declarative relational statements.
        score = max(score, 0.5 if any(x in tl for x in (" is ", " was ", " are ", " were ")) else 0.0)
    if "country" in cues:
        score = max(score, 1.0 if any(x in tl for x in (" country", "nation", "republic")) else 0.0)
    if "date" in cues or "year" in cues:
        has_digit = any(ch.isdigit() for ch in t)
        score = max(score, 0.9 if has_digit else 0.0)
    if "place" in cues:
        score = max(score, 0.8 if any(x in tl for x in (" city", "town", "located", "province", "state")) else 0.0)
    if "person" in cues:
        # Weak heuristic: person-like sentences tend to start with named entities.
        score = max(score, 0.6 if (len(t.split()) > 1 and t[:1].isupper()) else 0.0)
    if "organization" in cues:
        score = max(score, 0.8 if any(x in tl for x in ("company", "organization", "university", "inc.")) else 0.0)
    if "number" in cues:
        score = max(score, 0.8 if any(ch.isdigit() for ch in t) else 0.0)
    if "title" in cues:
        score = max(score, 0.6 if any(x in tl for x in ("\"","''","film","novel","album","song")) else 0.0)
    return float(max(0.0, min(1.0, score)))


def build_intent_slots_for_candidate(
    *,
    entity_cov: float,
    relation_cov: float,
    answer_type_compat: float,
) -> Set[str]:
    slots: Set[str] = set()
    if float(entity_cov) > 0.0:
        slots.add("entity")
    if float(relation_cov) > 0.0:
        slots.add("relation")
    if float(answer_type_compat) > 0.0:
        slots.add("answer_type")
    return slots

