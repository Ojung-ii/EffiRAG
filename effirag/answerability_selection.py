from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .utils import content_tokens, safe_div


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
DATE_TOKEN_RE = re.compile(
    r"\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    flags=re.IGNORECASE,
)
NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass
class EvidenceAtom:
    sentence_id: str
    source_id: str
    text: str
    token_count: int
    token_set: set[str]
    query_overlap: float
    entity_overlap: float
    relation_overlap: float
    answer_type_compat: float
    bridge_overlap: float
    structure_anchor: float
    structure_bridge: float
    structure_prior: float
    corridor_ids: Tuple[str, ...]
    atom_score: float


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _token_count(text: str) -> int:
    return int(len(list(content_tokens(str(text or "")))))


def _split_sentences(text: str) -> List[str]:
    raw = _normalize_space(text)
    if not raw:
        return []
    parts = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(raw) if piece.strip()]
    return parts or [raw]


def _source_id(sentence_id: str) -> str:
    sid = str(sentence_id or "").strip()
    if "::" not in sid:
        return sid
    return sid.split("::", 1)[0]


def _extract_entity_tokens(text: str) -> set[str]:
    raw = str(text or "")
    entities = {str(tok).lower() for tok in CAPITALIZED_TOKEN_RE.findall(raw)}
    return {tok for tok in entities if len(tok) >= 3}


def _question_relation_tokens(question_tokens: set[str]) -> set[str]:
    relation_keywords = {
        "born",
        "founded",
        "located",
        "capital",
        "president",
        "director",
        "wrote",
        "won",
        "married",
        "member",
        "country",
        "city",
        "state",
        "river",
        "who",
        "where",
        "when",
        "which",
        "what",
        "how",
    }
    out = set()
    for tok in question_tokens:
        if tok in relation_keywords:
            out.add(tok)
    return out


def _expected_answer_type(question_text: str) -> str:
    question = str(question_text or "").strip().lower()
    if question.startswith("when"):
        return "date"
    if question.startswith("who"):
        return "person"
    if question.startswith("where"):
        return "location"
    if question.startswith("how many") or question.startswith("how much"):
        return "number"
    return "generic"


def _answer_type_compatibility(text: str, expected_type: str) -> float:
    body = str(text or "")
    if expected_type == "date":
        return 1.0 if DATE_TOKEN_RE.search(body) else 0.0
    if expected_type == "number":
        return 1.0 if NUMBER_TOKEN_RE.search(body) else 0.0
    if expected_type in {"person", "location"}:
        capitals = CAPITALIZED_TOKEN_RE.findall(body)
        return 1.0 if len(capitals) >= 1 else 0.0
    return 0.5


def _build_bridge_tokens(
    *,
    sentence_texts: Sequence[str],
    query_tokens: set[str],
    question_entities: set[str],
) -> set[str]:
    token_counts: Counter[str] = Counter()
    for text in sentence_texts:
        toks = {
            str(tok)
            for tok in content_tokens(str(text or ""))
            if str(tok) and len(str(tok)) >= 4 and str(tok) not in query_tokens and str(tok) not in question_entities
        }
        for tok in toks:
            token_counts[tok] += 1
    return {tok for tok, count in token_counts.items() if int(count) >= 2}


def _atomize_selected_evidence(
    *,
    sentence_ids: Sequence[str],
    sentence_texts: Sequence[str],
    question_tokens: set[str],
    question_entities: set[str],
    relation_tokens: set[str],
    bridge_tokens: set[str],
    expected_answer_type: str,
    sentence_feature_table: Mapping[str, Mapping[str, Any]] | None,
    max_span_sentences: int,
    length_penalty_weight: float,
) -> List[EvidenceAtom]:
    feature_table = dict(sentence_feature_table or {})
    atoms: List[EvidenceAtom] = []
    span_cap = max(1, int(max_span_sentences))

    for sid, raw_text in zip(sentence_ids, sentence_texts):
        sid_norm = str(sid or "").strip()
        text_norm = _normalize_space(str(raw_text or ""))
        if not sid_norm or not text_norm:
            continue

        sentences = _split_sentences(text_norm)
        span_candidates: List[str] = []
        for start in range(len(sentences)):
            for end in range(start, min(len(sentences), start + span_cap)):
                span_text = _normalize_space(" ".join(sentences[start : end + 1]))
                if span_text:
                    span_candidates.append(span_text)
        if not span_candidates:
            span_candidates = [text_norm]

        best_span = span_candidates[0]
        best_score = -math.inf
        best_feats = {
            "query_overlap": 0.0,
            "entity_overlap": 0.0,
            "relation_overlap": 0.0,
            "answer_type_compat": 0.0,
            "bridge_overlap": 0.0,
            "token_set": set(),
            "token_count": 0,
        }
        for span_text in span_candidates:
            span_tokens = {str(tok) for tok in content_tokens(span_text)}
            query_overlap = float(len(span_tokens.intersection(question_tokens)))
            entity_overlap = float(len(span_tokens.intersection(question_entities)))
            relation_overlap = float(len(span_tokens.intersection(relation_tokens)))
            answer_type_compat = float(_answer_type_compatibility(span_text, expected_answer_type))
            bridge_overlap = float(len(span_tokens.intersection(bridge_tokens)))
            token_count = int(max(1, _token_count(span_text)))
            length_penalty = float(length_penalty_weight * float(max(0, token_count - 18)))
            score = float(
                (1.8 * query_overlap)
                + (1.2 * entity_overlap)
                + (1.0 * relation_overlap)
                + (0.8 * answer_type_compat)
                + (0.8 * bridge_overlap)
                - length_penalty
            )
            if score > best_score:
                best_score = float(score)
                best_span = span_text
                best_feats = {
                    "query_overlap": float(query_overlap),
                    "entity_overlap": float(entity_overlap),
                    "relation_overlap": float(relation_overlap),
                    "answer_type_compat": float(answer_type_compat),
                    "bridge_overlap": float(bridge_overlap),
                    "token_set": set(span_tokens),
                    "token_count": int(token_count),
                }

        row = dict(feature_table.get(sid_norm, {}) or {})
        structure_anchor = 1.0 if bool(row.get("is_main_candidate", False)) else (0.4 if best_feats["query_overlap"] > 0.0 else 0.0)
        structure_bridge = 1.0 if bool(row.get("is_connector_adjacent", False)) else (0.7 if bool(row.get("is_support_candidate", False)) else 0.0)
        locality = float(max(0.0, min(1.0, _safe_float(row.get("locality_score", 0.0), 0.0))))
        structure_prior = float((0.5 * structure_anchor) + (0.5 * structure_bridge) + (0.25 * locality))

        corridor_ids = tuple(_ordered_unique(list(row.get("corridor_ids", []) or [])))
        atoms.append(
            EvidenceAtom(
                sentence_id=sid_norm,
                source_id=_source_id(sid_norm),
                text=str(best_span),
                token_count=int(best_feats["token_count"]),
                token_set=set(best_feats["token_set"]),
                query_overlap=float(best_feats["query_overlap"]),
                entity_overlap=float(best_feats["entity_overlap"]),
                relation_overlap=float(best_feats["relation_overlap"]),
                answer_type_compat=float(best_feats["answer_type_compat"]),
                bridge_overlap=float(best_feats["bridge_overlap"]),
                structure_anchor=float(structure_anchor),
                structure_bridge=float(structure_bridge),
                structure_prior=float(structure_prior),
                corridor_ids=tuple(corridor_ids),
                atom_score=float(best_score + (0.35 * structure_prior)),
            )
        )
    return atoms


def _pairwise_jaccard_mean(token_sets: Sequence[set[str]]) -> float:
    n = len(token_sets)
    if n <= 1:
        return 0.0
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            union = token_sets[i].union(token_sets[j])
            if not union:
                continue
            sims.append(float(len(token_sets[i].intersection(token_sets[j])) / float(len(union))))
    if not sims:
        return 0.0
    return float(sum(sims) / float(len(sims)))


def _score_selected_set(
    *,
    atoms: Sequence[EvidenceAtom],
    selected_indices: Sequence[int],
    question_tokens: set[str],
    question_entities: set[str],
    relation_tokens: set[str],
    max_tokens: int,
    max_atoms: int,
    use_answerability: bool,
    use_structure: bool,
    use_noise_penalty: bool,
    use_cost_penalty: bool,
    structure_weight: float,
    noise_weight: float,
    cost_weight: float,
) -> Dict[str, float]:
    if not selected_indices:
        return {
            "A": 0.0,
            "S": 0.0,
            "N": 0.0,
            "C": 0.0,
            "score": 0.0,
            "query_coverage": 0.0,
            "entity_coverage": 0.0,
            "relation_coverage": 0.0,
            "answer_type_coverage": 0.0,
            "structure_coverage": 0.0,
            "redundancy_rate": 0.0,
            "selected_tokens": 0.0,
        }

    picked = [atoms[idx] for idx in selected_indices]
    union_tokens: set[str] = set()
    total_occ_tokens = 0
    source_counts: Counter[str] = Counter()
    for atom in picked:
        union_tokens.update(atom.token_set)
        total_occ_tokens += int(len(atom.token_set))
        source_counts[atom.source_id] += 1

    query_cov = float(safe_div(float(len(union_tokens.intersection(question_tokens))), float(max(1, len(question_tokens)))))
    entity_cov = float(safe_div(float(len(union_tokens.intersection(question_entities))), float(max(1, len(question_entities)))))
    relation_cov = float(safe_div(float(len(union_tokens.intersection(relation_tokens))), float(max(1, len(relation_tokens)))))
    answer_type_cov = float(max((atom.answer_type_compat for atom in picked), default=0.0))
    answerability = float((2.0 * query_cov) + (1.2 * entity_cov) + (0.9 * relation_cov) + (0.8 * answer_type_cov))
    if not use_answerability:
        answerability = 0.0

    all_corridors = set()
    selected_corridors = set()
    anchor_corridors = set()
    bridge_corridors = set()
    for atom in atoms:
        all_corridors.update(atom.corridor_ids)
    for atom in picked:
        selected_corridors.update(atom.corridor_ids)
        if atom.structure_anchor > 0.0:
            anchor_corridors.update(atom.corridor_ids)
        if atom.structure_bridge > 0.0:
            bridge_corridors.update(atom.corridor_ids)
    corridor_cov = float(safe_div(float(len(selected_corridors)), float(max(1, len(all_corridors)))))
    anchor_presence = float(max((atom.structure_anchor for atom in picked), default=0.0))
    bridge_presence = float(max((atom.structure_bridge for atom in picked), default=0.0))
    path_closure = 1.0 if anchor_corridors.intersection(bridge_corridors) else 0.0
    avg_structure_prior = float(safe_div(sum(float(atom.structure_prior) for atom in picked), float(max(1, len(picked)))))
    structure = float(
        (0.9 * corridor_cov)
        + (0.6 * anchor_presence)
        + (0.6 * bridge_presence)
        + (0.8 * path_closure)
        + (0.4 * avg_structure_prior)
    )
    if not use_structure:
        structure = 0.0

    source_repetition = float(
        safe_div(
            float(sum(max(0, int(count) - 1) for count in source_counts.values())),
            float(max(1, len(picked))),
        )
    )
    pair_overlap = float(_pairwise_jaccard_mean([atom.token_set for atom in picked]))
    token_dup = float(max(0.0, 1.0 - safe_div(float(len(union_tokens)), float(max(1, total_occ_tokens)))))
    noise = float((0.8 * pair_overlap) + (0.6 * source_repetition) + (0.6 * token_dup))
    if not use_noise_penalty:
        noise = 0.0

    selected_tokens = int(sum(int(atom.token_count) for atom in picked))
    norm_token_budget = max(1, int(max_tokens))
    norm_atom_budget = max(1, int(max_atoms))
    token_cost = float(safe_div(float(selected_tokens), float(norm_token_budget)))
    atom_cost = float(safe_div(float(len(picked)), float(norm_atom_budget)))
    cost = float((0.8 * token_cost) + (0.2 * atom_cost))
    if not use_cost_penalty:
        cost = 0.0

    score = float(answerability + (structure_weight * structure) - (noise_weight * noise) - (cost_weight * cost))
    return {
        "A": float(answerability),
        "S": float(structure),
        "N": float(noise),
        "C": float(cost),
        "score": float(score),
        "query_coverage": float(query_cov),
        "entity_coverage": float(entity_cov),
        "relation_coverage": float(relation_cov),
        "answer_type_coverage": float(answer_type_cov),
        "structure_coverage": float(corridor_cov),
        "redundancy_rate": float(pair_overlap),
        "selected_tokens": float(selected_tokens),
    }


def _order_selected_atoms(
    atoms: Sequence[EvidenceAtom],
    selected_indices: Sequence[int],
    ordering_enabled: bool,
) -> Tuple[List[int], str]:
    if not ordering_enabled:
        return list(selected_indices), "original"

    def _priority(atom: EvidenceAtom) -> Tuple[int, float, float]:
        focus = float(atom.query_overlap + atom.entity_overlap + atom.relation_overlap)
        bridge = float(max(atom.structure_bridge, atom.bridge_overlap))
        answer = float(atom.answer_type_compat)
        if focus > 0.0:
            group = 0
        elif bridge > 0.0:
            group = 1
        elif answer > 0.0:
            group = 2
        else:
            group = 3
        return (group, -float(atom.atom_score), float(atom.token_count))

    ordered = sorted(list(selected_indices), key=lambda idx: _priority(atoms[idx]))
    return ordered, "answerability"


def _select_greedy(
    *,
    atoms: Sequence[EvidenceAtom],
    question_tokens: set[str],
    question_entities: set[str],
    relation_tokens: set[str],
    max_atoms: int,
    max_tokens: int,
    hard_budget_enabled: bool,
    use_answerability: bool,
    use_structure: bool,
    use_noise_penalty: bool,
    use_cost_penalty: bool,
    structure_weight: float,
    noise_weight: float,
    cost_weight: float,
) -> Tuple[List[int], Dict[str, float]]:
    selected: List[int] = []
    current = _score_selected_set(
        atoms=atoms,
        selected_indices=selected,
        question_tokens=question_tokens,
        question_entities=question_entities,
        relation_tokens=relation_tokens,
        max_tokens=max_tokens,
        max_atoms=max_atoms,
        use_answerability=use_answerability,
        use_structure=use_structure,
        use_noise_penalty=use_noise_penalty,
        use_cost_penalty=use_cost_penalty,
        structure_weight=structure_weight,
        noise_weight=noise_weight,
        cost_weight=cost_weight,
    )
    while len(selected) < max_atoms:
        best_idx = None
        best_delta = -math.inf
        best_score = None
        for idx, atom in enumerate(atoms):
            if idx in selected:
                continue
            candidate_indices = list(selected) + [idx]
            candidate_tokens = sum(int(atoms[i].token_count) for i in candidate_indices)
            if hard_budget_enabled and selected and candidate_tokens > max_tokens:
                continue
            scored = _score_selected_set(
                atoms=atoms,
                selected_indices=candidate_indices,
                question_tokens=question_tokens,
                question_entities=question_entities,
                relation_tokens=relation_tokens,
                max_tokens=max_tokens,
                max_atoms=max_atoms,
                use_answerability=use_answerability,
                use_structure=use_structure,
                use_noise_penalty=use_noise_penalty,
                use_cost_penalty=use_cost_penalty,
                structure_weight=structure_weight,
                noise_weight=noise_weight,
                cost_weight=cost_weight,
            )
            delta = float(scored["score"] - current["score"])
            if (delta > best_delta) or (delta == best_delta and (best_score is None or scored["score"] > best_score["score"])):
                best_delta = float(delta)
                best_idx = int(idx)
                best_score = scored
        if best_idx is None or best_score is None:
            break
        if best_delta <= 0.0 and selected:
            break
        selected.append(best_idx)
        current = dict(best_score)
        if hard_budget_enabled and int(current.get("selected_tokens", 0.0)) >= max_tokens:
            break

    if not selected and atoms:
        singleton_scores: List[Tuple[float, int, Dict[str, float]]] = []
        for idx, atom in enumerate(atoms):
            if hard_budget_enabled and atom.token_count > max_tokens:
                continue
            scored = _score_selected_set(
                atoms=atoms,
                selected_indices=[idx],
                question_tokens=question_tokens,
                question_entities=question_entities,
                relation_tokens=relation_tokens,
                max_tokens=max_tokens,
                max_atoms=max_atoms,
                use_answerability=use_answerability,
                use_structure=use_structure,
                use_noise_penalty=use_noise_penalty,
                use_cost_penalty=use_cost_penalty,
                structure_weight=structure_weight,
                noise_weight=noise_weight,
                cost_weight=cost_weight,
            )
            singleton_scores.append((float(scored["score"]), int(idx), scored))
        if singleton_scores:
            singleton_scores.sort(key=lambda item: item[0], reverse=True)
            _, best_idx, current = singleton_scores[0]
            selected = [int(best_idx)]

    return list(selected), dict(current)


def _select_beam(
    *,
    atoms: Sequence[EvidenceAtom],
    question_tokens: set[str],
    question_entities: set[str],
    relation_tokens: set[str],
    max_atoms: int,
    max_tokens: int,
    beam_size: int,
    hard_budget_enabled: bool,
    use_answerability: bool,
    use_structure: bool,
    use_noise_penalty: bool,
    use_cost_penalty: bool,
    structure_weight: float,
    noise_weight: float,
    cost_weight: float,
) -> Tuple[List[int], Dict[str, float]]:
    beam_k = max(1, int(beam_size))
    beams: List[Tuple[Tuple[int, ...], Dict[str, float]]] = [
        (
            tuple(),
            _score_selected_set(
                atoms=atoms,
                selected_indices=[],
                question_tokens=question_tokens,
                question_entities=question_entities,
                relation_tokens=relation_tokens,
                max_tokens=max_tokens,
                max_atoms=max_atoms,
                use_answerability=use_answerability,
                use_structure=use_structure,
                use_noise_penalty=use_noise_penalty,
                use_cost_penalty=use_cost_penalty,
                structure_weight=structure_weight,
                noise_weight=noise_weight,
                cost_weight=cost_weight,
            ),
        )
    ]

    for _ in range(max_atoms):
        next_states: Dict[Tuple[int, ...], Dict[str, float]] = {state: score for state, score in beams}
        for state, _base_score in beams:
            used = set(state)
            for idx, atom in enumerate(atoms):
                if idx in used:
                    continue
                candidate = tuple(list(state) + [idx])
                candidate_tokens = sum(int(atoms[i].token_count) for i in candidate)
                if hard_budget_enabled and candidate_tokens > max_tokens:
                    continue
                scored = _score_selected_set(
                    atoms=atoms,
                    selected_indices=list(candidate),
                    question_tokens=question_tokens,
                    question_entities=question_entities,
                    relation_tokens=relation_tokens,
                    max_tokens=max_tokens,
                    max_atoms=max_atoms,
                    use_answerability=use_answerability,
                    use_structure=use_structure,
                    use_noise_penalty=use_noise_penalty,
                    use_cost_penalty=use_cost_penalty,
                    structure_weight=structure_weight,
                    noise_weight=noise_weight,
                    cost_weight=cost_weight,
                )
                prev = next_states.get(candidate)
                if prev is None or float(scored["score"]) > float(prev["score"]):
                    next_states[candidate] = scored
        ranked = sorted(
            next_states.items(),
            key=lambda item: (
                float(item[1].get("score", 0.0)),
                -float(item[1].get("selected_tokens", 0.0)),
                -float(len(item[0])),
            ),
            reverse=True,
        )
        beams = ranked[:beam_k]

    best_state = tuple()
    best_score = beams[0][1] if beams else {
        "A": 0.0,
        "S": 0.0,
        "N": 0.0,
        "C": 0.0,
        "score": 0.0,
        "query_coverage": 0.0,
        "entity_coverage": 0.0,
        "relation_coverage": 0.0,
        "answer_type_coverage": 0.0,
        "structure_coverage": 0.0,
        "redundancy_rate": 0.0,
        "selected_tokens": 0.0,
    }
    for state, scored in beams:
        if not state:
            continue
        if float(scored.get("score", -math.inf)) > float(best_score.get("score", -math.inf)):
            best_state = state
            best_score = scored
    if not best_state and atoms:
        fallback_indices, fallback_score = _select_greedy(
            atoms=atoms,
            question_tokens=question_tokens,
            question_entities=question_entities,
            relation_tokens=relation_tokens,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            hard_budget_enabled=hard_budget_enabled,
            use_answerability=use_answerability,
            use_structure=use_structure,
            use_noise_penalty=use_noise_penalty,
            use_cost_penalty=use_cost_penalty,
            structure_weight=structure_weight,
            noise_weight=noise_weight,
            cost_weight=cost_weight,
        )
        return list(fallback_indices), dict(fallback_score)
    return list(best_state), dict(best_score)


def apply_answerability_constrained_selection(
    *,
    question_text: str,
    selected_sentence_ids: Sequence[str],
    selected_sentences: Sequence[str],
    sentence_feature_table: Mapping[str, Mapping[str, Any]] | None,
    cfg: Any,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    enabled = bool(getattr(cfg, "answerability_selection_enabled", False))
    diag: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
    }
    base_ids = _ordered_unique(list(selected_sentence_ids or []))
    text_map = {str(sid): str(text or "") for sid, text in zip(selected_sentence_ids or [], selected_sentences or [])}
    base_texts = [str(text_map.get(sid, "") or "") for sid in base_ids]
    if (not enabled) or len(base_ids) <= 1:
        return base_ids, base_texts, diag

    question_tokens = {str(tok) for tok in content_tokens(str(question_text or ""))}
    question_entities = _extract_entity_tokens(str(question_text or ""))
    relation_tokens = _question_relation_tokens(question_tokens)
    expected_answer_type = _expected_answer_type(str(question_text or ""))
    bridge_tokens = _build_bridge_tokens(
        sentence_texts=base_texts,
        query_tokens=question_tokens,
        question_entities=question_entities,
    )

    max_span_sentences = max(1, _safe_int(getattr(cfg, "answerability_selection_atom_span_max_sentences", 2), 2))
    length_penalty_weight = max(
        0.0,
        _safe_float(getattr(cfg, "answerability_selection_length_penalty_weight", 0.04), 0.04),
    )
    atoms = _atomize_selected_evidence(
        sentence_ids=base_ids,
        sentence_texts=base_texts,
        question_tokens=question_tokens,
        question_entities=question_entities,
        relation_tokens=relation_tokens,
        bridge_tokens=bridge_tokens,
        expected_answer_type=expected_answer_type,
        sentence_feature_table=sentence_feature_table,
        max_span_sentences=max_span_sentences,
        length_penalty_weight=length_penalty_weight,
    )
    if not atoms:
        diag["reason"] = "empty_atom_pool"
        return base_ids, base_texts, diag

    max_atoms = max(1, _safe_int(getattr(cfg, "answerability_selection_max_atoms", 8), 8))
    max_tokens = max(1, _safe_int(getattr(cfg, "answerability_selection_max_tokens", 220), 220))
    beam_size = max(1, _safe_int(getattr(cfg, "answerability_selection_beam_size", 3), 3))
    mode = str(getattr(cfg, "answerability_selection_mode", "greedy") or "greedy").strip().lower()
    if mode not in {"greedy", "beam", "beam3"}:
        mode = "greedy"

    hard_budget_enabled = _safe_bool(getattr(cfg, "answerability_selection_hard_token_budget_enabled", True), True)
    use_answerability = _safe_bool(getattr(cfg, "answerability_selection_use_answerability", True), True)
    use_structure = _safe_bool(getattr(cfg, "answerability_selection_use_structure", True), True)
    use_noise_penalty = _safe_bool(getattr(cfg, "answerability_selection_use_noise_penalty", True), True)
    use_cost_penalty = _safe_bool(getattr(cfg, "answerability_selection_use_cost_penalty", True), True)
    ordering_enabled = _safe_bool(getattr(cfg, "answerability_selection_ordering_enabled", True), True)
    structure_weight = _safe_float(getattr(cfg, "answerability_selection_structure_weight", 0.65), 0.65)
    noise_weight = _safe_float(getattr(cfg, "answerability_selection_noise_weight", 0.55), 0.55)
    cost_weight = _safe_float(getattr(cfg, "answerability_selection_cost_weight", 0.35), 0.35)

    if mode in {"beam", "beam3"}:
        selected_indices, score_breakdown = _select_beam(
            atoms=atoms,
            question_tokens=question_tokens,
            question_entities=question_entities,
            relation_tokens=relation_tokens,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            beam_size=beam_size,
            hard_budget_enabled=hard_budget_enabled,
            use_answerability=use_answerability,
            use_structure=use_structure,
            use_noise_penalty=use_noise_penalty,
            use_cost_penalty=use_cost_penalty,
            structure_weight=structure_weight,
            noise_weight=noise_weight,
            cost_weight=cost_weight,
        )
        selection_mode = "beam"
    else:
        selected_indices, score_breakdown = _select_greedy(
            atoms=atoms,
            question_tokens=question_tokens,
            question_entities=question_entities,
            relation_tokens=relation_tokens,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            hard_budget_enabled=hard_budget_enabled,
            use_answerability=use_answerability,
            use_structure=use_structure,
            use_noise_penalty=use_noise_penalty,
            use_cost_penalty=use_cost_penalty,
            structure_weight=structure_weight,
            noise_weight=noise_weight,
            cost_weight=cost_weight,
        )
        selection_mode = "greedy"

    if not selected_indices:
        diag["reason"] = "selector_empty"
        return base_ids, base_texts, diag

    ordered_indices, ordering_type = _order_selected_atoms(
        atoms=atoms,
        selected_indices=selected_indices,
        ordering_enabled=ordering_enabled,
    )
    selected_atoms = [atoms[idx] for idx in ordered_indices]
    out_ids = [atom.sentence_id for atom in selected_atoms]
    out_texts = [_normalize_space(atom.text) for atom in selected_atoms]

    diag.update(
        {
            "applied": True,
            "mode": str(selection_mode),
            "ordering_enabled": bool(ordering_enabled),
            "evidence_ordering_type": str(ordering_type),
            "num_candidate_atoms": int(len(atoms)),
            "num_selected_atoms": int(len(selected_atoms)),
            "selected_tokens": int(sum(int(atom.token_count) for atom in selected_atoms)),
            "rendered_tokens": int(sum(int(atom.token_count) for atom in selected_atoms)),
            "answerability_score_A": float(score_breakdown.get("A", 0.0)),
            "structure_score_S": float(score_breakdown.get("S", 0.0)),
            "noise_penalty_N": float(score_breakdown.get("N", 0.0)),
            "cost_C": float(score_breakdown.get("C", 0.0)),
            "final_score": float(score_breakdown.get("score", 0.0)),
            "answerability_gain_per_atom": float(
                safe_div(float(score_breakdown.get("score", 0.0)), float(max(1, len(selected_atoms))))
            ),
            "structure_coverage": float(score_breakdown.get("structure_coverage", 0.0)),
            "redundancy_rate": float(score_breakdown.get("redundancy_rate", 0.0)),
            "query_coverage": float(score_breakdown.get("query_coverage", 0.0)),
            "entity_coverage": float(score_breakdown.get("entity_coverage", 0.0)),
            "relation_coverage": float(score_breakdown.get("relation_coverage", 0.0)),
            "answer_type_coverage": float(score_breakdown.get("answer_type_coverage", 0.0)),
            "selected_atom_ids": list(out_ids),
            "candidate_atom_ids": [str(atom.sentence_id) for atom in atoms],
        }
    )
    return out_ids, out_texts, diag

