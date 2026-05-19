from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .utils import content_tokens, safe_div


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
DATE_TOKEN_RE = re.compile(
    r"\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    flags=re.IGNORECASE,
)
NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class UnifiedEvidenceAtom:
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
    locality_score: float
    corridor_ids: Tuple[str, ...]
    bridge_gain: float
    atom_base_score: float


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
    return {tok for tok in question_tokens if tok in relation_keywords}


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
        return 1.0 if CAPITALIZED_TOKEN_RE.search(body) else 0.0
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


def _bridge_signal_from_feature_row(row: Mapping[str, Any]) -> float:
    candidates = [
        _safe_float(row.get("bridge_gain", 0.0), 0.0),
        _safe_float(row.get("bridge_path_score", 0.0), 0.0),
        _safe_float(row.get("best_corridor_score", 0.0), 0.0),
        1.0 if bool(row.get("is_connector_adjacent", False)) else 0.0,
        0.7 if bool(row.get("is_support_candidate", False)) else 0.0,
        _safe_float(row.get("locality_score", 0.0), 0.0),
    ]
    return float(max(0.0, min(1.0, max(candidates) if candidates else 0.0)))


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
    answerability_weight: float,
    lambda_bridge: float,
    use_answerability_gain: bool,
    use_bridge_gain: bool,
) -> List[UnifiedEvidenceAtom]:
    feature_table = dict(sentence_feature_table or {})
    span_cap = max(1, int(max_span_sentences))
    atoms: List[UnifiedEvidenceAtom] = []

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

        row = dict(feature_table.get(sid_norm, {}) or {})
        row_bridge = _bridge_signal_from_feature_row(row)
        structure_anchor = 1.0 if bool(row.get("is_main_candidate", False)) else 0.0
        structure_bridge = 1.0 if bool(row.get("is_connector_adjacent", False)) else (
            0.7 if bool(row.get("is_support_candidate", False)) else 0.0
        )
        locality = float(max(0.0, min(1.0, _safe_float(row.get("locality_score", 0.0), 0.0))))
        corridor_ids = tuple(_ordered_unique(list(row.get("corridor_ids", []) or [])))

        best_span = span_candidates[0]
        best_score = float("-inf")
        best_payload: Dict[str, Any] = {
            "token_set": set(),
            "token_count": 0,
            "query_overlap": 0.0,
            "entity_overlap": 0.0,
            "relation_overlap": 0.0,
            "answer_type_compat": 0.0,
            "bridge_overlap": 0.0,
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

            answerability_score = float(
                (2.0 * query_overlap)
                + (1.2 * entity_overlap)
                + (0.9 * relation_overlap)
                + (0.8 * answer_type_compat)
            )
            if not use_answerability_gain:
                answerability_score = 0.0
            span_bridge = float(max(row_bridge, min(1.0, bridge_overlap / float(max(1, len(bridge_tokens))))))
            bridge_score = float(lambda_bridge * span_bridge) if use_bridge_gain else 0.0
            total = float((answerability_weight * answerability_score) + bridge_score - length_penalty)
            if total > best_score:
                best_score = float(total)
                best_span = span_text
                best_payload = {
                    "token_set": set(span_tokens),
                    "token_count": int(token_count),
                    "query_overlap": float(query_overlap),
                    "entity_overlap": float(entity_overlap),
                    "relation_overlap": float(relation_overlap),
                    "answer_type_compat": float(answer_type_compat),
                    "bridge_overlap": float(bridge_overlap),
                }

        atoms.append(
            UnifiedEvidenceAtom(
                sentence_id=sid_norm,
                source_id=_source_id(sid_norm),
                text=str(best_span),
                token_count=int(best_payload["token_count"]),
                token_set=set(best_payload["token_set"]),
                query_overlap=float(best_payload["query_overlap"]),
                entity_overlap=float(best_payload["entity_overlap"]),
                relation_overlap=float(best_payload["relation_overlap"]),
                answer_type_compat=float(best_payload["answer_type_compat"]),
                bridge_overlap=float(best_payload["bridge_overlap"]),
                structure_anchor=float(structure_anchor),
                structure_bridge=float(structure_bridge),
                locality_score=float(locality),
                corridor_ids=tuple(corridor_ids),
                bridge_gain=float(row_bridge),
                atom_base_score=float(best_score),
            )
        )
    return atoms


def _question_coverage_score(
    *,
    selected_token_union: set[str],
    selected_answer_type_compat: float,
    question_tokens: set[str],
    question_entities: set[str],
    relation_tokens: set[str],
) -> float:
    query_cov = float(
        safe_div(
            float(len(selected_token_union.intersection(question_tokens))),
            float(max(1, len(question_tokens))),
        )
    )
    entity_cov = float(
        safe_div(
            float(len(selected_token_union.intersection(question_entities))),
            float(max(1, len(question_entities))),
        )
    )
    relation_cov = float(
        safe_div(
            float(len(selected_token_union.intersection(relation_tokens))),
            float(max(1, len(relation_tokens))),
        )
    )
    answer_cov = float(max(0.0, min(1.0, selected_answer_type_compat)))
    return float((2.0 * query_cov) + (1.2 * entity_cov) + (0.9 * relation_cov) + (0.8 * answer_cov))


def _bridge_gain(
    atom: UnifiedEvidenceAtom,
    selected_atoms: Sequence[UnifiedEvidenceAtom],
) -> float:
    if not selected_atoms:
        return float(max(atom.bridge_gain, atom.structure_bridge, atom.structure_anchor))

    selected_corridors = set()
    anchor_corridors = set()
    bridge_corridors = set()
    for item in selected_atoms:
        selected_corridors.update(item.corridor_ids)
        if item.structure_anchor > 0.0:
            anchor_corridors.update(item.corridor_ids)
        if item.structure_bridge > 0.0:
            bridge_corridors.update(item.corridor_ids)

    atom_corridors = set(atom.corridor_ids)
    corridor_overlap = 1.0 if selected_corridors.intersection(atom_corridors) else 0.0
    path_closure = 0.0
    if atom.structure_bridge > 0.0 and anchor_corridors.intersection(atom_corridors):
        path_closure = 1.0
    if atom.structure_anchor > 0.0 and bridge_corridors.intersection(atom_corridors):
        path_closure = 1.0
    if atom.structure_bridge > 0.0 and bridge_corridors.intersection(atom_corridors):
        path_closure = max(path_closure, 0.7)
    anchor_bridge_pair_gain = float(max(atom.structure_anchor, atom.structure_bridge))
    return float(max(atom.bridge_gain, corridor_overlap, path_closure, anchor_bridge_pair_gain))


def _chain_complement_gain(
    atom: UnifiedEvidenceAtom,
    selected_atoms: Sequence[UnifiedEvidenceAtom],
) -> float:
    """Reward complementary anchor/bridge evidence that closes a chain."""
    if not selected_atoms:
        return float(max(atom.structure_anchor, atom.structure_bridge))

    atom_corridors = set(atom.corridor_ids)
    has_anchor = any(item.structure_anchor > 0.0 for item in selected_atoms)
    has_bridge = any(item.structure_bridge > 0.0 for item in selected_atoms)
    atom_is_anchor = atom.structure_anchor > 0.0
    atom_is_bridge = atom.structure_bridge > 0.0

    role_complement = 1.0 if ((atom_is_anchor and has_bridge) or (atom_is_bridge and has_anchor)) else 0.0

    corridor_complement = 0.0
    if atom_corridors:
        for item in selected_atoms:
            if set(item.corridor_ids).intersection(atom_corridors):
                if (item.structure_anchor > 0.0 and atom_is_bridge) or (item.structure_bridge > 0.0 and atom_is_anchor):
                    corridor_complement = 1.0
                    break
                corridor_complement = max(corridor_complement, 0.6)

    bridge_synergy = float(max(atom.bridge_gain, atom.bridge_overlap, atom.structure_bridge))
    return float(max(role_complement, corridor_complement, bridge_synergy * 0.5))


def _infer_atom_role(atom: UnifiedEvidenceAtom) -> str:
    anchor_score = float(atom.query_overlap + atom.entity_overlap + atom.structure_anchor)
    bridge_score = float(atom.bridge_gain + atom.structure_bridge + atom.bridge_overlap)
    answer_score = float(atom.answer_type_compat + atom.relation_overlap)
    best = max(anchor_score, bridge_score, answer_score)
    if best <= 0.0:
        return "generic"
    if anchor_score >= bridge_score and anchor_score >= answer_score:
        return "anchor"
    if bridge_score >= answer_score:
        return "bridge"
    return "answer"


def _has_evidence_signal(atom: UnifiedEvidenceAtom, bridge_threshold: float = 0.35) -> bool:
    if float(atom.answer_type_compat) > 0.0:
        return True
    if float(atom.relation_overlap) > 0.0:
        return True
    if float(atom.entity_overlap) > 0.0:
        return True
    bridge_like = max(float(atom.bridge_gain), float(atom.structure_bridge), float(atom.bridge_overlap))
    return bridge_like >= float(max(0.0, bridge_threshold))


def _has_shared_source_or_corridor(a: UnifiedEvidenceAtom, b: UnifiedEvidenceAtom) -> bool:
    if str(a.source_id) and str(a.source_id) == str(b.source_id):
        return True
    return bool(set(a.corridor_ids).intersection(set(b.corridor_ids)))


def _token_jaccard(a: set[str], b: set[str]) -> float:
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(len(a.intersection(b)) / float(union))


def _redundancy_penalty(
    atom: UnifiedEvidenceAtom,
    selected_atoms: Sequence[UnifiedEvidenceAtom],
) -> float:
    if not selected_atoms:
        return 0.0
    token_jaccards = [_token_jaccard(atom.token_set, item.token_set) for item in selected_atoms]
    source_repeat = 1.0 if any(item.source_id == atom.source_id for item in selected_atoms) else 0.0
    selected_corridors = set()
    for item in selected_atoms:
        selected_corridors.update(item.corridor_ids)
    atom_corridors = set(atom.corridor_ids)
    corridor_repeat = 0.0
    if atom_corridors:
        corridor_repeat = float(
            safe_div(
                float(len(selected_corridors.intersection(atom_corridors))),
                float(max(1, len(atom_corridors))),
            )
        )
    return float(max(max(token_jaccards) if token_jaccards else 0.0, source_repeat, corridor_repeat))


def _cost_penalty(
    *,
    selected_tokens: int,
    candidate_tokens: int,
    max_tokens: int,
) -> float:
    budget = max(1, int(max_tokens))
    return float(max(0.0, min(1.0, float(selected_tokens + candidate_tokens) / float(budget))))


def _in_budget(
    *,
    selected_atoms: Sequence[UnifiedEvidenceAtom],
    selected_tokens: int,
    candidate_tokens: int,
    max_tokens: int,
    hard_budget_enabled: bool,
) -> bool:
    if not hard_budget_enabled:
        return True
    if not selected_atoms:
        return True
    return int(selected_tokens + candidate_tokens) <= int(max_tokens)


def _marginal_delta(
    *,
    atom: UnifiedEvidenceAtom,
    selected_atoms: Sequence[UnifiedEvidenceAtom],
    selected_tokens: int,
    selected_union_tokens: set[str],
    selected_answer_type_compat: float,
    question_tokens: set[str],
    question_entities: set[str],
    relation_tokens: set[str],
    max_tokens: int,
    use_answerability_gain: bool,
    use_bridge_gain: bool,
    use_redundancy_penalty: bool,
    use_cost_penalty: bool,
    chain_aware_enabled: bool,
    role_aware_redundancy_enabled: bool,
    role_balanced_enabled: bool,
    redundancy_recalibrated_enabled: bool,
    lambda_bridge: float,
    mu_redundancy: float,
    chain_gain_weight: float,
    role_balance_weight: float,
    role_balance_max_gain_per_step: float,
    role_balance_max_token_jaccard: float,
    role_balance_missing_only: bool,
    role_redundancy_relax: float,
    role_redundancy_max_overlap: float,
    answerability_weight: float,
    timing_diag: Dict[str, float] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    if use_answerability_gain:
        t0 = time.perf_counter()
        current_a = _question_coverage_score(
            selected_token_union=selected_union_tokens,
            selected_answer_type_compat=selected_answer_type_compat,
            question_tokens=question_tokens,
            question_entities=question_entities,
            relation_tokens=relation_tokens,
        )
        next_union = set(selected_union_tokens)
        next_union.update(atom.token_set)
        next_answer = max(float(selected_answer_type_compat), float(atom.answer_type_compat))
        next_a = _question_coverage_score(
            selected_token_union=next_union,
            selected_answer_type_compat=next_answer,
            question_tokens=question_tokens,
            question_entities=question_entities,
            relation_tokens=relation_tokens,
        )
        a_gain = float(max(0.0, next_a - current_a))
        if timing_diag is not None:
            timing_diag["answerability_feature_ms"] = float(
                timing_diag.get("answerability_feature_ms", 0.0) + ((time.perf_counter() - t0) * 1000.0)
            )
    else:
        a_gain = 0.0

    if use_bridge_gain:
        t0 = time.perf_counter()
        b_gain = _bridge_gain(atom, selected_atoms)
        if timing_diag is not None:
            timing_diag["bridge_feature_ms"] = float(
                timing_diag.get("bridge_feature_ms", 0.0) + ((time.perf_counter() - t0) * 1000.0)
            )
    else:
        b_gain = 0.0

    chain_gain = 0.0
    if chain_aware_enabled:
        chain_gain = _chain_complement_gain(atom, selected_atoms)

    atom_role = _infer_atom_role(atom)
    selected_role_counts = Counter(_infer_atom_role(item) for item in selected_atoms)
    role_coverage_gain = 0.0
    role_coverage_gain_weighted = 0.0
    role_balance_activated = 0
    role_balance_rejected_reason = "disabled"
    if role_balanced_enabled:
        max_overlap = 0.0
        if selected_atoms:
            max_overlap = max(_token_jaccard(atom.token_set, item.token_set) for item in selected_atoms)
        missing_role = int(selected_role_counts.get(atom_role, 0)) <= 0
        if atom_role == "generic":
            role_balance_rejected_reason = "generic_role"
        elif bool(role_balance_missing_only) and not missing_role:
            role_balance_rejected_reason = "role_already_present"
        elif not _has_evidence_signal(atom):
            role_balance_rejected_reason = "weak_signal"
        elif selected_atoms and float(max_overlap) >= float(max(0.0, role_balance_max_token_jaccard)):
            role_balance_rejected_reason = "high_overlap"
        else:
            role_balance_activated = 1
            role_balance_rejected_reason = ""
            role_coverage_gain = 1.0
            role_coverage_gain_weighted = float(
                min(
                    max(0.0, float(role_balance_max_gain_per_step)),
                    max(0.0, float(role_balance_weight)) * float(role_coverage_gain),
                )
            )

    if use_redundancy_penalty:
        t0 = time.perf_counter()
        redundancy = _redundancy_penalty(atom, selected_atoms)
        if timing_diag is not None:
            timing_diag["redundancy_scoring_ms"] = float(
                timing_diag.get("redundancy_scoring_ms", 0.0) + ((time.perf_counter() - t0) * 1000.0)
            )
    else:
        redundancy = 0.0

    redundancy_before = float(redundancy)
    redundancy_effective = float(redundancy_before)
    redundancy_relax_applied = 0.0
    redundancy_recalibrated = 0
    redundancy_recalibration_reason = "disabled"
    if redundancy_recalibrated_enabled and use_redundancy_penalty and selected_atoms:
        relax_applied = False
        atom_signal = _has_evidence_signal(atom)
        if not atom_signal:
            redundancy_recalibration_reason = "weak_signal"
        else:
            atom_role_local = atom_role
            max_overlap = float(max(0.0, role_redundancy_max_overlap))
            saw_generic_or_same = False
            saw_overlap_block = False
            saw_no_shared = False
            for item in selected_atoms:
                other_role = _infer_atom_role(item)
                if atom_role_local == "generic" or other_role == "generic":
                    saw_generic_or_same = True
                    continue
                if atom_role_local == other_role:
                    saw_generic_or_same = True
                    continue
                if not _has_shared_source_or_corridor(atom, item):
                    saw_no_shared = True
                    continue
                overlap = _token_jaccard(atom.token_set, item.token_set)
                if overlap >= max_overlap:
                    saw_overlap_block = True
                    continue
                relax = max(0.0, min(1.0, float(role_redundancy_relax)))
                redundancy_relax_applied = float(relax)
                redundancy_effective = float(redundancy_before * (1.0 - relax))
                redundancy_recalibrated = 1
                redundancy_recalibration_reason = "applied"
                relax_applied = True
                break
            if not relax_applied:
                if saw_overlap_block:
                    redundancy_recalibration_reason = "high_overlap"
                elif saw_no_shared:
                    redundancy_recalibration_reason = "no_shared_source_or_corridor"
                elif saw_generic_or_same:
                    redundancy_recalibration_reason = "role_not_complementary"
                else:
                    redundancy_recalibration_reason = "conditions_not_met"

    if (
        (not redundancy_recalibrated)
        and role_aware_redundancy_enabled
        and use_redundancy_penalty
        and selected_atoms
    ):
        atom_is_anchor = atom.structure_anchor > 0.0
        atom_is_bridge = atom.structure_bridge > 0.0
        has_anchor = any(item.structure_anchor > 0.0 for item in selected_atoms)
        has_bridge = any(item.structure_bridge > 0.0 for item in selected_atoms)
        complementary = (atom_is_anchor and has_bridge) or (atom_is_bridge and has_anchor)
        if complementary:
            relax = max(0.0, min(1.0, float(role_redundancy_relax)))
            redundancy_relax_applied = float(relax)
            redundancy_effective = float(redundancy_before * (1.0 - relax))
            redundancy_recalibration_reason = "legacy_role_aware_relax"
            redundancy_recalibrated = int(redundancy_recalibrated or 0)

    cost = _cost_penalty(
        selected_tokens=selected_tokens,
        candidate_tokens=int(atom.token_count),
        max_tokens=max_tokens,
    ) if use_cost_penalty else 0.0

    delta = float(
        (answerability_weight * a_gain)
        + (lambda_bridge * b_gain)
        + (chain_gain_weight * chain_gain)
        + float(role_coverage_gain_weighted)
        - (mu_redundancy * redundancy_effective)
        - cost
    )
    return delta, {
        "a_gain": float(a_gain),
        "b_gain": float(b_gain),
        "chain_gain": float(chain_gain),
        "atom_role": str(atom_role),
        "selected_role_counts": {
            "anchor": int(selected_role_counts.get("anchor", 0)),
            "bridge": int(selected_role_counts.get("bridge", 0)),
            "answer": int(selected_role_counts.get("answer", 0)),
            "generic": int(selected_role_counts.get("generic", 0)),
        },
        "role_coverage_gain": float(role_coverage_gain),
        "role_coverage_gain_weighted": float(role_coverage_gain_weighted),
        "role_balance_activated": int(role_balance_activated),
        "role_balance_rejected": int(1 if role_balanced_enabled and not role_balance_activated else 0),
        "role_balance_rejected_reason": str(role_balance_rejected_reason),
        "redundancy_before": float(redundancy_before),
        "redundancy": float(redundancy_effective),
        "redundancy_raw": float(redundancy_before),
        "redundancy_after": float(redundancy_effective),
        "redundancy_relax_applied": float(redundancy_relax_applied),
        "redundancy_recalibrated": int(redundancy_recalibrated),
        "redundancy_recalibration_reason": str(redundancy_recalibration_reason),
        "cost": float(cost),
    }


def _order_selected_atoms(selected_atoms: Sequence[UnifiedEvidenceAtom]) -> Tuple[List[UnifiedEvidenceAtom], str]:
    def _priority(atom: UnifiedEvidenceAtom) -> Tuple[int, float, float]:
        focus = float(atom.query_overlap + atom.entity_overlap + atom.relation_overlap)
        bridge = float(max(atom.structure_bridge, atom.bridge_overlap, atom.bridge_gain))
        answer = float(atom.answer_type_compat)
        if focus > 0.0:
            group = 0
        elif bridge > 0.0:
            group = 1
        elif answer > 0.0:
            group = 2
        else:
            group = 3
        return (group, -float(atom.atom_base_score), float(atom.token_count))

    return sorted(list(selected_atoms), key=_priority), "question_bridge_answer"


def _select_greedy(
    *,
    atoms: Sequence[UnifiedEvidenceAtom],
    question_tokens: set[str],
    question_entities: set[str],
    relation_tokens: set[str],
    max_atoms: int,
    max_tokens: int,
    hard_budget_enabled: bool,
    use_answerability_gain: bool,
    use_bridge_gain: bool,
    use_redundancy_penalty: bool,
    use_cost_penalty: bool,
    chain_aware_enabled: bool,
    role_aware_redundancy_enabled: bool,
    role_balanced_enabled: bool,
    redundancy_recalibrated_enabled: bool,
    lambda_bridge: float,
    mu_redundancy: float,
    chain_gain_weight: float,
    role_balance_weight: float,
    role_balance_max_gain_per_step: float,
    role_balance_max_token_jaccard: float,
    role_balance_missing_only: bool,
    role_redundancy_relax: float,
    role_redundancy_max_overlap: float,
    answerability_weight: float,
) -> Tuple[List[int], Dict[str, Any]]:
    selected_indices: List[int] = []
    selected_tokens = 0
    selected_union_tokens: set[str] = set()
    selected_answer_type_compat = 0.0
    a_gain_total = 0.0
    b_gain_total = 0.0
    chain_gain_total = 0.0
    role_coverage_gain_total = 0.0
    role_coverage_gain_weighted_total = 0.0
    redundancy_total = 0.0
    redundancy_raw_total = 0.0
    role_redundancy_relax_applied_total = 0.0
    redundancy_before_total = 0.0
    redundancy_after_total = 0.0
    score_total = 0.0
    objective_eval_calls = 0
    role_balance_activated_count = 0
    role_balance_rejected_count = 0
    role_balance_rejected_reasons: Counter[str] = Counter()
    redundancy_recalibrated_count = 0
    redundancy_recalibration_reasons: Counter[str] = Counter()
    score_trace: List[Dict[str, Any]] = []
    timing_diag = {
        "answerability_feature_ms": 0.0,
        "bridge_feature_ms": 0.0,
        "redundancy_scoring_ms": 0.0,
    }

    while len(selected_indices) < max_atoms:
        best_idx = None
        best_delta = float("-inf")
        best_components: Dict[str, Any] | None = None
        selected_atoms = [atoms[i] for i in selected_indices]
        for idx, atom in enumerate(atoms):
            if idx in selected_indices:
                continue
            if not _in_budget(
                selected_atoms=selected_atoms,
                selected_tokens=selected_tokens,
                candidate_tokens=int(atom.token_count),
                max_tokens=max_tokens,
                hard_budget_enabled=hard_budget_enabled,
            ):
                continue
            delta, components = _marginal_delta(
                atom=atom,
                selected_atoms=selected_atoms,
                selected_tokens=selected_tokens,
                selected_union_tokens=selected_union_tokens,
                selected_answer_type_compat=selected_answer_type_compat,
                question_tokens=question_tokens,
                question_entities=question_entities,
                relation_tokens=relation_tokens,
                max_tokens=max_tokens,
                use_answerability_gain=use_answerability_gain,
                use_bridge_gain=use_bridge_gain,
                use_redundancy_penalty=use_redundancy_penalty,
                use_cost_penalty=use_cost_penalty,
                chain_aware_enabled=chain_aware_enabled,
                role_aware_redundancy_enabled=role_aware_redundancy_enabled,
                role_balanced_enabled=role_balanced_enabled,
                redundancy_recalibrated_enabled=redundancy_recalibrated_enabled,
                lambda_bridge=lambda_bridge,
                mu_redundancy=mu_redundancy,
                chain_gain_weight=chain_gain_weight,
                role_balance_weight=role_balance_weight,
                role_balance_max_gain_per_step=role_balance_max_gain_per_step,
                role_balance_max_token_jaccard=role_balance_max_token_jaccard,
                role_balance_missing_only=role_balance_missing_only,
                role_redundancy_relax=role_redundancy_relax,
                role_redundancy_max_overlap=role_redundancy_max_overlap,
                answerability_weight=answerability_weight,
                timing_diag=timing_diag,
            )
            objective_eval_calls += 1
            role_balance_activated_count += int(components.get("role_balance_activated", 0))
            role_balance_rejected_count += int(components.get("role_balance_rejected", 0))
            rejected_reason = str(components.get("role_balance_rejected_reason", "") or "")
            if rejected_reason:
                role_balance_rejected_reasons[rejected_reason] += 1
            if int(components.get("redundancy_recalibrated", 0)) > 0:
                redundancy_recalibrated_count += 1
            recal_reason = str(components.get("redundancy_recalibration_reason", "") or "")
            if recal_reason:
                redundancy_recalibration_reasons[recal_reason] += 1
            if (delta > best_delta) or (
                delta == best_delta and (best_idx is None or atoms[idx].atom_base_score > atoms[best_idx].atom_base_score)
            ):
                best_delta = float(delta)
                best_idx = int(idx)
                best_components = dict(components)

        if best_idx is None or best_components is None:
            break
        if best_delta <= 0.0 and selected_indices:
            break

        chosen = atoms[best_idx]
        selected_indices.append(best_idx)
        selected_tokens += int(chosen.token_count)
        selected_union_tokens.update(chosen.token_set)
        selected_answer_type_compat = max(float(selected_answer_type_compat), float(chosen.answer_type_compat))
        a_gain_total += float(best_components["a_gain"])
        b_gain_total += float(best_components["b_gain"])
        chain_gain_total += float(best_components.get("chain_gain", 0.0))
        role_coverage_gain_total += float(best_components.get("role_coverage_gain", 0.0))
        role_coverage_gain_weighted_total += float(best_components.get("role_coverage_gain_weighted", 0.0))
        redundancy_before_total += float(best_components.get("redundancy_before", best_components.get("redundancy_raw", 0.0)))
        redundancy_after_total += float(best_components.get("redundancy_after", best_components.get("redundancy", 0.0)))
        redundancy_total += float(best_components["redundancy"])
        redundancy_raw_total += float(best_components.get("redundancy_raw", best_components.get("redundancy", 0.0)))
        role_redundancy_relax_applied_total += float(best_components.get("redundancy_relax_applied", 0.0))
        score_total += float(best_delta)
        score_trace.append(
            {
                "step": int(len(selected_indices)),
                "sentence_id": str(chosen.sentence_id),
                "delta": float(best_delta),
                "a_gain": float(best_components["a_gain"]),
                "b_gain": float(best_components["b_gain"]),
                "chain_gain": float(best_components.get("chain_gain", 0.0)),
                "role_coverage_gain": float(best_components.get("role_coverage_gain", 0.0)),
                "role_coverage_gain_weighted": float(best_components.get("role_coverage_gain_weighted", 0.0)),
                "atom_role": str(best_components.get("atom_role", "generic")),
                "redundancy": float(best_components["redundancy"]),
                "redundancy_raw": float(best_components.get("redundancy_raw", best_components.get("redundancy", 0.0))),
                "redundancy_before": float(best_components.get("redundancy_before", best_components.get("redundancy_raw", 0.0))),
                "redundancy_after": float(best_components.get("redundancy_after", best_components.get("redundancy", 0.0))),
                "redundancy_relax_applied": float(best_components.get("redundancy_relax_applied", 0.0)),
                "redundancy_recalibrated": int(best_components.get("redundancy_recalibrated", 0)),
                "redundancy_recalibration_reason": str(
                    best_components.get("redundancy_recalibration_reason", "")
                ),
                "cost": float(best_components["cost"]),
                "selected_tokens": int(selected_tokens),
            }
        )
        if hard_budget_enabled and int(selected_tokens) >= int(max_tokens):
            break

    if not selected_indices and atoms:
        best_idx = None
        best_delta = float("-inf")
        best_components = None
        for idx, atom in enumerate(atoms):
            delta, components = _marginal_delta(
                atom=atom,
                selected_atoms=[],
                selected_tokens=0,
                selected_union_tokens=set(),
                selected_answer_type_compat=0.0,
                question_tokens=question_tokens,
                question_entities=question_entities,
                relation_tokens=relation_tokens,
                max_tokens=max_tokens,
                use_answerability_gain=use_answerability_gain,
                use_bridge_gain=use_bridge_gain,
                use_redundancy_penalty=use_redundancy_penalty,
                use_cost_penalty=use_cost_penalty,
                chain_aware_enabled=chain_aware_enabled,
                role_aware_redundancy_enabled=role_aware_redundancy_enabled,
                role_balanced_enabled=role_balanced_enabled,
                redundancy_recalibrated_enabled=redundancy_recalibrated_enabled,
                lambda_bridge=lambda_bridge,
                mu_redundancy=mu_redundancy,
                chain_gain_weight=chain_gain_weight,
                role_balance_weight=role_balance_weight,
                role_balance_max_gain_per_step=role_balance_max_gain_per_step,
                role_balance_max_token_jaccard=role_balance_max_token_jaccard,
                role_balance_missing_only=role_balance_missing_only,
                role_redundancy_relax=role_redundancy_relax,
                role_redundancy_max_overlap=role_redundancy_max_overlap,
                answerability_weight=answerability_weight,
                timing_diag=timing_diag,
            )
            objective_eval_calls += 1
            role_balance_activated_count += int(components.get("role_balance_activated", 0))
            role_balance_rejected_count += int(components.get("role_balance_rejected", 0))
            rejected_reason = str(components.get("role_balance_rejected_reason", "") or "")
            if rejected_reason:
                role_balance_rejected_reasons[rejected_reason] += 1
            if int(components.get("redundancy_recalibrated", 0)) > 0:
                redundancy_recalibrated_count += 1
            recal_reason = str(components.get("redundancy_recalibration_reason", "") or "")
            if recal_reason:
                redundancy_recalibration_reasons[recal_reason] += 1
            if delta > best_delta:
                best_delta = float(delta)
                best_idx = int(idx)
                best_components = dict(components)
        if best_idx is not None and best_components is not None:
            chosen = atoms[best_idx]
            selected_indices = [best_idx]
            selected_tokens = int(chosen.token_count)
            a_gain_total = float(best_components["a_gain"])
            b_gain_total = float(best_components["b_gain"])
            chain_gain_total = float(best_components.get("chain_gain", 0.0))
            role_coverage_gain_total = float(best_components.get("role_coverage_gain", 0.0))
            role_coverage_gain_weighted_total = float(best_components.get("role_coverage_gain_weighted", 0.0))
            redundancy_before_total = float(
                best_components.get("redundancy_before", best_components.get("redundancy_raw", 0.0))
            )
            redundancy_after_total = float(
                best_components.get("redundancy_after", best_components.get("redundancy", 0.0))
            )
            redundancy_total = float(best_components["redundancy"])
            redundancy_raw_total = float(best_components.get("redundancy_raw", best_components.get("redundancy", 0.0)))
            role_redundancy_relax_applied_total = float(best_components.get("redundancy_relax_applied", 0.0))
            score_total = float(best_delta)
            score_trace = [
                {
                    "step": 1,
                    "sentence_id": str(chosen.sentence_id),
                    "delta": float(best_delta),
                    "a_gain": float(best_components["a_gain"]),
                    "b_gain": float(best_components["b_gain"]),
                    "chain_gain": float(best_components.get("chain_gain", 0.0)),
                    "role_coverage_gain": float(best_components.get("role_coverage_gain", 0.0)),
                    "role_coverage_gain_weighted": float(best_components.get("role_coverage_gain_weighted", 0.0)),
                    "atom_role": str(best_components.get("atom_role", "generic")),
                    "redundancy": float(best_components["redundancy"]),
                    "redundancy_raw": float(best_components.get("redundancy_raw", best_components.get("redundancy", 0.0))),
                    "redundancy_before": float(
                        best_components.get("redundancy_before", best_components.get("redundancy_raw", 0.0))
                    ),
                    "redundancy_after": float(
                        best_components.get("redundancy_after", best_components.get("redundancy", 0.0))
                    ),
                    "redundancy_relax_applied": float(best_components.get("redundancy_relax_applied", 0.0)),
                    "redundancy_recalibrated": int(best_components.get("redundancy_recalibrated", 0)),
                    "redundancy_recalibration_reason": str(
                        best_components.get("redundancy_recalibration_reason", "")
                    ),
                    "cost": float(best_components["cost"]),
                    "selected_tokens": int(selected_tokens),
                }
            ]

    return list(selected_indices), {
        "selected_tokens": int(selected_tokens),
        "A_gain_total": float(a_gain_total),
        "B_gain_total": float(b_gain_total),
        "chain_gain_total": float(chain_gain_total),
        "role_coverage_gain_total": float(role_coverage_gain_total),
        "role_coverage_gain_weighted_total": float(role_coverage_gain_weighted_total),
        "redundancy_total": float(redundancy_total),
        "redundancy_raw_total": float(redundancy_raw_total),
        "redundancy_before_total": float(redundancy_before_total),
        "redundancy_after_total": float(redundancy_after_total),
        "role_aware_redundancy_applied_total": float(role_redundancy_relax_applied_total),
        "role_balance_activated_count": int(role_balance_activated_count),
        "role_balance_rejected_count": int(role_balance_rejected_count),
        "role_balance_rejected_reasons": dict(role_balance_rejected_reasons),
        "redundancy_recalibrated_count": int(redundancy_recalibrated_count),
        "redundancy_recalibration_reasons": dict(redundancy_recalibration_reasons),
        "selected_role_counts": dict(Counter(_infer_atom_role(atoms[i]) for i in selected_indices)),
        "score_total": float(score_total),
        "objective_eval_calls": int(objective_eval_calls),
        "score_trace": list(score_trace),
        "answerability_feature_ms": float(timing_diag.get("answerability_feature_ms", 0.0)),
        "bridge_feature_ms": float(timing_diag.get("bridge_feature_ms", 0.0)),
        "redundancy_scoring_ms": float(timing_diag.get("redundancy_scoring_ms", 0.0)),
    }


def _select_beam(
    *,
    atoms: Sequence[UnifiedEvidenceAtom],
    question_tokens: set[str],
    question_entities: set[str],
    relation_tokens: set[str],
    max_atoms: int,
    max_tokens: int,
    beam_size: int,
    hard_budget_enabled: bool,
    use_answerability_gain: bool,
    use_bridge_gain: bool,
    use_redundancy_penalty: bool,
    use_cost_penalty: bool,
    chain_aware_enabled: bool,
    role_aware_redundancy_enabled: bool,
    role_balanced_enabled: bool,
    redundancy_recalibrated_enabled: bool,
    lambda_bridge: float,
    mu_redundancy: float,
    chain_gain_weight: float,
    role_balance_weight: float,
    role_balance_max_gain_per_step: float,
    role_balance_max_token_jaccard: float,
    role_balance_missing_only: bool,
    role_redundancy_relax: float,
    role_redundancy_max_overlap: float,
    answerability_weight: float,
) -> Tuple[List[int], Dict[str, Any]]:
    beam_k = max(1, int(beam_size))
    objective_eval_calls = 0
    role_balance_activated_count = 0
    role_balance_rejected_count = 0
    role_balance_rejected_reasons: Counter[str] = Counter()
    redundancy_recalibrated_count = 0
    redundancy_recalibration_reasons: Counter[str] = Counter()
    timing_diag = {
        "answerability_feature_ms": 0.0,
        "bridge_feature_ms": 0.0,
        "redundancy_scoring_ms": 0.0,
    }

    # tuple(indices) -> state
    beams: Dict[Tuple[int, ...], Dict[str, Any]] = {
        tuple(): {
            "score_total": 0.0,
            "selected_tokens": 0,
            "A_gain_total": 0.0,
            "B_gain_total": 0.0,
            "chain_gain_total": 0.0,
            "role_coverage_gain_total": 0.0,
            "role_coverage_gain_weighted_total": 0.0,
            "redundancy_total": 0.0,
            "redundancy_raw_total": 0.0,
            "redundancy_before_total": 0.0,
            "redundancy_after_total": 0.0,
            "role_aware_redundancy_applied_total": 0.0,
            "score_trace": [],
        }
    }

    for _ in range(max_atoms):
        next_states: Dict[Tuple[int, ...], Dict[str, Any]] = dict(beams)
        expanded = False
        for state_indices, state in list(beams.items()):
            selected_atoms = [atoms[i] for i in state_indices]
            selected_union_tokens: set[str] = set()
            selected_answer_type = 0.0
            for atom in selected_atoms:
                selected_union_tokens.update(atom.token_set)
                selected_answer_type = max(float(selected_answer_type), float(atom.answer_type_compat))

            for idx, atom in enumerate(atoms):
                if idx in state_indices:
                    continue
                if not _in_budget(
                    selected_atoms=selected_atoms,
                    selected_tokens=int(state["selected_tokens"]),
                    candidate_tokens=int(atom.token_count),
                    max_tokens=max_tokens,
                    hard_budget_enabled=hard_budget_enabled,
                ):
                    continue
                delta, components = _marginal_delta(
                    atom=atom,
                    selected_atoms=selected_atoms,
                    selected_tokens=int(state["selected_tokens"]),
                    selected_union_tokens=selected_union_tokens,
                    selected_answer_type_compat=selected_answer_type,
                    question_tokens=question_tokens,
                    question_entities=question_entities,
                    relation_tokens=relation_tokens,
                    max_tokens=max_tokens,
                    use_answerability_gain=use_answerability_gain,
                    use_bridge_gain=use_bridge_gain,
                    use_redundancy_penalty=use_redundancy_penalty,
                    use_cost_penalty=use_cost_penalty,
                    chain_aware_enabled=chain_aware_enabled,
                    role_aware_redundancy_enabled=role_aware_redundancy_enabled,
                    role_balanced_enabled=role_balanced_enabled,
                    redundancy_recalibrated_enabled=redundancy_recalibrated_enabled,
                    lambda_bridge=lambda_bridge,
                    mu_redundancy=mu_redundancy,
                    chain_gain_weight=chain_gain_weight,
                    role_balance_weight=role_balance_weight,
                    role_balance_max_gain_per_step=role_balance_max_gain_per_step,
                    role_balance_max_token_jaccard=role_balance_max_token_jaccard,
                    role_balance_missing_only=role_balance_missing_only,
                    role_redundancy_relax=role_redundancy_relax,
                    role_redundancy_max_overlap=role_redundancy_max_overlap,
                    answerability_weight=answerability_weight,
                    timing_diag=timing_diag,
                )
                objective_eval_calls += 1
                role_balance_activated_count += int(components.get("role_balance_activated", 0))
                role_balance_rejected_count += int(components.get("role_balance_rejected", 0))
                rejected_reason = str(components.get("role_balance_rejected_reason", "") or "")
                if rejected_reason:
                    role_balance_rejected_reasons[rejected_reason] += 1
                if int(components.get("redundancy_recalibrated", 0)) > 0:
                    redundancy_recalibrated_count += 1
                recal_reason = str(components.get("redundancy_recalibration_reason", "") or "")
                if recal_reason:
                    redundancy_recalibration_reasons[recal_reason] += 1
                if delta <= 0.0 and state_indices:
                    continue
                new_indices = tuple(list(state_indices) + [idx])
                new_score = float(state["score_total"] + delta)
                prev = next_states.get(new_indices)
                if prev is not None and float(prev.get("score_total", -math.inf)) >= new_score:
                    continue
                expanded = True
                next_states[new_indices] = {
                    "score_total": float(new_score),
                    "selected_tokens": int(state["selected_tokens"]) + int(atom.token_count),
                    "A_gain_total": float(state["A_gain_total"]) + float(components["a_gain"]),
                    "B_gain_total": float(state["B_gain_total"]) + float(components["b_gain"]),
                    "chain_gain_total": float(state.get("chain_gain_total", 0.0))
                    + float(components.get("chain_gain", 0.0)),
                    "role_coverage_gain_total": float(state.get("role_coverage_gain_total", 0.0))
                    + float(components.get("role_coverage_gain", 0.0)),
                    "role_coverage_gain_weighted_total": float(state.get("role_coverage_gain_weighted_total", 0.0))
                    + float(components.get("role_coverage_gain_weighted", 0.0)),
                    "redundancy_total": float(state["redundancy_total"]) + float(components["redundancy"]),
                    "redundancy_raw_total": float(state.get("redundancy_raw_total", 0.0))
                    + float(components.get("redundancy_raw", components.get("redundancy", 0.0))),
                    "redundancy_before_total": float(state.get("redundancy_before_total", 0.0))
                    + float(components.get("redundancy_before", components.get("redundancy_raw", 0.0))),
                    "redundancy_after_total": float(state.get("redundancy_after_total", 0.0))
                    + float(components.get("redundancy_after", components.get("redundancy", 0.0))),
                    "role_aware_redundancy_applied_total": float(
                        state.get("role_aware_redundancy_applied_total", 0.0)
                    )
                    + float(components.get("redundancy_relax_applied", 0.0)),
                    "score_trace": list(state["score_trace"])
                    + [
                        {
                            "step": int(len(new_indices)),
                            "sentence_id": str(atom.sentence_id),
                            "delta": float(delta),
                            "a_gain": float(components["a_gain"]),
                            "b_gain": float(components["b_gain"]),
                            "chain_gain": float(components.get("chain_gain", 0.0)),
                            "role_coverage_gain": float(components.get("role_coverage_gain", 0.0)),
                            "role_coverage_gain_weighted": float(components.get("role_coverage_gain_weighted", 0.0)),
                            "atom_role": str(components.get("atom_role", "generic")),
                            "redundancy": float(components["redundancy"]),
                            "redundancy_raw": float(components.get("redundancy_raw", components.get("redundancy", 0.0))),
                            "redundancy_before": float(
                                components.get("redundancy_before", components.get("redundancy_raw", 0.0))
                            ),
                            "redundancy_after": float(
                                components.get("redundancy_after", components.get("redundancy", 0.0))
                            ),
                            "redundancy_relax_applied": float(components.get("redundancy_relax_applied", 0.0)),
                            "redundancy_recalibrated": int(components.get("redundancy_recalibrated", 0)),
                            "redundancy_recalibration_reason": str(
                                components.get("redundancy_recalibration_reason", "")
                            ),
                            "cost": float(components["cost"]),
                            "selected_tokens": int(state["selected_tokens"]) + int(atom.token_count),
                        }
                    ],
                }

        ranked = sorted(
            list(next_states.items()),
            key=lambda item: (
                float(item[1].get("score_total", 0.0)),
                -float(item[1].get("selected_tokens", 0.0)),
                -float(len(item[0])),
            ),
            reverse=True,
        )
        beams = dict(ranked[:beam_k])
        if not expanded:
            break

    non_empty = [(indices, state) for indices, state in beams.items() if indices]
    if non_empty:
        best_indices, best_state = max(non_empty, key=lambda item: float(item[1].get("score_total", -math.inf)))
        return list(best_indices), {
            "selected_tokens": int(best_state["selected_tokens"]),
            "A_gain_total": float(best_state["A_gain_total"]),
            "B_gain_total": float(best_state["B_gain_total"]),
            "chain_gain_total": float(best_state.get("chain_gain_total", 0.0)),
            "role_coverage_gain_total": float(best_state.get("role_coverage_gain_total", 0.0)),
            "role_coverage_gain_weighted_total": float(best_state.get("role_coverage_gain_weighted_total", 0.0)),
            "redundancy_total": float(best_state["redundancy_total"]),
            "redundancy_raw_total": float(best_state.get("redundancy_raw_total", 0.0)),
            "redundancy_before_total": float(best_state.get("redundancy_before_total", 0.0)),
            "redundancy_after_total": float(best_state.get("redundancy_after_total", 0.0)),
            "role_aware_redundancy_applied_total": float(
                best_state.get("role_aware_redundancy_applied_total", 0.0)
            ),
            "role_balance_activated_count": int(role_balance_activated_count),
            "role_balance_rejected_count": int(role_balance_rejected_count),
            "role_balance_rejected_reasons": dict(role_balance_rejected_reasons),
            "redundancy_recalibrated_count": int(redundancy_recalibrated_count),
            "redundancy_recalibration_reasons": dict(redundancy_recalibration_reasons),
            "selected_role_counts": dict(Counter(_infer_atom_role(atoms[i]) for i in best_indices)),
            "score_total": float(best_state["score_total"]),
            "objective_eval_calls": int(objective_eval_calls),
            "score_trace": list(best_state["score_trace"]),
            "answerability_feature_ms": float(timing_diag.get("answerability_feature_ms", 0.0)),
            "bridge_feature_ms": float(timing_diag.get("bridge_feature_ms", 0.0)),
            "redundancy_scoring_ms": float(timing_diag.get("redundancy_scoring_ms", 0.0)),
        }

    greedy_indices, greedy_state = _select_greedy(
        atoms=atoms,
        question_tokens=question_tokens,
        question_entities=question_entities,
        relation_tokens=relation_tokens,
        max_atoms=max_atoms,
        max_tokens=max_tokens,
        hard_budget_enabled=hard_budget_enabled,
        use_answerability_gain=use_answerability_gain,
        use_bridge_gain=use_bridge_gain,
        use_redundancy_penalty=use_redundancy_penalty,
        use_cost_penalty=use_cost_penalty,
        chain_aware_enabled=chain_aware_enabled,
        role_aware_redundancy_enabled=role_aware_redundancy_enabled,
        role_balanced_enabled=role_balanced_enabled,
        redundancy_recalibrated_enabled=redundancy_recalibrated_enabled,
        lambda_bridge=lambda_bridge,
        mu_redundancy=mu_redundancy,
        chain_gain_weight=chain_gain_weight,
        role_balance_weight=role_balance_weight,
        role_balance_max_gain_per_step=role_balance_max_gain_per_step,
        role_balance_max_token_jaccard=role_balance_max_token_jaccard,
        role_balance_missing_only=role_balance_missing_only,
        role_redundancy_relax=role_redundancy_relax,
        role_redundancy_max_overlap=role_redundancy_max_overlap,
        answerability_weight=answerability_weight,
    )
    greedy_state["objective_eval_calls"] = int(
        _safe_int(greedy_state.get("objective_eval_calls", 0), 0) + objective_eval_calls
    )
    greedy_state["answerability_feature_ms"] = float(
        _safe_float(greedy_state.get("answerability_feature_ms", 0.0), 0.0)
        + float(timing_diag.get("answerability_feature_ms", 0.0))
    )
    greedy_state["bridge_feature_ms"] = float(
        _safe_float(greedy_state.get("bridge_feature_ms", 0.0), 0.0)
        + float(timing_diag.get("bridge_feature_ms", 0.0))
    )
    greedy_state["redundancy_scoring_ms"] = float(
        _safe_float(greedy_state.get("redundancy_scoring_ms", 0.0), 0.0)
        + float(timing_diag.get("redundancy_scoring_ms", 0.0))
    )
    return greedy_indices, greedy_state


def apply_unified_acr_rcedr_selection(
    *,
    question_text: str,
    selected_sentence_ids: Sequence[str],
    selected_sentences: Sequence[str],
    sentence_feature_table: Mapping[str, Mapping[str, Any]] | None,
    cfg: Any,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    enabled = bool(getattr(cfg, "unified_acr_rcedr_enabled", False))
    diag: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
    }
    base_ids = _ordered_unique(list(selected_sentence_ids or []))
    text_map = {str(sid): str(text or "") for sid, text in zip(selected_sentence_ids or [], selected_sentences or [])}
    base_texts = [str(text_map.get(sid, "") or "") for sid in base_ids]
    if (not enabled) or len(base_ids) <= 1:
        return base_ids, base_texts, diag

    started = time.perf_counter()
    question_tokens = {str(tok) for tok in content_tokens(str(question_text or ""))}
    question_entities = _extract_entity_tokens(str(question_text or ""))
    relation_tokens = _question_relation_tokens(question_tokens)
    expected_answer_type = _expected_answer_type(str(question_text or ""))
    bridge_tokens = _build_bridge_tokens(
        sentence_texts=base_texts,
        query_tokens=question_tokens,
        question_entities=question_entities,
    )

    max_span_sentences = max(1, _safe_int(getattr(cfg, "unified_acr_rcedr_atom_span_max_sentences", 2), 2))
    length_penalty_weight = max(
        0.0,
        _safe_float(getattr(cfg, "unified_acr_rcedr_length_penalty_weight", 0.04), 0.04),
    )
    max_atoms = max(1, _safe_int(getattr(cfg, "unified_acr_rcedr_max_atoms", 8), 8))
    max_tokens = max(1, _safe_int(getattr(cfg, "unified_acr_rcedr_max_tokens", 220), 220))
    mode = str(getattr(cfg, "unified_acr_rcedr_mode", "greedy") or "greedy").strip().lower()
    beam_size = max(1, _safe_int(getattr(cfg, "unified_acr_rcedr_beam_size", 3), 3))
    hard_budget_enabled = _safe_bool(getattr(cfg, "unified_acr_rcedr_hard_token_budget_enabled", True), True)
    use_answerability_gain = _safe_bool(getattr(cfg, "unified_acr_rcedr_use_answerability_gain", True), True)
    use_bridge_gain = _safe_bool(getattr(cfg, "unified_acr_rcedr_use_bridge_gain", True), True)
    use_redundancy_penalty = _safe_bool(getattr(cfg, "unified_acr_rcedr_use_redundancy_penalty", True), True)
    use_cost_penalty = _safe_bool(getattr(cfg, "unified_acr_rcedr_use_cost_penalty", False), False)
    chain_aware_enabled = _safe_bool(getattr(cfg, "unified_acr_rcedr_chain_aware_enabled", False), False)
    role_aware_redundancy_enabled = _safe_bool(
        getattr(cfg, "unified_acr_rcedr_role_aware_redundancy_enabled", False),
        False,
    )
    role_balanced_enabled = _safe_bool(
        getattr(cfg, "unified_acr_rcedr_role_balanced_enabled", False),
        False,
    )
    redundancy_recalibrated_enabled = _safe_bool(
        getattr(cfg, "unified_acr_rcedr_redundancy_recalibrated_enabled", False),
        False,
    )
    lambda_bridge = _safe_float(getattr(cfg, "unified_acr_rcedr_lambda_bridge", 0.28), 0.28)
    mu_redundancy = _safe_float(getattr(cfg, "unified_acr_rcedr_mu_redundancy", 0.22), 0.22)
    chain_gain_weight = _safe_float(getattr(cfg, "unified_acr_rcedr_chain_gain_weight", 0.15), 0.15)
    role_balance_weight = _safe_float(getattr(cfg, "unified_acr_rcedr_role_balance_weight", 0.08), 0.08)
    role_balance_max_gain_per_step = _safe_float(
        getattr(cfg, "unified_acr_rcedr_role_balance_max_gain_per_step", 0.08),
        0.08,
    )
    role_balance_max_token_jaccard = _safe_float(
        getattr(cfg, "unified_acr_rcedr_role_balance_max_token_jaccard", 0.45),
        0.45,
    )
    role_balance_missing_only = _safe_bool(
        getattr(cfg, "unified_acr_rcedr_role_balance_missing_only", True),
        True,
    )
    role_redundancy_relax = _safe_float(getattr(cfg, "unified_acr_rcedr_role_redundancy_relax", 0.5), 0.5)
    role_redundancy_max_overlap = _safe_float(
        getattr(cfg, "unified_acr_rcedr_role_redundancy_max_overlap", 0.45),
        0.45,
    )
    answerability_weight = _safe_float(getattr(cfg, "unified_acr_rcedr_answerability_weight", 1.0), 1.0)

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
        answerability_weight=answerability_weight,
        lambda_bridge=lambda_bridge,
        use_answerability_gain=use_answerability_gain,
        use_bridge_gain=use_bridge_gain,
    )
    if not atoms:
        diag["reason"] = "empty_atom_pool"
        return base_ids, base_texts, diag

    if mode in {"beam", "beam3"}:
        selected_indices, score_diag = _select_beam(
            atoms=atoms,
            question_tokens=question_tokens,
            question_entities=question_entities,
            relation_tokens=relation_tokens,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            beam_size=beam_size,
            hard_budget_enabled=hard_budget_enabled,
            use_answerability_gain=use_answerability_gain,
            use_bridge_gain=use_bridge_gain,
            use_redundancy_penalty=use_redundancy_penalty,
            use_cost_penalty=use_cost_penalty,
            chain_aware_enabled=chain_aware_enabled,
            role_aware_redundancy_enabled=role_aware_redundancy_enabled,
            role_balanced_enabled=role_balanced_enabled,
            redundancy_recalibrated_enabled=redundancy_recalibrated_enabled,
            lambda_bridge=lambda_bridge,
            mu_redundancy=mu_redundancy,
            chain_gain_weight=chain_gain_weight,
            role_balance_weight=role_balance_weight,
            role_balance_max_gain_per_step=role_balance_max_gain_per_step,
            role_balance_max_token_jaccard=role_balance_max_token_jaccard,
            role_balance_missing_only=role_balance_missing_only,
            role_redundancy_relax=role_redundancy_relax,
            role_redundancy_max_overlap=role_redundancy_max_overlap,
            answerability_weight=answerability_weight,
        )
        selection_mode = "beam"
    else:
        selected_indices, score_diag = _select_greedy(
            atoms=atoms,
            question_tokens=question_tokens,
            question_entities=question_entities,
            relation_tokens=relation_tokens,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            hard_budget_enabled=hard_budget_enabled,
            use_answerability_gain=use_answerability_gain,
            use_bridge_gain=use_bridge_gain,
            use_redundancy_penalty=use_redundancy_penalty,
            use_cost_penalty=use_cost_penalty,
            chain_aware_enabled=chain_aware_enabled,
            role_aware_redundancy_enabled=role_aware_redundancy_enabled,
            role_balanced_enabled=role_balanced_enabled,
            redundancy_recalibrated_enabled=redundancy_recalibrated_enabled,
            lambda_bridge=lambda_bridge,
            mu_redundancy=mu_redundancy,
            chain_gain_weight=chain_gain_weight,
            role_balance_weight=role_balance_weight,
            role_balance_max_gain_per_step=role_balance_max_gain_per_step,
            role_balance_max_token_jaccard=role_balance_max_token_jaccard,
            role_balance_missing_only=role_balance_missing_only,
            role_redundancy_relax=role_redundancy_relax,
            role_redundancy_max_overlap=role_redundancy_max_overlap,
            answerability_weight=answerability_weight,
        )
        selection_mode = "greedy"

    if not selected_indices:
        diag["reason"] = "selector_empty"
        return base_ids, base_texts, diag

    selected_atoms = [atoms[idx] for idx in selected_indices]
    ordered_atoms, ordering_type = _order_selected_atoms(selected_atoms)
    out_ids = [atom.sentence_id for atom in ordered_atoms]
    out_texts = [_normalize_space(atom.text) for atom in ordered_atoms]
    selected_tokens = int(sum(int(atom.token_count) for atom in ordered_atoms))
    selected_atom_count = max(1, int(len(ordered_atoms)))

    diag.update(
        {
            "applied": True,
            "mode": str(selection_mode),
            "num_input_candidates": int(len(base_ids)),
            "num_atoms": int(len(atoms)),
            "num_selected_atoms": int(len(ordered_atoms)),
            "selected_sentence_ids": list(out_ids),
            "selected_tokens": int(selected_tokens),
            "max_tokens": int(max_tokens),
            "objective": "A_gain + lambda*B_gain - mu*Redundancy under hard token budget",
            "A_gain_total": float(score_diag.get("A_gain_total", 0.0)),
            "B_gain_total": float(score_diag.get("B_gain_total", 0.0)),
            "chain_gain_total_avg": float(score_diag.get("chain_gain_total", 0.0)),
            "role_coverage_gain_total_avg": float(score_diag.get("role_coverage_gain_total", 0.0)),
            "role_coverage_gain_weighted_total_avg": float(
                score_diag.get("role_coverage_gain_weighted_total", 0.0)
            ),
            "redundancy_total": float(score_diag.get("redundancy_total", 0.0)),
            "redundancy_raw_total": float(score_diag.get("redundancy_raw_total", 0.0)),
            "redundancy_before_avg": float(score_diag.get("redundancy_before_total", 0.0)) / float(selected_atom_count),
            "redundancy_after_avg": float(score_diag.get("redundancy_after_total", 0.0)) / float(selected_atom_count),
            "role_aware_redundancy_applied_avg": float(
                score_diag.get("role_aware_redundancy_applied_total", 0.0)
            ),
            "role_balance_activated_count": int(score_diag.get("role_balance_activated_count", 0)),
            "role_balance_rejected_count": int(score_diag.get("role_balance_rejected_count", 0)),
            "role_balance_rejected_reasons": dict(score_diag.get("role_balance_rejected_reasons", {}) or {}),
            "selected_role_counts": dict(score_diag.get("selected_role_counts", {}) or {}),
            "redundancy_relaxed_count": int(score_diag.get("redundancy_recalibrated_count", 0)),
            "redundancy_recalibrated_count": int(score_diag.get("redundancy_recalibrated_count", 0)),
            "redundancy_recalibration_reasons": dict(
                score_diag.get("redundancy_recalibration_reasons", {}) or {}
            ),
            "redundancy_recalibration_applied_avg": float(
                score_diag.get("redundancy_recalibrated_count", 0)
            ) / float(selected_atom_count),
            "score_total": float(score_diag.get("score_total", 0.0)),
            "objective_eval_calls": int(score_diag.get("objective_eval_calls", 0)),
            "selection_ms": float((time.perf_counter() - started) * 1000.0),
            "answerability_feature_ms": float(score_diag.get("answerability_feature_ms", 0.0)),
            "bridge_feature_ms": float(score_diag.get("bridge_feature_ms", 0.0)),
            "redundancy_scoring_ms": float(score_diag.get("redundancy_scoring_ms", 0.0)),
            "score_trace": list(score_diag.get("score_trace", [])),
            "ordering_type": str(ordering_type),
            "use_answerability_gain": bool(use_answerability_gain),
            "use_bridge_gain": bool(use_bridge_gain),
            "use_redundancy_penalty": bool(use_redundancy_penalty),
            "use_cost_penalty": bool(use_cost_penalty),
            "chain_aware_enabled": bool(chain_aware_enabled),
            "role_aware_redundancy_enabled": bool(role_aware_redundancy_enabled),
            "role_balanced_enabled": bool(role_balanced_enabled),
            "redundancy_recalibrated_enabled": bool(redundancy_recalibrated_enabled),
            "chain_gain_weight": float(chain_gain_weight),
            "role_balance_weight": float(role_balance_weight),
            "role_balance_max_gain_per_step": float(role_balance_max_gain_per_step),
            "role_balance_max_token_jaccard": float(role_balance_max_token_jaccard),
            "role_balance_missing_only": bool(role_balance_missing_only),
            "role_redundancy_relax": float(role_redundancy_relax),
            "role_redundancy_max_overlap": float(role_redundancy_max_overlap),
            "hard_token_budget_enabled": bool(hard_budget_enabled),
        }
    )
    return out_ids, out_texts, diag
