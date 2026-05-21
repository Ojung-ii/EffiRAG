from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, Optional, Set, Tuple


_TITLE_SENT_RE = re.compile(
    r"^(?P<title>.+?)::(?:(?:sent(?:ence)?(?:_idx)?|idx)\s*=\s*)?(?P<idx>-?\d+)$",
    re.IGNORECASE,
)
_KV_TITLE_RE = re.compile(r"(?:^|[|;,])\s*title\s*[:=]\s*(?P<title>[^|;,]+)", re.IGNORECASE)
_KV_SENT_RE = re.compile(
    r"(?:^|[|;,])\s*(?:sent(?:ence)?(?:_idx)?|idx)\s*[:=]\s*(?P<idx>-?\d+)",
    re.IGNORECASE,
)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def normalize_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def canonical_support_key(title: Any, sent_idx: Any) -> str:
    t = normalize_title(title)
    i = _safe_int(sent_idx, -1)
    if not t or i < 0:
        return ""
    return f"{t}::{i}"


def split_evidence_id(value: Any) -> Tuple[str, Optional[int]]:
    if isinstance(value, dict):
        title = value.get("title", value.get("doc_title", ""))
        sent_idx = value.get("sent_idx", value.get("sentence_idx", value.get("idx", None)))
        idx = _safe_int(sent_idx, -1)
        if str(title or "").strip() and idx >= 0:
            return str(title), int(idx)

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return "", None

    m = _TITLE_SENT_RE.match(text)
    if m:
        return m.group("title"), _safe_int(m.group("idx"), -1)

    mt = _KV_TITLE_RE.search(text)
    mi = _KV_SENT_RE.search(text)
    if mt and mi:
        return mt.group("title"), _safe_int(mi.group("idx"), -1)

    parts = text.split("/")
    if len(parts) >= 2 and re.fullmatch(r"-?\d+", parts[-1] or ""):
        title = parts[-2]
        idx = _safe_int(parts[-1], -1)
        if title and idx >= 0:
            return title, idx

    return text, None


def normalize_evidence_id(value: Any) -> str:
    title, sent_idx = split_evidence_id(value)
    if title and sent_idx is not None and sent_idx >= 0:
        return canonical_support_key(title, sent_idx)
    raw = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"\s+", " ", raw).casefold() if raw else ""


def normalized_evidence_set(values: Iterable[Any]) -> Set[str]:
    out: Set[str] = set()
    for value in list(values or []):
        nid = normalize_evidence_id(value)
        if nid:
            out.add(nid)
    return out


def coverage_stats(gold_ids: Set[str], stage_ids: Set[str]) -> Dict[str, float]:
    g = set(gold_ids or set())
    s = set(stage_ids or set())
    if not g:
        return {
            "gold_count": 0.0,
            "gold_in_stage": 0.0,
            "partial_hit": 0.0,
            "full_coverage": 0.0,
            "gold_recall": 0.0,
        }
    inter = g.intersection(s)
    gtot = float(len(g))
    gint = float(len(inter))
    return {
        "gold_count": gtot,
        "gold_in_stage": gint,
        "partial_hit": 1.0 if gint > 0.0 else 0.0,
        "full_coverage": 1.0 if gint >= gtot else 0.0,
        "gold_recall": float(gint / gtot),
    }

