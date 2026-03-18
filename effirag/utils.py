import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "who",
    "what",
    "when",
    "where",
    "which",
    "how",
}


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_tokens(text):
    return [t for t in tokenize(text) if len(t) > 2 and t not in STOPWORDS]


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(mean(values))


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def timestamp_for_filename():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def timestamp_iso_utc():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def markdown_table(headers, rows):
    safe_headers = [str(h) for h in headers]
    lines = [
        "| " + " | ".join(safe_headers) + " |",
        "| " + " | ".join(["---"] * len(safe_headers)) + " |",
    ]
    for row in rows:
        safe_row = [str(cell) for cell in row]
        lines.append("| " + " | ".join(safe_row) + " |")
    return "\n".join(lines)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def load_yaml(path):
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in yaml file: {path}")
    return data
