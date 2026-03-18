import json
from pathlib import Path
from typing import List

from .registry import register_dataset
from .types import ContextDocument, Sample


def _demo_hotpot_samples():
    return [
        Sample(
            qid="demo-1",
            question="Which city is known as the City of Light and is the capital of France?",
            answer="Paris",
            contexts=[
                ContextDocument(
                    title="Paris",
                    sentences=[
                        "Paris is the capital and most populous city of France.",
                        "Paris is often called the City of Light.",
                    ],
                ),
                ContextDocument(
                    title="France",
                    sentences=[
                        "France is a country in Western Europe.",
                        "Its capital city is Paris.",
                    ],
                ),
            ],
            supporting_facts=[("Paris", 0), ("Paris", 1)],
        ),
        Sample(
            qid="demo-2",
            question="What is the chemical symbol of the element with atomic number 8?",
            answer="O",
            contexts=[
                ContextDocument(
                    title="Oxygen",
                    sentences=[
                        "Oxygen is the chemical element with atomic number 8.",
                        "The chemical symbol for oxygen is O.",
                    ],
                ),
                ContextDocument(
                    title="Hydrogen",
                    sentences=[
                        "Hydrogen has the symbol H.",
                        "Hydrogen has atomic number 1.",
                    ],
                ),
            ],
            supporting_facts=[("Oxygen", 0), ("Oxygen", 1)],
        ),
    ]


def _parse_hotpot_record(rec):
    qid = str(rec.get("_id") or rec.get("id") or rec.get("qid") or "")
    question = rec.get("question", "")
    answer = rec.get("answer", "")

    contexts = []
    raw_context = rec.get("context", [])
    if isinstance(raw_context, dict):
        # HF hotpot_qa schema: {"title": [...], "sentences": [[...], ...]}
        titles = raw_context.get("title", [])
        sentences = raw_context.get("sentences", [])
        for title, sents in zip(titles, sentences):
            contexts.append(
                ContextDocument(
                    title=str(title),
                    sentences=[str(x) for x in (sents or [])],
                )
            )
    elif isinstance(raw_context, list):
        # Local schema: [[title, [sent1, sent2]], ...]
        for c in raw_context:
            if not isinstance(c, list) or len(c) != 2:
                continue
            title, sents = c
            contexts.append(ContextDocument(title=str(title), sentences=[str(x) for x in (sents or [])]))

    supporting_facts = []
    raw_supporting_facts = rec.get("supporting_facts", [])
    if isinstance(raw_supporting_facts, dict):
        # HF hotpot_qa schema: {"title": [...], "sent_id": [...]}
        titles = raw_supporting_facts.get("title", [])
        sent_ids = raw_supporting_facts.get("sent_id", [])
        for title, sent_idx in zip(titles, sent_ids):
            supporting_facts.append((str(title), int(sent_idx)))
    elif isinstance(raw_supporting_facts, list):
        for sf in raw_supporting_facts:
            if not isinstance(sf, list) or len(sf) != 2:
                continue
            title, sent_idx = sf
            supporting_facts.append((str(title), int(sent_idx)))

    return Sample(
        qid=qid,
        question=str(question),
        answer=str(answer),
        contexts=contexts,
        supporting_facts=supporting_facts,
        metadata={"type": rec.get("type", "")},
    )


def _load_hotpot_from_path(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    rows = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            rows = data.get("data", [])
        else:
            rows = data

    return [_parse_hotpot_record(r) for r in rows]


def _load_hotpot_hf(split):
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    return [_parse_hotpot_record(row) for row in ds]


@register_dataset("hotpotqa")
def load_hotpotqa(
    split="validation",
    limit=None,
    data_path=None,
):
    if data_path:
        samples = _load_hotpot_from_path(data_path)
    else:
        try:
            samples = _load_hotpot_hf(split=split)
        except Exception:
            samples = _demo_hotpot_samples()

    if limit is not None:
        samples = samples[: max(0, limit)]
    return samples
