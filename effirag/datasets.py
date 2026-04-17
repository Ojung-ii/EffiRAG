import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .registry import register_dataset
from .types import ContextDocument, Sample


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _split_sentences(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return pieces or [text]


def _normalize_sentences(value) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return _split_sentences(value)
    return []


def _dedupe_supporting_facts(facts: Iterable[Tuple[str, int]]) -> List[Tuple[str, int]]:
    uniq = []
    seen = set()
    for title, sent_idx in facts:
        key = (str(title), int(sent_idx))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _parse_possible_answers(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [text]
    return [str(raw).strip()]


def _pick_answer(rec: dict) -> str:
    for key in ("answer", "obj"):
        if key in rec and str(rec.get(key) or "").strip():
            return str(rec[key]).strip()

    for key in ("possible_answers", "answers", "answer_aliases", "o_aliases"):
        vals = _parse_possible_answers(rec.get(key))
        if vals:
            return vals[0]
    return ""


def _parse_supporting_facts(raw_supporting_facts) -> List[Tuple[str, int]]:
    supporting_facts = []
    if isinstance(raw_supporting_facts, dict):
        titles = raw_supporting_facts.get("title", [])
        sent_ids = (
            raw_supporting_facts.get("sent_id")
            or raw_supporting_facts.get("sent_ids")
            or raw_supporting_facts.get("sentence_id")
            or raw_supporting_facts.get("sentence_ids")
            or []
        )
        for title, sent_idx in zip(titles, sent_ids):
            supporting_facts.append((str(title), _safe_int(sent_idx)))
        return _dedupe_supporting_facts(supporting_facts)

    if isinstance(raw_supporting_facts, list):
        for sf in raw_supporting_facts:
            title = None
            sent_idx = 0
            if isinstance(sf, list) and len(sf) >= 2:
                title, sent_idx = sf[0], sf[1]
            elif isinstance(sf, dict):
                title = sf.get("title") or sf.get("doc_title") or sf.get("document_title")
                sent_idx = sf.get("sent_id", sf.get("sent_idx", sf.get("sentence_id", sf.get("index", 0))))
            if title is None:
                continue
            supporting_facts.append((str(title), _safe_int(sent_idx)))
    return _dedupe_supporting_facts(supporting_facts)


def _parse_contexts(raw_context) -> List[ContextDocument]:
    contexts = []

    if isinstance(raw_context, dict):
        titles = raw_context.get("title", [])
        sentences = raw_context.get("sentences", [])
        for title, sents in zip(titles, sentences):
            norm = _normalize_sentences(sents)
            if norm:
                contexts.append(ContextDocument(title=str(title), sentences=norm))
        return contexts

    if not isinstance(raw_context, list):
        return contexts

    for idx, item in enumerate(raw_context):
        title = None
        sentences = []

        if isinstance(item, list) and len(item) >= 2:
            title = str(item[0])
            sentences = _normalize_sentences(item[1])
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("document_title") or item.get("name") or f"doc-{idx}")
            for key in ("sentences", "paragraph_text", "paragraph", "text", "context", "sentence_text"):
                if key in item:
                    sentences = _normalize_sentences(item.get(key))
                    if sentences:
                        break

        if title and sentences:
            contexts.append(ContextDocument(title=title, sentences=sentences))

    return contexts


def _supporting_from_paragraph_flags(paragraphs, contexts: List[ContextDocument]) -> List[Tuple[str, int]]:
    if not isinstance(paragraphs, list):
        return []

    sent_count_by_title: Dict[str, int] = {}
    for doc in contexts:
        sent_count_by_title[doc.title] = len(doc.sentences)

    facts = []
    for idx, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            continue

        is_supporting = paragraph.get("is_supporting", paragraph.get("supporting", paragraph.get("is_support", False)))
        if not bool(is_supporting):
            continue

        title = str(paragraph.get("title") or paragraph.get("document_title") or f"doc-{idx}")

        sent_ids = None
        for key in (
            "supporting_sentence_ids",
            "supporting_sent_ids",
            "supporting_sent_idxs",
            "sent_ids",
            "sentence_ids",
            "sent_id",
            "sentence_id",
        ):
            if key not in paragraph:
                continue
            raw = paragraph.get(key)
            if isinstance(raw, list):
                sent_ids = [_safe_int(x) for x in raw]
            else:
                sent_ids = [_safe_int(raw)]
            break

        if sent_ids is None:
            n_sents = sent_count_by_title.get(title, 0)
            if n_sents <= 0:
                n_sents = 1
            sent_ids = list(range(n_sents))

        for sent_idx in sent_ids:
            facts.append((title, sent_idx))

    return _dedupe_supporting_facts(facts)


def _load_records_from_path(path):
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
            rows = data.get("data", data.get("examples", []))
        else:
            rows = data

    return rows


def _discover_local_data_path(dataset_name: str):
    candidates = [
        Path.cwd() / "data" / "qa" / f"{dataset_name}.json",
        Path.cwd() / "data" / f"{dataset_name}.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _load_hf_rows(candidates, split: str):
    from datasets import load_dataset

    split_candidates = []
    for s in [split, "validation", "dev", "test", "train"]:
        if s and s not in split_candidates:
            split_candidates.append(s)

    last_error = None
    for dataset_name, subset_name in candidates:
        for split_name in split_candidates:
            try:
                if subset_name is None:
                    ds = load_dataset(dataset_name, split=split_name)
                else:
                    ds = load_dataset(dataset_name, subset_name, split=split_name)
                return list(ds), dataset_name, subset_name, split_name
            except Exception as exc:
                last_error = exc

    if last_error is None:
        raise RuntimeError("No Hugging Face dataset candidate could be evaluated.")
    raise last_error


def _ensure_qid(sample: Sample, fallback_prefix: str, index: int) -> Sample:
    if str(sample.qid).strip():
        return sample
    sample.qid = f"{fallback_prefix}-{index}"
    return sample


def _parse_hotpot_record(rec):
    qid = str(rec.get("_id") or rec.get("id") or rec.get("qid") or "")
    question = rec.get("question", "")
    answer = rec.get("answer", "")

    contexts = _parse_contexts(rec.get("context", []))
    supporting_facts = _parse_supporting_facts(rec.get("supporting_facts", []))

    return Sample(
        qid=qid,
        question=str(question),
        answer=str(answer),
        contexts=contexts,
        supporting_facts=supporting_facts,
        metadata={"type": rec.get("type", "")},
    )


def _parse_musique_record(rec):
    qid = str(rec.get("id") or rec.get("_id") or rec.get("qid") or "")
    question = str(rec.get("question", ""))
    answer = _pick_answer(rec)

    contexts = _parse_contexts(rec.get("context", []))
    paragraphs = rec.get("paragraphs", [])
    if not contexts and paragraphs:
        contexts = _parse_contexts(paragraphs)

    supporting_facts = _parse_supporting_facts(rec.get("supporting_facts", []))
    if not supporting_facts:
        supporting_facts = _supporting_from_paragraph_flags(paragraphs, contexts)

    return Sample(
        qid=qid,
        question=question,
        answer=answer,
        contexts=contexts,
        supporting_facts=supporting_facts,
        metadata={
            "answerable": rec.get("answerable", True),
            "question_decomposition": rec.get("question_decomposition", []),
        },
    )


def _parse_2wikimultihopqa_record(rec):
    base = _parse_hotpot_record(rec)
    base.metadata["answer_id"] = rec.get("answer_id", "")
    base.metadata["type"] = rec.get("type", base.metadata.get("type", ""))
    return base


def _parse_popqa_record(rec):
    qid = str(rec.get("id") or rec.get("_id") or rec.get("qid") or "")
    question = str(rec.get("question", ""))
    answer = _pick_answer(rec)

    contexts = _parse_contexts(rec.get("context", []))
    paragraphs = rec.get("paragraphs", rec.get("ctxs", []))
    if not contexts and paragraphs:
        contexts = _parse_contexts(paragraphs)

    if not contexts:
        subject = str(rec.get("subj") or rec.get("s_wiki_title") or "")
        relation = str(rec.get("prop") or "")
        obj = str(rec.get("obj") or answer or "")
        synthetic = " ".join([x for x in [subject, relation, obj] if x]).strip()
        if synthetic:
            contexts = [ContextDocument(title=subject or "popqa", sentences=[synthetic])]

    supporting_facts = _parse_supporting_facts(rec.get("supporting_facts", []))
    if not supporting_facts:
        supporting_facts = _supporting_from_paragraph_flags(paragraphs, contexts)

    return Sample(
        qid=qid,
        question=question,
        answer=answer,
        contexts=contexts,
        supporting_facts=supporting_facts,
        metadata={
            "subject": rec.get("subj", ""),
            "property": rec.get("prop", ""),
            "object": rec.get("obj", ""),
        },
    )


def _demo_hotpot_samples():
    return [
        Sample(
            qid="hotpot-demo-1",
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
            metadata={"source": "demo"},
        )
    ]


def _demo_musique_samples():
    return [
        Sample(
            qid="musique-demo-1",
            question="Which city is in France and known as the City of Light?",
            answer="Paris",
            contexts=[
                ContextDocument(
                    title="Paris",
                    sentences=[
                        "Paris is the capital city of France.",
                        "Paris is known as the City of Light.",
                    ],
                )
            ],
            supporting_facts=[("Paris", 0), ("Paris", 1)],
            metadata={"source": "demo"},
        )
    ]


def _demo_2wiki_samples():
    return [
        Sample(
            qid="2wiki-demo-1",
            question="Which country has Paris as its capital?",
            answer="France",
            contexts=[
                ContextDocument(title="Paris", sentences=["Paris is the capital of France."]),
                ContextDocument(title="France", sentences=["France is a country in Europe."]),
            ],
            supporting_facts=[("Paris", 0)],
            metadata={"source": "demo"},
        )
    ]


def _demo_popqa_samples():
    return [
        Sample(
            qid="popqa-demo-1",
            question="What is the occupation of George Rankin?",
            answer="politician",
            contexts=[
                ContextDocument(
                    title="George Rankin",
                    sentences=["George Rankin was an Australian soldier and politician."],
                )
            ],
            supporting_facts=[("George Rankin", 0)],
            metadata={"source": "demo"},
        )
    ]


def _load_dataset_generic(
    dataset_name: str,
    parser,
    split: str,
    limit,
    data_path,
    hf_candidates,
    demo_factory,
):
    source = "local"
    rows = None

    if data_path:
        rows = _load_records_from_path(data_path)
    else:
        discovered = _discover_local_data_path(dataset_name)
        if discovered:
            rows = _load_records_from_path(discovered)
            source = "auto_local"
        else:
            try:
                rows, hf_dataset, hf_subset, hf_split = _load_hf_rows(hf_candidates, split=split)
                source = "hf"
            except Exception:
                rows = None

    if rows is None:
        samples = demo_factory()
        if limit is not None:
            samples = samples[: max(0, limit)]
        for sample in samples:
            sample.metadata.setdefault("dataset", dataset_name)
            sample.metadata.setdefault("source", "demo")
        return samples

    samples = []
    for i, row in enumerate(rows):
        sample = _ensure_qid(parser(row), dataset_name, i)
        sample.metadata.setdefault("dataset", dataset_name)
        if source == "hf":
            sample.metadata.setdefault("source", "hf")
            sample.metadata.setdefault("hf_dataset", hf_dataset)
            sample.metadata.setdefault("hf_subset", hf_subset or "")
            sample.metadata.setdefault("hf_split", hf_split)
        else:
            sample.metadata.setdefault("source", source)
        samples.append(sample)

    if limit is not None:
        samples = samples[: max(0, limit)]
    return samples


@register_dataset("hotpotqa")
def load_hotpotqa(split="validation", limit=None, data_path=None):
    return _load_dataset_generic(
        dataset_name="hotpotqa",
        parser=_parse_hotpot_record,
        split=split,
        limit=limit,
        data_path=data_path,
        hf_candidates=[("hotpot_qa", "distractor")],
        demo_factory=_demo_hotpot_samples,
    )


@register_dataset("musique")
def load_musique(split="validation", limit=None, data_path=None):
    return _load_dataset_generic(
        dataset_name="musique",
        parser=_parse_musique_record,
        split=split,
        limit=limit,
        data_path=data_path,
        hf_candidates=[
            ("dgslibisey/MuSiQue", None),
            ("musique", None),
        ],
        demo_factory=_demo_musique_samples,
    )


@register_dataset("2wikimultihopqa")
@register_dataset("twowikimultihopqa")
def load_2wikimultihopqa(split="validation", limit=None, data_path=None):
    return _load_dataset_generic(
        dataset_name="2wikimultihopqa",
        parser=_parse_2wikimultihopqa_record,
        split=split,
        limit=limit,
        data_path=data_path,
        hf_candidates=[
            ("scholarly-shadows-syndicate/2wikimultihopqa", None),
            ("framolfese/2WikiMultihopQA", None),
            ("2wikimultihopqa", None),
        ],
        demo_factory=_demo_2wiki_samples,
    )


@register_dataset("popqa")
def load_popqa(split="test", limit=None, data_path=None):
    return _load_dataset_generic(
        dataset_name="popqa",
        parser=_parse_popqa_record,
        split=split,
        limit=limit,
        data_path=data_path,
        hf_candidates=[
            ("akariasai/PopQA", None),
            ("ibm/popqa-tp", "popqa-tp"),
            ("popqa", None),
        ],
        demo_factory=_demo_popqa_samples,
    )
