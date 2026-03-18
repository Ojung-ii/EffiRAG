from __future__ import annotations

import time

from .registry import register_generator
from .types import GenerationResult, RenderedContext, Sample
from .utils import content_tokens


@register_generator("heuristic")
def generate_heuristic(sample: Sample, rendered: RenderedContext, model_name: str = "") -> GenerationResult:
    start = time.perf_counter()

    q_tokens = set(content_tokens(sample.question))
    best_sentence = ""
    best_overlap = -1

    for sent in rendered.sentences:
        s_tokens = set(content_tokens(sent))
        overlap = len(q_tokens.intersection(s_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sent

    prediction = best_sentence if best_sentence else (rendered.sentences[0] if rendered.sentences else "")

    latency_ms = (time.perf_counter() - start) * 1000.0
    return GenerationResult(
        sample_id=sample.qid,
        generator="heuristic",
        model_name=model_name,
        prediction=prediction,
        raw_text=prediction,
        latency_ms=latency_ms,
    )


@register_generator("oracle")
def generate_oracle(sample: Sample, rendered: RenderedContext, model_name: str = "") -> GenerationResult:
    start = time.perf_counter()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return GenerationResult(
        sample_id=sample.qid,
        generator="oracle",
        model_name=model_name,
        prediction=sample.answer,
        raw_text=sample.answer,
        latency_ms=latency_ms,
    )


@register_generator("hf")
def generate_hf(sample: Sample, rendered: RenderedContext, model_name: str = "") -> GenerationResult:
    start = time.perf_counter()

    try:
        from transformers import pipeline

        resolved_model = model_name or "google/flan-t5-small"
        qa_pipe = pipeline("text2text-generation", model=resolved_model)
        prompt = (
            "Answer the question using only the provided context.\n"
            f"Question: {sample.question}\n"
            f"Context:\n{rendered.text}\n"
            "Answer:"
        )
        out = qa_pipe(prompt, max_new_tokens=64, do_sample=False)
        text = out[0]["generated_text"].strip() if out else ""
        prediction = text
    except Exception as exc:
        prediction = ""
        text = f"HF generation failed: {exc}"

    latency_ms = (time.perf_counter() - start) * 1000.0
    return GenerationResult(
        sample_id=sample.qid,
        generator="hf",
        model_name=model_name,
        prediction=prediction,
        raw_text=text,
        latency_ms=latency_ms,
    )
