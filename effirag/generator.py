from __future__ import annotations

import re
import time

from .registry import register_generator
from .types import GenerationResult, RenderedContext, Sample
from .utils import content_tokens

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_PREFIX_RE = re.compile(r"^\s*(answer|final answer)\s*:\s*", re.IGNORECASE)
_HF_PIPELINE_CACHE = {}
_HF_PIPELINE_ERRORS = {}
_HF_VERBOSITY_SET = False


def _hf_accel_kwargs():
    try:
        import torch
    except Exception:
        return {}

    if not torch.cuda.is_available():
        return {}

    dtype = torch.float16
    try:
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
    except Exception:
        pass
    return {"device_map": "auto", "torch_dtype": dtype}


def _postprocess_prediction(raw_text: str, question: str) -> str:
    text = raw_text or ""
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = text.replace("</s>", " ").strip()

    if "Answer:" in text:
        text = text.rsplit("Answer:", 1)[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    text = _ANSWER_PREFIX_RE.sub("", text).strip().strip("\"'` ")

    q = (question or "").strip().lower()
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ")):
        low = text.lower()
        if low.startswith("yes"):
            return "yes"
        if low.startswith("no"):
            return "no"

    return text


def _get_hf_pipeline(task: str, model_name: str):
    global _HF_VERBOSITY_SET
    key = (task, model_name)
    if key in _HF_PIPELINE_CACHE:
        return _HF_PIPELINE_CACHE[key]
    if key in _HF_PIPELINE_ERRORS:
        raise RuntimeError(_HF_PIPELINE_ERRORS[key])

    from transformers import pipeline

    if not _HF_VERBOSITY_SET:
        # Reduce noisy generation warnings in CLI logs for toy experiments.
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:
            pass
        _HF_VERBOSITY_SET = True

    errors = []
    accel_kwargs = _hf_accel_kwargs()
    attempts = []
    if accel_kwargs:
        attempts.append({"model": model_name, **accel_kwargs})
        attempts.append({"model": model_name, "tokenizer": model_name, "local_files_only": True, **accel_kwargs})
    attempts.extend(
        [
            {"model": model_name},
            # When network is flaky, prefer already-downloaded local cache.
            {"model": model_name, "tokenizer": model_name, "local_files_only": True},
        ]
    )
    for kwargs in attempts:
        try:
            pipe = pipeline(task, **kwargs)
            _HF_PIPELINE_CACHE[key] = pipe
            return pipe
        except Exception as exc:
            errors.append(exc)

    message = str(errors[-1]) if errors else f"Failed to create HF pipeline: task={task}, model={model_name}"
    _HF_PIPELINE_ERRORS[key] = message
    raise RuntimeError(message)


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
        resolved_model = model_name or "google/flan-t5-small"
        prompt = (
            "You are a QA assistant.\n"
            "Use only the provided context.\n"
            "Return only the final answer span.\n"
            "Do not output reasoning, explanations, or <think> tags.\n"
            "If the question is yes/no, output exactly yes or no.\n"
            f"Question: {sample.question}\n"
            f"Context:\n{rendered.text}\n"
            "Final answer:"
        )
        task = "text2text-generation"
        try:
            qa_pipe = _get_hf_pipeline(task, resolved_model)
            out = qa_pipe(prompt, max_new_tokens=64, do_sample=False)
            raw_text = out[0]["generated_text"].strip() if out else ""
        except Exception:
            # Fallback for causal LMs (e.g., Qwen-family) that support text-generation only.
            task = "text-generation"
            qa_pipe = _get_hf_pipeline(task, resolved_model)
            out = qa_pipe(prompt, max_new_tokens=64, do_sample=False, return_full_text=False)
            raw_text = out[0]["generated_text"].strip() if out else ""

        prediction = _postprocess_prediction(raw_text, sample.question)
        text = raw_text
    except Exception as exc:
        fallback = generate_heuristic(sample, rendered, model_name=model_name)
        prediction = fallback.prediction
        text = f"HF generation failed: {exc}\nFallback(heuristic): {prediction}"

    latency_ms = (time.perf_counter() - start) * 1000.0
    return GenerationResult(
        sample_id=sample.qid,
        generator="hf",
        model_name=model_name,
        prediction=prediction,
        raw_text=text,
        latency_ms=latency_ms,
    )
