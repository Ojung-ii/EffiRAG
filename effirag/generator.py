from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from .registry import register_generator
from .types import GenerationResult, RenderedContext, Sample
from .utils import content_tokens

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_PREFIX_RE = re.compile(r"^\s*(answer|final answer)\s*:\s*", re.IGNORECASE)
_HF_PIPELINE_CACHE = {}
_HF_PIPELINE_ERRORS = {}
_HF_VERBOSITY_SET = False
_OPENAI_CLIENT_CACHE = {}


def _uses_max_completion_tokens(model_name: str) -> bool:
    """HippoRAG2-style token arg compatibility: GPT models prefer max_completion_tokens.

    Some OpenAI-compatible backends (e.g., vLLM OpenAI server) still expect max_tokens.
    We handle fallback at call time if needed.
    """
    return "gpt" in str(model_name or "").lower()


def _build_openai_chat_params(model_name: str, messages: list, max_new_tokens: int) -> dict:
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
    }
    if _uses_max_completion_tokens(model_name):
        params["max_completion_tokens"] = int(max_new_tokens)
    else:
        params["max_tokens"] = int(max_new_tokens)
    return params


def _retry_with_max_tokens_if_needed(params: dict, err: Exception) -> dict:
    """If backend rejects max_completion_tokens, retry with max_tokens."""
    text = str(err or "")
    if "max_completion_tokens" not in text:
        raise err
    patched = dict(params)
    if "max_completion_tokens" in patched:
        patched["max_tokens"] = patched.pop("max_completion_tokens")
    return patched


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


def _extract_chat_message_content(message_obj) -> str:
    if message_obj is None:
        return ""
    content = getattr(message_obj, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = str(item.get("text", "") or "").strip()
            else:
                txt = str(getattr(item, "text", "") or "").strip()
            if txt:
                parts.append(txt)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _extract_openai_usage_metadata(response_obj) -> dict:
    meta = {}
    if response_obj is None:
        return meta

    usage = getattr(response_obj, "usage", None)
    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        try:
            meta["prompt_tokens"] = int(prompt_tokens or 0)
        except Exception:
            meta["prompt_tokens"] = 0
        try:
            meta["completion_tokens"] = int(completion_tokens or 0)
        except Exception:
            meta["completion_tokens"] = 0

    choices = getattr(response_obj, "choices", None)
    if choices:
        first = choices[0]
        finish_reason = getattr(first, "finish_reason", "")
        if not finish_reason and isinstance(first, dict):
            finish_reason = str(first.get("finish_reason", "") or "")
        meta["finish_reason"] = str(finish_reason or "")
    return meta


def _prefers_text_generation(model_name: str) -> bool:
    lower = str(model_name or "").lower()
    causal_markers = ("qwen", "llama", "mistral", "deepseek", "phi", "gemma")
    return any(tag in lower for tag in causal_markers)


def _lightweight_interface_instruction(rendered: RenderedContext) -> str:
    meta = dict((getattr(rendered, "metadata", {}) or {}))
    enabled = bool(meta.get("raw_focus_scaffold_light_enabled", False))
    if not enabled:
        return ""
    text = str(meta.get("raw_focus_scaffold_light_text", "") or "").strip()
    if not text:
        text = "Use the earliest evidence chain that links entity, bridge, and answer."
    return " ".join(text.split())


def _generation_intervention_instruction(rendered: RenderedContext) -> str:
    meta = dict((getattr(rendered, "metadata", {}) or {}))
    enabled = bool(meta.get("generation_intervention_enabled", False))
    if not enabled:
        return ""
    text = str(meta.get("generation_intervention_text", "") or "").strip()
    return " ".join(text.split())


def _build_qa_prompt(sample: Sample, rendered: RenderedContext) -> str:
    parts = [
        "You are a QA assistant.",
        "Use only the provided context.",
        "Return only the final answer span.",
        "Do not output reasoning, explanations, or <think> tags.",
        "If the question is yes/no, output exactly yes or no.",
    ]
    intervention_instruction = _generation_intervention_instruction(rendered)
    if intervention_instruction:
        parts.append(intervention_instruction)
    else:
        light_instruction = _lightweight_interface_instruction(rendered)
        if light_instruction:
            parts.append(light_instruction)
    parts.append(f"Question: {sample.question}")
    parts.append(f"Context:\n{rendered.text}")
    parts.append("Final answer:")
    return "\n".join(parts)


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


def _get_openai_client(base_url: str, api_key: str, timeout_sec: float):
    key = (str(base_url or "").strip(), str(api_key or "").strip(), float(timeout_sec))
    if key in _OPENAI_CLIENT_CACHE:
        return _OPENAI_CLIENT_CACHE[key]

    from openai import OpenAI

    kwargs = {"api_key": str(api_key or "").strip() or os.getenv("OPENAI_API_KEY", "EMPTY")}
    if str(base_url or "").strip():
        kwargs["base_url"] = str(base_url).strip()
    if float(timeout_sec) > 0:
        kwargs["timeout"] = float(timeout_sec)
    client = OpenAI(**kwargs)
    _OPENAI_CLIENT_CACHE[key] = client
    return client


def _openai_http_chat_completion(base_url: str, api_key: str, params: dict, timeout_sec: float):
    root = str(base_url or "").strip().rstrip("/")
    if not root:
        raise RuntimeError("llm_base_url is required when openai package is not installed.")
    if not root.endswith("/v1"):
        root = root + "/v1"
    url = root + "/chat/completions"
    req = urllib.request.Request(
        url=url,
        data=json.dumps(params).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {str(api_key or '').strip() or 'EMPTY'}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_sec))) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"http_error {exc.code}: {detail[:280]}") from exc
    except Exception as exc:
        raise RuntimeError(f"http_request_failed: {exc}") from exc

    choices = payload.get("choices", []) if isinstance(payload, dict) else []
    metadata = {}
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    if isinstance(usage, dict):
        metadata["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
        metadata["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
    if not choices:
        return "", metadata
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    if isinstance(choices[0], dict):
        metadata["finish_reason"] = str(choices[0].get("finish_reason", "") or "")
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content.strip(), metadata
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = str(item.get("text", "") or "").strip()
                if txt:
                    parts.append(txt)
        return "\n".join(parts).strip(), metadata
    return str(content or "").strip(), metadata


@register_generator("heuristic")
def generate_heuristic(sample: Sample, rendered: RenderedContext, model_name: str = "", cfg=None) -> GenerationResult:
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
def generate_oracle(sample: Sample, rendered: RenderedContext, model_name: str = "", cfg=None) -> GenerationResult:
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
def generate_hf(sample: Sample, rendered: RenderedContext, model_name: str = "", cfg=None) -> GenerationResult:
    start = time.perf_counter()

    try:
        resolved_model = model_name or "google/flan-t5-small"
        prompt = _build_qa_prompt(sample=sample, rendered=rendered)
        max_new_tokens = int(getattr(cfg, "llm_max_new_tokens", 64) if cfg is not None else 64)
        prefer_text_gen = _prefers_text_generation(resolved_model)
        if prefer_text_gen:
            qa_pipe = _get_hf_pipeline("text-generation", resolved_model)
            out = qa_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
            raw_text = out[0]["generated_text"].strip() if out else ""
        else:
            try:
                qa_pipe = _get_hf_pipeline("text2text-generation", resolved_model)
                out = qa_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
                raw_text = out[0]["generated_text"].strip() if out else ""
            except Exception:
                qa_pipe = _get_hf_pipeline("text-generation", resolved_model)
                out = qa_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
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


@register_generator("openai_compat")
def generate_openai_compat(sample: Sample, rendered: RenderedContext, model_name: str = "", cfg=None) -> GenerationResult:
    start = time.perf_counter()

    try:
        resolved_model = model_name or "Qwen/Qwen2.5-7B-Instruct"
        base_url = str(getattr(cfg, "llm_base_url", "") or "").strip() if cfg is not None else ""
        api_key = str(getattr(cfg, "llm_api_key", "") or "").strip() if cfg is not None else ""
        timeout_sec = float(getattr(cfg, "llm_timeout_sec", 120.0) if cfg is not None else 120.0)
        max_new_tokens = int(getattr(cfg, "llm_max_new_tokens", 64) if cfg is not None else 64)

        prompt = _build_qa_prompt(sample=sample, rendered=rendered)
        messages = [{"role": "user", "content": prompt}]
        generation_meta = {}

        params = _build_openai_chat_params(
            model_name=resolved_model,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )

        try:
            client = _get_openai_client(base_url=base_url, api_key=api_key, timeout_sec=timeout_sec)
            try:
                response = client.chat.completions.create(**params)
            except Exception as api_exc:
                retry_params = _retry_with_max_tokens_if_needed(params, api_exc)
                response = client.chat.completions.create(**retry_params)
            raw_text = ""
            generation_meta = _extract_openai_usage_metadata(response)
            if getattr(response, "choices", None):
                raw_text = _extract_chat_message_content(getattr(response.choices[0], "message", None))
        except Exception as sdk_exc:
            if "No module named 'openai'" not in str(sdk_exc):
                raise
            try:
                raw_text, generation_meta = _openai_http_chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    params=params,
                    timeout_sec=timeout_sec,
                )
            except Exception as http_exc:
                retry_params = _retry_with_max_tokens_if_needed(params, http_exc)
                raw_text, generation_meta = _openai_http_chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    params=retry_params,
                    timeout_sec=timeout_sec,
                )
        prediction = _postprocess_prediction(raw_text, sample.question)
        text = raw_text
    except Exception as exc:
        fallback = generate_heuristic(sample, rendered, model_name=model_name, cfg=cfg)
        prediction = fallback.prediction
        text = f"OpenAI-compatible generation failed: {exc}\nFallback(heuristic): {prediction}"
        generation_meta = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "fallback_heuristic",
            "fallback_used": True,
        }

    latency_ms = (time.perf_counter() - start) * 1000.0
    return GenerationResult(
        sample_id=sample.qid,
        generator="openai_compat",
        model_name=model_name,
        prediction=prediction,
        raw_text=text,
        latency_ms=latency_ms,
        metadata=generation_meta,
    )


@register_generator("vllm")
def generate_vllm_compat(sample: Sample, rendered: RenderedContext, model_name: str = "", cfg=None) -> GenerationResult:
    """Alias of openai_compat for user-facing vLLM terminology."""
    result = generate_openai_compat(sample, rendered, model_name=model_name, cfg=cfg)
    return GenerationResult(
        sample_id=result.sample_id,
        generator="vllm",
        model_name=result.model_name,
        prediction=result.prediction,
        raw_text=result.raw_text,
        latency_ms=result.latency_ms,
        metadata=dict(result.metadata or {}),
    )
