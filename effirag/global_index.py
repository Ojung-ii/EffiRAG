import ast
import hashlib
import json
import os
import pickle
import re
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.lib.format import open_memmap

from .embedding import encode_texts
from .utils import content_tokens, timestamp_iso_utc

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CODE_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.IGNORECASE | re.DOTALL)
SEMANTIC_TEXT_CONSTRUCTION_VERSION = "entity_alias_context_v1__chunk_title_passage_v1"
SEMANTIC_NORMALIZATION_VERSION = "l2_unit_norm"
SEMANTIC_DATA_ARTIFACT_NAMES = [
    "semantic_entities_ids.json",
    "semantic_entities_embeddings.f16.npy",
    "semantic_chunks_ids.json",
    "semantic_chunks_embeddings.f16.npy",
    "entity_chunk_support_map.json",
    "entity_topk_chunks_cache.json",
    "chunk_topk_entities_cache.json",
]


def _uses_max_completion_tokens(model_name):
    # Align with HippoRAG2 behavior: GPT models may use max_completion_tokens.
    # Some OpenAI-compatible backends (e.g., vLLM server) still require max_tokens.
    return "gpt" in str(model_name or "").lower()


def _build_openai_chat_params(model_name, messages, max_new_tokens):
    params = {
        "model": str(model_name or ""),
        "messages": messages,
        "temperature": 0.0,
    }
    if _uses_max_completion_tokens(model_name):
        params["max_completion_tokens"] = int(max_new_tokens)
    else:
        params["max_tokens"] = int(max_new_tokens)
    return params


def _retry_with_max_tokens_if_needed(params, err):
    text = str(err or "")
    if "max_completion_tokens" not in text:
        raise err
    patched = dict(params)
    if "max_completion_tokens" in patched:
        patched["max_tokens"] = patched.pop("max_completion_tokens")
    return patched


def _resolve_local_hf_snapshot(model_name):
    raw = str(model_name or "").strip()
    if not raw:
        return "", False

    candidate = Path(raw).expanduser()
    if candidate.exists():
        return str(candidate.resolve()), True

    if "/" not in raw:
        return raw, False
    org, repo = raw.split("/", 1)
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}" / "snapshots"
    if not hub_dir.exists():
        return raw, False

    snapshots = [p for p in hub_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return raw, False
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(snapshots[0].resolve()), True


def _openie_sentence_key(doc_idx, sent_idx, text):
    digest = hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:16]
    return f"{int(doc_idx)}:{int(sent_idx)}:{digest}"


def _load_openie_sentence_cache(path, show_progress=True):
    p = Path(path)
    if not p.exists():
        return {}
    cache = {}
    with p.open("r", encoding="utf-8") as f:
        for line in tqdm(
            f,
            desc="Load OpenIE cache",
            unit="line",
            leave=True,
            disable=not show_progress,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = str(row.get("key", "")).strip()
            if not key:
                continue
            triples = []
            for item in row.get("triples", []) or []:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    subj = str(item[0] or "").strip()
                    pred = str(item[1] or "").strip()
                    obj = str(item[2] or "").strip()
                    if subj and pred and obj:
                        triples.append((subj, pred, obj))
            cache[key] = {
                "parsed": bool(row.get("parsed", True)),
                "triples": triples,
            }
    return cache


def _write_openie_sentence_cache(path, cache, show_progress=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        keys = sorted(cache.keys())
        for key in tqdm(
            keys,
            total=len(keys),
            desc="Persist OpenIE cache",
            unit="sent",
            leave=True,
            disable=not show_progress,
        ):
            item = cache[key]
            payload = {
                "key": key,
                "parsed": bool(item.get("parsed", True)),
                "triples": [list(t) for t in (item.get("triples", []) or [])],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _split_sentences(text):
    text = str(text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def _normalize_corpus_rows(data):
    if isinstance(data, dict):
        rows = data.get("docs", data.get("data", data.get("examples", [])))
    else:
        rows = data

    if not isinstance(rows, list):
        raise ValueError("Corpus file must contain a list-like payload.")

    normalized = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or row.get("name") or f"doc-{idx}").strip()
        raw_text = row.get("text", row.get("context", row.get("passage", "")))
        if isinstance(raw_text, list):
            text = " ".join(str(x).strip() for x in raw_text if str(x).strip())
        else:
            text = str(raw_text or "").strip()
        if not text:
            continue
        doc_idx = row.get("idx", idx)
        try:
            doc_idx = int(doc_idx)
        except Exception:
            doc_idx = idx
        normalized.append({"idx": doc_idx, "title": title, "text": text})
    return normalized


def load_corpus_rows(path, show_progress=True):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in tqdm(
                f,
                desc="Load corpus(jsonl)",
                unit="line",
                leave=True,
                disable=not show_progress,
            ):
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return _normalize_corpus_rows(rows)

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize_corpus_rows(data)


def _entity_tokens(text):
    toks = content_tokens(text)
    if toks:
        return toks
    raw = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", str(text or ""))]
    return [t for t in raw if len(t) > 1][:4]


def _ensure_entity_node(g, token):
    token = str(token or "").strip().lower()
    if not token:
        return None
    node = f"e::{token}"
    if node not in g:
        g.add_node(node, node_type="entity", token=token)
    return node


def _extract_json_candidates(raw_text):
    text = str(raw_text or "").strip()
    if not text:
        return []

    candidates = [text]
    for match in CODE_FENCE_RE.finditer(text):
        block = str(match.group(1) or "").strip()
        if block:
            candidates.append(block)

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if 0 <= start_obj < end_obj:
        candidates.append(text[start_obj : end_obj + 1])

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if 0 <= start_arr < end_arr:
        candidates.append(text[start_arr : end_arr + 1])

    ordered = []
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        ordered.append(cand)
    return ordered


def _try_parse_payload(text):
    txt = str(text or "").strip()
    if not txt:
        return None

    try:
        return json.loads(txt)
    except Exception:
        pass

    try:
        return ast.literal_eval(txt)
    except Exception:
        return None


def _normalize_triple_item(item):
    subj = ""
    pred = ""
    obj = ""

    if isinstance(item, (list, tuple)) and len(item) >= 3:
        subj, pred, obj = item[0], item[1], item[2]
    elif isinstance(item, dict):
        subj = item.get("subject", item.get("head", item.get("s", "")))
        pred = item.get("predicate", item.get("relation", item.get("p", "")))
        obj = item.get("object", item.get("tail", item.get("o", "")))
    elif isinstance(item, str):
        parts = [p.strip() for p in re.split(r"\s*\|\s*|\s*;\s*|\s*,\s*", item) if p.strip()]
        if len(parts) >= 3:
            subj, pred, obj = parts[0], parts[1], parts[2]

    subj = str(subj or "").strip()
    pred = str(pred or "").strip()
    obj = str(obj or "").strip()
    if not subj or not pred or not obj:
        return None
    return subj, pred, obj


def _normalize_triples_from_payload(payload):
    if isinstance(payload, dict):
        items = None
        for key in ("triples", "relations", "facts"):
            val = payload.get(key)
            if isinstance(val, list):
                items = val
                break
        if items is None:
            return None
    elif isinstance(payload, list):
        items = payload
    else:
        return None

    triples = []
    seen = set()
    for item in items:
        triple = _normalize_triple_item(item)
        if triple is None:
            continue
        norm = tuple(str(x).strip() for x in triple)
        if norm in seen:
            continue
        seen.add(norm)
        triples.append(norm)
    return triples


def _parse_llm_triples(raw_text):
    saw_json_payload = False
    for cand in _extract_json_candidates(raw_text):
        payload = _try_parse_payload(cand)
        if payload is None:
            continue
        saw_json_payload = True
        triples = _normalize_triples_from_payload(payload)
        if triples is not None:
            return triples, True
    return [], saw_json_payload


def _extract_chat_message_content(message_obj):
    content = getattr(message_obj, "content", "")
    if isinstance(content, str):
        return content
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


def _openai_http_chat_completion(base_url, api_key, params, timeout_sec):
    root = str(base_url or "").strip().rstrip("/")
    if not root:
        raise RuntimeError("openie_api_base_url is required for HTTP OpenAI-compatible calls.")
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
    if not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = str(item.get("text", "") or "").strip()
                if txt:
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _attach_relation_triples(g, doc_idx, sent_idx, sentence_id, triples):
    added_relations = 0
    for tri_idx, triple in enumerate(triples):
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        subj, pred, obj = str(triple[0]), str(triple[1]), str(triple[2])
        rel_node = f"r::{int(doc_idx)}:{int(sent_idx)}:{int(tri_idx)}"
        sent_node = f"s::{int(doc_idx)}:{int(sent_idx)}"
        g.add_node(
            rel_node,
            node_type="relation",
            predicate=pred,
            subject=subj,
            object=obj,
            doc_idx=int(doc_idx),
            sent_idx=int(sent_idx),
            sentence_id=str(sentence_id),
        )
        g.add_edge(sent_node, rel_node, edge_type="expressed_in")
        added_relations += 1

        for tok in _entity_tokens(subj):
            ent = _ensure_entity_node(g, tok)
            if ent is not None:
                g.add_edge(rel_node, ent, edge_type="triple_subject")
        for tok in _entity_tokens(obj):
            ent = _ensure_entity_node(g, tok)
            if ent is not None:
                g.add_edge(rel_node, ent, edge_type="triple_object")
    return added_relations


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


class _LLMTripleExtractor:
    def __init__(self, model_name, text_max_chars, max_new_tokens, local_files_only, retry_attempts, retry_backoff_sec):
        self.model_name = str(model_name or "").strip() or "Qwen/Qwen2.5-7B-Instruct"
        self.model_ref, self.model_ref_is_local = _resolve_local_hf_snapshot(self.model_name)
        self.text_max_chars = int(text_max_chars)
        self.max_new_tokens = int(max_new_tokens)
        self.local_files_only = bool(local_files_only)
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.task = ""
        self.pipe = None
        self.available = False
        self.error_message = ""
        self._init_pipeline()

    def _init_pipeline(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from transformers.utils import logging as hf_logging
        except Exception as exc:
            self.error_message = f"transformers_import_failed: {exc}"
            return

        try:
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        attempts = []
        if self.model_ref_is_local:
            attempts.append({"model_ref": self.model_ref, "local_only": True})
        if self.local_files_only:
            attempts.append({"model_ref": self.model_name, "local_only": True})
        else:
            attempts.append({"model_ref": self.model_name, "local_only": False})
        unique_attempts = []
        seen = set()
        for att in attempts:
            key = (att["model_ref"], bool(att["local_only"]))
            if key in seen:
                continue
            seen.add(key)
            unique_attempts.append(att)

        errors = []
        for att in unique_attempts:
            model_ref = att["model_ref"]
            local_only = bool(att["local_only"])
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_ref,
                    local_files_only=local_only,
                    trust_remote_code=True,
                )
                model_kwargs = _hf_accel_kwargs()
                model = AutoModelForCausalLM.from_pretrained(
                    model_ref,
                    local_files_only=local_only,
                    trust_remote_code=True,
                    **model_kwargs,
                )
                self.pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                self.task = "text-generation"
                self.available = True
                return
            except Exception as exc:
                errors.append(f"text-generation(local_only={local_only}, model_ref={model_ref}): {exc}")

        self.error_message = errors[-1] if errors else "hf_pipeline_init_failed"

    @staticmethod
    def _prompt(text):
        return (
            "Extract factual relation triples from the passage.\n"
            "Return strict JSON only: {\"triples\": [[\"subject\", \"predicate\", \"object\"], ...]}.\n"
            "If no valid triple exists, return {\"triples\": []}.\n"
            "Passage:\n"
            f"{text}\n"
            "JSON:"
        )

    def extract(self, text):
        if not self.available:
            return [], "", False, self.error_message or "openie_llm_unavailable"

        passage = str(text or "").strip()
        if not passage:
            return [], "", True, ""
        if self.text_max_chars > 0:
            passage = passage[: self.text_max_chars]

        prompt = self._prompt(passage)
        last_err = ""
        for attempt_idx in range(self.retry_attempts):
            try:
                out = self.pipe(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                )

                raw = ""
                if isinstance(out, list) and out:
                    first = out[0]
                    if isinstance(first, dict):
                        raw = str(first.get("generated_text", "")).strip()
                    else:
                        raw = str(first).strip()

                triples, parsed = _parse_llm_triples(raw)
                return triples, raw, parsed, ""
            except Exception as exc:
                last_err = str(exc)
                if attempt_idx < self.retry_attempts - 1 and self.retry_backoff_sec > 0.0:
                    time.sleep(self.retry_backoff_sec)
        return [], "", False, last_err or "openie_runtime_error"


class _OpenAICompatTripleExtractor:
    def __init__(
        self,
        model_name,
        text_max_chars,
        max_new_tokens,
        base_url,
        api_key,
        retry_attempts,
        retry_backoff_sec,
        timeout_sec,
    ):
        self.model_name = str(model_name or "").strip() or "Qwen/Qwen2.5-7B-Instruct"
        self.text_max_chars = int(text_max_chars)
        self.max_new_tokens = int(max_new_tokens)
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or os.getenv("OPENAI_API_KEY", "EMPTY"))
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.available = False
        self.error_message = ""
        self.client = None
        self.client_mode = "none"
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
        except Exception as exc:
            if self.base_url:
                self.client_mode = "http"
                self.available = True
                self.error_message = f"openai_import_failed_fallback_http: {exc}"
                return
            self.error_message = f"openai_import_failed: {exc}"
            return

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout_sec > 0:
            kwargs["timeout"] = self.timeout_sec

        try:
            self.client = OpenAI(**kwargs)
            self.available = True
            self.client_mode = "sdk"
        except Exception as exc:
            self.error_message = f"openai_client_init_failed: {exc}"

    @staticmethod
    def _prompt_messages(passage):
        system = (
            "Extract factual relation triples.\n"
            "Return strict JSON only: {\"triples\": [[\"subject\", \"predicate\", \"object\"], ...]}.\n"
            "If no valid triple exists, return {\"triples\": []}."
        )
        user = f"Passage:\n{passage}\nJSON:"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def extract(self, text):
        if not self.available:
            return [], "", False, self.error_message or "openie_llm_api_unavailable"

        passage = str(text or "").strip()
        if not passage:
            return [], "", True, ""
        if self.text_max_chars > 0:
            passage = passage[: self.text_max_chars]

        params = {
            "model": self.model_name,
            "messages": self._prompt_messages(passage),
            "temperature": 0.0,
        }
        params = _build_openai_chat_params(
            model_name=self.model_name,
            messages=params["messages"],
            max_new_tokens=self.max_new_tokens,
        )

        last_err = ""
        for attempt_idx in range(self.retry_attempts):
            try:
                if self.client_mode == "sdk" and self.client is not None:
                    try:
                        resp = self.client.chat.completions.create(**params)
                    except Exception as api_exc:
                        retry_params = _retry_with_max_tokens_if_needed(params, api_exc)
                        resp = self.client.chat.completions.create(**retry_params)
                    raw = ""
                    if getattr(resp, "choices", None):
                        msg = getattr(resp.choices[0], "message", None)
                        raw = _extract_chat_message_content(msg)
                else:
                    try:
                        raw = _openai_http_chat_completion(
                            base_url=self.base_url,
                            api_key=self.api_key,
                            params=params,
                            timeout_sec=self.timeout_sec,
                        )
                    except Exception as http_exc:
                        retry_params = _retry_with_max_tokens_if_needed(params, http_exc)
                        raw = _openai_http_chat_completion(
                            base_url=self.base_url,
                            api_key=self.api_key,
                            params=retry_params,
                            timeout_sec=self.timeout_sec,
                        )
                triples, parsed = _parse_llm_triples(raw)
                return triples, raw, parsed, ""
            except Exception as exc:
                last_err = str(exc)
                if attempt_idx < self.retry_attempts - 1 and self.retry_backoff_sec > 0.0:
                    time.sleep(self.retry_backoff_sec)
        return [], "", False, last_err or "openie_llm_api_runtime_error"


def build_corpus_graph(
    corpus_rows,
    openie_mode="llm",
    openie_model_name="Qwen/Qwen2.5-7B-Instruct",
    openie_text_max_chars=2200,
    openie_max_new_tokens=256,
    openie_local_files_only=True,
    openie_retry_attempts=3,
    openie_retry_backoff_sec=0.2,
    openie_error_sample_limit=20,
    openie_api_base_url="",
    openie_api_key="",
    openie_api_timeout_sec=120.0,
    openie_parallel_workers=1,
    openie_log_every=200,
    openie_sentence_cache_path="",
    show_progress=True,
):
    g = nx.Graph()
    num_sentences = 0
    num_relation_nodes = 0
    num_triples = 0

    requested_openie_mode = str(openie_mode or "llm").strip().lower()
    if requested_openie_mode not in {"lexical", "llm"}:
        requested_openie_mode = "lexical"

    effective_openie_mode = requested_openie_mode
    openie_init_error = ""
    extractor = None
    llm_calls = 0
    llm_parse_failures = 0
    llm_zero_triple_sentences = 0
    llm_runtime_errors = 0
    llm_cache_hits = 0
    llm_cache_misses = 0
    llm_error_counter = {}
    llm_error_samples = []
    openie_sentence_cache = {}
    cache_dirty = False
    cache_path = str(openie_sentence_cache_path or "").strip()
    openie_backend = "lexical"
    openie_api_base_url = str(openie_api_base_url or "").strip()
    log_every = max(1, int(openie_log_every))

    def _log(msg):
        if show_progress:
            print(str(msg), flush=True)

    if requested_openie_mode == "llm":
        for _ in tqdm(
            [0],
            total=1,
            desc="Init OpenIE model",
            unit="step",
            leave=True,
            disable=not show_progress,
        ):
            if openie_api_base_url:
                extractor = _OpenAICompatTripleExtractor(
                    model_name=openie_model_name,
                    text_max_chars=openie_text_max_chars,
                    max_new_tokens=openie_max_new_tokens,
                    base_url=openie_api_base_url,
                    api_key=openie_api_key,
                    retry_attempts=openie_retry_attempts,
                    retry_backoff_sec=openie_retry_backoff_sec,
                    timeout_sec=openie_api_timeout_sec,
                )
                openie_backend = "openai_compat"
            else:
                extractor = _LLMTripleExtractor(
                    model_name=openie_model_name,
                    text_max_chars=openie_text_max_chars,
                    max_new_tokens=openie_max_new_tokens,
                    local_files_only=openie_local_files_only,
                    retry_attempts=openie_retry_attempts,
                    retry_backoff_sec=openie_retry_backoff_sec,
                )
                openie_backend = "hf_local"
        if not extractor.available:
            effective_openie_mode = "lexical"
            openie_init_error = str(extractor.error_message or "")
            openie_backend = "lexical"
            _log(f"[Index/OpenIE] fallback to lexical (init_error={openie_init_error[:180]})")
        if effective_openie_mode == "llm" and cache_path:
            openie_sentence_cache = _load_openie_sentence_cache(cache_path, show_progress=show_progress)
        if effective_openie_mode == "llm":
            _log(
                "[Index/OpenIE] "
                f"backend={openie_backend} model={openie_model_name} workers={max(1, int(openie_parallel_workers))} "
                f"timeout={float(openie_api_timeout_sec):.1f}s retries={int(openie_retry_attempts)} log_every={log_every}"
            )
            if openie_api_base_url:
                _log(f"[Index/OpenIE] endpoint={openie_api_base_url}")

    total_sentences_expected = None
    if show_progress:
        total_sentences_expected = sum(len(_split_sentences(str(row.get("text", "")))) for row in corpus_rows)
    sentence_iter = (
        (row, sent_idx, sent)
        for row in corpus_rows
        for sent_idx, sent in enumerate(_split_sentences(str(row.get("text", ""))))
    )
    sentence_records = []
    try:
        for row, sent_idx, sent in tqdm(
            sentence_iter,
            total=total_sentences_expected,
            desc="Build sentence/entity graph",
            unit="sent",
            leave=True,
            disable=not show_progress,
        ):
            doc_idx = int(row["idx"])
            title = str(row["title"])
            node = f"s::{doc_idx}:{sent_idx}"
            sentence_id = f"{title}::{sent_idx}"
            g.add_node(
                node,
                node_type="sentence",
                sentence_id=sentence_id,
                title=title,
                sent_idx=sent_idx,
                text=sent,
                doc_idx=doc_idx,
            )
            num_sentences += 1

            for tok in content_tokens(sent):
                entity_node = _ensure_entity_node(g, tok)
                if entity_node is None:
                    continue
                g.add_edge(
                    node,
                    entity_node,
                    edge_type="mentions",
                    support_layer="entity_chunk",
                )

            if effective_openie_mode == "llm" and extractor is not None:
                sentence_records.append(
                    {
                        "doc_idx": int(doc_idx),
                        "sent_idx": int(sent_idx),
                        "sentence_id": str(sentence_id),
                        "text": str(sent),
                    }
                )

        if effective_openie_mode == "llm" and extractor is not None:
            cached_items = []
            pending_items = []
            for rec in tqdm(
                sentence_records,
                total=len(sentence_records),
                desc="OpenIE cache lookup",
                unit="sent",
                leave=True,
                disable=not show_progress,
            ):
                cache_key = _openie_sentence_key(rec["doc_idx"], rec["sent_idx"], rec["text"])
                cached = openie_sentence_cache.get(cache_key)
                if cached is not None:
                    llm_cache_hits += 1
                    cached_items.append((rec, cache_key, cached))
                else:
                    llm_cache_misses += 1
                    pending_items.append((rec, cache_key))
            _log(
                f"[Index/OpenIE] targets={len(sentence_records)} cache_hit={len(cached_items)} pending={len(pending_items)}"
            )

            def _consume_openie_result(rec, cache_key, triples, parsed, err, from_cache):
                nonlocal llm_runtime_errors, llm_parse_failures, llm_zero_triple_sentences
                nonlocal num_relation_nodes, num_triples, cache_dirty
                if err:
                    llm_runtime_errors += 1
                    err_key = str(err).strip().splitlines()[0][:280]
                    llm_error_counter[err_key] = llm_error_counter.get(err_key, 0) + 1
                    if len(llm_error_samples) < max(0, int(openie_error_sample_limit)):
                        llm_error_samples.append(
                            {
                                "sentence_id": rec["sentence_id"],
                                "doc_idx": int(rec["doc_idx"]),
                                "sent_idx": int(rec["sent_idx"]),
                                "error": err_key,
                                "text_head": str(rec["text"])[:240],
                            }
                        )
                    return

                if not parsed:
                    llm_parse_failures += 1

                norm_triples = []
                for item in triples or []:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        norm_triples.append((str(item[0]), str(item[1]), str(item[2])))

                if (not from_cache) and (cache_key not in openie_sentence_cache):
                    openie_sentence_cache[cache_key] = {
                        "parsed": bool(parsed),
                        "triples": norm_triples,
                    }
                    cache_dirty = True

                if not norm_triples:
                    llm_zero_triple_sentences += 1
                    return

                added = _attach_relation_triples(
                    g=g,
                    doc_idx=rec["doc_idx"],
                    sent_idx=rec["sent_idx"],
                    sentence_id=rec["sentence_id"],
                    triples=norm_triples,
                )
                num_relation_nodes += int(added)
                num_triples += int(added)

            for rec, cache_key, cached in tqdm(
                cached_items,
                total=len(cached_items),
                desc="Attach cached OpenIE",
                unit="sent",
                leave=True,
                disable=not show_progress,
            ):
                triples = list(cached.get("triples", []) or [])
                parsed = bool(cached.get("parsed", True))
                _consume_openie_result(
                    rec=rec,
                    cache_key=cache_key,
                    triples=triples,
                    parsed=parsed,
                    err="",
                    from_cache=True,
                )

            llm_calls += int(len(pending_items))
            workers = max(1, int(openie_parallel_workers))
            openie_start = time.perf_counter()
            openie_done = 0

            def _heartbeat(force=False):
                nonlocal openie_done
                if not show_progress:
                    return
                if (not force) and (openie_done == 0 or (openie_done % log_every != 0)):
                    return
                elapsed = max(1e-9, time.perf_counter() - openie_start)
                speed = float(openie_done) / elapsed
                ok = max(0, int(openie_done - llm_runtime_errors))
                print(
                    "[Index/OpenIE] "
                    f"done={openie_done}/{len(pending_items)} ok={ok} err={llm_runtime_errors} "
                    f"parse_fail={llm_parse_failures} zero={llm_zero_triple_sentences} "
                    f"speed={speed:.2f} sent/s",
                    flush=True,
                )

            if workers <= 1:
                pbar = tqdm(
                    pending_items,
                    total=len(pending_items),
                    desc="OpenIE triples",
                    unit="sent",
                    leave=True,
                    disable=not show_progress,
                )
                for rec, cache_key in pbar:
                    triples, _, parsed, err = extractor.extract(rec["text"])
                    _consume_openie_result(
                        rec=rec,
                        cache_key=cache_key,
                        triples=triples,
                        parsed=parsed,
                        err=err,
                        from_cache=False,
                    )
                    openie_done += 1
                    _heartbeat(force=False)
                    if show_progress:
                        pbar.set_postfix(
                            {
                                "ok": int(len(openie_sentence_cache)),
                                "err": int(llm_runtime_errors),
                            }
                        )
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    fut_to_item = {
                        executor.submit(extractor.extract, rec["text"]): (rec, cache_key)
                        for rec, cache_key in pending_items
                    }
                    pbar = tqdm(
                        as_completed(fut_to_item),
                        total=len(fut_to_item),
                        desc=f"OpenIE triples (parallel={workers})",
                        unit="sent",
                        leave=True,
                        disable=not show_progress,
                    )
                    for fut in pbar:
                        rec, cache_key = fut_to_item[fut]
                        try:
                            triples, _, parsed, err = fut.result()
                        except Exception as exc:
                            triples, parsed, err = [], False, f"executor_error: {exc}"
                        _consume_openie_result(
                            rec=rec,
                            cache_key=cache_key,
                            triples=triples,
                            parsed=parsed,
                            err=err,
                            from_cache=False,
                        )
                        openie_done += 1
                        _heartbeat(force=False)
                        if show_progress:
                            pbar.set_postfix(
                                {
                                    "ok": int(len(openie_sentence_cache)),
                                    "err": int(llm_runtime_errors),
                                }
                            )
            _heartbeat(force=True)
            _log(
                f"[Index/OpenIE] completed calls={llm_calls} cache_hits={llm_cache_hits} "
                f"errors={llm_runtime_errors} triples={num_triples}"
            )
    finally:
        if effective_openie_mode == "llm" and cache_path and cache_dirty:
            _write_openie_sentence_cache(cache_path, openie_sentence_cache, show_progress=show_progress)

    num_entities = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "entity")
    num_relations = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "relation")
    num_docs = len({int(row["idx"]) for row in corpus_rows})
    num_support_edges = sum(1 for u, v in g.edges if str(g[u][v].get("support_layer", "")) == "entity_chunk")
    top_errors = sorted(llm_error_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    stats = {
        "num_docs": int(num_docs),
        "num_sentences": int(num_sentences),
        "num_entities": int(num_entities),
        "num_relation_nodes": int(num_relations or num_relation_nodes),
        "num_extracted_triples": int(num_triples),
        "num_nodes": int(g.number_of_nodes()),
        "num_edges": int(g.number_of_edges()),
        "num_entity_chunk_support_edges": int(num_support_edges),
        "openie_mode_requested": requested_openie_mode,
        "openie_mode_effective": effective_openie_mode,
        "openie_backend_effective": openie_backend,
        "openie_model_name": str(openie_model_name or ""),
        "openie_text_max_chars": int(openie_text_max_chars),
        "openie_max_new_tokens": int(openie_max_new_tokens),
        "openie_local_files_only": bool(openie_local_files_only),
        "openie_api_base_url": str(openie_api_base_url or ""),
        "openie_api_timeout_sec": float(openie_api_timeout_sec),
        "openie_parallel_workers": int(openie_parallel_workers),
        "openie_log_every": int(log_every),
        "openie_retry_attempts": int(openie_retry_attempts),
        "openie_retry_backoff_sec": float(openie_retry_backoff_sec),
        "openie_sentence_cache_path": str(cache_path),
        "openie_init_error": str(openie_init_error or ""),
        "llm_calls": int(llm_calls),
        "llm_cache_hits": int(llm_cache_hits),
        "llm_cache_misses": int(llm_cache_misses),
        "llm_parse_failures": int(llm_parse_failures),
        "llm_zero_triple_sentences": int(llm_zero_triple_sentences),
        "llm_runtime_errors": int(llm_runtime_errors),
        "llm_error_top": [{"error": str(msg), "count": int(cnt)} for msg, cnt in top_errors],
        "llm_error_samples": llm_error_samples,
    }
    return g, stats


def _fingerprint(path):
    p = Path(path).resolve()
    st = p.stat()
    return {
        "path": str(p),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _normalize_build_config(
    prebuilt_igraph_path,
    prebuilt_igraph_format,
    prebuilt_entity_token_limit,
    embedding_enabled,
    embedding_model_name,
    embedding_batch_size,
    embedding_max_length,
    embedding_text_max_chars,
    openie_mode,
    openie_model_name,
    openie_text_max_chars,
    openie_max_new_tokens,
    openie_local_files_only,
    openie_api_base_url,
    openie_api_timeout_sec,
    openie_retry_attempts,
    openie_retry_backoff_sec,
    openie_error_sample_limit,
):
    mode = str(openie_mode or "llm").strip().lower()
    if mode not in {"llm", "lexical"}:
        mode = "lexical"
    prebuilt_path = str(prebuilt_igraph_path or "").strip()
    if prebuilt_path:
        try:
            prebuilt_path = str(Path(prebuilt_path).resolve())
        except Exception:
            prebuilt_path = str(prebuilt_igraph_path or "")
    return {
        "prebuilt_igraph_path": prebuilt_path,
        "prebuilt_igraph_format": str(prebuilt_igraph_format or "hipporag_pickle"),
        "prebuilt_entity_token_limit": int(prebuilt_entity_token_limit),
        "embedding_enabled": bool(embedding_enabled),
        "embedding_model_name": str(embedding_model_name or ""),
        "embedding_batch_size": int(embedding_batch_size),
        "embedding_max_length": int(embedding_max_length),
        "embedding_text_max_chars": int(embedding_text_max_chars),
        "openie_mode": mode,
        "openie_model_name": str(openie_model_name or ""),
        "openie_text_max_chars": int(openie_text_max_chars),
        "openie_max_new_tokens": int(openie_max_new_tokens),
        "openie_local_files_only": bool(openie_local_files_only),
        "openie_api_base_url": str(openie_api_base_url or ""),
        "openie_api_timeout_sec": float(openie_api_timeout_sec),
        "openie_retry_attempts": int(openie_retry_attempts),
        "openie_retry_backoff_sec": float(openie_retry_backoff_sec),
        "openie_error_sample_limit": int(openie_error_sample_limit),
    }


def _index_dir(source_path, cache_dir, build_config):
    source_abs = str(Path(source_path).resolve()) if str(source_path or "").strip() else "<none>"
    signature = json.dumps(build_config, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha1(f"{source_abs}::{signature}".encode("utf-8")).hexdigest()[:12]
    stem = Path(source_path).stem if str(source_path or "").strip() else "prebuilt_graph"
    mode = str(build_config.get("openie_mode", "lexical"))
    return Path(cache_dir) / f"{stem}_{mode}_{key}"


def _normalize_semantic_build_config(build_config):
    cfg = dict(build_config or {})
    return {
        "embedding_enabled": bool(cfg.get("embedding_enabled", False)),
        "embedding_model_name": str(cfg.get("embedding_model_name", "") or ""),
        "embedding_max_length": int(cfg.get("embedding_max_length", 192)),
        "embedding_text_max_chars": int(cfg.get("embedding_text_max_chars", 600)),
        "text_construction_version": SEMANTIC_TEXT_CONSTRUCTION_VERSION,
        "normalization_version": SEMANTIC_NORMALIZATION_VERSION,
    }


def _semantic_cache_key(source_fingerprint, semantic_build_config):
    payload = {
        "source_fingerprint": dict(source_fingerprint or {}),
        "semantic_build_config": dict(semantic_build_config or {}),
    }
    signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]


def _semantic_node_signature_from_ids(entity_ids, chunk_ids):
    h = hashlib.sha1()
    for node_id in sorted([str(x) for x in (entity_ids or [])]):
        h.update(node_id.encode("utf-8"))
        h.update(b"\n")
    h.update(b"--")
    for node_id in sorted([str(x) for x in (chunk_ids or [])]):
        h.update(node_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:20]


def _graph_semantic_node_signature(g):
    entity_ids = []
    chunk_ids = []
    for node in g.nodes:
        node_type = str(g.nodes[node].get("node_type", "") or "")
        if node_type == "entity":
            entity_ids.append(str(node))
        elif node_type == "sentence":
            chunk_ids.append(str(node))
    return _semantic_node_signature_from_ids(entity_ids, chunk_ids)


def _semantic_data_artifacts_ready(index_dir):
    root = Path(index_dir)
    for name in SEMANTIC_DATA_ARTIFACT_NAMES:
        if not (root / name).exists():
            return False
    return True


def _materialize_semantic_meta_paths(
    *,
    index_dir,
    semantic_meta,
    semantic_cache_key,
    semantic_build_config,
    graph_semantic_node_signature,
    build_ms_override=None,
    reused_from_index_dir="",
):
    root = Path(index_dir).resolve()
    payload = dict(semantic_meta or {})
    payload["enabled"] = bool(payload.get("enabled", True))
    payload["index_dir"] = str(root)
    payload["entity_ids_path"] = str((root / "semantic_entities_ids.json").resolve())
    payload["entity_embeddings_path"] = str((root / "semantic_entities_embeddings.f16.npy").resolve())
    payload["chunk_ids_path"] = str((root / "semantic_chunks_ids.json").resolve())
    payload["chunk_embeddings_path"] = str((root / "semantic_chunks_embeddings.f16.npy").resolve())
    payload["entity_chunk_support_map_path"] = str((root / "entity_chunk_support_map.json").resolve())
    payload["entity_topk_chunks_cache_path"] = str((root / "entity_topk_chunks_cache.json").resolve())
    payload["chunk_topk_entities_cache_path"] = str((root / "chunk_topk_entities_cache.json").resolve())
    payload["semantic_build_config"] = dict(semantic_build_config or {})
    payload["semantic_cache_key"] = str(semantic_cache_key or "")
    payload["text_construction_version"] = str(
        payload.get("text_construction_version", SEMANTIC_TEXT_CONSTRUCTION_VERSION)
    )
    payload["normalization_version"] = str(payload.get("normalization_version", SEMANTIC_NORMALIZATION_VERSION))
    if graph_semantic_node_signature:
        payload["semantic_node_signature"] = str(graph_semantic_node_signature)
    if build_ms_override is not None:
        payload["build_ms"] = float(build_ms_override)
    if reused_from_index_dir:
        payload["reused_from_index_dir"] = str(reused_from_index_dir)
    return payload


def _link_or_copy_file(src, dst):
    src_p = Path(src).resolve()
    dst_p = Path(dst).resolve()
    if dst_p.exists():
        dst_p.unlink()
    try:
        os.link(str(src_p), str(dst_p))
    except Exception:
        shutil.copy2(str(src_p), str(dst_p))


def _try_reuse_semantic_artifacts(
    *,
    index_dir,
    cache_dir,
    source_id_path,
    source_fingerprint,
    semantic_build_config,
    semantic_cache_key,
    graph_semantic_node_signature,
    show_progress=True,
):
    info = {
        "semantic_reuse_attempted": bool(semantic_build_config.get("embedding_enabled", False)),
        "semantic_reuse_applied": False,
        "semantic_reuse_reason": "",
        "semantic_cache_key": str(semantic_cache_key or ""),
    }
    if not bool(semantic_build_config.get("embedding_enabled", False)):
        info["semantic_reuse_reason"] = "embedding_disabled"
        return None, info

    root = Path(index_dir)
    if _semantic_data_artifacts_ready(root) and (root / "semantic_index_meta.json").exists():
        local_meta = _read_json_obj(root / "semantic_index_meta.json")
        local_meta = _materialize_semantic_meta_paths(
            index_dir=root,
            semantic_meta=local_meta,
            semantic_cache_key=semantic_cache_key,
            semantic_build_config=semantic_build_config,
            graph_semantic_node_signature=graph_semantic_node_signature,
            build_ms_override=float(local_meta.get("build_ms", 0.0) or 0.0),
        )
        _write_json_obj(root / "semantic_index_meta.json", local_meta)
        info["semantic_reuse_reason"] = "already_present_locally"
        return local_meta, info

    cache_root = Path(cache_dir)
    if not cache_root.exists():
        info["semantic_reuse_reason"] = "cache_dir_not_found"
        return None, info

    stem = Path(str(source_id_path or "")).stem
    if not stem:
        info["semantic_reuse_reason"] = "empty_source_stem"
        return None, info

    candidates = []
    pattern = f"{stem}_*_*"
    for cand_dir in cache_root.glob(pattern):
        if not cand_dir.is_dir():
            continue
        if cand_dir.resolve() == root.resolve():
            continue
        meta_path = cand_dir / "meta.json"
        sem_meta_path = cand_dir / "semantic_index_meta.json"
        if (not meta_path.exists()) or (not sem_meta_path.exists()):
            continue
        if not _semantic_data_artifacts_ready(cand_dir):
            continue
        meta = _read_json_obj(meta_path)
        if not meta:
            continue
        if meta.get("fingerprint", {}) != dict(source_fingerprint or {}):
            continue
        sem_meta = _read_json_obj(sem_meta_path)
        if not sem_meta or not bool(sem_meta.get("enabled", False)):
            continue

        cand_key = str(sem_meta.get("semantic_cache_key", "") or "")
        if not cand_key:
            cand_sem_cfg = sem_meta.get("semantic_build_config", {}) if isinstance(sem_meta, dict) else {}
            if not cand_sem_cfg:
                cand_sem_cfg = _normalize_semantic_build_config(meta.get("build_config", {}) or {})
            cand_key = _semantic_cache_key(meta.get("fingerprint", {}), cand_sem_cfg)
        if cand_key != str(semantic_cache_key):
            continue

        cand_sig = str(sem_meta.get("semantic_node_signature", "") or "")
        if not cand_sig:
            ent_ids = _read_json_list(cand_dir / "semantic_entities_ids.json")
            chk_ids = _read_json_list(cand_dir / "semantic_chunks_ids.json")
            cand_sig = _semantic_node_signature_from_ids(ent_ids, chk_ids)
        if graph_semantic_node_signature and cand_sig != graph_semantic_node_signature:
            continue

        mtime = float(sem_meta_path.stat().st_mtime)
        candidates.append((mtime, cand_dir, sem_meta))

    if not candidates:
        info["semantic_reuse_reason"] = "no_matching_semantic_cache"
        return None, info

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, source_dir, source_sem_meta = candidates[0]
    root.mkdir(parents=True, exist_ok=True)
    for name in SEMANTIC_DATA_ARTIFACT_NAMES:
        _link_or_copy_file(source_dir / name, root / name)

    reused_meta = _materialize_semantic_meta_paths(
        index_dir=root,
        semantic_meta=source_sem_meta,
        semantic_cache_key=semantic_cache_key,
        semantic_build_config=semantic_build_config,
        graph_semantic_node_signature=graph_semantic_node_signature,
        build_ms_override=0.0,
        reused_from_index_dir=str(source_dir.resolve()),
    )
    _write_json_obj(root / "semantic_index_meta.json", reused_meta)

    info["semantic_reuse_applied"] = True
    info["semantic_reuse_reason"] = "copied_matching_semantic_cache"
    info["semantic_reuse_source"] = str(source_dir.resolve())
    if show_progress:
        print(
            "[Index/Semantic] "
            f"reuse semantic cache: {info['semantic_reuse_source']} -> {str(root.resolve())}",
            flush=True,
        )
    return reused_meta, info


def _maybe_seed_openie_cache_from_latest(
    *,
    cache_dir,
    source_id_path,
    target_cache_path,
    openie_mode,
    show_progress=True,
):
    mode = str(openie_mode or "").strip().lower()
    if mode != "llm":
        return {"openie_cache_seeded": False, "openie_cache_seed_reason": "mode_not_llm"}

    target = Path(target_cache_path)
    if target.exists() and target.stat().st_size > 0:
        return {
            "openie_cache_seeded": False,
            "openie_cache_seed_reason": "target_already_exists",
            "openie_cache_seed_target": str(target.resolve()),
        }

    root = Path(cache_dir)
    if not root.exists():
        return {"openie_cache_seeded": False, "openie_cache_seed_reason": "cache_dir_not_found"}

    stem = Path(str(source_id_path or "")).stem
    if not stem:
        return {"openie_cache_seeded": False, "openie_cache_seed_reason": "empty_source_stem"}

    pattern = f"{stem}_llm_*"
    candidates = []
    for idx_dir in root.glob(pattern):
        if not idx_dir.is_dir():
            continue
        cand = idx_dir / "openie_sentence_cache.jsonl"
        if not cand.exists():
            continue
        try:
            st = cand.stat()
        except Exception:
            continue
        if st.st_size <= 0:
            continue
        if cand.resolve() == target.resolve():
            continue
        candidates.append((int(st.st_size), float(st.st_mtime), cand))

    if not candidates:
        return {
            "openie_cache_seeded": False,
            "openie_cache_seed_reason": "no_candidate_cache_found",
            "openie_cache_seed_pattern": pattern,
        }

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, source = candidates[0]

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target))
    info = {
        "openie_cache_seeded": True,
        "openie_cache_seed_reason": "copied_latest_dataset_cache",
        "openie_cache_seed_source": str(source.resolve()),
        "openie_cache_seed_target": str(target.resolve()),
        "openie_cache_seed_pattern": pattern,
    }
    if show_progress:
        print(
            "[Index/OpenIE] "
            f"seed cache from latest dataset cache: {info['openie_cache_seed_source']} -> {info['openie_cache_seed_target']}",
            flush=True,
        )
    return info


def _read_gpickle(path):
    try:
        return nx.read_gpickle(path)
    except AttributeError:
        with Path(path).open("rb") as f:
            return pickle.load(f)


def _write_gpickle(graph, path):
    try:
        nx.write_gpickle(graph, path)
    except AttributeError:
        with Path(path).open("wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def _clean_prebuilt_entity_content(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"^\s*\d+\s+\d+\s*", "", raw)
    return raw.strip()


def _node_label(node_id):
    raw = str(node_id or "")
    if "::" in raw:
        return raw.split("::", 1)[1]
    if "-" in raw:
        return raw.split("-", 1)[1]
    return raw


def _collect_entity_support_texts(g, node_id, limit=2):
    texts = []
    seen = set()
    max_items = max(0, int(limit))
    if max_items <= 0:
        return texts

    def _push_sentence(sent_node):
        if sent_node not in g:
            return
        data = g.nodes[sent_node]
        if data.get("node_type") != "sentence":
            return
        text = str(data.get("text", data.get("content", "")) or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        texts.append(text)

    for nbr in g.neighbors(node_id):
        if len(texts) >= max_items:
            break
        nbr_type = g.nodes[nbr].get("node_type")
        if nbr_type == "sentence":
            _push_sentence(nbr)
        elif nbr_type == "relation":
            for nbr2 in g.neighbors(nbr):
                if len(texts) >= max_items:
                    break
                _push_sentence(nbr2)

    return texts[:max_items]


def _build_entity_semantic_text(g, node_id, support_limit=2):
    data = g.nodes[node_id]
    token = str(data.get("token", "") or "").strip()
    content = str(data.get("content", "") or "").strip()
    label = token or _clean_prebuilt_entity_content(content) or _node_label(node_id)
    label = str(label or "").strip()
    if not label:
        return ""

    aliases = []
    alias_seen = set()
    for nbr in g.neighbors(node_id):
        nbr_data = g.nodes[nbr]
        if nbr_data.get("node_type") != "entity":
            continue
        alias = str(nbr_data.get("token", "") or nbr_data.get("content", "") or _node_label(nbr)).strip()
        if not alias or alias == label or alias in alias_seen:
            continue
        alias_seen.add(alias)
        aliases.append(alias)
        if len(aliases) >= 4:
            break

    support = _collect_entity_support_texts(g, node_id, limit=support_limit)
    parts = [f"Entity: {label}"]
    if aliases:
        parts.append("Aliases: " + ", ".join(aliases))
    if support:
        parts.append("Context: " + " ".join(support))
    return "\n".join(parts).strip()


def _build_chunk_semantic_text(g, node_id):
    data = g.nodes[node_id]
    title = str(data.get("title", "") or "").strip()
    text = str(data.get("text", data.get("content", "")) or "").strip()
    if not text and not title:
        text = _node_label(node_id)
    if title:
        return f"Title: {title}\nPassage: {text}".strip()
    return str(text or "").strip()


def _collect_semantic_records(g):
    entity_records = []
    chunk_records = []
    for node_id in g.nodes:
        node_type = str(g.nodes[node_id].get("node_type", "") or "")
        if node_type == "entity":
            text = _build_entity_semantic_text(g, node_id)
            if text:
                entity_records.append({"node_id": str(node_id), "text": text})
        elif node_type == "sentence":
            text = _build_chunk_semantic_text(g, node_id)
            if text:
                chunk_records.append({"node_id": str(node_id), "text": text})
    return entity_records, chunk_records


def _write_json_list(path, items):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(list(items), f, ensure_ascii=False)


def _read_json_list(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return []


def _write_json_obj(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _read_json_obj(path):
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    return {}


def _build_entity_chunk_support_mapping(g):
    entity_to_chunks = {}
    chunk_to_entities = {}

    for u, v in g.edges:
        u_type = str(g.nodes[u].get("node_type", "") or "")
        v_type = str(g.nodes[v].get("node_type", "") or "")
        support_layer = str(g[u][v].get("support_layer", "") or "")

        is_support_edge = (support_layer == "entity_chunk") or ({u_type, v_type} == {"entity", "sentence"})
        if not is_support_edge:
            continue

        if u_type == "entity" and v_type == "sentence":
            entity = str(u)
            chunk = str(v)
        elif v_type == "entity" and u_type == "sentence":
            entity = str(v)
            chunk = str(u)
        else:
            continue

        entity_to_chunks.setdefault(entity, set()).add(chunk)
        chunk_to_entities.setdefault(chunk, set()).add(entity)

    entity_to_chunks = {
        key: sorted(list(vals))
        for key, vals in entity_to_chunks.items()
    }
    chunk_to_entities = {
        key: sorted(list(vals))
        for key, vals in chunk_to_entities.items()
    }
    return entity_to_chunks, chunk_to_entities


def _build_support_lookup_cache(mapping, topk=32):
    k = max(1, int(topk))
    cache = {}
    for key, values in (mapping or {}).items():
        seq = [str(v) for v in list(values or []) if str(v).strip()]
        cache[str(key)] = seq[:k]
    return cache


def _encode_records_to_files(
    records,
    index_dir,
    prefix,
    model_name,
    batch_size,
    max_length,
    max_chars,
    show_progress=True,
):
    ids_path = Path(index_dir) / f"{prefix}_ids.json"
    emb_path = Path(index_dir) / f"{prefix}_embeddings.f16.npy"
    ids = [str(r.get("node_id", "")) for r in records]
    texts = [str(r.get("text", "")) for r in records]

    if not records:
        np.save(emb_path, np.zeros((0, 0), dtype=np.float16))
        _write_json_list(ids_path, ids)
        return {
            "count": 0,
            "dim": 0,
            "ids_path": str(ids_path.resolve()),
            "embeddings_path": str(emb_path.resolve()),
        }

    bs = max(1, int(batch_size))
    first_end = min(len(texts), bs)
    first = encode_texts(
        texts=texts[:first_end],
        model_name=model_name,
        batch_size=bs,
        max_length=max_length,
        max_chars=max_chars,
        instruction="",
    )
    if not first.get("ok", False):
        raise RuntimeError(str(first.get("error", "semantic_embedding_encode_failed")))

    first_vecs = np.asarray(first.get("vectors", []), dtype=np.float32)
    if first_vecs.ndim != 2 or first_vecs.shape[0] <= 0:
        raise RuntimeError("semantic_embedding_empty_first_batch")
    dim = int(first_vecs.shape[1])
    mmap = open_memmap(str(emb_path), mode="w+", dtype=np.float16, shape=(len(records), dim))
    mmap[:first_vecs.shape[0], :] = first_vecs.astype(np.float16)

    starts = range(first_end, len(texts), bs)
    for start in tqdm(
        starts,
        total=(max(0, len(texts) - first_end) + bs - 1) // bs,
        desc=f"Encode {prefix}",
        unit="batch",
        leave=True,
        disable=not show_progress,
    ):
        end = min(len(texts), start + bs)
        out = encode_texts(
            texts=texts[start:end],
            model_name=model_name,
            batch_size=bs,
            max_length=max_length,
            max_chars=max_chars,
            instruction="",
        )
        if not out.get("ok", False):
            raise RuntimeError(str(out.get("error", "semantic_embedding_encode_failed")))
        vecs = np.asarray(out.get("vectors", []), dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != (end - start):
            raise RuntimeError("semantic_embedding_batch_shape_mismatch")
        mmap[start:end, :] = vecs.astype(np.float16)

    del mmap
    _write_json_list(ids_path, ids)
    return {
        "count": int(len(records)),
        "dim": int(dim),
        "ids_path": str(ids_path.resolve()),
        "embeddings_path": str(emb_path.resolve()),
    }


def _build_semantic_index_artifacts(
    g,
    index_dir,
    model_name,
    batch_size,
    max_length,
    max_chars,
    semantic_build_config=None,
    semantic_cache_key="",
    graph_semantic_node_signature="",
    show_progress=True,
):
    start = time.perf_counter()
    entity_records, chunk_records = _collect_semantic_records(g)
    entity_info = _encode_records_to_files(
        records=entity_records,
        index_dir=index_dir,
        prefix="semantic_entities",
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        max_chars=max_chars,
        show_progress=show_progress,
    )
    chunk_info = _encode_records_to_files(
        records=chunk_records,
        index_dir=index_dir,
        prefix="semantic_chunks",
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        max_chars=max_chars,
        show_progress=show_progress,
    )

    entity_to_chunks, chunk_to_entities = _build_entity_chunk_support_mapping(g)
    support_map_path = Path(index_dir) / "entity_chunk_support_map.json"
    support_map_payload = {
        "entity_to_chunks": entity_to_chunks,
        "chunk_to_entities": chunk_to_entities,
    }
    _write_json_obj(support_map_path, support_map_payload)

    entity_topk_cache_path = Path(index_dir) / "entity_topk_chunks_cache.json"
    chunk_topk_cache_path = Path(index_dir) / "chunk_topk_entities_cache.json"
    _write_json_obj(entity_topk_cache_path, _build_support_lookup_cache(entity_to_chunks, topk=32))
    _write_json_obj(chunk_topk_cache_path, _build_support_lookup_cache(chunk_to_entities, topk=32))

    dim = int(entity_info.get("dim", 0) or chunk_info.get("dim", 0))
    if entity_info.get("count", 0) > 0 and chunk_info.get("count", 0) > 0:
        if int(entity_info.get("dim", 0)) != int(chunk_info.get("dim", 0)):
            raise RuntimeError("semantic_embedding_dimension_mismatch_between_entity_and_chunk")

    payload = {
        "enabled": True,
        "index_dir": str(Path(index_dir).resolve()),
        "model_name": str(model_name or ""),
        "normalized": True,
        "dtype": "float16",
        "dim": int(dim),
        "entity_count": int(entity_info.get("count", 0)),
        "chunk_count": int(chunk_info.get("count", 0)),
        "entity_ids_path": str(entity_info.get("ids_path", "")),
        "entity_embeddings_path": str(entity_info.get("embeddings_path", "")),
        "chunk_ids_path": str(chunk_info.get("ids_path", "")),
        "chunk_embeddings_path": str(chunk_info.get("embeddings_path", "")),
        "entity_chunk_support_map_path": str(support_map_path.resolve()),
        "entity_topk_chunks_cache_path": str(entity_topk_cache_path.resolve()),
        "chunk_topk_entities_cache_path": str(chunk_topk_cache_path.resolve()),
        "entity_support_count": int(len(entity_to_chunks)),
        "chunk_support_count": int(len(chunk_to_entities)),
        "build_ms": float((time.perf_counter() - start) * 1000.0),
    }
    if not graph_semantic_node_signature:
        graph_semantic_node_signature = _semantic_node_signature_from_ids(
            [r.get("node_id", "") for r in entity_records],
            [r.get("node_id", "") for r in chunk_records],
        )
    payload = _materialize_semantic_meta_paths(
        index_dir=index_dir,
        semantic_meta=payload,
        semantic_cache_key=semantic_cache_key,
        semantic_build_config=semantic_build_config or {},
        graph_semantic_node_signature=graph_semantic_node_signature,
        build_ms_override=float(payload.get("build_ms", 0.0) or 0.0),
    )
    meta_path = Path(index_dir) / "semantic_index_meta.json"
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_semantic_index(semantic_meta, mmap_mode="r"):
    meta = dict(semantic_meta or {})
    if not bool(meta.get("enabled", False)):
        return None

    root_dir = str(meta.get("index_dir", "") or "").strip()
    root = Path(root_dir) if root_dir else None

    def _resolve(path_hint):
        raw = str(path_hint or "").strip()
        if not raw:
            return ""
        p = Path(raw)
        if p.exists():
            return str(p.resolve())
        if root is not None:
            q = (root / raw).resolve()
            if q.exists():
                return str(q)
        return raw

    entity_ids_path = _resolve(meta.get("entity_ids_path", ""))
    chunk_ids_path = _resolve(meta.get("chunk_ids_path", ""))
    entity_emb_path = _resolve(meta.get("entity_embeddings_path", ""))
    chunk_emb_path = _resolve(meta.get("chunk_embeddings_path", ""))
    support_map_path = _resolve(meta.get("entity_chunk_support_map_path", ""))
    entity_topk_cache_path = _resolve(meta.get("entity_topk_chunks_cache_path", ""))
    chunk_topk_cache_path = _resolve(meta.get("chunk_topk_entities_cache_path", ""))

    if not Path(entity_ids_path).exists() or not Path(chunk_ids_path).exists():
        return None
    if not Path(entity_emb_path).exists() or not Path(chunk_emb_path).exists():
        return None

    entity_ids = _read_json_list(entity_ids_path)
    chunk_ids = _read_json_list(chunk_ids_path)
    entity_embeddings = np.load(entity_emb_path, mmap_mode=mmap_mode)
    chunk_embeddings = np.load(chunk_emb_path, mmap_mode=mmap_mode)

    if int(len(entity_ids)) != int(len(entity_embeddings)):
        raise RuntimeError("semantic_entity_id_embedding_count_mismatch")
    if int(len(chunk_ids)) != int(len(chunk_embeddings)):
        raise RuntimeError("semantic_chunk_id_embedding_count_mismatch")

    entity_id_to_idx = {str(node_id): idx for idx, node_id in enumerate(entity_ids)}
    chunk_id_to_idx = {str(node_id): idx for idx, node_id in enumerate(chunk_ids)}
    support_map_payload = _read_json_obj(support_map_path) if support_map_path else {}
    entity_to_chunks = support_map_payload.get("entity_to_chunks", {}) if isinstance(support_map_payload, dict) else {}
    chunk_to_entities = support_map_payload.get("chunk_to_entities", {}) if isinstance(support_map_payload, dict) else {}
    entity_topk_cache = _read_json_obj(entity_topk_cache_path) if entity_topk_cache_path else {}
    chunk_topk_cache = _read_json_obj(chunk_topk_cache_path) if chunk_topk_cache_path else {}
    return {
        "meta": meta,
        "entity_ids": entity_ids,
        "chunk_ids": chunk_ids,
        "entity_embeddings": entity_embeddings,
        "chunk_embeddings": chunk_embeddings,
        "entity_id_to_idx": entity_id_to_idx,
        "chunk_id_to_idx": chunk_id_to_idx,
        "entity_to_chunks": entity_to_chunks if isinstance(entity_to_chunks, dict) else {},
        "chunk_to_entities": chunk_to_entities if isinstance(chunk_to_entities, dict) else {},
        "entity_topk_chunks_cache": entity_topk_cache if isinstance(entity_topk_cache, dict) else {},
        "chunk_topk_entities_cache": chunk_topk_cache if isinstance(chunk_topk_cache, dict) else {},
    }


def _build_graph_from_prebuilt_igraph(
    prebuilt_igraph_path,
    prebuilt_igraph_format="hipporag_pickle",
    prebuilt_entity_token_limit=6,
    show_progress=True,
):
    fmt = str(prebuilt_igraph_format or "hipporag_pickle").strip().lower()
    if fmt not in {"hipporag_pickle", "igraph_pickle"}:
        raise ValueError(f"Unsupported prebuilt igraph format: {prebuilt_igraph_format}")

    try:
        import igraph as ig
    except Exception as exc:
        raise RuntimeError(
            "python-igraph is required to load --prebuilt-igraph-path. "
            "Install with `pip install python-igraph` in the current environment."
        ) from exc

    src = Path(prebuilt_igraph_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    ig_graph = ig.Graph.Read_Pickle(str(src))
    names = list(ig_graph.vs["name"]) if "name" in ig_graph.vs.attributes() else [f"v::{i}" for i in range(ig_graph.vcount())]
    contents = list(ig_graph.vs["content"]) if "content" in ig_graph.vs.attributes() else ["" for _ in range(ig_graph.vcount())]

    g = nx.Graph()
    entity_nodes = []
    chunk_nodes = []

    for vid in tqdm(
        range(ig_graph.vcount()),
        total=ig_graph.vcount(),
        desc="Convert prebuilt KG nodes",
        unit="node",
        leave=True,
        disable=not show_progress,
    ):
        name = str(names[vid])
        content = str(contents[vid] or "")

        if name.startswith("chunk-"):
            g.add_node(
                name,
                node_type="sentence",
                sentence_id=name,
                title="prebuilt_chunk",
                sent_idx=0,
                text=content,
                source_node_type="chunk",
            )
            chunk_nodes.append(name)
        elif name.startswith("entity-"):
            cleaned = _clean_prebuilt_entity_content(content)
            token_hint = content_tokens(cleaned)
            g.add_node(
                name,
                node_type="entity",
                token=(token_hint[0] if token_hint else ""),
                content=content,
                source_node_type="entity",
            )
            entity_nodes.append(name)
        else:
            g.add_node(
                name,
                node_type="entity",
                token="",
                content=content,
                source_node_type="other",
            )

    weights = list(ig_graph.es["weight"]) if "weight" in ig_graph.es.attributes() else None
    for eidx in tqdm(
        range(ig_graph.ecount()),
        total=ig_graph.ecount(),
        desc="Convert prebuilt KG edges",
        unit="edge",
        leave=True,
        disable=not show_progress,
    ):
        u_idx, v_idx = ig_graph.es[eidx].tuple
        u = str(names[u_idx])
        v = str(names[v_idx])
        weight = float(weights[eidx]) if weights is not None else 1.0
        if g.has_edge(u, v):
            prev = float(g[u][v].get("weight", 0.0))
            g[u][v]["weight"] = max(prev, weight)
        else:
            g.add_edge(u, v, edge_type="hipporag_link", weight=weight)
        u_type = g.nodes[u].get("node_type")
        v_type = g.nodes[v].get("node_type")
        if {u_type, v_type} == {"sentence", "entity"}:
            g[u][v]["support_layer"] = "entity_chunk"

    alias_nodes_added = 0
    alias_edges_added = 0
    alias_limit = max(0, int(prebuilt_entity_token_limit))
    for entity_node in tqdm(
        entity_nodes,
        total=len(entity_nodes),
        desc="Build lexical anchor aliases",
        unit="entity",
        leave=True,
        disable=not show_progress,
    ):
        raw = str(g.nodes[entity_node].get("content", ""))
        cleaned = _clean_prebuilt_entity_content(raw)
        toks = content_tokens(cleaned)
        if alias_limit > 0:
            toks = toks[:alias_limit]
        for tok in toks:
            alias = f"e::{tok}"
            if alias not in g:
                g.add_node(alias, node_type="entity", token=tok, source_node_type="alias")
                alias_nodes_added += 1
            if not g.has_edge(alias, entity_node):
                g.add_edge(alias, entity_node, edge_type="entity_alias", weight=1.0)
                alias_edges_added += 1

    token_to_entities = {}
    for node in g.nodes:
        if g.nodes[node].get("node_type") != "entity":
            continue
        token = str(g.nodes[node].get("token", "") or "").strip().lower()
        if not token:
            continue
        token_to_entities.setdefault(token, set()).add(node)

    support_edges_added = 0
    for chunk_node in tqdm(
        chunk_nodes,
        total=len(chunk_nodes),
        desc="Build entity-chunk support edges",
        unit="chunk",
        leave=True,
        disable=not show_progress,
    ):
        text = str(g.nodes[chunk_node].get("text", "") or "")
        toks = content_tokens(text)
        if alias_limit > 0:
            toks = toks[: max(alias_limit * 2, alias_limit)]
        for tok in toks:
            for ent in token_to_entities.get(tok, set()):
                if g.has_edge(chunk_node, ent):
                    g[chunk_node][ent]["support_layer"] = "entity_chunk"
                    continue
                g.add_edge(chunk_node, ent, edge_type="entity_chunk_support", support_layer="entity_chunk", weight=1.0)
                support_edges_added += 1

    num_entities = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "entity")
    num_sentences = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "sentence")
    num_support_edges = sum(1 for u, v in g.edges if str(g[u][v].get("support_layer", "")) == "entity_chunk")
    stats = {
        "source_type": "prebuilt_igraph",
        "prebuilt_igraph_path": str(src.resolve()),
        "prebuilt_igraph_format": fmt,
        "prebuilt_igraph_vcount": int(ig_graph.vcount()),
        "prebuilt_igraph_ecount": int(ig_graph.ecount()),
        "prebuilt_entity_nodes": int(len(entity_nodes)),
        "prebuilt_chunk_nodes": int(len(chunk_nodes)),
        "prebuilt_entity_token_limit": int(alias_limit),
        "prebuilt_alias_nodes_added": int(alias_nodes_added),
        "prebuilt_alias_edges_added": int(alias_edges_added),
        "prebuilt_support_edges_added": int(support_edges_added),
        "num_nodes": int(g.number_of_nodes()),
        "num_edges": int(g.number_of_edges()),
        "num_entities": int(num_entities),
        "num_sentences": int(num_sentences),
        "num_entity_chunk_support_edges": int(num_support_edges),
    }
    return g, stats


def load_or_build_global_index(
    corpus_path,
    cache_dir,
    force_rebuild=False,
    prebuilt_igraph_path="",
    prebuilt_igraph_format="hipporag_pickle",
    prebuilt_entity_token_limit=6,
    embedding_enabled=False,
    embedding_model_name="nvidia/NV-Embed-v2",
    embedding_batch_size=8,
    embedding_max_length=192,
    embedding_text_max_chars=600,
    openie_mode="llm",
    openie_model_name="Qwen/Qwen2.5-7B-Instruct",
    openie_text_max_chars=2200,
    openie_max_new_tokens=256,
    openie_local_files_only=True,
    openie_api_base_url="",
    openie_api_key="",
    openie_api_timeout_sec=120.0,
    openie_parallel_workers=1,
    openie_log_every=200,
    openie_retry_attempts=3,
    openie_retry_backoff_sec=0.2,
    openie_error_sample_limit=20,
    show_progress=True,
):
    index_total_start = time.perf_counter()
    build_config = _normalize_build_config(
        prebuilt_igraph_path=prebuilt_igraph_path,
        prebuilt_igraph_format=prebuilt_igraph_format,
        prebuilt_entity_token_limit=prebuilt_entity_token_limit,
        embedding_enabled=embedding_enabled,
        embedding_model_name=embedding_model_name,
        embedding_batch_size=embedding_batch_size,
        embedding_max_length=embedding_max_length,
        embedding_text_max_chars=embedding_text_max_chars,
        openie_mode=openie_mode,
        openie_model_name=openie_model_name,
        openie_text_max_chars=openie_text_max_chars,
        openie_max_new_tokens=openie_max_new_tokens,
        openie_local_files_only=openie_local_files_only,
        openie_api_base_url=openie_api_base_url,
        openie_api_timeout_sec=openie_api_timeout_sec,
        openie_retry_attempts=openie_retry_attempts,
        openie_retry_backoff_sec=openie_retry_backoff_sec,
        openie_error_sample_limit=openie_error_sample_limit,
    )
    semantic_build_config = _normalize_semantic_build_config(build_config)
    prebuilt_path = str(prebuilt_igraph_path or "").strip()
    source_id_path = prebuilt_path if prebuilt_path else str(corpus_path or "").strip()
    if not source_id_path:
        raise ValueError("Provide either corpus_path or prebuilt_igraph_path.")

    index_dir = _index_dir(source_id_path, cache_dir, build_config)
    index_dir.mkdir(parents=True, exist_ok=True)
    graph_path = index_dir / "graph.gpickle"
    meta_path = index_dir / "meta.json"
    semantic_meta_path = index_dir / "semantic_index_meta.json"
    openie_sentence_cache_path = index_dir / "openie_sentence_cache.jsonl"
    fp = _fingerprint(prebuilt_path) if prebuilt_path else _fingerprint(corpus_path)
    semantic_key = _semantic_cache_key(fp, semantic_build_config)

    if not force_rebuild and graph_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint", {}) == fp and meta.get("build_config", {}) == build_config:
            load_start = time.perf_counter()
            graph = _read_gpickle(graph_path)
            load_ms = (time.perf_counter() - load_start) * 1000.0
            out_meta = dict(meta)
            out_meta_stats = dict(out_meta.get("stats", {}) or {})
            semantic_meta = dict(out_meta.get("semantic_index", {}) or {})
            semantic_build_ms = 0.0
            semantic_reuse_info = {}
            if bool(build_config.get("embedding_enabled", False)):
                graph_semantic_signature = _graph_semantic_node_signature(graph)
                if semantic_meta_path.exists() and _semantic_data_artifacts_ready(index_dir):
                    semantic_meta = _read_json_obj(semantic_meta_path)
                    semantic_meta = _materialize_semantic_meta_paths(
                        index_dir=index_dir,
                        semantic_meta=semantic_meta,
                        semantic_cache_key=semantic_key,
                        semantic_build_config=semantic_build_config,
                        graph_semantic_node_signature=graph_semantic_signature,
                        build_ms_override=float(semantic_meta.get("build_ms", 0.0) or 0.0),
                    )
                    _write_json_obj(semantic_meta_path, semantic_meta)
                else:
                    semantic_meta, semantic_reuse_info = _try_reuse_semantic_artifacts(
                        index_dir=index_dir,
                        cache_dir=cache_dir,
                        source_id_path=source_id_path,
                        source_fingerprint=fp,
                        semantic_build_config=semantic_build_config,
                        semantic_cache_key=semantic_key,
                        graph_semantic_node_signature=graph_semantic_signature,
                        show_progress=show_progress,
                    )
                    if semantic_meta is None:
                        semantic_meta = _build_semantic_index_artifacts(
                            g=graph,
                            index_dir=index_dir,
                            model_name=build_config.get("embedding_model_name", "nvidia/NV-Embed-v2"),
                            batch_size=int(build_config.get("embedding_batch_size", 8)),
                            max_length=int(build_config.get("embedding_max_length", 192)),
                            max_chars=int(build_config.get("embedding_text_max_chars", 600)),
                            semantic_build_config=semantic_build_config,
                            semantic_cache_key=semantic_key,
                            graph_semantic_node_signature=graph_semantic_signature,
                            show_progress=show_progress,
                        )
                        semantic_build_ms = float(semantic_meta.get("build_ms", 0.0) or 0.0)
                    else:
                        semantic_build_ms = 0.0

                out_meta_stats["semantic_embedding_enabled"] = True
                out_meta_stats["semantic_embedding_model_name"] = str(semantic_meta.get("model_name", ""))
                out_meta_stats["semantic_embedding_dim"] = int(semantic_meta.get("dim", 0) or 0)
                out_meta_stats["semantic_entity_count"] = int(semantic_meta.get("entity_count", 0) or 0)
                out_meta_stats["semantic_chunk_count"] = int(semantic_meta.get("chunk_count", 0) or 0)
                out_meta_stats["semantic_build_ms"] = float(semantic_build_ms)
                out_meta_stats["semantic_cache_key"] = str(semantic_key)
                out_meta_stats["semantic_build_config"] = dict(semantic_build_config)
                out_meta_stats["semantic_node_signature"] = str(semantic_meta.get("semantic_node_signature", "") or "")
                if semantic_reuse_info:
                    out_meta_stats.update(semantic_reuse_info)
            else:
                semantic_meta = {
                    "enabled": False,
                    "model_name": str(build_config.get("embedding_model_name", "")),
                    "entity_count": 0,
                    "chunk_count": 0,
                    "dim": 0,
                    "build_ms": 0.0,
                }
                out_meta_stats["semantic_embedding_enabled"] = False
                out_meta_stats["semantic_cache_key"] = str(semantic_key)
                out_meta_stats["semantic_build_config"] = dict(semantic_build_config)

            out_meta["cache_hit"] = True
            out_meta["index_operation"] = "cache_hit" if semantic_build_ms <= 0.0 else "cache_hit+semantic_build"
            out_meta["index_load_graph_ms"] = float(load_ms)
            out_meta["index_build_ms"] = float(out_meta.get("index_build_ms", 0.0) or 0.0) + float(semantic_build_ms)
            out_meta["index_write_ms"] = float(out_meta.get("index_write_ms", 0.0) or 0.0)
            out_meta["semantic_index"] = semantic_meta
            out_meta["semantic_cache_key"] = str(semantic_key)
            out_meta["semantic_build_config"] = dict(semantic_build_config)
            out_meta["stats"] = out_meta_stats
            out_meta["index_total_ms"] = float((time.perf_counter() - index_total_start) * 1000.0)
            if semantic_build_ms > 0.0 or bool(semantic_reuse_info.get("semantic_reuse_applied", False)):
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(out_meta, f, ensure_ascii=False, indent=2)
            return graph, out_meta

    build_start = time.perf_counter()
    if prebuilt_path:
        graph, stats = _build_graph_from_prebuilt_igraph(
            prebuilt_igraph_path=prebuilt_path,
            prebuilt_igraph_format=build_config.get("prebuilt_igraph_format", "hipporag_pickle"),
            prebuilt_entity_token_limit=build_config.get("prebuilt_entity_token_limit", 6),
            show_progress=show_progress,
        )
    else:
        openie_cache_seed_info = _maybe_seed_openie_cache_from_latest(
            cache_dir=cache_dir,
            source_id_path=source_id_path,
            target_cache_path=str(openie_sentence_cache_path),
            openie_mode=build_config["openie_mode"],
            show_progress=show_progress,
        )
        rows = load_corpus_rows(corpus_path, show_progress=show_progress)
        graph, stats = build_corpus_graph(
            rows,
            openie_mode=build_config["openie_mode"],
            openie_model_name=build_config["openie_model_name"],
            openie_text_max_chars=build_config["openie_text_max_chars"],
            openie_max_new_tokens=build_config["openie_max_new_tokens"],
            openie_local_files_only=build_config["openie_local_files_only"],
            openie_api_base_url=build_config.get("openie_api_base_url", ""),
            openie_api_key=openie_api_key,
            openie_api_timeout_sec=build_config.get("openie_api_timeout_sec", 120.0),
            openie_parallel_workers=int(openie_parallel_workers),
            openie_log_every=int(openie_log_every),
            openie_retry_attempts=build_config["openie_retry_attempts"],
            openie_retry_backoff_sec=build_config["openie_retry_backoff_sec"],
            openie_error_sample_limit=build_config["openie_error_sample_limit"],
            openie_sentence_cache_path=str(openie_sentence_cache_path),
            show_progress=show_progress,
        )
        stats = dict(stats or {})
        stats.update(openie_cache_seed_info)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    semantic_meta = {
        "enabled": False,
        "model_name": str(build_config.get("embedding_model_name", "")),
        "entity_count": 0,
        "chunk_count": 0,
        "dim": 0,
        "build_ms": 0.0,
    }
    semantic_reuse_info = {}
    if bool(build_config.get("embedding_enabled", False)):
        graph_semantic_signature = _graph_semantic_node_signature(graph)
        semantic_meta, semantic_reuse_info = _try_reuse_semantic_artifacts(
            index_dir=index_dir,
            cache_dir=cache_dir,
            source_id_path=source_id_path,
            source_fingerprint=fp,
            semantic_build_config=semantic_build_config,
            semantic_cache_key=semantic_key,
            graph_semantic_node_signature=graph_semantic_signature,
            show_progress=show_progress,
        )
        if semantic_meta is None:
            semantic_meta = _build_semantic_index_artifacts(
                g=graph,
                index_dir=index_dir,
                model_name=build_config.get("embedding_model_name", "nvidia/NV-Embed-v2"),
                batch_size=int(build_config.get("embedding_batch_size", 8)),
                max_length=int(build_config.get("embedding_max_length", 192)),
                max_chars=int(build_config.get("embedding_text_max_chars", 600)),
                semantic_build_config=semantic_build_config,
                semantic_cache_key=semantic_key,
                graph_semantic_node_signature=graph_semantic_signature,
                show_progress=show_progress,
            )
            build_ms += float(semantic_meta.get("build_ms", 0.0) or 0.0)

    stats = dict(stats or {})
    stats["semantic_embedding_enabled"] = bool(semantic_meta.get("enabled", False))
    stats["semantic_embedding_model_name"] = str(semantic_meta.get("model_name", ""))
    stats["semantic_embedding_dim"] = int(semantic_meta.get("dim", 0) or 0)
    stats["semantic_entity_count"] = int(semantic_meta.get("entity_count", 0) or 0)
    stats["semantic_chunk_count"] = int(semantic_meta.get("chunk_count", 0) or 0)
    stats["semantic_build_ms"] = float(semantic_meta.get("build_ms", 0.0) or 0.0)
    stats["semantic_cache_key"] = str(semantic_key)
    stats["semantic_build_config"] = dict(semantic_build_config)
    stats["semantic_node_signature"] = str(semantic_meta.get("semantic_node_signature", "") or "")
    if semantic_reuse_info:
        stats.update(semantic_reuse_info)

    write_start = time.perf_counter()
    _write_gpickle(graph, graph_path)
    write_ms = (time.perf_counter() - write_start) * 1000.0
    meta = {
        "index_dir": str(index_dir),
        "graph_path": str(graph_path),
        "source_id_path": str(Path(source_id_path).resolve()),
        "prebuilt_igraph_path": str(Path(prebuilt_path).resolve()) if prebuilt_path else "",
        "corpus_path": str(Path(corpus_path).resolve()) if str(corpus_path or "").strip() else "",
        "fingerprint": fp,
        "build_config": build_config,
        "built_at_utc": timestamp_iso_utc(),
        "stats": stats,
        "semantic_index": semantic_meta,
        "semantic_cache_key": str(semantic_key),
        "semantic_build_config": dict(semantic_build_config),
        "cache_hit": False,
        "index_operation": "build",
        "index_load_graph_ms": 0.0,
        "index_build_ms": float(build_ms),
        "index_write_ms": float(write_ms),
        "index_total_ms": float((time.perf_counter() - index_total_start) * 1000.0),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return graph, meta
