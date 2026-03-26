import ast
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import networkx as nx

from .utils import content_tokens, timestamp_iso_utc

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CODE_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.IGNORECASE | re.DOTALL)


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
            leave=False,
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


def _write_openie_sentence_cache(path, cache):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for key in sorted(cache.keys()):
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
                leave=False,
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

    if requested_openie_mode == "llm":
        for _ in tqdm(
            [0],
            total=1,
            desc="Init OpenIE model",
            unit="step",
            leave=False,
            disable=not show_progress,
        ):
            extractor = _LLMTripleExtractor(
                model_name=openie_model_name,
                text_max_chars=openie_text_max_chars,
                max_new_tokens=openie_max_new_tokens,
                local_files_only=openie_local_files_only,
                retry_attempts=openie_retry_attempts,
                retry_backoff_sec=openie_retry_backoff_sec,
            )
        if not extractor.available:
            effective_openie_mode = "lexical"
            openie_init_error = str(extractor.error_message or "")
        if effective_openie_mode == "llm" and cache_path:
            openie_sentence_cache = _load_openie_sentence_cache(cache_path, show_progress=show_progress)

    total_sentences_expected = None
    if show_progress:
        total_sentences_expected = sum(len(_split_sentences(str(row.get("text", "")))) for row in corpus_rows)
    sentence_iter = (
        (row, sent_idx, sent)
        for row in corpus_rows
        for sent_idx, sent in enumerate(_split_sentences(str(row.get("text", ""))))
    )
    try:
        for row, sent_idx, sent in tqdm(
            sentence_iter,
            total=total_sentences_expected,
            desc="Build global KG",
            unit="sent",
            leave=False,
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
                g.add_edge(node, entity_node, edge_type="mentions")

            if effective_openie_mode != "llm" or extractor is None:
                continue

            cache_key = _openie_sentence_key(doc_idx, sent_idx, sent)
            cached = openie_sentence_cache.get(cache_key)
            if cached is not None:
                llm_cache_hits += 1
                triples = list(cached.get("triples", []) or [])
                parsed = bool(cached.get("parsed", True))
                err = ""
            else:
                llm_cache_misses += 1
                llm_calls += 1
                triples, _, parsed, err = extractor.extract(sent)
                if not err:
                    openie_sentence_cache[cache_key] = {
                        "parsed": bool(parsed),
                        "triples": [tuple(t) for t in triples],
                    }
                    cache_dirty = True

            if err:
                llm_runtime_errors += 1
                err_key = str(err).strip().splitlines()[0][:280]
                llm_error_counter[err_key] = llm_error_counter.get(err_key, 0) + 1
                if len(llm_error_samples) < max(0, int(openie_error_sample_limit)):
                    llm_error_samples.append(
                        {
                            "sentence_id": sentence_id,
                            "doc_idx": int(doc_idx),
                            "sent_idx": int(sent_idx),
                            "error": err_key,
                            "text_head": str(sent)[:240],
                        }
                    )
                continue
            if not parsed:
                llm_parse_failures += 1
            if not triples:
                llm_zero_triple_sentences += 1
                continue

            for tri_idx, (subj, pred, obj) in enumerate(triples):
                rel_node = f"r::{doc_idx}:{sent_idx}:{tri_idx}"
                g.add_node(
                    rel_node,
                    node_type="relation",
                    predicate=str(pred),
                    subject=str(subj),
                    object=str(obj),
                    doc_idx=doc_idx,
                    sent_idx=sent_idx,
                    sentence_id=sentence_id,
                )
                g.add_edge(node, rel_node, edge_type="expressed_in")
                num_relation_nodes += 1
                num_triples += 1

                for tok in _entity_tokens(subj):
                    ent = _ensure_entity_node(g, tok)
                    if ent is not None:
                        g.add_edge(rel_node, ent, edge_type="triple_subject")
                for tok in _entity_tokens(obj):
                    ent = _ensure_entity_node(g, tok)
                    if ent is not None:
                        g.add_edge(rel_node, ent, edge_type="triple_object")
    finally:
        if effective_openie_mode == "llm" and cache_path and cache_dirty:
            _write_openie_sentence_cache(cache_path, openie_sentence_cache)

    num_entities = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "entity")
    num_relations = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "relation")
    num_docs = len({int(row["idx"]) for row in corpus_rows})
    top_errors = sorted(llm_error_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    stats = {
        "num_docs": int(num_docs),
        "num_sentences": int(num_sentences),
        "num_entities": int(num_entities),
        "num_relation_nodes": int(num_relations or num_relation_nodes),
        "num_extracted_triples": int(num_triples),
        "num_nodes": int(g.number_of_nodes()),
        "num_edges": int(g.number_of_edges()),
        "openie_mode_requested": requested_openie_mode,
        "openie_mode_effective": effective_openie_mode,
        "openie_model_name": str(openie_model_name or ""),
        "openie_text_max_chars": int(openie_text_max_chars),
        "openie_max_new_tokens": int(openie_max_new_tokens),
        "openie_local_files_only": bool(openie_local_files_only),
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
    openie_mode,
    openie_model_name,
    openie_text_max_chars,
    openie_max_new_tokens,
    openie_local_files_only,
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
        "openie_mode": mode,
        "openie_model_name": str(openie_model_name or ""),
        "openie_text_max_chars": int(openie_text_max_chars),
        "openie_max_new_tokens": int(openie_max_new_tokens),
        "openie_local_files_only": bool(openie_local_files_only),
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
        leave=False,
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
        leave=False,
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

    alias_nodes_added = 0
    alias_edges_added = 0
    alias_limit = max(0, int(prebuilt_entity_token_limit))
    for entity_node in tqdm(
        entity_nodes,
        total=len(entity_nodes),
        desc="Build lexical anchor aliases",
        unit="entity",
        leave=False,
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

    num_entities = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "entity")
    num_sentences = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "sentence")
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
        "num_nodes": int(g.number_of_nodes()),
        "num_edges": int(g.number_of_edges()),
        "num_entities": int(num_entities),
        "num_sentences": int(num_sentences),
    }
    return g, stats


def load_or_build_global_index(
    corpus_path,
    cache_dir,
    force_rebuild=False,
    prebuilt_igraph_path="",
    prebuilt_igraph_format="hipporag_pickle",
    prebuilt_entity_token_limit=6,
    openie_mode="llm",
    openie_model_name="Qwen/Qwen2.5-7B-Instruct",
    openie_text_max_chars=2200,
    openie_max_new_tokens=256,
    openie_local_files_only=True,
    openie_retry_attempts=3,
    openie_retry_backoff_sec=0.2,
    openie_error_sample_limit=20,
    show_progress=True,
):
    build_config = _normalize_build_config(
        prebuilt_igraph_path=prebuilt_igraph_path,
        prebuilt_igraph_format=prebuilt_igraph_format,
        prebuilt_entity_token_limit=prebuilt_entity_token_limit,
        openie_mode=openie_mode,
        openie_model_name=openie_model_name,
        openie_text_max_chars=openie_text_max_chars,
        openie_max_new_tokens=openie_max_new_tokens,
        openie_local_files_only=openie_local_files_only,
        openie_retry_attempts=openie_retry_attempts,
        openie_retry_backoff_sec=openie_retry_backoff_sec,
        openie_error_sample_limit=openie_error_sample_limit,
    )
    prebuilt_path = str(prebuilt_igraph_path or "").strip()
    source_id_path = prebuilt_path if prebuilt_path else str(corpus_path or "").strip()
    if not source_id_path:
        raise ValueError("Provide either corpus_path or prebuilt_igraph_path.")

    index_dir = _index_dir(source_id_path, cache_dir, build_config)
    index_dir.mkdir(parents=True, exist_ok=True)
    graph_path = index_dir / "graph.gpickle"
    meta_path = index_dir / "meta.json"
    openie_sentence_cache_path = index_dir / "openie_sentence_cache.jsonl"
    fp = _fingerprint(prebuilt_path) if prebuilt_path else _fingerprint(corpus_path)

    if not force_rebuild and graph_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint", {}) == fp and meta.get("build_config", {}) == build_config:
            graph = _read_gpickle(graph_path)
            out_meta = dict(meta)
            out_meta["cache_hit"] = True
            return graph, out_meta

    if prebuilt_path:
        graph, stats = _build_graph_from_prebuilt_igraph(
            prebuilt_igraph_path=prebuilt_path,
            prebuilt_igraph_format=build_config.get("prebuilt_igraph_format", "hipporag_pickle"),
            prebuilt_entity_token_limit=build_config.get("prebuilt_entity_token_limit", 6),
            show_progress=show_progress,
        )
    else:
        rows = load_corpus_rows(corpus_path, show_progress=show_progress)
        graph, stats = build_corpus_graph(
            rows,
            openie_mode=build_config["openie_mode"],
            openie_model_name=build_config["openie_model_name"],
            openie_text_max_chars=build_config["openie_text_max_chars"],
            openie_max_new_tokens=build_config["openie_max_new_tokens"],
            openie_local_files_only=build_config["openie_local_files_only"],
            openie_retry_attempts=build_config["openie_retry_attempts"],
            openie_retry_backoff_sec=build_config["openie_retry_backoff_sec"],
            openie_error_sample_limit=build_config["openie_error_sample_limit"],
            openie_sentence_cache_path=str(openie_sentence_cache_path),
            show_progress=show_progress,
        )

    _write_gpickle(graph, graph_path)
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
        "cache_hit": False,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return graph, meta
