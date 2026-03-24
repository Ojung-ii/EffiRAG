import ast
import hashlib
import json
import re
from pathlib import Path

import networkx as nx

from .utils import content_tokens, timestamp_iso_utc

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CODE_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.IGNORECASE | re.DOTALL)


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


def load_corpus_rows(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
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
    def __init__(self, model_name, text_max_chars, max_new_tokens):
        self.model_name = str(model_name or "").strip() or "Qwen/Qwen2.5-7B-Instruct"
        self.text_max_chars = int(text_max_chars)
        self.max_new_tokens = int(max_new_tokens)
        self.task = ""
        self.pipe = None
        self.available = False
        self.error_message = ""
        self._init_pipeline()

    def _init_pipeline(self):
        try:
            from transformers import pipeline
            from transformers.utils import logging as hf_logging
        except Exception as exc:
            self.error_message = f"transformers_import_failed: {exc}"
            return

        try:
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        attempts = []
        accel_kwargs = _hf_accel_kwargs()
        if accel_kwargs:
            attempts.append({"model": self.model_name, **accel_kwargs})
            attempts.append({"model": self.model_name, "tokenizer": self.model_name, "local_files_only": True, **accel_kwargs})
        attempts.extend(
            [
                {"model": self.model_name},
                {"model": self.model_name, "tokenizer": self.model_name, "local_files_only": True},
            ]
        )

        errors = []
        for task in ("text2text-generation", "text-generation"):
            for kwargs in attempts:
                try:
                    self.pipe = pipeline(task, **kwargs)
                    self.task = task
                    self.available = True
                    return
                except Exception as exc:
                    errors.append(f"{task}: {exc}")

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
        try:
            if self.task == "text-generation":
                out = self.pipe(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                )
            else:
                out = self.pipe(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
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
            return [], "", False, str(exc)


def build_corpus_graph(
    corpus_rows,
    openie_mode="llm",
    openie_model_name="Qwen/Qwen2.5-7B-Instruct",
    openie_text_max_chars=2200,
    openie_max_new_tokens=256,
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

    if requested_openie_mode == "llm":
        extractor = _LLMTripleExtractor(
            model_name=openie_model_name,
            text_max_chars=openie_text_max_chars,
            max_new_tokens=openie_max_new_tokens,
        )
        if not extractor.available:
            effective_openie_mode = "lexical"
            openie_init_error = str(extractor.error_message or "")

    for row in corpus_rows:
        doc_idx = int(row["idx"])
        title = str(row["title"])
        text = str(row["text"])
        sentences = _split_sentences(text)
        for sent_idx, sent in enumerate(sentences):
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

            llm_calls += 1
            triples, _, parsed, err = extractor.extract(sent)
            if err:
                llm_runtime_errors += 1
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

    num_entities = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "entity")
    num_relations = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "relation")
    num_docs = len({int(row["idx"]) for row in corpus_rows})
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
        "openie_init_error": str(openie_init_error or ""),
        "llm_calls": int(llm_calls),
        "llm_parse_failures": int(llm_parse_failures),
        "llm_zero_triple_sentences": int(llm_zero_triple_sentences),
        "llm_runtime_errors": int(llm_runtime_errors),
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


def _normalize_build_config(openie_mode, openie_model_name, openie_text_max_chars, openie_max_new_tokens):
    mode = str(openie_mode or "llm").strip().lower()
    if mode not in {"llm", "lexical"}:
        mode = "lexical"
    return {
        "openie_mode": mode,
        "openie_model_name": str(openie_model_name or ""),
        "openie_text_max_chars": int(openie_text_max_chars),
        "openie_max_new_tokens": int(openie_max_new_tokens),
    }


def _index_dir(corpus_path, cache_dir, build_config):
    corpus_abs = str(Path(corpus_path).resolve())
    signature = json.dumps(build_config, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha1(f"{corpus_abs}::{signature}".encode("utf-8")).hexdigest()[:12]
    stem = Path(corpus_path).stem
    mode = str(build_config.get("openie_mode", "lexical"))
    return Path(cache_dir) / f"{stem}_{mode}_{key}"


def load_or_build_global_index(
    corpus_path,
    cache_dir,
    force_rebuild=False,
    openie_mode="llm",
    openie_model_name="Qwen/Qwen2.5-7B-Instruct",
    openie_text_max_chars=2200,
    openie_max_new_tokens=256,
):
    build_config = _normalize_build_config(
        openie_mode=openie_mode,
        openie_model_name=openie_model_name,
        openie_text_max_chars=openie_text_max_chars,
        openie_max_new_tokens=openie_max_new_tokens,
    )
    index_dir = _index_dir(corpus_path, cache_dir, build_config)
    index_dir.mkdir(parents=True, exist_ok=True)
    graph_path = index_dir / "graph.gpickle"
    meta_path = index_dir / "meta.json"
    fp = _fingerprint(corpus_path)

    if not force_rebuild and graph_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint", {}) == fp and meta.get("build_config", {}) == build_config:
            graph = nx.read_gpickle(graph_path)
            out_meta = dict(meta)
            out_meta["cache_hit"] = True
            return graph, out_meta

    rows = load_corpus_rows(corpus_path)
    graph, stats = build_corpus_graph(
        rows,
        openie_mode=build_config["openie_mode"],
        openie_model_name=build_config["openie_model_name"],
        openie_text_max_chars=build_config["openie_text_max_chars"],
        openie_max_new_tokens=build_config["openie_max_new_tokens"],
    )

    nx.write_gpickle(graph, graph_path)
    meta = {
        "index_dir": str(index_dir),
        "graph_path": str(graph_path),
        "fingerprint": fp,
        "build_config": build_config,
        "built_at_utc": timestamp_iso_utc(),
        "stats": stats,
        "cache_hit": False,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return graph, meta
