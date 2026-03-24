import hashlib
import json
import re
from pathlib import Path

import networkx as nx

from .utils import content_tokens, timestamp_iso_utc

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


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


def build_corpus_graph(corpus_rows):
    g = nx.Graph()
    num_sentences = 0

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
                entity_node = f"e::{tok}"
                if entity_node not in g:
                    g.add_node(entity_node, node_type="entity", token=tok)
                g.add_edge(node, entity_node, edge_type="mentions")

    num_entities = sum(1 for n in g.nodes if g.nodes[n].get("node_type") == "entity")
    num_docs = len({int(row["idx"]) for row in corpus_rows})
    stats = {
        "num_docs": int(num_docs),
        "num_sentences": int(num_sentences),
        "num_entities": int(num_entities),
        "num_nodes": int(g.number_of_nodes()),
        "num_edges": int(g.number_of_edges()),
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


def _index_dir(corpus_path, cache_dir):
    corpus_abs = str(Path(corpus_path).resolve())
    key = hashlib.sha1(corpus_abs.encode("utf-8")).hexdigest()[:12]
    stem = Path(corpus_path).stem
    return Path(cache_dir) / f"{stem}_{key}"


def load_or_build_global_index(corpus_path, cache_dir, force_rebuild=False):
    index_dir = _index_dir(corpus_path, cache_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    graph_path = index_dir / "graph.gpickle"
    meta_path = index_dir / "meta.json"
    fp = _fingerprint(corpus_path)

    if not force_rebuild and graph_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint", {}) == fp:
            graph = nx.read_gpickle(graph_path)
            out_meta = dict(meta)
            out_meta["cache_hit"] = True
            return graph, out_meta

    rows = load_corpus_rows(corpus_path)
    graph, stats = build_corpus_graph(rows)

    nx.write_gpickle(graph, graph_path)
    meta = {
        "index_dir": str(index_dir),
        "graph_path": str(graph_path),
        "fingerprint": fp,
        "built_at_utc": timestamp_iso_utc(),
        "stats": stats,
        "cache_hit": False,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return graph, meta
