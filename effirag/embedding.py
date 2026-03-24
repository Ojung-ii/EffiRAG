_EMBEDDER_CACHE = {}


def _clip_text(text, max_chars):
    val = str(text or "").strip()
    if max_chars is None or int(max_chars) <= 0:
        return val
    return val[: int(max_chars)]


def _minmax(values):
    vals = [float(v) for v in values]
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return [0.0 for _ in vals]
    scale = hi - lo
    return [(v - lo) / scale for v in vals]


def _get_embedder(model_name):
    key = str(model_name or "").strip() or "sentence-transformers/all-MiniLM-L6-v2"
    if key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[key]

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        _EMBEDDER_CACHE[key] = {"ok": False, "error": f"embedding_import_failed: {exc}"}
        return _EMBEDDER_CACHE[key]

    try:
        tokenizer = AutoTokenizer.from_pretrained(key)
        model = AutoModel.from_pretrained(key)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        _EMBEDDER_CACHE[key] = {
            "ok": True,
            "error": "",
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "model_name": key,
        }
    except Exception as exc:
        _EMBEDDER_CACHE[key] = {"ok": False, "error": f"embedding_model_load_failed: {exc}"}

    return _EMBEDDER_CACHE[key]


def _encode_texts(texts, model_name, batch_size=16, max_length=256):
    cfg = _get_embedder(model_name)
    if not cfg.get("ok", False):
        return None, cfg.get("error", "embedding_unavailable")

    try:
        import torch
    except Exception as exc:
        return None, f"embedding_torch_unavailable: {exc}"

    tokenizer = cfg["tokenizer"]
    model = cfg["model"]
    device = cfg["device"]

    vectors = []
    bs = max(1, int(batch_size))
    max_len = max(8, int(max_length))

    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            masked = hidden * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.detach().cpu().tolist())

    return vectors, ""


def rerank_sentences_by_embedding(
    question,
    sentence_ids,
    sentence_texts,
    base_score_map,
    model_name,
    weight=0.35,
    rerank_topn=80,
    batch_size=16,
    max_length=256,
    max_chars=600,
):
    ids = list(sentence_ids or [])
    texts = list(sentence_texts or [])
    if len(ids) != len(texts):
        return {
            "applied": False,
            "error": "length_mismatch_between_sentence_ids_and_texts",
            "ranked_sentence_ids": ids,
            "ranked_sentence_texts": texts,
            "similarity_by_sentence_id": {},
            "fused_score_by_sentence_id": {},
            "rerank_topn": int(max(0, int(rerank_topn))),
            "weight": float(weight),
        }

    n = len(ids)
    if n <= 1:
        return {
            "applied": False,
            "error": "",
            "ranked_sentence_ids": ids,
            "ranked_sentence_texts": texts,
            "similarity_by_sentence_id": {},
            "fused_score_by_sentence_id": {},
            "rerank_topn": int(max(0, int(rerank_topn))),
            "weight": float(weight),
        }

    topn = int(rerank_topn) if rerank_topn is not None else n
    if topn <= 0:
        topn = n
    topn = min(topn, n)

    head_ids = ids[:topn]
    head_texts = texts[:topn]
    tail_ids = ids[topn:]
    tail_texts = texts[topn:]

    clipped_head = [_clip_text(t, max_chars=max_chars) for t in head_texts]
    query = _clip_text(question, max_chars=max_chars)
    all_texts = [query] + clipped_head

    vectors, err = _encode_texts(
        texts=all_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )
    if vectors is None or len(vectors) != len(all_texts):
        return {
            "applied": False,
            "error": err or "embedding_encode_failed",
            "ranked_sentence_ids": ids,
            "ranked_sentence_texts": texts,
            "similarity_by_sentence_id": {},
            "fused_score_by_sentence_id": {},
            "rerank_topn": int(topn),
            "weight": float(weight),
        }

    qvec = vectors[0]
    svecs = vectors[1:]
    similarities = []
    for vec in svecs:
        dot = 0.0
        for a, b in zip(qvec, vec):
            dot += float(a) * float(b)
        similarities.append(dot)

    base_vals = [float((base_score_map or {}).get(sid, 0.0)) for sid in head_ids]
    base_norm = _minmax(base_vals)
    sim_norm = _minmax(similarities)
    w = max(0.0, min(1.0, float(weight)))

    fused = []
    for i, sid in enumerate(head_ids):
        score = (1.0 - w) * base_norm[i] + w * sim_norm[i]
        fused.append((sid, float(score), i, float(similarities[i])))
    fused.sort(key=lambda x: (x[1], x[3], -x[2]), reverse=True)

    ranked_head_ids = [sid for sid, _, _, _ in fused]
    text_map = {sid: txt for sid, txt in zip(head_ids, head_texts)}
    ranked_head_texts = [text_map.get(sid, "") for sid in ranked_head_ids]

    similarity_by_sid = {sid: float(sim) for sid, _, _, sim in fused}
    fused_by_sid = {sid: float(score) for sid, score, _, _ in fused}

    return {
        "applied": True,
        "error": "",
        "ranked_sentence_ids": ranked_head_ids + tail_ids,
        "ranked_sentence_texts": ranked_head_texts + tail_texts,
        "similarity_by_sentence_id": similarity_by_sid,
        "fused_score_by_sentence_id": fused_by_sid,
        "rerank_topn": int(topn),
        "weight": float(w),
    }
