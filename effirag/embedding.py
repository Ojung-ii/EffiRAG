import numpy as np

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


def _l2_normalize_rows(array_2d):
    if array_2d is None:
        return None
    arr = np.asarray(array_2d, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1.0e-12, None)
    return arr / norms


def _format_instruction(model_name, instruction):
    inst = str(instruction or "").strip()
    if not inst:
        return ""
    name = str(model_name or "").strip().lower()
    if "nv-embed" in name:
        # Match HippoRAG2 NV-Embed-v2 convention.
        return f"Instruct: {inst}\\nQuery: "
    return inst


def _pick_torch_dtype(torch):
    if not torch.cuda.is_available():
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _is_nvembed_tied_weights_error(key, err):
    msg = str(err or "")
    return ("nv-embed" in str(key or "").strip().lower()) and ("all_tied_weights_keys" in msg)


def _is_nvembed_dynamic_cache_error(key, err):
    msg = str(err or "")
    lowered = str(key or "").strip().lower()
    if "nv-embed" not in lowered:
        return False
    return ("from_legacy_cache" in msg) or ("get_usable_length" in msg)


def _patch_transformers_tied_weights_key():
    try:
        import transformers
    except Exception:
        return False, "transformers_import_failed"

    try:
        base_cls = getattr(transformers, "PreTrainedModel", None)
        if base_cls is None:
            return False, "pretrained_model_class_missing"
        if hasattr(base_cls, "all_tied_weights_keys"):
            cur = getattr(base_cls, "all_tied_weights_keys")
            if isinstance(cur, dict):
                return False, "already_present_dict"
            setattr(base_cls, "all_tied_weights_keys", {})
            return True, "replaced_pretrainedmodel_all_tied_weights_keys_with_dict"
        setattr(base_cls, "all_tied_weights_keys", {})
        return True, "patched_pretrainedmodel_all_tied_weights_keys_dict"
    except Exception as exc:
        return False, f"patch_failed: {exc}"


def _patch_transformers_dynamic_cache_compat():
    try:
        from transformers.cache_utils import DynamicCache
    except Exception as exc:
        return False, f"dynamic_cache_import_failed: {exc}"

    patched = []
    try:
        if not hasattr(DynamicCache, "from_legacy_cache"):
            @classmethod
            def _from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is None:
                    return cache
                try:
                    for layer_idx, kv in enumerate(past_key_values):
                        if not isinstance(kv, (list, tuple)) or len(kv) < 2:
                            continue
                        key_states, value_states = kv[0], kv[1]
                        if key_states is None or value_states is None:
                            continue
                        cache.update(key_states, value_states, layer_idx)
                except Exception:
                    # Best-effort conversion only; empty cache is acceptable.
                    return cache
                return cache
            setattr(DynamicCache, "from_legacy_cache", _from_legacy_cache)
            patched.append("from_legacy_cache")

        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length, layer_idx=0):
                try:
                    return int(self.get_seq_length(layer_idx=layer_idx))
                except TypeError:
                    return int(self.get_seq_length())
                except Exception:
                    return 0
            setattr(DynamicCache, "get_usable_length", _get_usable_length)
            patched.append("get_usable_length")

        if patched:
            return True, "patched_dynamic_cache:" + ",".join(patched)
        return False, "already_compatible"
    except Exception as exc:
        return False, f"dynamic_cache_patch_failed: {exc}"


def _patch_nvembed_runtime_compat(key):
    notes = []
    if "nv-embed" not in str(key or "").strip().lower():
        return notes

    tied_ok, tied_msg = _patch_transformers_tied_weights_key()
    if tied_ok or str(tied_msg).startswith("already_present"):
        notes.append(f"tied_weights={tied_msg}")
    else:
        notes.append(f"tied_weights_patch_issue={tied_msg}")

    cache_ok, cache_msg = _patch_transformers_dynamic_cache_compat()
    if cache_ok or str(cache_msg).startswith("already_compatible"):
        notes.append(f"dynamic_cache={cache_msg}")
    else:
        notes.append(f"dynamic_cache_patch_issue={cache_msg}")
    return notes


def _load_model_with_optional_nvembed_patch(key, AutoModel, load_kwargs):
    kwargs = dict(load_kwargs or {})
    try:
        return AutoModel.from_pretrained(key, trust_remote_code=True, **kwargs), []
    except Exception as exc:
        need_tied_patch = _is_nvembed_tied_weights_error(key=key, err=exc)
        need_cache_patch = _is_nvembed_dynamic_cache_error(key=key, err=exc)
        if not (need_tied_patch or need_cache_patch):
            raise

        patch_notes = []
        if need_tied_patch:
            patched, patch_msg = _patch_transformers_tied_weights_key()
            patch_notes.append(f"tied_weights={patch_msg}")
            if (not patched) and ("already_present_dict" not in patch_msg):
                raise RuntimeError(f"{exc} (nvembed_patch={patch_msg})") from exc

        if need_cache_patch:
            patched, patch_msg = _patch_transformers_dynamic_cache_compat()
            patch_notes.append(f"dynamic_cache={patch_msg}")
            if (not patched) and ("already_compatible" not in patch_msg):
                raise RuntimeError(f"{exc} (nvembed_patch={patch_msg})") from exc

        model = AutoModel.from_pretrained(key, trust_remote_code=True, **kwargs)
        return model, [f"nvembed_compat_patch_applied:{';'.join(patch_notes)}"]


def _init_model_with_fallbacks(key, torch, AutoModel):
    errors = []
    if torch.cuda.is_available():
        dtype = _pick_torch_dtype(torch)
        try:
            model, notes = _load_model_with_optional_nvembed_patch(
                key=key,
                AutoModel=AutoModel,
                load_kwargs={
                    "device_map": "auto",
                    "torch_dtype": dtype,
                },
            )
            errors.extend(notes)
            return model, "cuda", errors
        except Exception as exc:
            errors.append(f"device_map_auto_failed: {exc}")

    try:
        model, notes = _load_model_with_optional_nvembed_patch(
            key=key,
            AutoModel=AutoModel,
            load_kwargs={},
        )
        errors.extend(notes)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model, device, errors
    except Exception as exc:
        errors.append(f"model_load_failed: {exc}")

    return None, "cpu", errors


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

    # Apply NV-Embed compatibility patches up front so encode-time calls are also covered.
    prepatch_notes = _patch_nvembed_runtime_compat(key=key)

    model, device, load_errors = _init_model_with_fallbacks(key=key, torch=torch, AutoModel=AutoModel)
    if model is None:
        merged_errors = list(prepatch_notes) + list(load_errors)
        _EMBEDDER_CACHE[key] = {
            "ok": False,
            "error": "embedding_model_load_failed: " + " | ".join(merged_errors),
        }
        return _EMBEDDER_CACHE[key]

    tokenizer = None
    backend = "native_encode" if hasattr(model, "encode") else "hf_pool"
    if backend == "hf_pool":
        try:
            tokenizer = AutoTokenizer.from_pretrained(key, trust_remote_code=True)
        except Exception as exc:
            _EMBEDDER_CACHE[key] = {"ok": False, "error": f"embedding_tokenizer_load_failed: {exc}"}
            return _EMBEDDER_CACHE[key]

    try:
        model.eval()
    except Exception:
        pass

    _EMBEDDER_CACHE[key] = {
        "ok": True,
        "error": "",
        "backend": backend,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "model_name": key,
    }
    return _EMBEDDER_CACHE[key]


def _encode_texts(texts, model_name, batch_size=16, max_length=256, instruction=""):
    cfg = _get_embedder(model_name)
    if not cfg.get("ok", False):
        return None, cfg.get("error", "embedding_unavailable")

    try:
        import torch
    except Exception as exc:
        return None, f"embedding_torch_unavailable: {exc}"

    model = cfg["model"]
    device = cfg["device"]
    backend = str(cfg.get("backend", "hf_pool"))

    bs = max(1, int(batch_size))
    max_len = max(8, int(max_length))
    vectors = []

    if backend == "native_encode":
        inst = _format_instruction(model_name=model_name, instruction=instruction)
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                params = {
                    "prompts": batch,
                    "max_length": max_len,
                }
                if inst:
                    params["instruction"] = inst
                out = model.encode(**params)
                arr = out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
                arr = _l2_normalize_rows(arr)
                vectors.extend(arr.tolist())
        return vectors, ""

    tokenizer = cfg["tokenizer"]
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


def encode_texts(
    texts,
    model_name,
    batch_size=16,
    max_length=256,
    max_chars=None,
    instruction="",
):
    payload = [_clip_text(t, max_chars=max_chars) for t in list(texts or [])]
    if not payload:
        return {
            "ok": True,
            "error": "",
            "vectors": [],
            "dim": 0,
            "model_name": str(model_name or ""),
        }

    vectors, err = _encode_texts(
        texts=payload,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        instruction=instruction,
    )
    if vectors is None:
        return {
            "ok": False,
            "error": err or "embedding_encode_failed",
            "vectors": [],
            "dim": 0,
            "model_name": str(model_name or ""),
        }

    dim = int(len(vectors[0])) if vectors else 0
    return {
        "ok": True,
        "error": "",
        "vectors": vectors,
        "dim": dim,
        "model_name": str(model_name or ""),
    }


def cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.dot(a, b))


def topk_cosine_similarity(query_vector, matrix, topn=50, scan_batch_size=8192):
    if matrix is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    arr = np.asarray(query_vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    q = arr / norm

    n = int(len(matrix))
    k = max(0, int(topn))
    if n <= 0 or k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    k = min(k, n)

    score_parts = []
    idx_parts = []
    step = max(1, int(scan_batch_size))

    for start in range(0, n, step):
        end = min(n, start + step)
        chunk = np.asarray(matrix[start:end], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            continue
        sims = np.dot(chunk, q)
        local_k = min(k, sims.shape[0])
        if local_k <= 0:
            continue
        local_idx = np.argpartition(sims, -local_k)[-local_k:]
        score_parts.append(sims[local_idx])
        idx_parts.append(local_idx + start)

    if not score_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    all_scores = np.concatenate(score_parts)
    all_indices = np.concatenate(idx_parts)

    final_k = min(k, all_scores.shape[0])
    top_local = np.argpartition(all_scores, -final_k)[-final_k:]
    order = top_local[np.argsort(all_scores[top_local])[::-1]]
    return all_indices[order].astype(np.int64), all_scores[order].astype(np.float32)


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

    encoded = encode_texts(
        texts=all_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        max_chars=None,
        instruction="",
    )
    vectors = encoded.get("vectors", []) if encoded.get("ok", False) else None
    if vectors is None or len(vectors) != len(all_texts):
        return {
            "applied": False,
            "error": encoded.get("error", "embedding_encode_failed"),
            "ranked_sentence_ids": ids,
            "ranked_sentence_texts": texts,
            "similarity_by_sentence_id": {},
            "fused_score_by_sentence_id": {},
            "rerank_topn": int(topn),
            "weight": float(weight),
        }

    qvec = np.asarray(vectors[0], dtype=np.float32)
    svecs = np.asarray(vectors[1:], dtype=np.float32)
    similarities = np.dot(svecs, qvec).tolist()

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
