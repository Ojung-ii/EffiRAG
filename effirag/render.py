from .types import RenderedContext


def _sample_sentence_lookup(sample):
    lookup = {}
    for doc in sample.contexts:
        for sent_idx, sent in enumerate(doc.sentences):
            sid = "%s::%d" % (doc.title, sent_idx)
            lookup[sid] = sent
    return lookup


def render_context(sample, retrieval_result, max_context_sentences):
    max_n = max(1, int(max_context_sentences))
    lookup = _sample_sentence_lookup(sample)

    ids = list(retrieval_result.selected_sentence_ids)
    texts = list(retrieval_result.selected_sentences)

    if len(texts) != len(ids):
        texts = [lookup.get(sid, "") for sid in ids]

    pairs = [(sid, txt) for sid, txt in zip(ids, texts) if sid and txt]

    truncated = len(pairs) > max_n
    pairs = pairs[:max_n]

    lines = []
    for i, (sid, sent) in enumerate(pairs, start=1):
        lines.append("[%d] (%s) %s" % (i, sid, sent))

    text = "\n".join(lines)
    return RenderedContext(
        sample_id=sample.qid,
        method=retrieval_result.method,
        text=text,
        sentences=[sent for _, sent in pairs],
        sentence_ids=[sid for sid, _ in pairs],
        truncated=truncated,
    )
