from .types import RetrievalResult, Sample
from .utils import mean_or_zero, safe_div

DEFAULT_RECALL_KS = (1, 2, 5, 10, 20, 30, 50, 100, 150, 200)


def supporting_fact_ids(sample):
    return {f"{title}::{sent_idx}" for title, sent_idx in sample.supporting_facts}


def supporting_fact_recall(sample, retrieval):
    gold = supporting_fact_ids(sample)
    pred = set(retrieval.selected_sentence_ids)
    return safe_div(len(gold.intersection(pred)), len(gold))


def supporting_fact_precision(sample, retrieval):
    gold = supporting_fact_ids(sample)
    pred = set(retrieval.selected_sentence_ids)
    return safe_div(len(gold.intersection(pred)), len(pred))


def supporting_fact_recall_at_k(sample, retrieval, k: int):
    if k <= 0:
        return 0.0
    gold = supporting_fact_ids(sample)
    pred = set((retrieval.selected_sentence_ids or [])[:k])
    return safe_div(len(gold.intersection(pred)), len(gold))


def supporting_fact_recall_at_ks(sample, retrieval, ks=DEFAULT_RECALL_KS):
    return {str(int(k)): supporting_fact_recall_at_k(sample, retrieval, int(k)) for k in ks}


def aggregate_retrieval_metrics(rows, recall_ks=DEFAULT_RECALL_KS):
    recall = [float(r.get("supporting_fact_recall", 0.0)) for r in rows]
    precision = [float(r.get("supporting_fact_precision", 0.0)) for r in rows]
    latency = [float(r.get("retrieval_latency_ms", 0.0)) for r in rows]

    summary = {
        "n_samples": float(len(rows)),
        "supporting_fact_recall": mean_or_zero(recall),
        "supporting_fact_precision": mean_or_zero(precision),
        "retrieval_latency_ms": mean_or_zero(latency),
    }

    recall_at_k_summary = {}
    for k in recall_ks:
        key = str(int(k))
        vals = []
        for row in rows:
            per_row = row.get("recall_at_k", {}) or {}
            vals.append(float(per_row.get(key, 0.0)))
        agg = mean_or_zero(vals)
        recall_at_k_summary[key] = agg
        summary[f"supporting_fact_recall_at_{key}"] = agg

    summary["supporting_fact_recall_at_k"] = recall_at_k_summary
    return summary
