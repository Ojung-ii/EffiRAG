from .types import RetrievalResult, Sample
from .utils import mean_or_zero, safe_div


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


def aggregate_retrieval_metrics(rows):
    recall = [float(r.get("supporting_fact_recall", 0.0)) for r in rows]
    precision = [float(r.get("supporting_fact_precision", 0.0)) for r in rows]
    latency = [float(r.get("retrieval_latency_ms", 0.0)) for r in rows]

    return {
        "n_samples": float(len(rows)),
        "supporting_fact_recall": mean_or_zero(recall),
        "supporting_fact_precision": mean_or_zero(precision),
        "retrieval_latency_ms": mean_or_zero(latency),
    }
