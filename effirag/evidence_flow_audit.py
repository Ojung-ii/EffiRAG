from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .datasets import load_2wikimultihopqa, load_hotpotqa, load_musique, load_popqa
from .metrics import supporting_fact_match_details
from .utils import content_tokens, mean_or_zero, safe_div


DATASET_ORDER = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]
FAILURE_ORDER = [
    "retrieval_miss",
    "selector_drop",
    "render_drop",
    "low_density",
    "generation_ready",
    "ambiguous_id_mapping",
]


def ordered_unique_ids_texts(unit_ids: Iterable[str], unit_texts: Iterable[str]) -> Tuple[List[str], List[str]]:
    ids = [str(x or "").strip() for x in list(unit_ids or [])]
    texts = [str(x or "") for x in list(unit_texts or [])]
    seen = set()
    out_ids: List[str] = []
    out_texts: List[str] = []
    for idx, sid in enumerate(ids):
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out_ids.append(sid)
        out_texts.append(texts[idx] if idx < len(texts) else "")
    return out_ids, out_texts


def token_count_for_texts(texts: Iterable[str]) -> int:
    total = 0
    for text in list(texts or []):
        total += int(len(content_tokens(str(text or ""))))
    return int(total)


def stage_metrics(sample, unit_ids: List[str], unit_texts: List[str], graph_mode: str) -> Dict[str, Any]:
    if sample is None:
        return {
            "sf_P": 0.0,
            "sf_R": 0.0,
            "sf_F1": 0.0,
            "support_hit": 0,
            "tokens": token_count_for_texts(unit_texts),
            "sf_F1_per_1k_tokens": 0.0,
            "count": int(len(unit_ids)),
            "gold_support_count": 0,
            "matched_gold_count": 0,
            "matched_gold_ids": [],
            "predicted_ids": list(unit_ids),
            "id_mapping_available": False,
        }

    ids, texts = ordered_unique_ids_texts(unit_ids, unit_texts)
    match = supporting_fact_match_details(
        sample=sample,
        unit_ids=ids,
        unit_texts=texts,
        graph_mode=graph_mode,
    )
    gold_total = int(match.get("gold_total", 0) or 0)
    matched_gold_total = int(match.get("matched_gold_total", 0) or 0)
    predicted_total = int(match.get("predicted_unit_total", len(ids)) or len(ids))
    matched_unit_total = int(match.get("matched_unit_total", 0) or 0)
    sf_p = float(safe_div(float(matched_unit_total), float(predicted_total)))
    sf_r = float(safe_div(float(matched_gold_total), float(gold_total)))
    sf_f1 = float(safe_div(2.0 * sf_p * sf_r, sf_p + sf_r)) if (sf_p + sf_r) > 0.0 else 0.0
    predicted_units = list(match.get("predicted_units", []) or [])
    tokens = token_count_for_texts([str((unit or {}).get("text", "") or "") for unit in predicted_units])
    density = float(safe_div(float(sf_f1) * 1000.0, float(max(1, tokens))))
    return {
        "sf_P": float(sf_p),
        "sf_R": float(sf_r),
        "sf_F1": float(sf_f1),
        "support_hit": int(1 if matched_gold_total > 0 else 0),
        "tokens": int(tokens),
        "sf_F1_per_1k_tokens": float(density),
        "count": int(predicted_total),
        "gold_support_count": int(gold_total),
        "matched_gold_count": int(matched_gold_total),
        "matched_gold_ids": list(match.get("matched_gold_sentence_ids", []) or []),
        "predicted_ids": [str((unit or {}).get("unit_id", "") or "") for unit in predicted_units],
        "id_mapping_available": True,
    }


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aset = {str(x) for x in list(a or []) if str(x)}
    bset = {str(x) for x in list(b or []) if str(x)}
    if not aset and not bset:
        return 1.0
    denom = len(aset.union(bset))
    if denom <= 0:
        return 0.0
    return float(len(aset.intersection(bset)) / float(denom))


def classify_failure_type(
    candidate_stage: Mapping[str, Any],
    selected_stage: Mapping[str, Any],
    rendered_stage: Mapping[str, Any],
    *,
    density_threshold: float = 0.30,
) -> str:
    if not bool(candidate_stage.get("id_mapping_available", False)):
        return "ambiguous_id_mapping"
    if int(candidate_stage.get("matched_gold_count", 0)) <= 0:
        return "retrieval_miss"
    if float(selected_stage.get("sf_R", 0.0)) + 1e-12 < float(candidate_stage.get("sf_R", 0.0)):
        return "selector_drop"
    if float(rendered_stage.get("sf_R", 0.0)) + 1e-12 < float(selected_stage.get("sf_R", 0.0)):
        return "render_drop"
    if float(rendered_stage.get("sf_F1_per_1k_tokens", 0.0)) < float(density_threshold):
        return "low_density"
    return "generation_ready"


def load_samples_for_dataset(dataset: str):
    name = str(dataset or "").strip().lower()
    if name == "hotpotqa":
        samples = load_hotpotqa(split="validation", limit=None, data_path=None)
    elif name == "2wikimultihopqa":
        samples = load_2wikimultihopqa(split="validation", limit=None, data_path=None)
    elif name == "musique":
        samples = load_musique(split="validation", limit=None, data_path=None)
    elif name == "popqa":
        samples = load_popqa(split="test", limit=None, data_path=None)
    else:
        raise ValueError(f"Unsupported dataset for audit: {dataset}")
    return {str(sample.qid): sample for sample in list(samples or [])}


def find_latest_query_results(root: Path, dataset: str) -> Optional[Path]:
    dataset_root = root / dataset
    paths = []
    if dataset_root.exists():
        paths.extend(dataset_root.rglob("rag_query_results.jsonl"))
    paths.extend(root.glob(f"**/{dataset}/**/rag_query_results.jsonl"))
    uniq = sorted({path.resolve() for path in paths})
    return uniq[-1] if uniq else None


def parse_roots(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in list(items or []):
        token = str(item or "").strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            continue
        out[k] = Path(v).resolve()
    return out


def aggregate_by_profile_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        buckets[(str(row.get("dataset", "")), str(row.get("profile", "")))].append(row)

    out: List[Dict[str, Any]] = []
    for (dataset, profile), group in sorted(buckets.items()):
        num_queries = len(group)
        failure_counts = Counter([str(item.get("failure_type", "")) for item in group])
        out.append(
            {
                "dataset": dataset,
                "profile": profile,
                "num_queries": int(num_queries),
                "candidate_sf_P": float(mean_or_zero([float(item.get("candidate_sf_P", 0.0)) for item in group])),
                "candidate_sf_R": float(mean_or_zero([float(item.get("candidate_sf_R", 0.0)) for item in group])),
                "candidate_sf_F1": float(mean_or_zero([float(item.get("candidate_sf_F1", 0.0)) for item in group])),
                "candidate_tokens_avg": float(mean_or_zero([float(item.get("candidate_tokens", 0.0)) for item in group])),
                "candidate_sf_F1_per_1k_tokens": float(
                    mean_or_zero([float(item.get("candidate_sf_F1_per_1k_tokens", 0.0)) for item in group])
                ),
                "selected_sf_P": float(mean_or_zero([float(item.get("selected_sf_P", 0.0)) for item in group])),
                "selected_sf_R": float(mean_or_zero([float(item.get("selected_sf_R", 0.0)) for item in group])),
                "selected_sf_F1": float(mean_or_zero([float(item.get("selected_sf_F1", 0.0)) for item in group])),
                "selected_tokens_avg": float(mean_or_zero([float(item.get("selected_tokens", 0.0)) for item in group])),
                "selected_sf_F1_per_1k_tokens": float(
                    mean_or_zero([float(item.get("selected_sf_F1_per_1k_tokens", 0.0)) for item in group])
                ),
                "rendered_sf_P": float(mean_or_zero([float(item.get("rendered_sf_P", 0.0)) for item in group])),
                "rendered_sf_R": float(mean_or_zero([float(item.get("rendered_sf_R", 0.0)) for item in group])),
                "rendered_sf_F1": float(mean_or_zero([float(item.get("rendered_sf_F1", 0.0)) for item in group])),
                "rendered_tokens_avg": float(mean_or_zero([float(item.get("rendered_tokens", 0.0)) for item in group])),
                "rendered_sf_F1_per_1k_tokens": float(
                    mean_or_zero([float(item.get("rendered_sf_F1_per_1k_tokens", 0.0)) for item in group])
                ),
                "retrieval_miss_rate": float(safe_div(float(failure_counts.get("retrieval_miss", 0)), float(max(1, num_queries)))),
                "selector_drop_rate": float(safe_div(float(failure_counts.get("selector_drop", 0)), float(max(1, num_queries)))),
                "render_drop_rate": float(safe_div(float(failure_counts.get("render_drop", 0)), float(max(1, num_queries)))),
                "low_density_rate": float(safe_div(float(failure_counts.get("low_density", 0)), float(max(1, num_queries)))),
                "generation_ready_rate": float(
                    safe_div(float(failure_counts.get("generation_ready", 0)), float(max(1, num_queries)))
                ),
            }
        )
    return out


def overlap_ratio(reference_ids: Iterable[str], probe_ids: Iterable[str]) -> float:
    ref = {str(x) for x in list(reference_ids or []) if str(x)}
    if not ref:
        return 0.0
    probe = {str(x) for x in list(probe_ids or []) if str(x)}
    return float(safe_div(float(len(ref.intersection(probe))), float(len(ref))))


def infer_lost_stage(
    legacy_rendered_sf_r: float,
    profile_candidate_sf_r: float,
    profile_selected_sf_r: float,
    profile_rendered_sf_r: float,
) -> str:
    legacy_r = float(legacy_rendered_sf_r)
    cand_r = float(profile_candidate_sf_r)
    sel_r = float(profile_selected_sf_r)
    ren_r = float(profile_rendered_sf_r)
    if legacy_r <= 0.0:
        return "unknown"
    if cand_r + 1e-12 < legacy_r:
        return "candidate"
    if sel_r + 1e-12 < cand_r:
        return "selected"
    if ren_r + 1e-12 < sel_r:
        return "rendered"
    if ren_r + 1e-12 < legacy_r:
        return "rendered"
    return "not_lost"
