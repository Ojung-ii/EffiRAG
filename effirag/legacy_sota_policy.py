from __future__ import annotations

from typing import Any, Dict

from effirag.profiles import COPY_SPAN_DATASET_LOCKS


# Legacy compatibility settings for reproducing previous SOTA runs.
# These settings may contain dataset-specific behavior and must not be used
# as the paper's strict unified main method.
LEGACY_COPY_SPAN_DATASET_LOCKS: Dict[str, Dict[str, Any]] = {
    dataset: dict(locks) for dataset, locks in COPY_SPAN_DATASET_LOCKS.items()
}

LEGACY_COPY_SPAN_PROFILE_NAME = "copy_span_instruction_legacy_sota_replay"


def normalize_dataset_name(dataset_name: str | None) -> str:
    raw = str(dataset_name or "").strip().lower()
    aliases = {
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "2wiki": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
        "wikimultihopqa": "2wikimultihopqa",
        "musique": "musique",
        "popqa": "popqa",
    }
    return aliases.get(raw, raw)


def resolve_legacy_sota_copy_span_mode(dataset_name: str | None = None) -> str:
    ds = normalize_dataset_name(dataset_name)
    return str(LEGACY_COPY_SPAN_DATASET_LOCKS.get(ds, {}).get("retrieval_objective_mode", "baseline"))


def legacy_sota_copy_span_locks(dataset_name: str | None) -> Dict[str, Any]:
    ds = normalize_dataset_name(dataset_name)
    if ds not in LEGACY_COPY_SPAN_DATASET_LOCKS:
        supported = ", ".join(sorted(LEGACY_COPY_SPAN_DATASET_LOCKS))
        raise ValueError(f"Unsupported legacy SOTA dataset '{dataset_name}'. Supported: {supported}")
    return dict(LEGACY_COPY_SPAN_DATASET_LOCKS[ds])
