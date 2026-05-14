from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from effirag.canonical_objective import resolve_unified_copy_span_mode as _resolve_unified_copy_span_mode
from effirag.grouped_profiles import INSTRUCTION_INTERFACE_CONTRACT, REFERENCE_LABEL
from effirag.profiles import (
    COPY_SPAN_DISABLED_MODULES,
    COPY_SPAN_RENDER_WEIGHTS,
    COPY_SPAN_RETRIEVAL_WEIGHTS,
)


DATASET_ORDER = ("hotpotqa", "2wikimultihopqa", "musique", "popqa")
STRICT_UNIFIED_VARIANT_NAME = "phase6_strict_unified_v1"
STRICT_UNIFIED_PROFILE_FAMILY = "copy_span_instruction_unified"

COPY_SPAN_INSTRUCTION_UNIFIED_LARGE = "copy_span_instruction_unified_large"
COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM = "copy_span_instruction_unified_medium"
COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT = "copy_span_instruction_unified_compact"
COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP = "copy_span_instruction_unified_large_lightsep"
COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL = "copy_span_instruction_unified_large_reorder_all"
COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL = "copy_span_instruction_unified_large_lightsep_reorder_all"
COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL = "copy_span_instruction_unified_medium_lightsep_reorder_all"

UNIFIED_CLEAN_PROFILE_NAMES = (
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT,
)

UNIFIED_ENHANCED_PROFILE_NAMES = (
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL,
)

UNIFIED_PROFILE_NAMES = UNIFIED_CLEAN_PROFILE_NAMES + UNIFIED_ENHANCED_PROFILE_NAMES

UNIFIED_PROFILE_ALIASES = {
    "large": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE,
    "unified_large": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE: COPY_SPAN_INSTRUCTION_UNIFIED_LARGE,
    "medium": COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
    "unified_medium": COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM: COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
    "compact": COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT,
    "unified_compact": COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT,
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT: COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT,
    "large_lightsep": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP,
    "unified_large_lightsep": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP: COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP,
    "large_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL,
    "unified_large_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL,
    "large_lightsep_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL,
    "unified_large_lightsep_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL,
    "medium_lightsep_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL,
    "unified_medium_lightsep_reorder_all": COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL,
}

UNIFIED_PROFILE_DIRS = {
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE: "unified_large",
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM: "unified_medium",
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT: "unified_compact",
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP: "unified_large_lightsep",
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: "unified_large_reorder_all",
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: "unified_large_lightsep_reorder_all",
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: "unified_medium_lightsep_reorder_all",
}

UNIFIED_BUDGETS = {
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM: {
        "seed_topn": 36,
        "run_topn": 18,
        "render_topn": 18,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT: {
        "seed_topn": 30,
        "run_topn": 15,
        "render_topn": 15,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: {
        "seed_topn": 45,
        "run_topn": 24,
        "render_topn": 24,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: {
        "seed_topn": 36,
        "run_topn": 18,
        "render_topn": 18,
    },
}

UNIFIED_BUDGET_CONFIG_KEYS = (
    "semantic_topn_entity",
    "semantic_topn_chunk",
    "graph_reserve_topn",
)

UNIFIED_RENDERING_LOCK = {
    "answer_support_pinning_enabled": False,
    "answer_support_pinning_min": 1,
    "final_top_slice_reorder_enabled": False,
    "final_top_slice_reorder_topk": 4,
    "corridor_answer_preserve_guarded_hotpot_enabled": False,
    "oracle_support_injection_enabled": False,
}

UNIFIED_DEFAULT_INTERFACE_LOCK = {
    "order_strategy": "score",
    "prompt_variant": "default",
    "prompt_variant_label": "default",
    "render_variant": "score_ordered_corridor_aware_flat",
    "added_instruction": "none",
}

UNIFIED_LIGHTSEP_INTERFACE_LOCK = dict(INSTRUCTION_INTERFACE_CONTRACT)

UNIFIED_METHOD_LOCK = {
    "retrieval_objective_mode": "baseline",
    "canonical_variant_name": STRICT_UNIFIED_VARIANT_NAME,
    "shared_budget_profile": "off",
    "max_anchors": 4,
    "samples_per_anchor": 4,
    "hybrid_anchor_recall_enabled": False,
    "bridge_candidate_induction_enabled": False,
    "role_aware_chunk_scoring_enabled": False,
    "coverage_selection_enabled": False,
    "corridor_compact_shaping_enabled": False,
    "corridor_answer_preserve_enabled": False,
    "corridor_bridge_purity_shaping_enabled": False,
    "corridor_path_preserve_enabled": False,
    "corridor_path_preserve_compact_enabled": False,
    "corridor_path_preserve_guarded_enabled": False,
    "corridor_path_preserve_compact_lite_enabled": False,
    "corridor_answer_preserve_confidence_gated_enabled": False,
    "top1_correction_enabled": False,
    "run_light_rerank_enabled": False,
    "chunk_grounding_enabled": False,
    **UNIFIED_RENDERING_LOCK,
}

UNIFIED_PROFILE_INTERFACE_LOCKS = {
    # Preserve the generated Phase-6 clean profile behavior that already uses
    # the copy-span light-separator interface inherited from the SOTA source pack.
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_COMPACT: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: UNIFIED_DEFAULT_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: UNIFIED_LIGHTSEP_INTERFACE_LOCK,
}

UNIFIED_PROFILE_RENDERING_OVERRIDES = {
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_REORDER_ALL: {
        "final_top_slice_reorder_enabled": True,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_LARGE_LIGHTSEP_REORDER_ALL: {
        "final_top_slice_reorder_enabled": True,
    },
    COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM_LIGHTSEP_REORDER_ALL: {
        "final_top_slice_reorder_enabled": True,
    },
}

UNIFIED_RUNTIME_DEFAULTS = {
    "precomputed_retrieval_path": "",
    "precomputed_retrieval_strict": False,
}

UNIFIED_INTERFACE_SETTING_KEYS = (
    "order_strategy",
    "prompt_variant",
)

UNIFIED_METHOD_SETTING_KEYS = tuple(UNIFIED_METHOD_LOCK.keys()) + UNIFIED_INTERFACE_SETTING_KEYS


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


def normalize_unified_profile_name(profile_name: str | None) -> str:
    raw = str(profile_name or COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM).strip().lower()
    normalized = UNIFIED_PROFILE_ALIASES.get(raw)
    if not normalized:
        supported = ", ".join(sorted(UNIFIED_PROFILE_ALIASES))
        raise ValueError(f"Unsupported unified copy-span profile '{profile_name}'. Supported: {supported}")
    return normalized


def unified_profile_dir(profile_name: str | None) -> str:
    return UNIFIED_PROFILE_DIRS[normalize_unified_profile_name(profile_name)]


def resolve_unified_copy_span_mode(dataset_name: str | None = None) -> str:
    return _resolve_unified_copy_span_mode(dataset_name)


def unified_budget(profile_name: str | None) -> Dict[str, int]:
    normalized = normalize_unified_profile_name(profile_name)
    return dict(UNIFIED_BUDGETS[normalized])


def unified_interface_config(profile_name: str | None) -> Dict[str, Any]:
    normalized = normalize_unified_profile_name(profile_name)
    return dict(UNIFIED_PROFILE_INTERFACE_LOCKS[normalized])


def unified_rendering_overrides(profile_name: str | None) -> Dict[str, Any]:
    normalized = normalize_unified_profile_name(profile_name)
    return dict(UNIFIED_PROFILE_RENDERING_OVERRIDES.get(normalized, {}))


def unified_budget_config(profile_name: str | None) -> Dict[str, int]:
    budget = unified_budget(profile_name)
    return {
        "semantic_topn_entity": int(budget["seed_topn"]),
        "semantic_topn_chunk": int(budget["run_topn"]),
        "graph_reserve_topn": int(budget["render_topn"]),
    }


def apply_unified_profile(
    cfg: Mapping[str, Any] | None,
    dataset_name: str | None,
    profile_name: str | None = COPY_SPAN_INSTRUCTION_UNIFIED_MEDIUM,
) -> Dict[str, Any]:
    ds = normalize_dataset_name(dataset_name)
    if ds not in DATASET_ORDER:
        supported = ", ".join(DATASET_ORDER)
        raise ValueError(f"Unsupported strict unified dataset '{dataset_name}'. Supported: {supported}")

    normalized_profile = normalize_unified_profile_name(profile_name)
    out = deepcopy(dict(cfg or {}))
    out["dataset"] = ds
    out.update(COPY_SPAN_RETRIEVAL_WEIGHTS)
    out.update(COPY_SPAN_RENDER_WEIGHTS)
    out.update(COPY_SPAN_DISABLED_MODULES)
    out.update(UNIFIED_METHOD_LOCK)
    out["retrieval_objective_mode"] = resolve_unified_copy_span_mode(ds)
    out.update(unified_interface_config(normalized_profile))
    out.update(unified_rendering_overrides(normalized_profile))
    out.update(unified_budget_config(normalized_profile))
    out.update(UNIFIED_RUNTIME_DEFAULTS)
    out.update(
        {
            "method_name": "copy-span",
            "method_profile": normalized_profile,
            "profile_family": STRICT_UNIFIED_PROFILE_FAMILY,
            "reference_label": REFERENCE_LABEL,
            "sota_reference": False,
            "strict_unified_main": True,
            "dataset_independent_method": True,
        }
    )
    return out


def unified_method_signature(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: cfg.get(key) for key in UNIFIED_METHOD_SETTING_KEYS}


def unified_budget_signature(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: cfg.get(key) for key in UNIFIED_BUDGET_CONFIG_KEYS}


def audit_unified_configs(configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    datasets = sorted(configs)
    errors = []
    method_signatures = {dataset: unified_method_signature(cfg) for dataset, cfg in configs.items()}
    budget_signatures = {dataset: unified_budget_signature(cfg) for dataset, cfg in configs.items()}

    if datasets:
        first_method = method_signatures[datasets[0]]
        first_budget = budget_signatures[datasets[0]]
        for dataset in datasets[1:]:
            if method_signatures[dataset] != first_method:
                errors.append(f"method setting drift for {dataset}")
            if budget_signatures[dataset] != first_budget:
                errors.append(f"budget setting drift for {dataset}")

    for dataset, cfg in configs.items():
        if cfg.get("retrieval_objective_mode") not in {"baseline", "canonical_copy_span_unified"}:
            errors.append(f"unsupported unified objective for {dataset}: {cfg.get('retrieval_objective_mode')}")
        for key, expected in UNIFIED_RENDERING_LOCK.items():
            if key == "final_top_slice_reorder_enabled":
                # Reorder-all profiles may enable this globally. The signature
                # comparison above ensures it cannot drift by dataset.
                continue
            if bool(cfg.get(key)) is not bool(expected):
                errors.append(f"unified rendering lock violated for {dataset}.{key}")

    return {
        "datasets": datasets,
        "method_signatures": method_signatures,
        "budget_signatures": budget_signatures,
        "errors": errors,
        "ok": not errors,
    }
