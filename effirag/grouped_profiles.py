from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from effirag.profiles import (
    COPY_SPAN_DATASET_LOCKS,
    COPY_SPAN_RENDER_WEIGHTS,
    COPY_SPAN_RETRIEVAL_WEIGHTS,
)

REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN = "light_separator_copy_span_instruction"
PREVIOUS_REFERENCE_PROFILE = "champion_reconfirm"

COPY_SPAN_INSTRUCTION_GROUPED_V0 = "copy_span_instruction_grouped_v0"
COPY_SPAN_INSTRUCTION_GROUPED_V1 = "copy_span_instruction_grouped_v1"
COPY_SPAN_INSTRUCTION_MINIMAL_V1 = "copy_span_instruction_minimal_v1"
COPY_SPAN_INSTRUCTION_NO_DATASET_TOGGLE = "copy_span_instruction_no_dataset_toggle"

INSTRUCTION_GROUPED_PROFILE_NAMES = (
    COPY_SPAN_INSTRUCTION_GROUPED_V0,
    COPY_SPAN_INSTRUCTION_GROUPED_V1,
    COPY_SPAN_INSTRUCTION_MINIMAL_V1,
    COPY_SPAN_INSTRUCTION_NO_DATASET_TOGGLE,
)

REFERENCE_LABEL = "copy-span/light_separator_copy_span_instruction"

INSTRUCTION_INTERFACE_CONTRACT: Dict[str, Any] = {
    "order_strategy": "score+light_separator_render",
    "prompt_variant": "light_separator_copy_span_instruction",
    "prompt_variant_label": "light_separator_copy_span_instruction",
    "render_variant": "light_separator_render",
    "added_instruction": "copy_span_only",
}

NO_DATASET_TOGGLE_LOCK: Dict[str, Any] = {
    "retrieval_objective_mode": "baseline",
    "answer_support_pinning_enabled": False,
    "final_top_slice_reorder_enabled": False,
    "precomputed_retrieval_strict": False,
}

INSTRUCTION_REFERENCE_WEIGHTS: Dict[str, float] = {
    **COPY_SPAN_RETRIEVAL_WEIGHTS,
    **COPY_SPAN_RENDER_WEIGHTS,
}

_GROUPED_V0_SEED: Dict[str, float] = {
    "seed_group_relevance_weight": 0.55,
    "seed_group_structure_weight": 0.45,
    "seed_relevance_semantic_ratio": 0.6363636364,
    "seed_relevance_anchor_ratio": 0.3636363636,
}

_GROUPED_V0_RUN: Dict[str, float] = {
    "run_group_relevance_weight": 0.50,
    "run_group_structure_weight": 0.40,
    "run_group_redundancy_weight": 0.10,
    "run_relevance_semantic_ratio": 0.60,
    "run_relevance_anchor_ratio": 0.40,
    "run_structure_structure_ratio": 0.625,
    "run_structure_bridge_ratio": 0.375,
}

_GROUPED_V0_RENDER: Dict[str, float] = {
    "alpha": 1.0,
    "beta": 0.35,
    "render_group_evidence_weight": 0.85,
    "render_group_query_weight": 0.12,
    "render_group_locality_weight": 0.08,
    "render_group_redundancy_weight": 0.25,
    "evidence_main_ratio": 0.5294117647,
    "evidence_support_ratio": 0.2352941176,
    "evidence_connector_ratio": 0.2352941176,
}

GROUPED_PROFILE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    COPY_SPAN_INSTRUCTION_GROUPED_V0: {
        "grouped_profile_version": "v0",
        "seed": dict(_GROUPED_V0_SEED),
        "run": dict(_GROUPED_V0_RUN),
        "render": dict(_GROUPED_V0_RENDER),
    },
    COPY_SPAN_INSTRUCTION_GROUPED_V1: {
        "grouped_profile_version": "v1",
        "seed": dict(_GROUPED_V0_SEED),
        "run": {
            **_GROUPED_V0_RUN,
            # Alias name for family-level semantics while preserving behavior.
            "run_group_structural_utility_weight": _GROUPED_V0_RUN["run_group_structure_weight"],
        },
        "render": dict(_GROUPED_V0_RENDER),
    },
    COPY_SPAN_INSTRUCTION_MINIMAL_V1: {
        "grouped_profile_version": "minimal_v1",
        "seed": dict(_GROUPED_V0_SEED),
        "run": dict(_GROUPED_V0_RUN),
        "render": dict(_GROUPED_V0_RENDER),
    },
    COPY_SPAN_INSTRUCTION_NO_DATASET_TOGGLE: {
        "grouped_profile_version": "no_dataset_toggle",
        "seed": dict(_GROUPED_V0_SEED),
        "run": dict(_GROUPED_V0_RUN),
        "render": dict(_GROUPED_V0_RENDER),
        "disable_dataset_toggle": True,
    },
}


def _normalize_dataset_name(dataset: str) -> str:
    raw = str(dataset or "").strip().lower()
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


def _f(mapping: Mapping[str, Any], key: str) -> float:
    return float(mapping[key])


def expand_grouped_seed_weights(profile: Mapping[str, Any]) -> Dict[str, float]:
    relevance_weight = _f(profile, "seed_group_relevance_weight")
    structure_weight = _f(profile, "seed_group_structure_weight")
    semantic_ratio = _f(profile, "seed_relevance_semantic_ratio")
    anchor_ratio = _f(profile, "seed_relevance_anchor_ratio")
    return {
        "seed_score_semantic_weight": relevance_weight * semantic_ratio,
        "seed_score_graph_weight": structure_weight,
        "seed_score_anchor_weight": relevance_weight * anchor_ratio,
    }


def expand_grouped_run_weights(profile: Mapping[str, Any]) -> Dict[str, float]:
    relevance_weight = _f(profile, "run_group_relevance_weight")
    structure_weight = float(
        profile.get("run_group_structural_utility_weight", profile["run_group_structure_weight"])
    )
    redundancy_weight = _f(profile, "run_group_redundancy_weight")
    semantic_ratio = _f(profile, "run_relevance_semantic_ratio")
    anchor_ratio = _f(profile, "run_relevance_anchor_ratio")
    structure_ratio = _f(profile, "run_structure_structure_ratio")
    bridge_ratio = _f(profile, "run_structure_bridge_ratio")
    return {
        "run_score_semantic_weight": relevance_weight * semantic_ratio,
        "run_score_anchor_weight": relevance_weight * anchor_ratio,
        "run_score_structure_weight": structure_weight * structure_ratio,
        "run_score_bridge_weight": structure_weight * bridge_ratio,
        "run_score_redundancy_weight": redundancy_weight,
    }


def expand_grouped_render_weights(profile: Mapping[str, Any]) -> Dict[str, float]:
    evidence_weight = _f(profile, "render_group_evidence_weight")
    main_ratio = _f(profile, "evidence_main_ratio")
    support_ratio = _f(profile, "evidence_support_ratio")
    connector_ratio = _f(profile, "evidence_connector_ratio")
    return {
        "alpha": float(profile.get("alpha", 1.0)),
        "beta": float(profile.get("beta", 0.35)),
        "gamma_main": evidence_weight * main_ratio,
        "delta_support": evidence_weight * support_ratio,
        "eta_connector": evidence_weight * connector_ratio,
        "zeta_query": _f(profile, "render_group_query_weight"),
        "xi_locality": _f(profile, "render_group_locality_weight"),
        "lambda_redundancy": _f(profile, "render_group_redundancy_weight"),
    }


def _apply_dataset_locks(cfg: Dict[str, Any], dataset: str, disable_dataset_toggle: bool) -> None:
    if disable_dataset_toggle:
        cfg.update(NO_DATASET_TOGGLE_LOCK)
        return
    ds = _normalize_dataset_name(dataset)
    if ds not in COPY_SPAN_DATASET_LOCKS:
        raise ValueError(f"Unsupported dataset '{dataset}' for copy-span instruction grouped profiles")
    cfg.update(dict(COPY_SPAN_DATASET_LOCKS[ds]))


def apply_grouped_profile(
    cfg: Mapping[str, Any],
    dataset: str,
    profile_name: str,
    *,
    reference_profile: str = REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
) -> Dict[str, Any]:
    if str(reference_profile) != REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN:
        raise ValueError(
            "instruction grouped profiles require explicit reference_profile="
            f"{REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN}"
        )

    allowed = set(INSTRUCTION_GROUPED_PROFILE_NAMES) | {REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN}
    if profile_name not in allowed:
        raise ValueError(f"Unsupported profile '{profile_name}'")

    out = deepcopy(dict(cfg or {}))
    out.update(INSTRUCTION_INTERFACE_CONTRACT)

    if profile_name == REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN:
        _apply_dataset_locks(out, dataset, disable_dataset_toggle=False)
        out.update(INSTRUCTION_REFERENCE_WEIGHTS)
        out.update(
            {
                "method_name": "copy-span",
                "method_profile": REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
                "reference_profile": REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
                "reference_label": REFERENCE_LABEL,
                "sota_reference": True,
                "previous_reference_profile": PREVIOUS_REFERENCE_PROFILE,
                "retarget_reason": "generation_interface_sota",
                "profile_family": "copy_span_instruction_grouped",
                "grouped_profile_version": "reference",
                "retargeted_from": "champion_reconfirm_grouped_phase2",
            }
        )
        return out

    definition = GROUPED_PROFILE_DEFINITIONS[profile_name]
    seed = expand_grouped_seed_weights(definition["seed"])
    run = expand_grouped_run_weights(definition["run"])
    render = expand_grouped_render_weights(definition["render"])

    out.update(seed)
    out.update(run)
    out.update(render)
    out["embedding_weight"] = float(INSTRUCTION_REFERENCE_WEIGHTS["embedding_weight"])
    _apply_dataset_locks(out, dataset, disable_dataset_toggle=bool(definition.get("disable_dataset_toggle", False)))

    out.update(
        {
            "method_name": "copy-span",
            "method_profile": profile_name,
            "reference_profile": REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
            "reference_label": REFERENCE_LABEL,
            "sota_reference": True,
            "previous_reference_profile": PREVIOUS_REFERENCE_PROFILE,
            "retarget_reason": "generation_interface_sota",
            "profile_family": "copy_span_instruction_grouped",
            "grouped_profile_version": str(definition["grouped_profile_version"]),
            "retargeted_from": "champion_reconfirm_grouped_phase2",
        }
    )
    if profile_name == COPY_SPAN_INSTRUCTION_GROUPED_V0:
        out["expanded_seed_weights"] = dict(seed)
        out["expanded_run_weights"] = dict(run)
        out["expanded_render_weights"] = dict(render)
    return out


def instruction_reference_contract(dataset: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    cfg.update(INSTRUCTION_REFERENCE_WEIGHTS)
    cfg.update(INSTRUCTION_INTERFACE_CONTRACT)
    _apply_dataset_locks(cfg, dataset, disable_dataset_toggle=False)
    return cfg
