from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from effirag.canonical_objective import (
    CANONICAL_COPY_SPAN_MODE,
    CANONICAL_OBJECTIVE_MODES,
    apply_canonical_copy_span_objective,
)
from effirag.canonical_scoring import (
    CANONICAL_CORE_WEIGHT_FIELDS,
    CANONICAL_PRUNED_WEIGHT_FIELDS,
)
from effirag.config import RagConfig, dataclass_from_dict
from effirag.grouped_profiles import (
    INSTRUCTION_GROUPED_PROFILE_NAMES,
    REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
    apply_grouped_profile,
)
from effirag.profiles import COPY_SPAN_DATASET_LOCKS
from effirag.retrieval import _apply_connector_objective_profile, _resolve_retrieval_objective_flags


TRACE_FLAG_FIELDS = (
    "bridge_candidate_induction_enabled",
    "role_aware_chunk_scoring_enabled",
    "coverage_selection_enabled",
    "corridor_compact_shaping_enabled",
    "corridor_answer_preserve_enabled",
    "corridor_bridge_purity_shaping_enabled",
    "corridor_path_preserve_enabled",
    "corridor_path_preserve_compact_enabled",
    "corridor_path_preserve_guarded_enabled",
    "corridor_path_preserve_compact_lite_enabled",
    "corridor_answer_preserve_guarded_hotpot_enabled",
    "corridor_answer_preserve_confidence_gated_enabled",
    "answer_support_pinning_enabled",
    "copy_span_instruction_enabled",
)

TRACE_WEIGHT_FIELDS = (
    "seed_score_semantic_weight",
    "seed_score_graph_weight",
    "seed_score_anchor_weight",
    "seed_score_bridge_weight",
    "seed_score_chunk_grounding_weight",
    "seed_objective_bridge_weight",
    "seed_objective_chunk_grounding_weight",
    "seed_objective_anchor_coverage_weight",
    "run_score_semantic_weight",
    "run_score_anchor_weight",
    "run_score_structure_weight",
    "run_score_bridge_weight",
    "run_score_redundancy_weight",
    "run_score_pair_coverage_weight",
    "run_score_bridge_completeness_weight",
    "run_score_entity_chunk_grounding_weight",
    "run_score_anchor_dispersion_penalty",
    "corridor_score_structure_weight",
    "corridor_score_semantic_weight",
    "corridor_score_answer_weight",
    "corridor_score_pair_weight",
    "corridor_score_chunk_support_weight",
    "corridor_score_answer_alignment_weight",
    "alpha",
    "beta",
    "gamma_main",
    "delta_support",
    "eta_connector",
    "zeta_query",
    "xi_locality",
    "lambda_redundancy",
)

OBJECTIVE_FLAG_KEY_MAP = {
    "bridge_candidate_induction_enabled": "bridge_candidate_induction",
    "role_aware_chunk_scoring_enabled": "role_aware_chunk_scoring",
    "coverage_selection_enabled": "coverage_selection",
    "corridor_compact_shaping_enabled": "corridor_compact_shaping",
    "corridor_answer_preserve_enabled": "corridor_answer_preserve_shaping",
    "corridor_bridge_purity_shaping_enabled": "corridor_bridge_purity_shaping",
    "corridor_path_preserve_enabled": "corridor_path_preserve_shaping",
    "corridor_path_preserve_compact_enabled": "corridor_path_preserve_compact_shaping",
    "corridor_path_preserve_guarded_enabled": "corridor_path_preserve_guarded_shaping",
    "corridor_path_preserve_compact_lite_enabled": "corridor_path_preserve_compact_lite_shaping",
    "corridor_answer_preserve_guarded_hotpot_enabled": "corridor_answer_preserve_guarded_hotpot",
    "corridor_answer_preserve_confidence_gated_enabled": "corridor_answer_preserve_confidence_gated",
}


def _normalize_dataset_name(dataset: str | None) -> str:
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


def _as_rag_config(cfg: Any) -> RagConfig:
    if isinstance(cfg, RagConfig):
        return deepcopy(cfg)
    if hasattr(cfg, "__dict__") and not isinstance(cfg, Mapping):
        payload = dict(vars(cfg))
    elif isinstance(cfg, Mapping):
        payload = dict(cfg)
    else:
        payload = {}
    return dataclass_from_dict(RagConfig, payload, warn_unknown_keys=False)


def _maybe_apply_profile(
    cfg: Mapping[str, Any],
    *,
    dataset: str,
    profile: str | None,
) -> Dict[str, Any]:
    out = deepcopy(dict(cfg or {}))
    if not profile:
        return out

    if profile in set(INSTRUCTION_GROUPED_PROFILE_NAMES) | {REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN}:
        return apply_grouped_profile(
            out,
            dataset=dataset,
            profile_name=profile,
            reference_profile=REFERENCE_PROFILE_LIGHT_SEPARATOR_COPY_SPAN,
        )

    if profile in CANONICAL_OBJECTIVE_MODES:
        out["retrieval_objective_mode"] = profile
        return out

    return out


def _get_attr(cfg: Any, key: str) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return None


def _split_flags(cfg: RagConfig, field_names: Iterable[str]) -> tuple[Dict[str, bool], Dict[str, bool], Dict[str, None]]:
    active: Dict[str, bool] = {}
    inactive: Dict[str, bool] = {}
    absent: Dict[str, None] = {}
    for name in field_names:
        if not hasattr(cfg, name):
            absent[name] = None
            continue
        vv = bool(getattr(cfg, name))
        if vv:
            active[name] = True
        else:
            inactive[name] = False
    return active, inactive, absent


def _overlay_effective_objective_flags(
    active: Dict[str, bool],
    inactive: Dict[str, bool],
    objective_flags: Mapping[str, Any],
) -> tuple[Dict[str, bool], Dict[str, bool]]:
    active_out = dict(active)
    inactive_out = dict(inactive)
    for field_name, objective_key in OBJECTIVE_FLAG_KEY_MAP.items():
        if objective_key not in objective_flags:
            continue
        vv = bool(objective_flags.get(objective_key))
        if vv:
            active_out[field_name] = True
            inactive_out.pop(field_name, None)
        else:
            inactive_out[field_name] = False
            active_out.pop(field_name, None)
    return active_out, inactive_out


def _split_weights(cfg: RagConfig, field_names: Iterable[str]) -> tuple[Dict[str, float], Dict[str, float], Dict[str, None]]:
    active: Dict[str, float] = {}
    inactive: Dict[str, float] = {}
    absent: Dict[str, None] = {}
    for name in field_names:
        if not hasattr(cfg, name):
            absent[name] = None
            continue
        raw = getattr(cfg, name)
        try:
            vv = float(raw)
        except Exception:
            absent[name] = None
            continue
        if abs(vv) > 0.0:
            active[name] = vv
        else:
            inactive[name] = vv
    return active, inactive, absent


def trace_active_pipeline(cfg, *, dataset: str | None = None, profile: str | None = None) -> dict:
    ds = _normalize_dataset_name(dataset)
    if not ds and isinstance(cfg, Mapping):
        ds = _normalize_dataset_name(cfg.get("dataset"))
    if not ds:
        ds = "hotpotqa"

    cfg_dict = _maybe_apply_profile(cfg if isinstance(cfg, Mapping) else vars(cfg), dataset=ds, profile=profile)
    if not cfg_dict.get("dataset"):
        cfg_dict["dataset"] = ds

    cfg_obj = _as_rag_config(cfg_dict)
    requested_mode = str(getattr(cfg_obj, "retrieval_objective_mode", "baseline") or "baseline").strip().lower()

    objective_flags = _resolve_retrieval_objective_flags(cfg_obj)
    objective_flags, objective_profile_diag = _apply_connector_objective_profile(cfg=cfg_obj, objective_flags=objective_flags)

    # canonical mode keeps one explicit path; ensure trace captures post-application state.
    if requested_mode in CANONICAL_OBJECTIVE_MODES:
        apply_canonical_copy_span_objective(cfg_obj, dataset=ds, mode=requested_mode)

    active_flags, inactive_flags, absent_flags = _split_flags(cfg_obj, TRACE_FLAG_FIELDS)
    active_flags, inactive_flags = _overlay_effective_objective_flags(active_flags, inactive_flags, objective_flags)
    active_weights, inactive_weights, absent_weights = _split_weights(cfg_obj, TRACE_WEIGHT_FIELDS)

    canonical_mode = requested_mode in set(CANONICAL_OBJECTIVE_MODES)
    canonical_core_active_weights: Dict[str, float] = {}
    canonical_legacy_or_pruned_nonzero_weights: Dict[str, float] = {}
    canonical_pruned_zero_weights: Dict[str, float] = {}
    if canonical_mode:
        for key in CANONICAL_PRUNED_WEIGHT_FIELDS:
            if key in active_weights:
                canonical_legacy_or_pruned_nonzero_weights[key] = float(active_weights.pop(key))
            elif key in inactive_weights:
                canonical_pruned_zero_weights[key] = float(inactive_weights.get(key, 0.0))
        canonical_core_active_weights = {
            key: float(val)
            for key, val in active_weights.items()
            if key in set(CANONICAL_CORE_WEIGHT_FIELDS)
        }

    core_modules = {
        "query_conditioned_proposal": {
            "bridge_candidate_induction_enabled": bool(_get_attr(cfg_obj, "bridge_candidate_induction_enabled")),
        },
        "stochastic_multi_run_ppr": {
            "ppr_engine": str(getattr(cfg_obj, "ppr_engine", "auto")),
            "tau": int(getattr(cfg_obj, "tau", 4)),
            "samples_per_anchor": int(getattr(cfg_obj, "samples_per_anchor", 3)),
        },
        "bounded_local_refinement": {
            "phase2_refine_mode": str(getattr(cfg_obj, "phase2_refine_mode", "")),
            "pair_top_lp": int(getattr(cfg_obj, "pair_top_lp", 0)),
        },
        "compact_rendering": {
            "render_mode": str(getattr(cfg_obj, "render_mode", "")),
            "order_strategy": str(getattr(cfg_obj, "order_strategy", "")),
        },
        "copy_span_interface": {
            "prompt_variant": str(getattr(cfg_obj, "prompt_variant", "")),
            "added_instruction": str(getattr(cfg_obj, "added_instruction", "")),
        },
    }

    legacy_like_modules = {
        "path_preserve_variants": {
            "corridor_path_preserve_enabled": bool(_get_attr(cfg_obj, "corridor_path_preserve_enabled")),
            "corridor_path_preserve_compact_enabled": bool(_get_attr(cfg_obj, "corridor_path_preserve_compact_enabled")),
            "corridor_path_preserve_guarded_enabled": bool(_get_attr(cfg_obj, "corridor_path_preserve_guarded_enabled")),
            "corridor_path_preserve_compact_lite_enabled": bool(
                _get_attr(cfg_obj, "corridor_path_preserve_compact_lite_enabled")
            ),
        },
        "corridor_experimental_variants": {
            "corridor_compact_shaping_enabled": bool(_get_attr(cfg_obj, "corridor_compact_shaping_enabled")),
            "corridor_bridge_purity_shaping_enabled": bool(_get_attr(cfg_obj, "corridor_bridge_purity_shaping_enabled")),
            "corridor_answer_preserve_confidence_gated_enabled": bool(
                _get_attr(cfg_obj, "corridor_answer_preserve_confidence_gated_enabled")
            ),
        },
    }
    if canonical_mode:
        legacy_like_modules["canonical_pruned_weight_terms"] = {
            "nonzero_pruned_or_legacy_weights": dict(canonical_legacy_or_pruned_nonzero_weights),
            "zero_pruned_weights": dict(canonical_pruned_zero_weights),
        }

    notes = []
    if absent_flags:
        notes.append(f"absent_flag_fields={sorted(absent_flags.keys())}")
    if absent_weights:
        notes.append(f"absent_weight_fields={sorted(absent_weights.keys())}")
    if "seed_objective_chunk_grounding_weight" in absent_weights and hasattr(cfg_obj, "seed_objective_grounding_weight"):
        notes.append(
            "seed_objective_chunk_grounding_weight absent; using existing field seed_objective_grounding_weight in current config"
        )
    if ds in COPY_SPAN_DATASET_LOCKS:
        notes.append(f"dataset_lock={COPY_SPAN_DATASET_LOCKS[ds]}")
    if canonical_mode and canonical_legacy_or_pruned_nonzero_weights:
        notes.append(
            "canonical_mode_pruned_weights_present_nonzero="
            + str(sorted(canonical_legacy_or_pruned_nonzero_weights.keys()))
        )

    return {
        "dataset": ds,
        "profile": profile,
        "retrieval_objective_mode": str(objective_flags.get("mode", requested_mode)),
        "requested_retrieval_objective_mode": requested_mode,
        "active_flags": active_flags,
        "inactive_flags": inactive_flags,
        "active_nonzero_weights": active_weights,
        "inactive_zero_weights": inactive_weights,
        "canonical_mode": bool(canonical_mode),
        "canonical_active_core_weights": canonical_core_active_weights,
        "canonical_legacy_or_pruned_nonzero_weights": canonical_legacy_or_pruned_nonzero_weights,
        "core_modules": core_modules,
        "legacy_like_modules": legacy_like_modules,
        "objective_profile_diag": objective_profile_diag,
        "notes": notes,
    }
