from __future__ import annotations

from typing import Any, Dict

from .canonical_scoring import CANONICAL_PRUNED_WEIGHT_FIELDS

CANONICAL_COPY_SPAN_MODE = "canonical_copy_span"
CANONICAL_NO_SEED_OPTIONAL_MODE = "canonical_no_seed_optional"
CANONICAL_NO_RUN_OPTIONAL_MODE = "canonical_no_run_optional"
CANONICAL_NO_PAIR_COVERAGE_MODE = "canonical_no_pair_coverage"
CANONICAL_NO_BRIDGE_COMPLETENESS_MODE = "canonical_no_bridge_completeness"
CANONICAL_RENDER_CORE_ONLY_MODE = "canonical_render_core_only"
CANONICAL_NO_DATASET_GUARD_MODE = "canonical_no_dataset_guard"

CANONICAL_OBJECTIVE_MODES = (
    CANONICAL_COPY_SPAN_MODE,
    CANONICAL_NO_SEED_OPTIONAL_MODE,
    CANONICAL_NO_RUN_OPTIONAL_MODE,
    CANONICAL_NO_PAIR_COVERAGE_MODE,
    CANONICAL_NO_BRIDGE_COMPLETENESS_MODE,
    CANONICAL_RENDER_CORE_ONLY_MODE,
    CANONICAL_NO_DATASET_GUARD_MODE,
)


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


def _normalize_mode(mode: str | None) -> str:
    return str(mode or "").strip().lower()


def is_canonical_copy_span_mode(mode: str | None) -> bool:
    return _normalize_mode(mode) in set(CANONICAL_OBJECTIVE_MODES)


def _mode_options(mode: str | None) -> Dict[str, bool]:
    mm = _normalize_mode(mode)
    return {
        "no_seed_optional": mm == CANONICAL_NO_SEED_OPTIONAL_MODE,
        "no_run_optional": mm == CANONICAL_NO_RUN_OPTIONAL_MODE,
        "no_pair_coverage": mm == CANONICAL_NO_PAIR_COVERAGE_MODE,
        "no_bridge_completeness": mm == CANONICAL_NO_BRIDGE_COMPLETENESS_MODE,
        "render_core_only": mm == CANONICAL_RENDER_CORE_ONLY_MODE,
        "no_dataset_guard": mm == CANONICAL_NO_DATASET_GUARD_MODE,
    }


def canonical_effective_reference_mode(dataset: str | None, *, mode: str | None = None) -> str:
    """
    Resolve compatibility mode for canonical_copy_span.

    Notes
    -----
    The canonical retrieval/scoring core is dataset-agnostic. This function
    only exposes a compatibility adapter layer for dataset-specific guards
    (currently hotpot guarded answer-preserve), so behavior remains consistent
    with previously validated reference runs.
    """
    ds = _normalize_dataset_name(dataset)
    opts = _mode_options(mode)
    if ds == "hotpotqa":
        if opts["no_dataset_guard"]:
            return "p3_answer_preserve_base"
        return "p3_answer_preserve_guarded_hotpot"
    return "baseline"


def is_dataset_specific_guard_enabled(dataset: str | None, *, mode: str | None = None) -> bool:
    ds = _normalize_dataset_name(dataset)
    if ds != "hotpotqa":
        return False
    return canonical_effective_reference_mode(ds, mode=mode) == "p3_answer_preserve_guarded_hotpot"


def _canonical_flags(dataset: str | None, *, mode: str | None = None) -> Dict[str, bool]:
    ds = _normalize_dataset_name(dataset)
    effective_mode = canonical_effective_reference_mode(ds, mode=mode)
    answer_preserve = effective_mode in {"p3_answer_preserve_base", "p3_answer_preserve_guarded_hotpot"}
    guarded_hotpot = is_dataset_specific_guard_enabled(ds, mode=mode)
    bridge_induction = effective_mode in {"p3_answer_preserve_base", "p3_answer_preserve_guarded_hotpot"}
    return {
        "bridge_candidate_induction_enabled": bool(bridge_induction),
        "role_aware_chunk_scoring_enabled": False,
        "coverage_selection_enabled": False,
        "corridor_compact_shaping_enabled": False,
        "corridor_answer_preserve_enabled": bool(answer_preserve),
        "corridor_bridge_purity_shaping_enabled": False,
        "corridor_path_preserve_enabled": False,
        "corridor_path_preserve_compact_enabled": False,
        "corridor_path_preserve_guarded_enabled": False,
        "corridor_path_preserve_compact_lite_enabled": False,
        "corridor_answer_preserve_guarded_hotpot_enabled": bool(guarded_hotpot),
        "corridor_answer_preserve_confidence_gated_enabled": False,
    }


def _enforce_canonical_pruned_weights_zero(cfg) -> Dict[str, float]:
    changed: Dict[str, float] = {}
    for field in CANONICAL_PRUNED_WEIGHT_FIELDS:
        if not hasattr(cfg, field):
            continue
        raw = getattr(cfg, field)
        try:
            current = float(raw)
        except Exception:
            current = 0.0
        if abs(current) <= 0.0:
            continue
        changed[field] = float(current)
        setattr(cfg, field, 0.0)
    return changed


def apply_canonical_copy_span_objective(cfg, *, dataset: str | None = None, mode: str | None = None):
    """
    Apply the canonical copy-span retrieval objective.

    This reproduces the validated copy-span instruction behavior by mapping
    canonical modes onto existing reference behavior (hotpot guarded answer-
    preserve vs other datasets baseline), then optionally applying explicit
    ablation overrides.
    """
    ds = _normalize_dataset_name(dataset or getattr(cfg, "dataset", ""))
    mm = _normalize_mode(mode or getattr(cfg, "retrieval_objective_mode", CANONICAL_COPY_SPAN_MODE))
    if mm not in set(CANONICAL_OBJECTIVE_MODES):
        mm = CANONICAL_COPY_SPAN_MODE

    setattr(cfg, "retrieval_objective_mode", mm)

    flags = _canonical_flags(ds, mode=mm)
    for key, value in flags.items():
        if hasattr(cfg, key):
            setattr(cfg, key, bool(value))

    opts = _mode_options(mm)
    pruned_weight_overrides: Dict[str, float] = {}

    # Canonical mainline invariant: optional/pruned score terms never influence
    # canonical_copy_span behavior, even if a config accidentally sets them.
    if mm == CANONICAL_COPY_SPAN_MODE:
        pruned_weight_overrides = _enforce_canonical_pruned_weights_zero(cfg)

    if opts["no_seed_optional"]:
        if hasattr(cfg, "seed_score_bridge_weight"):
            setattr(cfg, "seed_score_bridge_weight", 0.0)
        if hasattr(cfg, "seed_score_chunk_grounding_weight"):
            setattr(cfg, "seed_score_chunk_grounding_weight", 0.0)

    if opts["no_run_optional"]:
        if hasattr(cfg, "run_score_entity_chunk_grounding_weight"):
            setattr(cfg, "run_score_entity_chunk_grounding_weight", 0.0)
        if hasattr(cfg, "run_score_anchor_dispersion_penalty"):
            setattr(cfg, "run_score_anchor_dispersion_penalty", 0.0)

    if opts["no_pair_coverage"] and hasattr(cfg, "run_score_pair_coverage_weight"):
        setattr(cfg, "run_score_pair_coverage_weight", 0.0)

    if opts["no_bridge_completeness"] and hasattr(cfg, "run_score_bridge_completeness_weight"):
        setattr(cfg, "run_score_bridge_completeness_weight", 0.0)

    if opts["render_core_only"]:
        if hasattr(cfg, "zeta_query"):
            setattr(cfg, "zeta_query", 0.0)
        if hasattr(cfg, "xi_locality"):
            setattr(cfg, "xi_locality", 0.0)

    return {
        "mode": mm,
        "dataset": ds,
        "effective_reference_mode": canonical_effective_reference_mode(ds, mode=mm),
        "flags": dict(flags),
        "options": dict(opts),
        "pruned_weight_overrides": dict(pruned_weight_overrides),
    }
