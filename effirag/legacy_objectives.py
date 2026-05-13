from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from effirag.canonical_objective import CANONICAL_OBJECTIVE_MODES

CORE_OBJECTIVE_MODES = {
    "baseline",
    "hybrid_anchor_recall",
    "bridge_candidate_induction",
    "role_aware_chunk_scoring",
    "coverage_selection",
    "bridge_coverage_full",
}

LEGACY_OBJECTIVE_MODES = {
    # Connector-lite refinement modes
    "r2_bridge_only",
    "r2_connector_core",
    "r2_plus_r3_anchor_light",
    "r2_plus_r3_answer_light",
    "r2_plus_path_preserve",
    "r2_plus_path_preserve_compact",
    "r2_plus_path_preserve_guarded",
    "r2_plus_path_preserve_compact_lite",
    # PAMAE-style bounded-budget connector rounds
    "seed_quality_analysis",
    "run_objective_bridge_aware",
    "run_objective_role_balanced",
    "corridor_role_constrained",
    "bridge_aware_run_plus_role_constrained_corridor",
    # PAMAE seed-run-corridor refinement variants
    "seed_run_connector_core",
    "seed_run_connector_core_corridor_compact",
    "seed_run_connector_core_corridor_answer_preserve",
    "seed_run_connector_core_corridor_bridge_purity",
    "seed_run_connector_core_corridor_compact_answer_preserve",
    # Guarded answer-preserve refinement round
    "p3_answer_preserve_base",
    "p3_answer_preserve_guarded_hotpot",
    "p3_answer_preserve_confidence_gated",
}

OBJECTIVE_MODE_ALIASES = {
    "off": "baseline",
    "default": "baseline",
    "r1": "hybrid_anchor_recall",
    "r2": "bridge_candidate_induction",
    "r3": "role_aware_chunk_scoring",
    "r4": "coverage_selection",
    "full": "bridge_coverage_full",
    "bridge_coverage": "bridge_coverage_full",
    # Connector-lite refinement aliases
    "r2_bridge": "r2_bridge_only",
    "r2_bridge_only": "r2_bridge_only",
    "r2_connector_core": "r2_connector_core",
    "r2+r3_anchor_light": "r2_plus_r3_anchor_light",
    "r2_plus_r3_anchor_light": "r2_plus_r3_anchor_light",
    "r2+r3_answer_light": "r2_plus_r3_answer_light",
    "r2_plus_r3_answer_light": "r2_plus_r3_answer_light",
    "r2_path": "r2_plus_path_preserve",
    "r2_plus_path_preserve": "r2_plus_path_preserve",
    "r2_path_compact": "r2_plus_path_preserve_compact",
    "r2_plus_path_preserve_compact": "r2_plus_path_preserve_compact",
    "r2_path_guarded": "r2_plus_path_preserve_guarded",
    "r2_plus_path_preserve_guarded": "r2_plus_path_preserve_guarded",
    "r2_path_compact_lite": "r2_plus_path_preserve_compact_lite",
    "r2_plus_path_preserve_compact_lite": "r2_plus_path_preserve_compact_lite",
    # PAMAE-style bounded-budget connector rounds
    "p1": "seed_quality_analysis",
    "p2_bridge": "run_objective_bridge_aware",
    "p2_role": "run_objective_role_balanced",
    "p3_corridor": "corridor_role_constrained",
    "p3_combo": "bridge_aware_run_plus_role_constrained_corridor",
    # PAMAE seed-run-corridor refinement variants
    "pamae_seed_run_core": "seed_run_connector_core",
    "seed_run_connector_core": "seed_run_connector_core",
    "p2_corridor_compact": "seed_run_connector_core_corridor_compact",
    "seed_run_connector_core_corridor_compact": "seed_run_connector_core_corridor_compact",
    "p3_corridor_answer_preserve": "seed_run_connector_core_corridor_answer_preserve",
    "seed_run_connector_core_corridor_answer_preserve": "seed_run_connector_core_corridor_answer_preserve",
    "p4_corridor_bridge_purity": "seed_run_connector_core_corridor_bridge_purity",
    "seed_run_connector_core_corridor_bridge_purity": "seed_run_connector_core_corridor_bridge_purity",
    "p5_compact_answer_preserve": "seed_run_connector_core_corridor_compact_answer_preserve",
    "seed_run_connector_core_corridor_compact_answer_preserve": "seed_run_connector_core_corridor_compact_answer_preserve",
    # Guarded answer-preserve refinement round
    "p3_base": "p3_answer_preserve_base",
    "p3_answer_preserve_base": "p3_answer_preserve_base",
    "p3_guarded_hotpot": "p3_answer_preserve_guarded_hotpot",
    "p3_answer_preserve_guarded_hotpot": "p3_answer_preserve_guarded_hotpot",
    "p3_confidence_gated": "p3_answer_preserve_confidence_gated",
    "p3_answer_preserve_confidence_gated": "p3_answer_preserve_confidence_gated",
}

ALL_OBJECTIVE_MODES = set(CORE_OBJECTIVE_MODES) | set(LEGACY_OBJECTIVE_MODES) | set(CANONICAL_OBJECTIVE_MODES)


def normalize_objective_mode(mode: str | None) -> str:
    raw = str(mode or "baseline").strip().lower()
    resolved = OBJECTIVE_MODE_ALIASES.get(raw, raw)
    if resolved not in ALL_OBJECTIVE_MODES:
        return "baseline"
    return resolved


def is_legacy_objective_mode(mode: str | None) -> bool:
    return normalize_objective_mode(mode) in LEGACY_OBJECTIVE_MODES


def apply_legacy_objective_flag_overrides(mode: str, flags: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(flags or {})

    if mode in {"r2_bridge_only", "r2_connector_core"}:
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
            }
        )
    elif mode in {"r2_plus_r3_anchor_light", "r2_plus_r3_answer_light"}:
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": True,
                "coverage_selection": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
            }
        )
    elif mode == "r2_plus_path_preserve":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": False,
                "corridor_answer_preserve_shaping": False,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": True,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
                "corridor_answer_preserve_guarded_hotpot": False,
                "corridor_answer_preserve_confidence_gated": False,
            }
        )
    elif mode == "r2_plus_path_preserve_compact":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": True,
                "corridor_answer_preserve_shaping": False,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": True,
                "corridor_path_preserve_compact_shaping": True,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
                "corridor_answer_preserve_guarded_hotpot": False,
                "corridor_answer_preserve_confidence_gated": False,
            }
        )
    elif mode == "r2_plus_path_preserve_guarded":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": False,
                "corridor_answer_preserve_shaping": False,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": True,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": True,
                "corridor_path_preserve_compact_lite_shaping": False,
                "corridor_answer_preserve_guarded_hotpot": False,
                "corridor_answer_preserve_confidence_gated": False,
            }
        )
    elif mode == "r2_plus_path_preserve_compact_lite":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": True,
                "corridor_answer_preserve_shaping": False,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": True,
                "corridor_path_preserve_compact_shaping": True,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": True,
                "corridor_answer_preserve_guarded_hotpot": False,
                "corridor_answer_preserve_confidence_gated": False,
            }
        )
    elif mode == "corridor_role_constrained":
        out.update({"coverage_selection": True})
    elif mode == "bridge_aware_run_plus_role_constrained_corridor":
        out.update({"coverage_selection": True})
    elif mode == "seed_run_connector_core":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": False,
                "corridor_answer_preserve_shaping": False,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
            }
        )
    elif mode == "seed_run_connector_core_corridor_compact":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": True,
                "corridor_answer_preserve_shaping": False,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
            }
        )
    elif mode == "seed_run_connector_core_corridor_answer_preserve":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": False,
                "corridor_answer_preserve_shaping": True,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
            }
        )
    elif mode == "seed_run_connector_core_corridor_bridge_purity":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": False,
                "corridor_answer_preserve_shaping": False,
                "corridor_bridge_purity_shaping": True,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
            }
        )
    elif mode == "seed_run_connector_core_corridor_compact_answer_preserve":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": True,
                "corridor_answer_preserve_shaping": True,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
                "corridor_answer_preserve_guarded_hotpot": False,
                "corridor_answer_preserve_confidence_gated": False,
            }
        )
    elif mode == "p3_answer_preserve_base":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": False,
                "corridor_answer_preserve_shaping": True,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
                "corridor_answer_preserve_guarded_hotpot": False,
                "corridor_answer_preserve_confidence_gated": False,
            }
        )
    elif mode == "p3_answer_preserve_guarded_hotpot":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": False,
                "corridor_answer_preserve_shaping": True,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
                "corridor_answer_preserve_guarded_hotpot": True,
                "corridor_answer_preserve_confidence_gated": False,
            }
        )
    elif mode == "p3_answer_preserve_confidence_gated":
        out.update(
            {
                "bridge_candidate_induction": True,
                "role_aware_chunk_scoring": False,
                "coverage_selection": False,
                "corridor_compact_shaping": False,
                "corridor_answer_preserve_shaping": True,
                "corridor_bridge_purity_shaping": False,
                "corridor_path_preserve_shaping": False,
                "corridor_path_preserve_compact_shaping": False,
                "corridor_path_preserve_guarded_shaping": False,
                "corridor_path_preserve_compact_lite_shaping": False,
                "corridor_answer_preserve_guarded_hotpot": False,
                "corridor_answer_preserve_confidence_gated": True,
            }
        )

    return out


def apply_legacy_objective_profile(
    cfg,
    mode: str,
    *,
    dataset: str | None = None,
    objective_flags: Mapping[str, Any],
    apply_fn: Callable[[Any, Mapping[str, Any]], tuple[Dict[str, Any], Dict[str, Any]]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Legacy boundary wrapper.

    The concrete score-shaping implementation remains in retrieval for now,
    but all legacy-mode routing goes through this module.
    """
    resolved = normalize_objective_mode(mode)
    if resolved not in LEGACY_OBJECTIVE_MODES:
        raise ValueError(f"Mode is not legacy: {mode}")
    flags = dict(objective_flags or {})
    flags["mode"] = resolved
    return apply_fn(cfg=cfg, objective_flags=flags)
