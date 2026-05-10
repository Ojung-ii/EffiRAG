import math

from effirag.canonical_objective import (
    CANONICAL_COPY_SPAN_MODE,
    CANONICAL_NO_DATASET_GUARD_MODE,
    apply_canonical_copy_span_objective,
    canonical_effective_reference_mode,
)
from effirag.canonical_scoring import CANONICAL_PRUNED_WEIGHT_FIELDS
from effirag.config import RagConfig, dataclass_from_dict
from effirag.grouped_profiles import COPY_SPAN_INSTRUCTION_GROUPED_V1, apply_grouped_profile
from effirag.retrieval import _apply_connector_objective_profile, _resolve_retrieval_objective_flags


KEY_FLAGS = [
    "bridge_candidate_induction",
    "role_aware_chunk_scoring",
    "coverage_selection",
    "corridor_compact_shaping",
    "corridor_answer_preserve_shaping",
    "corridor_bridge_purity_shaping",
    "corridor_path_preserve_shaping",
    "corridor_path_preserve_compact_shaping",
    "corridor_path_preserve_guarded_shaping",
    "corridor_path_preserve_compact_lite_shaping",
    "corridor_answer_preserve_guarded_hotpot",
    "corridor_answer_preserve_confidence_gated",
]


def _resolved_flags(cfg_dict):
    cfg = dataclass_from_dict(RagConfig, cfg_dict, warn_unknown_keys=False)
    flags = _resolve_retrieval_objective_flags(cfg)
    flags, diag = _apply_connector_objective_profile(cfg=cfg, objective_flags=flags)
    return cfg, flags, diag


def test_canonical_apply_does_not_modify_unrelated_config_fields():
    cfg = RagConfig()
    cfg.semantic_topn_entity = 77
    before = int(cfg.semantic_topn_entity)
    diag = apply_canonical_copy_span_objective(cfg, dataset="hotpotqa", mode=CANONICAL_COPY_SPAN_MODE)

    assert cfg.semantic_topn_entity == before
    assert cfg.bridge_candidate_induction_enabled is True
    assert cfg.corridor_answer_preserve_enabled is True
    assert cfg.corridor_answer_preserve_guarded_hotpot_enabled is True
    assert diag["effective_reference_mode"] == "p3_answer_preserve_guarded_hotpot"


def test_canonical_no_dataset_guard_changes_effective_hotpot_mode_only():
    assert canonical_effective_reference_mode("hotpotqa", mode=CANONICAL_COPY_SPAN_MODE) == "p3_answer_preserve_guarded_hotpot"
    assert canonical_effective_reference_mode("hotpotqa", mode=CANONICAL_NO_DATASET_GUARD_MODE) == "p3_answer_preserve_base"
    assert canonical_effective_reference_mode("2wikimultihopqa", mode=CANONICAL_NO_DATASET_GUARD_MODE) == "baseline"


def test_grouped_v1_and_canonical_copy_span_have_same_key_objective_flags():
    for dataset in ("hotpotqa", "2wikimultihopqa"):
        grouped_cfg = apply_grouped_profile({}, dataset, COPY_SPAN_INSTRUCTION_GROUPED_V1)
        grouped_cfg["dataset"] = dataset
        _, grouped_flags, _ = _resolved_flags(grouped_cfg)

        canonical_cfg = dict(grouped_cfg)
        canonical_cfg["dataset"] = dataset
        canonical_cfg["retrieval_objective_mode"] = CANONICAL_COPY_SPAN_MODE
        _, canonical_flags, _ = _resolved_flags(canonical_cfg)

        for key in KEY_FLAGS:
            assert bool(canonical_flags.get(key)) == bool(grouped_flags.get(key)), (
                f"flag mismatch dataset={dataset} key={key}: "
                f"canonical={canonical_flags.get(key)} grouped={grouped_flags.get(key)}"
            )


def test_legacy_objective_resolution_still_works_for_old_alias():
    cfg = RagConfig()
    cfg.retrieval_objective_mode = "p2_bridge"
    flags = _resolve_retrieval_objective_flags(cfg)
    assert flags["mode"] == "run_objective_bridge_aware"

    flags, diag = _apply_connector_objective_profile(cfg=cfg, objective_flags=flags)
    assert diag["profile_applied"] == "run_bridge_aware"
    assert math.isclose(float(cfg.run_score_bridge_weight), 0.22, rel_tol=0.0, abs_tol=1e-12)


def test_canonical_copy_span_enforces_pruned_weights_to_zero():
    cfg = RagConfig()
    cfg.retrieval_objective_mode = CANONICAL_COPY_SPAN_MODE
    for idx, field in enumerate(CANONICAL_PRUNED_WEIGHT_FIELDS, start=1):
        if hasattr(cfg, field):
            setattr(cfg, field, float(idx))

    diag = apply_canonical_copy_span_objective(cfg, dataset="hotpotqa", mode=CANONICAL_COPY_SPAN_MODE)

    for field in CANONICAL_PRUNED_WEIGHT_FIELDS:
        if hasattr(cfg, field):
            assert math.isclose(float(getattr(cfg, field)), 0.0, rel_tol=0.0, abs_tol=1e-12)
    overrides = dict(diag.get("pruned_weight_overrides", {}) or {})
    assert isinstance(overrides, dict)
    assert len(overrides) > 0


def test_noncanonical_baseline_mode_does_not_force_pruned_zero():
    cfg = RagConfig()
    cfg.retrieval_objective_mode = "baseline"
    if hasattr(cfg, "zeta_query"):
        cfg.zeta_query = 0.77
    if hasattr(cfg, "run_score_pair_coverage_weight"):
        cfg.run_score_pair_coverage_weight = 0.66

    _ = _resolve_retrieval_objective_flags(cfg)

    # baseline path should not trigger canonical invariant enforcement.
    assert math.isclose(float(cfg.zeta_query), 0.77, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(cfg.run_score_pair_coverage_weight), 0.66, rel_tol=0.0, abs_tol=1e-12)
