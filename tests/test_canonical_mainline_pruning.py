import math

from effirag.canonical_objective import (
    CANONICAL_COPY_SPAN_MODE,
    CANONICAL_NO_DATASET_GUARD_MODE,
    apply_canonical_copy_span_objective,
)
from effirag.config import RagConfig, dataclass_from_dict
from effirag.grouped_profiles import COPY_SPAN_INSTRUCTION_GROUPED_V1, apply_grouped_profile
from effirag.pipeline_trace import trace_active_pipeline
from effirag.retrieval import _apply_connector_objective_profile, _resolve_retrieval_objective_flags


def test_dataset_guard_is_preserved_for_canonical_copy_span():
    cfg = RagConfig()
    apply_canonical_copy_span_objective(cfg, dataset="hotpotqa", mode=CANONICAL_COPY_SPAN_MODE)
    assert cfg.corridor_answer_preserve_enabled is True
    assert cfg.corridor_answer_preserve_guarded_hotpot_enabled is True


def test_canonical_no_dataset_guard_is_not_merged_into_canonical_copy_span():
    cfg_keep_guard = RagConfig()
    apply_canonical_copy_span_objective(cfg_keep_guard, dataset="hotpotqa", mode=CANONICAL_COPY_SPAN_MODE)
    assert cfg_keep_guard.corridor_answer_preserve_guarded_hotpot_enabled is True

    cfg_drop_guard = RagConfig()
    apply_canonical_copy_span_objective(cfg_drop_guard, dataset="hotpotqa", mode=CANONICAL_NO_DATASET_GUARD_MODE)
    assert cfg_drop_guard.corridor_answer_preserve_guarded_hotpot_enabled is False


def test_trace_reports_pruned_weights_as_zero_not_active_canonical():
    cfg = apply_grouped_profile({}, dataset="hotpotqa", profile_name=COPY_SPAN_INSTRUCTION_GROUPED_V1)
    cfg["dataset"] = "hotpotqa"
    cfg["retrieval_objective_mode"] = CANONICAL_COPY_SPAN_MODE
    cfg["seed_score_bridge_weight"] = 0.11
    cfg["run_score_pair_coverage_weight"] = 0.22
    cfg["zeta_query"] = 0.33

    out = trace_active_pipeline(cfg, dataset="hotpotqa", profile=CANONICAL_COPY_SPAN_MODE)
    assert out.get("canonical_mode") is True

    active = out.get("active_nonzero_weights", {})
    assert "seed_score_bridge_weight" not in active
    assert "run_score_pair_coverage_weight" not in active
    assert "zeta_query" not in active

    legacy_like = out.get("canonical_legacy_or_pruned_nonzero_weights", {}) or {}
    assert "seed_score_bridge_weight" not in legacy_like
    assert "run_score_pair_coverage_weight" not in legacy_like
    assert "zeta_query" not in legacy_like

    zeros = out.get("inactive_zero_weights", {})
    assert math.isclose(float(zeros.get("seed_score_bridge_weight", 1.0)), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(zeros.get("run_score_pair_coverage_weight", 1.0)), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(zeros.get("zeta_query", 1.0)), 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_trace_canonical_active_core_weights_include_run_core_terms():
    cfg = apply_grouped_profile({}, dataset="2wikimultihopqa", profile_name=COPY_SPAN_INSTRUCTION_GROUPED_V1)
    cfg["dataset"] = "2wikimultihopqa"
    cfg["retrieval_objective_mode"] = CANONICAL_COPY_SPAN_MODE
    out = trace_active_pipeline(cfg, dataset="2wikimultihopqa", profile=CANONICAL_COPY_SPAN_MODE)
    core = out.get("canonical_active_core_weights", {})
    assert "run_score_semantic_weight" in core
    assert "run_score_anchor_weight" in core
    assert "run_score_structure_weight" in core
    assert "run_score_bridge_weight" in core
    assert "run_score_redundancy_weight" in core


def test_legacy_objective_resolution_still_works_through_legacy_boundary():
    cfg = RagConfig()
    cfg.retrieval_objective_mode = "p2_bridge"
    flags = _resolve_retrieval_objective_flags(cfg)
    assert flags["mode"] == "run_objective_bridge_aware"
    flags, diag = _apply_connector_objective_profile(cfg=cfg, objective_flags=flags)
    assert diag["profile_applied"] == "run_bridge_aware"
    assert math.isclose(float(cfg.run_score_bridge_weight), 0.22, rel_tol=0.0, abs_tol=1e-12)


def test_canonical_apply_does_not_modify_unrelated_fields():
    cfg_dict = {
        "dataset": "hotpotqa",
        "semantic_topn_entity": 77,
        "retrieval_objective_mode": CANONICAL_COPY_SPAN_MODE,
    }
    cfg = dataclass_from_dict(RagConfig, cfg_dict)
    before = int(cfg.semantic_topn_entity)
    apply_canonical_copy_span_objective(cfg, dataset="hotpotqa", mode=CANONICAL_COPY_SPAN_MODE)
    assert cfg.semantic_topn_entity == before
