from effirag.config import RagConfig, dataclass_from_dict
from effirag.retrieval import _apply_connector_objective_profile, _resolve_retrieval_objective_flags
from effirag.unified_copy_span_policy import apply_unified_profile, resolve_unified_copy_span_mode


def test_unified_resolver_ignores_hotpot_dataset_name():
    assert resolve_unified_copy_span_mode("hotpotqa") == "baseline"
    assert resolve_unified_copy_span_mode("2wikimultihopqa") == "baseline"


def test_hotpotqa_unified_profile_does_not_enable_hotpot_guard_at_runtime():
    raw_cfg = apply_unified_profile({}, "hotpotqa", "unified_medium")
    cfg = dataclass_from_dict(RagConfig, raw_cfg, warn_unknown_keys=False)

    flags = _resolve_retrieval_objective_flags(cfg)
    flags, diag = _apply_connector_objective_profile(cfg, flags)

    assert cfg.retrieval_objective_mode == "baseline"
    assert diag["effective_mode"] == "baseline"
    assert flags["corridor_answer_preserve_guarded_hotpot"] is False
    assert cfg.corridor_answer_preserve_guarded_hotpot_enabled is False
    assert cfg.corridor_answer_preserve_enabled is False
    assert cfg.answer_support_pinning_enabled is False
    assert cfg.final_top_slice_reorder_enabled is False
