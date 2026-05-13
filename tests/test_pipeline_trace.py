from effirag.grouped_profiles import COPY_SPAN_INSTRUCTION_GROUPED_V1
from effirag.pipeline_trace import trace_active_pipeline


def test_trace_active_pipeline_returns_dictionary():
    out = trace_active_pipeline({}, dataset="hotpotqa", profile=COPY_SPAN_INSTRUCTION_GROUPED_V1)
    assert isinstance(out, dict)
    assert out["dataset"] == "hotpotqa"
    assert "active_flags" in out
    assert "active_nonzero_weights" in out


def test_trace_does_not_fail_on_absent_optional_fields():
    out = trace_active_pipeline({}, dataset="hotpotqa")
    assert isinstance(out, dict)
    assert "notes" in out
    assert any("absent_flag_fields=" in note for note in out["notes"])
    assert any("absent_weight_fields=" in note for note in out["notes"])


def test_trace_separates_zero_and_nonzero_weights():
    cfg = {
        "dataset": "musique",
        "retrieval_objective_mode": "baseline",
        "run_score_semantic_weight": 0.31,
        "run_score_pair_coverage_weight": 0.0,
    }
    out = trace_active_pipeline(cfg, dataset="musique")
    assert float(out["active_nonzero_weights"]["run_score_semantic_weight"]) == 0.31
    assert "run_score_pair_coverage_weight" in out["inactive_zero_weights"]
