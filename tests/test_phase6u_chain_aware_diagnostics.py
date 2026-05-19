import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_phase6u_chain_aware_paired_diagnostic as diag  # noqa: E402


def test_missing_diagnostics_not_silently_zero():
    assert diag._fmt(None, 4) == "n/a"


def test_atom_to_sentence_mapping():
    assert diag.normalize_sentence_support_id("chunk::The Newcomers::1") == "The Newcomers::1"
    assert diag.normalize_sentence_support_id("sentence::Foo::2") == "Foo::2"
    assert diag.normalize_sentence_support_id("title::3") == "title::3"


def test_paired_query_ids():
    out = diag.pair_sample_ids(["a", "b", "c"], ["a", "b", "c"])
    assert out["identical_order"] is True
    assert out["intersection_count"] == 3

    out2 = diag.pair_sample_ids(["a", "b", "c"], ["b", "c", "d"])
    assert out2["identical_order"] is False
    assert out2["intersection_count"] == 2
    assert "a" in out2["baseline_only"]
    assert "d" in out2["variant_only"]


def test_latency_fields_present():
    row = {
        "efficiency": {
            "retrieval_latency_ms": 12.3,
            "total_latency_ms": 20.1,
            "render_ms": 0.5,
        },
        "latency_breakdown_ms": {
            "proposal_total_ms": 5.0,
            "phase1_run_scoring_ms": 1.1,
            "phase2_refine_ms": 0.8,
            "sentence_rerank_ms": 0.0,
            "unified_acr_rcedr_ms": 0.2,
            "render_ms": 0.5,
        },
        "retrieval": {
            "diagnostics": {
                "unified_acr_rcedr_diag": {
                    "chain_aware_enabled": True,
                    "objective_eval_calls": 7,
                    "num_atoms": 9,
                    "num_selected_atoms": 4,
                    "selected_tokens": 123,
                }
            }
        },
    }
    lat = diag.extract_latency_fields(row)
    for key in [
        "retrieval_ms",
        "proposal_total_ms",
        "phase1_run_scoring_ms",
        "phase2_refine_ms",
        "sentence_rerank_ms",
        "unified_acr_rcedr_ms",
        "render_ms",
        "total_ms",
        "objective_eval_calls",
        "num_atoms",
        "num_selected_atoms",
    ]:
        assert key in lat
