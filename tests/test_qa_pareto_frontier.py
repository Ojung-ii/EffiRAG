from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_qa_aware_bottlenecks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_qa_aware_bottlenecks", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pareto_and_dominated_and_reference_status():
    mod = _load_module()
    rows = [
        {
            "dataset": "hotpotqa",
            "profile": "unified_large",
            "qa_available": True,
            "retrieval_available": True,
            "F1": 0.45,
            "prompt_tokens_avg": 620.0,
            "rendered_tokens": 210.0,
            "rendered_sf_R": 0.55,
            "rendered_sf_F1_per_1k_tokens": 1.2,
            "F1_per_1k_prompt_tokens": 0.72,
        },
        {
            "dataset": "hotpotqa",
            "profile": "candidate_a",
            "qa_available": True,
            "retrieval_available": True,
            "F1": 0.42,
            "prompt_tokens_avg": 520.0,
            "rendered_tokens": 190.0,
            "rendered_sf_R": 0.50,
            "rendered_sf_F1_per_1k_tokens": 1.3,
            "F1_per_1k_prompt_tokens": 0.81,
        },
        {
            "dataset": "hotpotqa",
            "profile": "candidate_b",
            "qa_available": True,
            "retrieval_available": True,
            "F1": 0.36,
            "prompt_tokens_avg": 680.0,
            "rendered_tokens": 280.0,
            "rendered_sf_R": 0.40,
            "rendered_sf_F1_per_1k_tokens": 0.8,
            "F1_per_1k_prompt_tokens": 0.52,
        },
    ]

    frontier_rows = mod.compute_frontier_status_rows(
        merged_rows=rows,
        datasets=["hotpotqa"],
        profiles=["unified_large", "candidate_a", "candidate_b"],
        reference_profiles=("unified_large",),
    )
    index = {
        (r["profile"], r["frontier_name"]): r["pareto_status"]
        for r in frontier_rows
    }
    assert index[("unified_large", "qa_f1_vs_prompt_tokens")] == "reference_only"
    assert index[("candidate_a", "qa_f1_vs_prompt_tokens")] == "pareto"
    assert index[("candidate_b", "qa_f1_vs_prompt_tokens")] == "dominated"


def test_incomplete_metrics_handled_gracefully():
    mod = _load_module()
    rows = [
        {
            "dataset": "2wikimultihopqa",
            "profile": "candidate_missing",
            "qa_available": False,
            "retrieval_available": True,
            "F1": 0.0,
            "prompt_tokens_avg": 0.0,
            "rendered_tokens": 150.0,
            "rendered_sf_R": 0.45,
            "rendered_sf_F1_per_1k_tokens": 1.0,
            "F1_per_1k_prompt_tokens": 0.0,
        }
    ]
    frontier_rows = mod.compute_frontier_status_rows(
        merged_rows=rows,
        datasets=["2wikimultihopqa"],
        profiles=["candidate_missing"],
    )
    target = [r for r in frontier_rows if r["frontier_name"] == "qa_f1_vs_prompt_tokens"][0]
    assert target["pareto_status"] == "incomplete_metrics"
