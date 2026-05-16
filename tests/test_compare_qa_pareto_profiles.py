from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_qa_pareto_profiles.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_qa_pareto_profiles", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pareto_frontier_minimize_and_maximize_axes():
    mod = _load_module()
    rows = [
        {"profile": "a", "F1": 0.50, "prompt_tokens_avg": 800.0},
        {"profile": "b", "F1": 0.55, "prompt_tokens_avg": 780.0},
        {"profile": "c", "F1": 0.48, "prompt_tokens_avg": 760.0},
        {"profile": "d", "F1": 0.55, "prompt_tokens_avg": 900.0},
    ]

    frontier = mod._pareto_frontier(rows, maximize=["F1"], minimize=["prompt_tokens_avg"])
    profiles = {str(r.get("profile")) for r in frontier}

    assert "b" in profiles
    assert "c" in profiles
    assert "a" not in profiles
    assert "d" not in profiles


def test_find_retrieval_aggregate_csv_fallback_paths(tmp_path: Path):
    mod = _load_module()

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True)
    target = audit_dir / "aggregate_by_profile_dataset.csv"
    target.write_text("dataset,profile\n", encoding="utf-8")

    found = mod._find_retrieval_aggregate_csv(tmp_path)
    assert found == target
