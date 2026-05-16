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


def test_classify_over_compression_bottleneck():
    mod = _load_module()
    name, _ = mod.classify_dominant_bottleneck(
        {
            "F1": 0.20,
            "rendered_sf_R": 0.30,
            "rendered_sf_P": 0.10,
            "rendered_tokens": 150.0,
            "rendered_sf_F1_per_1k_tokens": 1.40,
            "prompt_tokens_avg": 700.0,
        }
    )
    assert name == "over_compression_bottleneck"


def test_classify_evidence_density_bottleneck():
    mod = _load_module()
    name, _ = mod.classify_dominant_bottleneck(
        {
            "F1": 0.40,
            "rendered_sf_R": 0.50,
            "rendered_sf_P": 0.10,
            "rendered_tokens": 300.0,
            "rendered_sf_F1_per_1k_tokens": 0.70,
            "prompt_tokens_avg": 950.0,
        }
    )
    assert name == "evidence_density_bottleneck"


def test_classify_prompt_usability_bottleneck():
    mod = _load_module()
    name, _ = mod.classify_dominant_bottleneck(
        {
            "F1": 0.30,
            "rendered_sf_R": 0.50,
            "rendered_sf_P": 0.14,
            "rendered_tokens": 220.0,
            "rendered_sf_F1_per_1k_tokens": 1.00,
            "prompt_tokens_avg": 780.0,
        }
    )
    assert name == "prompt_usability_bottleneck"


def test_classify_generation_bottleneck():
    mod = _load_module()
    name, _ = mod.classify_dominant_bottleneck(
        {
            "F1": 0.30,
            "rendered_sf_R": 0.62,
            "rendered_sf_P": 0.20,
            "rendered_tokens": 240.0,
            "rendered_sf_F1_per_1k_tokens": 1.00,
            "prompt_tokens_avg": 700.0,
        }
    )
    assert name == "generation_bottleneck"


def test_classify_evidence_recall_bottleneck():
    mod = _load_module()
    name, _ = mod.classify_dominant_bottleneck(
        {
            "F1": 0.24,
            "rendered_sf_R": 0.32,
            "rendered_sf_P": 0.11,
            "rendered_tokens": 260.0,
            "rendered_sf_F1_per_1k_tokens": 0.85,
            "prompt_tokens_avg": 820.0,
        }
    )
    assert name == "evidence_recall_bottleneck"
