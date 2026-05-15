from __future__ import annotations

from effirag.champion_bottleneck_audit import token_decomposition_from_row


def test_token_decomposition_splits_components():
    row = {
        "rendered": {
            "sentences": [
                "Alpha evidence sentence about entity A.",
                "Beta evidence sentence about entity B.",
            ],
            "sentence_ids": [
                "DocA::0",
                "DocB::1",
            ],
            "metadata": {"separator_line_count": 2},
        },
        "generation_diagnostics": {"prompt_tokens": 120},
    }
    out = token_decomposition_from_row(row)
    assert out["prompt_tokens"] == 120
    assert out["evidence_tokens"] > 0
    assert out["metadata_tokens"] > 0
    assert out["separator_tokens"] >= 2
    assert out["instruction_tokens"] >= 0


def test_token_decomposition_warns_on_large_gap():
    row = {
        "rendered": {
            "sentences": ["short evidence"],
            "sentence_ids": ["DocA::0"],
            "metadata": {"separator_line_count": 0},
        },
        "generation_diagnostics": {"prompt_tokens": 2000},
    }
    out = token_decomposition_from_row(row)
    assert out["decomposition_warning"] is True
    assert abs(int(out["decomposition_gap_tokens"])) > 64


def test_token_decomposition_fallback_on_missing_fields():
    out = token_decomposition_from_row({})
    assert out["prompt_tokens"] == 0
    assert out["evidence_tokens"] == 0
    assert out["metadata_tokens"] == 0
    assert out["instruction_tokens"] == 0
    assert out["decomposition_warning"] is False

