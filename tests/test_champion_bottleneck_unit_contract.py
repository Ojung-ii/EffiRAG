from __future__ import annotations

from effirag.champion_bottleneck_audit import unit_contract_from_row


def test_unit_contract_sentence_level_case():
    row = {
        "rendered": {
            "sentences": [
                "Alpha is the capital of A.",
                "Beta is the capital of B.",
            ],
            "sentence_ids": ["DocA::0", "DocB::1"],
            "metadata": {"selected_unit_type": "sentence"},
        }
    }
    out = unit_contract_from_row(row)
    assert out["is_sentence_level"] is True
    assert out["chunk_like_rate"] < 0.5
    assert out["avg_tokens_per_item"] > 0.0


def test_unit_contract_chunk_like_case():
    row = {
        "rendered": {
            "sentences": [
                "This is a long paragraph with multiple facts. It adds another fact here. "
                "And a third sentence with extra details to emulate chunk-like rendering."
            ],
            "sentence_ids": ["chunk::DocX::3"],
            "metadata": {"selected_unit_type": "chunk"},
        }
    }
    out = unit_contract_from_row(row)
    assert out["is_chunk_like"] is True
    assert out["chunk_like_rate"] >= 0.5
    assert out["avg_tokens_per_item"] >= 20.0

