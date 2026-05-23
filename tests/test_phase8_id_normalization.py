from effirag.phase8_diagnostics import canonical_support_key, normalize_evidence_id, normalize_title, normalized_query_diagnostics


def test_title_sent_id_matches_gold_support_key():
    assert normalize_title("A &amp; B") == "a & b"
    assert normalize_evidence_id("A &amp; B::1") == canonical_support_key("A & B", 1)


def test_normalized_candidate_selected_rendered_recall():
    query = {
        "phase7_diag": {
            "gold_supporting_facts": [
                {"title": "A &amp; B", "sent_idx": 1},
                {"title": "Other", "sent_idx": 0},
            ],
            "phase2_candidate_ids": ["A & B::1", "Distractor::0"],
            "selected_evidence_ids": ["A &amp; B::1"],
            "rendered_evidence_ids": ["A & B::1"],
        },
        "phase7": {"candidate_chain_feasibility": {"chain_unit_oracle_feasible": True}},
    }
    diag = normalized_query_diagnostics(query)
    assert diag["normalized_candidate_gold_partial"] is True
    assert diag["normalized_candidate_gold_full"] is False
    assert diag["normalized_candidate_gold_recall"] == 0.5
    assert diag["normalized_selected_gold_recall"] == 0.5
    assert diag["normalized_rendered_gold_recall"] == 0.5
    assert diag["chain_unit_oracle_feasible"] is True
