from effirag.phase8_diagnostics import failure_label_repaired


def test_source_balanced_never_gets_phase8_uq_miss():
    query = {
        "phase7_diag": {
            "gold_supporting_facts": [{"title": "Gold", "sent_idx": 0}],
            "phase2_candidate_ids": [],
            "selected_evidence_ids": [],
            "rendered_evidence_ids": [],
        },
        "result": {"metrics": {"supporting_fact_recall": 0.0, "f1": 0.0}},
    }
    assert failure_label_repaired("source_balanced_128", query) == "PHASE1_NOT_FOUND"


def test_phase8_with_candidates_can_report_seed_to_evidence_miss():
    query = {
        "phase7": {
            "phase8_diagnostics": {
                "entity_universe": {"num_entities": 2000},
                "best_seed_selection": {"seed_gold_hit_eval_only": True},
                "evidence_proposal": {"num_final_evidence_candidates": 50},
            }
        },
        "phase7_diag": {
            "gold_supporting_facts": [{"title": "Gold", "sent_idx": 0}],
            "phase2_candidate_ids": ["Distractor::0"],
            "selected_evidence_ids": ["Distractor::0"],
            "rendered_evidence_ids": ["Distractor::0"],
        },
        "result": {"metrics": {"supporting_fact_recall": 0.0, "f1": 0.0}},
    }
    assert failure_label_repaired("pamae_seed_k5", query) == "SEED_TO_EVIDENCE_MISS"
