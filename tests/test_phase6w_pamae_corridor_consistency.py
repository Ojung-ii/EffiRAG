from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

from effirag.unified_acr_rcedr_selector import (
    UnifiedEvidenceAtom,
    _build_atom_diag_row,
    apply_unified_acr_rcedr_selection,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_phase6w_pamae_corridor_consistency as phase6w_pamae  # noqa: E402


def _cfg():
    return SimpleNamespace(
        unified_acr_rcedr_enabled=True,
        unified_acr_rcedr_mode="greedy",
        unified_acr_rcedr_beam_size=3,
        unified_acr_rcedr_max_atoms=4,
        unified_acr_rcedr_max_tokens=240,
        unified_acr_rcedr_hard_token_budget_enabled=True,
        unified_acr_rcedr_use_answerability_gain=True,
        unified_acr_rcedr_use_bridge_gain=True,
        unified_acr_rcedr_use_redundancy_penalty=True,
        unified_acr_rcedr_use_cost_penalty=False,
        unified_acr_rcedr_lambda_bridge=0.28,
        unified_acr_rcedr_mu_redundancy=0.22,
        unified_acr_rcedr_answerability_weight=1.0,
        unified_acr_rcedr_atomization_enabled=True,
        unified_acr_rcedr_atom_span_max_sentences=2,
        unified_acr_rcedr_length_penalty_weight=0.04,
        unified_acr_rcedr_chain_aware_enabled=False,
        unified_acr_rcedr_role_aware_redundancy_enabled=False,
        unified_acr_rcedr_role_balanced_enabled=False,
        unified_acr_rcedr_redundancy_recalibrated_enabled=False,
        unified_acr_rcedr_semantic_sufficiency_enabled=False,
        unified_acr_rcedr_semantic_answerability_weight=0.20,
        unified_acr_rcedr_semantic_use_as_prior_only=True,
        unified_acr_rcedr_chain_gain_weight=0.15,
        unified_acr_rcedr_role_balance_weight=0.08,
        unified_acr_rcedr_role_balance_max_gain_per_step=0.08,
        unified_acr_rcedr_role_balance_max_token_jaccard=0.45,
        unified_acr_rcedr_role_balance_missing_only=True,
        unified_acr_rcedr_role_redundancy_relax=0.75,
        unified_acr_rcedr_role_redundancy_max_overlap=0.45,
    )


def test_corridor_id_is_deterministic():
    cid1 = phase6w_pamae._stable_corridor_id("hotpotqa", "s1", "A::0", "B::1", 3)
    cid2 = phase6w_pamae._stable_corridor_id("hotpotqa", "s1", "A::0", "B::1", 3)
    cid3 = phase6w_pamae._stable_corridor_id("hotpotqa", "s2", "A::0", "B::1", 3)
    assert cid1 == cid2
    assert cid1 != cid3
    assert cid1.startswith("c_")


def test_corridor_id_mapping_helpers_handle_missing_metadata():
    assert phase6w_pamae._corridor_ids_from_entry({}) == []
    assert phase6w_pamae._best_corridor_id({}) == ""
    score = phase6w_pamae._path_completeness_score([])
    assert score["anchor_side_hit"] == 0
    assert score["seed_side_hit"] == 0
    assert score["bridge_node_hit"] == 0
    assert score["path_completeness_score"] == 0.0


def test_sentence_candidate_corridor_ids_are_preserved():
    entry = {"origin_corridor_ids": ["c2", "c1"], "best_corridor_id": "c0"}
    ids = phase6w_pamae._corridor_ids_from_entry(entry)
    assert ids == ["c0", "c2", "c1"]
    assert phase6w_pamae._best_corridor_id(entry) == "c0"


def test_atom_diag_row_preserves_origin_corridor_ids():
    atom = UnifiedEvidenceAtom(
        sentence_id="Doc::1",
        source_id="Doc",
        text="Bridge clue sentence.",
        token_count=4,
        token_set={"bridge", "clue", "sentence"},
        query_overlap=0.5,
        entity_overlap=0.3,
        relation_overlap=0.1,
        answer_type_compat=0.2,
        bridge_overlap=0.7,
        structure_anchor=0.0,
        structure_bridge=0.6,
        locality_score=0.4,
        corridor_ids=("c1", "c2"),
        bridge_gain=0.8,
        atom_base_score=1.2,
    )
    row = _build_atom_diag_row(atom, selected_rank=1, abr_delta_score=0.42)
    assert row["origin_corridor_ids"] == ["c1", "c2"]
    assert row["best_corridor_id"] == "c1"
    assert row["selected_rank"] == 1
    assert row["abr_delta_score"] == 0.42


def test_selector_outputs_unchanged_with_phase6w_diag_tables_present():
    question = "Which region borders country X?"
    ids = ["A::0", "B::0", "C::0"]
    texts = [
        "Alpha city is in country X.",
        "Country X borders Beta region.",
        "Distractor sentence.",
    ]
    table = {
        "A::0": {"is_main_candidate": True, "corridor_ids": ["c1"]},
        "B::0": {"is_support_candidate": True, "is_connector_adjacent": True, "corridor_ids": ["c1"]},
        "C::0": {"corridor_ids": ["c2"]},
    }
    out1_ids, out1_texts, diag1 = apply_unified_acr_rcedr_selection(
        question_text=question,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_cfg(),
    )
    out2_ids, out2_texts, diag2 = apply_unified_acr_rcedr_selection(
        question_text=question,
        selected_sentence_ids=ids,
        selected_sentences=texts,
        sentence_feature_table=table,
        cfg=_cfg(),
    )
    assert out1_ids == out2_ids
    assert out1_texts == out2_texts
    assert isinstance(diag1.get("phase6w_atom_table", []), list)
    assert isinstance(diag2.get("phase6w_selected_atom_table", []), list)


def test_query_type_classifier_is_analysis_only_helper():
    assert phase6w_pamae._infer_query_type("Is Alpha city in country X?") == "yes_no"
    assert phase6w_pamae._infer_query_type("Which country is larger, A or B?") in {"comparison", "date_or_number"}
    src = inspect.getsource(phase6w_pamae._infer_query_type)
    assert "dataset" not in src.lower()
    assert "profile" not in src.lower()


def test_runner_is_gpu1_two_process_and_enables_pamae_audit():
    runner = (ROOT / "scripts" / "run_phase6w_pamae_corridor_consistency_2proc.sh").read_text(encoding="utf-8")
    assert "PROCESS_COUNT must be 2" in runner
    assert "This runner is GPU1-only" in runner
    assert "PHASE6W_PAMAE_AUDIT" in runner
    assert "group_a" in runner and "group_b" in runner
