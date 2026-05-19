from scripts.analyze_phase6u_persistent_evidence_loss import (
    CANONICAL_STAGE_ORDER,
    TRANSIENT_CLASS,
    classify_support_presence,
)


def _presence(**overrides):
    base = {stage: False for stage in CANONICAL_STAGE_ORDER}
    base.update(overrides)
    return base


def test_transient_pair_shortlist_recovery_classification():
    presence = _presence(
        proposal_candidates=True,
        phase1_run_seed_shortlist=True,
        anchor_seed_pair_shortlist=False,
        local_corridor_candidates=True,
        sentence_candidates_before_text_rerank=True,
        sentence_candidates_after_text_rerank=True,
        evidence_atoms_before_abr=True,
        abr_selected_evidence=True,
        rendered_compact_context=True,
    )
    assert classify_support_presence(presence) == TRANSIENT_CLASS


def test_abr_persistent_loss_classification():
    presence = _presence(
        proposal_candidates=True,
        phase1_run_seed_shortlist=True,
        local_corridor_candidates=True,
        corridor_after_rerank_trim=True,
        sentence_candidates_before_text_rerank=True,
        sentence_candidates_after_text_rerank=True,
        evidence_atoms_before_abr=True,
        abr_selected_evidence=False,
        rendered_compact_context=False,
    )
    assert classify_support_presence(presence) == "abr_selection_persistent_loss"


def test_phase1_persistent_loss_classification():
    presence = _presence(
        proposal_candidates=True,
        phase1_run_seed_shortlist=False,
        local_corridor_candidates=False,
        sentence_candidates_before_text_rerank=False,
        sentence_candidates_after_text_rerank=False,
        abr_selected_evidence=False,
        rendered_compact_context=False,
    )
    assert classify_support_presence(presence) == "run_seed_shortlist_persistent_loss"


def test_rendering_persistent_loss_classification():
    presence = _presence(
        proposal_candidates=True,
        phase1_run_seed_shortlist=True,
        local_corridor_candidates=True,
        sentence_candidates_before_text_rerank=True,
        sentence_candidates_after_text_rerank=True,
        evidence_atoms_before_abr=True,
        abr_selected_evidence=True,
        rendered_compact_context=False,
    )
    assert classify_support_presence(presence) == "rendering_persistent_loss"
