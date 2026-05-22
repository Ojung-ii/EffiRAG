from effirag.phase7_evidence_flow import Phase7EvidenceAtom, _merge_phase8_augmented_atoms


def _atom(source_id: str, *, tag: str, semantic: float = 0.1, flow: float = 0.2) -> Phase7EvidenceAtom:
    return Phase7EvidenceAtom(
        atom_id=f"atom::{source_id}",
        source_id=source_id,
        title="Title",
        text=f"text for {source_id}",
        token_count=4,
        semantic_score=semantic,
        flow_score=flow,
        anchor_distance=0.0,
        anchor_reachability=0.0,
        bridge_entity_coverage=0.0,
        relation_term_coverage=0.0,
        source_key="semantic",
        source_tags={tag},
        proposal_source=tag,
    )


def test_phase8_augmentation_preserves_base_and_adds_unique_atoms() -> None:
    base_atoms = [_atom("s1", tag="phase7"), _atom("s2", tag="phase7")]
    phase8_atoms = [_atom("s3", tag="phase8_pamae_seed")]

    merged, diagnostics = _merge_phase8_augmented_atoms(base_atoms, phase8_atoms, max_candidates=4)

    assert [atom.source_id for atom in merged] == ["s1", "s2", "s3"]
    assert diagnostics["base_candidate_count"] == 2
    assert diagnostics["phase8_unique_added_count"] == 1
    assert diagnostics["phase8_duplicate_count"] == 0
    assert diagnostics["augmented_candidate_count"] == 3


def test_phase8_augmentation_dedupes_by_source_id_and_merges_tags() -> None:
    base_atoms = [_atom("s1", tag="phase7", semantic=0.1, flow=0.2)]
    phase8_atoms = [_atom("s1", tag="phase8_pamae_seed", semantic=0.9, flow=0.8)]

    merged, diagnostics = _merge_phase8_augmented_atoms(base_atoms, phase8_atoms, max_candidates=4)

    assert len(merged) == 1
    assert merged[0].source_tags == {"phase7", "phase8_pamae_seed"}
    assert merged[0].semantic_score == 0.9
    assert merged[0].flow_score == 0.8
    assert diagnostics["phase8_unique_added_count"] == 0
    assert diagnostics["phase8_duplicate_count"] == 1


def test_phase8_augmentation_respects_total_candidate_cap_after_base() -> None:
    base_atoms = [_atom("s1", tag="phase7"), _atom("s2", tag="phase7")]
    phase8_atoms = [_atom("s3", tag="phase8_pamae_seed"), _atom("s4", tag="phase8_pamae_seed")]

    merged, diagnostics = _merge_phase8_augmented_atoms(base_atoms, phase8_atoms, max_candidates=3)

    assert [atom.source_id for atom in merged] == ["s1", "s2", "s3"]
    assert diagnostics["phase8_candidate_seen"] == 2
    assert diagnostics["phase8_unique_added_count"] == 1
    assert diagnostics["augmented_candidate_count"] == 3
