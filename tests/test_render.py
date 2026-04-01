from effirag.render import render_context
from effirag.types import ContextDocument, RetrievalResult, Sample


def test_render_respects_max_context_sentences():
    sample = Sample(
        qid="r1",
        question="q",
        answer="a",
        contexts=[ContextDocument(title="Doc", sentences=["s0", "s1", "s2"])],
        supporting_facts=[],
    )

    retrieval = RetrievalResult(
        sample_id="r1",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc::0", "Doc::1", "Doc::2"],
        selected_sentences=["s0", "s1", "s2"],
    )

    rendered = render_context(sample, retrieval, max_context_sentences=2)
    assert len(rendered.sentences) == 2
    assert rendered.truncated is True
    assert "Doc::0" in rendered.text


def test_render_corridor_mode_groups_and_tracks_truncation():
    sample = Sample(
        qid="r2",
        question="Who and when?",
        answer="a",
        contexts=[ContextDocument(title="Doc", sentences=["s0", "s1", "s2", "s3"])],
        supporting_facts=[],
    )

    retrieval = RetrievalResult(
        sample_id="r2",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc::0", "Doc::1", "Doc::2", "Doc::3"],
        selected_sentences=["s0", "s1", "s2", "s3"],
        corridors=[
            {
                "corridor_id": "c01",
                "corridor_score": 1.23,
                "anchors": ["a1", "a2"],
                "main_path_sentence_ids": ["Doc::0", "Doc::1", "Doc::2"],
                "support_sentence_ids": ["Doc::3"],
                "sentence_score_map": {"Doc::0": 1.0, "Doc::1": 0.8, "Doc::2": 0.5, "Doc::3": 0.2},
            }
        ],
    )

    rendered = render_context(
        sample,
        retrieval,
        max_context_sentences=10,
        render_mode="corridor",
        max_corridors_in_context=1,
        max_main_sentences_per_corridor=2,
        max_support_per_corridor=1,
        max_total_sentences=3,
    )

    assert rendered.render_mode == "corridor"
    assert rendered.rendered_corridor_ids == ["c01"]
    assert len(rendered.sentence_ids) == 3
    assert rendered.sentence_ids == ["Doc::0", "Doc::1", "Doc::3"]
    assert rendered.retrieval_selected_sentence_ids == ["Doc::0", "Doc::1", "Doc::2", "Doc::3"]
    assert rendered.truncated_corridor_count == 0
    assert rendered.truncated_sentence_count == 0
    assert "[Corridor 1 | score=1.230 | anchors=a1 ↔ a2]" in rendered.text
    assert "Support:" in rendered.text


def test_render_corridor_fallback_orders_by_local_score_when_main_missing():
    sample = Sample(
        qid="r3",
        question="q",
        answer="a",
        contexts=[ContextDocument(title="Doc", sentences=["s0", "s1", "s2"])],
        supporting_facts=[],
    )

    retrieval = RetrievalResult(
        sample_id="r3",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc::0", "Doc::1", "Doc::2"],
        selected_sentences=["s0", "s1", "s2"],
        corridors=[
            {
                "corridor_id": "c01",
                "corridor_score": 1.0,
                "anchors": ["a", "b"],
                "main_path_sentence_ids": [],
                "support_sentence_ids": ["Doc::0", "Doc::1", "Doc::2"],
                "sentence_score_map": {"Doc::0": 0.1, "Doc::1": 0.8, "Doc::2": 0.5},
            }
        ],
    )

    rendered = render_context(
        sample,
        retrieval,
        max_context_sentences=10,
        render_mode="corridor",
        max_corridors_in_context=1,
        max_main_sentences_per_corridor=2,
        max_support_per_corridor=0,
        max_total_sentences=2,
    )

    assert rendered.sentence_ids == ["Doc::1", "Doc::2"]


def test_render_corridor_overflow_drops_low_ranked_corridor_before_sentence_trim():
    sample = Sample(
        qid="r4",
        question="q",
        answer="a",
        contexts=[ContextDocument(title="Doc", sentences=["s0", "s1", "s2", "s3", "s4", "s5"])],
        supporting_facts=[],
    )

    retrieval = RetrievalResult(
        sample_id="r4",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Doc::0", "Doc::1", "Doc::2", "Doc::3", "Doc::4", "Doc::5"],
        selected_sentences=["s0", "s1", "s2", "s3", "s4", "s5"],
        corridors=[
            {
                "corridor_id": "c01",
                "corridor_score": 2.0,
                "anchors": ["a", "b"],
                "main_path_sentence_ids": ["Doc::0", "Doc::1"],
                "support_sentence_ids": ["Doc::2"],
                "sentence_score_map": {"Doc::0": 1.0, "Doc::1": 0.9, "Doc::2": 0.8},
            },
            {
                "corridor_id": "c02",
                "corridor_score": 1.0,
                "anchors": ["c", "d"],
                "main_path_sentence_ids": ["Doc::3", "Doc::4"],
                "support_sentence_ids": ["Doc::5"],
                "sentence_score_map": {"Doc::3": 0.7, "Doc::4": 0.6, "Doc::5": 0.5},
            },
        ],
    )

    rendered = render_context(
        sample,
        retrieval,
        max_context_sentences=12,
        render_mode="corridor",
        max_corridors_in_context=2,
        max_main_sentences_per_corridor=2,
        max_support_per_corridor=1,
        max_total_sentences=3,
    )

    assert rendered.rendered_corridor_ids == ["c01"]
    assert rendered.sentence_ids == ["Doc::0", "Doc::1", "Doc::2"]
    assert rendered.truncated_corridor_count >= 1
    assert rendered.truncated_sentence_count == 0


def test_render_entity_chunk_package_score_adds_chunk_excerpts():
    sample = Sample(
        qid="r5",
        question="Who discovered penicillin?",
        answer="Alexander Fleming",
        contexts=[
            ContextDocument(
                title="Penicillin",
                sentences=[
                    "Penicillin is an antibiotic.",
                    "Alexander Fleming discovered penicillin in 1928.",
                    "It changed medicine.",
                ],
            )
        ],
        supporting_facts=[],
    )

    chunk_text = " ".join(sample.contexts[0].sentences)
    retrieval = RetrievalResult(
        sample_id="r5",
        method="effirag",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["chunk::Penicillin::0"],
        selected_sentences=[chunk_text],
        diagnostics={"graph_mode": "entity_chunk_graph", "selected_text_map": {"chunk::Penicillin::0": chunk_text}},
    )

    rendered = render_context(
        sample,
        retrieval,
        max_context_sentences=6,
        render_mode="corridor_aware_flat",
        chunk_grounding_enabled=True,
        chunk_grounding_mode="package_score",
        chunk_grounding_top_k_packages=2,
        max_excerpt_sentences_per_package=2,
        chunk_excerpt_max_total_sentences=4,
    )

    assert rendered.metadata.get("chunk_grounding_enabled") is True
    assert rendered.metadata.get("evidence_package_count", 0) >= 1
    assert rendered.metadata.get("chunk_excerpt_sentence_count", 0) >= 1
    assert "[Grounded Chunk Excerpts]" in rendered.text
    assert any("#e" in sid for sid in rendered.sentence_ids)
