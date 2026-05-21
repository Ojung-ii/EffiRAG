from types import SimpleNamespace

from effirag.generator import _build_qa_prompt
from effirag.types import RenderedContext, Sample


def _sample_rendered():
    sample = Sample(qid="q1", question="Who wrote Hamlet?", answer="William Shakespeare", contexts=[])
    rendered = RenderedContext(
        sample_id="q1",
        method="phase7_evidence_flow",
        text="[Hamlet] Hamlet was written by William Shakespeare.",
        sentences=["Hamlet was written by William Shakespeare."],
        sentence_ids=["Hamlet::0"],
        truncated=False,
        render_mode="phase7_flat",
    )
    return sample, rendered


def test_current_phase7_prompt_mode_keeps_legacy_shape():
    sample, rendered = _sample_rendered()
    cfg = SimpleNamespace(prompt_variant="default", qa_prompt_mode="current_phase7", user_prompt="")
    prompt = _build_qa_prompt(sample=sample, rendered=rendered, cfg=cfg)
    assert "Use only the provided context." in prompt
    assert "Final answer:" in prompt


def test_lightrag_short_prompt_mode():
    sample, rendered = _sample_rendered()
    cfg = SimpleNamespace(prompt_variant="default", qa_prompt_mode="lightrag_short", user_prompt="")
    prompt = _build_qa_prompt(sample=sample, rendered=rendered, cfg=cfg)
    assert "insufficient information" in prompt
    assert "Return only the final short answer span." in prompt
    assert "Answer:" in prompt


def test_phase7_short_prompt_mode():
    sample, rendered = _sample_rendered()
    cfg = SimpleNamespace(prompt_variant="default", qa_prompt_mode="phase7_short", user_prompt="")
    prompt = _build_qa_prompt(sample=sample, rendered=rendered, cfg=cfg)
    assert "---Role---" in prompt
    assert "---Context---" in prompt
    assert "Answer:" in prompt
