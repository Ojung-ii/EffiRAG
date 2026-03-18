import sys
from types import SimpleNamespace

from effirag.generator import _HF_PIPELINE_CACHE, _HF_PIPELINE_ERRORS, _postprocess_prediction, generate_hf
from effirag.types import RenderedContext, Sample


def _sample_and_rendered():
    sample = Sample(
        qid="g1",
        question="What city is the capital of France?",
        answer="Paris",
        contexts=[],
    )
    rendered = RenderedContext(
        sample_id="g1",
        method="effirag",
        text="Paris is the capital of France.",
        sentences=["Paris is the capital of France."],
        sentence_ids=["Paris::0"],
        truncated=False,
    )
    return sample, rendered


def test_generate_hf_falls_back_to_text_generation(monkeypatch):
    _HF_PIPELINE_CACHE.clear()
    _HF_PIPELINE_ERRORS.clear()
    calls = []

    def fake_pipeline(task, **kwargs):
        calls.append(task)
        if task == "text2text-generation":
            raise RuntimeError("unsupported task")

        def _runner(prompt, max_new_tokens, do_sample, return_full_text):
            return [{"generated_text": "Paris"}]

        return _runner

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))

    sample, rendered = _sample_and_rendered()
    result = generate_hf(sample, rendered, model_name="Qwen/Qwen3.5-2B")

    assert calls[0] == "text2text-generation"
    assert calls[-1] == "text-generation"
    assert result.prediction == "Paris"
    assert result.raw_text == "Paris"


def test_postprocess_prediction_removes_think_block():
    raw = "<think>\ninternal reasoning\n</think>\n\nAnimorphs"
    pred = _postprocess_prediction(raw, question="What series is this?")
    assert pred == "Animorphs"


def test_postprocess_prediction_normalizes_yes_no():
    raw = "<think>...</think>\nNo, they are not the same."
    pred = _postprocess_prediction(raw, question="Were they of the same nationality?")
    assert pred == "no"


def test_generate_hf_falls_back_to_heuristic_when_model_unavailable(monkeypatch):
    _HF_PIPELINE_CACHE.clear()
    _HF_PIPELINE_ERRORS.clear()

    def fake_pipeline(task, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))

    sample, rendered = _sample_and_rendered()
    result = generate_hf(sample, rendered, model_name="Qwen/Qwen3.5-2B")

    assert result.prediction == "Paris is the capital of France."
    assert "HF generation failed" in result.raw_text
