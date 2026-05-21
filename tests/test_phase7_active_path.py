import ast
import importlib
import sys
from pathlib import Path

from effirag.render import render_phase7_flat_context
from effirag.types import ContextDocument, RetrievalResult, Sample


FORBIDDEN = {
    "candidate_recall_boost",
    "dynamic_compact_selector",
    "gl_rcedr",
    "unified_acr_rcedr_selector",
    "legacy_sota_policy",
}


def test_phase7_module_does_not_import_forbidden_modules():
    path = Path("effirag/phase7_evidence_flow.py")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(str(alias.name or "").split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            found.add(str(node.module or "").split(".")[-1])
    assert FORBIDDEN.isdisjoint(found)


def test_register_defaults_phase7_mode_does_not_import_forbidden():
    for name in list(FORBIDDEN):
        sys.modules.pop(f"effirag.{name}", None)
    reg = importlib.import_module("effirag.registry")
    before = set(sys.modules.keys())
    reg.register_defaults("phase7_evidence_flow")
    after = set(sys.modules.keys())
    loaded = set(after - before)
    for name in FORBIDDEN:
        assert f"effirag.{name}" not in loaded
    assert "phase7_evidence_flow" in reg.METHOD_REGISTRY


def test_phase7_flat_render_format():
    sample = Sample(
        qid="q1",
        question="Who wrote Hamlet?",
        answer="Shakespeare",
        contexts=[
            ContextDocument(
                title="Hamlet",
                sentences=["Hamlet is a tragedy.", "It was written by William Shakespeare."],
            )
        ],
        supporting_facts=[],
    )
    retrieval = RetrievalResult(
        sample_id="q1",
        method="phase7_evidence_flow",
        anchors=[],
        seeds=[],
        selected_nodes=[],
        selected_sentence_ids=["Hamlet::0", "Hamlet::1"],
        selected_sentences=["Hamlet is a tragedy.", "It was written by William Shakespeare."],
    )
    rendered = render_phase7_flat_context(sample, retrieval, max_context_sentences=10)
    lines = [ln for ln in rendered.text.splitlines() if ln.strip()]
    assert lines[0].startswith("[Hamlet] ")
    assert lines[1].startswith("[Hamlet] ")
    assert rendered.render_mode == "phase7_flat"

