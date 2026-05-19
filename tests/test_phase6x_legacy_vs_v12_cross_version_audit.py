from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_phase6x_legacy_vs_v12_cross_version_audit as phase6x  # noqa: E402


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_missing_legacy_stage_values_are_na_not_zero():
    stage = phase6x._stage_unavailable("S2_phase1_shortlist")
    assert stage["stage_available"] is False
    assert stage["SF_recall"] is None
    assert stage["candidate_count"] is None


def test_failure_stage_attribution_synthetic():
    stage_map = {
        "S1_proposal_candidates": {"stage_available": True, "SF_recall": 1.0},
        "S2_phase1_shortlist": {"stage_available": True, "SF_recall": 0.8},
        "S3_local_refinement_or_corridor": {"stage_available": True, "SF_recall": 0.8},
    }
    assert phase6x._failure_stage(stage_map) == "S2_phase1_shortlist"


def test_paired_case_classification_legacy_high_f1_low_sf():
    row = {
        "legacy_F1": 0.6,
        "v12_F1": 0.3,
        "legacy_rendered_SF_recall": 0.4,
        "v12_rendered_SF_recall": 0.5,
        "legacy_answer_string_hit": 1.0,
        "v12_answer_string_hit": 0.0,
    }
    assert phase6x._paired_case_classification(row, row) == "legacy_high_F1_low_SF"


def test_load_run_bundle_requires_final_outputs(tmp_path: Path):
    out_root = tmp_path / "out"
    run_root = out_root / "raw" / "legacy_hotpotqa"
    run_root.mkdir(parents=True, exist_ok=True)
    bundle, err = phase6x._load_run_bundle(
        out_root,
        {
            "run_name": "legacy_hotpotqa",
            "run_root": str(run_root),
            "dataset": "hotpotqa",
            "method_label": "legacy",
            "profile": "legacy",
            "config_path": "/tmp/fake.yaml",
        },
    )
    assert bundle is None
    assert "missing_output_files" in str(err)


def test_validation_detects_sample_id_mismatch(tmp_path: Path, monkeypatch):
    out_root = tmp_path / "phase6x"
    raw_legacy = out_root / "raw" / "legacy_hotpotqa" / "hotpotqa" / "ts1"
    raw_v12 = out_root / "raw" / "v12_hotpotqa" / "hotpotqa" / "ts1"
    raw_legacy.mkdir(parents=True, exist_ok=True)
    raw_v12.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": "hotpotqa",
        "em": 0.0,
        "f1": 0.0,
        "supporting_fact_precision": 0.0,
        "supporting_fact_recall": 0.0,
        "retrieval_ms": 1.0,
        "generation_ms": 1.0,
        "total_ms": 2.0,
    }
    _write_json(raw_legacy / "rag_summary.json", summary)
    _write_json(raw_v12 / "rag_summary.json", summary)

    legacy_row = {
        "sample_id": "s1",
        "question": "Q",
        "answer": "A",
        "prediction": "A",
        "qa_executed": True,
        "metrics": {"em": 1.0, "f1": 1.0, "supporting_fact_recall": 0.0, "supporting_fact_precision": 0.0},
        "efficiency": {"retrieval_latency_ms": 1.0, "generation_latency_ms": 1.0, "total_latency_ms": 2.0},
        "retrieval": {"selected_sentence_ids": ["Title::0"], "selected_sentences": ["A"], "diagnostics": {"graph_mode": "current_entity_graph"}},
        "rendered": {"sentence_ids": ["Title::0"], "sentences": ["A"]},
        "rendered_sentence_ids": ["Title::0"],
    }
    v12_row = dict(legacy_row)
    v12_row["sample_id"] = "s2"

    _write_jsonl(raw_legacy / "rag_query_results.jsonl", [legacy_row])
    _write_jsonl(raw_v12 / "rag_query_results.jsonl", [v12_row])

    manifest = {
        "phase": "phase6x_legacy_vs_v12_cross_version_audit",
        "legacy_commit": "56779c0",
        "latest_commit": "HEAD",
        "legacy_hotpotqa_config_path": "/tmp/legacy_hotpotqa.yaml",
        "legacy_2wiki_config_path": "/tmp/legacy_2wiki.yaml",
        "latest_hotpotqa_config_path": "/tmp/latest_hotpotqa.yaml",
        "latest_2wiki_config_path": "/tmp/latest_2wiki.yaml",
        "runs": [
            {
                "run_name": "legacy_hotpotqa",
                "method_label": "legacy",
                "dataset": "hotpotqa",
                "profile": "legacy",
                "config_path": "/tmp/legacy_hotpotqa.yaml",
                "run_root": str(out_root / "raw" / "legacy_hotpotqa"),
            },
            {
                "run_name": "v12_hotpotqa",
                "method_label": "v12",
                "dataset": "hotpotqa",
                "profile": "unified_acr_rcedr_v12",
                "config_path": "/tmp/latest_hotpotqa.yaml",
                "run_root": str(out_root / "raw" / "v12_hotpotqa"),
            },
        ],
    }
    _write_json(out_root / "phase6x_run_manifest.json", manifest)

    sample_map = {
        "s1": SimpleNamespace(qid="s1", question="Q", answer="A", supporting_facts=[], metadata={}),
        "s2": SimpleNamespace(qid="s2", question="Q", answer="A", supporting_facts=[], metadata={}),
    }
    monkeypatch.setattr(phase6x, "load_samples_for_dataset", lambda _: sample_map)

    phase6x.build_audit(out_root=out_root, datasets=["hotpotqa"], top_k=5)
    validation = json.loads((out_root / "phase6x_artifact_validation.json").read_text(encoding="utf-8"))
    assert validation["scheduled_runs"] == 2
    assert validation["completed_run_names"] == 2
    assert validation["sample_id_mismatch"] == 1
