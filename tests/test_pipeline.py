import json

from effirag.config import RetrievalConfig
from effirag.run_retrieval import execute_retrieval_experiment


def test_retrieval_pipeline_execution(tmp_path):
    data_path = tmp_path / "hotpot.json"
    payload = [
        {
            "_id": "p1",
            "question": "What city is the capital of France?",
            "answer": "Paris",
            "supporting_facts": [["Paris", 0]],
            "context": [
                ["Paris", ["Paris is the capital of France.", "It is in Europe."]],
                ["France", ["France is a country."]],
            ],
            "type": "bridge",
        }
    ]
    data_path.write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "out"
    cfg = RetrievalConfig(
        dataset="hotpotqa",
        data_path=str(data_path),
        output_dir=str(out_dir),
        limit=1,
        method="effirag",
        num_workers=1,
        samples_per_anchor=2,
    )

    rows, summary = execute_retrieval_experiment(cfg)

    assert len(rows) == 1
    assert summary["n_samples"] == 1.0
    assert (out_dir / "retrieval_query_results.jsonl").exists()
    assert (out_dir / "retrieval_summary.json").exists()
