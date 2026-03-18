# EffiRAG MVP (HotpotQA)

Modular minimum-viable GraphRAG experiment codebase with:
- Dataset: `hotpotqa`
- Method: `effirag`
- Baseline: `naive_graphrag`
- Retrieval eval: supporting-fact recall/precision
- QA eval: EM and token F1
- Efficiency eval: retrieval latency and total latency
- Optional efficiency eval: GPU peak memory and CPU RAM

## Repository Layout

```text
EffiRAG/
  README.md
  requirements-retrieval.txt
  requirements-rag.txt
  environment.template.yml
  scripts/
    run_retrieval.sh
    run_rag.sh
    run_ablation.sh
  configs/
    retrieval.yaml
    rag.yaml
    ablation.yaml
  effirag/
    __init__.py
    types.py
    config.py
    utils.py
    registry.py
    datasets.py
    graph.py
    anchors.py
    retrieval.py
    baselines.py
    render.py
    generator.py
    metrics.py
    qa_metrics.py
    efficiency.py
    run_retrieval.py
    run_rag.py
    run_ablation.py
  tests/
    test_retrieval.py
    test_render.py
    test_metrics.py
    test_pipeline.py
```

## Setup

### Retrieval-only dependencies

```bash
pip install -r requirements-retrieval.txt
```

### RAG dependencies

```bash
pip install -r requirements-rag.txt
```

### Conda template

```bash
conda env create -f environment.template.yml
conda activate effirag
```

## Data

Supported dataset name is currently `hotpotqa`.

You can run with:
- local file via `--data-path` (`.json` or `.jsonl`, HotpotQA-like schema), or
- Hugging Face `hotpot_qa` (when available), with automatic fallback demo samples for quick smoke tests.

## Retrieval Run Example

```bash
python3 -m effirag.run_retrieval \
  --dataset hotpotqa \
  --method effirag \
  --split validation \
  --limit 50 \
  --max-anchors 4 \
  --samples-per-anchor 8 \
  --num-workers 4 \
  --candidate-top-t 20 \
  --seed-k 4 \
  --pair-top-lp 4 \
  --corridor-top-bc 20 \
  --trim-on true \
  --trim-rho 0.75 \
  --output-dir outputs/retrieval
```

or

```bash
scripts/run_retrieval.sh
```

실행 종료 시 터미널에 요약 결과가 마크다운 테이블로 출력됩니다.

## RAG Run Example

```bash
python3 -m effirag.run_rag \
  --dataset hotpotqa \
  --method effirag \
  --split validation \
  --limit 50 \
  --max-anchors 4 \
  --samples-per-anchor 8 \
  --num-workers 1 \
  --candidate-top-t 20 \
  --seed-k 4 \
  --pair-top-lp 4 \
  --corridor-top-bc 20 \
  --trim-on true \
  --trim-rho 0.75 \
  --run-qa true \
  --generator heuristic \
  --model-name "" \
  --max-context-sentences 15 \
  --measure-gpu-peak false \
  --measure-cpu-ram true \
  --output-dir outputs/rag
```

or

```bash
scripts/run_rag.sh
```

실행 종료 시 터미널에 요약 결과가 마크다운 테이블로 출력됩니다.

## Ablation Run Example

```bash
python3 -m effirag.run_ablation \
  --dataset hotpotqa \
  --split validation \
  --limit 100 \
  --max-anchors 4 \
  --samples-per-anchor 8 \
  --num-workers 4 \
  --candidate-top-t 20 \
  --seed-k 4 \
  --pair-top-lp 4 \
  --corridor-top-bc 20 \
  --trim-rho 0.75 \
  --output-dir outputs/ablation
```

or

```bash
scripts/run_ablation.sh
```

실행 종료 시 터미널에 variant별 요약 결과가 마크다운 테이블로 출력됩니다.

## CLI Arguments

Implemented CLI parameters:
- `--max-anchors`
- `--samples-per-anchor`
- `--num-workers`
- `--candidate-top-t`
- `--seed-k`
- `--pair-top-lp`
- `--corridor-top-bc`
- `--trim-on`
- `--trim-rho`
- `--run-qa`
- `--generator`
- `--model-name`
- `--max-context-sentences`
- `--measure-gpu-peak`
- `--measure-cpu-ram`

## Output Files

### Retrieval runner
- `outputs/.../retrieval_query_results.jsonl`
- `outputs/.../retrieval_summary.json`
- `outputs/.../logs/config_<YYYYMMDD_HHMMSS_micro>.json`
- `outputs/.../logs/result_<YYYYMMDD_HHMMSS_micro>.json`
- `outputs/.../logs/config_history.jsonl`
- `outputs/.../logs/result_history.jsonl`

### RAG runner
- `outputs/.../rag_query_results.jsonl`
- `outputs/.../rag_summary.json`
- `outputs/.../logs/config_<YYYYMMDD_HHMMSS_micro>.json`
- `outputs/.../logs/result_<YYYYMMDD_HHMMSS_micro>.json`
- `outputs/.../logs/config_history.jsonl`
- `outputs/.../logs/result_history.jsonl`

### Ablation runner
- per-variant retrieval outputs under `outputs/.../<variant>/`
- `outputs/.../ablation_summary.json`

## Notes on Modularity

- `retrieval.py` does not depend on `generator.py`.
- `generator.py` consumes rendered text and does not know retrieval internals.
- `naive_graphrag` returns the same `RetrievalResult` schema as `effirag`.
