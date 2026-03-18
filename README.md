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
    run_toy.sh
    run_grid.sh
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

## Toy Run Shortcut

`effirag` vs `naive_graphrag`를 retrieval + RAG에서 한 번에 비교:

```bash
scripts/run_toy.sh
```

HF 생성기 비교까지 포함:

```bash
scripts/run_toy.sh --run-hf --hf-model Qwen/Qwen3.5-2B
```

oracle(검색 upper-bound sanity check)까지 포함:

```bash
scripts/run_toy.sh --run-oracle
```

비교 메서드 변경 예시:

```bash
scripts/run_toy.sh --methods effirag,naive_graphrag
```

실행 후 CLI에 통합 summary table이 출력되고, 아래 파일로 저장됩니다:
- `outputs/toy_compare_<timestamp>/logs/toy_summary.md`
- `outputs/toy_compare_<timestamp>/logs/toy_summary.json`
- `outputs/toy_compare_<timestamp>/logs/toy_summary.csv`

## Grid Run (Parameter Sweep)

EffiRAG 파라미터 조합을 cartesian grid로 실행하고, 통합 요약 테이블을 저장:

```bash
scripts/run_grid.sh \
  --dataset hotpotqa \
  --retrieval-limit 100 \
  --rag-limit 100 \
  --grid-method effirag \
  --include-naive-baseline true \
  --generators heuristic,hf \
  --hf-model Qwen/Qwen3.5-2B \
  --max-anchors-grid 4,6 \
  --candidate-top-t-grid 16,20 \
  --seed-k-grid 4,6 \
  --pair-top-lp-grid 3 \
  --corridor-top-bc-grid 14,20 \
  --trim-rho-grid 0.6,0.75 \
  --tau-grid 4 \
  --max-context-sentences-grid 10,15
```

시간 단축용(권장) 2-stage 실행 예시:
- 1단계: retrieval은 전체 조합 실행
- 2단계: retrieval 상위 `k`개 프로파일만 RAG 실행
- RAG는 `--rag-workers`로 프로파일 단위 병렬 실행
- `--reuse-retrieval true`(기본값)이면 RAG에서 retrieval을 재실행하지 않고 prepass 결과를 재사용

```bash
scripts/run_grid.sh \
  --dataset hotpotqa \
  --retrieval-limit 100 \
  --rag-limit 100 \
  --rag-topk-profiles 8 \
  --rag-topk-metric sf_recall \
  --rag-workers 2 \
  --reuse-retrieval true \
  --generators heuristic,hf \
  --max-anchors-grid 4,6 \
  --candidate-top-t-grid 16,20 \
  --seed-k-grid 4,6 \
  --pair-top-lp-grid 3 \
  --corridor-top-bc-grid 14,20 \
  --trim-rho-grid 0.6,0.75 \
  --tau-grid 4 \
  --max-context-sentences-grid 10,15
```

실행 후 CLI에 `**Total Eval Summary**`가 출력되고, 아래 파일로 저장됩니다:
- `outputs/grid_compare_<timestamp>/logs/grid_profiles.json`
- `outputs/grid_compare_<timestamp>/logs/rag_selected_profiles.json` (`--rag-topk-profiles` 사용 시)
- `outputs/grid_compare_<timestamp>/logs/grid_summary.md`
- `outputs/grid_compare_<timestamp>/logs/grid_summary.json`
- `outputs/grid_compare_<timestamp>/logs/grid_summary.csv`

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
