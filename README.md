# EffiRAG MVP (Multi-dataset)

Modular minimum-viable GraphRAG experiment codebase with:
- Dataset: `hotpotqa`, `musique`, `2wikimultihopqa`, `popqa`
- Method: `effirag`
- Baseline: `naive_graphrag`
- Retrieval eval: supporting-fact recall/precision + Recall@K (`K=1,2,5,10,20,30,50,100,150,200`)
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
    build_datasets.sh
    run_retrieval.sh
    run_rag.sh
    run_ablation.sh
    run_toy.sh
    run_grid.sh
    run_dataset_suite.sh
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
    prepare_datasets.py
    run_retrieval.py
    run_rag.py
    run_ablation.py
  tests/
    test_datasets.py
    test_generator.py
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

Supported dataset names:
- `hotpotqa`
- `musique`
- `2wikimultihopqa` (alias: `twowikimultihopqa`)
- `popqa`

You can run with:
- local file via `--data-path` (`.json` or `.jsonl`), or
- HF datasets (when available), with automatic fallback demo samples for quick smoke tests.

로컬 파일 없이 실행해도 아래 경로가 있으면 자동 탐색합니다:
- `./data/qa/<dataset>.json`
- `./data/<dataset>.json`
- `~/raptor/data/qa/<dataset>.json`
- `~/HippoRAG/data/<dataset>.json`

데이터셋 구축(정규화 json 생성) 예시:

```bash
scripts/build_datasets.sh \
  --datasets hotpotqa,musique,2wikimultihopqa,popqa \
  --source-root /home/ojungii/raptor/data/qa \
  --output-root data/qa \
  --overwrite true
```

`--source-root`를 생략하면 로더가 HF/자동탐색 경로를 시도합니다.

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

참고: `supporting_fact_recall`은 각 샘플의 최종 선택 문장 전체에 대한 recall이며, 추가로 `Recall@K` 평균(`supporting_fact_recall_at_<K>`)이 요약 json에 저장됩니다.

## Global Corpus KG Index (One-time Build + Reuse)

전역 코퍼스(예: `*_corpus.json`)에서 한 번 그래프를 구축해 캐시에 저장하고, 이후 retrieval/RAG에서 재사용할 수 있습니다.

```bash
python3 -m effirag.run_index \
  --corpus-path data/hipporag2/hotpotqa_corpus.json \
  --cache-dir outputs/index_cache \
  --force-rebuild false
```

retrieval/rag 실행 시 아래 옵션을 주면 query-local context 대신 전역 코퍼스 그래프를 사용합니다.

- `--global-corpus-path`
- `--graph-cache-dir`
- `--force-rebuild-graph-index true|false`

렌더링 모드:
- `effirag`: 기본 `corridor_aware_flat` (메서드 기반 자동 선택)
- `naive_graphrag`: 기본 `flat`
- 수동 지정: `--render-mode flat|corridor|corridor_aware_flat`

corridor 렌더 예산 파라미터:
- `--max-corridors-in-context` (default: 2)
- `--max-main-sentences-per-corridor` (default: 2)
- `--max-support-per-corridor` (default: 1)
- `--max-total-sentences` (default: 12)

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

## Multi-dataset RAG Run

`hotpotqa,musique,2wikimultihopqa,popqa`를 순차 실행:

```bash
scripts/run_dataset_suite.sh \
  --data-root data/qa \
  --model-name Qwen/Qwen2.5-7B-Instruct
```

샘플 제한 테스트:

```bash
scripts/run_dataset_suite.sh \
  --datasets musique,2wikimultihopqa,popqa \
  --data-root data/qa \
  --limit 100 \
  --model-name Qwen/Qwen2.5-7B-Instruct
```

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
