# Next Method Design Guide (Branch-Isolated)

This guide documents branch-isolated methodology experiments after the frozen operating point.

## Branch / Freeze Rules
- Frozen operating point is tagged as `freeze-20260329`.
- Next methodology work must run on branch `exp/method-next` (or child experiment branches).
- Do not edit frozen scripts/configs/reports in-place.

## Experiment Families
- `system_optimization`: bottleneck reduction and wall-clock optimization.
- `operating_point_selection`: speed/quality default selection and guardrail baselines.
- `method_ablation`: semantic/bridge/support ablations.
- `next_method_design`: objective redesign, grounding redesign, failure diagnosis.

## New Configs (next_method_design)
- `configs/rag_nextdesign_2wiki_run_objective_*.yaml`
- `configs/retrieval_nextdesign_2wiki_run_objective_*.yaml`
- `configs/rag_nextdesign_2wiki_grounding_*.yaml`
- `configs/retrieval_nextdesign_2wiki_grounding_*.yaml`
- `configs/rag_nextdesign_musique_diagnosis_baseline.yaml`
- `configs/retrieval_nextdesign_musique_diagnosis_baseline.yaml`

## New Scripts
- `scripts/run_2wiki_run_objective_ablation.sh`
- `scripts/run_2wiki_grounding_ablation.sh`
- `scripts/run_musique_failure_diagnosis.sh`
- `scripts/generate_next_phase_operating_strategy.sh`
- `scripts/run_next_method_design_suite.sh`

## Output Namespaces
- Profiling: `outputs/profiling/next_method_design/...`
- QA/RAG runs: `outputs/rag_experiments/next_method_design/...`
- Retrieval-only runs: `outputs/retrieval_experiments/next_method_design/...`

## Recommended Execution Order
1. 2Wiki run-objective ablation
2. 2Wiki grounding ablation
3. MuSiQue failure diagnosis
4. Strategy + final interpretation report generation

### One-shot run
```bash
cd /path/to/EffiRAG
CUDA_VISIBLE_DEVICES=0 LIMIT=200 bash scripts/run_next_method_design_suite.sh
```

### Individual runs
```bash
cd /path/to/EffiRAG
CUDA_VISIBLE_DEVICES=0 LIMIT=200 bash scripts/run_2wiki_run_objective_ablation.sh
CUDA_VISIBLE_DEVICES=0 LIMIT=200 bash scripts/run_2wiki_grounding_ablation.sh
CUDA_VISIBLE_DEVICES=0 MODE=qa LIMIT=200 bash scripts/run_musique_failure_diagnosis.sh
bash scripts/generate_next_phase_operating_strategy.sh
```

## Metadata Tracking
The summary generator now supports:
- `experiment_family`
- `question_being_answered`
- `baseline_reference`
- `frozen_config_reference`
- `dataset_scope`
- `changed_components`
- `git_branch`, `git_commit`, `git_tag`

Use these fields for branch/commit-aligned reproducibility in reports.
