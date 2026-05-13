# Copy-Span-Instruction SOTA Config (4 Datasets)

This directory is the single source of truth for reproducible 4-dataset
copy-span-instruction runs.

## Contract
- `prompt_variant=light_separator_copy_span_instruction`
- `order_strategy=score+light_separator_render`
- `added_instruction=copy_span_only` (enforced by prompt variant template)

## Profiles
- `locked_precomputed/`
  - Uses fixed `precomputed_retrieval_path` for stable quality reproduction.
  - Recommended for SOTA regression checks and comparison baselines.
  - Provenance of that precomputed retrieval is stored in `manifest.json` under
    `datasets.<name>.retrieval_provenance`.
- `on_the_fly/`
  - Clears `precomputed_retrieval_path` to measure retrieval latency in current runs.
  - Recommended for timing diagnostics and retrieval-system changes.

## Retrieval Provenance
- `manifest.json` now records the full precomputed chain for each dataset:
  - `retrieval_provenance.locked_precomputed_query_results`
  - `retrieval_provenance.chain[]` (per-hop run/config/summary and retrieval contract)
  - `retrieval_provenance.terminal_on_the_fly_query_results`
  - `retrieval_provenance.terminal_retrieval_contract`
- For strict score reproduction, use `locked_precomputed`.
- For retrieval regeneration, follow `terminal_retrieval_contract` and rebuild
  from `terminal_on_the_fly_query_results` lineage.

## Build / Refresh
```bash
cd /home/ojungii/EffiRAG
/home/ojungii/miniconda3/envs/effirag/bin/python scripts/update_sota_config_copy_span_instruction_4ds.py
```

## Run 4 Datasets
Locked (quality reproduction):
```bash
cd /home/ojungii/EffiRAG
/home/ojungii/miniconda3/envs/effirag/bin/python scripts/run_sota_config_4ds.py --mode locked_precomputed
```

On-the-fly (retrieval measured this run):
```bash
cd /home/ojungii/EffiRAG
/home/ojungii/miniconda3/envs/effirag/bin/python scripts/run_sota_config_4ds.py --mode on_the_fly
```

## Retrieval Repro (No Historical Precomputed Consumption)
Replays `retrieval_provenance.chain` from terminal hop to root hop, so precomputed
inputs are regenerated in the current run before being consumed by parent hops.

```bash
cd /home/ojungii/EffiRAG
CUDA_VISIBLE_DEVICES=1 /home/ojungii/miniconda3/envs/effirag/bin/python \
  scripts/replay_sota_retrieval_provenance.py \
  --datasets hotpotqa \
  --retrieval-only true
```

Hotpot + 2Wiki end-to-end repro (retrieval -> QA):
```bash
cd /home/ojungii/EffiRAG
CUDA_VISIBLE_DEVICES=1 /home/ojungii/miniconda3/envs/effirag/bin/python \
  scripts/replay_sota_retrieval_provenance.py \
  --datasets hotpotqa,2wikimultihopqa \
  --retrieval-only true \
  --run-qa-from-replayed-root true
```

## Update Policy
- When a new experiment beats current SOTA and passes sanity checks:
  1. Replace source run mapping in `scripts/update_sota_config_copy_span_instruction_4ds.py`.
  2. Re-run the update script.
  3. Commit updated YAML files and `manifest.json` together.
