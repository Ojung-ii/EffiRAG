# Strict Unified Copy-Span-Instruction Main Config

This directory contains the Phase-6 paper-main candidates for copy-span
instruction runs. Unlike `configs/SOTA_config/copy_span_instruction_4ds`, these
profiles do not use dataset-specific method toggles.

## Profiles

- `unified_large/`: seed/run/render topn = `45/24/24`
- `unified_medium/`: seed/run/render topn = `36/18/18`
- `unified_compact/`: seed/run/render topn = `30/15/15`

`unified_medium` is the primary main-method candidate.

## Strict Unified Lock

Every dataset in a profile uses the same method-level settings:

- `retrieval_objective_mode: baseline`
- `canonical_variant_name: phase6_strict_unified_v1`
- `answer_support_pinning_enabled: false`
- `final_top_slice_reorder_enabled: false`
- `corridor_answer_preserve_guarded_hotpot_enabled: false`
- `oracle_support_injection_enabled: false`
- `max_anchors: 4`
- `samples_per_anchor: 4`

Dataset names may select dataset files and corpus paths, but not method
behavior.

## Refresh

```bash
PYTHONPATH=. python scripts/update_unified_config_copy_span_instruction_4ds.py
```

## Audit

```bash
PYTHONPATH=. python -m effirag.config_audit configs/main_config/copy_span_instruction_unified
```

## Smoke

```bash
python scripts/run_unified_config_4ds.py \
  --profile unified_medium \
  --datasets hotpotqa \
  --sample-size 100 \
  --output-root outputs/phase6_strict_unified_smoke/hotpotqa_medium
```
