# Phase8 Chunk-Medoid Handoff

This branch is the colleague-facing Phase8 chunk-medoid handoff branch.

It is rebuilt from `phase7` and adds only the code, scripts, tests, and minimal
Phase7 compatibility layer needed to inspect and run the Phase8 chunk-medoid
proposal path. Historical output JSONL files and repaired report artifacts are
not included.

## High-Level Intent

Phase8 chunk-medoid tests a chunk/carrier-first proposal source:

```text
query
-> chunk/carrier universe
-> query-weighted chunk medoids
-> optional bridge-entity refinement
-> sentence atom candidates
-> existing Phase7 selection/rendering
```

The current implementation does not replace Phase7 rendering or QA prompting.
It only changes the proposal source that feeds candidate atoms into the existing
Phase7 evidence-flow path.

## Read The Code In This Order

1. `effirag/phase8_chunk_medoid.py`

   Main Phase8 chunk-medoid implementation.

   Useful entry points:

   - `run_phase8_chunk_medoid_proposal`
   - chunk/carrier universe builders
   - medoid sampling/scoring helpers
   - bridge-entity refinement helpers
   - seed-to-sentence candidate expansion

2. `effirag/phase7_evidence_flow.py`

   Integration point with Phase7. Search for:

   - `phase8_chunk_medoid_enabled`
   - `run_phase8_chunk_medoid_proposal`
   - `phase8_chunk_medoid_proposal`
   - `phase8_chunk_diagnostics`

   This is where Phase8 chunk candidates are converted into Phase7 candidate
   atoms and then passed to the existing selection/rendering path.

3. `effirag/phase8_chunk_logging.py`

   Runtime trace writer for chunk-medoid runs.

4. `effirag/config.py`

   Config/CLI surface. Search for:

   - `phase8_chunk_`
   - `phase8_pamae_`

   `phase8_pamae_entity_seeding.py` is also present because chunk-medoid reuses
   shared helpers from the entity-side Phase8 implementation.

5. `scripts/run_phase8_chunk_medoid_experiment.sh`

   Main chunk-medoid runner.

6. `scripts/run_phase8_chunk_only_probe.sh`

   Small diagnostic runner for chunk-only probes.

7. `scripts/summarize_phase8_chunk_medoid.py`

   Summary script for completed chunk-medoid runs.

8. `scripts/analyze_phase8_chunk_medoid_failures.py`

   Failure-analysis script for candidate/source diagnostics.

## Minimal Validation

```bash
git fetch origin
git checkout phase8-chunk-medoid-handoff

PYTHONPATH=. python -m py_compile effirag/*.py
PYTHONPATH=. python scripts/audit_phase7_method.py

PYTHONPATH=. /home/ojungii/miniconda3/envs/effirag/bin/python -m pytest -q \
  tests/test_phase8_chunk_medoid.py \
  tests/test_phase8_chunk_trace_isolation.py
```

## Small Probe

```bash
CUDA_VISIBLE_DEVICES=0 DATASET=2wikimultihopqa PROFILE=balanced_384_8 LIMIT=5 \
bash scripts/run_phase8_chunk_only_probe.sh
```

or the full chunk-medoid runner:

```bash
CUDA_VISIBLE_DEVICES=0 DATASET=2wikimultihopqa PROFILE=balanced_384_8 LIMIT=10 \
PHASE8_CHUNK_VARIANTS=chunk_pamae_k5 \
bash scripts/run_phase8_chunk_medoid_experiment.sh
```

Summarize:

```bash
python scripts/summarize_phase8_chunk_medoid.py \
  --root outputs/phase8_chunk_medoid
```

## Important Files

| Path | Role |
| --- | --- |
| `effirag/phase8_chunk_medoid.py` | Main chunk-medoid proposal logic |
| `effirag/phase8_chunk_logging.py` | Runtime trace/log helpers |
| `effirag/phase8_pamae_entity_seeding.py` | Shared Phase8 helper functions reused by chunk-medoid |
| `effirag/phase7_evidence_flow.py` | Phase8-to-Phase7 integration |
| `scripts/run_phase8_chunk_medoid_experiment.sh` | Main experiment launcher |
| `scripts/run_phase8_chunk_only_probe.sh` | Minimal probe launcher |
| `tests/test_phase8_chunk_medoid.py` | Core unit tests |
| `tests/test_phase8_chunk_trace_isolation.py` | Trace isolation tests |

## Compatibility Layer

This branch includes a small Phase7 compatibility layer:

- `effirag/phase7_config.py`
- `effirag/phase7_source_balanced.py`
- `effirag/phase7_diagnostics.py`
- `effirag/phase7_logging.py`

These are included because Phase8 chunk-medoid runs through the existing
Phase7 evidence-flow pipeline. They are not the conceptual focus of Phase8.

## Current Caveat

Prior diagnostics suggested that chunk-medoid is not yet a finished replacement
for Phase7 source-balanced proposal. In particular, the useful debugging lens is:

```text
Does the chunk/carrier universe contain the needed evidence?
If not, medoid selection cannot recover it.
```

So when reviewing results, check:

- chunk universe coverage,
- medoid seed quality,
- bridge-refinement additions,
- unique evidence candidates,
- selected support precision/recall,
- retrieval latency.

If the chunk universe misses the relevant carriers, the next fix should target
chunk universe construction rather than medoid scoring.
