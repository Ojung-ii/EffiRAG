# Entity-Chunk Refactor Summary

## Core strategy adopted
- Keep search / diffusion entity-centric for stability.
- Use passage chunks as grounded evidence/context units.
- Use bounded-local corridor only as local evidence packaging signal, not as a hard requirement that chunks themselves form the primary diffusion frontier.

## Implemented changes
1. Added config audit utility and filename/strategy drift warnings.
2. Added multi-chunk passage indexing with sliding sentence windows.
3. Fixed lexical anchor scoring to use text-support frequency instead of sentence-only frequency.
4. Added seed objective augmentation with bridge bonus, chunk grounding bonus, and anchor coverage bonus.
5. Generalized corridor payload from sentence-only interface to text-unit interface while preserving legacy aliases.
6. Added canonical configs for the recommended entity-first chunk-grounded setup.
7. Added smoke test for chunk indexing + chunk-aware rendering.

## Validation performed
- `python -m py_compile effirag/*.py`
- `PYTHONPATH=. python tests/smoke_entity_chunk.py`
- `PYTHONPATH=. python -m effirag.config_audit configs/canonical`
- `PYTHONPATH=. python -m effirag.config_audit configs` (used to expose legacy config naming drift)
