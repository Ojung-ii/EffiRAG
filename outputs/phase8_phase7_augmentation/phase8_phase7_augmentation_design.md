# Phase8 On Phase7 Augmentation Design

## Verdict From Prior Phase8 Runs

Phase8 should not replace Phase7 source-balanced proposal in its current form.

The repaired Phase8 diagnostics show:

- Raw entity PAMAE (`pamae_seed_k5`) is a weak replacement. Candidate recall and SF-R collapse on both HotpotQA and 2Wiki.
- Entity PAMAE with one-step refinement has useful signal on 2Wiki: candidate recall rises from about `0.03` to about `0.36`, but it remains below source-balanced and is slower.
- Chunk-first medoids are not ready. The minimal 2Wiki probe had `CHUNK_UNIVERSE_MISS = 5/5`, so the chunk universe fails before medoid selection can help.
- The realistic next test is therefore not a new standalone Phase8 method. It is a narrow augmentation test: preserve Phase7 source-balanced candidates and add a capped Phase8 entity-refine candidate tail.

## Hypothesis

Phase7 source-balanced proposal has high recall but may miss some bridge/local evidence. Phase8 entity-refine can sometimes recover additional evidence, especially on 2Wiki. If Phase8 is capped and added after Phase7 candidates, it may improve candidate or selected recall without destroying Phase7 precision.

## Active Variants

| variant | purpose |
| --- | --- |
| `source_balanced_128` | External Phase7 reference. This is not a Phase8 method. |
| `sb_plus_entity_refine_k5` | Primary test: Phase7 source-balanced backbone plus capped Phase8 entity PAMAE refine candidates. |

Optional diagnostic only:

| variant | purpose |
| --- | --- |
| `sb_plus_chunk_k5` | Disabled by default. Use only if checking whether chunk medoids add unique evidence after C_q is repaired. |

## Method

For augmentation runs:

1. Build normal Phase7 source-balanced local graph and candidate atoms.
2. Run Phase8 entity PAMAE with one-step refinement.
3. Add Phase8 evidence candidates to the local graph.
4. Build Phase8 atoms from the Phase8 candidate ids.
5. Merge candidates by `source_id`:
   - preserve all base Phase7 candidates first,
   - merge tags and feature maxima for duplicates,
   - add only unique Phase8 candidates up to `phase8_augmentation_extra_candidate_cap`.
6. Run existing Phase7 feature construction, A+Bq-R selection, rendering, QA, and metrics unchanged.

## First Probe

Run a small probe before any full experiment:

- dataset/profile: `2wikimultihopqa / balanced_384_8`
- variants: `source_balanced_128,sb_plus_entity_refine_k5`
- limit: `10`

If the Phase8 tail does not add unique evidence or does not improve candidate recall, stop and redesign Phase8. If candidate recall improves but selected recall/F1 do not, the bottleneck is selection interaction.

## Overnight Four-Dataset Sweep

The overnight sweep expands the same hypothesis to four datasets:

- `hotpotqa`
- `2wikimultihopqa`
- `musique`
- `popqa`

Default grid:

- profiles: `legacy_512_10`, `balanced_384_8`
- variants: `source_balanced_128`, `sb_plus_entity_refine_k5`
- total runs: `4 datasets * 2 profiles * 2 variants = 16`

The launcher uses six worker processes by default:

- GPU 0: 2 workers
- GPU 1: 4 workers

This keeps the experiment focused on the single live question: whether Phase8 entity-refine adds useful evidence to Phase7 without becoming a Phase7 replacement.

## Success Criteria

Continue to larger runs only if:

- `normalized_candidate_gold_recall` or `normalized_selected_gold_recall` improves over `source_balanced_128`, and
- `SF-P` and `F1` do not materially drop, and
- `retrieval_ms_ratio` remains close to `1.30` or below, and
- `phase8_unique_added` is nonzero enough to show Phase8 is not merely duplicating Phase7.

## Rejection Criteria

Reject this augmentation if:

- candidate recall does not improve,
- selected recall/F1 decline,
- SF-P collapses,
- retrieval latency grows beyond the acceptable band,
- or `phase8_unique_added` is near zero.

If rejected, return to a full Phase8 redesign rather than adding more augmentation variants.
