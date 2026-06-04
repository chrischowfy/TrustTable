# TrustTable-Bench Data Card

## Dataset Summary

TrustTable-Bench is a diagnostic TableQA reasoning-faithfulness dataset. Each
record contains a table, a question, a gold answer, and one generated reasoning
sample for a specific diagnostic type.

This GitHub release contains the `small` diagnostic split: the unified Track-2
base plus Track-3 increment. Only active, non-`_skipped` generated samples are
included. The planned full-scale Panel A/B/C datasets are not included in this
repository.

## Error Taxonomy

| Type | Shape | Meaning |
|---|---|---|
| Type 1 | `Z+ A+` | Faithful reasoning and correct answer |
| Type 2 grounding | `Z- A+` | Correct answer with fabricated or misgrounded table value |
| Type 2 arithmetic | `Z- A+` | Correct answer with erroneous arithmetic in the reasoning path |
| Type 2 logic | `Z- A+` | Correct answer with wrong rule, condition, quantifier, or aggregation |
| Type 3 | `Z- A-` | Wrong reasoning and wrong answer |
| Type 4 calc | `Z+ A-` | Correct grounding/logic with wrong calculation and wrong answer |
| Type 4 perturb | `Z+ A-` | Correct reasoning with perturbed final answer; WTQ only |

## Released Counts

| Panel | Files | Active Records |
|---|---:|---:|
| FinQA | 6 | 1102 |
| Med | 6 | 1144 |
| PubH | 6 | 989 |
| WTQ | 7 | 1365 |
| Total | 25 | 4600 |

Files are stored under `data/small/`. See `manifest.json` for per-type raw,
skipped, and active counts.

## Schema

Each record contains:

- `id`
- `original_question`
- `gold_answer`
- `table_md`
- optional `table_content`
- `generated_samples`, containing exactly one type-specific block
- optional record-level provenance fields:
  - `_track3_revised`
  - `_track2_pad_uniform`
  - `_track2_pad`

## Known Quality Notes

The `small` dataset is designed for diagnostic verification, not model training
at web scale. Some Track-2 base samples predate the stricter Track-3 audit
metadata.
For stricter slices, users can filter to records with type-specific `_gen_*`
audit fields or use the record-level provenance fields.

Known higher-risk subsets:

- Type 2 grounding samples without `_gen_grounding_*` metadata.
- Type 2 logic samples without `_gen_chosen_pattern`.
- Samples where financial or clinical terminology has multiple plausible
  accounting/domain interpretations.

## Licensing

TrustTable-generated annotations are released under CC BY 4.0, subject to
upstream dataset restrictions. See `LICENSE-DATA`.
