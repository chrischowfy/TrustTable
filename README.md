# TrustTable

TrustTable is a neuro-symbolic auditing framework for faithful TableQA
reasoning. This repository contains the public GitHub package: verifier code,
baseline runners, the TrustTable-Bench `small` diagnostic dataset, result
metadata, and release validation tools.

## What's Included

- `code/`: TrustTable and baseline verifier implementations.
- `data/small/`: public TrustTable-Bench small release data, 4 panels and 25
  JSON files.
- `results/table1.json`: paper/reproduction result table metadata.
- `manifest.json`: versions, counts, provenance, and release policy.
- `DATA_CARD.md`: dataset schema, taxonomy, counts, known quality notes.
- `scripts/validate_release.py`: checks data counts, schema, `_skipped`, and
  accidental secret/local-path leakage.

This repository intentionally publishes the `small` diagnostic release rather
than the planned full-scale panel data. Full-scale Panel A/B/C datasets are not
bundled here.

## Dataset: `small`

The public `small` data contains only active generated blocks. Records whose target
`generated_samples` block was marked `_skipped` are not exported, so placeholder
records do not inflate release size.

| Panel | Files | Total raw | Active | _skipped |
|---|---:|---:|---:|---:|
| FinQA | 6 | 1228 | 1102 | 126 |
| Med | 6 | 1210 | 1144 | 66 |
| PubH | 6 | 1208 | 989 | 219 |
| WTQ | 7 | 1583 | 1365 | 218 |
| Total | 25 | 5229 | 4600 | 629 |

The release is a unified Track-2 base plus Track-3 increment. Record-level
provenance fields are retained when present: `_track3_revised`,
`_track2_pad_uniform`, and `_track2_pad`.

Track-3 net increment, excluding Track-2 base records:

| Panel | Track-3 Net New |
|---|---:|
| FinQA | 867 |
| Med | 870 |
| PubH | 714 |
| WTQ | 920 |
| Total | 3371 |

WTQ includes 200 `type4_answer_perturb` records; perturb records are all
Track-2 base.

## Error Types

| Type | Shape | Description |
|---|---|---|
| `type1_correct` | `Z+ A+` | Faithful reasoning and correct answer |
| `type2_grounding_error` | `Z- A+` | Correct answer with fabricated or misgrounded table value |
| `type2_arithmetic_error` | `Z- A+` | Correct answer with erroneous arithmetic in the reasoning path |
| `type2_logic_error` | `Z- A+` | Correct answer with wrong rule, condition, quantifier, or aggregation |
| `type3_fully_wrong` | `Z- A-` | Wrong reasoning and wrong answer |
| `type4_calc_error` | `Z+ A-` | Correct grounding/logic with wrong calculation and wrong answer |
| `type4_answer_perturb` | `Z+ A-` | Correct reasoning with perturbed final answer; WTQ only |

## Layout

```text
.
├── code/
│   ├── run_pipeline_main.py
│   ├── run_pot_baseline.py
│   ├── run_atomic_skills_baseline.py
│   ├── run_llm_judge_stepwise.py
│   ├── run_vericot_baseline.py
│   ├── eval_cot_verifier.py
│   ├── src/
│   ├── utils/
│   ├── configs/
│   └── requirements.txt
├── data/
│   └── small/
│       ├── panel_a_finqa/
│       ├── panel_b_med/
│       ├── panel_b_pubh/
│       └── panel_c_wtq/
├── results/
├── scripts/
├── DATA_CARD.md
├── LICENSE
├── LICENSE-DATA
└── manifest.json
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt

export LLM_API_KEY=...
export LLM_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-chat

cd code
python run_pipeline_main.py \
  ../data/small/panel_c_wtq/type1_correct.json \
  /tmp/trusttable_wtq_type1.json
```

Run a baseline:

```bash
cd code
python run_pot_baseline.py \
  ../data/small/panel_a_finqa/type1_correct.json \
  /tmp/pot_finqa_type1.json
```

Available baseline runners:

- `run_pot_baseline.py`
- `run_atomic_skills_baseline.py`
- `run_llm_judge_stepwise.py`
- `run_vericot_baseline.py`

Validate the release tree:

```bash
python scripts/validate_release.py
```

## Configuration

Credentials are read from environment variables. Do not commit API keys.

```bash
cp .env.example .env
```

Supported variables:

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`

## Do Not Publish Legacy Artifacts

This release intentionally excludes legacy `processed_data/*` artifacts such as
`*_clean50_*`, `*_subset_100_*`, `*_subset_50_*`,
`*_enhanced_100.json`, `runner_input_cleaned_*_100.json`,
`wtq_subset_*_gemini3pro.json`, and `wtq_type1_clean_*.json`.

Those are Track-1 or early-stage artifacts superseded by this release.

## Citation

See `CITATION.cff`. If you use this repository, cite the TrustTable paper and
the upstream datasets listed in `LICENSE-DATA`.

## Licenses

- Code: MIT, see `LICENSE`.
- TrustTable-generated diagnostic annotations: CC BY 4.0, subject to upstream
  dataset restrictions. See `LICENSE-DATA`.
