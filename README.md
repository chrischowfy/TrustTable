# TrustTable

TrustTable is a neuro-symbolic auditing framework for faithful TableQA
reasoning. It evaluates whether a model-generated reasoning trace is grounded
in table evidence, mathematically consistent, and logically aligned with the
question, rather than scoring only the final answer.

This repository contains the public TrustTable release package: verifier code,
baseline runners, the TrustTable-Bench small diagnostic dataset, paper result
metadata, and a lightweight validation script.

## Highlights

- Audits table-based chain-of-thought reasoning with fact, arithmetic, logic,
  and consistency checks.
- Includes baseline runners for Program-of-Thoughts, atomic skill auditing,
  stepwise LLM judging, and VeriCoT-style verification.
- Provides a compact public benchmark split across finance, medical,
  public-health, and WikiTableQuestions-style tables.
- Ships with reproducibility metadata for the main reported result table.

## Repository Layout

```text
.
├── code/                  # TrustTable and baseline runner implementations
├── data/small/            # Public TrustTable-Bench small diagnostic release
├── results/table1.json    # Result metadata for the main comparison table
├── scripts/               # Release validation utilities
├── CITATION.cff
├── LICENSE
└── LICENSE-DATA
```

## Dataset

The public dataset is the `small` release of TrustTable-Bench. It contains
4,600 active records across 25 JSON files.

| Panel | Directory | Files | Records |
|---|---|---:|---:|
| Panel A: FinQA | `data/small/panel_a_finqa/` | 6 | 1,102 |
| Panel B: Medical | `data/small/panel_b_med/` | 6 | 1,144 |
| Panel B: Public Health | `data/small/panel_b_pubh/` | 6 | 989 |
| Panel C: WTQ | `data/small/panel_c_wtq/` | 7 | 1,365 |
| Total | `data/small/` | 25 | 4,600 |

Each record includes a question, table, gold answer, and one generated
reasoning block. The generated block encodes one diagnostic condition:

| Type | Meaning |
|---|---|
| `type1_correct` | Faithful reasoning with a correct answer |
| `type2_grounding_error` | Correct answer with unsupported or misgrounded evidence |
| `type2_arithmetic_error` | Correct answer with an arithmetic error in the reasoning |
| `type2_logic_error` | Correct answer with an invalid logical step |
| `type3_fully_wrong` | Wrong reasoning and wrong answer |
| `type4_calc_error` | Faithful grounding and logic with a wrong calculation or answer |
| `type4_answer_perturb` | Faithful reasoning with a perturbed final answer; WTQ only |

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

TrustTable runners read model credentials from environment variables:

```bash
export LLM_API_KEY=your_api_key
export LLM_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-chat
```

The same variables can also be placed in a local `.env` file based on
`.env.example`. Do not commit populated credential files.

## Quick Start

Run the TrustTable verifier on one dataset file:

```bash
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

Available runners:

- `run_pipeline_main.py`
- `run_pot_baseline.py`
- `run_atomic_skills_baseline.py`
- `run_llm_judge_stepwise.py`
- `run_vericot_baseline.py`

## License

- Code is released under the MIT License. See `LICENSE`.
- TrustTable-generated diagnostic annotations are released under CC BY 4.0,
  subject to upstream dataset restrictions. See `LICENSE-DATA`.
