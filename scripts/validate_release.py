#!/usr/bin/env python3
"""Validate the public TrustTable release tree.

This script intentionally uses only the Python standard library so it can run
immediately after cloning the release repository.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "small"

EXPECTED_COUNTS = {
    "panel_a_finqa": {
        "type1_correct": 191,
        "type2_arithmetic_error": 186,
        "type2_grounding_error": 163,
        "type2_logic_error": 181,
        "type3_fully_wrong": 191,
        "type4_calc_error": 190,
    },
    "panel_b_med": {
        "type1_correct": 198,
        "type2_arithmetic_error": 193,
        "type2_grounding_error": 165,
        "type2_logic_error": 192,
        "type3_fully_wrong": 200,
        "type4_calc_error": 196,
    },
    "panel_b_pubh": {
        "type1_correct": 172,
        "type2_arithmetic_error": 162,
        "type2_grounding_error": 151,
        "type2_logic_error": 163,
        "type3_fully_wrong": 173,
        "type4_calc_error": 168,
    },
    "panel_c_wtq": {
        "type1_correct": 200,
        "type2_arithmetic_error": 196,
        "type2_grounding_error": 188,
        "type2_logic_error": 188,
        "type3_fully_wrong": 200,
        "type4_answer_perturb": 200,
        "type4_calc_error": 193,
    },
}

REQUIRED_BLOCK_FIELDS = {
    "type1_correct": ("chain_of_thought", "answer"),
    "type2_grounding_error": ("flawed_chain_of_thought", "answer"),
    "type2_arithmetic_error": ("flawed_chain_of_thought", "answer"),
    "type2_logic_error": ("flawed_chain_of_thought", "answer"),
    "type3_fully_wrong": ("incorrect_chain_of_thought", "incorrect_answer"),
    "type4_calc_error": ("correct_logic_wrong_math_cot", "incorrect_answer"),
    "type4_answer_perturb": ("correct_logic_wrong_math_cot", "incorrect_answer"),
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"API_KEY\s*=\s*[\"'][^\"']{12,}[\"']"),
    re.compile("/root/" + "autodl"),
]


def present(value: object) -> bool:
    return value is not None and value != ""


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def validate_data() -> None:
    total = 0
    for panel, type_counts in EXPECTED_COUNTS.items():
        for typ, expected_count in type_counts.items():
            path = DATA / panel / f"{typ}.json"
            if not path.exists():
                fail(f"missing data file: {path.relative_to(ROOT)}")

            records = json.loads(path.read_text())
            if len(records) != expected_count:
                fail(f"{path.relative_to(ROOT)} has {len(records)} records; expected {expected_count}")
            total += len(records)

            ids = [record.get("id") for record in records]
            if len(ids) != len(set(ids)):
                fail(f"{path.relative_to(ROOT)} contains duplicate IDs")

            for idx, record in enumerate(records):
                for field in ("id", "original_question", "gold_answer"):
                    if not present(record.get(field)):
                        fail(f"{path.relative_to(ROOT)} record {idx} missing {field}")
                if not (present(record.get("table_md")) or present(record.get("table_content"))):
                    fail(f"{path.relative_to(ROOT)} record {idx} missing table")

                samples = record.get("generated_samples")
                if not isinstance(samples, dict) or list(samples) != [typ]:
                    fail(f"{path.relative_to(ROOT)} record {idx} has wrong generated_samples keys")
                block = samples[typ]
                if block.get("_skipped"):
                    fail(f"{path.relative_to(ROOT)} record {idx} still has _skipped")
                for field in REQUIRED_BLOCK_FIELDS[typ]:
                    if not present(block.get(field)):
                        fail(f"{path.relative_to(ROOT)} record {idx} missing block field {field}")

    if total != 4600:
        fail(f"release has {total} records; expected 4600")


def validate_manifest() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest.get("dataset_size") != "small":
        fail("manifest dataset_size must be 'small'")
    if manifest.get("data_root") != "data/small":
        fail("manifest data_root must be 'data/small'")
    active = manifest.get("panel_totals", {}).get("total", {}).get("active")
    if active != 4600:
        fail(f"manifest total active is {active}; expected 4600")


def validate_no_secrets() -> None:
    for path in ROOT.rglob("*"):
        if path.is_dir() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".cff", ".example"}:
            continue
        text = path.read_text(errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                fail(f"possible secret/local path in {path.relative_to(ROOT)}")


def main() -> None:
    validate_data()
    validate_manifest()
    validate_no_secrets()
    print("OK: small release tree is valid (25 files, 4600 active records, no _skipped blocks).")


if __name__ == "__main__":
    main()
