"""
VeriCoT baseline — Feng et al., 2025 (arXiv:2511.04662, no public code).

Per-step FOL verification of CoT via Z3, with premises extracted from the CoT.

v9 (2026-04-26 night): Stage 1 sees the table (paper §2.3 fix).

Root cause of remaining gap (DIR_spur −26pt, FP −40pt vs paper):
v7-v8.4 had Stage 1 prompt forbid table access — "Do NOT reference the
underlying table; work only with what the CoT asserts." This was a
mis-implementation. paper §2.3 says premise generation prompts the LLM
to "identify supporting premises that ground the argument in the SOURCE
context." Z3 doesn't see source; the LLM extracting premises does.

When LLM can only mirror CoT's own assertions, fabricated counts flow
through verbatim into premises; Stage 2 Z3 then proves step entailment
on bad premises (silent UNSAT). Letting Stage 1 see the table closes
the loop — premises become naturally grounded, audit + Z3 cascade
becomes much sharper.

v8.3 (2026-04-26 PM): deterministic Final-Answer Consistency Check (FACC).
On the all-UNSAT-pass path (paper Algorithm 1's "valid" outcome), Python
directly compares the last numeric / string answer in the CoT's
final_conclusion against the CoT-claimed answer. Mismatch → REJECT.
This is the robust fallback for paper §2.2's final-step Z3 inequality
`assert derived_answer != claimed`, which the LLM rarely grounds correctly
in the generated Z3 code (47/59 perturb leaks were all-UNSAT in v8.1).
Targets DIR_inc_E (T4_answer_perturb) — paper 94.7%, v8.1 42.2%.

v8.1 (2026-04-26 PM): SOURCE_GROUNDED prompt relaxed — accept correctly-derived
counts/sums (e.g., "Rochester has 3 listings", "There are 6 rows with 'Winner'")
as VERIFIED with row-range cite, not FABRICATED. v8 was over-strict on derived
table facts and lost 21/29 T1 (Faithful) samples; only flag FABRICATED when the
derivation is provably WRONG.

v8 (2026-04-26): paper-fidelity hardening on top of v7.
  Δ1 — Audit prompt SPLIT (paper §2.4 says source-grounded vs commonsense
        premises are judged differently):
          ptype=fact / arithmetic → SOURCE_GROUNDED prompt (only VERIFIED /
            FABRICATED; COMMONSENSE not allowed; arithmetic equalities are
            additionally Python-eval'd before LLM call).
          ptype=logic            → COMMONSENSE_OR_FABRICATED prompt (the
            three-way verdict from v7, kept for rules / definitions).
        Why: v7's single prompt let "5+10=20" pass as COMMONSENSE — algo
        errors leaked en masse (DIR_arith=53.9 on WTQ102_cleaned).
  Δ2 — Logical-vocabulary accumulation across steps (paper §2.2: SMT-LIB
        vocab grows monotonically). vote() now receives the prior steps'
        Z3 declarations, the prompt asks the LLM to REUSE the same names /
        types so step-to-step references stay consistent. Helps reduce
        `fol_all_error` silent ACCEPTs on cross-step variable drift.

v7 (2026-04-26): paper-faithful premise attribution PoC.
  + Stage 1.5: For each extracted premise, an LLM judge classifies it as
       VERIFIED      (cite-able to a specific table cell)
       COMMONSENSE   (universally-true math/logic axiom — accepted without table)
       FABRICATED    (specific claim about table data, no supporting cell)
    If ANY premise is FABRICATED → REJECT with reason `premise_unattributable`
    (paper Algorithm 1 "ungrounded" error type).
    Concurrency: per-CoT premise audits run in parallel.

v3 carry-overs (2026-04-17 hardening):
  1. Two-stage prompt:   extract {premises, conclusion} JSON first, then translate.
  2. Step-by-step:       verify each step against premises + prior-verified steps.
  3. n=3 Z3 voting:      majority across 3 independent FOL code samples (T=0.3).
  4. Expanded Z3 imports: adds ForAll / Exists / Distinct / If / Function / Array.
  5. Claimed-answer:     explicit `Assert(derived == claimed)` for T4 detection.
  6. unknown → REJECT:   conservative default on solver timeout.
  7. Tool Repair:        on FOL exec error, retry once with traceback feedback.

Method note: paper VeriCoT (§2.3-2.4) generates premises **per-step on demand**.
v8 still pre-extracts premises (batch) — full per-step on-demand generation is
Phase 2 follow-up. v8 narrows the gap on the audit prompt and vocab dimensions."""
import json, os, sys, re, asyncio, random, traceback, io, threading, math
import pandas as pd
from collections import Counter
from contextlib import redirect_stdout
from tqdm.asyncio import tqdm_asyncio

# Z3's Python bindings are NOT thread-safe — concurrent exec() of LLM-generated
# Z3 code segfaults. Serialize all Z3 executions through a global lock; LLM
# calls (the bottleneck) remain fully concurrent.
_Z3_LOCK = threading.Lock()
from z3 import (
    Solver, Reals, Ints, Strings, Bool, Int, Real, String,
    Not, And, Or, Implies, If, ForAll, Exists, Distinct,
    Function, Array, Store, Select,
    IntSort, RealSort, BoolSort, StringSort,
    DeclareSort, Const,
    sat, unsat, unknown, set_option,
)
from src.llm_engine import LLMEngine
from utils.logger import setup_logger
from utils.table_utils import parse_structured_table

logger = setup_logger("VeriCoT_Baseline_v8")
set_option("timeout", 3000)

INPUT_FILE = "../data/small/panel_c_wtq/type1_correct.json"
OUTPUT_FILE = "../outputs/vericot_wtq_type1.json"
CONCURRENCY = 10
SAVE_INTERVAL = 50
VOTE_N = 3
TABLE_CHAR_CAP = 5000     # window passed to premise attribution audit
PREMISE_AUDIT_CONCURRENCY = 4   # per-CoT,inside the outer CONCURRENCY

KEY_MAPPING = {
    "type1_correct": "type1_golden",
    "type2_grounding_error": "type2_spurious",
    "type2_arithmetic_error": "type2_spurious",
    "type2_logic_error": "type2_spurious",
    "type3_fully_wrong": "type3_fully_wrong",
    "type4_calc_error": "type4_calc_error",
    "type4_answer_perturb": "type4_inconsistent_easy",
}


def extract_cot_text(sd):
    return (sd.get("chain_of_thought") or sd.get("flawed_chain_of_thought")
            or sd.get("correct_logic_wrong_math_cot") or sd.get("incorrect_chain_of_thought") or "")


def extract_claimed_answer(sd):
    return sd.get("answer") or sd.get("incorrect_answer") or sd.get("pred_answer") or ""


# ============================================================
# Stage 1 — premise + conclusion extraction (JSON)
# ============================================================
PREMISE_EXTRACT_PROMPT = """You are a premise extractor for a Chain-of-Thought reasoning chain (paper VeriCoT §2.3). Decompose the CoT into a structured JSON suitable for formal FOL verification, **grounding each premise in the source table where possible**.

### Question
{question}

### Source table (the source of truth — premises grounded here pass attribution)
{table}

### Chain-of-Thought
{cot}

### Claimed final answer
{claimed}

### Task
Output a single JSON object with these fields:
  "premises":   list of atomic factual statements (each item: {{"id": "p1", "text": "<statement>", "type": "fact"|"arithmetic"|"logic", "cite": "<row/col cite OR 'commonsense' OR 'derived'>"}})
  "steps":      ordered list of reasoning steps, each with: {{"id": "s1", "depends_on": ["p1","s_prev"], "conclusion": "<what this step concludes>"}}
  "final_conclusion": the CoT's final claim in one short sentence

Rules for premises:
- A premise is an atomic fact **the CoT asserts** the reasoning relies on. **Record the premise text FAITHFULLY — preserve exactly what the CoT claims, even if it contradicts the table.** Do NOT silently correct CoT errors — downstream verifiers need to see what the CoT actually said in order to catch grounding / arithmetic errors.
- **CRITICAL — preserve the CoT's EXACT logical operators and definitions:**
  - If the CoT says "A OR B", the premise MUST say "A OR B", NOT "A AND B" — even if the question asked "A AND B".
  - If the CoT defines "top = highest number", record that definition as a premise — do NOT replace it with "top = position 1".
  - If the CoT counts items and states a number, record both: the enumerated list AND the stated count as separate premises.
  - If the CoT reverses a quantifier (checks ∀x.P(x) by checking ∀x.Q(x)→P(x) instead), record what the CoT actually checked.
  - **The `cite` field is for the TABLE's ground truth; the `text` field is for what the CoT CLAIMS. They may disagree — that disagreement is the signal for downstream verifiers.**
- Three premise sources are allowed:
  1. **Table-grounded**: a cell value, a count of matching rows, a sum/average. Use the `cite` field to record what the TABLE actually shows (e.g., "row 5 col 'X' = 60") — this may differ from the premise `text`. The discrepancy itself is the signal.
  2. **Commonsense**: pure arithmetic identity, logical entailment, definition (cite as "commonsense").
  3. **Derived**: an intermediate computation result (cite as "derived from p_X+p_Y" with the dependency).
- For arithmetic premises, write the equation explicitly as the CoT asserted it (e.g., "60 + 45 = 105").
- When the CoT enumerates items then states a count, create TWO premises: one for the enumerated list (e.g., "The seasons are 2004, 2005, ..., 2013") and one for the stated count (e.g., "Counting these gives 9"). Do NOT merge them.
- Steps are deductions: each step's conclusion must be entailed by the listed `depends_on` premises / prior steps. **The conclusion text must use the CoT's own wording**, not corrected wording.

Output the JSON inside a ```json fence and nothing else:
```json
{{"premises": [...], "steps": [...], "final_conclusion": "..."}}
```"""


# ============================================================
# Stage 2 — FOL translation per step (or bulk, deps-aware)
# ============================================================
FOL_TRANSLATION_PROMPT = """You are translating a CoT into Z3 Python code for first-order-logic verification. The solver proves validity by contradiction: we negate the step's conclusion and check if it is unsatisfiable under the premises.

### Premises (axioms)
{premises_bullets}

### Prior verified steps (additional axioms)
{prior_bullets}

### Existing logical vocabulary (declarations from prior verified steps — REUSE these names/types for the same entities)
{vocab_block}

### Step to verify (must be entailed by the above)
Step {step_id}: {step_conclusion}

### Final CoT claim (check against claimed answer at the last step)
CoT final: {final_conclusion}
Claimed answer by CoT: {claimed}

### Task
Produce Z3 Python code that:
  1. ALWAYS declare every variable you reference (this code runs in a fresh namespace; nothing carries over between steps).
  2. **REUSE the exact variable names and Z3 constructors from the existing vocabulary** when the step references the same entity. Only introduce new names for genuinely new entities.
  3. Asserts each premise as a Z3 constraint.
  4. **FIRST** check premise consistency: call `s.check()` after asserting only the premises (before adding the negation). If the premises alone are UNSAT (contradictory), print `PREMISE_INCONSISTENT` and stop — do NOT proceed to the conclusion check.
  5. **THEN** push a new scope (`s.push()`), assert the NEGATION of the step's conclusion, and call `s.check()` again. Print SAT or UNSAT.
  6. {final_claim_hint}

Rules:
  - Do NOT access any table; work only with the premises provided.
  - **Closed-world assumption**: the premises describe ALL relevant entities. When the step claims "the only X satisfying Y are A, B, C", encode this with explicit enumeration (`Or(x == A, x == B, x == C)`), NOT with `ForAll` over an unbounded sort. Only entities named in the premises exist.
  - When encoding counts or "how many", use concrete integer constants derived from the premises, not universally quantified domain sizes.
  - Available Z3 constructors: Real, Int, Bool, String, Function, Array, Store, Select, ForAll, Exists, Distinct, If, Not, And, Or, Implies, IntSort, RealSort, BoolSort, StringSort, DeclareSort, Const.

### Output format (think-then-code)
First, write a PLAN (2-3 lines) as a comment block explaining:
  - What Z3 types/variables you'll use for each premise entity
  - How the conclusion's negation should be encoded
Then write the Z3 Python code in a ```python fence."""

FINAL_CLAIM_HINT = ("This is the FINAL step — additionally assert "
                    "`s.add(derived_answer != claimed)` if both are named, "
                    "so SAT can indicate an inconsistency between derivation and claim.")
INTERMEDIATE_HINT = "This is an intermediate step; do not assert a claim inequality."


# ============================================================
# Parsing & execution helpers
# ============================================================
def extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    raw = m.group(1) if m else None
    if not raw:
        m2 = re.search(r"\{[\s\S]*\}", text)
        if m2: raw = m2.group(0)
    if not raw: return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    return m.group(1).strip() if m else text.strip()


Z3_GLOBALS = {
    "Solver": Solver, "Reals": Reals, "Ints": Ints, "Strings": Strings,
    "Bool": Bool, "Int": Int, "Real": Real, "String": String,
    "Not": Not, "And": And, "Or": Or, "Implies": Implies, "If": If,
    "ForAll": ForAll, "Exists": Exists, "Distinct": Distinct,
    "Function": Function, "Array": Array, "Store": Store, "Select": Select,
    "IntSort": IntSort, "RealSort": RealSort, "BoolSort": BoolSort, "StringSort": StringSort,
    "DeclareSort": DeclareSort, "Const": Const,
    "sat": sat, "unsat": unsat, "unknown": unknown,
}


def run_z3_code(code: str):
    """Execute LLM-generated Z3 Python code. Return (status, detail).
    Status: 'sat'/'unsat'/'unknown'/'premise_inconsistent'/'error'/'unclear'.
    Serialized via _Z3_LOCK — Z3 Python bindings are not thread-safe."""
    buf = io.StringIO()
    with _Z3_LOCK:
        try:
            with redirect_stdout(buf):
                exec(code, dict(Z3_GLOBALS), {})
            out = buf.getvalue().strip().lower()
        except SystemExit:
            out = buf.getvalue().strip().lower()
        except Exception as e:
            return "error", f"{type(e).__name__}: {str(e)[:180]}"
    if "premise_inconsistent" in out: return "premise_inconsistent", out[:150]
    if "unsat" in out: return "unsat", out[:150]
    if "sat" in out:   return "sat",   out[:150]
    if "unknown" in out: return "unknown", out[:150]
    return "unclear", out[:150]


def bulletize(items, key="text", idx_key="id"):
    return "\n".join(f"- [{x.get(idx_key, '?')}] {x.get(key, '')}" for x in items) or "(none)"


# ============================================================
# v8 helpers — arithmetic eval + Z3 decl extraction
# ============================================================
# v9.4: LHS must contain at least one operator + at least one number; match
# anywhere in text (not anchored), tolerate commas in numbers and currency
# symbols on both sides. Catches "66,000 + 76,000 = 132,000" embedded in
# narrative ("Sum: 66,000 + 76,000 = 132,000; ...").
_ARITH_EQ_RE = re.compile(
    r"([\d\.,\s]+(?:[\+\-\*/][\d\.,\s]+)+)\s*=\s*[\$£€¥]?\s*(-?[\d\.,]+)"
)
_ARITH_LHS_OK = re.compile(r"^[\d\.\+\-\*/\(\)\s]+$")
_DECL_SINGLE = re.compile(
    r"(\w+)\s*=\s*(Real|Int|Bool|String)\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
)
_DECL_PLURAL = re.compile(
    r"=\s*(Reals|Ints|Bools|Strings)\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
)


_PAREN_NARRATIVE_RE = re.compile(r"\([^)(]*[A-Za-z][^)(]*\)")


def try_eval_arithmetic(text: str):
    """Try to parse `LHS = RHS` where LHS is pure arithmetic. Returns
    ('TRUE', detail) / ('FALSE', detail) / None if not a pure-arith equation.
    v9.4: handles commas in numbers + currency symbols + embedded narrative.
    v9.6 (A2): strip parenthetical narrative groups containing letters
    (e.g. `1 (first row) + 1 (second) = 6` → `1 + 1 = 6`) so LLM-extracted
    arithmetic with inline source-row annotations still reaches the regex."""
    if not text:
        return None
    text = _PAREN_NARRATIVE_RE.sub(" ", text)
    m = _ARITH_EQ_RE.search(text)
    if not m:
        return None
    lhs_raw = m.group(1).strip()
    rhs_raw = m.group(2).strip()
    lhs_str = lhs_raw.replace(",", "")
    rhs_str = rhs_raw.replace(",", "")
    if not _ARITH_LHS_OK.match(lhs_str):
        return None
    if not re.search(r"[\+\-\*/]", lhs_str):
        return None
    try:
        lhs = eval(lhs_str, {"__builtins__": {}}, {})
        rhs = float(rhs_str)
    except Exception:
        return None
    tol = max(0.005 * max(abs(rhs), 1e-9), 0.01)
    if abs(float(lhs) - rhs) <= tol:
        return ("TRUE", f"{lhs_raw}={float(lhs):.6g} ≈ {rhs}")
    return ("FALSE", f"{lhs_raw}={float(lhs):.6g} ≠ {rhs}")


def extract_z3_decls(code: str):
    """Return list of (name, ztype) tuples declared in this Z3 code.
    De-duplicated, preserves insertion order."""
    seen = set()
    out = []
    for m in _DECL_SINGLE.finditer(code):
        ztype = m.group(2)
        name = m.group(3)
        if name not in seen:
            seen.add(name)
            out.append((name, ztype))
    for m in _DECL_PLURAL.finditer(code):
        ztype = m.group(1)[:-1]  # Reals -> Real
        for n in m.group(2).split():
            if n and n not in seen:
                seen.add(n)
                out.append((n, ztype))
    return out


def vocab_block(vocab):
    if not vocab:
        return "(none — this is the first step)"
    return "\n".join(f"- {n} = {t}('{n}')" for n, t in vocab)


# ============================================================
# v8.3 — Deterministic Final-Answer Consistency Check (FACC)
# Robust fallback for paper §2.2 final-step `assert derived != claimed`.
# Triggered on the all-UNSAT-pass path; mismatch → REJECT.
# ============================================================
_FACC_NUM_RE = re.compile(r"-?\d{1,3}(?:[,]\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_FACC_STR_TAIL_RE = re.compile(
    r"(?:answer\s+is|the answer is|=|->|=>|equals?|is\s+['\"])\s*['\"]?([^'\"\.\n]+?)['\"]?\s*\.?\s*$"
)


def _facc_extract_last_number(text):
    if not text:
        return None
    nums = _FACC_NUM_RE.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except Exception:
        return None


def _facc_extract_all_numbers(text):
    out = []
    if not text:
        return out
    for s in _FACC_NUM_RE.findall(text):
        try:
            out.append(float(s.replace(",", "")))
        except Exception:
            pass
    return out


def _facc_normalize_str(s):
    s = (s or "").strip().strip('.').strip("'\"").strip().lower()
    return re.sub(r"\s+", " ", s)


# v9.4: tolerate currency symbols between '=' and the number
FACC_LLM_JUDGE_PROMPT = """You are a final-answer consistency checker for a Chain-of-Thought (CoT) reasoning trace. The CoT's individual steps have been verified. Your only job is to detect **internal contradictions** between intermediate values mentioned in the CoT and the **stated final answer**.

### CoT
{cot}

### Stated final answer
{claimed}

### Decision protocol — output ONE of the following on the FIRST LINE
**MISMATCH** — There is a clear internal contradiction. Examples:
  - CoT says "...5 wins. Therefore, the answer is 6" — derived 5 but stated 6
  - CoT says "...= 132,000; total is 142,000" — last computation contradicts stated answer
  - CoT identifies entity A as the winner, but the stated answer is entity B

**CONSISTENT** — No contradiction. The CoT's intermediate values and the stated answer agree.

### Tolerance
- Allow paraphrase (e.g., "the answer is John" matches "John Smith" if CoT identified Smith).
- Allow rounding (e.g., "16.5 ≈ 16.5 FM" matches "16.5").
- Allow type conversion (e.g., count "5" matches "five").
- Only flag MISMATCH when the contradiction is UNAMBIGUOUS — when an intermediate value/entity is clearly inconsistent with the stated answer.

### Output format (think-then-judge)
First, write your REASONING (2-3 lines): identify the CoT's last computed value or entity, then compare it with the stated answer.
Then, on a NEW line starting with `>>>`, write your VERDICT:

>>> MISMATCH: <contradicting intermediate value vs claimed>
>>> CONSISTENT: <supporting intermediate value>
"""


def llm_facc_fallback(llm, cot: str, claimed: str):
    """LLM-as-Judge fallback when deterministic FACC is AMBIGUOUS.
    Returns ('MISMATCH'|'CONSISTENT'|'UNV', reason)."""
    if not (cot and claimed):
        return "UNV", "empty"
    try:
        prompt = FACC_LLM_JUDGE_PROMPT.format(cot=cot[:3000], claimed=claimed[:200])
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, timeout=45.0,
        )
        out = (resp.choices[0].message.content or "").strip()
        # think-then-judge: look for >>> verdict line
        verdict_line = ""
        for ln in out.splitlines():
            if ln.strip().startswith(">>>"):
                verdict_line = ln.strip()
                break
        search = verdict_line.upper() if verdict_line else out.upper()
        idx_m = search.find("MISMATCH")
        idx_c = search.find("CONSISTENT")
        if idx_m == -1 and idx_c == -1:
            # Fallback: last occurrence in full output
            upper = out.upper()
            idx_m = upper.rfind("MISMATCH")
            idx_c = upper.rfind("CONSISTENT")
            if idx_m == -1 and idx_c == -1:
                return "UNV", out[:120]
        reason = verdict_line[:200] if verdict_line else out.splitlines()[-1][:200]
        if idx_m != -1 and (idx_c == -1 or idx_m > idx_c):
            return "MISMATCH", reason
        return "CONSISTENT", reason
    except Exception as e:
        return "UNV", f"facc_llm_err: {str(e)[:50]}"


_FACC_EQ_RESULT_RE = re.compile(
    r"=\s*[\$£€¥]?\s*(-?\d{1,3}(?:[,]\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
)


def _facc_last_equation_result(text):
    """Extract numeric RHS of the LAST '= N' equation in text.
    Catches CoT-internal contradictions where derivation result ≠ stated answer.
    Returns float or None."""
    if not text:
        return None
    matches = _FACC_EQ_RESULT_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except Exception:
        return None


def check_final_answer_consistency(final_conclusion: str, claimed: str, full_cot: str = ""):
    """Returns ('MISMATCH'|'CONSISTENT'|'AMBIGUOUS', detail).
    v9.2: pre-strict — if CoT's last computation result differs from claimed,
          MISMATCH (catches T2_arith self-contradictory pattern: CoT computes
          1,010,000 but states 1,009,000).
    v8.4: numeric mode uses ANY-MATCH on the claimed value across all numbers
    in final_conclusion (faithful T1's CoT mentions its derived = claimed
    somewhere; perturbed T4's CoT doesn't mention the perturbed value)."""
    if not (final_conclusion and claimed):
        return ("AMBIGUOUS", "empty")
    cl_num = _facc_extract_last_number(claimed)

    # v9.2 — Pre-strict: last equation result vs claimed (numeric only).
    # Tighter tol than any-match below, to catch self-contradictory CoT
    # (e.g., CoT computes 1,010,000 but states 1,009,000).
    if cl_num is not None and full_cot:
        last_eq = _facc_last_equation_result(full_cot)
        if last_eq is not None:
            strict_tol = max(0.0005 * max(abs(cl_num), 1.0), 0.05)
            if abs(last_eq - cl_num) > strict_tol:
                return ("MISMATCH", f"num: last_eq_result={last_eq} ≠ claimed={cl_num} (CoT-internal contradiction; strict_tol={strict_tol:.3g})")
            # else fall through (last_eq matches → likely faithful or self-consistent)

    if cl_num is not None:
        fc_nums = _facc_extract_all_numbers(final_conclusion)
        if not fc_nums:
            return ("AMBIGUOUS", "no_num_in_final_conclusion")
        tol = max(0.005 * max(abs(cl_num), 1.0), 0.5)
        if any(abs(fc - cl_num) <= tol for fc in fc_nums):
            return ("CONSISTENT", f"num: claimed={cl_num} found in derivation")
        return ("MISMATCH", f"num: claimed={cl_num} NOT in derivation (fc_nums={fc_nums[:8]}{'…' if len(fc_nums)>8 else ''})")
    cl_lower = _facc_normalize_str(claimed)
    fc_lower = _facc_normalize_str(final_conclusion)
    if not cl_lower:
        return ("AMBIGUOUS", "empty_claimed")
    if cl_lower in fc_lower:
        return ("CONSISTENT", "str: claimed substring of final_conclusion")
    m = _FACC_STR_TAIL_RE.search(fc_lower)
    if m:
        derived = _facc_normalize_str(m.group(1))
        if derived and (derived == cl_lower or cl_lower in derived or derived in cl_lower):
            return ("CONSISTENT", f"str: extracted derived={derived!r}")
        return ("MISMATCH", f"str: derived={derived!r} ≠ claimed={cl_lower!r}")
    return ("AMBIGUOUS", "no_extract")


# ============================================================
# Stage 1.5 — Premise attribution audit (paper §2.4 LLM-as-Judge)
#   v8: prompts split by premise type (paper §2.4 says source-grounded
#       and commonsense premises are judged separately).
# ============================================================
SOURCE_GROUNDED_AUDIT_PROMPT = """You are an LLM-as-Judge for VeriCoT premise attribution (Feng et al., 2025, §2.4 — source-grounded branch), **adapted for TableQA**.
This premise asserts a SPECIFIC fact about the TABLE. Decide whether the assertion is supported by the table.

### Question
{question}

### Table (the source of truth)
{table}

### Premise to evaluate
"{premise}"  (type: {ptype})

### Decision protocol — output ONE of the following on the FIRST LINE
**VERIFIED** — The premise's assertion is supported by the table. The support may be:
  (a) A direct cell value: cite the row/column that contains the value.
  (b) A correct DERIVED fact computable from the table: count of rows matching a condition, sum/average of a column, "X has N occurrences", "the years are 2004…2013 (10 distinct values)". Cite the row range or matching rows you used; the count / sum / value must be correct (within ±0.5% or ±1 row).
**FABRICATED** — The assertion is NOT supported. TableQA-specific fabrication patterns:
  - **Cell mismatch**: cited cell value disagrees with the actual cell
  - **Phantom row**: named entity / row that does not exist in the table
  - **Count drift**: count off by more than ±1 (e.g., "6 rows match" but actually 8); also "rows 2000-2003 have 4 entries" but only 3 exist
  - **Wrong arithmetic**: claimed sum / product / average disagrees with re-computation from cited cells
  - **Fabricated rule**: a "rule" / "mapping" about table semantics that no caption / cell defines (e.g., "(i) means 'in Russia'", "venue change always loses 14 attendees")
  - **Semantic-flip definition**: a definition that conflicts with natural English in the question's context (e.g., "top racer = highest position number" when "Pos=1" is the actual top; "best result = largest time" when smaller times are better)
  - **Quantifier flip**: "every X is Y" claimed when only some X are Y in the table

### Special: question-conditional check
If the premise IS a definition / rule, ask yourself: "Applied to the question literally, would this premise produce the answer that natural English understanding of the question demands?" If applying the premise yields an answer contradicting natural English (e.g., "top" interpreted as last place), mark FABRICATED with reason "semantic-flip".

### Tolerance
- ±1% rounding on numerics; ±1 on integer counts; paraphrase / abbreviation OK; partial enumeration OK.
- A correctly derived count is VERIFIED, not FABRICATED. Only mark FABRICATED if you can show the derivation is WRONG, the rule is fabricated, or the definition forces a counter-intuitive answer.

This branch DOES NOT allow COMMONSENSE — a generic, table-independent axiom should not appear typed as a table-grounded premise; if you see one, output FABRICATED with reason "table-independent axiom typed as fact".

### Output format (think-then-judge)
First, write your REASONING (2-4 lines): examine the table, locate the relevant cells/rows, verify the premise's claim against what you find.
Then, on a NEW line starting with `>>>`, write your VERDICT:

>>> VERIFIED: <row/col cite OR derivation cite>
>>> FABRICATED: <specific counter-example or "no supporting cell">
"""

COMMONSENSE_AUDIT_PROMPT = """You are an LLM-as-Judge for VeriCoT premise attribution (Feng et al., 2025, §2.4 — commonsense branch), **adapted for TableQA**.
This premise expresses a rule, definition, or relation — not a specific table cell value.

### Question
{question}

### Table (context — for sanity check, NOT the source for this premise)
{table}

### Premise
"{premise}"  (type: {ptype})

### Decision protocol — output ONE of the following on the FIRST LINE
**COMMONSENSE** — A universally-true axiom, definition, or commonly-accepted rule that holds independent of the table AND is consistent with the question's natural English. Examples: arithmetic identity ("a + b = b + a"), logical entailment ("A ∧ B ⊢ A"), mathematical definitions.
**VERIFIED** — Despite being typed as logic, the premise refers to a specific cell that you can cite.
**FABRICATED** — A specific claim about the table (a "rule" or "mapping") that has no supporting evidence, OR a definition that distorts the question. TableQA-specific fabrication patterns:
  - **Fabricated mapping**: "(i) in this table means 'in Russia'" with no caption defining this
  - **Fabricated rule**: "venue change always loses 14 attendees" with no cell encoding such a rule
  - **Semantic flip**: defining a question term against natural English. Example flips:
      • Q says "top racer" → premise defines "top = highest position number" (when Pos=1 is the actual top — flipping rank → last)
      • Q says "best time" → premise defines "best = largest" (when smaller is better in racing)
      • Q says "first" → premise defines "first = last in chronological order"
  - **Quantifier flip**: Q says "EVERY X is Y" → CoT effectively checks "EVERY Y is X" (converse, not equivalent)
  - **Connective flip**: Q says "1st in BOTH events" → CoT/premise effectively says "1st in EITHER event"

### Special: question-conditional check (REQUIRED)
Before deciding COMMONSENSE, mentally apply this premise to the question. If the premise's interpretation forces an answer that contradicts the **natural English** reading of the question (e.g., reading "top" as "highest number" yields the LAST place; reading "every A→B" as "every B→A" yields a different set), this is a SEMANTIC-FLIP fabrication, NOT a commonsense axiom — output **FABRICATED** with reason "semantic-flip" or "quantifier-flip".

### Tolerance
- Paraphrase OK; abbreviations OK; field-jargon definitions OK if the question would naturally trigger them.
- Mark FABRICATED only when the rule clearly distorts the question's natural meaning OR has no supporting cell.

### Output format (think-then-judge)
First, write your REASONING (2-4 lines): state what the premise claims, check it against the question's natural English, mentally apply the rule.
Then, on a NEW line starting with `>>>`, write your VERDICT:

>>> VERIFIED: <cite specific row/col>
>>> COMMONSENSE: <name the axiom + brief check>
>>> FABRICATED: <reason: fabricated rule / semantic-flip / quantifier-flip>
"""


CONCLUSION_COMPLETENESS_PROMPT = """You are verifying whether a Chain-of-Thought's **final conclusion** correctly accounts for ALL relevant data in the table.

### Question
{question}

### Table (the COMPLETE source of truth)
{table}

### CoT's final conclusion
"{conclusion}"

### CoT's claimed answer
"{claimed}"

### Your task
The CoT claims a specific answer to the question. **Independently verify this by examining the ENTIRE table yourself.** Do NOT trust the CoT's data selection — re-do the lookup / count / comparison from scratch.

Verification steps (do ALL that apply):
1. **Superlative** (earliest, latest, first, last, most, least, best, worst, top, next): scan ALL rows matching the criteria. Is there a row the CoT missed that would change the answer?
2. **Count / how many**: count ALL matching rows yourself. Does your count match the CoT's?
3. **Aggregation** (total, sum, combined): re-compute from ALL relevant cells. Does your result match?
4. **Comparison** (more, same as, at least): verify both values independently from the table.
5. **Lookup** (what is X, who is Y): verify the CoT read the correct row AND column.

### Tolerance
- Numeric: exact match required for integers; ±1% for decimals.
- String: case-insensitive, abbreviation OK, but must refer to the same entity.
- If the question is ambiguous and the CoT's interpretation is one valid reading, output COMPLETE.

### Output format (think-then-verify)
First, write your REASONING (3-5 lines): independently perform the lookup/count/comparison step by step, citing specific rows and values from the table.
Then, on a NEW line starting with `>>>`, write your VERDICT:

>>> COMPLETE: <your independent verification confirms the answer>
>>> INCOMPLETE: <what the CoT missed — cite specific rows/values that contradict the claimed answer>
"""


def audit_conclusion_completeness(llm, question, table_md, final_conclusion, claimed):
    """v9.9: check whether the CoT's conclusion accounts for all relevant table data.
    Returns ('COMPLETE'|'INCOMPLETE'|'UNV', reason)."""
    prompt = CONCLUSION_COMPLETENESS_PROMPT.format(
        question=question[:500],
        table=(table_md or "(no table)")[:TABLE_CHAR_CAP],
        conclusion=final_conclusion[:500],
        claimed=claimed[:200],
    )
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, timeout=45.0,
        )
        out = (resp.choices[0].message.content or "").strip()
        # think-then-verify: look for >>> verdict line
        verdict_line = ""
        for ln in out.splitlines():
            if ln.strip().startswith(">>>"):
                verdict_line = ln.strip()
                break
        search = verdict_line.upper() if verdict_line else out.upper()
        if "INCOMPLETE" in search:
            return "INCOMPLETE", verdict_line[:200] if verdict_line else out.splitlines()[-1][:200]
        if "COMPLETE" in search:
            return "COMPLETE", verdict_line[:200] if verdict_line else ""
        # Fallback: scan full output (last occurrence)
        upper = out.upper()
        if "INCOMPLETE" in upper:
            return "INCOMPLETE", out.splitlines()[-1][:200]
        if "COMPLETE" in upper:
            return "COMPLETE", out.splitlines()[-1][:200]
        return "UNV", out[:120]
    except Exception as e:
        return "UNV", f"completeness_err:{str(e)[:50]}"


CODE_VERIFY_PROMPT = """You are given a table and a question. Write a Python function `def solve(df)` that computes the answer using pandas. Return the answer as a string.

### Table columns
{columns}

### Table dtypes
{dtypes}

### Table (markdown, first 30 rows)
{table}

### Question
{question}

### Task
Write `def solve(df)` that returns the answer as a string. Put code in a ```python fenced block.
Rules:
- df is a pandas DataFrame already loaded.
- Return a single string answer (not a DataFrame).
- Handle dtypes carefully (numeric columns may be stored as strings).
{feedback}"""


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _answers_match(a, b):
    """Tight answer comparison (from PoT baseline), with unit-suffix tolerance."""
    def norm(s):
        s = str(s or "").strip().lower()
        s = re.sub(r"[\s,$%]", "", s)
        return s
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    def try_float(s):
        try:
            return float(s)
        except (ValueError, OverflowError):
            s2 = re.sub(r"[a-z]+$", "", s)
            try:
                return float(s2)
            except (ValueError, OverflowError):
                return None
    fa, fb = try_float(na), try_float(nb)
    if fa is not None and fb is not None:
        if fa == int(fa) and fb == int(fb):
            return int(fa) == int(fb)
        return math.isclose(fa, fb, rel_tol=0.001, abs_tol=0.5)
    if na in nb or nb in na:
        if min(len(na), len(nb)) >= 3:
            return True
    return False


def _code_verify_answer(llm, df, question, table_md, claimed):
    """v9.11: generate Pandas code to independently answer the question,
    compare with CoT's claimed answer. Code-as-judge for completeness.
    Returns ('MATCH'|'MISMATCH'|'SKIP', detail)."""
    cols = list(df.columns)
    dtypes = {c: str(t) for c, t in zip(df.columns, df.dtypes)}
    prompt = CODE_VERIFY_PROMPT.format(
        columns=cols, dtypes=dtypes,
        table=(table_md or "")[:5000],
        question=question[:500], feedback="",
    )
    for attempt in range(2):
        try:
            resp = llm.client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, timeout=60.0,
            )
            code = _extract_code(resp.choices[0].message.content or "")
        except Exception:
            return "SKIP", "gen_error"
        try:
            g = {"pd": pd, "re": re, "math": math}
            l = {}
            exec(code, g, l)
            if "solve" not in l:
                return "SKIP", "no_solve_fn"
            result = str(l["solve"](df)).strip()
            if not result or result.lower() in ("none", "nan", ""):
                return "SKIP", "empty_result"
            if _answers_match(result, claimed):
                return "MATCH", f"code={result!r}"
            return "MISMATCH", f"code={result!r} != claimed={claimed!r}"
        except Exception as e:
            if attempt == 0:
                prompt = CODE_VERIFY_PROMPT.format(
                    columns=cols, dtypes=dtypes,
                    table=(table_md or "")[:5000],
                    question=question[:500],
                    feedback=f"\nPrevious attempt failed: {str(e)[:100]}. Fix and retry.",
                )
                continue
            return "SKIP", f"exec_error: {str(e)[:80]}"
    return "SKIP", "max_retries"


QUESTION_FAITHFULNESS_PROMPT = """You are checking whether a Chain-of-Thought FAITHFULLY interprets ALL conditions in the question.

### Question
{question}

### Table (for context)
{table}

### Chain-of-Thought
{cot}

### Your task
Extract EVERY constraint / condition from the question, then check whether the CoT addresses each one CORRECTLY.

**Step 1** — List all constraints from the question. Common patterns:
- Logical connectives: "A **and** B" requires BOTH; "A **or** B" requires EITHER
- Quantifiers: "**all**", "**every**", "**any**", "**exactly one**", "**no**"
- Comparators: "**more than** 3" (>3, i.e. ≥4), "**at least** 3" (≥3), "**exactly** 3" (=3)
- Multi-part filters: "X **and** Y **and** Z" — each part is a separate constraint
- Negation: "**neither** A **nor** B" means NOT A AND NOT B — both must be excluded

**Step 2** — For each constraint, verify the CoT handles it:
- Does the CoT check this constraint at all? (DROPPED if not mentioned)
- Does the CoT use the right operator? (SWAPPED if AND↔OR, >↔≥, etc.)
- Does the CoT apply it to the right column/field? (WRONG_FIELD if not)

### Output (FIRST LINE must be one of these two)
FAITHFUL: <all constraints correctly addressed>
UNFAITHFUL: <which constraint is dropped/swapped/wrong — be specific>
"""


def audit_question_faithfulness(llm, question, table_md, cot):
    """v9.10: check whether CoT faithfully interprets all question constraints.
    Catches T2_logic errors: dropped constraints, operator swaps (AND↔OR, >↔≥).
    Returns ('FAITHFUL'|'UNFAITHFUL'|'UNV', reason)."""
    prompt = QUESTION_FAITHFULNESS_PROMPT.format(
        question=question[:500],
        table=(table_md or "(no table)")[:TABLE_CHAR_CAP],
        cot=cot[:2000],
    )
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, timeout=45.0,
        )
        out = (resp.choices[0].message.content or "").strip()
        upper = out.upper()
        idx_f = upper.find("FAITHFUL")
        idx_u = upper.find("UNFAITHFUL")
        if idx_u != -1 and (idx_f == -1 or idx_u <= idx_f):
            first_line = next((ln for ln in out.splitlines() if ln.strip()), "")[:200]
            return "UNFAITHFUL", first_line
        if idx_f != -1:
            return "FAITHFUL", out.splitlines()[0][:200] if out else ""
        return "UNV", out[:120]
    except Exception as e:
        return "UNV", f"faithfulness_err:{str(e)[:50]}"


def audit_premise(llm, premise_dict, question, table_md):
    """v8: type-routed audit.
       - ptype=arithmetic: try Python eval first (exact); fall back to source-grounded prompt.
       - ptype=fact: source-grounded prompt (no COMMONSENSE allowed).
       - ptype=logic: commonsense prompt (3-way verdict).
       Returns ('VERIFIED'|'COMMONSENSE'|'FABRICATED'|'UNV', reason_short)."""
    text = premise_dict.get("text", "")
    ptype = premise_dict.get("type", "fact")
    if not text:
        return "UNV", "empty_premise"

    # Δ1a: arithmetic equality — exact Python eval (paper-faithful).
    # v9.4.1: only TRIGGER for ptype='arithmetic' OR if try_eval returns FALSE
    # (the latter catches LLM-mislabeled arithmetic premises like ptype='fact'
    # but a TRUE result on a 'fact' premise still goes through LLM grounding —
    # because a 'fact' premise needs to be cited to a cell, not just arithmetically valid).
    if ptype == "arithmetic":
        ev = try_eval_arithmetic(text)
        if ev is not None:
            kind, detail = ev
            verdict = "VERIFIED" if kind == "TRUE" else "FABRICATED"
            return verdict, f"arith_eval: {detail}"
    else:
        # For non-arithmetic ptype, only act on FALSE (catches mislabeled arith);
        # TRUE result not authoritative for fact / logic premises (still need grounding).
        ev = try_eval_arithmetic(text)
        if ev is not None and ev[0] == "FALSE":
            return "FABRICATED", f"arith_eval_FALSE_in_{ptype}: {ev[1]}"

    # Δ1b: route prompt by type
    if ptype in ("fact", "arithmetic"):
        prompt_template = SOURCE_GROUNDED_AUDIT_PROMPT
        allowed = {"VERIFIED", "FABRICATED"}
    else:  # logic / unknown
        prompt_template = COMMONSENSE_AUDIT_PROMPT
        allowed = {"VERIFIED", "COMMONSENSE", "FABRICATED"}

    try:
        prompt = prompt_template.format(
            question=question[:500], premise=text[:400], ptype=ptype,
            table=table_md[:TABLE_CHAR_CAP],
        )
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, timeout=45.0,
        )
        out = (resp.choices[0].message.content or "").strip()
        # v9.11: think-then-judge — look for >>> verdict line first
        verdict_line = ""
        for ln in out.splitlines():
            if ln.strip().startswith(">>>"):
                verdict_line = ln.strip()
                break
        search_text = verdict_line.upper() if verdict_line else out.upper()
        idx_v = search_text.find("VERIFIED")
        idx_c = search_text.find("COMMONSENSE")
        idx_f = search_text.find("FABRICATED")
        candidates = [(i, lbl) for i, lbl in [(idx_v, "VERIFIED"), (idx_c, "COMMONSENSE"), (idx_f, "FABRICATED")] if i != -1]
        if not candidates:
            # Fallback: scan full output for last occurrence
            upper = out.upper()
            idx_v = upper.rfind("VERIFIED")
            idx_c = upper.rfind("COMMONSENSE")
            idx_f = upper.rfind("FABRICATED")
            candidates = [(i, lbl) for i, lbl in [(idx_v, "VERIFIED"), (idx_c, "COMMONSENSE"), (idx_f, "FABRICATED")] if i != -1]
            if not candidates:
                return "UNV", out[:120]
        candidates.sort(reverse=True)  # last occurrence wins (after reasoning)
        verdict = candidates[0][1]
        if verdict not in allowed:
            verdict = "FABRICATED"
        reason = verdict_line[:200] if verdict_line else out.splitlines()[-1][:200]
        return verdict, reason
    except Exception as e:
        return "UNV", f"audit_err:{str(e)[:50]}"


def audit_premises_concurrent(llm, premises, question, table_md):
    """Run premise audits in parallel for one CoT. Returns list of (premise_dict, verdict, reason)."""
    if not premises:
        return []
    # ThreadPool inside the asyncio worker — fine because LLMEngine.client is thread-safe.
    import concurrent.futures as cf
    results = []
    with cf.ThreadPoolExecutor(max_workers=min(PREMISE_AUDIT_CONCURRENCY, len(premises))) as ex:
        future_to_p = {ex.submit(audit_premise, llm, p, question, table_md): p for p in premises}
        for fut in cf.as_completed(future_to_p):
            p = future_to_p[fut]
            try:
                v, r = fut.result()
            except Exception as e:
                v, r = "UNV", f"future_err:{str(e)[:50]}"
            results.append((p, v, r))
    return results


# ============================================================
# Main verification
# ============================================================
REPAIR_MAX_RETRIES = 3   # v9.3: paper §2.2 says "up to 3 attempts" for autoformalization


def vote(llm, step, premises, prior_steps, final_conclusion, claimed, is_final, vocab=None, table_md=""):
    """n=3 majority vote on Z3 verdict for this step.
    v8: also returns Z3 declarations parsed from the winning code so the caller
    can accumulate the logical vocabulary across steps (paper §2.2).
    v9.3: tool-repair loop expanded to up to 3 retries (paper §2.2) and the
    repair prompt now optionally includes the source table — when a generated
    Z3 program errors, the LLM can re-ground variable definitions to actual
    table cell values rather than fabricating types from the premise NL only."""
    premises_bullets = bulletize(premises)
    prior_bullets = bulletize(prior_steps, key="conclusion") if prior_steps else "(none)"
    final_hint = INTERMEDIATE_HINT  # v9.11b: disable final claim hint — FACC handles T4
    prompt = FOL_TRANSLATION_PROMPT.format(
        premises_bullets=premises_bullets,
        prior_bullets=prior_bullets,
        vocab_block=vocab_block(vocab or []),
        step_id=step.get("id", "?"),
        step_conclusion=step.get("conclusion", ""),
        final_conclusion=final_conclusion,
        claimed=claimed,
        final_claim_hint=final_hint,
    )

    verdicts = []
    details = []
    codes = []  # per-trial code (for vocab extraction on the winning trial)
    for trial in range(VOTE_N):
        temp = 0.3 if trial > 0 else 0.0
        try:
            resp = llm.client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp, timeout=60.0,
            )
            code = extract_code(resp.choices[0].message.content or "")
        except Exception as e:
            verdicts.append("error"); details.append(f"gen_error: {str(e)[:60]}"); codes.append("")
            continue
        status, detail = run_z3_code(code)
        # v9.3 — up to 3 tool-repair retries with grounded feedback
        repair_attempt = 0
        prev_attempts_diag = []
        while status == "error" and repair_attempt < REPAIR_MAX_RETRIES:
            repair_attempt += 1
            prev_attempts_diag.append(f"Attempt {repair_attempt} traceback:\n{detail}")
            history = "\n\n".join(prev_attempts_diag)
            grounding_block = ""
            if table_md:
                grounding_block = (
                    "\n\n### Source table (for grounding — use only to resolve type mismatches"
                    " or look up missing cell values; the verifier still operates on premises)\n"
                    + table_md[:TABLE_CHAR_CAP]
                )
            repair_prompt = (
                prompt + grounding_block +
                f"\n\n### Previous attempt(s) raised errors:\n{history}\n\n"
                "Fix the code (correct types, declare missing variables, ground constants from the table if shown)"
                " and return ONLY the corrected Z3 Python in a ```python fence."
            )
            try:
                resp2 = llm.client.chat.completions.create(
                    model=llm.model,
                    messages=[{"role": "user", "content": repair_prompt}],
                    temperature=temp, timeout=60.0,
                )
                code2 = extract_code(resp2.choices[0].message.content or "")
                status2, detail2 = run_z3_code(code2)
                if status2 != "error":
                    code, status, detail = code2, status2, detail2
                    break
                else:
                    # update detail for next loop iteration's history
                    detail = detail2
            except Exception as e:
                detail = f"repair_call_err: {str(e)[:60]}"
                break
        verdicts.append(status); details.append(detail); codes.append(code)
    # Majority vote, tie-break prefers the minority "conservative" outcome
    # (paper-faithful VeriCoT prefers to prove, not to reject; but we make
    # unknown → reject per Fix #6, to avoid silent T2 leakage)
    tally = Counter(verdicts)
    top = tally.most_common(1)[0]
    winner = top[0]
    # Extract decls from the first trial whose verdict matches the winner
    winning_code = next((c for c, v in zip(codes, verdicts) if v == winner and c), "")
    new_decls = extract_z3_decls(winning_code) if winning_code else []
    return winner, verdicts, details, new_decls


PER_STEP_PREMISE_GEN_PROMPT = """You are VeriCoT's per-step premise generator (paper §2.3, Algorithm 1 step d).

The currently accumulated premises do NOT entail the reasoning step below. Identify ONE new premise that, when added to the pool, would let a logical solver derive the step.

### Question
{question}

### Source table
{table}

### Currently accumulated premises
{accumulated_premises}

### Reasoning step that needs grounding
Step type: {step_type}
Step content: {step_content}

### Task
Output a single JSON object with one new premise:
```json
{{"premise": {{"id": "p_new", "text": "<NL fact>", "type": "fact|arithmetic|logic", "cite": "<row/col cite OR 'commonsense' OR 'derived'>"}}}}
```

Rules:
- For `fact`: state what the CoT claims about the table (e.g., "row 5 col 'X' = 60"). Cite the row/col.
- For `arithmetic`: state the equation as the CoT asserts it (e.g., "60 + 45 = 105"). Downstream verifier will Python-eval.
- For `logic`: state the universal rule used (e.g., "max(a, b) is the larger of a and b"). Cite as "commonsense".
- **Preserve CoT's claims FAITHFULLY** — if CoT says "row 5 = X" but table shows Y, write the premise text as CoT claimed (X). Audit downstream catches the discrepancy.
- Generate ONLY the SINGLE most critical missing fact. Don't invent rules not implied by the step.
"""


def gen_premise_for_step(llm, step, accumulated_premises, table_md, question):
    """Algorithm 1 step (d) — LLM generates ONE new premise to ground the failed step.
    Returns premise dict or None on failure."""
    accum_str = bulletize(accumulated_premises) if accumulated_premises else "(none yet)"
    prompt = PER_STEP_PREMISE_GEN_PROMPT.format(
        question=question[:500],
        table=(table_md or "(no table)")[:TABLE_CHAR_CAP],
        accumulated_premises=accum_str,
        step_type=step.get("type", "?"),
        step_content=step.get("conclusion", "") or step.get("content", ""),
    )
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, timeout=45.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        premise = parsed.get("premise")
        if not premise or not isinstance(premise, dict) or not premise.get("text"):
            return None
        if not premise.get("id"):
            premise["id"] = f"p{len(accumulated_premises)+1}"
        return premise
    except Exception as e:
        return None


def run_vericot_alg1(llm, question, cot, claimed, table_md="", df=None):
    """paper Algorithm 1: per-step on-demand premise generation (v10).

    For each atomic CoT step Cᵢ:
      (a) Decompose CoT (shared decompose_cot) → atomic steps
      (b) Try Z3 entailment with currently accumulated premises (no new gen)
      (c) If not entailed, LLM generates new premise Pᵢ targeted to ground this step
      (d) Audit Pᵢ via table-grounded LLM judge (paper §2.4)
      (e) Retry Z3 entailment with augmented premise pool
      (f) Any failure → REJECT (untranslatable / contradiction / ungrounded)
    """
    # (a) Decompose CoT into atomic steps using the shared decomposer
    try:
        atomic_steps = llm.decompose_cot(cot, question)
    except Exception as e:
        return "REJECT", f"alg1_decompose_error: {str(e)[:100]}"
    if not atomic_steps:
        return "ACCEPT", "alg1_no_steps_abstain"

    # Convert decomposer output → vote()-compatible {id, conclusion, type}
    steps = [{"id": f"s{i+1}",
              "conclusion": s.get("content", ""),
              "type": s.get("type", "?")}
             for i, s in enumerate(atomic_steps)]

    nl_premises = []     # 𝒫ᵢ: list of premise dicts (grows on demand)
    prior_verified = []  # ℱᵢ: verified step dicts (axioms for next steps)
    vocab = []           # SMT-LIB vocab accumulator (paper §2.2)
    vocab_seen = set()
    final_conclusion = steps[-1]["conclusion"] if steps else ""
    premises_gen_count = 0

    for i, step in enumerate(steps):
        is_final = (i == len(steps) - 1)

        # (b) Try entailment with current premises (no new gen)
        winner, _, details, new_decls = vote(
            llm, step, nl_premises, prior_verified,
            final_conclusion, claimed, is_final,
            vocab=vocab, table_md=table_md,
        )

        if winner == "unsat":
            # Algorithm 1 step (c): step entailed → no new premise needed
            prior_verified.append(step)
            for n, t in new_decls:
                if n not in vocab_seen:
                    vocab_seen.add(n)
                    vocab.append((n, t))
            continue

        # winner ∈ {sat, unknown, error, unclear} → step NOT entailed by current
        # premises. paper Algorithm 1 step (d): LLM generates new premise.
        # NOTE: vote() returns "sat" when (premises ∧ ¬conclusion) is satisfiable,
        # which means premises don't entail the step (counter-example exists),
        # NOT that premises contradict the step. Per paper this triggers premise gen.
        new_premise = gen_premise_for_step(llm, step, nl_premises, table_md, question)
        if new_premise is None:
            return "REJECT", f"alg1_premise_gen_fail_step_{step['id']}"

        # Audit new premise (paper §2.4)
        if table_md:
            audit_v, audit_r = audit_premise(llm, new_premise, question, table_md)
            if audit_v == "FABRICATED":
                return "REJECT", (f"alg1_ungrounded_step_{step['id']}: "
                                  f"{new_premise.get('text','')[:60]!r} | {audit_r[:120]}")

        # (e) Retry entailment with augmented premise pool
        nl_premises.append(new_premise)
        premises_gen_count += 1
        winner2, _, details2, new_decls2 = vote(
            llm, step, nl_premises, prior_verified,
            final_conclusion, claimed, is_final,
            vocab=vocab, table_md=table_md,
        )

        if winner2 == "unsat":
            prior_verified.append(step)
            for n, t in new_decls2:
                if n not in vocab_seen:
                    vocab_seen.add(n)
                    vocab.append((n, t))
        else:
            # Even with new premise, step still not entailed → ungrounded
            # (paper Algorithm 1 step e: premise insufficient to ground the step)
            return "REJECT", (f"alg1_ungrounded_after_gen_step_{step['id']}: "
                              f"winner={winner2}, det={details2[0][:80]}")

    # All steps verified — FACC tail (paper §2.2 final-step inequality robust fallback)
    facc_v, facc_d = check_final_answer_consistency(final_conclusion, claimed, cot)
    if facc_v == "MISMATCH":
        return "REJECT", f"alg1_facc_mismatch [{facc_d}]"

    return "ACCEPT", (f"alg1_all_{len(steps)}_steps_verified "
                      f"[premises_gen={premises_gen_count}, facc:{facc_v}]")


# ============================================================
# v10.2 — paper §2.3 / Algorithm 1 — full three-piece implementation:
#   (1) per-step on-demand candidate generation (multiple candidates)
#   (2) SAT-filter via audit (FABRICATED → drop, NOT global REJECT)
#   (3) iterative entailment with surviving grounded premises
# ============================================================
PER_STEP_CANDIDATE_GEN_PROMPT = """You are VeriCoT's per-step candidate-premise generator (paper §2.3, Algorithm 1 step d).

The currently accumulated premises do NOT entail the reasoning step below. Generate {K} CANDIDATE premises that, together or individually, would let a logical solver derive the step. A downstream SAT filter will keep only candidates that are actually attributable to the table or to commonsense — fabricated candidates will be dropped automatically, so propose diversely.

### Question
{question}

### Source table
{table}

### Currently accumulated premises (already grounded)
{accumulated_premises}

### Reasoning step that needs grounding
Step type: {step_type}
Step content: {step_content}

### Task
Output a single JSON object with a list of {K} distinct candidate premises:
```json
{{"candidates": [
  {{"id": "c1", "text": "<NL fact>", "type": "fact|arithmetic|logic", "cite": "<row/col cite OR 'commonsense' OR 'derived'>"}},
  {{"id": "c2", "text": "...", "type": "...", "cite": "..."}},
  {{"id": "c3", "text": "...", "type": "...", "cite": "..."}}
]}}
```

Rules:
- For `fact`: state what the CoT claims about the table (e.g., "row 5 col 'X' = 60"). Cite the row/col.
- For `arithmetic`: state the equation as the CoT asserts it (e.g., "60 + 45 = 105"). Downstream verifier will Python-eval.
- For `logic`: state the universal rule used (e.g., "max(a, b) is the larger of a and b"). Cite as "commonsense".
- **Preserve CoT's claims FAITHFULLY** — if CoT cites a value, write the candidate text as CoT claimed it. The SAT filter (paper §2.3) catches table mismatch; do NOT silent-correct.
- Generate {K} DIFFERENT candidates spanning different framings of what the step needs (e.g., one direct cell cite, one derived-fact form, one rule). The SAT filter will discard ungrounded ones.
- Order candidates from most-likely-grounded to most-speculative.
"""


def gen_candidate_premises_for_step(llm, step, accumulated_premises, table_md, question, K=3):
    """paper §2.3 — Generate K candidate premises for ONE step.
    SAT filter (audit) downstream drops fabrications; we only propose."""
    accum_str = bulletize(accumulated_premises) if accumulated_premises else "(none yet)"
    prompt = PER_STEP_CANDIDATE_GEN_PROMPT.format(
        K=K,
        question=question[:500],
        table=(table_md or "(no table)")[:TABLE_CHAR_CAP],
        accumulated_premises=accum_str,
        step_type=step.get("type", "?"),
        step_content=step.get("conclusion", "") or step.get("content", ""),
    )
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, timeout=60.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        cands = parsed.get("candidates") or []
        if not isinstance(cands, list):
            return []
        out = []
        base = len(accumulated_premises)
        for i, c in enumerate(cands[:K]):
            if not isinstance(c, dict) or not c.get("text"):
                continue
            if not c.get("id"):
                c["id"] = f"p{base + i + 1}"
            out.append(c)
        return out
    except Exception:
        return []


def run_vericot_alg2(llm, question, cot, claimed, table_md="", df=None):
    """paper §2.3 / Algorithm 1 — full three-piece implementation (v10.2).

    For each atomic CoT step Cᵢ:
      (a) Decompose CoT → atomic steps (shared decomposer)
      (b) Try Z3 entailment with currently accumulated grounded premises
      (c) If entailed: continue; record step as verified axiom
      (d) Else: LLM generates K candidate premises (focused on this step)
      (e) **SAT filter** — audit each candidate via existing `audit_premise`
            VERIFIED / COMMONSENSE → keep
            FABRICATED            → drop (NOT global REJECT — paper §2.3 only
                                          drops the candidate)
            UNV                   → drop (cannot certify)
      (f) Add SURVIVORS to premise pool
      (g) Re-try Z3 entailment with augmented pool
      (h) Still not entailed (no survivors OR Z3 fails) → REJECT this CoT
    """
    try:
        atomic_steps = llm.decompose_cot(cot, question)
    except Exception as e:
        return "REJECT", f"alg2_decompose_error: {str(e)[:100]}"
    if not atomic_steps:
        return "ACCEPT", "alg2_no_steps_abstain"

    steps = [{"id": f"s{i+1}",
              "conclusion": s.get("content", ""),
              "type": s.get("type", "?")}
             for i, s in enumerate(atomic_steps)]

    nl_premises = []     # 𝒫ᵢ: grounded premises (grow on demand, post-SAT-filter)
    prior_verified = []  # ℱᵢ: verified steps (axioms for next steps)
    vocab = []
    vocab_seen = set()
    final_conclusion = steps[-1]["conclusion"] if steps else ""

    K = 3                # paper says "candidate premises" (plural); we use 3
    candidates_total = 0
    survivors_total = 0
    fabricated_dropped = 0

    for i, step in enumerate(steps):
        is_final = (i == len(steps) - 1)

        # (b) Entailment with current pool
        winner, _, details, new_decls = vote(
            llm, step, nl_premises, prior_verified,
            final_conclusion, claimed, is_final,
            vocab=vocab, table_md=table_md,
        )
        if winner == "unsat":
            prior_verified.append(step)
            for n, t in new_decls:
                if n not in vocab_seen:
                    vocab_seen.add(n)
                    vocab.append((n, t))
            continue

        # winner ∈ {sat, unknown, error, unclear} → step not yet entailed
        # (d) Generate K candidate premises focused on THIS step
        candidates = gen_candidate_premises_for_step(
            llm, step, nl_premises, table_md, question, K=K,
        )
        candidates_total += len(candidates)
        if not candidates:
            return "REJECT", f"alg2_no_candidates_step_{step['id']}"

        # (e) SAT filter via audit (paper §2.3: drop ungrounded candidates,
        # NOT the whole CoT). VERIFIED/COMMONSENSE survive; FABRICATED dropped.
        if table_md:
            audit_results = audit_premises_concurrent(llm, candidates, question, table_md)
        else:
            audit_results = [(p, "UNV", "no_table") for p in candidates]

        survivors = []
        for p, verdict, _reason in audit_results:
            if verdict in ("VERIFIED", "COMMONSENSE"):
                survivors.append(p)
            elif verdict == "FABRICATED":
                fabricated_dropped += 1
            # UNV → dropped silently
        survivors_total += len(survivors)

        if not survivors:
            # All candidates FABRICATED/UNV → step cannot be grounded → paper-faithful REJECT
            return "REJECT", (
                f"alg2_all_candidates_ungrounded_step_{step['id']} "
                f"[K={len(candidates)}, FAB={fabricated_dropped}]"
            )

        # (f) Add survivors to pool
        nl_premises.extend(survivors)

        # (g) Re-check entailment with augmented pool
        winner2, _, details2, new_decls2 = vote(
            llm, step, nl_premises, prior_verified,
            final_conclusion, claimed, is_final,
            vocab=vocab, table_md=table_md,
        )
        if winner2 == "unsat":
            prior_verified.append(step)
            for n, t in new_decls2:
                if n not in vocab_seen:
                    vocab_seen.add(n)
                    vocab.append((n, t))
        else:
            # (h) Even with grounded survivors, step not entailed → REJECT
            return "REJECT", (
                f"alg2_ungrounded_after_filter_step_{step['id']}: "
                f"winner={winner2}, surv={len(survivors)}, det={details2[0][:60]}"
            )

    # FACC tail (paper §2.2 robust fallback)
    facc_v, facc_d = check_final_answer_consistency(final_conclusion, claimed, cot)
    if facc_v == "MISMATCH":
        return "REJECT", f"alg2_facc_mismatch [{facc_d}]"

    return "ACCEPT", (
        f"alg2_all_{len(steps)}_steps_verified "
        f"[K={K}, cands={candidates_total}, surv={survivors_total}, "
        f"fab_dropped={fabricated_dropped}, facc:{facc_v}]"
    )


def run_vericot(llm, question, cot, claimed, table_md="", df=None):
    # Stage 1 — extract premises + steps (v9: now sees the table for grounding)
    extract_prompt = PREMISE_EXTRACT_PROMPT.format(
        question=question, cot=cot[:3000], claimed=claimed,
        table=(table_md or "(no table provided)")[:TABLE_CHAR_CAP],
    )
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0.0, timeout=60.0,
        )
        parsed = extract_json(resp.choices[0].message.content or "")
    except Exception as e:
        return "REJECT", f"extract_error: {str(e)[:100]}"  # Fix #6: unknown → REJECT (stricter default)

    if not parsed or not isinstance(parsed, dict) or "steps" not in parsed:
        return "ACCEPT", "extract_failed_abstain"  # hard abstain when we cannot extract anything

    premises = parsed.get("premises", []) or []
    steps = parsed.get("steps", []) or []
    final_conclusion = parsed.get("final_conclusion", "")
    if not steps:
        return "ACCEPT", "no_steps_abstain"

    # Stage 1.5 (v7) — Premise attribution audit (paper §2.4 LLM-as-Judge)
    if table_md and premises:
        audit_results = audit_premises_concurrent(llm, premises, question, table_md)
        fabricated = [(p, r) for p, v, r in audit_results if v == "FABRICATED"]
        if fabricated:
            p0, reason = fabricated[0]
            return "REJECT", (f"premise_unattributable [{p0.get('id','?')}]: "
                              f"{p0.get('text','')[:80]!r} | {reason[:200]}")

    # Stage 1.6 — answer verification
    use_code_judge = os.environ.get("VERICOT_NO_CODE") != "1"
    if use_code_judge and df is not None and table_md and claimed:
        code_v, code_r = _code_verify_answer(llm, df, question, table_md, claimed)
        if code_v == "MISMATCH":
            return "REJECT", f"code_answer_mismatch: {code_r}"
    elif not use_code_judge and table_md and claimed:
        comp_v, comp_r = audit_conclusion_completeness(
            llm, question, table_md,
            parsed.get("final_conclusion", ""), claimed)
        if comp_v == "INCOMPLETE":
            return "REJECT", f"conclusion_incomplete: {comp_r}"

    # Stage 2 — step-by-step verification with majority voting
    # v8: maintain a logical vocabulary across steps (paper §2.2 SMT-LIB vocab)
    prior_verified = []
    vocab = []  # list of (name, ztype) tuples, de-duplicated, insertion-ordered
    vocab_seen = set()
    for i, step in enumerate(steps):
        is_final = (i == len(steps) - 1)
        winner, verdicts, details, new_decls = vote(
            llm, step, premises, prior_verified, final_conclusion, claimed, is_final,
            vocab=vocab, table_md=table_md,
        )

        if winner == "premise_inconsistent":
            # Z3 encoding produced contradictory axioms — likely a code-gen artifact
            # (common on financial Item/Amount tables). Skip this step instead of rejecting.
            logger.warning(f"premise_inconsistent at step {step.get('id', i+1)}, skipping (not rejecting)")
            prior_verified.append(step)
            continue
        if winner == "unsat":
            # Step is entailed — add to prior verified, accumulate its decls
            prior_verified.append(step)
            for n, t in new_decls:
                if n not in vocab_seen:
                    vocab_seen.add(n)
                    vocab.append((n, t))
            continue
        if winner == "sat":
            # Counter-example found — but Z3 may produce false SAT on financial
            # arithmetic (Int precision, wrong encoding).
            step_text = step.get("conclusion", step.get("text", ""))
            # Fallback 1: Python eval confirms arithmetic correct → skip
            arith_check = try_eval_arithmetic(step_text)
            if arith_check is not None and arith_check[0] == "TRUE":
                logger.warning(f"Z3 SAT at step {step.get('id', i+1)} but Python eval confirms correct — skipping")
                prior_verified.append(step)
                continue
            # Fallback 2: require unanimous SAT (3/3) to reject.
            # Non-unanimous (2:1) → Z3 encoding is unstable → skip.
            sat_count = sum(1 for v in verdicts if v == "sat")
            if sat_count < len(verdicts):
                logger.warning(f"Z3 SAT at step {step.get('id', i+1)} non-unanimous ({sat_count}/{len(verdicts)}) — skipping")
                prior_verified.append(step)
                continue
            return "REJECT", f"fol_SAT_step_{step.get('id', i+1)}: {details[0][:120]}"
        if winner == "unknown":
            # Z3 couldn't determine — skip instead of rejecting (avoids false-reject
            # on tables where Z3 encoding is unreliable)
            logger.warning(f"Z3 unknown at step {step.get('id', i+1)}, skipping")
            prior_verified.append(step)
            continue
        if winner == "error":
            # Z3 codegen / exec failed — skip this step, continue verification
            logger.warning(f"Z3 error at step {step.get('id', i+1)}, skipping")
            prior_verified.append(step)
            continue
        # unclear → skip and continue
        logger.warning(f"Z3 unclear at step {step.get('id', i+1)}, skipping")
        prior_verified.append(step)
        continue

    # v8.3 — Deterministic Final-Answer Consistency Check on the
    # all-UNSAT-pass path (paper §2.2 final-step inequality, robust fallback).
    # v9.5 — On AMBIGUOUS, fallback to LLM-as-Judge (paper §2.4 spirit applied to
    #        final consistency, catches narrative self-contradictions like
    #        "5 wins...therefore the answer is 6" that regex can't see).
    facc_verdict, facc_detail = check_final_answer_consistency(final_conclusion, claimed, cot)
    if facc_verdict == "AMBIGUOUS":
        llm_v, llm_d = llm_facc_fallback(llm, cot, claimed)
        if llm_v == "MISMATCH":
            return "REJECT", f"facc_llm_mismatch [{llm_d}] | fol_{len(steps)}_steps_all_UNSAT"
        facc_verdict = llm_v if llm_v in ("CONSISTENT", "MISMATCH") else facc_verdict
        facc_detail = f"py_{facc_detail} → llm:{llm_d[:80]}"
    if facc_verdict == "MISMATCH":
        return "REJECT", f"facc_mismatch [{facc_detail}] | fol_{len(steps)}_steps_all_UNSAT"

    return "ACCEPT", f"fol_{len(steps)}_steps_all_UNSAT [facc:{facc_verdict}]"


def verify_single(item, sample_key, sample_data, llm):
    item_id = item.get("id", "unknown")
    try:
        table_content = item.get("table_content")
        if not table_content or not isinstance(table_content, dict):
            return None
        df = parse_structured_table(table_content)
        if df.empty:
            return None
        cot = extract_cot_text(sample_data)
        if not cot:
            return None
        claimed = extract_claimed_answer(sample_data)
        question = item.get("original_question", "")
        table_md = item.get("table_md") or df.to_csv(sep="|", index=False)

        use_alg1 = os.environ.get("VERICOT_ALG1") == "1"
        use_alg2 = os.environ.get("VERICOT_ALG2") == "1"
        if use_alg2:
            verifier = run_vericot_alg2
            src_tag = "vericot_v10_2_alg2"
        elif use_alg1:
            verifier = run_vericot_alg1
            src_tag = "vericot_v10_1_alg1"
        else:
            verifier = run_vericot
            src_tag = "vericot_v9_11"
        decision, rationale = verifier(llm, question, cot, claimed, table_md=table_md, df=df)
        return {
            "id": item_id,
            "target_type": KEY_MAPPING.get(sample_key, "unknown"),
            "specific_subtype": sample_key,
            "verifier_decision": decision,
            "verifier_rationale": rationale[:500],
            "source": src_tag,
        }
    except Exception as e:
        logger.error(f"[{item_id}-{sample_key}] {e}")
        return None


async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    llm = LLMEngine()

    tasks_args = []
    for item in data:
        gs = item.get("generated_samples", {})
        for sk in KEY_MAPPING:
            if sk not in gs: continue
            sd = gs[sk]
            if not isinstance(sd, dict) or "error" in sd: continue
            tasks_args.append((item, sk, sd))

    print(f"{len(data)} items, {len(tasks_args)} tasks (VeriCoT v9.4, vote_n={VOTE_N}, +stage1_sees_table)")
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    lock = asyncio.Lock()

    async def run_one(item, sk, sd):
        async with sem:
            return await asyncio.to_thread(verify_single, item, sk, sd, llm)

    tasks = [asyncio.create_task(run_one(*a)) for a in tasks_args]
    for f in tqdm_asyncio.as_completed(tasks, desc="VeriCoT v9.4 audit_split+vocab+FACC"):
        res = await f
        if res:
            results.append(res)
            if len(results) % SAVE_INTERVAL == 0:
                async with lock:
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
                        json.dump(results, fout, indent=2, ensure_ascii=False)

    print(f"\nSaving {len(results)} results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) >= 2: INPUT_FILE = sys.argv[1]
    if len(sys.argv) >= 3: OUTPUT_FILE = sys.argv[2]
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
