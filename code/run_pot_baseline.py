"""
PoT (Program-of-Thoughts) baseline — Chen et al., TMLR 2023.
arXiv:2211.12588, code: https://github.com/wenhuchen/Program-of-Thoughts

Adapted to TrustTable-Bench verification task (v6-equivalent, paper-faithful trim):
  Input:  (Q, T, CoT, claimed_answer, target_type)

Pipeline:
  1. Code-gen + exec: solve(df) → answer string. tool-error → REJECT.
  2. Answer match: pot_answer vs claimed_answer (tight numeric tolerance).
  2.5. Regex arith check: 'X op Y = Z' patterns, Python-eval, reject on
       arithmetic mismatch.
  2.6. LLM-extracted arith: LLM canonicalizes narrative arith to 'X op Y = Z',
       Python-eval each. Bypasses think-then-judge arith leniency.
  3. CoT Fact+Logic audit (think-then-judge): catches fabricated values
     (Type A) and fabricated logic (Type B). Reject if either found.

Note: previously had a Stage 2.7 F1-F7 logic-flip audit + worked examples
(v7-v10). Removed 2026-05-02 to reduce paper-deviation; logic-flip detection
deferred to Stage 3 TTJ.
"""
import json, operator, os, sys, re, asyncio, math
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from src.llm_engine import LLMEngine
from utils.logger import setup_logger
from utils.table_utils import parse_structured_table

logger = setup_logger("PoT_Baseline")

INPUT_FILE = "../data/small/panel_c_wtq/type1_correct.json"
OUTPUT_FILE = "../outputs/pot_wtq_type1.json"
CONCURRENCY = 10
SAVE_INTERVAL = 50

# Prompt-input truncation budgets (avoid token blow-up on long FinQA tables / CoTs)
MAX_TABLE_CHARS = 5000
MAX_COT_CHARS = 2500

# Arithmetic operators recognized by Stage 2.5 / 2.6
_OPS = {
    '+': operator.add, '-': operator.sub,
    '*': operator.mul, '×': operator.mul, 'x': operator.mul,
    '/': operator.truediv, '÷': operator.truediv,
}

# Tolerance shared by regex and LLM-extracted arith checks.
# Tightened from (0.003, 0.02) → (0.001, 0.005) for Med tables where embedded
# T2_a errors are often sub-percent sanity-check sums.
_ARITH_REL_TOL = 0.001
_ARITH_ABS_TOL = 0.005


def _eval_eq(a: float, op: str, b: float, c: float):
    """Return the expected value if `a op b != c` (within tolerance), else None.

    Returning the expected value lets the caller format an error message; None
    means the equation checks out (or the op was unrecognized / div by zero)."""
    fn = _OPS.get(op)
    if fn is None or (op in ('/', '÷') and b == 0):
        return None
    expected = fn(a, b)
    if math.isclose(expected, c, rel_tol=_ARITH_REL_TOL, abs_tol=_ARITH_ABS_TOL):
        return None
    return expected

KEY_MAPPING = {
    "type1_correct": "type1_golden",
    "type2_grounding_error": "type2_spurious",
    "type2_arithmetic_error": "type2_spurious",
    "type2_logic_error": "type2_spurious",
    "type3_fully_wrong": "type3_fully_wrong",
    "type4_calc_error": "type4_calc_error",
    "type4_answer_perturb": "type4_inconsistent_easy",
}


def extract_claimed_answer(sd):
    return sd.get("answer") or sd.get("incorrect_answer") or sd.get("pred_answer") or ""


POT_SYSTEM = """You are a Python programmer solving TableQA questions.
Write a self-contained Python script that reads the table and computes the answer.

Rules:
- The table is provided as a Pandas DataFrame variable `df`.
- Define a function `solve(df)` that returns the answer as a string.
- Use only standard Python and Pandas operations. Do NOT import anything else.
- The function must produce a single-line answer string (not a DataFrame).
- Handle string normalization (strip, case) as needed.
- If a value cannot be determined from the table, return the empty string ""."""


# v3: CoT Fact Audit — scan the CoT for any specific value/entity cited and
# verify it against the table. This catches Type-2 spurious reasoning where
# the CoT fabricates intermediate values but the answer is coincidentally right.
COT_AUDIT_PROMPT = """You are verifying a Chain-of-Thought against a table. Check for TWO types of errors:

### Question
{question}

### Table (ground truth)
{table_md}

### Chain-of-Thought under audit
{cot}

### Task — check for BOTH error types:

**Type A — Fabricated values**: The CoT cites a SPECIFIC value not in the table.
  - "X has Y = V" but table's actual V for X is different
  - An entity/row that does not exist in the table
  - A count off by more than 1 from the actual count

**Type B — Fabricated logic**: The CoT misinterprets the question's conditions.
  - Question says "A AND B" but CoT checks "A OR B" (connective flip)
  - Question says "more than 3" but CoT checks "at least 3" (operator swap: >3 vs ≥3)
  - Question says "neither A nor B" but CoT only excludes A (dropped constraint)
  - Question says "exactly one" but CoT checks "at least one" (quantifier weakening)

Do NOT flag: paraphrases, formatting differences, abbreviations, claims the table does not address.

### Output format (think-then-judge)
First, REASONING (3-5 lines):
  1. List each specific value/entity the CoT cites and verify against the table.
  2. List each condition in the question and check if the CoT interprets it correctly.
Then, on a NEW line starting with `>>>`, write your verdict:

>>> FABRICATED: <specific error — cite the CoT claim vs table/question truth>
>>> NO_FABRICATION
"""


def extract_code(text: str) -> str:
    """Pull Python code from fenced block if present, else return whole text."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def answers_equal(a: str, b: str) -> bool:
    """Answer matching with tight numeric tolerance and string-only containment.

    Fixes from v1:
      - numeric tolerance tightened: rel_tol=0.001 abs_tol=0.5 (was 0.01/0.01
        which false-matched 1972~1974, 10000~100000)
      - substring containment suppressed for numeric answers to stop
        '10000' ⊂ '100000' false matches
    """
    def norm(s):
        s = str(s or "").strip().lower()
        s = re.sub(r"[\s,$%]", "", s)
        return s
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True

    # Numeric match — tight tolerance
    both_numeric = False
    try:
        fa, fb = float(na), float(nb)
        both_numeric = True
        # Exact-integer: if both integer-valued, require exact match
        if fa == int(fa) and fb == int(fb):
            if int(fa) == int(fb):
                return True
            return False
        # Otherwise: very tight tolerance (0.1% rel, 0.5 abs) for WTQ-style answers
        if math.isclose(fa, fb, rel_tol=0.001, abs_tol=0.5):
            return True
        return False
    except (ValueError, OverflowError):
        pass

    # Non-numeric: allow substring containment for short-form answers
    if not both_numeric and (na in nb or nb in na):
        # Guard: avoid trivial containment ('a' in 'apple')
        if min(len(na), len(nb)) >= 3:
            return True
    return False


def _gen_and_exec(llm, df, question, table_md, feedback=""):
    """Single attempt at gen+exec. Returns (result_str_or_None, error_msg)."""
    user_prompt = f"""### Table
{table_md[:6000]}

### DataFrame columns
{list(df.columns)}

### DataFrame dtypes
{ {c: str(df[c].dtype) for c in df.columns} }

### Question
{question}

### Task
Write `def solve(df)` that returns the answer as a string. Put code in a fenced python block.
{feedback}"""
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": POT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0, timeout=60.0,
        )
        code = extract_code(resp.choices[0].message.content or "")
    except Exception as e:
        return None, f"gen_error: {str(e)[:100]}"
    try:
        g = {"pd": pd, "re": re}
        l = {}
        exec(code, g, l)
        if "solve" not in l:
            return None, "no_solve_fn"
        result = str(l["solve"](df)).strip()
        return result, ""
    except Exception as e:
        return None, f"exec_error: {str(e)[:100]}"


def _llm_judge(llm, prompt: str, timeout: float = 45.0) -> str:
    """One-shot LLM call returning the '>>> ...' verdict line (or last line if missing).

    Returns "" on any failure — caller decides pass/fail from the verdict text.
    """
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, timeout=timeout,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.debug("LLM judge call failed: %s", str(e)[:120])
        return ""
    for ln in out.splitlines():
        if ln.strip().startswith(">>>"):
            return ln.strip()
    return out.splitlines()[-1].strip() if out else ""


ARITH_EXTRACT_PROMPT = """You extract arithmetic equations from a Chain-of-Thought, including those expressed in natural language.

### CoT
{cot}

### Task
List EVERY arithmetic equation explicitly stated or directly computed in the CoT, in canonical form. Include narrative forms like "X plus Y is Z", "X minus Y gives Z", "Adding X and Y", "X percent of Y", "X divided by Y", etc.

For each equation, output exactly ONE line:
  X OP Y = Z

where:
  - X, Y, Z are numeric literals (NO $, commas, units, currency symbols, NO percent sign — convert "5%" to 0.05; convert "(1,000)" to -1000)
  - OP is one of: + - * /
  - "X percent of Y" → use X*Y/100 i.e. (X/100) * Y = Z, output as Y * 0.0X = Z
  - "X out of Y" → output as X / Y = Z

Skip:
  - Pure comparisons (no equation)
  - Equations whose Z is not stated as a concrete number
  - Self-corrections ("Wait, that should be …") — output only the FINAL stated value

If no arithmetic equation is computed, output the single line: NONE

Output ONLY the equation lines, one per line. No commentary."""


_LLM_EQ_RE = re.compile(
    r'^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)\s*$'
)


def _llm_extract_arith(llm, cot: str, timeout: float = 30.0):
    """Stage 2.6: LLM canonicalizes narrative arith to 'X op Y = Z'; we Python-eval each.
    Returns list of error strings (empty = all correct or nothing extracted)."""
    if not cot.strip():
        return []
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": ARITH_EXTRACT_PROMPT.format(
                cot=cot[:MAX_COT_CHARS])}],
            temperature=0.0, timeout=timeout,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.debug("LLM arith-extract call failed: %s", str(e)[:120])
        return []
    if not out or out.upper().startswith("NONE"):
        return []

    errors = []
    for ln in out.splitlines():
        m = _LLM_EQ_RE.match(ln.strip())
        if not m:
            continue
        a_s, op, b_s, c_s = m.groups()
        try:
            expected = _eval_eq(float(a_s), op, float(b_s), float(c_s))
        except (ValueError, ZeroDivisionError):
            continue
        if expected is not None:
            errors.append(f"{a_s} {op} {b_s} = {c_s} (should be {expected:.4g})")
    return errors


_NUM = r'\$?-?\d+(?:,\d{3})*(?:\.\d+)?'
_REGEX_EQ_RE = re.compile(rf'({_NUM})\s*([+\-×x*/÷])\s*({_NUM})\s*[=≈]\s*({_NUM})')


def _parse_finqa_num(s: str) -> float:
    return float(s.replace('$', '').replace(',', ''))


def _check_cot_arithmetic(cot: str):
    """Extract binary arithmetic equations from CoT and verify with Python eval.
    Returns list of error strings (empty = all correct). Chain guard skips
    matches inside chains like '1 + 2 + 3 = 6' (would otherwise flag '2 + 3 = 6')."""
    errors = []
    for m in _REGEX_EQ_RE.finditer(cot):
        j = m.start(1) - 1
        while j >= 0 and cot[j] in ' \t':
            j -= 1
        if j >= 0 and cot[j] in '+-*/×÷.,':
            continue
        a_s, op, b_s, c_s = m.groups()
        try:
            expected = _eval_eq(_parse_finqa_num(a_s), op,
                                _parse_finqa_num(b_s), _parse_finqa_num(c_s))
        except (ValueError, ZeroDivisionError):
            continue
        if expected is not None:
            errors.append(f"{a_s} {op} {b_s} = {c_s} (should be {expected:.4g})")
    return errors


def run_pot(llm, df: pd.DataFrame, question: str, claimed_answer: str, table_md: str, cot: str = ""):
    """Run the 5-stage PoT verification. See module docstring."""
    # Stage 1 — code-gen + exec (1 tool-repair retry). Failure → REJECT
    # (verifier could not certify correctness; paper-faithful interpretation).
    result, err = _gen_and_exec(llm, df, question, table_md)
    if result is None and err:
        feedback = f"\nPrevious attempt failed: {err}. Fix and retry; be robust to dtype coercion."
        result, err2 = _gen_and_exec(llm, df, question, table_md, feedback=feedback)
        if result is None:
            return "REJECT", f"pot_tool_error_no_certify: {err2 or err}"

    # Stage 2 — answer match
    if not answers_equal(result, claimed_answer):
        return "REJECT", f"pot_answer={result!r} != claimed={claimed_answer!r}"

    if cot:
        # Stage 2.5 — regex arith check (deterministic).
        regex_errs = _check_cot_arithmetic(cot)
        if regex_errs:
            return "REJECT", f"pot_cot_arith_mismatch: {regex_errs[0]}"

        # Stage 2.6 — LLM-extracted arith (catches narrative forms).
        llm_arith_errs = _llm_extract_arith(llm, cot)
        if llm_arith_errs:
            return "REJECT", f"pot_cot_arith_mismatch_llm: {llm_arith_errs[0]}"

        # Stage 3 — TTJ fact+logic audit.
        verdict = _llm_judge(llm, COT_AUDIT_PROMPT.format(
            question=question, table_md=table_md[:MAX_TABLE_CHARS], cot=cot[:MAX_COT_CHARS]))
        if "FABRICATED" in verdict.upper() and "NO_FABRICATION" not in verdict.upper():
            return "REJECT", f"pot_cot_fabrication: {verdict[4:].strip()[:200]}"

    return "ACCEPT", f"pot_answer={result!r} matches claimed={claimed_answer!r}"


def _extract_cot(sd):
    return (sd.get("chain_of_thought") or sd.get("flawed_chain_of_thought")
            or sd.get("correct_logic_wrong_math_cot") or sd.get("incorrect_chain_of_thought") or "")


def verify_single(item, sample_key, sample_data, llm):
    item_id = item.get("id", "unknown")
    try:
        table_content = item.get("table_content")
        if not table_content or not isinstance(table_content, dict):
            return None
        df = parse_structured_table(table_content)
        if df.empty:
            return None

        claimed = extract_claimed_answer(sample_data)
        if not claimed:
            return None

        question = item.get("original_question", "")
        table_md = item.get("table_md") or df.to_csv(sep="|", index=False)
        cot = _extract_cot(sample_data)

        decision, rationale = run_pot(llm, df, question, claimed, table_md, cot=cot)

        return {
            "id": item_id,
            "target_type": KEY_MAPPING.get(sample_key, "unknown"),
            "specific_subtype": sample_key,
            "verifier_decision": decision,
            "verifier_rationale": rationale[:500],
            "source": "pot",
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

    print(f"{len(data)} items, {len(tasks_args)} tasks")
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    lock = asyncio.Lock()

    async def run_one(item, sk, sd):
        async with sem:
            return await asyncio.to_thread(verify_single, item, sk, sd, llm)

    tasks = [asyncio.create_task(run_one(*a)) for a in tasks_args]
    for f in tqdm_asyncio.as_completed(tasks, desc="PoT Baseline"):
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
