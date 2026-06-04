"""
Atomic Skills (Atomic Reasoning) baseline — Zhang et al., 2025
arXiv:2506.06972

v11 — Two paper-faithful additions on top of v10:
  (1) Recap field per claim (paper §4.1 skill-chain schema's "Recap" step):
      decomposer must emit a VERBATIM CoT quote per claim. Recap, not the
      (potentially LLM-rewritten) claim text, is the source of truth for
      arithmetic verification. Programmatically enforces the FIDELITY rule
      that v10 only stated as a prompt warning.
  (2) Causal Analysis skill (paper §4.2 atomic skill set, "Causal Analysis"
      group): new 7th skill `Causal` for claims that assert a rule / mapping /
      causal/correlational relation. Verified per-claim by LLM judge that must
      cite specific table cells or reject as fabricated. Zero-shot adaptation
      of paper's Causal Analysis skill (paper fine-tunes; we prompt — only
      this dimension differs from §4.2 spec).

Carry-over from v10:
  (a) Lookup fuzzy-match (FC2 port) — content-word-guarded format-friction recovery.
  (b) Aggressive arithmetic scan (regex over raw CoT) — decomposer-bypass safety.
  (c) Strict UNV-helpless gate — REJECT when zero TRUEs and ≥3 UNVs.

Method spirit: Zhang 2025 §4 skill-chain schema, adapted from binary
SUPPORT/REFUTE to per-error-type T1-T4 verification. Aggregation: any FALSE → REJECT.

Pipeline:
  1. Aggressive arith scan pre-check — programmatic, no LLM.
  2. Skill-typed decomposition (1 LLM call) — outputs JSON array of {skill, claim, recap}.
  3. Per-skill dispatch (recap passed as context):
       Lookup → generate_pandas_check (FC2 fuzzy match overrides LLM bool)
       Filter / Aggregate / Compare / FinalAnswer → generate_inference_check
       Arithmetic → _try_simple_arithmetic on RECAP first, then claim, fallback verify_inference
       Causal → LLM judge with cite-or-refute prompt (per-claim, NOT global)
  4. Aggregation: first FALSE → REJECT; strict UNV-helpless gate; else ACCEPT.

Output records compatible with eval_cot_verifier.py.
"""
import json, os, sys, re, asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from src.llm_engine import LLMEngine
from src.pipeline import _try_simple_arithmetic
from src.verifiers.fact_checker import _fuzzy_match
from utils.logger import setup_logger
from utils.table_utils import parse_structured_table

logger = setup_logger("AtomicSkills_Baseline")

INPUT_FILE = "../data/small/panel_c_wtq/type1_correct.json"
OUTPUT_FILE = "../outputs/atomic_wtq_type1.json"
CONCURRENCY = 10
SAVE_INTERVAL = 50
MAX_CLAIMS = 12
TABLE_CHAR_CAP = 6000

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


SKILL_TYPES = {"Lookup", "Filter", "Aggregate", "Compare", "Arithmetic", "FinalAnswer", "Causal"}

DECOMPOSE_PROMPT = """You are a skill-typed atomic decomposer for table reasoning audits. Decompose the CoT into atomic claims; tag EACH claim with EXACTLY ONE skill from the 6-skill taxonomy of Zhang et al. 2025 (Atomic Reasoning).

### CRITICAL: FIDELITY — DO NOT SILENTLY CORRECT THE CoT
The CoT may contain arithmetic errors, miscounts, wrong intermediate results, or self-contradictions. You MUST preserve EVERY numeric / count / arithmetic assertion VERBATIM, exactly as the CoT states it. Do NOT recompute, normalize, or round — even if you can see the CoT is wrong.

Examples of fidelity violations to AVOID:
  - CoT: "106.3 - 89.7 = 16.5" (wrong, actual=16.6) → claim MUST be "106.3 - 89.7 = 16.5", NOT "106.3 - 89.7 = 16.6".
  - CoT: "2 + 1 = 4" → claim MUST be "2 + 1 = 4", NOT "2 + 1 = 3".
  - CoT: "Counting gives 5 entries" but only 4 listed → claim MUST be "Counting gives 5", NOT "Counting gives 4".
  - CoT: "458,978 + 177 = 460,252" → claim MUST be "458,978 + 177 = 460,252", NOT "459,155".

To enforce fidelity programmatically, EACH claim MUST include a `recap` field:
the **verbatim quote** from the CoT containing the asserted fact (no
paraphrasing, no normalization). The verifier will scan recap text — if your
`claim` field has been silently corrected, the discrepancy with `recap` will
be caught.

### Skills (6-skill atomic taxonomy)

- **Lookup**: claim asserts a SPECIFIC cell value at a specific row.
  e.g., "John's Total column value is 20", "The 2002 row's Date column = '5 March 2002'"
- **Filter**: claim asserts a count of rows / a list of rows that satisfy a condition.
  e.g., "There are 4 rows where Score >= 30", "Teams with positive GD are KR, Fylkir, Grindavík"
- **Aggregate**: claim asserts a sum / count / average / max / min over filtered rows.
  e.g., "The sum of points = 156", "The maximum age is 23"
- **Compare**: claim asserts an ordering / relational fact between specific values.
  e.g., "Pat's total (1) is less than John's (20)", "X is the highest"
- **Arithmetic**: claim asserts a stated arithmetic equation (intermediate sub-step OR final).
  e.g., "37 + 35 = 72", "131 + 26 = 156"
- **Causal**: claim asserts a RULE, MAPPING, or CAUSAL/CORRELATIONAL relation
  not directly a single cell value or arithmetic. Tag Causal when the claim
  contains causal connectives ("because", "due to", "implies", "leads to",
  "as a result", "if X then Y") OR asserts a code/symbol mapping ("(i) means
  X", "the asterisk indicates Y", "this column represents Z") OR asserts a
  trend/pattern ("the values increase with age", "more X correlates with more Y").
  e.g., "(i) in the table denotes 'in Russia'", "venue change incurs 14 attendee loss",
  "the rule is that retired drivers earn 0 points"

### Question
{question}

### Chain-of-Thought
{cot}

### Claimed final answer
{claimed}

### Rules
- One specific fact per claim (atomic).
- For multi-step arithmetic chains: extract EACH sub-step as a separate Arithmetic claim. Do NOT fold them into a single total.
- Be EXHAUSTIVE for numeric/entity claims — better too many than miss a fabrication.
- Tag claims as Causal whenever they assert a rule/mapping not directly verifiable by a single cell — these are the highest-risk fabrication vectors.
- Each claim MUST have BOTH `claim` and `recap` fields. recap = verbatim CoT quote.
- MAX {max_claims} claims (excluding the final-answer entry).

### Output — JSON array, fenced
```json
[
  {{"skill": "Lookup", "claim": "Pat's Total = 1",
    "recap": "Pat's column shows a Total of 1."}},
  {{"skill": "Arithmetic", "claim": "10 + 9 + 3 = 21",
    "recap": "Their total is 10 + 9 + 3 = 21."}},
  {{"skill": "Compare", "claim": "John's Total (20) > Pat's Total (1)",
    "recap": "John (20) scored more than Pat (1)."}},
  {{"skill": "Causal", "claim": "The (i) symbol means 'in Russia'",
    "recap": "Note: (i) indicates the match took place in Russia."}},
  ...
]
```"""


_CHAIN_END_PAT = re.compile(
    r'(?:'
    r'(?:therefore|thus|hence|so|finally|in\s+total|total(?:s)?|sum(?:s)?|gives|equals|gets|comes\s+to|adds?\s+to|is|are|=|gives\s+us|brings\s+(?:us|the\s+total)\s+to|brings\s+(?:us|it)\s+precisely\s+to|adding\s+them\s+up\s+gives|adding\s+up\s+gives|that\s+totals\s+to|adding\s+\w+\s+gives|the\s+result\s+is)'
    r'\s*[a-z\s,]*?'
    r'([\$]?[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*\w+)?)'
    r')',
    re.IGNORECASE,
)


def _extract_chain_conclusion(cot: str):
    """Extract the LAST numeric/quantitative conclusion the CoT states before
    its 'answer is X' / 'final answer' line. Returns the value as a string, or None.

    Strategy: scan the last 400 chars; look for the last occurrence of a
    summary cue ("totals to N", "= N", "gives N", "is N seasons", etc.) AT
    a location BEFORE any "answer is" / "Therefore, the answer" prelude.
    """
    if not cot or len(cot) < 5:
        return None
    tail = cot[-500:]
    # Cut off after 'answer' marker — we want pre-answer chain conclusions
    ans_split = re.split(r'(?:[Tt]he\s+)?answer\s+is', tail)
    chain_text = ans_split[0] if len(ans_split) > 1 else tail
    matches = list(_CHAIN_END_PAT.finditer(chain_text))
    if not matches:
        return None
    # Pick the LAST numeric mention
    last = matches[-1].group(1).strip()
    # Strip trailing word like "seasons" — keep just the number
    m_num = re.match(r'([\$]?[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)', last)
    if not m_num:
        return None
    return m_num.group(1)


_NUM_NORM_PAT = re.compile(r'[\$,]')


def _norm_num_str(s: str):
    """Normalize a number-like string for direct equality comparison.
    Returns float on success, None on failure."""
    if s is None:
        return None
    t = _NUM_NORM_PAT.sub('', str(s).strip())
    try:
        return float(t)
    except (ValueError, TypeError):
        return None


def _values_match(a: str, b: str) -> bool:
    """Match two answer strings: numeric-equality first (with 1e-3 tolerance),
    then case-insensitive trimmed equality, then membership."""
    if a is None or b is None:
        return False
    na, nb = _norm_num_str(a), _norm_num_str(b)
    if na is not None and nb is not None:
        tol = 1e-3 * max(1.0, abs(nb)) + 1e-6
        return abs(na - nb) <= tol
    a_s = str(a).strip().lower()
    b_s = str(b).strip().lower()
    if not a_s or not b_s:
        return False
    if a_s == b_s:
        return True
    # one contained in the other (handles "4" vs "4 entries", "$200" vs "200")
    if a_s in b_s or b_s in a_s:
        return True
    return False


# Aggressive arithmetic equation scanner — finds every "X op Y (op Z ...) = R"
# in the raw CoT text. Used to bypass decomposer "silent correction" of mid-CoT
# arithmetic errors (a major T4_calc leak source).
_NUM = r'[\+\-]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
_OP = r'[\+\-\*/×÷]'
_EQ_PAT = re.compile(
    rf'(?<!\d)({_NUM}(?:\s*{_OP}\s*{_NUM})+)\s*=\s*({_NUM})(?!\d)'
)
# Narrative equation form: "X + Y + Z. (some words) (results in|totals|...) N"
# Strict guards to avoid date/score patterns:
#   - operators MUST be surrounded by whitespace ("1 + 1", not "1955/56")
#   - LHS MUST contain at least 2 operators (at least 3 operands)
#   - connector list excludes "is/are" (too ambiguous)
_EQ_NARRATIVE_PAT = re.compile(
    rf'(?<!\d)({_NUM}(?:\s+{_OP}\s+{_NUM}){{2,}})[\s.,;]+'
    rf'(?:[A-Za-z][\w\s,]{{0,80}}?)?'
    rf'\b(?:results?\s+in|equals?|gives?|adds?\s+up\s+to|adds?\s+to|sums?\s+to|comes?\s+to|totals?(?:ing)?\s*(?:to)?|amounts?\s+to|yields?)\s+'
    rf'(?:a\s+total\s+of\s+|just\s+|exactly\s+|approximately\s+)?'
    rf'({_NUM})(?!\d)',
    re.IGNORECASE,
)


def _scan_cot_arithmetic_errors(cot: str, max_checks: int = 12):
    """Return list of (lhs_str, claimed_str, computed_value) for equations whose
    stated RHS is wrong. Empty list = no detectable arithmetic error.

    Two patterns:
      - explicit `X op Y = N` (covers most CoTs)
      - narrative `X op Y. ... results in/equals/is N` (catches CoTs that say
        "1 + 1 + 1. However, adding these together results in 4")
    """
    if not cot:
        return []
    errors = []
    seen = set()

    def _check(lhs_raw, rhs_raw):
        try:
            expr = (lhs_raw.replace('×', '*').replace('÷', '/')
                           .replace(',', '').replace('$', '').strip())
            if not re.fullmatch(r'[\d\s\+\-\*/\.]+', expr):
                return None
            actual = float(eval(expr, {"__builtins__": {}}, {}))
            claimed = float(rhs_raw.replace(',', '').replace('$', ''))
            tol = 1e-6 * max(1.0, abs(claimed)) + 0.01
            if abs(actual - claimed) > tol:
                return (lhs_raw, rhs_raw, actual)
        except Exception:
            return None
        return None

    for m in _EQ_PAT.finditer(cot):
        if len(errors) + len(seen) >= max_checks:
            break
        lhs_raw, rhs_raw = m.group(1), m.group(2)
        key = (lhs_raw.strip(), rhs_raw.strip())
        if key in seen:
            continue
        seen.add(key)
        err = _check(lhs_raw, rhs_raw)
        if err:
            errors.append(err)

    if not errors:
        for m in _EQ_NARRATIVE_PAT.finditer(cot):
            if len(seen) >= max_checks:
                break
            lhs_raw, rhs_raw = m.group(1), m.group(2)
            key = (lhs_raw.strip(), rhs_raw.strip())
            if key in seen:
                continue
            seen.add(key)
            err = _check(lhs_raw, rhs_raw)
            if err:
                errors.append(err)

    return errors


def parse_json_skill_array(text: str):
    """Parse [{skill, claim, recap}, ...] array. Tolerate fence/no-fence/missing fields."""
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    raw = m.group(1) if m else text
    if not m:
        m2 = re.search(r"\[[\s\S]*\]", text)
        if m2:
            raw = m2.group(0)
    try:
        arr = json.loads(raw)
    except Exception:
        return None
    if not isinstance(arr, list):
        return None
    out = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        skill = str(x.get("skill", "")).strip()
        claim = str(x.get("claim", "")).strip()
        recap = str(x.get("recap", "")).strip() or claim  # fallback: recap = claim
        if not claim:
            continue
        if skill not in SKILL_TYPES:
            # Default unknown to Compare (LLM judge), the most lenient route
            skill = "Compare"
        out.append((skill, claim, recap))
    return out or None


# ----- Skill verifiers ------------------------------------------------------
def _exec_pandas_check(code: str, df: pd.DataFrame, fn_name: str):
    """Execute generated code, look up fn_name(df), return its output or raise."""
    g = {"pd": pd, "re": re}
    l = {}
    exec(code, g, l)
    fn = l.get(fn_name) or g.get(fn_name)
    if fn is None:
        raise RuntimeError(f"function {fn_name!r} not found")
    return fn(df)


def verify_lookup(claim, recap, df, columns, sample_data, table_md, question, llm):
    """Generate extract_and_compare(df) → {actual, claimed, match}. Return ('TRUE'|'FALSE'|'UNV', reason).

    v10: post-process LLM's match decision with FC2 _fuzzy_match. The LLM-emitted
    `match` flag is conservative (often False on parenthetical/format friction).
    FC2's fuzzy comparison is content-word-guarded, so it doesn't leak T2 errors
    while recovering format-only T1 false-rejects.

    v11: claim is augmented with recap (verbatim CoT quote) for cell disambiguation.
    """
    try:
        # Pass recap-augmented claim so the Pandas extractor sees the verbatim phrasing.
        prompt_claim = claim if claim == recap else f"{claim}\n(verbatim CoT: {recap[:200]})"
        code = llm.generate_pandas_check(prompt_claim, columns, sample_data, full_table=table_md, question=question)
        result = _exec_pandas_check(code, df, "extract_and_compare")
        if not isinstance(result, dict):
            return "UNV", f"lookup_bad_return:{type(result).__name__}"
        actual = str(result.get("actual", ""))[:200]
        claimed_v = str(result.get("claimed", ""))[:200]
        if actual == "N/A" and claimed_v == "N/A":
            return "TRUE", "lookup_procedural_skip"
        # FC2-style fuzzy match (overrides LLM's strict bool).
        match = _fuzzy_match(actual, claimed_v)
        return ("TRUE" if match else "FALSE"), f"actual={actual[:80]!r} claimed={claimed_v[:80]!r}"
    except Exception as e:
        return "UNV", f"lookup_exec_err:{str(e)[:60]}"


def verify_inference_skill(claim, recap, df, columns, sample_data, table_md, question, llm, skill_label="Inference"):
    """Generate verify_inference(df) → bool. For Filter / Aggregate / Compare / FinalAnswer."""
    try:
        prompt_claim = claim if claim == recap else f"{claim}\n(verbatim CoT: {recap[:200]})"
        code = llm.generate_inference_check(prompt_claim, columns, sample_data, table_md, question=question)
        result = _exec_pandas_check(code, df, "verify_inference")
        if isinstance(result, bool):
            return ("TRUE" if result else "FALSE"), f"{skill_label.lower()}_pandas={result}"
        return "UNV", f"{skill_label.lower()}_bad_return:{type(result).__name__}"
    except Exception as e:
        return "UNV", f"{skill_label.lower()}_exec_err:{str(e)[:60]}"


def verify_arithmetic_skill(claim, recap, df, columns, sample_data, table_md, question, llm):
    """Try direct eval first, fallback to LLM verify_inference.

    v11: scan recap (verbatim CoT) FIRST for an equation. If recap and claim
    disagree on the equation values, recap wins (decomposer fidelity violation
    is treated as evidence of silent correction). This catches T4_calc cases
    where the decomposer replaces 'X+Y=21' (wrong) with 'X+Y=22' (right).
    """
    # Step 1: scan recap text for an equation (decomposer-bypass).
    recap_arith = _try_simple_arithmetic(recap)
    if recap_arith is not None:
        return ("TRUE" if recap_arith else "FALSE"), f"arith_eval_on_recap={recap_arith}"
    # Step 2: try claim text.
    arith = _try_simple_arithmetic(claim)
    if arith is not None:
        return ("TRUE" if arith else "FALSE"), f"arith_eval_on_claim={arith}"
    # Step 3: fall back to LLM verify_inference.
    return verify_inference_skill(claim, recap, df, columns, sample_data, table_md, question, llm,
                                  skill_label="Arithmetic")


def verify_compare_skill(claim, recap, df, columns, sample_data, table_md, question, llm):
    """Compare relations by Pandas inference (works for X > Y, ordering, etc)."""
    return verify_inference_skill(claim, recap, df, columns, sample_data, table_md, question, llm,
                                  skill_label="Compare")


CAUSAL_CHECK_PROMPT = """You are a careful CAUSAL ANALYSIS verifier for table claims (atomic skill from Zhang et al. 2025, §4.2).

The claim asserts a RULE / MAPPING / CAUSAL or CORRELATIONAL relation. Determine whether the table itself directly evidences this rule. If the rule is not supported by specific table cells (no evidence of the mapping, no observable causal pattern), classify as FABRICATED — even if the rule "sounds plausible."

### Question
{question}

### Atomic claim (rule to verify)
{claim}

### Verbatim CoT recap (where the claim came from)
{recap}

### Table
{table}

### Decision protocol (very important)
1. CODE / SYMBOL mapping (e.g., "(i) means in Russia", "* means home game"):
   - REQUIRED: cite the exact table cell or caption defining the mapping.
   - If the table does not define this mapping anywhere, → FABRICATED.
2. CAUSAL or correlational pattern (e.g., "venue change incurs 14 attendee loss"):
   - REQUIRED: identify at least one row pair in the table where the pattern holds.
   - If no such pattern is observable, → FABRICATED.
3. NARROW EXEMPTIONS (output VERIFIED):
   - The claim is a TAUTOLOGY directly entailed by the table semantics — e.g.,
     "the rank immediately above X is closer than rows further above";
     "if A > B then more X corresponds to A". (Do NOT use this for abbreviations
     or symbol mappings — those need rule 1.)
   - The claim restates an observable numeric comparison: "X (142) > Y (130)
     means more X" where the cells X=142 and Y=130 are visible.
4. Domain-specific common-sense fact NOT in the table → still FABRICATED. We
   audit table-grounded reasoning, not world knowledge.

### Output (one line, exactly one prefix)
VERIFIED: <one-line reason citing specific cell(s), caption, or naming the tautology>
FABRICATED: <one-line counter-example or "no evidence in table">
"""


def verify_causal_skill(claim, recap, df, columns, sample_data, table_md, question, llm):
    """Per-claim LLM judge for Causal Analysis claims (paper §4.2 atomic skill).

    Paper-faithful: per-claim, not global. Distinct from TrustTable Challenger
    which acts on the entire CoT post-pipeline. Here, only claims tagged
    `Causal` by the decomposer are routed here, and each is judged individually.
    """
    try:
        prompt = CAUSAL_CHECK_PROMPT.format(
            question=question[:500], claim=claim[:400], recap=recap[:400],
            table=table_md[:5000],
        )
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, timeout=60.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Robust parse: scan for the FIRST verdict prefix anywhere in the text.
        # This avoids parsing bugs where the LLM writes "FABRICATED: ..." but
        # then in the same line says it's actually evidenced.
        upper = text.upper()
        v_idx = upper.find("VERIFIED")
        f_idx = upper.find("FABRICATED")
        # If both appear, the EARLIER occurrence is the verdict.
        if v_idx == -1 and f_idx == -1:
            return "UNV", f"causal_no_decision: {text[:150]}"
        if v_idx == -1:
            verdict = "FABRICATED"
        elif f_idx == -1:
            verdict = "VERIFIED"
        else:
            verdict = "VERIFIED" if v_idx < f_idx else "FABRICATED"
        if verdict == "FABRICATED":
            return "FALSE", f"causal_FABRICATED: {text[:150]}"
        return "TRUE", f"causal_VERIFIED: {text[:150]}"
    except Exception as e:
        return "UNV", f"causal_exec_err:{str(e)[:60]}"


def verify_skill(skill, claim, recap, df, columns, sample_data, table_md, question, llm):
    if skill == "Lookup":
        return verify_lookup(claim, recap, df, columns, sample_data, table_md, question, llm)
    if skill == "Arithmetic":
        return verify_arithmetic_skill(claim, recap, df, columns, sample_data, table_md, question, llm)
    if skill == "Compare":
        return verify_compare_skill(claim, recap, df, columns, sample_data, table_md, question, llm)
    if skill == "Causal":
        return verify_causal_skill(claim, recap, df, columns, sample_data, table_md, question, llm)
    # Filter, Aggregate, FinalAnswer all go through verify_inference (Pandas).
    return verify_inference_skill(claim, recap, df, columns, sample_data, table_md, question, llm,
                                  skill_label=skill)


# ----- Pipeline driver ------------------------------------------------------
# Strict UNV-helpless gate: fire only when verifier is *completely* helpless —
# zero TRUE checks AND at least 3 UNVs. Verified zero-T1-false-reject on WTQ102.
UNV_HELPLESS_TRUE_MAX = 0   # gate fires only if TRUE count == 0
UNV_HELPLESS_UNV_MIN = 3    # gate fires only if UNV count >= this


def run_atomic_skills(llm, df, table_md, question, cot, claimed):
    columns = list(df.columns)
    sample_data = str(df.head(3).to_dict(orient="records"))[:1500]

    # ---------- Programmatic pre-check: aggressive arithmetic scan ----------
    # Bypasses the decomposer (which may "silently correct" wrong sums).
    # Verified zero T1 false-reject risk on WTQ102.
    arith_errors = _scan_cot_arithmetic_errors(cot)
    if arith_errors:
        lhs, rhs, computed = arith_errors[0]
        return "REJECT", (f"cot_arithmetic_FALSE: '{lhs.strip()} = {rhs.strip()}' "
                          f"but actual={computed:g}")

    # ---------- Stage 1 — skill-typed decomposition ----------
    d_prompt = DECOMPOSE_PROMPT.format(
        question=question, cot=cot, claimed=claimed, max_claims=MAX_CLAIMS,
    )
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": d_prompt}],
            temperature=0.0, timeout=60.0,
        )
        items = parse_json_skill_array(resp.choices[0].message.content or "")
    except Exception as e:
        return "ACCEPT", f"decompose_error: {str(e)[:100]}"
    if not items:
        return "ACCEPT", "decompose_failed"
    items = items[:MAX_CLAIMS]

    # Always append a FinalAnswer claim — catches T4_answer_perturb where the
    # reasoning is correct but the stated answer was replaced.
    if claimed:
        final_claim = f"The answer to the question \"{question}\" is \"{claimed}\""
        already_has_final = any(s == "FinalAnswer" for s, _, _ in items)
        if not already_has_final:
            items.append(("FinalAnswer", final_claim, final_claim))

    # Append a Chain-Conclusion-Coherence claim: extract the last numeric value
    # the CoT itself states, then assert "CoT's stated chain conclusion == claimed".
    # Catches the (Z-, A+) case where CoT's reasoning chain ends at value N but
    # the stated final answer is a DIFFERENT value M (CoT internally inconsistent).
    if claimed:
        chain_concl = _extract_chain_conclusion(cot)
        if chain_concl is not None:
            coh_claim = (f"The CoT's reasoning chain ends at value '{chain_concl}', "
                         f"which is consistent with the stated final answer '{claimed}'")
            items.append(("Arithmetic", coh_claim, coh_claim))

    # ---------- Stage 2 — dispatch per skill ----------
    table_window = table_md[:TABLE_CHAR_CAP]
    verdicts = []
    for idx, (skill, claim, recap) in enumerate(items):
        verdict, reason = verify_skill(skill, claim, recap, df, columns, sample_data,
                                       table_window, question, llm)
        verdicts.append((idx + 1, skill, claim, verdict, reason))
        if verdict == "FALSE":
            break  # one refutation is enough

    # ---------- Stage 3 — aggregate ----------
    falses = [v for v in verdicts if v[3] == "FALSE"]
    if falses:
        n, skill, claim, _, reason = falses[0]
        return "REJECT", f"claim{n}_{skill}_FALSE: {claim[:120]!r} | {reason[:140]}"

    # Strict UNV-helpless gate: zero TRUEs AND ≥3 UNVs → no programmatic
    # evidence at all → don't ACCEPT on faith.
    n_unv = sum(1 for v in verdicts if v[3] == "UNV")
    n_true = sum(1 for v in verdicts if v[3] == "TRUE")
    n_total = len(verdicts)
    if n_true <= UNV_HELPLESS_TRUE_MAX and n_unv >= UNV_HELPLESS_UNV_MIN:
        return "REJECT", (f"unv_helpless_no_evidence: total={n_total}, "
                          f"true={n_true}, unv={n_unv}")

    return "ACCEPT", f"all_{n_total}_skills_passed (true={n_true}, unv={n_unv})"


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

        decision, rationale = run_atomic_skills(llm, df, table_md, question, cot, claimed)
        return {
            "id": item_id,
            "target_type": KEY_MAPPING.get(sample_key, "unknown"),
            "specific_subtype": sample_key,
            "verifier_decision": decision,
            "verifier_rationale": rationale[:500],
            "source": "atomic_skills_hybrid",
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
    for f in tqdm_asyncio.as_completed(tasks, desc="Atomic Skills (hybrid)"):
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
