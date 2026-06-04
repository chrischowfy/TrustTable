"""
Pipeline + LLM Challenger (Adversarial Flaw Hunter)
====================================================
Variant of run_pipeline_with_llm_confirm.py with an ADVERSARIAL prompt:
  1. Pipeline decides first (with Tool Repair, FC2 fuzzy match)
  2. If Pipeline REJECTS → final REJECT
  3. If Pipeline ACCEPTS → LLM Challenger tries to find a flaw
     - FLAW_FOUND → final REJECT (plug T2_logic semantic leaks)
     - NO_FLAW    → final ACCEPT

Difference from llm_confirm: Challenger prompt is biased toward finding flaws
instead of confirming. Targets Type2_logic_error blindspot (fabricated rules).
Must use TRUST_K=0 (refinement is hazardous for diagnostic eval).

Usage:
    python run_pipeline_main.py [input_file] [output_file]
"""

import json, os, sys, asyncio
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from src.llm_engine import LLMEngine
from src.schema import CoTTrace, ReasoningStep
from src.pipeline import TrustTablePipeline
from utils.logger import setup_logger
from utils.table_utils import parse_structured_table

logger = setup_logger("PipelineChallenger")

INPUT_FILE = "../data/small/panel_c_wtq/type1_correct.json"
OUTPUT_FILE = "../outputs/trusttable_wtq_type1.json"
CONCURRENCY = int(os.environ.get("TRUST_CONCURRENCY", "10"))
SAVE_INTERVAL = 50

KEY_MAPPING = {
    "type1_correct": "type1_golden",
    "type2_grounding_error": "type2_spurious",
    "type2_arithmetic_error": "type2_spurious",
    "type2_logic_error": "type2_spurious",
    "type3_fully_wrong": "type3_fully_wrong",
    "type4_calc_error": "type4_calc_error",                  # hard (Appendix H Prompt D)
    "type4_answer_perturb": "type4_inconsistent_easy",       # easy (§B.2, paper DIR_inc=99.5%)
}


def load_input(path):
    p = Path(path)
    if p.is_dir():
        data = []
        for json_path in sorted(p.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                chunk = json.load(f)
            if not isinstance(chunk, list):
                raise ValueError(f"{json_path} must contain a JSON list")
            data.extend(chunk)
        return data
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def result_key(result):
    return (result.get("id", "unknown"), result.get("specific_subtype", "unknown"))


def extract_cot_text(sd):
    return (sd.get("chain_of_thought") or sd.get("flawed_chain_of_thought")
            or sd.get("correct_logic_wrong_math_cot") or sd.get("incorrect_chain_of_thought") or "")


def extract_claimed_answer(sd):
    return sd.get("answer") or sd.get("incorrect_answer") or sd.get("pred_answer") or ""


CHALLENGER_SYSTEM_PROMPT = """You are a CRITICAL adversarial reviewer for TableQA reasoning chains.

The reasoning has already passed symbolic fact/arithmetic/consistency checks. Your job is to hunt for flaws that symbolic verifiers CANNOT detect. Check every category below:

### A. Fabrication flaws (old Challenger scope)
1. **Fabricated rules**: CoT invents a "standard" or "convention" not present in the table (e.g., "in professional tennis data, (i) stands for 'in Russia'").
2. **Term redefinition**: CoT silently redefines a term from the question (e.g., redefining "first" as "top-3 finish").
3. **Invalid inference**: the conclusion doesn't follow from the cited facts even though individual facts are correct.
4. **Hidden assumptions**: CoT relies on external "statistical patterns" or "industry standards" that are made up.

### B. Logical fallacies (NEW — CoT facts are correct but reasoning structure is unsound)
5. **Dropped conjunct**: Q requires "A AND B" (two or more independent conditions) but CoT only verifies A (or only B). Also fires when Q says "exactly / both / each / only if" and CoT silently drops a clause.
   - Check: LIST every condition mentioned in the Question. For each, verify CoT explicitly addresses it. If any clause is missing from CoT's reasoning, FLAW.
6. **Wrong negation scope / quantifier flip**: "no X is Y" is NOT equivalent to "all X are not Y" in CoT's reading. "not all" is NOT equivalent to "all not". "∃" is NOT "∀".
   - Check: when Q contains negation + quantifier, verify CoT's proof uses the correct scope.
7. **Necessary vs sufficient confusion**: "A is necessary for B" means "¬A → ¬B" (i.e., if not A then not B); it does NOT mean "A → B". Similarly, "sufficient" means "A → B" but not the reverse.
   - Check: when Q asks necessity/sufficiency/"only if"/"if and only if", verify CoT tests the correct implication direction.
8. **Boundary off-by-one / operator swap**: "exactly N" ≠ "≥N"; "more than N" ≠ "≥N" (i.e. "> N" vs "≥ N"); "between A and B" may or may not include endpoints depending on wording.
   - Check: when Q specifies a numeric boundary or comparison, verify CoT applies the exact operator (>, ≥, <, ≤, ==, !=).
9. **Selection-procedure mismatch (positional shortcut)**: when Q asks for a superlative defined by VALUE (largest/smallest/highest/lowest/maximum/minimum/most/least + a numeric metric), the CoT must justify the choice by VALUE COMPARISON, not by table position.
   - FLAW signals: CoT contains phrases like "the first listed", "in order", "first item", "last item", "first one", "in the table order", "first row", "last row" *as the basis for picking the extremum* (without showing the values support it).
   - NOT a flaw: CoT just states "the largest is X at $N" then visibly lists or ranks the values; or table happens to be sorted and CoT explicitly verifies value rank.
10. **Subset / scope qualifier drop**: when Q includes a category qualifier ("operating", "non-operating", "current", "non-current", "individual line items", "excluding totals/subtotals", "before tax", "after tax", a specific section header), CoT must operate on EXACTLY the rows that match that qualifier.
    - FLAW signals: CoT (a) includes a row that doesn't match the qualifier (e.g. counts Cost of revenue as an "operating expense" when the income statement breaks it out separately above operating expenses); (b) excludes a row that does match (e.g. drops Interest expense and Income taxes when Q says "expense lines" generically); (c) includes a Total/subtotal row when Q says "individual line items"; (d) ignores the qualifier and operates on a broader set.
    - Check: list each row CoT used; for each, confirm it satisfies every qualifier in Q.
11. **Direction-of-comparison flip (sign error)**: when Q phrases the answer as a directional difference ("by how much does A exceed B", "the amount A is above B", "X minus Y"), CoT must compute A − B (not B − A, not |A − B|).
    - FLAW signals: CoT computes the absolute value of the difference; CoT subtracts in reverse order and reports the magnitude as if it were the directional answer; CoT silently flips operands.

### Evaluation discipline (CRITICAL)
- ONLY flag a flaw if you can POINT TO a SPECIFIC claim (fabrication) or a SPECIFIC missing/misapplied logical operator (fallacy) and EXPLAIN why.
- DO NOT flag minor phrasing, word choice, or interpretation differences.
- DO NOT flag reasoning that is substantively faithful to the question AND grounded in real table data.
- If you are flagging a logical fallacy (categories 5–11), you MUST quote (a) the exact Question fragment that imposes the constraint and (b) the exact CoT fragment that violates or omits it.
- If the reasoning substantively answers the question and is grounded in the table, answer NO_FLAW.

### ANTI–FALSE POSITIVE rules (MUST follow before flagging)
- **Arithmetic**: do NOT flag a numeric step as flawed unless you have RECOMPUTED it
  yourself and confirmed the CoT's number is wrong. Differences within ±0.01 (or
  within rounding precision of the claim, e.g. 0.5×10⁻ᵈ for d-decimal claims) are
  NOT flaws — they are normal floating-point or rounding behavior.
- **Format equivalences are not flaws**: `($X)` ≡ `-$X`; `$1,234` ≡ `1234`; "—" /
  blank in numeric columns ≡ 0 in analyst memos. Do NOT flag CoTs for using these
  conventions.
- **Plausible alternative methods are not flaws**: if the question can be answered
  multiple equivalent ways (e.g., "three smallest = total minus three largest"
  vs direct sort), the CoT's choice is not a flaw as long as it lands on the gold.
- **Implicit-but-correct reasoning is not a flaw**: if the CoT skips an obvious
  intermediate step (e.g., "the Total row sums to X" without re-deriving it from
  components) but the assertion is verifiable from the table, do NOT flag.

### Output format (MUST end with one of these)
If you find a concrete flaw, quote the offending claim or missing/misapplied operator and end with:
FLAW_FOUND: <one sentence, category A1-A4 or B5-B11 + specific reason>

If the reasoning is substantively faithful, end with:
NO_FLAW
"""


def llm_challenge(llm, table_str, question, reasoning, answer):
    """Adversarial LLM review — only called when Pipeline ACCEPTs."""
    user_prompt = f"""### Table Context
{table_str}

### Question
{question}

### Reasoning Trace
{reasoning}

### Predicted Answer
{answer}

### Required Workflow (follow in order)

**Step 1 — Question Constraint Enumeration**
Read the Question and list EVERY logical constraint it imposes. Be exhaustive. Constraints fall into these classes:
  - Conjunctions: "X AND Y", "both", "and also", multiple required conditions joined by "and"
  - Disjunctions: "X OR Y", "either", "at least one of"
  - Negations / scope: "no", "not", "none", "without", "every X is not Y"
  - Quantifiers: "all", "every", "any", "exactly N", "at least N", "more than N"
  - Boundaries: distinguish "more than N" (> N) from "at least N" (>= N) from "exactly N"
  - Implications: "if X then Y", "necessary for", "sufficient for", "only if"
  - Comparators / superlatives: "highest", "most", "the only X with property Y"
  - Category qualifiers / scope: "operating" vs "non-operating", "current" vs "non-current", "individual line items" vs "totals/subtotals", "before-tax" vs "after-tax", section-specific qualifiers
  - Directional comparison: "by how much does A exceed B" requires A − B, not |A − B|

Output as: "Q-Constraints: [C1: ..., C2: ..., C3: ..., ...]"

**Step 2 — CoT Coverage Audit**
For EACH constraint Ci you listed, check whether the Reasoning Trace EXPLICITLY verifies it (with the correct logical operator).
  - COVERED: CoT performs an operation that matches Ci's exact semantics.
  - MISSED:  CoT silently drops Ci, or substitutes a different operator (AND→OR, > → >=, every→some, necessary→sufficient, etc.)
  - For superlative constraints (largest/smallest/highest/lowest/maximum/minimum/most/least): COVERED only if CoT shows VALUE-BASED COMPARISON across all candidates. MISSED if CoT picks by table position ("the first listed", "in order", "first item", "last item", "first row", "last row") without verifying that ordering matches value rank.
  - For scope/qualifier constraints (operating/non-operating, current/non-current, individual/total): COVERED only if CoT enumerates rows that match the qualifier exactly. MISSED if CoT includes a row that doesn't match the qualifier, excludes a row that does, or pulls in totals/subtotals when "individual" is required.
  - For directional differences ("A exceeds B by N", "A minus B"): COVERED only if CoT computes A − B in the correct direction. MISSED if CoT computes B − A and reports the absolute value.

Output as a checklist: "C1: COVERED/MISSED [+ short evidence]"

**Step 3 — Fabrication Check**
Scan for invented rules / fabricated conventions / hidden domain assumptions per system-prompt categories A1–A4.

**Step 4 — Final Verdict**
- If ANY Ci is MISSED or substituted, end with FLAW_FOUND quoting the missed/substituted constraint.
- If a fabrication or invalid inference is found, end with FLAW_FOUND.
- Otherwise (all Q-constraints covered AND no fabrication), end with NO_FLAW.

Show Step 1 / Step 2 / Step 3 reasoning explicitly before the final verdict line."""

    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": CHALLENGER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}],
            temperature=0.0, timeout=60.0,
        )
        content = resp.choices[0].message.content or ""
        if "FLAW_FOUND" in content:
            return "REJECT", content
        if "NO_FLAW" in content:
            return "ACCEPT", content
        last = content.strip().split('\n')[-1].upper()
        if "FLAW" in last and "NO_FLAW" not in last:
            return "REJECT", content
        return "ACCEPT", content
    except Exception as e:
        logger.error(f"LLM challenge failed: {e}")
        return "UNKNOWN", f"challenger_error: {e}"


def verify_single_sample(item, sample_key, sample_data, llm):
    item_id = item.get("id", "unknown")
    try:
        table_content = item.get("table_content")
        if not table_content or not isinstance(table_content, dict):
            return None
        df = parse_structured_table(table_content)
        if df.empty:
            return None

        cot_text = extract_cot_text(sample_data)
        if not cot_text:
            return None
        claimed_answer = extract_claimed_answer(sample_data)
        question = item["original_question"]
        table_str = item.get("table_md", df.to_csv(sep="|", index=False))

        raw_steps = llm.decompose_cot(cot_text, question=question)
        steps = [ReasoningStep(step_id=i + 1, content=s['content'], step_type=s['type'],
                               aligned=s.get('aligned', True))
                 for i, s in enumerate(raw_steps)]
        if not steps:
            return None

        trace = CoTTrace(question=question, steps=steps,
                         final_answer=claimed_answer, raw_text=cot_text)
        pipeline = TrustTablePipeline(df)
        pipe_valid, pipe_error = pipeline.run(trace)

        if not pipe_valid:
            return {
                "id": item_id,
                "target_type": KEY_MAPPING.get(sample_key, "unknown"),
                "specific_subtype": sample_key,
                "verifier_decision": "REJECT",
                "verifier_rationale": f"Pipeline: {json.dumps(pipe_error, ensure_ascii=False)[:300]}",
                "source": "pipeline",
            }

        chal_decision, chal_rationale = llm_challenge(llm, table_str, question, cot_text, claimed_answer)

        if chal_decision == "REJECT":
            return {
                "id": item_id,
                "target_type": KEY_MAPPING.get(sample_key, "unknown"),
                "specific_subtype": sample_key,
                "verifier_decision": "REJECT",
                "verifier_rationale": f"Pipeline accepted but Challenger found flaw: {chal_rationale[:400]}",
                "source": "challenger",
            }
        if chal_decision == "UNKNOWN":
            return {
                "id": item_id,
                "target_type": KEY_MAPPING.get(sample_key, "unknown"),
                "specific_subtype": sample_key,
                "verifier_decision": "UNKNOWN",
                "verifier_rationale": f"Pipeline accepted; Challenger did not complete: {chal_rationale[:400]}",
                "source": "challenger_unknown",
            }

        return {
            "id": item_id,
            "target_type": KEY_MAPPING.get(sample_key, "unknown"),
            "specific_subtype": sample_key,
            "verifier_decision": "ACCEPT",
            "verifier_rationale": "Pipeline accepted; Challenger found no flaw.",
            "source": "both_accept",
        }
    except Exception as e:
        logger.error(f"[{item_id}-{sample_key}] {e}")
        return None


async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("LLM_API_KEY is required. Export it before running TrustTable evaluation.")
    data = load_input(INPUT_FILE)
    llm = LLMEngine()
    if os.environ.get("TRUST_SKIP_PREFLIGHT", "0") != "1":
        try:
            llm.preflight()
        except Exception as e:
            raise SystemExit(f"LLM preflight failed: {e}") from e

    results = []
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            completed = {result_key(r) for r in results if isinstance(r, dict)}
            print(f"Resuming from {len(completed)} completed results in {OUTPUT_FILE}")
        except Exception as e:
            raise SystemExit(f"Cannot resume from existing output file {OUTPUT_FILE}: {e}") from e

    task_args = []
    for item in data:
        gs = item.get("generated_samples", {})
        for sk in KEY_MAPPING:
            if sk not in gs: continue
            sd = gs[sk]
            if not isinstance(sd, dict) or "error" in sd: continue
            if (item.get("id", "unknown"), sk) in completed:
                continue
            task_args.append((item, sk, sd))

    print(f"{len(data)} items, {len(task_args)} tasks")
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()

    async def run_one(item, sk, sd):
        async with sem:
            res = await asyncio.to_thread(verify_single_sample, item, sk, sd, llm)
            if res is not None:
                return res
            return {
                "id": item.get("id", "unknown"),
                "target_type": KEY_MAPPING.get(sk, "unknown"),
                "specific_subtype": sk,
                "verifier_decision": "UNKNOWN",
                "verifier_rationale": "Sample could not be verified due to parsing or runtime failure.",
                "source": "runtime_unknown",
            }

    tasks = [asyncio.create_task(run_one(*a)) for a in task_args]
    for f in tqdm_asyncio.as_completed(tasks, desc="Pipeline + Challenger"):
        res = await f
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
