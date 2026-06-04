"""Step-wise LLM-as-Judge — Standard CoT baseline (canonical implementation)
==========================================================================
This is the canonical Standard CoT verifier as described in our paper:
  (1) Decompose the CoT into atomic steps using the shared Decomposer
      (with Decomposer Fidelity — self-consistent and faithful to the
      original CoT, no silent correction).
  (2) For each step, an LLM judges ACCEPT / REJECT against the table.
  (3) Aggregation: any step REJECT → overall REJECT.

There is NO final answer-vs-reasoning consistency check (no "stage 3").
Pure step-wise per-step LLM judgment is the design contract — the verifier
is structurally blind to T4_answer_perturb errors (where the reasoning is
correct but the stated final answer is replaced), which is the intended
limitation of this baseline.

Compared to monolithic `run_llm_judge.py`:
  monolithic:  1 LLM call  / item   (whole-CoT judgment)
  step-wise:   1 + N calls / item   (1 decompose + N per-step judgments)

Output records are compatible with eval_cot_verifier.py.
"""
import json, os, sys, asyncio
from tqdm.asyncio import tqdm_asyncio
from src.llm_engine import LLMEngine
from utils.logger import setup_logger

logger = setup_logger("StepwiseLLMJudge")

INPUT_FILE = "../data/small/panel_c_wtq/type1_correct.json"
OUTPUT_FILE = "../outputs/llm_judge_stepwise_wtq_type1.json"
CONCURRENCY = 10

KEY_MAPPING = {
    "type1_correct": "type1_golden",
    "type2_grounding_error": "type2_spurious",
    "type2_arithmetic_error": "type2_spurious",
    "type2_logic_error": "type2_spurious",
    "type3_fully_wrong": "type3_fully_wrong",
    "type4_calc_error": "type4_calc_error",
    "type4_answer_perturb": "type4_inconsistent_easy",
}

STEP_JUDGE_PROMPT = """You are a calibrated Logic Auditor verifying ONE atomic reasoning step against a Table. Be fair, not over-strict — minor paraphrase, formatting, and reasonable interpretation are NOT errors.

### Table
{table}

### Question (for context)
{question}

### Reasoning Step to verify (one atomic claim)
Step type: {step_type}
Step content: {step}

### Task
Is this ONE reasoning step correct based on the table?
- For 'fact' steps: does the step cite a value that actually appears in the table? Tolerate paraphrase ("Voted Out" ≡ "voted out"), partial enumeration of long lists, and equivalent formulations of the same value.
- For 'inference' steps: is the computation/aggregation correct? Tolerate ±1 off-by-one in counts when the qualifying criterion is genuinely ambiguous, and accept the step if the cited intermediate values appear in the table and the arithmetic is plausible.
- For 'logic' steps: is the deduction/comparison valid? Accept reasonable interpretations of question wording.

Decision policy: REJECT only if you have CLEAR, table-grounded evidence the step is wrong (a specific cited value disagrees with the table, an arithmetic result is off by more than rounding, or a logical claim contradicts the table). Otherwise ACCEPT. When in doubt, ACCEPT.

Respond with a short analysis, then end with exactly "STEP_JUDGMENT: ACCEPT" or "STEP_JUDGMENT: REJECT"."""

# NOTE: FINAL_ANSWER_JUDGE_PROMPT and judge_final() are no longer invoked.
# Kept here only as historical reference for the v6.2-with-stage3 ablation.
# The canonical Standard CoT baseline does NOT do a final consistency check —
# it is pure step-wise per-step LLM judgment.
_LEGACY_FINAL_ANSWER_JUDGE_PROMPT_UNUSED = """LEGACY_REFERENCE_ONLY"""


def _extract_reasoning_and_answer(sd):
    reasoning = (sd.get("chain_of_thought")
                 or sd.get("flawed_chain_of_thought")
                 or sd.get("correct_logic_wrong_math_cot")
                 or sd.get("incorrect_chain_of_thought") or "")
    answer = sd.get("answer") or sd.get("incorrect_answer") or ""
    return reasoning, answer


def judge_step(llm, step, question, table_str, timeout=45.0):
    """Return ACCEPT/REJECT for one atomic step."""
    prompt = STEP_JUDGE_PROMPT.format(
        table=table_str[:4000],
        question=question,
        step_type=step.get("type", "?"),
        step=step.get("content", ""),
    )
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "system", "content": "You are a strict per-step Logic Auditor."},
                      {"role": "user", "content": prompt}],
            temperature=0.0, timeout=timeout,
        )
        out = (resp.choices[0].message.content or "").upper()
        if "STEP_JUDGMENT: REJECT" in out: return "REJECT"
        if "STEP_JUDGMENT: ACCEPT" in out: return "ACCEPT"
        return "REJECT" if "REJECT" in out.split("\n")[-1] else "ACCEPT"
    except Exception as e:
        return f"ERR:{str(e)[:40]}"


def judge_one(llm, item, sample_key, sample_data):
    item_id = item.get("id", "?")
    try:
        table_str = item.get("table_md", "") or str(item.get("table_content", ""))[:2000]
        question = item.get("original_question", "")
        reasoning, answer = _extract_reasoning_and_answer(sample_data)
        if not reasoning:
            return None

        # Stage 1: decompose (same as TrustTable — shared DS-chat decomposer)
        try:
            steps = llm.decompose_cot(reasoning, question)
        except Exception as e:
            return {"id": item_id, "target_type": KEY_MAPPING.get(sample_key, "unknown"),
                    "specific_subtype": sample_key,
                    "verifier_decision": "REJECT",
                    "verifier_rationale": f"decompose_error: {e}"[:500],
                    "n_steps": 0, "source": "stepwise_llm_judge"}

        if not isinstance(steps, list) or not steps:
            return {"id": item_id, "target_type": KEY_MAPPING.get(sample_key, "unknown"),
                    "specific_subtype": sample_key,
                    "verifier_decision": "REJECT",
                    "verifier_rationale": "empty_decomposition",
                    "n_steps": 0, "source": "stepwise_llm_judge"}

        # Stage 2: per-step judgment
        step_verdicts = []
        first_reject_idx = None
        for i, s in enumerate(steps):
            v = judge_step(llm, s, question, table_str)
            step_verdicts.append(v)
            if v == "REJECT" and first_reject_idx is None:
                first_reject_idx = i
                # early-exit: one rejection is enough (fair: same as TrustTable)
                break

        if first_reject_idx is not None:
            failed = steps[first_reject_idx]
            return {"id": item_id, "target_type": KEY_MAPPING.get(sample_key, "unknown"),
                    "specific_subtype": sample_key,
                    "verifier_decision": "REJECT",
                    "verifier_rationale": f"step_{first_reject_idx}_rejected ({failed.get('type','?')}): {str(failed.get('content',''))[:200]}",
                    "n_steps": len(steps), "first_reject_idx": first_reject_idx,
                    "source": "stepwise_llm_judge"}

        # Pure step-wise: if all atomic steps pass, ACCEPT.
        # No final answer-vs-reasoning consistency check by design.
        return {"id": item_id, "target_type": KEY_MAPPING.get(sample_key, "unknown"),
                "specific_subtype": sample_key,
                "verifier_decision": "ACCEPT",
                "verifier_rationale": f"all_{len(steps)}_steps_pass",
                "n_steps": len(steps), "source": "stepwise_llm_judge"}

    except Exception as e:
        logger.error(f"[{item_id}-{sample_key}] {e}")
        return None


async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    llm = LLMEngine()

    task_args = []
    for item in data:
        gs = item.get("generated_samples", {})
        for sk in KEY_MAPPING:
            if sk not in gs: continue
            sd = gs[sk]
            if not isinstance(sd, dict) or "error" in sd: continue
            task_args.append((item, sk, sd))

    print(f"{len(data)} items, {len(task_args)} tasks (step-wise LLM-Judge)")
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    async def run_one(item, sk, sd):
        async with sem:
            return await asyncio.to_thread(judge_one, llm, item, sk, sd)

    tasks = [asyncio.create_task(run_one(*a)) for a in task_args]
    for f in tqdm_asyncio.as_completed(tasks, desc="Step-wise LLM-Judge"):
        res = await f
        if res: results.append(res)

    print(f"Saving {len(results)} results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) >= 2: INPUT_FILE = sys.argv[1]
    if len(sys.argv) >= 3: OUTPUT_FILE = sys.argv[2]
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
