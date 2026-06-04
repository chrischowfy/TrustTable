import os
import re
from typing import Tuple, Optional
import pandas as pd

# P1: Internal arithmetic self-consistency (imported by _check_internal_arithmetic_chain)
from src.schema import CoTTrace, VerificationResult, ReasoningStep
from src.verifiers.fact_checker import FactChecker
from src.verifiers.z3_auditor import Z3Auditor
from src.verifiers.consistency_monitor import ConsistencyMonitor
from src.llm_engine import LLMEngine
from utils.logger import setup_logger
from utils.sandbox import build_exec_globals

logger = setup_logger("TrustTablePipeline")

# Ablation toggles (Appendix Table 3 Panel A) — set via env vars.
# Defaults: all modules ACTIVE.
ABLATE_FC = os.environ.get("ABLATE_FC", "0") == "1"     # disable FactChecker (fact steps auto-ACCEPT)
ABLATE_Z3 = os.environ.get("ABLATE_Z3", "0") == "1"     # disable Z3Auditor (logic steps auto-ACCEPT after Pandas fallback)
ABLATE_CM = os.environ.get("ABLATE_CM", "0") == "1"     # disable ConsistencyMonitor (skip final A_exec vs A_text check)
ABLATE_PI2 = os.environ.get("ABLATE_PI2", "0") == "1"   # disable PandasInference 2nd-opinion
if ABLATE_FC or ABLATE_Z3 or ABLATE_CM or ABLATE_PI2:
    logger.warning(f"ABLATION MODE active: FC={ABLATE_FC} Z3={ABLATE_Z3} CM={ABLATE_CM} PI2={ABLATE_PI2}")


def _num(s: str) -> Optional[float]:
    """Parse a numeric token, tolerating commas, $/%, surrounding punctuation."""
    s = s.strip().rstrip('.,;:)}').lstrip('({$')
    s = s.replace(',', '').replace('$', '').rstrip('%')
    try:
        return float(s)
    except ValueError:
        return None


def _check_internal_arithmetic_chain(cot_text: str) -> Optional[Tuple[bool, str]]:
    """Scan the entire CoT for explicit arithmetic equations and running sums;
    flag any internal inconsistency.

    Returns:
        (True,  msg)  — found >=1 arithmetic assertion, all check out
        (False, msg)  — found a concrete internal arithmetic error
        None          — no verifiable arithmetic found (nothing to assert)

    Handles two forms often produced by T2_arith / T4_calc CoTs:
      a) Explicit equations: "1+1+3 = 6", "44963 + 164134 = 209097"
      b) Stepwise running sum: "150+200=350; 350+150=500; 500+50=550"
      c) Count enumeration: "1, 2, 3, 4 — 4 entries" vs claimed "5"

    Intentionally CONSERVATIVE: only flags when arithmetic is unambiguous.
    Pattern requires both operands AND the asserted result explicit in the text.
    """
    if not cot_text:
        return None

    # Strip thousand separators inside a numeric token but keep decimals intact
    text = cot_text

    # Form (a) + (b): explicit equation "num op num (op num)* = result",
    # with support for CHAIN equations "A = B = C".
    # Use explicit number-and-operator structure to avoid non-greedy mis-matches.
    # NUM: strict digit-groups (thousand-separated or plain), optional decimal.
    # Disallows trailing comma (prevents gobbling "13, +6" across running-sum clauses).
    NUM = r'[\+\-]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[\+\-]?\$?\d+(?:\.\d+)?'
    OP = r'[\+\-\*/×÷]'
    chain_pattern = re.compile(
        rf'(?<!\d)(?:{NUM})(?:\s*{OP}\s*(?:{NUM}))+(?:\s*=\s*(?:{NUM})(?:\s*{OP}\s*(?:{NUM}))*)+(?!\d)'
    )

    def _safe_eval(e):
        e_norm = e.replace('×', '*').replace('÷', '/').replace(',', '').replace('$', '').strip()
        if not re.fullmatch(r'[\d\s\+\-\*/\.]+', e_norm):
            return None
        try:
            return eval(e_norm, {"__builtins__": {}}, {})
        except Exception:
            return None

    errors = []
    confirmed = 0
    for m in chain_pattern.finditer(text):
        chain = m.group(0)
        parts = [p.strip() for p in chain.split('=')]
        if len(parts) < 2:
            continue
        first_val = _safe_eval(parts[0])
        last_val = _safe_eval(parts[-1])
        if first_val is None or last_val is None:
            continue
        tol = 1e-6 * max(1.0, abs(last_val)) + 0.01
        if abs(first_val - last_val) > tol:
            errors.append(f"'{parts[0]} = {parts[-1]}' (actual {first_val})")
        else:
            confirmed += 1

    # Form (c): count enumeration followed by a different asserted count.
    # "one, two, three, four — thus 5 individuals" / "1, 2, 3 — that's 4"
    count_pattern = re.compile(
        r'(?:'
        r'(?:one|two|three|four|five|six|seven|eight|nine|ten)'
        r'(?:\s*,?\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)){1,}'
        r'|(?:\d+)(?:\s*,\s*\d+){2,}'
        r')\s*[—\-–,\.]*\s*(?:that\'?s|thus|making|totalling|totaling|gives|therefore,?\s*the\s+total\s+is|=)\s+(\d+)',
        re.I,
    )
    word2num = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,
                'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,
                'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20}
    for m in count_pattern.finditer(text):
        chunk = m.group(0)
        claimed = _num(m.group(1))
        if claimed is None:
            continue
        # Extract items in the enumeration
        words = re.findall(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b', chunk[:-len(m.group(1))-2].lower())
        digits = re.findall(r'\d+', chunk[:-len(m.group(1))-2])
        if words:
            enumerated_count = len(words)
            # Also verify the words form a strict ascending sequence 1,2,3,...
            seq = [word2num[w] for w in words]
            if seq != list(range(1, len(seq)+1)):
                continue  # not a clean count enumeration
        elif len(digits) >= 3:
            enumerated_count = len(digits)
            seq = [int(d) for d in digits]
            if seq != list(range(1, len(seq)+1)):
                continue
        else:
            continue
        if enumerated_count != int(claimed):
            errors.append(f"enumerated {enumerated_count} items but claimed {int(claimed)}")
        else:
            confirmed += 1

    # Form (d): "makes/totals/gives X" followed later by "answer is Y" (X != Y).
    # Captures T2_arith signature where CoT's internal tally contradicts the stated answer.
    # e.g. "That makes 3 wins. So the answer is 4."
    # Guards:
    #   - The captured "N" must be a STANDALONE result (not start of an expr like "gives 30 + 11").
    #   - Tally and stated answer must be within 120 chars (same narrative unit).
    #   - Stated answer connector must be specific ("the answer is / final answer / the total is"),
    #     not any bare "is" — avoids false matches across unrelated sentences.
    internal_vs_stated = re.compile(
        r'(?:that\s+)?(?:makes?|totals?|totalling|totaling|gives?|summing\s+these\s+gives|adds?\s+up\s+to|sums?\s+to)'
        r'\s+(\d+)\b(?!\s*[\+\-\*/×÷=])'   # internal tally: must NOT be start of expr
        r'[^.]{0,40}'
        r'[\.;]'
        r'[^.]{0,80}?(?:so\s+the\s+(?:answer|total)\s+is|therefore,?\s+the\s+(?:answer|total)\s+is|'
        r'thus,?\s+the\s+(?:answer|total)\s+is|hence,?\s+the\s+(?:answer|total)\s+is|'
        r'final\s+answer\s+is|the\s+answer\s+is)'
        r'[^.]*?\b(\d+)\b(?!\s*[\+\-\*/×÷=])',   # stated answer: same guard
        re.I,
    )
    for m in internal_vs_stated.finditer(text):
        internal = int(m.group(1))
        stated = int(m.group(2))
        if internal != stated:
            errors.append(f"internal tally {internal} contradicts stated answer {stated}")
        else:
            confirmed += 1

    if errors:
        return (False, f"Internal arithmetic error: {errors[0]}")
    if confirmed > 0:
        return (True, f"Internal arithmetic verified ({confirmed} checks).")
    return None


def _try_simple_arithmetic(content: str) -> Optional[bool]:
    """Verify arithmetic claims by Python-evaluating the full expression.

    Handles multi-term expressions like '25+20+16+13+7+6+5+1 = 93'
    and '106.3 - 89.7 = 16.5' and '$150,000 + $200,000 = $350,000' by
    capturing the full LHS (numbers + operators) and evaluating in Python,
    then comparing to the RHS. Supports thousand separators and $ prefix.

    Returns:
        True  - expression evaluates == claimed
        False - expression evaluates != claimed
        None  - no verifiable equation present (fall through to Pandas)
    """
    NUM = r'[\+\-]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    OP = r'[\+\-\*/×÷]'
    eq_pattern = re.compile(
        rf'(?<!\d)({NUM}(?:\s*{OP}\s*{NUM})+)\s*=\s*({NUM})(?!\d)'
    )
    confirmed = 0
    for m in eq_pattern.finditer(content):
        expr_raw, claimed_raw = m.group(1), m.group(2)
        try:
            expr_norm = (expr_raw.replace('×', '*').replace('÷', '/')
                                 .replace(',', '').replace('$', '').strip())
            if not re.fullmatch(r'[\d\s\+\-\*/\.]+', expr_norm):
                continue
            actual = eval(expr_norm, {"__builtins__": {}}, {})
            claimed = float(claimed_raw.replace(',', '').replace('$', ''))
            tol = 1e-6 * max(1.0, abs(claimed)) + 0.01
            if abs(float(actual) - claimed) > tol:
                return False
            confirmed += 1
        except Exception:
            continue

    if confirmed:
        return True

    c = content.lower()
    nl_patterns = [
        r'[Aa]dd\s+([\d.,]+)\s+and\s+([\d.,]+)\s+to\s+get\s+(?:a\s+)?(?:[\w\s]*?)([\d.,]+)',
        r'[Ss]um\s+(?:of\s+)?([\d.,]+)\s+and\s+([\d.,]+)\s+is\s+([\d.,]+)',
        r'([\d.,]+)\s+and\s+([\d.,]+)\s+total(?:ing)?\s+([\d.,]+)',
    ]
    for pat in nl_patterns:
        m2 = re.search(pat, content)
        if m2:
            try:
                a = float(m2.group(1).replace(',', ''))
                b = float(m2.group(2).replace(',', ''))
                rr = float(m2.group(3).replace(',', ''))
                if abs(a + b - rr) < 0.01:
                    return True
                return False
            except ValueError:
                pass
    return None



class TrustTablePipeline:
    def __init__(self, table_df: pd.DataFrame, enable_consistency_monitor: bool = True):
        self.table = table_df
        # Ablation: env var ABLATE_CM=1 overrides the constructor flag
        if ABLATE_CM:
            enable_consistency_monitor = False
        self.fact_checker = FactChecker(table_df)
        self.z3_auditor = Z3Auditor(table_df)
        self.enable_cm = enable_consistency_monitor
        self.consistency_monitor = ConsistencyMonitor(table_df) if enable_consistency_monitor else None
        self.llm = LLMEngine()

    def _pandas_inference_fresh_attempt(self, step: ReasoningStep, question: str,
                                         columns, sample_row, table_ctx):
        """Independent (no-feedback) verification attempt. Returns True/False/None
        (None on exec error or missing function)."""
        try:
            code = self.llm.generate_inference_check(
                step.content, columns, str(sample_row), table_ctx,
                question=question, error_feedback=""
            )
            exec_globals = build_exec_globals({'pd': pd, 're': re})
            exec_locals = {}
            exec(code, exec_globals, exec_locals)
            if 'verify_inference' not in exec_locals:
                return None
            return bool(exec_locals['verify_inference'](self.fact_checker.exec_table))
        except Exception as e:
            logger.info(f"PandasInference 2nd opinion errored: {e}")
            return None

    def _verify_inference_with_pandas(self, step: ReasoningStep, question: str,
                                       max_retries: int = 2) -> Optional[VerificationResult]:
        """Try to verify inference step with Pandas, with Tool Repair on failure.

        On consistent False after tool-repair, runs an independent fresh attempt
        (no error feedback) as a second opinion. Disagreement → ABSTAIN (return
        None) to fall through to Z3, avoiding LLM-code-bug false rejections on
        complex tables. Unanimous False across tool-repair + fresh → REJECT.
        """
        columns = self.fact_checker.clean_columns
        sample_row = self.fact_checker.exec_table.head(3).to_dict(orient='records')
        table_ctx = self.fact_checker.exec_table.to_csv(sep="|", index=False)

        last_error = None
        last_result_info = None

        for attempt in range(max_retries + 1):
            # Build feedback for retry
            if last_result_info:
                feedback = (f"Your verify_inference code returned False, but this may be a code bug. "
                            f"Previous error context: {last_result_info}. "
                            f"Please re-examine the data extraction logic and try again.")
            elif last_error:
                feedback = f"Previous attempt raised an error: {last_error}. Please fix."
            else:
                feedback = ""

            code = self.llm.generate_inference_check(
                step.content, columns, str(sample_row), table_ctx,
                question=question, error_feedback=feedback
            )

            try:
                exec_globals = build_exec_globals({'pd': pd, 're': re})
                exec_locals = {}
                exec(code, exec_globals, exec_locals)

                if 'verify_inference' not in exec_locals:
                    last_error = "No verify_inference function generated"
                    last_result_info = None
                    continue

                result = exec_locals['verify_inference'](self.fact_checker.exec_table)

                if result:
                    return VerificationResult(True, "PandasInference", "Inference verified via Pandas.")
                else:
                    # Tool Repair: rejection might be code bug
                    if attempt < max_retries:
                        logger.info(f"PandasInference rejected (attempt {attempt+1}), repairing code...")
                        last_result_info = f"Code returned False for claim: '{step.content[:200]}'"
                        last_error = None
                        continue
                    else:
                        # End of tool-repair chain; consult independent 2nd opinion
                        # before concluding reject (handles sticky LLM-code-gen bugs).
                        fresh = self._pandas_inference_fresh_attempt(
                            step, question, columns, sample_row, table_ctx)
                        if fresh is True:
                            logger.info("PandasInference 2nd opinion disagreed (True) → ABSTAIN → Z3")
                            return None
                        if fresh is None:
                            logger.info("PandasInference 2nd opinion errored → ABSTAIN → Z3")
                            return None
                        # fresh is False: unanimous → confident reject
                        logger.info("PandasInference 2nd opinion confirmed False → REJECT")
                        return VerificationResult(
                            False, "PandasInference",
                            "Inference contradicted by table data (confirmed by 2nd opinion).")

            except Exception as e:
                last_error = str(e)
                last_result_info = None
                if attempt < max_retries:
                    logger.info(f"Pandas inference exec failed (attempt {attempt+1}), retrying: {e}")
                else:
                    logger.info(f"Pandas inference check failed: {e}, falling back to Z3")
                    return None  # Fall through to Z3

        return None  # All retries exhausted with errors

    def run(self, trace: CoTTrace) -> Tuple[bool, Optional[dict]]:
        logger.info(f"Starting verification pipeline for Q: {trace.question}")

        verified_facts = []

        for i, step in enumerate(trace.steps):
            logger.info(f"--- Verifying Step {step.step_id} [{step.step_type.upper()}] ---")

            res = VerificationResult(True, "Pipeline", "Skipped")

            # Method alignment check (from decompose)
            if not step.aligned:
                logger.warning(f">>> REJECTED at Step {step.step_id}: method misalignment (from decompose)")
                return False, {
                    "step_index": step.step_id,
                    "step_content": step.content,
                    "module": "Decomposer",
                    "reason": "Semantic Mismatch: Step uses wrong column/method (detected during decomposition)."
                }

            if step.step_type == "fact":
                if ABLATE_FC:
                    res = VerificationResult(True, "FactChecker[ABLATED]", "Ablation: FactChecker disabled, step auto-ACCEPT.")
                else:
                    res = self.fact_checker.verify(step, context=verified_facts, question=trace.question)

            elif step.step_type in ("logic", "inference"):
                # All inference steps → try Pandas first, fallback to Z3
                arith = _try_simple_arithmetic(step.content)
                if arith is True:
                    res = VerificationResult(True, "ArithCheck", "Arithmetic verified.")
                elif arith is False:
                    res = VerificationResult(False, "ArithCheck", "Arithmetic error.")
                else:
                    # Try Pandas first (more reliable for data operations)
                    pandas_res = self._verify_inference_with_pandas(step, trace.question)
                    if pandas_res is not None:
                        res = pandas_res
                    elif ABLATE_Z3:
                        # Ablation: no Z3 fallback; auto-accept inference we can't verify via Pandas
                        res = VerificationResult(True, "Z3Auditor[ABLATED]", "Ablation: Z3 disabled, Pandas-abstain auto-ACCEPT.")
                    else:
                        # Pandas failed → fall back to Z3
                        res = self.z3_auditor.verify(step, context=verified_facts)

            else:
                # Unknown type → try Z3 (unless ablated)
                if ABLATE_Z3:
                    res = VerificationResult(True, "Z3Auditor[ABLATED]", "Ablation: Z3 disabled, unknown-type step auto-ACCEPT.")
                else:
                    res = self.z3_auditor.verify(step, context=verified_facts)

            if not res.is_valid:
                logger.warning(f">>> REJECTED at Step {step.step_id}")
                return False, {
                    "step_index": step.step_id,
                    "step_content": step.content,
                    "module": res.component,
                    "reason": res.reason
                }

            verified_facts.append(step)

        # Consistency Monitor
        if self.enable_cm and self.consistency_monitor and trace.final_answer:
            cm_res = self.consistency_monitor.check(trace.question, trace.final_answer)
            if not cm_res.is_valid:
                logger.warning(">>> REJECTED by Consistency Monitor")
                return False, {
                    "step_index": -1,
                    "step_content": f"Final Answer: {trace.final_answer}",
                    "module": cm_res.component,
                    "reason": cm_res.reason
                }

        logger.info(">>> PIPELINE VERIFICATION PASSED")
        return True, None
