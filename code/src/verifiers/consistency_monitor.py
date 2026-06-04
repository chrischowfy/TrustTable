"""
Consistency Monitor (Paper Section 4.4, Module 3)
Compares the deterministically computed answer (A_exec) with
the CoT's textual answer (A_text) to intercept Type 4 errors.
"""
import re
import pandas as pd
from src.verifiers.base import BaseVerifier
from src.schema import ReasoningStep, VerificationResult
from src.llm_engine import LLMEngine
from utils.logger import setup_logger
from utils.sandbox import build_exec_globals

logger = setup_logger("ConsistencyMonitor")


class ConsistencyMonitor(BaseVerifier):
    def __init__(self, table: pd.DataFrame):
        super().__init__(table)
        self.llm = LLMEngine()

    def verify(self, step: ReasoningStep, context: list) -> VerificationResult:
        """BaseVerifier interface — not used directly; use check() instead."""
        return VerificationResult(True, "ConsistencyMonitor", "N/A")

    def check(self, question: str, claimed_answer: str, max_retries: int = 1) -> VerificationResult:
        """
        Generate code to compute the answer, with TOOL REPAIR on mismatch.
        """
        if not claimed_answer or not claimed_answer.strip():
            return VerificationResult(True, "ConsistencyMonitor", "No claimed answer to check.")

        columns = self.table.columns.tolist()
        sample_data = str(self.table.head(3).to_dict(orient='records'))
        full_table_csv = self.table.to_csv(index=False)
        if len(full_table_csv) > 4000:
            full_table_csv = full_table_csv[:4000]
        unreliable = {'', 'none', 'nan', 'error', 'unknown', 'n/a', 'total'}

        last_feedback = ""
        for attempt in range(max_retries + 1):
            # Generate code (with repair feedback on retry)
            if last_feedback:
                code = self.llm.generate_answer_code(
                    f"{question}\n\n### REPAIR FEEDBACK\n{last_feedback}",
                    columns, sample_data, full_table=full_table_csv
                )
            else:
                code = self.llm.generate_answer_code(question, columns, sample_data, full_table=full_table_csv)

            try:
                exec_globals = build_exec_globals({'pd': pd, 're': re})
                exec_locals = {}
                exec(code, exec_globals, exec_locals)

                if 'compute_answer' not in exec_locals:
                    last_feedback = "No compute_answer function generated."
                    continue

                computed = str(exec_locals['compute_answer'](self.table))

                if self._normalize(computed) in unreliable:
                    logger.warning(f"CM computed unreliable value '{computed}'.")
                    return VerificationResult(False, "ConsistencyMonitor",
                                              f"Unable to verify answer: unreliable computed='{computed}'.")

                if self._answers_match(computed, claimed_answer):
                    logger.info(f"Consistency OK: computed='{computed}', claimed='{claimed_answer}'")
                    return VerificationResult(True, "ConsistencyMonitor",
                                              "Answer consistent with computation.")
                else:
                    # TOOL REPAIR: mismatch might be code bug
                    if attempt < max_retries:
                        last_feedback = (f"Your code computed '{computed}' but the expected answer is "
                                         f"'{claimed_answer}'. Check if your code reads the correct "
                                         f"row/column. The table has a 'Total' row if relevant.")
                        logger.info(f"CM mismatch (attempt {attempt+1}), repairing: "
                                    f"computed='{computed}' vs claimed='{claimed_answer}'")
                        continue
                    else:
                        logger.warning(f"Inconsistency (final): computed='{computed}' vs claimed='{claimed_answer}'")
                        return VerificationResult(
                            False, "ConsistencyMonitor",
                            f"Execution Inconsistency: computed '{computed}' but CoT claims '{claimed_answer}'"
                        )

            except Exception as e:
                if attempt < max_retries:
                    last_feedback = f"Code execution error: {e}. Please fix."
                    continue
                logger.warning(f"CM exec error: {e}")
                return VerificationResult(False, "ConsistencyMonitor",
                                          f"Unable to verify answer: execution error ({e})")

        return VerificationResult(False, "ConsistencyMonitor", "Unable to verify answer after retries.")

    # ------------------------------------------------------------------
    # Answer normalization & fuzzy matching
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(answer: str) -> str:
        answer = str(answer).strip().lower()
        # Remove common formatting noise
        for ch in [',', '$', '%', '"', "'", '\n', '\\n']:
            answer = answer.replace(ch, '')
        answer = re.sub(r'\s+', ' ', answer).strip()
        # Try numeric normalization
        try:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return f"{num:g}"
        except (ValueError, OverflowError):
            return answer

    def _answers_match(self, a: str, b: str) -> bool:
        na = self._normalize(a)
        nb = self._normalize(b)
        if not na or not nb:
            return False
        # Exact match
        if na == nb:
            return True
        # Numeric near-match (tolerant for rounding: 38.17 ≈ 38.2)
        try:
            na_num = float(na)
            nb_num = float(nb)
            if abs(na_num - nb_num) < 0.1:
                return True
            if abs(na_num) > 1 and abs(na_num - nb_num) / max(abs(na_num), 1e-9) < 0.005:
                return True
        except (ValueError, OverflowError):
            pass
        # Number word matching: "six" == "6"
        word2num = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
                    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
        na_w = word2num.get(na, na)
        nb_w = word2num.get(nb, nb)
        if na_w == nb_w:
            return True
        # Mixed unit answers: allow "16.6" == "16.6 FM", but do not let
        # unrelated entities match just because they share a number.
        a_nums = re.findall(r'[\d.]+', na)
        b_nums = re.findall(r'[\d.]+', nb)
        if a_nums and b_nums:
            try:
                a_alpha = {w for w in re.findall(r'[a-z]+', na)}
                b_alpha = {w for w in re.findall(r'[a-z]+', nb)}
                compatible_text = not a_alpha or not b_alpha or bool(a_alpha & b_alpha)
                if compatible_text and abs(float(a_nums[0]) - float(b_nums[0])) < 0.01:
                    return True
            except (ValueError, OverflowError):
                pass
        # Text aliases: require complete token containment, never raw substring
        # containment. This allows "John" vs "John O'Flynn" but not "Bra" vs
        # "Brazil", and is skipped when either side contains digits.
        if not re.search(r'\d', na + nb):
            a_tokens = {w for w in re.findall(r"[a-z][a-z']*", na)}
            b_tokens = {w for w in re.findall(r"[a-z][a-z']*", nb)}
            if a_tokens and b_tokens and (a_tokens.issubset(b_tokens) or b_tokens.issubset(a_tokens)):
                return True
        return False
