import gc
import os
import json
import threading
import z3
from src.verifiers.base import BaseVerifier
from src.schema import ReasoningStep, VerificationResult
from src.llm_engine import LLMEngine
from utils.logger import setup_logger
from utils.sandbox import build_exec_globals

logger = setup_logger("Z3Auditor")

# Optional per-run diagnostic dump. When env var Z3_DIAG_LOG is set, every
# Z3 verification call appends a JSON line containing code, votes, models
# so we can later distinguish silent encoding bugs (A) from genuine
# counter-examples (B).
_Z3_DIAG_LOG = os.environ.get("Z3_DIAG_LOG", "")
_Z3_DIAG_LOCK = threading.Lock()
_Z3_EXEC_LOCK = threading.Lock()


def _z3_diag_emit(record: dict):
    if not _Z3_DIAG_LOG:
        return
    try:
        with _Z3_DIAG_LOCK, open(_Z3_DIAG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Z3 diag emit failed: {e}")

class Z3Auditor(BaseVerifier):
    def __init__(self, table):
        super().__init__(table)
        self.llm = LLMEngine()



    def _build_exec_globals(self):
        """构建 Z3 沙盒执行环境"""
        return build_exec_globals({
            "z3": z3, "Solver": z3.Solver, "Optimize": z3.Optimize,
            "Int": z3.Int, "Ints": z3.Ints, "IntVal": z3.IntVal, "IntSort": z3.IntSort,
            "Real": z3.Real, "Reals": z3.Reals, "RealVal": z3.RealVal, "RealSort": z3.RealSort,
            "Bool": z3.Bool, "Bools": z3.Bools, "BoolVal": z3.BoolVal, "BoolSort": z3.BoolSort,
            "String": z3.String, "StringVal": z3.StringVal,
            "Not": z3.Not, "And": z3.And, "Or": z3.Or,
            "Implies": z3.Implies, "If": z3.If, "Xor": z3.Xor,
            "Distinct": z3.Distinct, "Const": z3.Const, "Function": z3.Function,
            "Sum": z3.Sum, "Product": z3.Product,
            "ForAll": z3.ForAll, "Exists": z3.Exists,
            "Array": z3.Array, "Select": z3.Select, "Store": z3.Store,
            "ArithRef": z3.ArithRef, "is_true": z3.is_true, "is_false": z3.is_false,
            "simplify": z3.simplify, "substitute": z3.substitute,
            "sat": z3.sat, "unsat": z3.unsat, "unknown": z3.unknown,
        })

    def verify(self, step: ReasoningStep, context: list, max_retries: int = 1,
               question: str = "", full_cot: str = "", n_votes: int = 3) -> VerificationResult:
        if step.step_type not in ("inference", "logic"):
            return VerificationResult(True, "Z3Auditor", "Skipping.")

        # 1. 准备上下文
        verified_facts = [s.content for s in context if s.step_type == "fact"]
        premise_text = "\n".join(verified_facts) if verified_facts else "No factual context"
        conclusion_text = step.content

        table_str = self.table.to_csv(sep="|", index=False)
        if len(self.table) > 100:
            table_str = self.table.head(100).to_csv(sep="|", index=False)

        logger.info(f"Auditing with FULL Table Context ({len(self.table)} rows)...")

        # Majority voting: run n_votes times, take majority
        votes_valid = 0
        votes_invalid = 0
        last_model_str = None
        last_error = None
        per_run_diag = []  # list of {verdict, model, code, error}

        for vote in range(n_votes):
            run_payload = {}
            result = self._single_z3_run(
                premise_text, conclusion_text, table_str, max_retries,
                _diag_payload=run_payload,
            )
            if result is None:
                # Execution failed completely → count as inconclusive
                last_error = "execution_failed"
                per_run_diag.append({
                    "verdict": "exec_error",
                    "model": None,
                    "code": run_payload.get("code"),
                    "error": run_payload.get("error"),
                })
                continue
            is_valid, model_str = result
            per_run_diag.append({
                "verdict": "valid" if is_valid else "invalid",
                "model": model_str,
                "code": run_payload.get("code"),
                "error": None,
            })
            if is_valid:
                votes_valid += 1
            else:
                votes_invalid += 1
                last_model_str = model_str

        logger.info(f"Z3 voting: valid={votes_valid}, invalid={votes_invalid}")

        # Diagnostic emission (if env var set)
        if _Z3_DIAG_LOG:
            qid = os.environ.get("Z3_DIAG_QID", "")
            _z3_diag_emit({
                "qid": qid,
                "step_index": getattr(step, "step_index", None) or getattr(step, "index", None),
                "step_content": step.content,
                "premise_text": premise_text[:2000],
                "conclusion_text": conclusion_text,
                "votes_valid": votes_valid,
                "votes_invalid": votes_invalid,
                "votes_error": n_votes - votes_valid - votes_invalid,
                "per_run": per_run_diag,
            })

        # Majority decision
        if votes_valid > votes_invalid:
            return VerificationResult(True, "Z3Auditor", f"Logic sound (vote: {votes_valid}/{votes_valid+votes_invalid}).")
        elif votes_invalid > 0:
            return VerificationResult(
                False, "Z3Auditor",
                f"Logic Error: Counter-example found (vote: {votes_invalid}/{votes_valid+votes_invalid}).",
                counter_example=last_model_str
            )
        else:
            logger.warning("Z3 all votes inconclusive, rejecting as unverified")
            return VerificationResult(False, "Z3Auditor", "Z3 inconclusive; unable to verify logic.")

    def _single_z3_run(self, premise_text, conclusion_text, table_str, max_retries,
                       _diag_payload=None):
        """Single Z3 verification attempt. Returns (is_valid, model_str) or None on failure.
        If `_diag_payload` is a dict, populate it with last code + error for diagnostics.
        """
        last_feedback = ""
        for attempt in range(max_retries + 1):
            z3_code = self.llm.autoformalize_to_z3(premise_text, conclusion_text, table_str, error_feedback=last_feedback)
            if isinstance(_diag_payload, dict):
                _diag_payload["code"] = z3_code

            try:
                exec_globals = self._build_exec_globals()
                exec_locals = {}
                with _Z3_EXEC_LOCK:
                    exec(z3_code, exec_globals, exec_locals)

                    if "solve_logic" not in exec_locals:
                        last_feedback = "No solve_logic function. Define def solve_logic() returning (bool, model)."
                        if isinstance(_diag_payload, dict):
                            _diag_payload["error"] = "no_solve_logic"
                        continue

                    is_valid, model = exec_locals["solve_logic"]()
                model_str = str(model) if model else None

                exec_locals.clear()
                exec_globals.clear()
                gc.collect()

                return (is_valid, model_str)

            except Exception as e:
                last_feedback = f"Error: {e}. Fix the code."
                if isinstance(_diag_payload, dict):
                    _diag_payload["error"] = str(e)
                gc.collect()
                if attempt < max_retries:
                    logger.info(f"Z3 exec failed (attempt {attempt+1}), retrying: {e}")
                else:
                    logger.error(f"Z3 Execution Failed (final): {e}")
                    return None
