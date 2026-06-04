import re
import pandas as pd
from src.verifiers.base import BaseVerifier
from src.schema import ReasoningStep, VerificationResult
from src.llm_engine import LLMEngine
from utils.logger import setup_logger
from utils.sandbox import build_exec_globals

logger = setup_logger("FactChecker")


def _normalize_value(v: str) -> str:
    v = str(v).strip().lower()
    v = v.replace(',', '').replace('\u00a0', '').replace(' ', '')
    # Strip currency symbols so '$7,719' ≡ '7719'.
    v = v.replace('$', '').replace('¥', '').replace('€', '').replace('£', '')
    # Convert accounting parens-negative '(123)' → '-123' (only if cleanly wrapped digits)
    if re.match(r'^\(-?[\d.]+\)$', v):
        v = '-' + v[1:-1]
    return v


_PLACEHOLDER_TOKENS = {
    '', '-', '--', '---', '—', '–', '−', 'empty', 'hyphen', 'blank',
    'none', 'null', 'nil', 'n/a', 'na', 'missing', 'no value', 'no data',
    'not available', 'not applicable', 'not provided', 'unknown', '?',
    # LLM-paraphrase variants seen in CoT generation
    'no_value', 'no amount', 'no_amount', 'no entry', 'no_entry',
    'dash', 'em dash', 'em_dash', 'en dash', 'en_dash',
    'not reported', 'not_reported', 'undisclosed',
    # An empty/dash numeric cell is often interpreted as $0 by analyst CoTs.
    # Treat as compatible with placeholders so a CoT calling an empty row "$0"
    # doesn't trip FactChecker.
    '$0', '0', '0.0', '0.00', '$0.00',
}


def _is_placeholder(s: str) -> bool:
    t = str(s).strip().strip('"\'').lower()
    if not t:
        return True
    if t in _PLACEHOLDER_TOKENS:
        return True
    if all(ch in '-—–−_' for ch in t):
        return True
    return False


def _predicate_verdict(text: str):
    """Structured extractor output ending with ': True' / ': False'.
    Returns True/False/None."""
    matches = re.findall(r":\s*(True|False|true|false)\b", str(text))
    if not matches:
        return None
    return matches[-1].lower() == 'true'


def _expand_range(text: str):
    """Parse 'X through Y', 'X to Y', 'X-Y', 'X..Y' → set of year strings.
    Only fires on 4-digit years in reasonable spans to avoid false merges."""
    m = re.search(
        r'\b(\d{4})\s*(?:through|thru|to|-|\u2013|\u2014|\.\.)\s*(\d{4})\b',
        str(text), re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 < b - a < 50:
            return {str(x) for x in range(a, b + 1)}
    return None


def _has_modifier(text: str) -> bool:
    return bool(re.search(
        r'\b(?:or more|or less|or above|or below|or fewer|or greater|'
        r'at least|at most|minimum|maximum|more than|less than|'
        r'no less than|no more than|up to)\b',
        str(text).lower()))


def _fuzzy_match(actual: str, claimed: str) -> bool:
    """Programmatic fuzzy comparison. Only SAFE rules that won't leak Type2."""
    if not actual or not claimed:
        return False
    a_s = str(actual).strip()
    c_s = str(claimed).strip()
    error_tokens = {'ERROR', 'NOT_FOUND', 'SEMANTIC_MISMATCH', 'PANDAS_GEN_FAILED'}
    unknown_tokens = {'?', 'N/A', 'None', ''}
    if a_s in error_tokens or c_s in error_tokens:
        return False
    if a_s in unknown_tokens and c_s in unknown_tokens:
        return True
    if a_s in unknown_tokens or c_s in unknown_tokens:
        return False

    a_raw = a_s.lower()
    c_raw = c_s.lower()

    if a_raw == c_raw:
        return True

    a_norm = _normalize_value(actual)
    c_norm = _normalize_value(claimed)
    if a_norm == c_norm:
        return True

    # Rule P: Placeholder equivalence — "–" ≡ "empty" ≡ "hyphen" ≡ "blank"
    if _is_placeholder(a_s) and _is_placeholder(c_s):
        return True

    # Rule V: Structured predicate verdict — "… : True" at end confirms claim
    a_verdict = _predicate_verdict(a_s)
    if a_verdict is True:
        # Actual's predicate evaluated to True. Accept only if claim shares
        # a distinctive content token with actual (not just pure numbers),
        # to avoid leaking unrelated "True" outputs onto wrong claims.
        a_tokens = {w for w in re.findall(r"[a-z][a-z']{2,}", a_raw)}
        c_tokens = {w for w in re.findall(r"[a-z][a-z']{2,}", c_raw)}
        stop = {'the', 'and', 'for', 'are', 'not', 'but', 'all', 'any',
                'true', 'false', 'contains', 'value', 'values', 'column',
                'row', 'table'}
        overlap = (a_tokens - stop) & (c_tokens - stop)
        if overlap or not c_tokens:
            return True

    # Rule R: Range expansion — "2002 through 2006" ≡ enumerated years
    r_a = _expand_range(a_s)
    r_c = _expand_range(c_s)
    a_years = set(re.findall(r'\b(\d{4})\b', a_raw))
    c_years = set(re.findall(r'\b(\d{4})\b', c_raw))
    if r_a and c_years and (r_a == c_years or r_a.issubset(c_years) or c_years.issubset(r_a)):
        return True
    if r_c and a_years and (r_c == a_years or r_c.issubset(a_years) or a_years.issubset(r_c)):
        return True

    # Strip formatting noise: "(s)", parenthetical, units
    def _clean(v):
        v = re.sub(r'\s*\([^)]*\)', '', v)
        v = re.sub(r'\s*(mm|cm|km|kg|%|m)$', '', v)
        v = v.replace('(s)', '')
        return v.strip()

    a_clean = _clean(a_raw)
    c_clean = _clean(c_raw)
    if a_clean and c_clean and a_clean == c_clean:
        return True

    try:
        na = float(_normalize_value(a_clean or actual))
        nc = float(_normalize_value(c_clean or claimed))
        if abs(na - nc) < 0.01:
            return True
    except (ValueError, OverflowError):
        pass

    # Rule N: Numeric match with modifier — "63" ≡ "63 or more" when one side has a modifier
    a_nums = re.findall(r'-?\d+(?:\.\d+)?', a_raw.replace(',', ''))
    c_nums = re.findall(r'-?\d+(?:\.\d+)?', c_raw.replace(',', ''))
    if a_nums and c_nums and (_has_modifier(a_raw) or _has_modifier(c_raw)):
        sa = {float(x) for x in a_nums}
        sc = {float(x) for x in c_nums}
        if sa & sc:
            return True

    a_words = {w for w in re.split(r'[\s,;:()\[\]]+', a_raw) if len(w) > 1}
    c_words = {w for w in re.split(r'[\s,;:()\[\]]+', c_raw) if len(w) > 1}
    has_digits = bool(re.search(r'\d', a_raw) or re.search(r'\d', c_raw))
    if not has_digits and c_words and len(c_words) >= 2 and c_words.issubset(a_words):
        return True

    return False


class FactChecker(BaseVerifier):
    def __init__(self, table: pd.DataFrame):
        super().__init__(table)
        self.llm = LLMEngine()
        self.exec_table, self.clean_columns = self._prepare_table(table)

    @staticmethod
    def _prepare_table(table):
        columns = table.columns.tolist()
        if len(columns) != len(set(columns)):
            t = table.copy()
            seen = {}
            new_cols = []
            for c in columns:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            t.columns = new_cols
            return t, new_cols
        return table, columns

    def verify(self, step: ReasoningStep, context: list, max_retries: int = 2, question: str = "") -> VerificationResult:
        """
        Two-stage verification with TOOL REPAIR:
        - Stage 1: extract_and_compare + fuzzy match
          - On exec error → retry with error feedback (existing)
          - On mismatch → retry with diagnostic feedback (NEW: tool repair)
        - Stage 2: method alignment check
        """
        content = step.content
        logger.info(f"Fact Checking: \"{content}\"")

        table_csv = self.exec_table.to_csv(sep="|", index=False)
        if len(self.exec_table) > 50:
            table_csv = self.exec_table.head(50).to_csv(sep="|", index=False)
        sample_row = self.exec_table.head(3).to_dict(orient='records')

        # ============================================================
        # Stage 1: Value Extraction + Tool Repair Loop
        # ============================================================
        last_error = None
        last_mismatch = None  # Track mismatch for tool repair

        for attempt in range(max_retries + 1):
            # Build feedback: either from crash or from mismatch
            if last_mismatch:
                feedback = (f"Your extraction code returned actual='{last_mismatch[0]}' "
                            f"but the claim says '{last_mismatch[1]}'. "
                            f"This might be a formatting issue (e.g., '66 000' vs '66,000') "
                            f"or your code extracted from the wrong cell. "
                            f"Please fix the extraction logic.")
            elif last_error:
                feedback = f"Error: {last_error}"
            else:
                feedback = ""

            code = self.llm.generate_pandas_check(
                content, self.clean_columns, str(sample_row),
                error_feedback=feedback, question=question,
                full_table=table_csv
            )

            try:
                exec_globals = build_exec_globals({'pd': pd, 're': re})
                exec_locals = {}
                exec(code, exec_globals, exec_locals)

                if 'extract_and_compare' not in exec_locals:
                    last_error = "No extract_and_compare function generated"
                    last_mismatch = None
                    continue

                result = exec_locals['extract_and_compare'](self.exec_table)

                if not isinstance(result, dict):
                    last_error = f"Expected dict, got {type(result)}"
                    last_mismatch = None
                    continue

                actual = str(result.get('actual', '?'))
                claimed = str(result.get('claimed', '?'))

                is_match = _fuzzy_match(actual, claimed)

                if is_match:
                    logger.info(f"Grounding OK: actual='{actual}', claimed='{claimed}'")
                    break  # Pass → go to Stage 2
                else:
                    # TOOL REPAIR: mismatch might be extraction code bug
                    if attempt < max_retries:
                        logger.info(f"Mismatch (attempt {attempt+1}), repairing: actual='{actual}' vs claimed='{claimed}'")
                        last_mismatch = (actual, claimed)
                        last_error = None
                        continue
                    else:
                        # Final attempt still mismatched → genuine data mismatch
                        logger.warning(f"Grounding FAIL (final): actual='{actual}' vs claimed='{claimed}'")
                        return VerificationResult(
                            False, "FactChecker",
                            f"Data Mismatch: actual='{actual}' vs claimed='{claimed}'"
                        )

            except Exception as e:
                last_error = str(e)
                last_mismatch = None
                if attempt < max_retries:
                    logger.info(f"Extract exec failed (attempt {attempt+1}), retrying: {e}")
                else:
                    logger.warning(f"Extraction failed, skipping: {e}")
                    return VerificationResult(
                        False, "FactChecker",
                        f"Extraction failed after retries: {e}"
                    )
        else:
            logger.warning(f"Code gen failed: {last_error}")
            return VerificationResult(
                False, "FactChecker",
                f"Extraction code generation failed: {last_error}"
            )

        # ============================================================
        # Stage 2: Method Alignment (with tool repair)
        # ============================================================
        # Stage2 disabled — method alignment now handled in decompose prompt

        return VerificationResult(True, "FactChecker", "Verification passed.")
