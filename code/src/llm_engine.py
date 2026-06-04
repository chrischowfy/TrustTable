import json
import os
import re
from openai import OpenAI
from configs.config import Config
from utils.logger import setup_logger
from typing import List, Dict
logger = setup_logger("LLMEngine")

class LLMEngine:
    def __init__(self):
        # 初始化 OpenAI/DeepSeek 客户端
        if not Config.API_KEY:
            raise RuntimeError(
                "LLM_API_KEY is required. Set LLM_API_KEY in the environment; "
                "the codebase no longer falls back to committed credentials."
            )
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)
        self.model = Config.MODEL_NAME

        # deepseek-v4-pro requires thinking mode to produce non-empty content.
        # Auto-inject reasoning_effort + thinking enabled so existing call sites
        # (which pass plain messages) work transparently.
        if "v4-pro" in (self.model or "") or "v4pro" in (self.model or ""):
            import time as _time
            _orig_create = self.client.chat.completions.create
            # Default to high per user request 2026-04-30; overrideable via env.
            _v4pro_effort = os.environ.get("DEEPSEEK_V4PRO_EFFORT", "high")  # noqa: pragma  (default high)

            def _create_v4pro(*args, **kwargs):
                kwargs.setdefault("reasoning_effort", _v4pro_effort)
                eb = kwargs.get("extra_body") or {}
                eb.setdefault("thinking", {"type": "enabled"})
                kwargs["extra_body"] = eb
                # retry on transient errors
                last_exc = None
                for attempt in range(4):
                    try:
                        return _orig_create(*args, **kwargs)
                    except Exception as e:
                        last_exc = e
                        msg = str(e)
                        if "429" in msg or "rate" in msg.lower() or "timeout" in msg.lower() or "503" in msg:
                            _time.sleep(2 + attempt * 2)
                            continue
                        raise
                raise last_exc

            self.client.chat.completions.create = _create_v4pro

        # When routing via OpenRouter, pin the provider so experiments are
        # reproducible against a single backend. Novita hosts FP8-quantized
        # DeepSeek V3.2 with the same weights and reliable low latency (~3s).
        # Also add retry-on-429 with exponential backoff, because Novita
        # occasionally returns upstream rate-limit 429s under heavy load.
        if "openrouter.ai" in (Config.BASE_URL or ""):
            import time as _time
            import random as _random
            _orig_create = self.client.chat.completions.create
            _provider_cfg = {"provider": {"only": ["Novita"], "allow_fallbacks": False}}

            def _create_with_provider(*args, **kwargs):
                eb = kwargs.get("extra_body") or {}
                merged = {**_provider_cfg, **eb}
                if "provider" in eb:
                    merged["provider"] = eb["provider"]
                kwargs["extra_body"] = merged

                # Retry on 429 with exponential backoff + jitter
                last_exc = None
                for attempt in range(6):  # up to 6 attempts total
                    try:
                        return _orig_create(*args, **kwargs)
                    except Exception as e:
                        last_exc = e
                        msg = str(e)
                        # Retry on rate limit or transient provider errors
                        if "429" in msg or "rate-limit" in msg.lower() or "rate limit" in msg.lower() or "temporarily" in msg.lower():
                            # 1,2,4,8,16,32 s base + jitter
                            sleep_s = (2 ** attempt) + _random.uniform(0, 1.5)
                            _time.sleep(sleep_s)
                            continue
                        raise
                # All retries exhausted
                raise last_exc

            self.client.chat.completions.create = _create_with_provider

    def preflight(self):
        """Fail fast on invalid credentials/model before starting a batch run."""
        self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=16,
        )

    def autoformalize_to_z3(self, premise_text: str, conclusion_text: str, table_context: str = "", error_feedback: str = "") -> str:

        system_prompt = """You are an expert in Formal Verification.
Verify if a Conclusion follows ONLY from the Premise and Table Data.

### IMPORTANT: Think step by step before writing code.
First, reason about:
1. What are the key variables and their concrete values from the table/premise?
2. What exactly does the conclusion claim?
3. How should I encode the negation for proof-by-contradiction?

Then write the Z3 code in a ```python block.

### CRITICAL: PREMISE GROUNDING CHECK
The Conclusion must be derivable ONLY from:
1. The verified facts in the Premise section
2. Values directly from the Table Context
3. Basic arithmetic (addition, subtraction, counting, comparison)

If the Conclusion introduces ANY of the following, return `(False, "FABRICATED_PREMISE")`:
- External rules or conventions ("Format Size Rule", "standard procedure", "in radio demographics")
- Assumptions about table structure ("table is sorted by", "alphabetically ordered")
- Domain knowledge not in the table ("teams are prohibited from", "by definition")

### STRATEGY
1. **Closed World Assumption**: Table values are ground truth. Use them as hard constraints.
2. **Superlatives (max/min/first/last)**: Compare against ALL values in the relevant column.
3. **Simple arithmetic**: Use plain Python, not Z3. E.g., `return (2 + 0 == 2, None)`.

### CRITICAL: NO FREE VARIABLES
Every Z3 variable MUST be constrained to a concrete value from the table or premise.
BAD:  `Total = Int('Total'); s.add(Not(Total == A + B))` → Total is free, Z3 finds trivial counter-example
GOOD: `s.add(Total == 2); s.add(Not(Total == A + B))` → Total is constrained to the claimed value

### CRITICAL: NUMERIC TOLERANCE (avoid silent floating-point false rejects)
For ANY comparison involving a rounded percentage / decimal / division result, use
TOLERANCE-BASED equality, not strict ==. Floating-point binary representation +
banker's rounding cause silent counterexamples otherwise.

Tolerance rule: tolerance = 0.5 × 10^(-d) where d = decimals in the claimed value.
  Claimed "48.6186%" → d=4 → tolerance = 0.5e-4 = 0.00005
  Claimed "13.0%"    → d=1 → tolerance = 0.05
  Claimed "$25,971"  → d=0 → tolerance = 0.5

BAD (causes silent A_NUMERIC false-reject):
  pct = (a / b) * 100
  rounded = round(pct, 4)
  if rounded == 48.6186: return True, None
  else: return False, f"got {rounded}"        # ← banker's rounding may flip

GOOD (Python check, tolerance-based):
  pct = (a / b) * 100
  if abs(pct - 48.6186) < 0.5e-4:
      return True, None
  return False, f"computed {pct:.6f} differs from claimed 48.6186 by {abs(pct-48.6186):.6f}"

GOOD (Z3 with Real arithmetic, tolerance-based):
  s.add(pct_z3 == apic_re_sum * 100 / total)   # premise
  s.add(Not(And(pct_z3 - 48.6186 >= -Q(1, 20000),
                pct_z3 - 48.6186 <=  Q(1, 20000))))
  if s.check() == sat: return False, s.model()
  return True, None

For pure equality on integers / strings / unrounded values, strict == is fine.
The tolerance rule applies ONLY when the claimed value is a rounded/truncated decimal.

### CRITICAL: SUPERLATIVE → CLOSED-WORLD ENUMERATION
When verifying "X is the largest / smallest / max / min / Nth-largest" claim:
You MUST enumerate ALL candidates and prove X dominates them, NOT use unbounded ForAll.

BAD (open-world fallacy → spurious counterexample):
  x = Int('x'); m = Int('m')
  s.add(m == 50647)
  s.add(Not(ForAll(x, m >= x)))     # ← Z3 picks x = 99999 outside table

GOOD (closed-world enumeration):
  vals = [50647, 1653, 5264, 218, 318, 228, 23]   # ALL candidate values from table
  m = max(vals)                                     # 50647 by construction
  if m == 50647: return True, None
  return False, f"max is {m}, not 50647"

For "X is the second largest", sort and check index 1; for "Nth", index N-1.

### Template
```python
def solve_logic():
    # Step 0: Check if conclusion introduces fabricated premises → return (False, "FABRICATED_PREMISE")
    # Step 1: Extract concrete values from table/premise
    # Step 2: The claim to verify (as Python expression or Z3 constraint)
    # Step 3: Return (True/False, model_or_None)
```
"""

        user_prompt = f"""
### Table Context (Ground Truth)
{table_context}

### Premise
"{premise_text}"

### Conclusion
"{conclusion_text}"
"""
        if error_feedback:
            user_prompt += f"""
### Previous Attempt Failed
{error_feedback}
Please fix the code and try again.
"""

        user_prompt += """
### Task
Write Python Z3 code to verify the conclusion.
### Example Template
```python
def solve_logic():
    s = Solver()
    # 1. Variables
    A_Total, A_Gold = Ints('A_Total A_Gold')

    # 2. Premise Constraints (from Table Context)
    s.add(A_Total == 19)

    # 3. The Logic to Verify: "Total > 10 implies Gold > 5"
    conclusion = Implies(A_Total > 10, A_Gold > 5)

    # 4. Proof by Contradiction (Find Counter-example)
    s.add(Not(conclusion))

    if s.check() == sat:
        return False, s.model() # Invalid
    return True, None # Valid
```"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0  # 确定性代码生成
            )

            raw_content = response.choices[0].message.content
            return self._clean_code(raw_content)

        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}")
            raise RuntimeError(f"LLM Z3 formalization failed: {e}") from e
        
        
    def _clean_code(self, text: str) -> str:
        """
        从 LLM 回复中提取 python 代码块
        """
        # 匹配 ```python ... ``` 或 ``` ... ```
        pattern = r"```(?:python)?\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 如果没有 markdown 标记，假设全文就是代码（但在 GPT-4/DeepSeek 中很少见）
        return text.strip()
    

    def decompose_cot(self, cot_text: str, question: str = "") -> List[Dict]:
        """
        [New Feature] 使用 LLM 将一段 CoT 文本智能拆解为原子步骤，
        并标注每一步是 'fact' (查表) 还是 'inference' (逻辑推导)。
        """
        
        system_prompt = """You are a Chain-of-Thought Parser for TableQA verification.
Parse a CoT reasoning trace into atomic, self-contained steps for verification.

### CRITICAL: FIDELITY — DO NOT SILENTLY CORRECT THE CoT
The CoT may contain arithmetic errors, miscounts, wrong intermediate results,
or self-contradictions (e.g., "A + B = 100" followed by "so the total is 110").
You MUST preserve EVERY numeric/count/arithmetic assertion the CoT makes,
EXACTLY as stated, as its own step. Do NOT merge contradictory numbers. Do
NOT "fix" an arithmetic result you know is wrong. Do NOT drop an intermediate
tally in favor of a later one. Your output is a FAITHFUL trace of what the
CoT claims; verification happens downstream against the table.

Examples of the fidelity rule:
- CoT: "106.3 - 89.7 = 16.5. Thus the difference is 16.6 FM."
  Produce TWO inference steps verbatim:
  - {"content": "106.3 - 89.7 = 16.5", "type": "inference"}
  - {"content": "The difference between highest (106.3) and lowest (89.7) frequency is 16.6 FM", "type": "inference"}
- CoT: "Adding these: 20+33+32+25+19 = 128. So the total is 129."
  Produce:
  - {"content": "20 + 33 + 32 + 25 + 19 = 128", "type": "inference"}
  - {"content": "The sum of 20, 33, 32, 25, 19 is 129", "type": "inference"}
- CoT: "Counting gives 5 seasons. Thus the answer is 6."
  Produce:
  - {"content": "Counting the values under 50 gives 5", "type": "inference"}
  - {"content": "The number of seasons with Overall ranking less than 50 is 6", "type": "inference"}

### CRITICAL: Each step must be SELF-CONTAINED
- Do NOT use pronouns or deictic references (this, that, it, these, those, they, them).
- Every step must explicitly name the entity, column, or value it refers to.
- BAD:  "Check the date column for this entry" (what entry?)
- GOOD: "Check the date column for the 'Kodachrome film' entry"
- BAD:  "It is greater than 10" (what is?)
- GOOD: "Brazil's gold medal count of 19 is greater than 10"

### CRITICAL: ONE CLAIM PER STEP
Each step must contain exactly ONE verifiable claim. Split multiple claims into separate steps.
- BAD (multiple claims): "5th is listed in 2005, 2008, and 2010"
- GOOD (one claim per step):
  - "The Reg. Season for 2005 is '5th, Western'"
  - "The Reg. Season for 2008 is '5th, Western'"
  - "The Reg. Season for 2010 is '5th, Western'"
- BAD: "Elie Wiesel, Cynthia Ozick, and Arthur Cohn are all Professional writers"
- GOOD:
  - "Elie Wiesel (1997) has Profession 'Professional writer'"
  - "Cynthia Ozick (2001) has Profession 'Professional writer'"
  - "Arthur Cohn (2004) has Profession 'Professional writer'"

### Step Types
1. **fact**: A claim about a specific value in a specific row/cell.
   Format: "[Entity] has [Column] = [Value]" or "The [Column] for [Row] is [Value]".
   Also includes: reading a cell value, identifying a row by condition.

2. **inference**: Calculation, comparison, or data-derived conclusion. Sub-types:
   - **arithmetic**: "3 + 2 = 5", "the total is 10"
   - **counting**: "there are 3 entries matching X"
   - **comparison**: "A is greater than B", "X is the maximum"
   - **aggregation**: "the sum/average/max of column X is Y"

3. **logic**: Reasoning that introduces external rules, conventions, or assumptions.
   - "according to the Format Size Rule..."
   - "in standard data cleaning procedure..."
   - "the table is sorted alphabetically..."
   - "by definition, X means Y..."
   WARNING: Steps of type "logic" often indicate FABRICATED PREMISES not supported by the table.

### Output Format
Return JSON: {"steps": [{"content": "...", "type": "fact|inference|logic", "aligned": true}, ...]}
"""

        user_prompt = f"""
### Original Question
"{question}"

### Chain-of-Thought to Parse
"{cot_text}"

### Task
Parse the CoT into atomic steps. For each step, check alignment:

Set "aligned": false ONLY if the step searches for a CLEARLY WRONG entity/column:
  - Question asks "Professional writer" but step searches for "Author" → false
  - Question asks "inner diameter" but step checks "outer diameter" → false

Set "aligned": true for everything else (data reads, counts, comparisons, calculations).

Return JSON: {{"steps": [{{"content": "...", "type": "fact", "aligned": true}}, ...]}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, # 强制 JSON 模式
                temperature=0.0 # 保持确定性
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("steps", [])

        except Exception as e:
            logger.error(f"CoT Decomposition Failed: {e}")
            raise RuntimeError(f"CoT decomposition failed: {e}") from e
        
    def generate_pandas_check(self, claim: str, columns: list, sample_data: str, error_feedback: str = "", question: str = "", full_table: str = "") -> str:
        """
        [Value Extraction] 从表格中提取 claim 引用的实际值，供程序化比较。
        返回格式: extract_and_compare(df) → dict with 'actual', 'claimed', 'match'
        """
        system_prompt = """You are a Python Pandas Expert for TableQA verification.
Write a function `extract_and_compare(df)` that extracts the ACTUAL value from the DataFrame and compares it with the CLAIMED value.

### IMPORTANT: Think step by step before writing code.
First, reason about:
1. What specific value does the claim assert? (the "claimed" value)
2. Which column and row in the DataFrame contain this value?
3. How should I extract and compare it?

Then write the code in a ```python block.

### Return format
Return a dict: {"actual": str, "claimed": str, "match": bool}

### IMPORTANT RULES
- The claim is an INTERMEDIATE reasoning step. Just verify its DATA FACTS.
- If the claim is procedural (e.g., "Scan the table", "Look at column X", "Count the entries"), it has no specific value to check → return {"actual": "N/A", "claimed": "N/A", "match": True}
- If the claim mentions a specific number, entity, or value → EXTRACT the real value from df and compare.
- ALWAYS use `df['col'].astype(str).str.strip()` before `.str` operations.
- ALWAYS strip commas from numbers: `pd.to_numeric(s.str.replace(',','',regex=False).str.replace(' ','',regex=False), errors='coerce')`
- Use `.str.contains(..., case=False, na=False)` for entity matching.
- Check `len(filtered) > 0` before `.values[0]`.
- The `df` contains ALL rows (sample shows only first rows).

### Semantic Check (IMPORTANT):
- If the claim uses a CLEARLY DIFFERENT column/entity than what the question asks about (e.g., Question="Professional writer" but claim searches for "Author"), set match=False and actual="SEMANTIC_MISMATCH".
- But intermediate steps (scanning, counting, looking up data) are always valid.

### Example
Claim: "Brazil has 7 gold medals"

The claim asserts Brazil has 7 gold medals. I need to look up the 'Nation' column for 'Brazil' and check the 'Gold' column value.

```python
def extract_and_compare(df):
    filtered = df[df['Nation'].astype(str).str.contains('Brazil', case=False, na=False)]
    if len(filtered) == 0:
        return {"actual": "NOT_FOUND", "claimed": "7", "match": False}
    actual = filtered['Gold'].astype(str).str.replace(',','',regex=False).str.strip().values[0]
    return {"actual": actual, "claimed": "7", "match": str(actual).strip() == "7"}
```
"""

        user_prompt = f"""### Table Schema
- Columns: {columns}
- Sample Data: {sample_data}
"""
        if full_table:
            user_prompt += f"""
### Full Table Data (CSV)
{full_table}
"""
        user_prompt += f"""
### Original Question
"{question}"

### Claim to Verify
"{claim}"
"""
        if error_feedback:
            user_prompt += f"\n### Previous Attempt Failed\n{error_feedback}\nPlease fix.\n"
        user_prompt += "\nWrite the `extract_and_compare(df)` function."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return self._clean_code(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Pandas Gen Failed: {e}")
            raise RuntimeError(f"Pandas check generation failed: {e}") from e

    def generate_inference_check(self, claim: str, columns: list, sample_data: str, table_context: str, question: str = "", error_feedback: str = "") -> str:
        """
        [Inference via Pandas] 用 Pandas 验证推理步骤（比较、聚合、排序等），
        替代 Z3 处理大多数 inference 步骤。
        """
        system_prompt = """You are a Python Pandas Expert for verifying reasoning claims about tables.
Write a function `verify_inference(df)` that checks if a reasoning/inference claim is correct.

### IMPORTANT: Think step by step before writing code.
First, reason about:
1. What does the claim assert? (e.g., a comparison, a count, an arithmetic result)
2. How can I verify this using the DataFrame? (which columns, what operations)
3. What does True/False mean for this claim?

Then write the code in a ```python block.

### Common inference patterns and how to verify:
- "X is the highest/maximum" → check df['col'].max() and compare
- "X is the first/earliest" → check df sorted by date/order
- "There are N items matching condition" → count and compare
- "X + Y = Z" → compute and compare
- "X comes before/after Y" → check ordering

### Rules
- ALWAYS query the DataFrame to verify. Never return True/False without checking.
- Use `df['col'].astype(str)` before `.str` ops; strip commas from numbers.
- If the claim is too abstract to verify with data (e.g., "therefore, the answer is..."), return True.

### CRITICAL: NUMERIC TOLERANCE (avoid silent floating-point false rejects)
For ANY comparison involving a rounded percentage / decimal / division result, use
TOLERANCE-BASED equality, not strict ==. Floating-point representation + banker's
rounding cause silent counterexamples otherwise.

Tolerance rule: tolerance = 0.5 × 10^(-d) where d = decimals in the claimed value.
  Claimed "21.568%" → d=3 → tolerance = 0.0005
  Claimed "48.6186%" → d=4 → tolerance = 0.00005
  Claimed "13.0%"   → d=1 → tolerance = 0.05
  Claimed "$25,971" → d=0 → tolerance = 0.5

BAD (silent A_NUMERIC false-reject):
    pct = (df['gp'].iloc[0] / df['rev'].iloc[0]) * 100
    return round(pct, 3) == 21.568          # floating-point + banker's rounding flip

GOOD (tolerance-based):
    pct = (df['gp'].iloc[0] / df['rev'].iloc[0]) * 100
    return abs(pct - 21.568) < 0.0005

GOOD (with explicit logging when rejecting):
    pct = ...
    if abs(pct - 21.568) < 0.0005:
        return True
    print(f"computed {pct:.6f} vs claimed 21.568, diff {abs(pct-21.568):.6f}")
    return False

Strict equality is fine for INTEGERS, COUNTS, BOOLEANS, STRINGS, EXACT NUMERIC EQUALITY.
The tolerance rule applies ONLY when the claim's value is a rounded/truncated decimal.

### CRITICAL: ACCOUNTING FORMAT NORMALIZATION
Tables often use accounting conventions that look different from CoT phrasing:
- `($X)` = parens-wrapped = NEGATIVE: `($6,881)` is `-6881`. Strip parens, prefix minus.
- `$X` and `X` are the same number (strip dollar sign).
- `1,234,567` and `1234567` are the same (strip thousand commas).
- "—" / "–" / blank = missing/zero in numeric columns; CoT may interpret as `0`.

When parsing cell values for comparison, normalize: strip $, strip commas, convert
parens-wrapped digits to negative. Example:
    def to_num(s):
        s = str(s).strip().replace('$','').replace(',','').replace(' ','')
        if s.startswith('(') and s.endswith(')'): s = '-' + s[1:-1]
        if s in ('—','–','-','','nan','none','null'): return 0.0
        return float(s)
"""
        user_prompt = f"""### Table Schema
- Columns: {columns}
- Sample Data: {sample_data}

### Full Table Context
{table_context[:2000]}

### Original Question
"{question}"

### Inference Claim to Verify
"{claim}"

Write `verify_inference(df)` returning bool."""

        if error_feedback:
            user_prompt += f"\n\n### Previous Attempt Failed\n{error_feedback}\nPlease fix the code."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return self._clean_code(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Inference Check Gen Failed: {e}")
            raise RuntimeError(f"Inference check generation failed: {e}") from e
        

    def check_method_alignment(self, question: str, claim: str, columns: list, table_sample: str = "") -> bool:
        """
        [Stage 2] 轻量级检查：CoT 步骤操作的列/方法是否与问题匹配。
        返回 True=对齐, False=语义偷换。
        """
        system_prompt = """You are a semantic alignment checker for TableQA.
Given a Question and a Reasoning Step, determine if the step uses the CORRECT column/field/entity to answer the question.

MISALIGNED means the step searches for a WRONG ENTITY or uses a WRONG COLUMN:
  * Question="Professional writer" but step searches for "Author" → MISALIGNED (wrong entity)
  * Question="inner diameter" but step checks "outer diameter" → MISALIGNED (wrong column)
  * Question="first winner" but step sorts alphabetically → MISALIGNED (wrong method)
  * Question="how many X" but step counts a COMPLETELY DIFFERENT entity Y → MISALIGNED

ALIGNED — return this for ALL of the following (even if they seem unrelated at first glance):
  * Reading ANY specific value from the table → ALIGNED
  * "Scan down to the Total row" → ALIGNED (navigating the table)
  * "Count entries for Rochester" → ALIGNED (counting operation)
  * "Check value for period 1960-1965" → ALIGNED (looking up data)
  * "The prize money is $50,000" → ALIGNED (reading a cell value)
  * "Look through the table to find entries for Me-109" → ALIGNED (scanning)
  * "Continue down to 2002" → ALIGNED (navigation)
  * "Find the affiliate count for Azteca 13" → ALIGNED (reading data)
  * "Look at the seventh row" → ALIGNED (positional reference)
  * ANY step that reads, counts, filters, or compares actual table data → ALIGNED

The ONLY reason to return MISALIGNED is if the step explicitly uses a WRONG column name or searches for a WRONG entity that contradicts the question.
Reply with ONLY one word: ALIGNED or MISALIGNED."""

        user_prompt = f"""Table columns: {columns}
"""
        if table_sample:
            user_prompt += f"Table data (first rows):\n{table_sample}\n"
        user_prompt += f"""
Question: "{question}"

Reasoning Step: "{claim}"
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip().upper()
            return "MISALIGNED" not in answer
        except Exception as e:
            logger.error(f"Method Alignment Check Failed: {e}")
            return True  # Default to aligned on error

    def generate_answer_code(self, question: str, columns: list, sample_data: str, full_table: str = "") -> str:
        """
        [Consistency Monitor] 生成 compute_answer(df) 函数，
        直接从表格计算问题的答案，用于与 CoT 声称的答案做一致性对比。
        """
        system_prompt = """You are a Python Pandas Expert for TableQA.
Write a function `compute_answer(df)` that directly answers the given question using the DataFrame.

### IMPORTANT: Think step by step before writing code.
First, reason about:
1. What does the question ask for? (a specific value, a calculation, a comparison)
2. Which columns and rows are relevant?
3. What operations are needed? (lookup, sum, divide, compare, etc.)

Then write the code in a ```python block.

### Requirements
1. **Function Signature**: `def compute_answer(df): -> str`
2. **Return**: The answer as a string (e.g., "Brazil", "7", "3.5").
3. **Robustness**:
   - ALWAYS use `df['col'].astype(str).str.strip()` before `.str` operations.
   - Use `pd.to_numeric(df['col'].astype(str).str.replace(',','',regex=False).str.replace('$','',regex=False), errors='coerce')` for numbers.
   - Strip $, commas; convert accounting parens `($X)` → `-X` before numeric ops.
   - Handle edge cases (empty results, missing values, "—"/"–"/blank → 0 in numeric columns).
   - The `df` contains ALL rows. Sample data shows only first rows.

### CRITICAL: PARSE QUESTION QUALIFIERS PRECISELY
Distinguish carefully:
- "the largest INDIVIDUAL X" / "the SINGLE largest X" / "the largest X line ITEM"
  → pick ONE row by argmax. NEVER sum multiple rows.
- "the total of all X" / "combined X" → SUM matching rows.
- "the largest X" alone is ambiguous; treat as argmax of one row when columns make subtotals available.
- If both "Total X" subtotal row AND individual line items exist, "individual" / "single" / "line item"
  modifiers EXCLUDE the subtotal row.

BAD (silent A — misreads "largest individual cost item" as "sum of all cost items"):
    total_cost = pd.to_numeric(df[df['Section']=='Costs']['Amount']).sum()
    return f"{total_cost / net_revenue * 100:.2f}%"     # 92.75% — wrong!

GOOD (correctly picks ONE largest line):
    cost_items = df[df['Section']=='Costs']  # exclude subtotal/total rows
    cost_items = cost_items[~cost_items['Item'].str.contains('Total', case=False, na=False)]
    largest = pd.to_numeric(cost_items['Amount'].astype(str).str.replace(',','').str.replace('$','')).max()
    return f"{largest / net_revenue * 100:.1f}%"        # 80.5% — right
"""
        user_prompt = f"""
### Table Schema
- Columns: {columns}
- Sample Data (First 3 rows): {sample_data}
"""
        if full_table:
            user_prompt += f"""
### Full Table Data (CSV)
{full_table}
"""
        user_prompt += f"""
### Question
"{question}"

### Task
Write the `compute_answer(df)` function.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return self._clean_code(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Answer Code Gen Failed: {e}")
            raise RuntimeError(f"Answer code generation failed: {e}") from e

    def refine_logic_proof(self, question: str, old_cot: str, error_report: dict) -> str:
        """
        [Role: Logic Auditor] 
        LLM 审视符号引擎给出的反例，通过追加全局约束或修正错误逻辑来完善证明。
        """
        module = error_report.get("module", "")
        reason = error_report.get("reason", "")
        failed_step = error_report.get("step_content", "")

        system_prompt = """You are a Formal Logic Auditor. Your role is to evaluate and fortify a reasoning chain (CoT) that failed a symbolic verifier (Z3/Pandas).

### AUDIT PHILOSOPHY:
- **Case A: Logic Leak (Incomplete Proof)**: The reasoning is correct but "leaky". For example, saying "7 is the max" without proving others are smaller. 
- *Refinement*: You must explicitly cite the values of ALL competitors from the table to "close the logical world".
- **Case B: Spurious Logic (Wrong Rule)**: The reasoning uses a rule that isn't supported by the table (e.g., "Sail number determines speed").
- *Refinement*: You must acknowledge the error and switch to a standard lookup or sequential logic.
- **Case C: Hallucination**: The reasoning cited data that isn't in the table.
- *Refinement*: Re-check the table and use grounded facts.

### OUTPUT:
Provide a refined, step-by-step Chain-of-Thought that is robust enough to be logically irrefutable.
"""

        user_prompt = f"""
### Original Question
"{question}"

### Failed Reasoning Trace
"{old_cot}"

### Verifier Feedback
- **Failed Module**: {module}
- **Faulty Step**: "{failed_step}"
- **Technical Objection**: {reason}

### Audit Instruction
Analyze the Technical Objection. If a counter-example was found, the logic is "leaky". 
Rewrite the Reasoning Chain to be a "Strict Proof". If it's a comparison task, you MUST explicitly enumerate the values of the other candidates to block the solver from finding counter-examples.

### Fortified Reasoning Chain:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # 验证者需要严谨，低温度
                top_p=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Refinement Failed: {e}")
            return old_cot
