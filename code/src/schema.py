# src/schema.py
from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class ReasoningStep:
    step_id: int
    content: str
    step_type: str = "inference"  # 'fact' or 'inference'
    aligned: bool = True  # Whether step uses correct column/method for the question

    # 存储 LLM 自动形式化后的代码 (Pandas 或 Z3)
    formalized_code: Optional[str] = None

@dataclass
class VerificationResult:
    is_valid: bool
    component: str  # "FactChecker" or "LogicAuditor"
    reason: str
    counter_example: Optional[Any] = None

@dataclass
class CoTTrace:
    question: str
    steps: List[ReasoningStep]
    final_answer: str
    raw_text: Optional[str] = None  # Original CoT text, used by refinement loop