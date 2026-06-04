from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from src.schema import ReasoningStep, VerificationResult

class BaseVerifier(ABC):
    def __init__(self, table: pd.DataFrame):
        self.table = table

    @abstractmethod
    def verify(self, step: ReasoningStep, context: List[ReasoningStep]) -> VerificationResult:
        pass