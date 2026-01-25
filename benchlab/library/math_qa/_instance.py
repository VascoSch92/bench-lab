from dataclasses import dataclass
from typing import Any

from benchlab._instance import Instance
from benchlab._types import AnswerType


@dataclass(slots=True, frozen=True, kw_only=True)
class MathQAInstance(Instance):
    problem: str
    rationale: str
    options: str
    correct: AnswerType
    annotated_formula: str
    linear_formula: str
    category: str

    @property
    def ground_truth(self) -> AnswerType:
        return self.correct

    def _to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "rationale": self.rationale,
            "options": self.options,
            "correct": self.correct,
            "annotated_formula": self.annotated_formula,
            "linear_formula": self.linear_formula,
            "category": self.category,
        }
