from dataclasses import dataclass
from typing import Any

from benchlab._core._instances import Instance
from benchlab._core._types import AnswerType


@dataclass(slots=True, frozen=True, kw_only=True)
class JailbreakLLMsInstance(Instance):
    content_policy_id: str
    content_policy_name: str
    question: str

    @property
    def ground_truth(self) -> AnswerType:
        return None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "content_policy_id": self.content_policy_id,
            "content_policy_name": self.content_policy_name,
            "question": self.question,
        }
