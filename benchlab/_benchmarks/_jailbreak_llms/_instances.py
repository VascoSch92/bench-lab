from benchlab._core._instances import Instance, AnswerType
from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class JailbreakLLMsInstance(Instance):
    content_policy_id: str
    content_policy_name: str
    question: str

    @property
    def ground_truth(self) -> AnswerType:
        return None
