import re

from benchlab._metrics.base import Metric, MetricType
from benchlab._instance import Attempt
from benchlab._types import InstanceType


class LLMJudgeMetric(Metric[InstanceType, bool | None]):
    """Implementation of the exact match metric."""

    name = "llm_as_judge"
    type_ = MetricType.BOOLEAN

    model_name: str
    system_prompt: str
    user_prompt: str

    def _eval_logic(self, instance: InstanceType, attempt: Attempt) -> bool | None:
        if attempt.response is None:
            return None

        ground_truth = instance.ground_truth

        assert ground_truth is not None
        escaped_gt = re.escape(ground_truth)

        pattern = rf"\b{escaped_gt}\b"
        match = re.search(pattern, attempt.response, re.IGNORECASE)

        return match is not None
