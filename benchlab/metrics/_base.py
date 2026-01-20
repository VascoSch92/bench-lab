from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, ClassVar, final, TYPE_CHECKING, Any

from benchlab._instance import Attempt
from benchlab._types import MetricOutputType, InstanceType

if TYPE_CHECKING:
    pass


class MetricType(StrEnum):
    """Enumeration of metric types for evaluation."""

    REGRESSION = "regression"
    """Metrics that produce continuous numerical values."""

    BOOLEAN = "boolean"
    """Metrics that produce True/False outcomes."""

    CATEGORICAL = "categorical"
    """Metrics that produce discrete category labels."""


@dataclass(frozen=True, slots=True)
class Metric(ABC, Generic[InstanceType, MetricOutputType]):
    """Base class for a metric."""

    name: ClassVar[str]
    """"Name of the metric"""

    type_: ClassVar[MetricType]
    """Type of the metric."""

    @final
    def evaluate(
        self, instance: InstanceType, attempts: list[Attempt]
    ) -> list[MetricOutputType]:
        if self.name in instance.evaluations:
            # todo: what are we doing here?
            return []

        values = [
            self._eval_logic(instance=instance, attempt=attempt) for attempt in attempts
        ]

        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }

    @abstractmethod
    def _eval_logic(
        self, instance: InstanceType, attempt: Attempt
    ) -> MetricOutputType: ...

    async def evaluate_async(
        self, instance: InstanceType, attempts: list[Attempt]
    ) -> list[MetricOutputType]:
        # todo: complete here
        return []

    async def _eval_logic_async(
        self,
        instance: InstanceType,
        attempt: Attempt,
    ) -> MetricOutputType:
        """Override this method if you want to provide an async evaluation logic."""
        raise NotImplementedError("An async logic is not implemented for this metric.")
