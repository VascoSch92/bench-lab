import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, ClassVar, Final, final, Type, TYPE_CHECKING, Any

from benchlab._core._evaluation._stats import (
    RegressionMetricStats,
    BooleanMetricStats,
    CategoricalMetricStats,
)
from benchlab._core._instance import Attempt
from benchlab._core._types import MetricOutputType, InstanceType

if TYPE_CHECKING:
    from benchlab._core._evaluation._stats import MetricStats


class MetricType(StrEnum):
    """Enumeration of metric types for evaluation."""

    REGRESSION = "regression"
    """Metrics that produce continuous numerical values."""

    BOOLEAN = "boolean"
    """Metrics that produce True/False outcomes."""

    CATEGORICAL = "categorical"
    """Metrics that produce discrete category labels."""


# todo: we can delete that
_METRIC_TYPE_TO_STATS: Final[dict[MetricType, Type["MetricStats"]]] = {
    MetricType.REGRESSION: RegressionMetricStats,
    MetricType.BOOLEAN: BooleanMetricStats,
    MetricType.CATEGORICAL: CategoricalMetricStats,
}


@dataclass(frozen=True, slots=True)
class Metric(ABC, Generic[InstanceType, MetricOutputType]):
    """Base class for a metric."""

    name: ClassVar[str]
    """"Name of the metric"""

    type_: ClassVar[MetricType]
    """Type of the metric."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))
    """Logger for the metric."""

    @final
    def evaluate(
        self, instance: InstanceType, attempts: list[Attempt]
    ) -> list[MetricOutputType]:
        if self.name in instance.evaluations:
            self.logger.warning(
                f"Metric `{self.name}` already evaluated. It will be overwritten."
            )

        # TODO: log
        values = [
            self._eval_logic(instance=instance, attempt=attempt) for attempt in attempts
        ]

        self.logger.debug(f"Instance {instance.id} evaluate on metric {self.name}")

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
