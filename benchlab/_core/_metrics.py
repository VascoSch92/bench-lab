import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, ClassVar, Final, Type

from benchlab._core._instances import MetricStats, Attempt
from benchlab._core._stats import (
    RegressionMetricStats,
    BooleanMetricStats,
    CategoricalMetricStats,
)
from benchlab._core._types import MetricOutputType, InstanceType


class MetricType(StrEnum):
    """Enumeration of metric types for evaluation."""

    REGRESSION = "regression"
    """Metrics that produce continuous numerical values."""

    BOOLEAN = "boolean"
    """Metrics that produce True/False outcomes."""

    CATEGORICAL = "categorical"
    """Metrics that produce discrete category labels."""


_METRIC_TYPE_TO_STATS: Final[dict[MetricType, Type[MetricStats]]] = {
    MetricType.REGRESSION: RegressionMetricStats,
    MetricType.BOOLEAN: BooleanMetricStats,
    MetricType.CATEGORICAL: CategoricalMetricStats,
}


@dataclass(frozen=True, slots=True)
class Metric(ABC, Generic[InstanceType, MetricOutputType]):
    """Base class for a metric."""

    name: ClassVar[str]
    """"Name of the metric"""

    # todo: do we need that?
    benchmarks: ClassVar[list[str]]
    """Benchmarks on which the metric can be computed."""

    type_: ClassVar[MetricType]
    """Type of the metric."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))
    """Logger for the metric."""

    def evaluate(
        self, instance: InstanceType, attempts: list[Attempt]
    ) -> list[MetricOutputType]:
        if self.name in instance.stats:
            self.logger.warning(
                f"Metric `{self.name}` already evaluated. It will be overwritten."
            )

        # TODO: log
        values = [
            self._eval_logic(instance=instance, attempt=attempt) for attempt in attempts
        ]

        self.logger.debug(f"Instance {instance.id} evaluate on metric {self.name}")

        return values

    @abstractmethod
    def _eval_logic(
        self, instance: InstanceType, attempt: Attempt
    ) -> MetricOutputType: ...

    async def evaluate_async(self, instance: InstanceType) -> list[MetricOutputType]:
        # todo: complete here
        return []

    @abstractmethod
    async def _eval_logic_async(
        self, instance: InstanceType, attempt: Attempt
    ) -> MetricOutputType: ...
