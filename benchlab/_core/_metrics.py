import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, final, ClassVar, TypeAlias, Final, Type

from benchlab._core._benchmark import InstanceType
from benchlab._core._instances import MetricStats, Attempt
from benchlab._core._stats import (
    RegressionMetricStats,
    BooleanMetricStats,
    CategoricalMetricStats,
)

MetricOutput: TypeAlias = int | float | None


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
class Metric(ABC, Generic[InstanceType]):
    """Base class for a metric."""

    name: ClassVar[str]
    """"Name of the metric"""

    benchmarks: ClassVar[list[str]]
    """Benchmarks on which the metric can be computed."""

    type_: ClassVar[MetricType]
    """Type of the metric."""

    logger: logging.Logger
    """Logger for the metric."""

    def evaluate(self, instance: InstanceType) -> None:
        if self.name in instance.metrics:
            self.logger.warning(
                f"Metric `{self.name}` already evaluated. It will be overwritten."
            )

        # TODO: notify and log
        values = [
            self.eval_logic(instance=instance, attempt=attempt)
            for attempt in instance.attempts
        ]

        stats = _METRIC_TYPE_TO_STATS[self.type_].from_eval(values)

        instance.update_metric_stats(
            name=self.name,
            stats=stats,
        )

        self.logger.debug(f"Instance {instance.id} evaluate on metric {self.name}")

    @abstractmethod
    def eval_logic(self, instance: InstanceType, attempt: Attempt) -> MetricOutput: ...

    @final
    async def evaluate_async(self, instance: InstanceType) -> None:
        if self.name in instance.metrics:
            # TODO: log and notify here
            return None
        # TODO: notify and log
        # todo: this should be change because is not async
        values = [
            await self.eval_logic_async(instance=instance, attempt=attempt)
            for attempt in instance.attempts
        ]
        instance.update_metric_stats(
            name=self.name, stats=MetricStats.from_eval(values)
        )

    @abstractmethod
    async def eval_logic_async(
        self, instance: InstanceType, attempt: Attempt
    ) -> MetricOutput: ...
