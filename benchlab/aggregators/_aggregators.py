from dataclasses import dataclass

import numpy as np

from benchlab._types import MetricOutputType, InstanceType
from ._base import Aggregator, AggregatorType, Report

__all__ = [
    "RuntimesAggregator",
    "StatusAggregator",
]


@dataclass(frozen=True, slots=True)
class RuntimesAggregator(Aggregator[MetricOutputType]):
    name = "runtime_aggregator"
    type_ = AggregatorType.RUNTIME

    def aggregate(self, instances: list[InstanceType]) -> Report:
        # Step 1: Intra-instance aggregation (Median)
        instance_medians = [
            self._inner(
                [runtime for runtime in instance.runtimes if runtime is not None]
            )
            for instance in instances
        ]

        # Step 2: Inter-instance aggregation (Geometric Mean)
        final_score = self._outer(instance_medians)

        return Report(
            aggregator_name=self.name,
            outer_output=final_score,
            inner_output={
                instance.id: inner_
                for instance, inner_ in zip(instances, instance_medians)
            },
        )

    def _inner(self, runtimes: list[float]) -> float:
        """Computes the median using numpy."""
        return float(np.median(runtimes))

    def _outer(self, medians: list[float]) -> float:
        """Computes the geometric mean using numpy log-space arithmetic."""
        # Note: medians must be positive (> 0) for Geometric Mean
        arr = np.array(medians)
        if np.any(arr <= 0):
            # Fallback or handling for 0.0 runtimes (which shouldn't happen)
            return 0.0

        return float(np.exp(np.log(arr).mean()))


@dataclass(frozen=True, slots=True)
class StatusAggregator(Aggregator[MetricOutputType]):
    name = "status_success_rate_aggregator"
    type_ = AggregatorType.STATUS

    def aggregate(self, instances: list[InstanceType]) -> Report:
        instance_metrics = []
        inner_output: dict[str, float] = {}
        weights: list[int] = []

        for instance in instances:
            # 1. Collect statuses (True for COMPLETED, False otherwise)
            # Assuming run.status is an Enum or String
            success_flags = [
                1 if status == "success" else 0 for status in instance.statuses
            ]

            if not success_flags:
                continue

            instance_success_rate = self._inner(success_flags)
            inner_output[instance.id] = instance_success_rate

            weights.append(len(success_flags))
            instance_metrics.append(instance_success_rate)

        return Report(
            aggregator_name=self.name,
            inner_output=inner_output,
            outer_output=self._outer(success_rates=instance_metrics, weights=weights),
        )

    def _inner(self, statuses: list[int]) -> float:
        """Computes the median success rate for a single instance."""
        if not statuses:
            return 0.0
        return float(np.median(statuses))

    def _outer(self, success_rates: list[float], weights: list[int]) -> float:
        """Computes the weighted arithmetic mean."""
        return float(np.average(success_rates, weights=weights))


# todo: complete here
"""
@dataclass(frozen=True, slots=True)
class MacroAverageAggregator(Aggregator[BooleanOutputType]):
    name: ClassVar[str] = "macro average"
    value: float

    def aggregate(self, instances: list[InstanceType]) -> Report:
        return Report(0, {})

    def _inner(self, data: np.ndarray) -> float:

        if data.size == 0:
            return 0.0
        return np.nanmean(data)

    def _outer(self, inner_value: float) -> float:
        return float(inner_value)
"""
