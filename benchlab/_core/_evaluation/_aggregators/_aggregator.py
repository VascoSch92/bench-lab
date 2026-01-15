from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, ClassVar

import numpy as np

from benchlab._core._types import MetricOutputType, BooleanOutputType, InstanceType


@dataclass(frozen=True, slots=True)
class Report:
    outer_output: float
    inner_output: dict[str, float]


@dataclass(frozen=True, slots=True)
class Aggregator(ABC, Generic[MetricOutputType]):
    name: ClassVar[str]
    """Name of the entity being aggregated."""

    @abstractmethod
    def aggregate(self, instances: list[InstanceType]) -> Report: ...

    @abstractmethod
    def _inner(self, *args, **kwargs) -> float: ...

    @abstractmethod
    def _outer(self, *args, **kwargs) -> float: ...


@dataclass(frozen=True, slots=True)
class MacroAverageAggregator(Aggregator[BooleanOutputType]):
    name: ClassVar[str] = "macro average"
    value: float

    def aggregate(self, instances: list[InstanceType]) -> Report:
        return Report(0, {})

    def _inner(self, data: np.ndarray) -> float:
        """Calculates the mean score (accuracy)."""
        if data.size == 0:
            return 0.0
        return np.nanmean(data)

    def _outer(self, inner_value: float) -> float:
        """Final aggregation step."""
        return float(inner_value)
