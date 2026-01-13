from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self

import numpy as np

from benchlab._core._types import MetricOutputType, BooleanOutputType


@dataclass(frozen=True, slots=True)
class Aggregator(ABC, Generic[MetricOutputType]):
    name: str
    """Name of the entity being aggregated."""

    @classmethod
    @abstractmethod
    def from_stats(cls, name: str, stats: list[list[MetricOutputType]]) -> Self: ...


@dataclass(frozen=True, slots=True)
class BooleanAggregator(Aggregator[BooleanOutputType]):
    macro_average: float

    micro_average: float

    pass_at_k: float

    @classmethod
    def from_stats(cls, name: str, stats: list[list[bool | None]]) -> Self:
        total_tasks = len(stats)
        if total_tasks == 0:
            return cls(
                name=name,
                macro_average=0,
                micro_average=0,
                pass_at_k=0,
            )

        # We use a list comprehension to get counts per task,
        # then convert to a NumPy matrix for vectorized math.
        # Row format: [n_true, n_valid]
        counts = np.array(
            [
                [
                    sum(1 for attempt in task if attempt is True),
                    sum(1 for attempt in task if attempt is not None),
                ]
                for task in stats
            ]
        )

        n_true = counts[:, 0]
        n_valid = counts[:, 1]

        # 1. Macro Average: Mean of (true/valid) per task
        # np.divide handles the "if valid > 0" logic via the 'where' parameter
        task_accuracies = np.divide(
            n_true, n_valid, out=np.zeros_like(n_true, dtype=float), where=n_valid > 0
        )
        macro_average = np.mean(task_accuracies)

        # 2. Micro Average: Total True / Total Valid (Global)
        total_true_sum = np.sum(n_true)
        total_valid_sum = np.sum(n_valid)
        micro_average = (
            float(total_true_sum / total_valid_sum) if total_valid_sum > 0 else 0.0
        )

        # 3. Pass@k: % of tasks where at least one attempt was True
        pass_at_k = np.mean(n_true > 0)

        return cls(
            name=name,
            macro_average=float(macro_average),
            micro_average=float(micro_average),
            pass_at_k=float(pass_at_k),
        )
