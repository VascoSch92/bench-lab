import collections
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, Generic

import numpy as np

from benchlab._core._types import (
    MetricOutputType,
    RegressionOutputType,
    BooleanOutputType,
    CategoricalOutputType,
)

# todo: we can implement the ConfusionMatrixStats
# todo: implement Rouge score with stats
# todo: implement BLUE score with stats


@dataclass(frozen=True)
class MetricStats(ABC, Generic[MetricOutputType]):
    metric_name: str
    """Name of the metric associated to the stats."""

    n_attempts: int
    """Total number of attempts."""

    n_valid_attempts: int
    """Total number of valid attempts, i.e., not `None`"""

    @classmethod
    @abstractmethod
    def from_eval(cls, metric_name: str, values: list[MetricOutputType]) -> Self: ...

    @classmethod
    def aggregate(cls, stats: list[Self]) -> Self:
        if len(stats) == 0:
            raise ValueError("Empty stats list")
        if len({s.metric_name for s in stats}) != 1:
            raise ValueError("All stats must have the same metric")
        return cls._aggregate(values=stats)

    @classmethod
    @abstractmethod
    def _aggregate(cls, values: list[Self]) -> Self: ...


@dataclass(frozen=True, slots=True)
class RegressionMetricStats(MetricStats[RegressionOutputType]):
    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def from_eval(cls, metric_name: str, values: list[RegressionOutputType]) -> Self:
        vector = np.array(values, dtype=float)

        n_valid_attempts = int(vector.size - np.count_nonzero(~np.isnan(vector)))
        if n_valid_attempts == 0:
            # todo: do we want to raise an error here?
            raise ValueError("Cannot compute MetricStats: all values are None")

        mean = np.nanmean(vector)
        std = np.nanstd(vector)

        return cls(
            metric_name=metric_name,
            n_attempts=vector.size,
            n_valid_attempts=n_valid_attempts,
            mean=float(mean),
            std=float(std),
            min=np.nanmin(vector),
            max=np.nanmax(vector),
        )

    @classmethod
    def _aggregate(cls, stats: list[Self]) -> Self:
        # Convert attributes to numpy array
        n_attempts = np.array([v.n_attempts for v in stats])
        n_valid = np.array([v.n_valid_attempts for v in stats])
        means = np.array([v.mean for v in stats])
        stds = np.array([v.std for v in stats])
        mins = np.array([v.min for v in stats])
        maxs = np.array([v.max for v in stats])

        total_valid = np.sum(n_valid)
        if total_valid == 0:
            raise ValueError("No valid attempts to aggregate")

        # weighted mean
        mean_value = np.average(means, weights=n_valid)

        # combined standard deviation
        variances = stds**2
        squared_diffs = (means - mean_value) ** 2
        pooled_var = np.sum(n_valid * (variances + squared_diffs)) / total_valid
        std_value = np.sqrt(pooled_var)

        return cls(
            metric_name=stats[0].metric_name,
            n_attempts=int(np.sum(n_attempts)),
            n_valid_attempts=int(total_valid),
            mean=float(mean_value),
            std=float(std_value),
            min=float(np.min(mins)),
            max=float(np.max(maxs)),
        )


@dataclass(frozen=True)
class BooleanMetricStats(MetricStats[BooleanOutputType]):
    n_true: int
    n_false: int

    def confidence_interval(
        self,
        confidence_level: float = 0.95,
    ) -> tuple[float, float]:
        """
        Compute the Wilson score confidence interval for a Bernoulli proportion.

        Args:
            successes: Number of successful trials.
            trials: Total number of trials.
            confidence_level: Confidence level (e.g. 0.90, 0.95, 0.99).

        Returns:
            (lower_bound, upper_bound) as floats in [0, 1].

        Raises:
            ValueError: If trials < 0, successes < 0, or successes > trials.
            ValueError: If confidence_level is not supported.
        """
        if self.n_valid_attempts == 0:
            return 0.0, 0.0

        z_table = {
            0.90: 1.6448536269514722,
            0.95: 1.959963984540054,
            0.99: 2.5758293035489004,
        }

        try:
            z = z_table[confidence_level]
        except KeyError:
            raise ValueError(
                "Unsupported confidence_level. Use one of: 0.90, 0.95, 0.99"
            )

        p_hat = self.n_true / self.n_valid_attempts

        denom = 1.0 + (z**2) / self.n_valid_attempts
        center = (p_hat + (z**2) / (2 * self.n_valid_attempts)) / denom
        margin = (
            z
            * math.sqrt(
                (p_hat * (1 - p_hat) + (z**2) / (4 * self.n_valid_attempts))
                / self.n_valid_attempts
            )
            / denom
        )

        return max(0.0, center - margin), min(1.0, center + margin)

    @classmethod
    def from_eval(cls, metric_name: str, values: list[BooleanOutputType]) -> Self:
        vector: np.ndarray = np.array(values, dtype=bool)

        n_valid_attempts = int(vector.size - np.count_nonzero(~np.isnan(vector)))
        if n_valid_attempts == 0:
            raise ValueError("Cannot compute MetricStats: all values are None")

        n_true = np.nansum(vector)
        return cls(
            metric_name=metric_name,
            n_attempts=vector.size,
            n_valid_attempts=n_valid_attempts,
            n_true=np.nansum(vector),
            n_false=n_valid_attempts - n_true,
        )

    @classmethod
    def _aggregate(cls, stats: list[Self]) -> Self:
        # create numpy array of shape ( # stats, 3)
        data = np.array([(v.n_attempts, v.n_valid_attempts, v.n_true) for v in stats])

        # sums will contain [total_attempts, total_valid_attempts, total_true]
        n_attempts, n_valid_attempts, n_true = data.sum(axis=0)

        return cls(
            metric_name=stats[0].metric_name,
            n_attempts=int(n_attempts),
            n_valid_attempts=int(n_valid_attempts),
            n_true=int(n_true),
            n_false=n_valid_attempts - n_true,
        )


@dataclass(frozen=True)
class CategoricalMetricStats(MetricStats[CategoricalOutputType]):
    metric_name: str
    counts: dict[CategoricalOutputType, int]
    frequencies: dict[CategoricalOutputType, float]
    mode: CategoricalOutputType

    @classmethod
    def from_eval(cls, metric_name: str, values: list[CategoricalOutputType]) -> Self:
        if not values:
            return cls(
                n_attempts=0,
                n_valid_attempts=0,
                metric_name=metric_name,
                counts={},
                frequencies={},
                mode=None,
            )

        vector = np.array(values)

        # unique elements and their counts
        unique, counts = np.unique(vector, return_counts=True)

        # build dict
        counts_dict = dict(zip(unique.tolist(), counts.tolist()))
        total = len(values)
        freq_dict = {k: v / total for k, v in counts_dict.items()}

        # find mode (the element with the maximum count)
        mode_val = unique[np.argmax(counts)]
        if str(mode_val).isdigit():
            mode_val = int(mode_val)
        else:
            mode_val = str(mode_val)

        return cls(
            metric_name=metric_name,
            n_attempts=vector.size,
            n_valid_attempts=int(vector.size - np.count_nonzero(~np.isnan(vector))),
            counts=counts_dict,
            frequencies=freq_dict,
            mode=mode_val,
        )

    @classmethod
    def _aggregate(cls, stats: list[Self]) -> Self:
        if not stats:
            raise ValueError("No stats provided to aggregate")

        # aggregate counts using a Counter
        total_counts: collections.Counter[CategoricalOutputType] = collections.Counter()
        n_attempts, n_valid_attempts = 0, 0
        for s in stats:
            n_attempts += s.n_attempts
            n_valid_attempts += s.n_valid_attempts
            total_counts.update(s.counts)

        # 2. Convert back to dict and calculate new frequencies
        final_counts = dict(total_counts)
        total_sum = sum(final_counts.values())
        final_freqs = {k: v / total_sum for k, v in final_counts.items()}

        # determine the new mode from the aggregated counts
        common = total_counts.most_common(1)
        final_mode = common[0][0] if common else None

        return cls(
            metric_name=stats[0].metric_name,
            n_attempts=n_attempts,
            n_valid_attempts=n_valid_attempts,
            counts=final_counts,
            frequencies=final_freqs,
            mode=final_mode,
        )
