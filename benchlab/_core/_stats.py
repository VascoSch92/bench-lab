import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricStats(ABC):
    n_attempts: int
    n_valid_attempts: int

    @classmethod
    @abstractmethod
    def from_eval(cls, values: list[Any | None]) -> "MetricStats": ...

    @classmethod
    @abstractmethod
    def aggregate(cls, values: list["MetricStats"]) -> "MetricStats": ...


@dataclass(frozen=True)
class RegressionMetricStats(MetricStats):
    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def from_eval(cls, values: list[int | float | None]) -> "MetricStats":
        n_attempts = len(values)

        valid_values = [v for v in values if v is not None]
        n_valid_attempts = len(valid_values)

        if n_valid_attempts == 0:
            raise ValueError("Cannot compute MetricStats: all values are None")

        mean = sum(valid_values) / n_valid_attempts

        if n_valid_attempts == 1:
            std = 0.0
        else:
            variance = sum((v - mean) ** 2 for v in valid_values) / n_valid_attempts
            std = math.sqrt(variance)

        return cls(
            n_attempts=n_attempts,
            n_valid_attempts=n_valid_attempts,
            mean=mean,
            std=std,
            min=min(valid_values),
            max=max(valid_values),
        )

    @classmethod
    def aggregate(
        cls, values: list["RegressionMetricStats"]
    ) -> "RegressionMetricStats":
        if not values:
            raise ValueError("Cannot aggregate empty list of MetricStats")

        total_attempts = sum(v.n_attempts for v in values)
        total_valid = sum(v.n_valid_attempts for v in values)

        if total_valid == 0:
            raise ValueError("No valid attempts to aggregate")

        # Weighted mean
        weighted_sum = sum(v.mean * v.n_valid_attempts for v in values)
        mean_value = weighted_sum / total_valid

        # Weighted population variance
        weighted_var_sum = sum(
            v.n_valid_attempts * (v.std**2 + (v.mean - mean_value) ** 2) for v in values
        )
        std_value = math.sqrt(weighted_var_sum / total_valid)

        min_value = min(v.min for v in values)
        max_value = max(v.max for v in values)

        return cls(
            n_attempts=total_attempts,
            n_valid_attempts=total_valid,
            mean=mean_value,
            std=std_value,
            min=min_value,
            max=max_value,
        )


@dataclass(frozen=True)
class BooleanMetricStats(MetricStats):
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
    def from_eval(cls, values: list[bool | None]) -> "BooleanMetricStats":
        values = list(values)

        n_attempts = len(values)
        valid = [v for v in values if v is not None]

        n_valid_attempts = len(valid)
        n_true = sum(1 for v in valid if v)
        n_false = n_valid_attempts - n_true

        return cls(
            n_attempts=n_attempts,
            n_valid_attempts=n_valid_attempts,
            n_true=n_true,
            n_false=n_false,
        )

    @classmethod
    def aggregate(cls, values: list["BooleanMetricStats"]) -> "BooleanMetricStats":
        n_attempts = sum(v.n_attempts for v in values)
        n_valid_attempts = sum(v.n_valid_attempts for v in values)
        n_true = sum(v.n_true for v in values)
        n_false = n_valid_attempts - n_true
        return cls(
            n_attempts=n_attempts,
            n_valid_attempts=n_valid_attempts,
            n_true=n_true,
            n_false=n_false,
        )


@dataclass(frozen=True)
class CategoricalMetricStats(MetricStats):
    counts: dict[str, int]
    frequencies: dict[str, float]
    mode: str | None

    @classmethod
    def from_eval(cls, values: list[Any | None]) -> "CategoricalMetricStats":
        # todo: complete
        raise NotImplementedError

    @classmethod
    def aggregate(cls, values: list["MetricStats"]) -> "CategoricalMetricStats":
        raise NotImplementedError
