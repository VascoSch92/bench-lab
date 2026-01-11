from dataclasses import dataclass
from typing import Generic, Self
from abc import ABC, abstractmethod
from benchlab._core._types import AggregatorType, AggregatorBooleanType


@dataclass(frozen=True, slots=True)
class Aggregator(ABC, Generic[AggregatorType]):
    @classmethod
    @abstractmethod
    def from_stats(cls, stats: list[AggregatorType]) -> Self: ...


@dataclass(frozen=True, slots=True)
class BooleanAggregator(Aggregator[AggregatorBooleanType]):
    macro_average: float

    micro_average: float

    pass_at_k: float

    consensus: float
    """Also called `majority voting`"""

    @classmethod
    def from_stats(cls, stats: list[AggregatorBooleanType]) -> Self:
        pass

    def estimated_pass_at_k(self) -> float: ...
