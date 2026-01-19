from dataclasses import dataclass, field

from benchlab._core._benchmark._states._base import BaseBenchmark
from benchlab._core._types import InstanceType
from benchlab._core._evaluation._aggregators._aggregator import Report


__all__ = ["BenchmarkReport"]


@dataclass(frozen=True, slots=True)
class BenchmarkReport(BaseBenchmark[InstanceType]):
    _reports: list[Report] = field(default_factory=list)
