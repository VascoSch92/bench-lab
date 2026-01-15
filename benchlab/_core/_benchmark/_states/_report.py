from dataclasses import dataclass

from benchlab._core._benchmark._states._base import BaseBenchmark
from benchlab._core._types import InstanceType

__all__ = ["BenchmarkReport"]


@dataclass(frozen=True, slots=True)
class BenchmarkReport(BaseBenchmark[InstanceType]):
    pass
