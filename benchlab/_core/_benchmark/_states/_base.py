import logging
from dataclasses import dataclass, field

from benchlab._core._benchmark._artifacts import BenchmarkArtifact
from benchlab._core._benchmark._spec import Spec
from benchlab._core._evaluation._aggregators._aggregator import Aggregator
from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._types import InstanceType


@dataclass(frozen=True, slots=True)
class BaseBenchmark(BenchmarkArtifact[InstanceType]):
    """
    Base class to enforce structure across Benchmark execution and evaluation.
    """

    _spec: Spec = field(default_factory=Spec.new)
    _instances: list[InstanceType] = field(default_factory=list)
    _metrics: list[Metric] = field(default_factory=list)
    _aggregators: list[Aggregator] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))

    @property
    def spec(self) -> Spec:
        return self._spec

    @property
    def instances(self) -> list[InstanceType]:
        return self._instances

    @property
    def metrics(self) -> list[Metric]:
        return self._metrics

    @property
    def aggregators(self) -> list[Aggregator]:
        return self._aggregators
