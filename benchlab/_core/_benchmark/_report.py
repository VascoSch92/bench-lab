import logging
from dataclasses import dataclass, field

from benchlab._core._benchmark._artifacts import BenchmarkArtifact
from benchlab._core._benchmark._spec import Spec
from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._types import InstanceType

__all__ = ["BenchmarkReport"]


@dataclass(frozen=True, slots=True)
class BenchmarkReport(BenchmarkArtifact[InstanceType]):
    instances: list[InstanceType]
    spec: Spec = field(default_factory=Spec.new)
    metrics: list[Metric] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))
