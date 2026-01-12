import logging
from dataclasses import dataclass, field
from typing import Any

from benchlab._core._types import InstanceType
from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._benchmark._artifacts import BenchmarkArtifact, ArtifactType


__all__ = ["BenchmarkReport"]


@dataclass(frozen=True, slots=True)
class BenchmarkReport(BenchmarkArtifact[InstanceType]):
    instances: list[InstanceType]
    spec: dict[str, Any] = field(default_factory=dict)
    metrics: list[Metric] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))

    @staticmethod
    def _artifact_type() -> ArtifactType:
        return ArtifactType.REPORT

    def _artifact(self) -> dict[str, Any]:
        return {"spec": self.spec, "instances": self.instances, "metrics": self.metrics}
