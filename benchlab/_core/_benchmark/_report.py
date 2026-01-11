import logging
from dataclasses import dataclass, field
from typing import Any

from benchlab._core._types import InstanceType
from benchlab._core._evaluation._metrics import Metric
from benchlab._core._benchmark._artifacts import BenchmarkArtifact, ArtifactType


__all__ = ["BenchmarkReport"]


@dataclass(frozen=True, slots=True)
class BenchmarkReport(BenchmarkArtifact[InstanceType]):
    spec: dict[str, Any] = field(default_factory=dict)
    instances: list[InstanceType]
    metrics: list[Metric] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))

    def _artifact_type(self) -> ArtifactType:
        return ArtifactType.REPORT
