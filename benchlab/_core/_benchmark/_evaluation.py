import logging
from dataclasses import dataclass, field
from typing import Any

from benchlab._core._benchmark._artifacts import BenchmarkArtifact, ArtifactType
from benchlab._core._benchmark._report import BenchmarkReport
from benchlab._core._evaluation._aggregator import BooleanAggregator
from benchlab._core._evaluation._metrics._metric import Metric, MetricType
from benchlab._core._types import InstanceType

__all__ = ["BenchmarkEval"]


@dataclass(frozen=True, slots=True)
class BenchmarkEval(BenchmarkArtifact[InstanceType]):
    spec: dict[str, Any] = field(default_factory=dict)
    instances: list[InstanceType] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))

    @staticmethod
    def _artifact_type() -> ArtifactType:
        return ArtifactType.EVALUATION

    def _artifact(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "instances": self.instances,
            "metrics": self.metrics,
        }

    def report(self) -> "BenchmarkReport":
        agg_map: dict[str, Any] = {}
        for metric in self.metrics:
            metric_name = metric.name
            metric_type = metric.type_
            match metric_type:
                case MetricType.BOOLEAN:
                    agg_cls = BooleanAggregator
                case _:
                    raise RuntimeError

            agg = agg_cls.from_stats(
                metric_name,
                [instance.evaluations[metric_name] for instance in self.instances],
            )
            agg_map[metric_name] = agg

        return BenchmarkReport(
            spec=self.spec,
            instances=self.instances,
            metrics=self.metrics,
            logger=self.logger,
        )
