import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from benchlab._core._benchmark._evaluation import BenchmarkEval
from benchlab._core._types import InstanceType
from benchlab._core._evaluation._metrics import Metric
from benchlab._core._benchmark._artifacts import BenchmarkArtifact, ArtifactType


__all__ = ["BenchmarkExec"]


@dataclass(frozen=True, slots=True)
class BenchmarkExec(BenchmarkArtifact[InstanceType]):
    spec: dict[str, Any] = field(default_factory=dict)
    instances: list[InstanceType] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))

    @staticmethod
    def _artifact_type() -> ArtifactType:
        return ArtifactType.EXECUTION

    def _artifact(self) -> dict[str, Any]:
        artifact: dict[str, Any] = {}

        artifact["spec"] = self.spec.get("spec", {})
        artifact["instances"] = [instance.to_dict() for instance in self.instances]
        artifact["metrics"] = [metric.to_dict() for metric in self.metrics]

        return artifact

    def evaluate(self) -> BenchmarkEval:
        instances = copy.deepcopy(self.instances)

        for metric in self.metrics:
            for instance in instances:
                evals = metric.evaluate(instance=instance, attempts=instance.attempts)
                # from here we can retrieve the metric type and compute directly the metric for the eval :-) 
                instance.add_eval(metric_name=metric.name, value=evals)

        return BenchmarkEval(
            spec=self.spec,
            instances=instances,
            logger=self.logger,
            metrics=self.metrics,
        )

    async def evaluate_async(self) -> BenchmarkEval:
        pass
