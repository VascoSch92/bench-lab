import copy
import logging
from dataclasses import dataclass, field

from benchlab._core._benchmark._artifacts import BenchmarkArtifact
from benchlab._core._benchmark._evaluation import BenchmarkEval
from benchlab._core._benchmark._spec import Spec
from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._types import InstanceType

__all__ = ["BenchmarkExec"]


@dataclass(frozen=True, slots=True)
class BenchmarkExec(BenchmarkArtifact[InstanceType]):
    spec: Spec = field(default_factory=Spec.new)
    instances: list[InstanceType] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))

    def add_metric(self, metric: Metric) -> None:
        if metric in self.metrics:
            raise ValueError(f"Metric {metric} is already present.")

        self.metrics.append(metric)
        self.logger.info(f"Metric {metric.name} added successfully.")

    def evaluate(self) -> BenchmarkEval:
        instances = copy.deepcopy(self.instances)

        for metric in self.metrics:
            for instance in instances:
                evals = metric.evaluate(instance=instance, attempts=instance.attempts)
                instance.add_eval(metric_name=metric.name, evals=evals)

        return BenchmarkEval(
            spec=self.spec,
            instances=instances,
            logger=self.logger,
            metrics=self.metrics,
        )
