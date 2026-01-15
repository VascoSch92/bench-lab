import copy
from dataclasses import dataclass

from benchlab._core._benchmark._states._base import BaseBenchmark
from benchlab._core._benchmark._states._evaluation import BenchmarkEval
from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._types import InstanceType

__all__ = ["BenchmarkExec"]


@dataclass(frozen=True, slots=True)
class BenchmarkExec(BaseBenchmark[InstanceType]):
    def add_metric(self, metric: Metric) -> None:
        if metric in self._metrics:
            raise ValueError(f"Metric {metric} is already present.")

        self._metrics.append(metric)
        self.logger.info(f"Metric {metric.name} added successfully.")

    def evaluate(self) -> BenchmarkEval[InstanceType]:
        instances = copy.deepcopy(self._instances)

        for metric in self._metrics:
            for instance in instances:
                evals = metric.evaluate(instance=instance, attempts=instance.attempts)
                instance.add_eval(metric_name=metric.name, evals=evals)

        return BenchmarkEval(
            _spec=self._spec,
            _instances=instances,
            logger=self.logger,
            _metrics=self._metrics,
        )
