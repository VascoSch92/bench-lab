import logging
from dataclasses import dataclass, field
from typing import Any

from benchlab._core._evaluation._metrics import _METRIC_TYPE_TO_STATS
from benchlab._core._evaluation._stats import MetricStats
from benchlab._core._types import InstanceType
from benchlab._core._evaluation._metrics import Metric
from benchlab._core._benchmark._artifacts import BenchmarkArtifact, ArtifactType
from benchlab._core._benchmark._report import BenchmarkReport
import copy


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
        artifact: dict[str, Any] = {}

        artifact["spec"] = self.spec.get("spec", {})
        artifact["instances"] = [instance.to_dict() for instance in self.instances]
        artifact["metrics"] = [metric.to_dict() for metric in self.metrics]

        return artifact

    def _name_to_type(self) -> dict:
        return {
            metric.name : metric.type_
            for metric in self.metrics
        }
    
    def report(self) -> BenchmarkReport:
        instances = copy.deepcopy(self.instances)
        _name_to_type = self._name_to_type()

        for instance in instances:
            for metric_name, eval in instance.evaluations.items():
                metric_type = _name_to_type[metric_name]
                stats_cls = _METRIC_TYPE_TO_STATS[metric_type]
                stats = stats_cls.from_eval(metric_name=metric_name, values=eval)
        
        

    def get_results_for(self, metric_name: str) -> MetricStats:
        evals = []
        for instance in self.instances:
            evals.extend(instance.evaluations[metric_name])

        metric_type = [m.type_ for m in self.metrics if m.name == metric_name][0]
        stats = _METRIC_TYPE_TO_STATS[metric_type].from_eval(
            metric_name=metric_name, values=evals
        )
        return stats

    # def display(self) -> None:
    #     if not self.metrics and not self.instances:
    #         return None
    #
    #     aggregate_metrics: list[MetricStats] = []
    #     for metric in self.metrics:
    #         values = [instance.stats[metric.name] for instance in self.instances]
    #         agg_metric = MetricStats.aggregate(stats=values)
    #         aggregate_metrics.append(agg_metric)
    #
    #     table = []
    #     for metric_stats in aggregate_metrics:
    #         table.append(f"{metric_stats.metric_name}: {metric_stats.n_valid_attempts}")
    #
    #     print(table)

    def to_csv(self) -> None:
        pass
