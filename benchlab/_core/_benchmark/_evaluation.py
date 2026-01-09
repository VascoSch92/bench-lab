import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Self

from benchlab._core._benchmark._load_utils import get_instances_from_json
from benchlab._core._metrics import _METRIC_TYPE_TO_STATS
from benchlab._core._stats import MetricStats
from benchlab._core._types import InstanceType
from benchlab._core._metrics import Metric

__all__ = ["BenchmarkEval"]


@dataclass(frozen=True, slots=True)
class BenchmarkEval(Generic[InstanceType]):
    instances: list[InstanceType]
    metrics: list[Metric] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))
    metadata: dict[str, Any] = field(default_factory=dict)

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

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        instances: list[InstanceType] = get_instances_from_json(data["instances"])

        return cls(
            metadata=data["metadata"],
            instances=instances,
        )

    def to_json(self, output_path: Path | str) -> None:
        output_path = Path(output_path)

        if not output_path.suffix:
            output_path = output_path.with_suffix(".json")

        file = {
            "metadata": self.metadata,
            "instances": [instance.to_dict() for instance in self.instances],
        }
        with output_path.open("w") as f:
            json.dump(file, f, indent=4)

    @classmethod
    def from_exec(
        cls,
    ) -> "BenchmarkEval":
        return None

    def to_csv(self) -> None:
        pass
