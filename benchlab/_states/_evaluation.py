import collections
import time
from dataclasses import dataclass
from functools import cached_property
from typing import DefaultDict

from rich import table

from benchlab._metrics.base import MetricType, Metric
from benchlab._states._base import BaseBenchmark
from benchlab._states._report import BenchmarkReport
from benchlab._types import InstanceType
from benchlab.aggregators._base import Report

__all__ = ["BenchmarkEval"]


@dataclass(frozen=True, slots=True)
class BenchmarkEval(BaseBenchmark[InstanceType]):
    """
    Represents the state of a benchmark where all instances have been evaluated
    against registered metrics.

    This class serves as the analytical layer of the benchmarking process. It organizes
    raw metric scores across instances and attempts, providing the logic to aggregate
    these scores into a final structured report.
    """

    @cached_property
    def _metric_type_to_metrics(self) -> DefaultDict[MetricType, list[Metric]]:
        map_ = collections.defaultdict(list)
        for metric in self.metrics:
            map_[metric.type_].append(metric)
        return map_

    def report(self) -> BenchmarkReport[InstanceType]:
        start_time = time.perf_counter()

        reports: list[Report] = [
            aggregator.aggregate(list(self.instances))
            for aggregator in self.aggregators
        ]

        updated_spec = self._spec.set_execution_time(time.perf_counter() - start_time)
        return BenchmarkReport.new(
            source=list(self.instances),
            metrics=self.metrics,
            aggregators=self.aggregators,
            logger=self.logger,
            **updated_spec.to_dict(),
            _reports=reports,
        )

    def _generate_summary_table(self) -> table.Table:
        summary_table = table.Table(title="Evaluation Summary")

        summary_table.add_column("Instance Id")

        metrics = sorted(metric.name for metric in self.metrics)
        for metric_name in metrics:
            for j in range(1, self.spec.n_attempts + 1):
                summary_table.add_column(f"{metric_name} (Attempt {j})")

        for instance in self.instances:
            row = [instance.id]
            evaluations = instance.evaluations
            for metric_name in metrics:
                attempts = evaluations[metric_name]

                for attempt in attempts:
                    row.append(attempt if attempt else "None")

            summary_table.add_row(*row)

        return summary_table
