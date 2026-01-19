import collections
from dataclasses import dataclass
from functools import cached_property
from typing import DefaultDict

from rich import table

from benchlab._core._benchmark._states._base import BaseBenchmark
from benchlab._core._benchmark._states._report import BenchmarkReport
from benchlab._core._evaluation._aggregators._aggregator import AggregatorType, Report
from benchlab._core._evaluation._metrics._metric import MetricType, Metric
from benchlab._core._types import InstanceType

__all__ = ["BenchmarkEval"]


@dataclass(frozen=True, slots=True)
class BenchmarkEval(BaseBenchmark[InstanceType]):
    def _task_specific_checks(self) -> None:
        # todo: complete with check for instance
        pass

    @cached_property
    def _metric_type_to_metrics(self) -> DefaultDict[MetricType, list[Metric]]:
        map_ = collections.defaultdict(list)
        for metric in self.metrics:
            map_[metric.type_].append(metric)
        return map_

    def report(self) -> BenchmarkReport[InstanceType]:
        reports: list[Report] = []
        for aggregator in self.aggregators:
            match aggregator.type_:
                case AggregatorType.RUNTIME:
                    report = aggregator.aggregate(instances=self._instances)
                    reports.append(report)
                case AggregatorType.STATUS:
                    report = aggregator.aggregate(instances=self._instances)
                    reports.append(report)
                case AggregatorType.BOOLEAN_METRICS:
                    for metric in self._metric_type_to_metrics[aggregator.type_]:
                        if metric.type_ == aggregator.type_:
                            report = aggregator.aggregate(self.instances)
                            reports.append(report)
                case _:
                    raise RuntimeError(f"Unknown aggregator type: {aggregator}")

        return BenchmarkReport(
            _spec=self._spec,
            _instances=self._instances,
            _metrics=self._metrics,
            _aggregators=self.aggregators,
            _reports=reports,
            logger=self.logger,
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
