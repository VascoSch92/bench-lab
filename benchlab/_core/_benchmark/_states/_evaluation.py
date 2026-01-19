import collections
from dataclasses import dataclass
from functools import cached_property
from typing import DefaultDict

from benchlab._core._benchmark._states._base import BaseBenchmark
from benchlab._core._benchmark._states._report import BenchmarkReport
from benchlab._core._evaluation._aggregators._aggregator import AggregatorType
from benchlab._core._evaluation._metrics._metric import MetricType, Metric
from benchlab._core._types import InstanceType

__all__ = ["BenchmarkEval"]


@dataclass(frozen=True, slots=True)
class BenchmarkEval(BaseBenchmark[InstanceType]):
    @cached_property
    def _metric_type_to_metrics(self) -> DefaultDict[MetricType, list[Metric]]:
        map_ = collections.defaultdict(list)
        for metric in self.metrics:
            map_[metric.type_].append(metric)
        return map_

    def report(self) -> BenchmarkReport[InstanceType]:
        reports = []
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
