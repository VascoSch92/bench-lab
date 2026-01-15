from dataclasses import dataclass
from typing import Any

from benchlab._core._benchmark._states._base import BaseBenchmark
from benchlab._core._benchmark._states._report import BenchmarkReport
from benchlab._core._types import InstanceType

__all__ = ["BenchmarkEval"]


@dataclass(frozen=True, slots=True)
class BenchmarkEval(BaseBenchmark[InstanceType]):
    def report(self) -> BenchmarkReport[InstanceType]:
        agg_map: dict[str, Any] = {}
        for metric in self._metrics:
            metric_name = metric.name
            metric_type = metric.type_
            match metric_type:
                case _:
                    raise RuntimeError

            agg = agg_cls.aggregate(
                metric_name,
                [instance.evaluations[metric_name] for instance in self._instances],
            )
            agg_map[metric_name] = agg

        return BenchmarkReport(
            _spec=self._spec,
            _instances=self._instances,
            _metrics=self._metrics,
            logger=self.logger,
        )
