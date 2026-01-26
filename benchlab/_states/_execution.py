import collections
import time
from dataclasses import dataclass
from typing import DefaultDict

from rich import table

from benchlab._states._base import BaseBenchmark
from benchlab._states._evaluation import BenchmarkEval
from benchlab._metrics.base import Metric
from benchlab._types import InstanceType

__all__ = ["BenchmarkExec"]


@dataclass(frozen=True, slots=True)
class BenchmarkExec(BaseBenchmark[InstanceType]):
    """
    Represents the state of a benchmark after execution but before evaluation.

    This class holds the raw results (responses, runtimes, and statuses) for all
    benchmark instances. it provides methods to attach metrics and transition the
    benchmark data into the evaluation phase.
    """

    def add_metric(self, metric: Metric) -> None:
        if metric in self._metrics:
            self.logger.info(f"Metric {metric.name} already present.")

        self._metrics.append(metric)
        self.logger.info(f"Metric {metric.name} added successfully.")

    def evaluate(self) -> BenchmarkEval[InstanceType]:
        start_time = time.perf_counter()

        for instance in self.instances:
            for metric in self._metrics:
                evals = metric.evaluate(instance=instance, attempts=instance.attempts)
                instance.add_eval(metric_name=metric.name, evals=evals)

        updated_spec = self._spec.set_evaluation_time(time.perf_counter() - start_time)
        return BenchmarkEval.new(
            source=list(self.instances),
            metrics=self.metrics,
            aggregators=self.aggregators,
            logger=self.logger,
            **updated_spec.to_dict(),
        )

    def _generate_summary_table(self) -> table.Table:
        stats: DefaultDict[str, int] = collections.defaultdict(int)

        for instance in self.instances:
            for status in instance.statuses:
                stats[status] += 1

        summary_table = table.Table(title="Execution Summary")

        summary_table.add_column("Success Runs", style="green")
        summary_table.add_column("Failure Runs", style="red")
        summary_table.add_column("Timeout runs", style="yellow")
        summary_table.add_column("Execution Time", style="cyan")

        summary_table.add_row(
            str(stats["success"]),
            str(stats["failure"]),
            str(stats["timeout"]),
            str(self._spec.execution_time),
        )

        return summary_table
