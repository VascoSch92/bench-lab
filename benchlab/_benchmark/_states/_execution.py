import collections
import copy
from dataclasses import dataclass
from typing import DefaultDict

from rich import table

from benchlab._benchmark._states._base import BaseBenchmark
from benchlab._benchmark._states._evaluation import BenchmarkEval
from benchlab.metrics._base import Metric
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

    def _task_specific_checks(self) -> None:
        # todo: complete with check that every instance was runned the correct number of times
        pass

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
            _aggregators=self._aggregators,
        )

    def _generate_summary_table(self) -> table.Table:
        stats: DefaultDict[str, int] = collections.defaultdict(int)

        for instance in self.instances:
            for status in instance.statuses:
                match status:
                    case "success":
                        stats["success"] += 1
                    case "failure":
                        stats["failure"] += 1
                    case "timeout":
                        stats["timeout"] += 1

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
