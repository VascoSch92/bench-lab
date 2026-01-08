from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, final, ClassVar, TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table
from benchlab._core._instances import Instance, AnswerType, MetricStats
from benchlab._core._logging import get_logger
from benchlab._core._tables import get_table_for_instance, export_table
from benchlab._core._time import _timed_exec

if TYPE_CHECKING:
    from benchlab._core._metrics import Metric


InstanceType = TypeVar("InstanceType", bound=Instance)

# TODO: add token usage or better usage
# todo: check how logger works if we have a logger in our main program
# todo: update logging to use rich
# todo: add callback method to stop retrying
# todo: divide run and eval from the API
# todo: method to save the output from a run

__all__ = ["Benchmark"]


class Benchmark(ABC, Generic[InstanceType]):
    """Class to run benchmarks"""

    """Map of the metrics for the benchmark"""
    _METRICS: ClassVar[dict[str, type["Metric"]]]

    def __init__(
        self,
        name: str,
        metrics: list[str] | None = None,
        instance_ids: list[str] | None = None,
        n_instance: int | None = None,
        n_attempts: int = 1,
        timeout: float | None = None,
        logs_filepath: str | None = None,
    ) -> None:
        self.logger = get_logger(name=__name__, path=logs_filepath, console=True)

        self.name = name

        self._metrics: dict[str, "Metric"] = {}
        self._register_metric(metrics)
        self._metrics_table: Table | None = None

        if n_instance is not None and n_instance <= 0:
            raise ValueError(
                "Argument `n_instance` must be a strictly positive integer, or `None` to select all the instances."
            )
        if n_instance is not None and instance_ids is not None:
            self.logger.warning(
                "Arguments `n_instance` and `instance_ids` are specified at the same time.\n"
                "`n_instance` instances will be selected from `instance_ids`."
            )
        self.instance_ids = instance_ids or []
        self.n_instance = n_instance

        self.n_attempts = n_attempts  # todo: check that it is positive

        if timeout is not None and timeout <= 0.0:
            raise ValueError(
                f"Argument `timeout` must be strictly positive. Got {timeout}"
            )
        self.timeout = timeout

        self._instances: list[Instance] | None = None

        self._instances_table: Table | None = None

    def _register_metric(self, metric_names: list[str] | None) -> None:
        if metric_names is None:
            return None
        if len(metric_names) != len(set(metric_names)):
            raise ValueError(
                "Duplicated metric names detected. Metric names must be unique."
            )

        for metric_name in metric_names:
            metric_cls = type(self)._METRICS.get(metric_name)

            if metric_cls is None:
                raise ValueError(
                    f"Metric `{metric_name}` is not supported by benchmark `{self.name}`.\n"
                    f"Available metrics: {sorted(type(self)._METRICS.keys())}"
                )

            self._metrics[metric_name] = metric_cls(logger=self.logger)

    @classmethod
    def new(
        cls,
        name: str,
        metrics: list[str] | None = None,
        instance_ids: list[str] | None = None,
        n_instance: int | None = None,
        n_attempts: int = 1,
        timeout: float | None = None,
        logs_filepath: str | None = None,
    ) -> "Benchmark":
        match name:
            case "GPQA":
                from benchlab._benchmarks._gpqa._benchmark import GPQABenchmark

                return GPQABenchmark(name=name, metrics=metrics)
            case "JailbreakLLMs":
                from benchlab._benchmarks._jailbreak_llms._benchmark import (
                    JailbreakLLMsBenchmark,
                )

                return JailbreakLLMsBenchmark(
                    name=name,
                    metrics=metrics,
                    logs_filepath=logs_filepath,
                    instance_ids=instance_ids,
                    n_instance=n_instance,
                    n_attempts=n_attempts,
                    timeout=timeout,
                )
            case _:
                raise ValueError(
                    f"`{name}` is not a valid benchmark name.\n"
                    f"Visit XXX to see implemented benchmark."
                )

    @final
    def load_dataset(self) -> list[Instance]:
        if self._instances is None:
            self.logger.info("Loading dataset from source")

            dataset = self._load_dataset()
            self._instances = self._filter_instances(dataset)

            self.logger.info(f"Loaded {len(self._instances)} instances")
            return self._instances

        self.logger.info("Dataset retrieved from cache")
        return self._instances

    def _filter_instances(self, dataset: list[Instance]) -> list[Instance]:
        """Private method to filter a dataset"""
        filter_dataset = dataset
        if self.instance_ids:
            filter_dataset = [
                instance for instance in dataset if instance.id in self.instance_ids
            ]
        if self.n_instance is not None:
            filter_dataset = filter_dataset[: self.n_instance]
        return filter_dataset

    @abstractmethod
    def _load_dataset(self) -> list[Instance]: ...

    def run(
        self,
        fn: Callable[[InstanceType, ...], AnswerType],
        args: dict[str, Any] | None = None,
    ) -> None:
        self.logger.info(f"Running benchmark for {fn.__name__}")
        if self._instances is None:
            self._instances = self.load_dataset()

        for instance in self._instances:
            for attempt_id in range(1, self.n_attempts + 1):
                timed_exec = _timed_exec(fn, self.timeout, instance, **(args or {}))

                if timed_exec.is_success():
                    self.logger.info(f"Instance {instance.id} successfull benchmarked")
                    status = "success"
                elif timed_exec.is_timeout():
                    self.logger.info(f"Instance {instance.id} timed out")
                    status = "timeout"
                else:
                    self.logger.error(
                        f"Error evaluating instance {instance.id}: {timed_exec.exception}"
                    )
                    status = "failure"

                instance.add_attempt(
                    response=timed_exec.result,
                    runtime=timed_exec.runtime,
                    status=status,
                )

            for metric in self._metrics.values():
                metric.evaluate(instance)

    async def run_async(self) -> None: ...

    def display_results(self) -> None:
        """
        Display benchmark results in a formatted console table using Rich.
        Caches the table after first generation.
        """
        if self._instances is None:
            raise ValueError("No instances available.")

        console = Console()

        if self._instances_table is not None:
            # if table is cached, we print it
            console.print(self._instances_table)
            return None

        table = get_table_for_instance(instances=self._instances, metrics=self._metrics)
        # cache table
        self._instances_table = table

        console.print(table)

    def export_results(self, filepath: str) -> None:
        if self._instances_table is None:
            self._instances_table = get_table_for_instance(
                instances=self._instances or [], metrics=self._metrics
            )

        export_table(self._instances_table, filepath=filepath)

    def display_metrics(self) -> None:
        if not self._metrics or not self._instances:
            raise ValueError("No metrics available.")

        if self._metrics_table is not None:
            print(self._metrics_table)
            return None

        aggregate_metric: dict[str, MetricStats] = {}
        for metric_name, metric in self._metrics.items():
            values = [instance._metrics[metric_name] for instance in self._instances]

            aggregate_metric[metric_name] = MetricStats.aggregate(values=values)

        table = []
        for metric_name, agg_value in aggregate_metric.items():
            table.append(f"{metric_name}: {agg_value}")

        self._metrics_table = table
        print(table)
