import copy
import importlib
import re
import time
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Self, ClassVar

from rich import table

from benchlab._core._benchmark._spec import Spec
from benchlab._core._benchmark._states._base import BaseBenchmark
from benchlab._core._benchmark._states._execution import BenchmarkExec
from benchlab._core._evaluation._aggregators._aggregator import Aggregator
from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._logging import get_logger
from benchlab._core._time import timed_exec
from benchlab._core._types import BenchmarkCallable, InstanceType

# todo: add token usage or better usage
# todo: check how logger works if we have a logger in our main program
# todo: update logging to use rich
# todo: add callback method to stop retrying

__all__ = ["Benchmark"]


@dataclass(frozen=True, slots=True)
class Benchmark(BaseBenchmark[InstanceType]):
    """Class to run benchmarks"""

    _METRICS: ClassVar[dict[str, type["Metric"]]]
    """Map of the metrics for the benchmark"""

    def _task_specific_checks(self) -> None:
        pass

    @classmethod
    def new(
        cls,
        name: str,
        instances: list[InstanceType] | None = None,
        metrics: list[Metric] | None = None,
        aggregators: list[Aggregator] | None = None,
        instance_ids: list[str] | None = None,
        n_instance: int | None = None,
        n_attempts: int = 1,
        timeout: float | None = None,
        logs_filepath: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        logger = get_logger(name=__name__, path=logs_filepath, console=True)
        spec = Spec(
            name=name,
            instance_ids=instance_ids or [],
            n_instance=n_instance,
            n_attempts=n_attempts,
            timeout=timeout,
            logs_filepath=logs_filepath,
        )
        if n_instance is not None and instance_ids is not None:
            logger.warning(
                "Arguments `n_instance` and `instance_ids` are specified at the same time.\n"
                "`n_instance` instances will be selected from `instance_ids`."
            )
        return cls(
            _spec=spec,
            _instances=instances or [],
            _metrics=metrics or [],
            _aggregators=aggregators or [],
            logger=logger,
        )

    @classmethod
    def from_library(
        cls, name: str, metric_names: list[str], **kwargs
    ) -> "Benchmark[Any]":
        # convert from camel case to snake case
        snake_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        module_path = f"benchlab._library._{snake_name}._benchmark"
        class_name = f"{name}Benchmark"

        try:
            module = importlib.import_module(module_path)
            benchmark_cls = getattr(module, class_name)

            metric_cls = cls._convert_metric_names_to_cls(
                benchmark_cls, metric_names, name
            )

            return benchmark_cls.new(
                name=name, metrics=[metric() for metric in metric_cls], **kwargs
            )
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not find benchmark `{name}`. "
                f"Check naming conventions. Error: {e}"
            )

    @staticmethod
    def _convert_metric_names_to_cls(
        benchmark_cls: type["Benchmark"], metric_names, name
    ) -> list[type[Metric]]:
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("Duplicated metrics. Metrics must be unique.")
        metric_cls = []
        for metric_name in metric_names:
            if metric_name not in benchmark_cls._METRICS:
                raise ValueError(
                    f"Metric `{metric_name}` is not supported by benchmark `{name}`.\n"
                    f"Available metrics: {sorted(benchmark_cls._METRICS.keys())}"
                )
            metric_cls.append(benchmark_cls._METRICS[metric_name])
        return metric_cls

    @cached_property
    def instances(self) -> list[InstanceType]:
        self.logger.info("Loading benchmark instances...")
        dataset = self._load_dataset()
        instances = self._filter_instances(dataset)
        self.logger.info(f"Loaded {len(instances)} instances")
        return instances

    def _filter_instances(self, dataset: list[InstanceType]) -> list[InstanceType]:
        """Private method to filter a dataset"""
        filter_dataset = dataset
        if self._spec.instance_ids:
            filter_dataset = [
                instance
                for instance in dataset
                if instance.id in self._spec.instance_ids
            ]
        if self._spec.n_instance is not None:
            filter_dataset = filter_dataset[: self._spec.n_instance]
        return filter_dataset

    @abstractmethod
    def _load_dataset(self) -> list[InstanceType]: ...

    def run(
        self,
        fn: BenchmarkCallable,
        args: dict[str, Any] | None = None,
    ) -> BenchmarkExec:
        start_time = time.perf_counter()

        self.logger.info(f"Running benchmark {self._spec.name} for {fn.__name__}")

        return_type = fn.__annotations__.get("return", None)
        if return_type is None:
            self.logger.warning("No return type detected")

        # deepcopy instances to not change the ones owned by the benchmark
        instances = copy.deepcopy(self.instances)

        for instance in instances:
            for attempt_id in range(1, self._spec.n_attempts + 1):
                timed_execution = timed_exec(
                    fn, self._spec.timeout, instance, **(args or {})
                )

                if timed_execution.is_success:
                    self.logger.info(f"Instance {instance.id} successfull benchmarked")
                    status = "success"
                elif timed_execution.is_timeout:
                    self.logger.info(f"Instance {instance.id} timed out")
                    status = "timeout"
                elif timed_execution.is_error:
                    self.logger.error(
                        f"Error evaluating instance {instance.id}: {timed_execution.exception}"
                    )
                    status = "failure"
                else:
                    raise RuntimeError("This should never happens.")

                instance.add_attempt(
                    response=timed_execution.result,
                    runtime=timed_execution.runtime,
                    status=status,
                )

        update_spec = self._spec.set_execution_time(time.perf_counter() - start_time)
        return BenchmarkExec(
            _instances=instances,
            _metrics=self._metrics,
            _aggregators=self.aggregators,
            logger=self.logger,
            _spec=update_spec,
        )

    async def run_async(self) -> None: ...

    def _generate_summary_table(self) -> table.Table:
        """
        Generates a rich table summary of the benchmark configuration,
        including metrics, aggregators, and instance constraints.
        """
        summary_table = table.Table(title="Benchmark Summary")

        summary_table.add_column("Property", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row("Benchmark name", self._spec.name)
        summary_table.add_row("Number of Instances", str(len(self.instances)))
        summary_table.add_row("Attempts per Instance", str(self._spec.n_attempts))
        summary_table.add_row(
            "Timeout", f"{self._spec.timeout}s" if self._spec.timeout else "None"
        )

        # Metrics & Aggregators
        metrics_list = ", ".join([m.name for m in self._metrics]) or "None"
        aggs_list = ", ".join([a.name for a in self.aggregators]) or "None"

        summary_table.add_row("Metrics", metrics_list)
        summary_table.add_row("Aggregators", aggs_list)
        if self._spec.logs_filepath:
            summary_table.add_row("Logs Path", self._spec.logs_filepath)

        return summary_table
