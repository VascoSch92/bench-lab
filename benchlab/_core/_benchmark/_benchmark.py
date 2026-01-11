import copy
from abc import abstractmethod
from typing import Any
from typing import final, ClassVar

from benchlab._core._benchmark._execution import BenchmarkExec
from benchlab._core._logging import get_logger
from benchlab._core._evaluation._metrics import Metric
from benchlab._core._time import _timed_exec
from benchlab._core._types import BenchmarkCallable, InstanceType
from benchlab._core._benchmark._artifacts import BenchmarkArtifact, ArtifactType


# todo: add token usage or better usage
# todo: check how logger works if we have a logger in our main program
# todo: update logging to use rich
# todo: add callback method to stop retrying

__all__ = ["Benchmark"]


class Benchmark(
    BenchmarkArtifact[InstanceType],
):
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

        self.metrics: list[Metric] = self._register_metric(metrics or [])

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

        if n_attempts <= 0:
            raise ValueError("Argument `n_attempts` must be strictly positive integer.")
        self.n_attempts = n_attempts  # todo: check that it is positive

        if timeout is not None and timeout <= 0.0:
            raise ValueError(
                f"Argument `timeout` must be strictly positive. Got {timeout}"
            )
        self.timeout = timeout

        self._instances: list[InstanceType] | None = None

    def _register_metric(self, metric_names: list[str]) -> list[Metric]:
        if len(metric_names) != len(set(metric_names)):
            raise ValueError(
                "Duplicated metric names detected. Metric names must be unique."
            )

        metrics: list[Metric] = []
        for metric_name in metric_names:
            metric_cls = type(self)._METRICS.get(metric_name, None)

            if metric_cls is None:
                raise ValueError(
                    f"Metric `{metric_name}` is not supported by benchmark `{self.name}`.\n"
                    f"Available metrics: {sorted(type(self)._METRICS.keys())}"
                )

            metrics.append(metric_cls(logger=self.logger))
        return metrics

    @staticmethod
    def _artifact_type() -> ArtifactType:
        return ArtifactType.BENCHMARK

    def _artifact(self) -> dict[str, Any]:
        artifact: dict[str, Any] = {}

        artifact["spec"] = {}
        artifact["spec"]["name"] = self.name
        if self.n_instance is not None:
            artifact["spec"]["n_instance"] = self.n_instance
        if self.n_attempts is not None:
            artifact["spec"]["n_attempts"] = self.n_attempts
        if self.timeout is not None:
            artifact["spec"]["timeout"] = self.timeout

        if self._instansces is not None:
            artifact["instances"] = [instance.to_dict() for instance in self._instances]
        else:
            artifact["instances"] = []
            
        artifact["metrics"] = [metric.to_dict() for metric in self.metrics]

        return artifact

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
    ) -> "Benchmark[Any]":
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
    def load_dataset(self) -> list[InstanceType]:
        if self._instances is None:
            self.logger.info("Loading dataset from source")

            dataset = self._load_dataset()
            self._instances = self._filter_instances(dataset)

            self.logger.info(f"Loaded {len(self._instances)} instances")
            return self._instances

        self.logger.info("Dataset retrieved from cache")
        return self._instances

    def _filter_instances(self, dataset: list[InstanceType]) -> list[InstanceType]:
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
    def _load_dataset(self) -> list[InstanceType]: ...

    def run(
        self,
        fn: BenchmarkCallable,
        args: dict[str, Any] | None = None,
    ) -> BenchmarkExec:
        self.logger.info(f"Running benchmark {self.name} for {fn.__name__}")

        if self._instances is None:
            self._instances = self.load_dataset()

        # deepcopy instances to not change the ones owned by the benchmark
        instances = copy.deepcopy(self._instances)

        for instance in instances:
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

        return BenchmarkExec(
            instances=instances,
            metrics=self.metrics,
            logger=self.logger,
            spec=self._artifact(),
        )

    async def run_async(self) -> None: ...
