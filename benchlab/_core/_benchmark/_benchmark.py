import copy
import importlib
import re
from abc import abstractmethod
from typing import Any
from typing import final, ClassVar

from benchlab._core._benchmark._artifacts import BenchmarkArtifact, ArtifactType
from benchlab._core._benchmark._execution import BenchmarkExec
from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._logging import get_logger
from benchlab._core._time import _timed_exec
from benchlab._core._types import BenchmarkCallable, InstanceType

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
        instances: list[InstanceType] | None = None,
        metrics: list[type[Metric]] | None = None,
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

        self.instances = instances

    def _register_metric(self, metrics: list[type[Metric]]) -> list[Metric]:
        return [metric_cls(self.logger) for metric_cls in metrics]

    @classmethod
    def new(cls, instances: list[InstanceType], **kwargs) -> "Benchmark[Any]":
        cls._check_consistency_instances(instances)
        return cls(instances=instances, **kwargs)

    @staticmethod
    def _check_consistency_instances(instances: list[InstanceType]) -> None:
        if not instances:
            return None

        first_type = type(instances[0])
        if all(type(i) is first_type for i in instances):
            raise ValueError("All instances must have the same type.")

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

            return benchmark_cls(name=name, metrics=metric_cls, **kwargs)
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

    @staticmethod
    def _artifact_type() -> ArtifactType:
        return ArtifactType.BENCHMARK

    def _artifact(self) -> dict[str, Any]:
        artifact: dict[str, Any] = {"spec": {}}

        artifact["spec"]["name"] = self.name
        artifact["spec"]["is_from_library"] = "library" in self.__class__.__module__
        if self.n_instance is not None:
            artifact["spec"]["n_instance"] = self.n_instance
        if self.n_attempts is not None:
            artifact["spec"]["n_attempts"] = self.n_attempts
        if self.timeout is not None:
            artifact["spec"]["timeout"] = self.timeout

        if self.instances is not None:
            artifact["instances"] = self.instances
        else:
            artifact["instances"] = []

        artifact["metrics"] = self.metrics

        return artifact

    @final
    def load_dataset(self) -> list[InstanceType]:
        if self.instances is None:
            self.logger.info("Loading dataset from source")

            dataset = self._load_dataset()
            self.instances = self._filter_instances(dataset)

            self.logger.info(f"Loaded {len(self.instances)} instances")
            return self.instances

        self.logger.info("Dataset retrieved from cache")
        return self.instances

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

        return_type = fn.__annotations__.get("return", None)
        if return_type is None:
            self.logger.warning("No return type detected")

        if self.instances is None:
            self.instances = self.load_dataset()

        # deepcopy instances to not change the ones owned by the benchmark
        instances = copy.deepcopy(self.instances)

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
            spec=self._artifact()["spec"],
        )

    async def run_async(self) -> None: ...
