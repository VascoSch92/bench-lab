import csv
import importlib
import json
from abc import ABC
from dataclasses import replace, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, TYPE_CHECKING, Union, Self

from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._types import InstanceType
from benchlab._core._benchmark._spec import Spec

if TYPE_CHECKING:
    from benchlab._core._benchmark._benchmark import Benchmark
    from benchlab._core._benchmark._execution import BenchmarkExec
    from benchlab._core._benchmark._evaluation import BenchmarkEval
    from benchlab._core._benchmark._report import BenchmarkReport

__all__ = ["BenchmarkArtifact"]


class ArtifactType(StrEnum):
    """Type of the artifact."""

    BENCHMARK = "Benchmark"
    """Artifact for the class `Benchmark`."""

    EXECUTION = "BenchmarkExec"
    """Artifact for the class `BenchmarkExec`."""

    EVALUATION = "BenchmarkEval"
    """Artifact for the class `BenchmarkEval`."""

    REPORT = "BenchmarkReport"
    """Artifact for the class `BenchmarkReport`."""

    @classmethod
    def from_string(cls, input_: str):
        match input_:
            case (
                ArtifactType.BENCHMARK
                | ArtifactType.EXECUTION
                | ArtifactType.EVALUATION
                | ArtifactType.REPORT
            ):
                return ArtifactType(input_)
            case _:
                raise RuntimeError(f"Unexpected artifact type {input_}")

    @property
    def _rank(self) -> dict[str, int]:
        return {
            ArtifactType.BENCHMARK: 1,
            ArtifactType.EXECUTION: 2,
            ArtifactType.EVALUATION: 3,
            ArtifactType.REPORT: 4,
        }

    def __lt__(self, other) -> bool:
        if not isinstance(other, ArtifactType):
            return NotImplemented
        return self._rank[self] < other._rank[other]


@dataclass(frozen=True)
class Artifact(Generic[InstanceType]):
    """
    Represents a serialized snapshot of a benchmark, including its configuration and data.

    An Artifact serves as the "container" that bundles together the benchmark's
    operational specifications, the specific instances (tasks) it contains, the
    metrics used for evaluation, and the metadata required for dynamic reconstruction.
    """

    metadata: dict[str, str]
    """
    A dictionary containing essential class information. Must include 'class_name' 
    and 'class_module' to allow for dynamic re-instantiation of the benchmark.
    """

    spec: Spec
    """Instance of :class:`Spec` containing the execution parameters."""

    instances: list[InstanceType]
    """A list of instances of type `InstanceType`. All items must be of the exact same class."""

    metrics: list[Metric]
    """ list of :class:`Metric` objects"""

    def __post_init__(self):
        if "class_name" not in self.metadata or "class_module" not in self.metadata:
            # todo: implement our excpetion ArtifactCorrupted
            raise ValueError(
                "artifact metadata must contain 'class_name' and 'class_module'"
            )
        # todo: check also here that all the instances have the same class :-)

    @property
    def type_(self) -> ArtifactType:
        class_name, class_module = (
            self.metadata["class_name"],
            self.metadata["class_module"],
        )
        if class_name.endswith("Benchmark") and "._library" in class_module:
            return ArtifactType.BENCHMARK
        return ArtifactType.from_string(class_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "spec": self.spec.to_dict(),
            "instances": [instance.to_dict() for instance in self.instances],
            "metrics": [metric.to_dict() for metric in self.metrics],
        }

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        path = Path(path)
        with path.open("r") as f:
            json_artifact = json.load(f)

        # todo: check that we have good number of keys.
        # todo: they should be exactly 4
        spec = Spec(**json_artifact["spec"])
        instances = cls._get_instances_from_json(json_artifact["instances"])
        metrics = cls._get_metrics_from_json(json_artifact["metrics"])

        return cls(
            metadata=json_artifact["metadata"],
            spec=spec,
            instances=instances,
            metrics=metrics,
        )

    # todo: change exception in ArtifactCorruptedException
    @staticmethod
    def _get_instances_from_json(
        json_instances: list[dict[str, Any]],
    ) -> list[InstanceType]:
        instances: list[InstanceType] = []

        class_module: str | None = None
        class_name: str | None = None
        instance_cls: type[InstanceType] | None = None
        for instance in json_instances:
            instance_class_module = instance.pop("class_module")
            instance_class_name = instance.pop("class_name")

            if instance_cls is None:
                # we need to import the class cls just once as
                # we suppose all the classes have the same instance
                class_module = instance_class_module
                class_name = instance_class_name
                module = importlib.import_module(class_module)
                instance_cls = getattr(module, class_name, None)

                if instance_cls is None:
                    # todo: better error msg
                    raise ValueError

            if (
                instance_class_module != class_module
                or instance_class_name != class_name
            ):
                raise ValueError("All the instance must be of the same class.")

            assert instances is not None
            loaded_instance = instance_cls(**instance)
            instances.append(loaded_instance)
        return instances

    @staticmethod
    def _get_metrics_from_json(json_metrics: list[dict[str, Any]]) -> list[Metric]:
        metrics = []

        for metric in json_metrics:
            metric_class_module = metric.pop("class_module")
            metric_class_name = metric.pop("class_name")

            module = importlib.import_module(metric_class_module)
            metric_cls = getattr(module, metric_class_name, None)

            if metric_cls is None:
                # todo: better error message
                raise ValueError
            metrics.append(metric_cls())

        return metrics


class BenchmarkArtifact(ABC, Generic[InstanceType]):
    def _artifact(self) -> Artifact[InstanceType]:
        return Artifact(
            metadata={
                "class_name": self.__class__.__name__,
                "class_module": self.__class__.__module__,
            },
            spec=getattr(self, "spec", Spec.new()),
            instances=getattr(self, "instances", []),
            metrics=getattr(self, "metrics", []),
        )

    @classmethod
    def from_json(
        cls,
        path: Path | str,
    ) -> Union[
        "Benchmark[InstanceType]",
        "BenchmarkExec[InstanceType]",
        "BenchmarkEval[InstanceType]",
        "BenchmarkReport[InstanceType]",
    ]:
        artifact: Artifact = Artifact.from_json(path)

        artifact_type = artifact.type_
        cls_type = ArtifactType.from_string(cls.__name__)
        if artifact_type < cls_type:
            # todo: the error msg is not so clear :-)
            raise ValueError(
                f"It is not possible to instantiate an artifact type {cls_type} with \n"
                f"an artifact of type {artifact_type}."
            )

        match cls_type:
            case ArtifactType.BENCHMARK:
                from_library = "._library" in artifact.metadata["class_module"]
                return cls._instantiate_benchmark(
                    instances=artifact.instances,
                    metrics=artifact.metrics,
                    spec=artifact.spec,
                    from_library=from_library,
                )
            case ArtifactType.EXECUTION:
                return cls._instantiate_benchmark_exec(
                    instances=artifact.instances,
                    metrics=artifact.metrics,
                    spec=artifact.spec,
                )
            case ArtifactType.EVALUATION:
                return cls._instantiate_benchmark_eval(
                    instances=artifact.instances,
                    metrics=artifact.metrics,
                    spec=artifact.spec,
                )
            case ArtifactType.REPORT:
                return cls._instantiate_benchmark_report(
                    instances=artifact.instances,
                    metrics=artifact.metrics,
                    spec=artifact.spec,
                )
            case _:
                raise RuntimeError(f"Unexpected artifact type {cls_type}")

    @staticmethod
    def _instantiate_benchmark(
        instances: list[InstanceType],
        metrics: list[Metric],
        spec: Spec,
        from_library: bool,
    ) -> "Benchmark[InstanceType]":
        from benchlab._core._benchmark._benchmark import Benchmark

        for j in range(len(instances)):
            if instances[j].attempts or instances[j].evaluations:
                instances[j] = replace(
                    instances[j],
                    _attempts=[],
                    _evaluated_attempts={},
                )

        kwargs = spec.to_dict()
        kwargs.pop("name")
        if from_library:
            return Benchmark.from_library(
                name=spec.name,
                instances=instances,
                metric_names=[m.name for m in metrics],
                **kwargs,
            )
        return Benchmark.new(
            name=spec.name,
            instances=instances,
            metric_names=metrics,
            **kwargs,
        )

    @staticmethod
    def _instantiate_benchmark_exec(
        instances: list[InstanceType],
        metrics: list[Metric],
        spec: Spec,
    ) -> "BenchmarkExec[InstanceType]":
        from benchlab._core._benchmark._execution import BenchmarkExec

        for j in range(len(instances)):
            if instances[j].evaluations:
                instances[j] = replace(
                    instances[j],
                    _evaluated_attempts={},
                )

        return BenchmarkExec(
            spec=spec,
            instances=instances,
            metrics=metrics,
        )

    @staticmethod
    def _instantiate_benchmark_eval(
        instances: list[InstanceType],
        metrics: list[Metric],
        spec: Spec,
    ) -> "BenchmarkEval[InstanceType]":
        from benchlab._core._benchmark._evaluation import BenchmarkEval

        return BenchmarkEval(
            instances=instances,
            metrics=metrics,
            spec=spec,
        )

    @staticmethod
    def _instantiate_benchmark_report(
        instances: list[InstanceType],
        metrics: list[Metric],
        spec: Spec,
    ) -> "BenchmarkReport[InstanceType]":
        from benchlab._core._benchmark._report import BenchmarkReport

        return BenchmarkReport(
            instances=instances,
            metrics=metrics,
            spec=spec,
        )

    def to_json(self, output_path: Path | str | None = None) -> None:
        output_path = self._validate_path(output_path=output_path, extension=".json")

        artifact_dict = self._artifact().to_dict()

        with output_path.open("w") as f:
            json.dump(artifact_dict, f, indent=4)

    def to_csv(self, output_path: Path | str | None = None) -> None:
        output_path = self._validate_path(output_path=output_path, extension=".csv")

        artifact = self._artifact()
        instances: list[InstanceType] = artifact.instances

        # Base table
        headers: list[str] = ["id", "ground_truth"]
        for j in range(len(instances[0].attempts)):
            headers.extend([f"attempt_{j + 1}", f"runtime_{j + 1}", f"status_{j + 1}"])
            for metric_name in instances[0].evaluations:
                headers.append(f"{metric_name}_{j + 1}")

        rows: list[dict] = []
        for instance in instances:
            row: dict[str, Any] = {
                "id": instance.id,
                "ground_truth": instance.ground_truth,
            }
            for idx, attempt in enumerate(instance.attempts, 1):
                row[f"attempt_{idx}"] = attempt.response
                row[f"runtime_{idx}"] = round(attempt.runtime, 4)
                row[f"status_{idx}"] = attempt.status

                for metric, values in instance.evaluations.items():
                    row[f"{metric}_{idx}"] = values[idx - 1]

            assert len(row) == len(headers)
            rows.append(row)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def _validate_path(self, output_path: Path | str | None, extension: str) -> Path:
        if output_path is None:
            name = self.__class__.__name__.lower()
            uuid = int(datetime.now().strftime("%H%M%S%f")[:8])
            output_path = Path.cwd() / f"{name}_{uuid}"
        else:
            output_path = Path(output_path)

        if not output_path.suffix:
            output_path = output_path.with_suffix(extension)
        if output_path.suffix != extension:
            raise ValueError(
                f"Expected a {extension} extension, but got {output_path.suffix}"
            )

        return output_path
