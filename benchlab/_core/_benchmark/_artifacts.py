import csv
import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, TYPE_CHECKING, Union

from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._types import InstanceType

if TYPE_CHECKING:
    from benchlab._core._benchmark._benchmark import Benchmark
    from benchlab._core._benchmark._execution import BenchmarkExec
    from benchlab._core._benchmark._evaluation import BenchmarkEval
    from benchlab._core._benchmark._report import BenchmarkReport

__all__ = ["BenchmarkArtifact", "ArtifactType"]

# todo: json schema to validate the given json and or spec?


class ArtifactType(StrEnum):
    """Type of the artifact."""

    BENCHMARK = "benchmark"
    """Artifact for the class `Benchmark`."""

    EXECUTION = "execution"
    """Artifact for the class `BenchmarkExec`."""

    EVALUATION = "evaluation"
    """Artifact for the class `BenchmarkEval`."""

    REPORT = "report"
    """Artifact for the class `BenchmarkReport`."""

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


class BenchmarkArtifact(ABC, Generic[InstanceType]):
    @staticmethod
    @abstractmethod
    def _artifact_type() -> ArtifactType: ...

    @abstractmethod
    def _artifact(self) -> dict[str, Any]: ...

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
        path = Path(path)
        with path.open("r") as f:
            artifact = json.load(f)

        artifact_type = cls._get_artifact_type_from_json(artifact.pop("type", None))
        if artifact_type < cls._artifact_type():
            # todo: the error msg is not so clear :-)
            raise ValueError(
                f"It is not possible to instantiate an artifact type {cls._artifact_type()} with \n"
                "an artifact of type {artifact_type}."
            )

        spec_artifact = artifact.pop("spec", {})

        instance_artifact = artifact.pop("instances", [])
        instances: list[InstanceType] = cls._get_instances_from_json(instance_artifact)

        metric_artifact = artifact.pop("metrics", [])
        metrics: list[Metric] = cls._get_metrics_from_json(metric_artifact)

        match cls._artifact_type():
            case ArtifactType.BENCHMARK:
                return cls._instantiate_benchmark(
                    instances=instances, metrics=metrics, spec=spec_artifact
                )
            case ArtifactType.EXECUTION:
                return cls._instantiate_benchmark_exec(
                    instances=instances, metrics=metrics, spec=spec_artifact
                )
            case ArtifactType.EVALUATION:
                return cls._instantiate_benchmark_eval(
                    instances=instances, metrics=metrics, spec=spec_artifact
                )
            case ArtifactType.REPORT:
                return cls._instantiate_benchmark_report(
                    instances=instances, metrics=metrics, spec=spec_artifact
                )
            case _:
                raise RuntimeError(f"Unexpected artifact type {cls._artifact_type()}")

    @staticmethod
    def _get_artifact_type_from_json(artifact_type: str | None) -> ArtifactType:
        if artifact_type is None:
            raise ValueError("Artifact corrupted. Indeed, artifact type is `None`.")
        return ArtifactType(artifact_type)

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

    @staticmethod
    def _get_instances_from_json(
        json_instances: list[dict[str, Any]],
    ) -> list[InstanceType]:
        instances: list[InstanceType] = []

        class_module: str | None = None
        class_name: str | None = None
        instance_cls: InstanceType | None = None
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
    def _instantiate_benchmark(
        instances: list[InstanceType],
        metrics: list[Metric],
        spec: dict[str, Any],
    ) -> "Benchmark[InstanceType]":
        from benchlab._core._benchmark._benchmark import Benchmark

        for j in range(len(instances)):
            if instances[j].attempts or instances[j].evaluations:
                instances[j] = replace(
                    instances[j],
                    _attempts=[],
                    _evaluated_attempts={},
                )

        return Benchmark.new(
            name=spec["name"],
            metrics=[m.name for m in metrics],
            instance_ids=spec.get("instance_ids", None),
            n_instance=spec.get("n_instance", None),
            n_attempts=spec.get("n_attempts", 1),
            timeout=spec.get("timeout", None),
            logs_filepath=spec.get("logs_filepath", None),
        )

    @staticmethod
    def _instantiate_benchmark_exec(
        instances: list[InstanceType],
        metrics: list[Metric],
        spec: dict[str, Any],
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
        spec: dict[str, Any],
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
        spec: dict[str, Any],
    ) -> "BenchmarkReport[InstanceType]":
        from benchlab._core._benchmark._report import BenchmarkReport

        return BenchmarkReport(
            instances=instances,
            metrics=metrics,
            spec=spec,
        )

    def to_json(self, output_path: Path | str | None = None) -> None:
        output_path = self._validate_path(output_path=output_path, extension=".json")

        artifact = self._artifact()

        json_artifact = {
            "spec": artifact["spec"],
            "instances": [instance.to_dict() for instance in artifact["instances"]],
            "metrics": [metric.to_dict() for metric in artifact["metrics"]],
        }

        with output_path.open("w") as f:
            json.dump(json_artifact, f, indent=4)

    def to_csv(self, output_path: Path | str | None = None) -> None:
        output_path = self._validate_path(output_path=output_path, extension=".csv")

        artifact = self._artifact()
        instances: list[InstanceType] = artifact["instances"]

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
            name = self._artifact_type().name.lower()
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
