from pathlib import Path
import json
from benchlab._core._types import InstanceType
from benchlab._core._evaluation._metrics import Metric
import importlib
from typing import Self, Any, Generic
from abc import ABC, abstractmethod
from enum import StrEnum
from dataclasses import replace

from datetime import datetime

__all__ = ["BenchmarkArtifact"]

# todo: json schema to validate the given json


class ArtifactType(StrEnum):
    BENCHMARK = "benchmark"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    REPORT = "report"


class BenchmarkArtifact(ABC, Generic[InstanceType]):
    @staticmethod
    @abstractmethod
    def _artifact_type() -> ArtifactType: ...

    @abstractmethod
    def _artifact(self) -> dict[str, str | int | float | None]: ...

    @classmethod
    def from_json(cls, path: Path | str) -> "Benchmark[Any]":
        path = Path(path)
        with path.open("r") as f:
            artifact = json.load(f)

        instances: list[InstanceType] = cls._get_instances_from_json(
            artifact["instances"]
        )
        metrics: list[Metric] = cls._get_metrics_from_json(artifact["metrics"])

        # todo: check that we are loading the class in the correct direction
        match cls._artifact_type():
            case ArtifactType.BENCHMARK:
                for j in range(len(instances)):
                    if (
                        instances[j]._attempts
                        or instances[j]._evaluations
                        or instances[j]._stats
                    ):
                        instances[j] = replace(
                            instances[j],
                            _attemps=[],
                            _evaluations=[],
                            _stats=[],
                        )

                return cls(**artifact, instances=instances)

            case ArtifactType.EXECUTION:
                for j in range(len(instances)):
                    if instances[j]._evaluations or instances[j]._stats:
                        instances[j] = replace(
                            instances[j],
                            _evaluations=[],
                            _stats=[],
                        )

                return cls(
                    spec=artifact["spec"],
                    instances=instances,
                    metrics=metrics,
                )

            case ArtifactType.EVALUATION:
                for j in range(len(instances)):
                    if instances[j]._stats:
                        instances[j] = replace(
                            instances[j],
                            _stats=[],
                        )

                return cls(
                    spec=artifact["spec"],
                    instances=instances,
                    metrics=metrics,
                )

            case ArtifactType.REPORT:
                return cls(spec=artifact["spec"], instances=instances, metrics=metrics)
            case _:
                raise RuntimeError

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
        instances = []

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

            loaded_instance = instance_cls(**instance)
            instances.append(loaded_instance)
        return instances

    def to_json(self, output_path: Path | str | None) -> None:
        output_path = self._validate_path(output_path=output_path)

        artifact = self._artifact()

        with output_path.open("w") as f:
            json.dump(artifact, f, indent=4)

    def _validate_path(self, output_path: Path | str | None) -> Path:
        if output_path is None:
            output_path = Path.cwd() / f"{self._artifact_type}_{datetime.now()}"
        else:
            output_path = Path(output_path)

        if not output_path.suffix:
            output_path = output_path.with_suffix(".json")
        if output_path.suffix != ".json":
            raise ValueError(f"Expected a json format, but got {output_path.suffix}")

        return output_path
