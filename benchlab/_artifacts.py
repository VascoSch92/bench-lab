import csv
import importlib
import json
from dataclasses import replace, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, TYPE_CHECKING, Union, Self

from benchlab._spec import Spec
from benchlab.aggregators._base import Aggregator
from benchlab._metrics.base import Metric
from benchlab._exceptions import ArtifactCorruptedError
from benchlab._types import InstanceType

if TYPE_CHECKING:
    from ._states import BenchmarkReport, BenchmarkEval, BenchmarkExec, Benchmark


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
    def from_string(cls, input_: str) -> Self:
        """
        Converts a string input into a valid ArtifactType member.

        Args:
            input_: The string value to convert.

        Returns:
            The corresponding ArtifactType instance.

        Raises:
            RuntimeError: If the input string does not match any known
                ArtifactType values.
        """
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
        """
        Provides a numeric mapping for the logical order of artifacts.

        This internal mapping ensures that artifacts follow the sequential
        pipeline: Benchmark -> Execution -> Evaluation -> Report.
        """
        return {
            ArtifactType.BENCHMARK: 1,
            ArtifactType.EXECUTION: 2,
            ArtifactType.EVALUATION: 3,
            ArtifactType.REPORT: 4,
        }

    def __lt__(self, other) -> bool:
        """
        Determines if this artifact type precedes another in the pipeline.

        Args:
            other: The other ArtifactType to compare against.

        Returns:
            True if this artifact occurs earlier in the lifecycle than the other.

        Raises:
            TypeError: If compared against a non-ArtifactType object.
        """
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
    """List of :class:`Metric` objects"""

    aggregators: list[Aggregator]
    """List of :class:`Aggregator` objects"""

    def __post_init__(self):
        if "class_name" not in self.metadata or "class_module" not in self.metadata:
            raise ArtifactCorruptedError(
                "Artifact metadata must contain `class_name` and `class_module`fields."
            )
        if self.instances:
            instance_type = type(self.instances[0])
            if not all(type(instance) is instance_type for instance in self.instances):
                raise ArtifactCorruptedError(
                    "Artifact instances must all be of the same type."
                )

    @property
    def type_(self) -> ArtifactType:
        """Returns the articat type of the artifact."""
        return ArtifactType.from_string(self.metadata["class_name"])

    def to_dict(self) -> dict[str, Any]:
        """Convert an Artifact to a dictionary."""
        return {
            "metadata": self.metadata,
            "spec": self.spec.to_dict(),
            "instances": [instance.to_dict() for instance in self.instances],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "aggregators": [agg.to_dict() for agg in self.aggregators],
        }

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """
        Initializes an instance by loading and parsing a JSON file.

        Args:
            path: The file system path to the JSON file, provided as a string
                or a Path object.

        Returns:
            An initialized instance of the class (Self) populated with the
            data from the JSON file.

        Raises:
            FileNotFoundError: If the specified path does not exist.
            json.JSONDecodeError: If the file is not a valid JSON document.
            KeyError: If the JSON structure is missing required top-level keys.
        """
        path = Path(path)
        with path.open("r") as f:
            json_artifact = json.load(f)

        spec = Spec(**json_artifact["spec"])
        instances = cls._load_objects_from_json(json_artifact["instances"], True)
        metrics = cls._load_objects_from_json(json_artifact["metrics"], False)
        aggregators = cls._load_objects_from_json(json_artifact["aggregators"], False)

        return cls(
            metadata=json_artifact["metadata"],
            spec=spec,
            instances=instances,
            metrics=metrics,
            aggregators=aggregators,
        )

    @staticmethod
    def _load_objects_from_json(
        data_list: list[dict[str, Any]],
        enforce_single_class: bool,
    ) -> list[Any]:
        """
        Generic helper to dynamically import and instantiate classes from JSON data.

        Args:
            data_list: List of dictionaries containing 'class_module' and 'class_name'.
            enforce_single_class: If True, raises ValueError if the list contains
                different class types.

        Returns:
            A list of the instantiate objects.

        Raises:
            ArtifactCorruptedError: if `enforce_single_class` is True, and input contains
                different classes types.
        """
        results: list[Any] = []
        first_cls_info: str | None = None
        cached_cls = None

        for item in data_list:
            # Extract class info
            module_name = item.pop("class_module")
            class_name = item.pop("class_name")

            if enforce_single_class:
                if first_cls_info and (module_name, class_name) != first_cls_info:
                    raise ArtifactCorruptedError(
                        "All items must be of the same class type.\n"
                        f"But got {first_cls_info} and {module_name}.{class_name}."
                    )
                first_cls_info = f"{module_name}.{class_name}"

            # Import the class. If `enforce_single_class` is True, then reuse the cached one.
            if cached_cls is None or not enforce_single_class:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name, None)
                if cls is None:
                    raise ArtifactCorruptedError(
                        f"Class {class_name} not found in {module_name}"
                    )
                cached_cls = cls

            results.append(cached_cls(**item))

        return results


class BenchmarkArtifact(Generic[InstanceType]):
    """
    A base mixin or component class that provides serialization and
    deserialization capabilities for benchmark lifecycle states.

    This class facilitates the conversion between live Python objects
    (Benchmark, Execution, Evaluation, Report) and persistent formats
    like JSON and CSV.
    """

    def _generate_artifact(self) -> Artifact[InstanceType]:
        """Creates an Artifact representation of the current instance."""
        return Artifact(
            metadata={
                "class_name": self.__class__.__name__,
                "class_module": self.__class__.__module__,
            },
            spec=getattr(self, "spec", Spec.new()),
            instances=getattr(self, "instances", []),
            metrics=getattr(self, "metrics", []),
            aggregators=getattr(self, "aggregators", []),
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
        """
        Factory method to reconstruct a benchmark state from a JSON file.

        The method also validates that the artifact type stored in the JSON is
        compatible with the class being called (e.g., you cannot load a raw
        Benchmark as a BenchmarkReport).

        Args:
            path: Path to the JSON artifact file.

        Returns:
            A concrete instance of a Benchmark state (`Benchmark`, `Exec`, `Eval`, or `Report`).

        Raises:
            ValueError: If the artifact type in the file is logically
                incompatible with the calling class.
            RuntimeError: If an unknown artifact type is encountered.
        """
        artifact: Artifact = Artifact.from_json(path)

        artifact_type = artifact.type_
        cls_type = ArtifactType.from_string(cls.__name__)
        if artifact_type < cls_type:
            raise ValueError(
                f"Incompatible Artifact Stage: Cannot initialize a '{cls.__name__}' "
                f"from a '{artifact_type}' file.\n"
                f"The requested class requires data from the '{cls_type}' stage or later, "
                f"but the provided file is only at the '{artifact_type}' stage.\n"
            )

        match cls_type:
            case ArtifactType.BENCHMARK:
                return cls._instantiate_benchmark(
                    instances=artifact.instances,
                    metrics=artifact.metrics,
                    aggregators=artifact.aggregators,
                    spec=artifact.spec,
                )
            case ArtifactType.EXECUTION:
                return cls._instantiate_benchmark_exec(
                    instances=artifact.instances,
                    metrics=artifact.metrics,
                    aggregators=artifact.aggregators,
                    spec=artifact.spec,
                )
            case ArtifactType.EVALUATION:
                return cls._instantiate_benchmark_eval(
                    instances=artifact.instances,
                    metrics=artifact.metrics,
                    aggregators=artifact.aggregators,
                    spec=artifact.spec,
                )
            case ArtifactType.REPORT:
                return cls._instantiate_benchmark_report(
                    instances=artifact.instances,
                    metrics=artifact.metrics,
                    aggregators=artifact.aggregators,
                    spec=artifact.spec,
                )
            case _:
                raise RuntimeError(f"Unexpected artifact type {cls_type}")

    @staticmethod
    def _instantiate_benchmark(
        instances: list[InstanceType],
        metrics: list[Metric],
        aggregators: list[Aggregator],
        spec: Spec,
    ) -> "Benchmark[InstanceType]":
        """
        Internal factory to create a Benchmark instance.

        Cleans instances of previous run data (attempts/evaluations) to ensure
        a fresh state and handles initialization via the library or direct creation.
        """
        from benchlab._states._benchmark import Benchmark

        for j in range(len(instances)):
            if instances[j].attempts or instances[j].evaluations:
                instances[j] = replace(
                    instances[j],
                    _attempts=[],
                    _evaluated_attempts={},
                )

        return Benchmark.new(
            source=instances,
            metrics=metrics,
            aggregators=aggregators,
            **spec.to_dict(),
        )

    @staticmethod
    def _instantiate_benchmark_exec(
        instances: list[InstanceType],
        metrics: list[Metric],
        aggregators: list[Aggregator],
        spec: Spec,
    ) -> "BenchmarkExec[InstanceType]":
        """
        Internal factory to create a BenchmarkExec instance.

        Ensures that evaluation data is cleared while preserving execution attempts.
        """
        from benchlab._states._execution import BenchmarkExec

        for j in range(len(instances)):
            if instances[j].evaluations:
                instances[j] = replace(
                    instances[j],
                    _evaluated_attempts={},
                )

        return BenchmarkExec.new(
            source=instances,
            metrics=metrics,
            aggregators=aggregators,
            **spec.to_dict(),
        )

    @staticmethod
    def _instantiate_benchmark_eval(
        instances: list[InstanceType],
        metrics: list[Metric],
        aggregators: list[Aggregator],
        spec: Spec,
    ) -> "BenchmarkEval[InstanceType]":
        """Internal factory to create a BenchmarkEval instance."""
        from benchlab._states._evaluation import BenchmarkEval

        return BenchmarkEval.new(
            source=instances,
            metrics=metrics,
            aggregators=aggregators,
            **spec.to_dict(),
        )

    @staticmethod
    def _instantiate_benchmark_report(
        instances: list[InstanceType],
        metrics: list[Metric],
        aggregators: list[Aggregator],
        spec: Spec,
    ) -> "BenchmarkReport[InstanceType]":
        """Internal factory to create a BenchmarkReport instance."""

        from benchlab._states._report import BenchmarkReport

        return BenchmarkReport.new(
            source=instances,
            metrics=metrics,
            aggregators=aggregators,
            **spec.to_dict(),
        )

    def to_json(self, output_path: Path | str | None = None) -> None:
        """
        Serializes the current benchmark state to a JSON file.

        Args:
            output_path: Destination path. If `None`, a default name is generated
                using the class name and a timestamp.
        """
        output_path = self._validate_path(output_path=output_path, extension=".json")

        artifact = self._generate_artifact()

        with output_path.open("w") as f:
            json.dump(artifact.to_dict(), f, indent=4)

    def to_csv(self, output_path: Path | str | None = None) -> None:
        """
        Exports the benchmark results to a flattened CSV format.

        This method flattens the nested structure of instances, attempts, and
        evaluations into a tabular format suitable for data analysis tools.

        Args:
            output_path: Destination path for the CSV file. If `None`, the current
                working directory is taken as `output_path`.
        """
        output_path = self._validate_path(output_path=output_path, extension=".csv")

        artifact = self._generate_artifact()
        instances: list[InstanceType] = artifact.instances

        # base header of the table
        rows: list[dict[str, Any]] = []
        for instance in instances:
            # base data about the instance
            row: dict[str, Any] = {
                "id": instance.id,
                "ground_truth": instance.ground_truth,
            }

            # adding data relative to the instance attempts
            for idx, attempt in enumerate(instance.attempts, 1):
                row[f"attempt_{idx}_response"] = attempt.response
                row[f"attempt_{idx}_status"] = attempt.status
                row[f"attempt_{idx}_runtime"] = (
                    round(attempt.runtime, 2) if attempt.runtime else None
                )
                if attempt.token_usage:
                    for k, v in attempt.token_usage.items():
                        row[f"attempt_{idx}_{k}"] = v

            # metrics data
            for metric_name, evals in instance.evaluations.items():
                for idx, eval_ in enumerate(evals, 1):
                    row[f"attempt_{idx}_{metric_name}"] = eval_

            rows.append(row)

        headers = list(rows[0].keys()) if len(rows) > 0 else []
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def _validate_path(self, output_path: Path | str | None, extension: str) -> Path:
        """
        Ensures the provided path is valid and has the correct file extension.
        If `None` is provided, the default output path is used, i.e., <CLASS_NAME>_uuid.

        Args:
            output_path: The user-provided path or None.
            extension: The required file extension (e.g., '.json' or '.csv').

        Returns:
            A validated Path object.

        Raises:
            ValueError: If the provided path has an incorrect extension.
        """
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
