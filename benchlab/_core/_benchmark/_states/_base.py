import logging
from abc import abstractmethod
from dataclasses import dataclass, field

from rich import console, table

from benchlab._core._benchmark._artifacts import BenchmarkArtifact
from benchlab._core._benchmark._spec import Spec
from benchlab._core._evaluation._aggregators._aggregator import Aggregator
from benchlab._core._evaluation._metrics._metric import Metric
from benchlab._core._types import InstanceType


@dataclass(frozen=True, slots=True)
class BaseBenchmark(BenchmarkArtifact[InstanceType]):
    """
    Base class to enforce structure across benchmark states.

    This class serves as a blueprint for implementing specific benchmarking logic,
    ensuring consistency in how instances, metrics, and aggregators are stored
    and accessed.
    """

    _spec: Spec = field(default_factory=Spec.new)
    """Configuration specifications for the benchmark."""

    _instances: list[InstanceType] = field(default_factory=list)
    """Collection of instances to be processed during the benchmark."""

    _metrics: list[Metric] = field(default_factory=list)
    """List of metrics used to evaluate individual instance performance."""

    _aggregators: list[Aggregator] = field(default_factory=list)
    """A list of aggregators used to summarize results across all instances."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))

    def __post_init__(self) -> None:
        self._check_consistency_instances()
        self._task_specific_checks()

    def _check_consistency_instances(self) -> None:
        if not self._instances:
            return None

        first_type = type(self._instances[0])
        if not all(type(i) is first_type for i in self._instances):
            raise ValueError("All instances must have the same type.")

    @abstractmethod
    def _task_specific_checks(self) -> None: ...

    @property
    def spec(self) -> Spec:
        return self._spec

    @property
    def instances(self) -> list[InstanceType]:
        return self._instances

    @property
    def metrics(self) -> list[Metric]:
        return self._metrics

    @property
    def aggregators(self) -> list[Aggregator]:
        return self._aggregators

    def summary(self) -> None:
        summary_table = self._generate_summary_table()
        console_ = console.Console()
        console_.print(summary_table)

    @abstractmethod
    def _generate_summary_table(self) -> table.Table: ...
