import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Self, Any, Sequence

from rich import console, table

from benchlab._benchmark._artifacts import BenchmarkArtifact
from benchlab._benchmark._spec import Spec
from benchlab.aggregators._base import Aggregator
from benchlab.metrics._base import Metric
from benchlab._types import InstanceType
from benchlab.dataset import Dataset, ListDataset
from benchlab.utils import get_logger


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

    _dataset: Dataset[InstanceType] | None = None
    """Dataset of the instances."""

    _instances: list[InstanceType] = field(default_factory=list, init=False)
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

    @classmethod
    def new(
        cls,
        name: str,
        source: list[InstanceType] | Dataset,
        metrics: list["Metric"] | None = None,
        aggregators: list["Aggregator"] | None = None,
        instance_ids: list[str] | None = None,
        n_instance: int | None = None,
        n_attempts: int = 1,
        timeout: float | None = None,
        logs_filepath: str | None = None,
        logger: logging.Logger = logging.getLogger("null"),
        **kwargs: Any,
    ) -> Self:
        if logger is None:
            logger = get_logger(name=__name__, path=logs_filepath, console=True)

        spec = Spec(
            name=name,
            instance_ids=instance_ids or [],
            n_instance=n_instance,
            n_attempts=n_attempts,
            timeout=timeout,
            logs_filepath=logs_filepath,
            execution_time=kwargs.get("execution_time", None),
            evaluation_time=kwargs.get("evaluation_time", None),
        )

        dataset: Dataset[InstanceType] = (
            ListDataset(source) if isinstance(source, list) else source
        )

        if n_instance is not None and instance_ids is not None:
            logger.warning(
                "Arguments `n_instance` and `instance_ids` are specified at the same time.\n"
                "`n_instance` instances will be selected from `instance_ids`."
            )

        return cls(
            _spec=spec,
            _dataset=dataset,
            _metrics=metrics or [],
            _aggregators=aggregators or [],
            logger=logger,
        )

    @abstractmethod
    def _task_specific_checks(self) -> None: ...

    @property
    def spec(self) -> Spec:
        return self._spec

    @cached_property
    def instances(self) -> tuple[InstanceType, ...]:
        if self._instances:
            return tuple(self._instances)
        if self._dataset is None:
            return tuple()

        idxs = self._select_instance_ids()
        return tuple(self._dataset.get(idx) for idx in idxs)

    def _select_instance_ids(self) -> Sequence[int | str]:
        """Select instance ids according to the benchmark spec."""
        if not self._spec.n_instance and not self._spec.instance_ids:
            if self._dataset is not None:
                return range(len(self._dataset))
            return range(len(self._instances))
        elif self._spec.n_instance and not self._spec.instance_ids:
            return range(self._spec.n_instance)
        elif not self._spec.n_instance and self._spec.instance_ids:
            return self._spec.instance_ids
        elif self._spec.n_instance and self._spec.instance_ids:
            size = min(
                self._spec.n_instance,
                len(self._spec.instance_ids),
            )
            return self._spec.instance_ids[:size]
        # todo: better error message
        raise RuntimeError

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
