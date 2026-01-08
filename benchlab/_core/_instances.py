from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias
from types import MappingProxyType
from benchlab._core._stats import MetricStats

AnswerType: TypeAlias = str | None


class AttemptStatus(StrEnum):
    """Enumeration representing the status of an attempt."""

    SUCCESS = "success"
    """The attempt completed successfully."""

    FAILURE = "failure"
    """The attempt failed due to an error."""

    TIMEOUT = "timeout"
    """The attempt exceeded the allowed time limit."""


@dataclass(frozen=True, slots=True)
class Attempt:
    _response: AnswerType | None
    _runtime: float
    _status: AttemptStatus

    @classmethod
    def new(
        cls, response: AnswerType | None, runtime: float, status: AttemptStatus
    ) -> "Attempt":
        return cls(response, runtime, status)

    @property
    def response(self) -> AnswerType | None:
        return self._response

    @property
    def runtime(self) -> float | None:
        return self._runtime

    @property
    def status(self) -> AttemptStatus:
        return self._status




@dataclass(slots=True, kw_only=True)
class Instance(ABC):
    """Instance of a benchmark."""

    id: str
    """Unique id of the instance."""

    _attempts: list[Attempt] = field(default_factory=list)
    """Answers produced for this instance, one per attempt."""

    _metrics: dict[str, MetricStats] = field(default_factory=dict)
    """Metrics stats produced for this instance."""

    @property
    def attempts(self) -> list[Attempt]:
        return self._attempts

    @property
    def responses(self) -> list[AnswerType | None]:
        return [attempt.response for attempt in self._attempts]

    @property
    def runtimes(self) -> list[float | None]:
        return [attempt.runtime for attempt in self._attempts]

    @property
    def statuses(self) -> list[AttemptStatus]:
        return [attempt.status for attempt in self._attempts]

    @property
    def metrics(self) -> MappingProxyType[str, MetricStats]:
        return MappingProxyType(self._metrics)

    def add_attempt(
        self, response: AnswerType | None, runtime: float, status: str
    ) -> None:
        if runtime < 0.0:
            raise ValueError(f"Runtime must be greater than zero. Got {runtime}")
        if status not in AttemptStatus:
            raise ValueError(f"Status must be one of {AttemptStatus.__members__}")
        self._attempts.append(Attempt.new(response, runtime, AttemptStatus(status)))

    def update_metric_stats(self, name: str, stats: MetricStats) -> None:
        if name in self._metrics:
            # TODO: should we raise an error?
            raise ValueError(f"Metric {name} already set")
        self._metrics[name] = stats

    @property
    @abstractmethod
    def ground_truth(self) -> AnswerType: ...
