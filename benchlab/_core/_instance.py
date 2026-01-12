from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from benchlab._core._types import AnswerType


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
    _response: AnswerType
    _runtime: float
    _status: AttemptStatus

    @classmethod
    def new(
        cls, response: AnswerType, runtime: float, status: AttemptStatus
    ) -> "Attempt":
        return cls(response, runtime, status)

    @property
    def response(self) -> AnswerType | None:
        return self._response

    @property
    def runtime(self) -> float:
        return self._runtime

    @property
    def status(self) -> AttemptStatus:
        return self._status

    def to_dict(self) -> dict[str, Any]:
        return {
            "_response": self.response,
            "_runtime": self.runtime,
            "_status": self.status,
        }


@dataclass(slots=True, frozen=True, kw_only=True)
class Instance(ABC):
    """Instance of a benchmark."""

    id: str
    """Unique id of the instance."""

    _attempts: list[Attempt] = field(default_factory=list)
    """Attempts produced by the benchmark."""

    _evaluated_attempts: dict[str, list[Any]] = field(default_factory=dict)
    """Map metric name to evaluated attempts."""

    @property
    def attempts(self) -> list[Attempt]:
        return self._attempts

    @property
    def responses(self) -> list[AnswerType | None]:
        return [attempt.response for attempt in self.attempts]

    @property
    def runtimes(self) -> list[float]:
        return [attempt.runtime for attempt in self.attempts]

    @property
    def statuses(self) -> list[AttemptStatus]:
        return [attempt.status for attempt in self.attempts]

    @property
    def evaluations(self) -> MappingProxyType[str, list[Any]]:
        return MappingProxyType(self._evaluated_attempts)

    def add_attempt(self, response: AnswerType, runtime: float, status: str) -> None:
        if runtime < 0.0:
            raise ValueError(f"Runtime must be greater than zero. Got {runtime}")
        if status not in AttemptStatus:
            raise ValueError(f"Status must be one of {AttemptStatus.__members__}")

        attempt = Attempt.new(response, runtime, AttemptStatus(status))
        self._attempts.append(attempt)

    def add_eval(self, metric_name: str, evals: list[Any]) -> None:
        self._evaluated_attempts[metric_name] = evals

    @property
    @abstractmethod
    def ground_truth(self) -> AnswerType: ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_module": self.__class__.__module__,
            "class_name": self.__class__.__name__,
            "id": self.id,
            "_attempts": [attempt.to_dict() for attempt in self._attempts],
            "_evaluated_attempts": self._evaluated_attempts,
            **self._to_dict(),
        }

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]: ...
