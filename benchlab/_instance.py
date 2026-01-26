from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from benchlab._types import AnswerType


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
    """Represents a single attempt to solve a benchmark instance."""

    _response: AnswerType
    """Response produced during the attempt."""

    _runtime: float | None
    """Wall-clock runtime in seconds. `None` if attempt was not successfully."""

    _status: AttemptStatus
    """Terminal status of the attempt."""

    _token_usage: dict[str, int] = field(default_factory=dict)
    """Token usage relative to the attempt."""

    @classmethod
    def new(
        cls,
        response: AnswerType,
        runtime: float | None,
        status: AttemptStatus,
        token_usage: dict[str, int],
    ) -> "Attempt":
        """Create a new Attempt instance."""
        return cls(response, runtime, status, token_usage)

    @property
    def response(self) -> AnswerType | None:
        """Return the response produced by the attempt."""
        return self._response

    @property
    def runtime(self) -> float | None:
        """Return the runtime of the attempt in seconds, if available."""
        return self._runtime

    @property
    def status(self) -> AttemptStatus:
        """Return the final status of the attempt."""
        return self._status

    @property
    def token_usage(self) -> MappingProxyType[str, int]:
        return MappingProxyType(self._token_usage)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the attempt to a dictionary."""
        return {field_.name: getattr(self, field_.name) for field_ in fields(self)}


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
    def runtimes(self) -> list[float | None]:
        return [attempt.runtime for attempt in self.attempts]

    @property
    def statuses(self) -> list[AttemptStatus]:
        return [attempt.status for attempt in self.attempts]

    @property
    def evaluations(self) -> MappingProxyType[str, list[Any]]:
        return MappingProxyType(self._evaluated_attempts)

    def token_usage(self) -> MappingProxyType[str, int]:
        token_usage: dict[str, int] = {}
        for attempt in self.attempts:
            for key, value in attempt.token_usage.items():
                token_usage[key] = token_usage.get(key, 0) + value
        return MappingProxyType(token_usage)

    def add_attempt(
        self,
        response: AnswerType,
        runtime: float | None,
        status: str,
        token_usage: dict[str, int],
    ) -> None:
        if runtime is not None and runtime < 0.0:
            raise ValueError(f"Runtime must be greater than zero. Got {runtime}")
        if status not in AttemptStatus:
            raise ValueError(f"Status must be one of {AttemptStatus.__members__}")

        attempt = Attempt.new(response, runtime, AttemptStatus(status), token_usage)
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
