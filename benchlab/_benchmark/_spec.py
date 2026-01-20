from dataclasses import dataclass, field, fields, replace
from typing import Self
from uuid import uuid4

__all_ = ["Spec"]


@dataclass(frozen=True, slots=True)
class Spec:
    """Class containing specification about a benchmark artifact."""

    name: str
    instance_ids: list[str] = field(default_factory=list)
    n_attempts: int = 1
    n_instance: int | None = None
    timeout: float | None = None
    logs_filepath: str | None = None
    execution_time: float | None = None
    evaluation_time: float | None = None

    @classmethod
    def new(cls) -> Self:
        return cls(name=str(uuid4()))

    def __post_init__(self) -> None:
        if self.n_instance is not None and self.n_instance <= 0:
            raise ValueError(
                "Argument `n_instance` must be a strictly positive integer, or `None` to select all the instances."
            )
        if self.n_attempts <= 0:
            raise ValueError("Argument `n_attempts` must be strictly positive integer.")
        if self.timeout is not None and self.timeout <= 0.0:
            raise ValueError(
                f"Argument `timeout` must be strictly positive. Got {self.timeout}"
            )

    def to_dict(self) -> dict:
        # note that this is faster for flat dataclasses
        return {field_.name: getattr(self, field_.name) for field_ in fields(self)}

    def set_execution_time(self, time: float) -> "Spec":
        """Returns a new instance of Spec with the updated execution_time."""
        return replace(self, execution_time=time)

    def set_evaluation_time(self, time: float) -> "Spec":
        """Returns a new instance of Spec with the updated evaluation_time."""
        return replace(self, evaluation_time=time)
