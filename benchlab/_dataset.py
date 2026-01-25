from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from enum import StrEnum
from typing import Generic
from typing import Self

from benchlab._types import InstanceType


class SplitType(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

    @classmethod
    def from_string(cls, input_: str) -> Self:
        """
        Converts a string input into a valid SplitType member.

        Args:
            input_: The string value to convert.

        Returns:
            The corresponding SplitType instance.

        Raises:
            RuntimeError: If the input string does not match any known
                ArtifactType values.
        """
        match input_:
            case SplitType.TRAIN | SplitType.VALIDATION | SplitType.TEST:
                return SplitType(input_)
            case _:
                raise ValueError(f"Unexpected split type {input_}")


@dataclass(slots=True)
class Dataset(ABC, Generic[InstanceType]):
    split: InitVar[str] = "train"
    _split: SplitType = field(init=False)

    def __post_init__(self, split: str) -> None:
        self._split = SplitType.from_string(split)

    @abstractmethod
    def get(self, idx: int | str) -> InstanceType: ...

    @abstractmethod
    def __len__(self) -> int: ...


class ListDataset(Dataset[InstanceType]):
    def __init__(self, instances: list[InstanceType]) -> None:
        self._instances = instances
        self._map_idx: dict[str, int] = {
            instance.id: idx for idx, instance in enumerate(self._instances)
        }

    def get(self, idx: int | str) -> InstanceType:
        if isinstance(idx, str):
            if idx in self._map_idx:
                return self._instances[self._map_idx[idx]]
            raise ValueError(f"Unknown index {idx}")
        return self._instances[idx]

    def __len__(self) -> int:
        return len(self._instances)
