from datasets import load_dataset

from benchlab.dataset import Dataset
from benchlab.library.math_qa._instance import MathQAInstance


class MathQADataset(Dataset[MathQAInstance]):
    def __post_init__(self, split: str):
        super().__post_init__(split=split)
        self._instances: list[MathQAInstance] = self._load_dataset(split=split)
        self._map_idx: dict[str, int] = {
            instance.id: idx for idx, instance in enumerate(self._instances)
        }

    def get(self, idx: int | str) -> MathQAInstance:
        if isinstance(idx, str):
            if idx in self._map_idx:
                return self._instances[self._map_idx[idx]]
            raise ValueError(f"Unknown index {idx}")
        return self._instances[idx]

    def __len__(self) -> int:
        return len(self._instances)

    @staticmethod
    def _load_dataset(split: str) -> list[MathQAInstance]:
        dataset = load_dataset("regisss/math_qa")
        split_ds = dataset[split]

        return [
            MathQAInstance(
                id=f"{hash(row['Problem'])}",
                problem=row["Problem"],
                rationale=row["Rationale"],
                options=row["options"],
                correct=row.get("correct", None),
                annotated_formula=row["annotated_formula"],
                linear_formula=row["linear_formula"],
                category=row["category"],
            )
            for row in split_ds
        ]
