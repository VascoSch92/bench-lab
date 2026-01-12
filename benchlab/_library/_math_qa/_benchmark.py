from typing import final

from datasets import load_dataset

from benchlab import Benchmark
from benchlab._core._evaluation._metrics._exact_match import ExactMatchMetric
from benchlab._library._math_qa._instance import MathQAInstance


class MathQABenchmark(Benchmark[MathQAInstance]):
    _METRICS = {
        "exact_match": ExactMatchMetric,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @final
    def _load_dataset(self) -> list[MathQAInstance]:
        dataset = load_dataset("regisss/math_qa")
        split_ds = dataset["train"]

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
