import os
from typing import final

from benchlab._core._benchmark._benchmark import Benchmark
from benchlab._library._gpqa._instances import GPQAInstance


class GPQABenchmark(Benchmark[GPQAInstance]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @final
    def _load_dataset(self) -> list[GPQAInstance]:
        from datasets import load_dataset

        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset(
            "Idavidrein/gpqa",
            "gpqa_diamond",
            token=os.getenv("HF_TOKEN"),
        )

        print(ds)
        return []
