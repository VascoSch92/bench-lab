import os
from typing import Callable, final

from benchlab._core._benchmark import Benchmark
from benchlab._benchmarks._gpqa._instances import GPQAInstance


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

    @final
    def run_eval(self, fn: Callable[[GPQAInstance], GPQAInstance]) -> None: ...

    @final
    async def run_eval_async(self) -> None: ...
