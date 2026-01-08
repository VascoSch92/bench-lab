from benchlab import Benchmark
from typing import final
from benchlab._benchmarks._jailbreak_llms._instances import JailbreakLLMsInstance
import csv
import urllib.request
import io

from benchlab._benchmarks._jailbreak_llms._metrics import JailbreakLLMsMetric


class JailbreakLLMsBenchmark(Benchmark[JailbreakLLMsInstance]):
    _METRICS = {"jailbreak_llms_checker": JailbreakLLMsMetric}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @final
    def _load_dataset(self) -> list[JailbreakLLMsInstance]:
        url = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/forbidden_question/forbidden_question_set.csv"

        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))

        return [
            JailbreakLLMsInstance(
                id=f"{row['content_policy_id']}__{row['q_id']}",
                content_policy_id=row["content_policy_id"],
                content_policy_name=row["content_policy_name"],
                question=row["question"],
            )
            for row in reader
        ]
