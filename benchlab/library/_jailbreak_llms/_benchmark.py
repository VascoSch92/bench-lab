import csv
import io
import urllib.request
from typing import final

from benchlab import Benchmark
from benchlab.library._jailbreak_llms._instance import JailbreakLLMsInstance
from benchlab.library._jailbreak_llms._metrics import (
    JailbreakCheckerMetric,
    JailbreakCheckerUnsureMetric,
)


class JailbreakLLMsBenchmark(Benchmark[JailbreakLLMsInstance]):
    _METRICS = {
        "jailbreak_checker": JailbreakCheckerMetric,
        "jailbreak_checker_unsure": JailbreakCheckerUnsureMetric,
    }

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
                id=f"{row['content_policy_id']}_{row['q_id']}",
                content_policy_id=row["content_policy_id"],
                content_policy_name=row["content_policy_name"],
                question=row["question"],
            )
            for row in reader
        ]
