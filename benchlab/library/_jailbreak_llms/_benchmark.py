from functools import partial

from benchlab import Benchmark
from benchlab.aggregators._aggregators import (
    StatusAggregator,
    RuntimesAggregator,
)
from benchlab.library._jailbreak_llms._dataset import JailbreakLLMsDataset
from benchlab.library._jailbreak_llms._instance import JailbreakLLMsInstance
from benchlab.library._jailbreak_llms._metrics import (
    JailbreakCheckerMetric,
    JailbreakCheckerUnsureMetric,
)

__all__ = ["JailbreakLLMsBench"]


JailbreakLLMsBench = partial(
    Benchmark[JailbreakLLMsInstance].new,
    name="JailbreakLLMsBench",
    source=JailbreakLLMsDataset(),
    metrics=[
        JailbreakCheckerMetric(),
        JailbreakCheckerUnsureMetric(),
    ],
    aggregators=[
        StatusAggregator(),
        RuntimesAggregator(),
    ],
)
