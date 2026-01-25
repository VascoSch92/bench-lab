from functools import partial

from benchlab import Benchmark
from benchlab._metrics import ExactMatchMetric
from benchlab.aggregators._aggregators import (
    ConsensusAggregator,
    StatusAggregator,
    RuntimesAggregator,
)
from benchlab.library.math_qa._dataset import MathQADataset
from benchlab.library.math_qa._instance import MathQAInstance

__all__ = ["MathQABench"]


MathQABench = partial(
    Benchmark[MathQAInstance].new,
    name="MathQABench",
    source=MathQADataset(),
    metrics=[ExactMatchMetric()],
    aggregators=[
        ConsensusAggregator(target=ExactMatchMetric.name),
        StatusAggregator(),
        RuntimesAggregator(),
    ],
)
