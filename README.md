# Bench Lab

> ⚠️ **Early Development Notice**
>
> This repository is a work in progress. Things may be incomplete, unstable, or subject to change.


Bench Lab is a framework for evaluating large language models (LLMs), agents, and RAG systems across various benchmarks. The project provides a unified interface for benchmarking while offering statistical tools to analyze and improve system performance.


## Usage Example

Simple example of the API.

```python
from benchlab import Benchmark
from benchlab.aggregators import RuntimesAggregator, StatusAggregator
import random


def mock_model(instance) -> str:
    random_answer = random.randint(1, 10)
    return f"The answer for question {instance.id} is {random_answer}."

# a benchmark is composed by 4 states: 
# definition, execution, evaluation, aggregation

# definition of the benchmark
benchmark = Benchmark.from_library(
    name="MathQA",
    metric_names=["exact_match"],
    aggregators=[RuntimesAggregator(), StatusAggregator()],
    timeout=None,
    n_instance=5,
    n_attempts=1,
)

# execution of the benchmark
execution = benchmark.run(mock_model)

# evaluation of the execution
evaluation = execution.evaluate()

# aggregation of the results 
report = evaluation.report()
report.summary()
```