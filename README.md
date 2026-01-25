# Bench Lab

> ⚠️ **Early Development Notice**
>
> This repository is a work in progress. Things may be incomplete, unstable, or subject to change.


Bench Lab is a framework for evaluating large language models (LLMs), agents, and RAG systems across various benchmarks. The project provides a unified interface for benchmarking while offering statistical tools to analyze and improve system performance.


## Usage Example

Simple example of the API.

```python
import random

# import your favorite benchmark from the library
from benchlab.library.math_qa._benchmark import MathQABench


def mock_model(instance) -> str:
    random_answer = random.randint(1, 10)
    return f"The answer for question {instance.id} is {random_answer}."


def main():
    # init the benchmark
    benchmark = MathQABench(n_instance=5)
    
    # run your implementation on the benchmark
    execution = benchmark.run(mock_model, args={"s": "ciao"})
    
    # evaluate them 
    evaluation = execution.evaluate()
    
    # finally, aggregate the results and print the benchmark report
    report = evaluation.report()
    report.summary()


if __name__ == "__main__":
    main()
```