from benchlab import Benchmark
from benchlab.aggregators import RuntimesAggregator, StatusAggregator
import random


def mock_model(instance) -> str:
    random_answer = random.randint(1, 10)
    return f"The answer for question {instance.id} is {random_answer}."


def main():
    benchmark = Benchmark.from_library(
        name="MathQA",
        metric_names=["exact_match"],
        aggregators=[RuntimesAggregator(), StatusAggregator()],
        timeout=None,
        n_instance=5,
        n_attempts=1,
    )
    execution = benchmark.run(mock_model, args={"s": "ciao"})
    evaluation = execution.evaluate()
    report = evaluation.report()
    report.summary()

    benchmark.summary()
    benchmark.to_json("my_json.json")
    bench = Benchmark.from_json("my_json.json")
    bench_exec = benchmark.run(mock_model, args={"s": "ciao"})
    bench_exec.summary()
    bench_eval = bench_exec.evaluate()
    bench_eval.summary()
    report = bench_eval.report()
    report.summary()


if __name__ == "__main__":
    main()
