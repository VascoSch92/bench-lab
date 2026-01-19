from benchlab import Benchmark
from benchlab._core._evaluation._aggregators._aggregator import (
    RuntimesAggregator,
    StatusAggregator,
)


def mock_model(instance, s: str) -> str:
    return " I don't know what to do!" + s


def main():
    benchmark = Benchmark.from_library(
        name="MathQA",
        metric_names=["exact_match"],
        aggregators=[RuntimesAggregator(), StatusAggregator()],
        timeout=None,
        n_instance=5,
        n_attempts=1,
    )
    _ = benchmark.instances
    benchmark.to_json("my_json.json")
    bench = Benchmark.from_json("my_json.json")
    bench_exec = benchmark.run(mock_model, args={"s": "ciao"})
    bench_eval = bench_exec.evaluate()
    report = bench_eval.report()
    print(report)


if __name__ == "__main__":
    main()
