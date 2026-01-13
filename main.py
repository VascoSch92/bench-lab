from benchlab import Benchmark


def mock_model(instance, s: str) -> str:
    return " I don't know what to do!" + s


def main():
    benchmark = Benchmark.from_library(
        name="MathQA",
        metric_names=["exact_match"],
        timeout=None,
        n_instance=5,
        n_attempts=1,
    )
    benchmark.load_dataset()
    benchmark.to_json("my_json.json")
    bench = Benchmark.from_json("my_json.json")
    bench_exec = benchmark.run(mock_model, args={"s": "ciao"})
    bench_eval = bench_exec.evaluate()
    bench_eval.report()


if __name__ == "__main__":
    main()
