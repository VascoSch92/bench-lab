from benchlab._core import Benchmark


def mock_model(instance, s: str) -> str:
    return " I don't know what to do!" + s


def main():
    benchmark = Benchmark.new(
        name="MathQA",
        metrics=["exact_match"],
        timeout=None,
        n_instance=5,
        n_attempts=1,
    )
    benchmark.load_dataset()
    bench_exec = benchmark.run(mock_model, args={"s": "ciao"})
    bench_eval = bench_exec.evaluate()
    bench_eval.report()


if __name__ == "__main__":
    main()
