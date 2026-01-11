from benchlab._core import BenchmarkEval, Benchmark


def mock_model(instance, s) -> str:
    return " I don't know what to do!" + s


def main():
    benchmark = Benchmark.new(
        name="JailbreakLLMs",
        metrics=["jailbreak_checker"],
        timeout=None,
        n_instance=10,
        n_attempts=3,
    )

    bench_exec = benchmark.run(mock_model, args={"s": "ciao"})
    bench_exec.to_json("bench_exec.json")
    bench_eval = bench_exec.evaluate()
    bench_eval.to_json("bench_eval.json")
    BenchmarkEval.from_json("bench_eval.json")


if __name__ == "__main__":
    main()
