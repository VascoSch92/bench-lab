from benchlab import Benchmark


def mock_model(instance, s) -> str:
    return " I don't know what to do!" + s


def main():
    benchmark = Benchmark.new(
        name="JailbreakLLMs",
        metrics=["jailbreak_llms_checker"],
        timeout=None,
        n_instance=10,
    )
    benchmark.run(mock_model, args={"s": "ciao"})
    benchmark.display_results()


if __name__ == "__main__":
    main()
