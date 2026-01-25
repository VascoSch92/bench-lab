import random

from benchlab.library.math_qa._benchmark import MathQABench


def mock_model(instance, s: str) -> str:
    random_answer = random.randint(1, 10)
    return f"The answer for question {instance.id} is {random_answer}. Or {s}"


def main():
    benchmark = MathQABench(n_instance=5)
    execution = benchmark.run(mock_model, kwargs={"s": "I don't know"})
    evaluation = execution.evaluate()
    report = evaluation.report()
    report.summary()


if __name__ == "__main__":
    main()
