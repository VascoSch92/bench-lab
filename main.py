import random

from benchlab.library.math_qa._benchmark import MathQABench


def mock_model(instance) -> str:
    random_answer = random.randint(1, 10)
    return f"The answer for question {instance.id} is {random_answer}."


def main():
    benchmark = MathQABench(n_instance=5)
    execution = benchmark.run(mock_model, args={"s": "ciao"})
    evaluation = execution.evaluate()
    report = evaluation.report()
    report.summary()


if __name__ == "__main__":
    main()
