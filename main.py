import logging
import random

from benchlab.library.math_qa import MathQABench
from benchlab._types import CallableOutput


def mock_model(instance, s: str) -> CallableOutput:
    random_answer = random.randint(1, 10)
    return {
        "answer": f"The answer for question {instance.id} is {random_answer}. Or {s}",
        "tokens_usage": {},
    }


def main():
    benchmark = MathQABench(n_instance=5, logging_level=logging.DEBUG)
    execution = benchmark.run(mock_model, kwargs={"s": "I don't know"})
    execution.summary()
    evaluation = execution.evaluate()
    report = evaluation.report()
    report.to_csv("report.csv")
    report.summary()


if __name__ == "__main__":
    main()
