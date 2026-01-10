import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Self

from benchlab._core._benchmark._evaluation import BenchmarkEval
from benchlab._core._benchmark._load_utils import get_instances_from_json
from benchlab._core._types import InstanceType
from benchlab._core._evaluation._metrics import Metric

__all__ = ["BenchmarkExec"]


@dataclass(frozen=True, slots=True)
class BenchmarkExec(Generic[InstanceType]):
    metadata: dict
    instances: list[InstanceType]
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("null"))
    metrics: list[Metric] = field(default_factory=list)

    def to_json(self, output_path: Path | str) -> None:
        output_path = Path(output_path)

        if not output_path.suffix:
            output_path = output_path.with_suffix(".json")

        file = {
            "metadata": self.metadata,
            "instances": [instance.to_dict() for instance in self.instances],
        }

        with output_path.open("w") as f:
            json.dump(file, f, indent=4)

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        instances: list[InstanceType] = get_instances_from_json(data["instances"])

        return cls(
            metadata=data["metadata"],
            instances=instances,
        )

    def evaluate(self) -> BenchmarkEval:
        instances = copy.deepcopy(self.instances)

        for metric in self.metrics:
            for instance in instances:
                evals = metric.evaluate(instance=instance, attempts=instance.attempts)
                instance.add_eval(metric_name=metric.name, value=evals)

        return BenchmarkEval(
            instances=instances,
            logger=self.logger,
            metrics=self.metrics,
        )

    def evaluate_async(self) -> BenchmarkEval:
        pass
