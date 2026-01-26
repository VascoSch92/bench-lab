import time
from dataclasses import dataclass
from typing import Any

from rich import table

from benchlab._states._base import BaseBenchmark
from benchlab._states._execution import BenchmarkExec
from benchlab._types import BenchmarkCallable, InstanceType
from benchlab.utils import timed_exec


# todo: check how logger works if we have a logger in our main program
# todo: update logging to use rich
# todo: update logging for warning
# todo: add callback method to stop retrying

__all__ = ["Benchmark"]


@dataclass(frozen=True, slots=True)
class Benchmark(BaseBenchmark[InstanceType]):
    """
    An immutable definition of a benchmark task used to evaluate function performance.

    This class serves as the entry point for the benchmarking lifecycle. It defines
    the data (instances), the success criteria (metrics), and the execution
    parameters (timeout, attempts). Once defined, calling the `run` method
    transitions the benchmark into a `BenchmarkExec` state.
    """

    def run(
        self,
        fn: BenchmarkCallable,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkExec:
        start_time = time.perf_counter()

        self.logger.info(f"Running benchmark {self._spec.name} for {fn.__name__}")

        self._check_consistency_signature(fn)

        for instance in self.instances:
            for attempt_id in range(1, self._spec.n_attempts + 1):
                timed_execution = timed_exec(
                    fn=fn,
                    timeout=self._spec.timeout,
                    instance=instance,
                    args=args,
                    kwargs=kwargs,
                )

                if timed_execution.is_success:
                    self.logger.info(f"Instance {instance.id} successfull benchmarked")
                    status = "success"
                elif timed_execution.is_timeout:
                    self.logger.info(f"Instance {instance.id} timed out")
                    status = "timeout"
                elif timed_execution.is_error:
                    self.logger.error(
                        f"Error evaluating instance {instance.id}: {timed_execution.exception}"
                    )
                    status = "failure"
                else:
                    raise RuntimeError("This should never happens.")

                response = (
                    timed_execution.result.pop("answer", None)
                    if timed_execution.result
                    else None
                )
                token_usage = (
                    timed_execution.result.pop("tokens_usage", None)
                    if timed_execution.result
                    else {}
                )
                instance.add_attempt(
                    response=response,
                    runtime=timed_execution.runtime,
                    status=status,
                    token_usage=token_usage,
                )

        updated_spec = self._spec.set_execution_time(time.perf_counter() - start_time)
        return BenchmarkExec.new(
            source=list(self.instances),
            metrics=self.metrics,
            aggregators=self.aggregators,
            logger=self.logger,
            **updated_spec.to_dict(),
        )

    def _check_consistency_signature(self, fn: BenchmarkCallable) -> None:
        annotations = fn.__annotations__
        return_type = annotations.pop("return", None)
        warning_msg: list[str] = []
        if not annotations:
            warning_msg.append("No annotations provided.")
        elif return_type is None:
            warning_msg.append("No return type provided.")

        if warning_msg:
            warning_msg.append(
                """
                Note that we expect an annotated callable, with return type `BenchmarkOutput`, or an equivalent dict.\n
                Example: 
                    def fn(instance, *args, **kwargs) -> BenchmarkOutput: ...\n
                """
            )
            self.logger.warning(warning_msg)

    async def run_async(self) -> None: ...

    def _generate_summary_table(self) -> table.Table:
        """
        Generates a rich table summary of the benchmark configuration,
        including metrics, aggregators, and instance constraints.
        """
        summary_table = table.Table(title="Benchmark Summary")

        summary_table.add_column("Property", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row("Benchmark name", self._spec.name)
        summary_table.add_row("Number of Instances", str(len(self.instances)))
        summary_table.add_row("Attempts per Instance", str(self._spec.n_attempts))
        summary_table.add_row(
            "Timeout", f"{self._spec.timeout}s" if self._spec.timeout else "None"
        )

        # Metrics & Aggregators
        metrics_list = ", ".join([m.name for m in self._metrics]) or "None"
        aggs_list = ", ".join([a.name for a in self.aggregators]) or "None"

        summary_table.add_row("Metrics", metrics_list)
        summary_table.add_row("Aggregators", aggs_list)
        if self._spec.logs_filepath:
            summary_table.add_row("Logs Path", self._spec.logs_filepath)

        return summary_table
