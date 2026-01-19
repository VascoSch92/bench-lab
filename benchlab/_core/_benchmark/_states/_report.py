from dataclasses import dataclass, field

from benchlab._core._benchmark._states._base import BaseBenchmark
from benchlab._core._types import InstanceType
from benchlab._core._evaluation._aggregators._aggregator import Report
from rich import table

__all__ = ["BenchmarkReport"]


@dataclass(frozen=True, slots=True)
class BenchmarkReport(BaseBenchmark[InstanceType]):
    _reports: list[Report] = field(default_factory=list)

    def _task_specific_checks(self) -> None:
        # todo: better error message
        if len(self.reports) != len(self.aggregators):
            raise ValueError

    @property
    def reports(self) -> list[Report]:
        return self._reports

    def _generate_summary_table(self) -> table.Table:
        summary_table = table.Table(title="Report Summary")

        summary_table.add_column("Aggregator")
        summary_table.add_column("Value")

        for report in self.reports:
            summary_table.add_row(report.aggregator_name, str(report.outer_output))

        return summary_table
