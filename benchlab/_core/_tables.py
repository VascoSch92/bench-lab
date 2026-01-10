import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
import statistics
import yaml
from rich.table import Table

if TYPE_CHECKING:
    from benchlab._core._evaluation._metrics import Metric
    from benchlab._core._instances import Instance


def get_table_for_instance(
    instances: list["Instance"],
    metrics: dict[str, "Metric"],
    mode: str = "grouped",
) -> Table:
    table = Table(title="Benchmark Results", show_lines=True)

    # Define columns
    table.add_column("Instance id", style="cyan", justify="left")
    table.add_column("Attempt", style="green", justify="center")

    runtime_header = "Runtime (s)"
    if mode == "grouped":
        runtime_header = "Average Runtime (s)"
    table.add_column(runtime_header, style="magenta", justify="center")
    for metric_name, metric in metrics.items():
        column_header = metric_name
        if mode == "grouped":
            column_header = column_header + f" ({metric.type_})"
        table.add_column(column_header, style="yellow", justify="center")

    # Fill table rows
    for idx, instance in enumerate(instances):
        # this is for grouped mode
        instance_id = str(instance.id)
        attempt = str(instance.responses[0])
        runtime = str(round(statistics.mean(instance.runtimes), 4))

        # Fill metrics with dummy 0 values (replace with real metric values if available)
        metrics_values = [
            "[green]True[/green]" if idx % 2 == 0 else "[red]False[/red]"
            for _ in range(len(metrics))
        ]

        table.add_row(instance_id, attempt, runtime, *metrics_values)

    return table


def _strip_rich_markup(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text) if text else text


def _convert_table_in_dict_format(table: Table) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []
    header_to_cell = {col.header: col.cells for col in table.columns}

    for j in range(table.row_count):
        row: dict[str, str] = {}
        for header, gen in header_to_cell.items():
            row[header] = _strip_rich_markup(next(gen))
        data.append(row)
    return data


def _export_to_json(table: Table, filepath: Path):
    """Save a Rich Table to JSON file."""
    data = _convert_table_in_dict_format(table)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _export_to_yaml(table: Table, filepath: Path):
    data = _convert_table_in_dict_format(table)

    with filepath.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


# todo: add different type of export: like markdown and html


def export_table(table: Table, filepath: Path | str) -> None:
    filepath = Path(filepath)
    match filepath.suffix:
        case ".json":
            _export_to_json(table, filepath)
        case ".yaml" | ".yml":
            _export_to_yaml(table, filepath)
        case _:
            raise ValueError
