import json
import logging
from pathlib import Path
from typing import Literal
from rich.logging import RichHandler

import yaml

__all__ = ["get_logger"]


class LogFormatter(logging.Formatter):
    """Formatter that outputs log records."""

    def __init__(self, format_type: Literal["json", "yaml", "text"]) -> None:
        super().__init__()
        if format_type not in ["json", "yaml", "text"]:
            raise ValueError("Format type must be either `json`, `yaml` or `text`")
        self.format_type = format_type

    def _get_log_record_dict(self, record: logging.LogRecord) -> dict[str, str]:
        """Returns a log record in dict form."""
        return {
            "time": self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

    def format(self, record: logging.LogRecord) -> str:
        match self.format_type:
            case "json":
                return self._format_json(record)
            case "yaml":
                return self._format_yaml(record)
            case "text":
                return self._format_text(record)
            case _:
                raise RuntimeError(f"Unknown log record type: {self.format_type}")

    def _format_json(self, record: logging.LogRecord) -> str:
        log_record = self._get_log_record_dict(record)
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

    def _format_yaml(self, record: logging.LogRecord) -> str:
        # todo: look at how the logging are saved
        log_record = self._get_log_record_dict(record)
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return f"---\n{yaml.dump(log_record).rstrip()}"

    def _format_text(self, record: logging.LogRecord) -> str:
        s = f"[{self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S')} - {record.name} - {record.levelname}] {record.getMessage()}"
        if record.exc_info:
            s += "\n" + self.formatException(record.exc_info)
        return s


def get_logger(
    name: str,
    path: Path | str | None = None,
    console: bool = True,
    level: int = logging.INFO,
) -> logging.Logger:
    """Returns a configured logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False

        if console:
            console_handler = RichHandler(
                level=level,
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=True,
            )
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                "[bold]OpenBench[/bold] | %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        if path is None:
            return logger

        match Path(path).suffix:
            case ".json":
                formatter = LogFormatter(format_type="json")
            case ".yaml" | ".yml":
                formatter = LogFormatter(format_type="yaml")
            case ".txt":
                formatter = LogFormatter(format_type="text")
            case _:
                raise ValueError(f"Unknown logger format: {name}")

        format_handler = logging.FileHandler(path)
        format_handler.setFormatter(formatter)
        logger.addHandler(format_handler)

    return logger
