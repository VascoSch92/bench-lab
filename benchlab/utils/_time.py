import time
from dataclasses import dataclass
from typing import Any, Callable

from func_timeout import func_timeout, FunctionTimedOut  # type: ignore[import-untyped]


__all__ = ["timed_exec"]

from benchlab._types import InstanceType


@dataclass(frozen=True, slots=True)
class TimedExec:
    runtime: float | None
    result: Any | None
    exception: Exception | None

    @property
    def is_success(self) -> bool:
        return self.exception is None

    @property
    def is_timeout(self) -> bool:
        return self.exception is not None and isinstance(
            self.exception, FunctionTimedOut
        )

    @property
    def is_error(self) -> bool:
        return self.exception is not None and not isinstance(
            self.exception, FunctionTimedOut
        )


def timed_exec(
    fn: Callable,
    timeout: float | None,
    instance: InstanceType,
    kwargs: dict[str, Any] | None = None,
) -> TimedExec:
    try:
        start = time.perf_counter()
        result = func_timeout(timeout, fn, args=(instance,), kwargs=kwargs)
        runtime = time.perf_counter() - start
        return TimedExec(
            result=result,
            runtime=runtime,
            exception=None,
        )
    except FunctionTimedOut as e:
        return TimedExec(
            result=None,
            runtime=None,
            exception=e,
        )
    except Exception as e:
        return TimedExec(
            result=None,
            runtime=None,
            exception=e,
        )
