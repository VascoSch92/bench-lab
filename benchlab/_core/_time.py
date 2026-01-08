import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any
from typing import Callable


# todo: the current implementation is very slow. Can we do better and ensure a correct timeout?

@dataclass(frozen=True, slots=True)
class TimedExec:
    runtime: float
    result: Any | None
    exception: Exception | None

    def is_success(self) -> bool:
        return self.exception is None

    def is_timeout(self) -> bool:
        return self.exception is not None and isinstance(self.exception, TimeoutError)

    def is_error(self) -> bool:
        return self.exception is not None


def _runner_wrapper(q: Queue, fn: Callable, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        q.put(("ok", result))
    except Exception as e:
        q.put(("err", e))


def _timed_exec(
    fn: Callable,
    timeout: float | None,
    *args,
    **kwargs,
) -> TimedExec:
    q: Queue = Queue()

    start = time.perf_counter()
    p = Process(target=_runner_wrapper, args=(q, fn, *args), kwargs=kwargs)
    p.start()

    p.join(timeout)
    runtime = time.perf_counter() - start

    if p.is_alive():
        p.terminate()
        p.join()  # Clean up the zombie process
        return TimedExec(
            result=None,
            runtime=runtime,
            exception=TimeoutError(f"Timeout after {timeout}s"),
        )

    # Use a small timeout on get to avoid hanging if the queue is corrupted
    try:
        if q.empty():
            return TimedExec(
                runtime=runtime,
                result=None,
                exception=Exception("Process exited without returning data"),
            )
        status, payload = q.get(timeout=0.1)
    except Exception as e:
        return TimedExec(runtime=runtime, result=None, exception=e)
    finally:
        q.close()  # Clean up resources

    if status == "ok":
        return TimedExec(runtime=runtime, result=payload, exception=None)
    return TimedExec(result=None, runtime=runtime, exception=payload)
