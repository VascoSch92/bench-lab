from typing import TypeAlias, Protocol, runtime_checkable, Any, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from benchlab._core._instances import Instance
    from benchlab._core._evaluation._stats import BooleanMetricStats


"""Typying Benchmark related."""

InstanceType = TypeVar("InstanceType", bound="Instance")
"""Type Var for an instance."""

@runtime_checkable
class BenchmarkCallable(Protocol):
    """Protocol for callable runnable in a benchmark instance."""

    __name__: str

    def __call__(
        self, instance: InstanceType, *args: Any, **kwargs: Any
    ) -> AnswerType: ...


AnswerType: TypeAlias = str | None
"""Type alias for the answer type of a LLM."""


"""Typying for metric stats."""

MetricOutputType = TypeVar("MetricOutputType")
"""Type Var for the output type of a metric."""

RegressionOutputType: TypeAlias = int | float | None
"""Type alias for the output type of a regression metric."""

BooleanOutputType: TypeAlias = bool | None
"""Type alias for the output type of a boolean metric."""

CategoricalOutputType: TypeAlias = str | int | None
"""Type alias for the output type of a categorical metric."""


"""Typying for aggregator."""

AggregatorType = TypeVar("AggregatorType")
"""Type var for the aggregator Type"""

AggregatorBooleanType: TypeAlias = BooleanMetricStats
"""Type for aggregator on boolean stats."""
