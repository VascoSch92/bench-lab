import pytest
from benchlab._core._evaluation._stats import MetricStats, BooleanMetricStats


@pytest.mark.parametrize(
    "stats, exception, msg", [([], ValueError, "Empty stats list")]
)
def test_init_exception(stats, exception, msg) -> None:
    with pytest.raises(exception, match=msg):
        _ = MetricStats.aggregate(stats)


@pytest.mark.parametrize(
    "metric_name, values, n_valid_attempts, n_true, n_false",
    [
        ("name_1", [], 0, 0, 0),
        ("name_2", [True], 1, 1, 0),
    ],
)
def test_boolean_metrics_stats_from_eval(
    metric_name, values, n_valid_attempts, n_true, n_false
):
    stats = BooleanMetricStats.from_eval(metric_name=metric_name, values=values)

    assert stats.metric_name == metric_name
    assert stats.n_attempts == len(values)
    assert stats.n_valid_attempts == n_valid_attempts
    assert stats.n_true == n_true
    assert stats.n_false == n_false
