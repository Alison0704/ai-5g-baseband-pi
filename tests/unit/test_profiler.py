import pytest

from src.metrics.profiler import profile_function


def sample_function(value):
    return value * 2


def test_profile_function_returns_valid_metrics():
    result = profile_function(
        sample_function,
        iterations=10,
        warmup_iterations=2,
        value=5,
    )

    assert result.iterations == 10
    assert result.mean_latency_ms > 0
    assert result.minimum_latency_ms > 0
    assert result.maximum_latency_ms >= (
        result.minimum_latency_ms
    )
    assert result.frames_per_second > 0


def test_rejects_zero_iterations():
    with pytest.raises(ValueError):
        profile_function(
            sample_function,
            iterations=0,
            value=5,
        )
