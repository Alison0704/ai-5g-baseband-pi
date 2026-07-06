"""Runtime profiling utilities."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass
class ProfileResult:
    """Execution-time statistics."""

    iterations: int
    mean_latency_ms: float
    minimum_latency_ms: float
    maximum_latency_ms: float
    standard_deviation_ms: float
    frames_per_second: float


def profile_function(
    function: Callable[..., Any],
    *,
    iterations: int = 100,
    warmup_iterations: int = 5,
    **kwargs: Any,
) -> ProfileResult:
    """Measure execution time across repeated function calls."""
    if iterations <= 0:
        raise ValueError("Iterations must be positive.")

    if warmup_iterations < 0:
        raise ValueError(
            "Warmup iterations cannot be negative."
        )

    for _ in range(warmup_iterations):
        function(**kwargs)

    latencies_ms: list[float] = []

    for _ in range(iterations):
        start = perf_counter()
        function(**kwargs)
        end = perf_counter()

        latencies_ms.append(
            (end - start) * 1000
        )

    latency_array = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    mean_latency = float(
        np.mean(latency_array)
    )

    return ProfileResult(
        iterations=iterations,
        mean_latency_ms=mean_latency,
        minimum_latency_ms=float(
            np.min(latency_array)
        ),
        maximum_latency_ms=float(
            np.max(latency_array)
        ),
        standard_deviation_ms=float(
            np.std(latency_array)
        ),
        frames_per_second=(
            1000 / mean_latency
        ),
    )
