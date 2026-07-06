import numpy as np
import pytest

from src.channel.interference import add_tone_interference


def test_interference_preserves_signal_shape():
    signal = np.ones(64, dtype=np.complex128)

    received, interference = add_tone_interference(
        signal,
        interference_to_signal_db=-20,
    )

    assert received.shape == signal.shape
    assert interference.shape == signal.shape


def test_interference_changes_signal():
    signal = np.ones(64, dtype=np.complex128)

    received, _ = add_tone_interference(
        signal,
        interference_to_signal_db=-20,
    )

    assert not np.array_equal(received, signal)


def test_interference_power_ratio():
    signal = np.ones(1024, dtype=np.complex128)

    _, interference = add_tone_interference(
        signal,
        interference_to_signal_db=-20,
        normalized_frequency=0.1,
    )

    signal_power = np.mean(np.abs(signal) ** 2)
    interference_power = np.mean(
        np.abs(interference) ** 2
    )

    expected_ratio = 10 ** (-20 / 10)

    assert interference_power / signal_power == pytest.approx(
        expected_ratio,
        rel=1e-12,
    )


def test_zero_db_interference_has_equal_power():
    signal = np.ones(128, dtype=np.complex128)

    _, interference = add_tone_interference(
        signal,
        interference_to_signal_db=0,
    )

    assert np.mean(
        np.abs(interference) ** 2
    ) == pytest.approx(
        np.mean(np.abs(signal) ** 2)
    )


def test_rejects_invalid_frequency():
    with pytest.raises(ValueError):
        add_tone_interference(
            np.ones(8),
            interference_to_signal_db=-10,
            normalized_frequency=0.5,
        )
