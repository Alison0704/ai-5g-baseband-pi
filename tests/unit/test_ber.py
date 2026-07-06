import pytest

from src.metrics.ber import calculate_ber


def test_zero_bit_errors():
    assert calculate_ber(
        [0, 1, 1, 0],
        [0, 1, 1, 0],
    ) == 0.0


def test_known_bit_error_rate():
    assert calculate_ber(
        [0, 1, 1, 0],
        [1, 1, 0, 0],
    ) == pytest.approx(0.5)


def test_rejects_different_lengths():
    with pytest.raises(ValueError):
        calculate_ber([0, 1], [0])
