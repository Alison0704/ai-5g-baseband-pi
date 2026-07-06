import numpy as np
import pytest

from src.metrics.evm import calculate_evm_rms, evm_to_percent


def test_zero_evm_for_identical_symbols():
    symbols = np.array([1 + 1j, -1 - 1j])

    assert calculate_evm_rms(symbols, symbols) == 0.0


def test_known_evm():
    reference = np.array([1 + 0j, 1 + 0j])
    received = np.array([1.1 + 0j, 0.9 + 0j])

    evm = calculate_evm_rms(reference, received)

    assert evm == pytest.approx(0.1)


def test_evm_percentage_conversion():
    assert evm_to_percent(0.125) == pytest.approx(12.5)


def test_rejects_different_lengths():
    with pytest.raises(ValueError):
        calculate_evm_rms([1 + 0j], [1 + 0j, 2 + 0j])
