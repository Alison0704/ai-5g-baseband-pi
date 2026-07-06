import pytest

from src.ml.failure_classifier import diagnose_fault


def test_rejects_unknown_modulation():
    with pytest.raises(ValueError):
        diagnose_fault(
            snr_db=30,
            modulation="256QAM",
            bits_per_symbol=8,
            ber=0,
            evm_percent=1,
            pilot_error_percent=1,
            average_symbol_power=1,
            estimated_channel_magnitude=1,
        )


def test_rejects_invalid_ber():
    with pytest.raises(ValueError):
        diagnose_fault(
            snr_db=30,
            modulation="16QAM",
            bits_per_symbol=4,
            ber=1.5,
            evm_percent=1,
            pilot_error_percent=1,
            average_symbol_power=1,
            estimated_channel_magnitude=1,
        )


def test_rejects_incorrect_bits_per_symbol():
    with pytest.raises(ValueError):
        diagnose_fault(
            snr_db=30,
            modulation="64QAM",
            bits_per_symbol=4,
            ber=0,
            evm_percent=1,
            pilot_error_percent=1,
            average_symbol_power=1,
            estimated_channel_magnitude=1,
        )
