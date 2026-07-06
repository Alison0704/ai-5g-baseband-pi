import numpy as np
import pytest

from src.fault_injection import (
    FAULT_TYPES,
    simulate_faulty_link_frame,
)


def test_clean_frame_has_zero_ber_at_high_snr():
    result = simulate_faulty_link_frame(
        fault_type="NONE",
        snr_db=60,
        rng=np.random.default_rng(123),
    )

    assert result.ber == 0.0
    assert result.evm_percent < 1.0


@pytest.mark.parametrize("fault_type", FAULT_TYPES)
def test_all_fault_types_produce_valid_measurements(
    fault_type,
):
    result = simulate_faulty_link_frame(
        fault_type=fault_type,
        snr_db=35,
        rng=np.random.default_rng(123),
    )

    assert 0 <= result.ber <= 1
    assert result.evm_percent >= 0
    assert result.pilot_error_percent >= 0
    assert result.average_symbol_power >= 0
    assert result.estimated_channel_magnitude >= 0


def test_pilot_corruption_increases_pilot_error():
    clean = simulate_faulty_link_frame(
        fault_type="NONE",
        snr_db=60,
        rng=np.random.default_rng(123),
    )

    corrupted = simulate_faulty_link_frame(
        fault_type="PILOT_CORRUPTION",
        snr_db=60,
        rng=np.random.default_rng(123),
    )

    assert (
        corrupted.pilot_error_percent
        > clean.pilot_error_percent
    )


def test_rejects_unknown_fault():
    with pytest.raises(ValueError):
        simulate_faulty_link_frame(
            fault_type="FFT_EXPLOSION",
        )
