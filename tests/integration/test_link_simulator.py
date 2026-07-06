import numpy as np
import pytest

from src.link_simulator import simulate_link_frame


@pytest.mark.parametrize(
    "modulation",
    ["QPSK", "16QAM", "64QAM"],
)
def test_link_simulator_at_high_snr(modulation):
    result = simulate_link_frame(
        modulation=modulation,
        snr_db=60,
        channel_coefficient=0.8 + 0.3j,
        rng=np.random.default_rng(123),
    )

    assert result.ber == 0.0

    assert abs(
        result.estimated_channel
        - result.true_channel
    ) < 0.01

    assert result.evm_percent < 1.0


def test_strong_interference_increases_evm():
    clean_result = simulate_link_frame(
        modulation="16QAM",
        snr_db=30,
        channel_coefficient=0.8 + 0.3j,
        rng=np.random.default_rng(123),
    )

    interfered_result = simulate_link_frame(
        modulation="16QAM",
        snr_db=30,
        channel_coefficient=0.8 + 0.3j,
        interference_to_signal_db=-5,
        rng=np.random.default_rng(123),
    )

    assert (
        interfered_result.evm_percent
        > clean_result.evm_percent
    )


def test_rejects_unknown_modulation():
    with pytest.raises(ValueError):
        simulate_link_frame(
            modulation="256QAM",
            snr_db=20,
        )
