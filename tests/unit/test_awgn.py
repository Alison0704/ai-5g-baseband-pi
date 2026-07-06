import numpy as np

from src.channel.awgn import add_awgn


def test_infinite_snr_adds_no_noise():
    signal = np.ones(8, dtype=np.complex128)

    received = add_awgn(signal, snr_db=np.inf)

    np.testing.assert_array_equal(received, signal)


def test_awgn_changes_signal():
    signal = np.ones(100, dtype=np.complex128)
    rng = np.random.default_rng(123)

    received = add_awgn(signal, snr_db=10, rng=rng)

    assert not np.array_equal(received, signal)


def test_awgn_preserves_signal_length():
    signal = np.ones(64, dtype=np.complex128)

    received = add_awgn(
        signal,
        snr_db=20,
        rng=np.random.default_rng(123),
    )

    assert received.shape == signal.shape
