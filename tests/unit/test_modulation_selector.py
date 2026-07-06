import pytest

from src.ml.modulation_selector import (
    select_best_modulation_from_ber,
)


def test_selects_64qam_when_all_modulations_pass():
    result = select_best_modulation_from_ber(
        {
            "QPSK": 0.0,
            "16QAM": 0.002,
            "64QAM": 0.008,
        }
    )

    assert result == "64QAM"


def test_selects_16qam_when_64qam_fails():
    result = select_best_modulation_from_ber(
        {
            "QPSK": 0.0,
            "16QAM": 0.005,
            "64QAM": 0.04,
        }
    )

    assert result == "16QAM"


def test_selects_qpsk_when_only_qpsk_passes():
    result = select_best_modulation_from_ber(
        {
            "QPSK": 0.005,
            "16QAM": 0.03,
            "64QAM": 0.10,
        }
    )

    assert result == "QPSK"


def test_falls_back_to_qpsk_during_outage():
    result = select_best_modulation_from_ber(
        {
            "QPSK": 0.20,
            "16QAM": 0.30,
            "64QAM": 0.40,
        }
    )

    assert result == "QPSK"


def test_rejects_missing_modulation():
    with pytest.raises(ValueError):
        select_best_modulation_from_ber(
            {
                "QPSK": 0.0,
                "16QAM": 0.01,
            }
        )
