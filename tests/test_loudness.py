import math

import numpy as np
import pytest

from app.components.loudness import (
    ABS_GATE_LUFS,
    BLOCK_S,
    SilenceError,
    TooShortError,
    measure,
)
from tests.signals import (
    bandlimited_sine,
    left_right_imbalance,
    narrow_impulse,
    short_click,
    silence,
)


def test_full_scale_sine_integrated_loudness():
    # 1 kHz sine at amplitude 1.0: K-weighting is ~transparent at 1 kHz,
    # integrated loudness must be -3.01 LUFS.
    rate = 48000
    n = rate * 5
    x = np.sin(2 * np.pi * 1000 * np.arange(n) / rate).reshape(-1, 1)
    m = measure(x, rate)
    assert abs(m.integrated_lufs - (-3.01)) < 0.2
    assert m.sample_peak_dbfs == pytest.approx(0.0, abs=0.01)


def test_half_scale_sine_is_6_db_quieter():
    rate = 48000
    n = rate * 5
    base = np.sin(2 * np.pi * 1000 * np.arange(n) / rate)
    full = measure(base.reshape(-1, 1), rate)
    half = measure((0.5 * base).reshape(-1, 1), rate)
    assert abs((full.integrated_lufs - half.integrated_lufs) - 6.0) < 0.1


def test_sample_peak_and_true_peak_are_distinct_measures():
    x, rate = bandlimited_sine()
    m = measure(x, rate)
    # The sampled sine peaks between samples; the interpolated true peak is
    # strictly higher than the sample peak.
    assert m.true_peak_dbtp > m.sample_peak_dbfs
    assert m.true_peak_dbtp <= 0.1  # but bounded, never absurdly over


def test_true_peak_of_well_sampled_tone_equals_sample_peak():
    rate = 48000
    n = rate * 4
    x = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / rate)).reshape(-1, 1)
    m = measure(x, rate)
    assert abs(m.true_peak_dbtp - m.sample_peak_dbfs) < 0.05


def test_silence_raises_and_is_reported_separately():
    x, rate = silence(2.0)
    with pytest.raises(SilenceError):
        measure(x, rate)
    m = measure(x, rate, allow_failure=True)
    assert m.integrated_lufs is None
    assert "digital_silence" in m.warnings
    assert m.sample_peak_dbfs <= -70.0


def test_short_segment_below_one_gating_block_raises():
    x, rate = short_click()
    assert x.shape[0] / rate < BLOCK_S
    with pytest.raises(TooShortError):
        measure(x, rate)
    m = measure(x, rate, allow_failure=True)
    assert "shorter_than_one_gating_block" in m.warnings


def test_narrow_impulse_high_peak_near_zero_energy():
    x, rate = narrow_impulse(2.0)
    m = measure(x, rate, allow_failure=True)
    assert m.sample_peak_dbfs == pytest.approx(0.0, abs=0.01)
    # RMS is tiny despite the full-scale sample.
    assert m.rms_dbfs < -45.0
    # The measurement stays finite — no infinities anywhere.
    assert np.isfinite(m.true_peak_dbtp) and np.isfinite(m.sample_peak_dbfs)


def test_stereo_imbalance_is_measured():
    x, rate = left_right_imbalance()
    m = measure(x, rate)
    assert m.n_channels == 2 and m.channel_layout == "stereo"
    assert m.integrated_lufs is not None and m.integrated_lufs < 0


def test_block_count_and_gating_flags():
    rate = 48000
    n = rate * 3
    x = (0.3 * np.sin(2 * np.pi * 500 * np.arange(n) / rate)).reshape(-1, 1)
    m = measure(x, rate)
    assert m.absolute_gated and m.relative_gated
    assert m.n_gated_blocks > 0
    # 3 s with 0.4 s blocks / 0.1 s hop -> 27 blocks.
    assert m.n_total_blocks == 27


def test_measurements_are_finite_for_all_fixtures():
    from tests.signals import ALL_FIXTURES

    for name, factory in ALL_FIXTURES.items():
        x, rate = factory()
        m = measure(x, rate, allow_failure=True)
        for value in (m.sample_peak_dbfs, m.true_peak_dbtp, m.rms_dbfs):
            assert math.isfinite(value), name
        if m.integrated_lufs is not None:
            assert math.isfinite(m.integrated_lufs), name


def test_non_finite_input_rejected():
    x = np.full((48000, 1), np.nan)
    with pytest.raises(ValueError):
        measure(x, 48000)
