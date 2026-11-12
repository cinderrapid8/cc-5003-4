import math

import numpy as np
import pytest

from app.components.resampler import resample


def _tone(freq, n, rate, amp=0.5):
    t = np.arange(n) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).reshape(-1, 1)


def test_identity_rate_returns_copy():
    x = _tone(1000, 4800, 48000)
    y, settings = resample(x, 48000, 48000)
    assert settings.up == 1 and settings.down == 1
    assert np.array_equal(y, x)


def test_integer_upsample_matches_analytic_tone():
    rate = 24000
    x = _tone(1000, 24000, rate)
    y, _ = resample(x, rate, 48000)
    ref = _tone(1000, y.shape[0], 48000)
    trim = 1500
    assert np.max(np.abs(y[trim:-trim] - ref[trim : y.shape[0] - trim])) < 1e-5


def test_integer_downsample_matches_analytic_tone():
    rate = 48000
    x = _tone(3000, 48000, rate)
    y, _ = resample(x, rate, 24000)
    ref = _tone(3000, y.shape[0], 24000)
    trim = 1500
    assert np.max(np.abs(y[trim:-trim] - ref[trim : y.shape[0] - trim])) < 1e-5


def test_rational_44100_to_48000_matches_analytic():
    in_rate, out_rate = 44100, 48000
    x = _tone(1000, 44100, in_rate)  # 1 s keeps the polyphase cost bounded
    y, settings = resample(x, in_rate, out_rate)
    assert settings.up == 160 and settings.down == 147
    ref = _tone(1000, y.shape[0], out_rate)
    trim = 2000
    assert np.max(np.abs(y[trim:-trim] - ref[trim : y.shape[0] - trim])) < 2e-5


def test_dc_gain_unity_at_4x_oversample():
    dc = np.full((12000, 1), 0.7)
    y, _ = resample(dc, 48000, 192000, transition_band_hz=4000.0)
    interior = y[1000:-1000]
    assert np.all(np.abs(interior - 0.7) < 1e-4)


def test_impulse_oversample_peak_position_and_amplitude():
    # A unit impulse reconstructed at 4x must peak at the aligned phase with
    # the filter's centre coefficient, never above unity.
    x = np.zeros((4800, 1))
    x[2400, 0] = 1.0
    y, _ = resample(x, 48000, 192000, transition_band_hz=4000.0)
    peak = float(np.max(np.abs(y)))
    # Within the Kaiser filter's 80 dB stop-band ripple (~1e-4).
    assert peak <= 1.0 + 2e-4
    assert int(np.argmax(np.abs(y))) == 2400 * 4


def test_output_length_follows_ratio():
    x = np.zeros((1000, 2))
    y, _ = resample(x, 48000, 16000)
    assert y.shape[0] == math.ceil(1000 / 3)
    assert y.shape[1] == 2


def test_rejects_invalid_rates():
    x = np.zeros((100, 1))
    with pytest.raises(ValueError):
        resample(x, 0, 48000)
    with pytest.raises(ValueError):
        resample(x, 48000, -1)


def test_settings_are_deterministic():
    x = _tone(1000, 4800, 48000)
    y1, s1 = resample(x, 48000, 192000, transition_band_hz=4000.0)
    y2, s2 = resample(x, 48000, 192000, transition_band_hz=4000.0)
    assert np.array_equal(y1, y2)
    assert s1.as_dict() == s2.as_dict()
