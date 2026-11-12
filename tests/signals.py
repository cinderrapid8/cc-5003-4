"""Deterministic test-signal generators used by the test suite and the
preview fixture script. Every generator returns (samples, sample_rate).
"""
from __future__ import annotations

import numpy as np

DEFAULT_RATE = 48000


def _tone(freq: float, n: int, rate: int, amp: float = 0.5, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / rate
    return amp * np.sin(2 * np.pi * freq * t + phase)


def silence(seconds: float = 2.0, rate: int = DEFAULT_RATE, stereo: bool = False):
    n = int(seconds * rate)
    shape = (n, 2) if stereo else (n, 1)
    return np.zeros(shape, dtype=np.float64), rate


def short_click(rate: int = DEFAULT_RATE):
    """0.2 s — shorter than one 0.4 s gating block."""
    x = np.zeros((int(0.2 * rate), 1), dtype=np.float64)
    x[rate // 100] = 0.9
    return x, rate


def narrow_impulse(seconds: float = 2.0, rate: int = DEFAULT_RATE):
    """One full-scale sample amid digital silence: high peak, ~zero energy."""
    x = np.zeros((int(seconds * rate), 1), dtype=np.float64)
    x[x.shape[0] // 2, 0] = 1.0
    return x, rate


def bandlimited_sine(seconds: float = 2.0, rate: int = DEFAULT_RATE, freq: float = 0.23 * 48000):
    """A sine at 0.23*rate peaks between samples when sampled: the 4x
    interpolated true peak must exceed the sample peak."""
    n = int(seconds * rate)
    x = _tone(freq, n, rate, amp=0.7).reshape(-1, 1)
    return x, rate


def left_right_imbalance(seconds: float = 3.0, rate: int = DEFAULT_RATE):
    """Stereo signal with a 12 dB level difference and a silent right tail."""
    n = int(seconds * rate)
    left = _tone(440.0, n, rate, amp=0.6)
    right = _tone(440.0, n, rate, amp=0.15)
    right[int(n * 0.75):] = 0.0
    return np.stack([left, right], axis=1), rate


def high_peak_low_average(seconds: float = 4.0, rate: int = DEFAULT_RATE):
    """Quiet speech-like floor (-45 dBFS RMS) with occasional 0 dBFS spikes."""
    rng = np.random.Generator(np.random.PCG64(20261112))
    n = int(seconds * rate)
    floor = _tone(300.0, n, rate, amp=0.005)
    floor += rng.normal(0, 0.0015, n)
    for center in np.linspace(n // 8, n - n // 8, 5).astype(int):
        floor[center] = 1.0
        floor[center + 1] = -1.0
    return floor.reshape(-1, 1), rate


def calibrated_speech_like(seconds: float = 10.0, rate: int = DEFAULT_RATE,
                           target_lufs: float = -23.0):
    """Band-limited pink-ish noise shaped to a requested loudness after a
    pilot measurement, giving a stable reference for end-to-end LUFS checks."""
    rng = np.random.Generator(np.random.PCG64(0xB17C))
    n = int(seconds * rate)
    white = rng.normal(0, 1, n)
    # crude pink shaping: cumulative low-pass mix
    shaped = np.zeros(n)
    acc = 0.0
    for i, v in enumerate(white):
        acc = 0.98 * acc + 0.02 * v
        shaped[i] = 0.7 * acc + 0.3 * v
    # gentle amplitude modulation to mimic speech dynamics
    t = np.arange(n) / rate
    shaped *= 0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t) ** 2
    shaped /= np.max(np.abs(shaped))
    shaped *= 0.8

    from app.components.loudness import measure

    probe = measure(shaped.reshape(-1, 1), rate)
    correction = 10.0 ** ((target_lufs - probe.integrated_lufs) / 20.0)
    return (shaped * correction).reshape(-1, 1), rate


ALL_FIXTURES = {
    "silence_mono": lambda: silence(2.0, stereo=False),
    "silence_stereo": lambda: silence(2.0, stereo=True),
    "short_click": short_click,
    "narrow_impulse": narrow_impulse,
    "bandlimited_sine": bandlimited_sine,
    "lr_imbalance": left_right_imbalance,
    "high_peak_low_average": high_peak_low_average,
    "speech_minus23": calibrated_speech_like,
}
