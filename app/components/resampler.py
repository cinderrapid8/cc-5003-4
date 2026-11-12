"""Rational-ratio polyphase resampling with a Kaiser-windowed sinc filter.

The same component serves two roles in the pipeline:
  * sample-rate conversion (e.g. 44.1 kHz -> 48 kHz), ratio L/M reduced
    by their greatest common divisor;
  * the 4x oversampling stage used for interpolated true-peak estimation.

Design model (upfirdn): the anti-alias / reconstruction low-pass is designed
on the *input-rate* grid with a physically meaningful transition band, then
zero-stuffed to the high-rate grid implicitly through the polyphase branch
indexing. For a kernel tap j = p + L*r the output high-rate index q with
q mod L == p pulls input frame q//L - r. The kernel parameters are recorded
per task so every result is reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResamplerSettings:
    up: int
    down: int
    cutoff_fraction: float
    transition_band_hz: float
    stopband_attenuation_db: float
    filter_taps: int
    input_rate: int
    output_rate: int

    def as_dict(self) -> dict:
        return {
            "up": self.up,
            "down": self.down,
            "cutoff_fraction_of_output_nyquist": self.cutoff_fraction,
            "transition_band_hz": self.transition_band_hz,
            "stopband_attenuation_db": self.stopband_attenuation_db,
            "filter_taps": self.filter_taps,
            "input_rate": self.input_rate,
            "output_rate": self.output_rate,
        }


def _kaiser_beta(attenuation_db: float) -> float:
    a = attenuation_db
    if a > 50.0:
        return 0.1102 * (a - 8.7)
    if a >= 21.0:
        return 0.5842 * (a - 21.0) ** 0.4 + 0.07886 * (a - 21.0)
    return 0.0


def _design_kernel(up: int, down: int, in_rate: int, high_grid_rate: int,
                   attenuation_db: float, transition_hz: float
                   ) -> tuple[np.ndarray, int, float]:
    # Prototype low-pass on the L-times upsampled (high) grid using Kaiser
    # window design (scipy firwin conventions), then scaled by L so the
    # zero-stuffed stream reconstructs with unity gain.
    high_rate = float(high_grid_rate)
    pass_edge = in_rate / 2.0
    cutoff_cycles = pass_edge / high_rate  # cycles per high-rate sample
    # Kaiser estimate in radians-per-sample: domega = 2*pi * transition/fs.
    domega = 2.0 * math.pi * transition_hz / high_rate
    if cutoff_cycles <= transition_hz / high_rate:
        raise ValueError("transition band too wide for requested rate ratio")
    beta = _kaiser_beta(attenuation_db)
    numtaps = int(math.ceil((attenuation_db - 8.0) / (2.285 * domega)))
    if numtaps % 2 == 0:
        numtaps += 1  # odd, symmetric, type I
    half_len = numtaps // 2
    n = np.arange(-half_len, half_len + 1, dtype=np.float64)
    window = np.kaiser(numtaps, beta)
    x = 2.0 * cutoff_cycles * n
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.where(x == 0.0, 1.0, np.sin(np.pi * x) / (np.pi * x))
    unit = sinc * window
    kernel = unit * (up / float(np.sum(unit)))
    fc_nyquist = cutoff_cycles / 0.5
    return kernel.astype(np.float64), half_len, fc_nyquist


def resample(samples: np.ndarray, input_rate: int, output_rate: int,
             *, stopband_attenuation_db: float = 80.0,
             transition_band_hz: float = 500.0) -> tuple[np.ndarray, ResamplerSettings]:
    """Resample (n_frames, n_channels) shaped float data from input_rate to output_rate."""
    if input_rate <= 0 or output_rate <= 0:
        raise ValueError("sample rates must be positive")
    g = math.gcd(input_rate, output_rate)
    up = output_rate // g
    down = input_rate // g

    if up == 1 and down == 1:
        settings = ResamplerSettings(1, 1, 1.0, 0.0, stopband_attenuation_db, 0,
                                     input_rate, output_rate)
        return samples.astype(np.float64, copy=True), settings

    # The polyphase prototype lives on the L-times upsampled grid:
    # input_rate * L == output_rate * M.
    high_grid_rate = input_rate * up
    kernel, half_len, fc_nyquist = _design_kernel(
        up, down, input_rate, high_grid_rate,
        stopband_attenuation_db, transition_band_hz,
    )
    if samples.ndim == 1:
        samples = samples[:, None]
    n_frames, n_channels = samples.shape
    n_out = int(math.ceil(n_frames * up / down))

    # Block-wise overlap-add upfirdn. Each input block is zero-stuffed onto
    # the high grid and linearly convolved with the prototype kernel in a
    # fixed-size FFT; block responses are summed at their global high-rate
    # offsets. Bounded block size keeps peak memory independent of length.
    fft_size = 1 << int(np.ceil(np.log2(max(kernel.size * 4, 16384))))
    block_in = max(1, (fft_size - kernel.size + 1) // up)
    kernel_h = np.fft.rfft(kernel, fft_size)
    total_conv = n_frames * up + kernel.size - 1
    full = np.zeros((total_conv, n_channels), dtype=np.float64)

    for start in range(0, n_frames, block_in):
        chunk = samples[start : min(start + block_in, n_frames)]
        stuffed_len = chunk.shape[0] * up
        padded = np.zeros((fft_size, n_channels), dtype=np.float64)
        padded[0:stuffed_len:up, :] = chunk
        offset = start * up
        for ch in range(n_channels):
            conv = np.fft.irfft(np.fft.rfft(padded[:, ch]) * kernel_h, fft_size)
            span = min(fft_size, total_conv - offset)
            full[offset : offset + span, ch] += conv[:span]

    take = np.arange(n_out, dtype=np.int64) * down + half_len
    take = take[take < full.shape[0]]
    out = np.zeros((n_out, n_channels), dtype=np.float64)
    out[: take.size] = full[take]

    settings = ResamplerSettings(
        up=up,
        down=down,
        cutoff_fraction=fc_nyquist,
        transition_band_hz=float(transition_band_hz),
        stopband_attenuation_db=float(stopband_attenuation_db),
        filter_taps=int(kernel.size),
        input_rate=int(input_rate),
        output_rate=int(output_rate),
    )
    return out, settings
