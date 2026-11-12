"""Loudness and peak measurement component.

Implements an ITU-R BS.1770-4 style K-weighted integrated loudness with the
two-stage gating procedure, plus two deliberately distinct peak measures:

  * sample peak: the largest absolute value present in the decoded PCM;
  * true peak: the largest absolute value after 4x polyphase oversampling
    (resampler component), an interpolated estimate of inter-sample peaks.

Filter coefficients below are the documented BS.1770-4 K-weighting stages at
48 kHz (pre-filter high shelf + RLB high-pass). When a measurement is needed
at another rate the signal is first moved to 48 kHz with the resampler, so
the coefficients stay fixed and every result is traceable to one filter set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .resampler import ResamplerSettings, resample

# --- BS.1770-4 K-weighting coefficients, prototype rate 48 kHz -------------
_PRE_NUM = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_PRE_DEN = np.array([1.0, -1.69065929318241, 0.73248077421585])
_RLB_NUM = np.array([1.0, -2.0, 1.0])
_RLB_DEN = np.array([1.0, -1.99004745483398, 0.99007225036621])

WEIGHTING_RATE = 48000
BLOCK_S = 0.4          # BS.1770 gating block length
HOP_S = 0.1            # 75% overlap
ABS_GATE_LUFS = -70.0
SILENCE_PEAK_DBFS = -70.0
TP_UPSAMPLE = 4
TP_TRANSITION_HZ = 4000.0
TP_STOPBAND_DB = 80.0


class MeasurementError(ValueError):
    """Base error for material that cannot support a reliable measurement."""

    code = "measurement_error"


class SilenceError(MeasurementError):
    code = "silence"


class TooShortError(MeasurementError):
    code = "too_short"


def _iir_filter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Transposed direct-form II IIR, vectorised across channels."""
    b0, b1, b2 = b
    _, a1, a2 = a
    y = np.empty_like(x)
    z1 = np.zeros(x.shape[1], dtype=np.float64)
    z2 = np.zeros_like(z1)
    for n in range(x.shape[0]):
        yn = b0 * x[n] + z1
        z1 = b1 * x[n] - a1 * yn + z2
        z2 = b2 * x[n] - a2 * yn
        y[n] = yn
    return y


def k_weight(samples: np.ndarray) -> np.ndarray:
    y = _iir_filter(_PRE_NUM, _PRE_DEN, samples)
    return _iir_filter(_RLB_NUM, _RLB_DEN, y)


def _to_weighting_rate(samples: np.ndarray, rate: int):
    if rate == WEIGHTING_RATE:
        return samples, None
    out, settings = resample(samples, rate, WEIGHTING_RATE)
    return out, settings


@dataclass
class Measurement:
    sample_rate: int
    duration_s: float
    n_channels: int
    channel_layout: str
    integrated_lufs: float | None
    lra_lu: float | None
    sample_peak_dbfs: float
    true_peak_dbtp: float
    rms_dbfs: float
    n_gated_blocks: int
    n_total_blocks: int
    absolute_gated: bool
    relative_gated: bool
    finite: bool
    oversample: ResamplerSettings | None = None
    rate_adjustment: ResamplerSettings | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 6),
            "n_channels": self.n_channels,
            "channel_layout": self.channel_layout,
            "integrated_lufs": _round_or_none(self.integrated_lufs),
            "lra_lu": _round_or_none(self.lra_lu),
            "sample_peak_dbfs": round(self.sample_peak_dbfs, 4),
            "true_peak_dbtp": round(self.true_peak_dbtp, 4),
            "rms_dbfs": round(self.rms_dbfs, 4),
            "n_gated_blocks": self.n_gated_blocks,
            "n_total_blocks": self.n_total_blocks,
            "absolute_gated": self.absolute_gated,
            "relative_gated": self.relative_gated,
            "finite": self.finite,
            "oversampling": self.oversample.as_dict() if self.oversample else None,
            "rate_adjustment": self.rate_adjustment.as_dict() if self.rate_adjustment else None,
            "warnings": list(self.warnings),
        }


def _round_or_none(value, digits=4):
    return None if value is None else round(float(value), digits)


def _db(linear: float, floor: float = 1e-12) -> float:
    return 20.0 * math.log10(max(linear, floor))


def _channel_layout(n_channels: int) -> str:
    return {1: "mono", 2: "stereo", 6: "5.1"}.get(n_channels, f"{n_channels}ch")


def _block_loudness(weighted: np.ndarray, block_len: int, hop: int) -> np.ndarray:
    n = weighted.shape[0]
    if n < block_len:
        return np.empty(0, dtype=np.float64)
    n_blocks = 1 + (n - block_len) // hop
    values = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        start = i * hop
        block = weighted[start : start + block_len]
        mean_square = float(np.mean(block**2, axis=0).sum())
        values[i] = -0.691 + 10.0 * math.log10(max(mean_square, 1e-12))
    return values


def _lra(weighted: np.ndarray, block_len: int, hop: int, rel_gate: float) -> float | None:
    # BS.1770 LRA uses 3 s / 2 s blocks; with shorter material fall back to
    # the 400 ms blocks so the statistic remains defined.
    n = weighted.shape[0]
    long_len = min(int(round(3.0 * WEIGHTING_RATE)), n)
    long_len = max(long_len, block_len)
    blocks = _block_loudness(weighted, long_len, hop)
    kept = blocks[blocks > ABS_GATE_LUFS]
    kept = kept[kept > rel_gate]
    if kept.size < 2:
        return None
    p10 = float(np.percentile(kept, 10))
    p95 = float(np.percentile(kept, 95))
    return p95 - p10


def measure(samples: np.ndarray, sample_rate: int, *, allow_failure: bool = False) -> Measurement:
    """Measure an in-memory signal. Raises SilenceError/TooShortError unless
    allow_failure is True, in which case the integrated value is returned None
    and the condition is reported via warnings instead."""
    if samples.ndim == 1:
        samples = samples[:, None]
    if not np.all(np.isfinite(samples)):
        raise MeasurementError("non-finite samples cannot be measured")

    n_frames, n_channels = samples.shape
    duration = n_frames / float(sample_rate)
    warnings: list[str] = []

    sample_peak_linear = float(np.max(np.abs(samples))) if n_frames else 0.0
    sample_peak_db = _db(sample_peak_linear)
    rms_linear = math.sqrt(float(np.mean(samples**2))) if n_frames else 0.0
    rms_db = _db(rms_linear)

    # True peak: oversample 4x on the original (un-weighted) signal.
    up, tp_settings = resample(
        samples, sample_rate, sample_rate * TP_UPSAMPLE,
        stopband_attenuation_db=TP_STOPBAND_DB,
        transition_band_hz=min(TP_TRANSITION_HZ, sample_rate * TP_UPSAMPLE * 0.2),
    )
    true_peak_linear = float(np.max(np.abs(up)))
    true_peak_db = _db(true_peak_linear)

    if sample_peak_linear <= 10.0 ** (SILENCE_PEAK_DBFS / 20.0):
        if not allow_failure:
            raise SilenceError(
                f"digital silence: sample peak {sample_peak_db:.1f} dBFS <= "
                f"{SILENCE_PEAK_DBFS} dBFS"
            )
        warnings.append("digital_silence")

    if duration < BLOCK_S:
        if not allow_failure:
            raise TooShortError(
                f"duration {duration:.3f}s is shorter than one {BLOCK_S:.1f}s "
                "gating block; integrated loudness is not defined"
            )
        warnings.append("shorter_than_one_gating_block")

    weighted, rate_settings = _to_weighting_rate(samples, sample_rate)
    weighted = k_weight(weighted)
    block_len = int(round(BLOCK_S * WEIGHTING_RATE))
    hop = int(round(HOP_S * WEIGHTING_RATE))
    block_loudness = _block_loudness(weighted, block_len, hop)

    integrated: float | None = None
    lra: float | None = None
    n_gated = 0
    abs_gated = rel_gated = False
    rel_gate = float("-inf")

    if block_loudness.size:
        above_abs = block_loudness[block_loudness > ABS_GATE_LUFS]
        abs_gated = above_abs.size > 0
        if abs_gated:
            rel_gate = float(np.mean(above_abs)) - 10.0
            gated = above_abs[above_abs > rel_gate]
            rel_gated = gated.size > 0
            if rel_gated:
                integrated = float(np.mean(gated))
                n_gated = int(gated.size)
                lra = _lra(weighted, block_len, hop, rel_gate)
        if integrated is None and not allow_failure:
            raise TooShortError("no blocks survived the BS.1770 gating stages")
        if integrated is None:
            warnings.append("no_blocks_survived_gating")

    return Measurement(
        sample_rate=int(sample_rate),
        duration_s=duration,
        n_channels=int(n_channels),
        channel_layout=_channel_layout(n_channels),
        integrated_lufs=integrated,
        lra_lu=lra,
        sample_peak_dbfs=sample_peak_db,
        true_peak_dbtp=true_peak_db,
        rms_dbfs=rms_db,
        n_gated_blocks=n_gated,
        n_total_blocks=int(block_loudness.size),
        absolute_gated=abs_gated,
        relative_gated=rel_gated,
        finite=True,
        oversample=tp_settings,
        rate_adjustment=rate_settings,
        warnings=warnings,
    )


def measure_file(path: str) -> Measurement:
    """Post-export re-measurement: read the actual exported file back from
    disk and measure it, rather than trusting filter parameters."""
    from .wavfile import read_wav

    audio = read_wav(path)
    return measure(audio.samples, audio.sample_rate)
