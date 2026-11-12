"""Loudness production chain.

Stages, in fixed order:
  1. decode/validate input (wavfile)
  2. layout decision (channels; mono->stereo duplication rule)
  3. sample-rate adjustment (resampler)
  4. source measurement (loudness; silence / too-short reported separately)
  5. finite static gain toward the target integrated loudness
  6. 4x oversampled look-ahead true-peak limiter when the true-peak ceiling
     cannot be met by gain alone
  7. TPDF dither for 16-bit export, quantise and write WAV
  8. re-read the exported file and re-measure it — the exported measurement
     is the only evidence used for the final verdict; filter parameters are not.

When the loudness target and true-peak ceiling cannot both be reached, the
chain still emits a bounded signal and reports `limited` / `target_missed`
with the remaining gap. An infinite or NaN gain is never produced or applied.

Optimized computation order (cache=... passed, see measure_cache):
  * Stage 4 is served from the content-addressed source-measurement cache
    when the same input was already measured under the same settings and
    channel interpretation; the gating/oversampling work is skipped only
    then. The stage order is unchanged.
  * Stage 6 computes the oversampled true-peak check on the gained signal
    via loudness.true_peak_check instead of a full measure(): the chain
    only ever consumes the after-gain true peak here, never the after-gain
    integrated loudness, so the K-weighted gating for it is dropped. The
    true-peak check itself always runs — a cache hit never skips it.
  * Stage 8 is identical in both orders: the exported file is always
    re-decoded and re-measured. Cached or derived numbers are never used
    as the post-processing re-measurement.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import telemetry
from .channels import LayoutDecision, apply_layout, channel_balance
from .loudness import (
    TP_UPSAMPLE,
    Measurement,
    SilenceError,
    TooShortError,
    measure,
    true_peak_check,
)
from .measure_cache import MeasureCache
from .resampler import ResamplerSettings, resample
from .wavfile import AudioFile, write_wav

MAX_GAIN_DB = 60.0          # hard finite-gain guard
MIN_GAIN_DB = -60.0
LIMITER_ATTACK_MS = 1.0
LIMITER_RELEASE_MS = 100.0
LIMITER_LOOKAHEAD_MS = 2.0
LIMITER_ITERATIONS = 4
LIMITER_HEADROOM_DB = 1.0  # margin so re-oversampling after decimation stays below ceiling
DITHER_AMPLITUDE = 2.0      # TPDF peak-to-peak of 2 LSB at the quantiser


class ProcessingError(ValueError):
    pass


@dataclass(frozen=True)
class Target:
    integrated_lufs: float
    true_peak_dbtp: float
    sample_rate: int
    layout: str
    bit_depth: int = 24

    @classmethod
    def from_dict(cls, raw: dict) -> "Target":
        try:
            target = cls(
                integrated_lufs=float(raw["integrated_lufs"]),
                true_peak_dbtp=float(raw["true_peak_dbtp"]),
                sample_rate=int(raw["sample_rate"]),
                layout=str(raw["layout"]),
                bit_depth=int(raw.get("bit_depth", 24)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProcessingError(f"invalid target descriptor: {exc}") from exc
        if not (-60.0 <= target.integrated_lufs <= 0.0):
            raise ProcessingError("target integrated loudness must lie in [-60, 0] LUFS")
        if not (-20.0 <= target.true_peak_dbtp <= 0.0):
            raise ProcessingError("true-peak ceiling must lie in [-20, 0] dBTP")
        if target.layout not in ("mono", "stereo"):
            raise ProcessingError("target layout must be 'mono' or 'stereo'")
        if target.bit_depth not in (16, 24, 32):
            raise ProcessingError("bit depth must be 16, 24 or 32")
        if target.sample_rate < 8000 or target.sample_rate > 192000:
            raise ProcessingError("target sample rate out of range")
        return target

    def as_dict(self) -> dict:
        return {
            "integrated_lufs": self.integrated_lufs,
            "true_peak_dbtp": self.true_peak_dbtp,
            "sample_rate": self.sample_rate,
            "layout": self.layout,
            "bit_depth": self.bit_depth,
        }


@dataclass
class LimiterTrace:
    enabled: bool
    threshold_dbtp: float
    gain_reduction_db: float
    iterations: int
    attack_ms: float
    release_ms: float
    lookahead_ms: float
    oversample: int

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "threshold_dbtp": round(self.threshold_dbtp, 4),
            "working_threshold_dbtp": round(self.threshold_dbtp - LIMITER_HEADROOM_DB, 4),
            "headroom_db": LIMITER_HEADROOM_DB,
            "max_gain_reduction_db": round(self.gain_reduction_db, 4),
            "iterations": self.iterations,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "lookahead_ms": self.lookahead_ms,
            "oversample": self.oversample,
        }


def _db_to_gain(db: float) -> float:
    gain = 10.0 ** (db / 20.0)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ProcessingError(f"non-finite or non-positive gain derived: {db} dB")
    return gain


def static_gain_db(source_lufs: float | None, target_lufs: float) -> float:
    """Finite-only gain. Missing source loudness (silence) yields zero gain."""
    if source_lufs is None or not math.isfinite(source_lufs):
        return 0.0
    gain = target_lufs - source_lufs
    return float(np.clip(gain, MIN_GAIN_DB, MAX_GAIN_DB))


def _smooth_envelope(hard_gain: np.ndarray, rate: int, lookahead: int) -> np.ndarray:
    """Smooth a per-frame hard-limiter gain into a click-free envelope.

    The hard gain already guarantees no frame exceeds the threshold; smoothing
    only shapes the transitions. A one-pole attack/release filter is run
    causally and the result is advanced by the look-ahead so gain reduction is
    fully in place by the time each peak arrives. Taking the element-wise
    minimum against the hard gain afterwards preserves the peak guarantee.
    """
    attack_coef = math.exp(-1.0 / (max(LIMITER_ATTACK_MS, 0.05) * 1e-3 * rate))
    release_coef = math.exp(-1.0 / (LIMITER_RELEASE_MS * 1e-3 * rate))
    smoothed = np.empty_like(hard_gain)
    previous = 1.0
    for n in range(hard_gain.size):
        target = hard_gain[n]
        coef = attack_coef if target < previous else release_coef
        previous = coef * previous + (1.0 - coef) * target
        smoothed[n] = previous
    if lookahead > 0:
        smoothed = np.concatenate([
            smoothed[lookahead:],
            np.full(lookahead, smoothed[-1]),
        ])
    return np.minimum(smoothed, hard_gain)


def apply_limiter(samples: np.ndarray, rate: int, ceiling_dbtp: float
                  ) -> tuple[np.ndarray, LimiterTrace, ResamplerSettings]:
    """True-peak limiter with a closed-loop verification against the same
    oversampling measurement used after export.

    Each iteration oversamples 4x, applies an attack/release-smoothed gain
    envelope with a hard safety floor, decimates back to the input grid and
    re-oversamples the decimated result to measure the peak the post-export
    measurement will actually see. Iteration stops when that verified peak is
    below the working threshold (ceiling minus a fixed headroom). Filter
    parameters are never treated as proof — the re-measured signal is.
    """
    working_ceiling = ceiling_dbtp - LIMITER_HEADROOM_DB
    threshold = _db_to_gain(working_ceiling)
    up_rate = rate * TP_UPSAMPLE
    iterations = 0
    total_reduction = 0.0
    current = samples.copy()
    up_settings: ResamplerSettings | None = None
    telemetry.event("limiter_runs")

    for iterations in range(1, LIMITER_ITERATIONS + 1):
        up, up_settings = resample(
            current, rate, up_rate,
            stopband_attenuation_db=80.0,
            transition_band_hz=min(4000.0, rate * 0.4),
        )
        level = np.max(np.abs(up), axis=1)
        telemetry.event("true_peak_checks")
        if float(np.max(level)) <= threshold * (1.0 + 1e-9):
            break
        desired = np.ones_like(level)
        over = level > threshold
        desired[over] = threshold / level[over]
        lookahead = int(round(LIMITER_LOOKAHEAD_MS * 1e-3 * up_rate))
        envelope = _smooth_envelope(desired, up_rate, lookahead)
        reduction = float(20.0 * math.log10(max(np.min(envelope), 1e-12)))
        total_reduction = min(total_reduction, reduction)
        limited_up = up * envelope[:, None]
        decimated = limited_up[::TP_UPSAMPLE, :][: samples.shape[0]]
        # Verify through the exact independent re-oversampling used at export.
        # If decimation/reconstruction re-introduces an overshoot, apply the
        # measured corrective gain directly to the decimated signal and loop.
        verify, _ = resample(
            decimated, rate, up_rate,
            stopband_attenuation_db=80.0,
            transition_band_hz=min(4000.0, rate * 0.4),
        )
        verify_peak = float(np.max(np.abs(verify)))
        telemetry.event("true_peak_checks")
        if verify_peak <= threshold * (1.0 + 1e-9):
            current = decimated
            break
        corrective = min(1.0, threshold / verify_peak)
        total_reduction = min(
            total_reduction,
            float(20.0 * math.log10(corrective)),
        )
        current = decimated * corrective

    trace = LimiterTrace(
        enabled=True,
        threshold_dbtp=ceiling_dbtp,
        gain_reduction_db=abs(total_reduction),
        iterations=max(iterations, 1),
        attack_ms=LIMITER_ATTACK_MS,
        release_ms=LIMITER_RELEASE_MS,
        lookahead_ms=LIMITER_LOOKAHEAD_MS,
        oversample=TP_UPSAMPLE,
    )
    return current, trace, up_settings


def _tpdf_dither(shape: tuple[int, int], lsb: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.uniform(-1.0, 1.0, size=shape)
            + rng.uniform(-1.0, 1.0, size=shape)) * (lsb / DITHER_AMPLITUDE)


@dataclass
class ChainResult:
    status: str
    reason: str | None
    source: Measurement | None
    output: Measurement | None
    target: Target
    layout: LayoutDecision | None
    rate_settings: ResamplerSettings | None
    limiter: LimiterTrace | None
    static_gain_db: float
    quantisation: dict
    balance: dict
    unmet: list[str] = field(default_factory=list)
    output_path: str | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "target": self.target.as_dict(),
            "source_measurement": self.source.as_dict() if self.source else None,
            "output_measurement": self.output.as_dict() if self.output else None,
            "layout_decision": self.layout.as_dict() if self.layout else None,
            "resampling": self.rate_settings.as_dict() if self.rate_settings else None,
            "static_gain_db": round(self.static_gain_db, 4),
            "limiter": self.limiter.as_dict() if self.limiter else None,
            "quantisation": self.quantisation,
            "channel_balance": self.balance,
            "unmet_conditions": self.unmet,
            "output_path": self.output_path,
        }


def run_chain(audio: AudioFile, target: Target, output_path: str,
              rng: np.random.Generator | None = None, *,
              cache: MeasureCache | None = None,
              input_sha256: str | None = None) -> ChainResult:
    """Run the production chain.

    With ``cache=None`` the original computation order runs: every stage
    measures fresh. With a MeasureCache and the input digest, stage 4 reuses
    a cached source measurement when the key (input digest, target rate,
    channel interpretation, settings fingerprint) matches, and stage 6 uses
    the true-peak-only check; stage 8 re-measurement is unchanged.
    """
    rng = rng or np.random.default_rng(0xA17C)
    samples = audio.samples.astype(np.float64, copy=True)
    balance = channel_balance(samples)

    # Stage 2: layout.
    samples, layout = apply_layout(samples, target.layout)

    # Stage 3: sample rate.
    samples, rate_settings = resample(samples, audio.sample_rate, target.sample_rate)

    # Stage 4: source measurement on the rate/layout-adjusted working signal.
    source: Measurement | None = None
    if cache is not None:
        if not input_sha256:
            raise ProcessingError(
                "input_sha256 is required when a measurement cache is used"
            )
        key = cache.source_key(
            input_sha256=input_sha256,
            sample_rate=target.sample_rate,
            layout_action=layout.action,
            channels_in=audio.n_channels,
            channels_out=samples.shape[1],
        )
        source = cache.get(key)
        if source is None:
            source = measure(samples, target.sample_rate, allow_failure=True)
            cache.put(key, source)
    else:
        source = measure(samples, target.sample_rate, allow_failure=True)
    unmet: list[str] = []

    if source.warnings:
        reason = ",".join(source.warnings)
        return ChainResult(
            status="rejected",
            reason=reason,
            source=source,
            output=None,
            target=target,
            layout=layout,
            rate_settings=rate_settings,
            limiter=None,
            static_gain_db=0.0,
            quantisation={},
            balance=balance,
            unmet=[reason],
        )

    # Stage 5: finite static gain (never infinite).
    gain_db = static_gain_db(source.integrated_lufs, target.integrated_lufs)
    gain_linear = _db_to_gain(gain_db)
    if not math.isfinite(gain_linear):
        raise ProcessingError("finite-gain guard tripped")
    processed = samples * gain_linear

    # Stage 6: true-peak limiter when the ceiling is not met by gain alone.
    # The optimized order uses the true-peak-only check on the gained signal;
    # the original order runs a full measurement. Both evaluate the same
    # oversampled peak on the actual processed samples.
    limiter: LimiterTrace | None = None
    if cache is not None:
        _, after_gain_tp = true_peak_check(processed, target.sample_rate)
    else:
        after_gain = measure(processed, target.sample_rate, allow_failure=True)
        after_gain_tp = after_gain.true_peak_dbtp
    if after_gain_tp > target.true_peak_dbtp + 0.001:
        processed, limiter, _ = apply_limiter(
            processed, target.sample_rate, target.true_peak_dbtp
        )

    if not np.all(np.isfinite(processed)):
        raise ProcessingError("non-finite samples produced by the chain")

    # Stage 7: dither and export.
    lsb = 2.0 ** (-(target.bit_depth - 1))
    quant = {
        "bit_depth": target.bit_depth,
        "lsb_linear": lsb,
        "dither": "tpdf_2lsb" if target.bit_depth == 16 else "none",
    }
    if target.bit_depth == 16:
        processed = np.clip(processed + _tpdf_dither(processed.shape, lsb, rng), -1.0, 1.0)

    write_wav(output_path, processed, target.sample_rate, target.bit_depth)

    # Stage 8: re-measure the exported file itself.
    exported = measure_file_safe(output_path)
    if exported is None:
        raise ProcessingError("post-export re-measurement failed")

    loudness_gap = target.integrated_lufs - (exported.integrated_lufs or 0.0)
    if abs(loudness_gap) > 0.5:
        unmet.append(
            f"integrated loudness {exported.integrated_lufs:.2f} LUFS differs from "
            f"target {target.integrated_lufs:.2f} LUFS by {loudness_gap:+.2f} dB"
        )
    # A peak "above" a dBTP/dBFS ceiling means numerically greater (less negative).
    if exported.true_peak_dbtp > target.true_peak_dbtp + 0.05:
        unmet.append(
            f"true peak {exported.true_peak_dbtp:.2f} dBTP remains above ceiling "
            f"{target.true_peak_dbtp:.2f} dBTP after limiting"
        )
    if exported.sample_peak_dbfs > target.true_peak_dbtp + 0.05:
        unmet.append(
            f"sample peak {exported.sample_peak_dbfs:.2f} dBFS above requested ceiling"
        )

    if unmet:
        status = "target_missed"
    elif limiter is not None and limiter.gain_reduction_db > 0.05:
        status = "limited"
    else:
        status = "matched"

    return ChainResult(
        status=status,
        reason=None,
        source=source,
        output=exported,
        target=target,
        layout=layout,
        rate_settings=rate_settings,
        limiter=limiter,
        static_gain_db=gain_db,
        quantisation=quant,
        balance=balance,
        unmet=unmet,
        output_path=output_path,
    )


def measure_file_safe(path: str) -> Measurement | None:
    try:
        from .loudness import measure_file

        return measure_file(path)
    except (SilenceError, TooShortError, ValueError):
        return None
