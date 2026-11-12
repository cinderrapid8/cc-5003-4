"""Runtime version and algorithm-setting provenance.

Every task record embeds the exact component versions and filter parameters
that ran, so a result can be re-computed later with the same settings.
"""
from __future__ import annotations

import platform
import sys

import numpy as np

from . import loudness, resampler
from .channels import DUPLICATION_GAIN_DB
from .measure_cache import DEFAULT_MAX_ENTRIES, SCHEMA_VERSION
from .processing import (
    LIMITER_ATTACK_MS,
    LIMITER_ITERATIONS,
    LIMITER_LOOKAHEAD_MS,
    LIMITER_RELEASE_MS,
    MAX_GAIN_DB,
    MIN_GAIN_DB,
)

SERVICE_VERSION = "1.0.0"
LOUDNESS_ALGORITHM = "BS.1770-4-style K-weighting, two-stage gating (in-house)"
TRUE_PEAK_ALGORITHM = (
    f"{loudness.TP_UPSAMPLE}x polyphase oversampled inter-sample peak "
    f"(Kaiser low-pass, {loudness.TP_STOPBAND_DB} dB stopband)"
)


def runtime_provenance() -> dict:
    return {
        "service_version": SERVICE_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "components": {
            "loudness": {
                "algorithm": LOUDNESS_ALGORITHM,
                "weighting_rate_hz": loudness.WEIGHTING_RATE,
                "block_s": loudness.BLOCK_S,
                "hop_s": loudness.HOP_S,
                "absolute_gate_lufs": loudness.ABS_GATE_LUFS,
                "silence_peak_dbfs": loudness.SILENCE_PEAK_DBFS,
                "pre_filter_num": loudness._PRE_NUM.tolist(),
                "pre_filter_den": loudness._PRE_DEN.tolist(),
                "rlb_num": loudness._RLB_NUM.tolist(),
                "rlb_den": loudness._RLB_DEN.tolist(),
                "true_peak": TRUE_PEAK_ALGORITHM,
                "tp_oversample": loudness.TP_UPSAMPLE,
                "tp_transition_hz": loudness.TP_TRANSITION_HZ,
            },
            "resampler": {
                "module": resampler.__name__,
                "method": "rational polyphase Kaiser-windowed sinc, FFT convolution",
                "default_stopband_db": 80.0,
            },
            "channel_rule": {
                "mono_to_stereo": "bit-identical duplication L=R=mono",
                "duplication_gain_db": round(float(DUPLICATION_GAIN_DB), 6),
            },
            "gain_guard": {
                "min_gain_db": MIN_GAIN_DB,
                "max_gain_db": MAX_GAIN_DB,
            },
            "limiter": {
                "attack_ms": LIMITER_ATTACK_MS,
                "release_ms": LIMITER_RELEASE_MS,
                "lookahead_ms": LIMITER_LOOKAHEAD_MS,
                "max_iterations": LIMITER_ITERATIONS,
                "oversample": loudness.TP_UPSAMPLE,
            },
            "measurement_cache": {
                "schema": SCHEMA_VERSION,
                "max_entries": DEFAULT_MAX_ENTRIES,
                "key": "input sha256 | target sample rate | layout action | "
                       "channel counts | settings fingerprint",
                "boundaries": [
                    "source measurement only; the exported file is always "
                    "re-decoded and re-measured",
                    "the post-gain true-peak check always runs on the "
                    "processed signal, cache hit or not",
                    "silence/too-short rejections are cached with warnings",
                    "limiter engagement and output are never cached",
                ],
            },
        },
        "executable": sys.executable,
    }
