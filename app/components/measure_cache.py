"""Content-addressed cache for source (working-signal) loudness measurements.

Repeatedly adjusting targets on the same uploaded material re-measures the
same working signal once per job. The measurement depends only on:

  * the input bytes — addressed by their SHA-256 digest;
  * the measurement settings — addressed by a fingerprint of every constant
    and filter coefficient that influences the result;
  * the channel interpretation — the layout decision (e.g. the
    mono->stereo duplication rule) and the channel counts, because the
    duplication rule changes integrated loudness by +3.01 dB;
  * the working sample rate after rate adjustment.

Cache boundaries (enforced here and in processing.run_chain):

  * Only the source measurement of the working signal is cached. The
    post-export re-measurement always re-decodes the exported file and
    measures it fresh — a cached source measurement is never substituted
    for the re-measurement of processed material.
  * A cache hit never skips the true-peak check: the limiter decision is
    always made from an oversampled peak measured on the actually gained
    signal of the current job.
  * Silence and too-short clips are cached *with their warnings*: a hit
    still rejects the job, it just does not re-run the gating to find out.
  * When the limiter actually engages, its iterations, envelope and output
    are never cached; only the pre-gain source measurement is reused.

Entries are copied in and out so callers can never mutate shared state.
"""
from __future__ import annotations

import hashlib
import inspect
from collections import OrderedDict
from dataclasses import replace

from . import loudness, resampler, telemetry
from .loudness import Measurement

SCHEMA_VERSION = "v1"
DEFAULT_MAX_ENTRIES = 128


def settings_fingerprint() -> str:
    """Fingerprint of every measurement-relevant constant and coefficient.

    Computed fresh on each call so a changed constant immediately yields a
    different fingerprint and therefore unreachable (invalidated) entries.
    """
    h = hashlib.sha256()
    h.update(f"bs1770-settings/{SCHEMA_VERSION}\n".encode())
    for name in (
        "WEIGHTING_RATE",
        "BLOCK_S",
        "HOP_S",
        "ABS_GATE_LUFS",
        "SILENCE_PEAK_DBFS",
        "TP_UPSAMPLE",
        "TP_TRANSITION_HZ",
        "TP_STOPBAND_DB",
    ):
        h.update(f"{name}={getattr(loudness, name)!r}\n".encode())
    for name in ("_PRE_NUM", "_PRE_DEN", "_RLB_NUM", "_RLB_DEN"):
        h.update(name.encode())
        h.update(getattr(loudness, name).tobytes())
    params = inspect.signature(resampler.resample).parameters
    defaults = (
        params["stopband_attenuation_db"].default,
        params["transition_band_hz"].default,
    )
    h.update(f"resampler_defaults={defaults!r}\n".encode())
    return h.hexdigest()


def _copy(measurement: Measurement) -> Measurement:
    return replace(measurement, warnings=list(measurement.warnings))


class MeasureCache:
    """Bounded LRU store of source measurements keyed by content + settings."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = int(max_entries)
        self._entries: "OrderedDict[str, Measurement]" = OrderedDict()

    @staticmethod
    def source_key(
        *,
        input_sha256: str,
        sample_rate: int,
        layout_action: str,
        channels_in: int,
        channels_out: int,
    ) -> str:
        return "|".join(
            [
                SCHEMA_VERSION,
                str(input_sha256),
                f"rate={int(sample_rate)}",
                f"layout={layout_action}",
                f"channels={int(channels_in)}>{int(channels_out)}",
                f"settings={settings_fingerprint()}",
            ]
        )

    def get(self, key: str) -> Measurement | None:
        measurement = self._entries.get(key)
        if measurement is None:
            telemetry.event("cache_misses")
            return None
        self._entries.move_to_end(key)
        telemetry.event("cache_hits")
        return _copy(measurement)

    def put(self, key: str, measurement: Measurement) -> None:
        self._entries[key] = _copy(measurement)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
