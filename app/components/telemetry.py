"""Execution-evidence counters for the loudness pipeline.

The components emit events (decode, measure, true-peak check, resample,
limiter run, cache hit/miss) into a thread-local recorder. Recording is
opt-in: without an active ``record()`` block, ``event()`` is a no-op, so
production paths pay only a thread-local lookup.

The counters are the cost decomposition used by the benchmark and are
attached to every service job as execution evidence, so decode counts and
measured peaks can be traced to the run that produced them.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass

FIELDS = (
    "decodes",
    "measures",
    "true_peak_checks",
    "resamples",
    "limiter_runs",
    "cache_hits",
    "cache_misses",
)


@dataclass
class Counters:
    decodes: int = 0          # WAV decodes (input and exported re-measurement)
    measures: int = 0         # full measure() calls (K-weighting + gating + peaks)
    true_peak_checks: int = 0 # every oversampled peak evaluation, incl. inside measure()
    resamples: int = 0        # polyphase resampler invocations
    limiter_runs: int = 0     # true-peak limiter engagements
    cache_hits: int = 0       # source-measurement cache hits
    cache_misses: int = 0     # source-measurement cache lookups that missed

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in FIELDS}


_local = threading.local()


@contextmanager
def record(counters: Counters | None = None):
    """Collect events in the current thread into ``counters`` (or a fresh one)."""
    counters = counters if counters is not None else Counters()
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = []
        _local.stack = stack
    stack.append(counters)
    try:
        yield counters
    finally:
        stack.pop()


def event(name: str, n: int = 1) -> None:
    stack = getattr(_local, "stack", None)
    if stack:
        counters = stack[-1]
        setattr(counters, name, getattr(counters, name) + n)
