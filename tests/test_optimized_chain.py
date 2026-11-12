"""Baseline (original order) vs. optimized (cached) chain equivalence.

The optimized order may only differ in the execution counters — every
recorded statistic must be identical. Decode counts are evidence and must
NOT drop: the input is decoded per job and the exported file is always
re-decoded for the re-measurement.
"""
import hashlib
import os

import pytest

from app.components import telemetry
from app.components.loudness import measure, true_peak_check
from app.components.measure_cache import MeasureCache
from app.components.processing import ProcessingError, Target, run_chain
from app.components.wavfile import read_wav, write_wav
from app.service import np_rng_seeded
from scripts.benchmark_loudness import compare_values
from tests.signals import (
    calibrated_speech_like,
    narrow_impulse,
    short_click,
    silence,
)


def _target(**over):
    base = dict(
        integrated_lufs=-23.0,
        true_peak_dbtp=-2.0,
        sample_rate=48000,
        layout="mono",
        bit_depth=24,
    )
    base.update(over)
    return Target(**base)


def _write_input(sig, rate, directory):
    path = os.path.join(directory, "in.wav")
    write_wav(path, sig, rate, 24)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return path, digest


def _run(input_path, digest, target, out_path, cache, seed="seed"):
    with telemetry.record() as counters:
        audio = read_wav(input_path)
        result = run_chain(
            audio, target, out_path,
            rng=np_rng_seeded(seed),
            cache=cache,
            input_sha256=digest if cache is not None else None,
        )
    return result, counters


def _chain_dict(result):
    d = result.as_dict()
    d.pop("output_path", None)
    return d


def _assert_equal_chains(baseline, optimized):
    failures = []
    compare_values(_chain_dict(baseline), _chain_dict(optimized), "chain", failures)
    assert failures == []


def test_repeated_targets_match_baseline_and_save_work(tmp_path):
    sig, rate = calibrated_speech_like(seconds=3)
    input_path, digest = _write_input(sig, rate, str(tmp_path))
    targets = [
        _target(integrated_lufs=-23.0),
        _target(integrated_lufs=-26.0),
        _target(integrated_lufs=-20.0, true_peak_dbtp=-4.0),
    ]

    baselines = []
    for i, target in enumerate(targets):
        result, counters = _run(input_path, digest, target,
                                str(tmp_path / f"base{i}.wav"), None, seed=f"t{i}")
        baselines.append((result, counters))

    cache = MeasureCache()
    optimized = []
    for i, target in enumerate(targets):
        result, counters = _run(input_path, digest, target,
                                str(tmp_path / f"opt{i}.wav"), cache, seed=f"t{i}")
        optimized.append((result, counters))

    for (base_result, _), (opt_result, _) in zip(baselines, optimized):
        _assert_equal_chains(base_result, opt_result)

    # Every baseline run pays three full measurements: source, post-gain,
    # post-export re-measurement.
    for _, counters in baselines:
        assert counters.measures == 3
        assert counters.decodes == 2
        assert counters.cache_hits == 0 and counters.cache_misses == 0

    # First optimized run measures the source once (miss); later runs hit.
    assert optimized[0][1].cache_misses == 1
    assert optimized[0][1].measures == 2
    for _, counters in optimized[1:]:
        assert counters.cache_hits == 1
        assert counters.measures == 1          # only the export re-measurement
        assert counters.true_peak_checks >= 1  # TP check never skipped on a hit
    # Decode counts are evidence, not savings: input + export every run.
    for _, counters in optimized:
        assert counters.decodes == 2


def test_limiter_still_engages_on_cache_hit(tmp_path):
    sig, rate = narrow_impulse(2.0)
    input_path, digest = _write_input(sig, rate, str(tmp_path))
    target = _target(true_peak_dbtp=-3.0)

    baseline, _ = _run(input_path, digest, target, str(tmp_path / "b.wav"), None)
    assert baseline.limiter is not None and baseline.limiter.gain_reduction_db > 0

    cache = MeasureCache()
    opt1, c1 = _run(input_path, digest, target, str(tmp_path / "o1.wav"), cache)
    opt2, c2 = _run(input_path, digest, target, str(tmp_path / "o2.wav"), cache)

    _assert_equal_chains(baseline, opt1)
    _assert_equal_chains(baseline, opt2)
    # Limiting actually happened in both optimized runs; the cache hit did
    # not skip the limiter or the true-peak check.
    assert c1.limiter_runs == 1 and c2.limiter_runs == 1
    assert c2.cache_hits == 1
    assert c2.true_peak_checks >= 1
    assert opt2.limiter is not None


def test_silence_rejection_is_cached_with_warnings(tmp_path):
    sig, rate = silence(1.5)
    input_path, digest = _write_input(sig, rate, str(tmp_path))
    target = _target()

    baseline, base_counters = _run(input_path, digest, target,
                                   str(tmp_path / "b.wav"), None)
    assert baseline.status == "rejected"
    assert "digital_silence" in baseline.reason
    assert base_counters.measures == 1

    cache = MeasureCache()
    opt1, c1 = _run(input_path, digest, target, str(tmp_path / "o1.wav"), cache)
    opt2, c2 = _run(input_path, digest, target, str(tmp_path / "o2.wav"), cache)
    for result in (opt1, opt2):
        assert result.status == "rejected"
        assert result.reason == baseline.reason
        assert "digital_silence" in result.source.warnings
    # Second run reuses the cached rejection: no re-measurement, no export.
    assert c2.cache_hits == 1
    assert c2.measures == 0
    assert c2.decodes == 1  # input only; a rejected job exports nothing
    _assert_equal_chains(baseline, opt2)


def test_short_clip_rejection_is_cached(tmp_path):
    sig, rate = short_click()
    input_path, digest = _write_input(sig, rate, str(tmp_path))
    target = _target()
    cache = MeasureCache()
    opt1, _ = _run(input_path, digest, target, str(tmp_path / "o1.wav"), cache)
    opt2, c2 = _run(input_path, digest, target, str(tmp_path / "o2.wav"), cache)
    assert opt1.status == opt2.status == "rejected"
    assert "shorter_than_one_gating_block" in opt2.reason
    assert c2.cache_hits == 1
    assert c2.measures == 0


def test_cache_key_isolates_rate_and_layout(tmp_path):
    sig, rate = calibrated_speech_like(seconds=2)
    input_path, digest = _write_input(sig, rate, str(tmp_path))
    cache = MeasureCache()
    runs = [
        (_target(), "a"),                                          # miss
        (_target(sample_rate=44100), "b"),                         # miss: rate
        (_target(layout="stereo"), "c"),                           # miss: layout
        (_target(), "d"),                                          # hit
    ]
    counters = []
    for target, name in runs:
        _, c = _run(input_path, digest, target, str(tmp_path / f"{name}.wav"), cache,
                    seed=name)
        counters.append(c)
    assert [c.cache_misses for c in counters] == [1, 1, 1, 0]
    assert counters[3].cache_hits == 1


def test_export_remeasurement_is_never_cached(tmp_path):
    sig, rate = calibrated_speech_like(seconds=2)
    input_path, digest = _write_input(sig, rate, str(tmp_path))
    target = _target()
    cache = MeasureCache()
    _, c1 = _run(input_path, digest, target, str(tmp_path / "o1.wav"), cache)
    r2, c2 = _run(input_path, digest, target, str(tmp_path / "o2.wav"), cache)
    # Even on a source-cache hit the exported file is decoded and measured.
    assert c2.cache_hits == 1
    assert c2.decodes == 2
    assert c2.measures == 1  # the export re-measurement itself
    assert r2.output is not None


def test_true_peak_check_matches_full_measure_exactly():
    for factory in (lambda: calibrated_speech_like(seconds=2),
                    lambda: narrow_impulse(1.0)):
        sig, rate = factory()
        gained = sig * 1.7
        sample_db, tp_db = true_peak_check(gained, rate)
        full = measure(gained, rate, allow_failure=True)
        assert tp_db == full.true_peak_dbtp
        assert sample_db == full.sample_peak_dbfs


def test_cache_requires_input_digest(tmp_path):
    sig, rate = calibrated_speech_like(seconds=1)
    input_path, _ = _write_input(sig, rate, str(tmp_path))
    audio = read_wav(input_path)
    with pytest.raises(ProcessingError):
        run_chain(audio, _target(), str(tmp_path / "x.wav"), cache=MeasureCache())
