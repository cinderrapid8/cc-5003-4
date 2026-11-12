from app.components import loudness, telemetry
from app.components.loudness import measure
from app.components.measure_cache import MeasureCache, settings_fingerprint
from tests.signals import narrow_impulse, silence


def _key(**over):
    base = dict(
        input_sha256="a" * 64,
        sample_rate=48000,
        layout_action="none",
        channels_in=1,
        channels_out=1,
    )
    base.update(over)
    return MeasureCache.source_key(**base)


def _measurement():
    x, rate = narrow_impulse(1.0)
    return measure(x, rate, allow_failure=True)


def test_fingerprint_is_stable_and_sensitive_to_settings(monkeypatch):
    fingerprint = settings_fingerprint()
    assert fingerprint == settings_fingerprint()
    monkeypatch.setattr(loudness, "ABS_GATE_LUFS", -60.0)
    assert settings_fingerprint() != fingerprint


def test_key_depends_on_digest_rate_layout_and_channels():
    base = _key()
    assert base == _key()  # identical inputs -> identical key
    assert base != _key(input_sha256="b" * 64)
    assert base != _key(sample_rate=44100)
    assert base != _key(layout_action="duplicate_mono_to_dual_mono", channels_out=2)
    assert base != _key(channels_in=2, channels_out=2)


def test_put_get_roundtrip_returns_independent_copy():
    cache = MeasureCache()
    key = _key()
    measurement = _measurement()
    cache.put(key, measurement)
    got = cache.get(key)
    assert got is not measurement
    assert got.as_dict() == measurement.as_dict()
    got.warnings.append("mutated by caller")
    assert cache.get(key).warnings == measurement.warnings


def test_lru_eviction_drops_oldest():
    cache = MeasureCache(max_entries=2)
    m = _measurement()
    cache.put("k1", m)
    cache.put("k2", m)
    cache.get("k1")          # k1 most recently used
    cache.put("k3", m)       # evicts k2
    assert cache.get("k2") is None
    assert cache.get("k1") is not None
    assert cache.get("k3") is not None
    assert len(cache) == 2


def test_hit_and_miss_are_counted():
    cache = MeasureCache()
    key = _key()
    with telemetry.record() as counters:
        assert cache.get(key) is None
        cache.put(key, _measurement())
        assert cache.get(key) is not None
    assert counters.cache_misses == 1
    assert counters.cache_hits == 1


def test_silence_warnings_survive_caching():
    x, rate = silence(1.0)
    measurement = measure(x, rate, allow_failure=True)
    assert "digital_silence" in measurement.warnings
    cache = MeasureCache()
    key = _key()
    cache.put(key, measurement)
    assert "digital_silence" in cache.get(key).warnings
