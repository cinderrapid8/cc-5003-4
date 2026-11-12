import numpy as np

from app.components import telemetry
from app.components.loudness import measure, true_peak_check
from app.components.wavfile import read_wav, write_wav


def _tone(path, rate=48000):
    x = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(rate) / rate)).reshape(-1, 1)
    write_wav(str(path), x, rate, 24)
    return x


def test_counters_record_pipeline_events(tmp_path):
    path = tmp_path / "a.wav"
    _tone(path)
    with telemetry.record() as counters:
        audio = read_wav(str(path))
        measure(audio.samples, audio.sample_rate)
    assert counters.decodes == 1
    assert counters.measures == 1
    assert counters.true_peak_checks == 1
    assert counters.resamples >= 1  # 4x oversampling for the true peak
    assert counters.limiter_runs == 0


def test_true_peak_check_counts_as_check_not_measure(tmp_path):
    path = tmp_path / "b.wav"
    x = _tone(path)
    with telemetry.record() as counters:
        true_peak_check(x, 48000)
    assert counters.true_peak_checks == 1
    assert counters.measures == 0


def test_event_without_recorder_is_noop():
    telemetry.event("decodes")  # must not raise and must not crash producers


def test_nested_recorders_are_scoped():
    with telemetry.record() as outer:
        with telemetry.record() as inner:
            telemetry.event("decodes")
        telemetry.event("measures")
    assert inner.decodes == 1 and inner.measures == 0
    assert outer.decodes == 0 and outer.measures == 1


def test_counters_as_dict_covers_all_fields():
    assert set(telemetry.Counters().as_dict()) == set(telemetry.FIELDS)
