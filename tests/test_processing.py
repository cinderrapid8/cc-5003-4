import math
import os
import tempfile

import numpy as np
import pytest

from app.components.loudness import measure_file
from app.components.processing import (
    MAX_GAIN_DB,
    MIN_GAIN_DB,
    ProcessingError,
    Target,
    run_chain,
    static_gain_db,
)
from app.components.wavfile import read_wav, write_wav
from tests.signals import (
    calibrated_speech_like,
    high_peak_low_average,
    narrow_impulse,
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


def _write(sig, rate, d, name="in.wav", bits=24):
    path = os.path.join(d, name)
    write_wav(path, sig, rate, bits)
    return read_wav(path)


def test_static_gain_is_always_finite_and_bounded():
    assert static_gain_db(-90.0, -23.0) == MAX_GAIN_DB
    assert static_gain_db(0.0, -60.0) == MIN_GAIN_DB
    # Missing source loudness (silence) must never produce infinite gain.
    assert static_gain_db(None, -23.0) == 0.0
    assert static_gain_db(float("nan"), -23.0) == 0.0
    g = static_gain_db(-30.0, -23.0)
    assert math.isfinite(10 ** (g / 20))


def test_target_validation_rejects_out_of_range():
    with pytest.raises(ProcessingError):
        Target.from_dict({"integrated_lufs": 5, "true_peak_dbtp": -2,
                          "sample_rate": 48000, "layout": "mono"})
    with pytest.raises(ProcessingError):
        Target.from_dict({"integrated_lufs": -23, "true_peak_dbtp": 5,
                          "sample_rate": 48000, "layout": "mono"})


def test_matched_chain_hits_loudness_and_peak_targets():
    sig, rate = calibrated_speech_like(seconds=4)
    with tempfile.TemporaryDirectory() as d:
        audio = _write(sig, rate, d)
        result = run_chain(audio, _target(), os.path.join(d, "out.wav"))
        assert result.status == "matched"
        assert result.limiter is None
        out = result.output
        assert abs(out.integrated_lufs - (-23.0)) < 0.5
        assert out.true_peak_dbtp <= -2.0 + 0.05
        # Exported file re-measurement matches the reported result.
        on_disk = measure_file(os.path.join(d, "out.wav"))
        assert abs(on_disk.integrated_lufs - out.integrated_lufs) < 0.05


def test_silence_is_rejected_and_no_output_listed():
    sig, rate = silence(2.0)
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "out.wav")
        audio = _write(sig, rate, d)
        result = run_chain(audio, _target(), out_path)
        assert result.status == "rejected"
        assert "digital_silence" in result.reason
        assert not os.path.exists(out_path)
        assert result.output is None


def test_conflicting_targets_reported_as_target_missed_with_gap():
    # Loud speech asked to go louder while held under a strict true-peak cap.
    sig, rate = calibrated_speech_like(seconds=4)
    with tempfile.TemporaryDirectory() as d:
        audio = _write(sig, rate, d)
        target = _target(integrated_lufs=-16.0, true_peak_dbtp=-10.0)
        result = run_chain(audio, target, os.path.join(d, "out.wav"))
        assert result.status == "target_missed"
        assert result.limiter is not None
        # The peak ceiling is still enforced even though loudness cannot be.
        assert result.output.true_peak_dbtp <= target.true_peak_dbtp + 0.05
        assert result.unmet  # the remaining loudness gap is explained


def test_impulse_never_gets_infinite_gain_and_peak_is_bounded():
    sig, rate = narrow_impulse(2.0)
    with tempfile.TemporaryDirectory() as d:
        audio = _write(sig, rate, d)
        result = run_chain(audio, _target(true_peak_dbtp=-2.0),
                           os.path.join(d, "out.wav"))
        assert math.isfinite(result.static_gain_db)
        assert result.output.true_peak_dbtp <= -2.0 + 0.05
        assert np.all(np.isfinite(read_wav(os.path.join(d, "out.wav")).samples))


def test_high_peak_low_average_is_limited_not_clipped_silently():
    sig, rate = high_peak_low_average(3.0)
    with tempfile.TemporaryDirectory() as d:
        audio = _write(sig, rate, d)
        result = run_chain(audio, _target(true_peak_dbtp=-3.0),
                           os.path.join(d, "out.wav"))
        assert result.limiter is not None
        assert result.limiter.gain_reduction_db > 0.0
        assert result.output.sample_peak_dbfs <= -3.0 + 0.05


def test_16_bit_export_applies_tpdf_dither():
    sig, rate = calibrated_speech_like(seconds=3)
    with tempfile.TemporaryDirectory() as d:
        audio = _write(sig, rate, d)
        result = run_chain(
            audio, _target(bit_depth=16), os.path.join(d, "out16.wav")
        )
        assert result.quantisation["dither"] == "tpdf_2lsb"
        exported = read_wav(os.path.join(d, "out16.wav"))
        assert exported.bit_depth == 16
        assert np.max(np.abs(exported.samples)) <= 1.0


def test_full_chain_settings_are_recorded():
    sig, rate = calibrated_speech_like(seconds=3)
    with tempfile.TemporaryDirectory() as d:
        audio = _write(sig, rate, d)
        result = run_chain(
            audio, _target(sample_rate=44100, layout="stereo"),
            os.path.join(d, "out.wav"),
        )
        record = result.as_dict()
        assert record["resampling"]["output_rate"] == 44100
        assert record["layout_decision"]["action"] == "duplicate_mono_to_dual_mono"
        assert record["target"]["layout"] == "stereo"
        assert record["output_measurement"]["sample_rate"] == 44100


def test_post_export_measurement_uses_file_not_filter_params():
    sig, rate = calibrated_speech_like(seconds=3)
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "out.wav")
        audio = _write(sig, rate, d)
        result = run_chain(audio, _target(), out_path)
        independent = measure_file(out_path)
        assert result.output.true_peak_dbtp == pytest.approx(
            independent.true_peak_dbtp, abs=1e-9
        )
        assert result.output.sample_peak_dbfs == pytest.approx(
            independent.sample_peak_dbfs, abs=1e-9
        )
