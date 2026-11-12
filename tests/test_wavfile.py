import os
import tempfile

import numpy as np
import pytest

from app.components.wavfile import WavFormatError, read_wav, write_wav


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_pcm_roundtrip_preserves_signal(bits):
    rate = 48000
    t = np.arange(rate) / rate
    mono = (0.6 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
    stereo = np.concatenate([mono, -mono], axis=1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.wav")
        write_wav(path, stereo, rate, bits)
        audio = read_wav(path)
    assert audio.sample_rate == rate
    assert audio.n_channels == 2
    assert audio.bit_depth == bits
    # Rounding quantisation keeps error within one LSB plus float rounding.
    lsb = 2.0 ** (-(bits - 1))
    assert np.max(np.abs(audio.samples - stereo)) <= 1.5 * lsb


def test_rejects_non_wav_bytes():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.wav")
        with open(path, "wb") as fh:
            fh.write(b"this is definitely not a wav file" * 100)
        with pytest.raises(WavFormatError):
            read_wav(path)


def test_rejects_truncated_chunk():
    rate = 8000
    x = np.zeros((100, 1))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.wav")
        write_wav(path, x, rate, 16)
        with open(path, "rb") as fh:
            blob = fh.read()
        cut = os.path.join(d, "cut.wav")
        with open(cut, "wb") as fh:
            fh.write(blob[: len(blob) // 2])
        with pytest.raises(WavFormatError):
            read_wav(cut)


def test_refuses_non_finite_float_write():
    x = np.full((100, 1), np.inf)
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            write_wav(os.path.join(d, "nan.wav"), x, 48000, 24)


def test_duration_and_shape():
    rate = 24000
    x = np.zeros((rate * 2, 3)) if False else np.zeros((rate * 2, 1))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "d.wav")
        write_wav(path, x, rate, 24)
        audio = read_wav(path)
    assert abs(audio.duration_s - 2.0) < 1e-9
    assert audio.samples.shape == (rate * 2, 1)
