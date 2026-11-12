"""Minimal WAV container reader/writer.

Supports PCM int16/int24/int32 and IEEE float32, mono or multi-channel.
Samples are always returned as float64 in the range [-1.0, 1.0].
No third-party decoder is involved; the container is parsed directly so
uploaded files are treated as untrusted data and validated explicitly.
"""
from __future__ import annotations

import struct
import wave
from dataclasses import dataclass

import numpy as np

from . import telemetry


class WavFormatError(ValueError):
    """Raised when a file is not a parseable supported WAV stream."""


@dataclass
class AudioFile:
    samples: np.ndarray  # shape (n_frames, n_channels), float64
    sample_rate: int
    bit_depth: int
    is_float: bool

    @property
    def n_channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / float(self.sample_rate)


_PCM_DTYPES = {
    16: ("<i2", 1 << 15),
    24: (None, 1 << 23),
    32: ("<i4", 1 << 31),
}


def _read_fmt_chunk(data: bytes, offset: int, size: int):
    if size < 16:
        raise WavFormatError("fmt chunk too small")
    (
        format_tag,
        n_channels,
        sample_rate,
        _byte_rate,
        _block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", data, offset)
    ext_size = 0
    if size >= 18:
        ext_size = struct.unpack_from("<H", data, offset + 16)[0]
    if format_tag == 0xFFFE and ext_size >= 22:
        # WAVE_FORMAT_EXTENSIBLE: real format tag lives in the extension GUID.
        format_tag = struct.unpack_from("<H", data, offset + 24)[0]
    return format_tag, n_channels, sample_rate, bits_per_sample, ext_size


def read_wav(path: str) -> AudioFile:
    telemetry.event("decodes")
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavFormatError("not a RIFF/WAVE file")

    fmt = None
    audio_blob: bytes | None = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if body + chunk_size > len(data):
            raise WavFormatError(f"truncated chunk {chunk_id!r}")
        if chunk_id == b"fmt ":
            fmt = _read_fmt_chunk(data, body, chunk_size)
        elif chunk_id == b"data":
            audio_blob = data[body : body + chunk_size]
        pos = body + chunk_size + (chunk_size & 1)

    if fmt is None:
        raise WavFormatError("missing fmt chunk")
    if audio_blob is None:
        raise WavFormatError("missing data chunk")

    format_tag, n_channels, sample_rate, bits_per_sample, _ = fmt
    if n_channels < 1 or n_channels > 6:
        raise WavFormatError(f"unsupported channel count {n_channels}")
    if sample_rate < 8000 or sample_rate > 384000:
        raise WavFormatError(f"unsupported sample rate {sample_rate}")

    frame_bytes = n_channels * bits_per_sample // 8
    n_frames = len(audio_blob) // frame_bytes
    if n_frames == 0:
        raise WavFormatError("file contains no audio frames")

    if format_tag == 1:  # PCM
        if bits_per_sample not in _PCM_DTYPES:
            raise WavFormatError(f"unsupported PCM bit depth {bits_per_sample}")
        samples = _decode_pcm(audio_blob, n_frames, n_channels, bits_per_sample)
        is_float = False
    elif format_tag == 3:  # IEEE float
        if bits_per_sample != 32:
            raise WavFormatError("only 32-bit float WAV is supported")
        raw = np.frombuffer(audio_blob, dtype="<f4", count=n_frames * n_channels)
        samples = raw.astype(np.float64).reshape(n_frames, n_channels)
        if not np.all(np.isfinite(samples)):
            raise WavFormatError("non-finite float samples")
        is_float = True
    else:
        raise WavFormatError(f"unsupported format tag {format_tag} (only PCM/float WAV)")

    return AudioFile(
        samples=samples,
        sample_rate=int(sample_rate),
        bit_depth=int(bits_per_sample),
        is_float=is_float,
    )


def _decode_pcm(blob: bytes, n_frames: int, n_channels: int, bits: int) -> np.ndarray:
    if bits == 24:
        raw = np.frombuffer(blob, dtype=np.uint8, count=n_frames * n_channels * 3)
        raw = raw.reshape(-1, 3).astype(np.int32)
        ints = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        scale = float(1 << 23)
    else:
        dtype, scale = _PCM_DTYPES[bits]
        ints = np.frombuffer(blob, dtype=dtype, count=n_frames * n_channels)
    samples = ints.astype(np.float64).reshape(n_frames, n_channels) / scale
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0 + 1.0 / scale:
        raise WavFormatError("PCM samples out of nominal full-scale range")
    return np.clip(samples, -1.0, 1.0)


def write_wav(path: str, samples: np.ndarray, sample_rate: int, bit_depth: int = 24) -> None:
    """Write a PCM WAV file. samples: (n_frames, n_channels), float in [-1, 1]."""
    if samples.ndim == 1:
        samples = samples[:, None]
    if not np.all(np.isfinite(samples)):
        raise ValueError("refusing to write non-finite samples")
    if bit_depth not in (16, 24, 32):
        raise ValueError(f"unsupported output bit depth {bit_depth}")

    clipped = np.clip(samples, -1.0, 1.0)
    if bit_depth == 16:
        ints = np.rint(clipped * 32767.0).astype("<i2")
        sampwidth = 2
    elif bit_depth == 24:
        ints32 = np.rint(clipped * 8388607.0).astype(np.int32)
        flat = ints32.reshape(-1)
        out = np.empty((flat.size, 3), dtype=np.uint8)
        out[:, 0] = flat & 0xFF
        out[:, 1] = (flat >> 8) & 0xFF
        out[:, 2] = (flat >> 16) & 0xFF
        blob = out.tobytes()
        sampwidth = 3
    else:
        ints = np.rint(clipped * 2147483647.0).astype("<i4")
        sampwidth = 4

    with wave.open(path, "wb") as wf:
        wf.setnchannels(samples.shape[1])
        wf.setsampwidth(sampwidth)
        wf.setframerate(int(sample_rate))
        if bit_depth == 24:
            wf.writeframes(blob)
        else:
            wf.writeframes(ints.tobytes())
