"""Channel-layout handling with one fixed, documented mono->stereo rule.

The duplication rule is part of the loudness plan and never changes at
runtime, because it changes both the measured integrated loudness
(+3.01 dB for an identical L/R pair) and the listening image (dual-mono,
no phantom centre):

    mono x[n] -> stereo [[x[n], x[n]]]   (L and R are bit-identical copies)

Down-mix is intentionally not provided; asking for mono output from a
stereo source is rejected so the balance information cannot be silently lost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loudness import _db

DUPLICATION_GAIN_DB = 20.0 * np.log10(np.sqrt(2.0))  # +3.0126 dB


class ChannelLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class LayoutDecision:
    input_layout: str
    output_layout: str
    action: str
    detail: str

    def as_dict(self) -> dict:
        return {
            "input_layout": self.input_layout,
            "output_layout": self.output_layout,
            "action": self.action,
            "detail": self.detail,
            "duplication_gain_db": round(float(DUPLICATION_GAIN_DB), 4),
        }


def _layout_name(n_channels: int) -> str:
    return {1: "mono", 2: "stereo", 6: "5.1"}.get(n_channels, f"{n_channels}ch")


def channel_balance(samples: np.ndarray) -> dict:
    """Per-channel RMS and L/R imbalance in dB (stereo only)."""
    n_channels = samples.shape[1]
    rms = [float(np.sqrt(np.mean(samples[:, c] ** 2))) for c in range(n_channels)]
    result = {
        "channel_rms_dbfs": [round(_db(v), 4) for v in rms],
        "lr_imbalance_db": None,
        "silent_channels": [c for c, v in enumerate(rms) if v <= 10 ** (-70 / 20)],
    }
    if n_channels == 2 and rms[0] > 0 and rms[1] > 0:
        result["lr_imbalance_db"] = round(_db(rms[0]) - _db(rms[1]), 4)
    return result


def apply_layout(samples: np.ndarray, target_layout: str) -> tuple[np.ndarray, LayoutDecision]:
    input_layout = _layout_name(samples.shape[1])
    if input_layout == target_layout:
        decision = LayoutDecision(
            input_layout, target_layout, "none",
            "input already matches requested layout",
        )
        return samples, decision

    if input_layout == "mono" and target_layout == "stereo":
        duplicated = np.repeat(samples, 2, axis=1)
        if not np.array_equal(duplicated[:, 0], duplicated[:, 1]):
            raise AssertionError("duplicated channels must be bit-identical")
        decision = LayoutDecision(
            "mono",
            "stereo",
            "duplicate_mono_to_dual_mono",
            "L = R = mono input (bit-identical); measured integrated loudness "
            f"rises {DUPLICATION_GAIN_DB:.2f} dB under channel summation",
        )
        return duplicated, decision

    raise ChannelLayoutError(
        f"layout conversion {input_layout} -> {target_layout} is not supported; "
        "only mono->stereo duplication is defined, down-mix is refused"
    )
