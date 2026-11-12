import numpy as np
import pytest

from app.components.channels import (
    DUPLICATION_GAIN_DB,
    ChannelLayoutError,
    apply_layout,
    channel_balance,
)
from app.components.loudness import measure


def test_mono_to_stereo_duplicates_bit_identical():
    x = np.linspace(-0.5, 0.5, 4800).reshape(-1, 1)
    y, decision = apply_layout(x, "stereo")
    assert y.shape == (4800, 2)
    assert np.array_equal(y[:, 0], y[:, 1])
    assert np.array_equal(y[:, 0], x[:, 0])
    assert decision.action == "duplicate_mono_to_dual_mono"


def test_duplication_raises_loudness_by_3_db():
    rate = 48000
    n = rate * 4
    x = (0.4 * np.sin(2 * np.pi * 800 * np.arange(n) / rate)).reshape(-1, 1)
    mono = measure(x, rate).integrated_lufs
    stereo, _ = apply_layout(x, "stereo")
    dual = measure(stereo, rate).integrated_lufs
    assert abs((dual - mono) - DUPLICATION_GAIN_DB) < 0.1


def test_mono_layout_is_passthrough():
    x = np.zeros((100, 1))
    y, decision = apply_layout(x, "mono")
    assert y is x
    assert decision.action == "none"


def test_stereo_to_mono_downmix_is_refused():
    x = np.zeros((100, 2))
    with pytest.raises(ChannelLayoutError):
        apply_layout(x, "mono")


def test_channel_balance_detects_imbalance_and_silence():
    left = np.full(4800, 0.5)
    right = np.full(4800, 0.125)  # 12 dB below left
    x = np.stack([left, right], axis=1)
    balance = channel_balance(x)
    assert abs(balance["lr_imbalance_db"] - 12.0) < 0.05
    x[:, 1] = 0.0
    balance = channel_balance(x)
    assert 1 in balance["silent_channels"]
