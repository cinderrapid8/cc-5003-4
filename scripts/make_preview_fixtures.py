"""Generate WAV previews for manual UI testing and demo.

Outputs one WAV per fixture into a chosen directory. These are real, playable
audio files (not synthetic placeholders rendered by the browser): a shaped
speech-like signal, an L/R imbalance stereo file, a high-peak/low-energy
file, a narrow impulse, digital silence and a sub-0.4 s short segment.

Run:  python3 scripts/make_preview_fixtures.py [out_dir]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.wavfile import write_wav
from tests import signals


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "data", "preview"
    )
    os.makedirs(out_dir, exist_ok=True)
    jobs = {
        "speech_minus23.wav": signals.calibrated_speech_like,
        "silence_mono.wav": lambda: signals.silence(2.0, stereo=False),
        "silence_stereo.wav": lambda: signals.silence(2.0, stereo=True),
        "short_click.wav": signals.short_click,
        "narrow_impulse.wav": signals.narrow_impulse,
        "bandlimited_sine.wav": signals.bandlimited_sine,
        "lr_imbalance.wav": signals.left_right_imbalance,
        "high_peak_low_average.wav": signals.high_peak_low_average,
    }
    for name, factory in jobs.items():
        samples, rate = factory()
        path = os.path.join(out_dir, name)
        write_wav(path, samples, rate, 24)
        print(f"wrote {path}  {samples.shape} @ {rate} Hz")


if __name__ == "__main__":
    main()
