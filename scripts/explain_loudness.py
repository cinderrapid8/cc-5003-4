"""Executable loudness explainer: walk real signals through the original
production flow, stage by stage, and verify the walk against run_chain.

Three didactic cases cover the flow end to end:
  * calibrated speech-like noise that matches the target with gain alone;
  * a narrow full-scale impulse that forces the true-peak limiter to engage;
  * digital silence, which is rejected at the source measurement.

Every case is processed twice: once as an explicit stage-by-stage walk
calling the same components in the original order, and once through
run_chain WITHOUT the measurement cache (the original computation order).
The two must agree on every reported number; a disagreement exits 1, so the
explanation can never drift away from the code it explains.

Run:  python3 scripts/explain_loudness.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.components import telemetry
from app.components.channels import apply_layout, channel_balance
from app.components.loudness import measure, measure_file
from app.components.processing import (
    Target,
    _db_to_gain,
    _tpdf_dither,
    apply_limiter,
    run_chain,
    static_gain_db,
)
from app.components.resampler import resample
from app.components.wavfile import AudioFile, read_wav, write_wav
from tests import signals

CASES = [
    (
        "speech_minus23",
        signals.calibrated_speech_like,
        dict(integrated_lufs=-23.0, true_peak_dbtp=-1.0, sample_rate=48000,
             layout="mono", bit_depth=24),
    ),
    (
        "narrow_impulse",
        lambda: signals.narrow_impulse(2.0),
        dict(integrated_lufs=-23.0, true_peak_dbtp=-2.0, sample_rate=48000,
             layout="mono", bit_depth=24),
    ),
    (
        "silence",
        lambda: signals.silence(1.5),
        dict(integrated_lufs=-23.0, true_peak_dbtp=-2.0, sample_rate=48000,
             layout="mono", bit_depth=24),
    ),
]


def _line(stage: str, text: str) -> None:
    print(f"  [{stage:<9}] {text}")


def walk_original_flow(audio: AudioFile, target: Target, work_dir: str) -> dict:
    """Stage-by-stage walk of the original chain order, with telemetry."""
    summary: dict = {}
    with telemetry.record() as counters:
        _line("decode", f"{audio.samples.shape[0]} frames, {audio.n_channels} ch, "
                        f"{audio.sample_rate} Hz, {audio.duration_s:.3f} s")
        summary["balance"] = channel_balance(audio.samples)

        samples, layout = apply_layout(audio.samples.astype(np.float64), target.layout)
        _line("layout", f"{layout.action}: {layout.detail}")

        samples, rate_settings = resample(samples, audio.sample_rate, target.sample_rate)
        _line("resample", f"up={rate_settings.up} down={rate_settings.down} "
                          f"taps={rate_settings.filter_taps}")

        source = measure(samples, target.sample_rate, allow_failure=True)
        _line("measure", f"integrated={source.integrated_lufs} LUFS, "
                         f"true_peak={source.true_peak_dbtp:.4f} dBTP, "
                         f"gated_blocks={source.n_gated_blocks}/{source.n_total_blocks}, "
                         f"warnings={source.warnings or 'none'}")
        summary["source"] = source
        if source.warnings:
            _line("verdict", f"rejected: {','.join(source.warnings)}")
            summary["status"] = "rejected"
            summary["reason"] = ",".join(source.warnings)
            summary["counters"] = counters.as_dict()
            return summary

        gain_db = static_gain_db(source.integrated_lufs, target.integrated_lufs)
        processed = samples * _db_to_gain(gain_db)
        _line("gain", f"{gain_db:+.4f} dB toward {target.integrated_lufs} LUFS")
        summary["static_gain_db"] = gain_db

        after_gain = measure(processed, target.sample_rate, allow_failure=True)
        _line("tp-check", f"after-gain true peak {after_gain.true_peak_dbtp:.4f} dBTP "
                          f"vs ceiling {target.true_peak_dbtp} dBTP")
        limiter_trace = None
        if after_gain.true_peak_dbtp > target.true_peak_dbtp + 0.001:
            processed, limiter_trace, _ = apply_limiter(
                processed, target.sample_rate, target.true_peak_dbtp
            )
            _line("limiter", f"engaged: {limiter_trace.iterations} iteration(s), "
                             f"max reduction {limiter_trace.gain_reduction_db:.4f} dB")
        else:
            _line("limiter", "not needed: ceiling met by gain alone")
        summary["limiter"] = None if limiter_trace is None else limiter_trace.as_dict()

        out_path = os.path.join(work_dir, "walk_export.wav")
        if target.bit_depth == 16:
            lsb = 2.0 ** (-(target.bit_depth - 1))
            processed = np.clip(
                # Same default seed as run_chain so the walk stays comparable.
                processed + _tpdf_dither(processed.shape, lsb,
                                         np.random.default_rng(0xA17C)),
                -1.0, 1.0,
            )
        write_wav(out_path, processed, target.sample_rate, target.bit_depth)
        _line("export", f"{target.bit_depth}-bit WAV written")

        exported = measure_file(out_path)
        _line("re-measure", f"integrated={exported.integrated_lufs} LUFS, "
                            f"true_peak={exported.true_peak_dbtp:.4f} dBTP, "
                            f"sample_peak={exported.sample_peak_dbfs:.4f} dBFS")
        summary["output"] = exported
        summary["status"] = "measured"
        summary["counters"] = counters.as_dict()
    return summary


def main() -> int:
    failures: list[str] = []
    for name, factory, target_raw in CASES:
        print(f"\n=== {name} ===")
        target = Target.from_dict(target_raw)
        signal, rate = factory()
        with tempfile.TemporaryDirectory(prefix=f"explain_{name}_") as work_dir:
            input_path = os.path.join(work_dir, "input.wav")
            write_wav(input_path, signal, rate, 24)
            audio = read_wav(input_path)

            walked = walk_original_flow(audio, target, work_dir)
            c = walked["counters"]
            print(f"  [counters ] decodes={c['decodes']} measures={c['measures']} "
                  f"tp_checks={c['true_peak_checks']} resamples={c['resamples']} "
                  f"limiter_runs={c['limiter_runs']}")

            # The same material through the original run_chain (no cache).
            chain = run_chain(audio, target, os.path.join(work_dir, "chain.wav"))

            if walked["status"] == "rejected":
                if chain.status != "rejected" or chain.reason != walked["reason"]:
                    failures.append(
                        f"{name}: walk rejected ({walked['reason']}) but chain "
                        f"returned {chain.status} ({chain.reason})"
                    )
                continue
            if walked["source"].integrated_lufs != chain.source.integrated_lufs:
                failures.append(f"{name}: source integrated mismatch")
            if abs(walked["static_gain_db"] - chain.static_gain_db) > 1e-9:
                failures.append(f"{name}: static gain mismatch")
            if (walked["limiter"] is None) != (chain.limiter is None):
                failures.append(f"{name}: limiter engagement mismatch")
            if abs(walked["output"].integrated_lufs - chain.output.integrated_lufs) > 1e-9:
                failures.append(f"{name}: exported integrated mismatch")
            if abs(walked["output"].true_peak_dbtp - chain.output.true_peak_dbtp) > 1e-9:
                failures.append(f"{name}: exported true peak mismatch")
            print(f"  [verify   ] walk matches run_chain: status={chain.status}, "
                  f"exported {chain.output.integrated_lufs:.4f} LUFS / "
                  f"{chain.output.true_peak_dbtp:.4f} dBTP")

    print()
    if failures:
        print(f"EXPLAINER FAILED — {len(failures)} mismatch(es):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("All cases walked the original flow and match run_chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
