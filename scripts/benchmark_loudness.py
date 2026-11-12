"""Loudness re-measurement cost benchmark: baseline vs. optimized chain.

Cost model
----------
Repeatedly adjusting loudness targets on the same material makes every job
pay for: input decode, source measurement (K-weighting + two-stage gating +
4x oversampled true peak), a post-gain measurement, the limiter (when the
ceiling is exceeded), export, and a post-export decode + re-measurement.
This benchmark decomposes that re-measurement overhead into counters —
decodes, full measures, true-peak checks, resamples, limiter runs, cache
hits/misses — and records them next to the peaks each run produced, so the
cost and the evidence stay tied together.

Load ladder
-----------
Level L is L narrow impulses (0.9 FS, one sample wide), each preceded by a
0.30 s short-silence gap, plus a 0.6 s tail: both the signal length and the
number of limiter-provoking peaks grow level by level. Every level is
processed under several targets (same rate/layout, then one different rate
to demonstrate the cache-key boundary), once with the original computation
order (no cache) and once with the optimized order (source-measurement
cache + true-peak-only post-gain check).

Computation-order changes being measured (see processing.run_chain):
  * stage 4 (source measurement) may be served from the content-addressed
    cache — the gating/oversampling work is skipped only on a key hit;
  * stage 6 (post-gain check) uses the oversampled true-peak check only —
    the after-gain integrated loudness is never consumed by the chain, so
    it is not computed; the true-peak check itself always runs;
  * stage 8 (post-export re-measurement) is identical in both orders.

Correctness gate
----------------
The optimized order is only allowed to differ in the counters. Every
recorded statistic (status, measurements, limiter trace, unmet conditions)
must match the baseline within 1e-12; any run that raises, any missing
value, or any mismatch is reported as a failure — failures are never
masked by substituting default results. Exit code is 1 when any failure
was recorded.

Run:  python3 scripts/benchmark_loudness.py [--levels 1,2,3,4] [--report out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.components import telemetry
from app.components.measure_cache import MeasureCache
from app.components.processing import Target, run_chain
from app.components.wavfile import read_wav, write_wav
from app.service import np_rng_seeded

RATE = 48000
SEGMENT_SILENCE_S = 0.30   # short-silence gap before each impulse
TAIL_S = 0.60              # keeps >= one 0.4 s gating block at level 1
IMPULSE_AMP = 0.9
FLOAT_TOL = 1e-12

DEFAULT_TARGETS = [
    dict(integrated_lufs=-23.0, true_peak_dbtp=-1.0, sample_rate=48000,
         layout="mono", bit_depth=24),
    dict(integrated_lufs=-18.0, true_peak_dbtp=-2.0, sample_rate=48000,
         layout="mono", bit_depth=24),
    dict(integrated_lufs=-30.0, true_peak_dbtp=-3.0, sample_rate=48000,
         layout="mono", bit_depth=16),
    # Different rate: same input must NOT reuse the 48 kHz measurement.
    dict(integrated_lufs=-23.0, true_peak_dbtp=-1.0, sample_rate=44100,
         layout="mono", bit_depth=24),
]


def level_signal(level: int) -> np.ndarray:
    """L x (short silence + one-sample narrow impulse) plus a silence tail."""
    segment = np.zeros(int(SEGMENT_SILENCE_S * RATE))
    segment[-1] = IMPULSE_AMP
    parts = [segment.copy() for _ in range(level)]
    parts.append(np.zeros(int(TAIL_S * RATE)))
    return np.concatenate(parts).reshape(-1, 1)


def _peaks(result) -> dict:
    """Peaks for the evidence row. None stays None — never defaulted."""
    out = result.output
    return {
        "status": result.status,
        "integrated_lufs": None if out is None else out.integrated_lufs,
        "true_peak_dbtp": None if out is None else out.true_peak_dbtp,
        "sample_peak_dbfs": None if out is None else out.sample_peak_dbfs,
        "limiter_reduction_db": (
            None if result.limiter is None else round(result.limiter.gain_reduction_db, 4)
        ),
    }


def _run_once(input_path: str, input_sha256: str, target_raw: dict,
              out_path: str, cache: MeasureCache | None, seed: str) -> dict:
    target = Target.from_dict(target_raw)
    started = time.perf_counter()
    with telemetry.record() as counters:
        audio = read_wav(input_path)
        result = run_chain(
            audio, target, out_path,
            rng=np_rng_seeded(seed),
            cache=cache,
            input_sha256=input_sha256 if cache is not None else None,
        )
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "result": result,
        "counters": counters.as_dict(),
        "wall_ms": round(wall_ms, 2),
    }


def compare_values(base, opt, path, failures):
    """Recursive equality: counters are compared by the caller, everything
    else must match. A value present on one side only is a failure, never
    defaulted."""
    if isinstance(base, dict) and isinstance(opt, dict):
        if set(base) != set(opt):
            failures.append(
                f"{path}: key mismatch {sorted(set(base) ^ set(opt))}"
            )
        for key in sorted(set(base) & set(opt)):
            compare_values(base[key], opt[key], f"{path}.{key}", failures)
        return
    if isinstance(base, list) and isinstance(opt, list):
        if len(base) != len(opt):
            failures.append(f"{path}: length {len(base)} != {len(opt)}")
            return
        for i, (x, y) in enumerate(zip(base, opt)):
            compare_values(x, y, f"{path}[{i}]", failures)
        return
    if base is None or opt is None:
        if base is not None or opt is not None:
            failures.append(f"{path}: {base!r} vs {opt!r} (missing value)")
        return
    if isinstance(base, float) or isinstance(opt, float):
        if not (math.isclose(float(base), float(opt), rel_tol=0.0, abs_tol=FLOAT_TOL)):
            failures.append(f"{path}: {base!r} != {opt!r}")
        return
    if base != opt:
        failures.append(f"{path}: {base!r} != {opt!r}")


def _compare_results(baseline: dict, optimized: dict) -> list[str]:
    failures: list[str] = []
    base_dict = baseline["result"].as_dict()
    opt_dict = optimized["result"].as_dict()
    base_dict.pop("output_path", None)
    opt_dict.pop("output_path", None)
    compare_values(base_dict, opt_dict, "chain", failures)
    return failures


def _total_counters(runs: list[dict]) -> dict:
    total = {name: 0 for name in telemetry.FIELDS}
    for run in runs:
        for name in telemetry.FIELDS:
            total[name] += run["counters"][name]
    return total


def run_ladder(work_dir: str, levels, targets) -> tuple[list[dict], list[str]]:
    """For each level: baseline runs (original order) and optimized runs
    (shared cache), then a per-target correctness comparison."""
    levels_report: list[dict] = []
    failures: list[str] = []
    for level in levels:
        signal = level_signal(level)
        input_path = os.path.join(work_dir, f"level{level}_input.wav")
        write_wav(input_path, signal, RATE, 24)
        with open(input_path, "rb") as fh:
            input_sha256 = hashlib.sha256(fh.read()).hexdigest()

        runs: list[dict] = []
        baseline_runs: list[dict] = []
        optimized_runs: list[dict] = []
        cache = MeasureCache()
        for mode in ("baseline", "optimized"):
            for ti, target_raw in enumerate(targets):
                seed = f"bench-L{level}-T{ti}"
                out_path = os.path.join(work_dir, f"L{level}_{mode}_T{ti}.wav")
                entry = {
                    "mode": mode,
                    "target_index": ti,
                    "target": dict(target_raw),
                    "seed": seed,
                }
                try:
                    run = _run_once(
                        input_path, input_sha256, target_raw, out_path,
                        cache if mode == "optimized" else None, seed,
                    )
                except Exception as exc:  # never masked: recorded as failure
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    failures.append(f"level {level} {mode} T{ti} raised: {entry['error']}")
                else:
                    entry.update(_peaks(run["result"]))
                    entry["counters"] = run["counters"]
                    entry["wall_ms"] = run["wall_ms"]
                    (baseline_runs if mode == "baseline" else optimized_runs).append(run)
                runs.append(entry)

        comparisons: list[dict] = []
        for ti in range(len(targets)):
            if ti >= len(baseline_runs) or ti >= len(optimized_runs):
                comparisons.append({"target_index": ti, "failures": ["run missing"]})
                continue
            pair_failures = _compare_results(baseline_runs[ti], optimized_runs[ti])
            comparisons.append({"target_index": ti, "failures": pair_failures})
            failures.extend(
                f"level {level} target {ti}: {f}" for f in pair_failures
            )

        base_totals = _total_counters(
            [{"counters": r["counters"]} for r in baseline_runs]
        )
        opt_totals = _total_counters(
            [{"counters": r["counters"]} for r in optimized_runs]
        )
        if opt_totals["measures"] >= base_totals["measures"]:
            failures.append(
                f"level {level}: optimized measures {opt_totals['measures']} "
                f"did not drop below baseline {base_totals['measures']}"
            )
        levels_report.append({
            "level": level,
            "n_frames": int(signal.shape[0]),
            "duration_s": round(signal.shape[0] / RATE, 6),
            "n_impulses": level,
            "runs": runs,
            "comparisons": comparisons,
            "totals": {"baseline": base_totals, "optimized": opt_totals},
        })
    return levels_report, failures


def run_boundaries(work_dir: str) -> tuple[list[dict], list[str]]:
    """Cache-boundary evidence for silence, short clips and actual limiting.

    Each case runs the optimized chain twice on identical material and
    target; the second run must reuse the source measurement (cache hit)
    while still applying the boundary rules.
    """
    from tests.signals import narrow_impulse, short_click, silence

    cases = [
        ("silence", silence(1.0), "digital_silence"),
        ("short_clip", short_click(), "shorter_than_one_gating_block"),
        ("limiter_engaged", narrow_impulse(1.0), None),
    ]
    target = dict(integrated_lufs=-23.0, true_peak_dbtp=-3.0,
                  sample_rate=48000, layout="mono", bit_depth=24)
    report: list[dict] = []
    failures: list[str] = []
    for name, (signal, rate), expected_warning in cases:
        input_path = os.path.join(work_dir, f"boundary_{name}.wav")
        write_wav(input_path, signal, rate, 24)
        with open(input_path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        cache = MeasureCache()
        runs = []
        for i in (1, 2):
            out_path = os.path.join(work_dir, f"boundary_{name}_{i}.wav")
            try:
                run = _run_once(input_path, digest, target, out_path, cache,
                                f"boundary-{name}-{i}")
            except Exception as exc:
                failures.append(f"boundary {name} run {i} raised: {exc}")
                runs.append({"run": i, "error": str(exc)})
                continue
            entry = {"run": i, "counters": run["counters"], **_peaks(run["result"])}
            entry["reason"] = run["result"].reason
            runs.append(entry)
        case_failures: list[str] = []
        if len(runs) == 2 and all("counters" in r for r in runs):
            first, second = runs
            if second["counters"]["cache_hits"] < 1:
                case_failures.append("second run did not hit the source cache")
            if expected_warning is not None:
                # Silence / short clip: cached rejection — the second run
                # rejects again without re-measuring.
                for r in runs:
                    if r["status"] != "rejected" or expected_warning not in (r["reason"] or ""):
                        case_failures.append(
                            f"run {r['run']} not rejected with {expected_warning}: "
                            f"{r['status']} {r['reason']}"
                        )
                if second["counters"]["measures"] != 0:
                    case_failures.append(
                        f"cached rejection still measured: {second['counters']['measures']}"
                    )
            else:
                # Actual limiting: the limiter runs in both passes and the
                # true-peak check is not skipped on the cache hit.
                for r in runs:
                    if r["counters"]["limiter_runs"] < 1:
                        case_failures.append(f"run {r['run']} did not engage the limiter")
                if second["counters"]["true_peak_checks"] < 1:
                    case_failures.append("cache hit skipped the true-peak check")
                if first["true_peak_dbtp"] != second["true_peak_dbtp"]:
                    case_failures.append("limited outputs diverged between runs")
            # The export re-measurement is never cached: every successful
            # run decodes input + export.
            for r in runs:
                if r["status"] != "rejected" and r["counters"]["decodes"] != 2:
                    case_failures.append(
                        f"run {r['run']} decoded {r['counters']['decodes']} files "
                        "(export re-measurement must always decode)"
                    )
        failures.extend(f"boundary {name}: {f}" for f in case_failures)
        report.append({
            "case": name,
            "expected_warning": expected_warning,
            "runs": runs,
            "failures": case_failures,
        })
    return report, failures


def run_benchmark(levels=(1, 2, 3, 4), targets=None, work_dir: str | None = None,
                  boundaries: bool = True) -> dict:
    targets = list(targets) if targets is not None else list(DEFAULT_TARGETS)
    failures: list[str] = []
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="loudness_bench_") as tmp:
            return run_benchmark(levels, targets, tmp, boundaries)
    levels_report, ladder_failures = run_ladder(work_dir, levels, targets)
    failures.extend(ladder_failures)
    boundaries_report: list[dict] = []
    if boundaries:
        boundaries_report, boundary_failures = run_boundaries(work_dir)
        failures.extend(boundary_failures)
    return {
        "levels": levels_report,
        "boundaries": boundaries_report,
        "failures": failures,
        "ok": not failures,
    }


def _fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def print_report(report: dict) -> None:
    print("LOUDNESS RE-MEASUREMENT COST BENCHMARK")
    print("counters: decodes / measures / tp_checks / resamples / limiter / cache h-m")
    for level in report["levels"]:
        print(f"\nlevel {level['level']}: {level['n_frames']} frames "
              f"({level['duration_s']} s), {level['n_impulses']} narrow impulse(s)")
        header = (f"  {'mode':<10}{'target':<26}{'dec':>4}{'mea':>4}{'tpc':>4}"
                  f"{'res':>4}{'lim':>4}{'cache':>8}  {'integrated':>10}"
                  f"{'true_peak':>10}{'smp_peak':>10}  {'status':<13}{'ms':>8}")
        print(header)
        for run in level["runs"]:
            t = run["target"]
            label = f"{t['integrated_lufs']}/{t['true_peak_dbtp']}/{t['sample_rate']}"
            if "error" in run:
                print(f"  {run['mode']:<10}{label:<26} ERROR: {run['error']}")
                continue
            c = run["counters"]
            cache_str = f"{c['cache_hits']}-{c['cache_misses']}"
            print(f"  {run['mode']:<10}{label:<26}"
                  f"{c['decodes']:>4}{c['measures']:>4}{c['true_peak_checks']:>4}"
                  f"{c['resamples']:>4}{c['limiter_runs']:>4}"
                  f"{cache_str:>8}  "
                  f"{_fmt(run['integrated_lufs']):>10}"
                  f"{_fmt(run['true_peak_dbtp']):>10}"
                  f"{_fmt(run['sample_peak_dbfs']):>10}  "
                  f"{run['status']:<13}{run['wall_ms']:>8.1f}")
        totals = level["totals"]
        print(f"  totals    baseline={totals['baseline']}  optimized={totals['optimized']}")
        for comp in level["comparisons"]:
            verdict = "OK" if not comp["failures"] else "FAIL"
            print(f"  compare target {comp['target_index']}: {verdict}")
            for failure in comp["failures"]:
                print(f"    {failure}")
    if report["boundaries"]:
        print("\nCACHE BOUNDARIES (optimized chain, two identical runs)")
        for case in report["boundaries"]:
            print(f"  {case['case']}: expected={case['expected_warning'] or 'limiter engages'}")
            for run in case["runs"]:
                if "error" in run:
                    print(f"    run {run['run']}: ERROR {run['error']}")
                    continue
                c = run["counters"]
                print(f"    run {run['run']}: status={run['status']} "
                      f"decodes={c['decodes']} measures={c['measures']} "
                      f"tp_checks={c['true_peak_checks']} limiter={c['limiter_runs']} "
                      f"cache h-m={c['cache_hits']}-{c['cache_misses']} "
                      f"true_peak={_fmt(run['true_peak_dbtp'])}")
            for failure in case["failures"]:
                print(f"    FAIL {failure}")
    print()
    if report["ok"]:
        print("RESULT: OK — optimized order differs only in counters")
    else:
        print(f"RESULT: FAIL — {len(report['failures'])} problem(s):")
        for failure in report["failures"]:
            print(f"  {failure}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--levels", default="1,2,3,4",
                        help="comma-separated load levels (impulse count)")
    parser.add_argument("--report", default=None,
                        help="optional path for the JSON report")
    args = parser.parse_args(argv)
    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    report = run_benchmark(levels=levels)
    print_report(report)
    if args.report:
        serializable = {
            **report,
            "levels": [
                {**level, "runs": [
                    {k: v for k, v in run.items()} for run in level["runs"]
                ]}
                for level in report["levels"]
            ],
        }
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2, default=str)
        print(f"report written to {args.report}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
