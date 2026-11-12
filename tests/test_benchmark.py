"""Benchmark harness tests: the benchmark must surface failures, never mask
them with defaults, and must certify that the optimized order differs from
the baseline only in the execution counters."""
import pytest

import scripts.benchmark_loudness as bench

TWO_TARGETS = bench.DEFAULT_TARGETS[:2]


def test_compare_values_flags_any_difference():
    failures = []
    bench.compare_values({"a": 1.0, "b": None}, {"a": 1.0, "b": None}, "x", failures)
    assert failures == []
    bench.compare_values({"a": 1.0}, {"a": 1.0 + 1e-6}, "x", failures)
    assert len(failures) == 1
    # A missing value on one side is a failure, never a defaulted 0.0.
    bench.compare_values({"a": None}, {"a": 0.0}, "x", failures)
    assert len(failures) == 2
    bench.compare_values({"a": "matched"}, {"a": "limited"}, "x", failures)
    assert len(failures) == 3


def test_benchmark_small_ladder_ok_and_saves_measures(tmp_path):
    report = bench.run_benchmark(levels=(1,), targets=TWO_TARGETS,
                                 work_dir=str(tmp_path), boundaries=False)
    assert report["ok"], report["failures"]
    totals = report["levels"][0]["totals"]
    assert totals["optimized"]["measures"] < totals["baseline"]["measures"]
    # Decode counts are evidence, not savings: input + export per run.
    assert totals["optimized"]["decodes"] == totals["baseline"]["decodes"] == 4
    # Every run recorded its peaks next to the counters.
    for run in report["levels"][0]["runs"]:
        assert run["integrated_lufs"] is not None
        assert run["true_peak_dbtp"] is not None
        assert run["counters"]["decodes"] == 2


def test_benchmark_cache_boundaries_ok(tmp_path):
    report = bench.run_benchmark(levels=(1,), targets=TWO_TARGETS[:1],
                                 work_dir=str(tmp_path), boundaries=True)
    assert report["ok"], report["failures"]
    cases = {case["case"] for case in report["boundaries"]}
    assert cases == {"silence", "short_clip", "limiter_engaged"}


def test_benchmark_flags_correctness_regression(tmp_path, monkeypatch):
    """If the optimized order degraded any statistic, the benchmark fails."""
    real_run_once = bench._run_once

    def tampered(input_path, digest, target_raw, out_path, cache, seed):
        run = real_run_once(input_path, digest, target_raw, out_path, cache, seed)
        if cache is not None:
            run["result"].static_gain_db += 0.5
        return run

    monkeypatch.setattr(bench, "_run_once", tampered)
    report = bench.run_benchmark(levels=(1,), targets=TWO_TARGETS[:1],
                                 work_dir=str(tmp_path), boundaries=False)
    assert not report["ok"]
    assert any("static_gain_db" in failure for failure in report["failures"])


def test_benchmark_never_masks_raising_runs(tmp_path, monkeypatch):
    def boom(input_path, digest, target_raw, out_path, cache, seed):
        if cache is not None:
            raise RuntimeError("optimized path exploded")
        return real_run_once(input_path, digest, target_raw, out_path, cache, seed)

    real_run_once = bench._run_once
    monkeypatch.setattr(bench, "_run_once", boom)
    report = bench.run_benchmark(levels=(1,), targets=TWO_TARGETS[:1],
                                 work_dir=str(tmp_path), boundaries=False)
    assert not report["ok"]
    assert any("raised" in failure for failure in report["failures"])
    # The failed run is an explicit error entry, not a defaulted row.
    errored = [run for run in report["levels"][0]["runs"] if "error" in run]
    assert errored and "optimized path exploded" in errored[0]["error"]
