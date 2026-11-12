"""Recompute measurements for a stored task and print a comparison table.

Independent re-measurement of both the uploaded input and every exported
output, placed next to the values recorded in task.json. This demonstrates
that the recorded numbers are reproducible from the files and algorithm
settings alone, rather than being trusted from filter parameters.

Run:  python3 scripts/recompute_task.py <task_dir>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.loudness import measure, measure_file
from app.components.wavfile import read_wav


def _row(label, recorded, recomputed, key):
    rv = "—" if not recorded or recorded.get(key) is None else f"{recorded[key]:.3f}"
    nv = "—" if recomputed.get(key) is None else f"{recomputed[key]:.3f}"
    return f"{label:<26} recorded={rv:>10}   recomputed={nv:>10}"


def main(task_dir: str) -> None:
    with open(os.path.join(task_dir, "task.json"), encoding="utf-8") as fh:
        record = json.load(fh)

    input_path = os.path.join(task_dir, record["upload"]["stored_as"])
    input_audio = read_wav(input_path)
    input_re = measure(
        input_audio.samples, input_audio.sample_rate, allow_failure=True
    ).as_dict()
    input_rec = (record.get("analysis") or {}).get("measurement") or {}

    print(f"task {record['task_id']}  ({record['upload']['original_filename']})")
    print("INPUT")
    for key, label in [
        ("integrated_lufs", "integrated (LUFS)"),
        ("true_peak_dbtp", "true peak (dBTP)"),
        ("sample_peak_dbfs", "sample peak (dBFS)"),
        ("rms_dbfs", "RMS (dBFS)"),
        ("lra_lu", "LRA (LU)"),
    ]:
        print("  " + _row(label, input_rec, input_re, key))

    for job in record.get("jobs", []):
        chain = job.get("chain")
        if not chain or not chain.get("output_path"):
            print(f"\njob {job['job_id']}: {job['status']} (no export)")
            continue
        output_path = os.path.join(task_dir, chain["output_path"])
        output_re = measure_file(output_path).as_dict()
        output_rec = chain.get("output_measurement") or {}
        print(f"\njob {job['job_id']}: {job['status']}")
        for key, label in [
            ("integrated_lufs", "integrated (LUFS)"),
            ("true_peak_dbtp", "true peak (dBTP)"),
            ("sample_peak_dbfs", "sample peak (dBFS)"),
            ("rms_dbfs", "RMS (dBFS)"),
            ("lra_lu", "LRA (LU)"),
        ]:
            print("  " + _row(label, output_rec, output_re, key))
        if chain.get("unmet_conditions"):
            print("  unmet: " + " | ".join(chain["unmet_conditions"]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: recompute_task.py <task_dir>")
    main(os.path.abspath(sys.argv[1]))
