"""Task orchestration: uploads, background processing, cancellation, cleanup.

A task directory holds the original upload, the export (only on success)
and a JSON record containing the input summary, the full processing chain
with every algorithm setting, runtime provenance and the re-measured result.

Failure and cancellation never produce a downloadable artifact: partial
outputs are deleted and the task is excluded from the download listing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from .components.loudness import MeasurementError, measure
from .components.processing import ProcessingError, Target, run_chain
from .components.versions import runtime_provenance
from .components.wavfile import WavFormatError, read_wav

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
ACTIVE_STATES = {"queued", "analyzing", "processing", "measuring"}


class CancellationRequested(Exception):
    pass


@dataclass
class Upload:
    task_id: str
    original_filename: str
    stored_path: str
    sha256: str
    size_bytes: int


class TaskManager:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.RLock()
        self._task_locks: dict[str, threading.RLock] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._records: dict[str, dict] = {}
        for task_id in os.listdir(self.root):
            record_path = os.path.join(self.root, task_id, "task.json")
            if os.path.isfile(record_path):
                try:
                    with open(record_path, encoding="utf-8") as fh:
                        record = json.load(fh)
                    self._records[task_id] = record
                    if record.get("status") in ACTIVE_STATES:
                        record["status"] = "failed"
                        record["error"] = "interrupted by service restart"
                        self._write_record(task_id, record)
                except (json.JSONDecodeError, OSError):
                    continue

    # ------------------------------------------------------------------ paths
    def _task_lock(self, task_id: str) -> threading.RLock:
        with self._lock:
            lock = self._task_locks.get(task_id)
            if lock is None:
                lock = threading.RLock()
                self._task_locks[task_id] = lock
            return lock
    def _task_dir(self, task_id: str) -> str:
        return os.path.join(self.root, task_id)

    def _record_path(self, task_id: str) -> str:
        return os.path.join(self._task_dir(task_id), "task.json")

    def _write_record(self, task_id: str, record: dict) -> None:
        tmp = self._record_path(task_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self._record_path(task_id))
        with self._lock:
            self._records[task_id] = record

    # ------------------------------------------------------------------ upload
    def create_task(self, filename: str, content: bytes) -> Upload:
        task_id = uuid.uuid4().hex
        task_dir = self._task_dir(task_id)
        os.makedirs(task_dir, exist_ok=True)
        safe_name = os.path.basename(filename).replace(os.sep, "_") or "upload.wav"
        stored = os.path.join(task_dir, "input" + os.path.splitext(safe_name)[1])
        with open(stored, "wb") as fh:
            fh.write(content)
        record = {
            "task_id": task_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": "queued",
            "upload": {
                "original_filename": safe_name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "stored_as": os.path.basename(stored),
            },
            "analysis": None,
            "jobs": [],
            "error": None,
            "provenance": runtime_provenance(),
        }
        self._write_record(task_id, record)
        with self._lock:
            self._cancel[task_id] = threading.Event()
        return Upload(task_id, safe_name, stored, record["upload"]["sha256"], len(content))

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in sorted(
                self._records.values(), key=lambda r: r["created_at"], reverse=True
            )]

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            record = self._records.get(task_id)
            return dict(record) if record else None

    def downloadable(self) -> list[dict]:
        """Only succeeded tasks with an existing export are downloadable."""
        items = []
        for record in self.list_tasks():
            for job in record.get("jobs", []):
                output = (job.get("chain") or {}).get("output_path")
                if record["status"] == "succeeded" and job.get("status") == "succeeded" and output:
                    full = os.path.join(self._task_dir(record["task_id"]), output)
                    if os.path.isfile(full):
                        items.append({
                            "task_id": record["task_id"],
                            "filename": f"{record['task_id']}_{os.path.basename(output)}",
                            "job_id": job["job_id"],
                            "output_path": output,
                        })
        return items

    # --------------------------------------------------------------- analysis
    def analyze(self, upload: Upload) -> dict:
        with self._task_lock(upload.task_id):
            return self._analyze_locked(upload)

    def _analyze_locked(self, upload: Upload) -> dict:
        record = self.get_task(upload.task_id)
        self._check_cancel(upload.task_id)
        self._update(upload.task_id, status="analyzing")
        try:
            audio = read_wav(upload.stored_path)
            # Strict mode: silence and sub-block segments raise and are
            # reported separately, but a non-fatal measurement is still stored
            # so the UI can show the numeric bounds that triggered the reject.
            measurement = measure(audio.samples, audio.sample_rate, allow_failure=True)
        except (WavFormatError, MeasurementError) as exc:
            self._fail(upload.task_id, f"analysis rejected: {exc}")
            raise
        analysis = {
            "container": {
                "sample_rate": audio.sample_rate,
                "n_channels": audio.n_channels,
                "bit_depth": audio.bit_depth,
                "is_float": audio.is_float,
                "duration_s": round(audio.duration_s, 6),
            },
            "measurement": measurement.as_dict(),
        }
        if measurement.warnings:
            # Silence / too-short material: reject explicitly, keep the numbers.
            record = self.get_task(upload.task_id)
            record["analysis"] = analysis
            record["status"] = "failed"
            record["error"] = "analysis rejected: " + ",".join(measurement.warnings)
            record["updated_at"] = time.time()
            self._write_record(upload.task_id, record)
            raise MeasurementError(record["error"])
        record = self.get_task(upload.task_id)
        record["analysis"] = analysis
        record["status"] = "analyzed"
        record["updated_at"] = time.time()
        self._write_record(upload.task_id, record)
        return analysis

    # ------------------------------------------------------------- processing
    def process(self, task_id: str, target_raw: dict) -> dict:
        # Serialize all mutating work on one task so concurrent jobs cannot
        # overwrite each other's task.json updates (lost-job bug).
        with self._task_lock(task_id):
            return self._process_locked(task_id, target_raw)

    def _process_locked(self, task_id: str, target_raw: dict) -> dict:
        record = self.get_task(task_id)
        if record is None:
            raise KeyError(task_id)
        if record["status"] not in ("analyzed", "succeeded"):
            raise ProcessingError(f"task in state {record['status']} cannot start processing")

        job_id = uuid.uuid4().hex[:12]
        target = Target.from_dict(target_raw)
        stored = os.path.join(
            self._task_dir(task_id), record["upload"]["stored_as"]
        )
        output_rel = os.path.join("exports", f"{job_id}_output.wav")
        output_abs = os.path.join(self._task_dir(task_id), output_rel)
        os.makedirs(os.path.dirname(output_abs), exist_ok=True)

        job = {
            "job_id": job_id,
            "submitted_at": time.time(),
            "status": "processing",
            "target": target.as_dict(),
            "chain": None,
            "error": None,
        }
        record["jobs"].append(job)
        record["status"] = "processing"
        self._write_record(task_id, record)

        try:
            self._check_cancel(task_id)
            audio = read_wav(stored)
            result = run_chain(audio, target, output_abs,
                               rng=np_rng_seeded(job_id))
            self._check_cancel(task_id)

            job["chain"] = result.as_dict()
            if result.status == "rejected":
                job["status"] = "rejected"
                self._remove_output(task_id, output_rel)
                job["chain"]["output_path"] = None
                record["status"] = "analyzed"
            else:
                # Persist a task-relative path so records stay portable and
                # resolve_path can confine downloads to the task directory.
                job["chain"]["output_path"] = output_rel
                job["status"] = "succeeded"
                record["status"] = "succeeded"
            job["finished_at"] = time.time()
            self._write_record(task_id, record)
            return dict(job)
        except CancellationRequested:
            self._remove_output(task_id, output_rel)
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
            record["status"] = "cancelled"
            self._write_record(task_id, record)
            raise
        except (ProcessingError, WavFormatError, MeasurementError) as exc:
            self._remove_output(task_id, output_rel)
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            record["status"] = "failed"
            record["error"] = str(exc)
            self._write_record(task_id, record)
            raise

    # ----------------------------------------------------------------- control
    def request_cancel(self, task_id: str) -> bool:
        with self._lock:
            event = self._cancel.get(task_id)
            record = self._records.get(task_id)
        if event is None or record is None:
            return False
        if record["status"] in TERMINAL_STATES:
            return False
        event.set()
        return True

    def _check_cancel(self, task_id: str) -> None:
        with self._lock:
            event = self._cancel.get(task_id)
        if event is not None and event.is_set():
            raise CancellationRequested(task_id)

    def delete_task(self, task_id: str) -> bool:
        record = self.get_task(task_id)
        if record is None:
            return False
        with self._task_lock(task_id):
            self.request_cancel(task_id)
            with self._lock:
                self._records.pop(task_id, None)
                self._cancel.pop(task_id, None)
                self._task_locks.pop(task_id, None)
            shutil.rmtree(self._task_dir(task_id), ignore_errors=True)
        return True

    # ---------------------------------------------------------------- helpers
    def _update(self, task_id: str, **changes) -> None:
        record = self.get_task(task_id)
        record.update(changes)
        record["updated_at"] = time.time()
        self._write_record(task_id, record)

    def _fail(self, task_id: str, message: str) -> None:
        record = self.get_task(task_id)
        record["status"] = "failed"
        record["error"] = message
        record["updated_at"] = time.time()
        exports = os.path.join(self._task_dir(task_id), "exports")
        if os.path.isdir(exports):
            shutil.rmtree(exports, ignore_errors=True)
        self._write_record(task_id, record)

    def _remove_output(self, task_id: str, output_rel: str) -> None:
        path = os.path.join(self._task_dir(task_id), output_rel)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def resolve_path(self, task_id: str, rel: str) -> str | None:
        """Safe path resolver confined to the task directory."""
        task_dir = self._task_dir(task_id)
        candidate = os.path.normpath(os.path.join(task_dir, rel))
        if not candidate.startswith(task_dir + os.sep) and candidate != task_dir:
            return None
        return candidate if os.path.isfile(candidate) else None


def np_rng_seeded(seed_text: str) -> "np.random.Generator":
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[: 8], "big")
    return np.random.Generator(np.random.PCG64(seed))
