import json
import os
import threading
import time

import numpy as np
import pytest

from app.components.wavfile import write_wav
from app.server import Handler, parse_multipart, build_server
from app.service import CancellationRequested, TaskManager
from tests.signals import calibrated_speech_like, silence


@pytest.fixture()
def store(tmp_path):
    return TaskManager(str(tmp_path / "tasks"))


def _wav_bytes(sig, rate, bits=24):
    import io
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    write_wav(path, sig, rate, bits)
    with open(path, "rb") as fh:
        data = fh.read()
    os.remove(path)
    return data


def test_upload_analyze_persists_input_summary(store):
    sig, rate = calibrated_speech_like(seconds=2)
    upload = store.create_task("guide.wav", _wav_bytes(sig, rate))
    analysis = store.analyze(upload)
    record = store.get_task(upload.task_id)
    assert record["status"] == "analyzed"
    assert record["upload"]["sha256"]
    assert analysis["measurement"]["integrated_lufs"] is not None
    assert record["provenance"]["numpy"]
    assert os.path.isfile(upload.stored_path)


def test_silent_upload_is_failed_and_not_downloadable(store):
    sig, rate = silence(1.5)
    upload = store.create_task("quiet.wav", _wav_bytes(sig, rate))
    with pytest.raises(Exception):
        store.analyze(upload)
    record = store.get_task(upload.task_id)
    assert record["status"] == "failed"
    assert store.downloadable() == []


def test_successful_job_records_full_chain_and_appears_in_downloads(store):
    sig, rate = calibrated_speech_like(seconds=2)
    upload = store.create_task("g.wav", _wav_bytes(sig, rate))
    store.analyze(upload)
    job = store.process(upload.task_id, {
        "integrated_lufs": -23, "true_peak_dbtp": -2,
        "sample_rate": 48000, "layout": "mono", "bit_depth": 24,
    })
    assert job["status"] == "succeeded"
    listing = store.downloadable()
    assert len(listing) == 1 and listing[0]["task_id"] == upload.task_id
    chain = job["chain"]
    assert chain["source_measurement"] and chain["output_measurement"]
    assert chain["target"]["integrated_lufs"] == -23


def test_failed_job_output_is_removed_and_not_downloadable(store):
    # A rejected (silent) job leaves no export behind.
    sig, rate = silence(1.5)
    upload = store.create_task("s.wav", _wav_bytes(sig, rate))
    with pytest.raises(Exception):
        store.analyze(upload)
    # Manually drive process against the stored file: it must reject.
    with pytest.raises(Exception):
        store.process(upload.task_id, {
            "integrated_lufs": -23, "true_peak_dbtp": -2,
            "sample_rate": 48000, "layout": "mono",
        })
    exports = os.path.join(store.root, upload.task_id, "exports")
    assert not os.path.exists(exports) or not os.listdir(exports)
    assert store.downloadable() == []


def test_cancellation_cleans_partial_output(store, monkeypatch):
    sig, rate = calibrated_speech_like(seconds=3)
    upload = store.create_task("c.wav", _wav_bytes(sig, rate))
    store.analyze(upload)
    store.request_cancel(upload.task_id)

    called = {"n": 0}
    real_run = __import__("app.service", fromlist=["run_chain"]).run_chain

    def cancelling_chain(audio, target, path, rng=None, **kwargs):
        called["n"] += 1
        raise CancellationRequested(upload.task_id)

    monkeypatch.setattr("app.service.run_chain", cancelling_chain)
    with pytest.raises(CancellationRequested):
        store.process(upload.task_id, {
            "integrated_lufs": -23, "true_peak_dbtp": -2,
            "sample_rate": 48000, "layout": "mono",
        })
    record = store.get_task(upload.task_id)
    assert record["status"] == "cancelled"
    assert store.downloadable() == []
    exports = os.path.join(store.root, upload.task_id, "exports")
    assert not os.listdir(exports)


def test_repeat_jobs_reuse_source_measurement_with_evidence(store):
    sig, rate = calibrated_speech_like(seconds=2)
    upload = store.create_task("g.wav", _wav_bytes(sig, rate))
    store.analyze(upload)
    job1 = store.process(upload.task_id, {
        "integrated_lufs": -23, "true_peak_dbtp": -2,
        "sample_rate": 48000, "layout": "mono", "bit_depth": 24,
    })
    job2 = store.process(upload.task_id, {
        "integrated_lufs": -24, "true_peak_dbtp": -2,
        "sample_rate": 48000, "layout": "mono", "bit_depth": 24,
    })
    assert job1["status"] == job2["status"] == "succeeded"
    ev1, ev2 = job1["evidence"], job2["evidence"]
    # Analysis seeded the cache: both jobs reuse the source measurement.
    assert ev1["counters"]["cache_hits"] >= 1
    assert ev2["counters"]["cache_hits"] >= 1
    # Only the export re-measurement remains as a full measure per job.
    assert ev1["counters"]["measures"] == 1
    assert ev2["counters"]["measures"] == 1
    # Decode counts are evidence, not savings: input + export every job.
    assert ev1["counters"]["decodes"] == 2
    assert ev2["counters"]["decodes"] == 2
    # The true-peak check ran on every job, cache hit or not.
    assert ev1["counters"]["true_peak_checks"] >= 1
    assert ev2["counters"]["true_peak_checks"] >= 1
    # Same source statistics, different targets -> different exports.
    assert ev1["source_integrated_lufs"] == ev2["source_integrated_lufs"]
    assert ev1["output_integrated_lufs"] != ev2["output_integrated_lufs"]


def test_delete_task_removes_everything(store):
    sig, rate = calibrated_speech_like(seconds=1)
    upload = store.create_task("d.wav", _wav_bytes(sig, rate))
    store.analyze(upload)
    task_dir = os.path.join(store.root, upload.task_id)
    assert os.path.isdir(task_dir)
    assert store.delete_task(upload.task_id)
    assert not os.path.exists(task_dir)


def test_path_resolver_cannot_escape_task_dir(store, tmp_path):
    sig, rate = calibrated_speech_like(seconds=1)
    upload = store.create_task("p.wav", _wav_bytes(sig, rate))
    assert store.resolve_path(upload.task_id, "../../etc/passwd") is None
    assert store.resolve_path(upload.task_id, upload.stored_path and
                              os.path.basename(upload.stored_path)) is not None


def test_interrupted_active_task_marked_failed_on_restart(tmp_path):
    root = str(tmp_path / "tasks")
    manager = TaskManager(root)
    sig, rate = calibrated_speech_like(seconds=1)
    upload = manager.create_task("r.wav", _wav_bytes(sig, rate))
    manager.analyze(upload)
    # Simulate a crash mid-processing.
    record = manager.get_task(upload.task_id)
    record["status"] = "processing"
    manager._write_record(upload.task_id, record)
    revived = TaskManager(root)
    assert revived.get_task(upload.task_id)["status"] == "failed"


# --------------------------------------------------------------------- HTTP
def _multipart(filename, content):
    boundary = "----testboundary123"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def test_parse_multipart_extracts_file():
    content = b"0" * 100
    body, ctype = _multipart("a.wav", content)
    name, payload, _ = parse_multipart(body, ctype)
    assert name == "a.wav" and payload == content


@pytest.fixture()
def http_server(tmp_path):
    httpd = build_server(str(tmp_path / "http_tasks"), "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    import urllib.request

    yield httpd, urllib.request
    httpd.shutdown()


def _url(httpd, path):
    host, port = httpd.server_address
    return f"http://{host}:{port}{path}"


def test_http_upload_job_and_download_flow(http_server):
    httpd, urllib = http_server
    sig, rate = calibrated_speech_like(seconds=2)
    body, ctype = _multipart("speech.wav", _wav_bytes(sig, rate))
    req = urllib.Request(
        _url(httpd, "/api/tasks"), data=body,
        headers={"Content-Type": ctype}, method="POST",
    )
    with urllib.urlopen(req) as resp:
        import json

        uploaded = json.loads(resp.read())
    assert uploaded["status"] == "analyzed"
    task_id = uploaded["task_id"]

    job_body = json.dumps({
        "integrated_lufs": -23, "true_peak_dbtp": -2,
        "sample_rate": 48000, "layout": "mono",
    }).encode()
    req = urllib.Request(
        _url(httpd, f"/api/tasks/{task_id}/jobs"), data=job_body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.urlopen(req) as resp:
        assert resp.status == 202

    deadline = time.time() + 30
    record = None
    while time.time() < deadline:
        with urllib.urlopen(_url(httpd, f"/api/tasks/{task_id}")) as resp:
            record = json.loads(resp.read())
        if record["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.2)
    assert record["status"] == "succeeded"

    with urllib.urlopen(_url(httpd, "/api/downloads")) as resp:
        downloads = json.loads(resp.read())["downloads"]
    assert len(downloads) == 1
    with urllib.urlopen(
        _url(httpd, f"/api/tasks/{task_id}/file?path="
             + record["jobs"][0]["chain"]["output_path"])
    ) as resp:
        assert resp.status == 200 and resp.read(4) == b"RIFF"


def test_http_version_endpoint_reports_versions(http_server):
    httpd, urllib = http_server
    with urllib.urlopen(_url(httpd, "/api/version")) as resp:
        provenance = json.loads(resp.read())
    assert provenance["service_version"]
    assert provenance["components"]["limiter"]["oversample"] == 4
