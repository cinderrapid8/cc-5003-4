"""Stdlib HTTP layer for the loudness production service.

Endpoints:
  GET  /                      web UI
  POST /api/tasks             multipart upload (field: file) -> analyze
  GET  /api/tasks             list task records
  GET  /api/tasks/{id}        one record
  GET  /api/tasks/{id}/events server-sent events for status changes
  POST /api/tasks/{id}/jobs   JSON target -> background processing
  POST /api/tasks/{id}/cancel cancellation request
  DELETE /api/tasks/{id}      remove task and all its files
  GET  /api/downloads         downloadable (successful) exports only
  GET  /api/tasks/{id}/file?path=...   stream input/export WAV (confined)
  GET  /api/version           runtime/algorithm provenance
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .components.loudness import MeasurementError
from .components.processing import ProcessingError
from .components.versions import runtime_provenance
from .components.wavfile import WavFormatError
from .service import CancellationRequested, TaskManager

WEB_ROOT = __file__.rsplit("/", 1)[0] + "/../web"
MAX_UPLOAD = 200 * 1024 * 1024


class MultipartError(ValueError):
    pass


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes, str]:
    marker = "boundary="
    idx = content_type.find(marker)
    if idx < 0:
        raise MultipartError("missing multipart boundary")
    boundary = content_type[idx + len(marker):].split(";")[0].strip().strip('"')
    delim = b"--" + boundary.encode("ascii")
    parts = body.split(delim)
    for part in parts:
        if not part or part in (b"--", b"--\r\n", b"\r\n"):
            continue
        if part.endswith(b"\r\n"):
            part = part[:-2]
        header_blob, _, payload = part.partition(b"\r\n\r\n")
        if not payload:
            continue
        headers = header_blob.decode("latin-1", errors="replace").lower()
        if 'name="file"' not in headers:
            continue
        filename = "upload.wav"
        for token in headers.split("\r\n"):
            if "filename=" in token:
                filename = token.split("filename=", 1)[1].strip().strip('"') or filename
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        return filename, payload, headers
    raise MultipartError("no file part in multipart body")


class Handler(BaseHTTPRequestHandler):
    manager: TaskManager = None
    _sse_clients: dict = {}
    _sse_lock = threading.Lock()
    sse_poll_interval = 0.3

    def log_message(self, fmt, *args):  # quiet stderr noise
        pass

    # ------------------------------------------------------------- helpers
    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=400, code=None):
        self._json({"error": message, "code": code or "bad_request"}, status)

    def _notify(self, task_id: str):
        with self._sse_lock:
            queues = list(self._sse_clients.get(task_id, ()))
        record = self.manager.get_task(task_id)
        for queue in queues:
            queue(record)

    # --------------------------------------------------------------- routing
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            return self._serve_static(path.split("/static/", 1)[1], None)
        if path == "/api/tasks":
            return self._json({"tasks": self.manager.list_tasks()})
        if path == "/api/downloads":
            return self._json({"downloads": self.manager.downloadable()})
        if path == "/api/version":
            return self._json(runtime_provenance())
        if path.startswith("/api/tasks/"):
            rest = path[len("/api/tasks/"):]
            if rest.endswith("/events"):
                return self._sse(rest[: -len("/events")])
            if rest.endswith("/file"):
                return self._serve_file(rest[: -len("/file")], parse_qs(parsed.query))
            if "/" not in rest:
                record = self.manager.get_task(rest)
                return self._json(record, 200) if record else self._error("not found", 404)
        return self._error("not found", 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/tasks":
            return self._upload()
        if path.startswith("/api/tasks/"):
            rest = path[len("/api/tasks/"):]
            if rest.endswith("/jobs"):
                return self._create_job(rest[: -len("/jobs")])
            if rest.endswith("/cancel"):
                task_id = rest[: -len("/cancel")]
                ok = self.manager.request_cancel(task_id)
                return self._json({"cancelled": ok})
        return self._error("not found", 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/tasks/"):
            task_id = path[len("/api/tasks/"):].strip("/")
            ok = self.manager.delete_task(task_id)
            return self._json({"deleted": ok}, 200 if ok else 404)
        return self._error("not found", 404)

    # ------------------------------------------------------------- upload/job
    def _upload(self):
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._error("expected multipart/form-data", 415)
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_UPLOAD:
            return self._error("empty or oversized upload", 413)
        body = self.rfile.read(length)
        try:
            filename, payload, _ = parse_multipart(body, ctype)
        except MultipartError as exc:
            return self._error(str(exc), 400)
        if not payload:
            return self._error("empty file", 400)
        upload = self.manager.create_task(filename, payload)
        try:
            analysis = self.manager.analyze(upload)
        except (WavFormatError, MeasurementError) as exc:
            record = self.manager.get_task(upload.task_id)
            self._notify(upload.task_id)
            return self._json({
                "task_id": upload.task_id,
                "status": record["status"],
                "error": str(exc),
                "code": getattr(exc, "code", "format_error"),
            }, 422)
        self._notify(upload.task_id)
        return self._json({
            "task_id": upload.task_id,
            "status": "analyzed",
            "analysis": analysis,
            "upload": self.manager.get_task(upload.task_id)["upload"],
        }, 201)

    def _create_job(self, task_id: str):
        length = int(self.headers.get("Content-Length", 0))
        try:
            target = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._error("invalid JSON target", 400)
        record = self.manager.get_task(task_id)
        if record is None:
            return self._error("not found", 404)

        def worker():
            try:
                self.manager.process(task_id, target)
            except (ProcessingError, MeasurementError, WavFormatError,
                    CancellationRequested):
                pass
            except Exception:
                import traceback
                record = self.manager.get_task(task_id)
                if record is not None:
                    record["status"] = "failed"
                    record["error"] = "unexpected worker error"
                    for job in reversed(record.get("jobs", [])):
                        if job.get("status") == "processing":
                            job["status"] = "failed"
                            job["error"] = traceback.format_exc(limit=4)
                            break
                    self.manager._write_record(task_id, record)
            finally:
                self._notify(task_id)

        threading.Thread(target=worker, daemon=True).start()
        return self._json({"started": True, "task_id": task_id}, 202)

    # ------------------------------------------------------------------ files
    def _serve_file(self, task_id: str, query: dict):
        rel = (query.get("path") or [""])[0]
        if not rel:
            return self._error("missing path", 400)
        full = self.manager.resolve_path(task_id, rel)
        if full is None:
            return self._error("not found", 404)
        with open(full, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, name: str, ctype: str | None):
        safe = name.replace("..", "").lstrip("/")
        full = f"{WEB_ROOT}/static/{safe}" if name != "index.html" else f"{WEB_ROOT}/index.html"
        try:
            with open(full, "rb") as fh:
                data = fh.read()
        except OSError:
            return self._error("not found", 404)
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -------------------------------------------------------------------- sse
    def _sse(self, task_id: str):
        if self.manager.get_task(task_id) is None:
            return self._error("not found", 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        cond = threading.Condition()

        def push(_record):
            with cond:
                cond.notify_all()

        with self._sse_lock:
            self._sse_clients.setdefault(task_id, []).append(push)
        try:
            last_status = None
            while True:
                record = self.manager.get_task(task_id)
                if record and record["status"] != last_status:
                    last_status = record["status"]
                    self.wfile.write(
                        f"event: status\ndata: {json.dumps(record)}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                if record and record["status"] in ("succeeded", "failed", "cancelled"):
                    break
                with cond:
                    cond.wait(self.sse_poll_interval)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self._sse_lock:
                clients = self._sse_clients.get(task_id, [])
                if push in clients:
                    clients.remove(push)


def build_server(data_root: str, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    manager = TaskManager(data_root)
    Handler.manager = manager
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.task_manager = manager
    return httpd
