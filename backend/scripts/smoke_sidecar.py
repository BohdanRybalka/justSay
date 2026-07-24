"""Smoke-test the PyInstaller-frozen JustSay backend sidecar.

Mirrors the v0.8.0 manual smoke matrix that proved the bundled binary
can boot without a system Python install:

  1. Verify the chosen port is free (bind/close test — portable across
     Windows + macOS, dodges the Test-NetConnection exit-code trap).
  2. Launch ``dist/justsay-backend/justsay-backend[.exe] --host <h> --port <p>``.
  3. Poll ``GET /health`` every 500 ms for up to 30 s; require
     ``{"status": "ok", ...}`` (exact match — ``"degraded"`` fails).
  4. GET ``/resources``, ``/settings``, ``/settings/cloud-status``,
     ``/audio/status`` — assert HTTP 200 on each. These cover the
     subsystems that have to import successfully under the frozen
     interpreter (psutil + GPU probe, settings I/O, cloud-key store,
     sounddevice).
  5. Terminate the child (SIGTERM on POSIX, ``taskkill /T /F /PID`` on
     Windows) and wait briefly for exit.

Exit 0 on success, 1 on any failure. Stdlib only — runs before the test
dependency installs in CI.

Usage:
  python backend/scripts/smoke_sidecar.py --host 127.0.0.1 --port 9377
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[smoke] FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def assert_port_free(host: str, port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
    except OSError as exc:
        s.close()
        fail(f"port {port} on {host} is occupied: {exc}")
    finally:
        s.close()
    log(f"port {host}:{port} is free")


def resolve_binary() -> Path:
    here = Path(__file__).resolve().parent
    backend_root = here.parent
    dist = backend_root / "dist" / "justsay-backend"
    name = "justsay-backend.exe" if os.name == "nt" else "justsay-backend"
    binary = dist / name
    if not binary.exists():
        fail(f"sidecar binary not found at {binary}")
    return binary


def http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() if exc.fp else b""
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        return 0, str(exc).encode("utf-8", errors="replace")


def poll_health(host: str, port: int, deadline_s: float = 30.0) -> None:
    url = f"http://{host}:{port}/health"
    start = time.monotonic()
    last_err = "no response"
    while time.monotonic() - start < deadline_s:
        status, body = http_get(url, timeout=1.0)
        if status == 200:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                last_err = f"non-JSON 200 body: {body[:200]!r}"
            else:
                if payload.get("status") == "ok":
                    log(f"/health OK (after {time.monotonic() - start:.1f}s)")
                    return
                last_err = f'/health returned status="{payload.get("status")!r}", want "ok"'
        else:
            last_err = f"HTTP {status} {body[:200]!r}"
        time.sleep(0.5)
    fail(f"/health never returned 200 ok within {deadline_s}s: {last_err}")


def assert_200(host: str, port: int, path: str) -> None:
    url = f"http://{host}:{port}{path}"
    status, body = http_get(url, timeout=10.0)
    if status != 200:
        fail(f"GET {path} returned HTTP {status}: {body[:200]!r}")
    log(f"GET {path} OK")


def terminate(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(child.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            child.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(prog="smoke_sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9377)
    args = parser.parse_args()

    assert_port_free(args.host, args.port)
    binary = resolve_binary()
    log(f"launching {binary}")

    creationflags = 0
    if os.name == "nt":
        creationflags = 0x08000000

    child = subprocess.Popen(
        [str(binary), "--host", args.host, "--port", str(args.port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )

    try:
        poll_health(args.host, args.port)
        for path in ("/resources", "/settings", "/settings/cloud-status", "/audio/status"):
            assert_200(args.host, args.port, path)
        log("all checks passed")
        return 0
    finally:
        terminate(child)


if __name__ == "__main__":
    sys.exit(main())
