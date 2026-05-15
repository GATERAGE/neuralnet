# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the Node.js UI server.js (v0.1.0a7).

Coverage:
  - server.js parses (node --check)            [requires `node` on PATH]
  - Has the expected routes (GET /, POST /ingest, POST /inference)
  - Spawn calls invoke `python -m neuralnet.inference` with `cwd: __dirname`
  - PORT is env-overridable via NEURALNET_PORT (added in v0.1.0a7)
  - Live start: actually launch `node server.js` on a free OS-assigned port,
    GET /, assert 200 + HTML, then clean shutdown
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_JS = ROOT / "server.js"


def _get_free_port() -> int:
    """Ask the OS for a free port. We immediately close the socket — there's
    a tiny race window between releasing the port here and Node grabbing it,
    but in practice this is fine for smoke tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _node_available() -> bool:
    if shutil.which("node") is None:
        return False
    try:
        r = subprocess.run(["node", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# ─── Static checks (only require `node --check` for the syntax test) ───


def test_server_js_exists():
    assert SERVER_JS.exists(), "server.js missing"


def test_server_js_has_expected_routes():
    """Structural assertions on the route handlers."""
    text = SERVER_JS.read_text(encoding="utf-8")
    assert "req.method === 'GET'" in text, "no GET handler"
    assert "req.method === 'POST' && req.url === '/ingest'" in text
    assert "req.method === 'POST' && req.url === '/inference'" in text
    assert "req.url === '/'" in text or "req.url === '/index.html'" in text
    assert "req.url === '/style.css'" in text


def test_server_js_spawn_uses_package_module():
    """v0.1.0a5+ contract: spawn calls invoke `python -m neuralnet.inference`
    with cwd pinned to server.js's own directory."""
    text = SERVER_JS.read_text(encoding="utf-8")
    # Both spawns should use the -m flag form and reference the package
    # module name, not the removed top-level shim `rag_inference.py`.
    assert "'-m', 'neuralnet.inference'" in text, (
        "server.js spawn should use the `-m neuralnet.inference` package form"
    )
    assert "rag_inference.py" not in text, (
        "server.js should not reference the removed rag_inference.py shim"
    )
    # cwd: __dirname is the cwd-safety contract from v0.1.0a6
    assert "cwd: __dirname" in text, (
        "server.js spawns must pin cwd to __dirname (v0.1.0a6 contract)"
    )


def test_server_js_port_env_overridable():
    """v0.1.0a7 contract: PORT respects NEURALNET_PORT (and PORT) env vars."""
    text = SERVER_JS.read_text(encoding="utf-8")
    assert "NEURALNET_PORT" in text, (
        "server.js should honor NEURALNET_PORT (added in v0.1.0a7)"
    )


@pytest.mark.skipif(not _node_available(), reason="`node` not on PATH")
def test_server_js_syntax_check():
    """`node --check server.js` must succeed."""
    result = subprocess.run(
        ["node", "--check", str(SERVER_JS)],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"node --check failed:\nSTDOUT: {result.stdout.decode()}\n"
        f"STDERR: {result.stderr.decode()}"
    )


# ─── Live start: actually launch the server ───


@pytest.mark.skipif(not _node_available(), reason="`node` not on PATH")
def test_server_js_starts_and_serves_static():
    """Launch `node server.js` on a free port, GET /, assert 200 + HTML,
    then SIGTERM. No Python pipeline involved — only the static-file path."""
    port = _get_free_port()
    env = {**os.environ, "NEURALNET_PORT": str(port)}
    proc = subprocess.Popen(
        ["node", str(SERVER_JS)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait up to 5s for the server to start.
        url = f"http://127.0.0.1:{port}/"
        deadline = time.time() + 5.0
        last_err = None
        body = None
        status = None
        content_type = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    status = resp.status
                    content_type = resp.headers.get("Content-Type", "")
                    body = resp.read().decode("utf-8", errors="ignore")
                break
            except Exception as e:
                last_err = e
                time.sleep(0.1)
        assert status == 200, (
            f"server didn't serve GET / (status={status}, last_err={last_err})"
        )
        assert "text/html" in content_type, f"unexpected content-type: {content_type!r}"
        assert "<html" in body.lower() or "<!doctype" in body.lower(), (
            f"body does not look like HTML; first 200 chars: {body[:200]!r}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


@pytest.mark.skipif(not _node_available(), reason="`node` not on PATH")
def test_server_js_404_on_unknown_route():
    """Unknown GET path returns 404."""
    port = _get_free_port()
    env = {**os.environ, "NEURALNET_PORT": str(port)}
    proc = subprocess.Popen(
        ["node", str(SERVER_JS)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        url = f"http://127.0.0.1:{port}/does-not-exist"
        deadline = time.time() + 5.0
        status = None
        while time.time() < deadline:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    status = resp.status
                break
            except urllib.error.HTTPError as e:
                status = e.code
                break
            except Exception:
                time.sleep(0.1)
        assert status == 404, f"unknown route should 404, got {status}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
