"""Browser capture page: start a session over the API, watch it land on disk in the
capture_props layout, tick the plan, and refuse paths outside data/props."""
import sys
import time
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import capture_web
from capture_props import open_camera


def _client(tmp_path, plan=("coke", "none"), n=3):
    rec = capture_web.Recorder(open_camera("hero", "synthetic"), tmp_path, "hero", list(plan), n, 0.0, 0.0)
    return TestClient(capture_web.make_app(rec))


def _wait_idle(client, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = client.get("/api/status").json()
        if s["state"] == "idle":
            return s
        time.sleep(0.02)
    raise AssertionError("session never finished")


def test_session_saves_frames_and_ticks_the_plan(tmp_path):
    client = _client(tmp_path)
    with client:
        assert client.get("/").status_code == 200
        client.post("/api/start", json={"prop": "coke"}).raise_for_status()
        s = _wait_idle(client)
        assert s["saved"] == 3 and s["error"] is None
        assert [p["done"] for p in s["plan"]] == [True, False]
        assert s["counts"]["coke"] == {"sessions": 1, "frames": 3}
        frames = sorted((tmp_path / "coke").glob("hero-*/*.jpg"))
        assert [f.name for f in frames] == ["0000.jpg", "0001.jpg", "0002.jpg"]
        assert all(cv2.imread(str(f)) is not None for f in frames)
        thumb = client.get(s["recent"][0])
        assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


def test_rejects_unknown_props_and_paths_outside_the_data(tmp_path):
    (tmp_path.parent / "secret.jpg").write_bytes(b"x")
    client = _client(tmp_path)
    with client:
        assert client.post("/api/start", json={"prop": "gun"}).status_code == 400
        # %2F, not "../": the client normalises a bare "../" away before it reaches the route
        assert client.get("/frames/..%2Fsecret.jpg").status_code == 404
        assert client.get("/fonts/..%2Fcapture_web.py").status_code == 404
        assert client.get("/fonts/plex-mono-400.woff2").status_code == 200


def test_stop_ends_a_session_early_and_a_second_start_waits(tmp_path):
    rec = capture_web.Recorder(open_camera("hero", "synthetic"), tmp_path, "hero", ["none"], 1000, 0.01, 0.0)
    with TestClient(capture_web.make_app(rec)) as client:
        client.post("/api/start", json={"prop": "none"}).raise_for_status()
        assert client.post("/api/start", json={"prop": "coke"}).status_code == 409
        client.post("/api/stop")
        s = _wait_idle(client)
        assert 0 < s["saved"] < 1000
